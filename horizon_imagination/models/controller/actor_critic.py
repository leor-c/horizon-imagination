from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Categorical, Independent, Normal
from tensordict.tensordict import TensorDict
from loguru import logger

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.modules.lightweight_seq_model import LightweightSeqModel
from horizon_imagination.modules.regression import RegressionHead


class ActorHead(nn.Module, Configurable, ABC):
    @dataclass(kw_only=True)
    class Config(BaseConfig):
        latent_dim: int
        actor_bias: Optional[tuple[float, ...]] = None
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.head = self._build_head()
        if config.actor_bias is not None:
            bias = torch.tensor(config.actor_bias, device=config.device, dtype=self.head.bias.dtype)
            self.head.bias.data = bias
            logger.info(f'Actor bias: {self.head.bias}')

    @abstractmethod
    def _build_head(self):
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Distribution:
        pass


class DiscreteActorHead(ActorHead):
    @dataclass(kw_only=True)
    class Config(ActorHead.Config):
        num_actions: int

    def _build_head(self):
        return nn.Linear(
            in_features=self.config.latent_dim,
            out_features=self.config.num_actions,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def forward(self, x: Tensor) -> Categorical:
        logits = self.head(x)

        return Categorical(logits=logits)


class GaussianActorHead(ActorHead):
    """
    A diagonal-Gaussian policy over a continuous (`Box`) action space.

    The head emits `2 * action_dim` values per step, read as [mean | log_std], so the
    spread is state-dependent. `Independent(..., 1)` sums the per-dimension log-probs
    and entropies over the action axis, which keeps `log_prob` and `entropy` shaped
    (B, T) -- exactly what the actor-critic losses expect from a `Categorical`.

    Actions are unbounded here: the `Box` bounds are enforced once, where the action
    reaches the environment (see `Controller.collect_data`), which keeps `log_prob`
    and `entropy` exact rather than needing a squashing correction.
    """

    @dataclass(kw_only=True)
    class Config(ActorHead.Config):
        action_dim: int
        # Initial spread, applied as the bias of the log_std half of the head. exp(-0.5)
        # ~ 0.6, a reasonable starting scale for the [-1, 1] boxes that are typical.
        init_log_std: float = -0.5
        log_std_min: float = -5.0
        log_std_max: float = 2.0

    def __init__(self, config: Config, *args, **kwargs):
        assert config.actor_bias is None, \
            "actor_bias is a per-env prior over discrete actions; it has no meaning " \
            "for a Gaussian head, whose output is a [mean | log_std] pair."
        super().__init__(config, *args, **kwargs)

        # Start with a near state-independent spread: the log_std rows begin at
        # `init_log_std` and barely respond to the input, so early training explores at
        # a predictable scale instead of at whatever the random projection produces.
        with torch.no_grad():
            self.head.bias[config.action_dim:] = config.init_log_std
            self.head.weight[config.action_dim:] *= 0.01

    def _build_head(self):
        return nn.Linear(
            in_features=self.config.latent_dim,
            out_features=2 * self.config.action_dim,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def forward(self, x: Tensor) -> Independent:
        mean, log_std = self.head(x).chunk(2, dim=-1)
        std = log_std.clamp(self.config.log_std_min, self.config.log_std_max).exp()

        return Independent(Normal(mean, std), 1)


class CriticHead(nn.Module, Configurable):
    @dataclass(kw_only=True)
    class Config(BaseConfig):
        latent_dim: int
        hl_gauss_num_bins: int = 129
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.head = RegressionHead(
            in_features=config.latent_dim,
            sym_log_normalize=True,
            sym_exp_order=1,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        values, logits = self.head(x)

        return values, logits

    def training_step(self, logits: Tensor, targets: Tensor) -> Tensor:
        loss = self.head.compute_loss(logits, target=targets)

        return loss


class OutputsBuffer:
    def __init__(self):
        self.actions_dist = []
        self.values = []
        self.v_logits = []


class ActorCritic(nn.Module, Configurable):
    @dataclass
    class Config(BaseConfig):
        backbone: LightweightSeqModel.Config
        shared_backbone: bool
        use_clean_diffused_actors: bool
        actor: ActorHead.Config
        critic: CriticHead.Config

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        assert config.backbone.ignore_actions, f"Currently not support for actions..."

        actor_backbone_cfg = config.backbone.clone()
        actor_backbone_cfg.condition_on_noise_level = True
        self.actor_backbone = actor_backbone_cfg.make_instance()
        self.actor_state = None

        if not config.shared_backbone:
            critic_cfg = config.clone()
            critic_cfg.backbone = config.backbone.clone()
            # critic_cfg.backbone.ignore_actions = False
            self.critic_backbone = critic_cfg.backbone.make_instance()
            self.critic_state = None

        config.actor.device = config.backbone.device
        config.critic.device = config.backbone.device

        config.actor.dtype = config.backbone.dtype
        config.critic.dtype = config.backbone.dtype

        self.actor: ActorHead = config.actor.make_instance()
        self.critic: CriticHead = config.critic.make_instance()

        self.clean_actor_backbone = self.actor_backbone
        self.clean_actor_head = self.actor
        if self.config.use_clean_diffused_actors:
            self.clean_actor_backbone = config.backbone.make_instance()
            self.clean_actor_head: ActorHead = config.actor.make_instance()

        self.outputs_buffer = None

    def _clean_actor_noise_level(self, obs: TensorDict) -> Optional[Tensor]:
        """
        `clean_actor_backbone` only requires a noise-level input when it is
        aliased to `actor_backbone` (use_clean_diffused_actors=False): that shared
        network is also conditioned on noise level during noisy rollout steps, so
        it needs the true (clean) t=1 value here to stay consistent. When it is a
        dedicated backbone, it never sees noisy input and needs no time input at all.
        """
        if self.clean_actor_backbone is not self.actor_backbone:
            return None
        dtype = self.config.backbone.dtype or torch.float32
        return torch.ones(obs.shape[:2], device=obs.device, dtype=dtype)

    def reset(
        self, context_actions = None, context_obs = None, pad_mask = None
    ) -> tuple[Distribution, Tensor, Tensor]:
        if context_actions is None or context_obs is None:
            assert context_actions is None and context_obs is None
            self.actor_state = None
            if not self.config.shared_backbone:
                self.critic_state = None
            return

        x, self.actor_state = self.clean_actor_backbone(
            context_actions, context_obs, None, pad_mask,
            noise_level=self._clean_actor_noise_level(context_obs),
        )
        last_action_dist = self.clean_actor_head(x[:, -1:])

        if not self.config.shared_backbone:
            x, self.critic_state = self.critic_backbone(
                context_actions, context_obs, None, pad_mask
            )
        last_value, last_v_logits = self.critic(x[:, -1:])

        self._record_if_needed(last_action_dist, last_value, last_v_logits)

        return last_action_dist, last_value, last_v_logits

    def forward(
        self,
        prev_actions,
        obs,
        advance_state: bool = False,
        compute_actor: bool = True,
        compute_critic: bool = True,
        noise_level: Tensor = None,
    ) -> tuple[Distribution, Tensor, Tensor]:
        if not self.config.shared_backbone:
            return self._forward_separate_backbones(
                prev_actions=prev_actions,
                obs=obs,
                advance_state=advance_state,
                compute_actor=compute_actor,
                compute_critic=compute_critic,
                noise_level=noise_level,
            )
        else:
            return self._forward_shared_backbone(
                prev_actions=prev_actions,
                obs=obs,
                advance_state=advance_state,
                compute_actor=compute_actor,
                compute_critic=compute_critic,
                noise_level=noise_level,
            )

    def _forward_shared_backbone(
        self,
        prev_actions,
        obs,
        advance_state: bool = False,
        compute_actor: bool = True,
        compute_critic: bool = True,
        noise_level: Tensor = None,
    ):
        assert self.config.shared_backbone

        x, actor_state = self.actor_backbone(prev_actions, obs, self.actor_state, noise_level=noise_level)

        if compute_actor:
            action_dist = self.actor(x)
        else:
            action_dist = None

        if compute_critic:
            value, v_logits = self.critic(x)
        else:
            value, v_logits = None, None

        if advance_state:
            self.actor_state = actor_state

        self._record_if_needed(action_dist, value, v_logits)

        return action_dist, value, v_logits
    
    def _forward_separate_backbones(
        self,
        prev_actions,
        obs,
        advance_state: bool = False,
        compute_actor: bool = True,
        compute_critic: bool = True,
        noise_level: Tensor = None,
    ):
        assert not self.config.shared_backbone

        if compute_actor:
            x, actor_state = self.actor_backbone(prev_actions, obs, self.actor_state, noise_level=noise_level)
            action_dist = self.actor(x)
        else:
            action_dist = None

        if compute_critic:
            x, critic_state = self.critic_backbone(prev_actions, obs, self.critic_state)
            value, v_logits = self.critic(x)
        else:
            value, v_logits = None, None

        if advance_state:
            if compute_actor:
                self.actor_state = actor_state
            if compute_critic:
                self.critic_state = critic_state

        self._record_if_needed(action_dist, value, v_logits)

        return action_dist, value, v_logits

    def clean_actor(
        self,
        prev_actions,
        obs,
        advance_state: bool = False,
    ):
        x, actor_state = self.clean_actor_backbone(
            prev_actions, obs, self.actor_state,
            noise_level=self._clean_actor_noise_level(obs),
        )
        if advance_state:
            self.actor_state = actor_state
        action_dist = self.clean_actor_head(x)
        self._record_if_needed(action_dist, None, None)
        return action_dist


    def generate(self, last_action, obs, advance_state: bool = False):
        """
        Auto-regressive generation of outputs.
        Appropriate during imagination where future actions are
        not available and must be generated one at a time.
        """
        # TODO: implement this
        pass

    def start_recording_outputs(self) -> None:
        """
        When this function is called all actor-critic experience from this
        moment is being recorded by the actor-critic.
        To stop recording and obtain the recorded data call `stop_recording`.
        """
        self.outputs_buffer = OutputsBuffer()

    def stop_recording(self) -> OutputsBuffer:
        buffer = self.outputs_buffer
        self.outputs_buffer = None

        return buffer

    def _record_if_needed(self, action_dist, value, v_logit):
        if self.outputs_buffer is not None:
            if action_dist is not None:
                self.outputs_buffer.actions_dist.append(action_dist)
            if value is not None:
                self.outputs_buffer.values.append(value)
            if v_logit is not None:
                self.outputs_buffer.v_logits.append(v_logit)
