from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Distribution, Categorical
from tensordict.tensordict import TensorDict
from loguru import logger

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.modules.lightweight_seq_model import LightweightSeqModel
from horizon_imagination.modules.regression import RegressionHead


class ActorHead(nn.Module, Configurable, ABC):
    @dataclass(kw_only=True)
    class Config(BaseConfig):
        latent_dim: int
        actor_bias: Optional[tuple[float, ...]] = None,
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
        actor: ActorHead.Config
        critic: CriticHead.Config

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        assert config.backbone.ignore_actions, f"Currently not support for actions..."

        self.actor_backbone = config.backbone.make_instance()
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

        self.outputs_buffer = None

    def reset(
        self, context_actions = None, context_obs = None, pad_mask = None
    ) -> tuple[Distribution, Tensor, Tensor]:
        if context_actions is None or context_obs is None:
            assert context_actions is None and context_obs is None
            self.actor_state = None
            if not self.config.shared_backbone:
                self.critic_state = None
            return

        x, self.actor_state = self.actor_backbone(
            context_actions, context_obs, None, pad_mask
        )
        last_action_dist = self.actor(x[:, -1:])

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
    ) -> tuple[Distribution, Tensor, Tensor]:
        if not self.config.shared_backbone:
            return self._forward_separate_backbones(
                prev_actions=prev_actions,
                obs=obs,
                advance_state=advance_state,
                compute_actor=compute_actor,
                compute_critic=compute_critic,
            )
        else:
            return self._forward_shared_backbone(
                prev_actions=prev_actions,
                obs=obs,
                advance_state=advance_state,
                compute_actor=compute_actor,
                compute_critic=compute_critic,
            )
        
    def _forward_shared_backbone(
        self,
        prev_actions, 
        obs, 
        advance_state: bool = False, 
        compute_actor: bool = True,
        compute_critic: bool = True,
    ):
        assert self.config.shared_backbone

        x, actor_state = self.actor_backbone(prev_actions, obs, self.actor_state)

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
    ):
        assert not self.config.shared_backbone

        if compute_actor:
            x, actor_state = self.actor_backbone(prev_actions, obs, self.actor_state)
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
