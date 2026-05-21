from typing import Any
import math

import lightning as L
import torch
from torch import Tensor
from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass, field
from horizon_imagination.utilities.types import ObsKey
from horizon_imagination.utilities import AdamWConfig, shift_fwd
from horizon_imagination.diffusion import RectifiedFlow, BetaTimeSampler, TimeSamplerBase
from horizon_imagination.diffusion.samplers import (
    EulerSampler, SamplerScheduler
)
from horizon_imagination.models.world_model.denoiser import DenoiserBase, VideoDiTDenoiser, KVCache
from horizon_imagination.models.world_model.action_producer import ActionProducer
from horizon_imagination.models.world_model.reward_done_model import RewardDoneModel
from horizon_imagination.modules.transform import PerModalityTransform


class DenoiserWithPolicyWrapper(DenoiserBase):
    """
    Wrap a denoiser model with a policy in the form of an ActionProducer.
    This wrapper provides policy actions to the denoiser, abstracting 
    them from the diffusion sampler.
    All the generated experience is stored for training the actor-critic.
    """
    def __init__(
            self, 
            denoiser: VideoDiTDenoiser, 
            action_producer: ActionProducer, 
            context_obs: Tensor = None, 
            context_actions: Tensor = None, 
            reward_done_model: RewardDoneModel = None,
            rd_state: tuple[Tensor, Tensor] = None,
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.denoiser = denoiser
        self.action_producer = action_producer
        self.reward_done_model = reward_done_model
        self.rd_state = rd_state

        self.context_actions = context_actions
        self.context_obs = context_obs
        ctx_t = None
        ctx_len = 0
        if context_actions is not None:
            ctx_t = torch.ones_like(context_actions)
            ctx_len = context_actions.shape[1]
        self.context_t = ctx_t
        self.context_len = ctx_len
        
        self.actions_buffer = []
        self.log_pi_buffer = []
        self.denoising_times = []
        self.denoiser_kv_cache: KVCache | None = None

    def denoise(
            self,
            x: TensorDict,
            t: Tensor,
            dt: Tensor = None,
            *args,
            **kwargs
    ) -> tuple[TensorDict, Any]:
        """
        :param x: the noisy observation to be denoised. shape (B T C H W)
        :param t: diffusion time indices (per-frame). shape (B T)
        :param state: optional. the state of the sequence model (e.g., KV-cache).
        :param dt_mask: a mask with values 1 for entries where denoising
        takes place, i.e., dt > 0. shape (B T). dt > 0 means that the diffusion time
        of the particular sequence element increases within the current iteration.
        """
        self.denoising_times.append(t)

        # the last step is used for critic value estimation (only).
        actions, log_pi = self.action_producer(x)

        self.actions_buffer.append(actions)
        self.log_pi_buffer.append(log_pi)

        # Shift actions to maintain causality (a_t affects o_{t+1}):
        assert self.context_len > 0
        actions = torch.cat([self.context_actions, actions], dim=1)
        actions = shift_fwd(actions)

        if self.denoiser_kv_cache is None:
            x_ = torch.cat([self.context_obs, x], dim=1)
            t_ = torch.cat([self.context_t, t], dim=1)
            dt_ = torch.cat([torch.zeros_like(self.context_t), dt], dim=1)
            start = self.context_len
            end = torch.logical_or(dt_[0] > 0, t_[0] > 0).sum()  # truncate suffix with dt == 0

            actions = actions[:, :end]
            x_ = x_[:, :end]
            t_ = t_[:, :end]

        else:
            t_ = t
            dt_ = dt
            actions = actions[:, self.context_len:]

            start = self.denoiser_kv_cache.length - self.context_len
            end = torch.logical_or(dt_[0] > 0, t_[0] > 0).sum()

            actions = actions[:, start:end]
            x_ = x[:, start:end]
            t_ = t[:, start:end]

        kv_cache = self.denoiser_kv_cache

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                denoised, new_kv_cache = self.denoiser(x_, t_, actions=actions, kv_cache=kv_cache)

        out = torch.zeros_like(x)
        n_clean = (t_[0] == 1).sum()
        if self.denoiser_kv_cache is None:
            assert n_clean == self.context_len, f"got {n_clean}, expected {self.context_len}"
            self.denoiser_kv_cache = new_kv_cache[
                :, :n_clean]  # keeping context + noise seq, to avoid concat at every step.
            # noise seq elements will be overridden in later calls.

            out[:, :end - self.context_len] = denoised[:, self.context_len:]
        else:
            if n_clean > 0:
                self.denoiser_kv_cache.concat_inplace(new_kv_cache[:, :n_clean])

            out[:, start:end] = denoised

        return out
    
    def get_buffers(self):
        return self.actions_buffer, self.log_pi_buffer, self.denoising_times


class RectifiedFlowWorldModel(L.LightningModule, Configurable):
    @dataclass
    class Config(BaseConfig):
        denoiser_config: VideoDiTDenoiser.Config
        sampler_scheduler: SamplerScheduler.Config
        denoiser_optim: AdamWConfig = field(default_factory=AdamWConfig)
        reward_optim: AdamWConfig = field(default_factory=AdamWConfig)
        obs_transform: PerModalityTransform = None
        reward_done_model: RewardDoneModel.Config = None
        time_sampler: TimeSamplerBase = BetaTimeSampler()

    def __init__(
            self, 
            config: Config, 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.obs_transform = config.obs_transform  
        self.flow_matching = RectifiedFlow(
            time_sampler=config.time_sampler,
            sigma_min=1e-5
        )
        self.denoiser = VideoDiTDenoiser(config=config.denoiser_config)
        self.reward_done_model = config.reward_done_model.make_instance() if config.reward_done_model is not None else None

    @torch.no_grad()
    def get_obs_from_batch(self, batch: TensorDict) -> TensorDict:
        obs = batch['observation']
        if self.obs_transform is not None:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                obs = self.obs_transform.transform(obs)
        return obs.float()

    def training_step(self, batch, batch_idx, log_dict_fn = None):
        self.train()

        obs = self.get_obs_from_batch(batch)
        mask = batch['mask'] if 'mask' in batch else None

        # It makes more sense to look at (action, obs) blocks!
        shifted_action = shift_fwd(batch['action'])

        loss, x_t = self.flow_matching.training_step(
            x_clean=obs, 
            time_dims=2, 
            denoiser=self.denoiser,
            denoiser_kwargs={
                'actions': shifted_action,
                'state': None
            },
            mask=mask
        )
        loss_dict = {'world_model/denoiser_loss': loss}

        if self.reward_done_model is not None:
            reward_end_mask = mask.clone()
            reward_end_mask[:, 0] = 0  # We only predict based on 2 obs (o_t, a_t, o_{t+1})
            reward_loss, done_loss, info = self.reward_done_model.training_step(
                shifted_action, 
                obs,  #x_t,
                shift_fwd(batch['reward']),
                shift_fwd(batch['terminated']),
                mask=reward_end_mask
            )
            loss_dict['world_model/reward_loss'] = reward_loss
            loss_dict['world_model/terminated_loss'] = done_loss
            for k, v in info.items():
                loss_dict[f'world_model/{k}'] = v

            loss = loss + reward_loss + done_loss
        
        if log_dict_fn is None:
            log_dict_fn = self.log_dict
        log_dict_fn(loss_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def imagine(
            self, 
            policy: ActionProducer, 
            batch_size: int, 
            horizon: int, 
            obs_shape: dict[ObsKey, tuple],
            denoising_steps: int,
            context: TensorDict = None, 
            context_noise_level: float = 0.05,
        ):
        self.eval()

        device = self.config.denoiser_config.device
        dtype = None  # self.config.denoiser_config.dtype
        
        # compute state from context:
        rd_state = None
        if context is not None:
            ctx_obs = self.get_obs_from_batch(context)
            ctx_actions = context['action']
            _, _, rd_state = self.reward_done_model(
                shift_fwd(ctx_actions),
                ctx_obs
            )
        else:
            ctx_obs, ctx_actions = None, None

        # sample noise:
        x = TensorDict(
            {
                k: self.flow_matching.noise.sample((batch_size, horizon, *shape_i[2:]), device=device, dtype=dtype)
                for k, shape_i in obs_shape.items()
            },
            batch_size=(batch_size, horizon),
            device=device,
        )

        # denoise:
        denoiser = DenoiserWithPolicyWrapper(
            self.denoiser,
            action_producer=policy,
            context_actions=ctx_actions,
            context_obs=ctx_obs,
            reward_done_model=self.reward_done_model,
            rd_state=rd_state
        )
        sampler = EulerSampler(
            denoiser=denoiser,
            scheduler=self.config.sampler_scheduler.make_instance()
        )
        segment_obs = sampler.sample(x, num_steps=denoising_steps)
        actions, log_pi, denoising_times = denoiser.get_buffers()

        # Compute the policy given the final (clean) obs:
        clean_actions, clean_log_pi = policy(segment_obs, is_clean=True)
        actions.append(clean_actions)
        log_pi.append(clean_log_pi)

        # Compute rewards and ends for the final (clean) obs:
        shifted_actions = torch.cat([context['action'][:, -1:], clean_actions[:, :-1]], dim=1)
        with torch.no_grad():
            rewards_clean, dones_probs_clean, rd_state = self.reward_done_model(
                shifted_actions,
                segment_obs,
                rd_state
            )

        segment = {
            'observation': segment_obs,
            'action': actions,
            'log_pi': log_pi,
            'reward': rewards_clean,
            'terminated': dones_probs_clean,
            'denoising_times': denoising_times
        }

        return segment
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {
                    "params": self.denoiser.parameters(), 
                    "lr": self.config.denoiser_optim.learning_rate,
                    "betas": self.config.denoiser_optim.betas,
                    "weight_decay": self.config.denoiser_optim.weight_decay,
                },
                {
                    "params": self.reward_done_model.parameters(), 
                    "lr": self.config.reward_optim.learning_rate,
                    "betas": self.config.reward_optim.betas,
                    "weight_decay": self.config.reward_optim.weight_decay,
                },
            ],
            eps=self.config.denoiser_optim.eps,
        )
