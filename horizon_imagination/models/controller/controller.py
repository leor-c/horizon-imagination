from typing import Literal, Optional
import lightning as L
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import gymnasium as gym
from tensordict.tensordict import TensorDict
from loguru import logger
from tqdm import tqdm

import episodata as ed

from horizon_imagination.models.controller.actor_critic import ActorCritic, OutputsBuffer
from horizon_imagination.models.controller.return_scaler import EMAScaler
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.utilities import AdamWConfig, shift_fwd, RawMultiModalObs, TensorDictRollingContextBuffer
from horizon_imagination.models.world_model import RectifiedFlowWorldModel
from horizon_imagination.models.world_model.action_producer import (
    StablePolicyActionProducer, NaivePolicyActionProducer
)
from horizon_imagination.data.statistics_collector import ExperienceStatisticsCollector


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2, f"got {rewards.shape}" 
    assert rewards.shape == ends.shape, f"{rewards.shape}, {ends.shape}"  # (B, T)
    assert values.dim() == 2 and values.shape[1] == rewards.shape[1] + 1

    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards + ends.logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in reversed(range(t - 1)):
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


def _to_tensor_dict_obs(obs: RawMultiModalObs):
    obs = TensorDict(
        {k: torch.from_numpy(v) for k, v in obs.items()},
    )
    return obs


def make_valid_mask(ends, t):
    B, T = ends.shape

    # Step 1: For each row, find the first index where A is True
    # If no True in row, set index to D (so the mask becomes all 0)
    # We use torch.cumsum to find the first True
    first_end_indices = torch.where(
        ends.any(dim=1),
        ends.float().cumsum(dim=1).float().argmax(dim=1),
        torch.full((B,), T, device=ends.device, dtype=torch.long)  # if no True found
    )

    # Step 2: Create range matrix
    range_matrix = torch.arange(T, device=ends.device).expand(B, T)

    # Step 3: Create mask
    mask = range_matrix <= first_end_indices.unsqueeze(1)

    # Step 4: Compute the denoising time valid mask:
    valid_denoising_masks = []
    t.append(torch.ones_like(t[-1]))  # to account for the last step
    for i in range(len(t)-1):
        # process the mask of each denoising time
        # Only steps where the denoising time of the next obs grows (dt > 0) are valid.
        # This way, each step contributes the same number of elements to the loss.
        # We want to optimize the policy at steps where input varies, once for each such input.
        # recall that mask steps here are one element ahead of `log_pi`
        step_mask = t[i+1] > t[i]  
        # step_mask[:, :-1] = step_mask[:, 1:]
        # step_mask[:, -1] = 0
        valid_denoising_masks.append(step_mask)

    # Finally, we consider the last 2 iterations: 
    # the first corresponds to the last denoising step,
    # the second iter is where the policy is appied to the final (clean) 
    # sequence:
    # valid_denoising_masks.append(torch.ones_like(valid_denoising_masks[-1]))
    valid_denoising_masks.append(torch.ones_like(valid_denoising_masks[-1]))
    num_noise_lvls = len(valid_denoising_masks)
    valid_denoising_masks = torch.cat(valid_denoising_masks, dim=0)

    # Now, the valid mask is the logical and of the two:
    mask = repeat(mask, 'b t -> (l b) t', l=num_noise_lvls)
    mask = torch.logical_and(mask.bool(), valid_denoising_masks.bool())

    return mask


def compute_original_actor_loss(
        traj_segment,
        lambda_returns,
        values,
        returns_scale,
        N,
        B,
        valid_mask,
        action_dist,
        actor_critic_outs
):
    log_probs = traj_segment['log_pi'][:, :-1]
    advantage = (lambda_returns - values).detach() / returns_scale.to(dtype=values.dtype)
    advantage = repeat(advantage, "B ... -> (N B) ...", N=N, B=B)
    loss_actions = -(log_probs * advantage.detach())
    # loss_actions = rearrange(loss_actions, '(N B) ... -> N B ...', N=N, B=B)
    # loss_actions = loss_actions[-1][torch.where(valid_mask[-1])].mean()
    loss_actions = loss_actions[torch.where(valid_mask)].mean()

    loss_actor = loss_actions

    entropy = torch.cat(
        [
            torch.cat([action_dist.entropy(), d.entropy()], dim=1)
            for d in actor_critic_outs.actions_dist
        ], dim=0
    )[:, :-1]
    entropy = entropy[torch.where(valid_mask)]
    entropy = entropy.mean()

    return loss_actor, entropy, advantage


def compute_clean_diffused_actor_loss(
        traj_segment,
        lambda_returns,
        values,
        returns_scale,
        N,
        B,
        valid_mask,
        action_dist,
        actor_critic_outs
):
    actions_logits = torch.stack(
        [
            torch.cat([action_dist.logits, d.logits], dim=1)
            for d in actor_critic_outs.actions_dist
        ], dim=0
    )[:, :, :-1]

    clean_log_probs = traj_segment['log_pi'][-1, :, :-1]
    advantage = (lambda_returns - values).detach() / returns_scale.to(dtype=values.dtype)
    # advantage = repeat(advantage, "B ... -> (N B) ...", N=N, B=B)
    loss_actions = -(clean_log_probs * advantage.detach())
    # loss_actions = rearrange(loss_actions, '(N B) ... -> N B ...', N=N, B=B)
    # loss_actions = loss_actions[-1][torch.where(valid_mask[-1])].mean()
    valid_mask_blocks = rearrange(valid_mask, '(N B) ... -> N B ...', N=N, B=B)
    loss_actions = loss_actions[torch.where(valid_mask_blocks[-1])].mean()

    noisy_action_logits = actions_logits[:-1]
    clean_action_logits = actions_logits[-1]
    targets = repeat(clean_action_logits.detach(), 'B T ... -> N B T ...', N=N - 1)
    loss_noisy_actions = F.cross_entropy(
        noisy_action_logits[torch.where(valid_mask_blocks[:-1])].flatten(0, 1),
        F.softmax(targets[torch.where(valid_mask_blocks[:-1])].flatten(0, 1), dim=-1)
        )

    loss_actor = loss_actions + loss_noisy_actions

    entropy = torch.stack(
        [
            torch.cat([action_dist.entropy(), d.entropy()], dim=1)
            for d in actor_critic_outs.actions_dist
        ], dim=0
    )[-1, :, :-1]
    entropy = entropy[torch.where(valid_mask_blocks[-1])]
    entropy = entropy.mean()

    return loss_actor, entropy, advantage


def compute_actor_loss(
        use_clean_diffused_actors: bool,
        traj_segment,
        lambda_returns,
        values,
        returns_scale,
        N,
        B,
        valid_mask,
        action_dist,
        actor_critic_outs
):
    if use_clean_diffused_actors:
        return compute_clean_diffused_actor_loss(
            traj_segment, lambda_returns, values, returns_scale, N, B, valid_mask, action_dist, actor_critic_outs
        )
    else:
        return compute_original_actor_loss(
            traj_segment, lambda_returns, values, returns_scale, N, B, valid_mask, action_dist, actor_critic_outs
        )


class Controller(L.LightningModule, Configurable):
    @dataclass
    class Config(BaseConfig):
        actor_critic: ActorCritic.Config
        world_model: RectifiedFlowWorldModel
        optim: AdamWConfig
        controller_context_length: int
        imagination_batch_size: int
        imagination_horizon: int
        num_denoising_steps: int
        context_noise_level: float = 0.0
        gae_gamma: float = 0.99
        gae_lambda: float = 0.95
        entropy_weight: float = 0.001
        return_scaler_decay: float = 0.005
        baseline: Literal['hi', 'ar', 'naive'] = 'hi'

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self._device = config.actor_critic.backbone.device
        self.action_space = config.actor_critic.backbone.action_space

        self.actor_critic: ActorCritic = config.actor_critic.make_instance()
        self.return_scaler = EMAScaler(decay=config.return_scaler_decay)

        self.stats_collector = ExperienceStatisticsCollector()

        self._last_obs = None
        self._context_buffer = TensorDictRollingContextBuffer(
            k=config.controller_context_length,
            action_spec=self.action_space,
        )
        self._episode_id = None

    @torch.no_grad()
    def forward(
            self, 
            env: gym.Env, 
            replay_buffer: ed.Dataset,
            num_steps: int, 
            log_dict_fn = None,
            pbar_update_fn = None
        ):
        """
        Inference - Collect data / act in a real env.
        """
        self.eval()
        self.collect_data(
            env,
            replay_buffer,
            num_steps,
            self._context_buffer,
            log_dict_fn,
            pbar_update_fn,
            self.stats_collector,
        )

    def collect_data(
            self,
            env: gym.Env,
            replay_buffer: ed.Dataset,
            num_steps: int,
            rolling_context_buffer: Optional[TensorDictRollingContextBuffer],
            log_dict_fn=None,
            pbar_update_fn=None,
            stats_collector=None,
        ):
        if num_steps <= 0:
            return

        def prepare_obs(x):
            x = x.to(device=self._device)
            # add batch dim and encode if necessary:
            return self.config.world_model.get_obs_from_batch({'all_observations': x[None, None]})

        def perform_episode_reset():
            obs, info = env.reset()
            td_obs = _to_tensor_dict_obs(obs)
            td_obs_latent = prepare_obs(td_obs)[0, 0]

            # update the rolling context buffer:
            if rolling_context_buffer is not None:
                rolling_context_buffer.reset(td_obs_latent)
                context_actions, context_obs = rolling_context_buffer.get_context()

            # add experience to the replay buffer:
            writer = replay_buffer.new_episode(obs, None)
            self._episode_id = writer.episode_id

            return context_actions, context_obs, writer

        # Set up the context:
        if rolling_context_buffer is not None:
            context = rolling_context_buffer.get_context()
            if context is None:
                """
                in this case, no previous experience exists:
                """
                # New episode, get first obs and set it as context:
                assert self._episode_id is None
                context_actions, context_obs, writer = perform_episode_reset()

            else:
                context_actions, context_obs = context
                assert self._episode_id is not None, f"Episode id not set: {self._episode_id}"
                writer = replay_buffer.episode(self._episode_id).writer()
        else:
            context_actions, context_obs, writer = perform_episode_reset()

        # Collect the data:
        for i in tqdm(range(num_steps), leave=False, desc="Data Collection"):
            # Compute the action with the policy:
            if i == 0:
                action_dist, _, _ = self.actor_critic.reset(context_actions[None], context_obs[None])
            else:
                action_dist = self.actor_critic.clean_actor(action, td_obs_latent, advance_state=True)
            action = action_dist.sample()

            # Step the environment:
            action_raw = action.item() if action.numel() == 1 else action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_raw)

            # Update the replay buffer:
            step = {
                'observation': {str(k): v for k, v in obs.items()},
                'action': action[0, 0].cpu().numpy(),
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated,
                # 'info': info,
            }
            writer.add_step(step)

            # Update the rolling context buffer:
            td_obs = _to_tensor_dict_obs(obs)
            td_obs_latent = prepare_obs(td_obs)
            if rolling_context_buffer is not None:
                rolling_context_buffer.add(action[0, 0], td_obs_latent[0, 0])

            # Update stats:
            if stats_collector is not None:
                stats_collector.step(None, action_raw, reward, terminated, truncated, info)

            if terminated or truncated:
                # perform env reset and update the replay buffer:
                context_actions, context_obs, writer = perform_episode_reset()
                action = context_actions[None, -1:]
                td_obs_latent = context_obs[None, -1:]

                # reset the actor-critic state:
                self.actor_critic.reset()
            
            if pbar_update_fn is not None:
                pbar_update_fn()
        
        if log_dict_fn is None:
            log_dict_fn = self.log_dict
        if stats_collector is not None:
            stats_collector.log_epoch_stats(log_dict_fn)

    @torch.no_grad()
    def collect_test_episodes(self, num_episodes: int, env: gym.Env, collect_stats_only: bool = True):
        # self.eval()
        # init a replay buffer. need to get a schema.
        # self.collect_data(
        #     env,
        #     replay_buffer,
        #     num_steps,
        #     rolling_context_buffer=None,
        # )
        raise NotImplementedError("Not implemented")

    def training_step(self, batch, batch_idx, log_dict_fn = None):
        # Assume suffix batch padding and prefix segment sampling.
        world_model = self.config.world_model

        # set actor critic context:
        context_actions = shift_fwd(batch['action'])
        context_obs = world_model.get_obs_from_batch(batch)
        pad_mask = batch['mask']
        action_dist, value, v_logits = self.actor_critic.reset(context_actions, context_obs, pad_mask)
        first_action = action_dist.sample()
        first_action_log_p = action_dist.log_prob(first_action)
        first_step_outs = (action_dist, first_action_log_p, value, v_logits)

        # set wm context:
        # take last K valid positions per row
        B = pad_mask.shape[0]
        K = 1
        if K == 1:
            wm_context = batch[torch.arange(pad_mask.shape[0]), pad_mask.sum(1) - 1][:, None]
        else:
            last = pad_mask.sum(1) - 1  # [B]
            offsets = torch.arange(K - 1, -1, -1, device=last.device)  # [K], e.g. [3,2,1,0]
            idx = last[:, None] - offsets[None, :]  # [B, K]
            idx = idx.clamp_min(0)  # optional safety

            wm_context = batch[torch.arange(B, device=last.device)[:, None], idx]  # [B, K, ...]
        wm_context['action'][:, -1:] = first_action

        # generate imagined data:
        self.actor_critic.start_recording_outputs()

        if self.config.baseline in ['hi', 'ar']:
            policy = StablePolicyActionProducer(self.actor_critic)
        else:
            assert self.config.baseline == 'naive', f"Got {self.config.baseline}"
            policy = NaivePolicyActionProducer(self.actor_critic)

        traj_segment = world_model.imagine(
            policy=policy,
            batch_size=self.config.imagination_batch_size,
            horizon=self.config.imagination_horizon,
            obs_shape={k: v.shape for k, v in context_obs.items()},
            denoising_steps=self.config.num_denoising_steps,
            context=wm_context,
            context_noise_level=self.config.context_noise_level,
        )

        actor_critic_outs = self.actor_critic.stop_recording()
        shifted_actions = torch.cat([wm_context['action'][:, -1:], traj_segment['action'][-1][:, :-1]], dim=1)
        _, values, v_logits = self.actor_critic(prev_actions=shifted_actions, obs=traj_segment['observation'], compute_actor=False)
        actor_critic_outs.values = values
        actor_critic_outs.v_logits = v_logits
        
        # compute RL losses:
        N, B = len(traj_segment['log_pi']), traj_segment['log_pi'][0].shape[0]
        self._process_imagined_data(traj_segment, actor_critic_outs, first_step_outs)
        ends = traj_segment['terminated']
        t = traj_segment['denoising_times']
        valid_mask = make_valid_mask(ends, t)
        # valid_mask = rearrange(valid_mask, '(N B) ... -> N B ...', N=N, B=B)
        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=traj_segment['reward'],
                values=actor_critic_outs.values,
                ends=ends,
                gamma=self.config.gae_gamma,
                lambda_=self.config.gae_lambda,
            )[:, :-1]

        self.return_scaler.update(lambda_returns.float())
        returns_scale = torch.maximum(torch.ones_like(self.return_scaler.scale), self.return_scaler.scale * 0.5)

        values = actor_critic_outs.values[:, :-1]

        loss_actor, entropy, advantage = compute_actor_loss(
            self.config.actor_critic.use_clean_diffused_actors,
            traj_segment, lambda_returns, values, returns_scale, N, B, valid_mask, action_dist, actor_critic_outs
        )
        loss_entropy = - self.config.entropy_weight * entropy

        valid_mask = rearrange(valid_mask, '(N B) ... -> N B ...', N=N, B=B)
        value_logits = actor_critic_outs.v_logits[:, :-1]
        
        loss_critic = self.actor_critic.critic.training_step(
            value_logits[torch.where(valid_mask[-1])],
            lambda_returns[torch.where(valid_mask[-1])],
        )
        loss = loss_actor + loss_critic + loss_entropy

        values = values[torch.where(valid_mask[-1])]
        valid_lambda_returns = lambda_returns[torch.where(valid_mask[-1])]
        avg_num_action_changes, avg_action_change_time = self._collect_action_changes_stats(
            actions=traj_segment['action'],
            denoising_times=traj_segment['denoising_times']
        )
        name = 'actor_critic'
        info = {
            f"{name}/loss_actor": loss_actor.detach().clone(),
            f"{name}/loss_critic": loss_critic.detach().clone(),
            f"{name}/loss_entropy": loss_entropy.detach().clone(),
            f"{name}/avg_entropy": entropy.detach().clone(),
            f"{name}/returns_avg": valid_lambda_returns.detach().mean(),
            f"{name}/returns_max": valid_lambda_returns.detach().max(),
            f"{name}/returns_min": valid_lambda_returns.detach().min(),
            f"{name}/values_avg": values.detach().mean(),
            f"{name}/values_max": values.detach().max(),
            f"{name}/values_min": values.detach().min(),
            f"{name}/return_scale": returns_scale,
            f"{name}/normalized_advantage_avg": advantage.detach().mean(),
            f"{name}/normalized_advantage_max": advantage.detach().max(),
            f"{name}/normalized_advantage_min": advantage.detach().min(),
            f"{name}/imagined_rewards_avg": traj_segment['reward'].detach().mean(),
            f"{name}/imagined_rewards_max": traj_segment['reward'].detach().max(),
            f"{name}/imagined_rewards_min": traj_segment['reward'].detach().min(),
            f"{name}/num_ends": traj_segment['terminated'].detach().float().sum(dim=1).mean(),
            f"{name}/avg_num_action_changes": avg_num_action_changes,
            f"{name}/avg_action_change_time": avg_action_change_time,
        }
        if log_dict_fn is None:
            log_dict_fn = self.log_dict
        log_dict_fn(info, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    def _process_imagined_data(self, traj_segment, actor_outs: OutputsBuffer, first_step_outs):
        first_action_dist, first_action_log_p, first_value, first_v_logits = first_step_outs

        actor_outs.values = torch.cat([first_value, actor_outs.values], dim=1)
        actor_outs.v_logits = torch.cat([first_v_logits, actor_outs.v_logits], dim=1)

        traj_segment['log_pi'] = [
            torch.cat([first_action_log_p, log_pi_i], dim=1) 
            for log_pi_i in traj_segment['log_pi']
        ]
        if self.config.actor_critic.use_clean_diffused_actors:
            traj_segment['log_pi'] = torch.stack(traj_segment['log_pi'], dim=0)
        else:
            traj_segment['log_pi'] = torch.cat(traj_segment['log_pi'], dim=0)

        # Optimize only for the final rewards & terminations:
        traj_segment['reward'] = traj_segment['reward']

        done_probs = traj_segment['terminated']
        dones = torch.distributions.Categorical(probs=done_probs).sample()
        traj_segment['terminated'] = dones

    def _collect_action_changes_stats(self, actions, denoising_times):
        actions = torch.stack(actions, dim=0)
        denoising_times = torch.stack(denoising_times, dim=0)
        action_changes = (actions[1:] != actions[:-1])
        avg_num_action_changes = action_changes.sum(dim=0).float()
        avg_action_change_time = action_changes.float() * denoising_times[1:]
        avg_action_change_time = avg_action_change_time.sum(dim=0)
        avg_action_change_time = (
            avg_action_change_time[torch.where(avg_num_action_changes > 0)] / 
            avg_num_action_changes[torch.where(avg_num_action_changes > 0)]
        )

        return avg_num_action_changes.mean(), avg_action_change_time.mean()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optim.learning_rate,
            betas=self.config.optim.betas,
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
