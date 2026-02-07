from typing import Literal
import lightning as L
import torch
from einops import rearrange, repeat

import gymnasium as gym
from tensordict.tensordict import TensorDict
from loguru import logger
from tqdm import tqdm

from torchrl.data import ListStorage

from horizon_imagination.models.controller.actor_critic import ActorCritic, OutputsBuffer
from horizon_imagination.models.controller.return_scaler import EMAScaler
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.utilities import AdamWConfig, shift_fwd, RawMultiModalObs
from horizon_imagination.models.world_model import RectifiedFlowWorldModel
from horizon_imagination.models.world_model.action_producer import (
    StablePolicyActionProducer, NaivePolicyActionProducer
)
from horizon_imagination.modules.transform import PerModalityTransform
from horizon_imagination.data.replay_buffer import TensorDictReplayBuffer
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


def _infer_current_episode_id(replay_buffer: TensorDictReplayBuffer, episode_key: str = 'episode'):
    if len(replay_buffer) == 0:
        return 0
    
    last_step = replay_buffer[-1]
    last_step_ep = last_step[episode_key]
    last_step_done = torch.logical_or(last_step['terminated'], last_step['truncated']).cpu().item()
    if last_step_done:
        return last_step_ep + 1
    
    return last_step_ep.cpu().item()


def get_episode_suffix(
        replay_buffer: TensorDictReplayBuffer, 
        episode_id: int,
        suffix_length: int, 
        episode_key: str = 'episode'
    ):
    if len(replay_buffer.storage) == 0:
        return None
    episode = replay_buffer[torch.where(replay_buffer[:][episode_key] == episode_id)]
    return episode[-suffix_length:]


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

    @torch.no_grad()
    def forward(
            self, 
            env: gym.Env, 
            replay_buffer: TensorDictReplayBuffer, 
            num_steps: int, 
            log_dict_fn = None,
            pbar_update_fn = None
        ):
        """
        Inference - Collect data / act in a real env.
        """
        self.eval()
        if num_steps <= 0:
            return
        
        episode_id = _infer_current_episode_id(replay_buffer)
        if isinstance(episode_id, int):
            episode_id = torch.tensor(episode_id)

        def prepare_obs(x):
            x = x.to(device=self._device)
            # add batch dim and encode if necessary:
            return self.config.world_model.get_obs_from_batch({'observation': x[None, None]})

        # Set up the context:
        context = get_episode_suffix(replay_buffer, episode_id, self.config.controller_context_length)
        if context is None or len(context) == 0:
            """
            in this case, either no previous experience exists,
            or the last step was also a terminal step.
            In both cases, we need to start a new episode:
            """
            # New episode, get first obs and set it as context:
            if self._last_obs is None:
                # Otherwise, an env reset has already been performed.
                obs, info = env.reset()
                self._last_obs = _to_tensor_dict_obs(obs)
            context_obs = prepare_obs(self._last_obs)
            # Dummy action (before first obs), will be ignored:
            action_shape = self.action_space.shape
            if len(action_shape) == 0:
                action_shape = (1,)
            size = (1, *action_shape)
            context_actions = torch.zeros(*size, device=self._device).long()
            pad_mask = None
        else:
            """
            Last obs must exist (this must be a step in the middle of an episode). 
            Instead of shifting action forward, shift obs backwards and place the 
            last obs in front (last).
            """
            assert self._last_obs is not None, f"Got {self._last_obs}"
            context = context[None].to(device=self._device)  # add batch dim
            context_actions = context['action']
            context['observation'] = torch.cat(
                [context['observation'][:, 1:], self._last_obs[None, None].to(device=self._device)], 
                dim=1
            )
            context_obs = self.config.world_model.get_obs_from_batch(context)
            pad_mask = context['mask'] if 'mask' in context else None

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            action_dist, _, _ = self.actor_critic.reset(context_actions, context_obs, pad_mask)
        action = action_dist.sample()

        # Collect the data:
        for i in tqdm(range(num_steps), leave=False, desc="Data Collection"):
            action_raw = action.item() if action.numel() == 1 else action.cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_raw)
            rb_device = replay_buffer.storage.device
            step = TensorDict({
                'observation': TensorDict({str(k): v for k, v in self._last_obs.items()}, 
                                          batch_size=self._last_obs.batch_size, 
                                          device=rb_device),
                'action': action[0, 0].to(device=rb_device),
                'reward': torch.tensor(reward, device=rb_device),
                'terminated': torch.tensor(terminated, device=rb_device),
                'truncated': torch.tensor(truncated, device=rb_device),
                'episode': episode_id.clone().to(device=rb_device),
            })
            replay_buffer.add(step)

            self.stats_collector.step(None, action_raw, reward, terminated, truncated, info)

            self._last_obs = _to_tensor_dict_obs(obs)

            if terminated or truncated:
                # log the last obs for reward / termination prediction:
                # set dummy action / reward / etc
                step = TensorDict({
                    'observation': TensorDict({str(k): v for k, v in self._last_obs.items()}, 
                                            batch_size=self._last_obs.batch_size, 
                                            device=rb_device),
                    'action': torch.zeros_like(action[0, 0]).to(device=rb_device),
                    'reward': torch.tensor(0, device=rb_device),
                    'terminated': torch.tensor(terminated, device=rb_device),
                    'truncated': torch.tensor(truncated, device=rb_device),
                    'episode': episode_id.clone().to(device=rb_device),
                })
                replay_buffer.add(step)

                # reset the actor-critic state:
                self.actor_critic.reset()

                # reset the env and start a new episode:
                episode_id += 1
                obs, info = env.reset()
                self._last_obs = _to_tensor_dict_obs(obs)

            if i < num_steps - 1:
                # Compute the action for the next step as long as we're not
                # at the last iteration:
                obs = prepare_obs(self._last_obs)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    action_dist, _, _ = self.actor_critic.forward(action, obs, advance_state=True, compute_critic=False)
                action = action_dist.sample()
            
            if pbar_update_fn is not None:
                pbar_update_fn()
        
        if log_dict_fn is None:
            log_dict_fn = self.log_dict 
        self.stats_collector.log_epoch_stats(log_dict_fn)

    @torch.no_grad()
    def collect_test_episodes(self, num_episodes: int, env: gym.Env, collect_stats_only: bool = True):
        replay_buffer = TensorDictReplayBuffer(storage=ListStorage())
        self.actor_critic.reset()

        obs, info = env.reset()
        last_obs = _to_tensor_dict_obs(obs)
        action = torch.zeros(1, *self.action_space.sample().shape, device=self._device)
        episode_id = torch.tensor(0)
        episode_returns = []

        def prepare_obs(x):
            x = x.to(device=self._device)
            # add batch dim and encode if necessary:
            return self.config.world_model.get_obs_from_batch({'observation': x[None, None]})

        for i in tqdm(range(num_episodes), leave=False, desc="Test Episodes Collection"):
            done = False
            episode_return = 0
            while not done:
                # compute action:
                obs = prepare_obs(last_obs)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    action_dist, _, _ = self.actor_critic.forward(action, obs, advance_state=True, compute_critic=False)
                action = action_dist.sample()

                # perform env step:
                action_raw = action.item() if action.numel() == 1 else action.cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(action_raw)

                # store interaction:
                rb_device = replay_buffer.storage.device
                if not collect_stats_only:
                    step = TensorDict({
                        'observation': TensorDict({str(k): v for k, v in last_obs.items()}, 
                                                batch_size=last_obs.batch_size, 
                                                device=rb_device),
                        'action': action[0, 0].to(device=rb_device),
                        'reward': torch.tensor(reward, device=rb_device),
                        'terminated': torch.tensor(terminated, device=rb_device),
                        'truncated': torch.tensor(truncated, device=rb_device),
                        'episode': episode_id.clone().to(device=rb_device),
                    })
                    replay_buffer.add(step)
                episode_return += reward

                last_obs = _to_tensor_dict_obs(obs)

                if terminated or truncated:
                    # log the last obs for reward / termination prediction:
                    # set dummy action / reward / etc
                    if not collect_stats_only:
                        step = TensorDict({
                            'observation': TensorDict({str(k): v for k, v in last_obs.items()}, 
                                                    batch_size=last_obs.batch_size, 
                                                    device=rb_device),
                            'action': torch.zeros_like(action[0, 0]).to(device=rb_device),
                            'reward': torch.tensor(0, device=rb_device),
                            'terminated': torch.tensor(terminated, device=rb_device),
                            'truncated': torch.tensor(truncated, device=rb_device),
                            'episode': episode_id.clone().to(device=rb_device),
                        })
                        replay_buffer.add(step)

                    # reset the actor-critic state:
                    self.actor_critic.reset()

                    # reset the env and start a new episode:
                    episode_id += 1
                    obs, info = env.reset()
                    last_obs = _to_tensor_dict_obs(obs)

                    done = True
            episode_returns.append(episode_return)

        return replay_buffer, episode_returns

    def training_step(self, batch, batch_idx, log_dict_fn = None):
        # Assume suffix batch padding and prefix segment sampling.
        world_model = self.config.world_model

        # set actor critc context:
        context_actions = shift_fwd(batch['action'])
        context_obs = world_model.get_obs_from_batch(batch)
        pad_mask = batch['mask']
        action_dist, value, v_logits = self.actor_critic.reset(context_actions, context_obs, pad_mask)
        first_action = action_dist.sample()
        first_action_log_p = action_dist.log_prob(first_action)
        first_step_outs = (action_dist, first_action_log_p, value, v_logits)

        # set wm context:
        wm_context = batch[torch.arange(pad_mask.shape[0]), pad_mask.sum(1)-1][:, None]
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

        log_probs = traj_segment['log_pi'][:, :-1]
        advantage = (lambda_returns - values).detach() / returns_scale.to(dtype=values.dtype)
        advantage = repeat(advantage, "B ... -> (N B) ...", N=N, B=B)
        loss_actions = -(log_probs * advantage.detach())
        # loss_actions = rearrange(loss_actions, '(N B) ... -> N B ...', N=N, B=B)
        # loss_actions = loss_actions[-1][torch.where(valid_mask[-1])].mean()
        loss_actions = loss_actions[torch.where(valid_mask)].mean()

        loss_actor = loss_actions
        
        entropy = torch.cat([
            torch.cat([action_dist.entropy(), d.entropy()], dim=1) 
            for d in actor_critic_outs.actions_dist
        ], dim=0)[:, :-1]
        entropy = entropy[torch.where(valid_mask)]
        entropy = entropy.mean()
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
