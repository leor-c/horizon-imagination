from typing import Literal, Optional
import lightning as L
import torch
from torch.distributions import Distribution, Categorical, kl_divergence
from einops import rearrange, repeat

import gymnasium as gym
from tensordict.tensordict import TensorDict
from loguru import logger
from tqdm import tqdm

import episodata as ed

from horizon_imagination.models.controller.actor_critic import ActorCritic, OutputsBuffer
from horizon_imagination.models.controller.return_scaler import EMAScaler
from horizon_imagination.modules.distributions import SquashedNormal, sample_with_log_prob
from horizon_imagination.models.controller.intrinsic_reward import LatentReconstructionResidual
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.utilities import AdamWConfig, shift_fwd, RawMultiModalObs, TensorDictRollingContextBuffer
from horizon_imagination.utilities.obs_codec import rgb_to_ycbcr_obs_np
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
    for i in reversed(range(t)):
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


def _to_tensor_dict_obs(obs: RawMultiModalObs):
    obs = TensorDict(
        {k: torch.from_numpy(v) for k, v in obs.items()},
    )
    return obs


def make_trajectory_mask(ends):
    """
    (B, T) bool mask that is True up to and including each row's first termination.

    Split out of ``make_valid_mask`` so the intrinsic reward can exclude post-terminal steps
    from its running statistics without also pulling in the denoising-time masking (which
    mutates its ``t`` argument and so cannot be called twice).
    """
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
    return range_matrix <= first_end_indices.unsqueeze(1)


def make_valid_mask(ends, t):
    mask = make_trajectory_mask(ends)

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


def _detach_dist(dist: Distribution) -> Distribution:
    """A copy of `dist` whose parameters carry no gradient."""
    if isinstance(dist, Categorical):
        return Categorical(logits=dist.logits.detach())
    if isinstance(dist, SquashedNormal):
        return SquashedNormal(dist.loc.detach(), dist.scale.detach())

    raise NotImplementedError(f"Cannot detach policy distribution {type(dist)}.")


def _slice_dist(dist: Distribution, index) -> Distribution:
    """Index a policy distribution along its batch dims, as if it were a tensor."""
    if isinstance(dist, Categorical):
        return Categorical(logits=dist.logits[index])
    if isinstance(dist, SquashedNormal):
        # `loc` is (..., A) and `index` addresses the batch dims only, so the action
        # axis survives and `SquashedNormal` re-infers its event shape from it.
        return SquashedNormal(dist.loc[index], dist.scale[index])

    raise NotImplementedError(f"Cannot index policy distribution {type(dist)}.")


def _cat_dists(dists: list[Distribution], dim: int) -> Distribution:
    """Concatenate policy distributions of the same family along a batch dim."""
    first = dists[0]
    if isinstance(first, Categorical):
        return Categorical(logits=torch.cat([d.logits for d in dists], dim=dim))
    if isinstance(first, SquashedNormal):
        return SquashedNormal(
            torch.cat([d.loc for d in dists], dim=dim),
            torch.cat([d.scale for d in dists], dim=dim),
        )

    raise NotImplementedError(f"Cannot concatenate policy distributions {type(first)}.")


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
    # One distribution per denoising step, over the whole (B, T) grid, with the
    # first (context) step prepended and the last (bootstrap) step dropped:
    step_dists = [
        _slice_dist(_cat_dists([action_dist, d], dim=1), (slice(None), slice(None, -1)))
        for d in actor_critic_outs.actions_dist
    ]

    clean_log_probs = traj_segment['log_pi'][-1, :, :-1]
    advantage = (lambda_returns - values).detach() / returns_scale.to(dtype=values.dtype)
    # advantage = repeat(advantage, "B ... -> (N B) ...", N=N, B=B)
    loss_actions = -(clean_log_probs * advantage.detach())
    # loss_actions = rearrange(loss_actions, '(N B) ... -> N B ...', N=N, B=B)
    # loss_actions = loss_actions[-1][torch.where(valid_mask[-1])].mean()
    valid_mask_blocks = rearrange(valid_mask, '(N B) ... -> N B ...', N=N, B=B)
    loss_actions = loss_actions[torch.where(valid_mask_blocks[-1])].mean()

    # Distill the clean actor into the noisy ones: KL(clean || noisy) per valid step,
    # with the clean distribution held fixed. Defined for any policy family, unlike the
    # cross-entropy-on-logits this replaces.
    #
    # NOTE this also fixes that older form. It read
    #   F.cross_entropy(noisy[valid].flatten(0, 1), softmax(clean[valid].flatten(0, 1)))
    # where `noisy[valid]` is already (num_valid_steps, num_actions), so `flatten(0, 1)`
    # collapsed it to a single 1-D vector and the softmax ran over every valid step and
    # action jointly -- one giant distribution over the batch, rather than one
    # distribution per step. For a Categorical the per-step KL below equals the intended
    # soft-target cross-entropy up to the constant H(clean), so `loss_actor` shifts by
    # that constant relative to previous runs.
    target = _detach_dist(step_dists[-1])
    loss_noisy_actions = torch.cat([
        kl_divergence(_slice_dist(target, mask_i), _slice_dist(noisy_i, mask_i))
        for noisy_i, mask_i in zip(
            step_dists[:-1], [torch.where(m) for m in valid_mask_blocks[:-1]]
        )
    ], dim=0).mean()

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
        # Weight on the latent reconstruction bonus (see
        # horizon_imagination.models.controller.intrinsic_reward), which arrives already
        # normalized per observation key. This is RND's fixed intrinsic coefficient
        # (Burda et al., 2018) applied to an already-normalized signal -- the per-key
        # normalizer plays the role RND's running return std plays there. Because the bonus
        # has a band of roughly 1, a weight of w produces a contribution comparable to an
        # extrinsic reward of magnitude w. 0.0 = disabled.
        intrinsic_reward_weight: float = 0.0
        # When True, imagination training optimizes only the scaled intrinsic reward --
        # extrinsic reward is excluded from the total reward entirely (pure exploration),
        # as in Burda et al., 2018, "Large-Scale Study of Curiosity-Driven Learning".
        pure_exploration: bool = False
        # Use the truncated latent round-trip (see ImageToLatentTransform.roundtrip): stops
        # short of pixel space, cutting the bonus's cost roughly in half and dropping the
        # uint8/chroma noise floor, at the price of measuring a weaker property.
        intrinsic_reward_truncated_roundtrip: bool = False

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self._device = config.actor_critic.backbone.device
        self.action_space = config.actor_critic.backbone.action_space

        self.actor_critic: ActorCritic = config.actor_critic.make_instance()
        self.return_scaler = EMAScaler(decay=config.return_scaler_decay)

        assert not (config.pure_exploration and config.intrinsic_reward_weight == 0.0), \
            "pure_exploration excludes the extrinsic reward, so a zero " \
            "intrinsic_reward_weight would train the actor on an all-zero reward."
        self.use_intrinsic_reward = config.intrinsic_reward_weight != 0.0
        self.intrinsic_residual = None
        if self.use_intrinsic_reward:
            assert config.world_model.obs_transform is not None, \
                "The intrinsic reward needs the world model's obs_transform to run the " \
                "encode(decode(z)) round-trip."
            self.intrinsic_residual = LatentReconstructionResidual(
                config.world_model.obs_transform,
                ema_decay=config.return_scaler_decay,
                truncated=config.intrinsic_reward_truncated_roundtrip,
            )

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

            # add experience to the replay buffer (stored as YCbCr, not RGB):
            writer = replay_buffer.new_episode(rgb_to_ycbcr_obs_np(obs), None)
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
            action = self._clip_action(action_dist.sample())

            # Step the environment:
            action_raw = self._action_for_env(action)
            obs, reward, terminated, truncated, info = env.step(action_raw)

            # Update the replay buffer (stored as YCbCr, not RGB):
            obs_stored = rgb_to_ycbcr_obs_np(obs)
            step = {
                'observation': {str(k): v for k, v in obs_stored.items()},
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

    def _clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Clip a sampled action into the action space (a no-op for discrete spaces).

        A safety net rather than the load-bearing bound: the policy is tanh-squashed,
        so it already emits `(-1, 1)`, and `RescaleActionWrapper` makes every `Box`
        exactly `[-1, 1]`. What is left for this to catch is `tanh` rounding to exactly
        +-1.0 in fp32, and an env reached without the rescaling wrapper.
        """
        if isinstance(self.action_space, gym.spaces.Discrete):
            return action

        assert isinstance(self.action_space, gym.spaces.Box), f"Got {self.action_space}"
        bounds = [
            torch.as_tensor(b, device=action.device, dtype=action.dtype)
            for b in (self.action_space.low, self.action_space.high)
        ]
        return action.clamp(*bounds)

    def _action_for_env(self, action: torch.Tensor):
        """Convert an action of shape (1, 1) / (1, 1, A) to what `env.step` expects."""
        if isinstance(self.action_space, gym.spaces.Discrete):
            return action.item()

        return action[0, 0].cpu().numpy().astype(self.action_space.dtype)

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
        first_action, first_action_log_p = sample_with_log_prob(action_dist)
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
        action_changes_stats = self._collect_action_changes_stats(
            actions=traj_segment['action'],
            denoising_times=traj_segment['denoising_times']
        )
        action_changes_stats.update(
            self._collect_policy_stats(actor_critic_outs, traj_segment, valid_mask[-1])
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
            # Deliberately the extrinsic reward, so these keys keep the meaning they had
            # before the intrinsic bonus existed and stay comparable across runs.
            f"{name}/imagined_rewards_avg": traj_segment['extrinsic_reward'].detach().mean(),
            f"{name}/imagined_rewards_max": traj_segment['extrinsic_reward'].detach().max(),
            f"{name}/imagined_rewards_min": traj_segment['extrinsic_reward'].detach().min(),
            f"{name}/total_rewards_avg": traj_segment['reward'].detach().mean(),
            f"{name}/num_ends": traj_segment['terminated'].detach().float().sum(dim=1).mean(),
        }
        info.update({f"{name}/{k}": v for k, v in action_changes_stats.items()})
        if self.use_intrinsic_reward:
            info.update({
                f"{name}/intrinsic_reward_avg": traj_segment['intrinsic_reward'].detach().mean(),
                f"{name}/intrinsic_reward_max": traj_segment['intrinsic_reward'].detach().max(),
                f"{name}/intrinsic_reward_min": traj_segment['intrinsic_reward'].detach().min(),
                f"{name}/intrinsic_reward_scaled_avg": traj_segment['intrinsic_reward_scaled'].detach().mean(),
                f"{name}/intrinsic_reward_scaled_max": traj_segment['intrinsic_reward_scaled'].detach().max(),
                f"{name}/intrinsic_reward_scaled_min": traj_segment['intrinsic_reward_scaled'].detach().min(),
            })
            # Per-key raw MSE / normalized means -- watch recon_mse's spread to confirm the
            # residual carries signal above the uint8+chroma round-trip noise floor.
            info.update({f"{name}/{k}": v for k, v in traj_segment['intrinsic_reward_info'].items()})
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
        done_probs = traj_segment['terminated']
        dones = torch.distributions.Categorical(probs=done_probs).sample()
        traj_segment['terminated'] = dones

        extrinsic_reward = traj_segment['reward']
        traj_segment['extrinsic_reward'] = extrinsic_reward
        if not self.use_intrinsic_reward:
            return

        # Post-termination the world model emits off-distribution frames, which is exactly
        # where the reconstruction residual spikes. Those steps are already dropped from the
        # losses, but letting them into the running statistics would inflate the scale and
        # shrink every advantage -- so mask them out of the estimators.
        on_trajectory = make_trajectory_mask(dones)
        intrinsic_reward, residual_info = self.intrinsic_residual(
            traj_segment['observation'], mask=on_trajectory
        )

        # The residual arrives normalized per key (band ~1), so all that is left is a fixed
        # coefficient -- RND's structure, with the per-key normalizer standing in for RND's
        # running return std. No second normalization stage: it would rescale an already
        # scale-stationary signal, and crucially it could not remove the residual's large
        # constant floor (measured at ~12x its own std on a trained tokenizer) -- only the
        # centering inside the per-key normalizer does that.
        scaled_intrinsic_reward = self.config.intrinsic_reward_weight * intrinsic_reward

        traj_segment['intrinsic_reward'] = intrinsic_reward
        traj_segment['intrinsic_reward_scaled'] = scaled_intrinsic_reward
        traj_segment['intrinsic_reward_info'] = residual_info
        if self.config.pure_exploration:
            traj_segment['reward'] = scaled_intrinsic_reward
        else:
            traj_segment['reward'] = extrinsic_reward + scaled_intrinsic_reward

    @torch.no_grad()
    def _collect_policy_stats(self, actor_critic_outs, traj_segment, clean_valid_mask) -> dict:
        """
        Where the squashed policy is actually sitting: its spread, how far the pre-tanh
        mean has travelled, and how much of the time tanh is saturated.

        These are the direct read-outs for the failure this head was rewritten to fix --
        a policy whose sigma ran away to its clamp, or whose mean ran far enough out that
        tanh pins every action to a corner of the box. `pre_tanh_mean_abs` is the one to
        watch: if it grows without bound the mean needs a bound of its own, which the
        squash alone does not provide.

        Read from the *clean* actor (the last recorded denoising step), which is the one
        whose distribution `clean_valid_mask` is shaped for.
        """
        dist = actor_critic_outs.actions_dist[-1]
        if not isinstance(dist, SquashedNormal):
            return {}

        valid = torch.where(clean_valid_mask)
        clean_actions = traj_segment['action'][-1]

        return {
            'policy_sigma_avg': dist.scale[valid].mean(),
            'policy_sigma_max': dist.scale[valid].max(),
            'pre_tanh_mean_abs': dist.loc[valid].abs().mean(),
            'tanh_saturation_frac': (clean_actions[valid].abs() > 0.99).float().mean(),
        }

    def _collect_action_changes_stats(self, actions, denoising_times) -> dict:
        """
        How much, and how late, the sampled action moves over the denoising process --
        the stability of the action producer.

        `actions` holds one entry per denoising step plus the final clean one, and the
        per-step move `actions[i+1] - actions[i]` is attributed to the time of the later
        action. For a discrete action the move is a change of index and its size is a
        count; for a continuous one it is the L2 distance travelled.

        Note this relies on `make_valid_mask` having already appended the clean step's
        time (all ones) to `denoising_times` in place -- which is why it must run first.
        """
        actions = torch.stack(actions, dim=0)
        change_times = torch.stack(denoising_times, dim=0)[1:]
        assert change_times.shape[0] == actions.shape[0] - 1, \
            f"Got {change_times.shape[0]} denoising times for {actions.shape[0]} actions."

        if isinstance(self.action_space, gym.spaces.Discrete):
            size_key = 'avg_num_action_changes'
            change_size = (actions[1:] != actions[:-1]).float()
        else:
            size_key = 'avg_action_drift'
            change_size = torch.linalg.vector_norm(actions[1:] - actions[:-1], dim=-1)

        total_change = change_size.sum(dim=0)
        moved = torch.where(total_change > 0)
        # Change-weighted mean denoising time, over the entries that moved at all:
        avg_change_time = (change_size * change_times).sum(dim=0)[moved] / total_change[moved]

        stats = {
            size_key: total_change.mean(),
            'avg_action_change_time': avg_change_time.mean(),
        }
        # `total_change` is a path *length* accumulated over every denoising step, so it
        # scales with the step budget and is not comparable across `--budget` settings.
        # The per-step figure is: for a squashed policy each step is bounded by the box
        # diameter `2 * sqrt(action_dim)`, and a settled policy should sit far below it.
        stats[f'{size_key}_per_step'] = total_change.mean() / change_size.shape[0]

        return stats
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optim.learning_rate,
            betas=self.config.optim.betas,
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )
