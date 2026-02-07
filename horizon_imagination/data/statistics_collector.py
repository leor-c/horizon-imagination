import numpy as np

        
class StableSum:
    """
    Kahan algorithm for numerically stable sum.
    see "improved Kahan–Babuška algorithm" at 
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """
    def __init__(self):
        self._sum = 0
        self.c = 0

    def __call__(self, x: float):
        t = self._sum + x

        if abs(self._sum) >= abs(x):
            self.c += (self._sum - t) + x
        else:
            self.c += (x - t) + self._sum
        
        self._sum = t

        return self._sum + self.c
    
    @property
    def sum(self):
        return self._sum + self.c
    
    def __iadd__(self, other):
        # override `+=` for convenience
        self.__call__(other)
        return self


class ExperienceStatisticsCollector:
    def __init__(self):
        self.episode_return = StableSum()
        self._complete_episode_returns = []
        self._complete_episode_nnz_rewards = []
        self._complete_episode_lengths = []
        self.epoch_return = StableSum()
        self.max_epoch_reward = None
        self.min_epoch_reward = None
        self.num_epoch_steps = 0
        self.num_episode_steps = 0
        self.num_epoch_non_zero_rewards = 0
        self.num_episode_non_zero_rewards = 0

    def step(self, obs, action, reward, terminated, truncated, info):
        self.num_epoch_steps += 1
        self.num_episode_steps += 1

        self.epoch_return += reward
        self.episode_return += reward

        if (self.max_epoch_reward is None) or (self.max_epoch_reward < reward):
            self.max_epoch_reward = reward
        
        if (self.min_epoch_reward is None) or (self.min_epoch_reward > reward):
            self.min_epoch_reward = reward

        if abs(reward) > 1e-8:
            self.num_epoch_non_zero_rewards += 1
            self.num_episode_non_zero_rewards += 1

        if terminated or truncated:
            self._complete_episode_returns.append(self.episode_return.sum)
            self._complete_episode_nnz_rewards.append(self.num_episode_non_zero_rewards)
            self._complete_episode_lengths.append(self.num_episode_steps)
            self.episode_return = StableSum()
            self.num_episode_steps = 0
            self.num_episode_non_zero_rewards = 0

    def log_epoch_stats(self, log_dict_fn):
        prefix = 'experience_stats'
        stats = {
            f'{prefix}/epoch_return': self.epoch_return.sum,
            f'{prefix}/epoch_avg_reward': self.epoch_return.sum / self.num_epoch_steps,
            f'{prefix}/epoch_max_reward': self.max_epoch_reward,
            f'{prefix}/epoch_min_reward': self.min_epoch_reward,
            f'{prefix}/epoch_frac_non_zero_rewards': self.num_epoch_non_zero_rewards / self.num_epoch_steps,
            f'{prefix}/epoch_nnz_rewards': self.num_epoch_non_zero_rewards,
        }

        ep_avg_rewards = []
        ep_nnz_frac_nnz_rewards = []
        for num_steps, ep_return, nnz_rewards in zip(
            self._complete_episode_lengths,
            self._complete_episode_returns,
            self._complete_episode_nnz_rewards
        ):
            ep_avg_rewards.append(ep_return / num_steps)
            ep_nnz_frac_nnz_rewards.append(nnz_rewards / num_steps)
        
        if len(ep_avg_rewards) > 0:
            stats[f"{prefix}/avg_episode_return"] = np.mean(self._complete_episode_returns)
            stats[f"{prefix}/avg_episode_reward"] = np.mean(ep_avg_rewards)
            stats[f"{prefix}/avg_episode_frac_nnz_reward"] = np.mean(ep_nnz_frac_nnz_rewards)
            stats[f"{prefix}/avg_episode_nnz_reward"] = np.mean(self._complete_episode_nnz_rewards)
            stats[f"{prefix}/avg_episode_length"] = np.mean(self._complete_episode_lengths)

        # Log:
        log_dict_fn(stats)

        # Reset:
        self._complete_episode_returns.clear()
        self._complete_episode_nnz_rewards.clear()
        self._complete_episode_lengths.clear()
        self.epoch_return = StableSum()
        self.max_epoch_reward = None
        self.min_epoch_reward = None
        self.num_epoch_steps = 0
        self.num_epoch_non_zero_rewards = 0
