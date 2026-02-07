from typing import Tuple
import gymnasium as gym
import numpy as np
from PIL import Image

from horizon_imagination.utilities.types import ObsKey, Modality, MultiModalObs


def auto_detect_modality(obs_space: gym.spaces.Space) -> Modality:
    if np.issubdtype(obs_space.dtype, np.uint8) and len(obs_space.shape) in [2, 3]:
        return Modality.image
    elif np.issubdtype(obs_space.dtype, np.floating) and len(obs_space.shape) == 1:
        return Modality.vector
    elif np.issubdtype(obs_space.dtype, np.integer) and len(obs_space.shape) == 1:
        return Modality.token
    elif np.issubdtype(obs_space.dtype, np.integer) and len(obs_space.shape) == 2:
        return Modality.token_2d
    else:
        raise ValueError(f"Observation space '{obs_space}' is not supported or could not be detected automatically.")


class ModalityDictObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            is_dict_env = True
            obs_keys = [ObsKey.from_parts(auto_detect_modality(v), k) 
                        for k, v in env.observation_space.items()]
            
        else:
            is_dict_env = False
            obs_keys = [ObsKey.from_parts(auto_detect_modality(env.observation_space), name='features')]
        self.is_dict_env = is_dict_env
        self.obs_keys = obs_keys
        self._modalities = set([k.modality for k in obs_keys])
        self.observation_space = self._make_obs_space(env.observation_space)

    def _make_obs_space(self, orig_space: gym.Space):
        if self.is_dict_env:
            assert isinstance(orig_space, gym.spaces.Dict)
            return gym.spaces.Dict({k: orig_space.spaces[k.name] for k in self.obs_keys})
        else:
            return gym.spaces.Dict({self.obs_keys[0]: orig_space})

    @property
    def modalities(self) -> set[Modality]:
        return self._modalities

    def observation(self, observation) -> MultiModalObs:
        if self.is_dict_env:
            assert isinstance(observation, dict), f"Expected a dict observation, got {observation} instead."
            return {k: observation[k.name] for k in self.obs_keys}
        
        return {self.obs_keys[0]: observation}


class ImageChannelsFirst(gym.ObservationWrapper):
    def observation(self, observation):
        assert isinstance(observation, np.ndarray)
        assert observation.shape[-1] == 3, f"Got shape {observation.shape}"
        return np.transpose(observation, (2, 0, 1))


class FrameSkip(gym.Wrapper):
    """Return only every `skip`-th frame"""
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0

        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break
        
        return obs, total_reward, terminated, truncated, info
    

class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        # assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0

        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        # lives = self.env.unwrapped.ale.lives()
        lives = int(info['lives'])
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises terminated.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        # self.lives = self.env.unwrapped.ale.lives()
        self.lives = int(info['lives'])
        return obs, info
    

class SignRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)
