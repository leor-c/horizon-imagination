from typing import Tuple
import gymnasium as gym
import numpy as np
from PIL import Image
from gymnasium import Env
from gymnasium.core import ObsType, ActType

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


class RescaleActionWrapper(gym.ActionWrapper):
    """
    Present every ``Box`` action space as ``[-1, 1]``, mapping back affinely on the way in.

    The tanh-squashed policy emits ``(-1, 1)``, so the agent, the replay buffer and the
    world model all speak that scale; this is what makes them agree with an environment
    whose native bounds are something else. A no-op for the MuJoCo tasks that are
    already ``[-1, 1]`` (HalfCheetah, Hopper, Walker2d, Ant), but load-bearing for
    Humanoid (``[-0.4, 0.4]``) and Pusher (``[-2, 2]``), which would otherwise be driven
    at the wrong scale with no error.

    Passes non-``Box`` spaces straight through, so it can be applied unconditionally --
    unlike ``gymnasium.wrappers.RescaleAction``, which asserts on the space in its
    constructor and re-asserts the incoming bounds on every step (fp32 ``tanh`` reaching
    exactly +-1.0 is enough to trip it).
    """

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        # `action_space` may be a property doing an RPC round-trip (portal-env), so read
        # it once here rather than per step.
        inner = env.action_space
        self._is_box = isinstance(inner, gym.spaces.Box)
        if not self._is_box:
            return

        self._low, self._high = np.asarray(inner.low), np.asarray(inner.high)
        assert np.all(np.isfinite(self._low)) and np.all(np.isfinite(self._high)), \
            f"An unbounded Box cannot be rescaled affinely; got {inner}."
        self._dtype = inner.dtype
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=inner.shape, dtype=inner.dtype
        )

    def action(self, action):
        if not self._is_box:
            return action

        action = np.clip(action, -1.0, 1.0)
        rescaled = self._low + (action + 1.0) * 0.5 * (self._high - self._low)
        return rescaled.astype(self._dtype)


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


class Float32ObsWrapper(gym.ObservationWrapper):
    """
    Downcast float64 observations (and their spaces) to float32.

    MuJoCo reports its state observations as float64, which would be stored at twice
    the size in the replay buffer for no benefit: the vector autoencoder normalizes
    through ``.float()`` anyway. Runs *before* ``ModalityDictObsWrapper``, so it sees
    either a single Box observation or a Dict of them; non-float64 entries (images,
    already-float32 vectors) pass through untouched.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        space = env.observation_space
        self.is_dict_env = isinstance(space, gym.spaces.Dict)
        if self.is_dict_env:
            self.observation_space = gym.spaces.Dict(
                {k: self._cast_space(v) for k, v in space.spaces.items()}
            )
        else:
            self.observation_space = self._cast_space(space)

    @staticmethod
    def _needs_cast(space: gym.Space) -> bool:
        return isinstance(space, gym.spaces.Box) and space.dtype == np.float64

    @classmethod
    def _cast_space(cls, space: gym.Space) -> gym.Space:
        if not cls._needs_cast(space):
            return space
        return gym.spaces.Box(
            low=space.low.astype(np.float32),
            high=space.high.astype(np.float32),
            shape=space.shape,
            dtype=np.float32,
        )

    @staticmethod
    def _cast(value: np.ndarray) -> np.ndarray:
        return value.astype(np.float32) if value.dtype == np.float64 else value

    def observation(self, observation):
        if self.is_dict_env:
            return {k: self._cast(v) for k, v in observation.items()}
        return self._cast(observation)


def _is_image_space(space: gym.Space) -> bool:
    return isinstance(space, gym.spaces.Box) and np.issubdtype(space.dtype, np.uint8) \
        and len(space.shape) == 3


class _PerImageKeyObservationWrapper(gym.ObservationWrapper):
    """
    Base for the raw-image preprocessing wrappers, which run *before*
    ``ModalityDictObsWrapper`` and therefore see either a single Box observation or a
    Dict of them. Subclasses only implement the single-image transform; this class
    applies it to every image entry of a Dict observation and leaves the rest alone.
    """

    def __init__(self, env: Env[ObsType, ActType]):
        super().__init__(env)

        space = env.observation_space
        self.is_dict_env = isinstance(space, gym.spaces.Dict)
        if self.is_dict_env:
            self.image_keys = [k for k, v in space.spaces.items() if _is_image_space(v)]
            assert self.image_keys, f"No image entries in observation space {space}."
            self.observation_space = gym.spaces.Dict({
                k: self.transform_space(v) if k in self.image_keys else v
                for k, v in space.spaces.items()
            })
        else:
            assert _is_image_space(space), f"Expected an image observation space, got {space}."
            self.image_keys = None
            self.observation_space = self.transform_space(space)

    def transform_space(self, space: gym.spaces.Box) -> gym.spaces.Box:
        raise NotImplementedError

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def observation(self, observation):
        if self.is_dict_env:
            return {
                k: self.transform_image(v) if k in self.image_keys else v
                for k, v in observation.items()
            }
        return self.transform_image(observation)


class ImageChannelsFirst(_PerImageKeyObservationWrapper):

    def transform_space(self, space: gym.spaces.Box) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=np.transpose(space.low, (2, 0, 1)),
            high=np.transpose(space.high, (2, 0, 1)),
            dtype=space.dtype,
        )

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        assert isinstance(image, np.ndarray)
        assert image.shape[-1] == 3, f"Got shape {image.shape}"
        return np.transpose(image, (2, 0, 1))


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
    

class ResizeObsWrapper(_PerImageKeyObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]) -> None:
        self.size = tuple(size)
        super().__init__(env)
        self.unwrapped.original_obs = None

    def transform_space(self, space: gym.spaces.Box) -> gym.spaces.Box:
        return gym.spaces.Box(low=0, high=255, shape=(self.size[0], self.size[1], 3), dtype=np.uint8)

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def transform_image(self, image: np.ndarray) -> np.ndarray:
        return self.resize(image)

    def observation(self, observation):
        self.unwrapped.original_obs = observation
        return super().observation(observation)


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
