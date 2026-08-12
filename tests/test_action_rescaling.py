"""Unit tests for `RescaleActionWrapper`.

The tanh-squashed policy speaks `[-1, 1]`; this wrapper is what makes an environment
whose native `Box` is something else agree with it, rather than being driven at the
wrong scale with no error.
"""
import gymnasium as gym
import numpy as np
import pytest

from horizon_imagination.envs.wrappers import (
    RescaleActionWrapper, ModalityDictObsWrapper, Float32ObsWrapper,
)


class _FakeEnv(gym.Env):
    """Records the action it was stepped with, so the mapping can be read back."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float64)
        self.last_action = None

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float64), {}

    def step(self, action):
        self.last_action = action
        return np.zeros(4, dtype=np.float64), 0.0, False, False, {}


def _box(low, high, n=3, dtype=np.float32):
    return gym.spaces.Box(low=low, high=high, shape=(n,), dtype=dtype)


class TestRescaling:
    def test_exposes_a_unit_box(self):
        env = RescaleActionWrapper(_FakeEnv(_box(-0.4, 0.4)))  # Humanoid-style

        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
        assert env.action_space.shape == (3,)

    @pytest.mark.parametrize("low,high", [(-0.4, 0.4), (-2.0, 2.0), (0.0, 1.0)])
    def test_endpoints_and_midpoint_map_correctly(self, low, high):
        inner = _FakeEnv(_box(low, high))
        env = RescaleActionWrapper(inner)

        for agent_action, expected in [
            (-1.0, low), (1.0, high), (0.0, (low + high) / 2),
        ]:
            env.step(np.full(3, agent_action, dtype=np.float32))
            assert np.allclose(inner.last_action, expected)

    def test_already_unit_box_is_an_identity(self):
        inner = _FakeEnv(_box(-1.0, 1.0))
        env = RescaleActionWrapper(inner)

        action = np.array([-0.7, 0.0, 0.35], dtype=np.float32)
        env.step(action)

        assert np.allclose(inner.last_action, action)

    def test_out_of_range_actions_are_clipped(self):
        """fp32 `tanh` reaches exactly +-1.0, and nothing above should reach the env."""
        inner = _FakeEnv(_box(-2.0, 2.0))
        env = RescaleActionWrapper(inner)

        env.step(np.array([-1.5, 1.5, 0.0], dtype=np.float32))
        assert np.allclose(inner.last_action, [-2.0, 2.0, 0.0])

    def test_dtype_is_preserved(self):
        inner = _FakeEnv(_box(-2.0, 2.0, dtype=np.float64))
        env = RescaleActionWrapper(inner)

        env.step(np.zeros(3, dtype=np.float32))
        assert inner.last_action.dtype == np.float64


class TestPassthroughAndComposition:
    def test_discrete_space_passes_through(self):
        """Applied unconditionally in `make_env`, so it must not touch discrete envs."""
        inner = _FakeEnv(gym.spaces.Discrete(5))
        env = RescaleActionWrapper(inner)

        assert env.action_space == gym.spaces.Discrete(5)
        env.step(3)
        assert inner.last_action == 3

    def test_unbounded_box_is_rejected(self):
        with pytest.raises(AssertionError, match="unbounded Box"):
            RescaleActionWrapper(_FakeEnv(_box(-np.inf, np.inf)))

    def test_unit_box_survives_the_observation_wrappers(self):
        """The action space the agent config reads is the outermost one."""
        env = ModalityDictObsWrapper(
            Float32ObsWrapper(RescaleActionWrapper(_FakeEnv(_box(-0.4, 0.4))))
        )

        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
