"""End-to-end check of the agent on dict observations.

Runs the full loop -- collection, one training step of every component (the
controller's step imagines through the world model), and validation-video
generation -- on a synthetic env, for both a single image observation (the
pre-existing setup, i.e. a no-regression check) and a multi-key one.
"""
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from horizon_imagination.envs.wrappers import (
    ImageChannelsFirst, ResizeObsWrapper, ModalityDictObsWrapper,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

RESOLUTION = 64
RAW_SIZE = 72  # exercises ResizeObsWrapper too


class _SyntheticEnv(gym.Env):
    """An env whose observation is a Box, or a Dict of images and vectors."""

    def __init__(self, dict_obs: bool, action_space: gym.Space = None):
        self.dict_obs = dict_obs
        image_space = gym.spaces.Box(0, 255, (RAW_SIZE, RAW_SIZE, 3), np.uint8)
        self.observation_space = gym.spaces.Dict({
            'rgb': image_space,
            'aux': image_space,
            'proprio': gym.spaces.Box(-1.0, 1.0, (7,), np.float32),
        }) if dict_obs else image_space
        self.action_space = action_space or gym.spaces.Discrete(4)
        self.rng = np.random.default_rng(0)
        self.t = 0

    def _image(self):
        return self.rng.integers(0, 255, (RAW_SIZE, RAW_SIZE, 3), dtype=np.uint8)

    def _obs(self):
        if not self.dict_obs:
            return self._image()
        return {
            'rgb': self._image(),
            'aux': self._image(),
            'proprio': self.rng.uniform(-1, 1, (7,)).astype(np.float32),
        }

    def reset(self, **kwargs):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), \
            f"{action} is outside {self.action_space} -- continuous actions must be clipped."
        self.t += 1
        return self._obs(), 1.0, self.t >= 12, False, {}


def _make_env(dict_obs: bool, action_space: gym.Space = None):
    env = _SyntheticEnv(dict_obs, action_space)
    env = ResizeObsWrapper(env, size=(RESOLUTION, RESOLUTION))
    env = ImageChannelsFirst(env)
    return ModalityDictObsWrapper(env)


@pytest.mark.parametrize("dict_obs,expected_keys,action_space", [
    (False, {'image|features'}, None),
    (True, {'image|rgb', 'image|aux', 'vector|proprio'}, None),
    # Continuous actions: a Gaussian policy, a linear action embedder in the denoiser
    # and the stable continuous producer driving imagination.
    (True, {'image|rgb', 'image|aux', 'vector|proprio'},
     gym.spaces.Box(-1.0, 1.0, (3,), np.float32)),
], ids=['single_image', 'dict_obs', 'dict_obs_box_actions'])
def test_agent_trains_every_component(dict_obs, expected_keys, action_space, tmp_path):
    from config.agent import get_agent_online_config
    from horizon_imagination.agent import Agent
    from horizon_imagination.data import EpochDataIterator

    env = _make_env(dict_obs, action_space)
    assert set(env.observation_space.spaces) == expected_keys

    cfg = get_agent_online_config(
        env=env, env_name='Synthetic/Test-v0',
        replay_buf_data_path=tmp_path / 'rb', resolution=RESOLUTION,
    )
    # Keep the imagination small enough for a test:
    cfg.controller.config.imagination_batch_size = 4
    cfg.controller.config.imagination_horizon = 4
    cfg.controller.config.num_denoising_steps = 2

    denoiser = cfg.world_model.denoiser
    assert set(map(str, denoiser.obs_spec)) == expected_keys
    # Image keys contribute their patch grid, vector keys one token each here:
    assert sum(denoiser.num_tokens.values()) == int(denoiser.token_positions.shape[0])

    agent = Agent(cfg).to('cuda')

    agent.controller.forward(
        env=env, replay_buffer=cfg.replay_buffer, num_steps=40,
        log_dict_fn=lambda *a, **k: None, pbar_update_fn=None,
    )
    cfg.replay_buffer.flush()
    # Encoding observations during collection leaves the tokenizer in eval mode;
    # Agent.on_train_epoch_start likewise re-enables training after collecting.
    agent.train()

    iterator = EpochDataIterator(EpochDataIterator.Config(
        replay_buffer=cfg.replay_buffer,
        tokenizer_steps=1, world_model_steps=1, controller_steps=1,
        tokenizer_batch_size=4,
        wm_segment_length=6, wm_min_segment_length=2, wm_batch_size=2,
        c_segment_length=cfg.controller.config.controller_context_length,
        c_min_segment_length=1,
        c_batch_size=cfg.controller.config.imagination_batch_size,
    ))

    losses = {}
    for batch, component_idx in iterator:
        loss = agent.components[component_idx].training_step(batch, 0, lambda *a, **k: None)
        losses[component_idx] = float(loss.detach())

    assert set(losses) == {0, 1, 2}
    assert all(np.isfinite(v) for v in losses.values()), losses
