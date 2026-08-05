"""Tests for the latent-reconstruction intrinsic reward.

The fast tests drive ``LatentReconstructionResidual`` with a fake transform so the
normalization behaviour can be checked exactly. The CUDA-gated tests exercise the real
``PerModalityTransform.inverse -> .transform`` round-trip, which has no other caller in the
codebase -- in particular the vector ``decode -> encode`` leg had never run end-to-end.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from tensordict.tensordict import TensorDict

from horizon_imagination.models.controller.intrinsic_reward import LatentReconstructionResidual
from horizon_imagination.modules.transform.base import BaseTransform


B, T = 6, 8


class _FakeTransform(BaseTransform):
    """Round-trips latents with a prescribed per-key distortion.

    ``inverse`` is the identity (the "raw" space is the latent space here) so that the
    residual seen by the class under test is exactly ``distortion ** 2``. Subclassing
    BaseTransform means these tests also cover its default ``roundtrip``.
    """

    def __init__(self, distortion: dict):
        self.distortion = distortion

    def inverse(self, z):
        return z

    def transform(self, x):
        return TensorDict(
            {k: v + self.distortion[k] for k, v in x.items()},
            batch_size=x.batch_size,
        )


def _latents(keys, shape=(4, 2, 2)):
    return TensorDict(
        {k: torch.zeros(B, T, *shape) for k in keys},
        batch_size=(B, T),
    )


def test_returns_per_step_residual_over_all_keys():
    keys = ['image|rgb', 'vector|proprio']
    z = _latents(keys)
    distortion = {k: torch.randn(B, T, 4, 2, 2) for k in keys}
    residual, info = LatentReconstructionResidual(_FakeTransform(distortion))(z)

    assert residual.shape == (B, T)
    assert torch.isfinite(residual).all()
    for k in keys:
        assert f'recon_mse/{k}' in info
        assert f'recon_normalized/{k}' in info


def test_raw_mse_matches_the_distortion():
    z = _latents(['image|rgb'])
    distortion = {'image|rgb': torch.full((B, T, 4, 2, 2), 0.5)}
    _, info = LatentReconstructionResidual(_FakeTransform(distortion))(z)

    assert info['recon_mse/image|rgb'] == pytest.approx(0.25, rel=1e-5)


def test_constant_reconstruction_error_contributes_nothing():
    """Centering means only *above-typical* error is rewarded.

    A residual that never varies is the tokenizer's noise floor, not novelty, so it must not
    turn into a constant per-step survival bonus.
    """
    z = _latents(['image|rgb'])
    distortion = {'image|rgb': torch.full((B, T, 4, 2, 2), 0.5)}
    residual, _ = LatentReconstructionResidual(_FakeTransform(distortion))(z)

    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)


def test_per_key_normalization_equalizes_wildly_different_scales():
    """Each key is normalized by its own band before averaging, so a key whose raw error is
    three orders of magnitude smaller still contributes comparably."""
    keys = ['image|rgb', 'vector|proprio']
    z = _latents(keys)
    g = torch.Generator().manual_seed(0)
    base = torch.randn(B, T, 4, 2, 2, generator=g)
    distortion = {'image|rgb': base * 1.0, 'vector|proprio': base * 0.001}

    residual_fn = LatentReconstructionResidual(_FakeTransform(distortion))
    residual_fn(z)
    _, info = residual_fn(z)

    big = info['recon_normalized/image|rgb']
    small = info['recon_normalized/vector|proprio']
    # Raw errors differ by ~1e6; normalized contributions must be within a small factor.
    assert info['recon_mse/image|rgb'] > 1e4 * info['recon_mse/vector|proprio']
    assert big == pytest.approx(float(small), rel=0.05)


def test_mask_excludes_steps_from_the_running_statistics():
    """Post-terminal steps spike the residual; letting them into the EMA would inflate the
    scale and shrink every advantage."""
    key = 'image|rgb'
    z = _latents([key])
    distortion = {key: torch.randn(B, T, 4, 2, 2)}
    distortion[key][:, -2:] *= 100.0  # simulate off-distribution post-terminal frames

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -2:] = False

    masked = LatentReconstructionResidual(_FakeTransform(distortion))
    masked(z, mask=mask)
    unmasked = LatentReconstructionResidual(_FakeTransform(distortion))
    unmasked(z)

    assert masked.scalers[key].scale < unmasked.scalers[key].scale


def test_all_off_trajectory_batch_yields_zero_without_crashing():
    """Reachable when the done head predicts termination at t=0 for every row."""
    z = _latents(['image|rgb'])
    distortion = {'image|rgb': torch.randn(B, T, 4, 2, 2)}
    mask = torch.zeros(B, T, dtype=torch.bool)

    residual, _ = LatentReconstructionResidual(_FakeTransform(distortion))(z, mask=mask)

    assert torch.equal(residual, torch.zeros_like(residual))


# --------------------------------------------------------------------------------------
# Scale-factor selection.
# --------------------------------------------------------------------------------------

class _ScalerStub:
    def __init__(self, scale):
        self.scale = torch.tensor(float(scale))


class _ConfigStub:
    def __init__(self, normalization, target_ratio=0.5, coeff=2.0):
        self.intrinsic_reward_normalization = normalization
        self.intrinsic_reward_target_ratio = target_ratio
        self.intrinsic_reward_coeff = coeff


def _scale_factor(normalization, extrinsic_scale, intrinsic_scale, **kw):
    from horizon_imagination.models.controller.controller import Controller

    ctl = Controller.__new__(Controller)  # bypass __init__; only the scalers are needed
    ctl.config = _ConfigStub(normalization, **kw)
    ctl.extrinsic_reward_scaler = _ScalerStub(extrinsic_scale)
    ctl.intrinsic_reward_scaler = _ScalerStub(intrinsic_scale)
    return float(Controller._intrinsic_scale_factor(ctl))


def test_quantile_ema_sizes_the_bonus_against_the_extrinsic_band():
    assert _scale_factor('quantile_ema', extrinsic_scale=0.08, intrinsic_scale=0.8) == \
        pytest.approx(0.5 * 0.08 / 0.8)


def test_degenerate_extrinsic_band_disables_the_bonus():
    """Rather than falling back to the raw, unnormalized residual -- which would inject an
    arbitrary magnitude exactly in the sparse-reward case."""
    assert _scale_factor('quantile_ema', extrinsic_scale=0.0, intrinsic_scale=0.8) == 0.0


def test_rnd_mode_ignores_the_extrinsic_scale():
    degenerate = _scale_factor('rnd_return_std', extrinsic_scale=0.0, intrinsic_scale=0.5)
    healthy = _scale_factor('rnd_return_std', extrinsic_scale=10.0, intrinsic_scale=0.5)
    assert degenerate == healthy == pytest.approx(2.0 / 0.5)


def test_running_return_std_scaler_tracks_discounted_return_spread():
    from horizon_imagination.models.controller.return_scaler import RunningReturnStdScaler

    scaler = RunningReturnStdScaler(gamma=0.99)
    scaler.update(torch.randn(16, 20) * 3.0)
    big = float(scaler.scale)

    scaler = RunningReturnStdScaler(gamma=0.99)
    scaler.update(torch.randn(16, 20) * 0.01)
    small = float(scaler.scale)

    assert big > 10 * small


# --------------------------------------------------------------------------------------
# The real tokenizer round-trip.
# --------------------------------------------------------------------------------------

RESOLUTION = 64
RAW_SIZE = 72


def _make_env():
    import gymnasium as gym
    from horizon_imagination.envs.wrappers import (
        ImageChannelsFirst, ResizeObsWrapper, ModalityDictObsWrapper,
    )

    class _SyntheticEnv(gym.Env):
        def __init__(self):
            image_space = gym.spaces.Box(0, 255, (RAW_SIZE, RAW_SIZE, 3), np.uint8)
            self.observation_space = gym.spaces.Dict({
                'rgb': image_space,
                'proprio': gym.spaces.Box(-1.0, 1.0, (7,), np.float32),
            })
            self.action_space = gym.spaces.Discrete(4)
            self.rng = np.random.default_rng(0)
            self.t = 0

        def _obs(self):
            return {
                'rgb': self.rng.integers(0, 255, (RAW_SIZE, RAW_SIZE, 3), dtype=np.uint8),
                'proprio': self.rng.uniform(-1, 1, (7,)).astype(np.float32),
            }

        def reset(self, **kwargs):
            self.t = 0
            return self._obs(), {}

        def step(self, action):
            self.t += 1
            return self._obs(), 1.0, self.t >= 12, False, {}

    env = _SyntheticEnv()
    env = ResizeObsWrapper(env, size=(RESOLUTION, RESOLUTION))
    env = ImageChannelsFirst(env)
    return ModalityDictObsWrapper(env)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_transform_roundtrip_over_image_and_vector_keys(tmp_path):
    from config.agent import get_agent_online_config

    cfg = get_agent_online_config(
        env=_make_env(), env_name='Synthetic/Test-v0',
        replay_buf_data_path=tmp_path / 'rb', resolution=RESOLUTION,
    )
    obs_transform = cfg.world_model.obs_transform
    assert obs_transform is not None

    spec = cfg.world_model.denoiser.obs_spec
    z = TensorDict(
        {str(k): torch.rand(2, 3, *shape, device='cuda') * 2 - 1 for k, shape in spec.items()},
        batch_size=(2, 3),
        device='cuda',
    )
    assert {'image|rgb', 'vector|proprio'} == set(z.keys())

    z_hat = obs_transform.transform(obs_transform.inverse(z))

    for k in z.keys():
        assert z_hat[k].shape == z[k].shape, k
        assert torch.isfinite(z_hat[k]).all(), k

    residual, info = LatentReconstructionResidual(obs_transform)(z)
    assert residual.shape == (2, 3)
    assert torch.isfinite(residual).all()
    # Random latents are off-manifold, so the raw error must be clearly non-zero.
    assert info['recon_mse/image|rgb'] > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("resolution", [64, 128])
def test_truncated_roundtrip_agrees_with_the_full_one(resolution):
    """The truncated cut is only valid at depths where the encoder and decoder widths
    coincide, which depends on channels_mult -- so check both shipped resolutions."""
    from config.tokenizer.image.cosmos import get_cosmos_tokenizer_online_config
    from horizon_imagination.modules.transform.image_to_latent import ImageToLatentTransform

    cfg = get_cosmos_tokenizer_online_config(dtype=None, resolution=resolution)
    tok = cfg.make_instance().to('cuda').eval()
    tr = ImageToLatentTransform(tok)

    latent_size = resolution // cfg.network_cfg.spatial_compression
    z = torch.rand(8, cfg.latent_channels, latent_size, latent_size, device='cuda') * 2 - 1

    full = tr.roundtrip(z)
    trunc = tr.roundtrip(z, truncated=True)

    assert trunc.shape == z.shape == full.shape
    assert trunc.dtype == z.dtype
    assert torch.isfinite(trunc).all()
    # Both are genuine round-trips of an off-manifold latent, so both must show real error.
    assert float((z - trunc).pow(2).mean()) > 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_truncated_roundtrip_preserves_a_batch_dim_of_any_rank():
    """The residual feeds it (B, T, C, H, W); the flatten/reshape must survive that."""
    from config.tokenizer.image.cosmos import get_cosmos_tokenizer_online_config
    from horizon_imagination.modules.transform.image_to_latent import ImageToLatentTransform

    cfg = get_cosmos_tokenizer_online_config(dtype=None, resolution=64)
    tr = ImageToLatentTransform(cfg.make_instance().to('cuda').eval())

    z = torch.rand(3, 5, 16, 8, 8, device='cuda') * 2 - 1
    assert tr.roundtrip(z, truncated=True).shape == z.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("normalization", ['quantile_ema', 'rnd_return_std'])
def test_controller_step_adds_the_bonus_and_logs_it(normalization, tmp_path):
    from config.agent import get_agent_online_config
    from horizon_imagination.agent import Agent
    from horizon_imagination.data import EpochDataIterator

    env = _make_env()
    cfg = get_agent_online_config(
        env=env, env_name='Synthetic/Test-v0',
        replay_buf_data_path=tmp_path / 'rb', resolution=RESOLUTION,
        intrinsic_reward_normalization=normalization,
        intrinsic_reward_target_ratio=0.5,
        # The truncated round-trip is the intended default cost profile; exercise it here.
        intrinsic_reward_truncated_roundtrip=(normalization == 'quantile_ema'),
    )
    cfg.controller.config.imagination_batch_size = 4
    cfg.controller.config.imagination_horizon = 4
    cfg.controller.config.num_denoising_steps = 2

    agent = Agent(cfg).to('cuda')
    assert agent.controller.use_intrinsic_reward

    agent.controller.forward(
        env=env, replay_buffer=cfg.replay_buffer, num_steps=40,
        log_dict_fn=lambda *a, **k: None, pbar_update_fn=None,
    )
    cfg.replay_buffer.flush()
    agent.train()

    iterator = EpochDataIterator(EpochDataIterator.Config(
        replay_buffer=cfg.replay_buffer,
        tokenizer_steps=0, world_model_steps=0, controller_steps=1,
        tokenizer_batch_size=4,
        wm_segment_length=6, wm_min_segment_length=2, wm_batch_size=2,
        c_segment_length=cfg.controller.config.controller_context_length,
        c_min_segment_length=1,
        c_batch_size=cfg.controller.config.imagination_batch_size,
    ))

    logged = {}
    for batch, component_idx in iterator:
        loss = agent.components[component_idx].training_step(
            batch, 0, lambda d, **k: logged.update(d)
        )
        assert np.isfinite(float(loss.detach()))

    assert 'actor_critic/intrinsic_reward_scale_factor' in logged
    assert 'actor_critic/recon_mse/image|rgb' in logged
    assert 'actor_critic/recon_mse/vector|proprio' in logged
    # imagined_rewards_* must keep meaning the extrinsic reward.
    assert 'actor_critic/imagined_rewards_avg' in logged
    assert 'actor_critic/total_rewards_avg' in logged

    # The bonus must actually be live -- not silently zeroed by the degenerate-band branch.
    scale_factor = float(logged['actor_critic/intrinsic_reward_scale_factor'])
    assert scale_factor > 0, "intrinsic bonus was disabled; the assertions below would be vacuous"
    assert float(logged['actor_critic/intrinsic_reward_avg']) > 0
    assert float(logged['actor_critic/total_rewards_avg']) != pytest.approx(
        float(logged['actor_critic/imagined_rewards_avg'])
    )
