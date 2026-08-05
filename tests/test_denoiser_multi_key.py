from collections import OrderedDict

import gymnasium as gym
import pytest
import torch
from tensordict.tensordict import TensorDict

from horizon_imagination.models.world_model.denoiser import (
    VideoDiTDenoiser, build_token_position_table,
)
from horizon_imagination.models.world_model.dit import DiT
from horizon_imagination.utilities.types import ObsKey, Modality


IMG_A = ObsKey.from_parts(Modality.image, 'features')
IMG_B = ObsKey.from_parts(Modality.image, 'wrist')
VEC_A = ObsKey.from_parts(Modality.vector, 'proprio')
VEC_B = ObsKey.from_parts(Modality.vector, 'chunked')

CHANNELS, LATENT_H, LATENT_W, PATCH, LATENT_DIM = 6, 8, 8, 2, 16
MODEL_CHANNELS = 64

MULTI_KEY_SPEC = OrderedDict([
    (IMG_A, (CHANNELS, LATENT_H, LATENT_W)),
    (IMG_B, (CHANNELS, LATENT_H, LATENT_W)),
    (VEC_A, (LATENT_DIM, 1, 1)),
    (VEC_B, (LATENT_DIM, 3, 1)),  # a vector key spanning several tokens
])


def make_denoiser(obs_spec, max_img_h=20, max_img_w=8):
    return VideoDiTDenoiser.Config(
        dit_cfg=DiT.Config(
            max_img_h=max_img_h, max_img_w=max_img_w, expected_max_frames=32,
            in_channels=CHANNELS, out_channels=CHANNELS,
            patch_spatial=PATCH, patch_temporal=1,
            model_channels=MODEL_CHANNELS, num_blocks=2, num_heads=4,
            device=torch.device('cpu'), ln_eps=1e-5,
        ),
        action_space=gym.spaces.Discrete(5),
        spatial_patch_size=PATCH,
        img_latent_channels=CHANNELS,
        obs_spec=obs_spec,
        vector_latent_dim=LATENT_DIM,
    ).make_instance().eval()


def make_obs(obs_spec, b, t):
    return TensorDict(
        {key: torch.randn(b, t, *shape) for key, shape in obs_spec.items()}, batch_size=(b, t)
    )


def test_a_single_image_key_keeps_the_row_major_meshgrid():
    denoiser = make_denoiser(OrderedDict([(IMG_A, (CHANNELS, LATENT_H, LATENT_W))]), max_img_h=8)
    rows = cols = LATENT_H // PATCH
    i, j = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
    expected = torch.stack([i.reshape(-1), j.reshape(-1)], dim=-1)

    assert torch.equal(denoiser.token_positions, expected)


def test_keys_get_disjoint_position_bands():
    denoiser = make_denoiser(MULTI_KEY_SPEC)

    assert dict(denoiser.num_tokens) == {IMG_A: 16, IMG_B: 16, VEC_A: 1, VEC_B: 3}
    positions = [tuple(p) for p in denoiser.token_positions.tolist()]
    assert len(set(positions)) == len(positions), "token positions collide"

    # Canonical order: images (by name) then vectors (by name).
    assert denoiser.obs_keys == [IMG_A, IMG_B, VEC_B, VEC_A]


def test_denoise_is_shape_preserving_across_modalities():
    torch.manual_seed(0)
    denoiser = make_denoiser(MULTI_KEY_SPEC)
    x = make_obs(MULTI_KEY_SPEC, b=2, t=4)

    with torch.no_grad():
        out, _ = denoiser.denoise(x, torch.rand(2, 4), actions=torch.randint(0, 5, (2, 4)))

    for key, shape in MULTI_KEY_SPEC.items():
        assert out[key].shape == x[key].shape
        assert torch.isfinite(out[key]).all()


def test_kv_cache_continuation_matches_the_full_forward():
    torch.manual_seed(0)
    denoiser = make_denoiser(MULTI_KEY_SPEC)
    b, t = 2, 4
    x = make_obs(MULTI_KEY_SPEC, b, t)
    time = torch.rand(b, t)
    actions = torch.randint(0, 5, (b, t))

    with torch.no_grad():
        full, _ = denoiser.denoise(x, time, actions=actions)
        _, cache = denoiser.denoise(x[:, :2], time[:, :2], actions=actions[:, :2])
        continued, _ = denoiser.denoise(x[:, 2:], time[:, 2:], actions=actions[:, 2:], kv_cache=cache)

    for key in MULTI_KEY_SPEC:
        assert torch.allclose(full[key][:, 2:], continued[key], atol=1e-5)


def test_denoise_rejects_unexpected_observation_keys():
    denoiser = make_denoiser(MULTI_KEY_SPEC)
    x = make_obs(MULTI_KEY_SPEC, b=1, t=2)
    del x[IMG_B]

    with pytest.raises(AssertionError, match="expected"):
        denoiser.denoise(x, torch.rand(1, 2), actions=torch.zeros(1, 2, dtype=torch.long))


def test_position_grid_capacity_is_checked():
    with pytest.raises(AssertionError, match="position grid"):
        make_denoiser(MULTI_KEY_SPEC, max_img_h=8, max_img_w=8)


def test_build_token_position_table_stacks_bands_vertically():
    positions, num_tokens = build_token_position_table(
        OrderedDict([(IMG_A, (2, 2)), (VEC_A, (1, 3))])
    )
    assert num_tokens == {IMG_A: 4, VEC_A: 3}
    assert positions.tolist() == [
        [0, 0], [0, 1], [1, 0], [1, 1],  # the image band
        [2, 0], [2, 1], [2, 2],          # the vector row, below it
    ]
