import numpy as np
import torch
from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.obs_codec import (
    rgb_to_ycbcr_obs, ycbcr_to_rgb_obs, rgb_to_ycbcr_obs_np, get_rgb_tensors,
    y_key, cbcr_key, rgb_keys,
)
from horizon_imagination.utilities.types import ObsKey, Modality


RGB_A = ObsKey.from_parts(Modality.image, 'features')
RGB_B = ObsKey.from_parts(Modality.image, 'wrist')
VEC = ObsKey.from_parts(Modality.vector, 'proprio')

H = W = 8


def make_image(batch, color, seed):
    """A near-flat image. 4:2:0 chroma subsampling averages 2x2 chroma blocks, so
    only spatially smooth images survive it -- fidelity itself is covered by
    tests/test_ycbcr.py; here the images just have to be distinguishable per key."""
    rng = np.random.default_rng(seed)
    planes = np.broadcast_to(np.array(color, dtype=np.int64)[:, None, None], (*batch, 3, H, W))
    return torch.from_numpy((planes + rng.integers(0, 8, planes.shape)).clip(0, 255).astype(np.uint8))


def make_obs(batch=(2,)):
    return {
        RGB_A: make_image(batch, color=(200, 30, 90), seed=0),
        RGB_B: make_image(batch, color=(10, 180, 240), seed=1),
        VEC: torch.randn(*batch, 5),
    }


def test_every_image_key_gets_its_own_components_and_vectors_pass_through():
    obs = make_obs()
    encoded = rgb_to_ycbcr_obs(obs)

    assert set(encoded) == {y_key(RGB_A), cbcr_key(RGB_A), y_key(RGB_B), cbcr_key(RGB_B), VEC}
    assert encoded[y_key(RGB_A)].shape == (2, 1, H, W)
    assert encoded[cbcr_key(RGB_A)].shape == (2, 2, H // 2, W // 2)
    assert torch.equal(encoded[VEC], obs[VEC])

    decoded = ycbcr_to_rgb_obs(encoded, drop_ycbcr=True)
    assert set(decoded) == {RGB_A, RGB_B, VEC}
    for key in (RGB_A, RGB_B):
        assert decoded[key].shape == obs[key].shape
        # 4:2:0 chroma subsampling is lossy, but stays close on smooth images:
        error = (decoded[key].float() - obs[key].float()).abs().mean()
        assert error < 5, error


def test_keys_do_not_leak_into_each_other():
    obs = make_obs()
    roundtrip = ycbcr_to_rgb_obs(rgb_to_ycbcr_obs(obs), drop_ycbcr=True)
    # A different image in key B must not change what key A decodes to:
    other = dict(obs)
    other[RGB_B] = torch.zeros_like(other[RGB_B])
    other_roundtrip = ycbcr_to_rgb_obs(rgb_to_ycbcr_obs(other), drop_ycbcr=True)
    assert torch.equal(roundtrip[RGB_A], other_roundtrip[RGB_A])


def test_tensordict_type_and_batch_size_are_preserved():
    obs = TensorDict(make_obs(), batch_size=(2,))
    encoded = rgb_to_ycbcr_obs(obs)
    assert isinstance(encoded, TensorDict) and encoded.batch_size == obs.batch_size


def test_get_rgb_tensors_returns_one_entry_per_image_key():
    encoded = rgb_to_ycbcr_obs(make_obs())
    tensors = get_rgb_tensors(encoded)
    assert set(tensors) == {RGB_A, RGB_B}
    assert all(t.shape == (2, 3, H, W) for t in tensors.values())


def test_numpy_variant_matches_the_tensor_variant():
    obs = make_obs(batch=())
    np_encoded = rgb_to_ycbcr_obs_np({k: v.numpy() for k, v in obs.items()})
    encoded = rgb_to_ycbcr_obs(obs)

    assert set(np_encoded) == set(encoded)
    for key in encoded:
        assert np.array_equal(np_encoded[key], encoded[key].numpy())


def test_component_keys_are_not_mistaken_for_rgb_keys():
    encoded = rgb_to_ycbcr_obs(make_obs())
    assert rgb_keys(encoded.keys()) == []
