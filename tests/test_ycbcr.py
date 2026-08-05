import torch
from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.ycbcr import YCbCrTensor
from horizon_imagination.utilities.obs_codec import (
    rgb_to_ycbcr_obs, ycbcr_to_rgb_obs, get_rgb_tensors, y_key, cbcr_key,
)
from horizon_imagination.utilities.types import ObsKey, Modality


RGB_KEY = ObsKey.from_parts(Modality.image, 'features')
Y_KEY = y_key(RGB_KEY)
CBCR_KEY = cbcr_key(RGB_KEY)


def _make_rgb(*shape):
    # 4:2:0 chroma subsampling relies on neighboring pixels being similar, as
    # in real images -- pure per-pixel noise has no such correlation and
    # would make the round trip look far lossier than it is in practice. So
    # build smooth, spatially-correlated content instead: a small random
    # image nearest-upsampled to full resolution.
    torch.manual_seed(0)
    low_res = torch.randint(0, 256, (*shape, 3, 2, 2), dtype=torch.uint8)
    return low_res.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)


def test_ycbcr_tensor_round_trip_is_close():
    rgb = _make_rgb(4)
    ycbcr = YCbCrTensor.from_rgb(rgb)
    rgb_hat = ycbcr.to_rgb().round().to(torch.uint8)

    # 4:2:0 chroma subsampling is lossy, so exact equality isn't expected --
    # but the round trip should stay close to the original.
    assert rgb_hat.shape == rgb.shape
    diff = (rgb_hat.float() - rgb.float()).abs()
    assert diff.mean() < 5.0
    assert diff.max() < 40.0


def test_ycbcr_tensor_chroma_is_subsampled():
    rgb = _make_rgb(2)
    ycbcr = YCbCrTensor.from_rgb(rgb, ratio=(2, 2))
    assert ycbcr._tensors['y'].shape[-2:] == (8, 8)
    assert ycbcr._tensors['cb'].shape[-2:] == (4, 4)
    assert ycbcr._tensors['cr'].shape[-2:] == (4, 4)


def test_obs_codec_round_trip_on_tensordict():
    rgb = _make_rgb(2, 3)
    obs = TensorDict({RGB_KEY: rgb}, batch_size=(2, 3))

    stored = rgb_to_ycbcr_obs(obs, drop_rgb=True)
    assert RGB_KEY not in stored.keys()
    assert Y_KEY in stored.keys()
    assert CBCR_KEY in stored.keys()
    assert stored[Y_KEY].shape[-3:] == (1, 8, 8)
    assert stored[CBCR_KEY].shape[-3:] == (2, 4, 4)

    reconstructed = ycbcr_to_rgb_obs(stored, drop_ycbcr=True)
    assert RGB_KEY in reconstructed.keys()
    assert Y_KEY not in reconstructed.keys()
    assert reconstructed[RGB_KEY].shape == rgb.shape

    diff = (reconstructed[RGB_KEY].float() - rgb.float()).abs()
    assert diff.mean() < 5.0


def test_get_rgb_tensors_is_noop_passthrough_when_already_rgb():
    rgb = _make_rgb(2)
    obs = TensorDict({RGB_KEY: rgb}, batch_size=(2,))
    assert torch.equal(get_rgb_tensors(obs)[RGB_KEY], rgb)
