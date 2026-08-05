from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.types import ObsKey, Modality
from horizon_imagination.utilities.ycbcr import YCbCrTensor


RGB_KEY = ObsKey.from_parts(Modality.image, "features")
Y_KEY = ObsKey.from_parts(Modality.image, "features_y")
CBCR_KEY = ObsKey.from_parts(Modality.image, "features_cbcr")


def _to_mapping(obs: Mapping | TensorDict) -> dict:
    return {k: v for k, v in obs.items()}


def _restore_type(obs: Mapping | TensorDict, data: dict):
    if isinstance(obs, TensorDict):
        return TensorDict(data, batch_size=obs.batch_size, device=obs.device)
    return data


def rgb_to_ycbcr_obs(
    obs: Mapping | TensorDict,
    ratio: tuple[int, int] = (2, 2),
    drop_rgb: bool = True,
):
    data = _to_mapping(obs)
    if RGB_KEY not in data:
        return obs

    ycbcr = YCbCrTensor.from_rgb(data[RGB_KEY], ratio=ratio, as_uint8=True)
    data[Y_KEY] = ycbcr._tensors["y"]
    data[CBCR_KEY] = torch.cat([ycbcr._tensors["cb"], ycbcr._tensors["cr"]], dim=-3)
    if drop_rgb:
        del data[RGB_KEY]

    return _restore_type(obs, data)


def ycbcr_to_rgb_obs(obs: Mapping | TensorDict, drop_ycbcr: bool = False):
    data = _to_mapping(obs)
    has_components = Y_KEY in data and CBCR_KEY in data
    if not has_components:
        return obs

    cbcr = data[CBCR_KEY]
    cb, cr = cbcr[..., 0:1, :, :], cbcr[..., 1:2, :, :]
    rgb = YCbCrTensor(data[Y_KEY], cb, cr).to_rgb().round().to(torch.uint8)
    data[RGB_KEY] = rgb
    if drop_ycbcr:
        del data[Y_KEY]
        del data[CBCR_KEY]

    return _restore_type(obs, data)


def get_rgb_tensor(obs: Mapping | TensorDict) -> torch.Tensor:
    obs_rgb = ycbcr_to_rgb_obs(obs, drop_ycbcr=False)
    return obs_rgb[RGB_KEY]


def rgb_to_ycbcr_obs_np(
    obs: Mapping[str, np.ndarray],
    ratio: tuple[int, int] = (2, 2),
) -> dict[str, np.ndarray]:
    """Numpy-dict variant of ``rgb_to_ycbcr_obs``.

    Raw environment observations (from ``env.reset()``/``env.step()``) are
    plain numpy dicts, not tensors/TensorDicts, and get handed straight to
    the episodata writer for storage -- so the RGB->YCbCr conversion at that
    write site needs a numpy-in/numpy-out form, converting through torch
    internally to reuse ``YCbCrTensor``.
    """
    data = dict(obs)
    if RGB_KEY not in data:
        return data

    ycbcr = YCbCrTensor.from_rgb(torch.from_numpy(data[RGB_KEY]), ratio=ratio, as_uint8=True)
    data[Y_KEY] = ycbcr._tensors["y"].numpy()
    data[CBCR_KEY] = torch.cat([ycbcr._tensors["cb"], ycbcr._tensors["cr"]], dim=-3).numpy()
    del data[RGB_KEY]

    return data
