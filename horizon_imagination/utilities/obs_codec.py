from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.types import ObsKey, Modality, image_keys
from horizon_imagination.utilities.ycbcr import YCbCrTensor


def y_key(rgb_key: ObsKey) -> ObsKey:
    return ObsKey.from_parts(Modality.image, f"{ObsKey(rgb_key).name}_y")


def cbcr_key(rgb_key: ObsKey) -> ObsKey:
    return ObsKey.from_parts(Modality.image, f"{ObsKey(rgb_key).name}_cbcr")


def rgb_key_of(component_key: ObsKey) -> ObsKey | None:
    """The RGB key a '<name>_y' / '<name>_cbcr' component belongs to, if any."""
    name = ObsKey(component_key).name
    for suffix in ('_y', '_cbcr'):
        if name.endswith(suffix):
            return ObsKey.from_parts(Modality.image, name[:-len(suffix)])
    return None


def rgb_keys(obs_keys) -> list[ObsKey]:
    """The RGB image keys of an observation, i.e. image keys that are not YCbCr components."""
    return [k for k in image_keys(obs_keys) if rgb_key_of(k) is None]


def ycbcr_source_keys(obs_keys) -> list[ObsKey]:
    """The RGB keys reconstructible from the '_y'/'_cbcr' components present in an observation."""
    keys = set(map(str, obs_keys))
    return [
        k for k in image_keys(obs_keys)
        if (rgb := rgb_key_of(k)) is not None and str(y_key(rgb)) in keys and str(cbcr_key(rgb)) in keys
        and k == y_key(rgb)
    ]


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
    keys = rgb_keys(data.keys())
    if not keys:
        return obs

    for key in keys:
        ycbcr = YCbCrTensor.from_rgb(data[key], ratio=ratio, as_uint8=True)
        data[y_key(key)] = ycbcr._tensors["y"]
        data[cbcr_key(key)] = torch.cat([ycbcr._tensors["cb"], ycbcr._tensors["cr"]], dim=-3)
        if drop_rgb:
            del data[key]

    return _restore_type(obs, data)


def ycbcr_to_rgb_obs(obs: Mapping | TensorDict, drop_ycbcr: bool = False):
    data = _to_mapping(obs)
    keys = ycbcr_source_keys(data.keys())
    if not keys:
        return obs

    for key in keys:
        rgb = rgb_key_of(key)
        cbcr = data[cbcr_key(rgb)]
        cb, cr = cbcr[..., 0:1, :, :], cbcr[..., 1:2, :, :]
        data[rgb] = YCbCrTensor(data[y_key(rgb)], cb, cr).to_rgb().round().to(torch.uint8)
        if drop_ycbcr:
            del data[y_key(rgb)]
            del data[cbcr_key(rgb)]

    return _restore_type(obs, data)


def get_rgb_tensors(obs: Mapping | TensorDict) -> dict[ObsKey, torch.Tensor]:
    """The RGB tensor of every image key, decoding YCbCr components where needed."""
    obs_rgb = ycbcr_to_rgb_obs(obs, drop_ycbcr=False)
    return {k: obs_rgb[k] for k in rgb_keys(obs_rgb.keys())}


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
    for key in rgb_keys(data.keys()):
        ycbcr = YCbCrTensor.from_rgb(torch.from_numpy(data[key]), ratio=ratio, as_uint8=True)
        data[y_key(key)] = ycbcr._tensors["y"].numpy()
        data[cbcr_key(key)] = torch.cat([ycbcr._tensors["cb"], ycbcr._tensors["cr"]], dim=-3).numpy()
        del data[key]

    return data
