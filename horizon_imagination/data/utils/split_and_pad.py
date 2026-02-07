from typing import Any, Literal
import torch
from torch import Tensor
from tensordict.tensordict import TensorDict


from tensordict import NestedKey, pad, set_lazy_legacy, TensorDictBase


@set_lazy_legacy(False)
def split_and_pad(
    rollout_tensordict: TensorDictBase,
    lengths: Tensor,
    *,
    prefix=None,
    trajectory_key: NestedKey | None = None,
    done_key: NestedKey | None = None,
    pad_direction: Literal['prefix', 'suffix'] = 'suffix'
) -> TensorDictBase:
    """
    Modified the implementation of torchrl.collectors.utils.split_trajectories.
    Most code here is copied from their implementation.
    This version enables padding direction control and better handling of
    variable-length segments splitting via the `lengths` argument.
    """
    mask_key = None
    if trajectory_key is not None:
        from torchrl.envs.utils import _replace_last

        traj_ids_key = trajectory_key
        mask_key = _replace_last(trajectory_key, "mask")
    else:
        if prefix is None and "collector" in rollout_tensordict.keys():
            prefix = "collector"
        if prefix is None:
            traj_ids_key = "traj_ids"
            mask_key = "mask"
        else:
            traj_ids_key = (prefix, "traj_ids")
            mask_key = (prefix, "mask")

    rollout_tensordict = rollout_tensordict.copy()
    traj_ids = rollout_tensordict.get(traj_ids_key, None)
    if traj_ids is None:
        if done_key is None:
            done_key = "done"
        done_key = ("next", done_key)
        done = rollout_tensordict.get(done_key)
        idx = (slice(None),) * (rollout_tensordict.ndim - 1) + (slice(None, -1),)
        done_sel = done[idx]
        pads = [1, 0]
        pads = [0, 0] * (done.ndim - rollout_tensordict.ndim) + pads
        done_sel = torch.nn.functional.pad(done_sel, pads)
        if done_sel.shape != done.shape:
            raise RuntimeError(
                f"done and done_sel have different shape {done.shape} - {done_sel.shape} "
            )
        traj_ids = done_sel.cumsum(rollout_tensordict.ndim - 1)
        traj_ids = traj_ids.squeeze(-1)
        if rollout_tensordict.ndim > 1:
            for i in range(1, rollout_tensordict.shape[0]):
                traj_ids[i] += traj_ids[i - 1].max() + 1
        rollout_tensordict.set(traj_ids_key, traj_ids)

    assert lengths.dim() == 1
    lengths = lengths.tolist()
    # if all splits are identical then we can skip this function
    if len(set(lengths)) == 1 and lengths[0] == traj_ids.shape[-1]:
        rollout_tensordict.set(
            mask_key,
            torch.ones(
                rollout_tensordict.shape,
                device=rollout_tensordict.device,
                dtype=torch.bool,
            ),
        )
        if rollout_tensordict.ndimension() == 1:
            rollout_tensordict = rollout_tensordict.unsqueeze(0)
        return rollout_tensordict

    out_splits = rollout_tensordict.reshape(-1)

    out_splits = out_splits.split(lengths, 0)

    for out_split in out_splits:
        out_split.set(
            mask_key,
            torch.ones(
                out_split.shape,
                dtype=torch.bool,
                device=out_split.device,
            ),
        )
    if len(out_splits) > 1:
        MAX = max(*[out_split.shape[0] for out_split in out_splits])
    else:
        MAX = out_splits[0].shape[0]

    if pad_direction == 'suffix':
        pad_size = lambda s: [0, MAX - s.shape[0]]
    else:
        assert pad_direction == 'prefix'
        pad_size = lambda s: [MAX - s.shape[0], 0]

    l = [pad(out_split, pad_size(out_split)) for out_split in out_splits]

    td = torch.stack(
        l, 0
    )
    return td
