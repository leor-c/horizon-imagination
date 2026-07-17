"""
TensorDictRollingContextBuffer: a fixed-size rolling context window over
(action, obs) pairs, built natively around `tensordict.TensorDict` for
the obs side -- intended for the case where obs is already an encoded
latent (or any nested structure of GPU tensors) that you want to buffer
directly, rather than re-encoding a raw observation sequence every step.

Same "action-in" convention and API shape as RollingContextBuffer (the
numpy version): `reset(obs)` / `add(action, obs)` / `get_context()`, and
the same episode-start placeholder-action handling (fully internal, no
`dummy_action` in the public API). The ring-buffer pointer bookkeeping
(the "2k mirrored-write" trick) is shared with the numpy version via
`RingBufferIndex` -- only the storage/write mechanics differ, since they
lean on TensorDict/torch primitives instead of numpy's.

Usage
-----
    obs_spec = TensorDict({"z": torch.zeros(256)}, batch_size=[], device="cuda")
    action_spec = torch.zeros(4, device="cuda")  # or a TensorDict, see below

    buf = TensorDictRollingContextBuffer(k=8, obs_spec=obs_spec, action_spec=action_spec)
    buf.reset(obs)                  # seeds slot 0, placeholder action synthesized internally
    action = policy(buf.get_context())
    obs = env.step(action)          # obs already encoded to a latent by caller
    buf.add(action, obs)

`obs_spec` is optional, same as `action_spec`: if given, GPU allocation
happens once, up front, at construction time -- recommended, since it
makes device/dtype/shape decisions explicit and catches problems (e.g.
the wrong device) immediately rather than at some later reset() call. If
omitted, the obs structure is instead inferred from whatever you pass to
the *first* `reset()` call -- including its device -- so make sure that
first obs already lives where you want the buffer to live; there's no
separate device argument to fall back on. `obs_spec` (or the first obs,
if inferring lazily) can be a `TensorDict` (any nesting) or a plain
`torch.Tensor`, each representing a *single* time step (no leading
batch/time dim).

`action_spec` is optional too, exactly as in the numpy version: if given
(TensorDict or Tensor, single time step), the episode-start placeholder
is ready immediately after reset(); if omitted, it's inferred lazily
from the first add() and back-filled retroactively.

Either `obs_spec` or `action_spec` can also be a gymnasium `Space`
directly (Box, Discrete, MultiDiscrete, MultiBinary, or Dict of these),
e.g. `action_spec=env.action_space`. It's converted to a template
Tensor/TensorDict automatically. Since spaces carry no device info of
their own (they're numpy-based), pass `device=...` to the constructor if
you want the resulting buffers on GPU -- otherwise they land on CPU.
gymnasium is an optional dependency, only imported if you actually pass
a Space.

Important: unlike the numpy version, there is **no real read-only
guarantee** here. `get_context()`'s default (copy=False) return value is
a torch view sharing memory with the internal buffer -- but PyTorch has
no equivalent of numpy's `.flags.writeable = False`; TensorDict's
`.lock_()` only blocks structural changes (adding/removing/reassigning
keys), not in-place leaf-tensor mutation. So the returned view is
genuinely mutable by the caller, not just semantically stale -- treat it
as read-only by convention, or pass `get_context(copy=True)` if you need
an actual safety guarantee (e.g. before handing it to code you don't
control).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict

from .ring_buffer_index import RingBufferIndex

try:
    import gymnasium as _gymnasium
except ImportError:  # gymnasium is optional; only needed if you pass a Space
    _gymnasium = None

ActionOrObs = Union[torch.Tensor, TensorDict]


def _space_to_tensor_template(
    space: "_gymnasium.Space", device: Optional[torch.device] = None
) -> ActionOrObs:
    """Convert a gymnasium Space into a template (torch.Tensor or
    TensorDict, shape/dtype only, values are zeros) usable as obs_spec
    or action_spec.

    Supported: Box, Discrete, MultiDiscrete, MultiBinary, and Dict (of
    any combination, nested arbitrarily deep) -- same coverage as the
    numpy version's space support. Spaces carry no device info of their
    own (they're numpy-based), so `device` controls where the resulting
    tensor(s) land; defaults to CPU if not given.
    """
    if isinstance(space, _gymnasium.spaces.Dict):
        return TensorDict(
            {
                key: _space_to_tensor_template(sub, device)
                for key, sub in space.spaces.items()
            },
            batch_size=[],
        )
    if hasattr(space, "shape") and hasattr(space, "dtype"):
        # Covers Box, Discrete, MultiDiscrete, MultiBinary. Route through
        # numpy only for dtype translation (np.dtype -> torch.dtype);
        # torch.from_numpy gets this exactly right for every dtype these
        # spaces use. Discrete's shape is () -- a 0-d tensor -- so
        # actions for it should be passed as e.g. torch.tensor(action_int),
        # not a bare python int.
        template = torch.from_numpy(np.zeros(space.shape, dtype=space.dtype))
        return template.to(device) if device is not None else template
    raise TypeError(
        f"Unsupported gymnasium space type {type(space).__name__}; "
        "only Box, Discrete, MultiDiscrete, MultiBinary, and Dict are "
        "supported. Convert it to a template Tensor/TensorDict manually "
        "and pass that as obs_spec/action_spec instead."
    )


def _alloc(spec: ActionOrObs, size: int) -> ActionOrObs:
    """Allocate a (size, *leaf_shape) buffer per leaf of `spec` (a single
    time step), preserving dtype/device, without touching CPU (no numpy
    round-trip)."""
    if isinstance(spec, TensorDict):
        return spec.apply(
            lambda t: torch.empty((size,) + t.shape, dtype=t.dtype, device=t.device),
            batch_size=[size],
        )
    if isinstance(spec, torch.Tensor):
        return torch.empty((size,) + spec.shape, dtype=spec.dtype, device=spec.device)
    raise TypeError(
        f"Unsupported spec type {type(spec).__name__}; expected torch.Tensor "
        "or tensordict.TensorDict."
    )


def _check_structure(buf: ActionOrObs, value: ActionOrObs, kind: str) -> None:
    """TensorDict item assignment silently ignores keys missing from
    `value` (stale data lingers) and silently adds keys present in
    `value` but not in `buf` (structure drifts). Neither raises on its
    own, so we check explicitly rather than relying on buf[pos] = value
    to catch it."""
    if isinstance(buf, TensorDict):
        if not isinstance(value, TensorDict):
            raise TypeError(f"{kind} structure changed: expected a TensorDict, got {type(value).__name__}")
        buf_keys = set(buf.keys(include_nested=True, leaves_only=True))
        value_keys = set(value.keys(include_nested=True, leaves_only=True))
        if buf_keys != value_keys:
            raise ValueError(
                f"{kind} structure changed between calls.\n"
                f"expected keys: {sorted(map(str, buf_keys))}\n"
                f"got keys:      {sorted(map(str, value_keys))}"
            )
        for key in buf_keys:
            expected_shape = buf[key].shape[1:]
            if value[key].shape != expected_shape:
                raise ValueError(
                    f"{kind} leaf {key} has shape {tuple(value[key].shape)}, "
                    f"expected {tuple(expected_shape)}"
                )
    else:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{kind} structure changed: expected a torch.Tensor, got {type(value).__name__}")
        if value.shape != buf.shape[1:]:
            raise ValueError(
                f"{kind} has shape {tuple(value.shape)}, expected {tuple(buf.shape[1:])}"
            )


class TensorDictRollingContextBuffer:
    """Fixed-size rolling context window over (action, obs) pairs, built
    around TensorDict/torch for the obs side (typically an encoded
    latent already living on GPU). Same action-in convention as
    RollingContextBuffer: action[t] is the action that produced obs[t].

    Parameters
    ----------
    k : int
        Context window size (number of most recent steps retained).
    obs_spec : torch.Tensor, TensorDict, or gymnasium.Space, optional
        A single-time-step template for obs (shape/dtype/device only;
        values are ignored), or a gymnasium Space. Recommended: without
        it, obs shape/dtype/device is instead inferred from the first
        reset() call, which means allocation timing and device placement
        become implicit -- fine if that's not a concern for you, but a
        source of hard-to-trace bugs if the first obs happens to land on
        the wrong device.
    action_spec : torch.Tensor, TensorDict, or gymnasium.Space, optional
        Same idea, for actions. If omitted, the action structure (and
        the episode-start placeholder) is inferred lazily from the first
        add() call.
    dummy_value : float, default 0.0
        Fill value used for the synthesized placeholder action.
    device : torch.device, optional
        Only used when `obs_spec`/`action_spec` is a gymnasium Space
        (spaces carry no device info of their own). Ignored for
        Tensor/TensorDict specs, which already carry their own device --
        and irrelevant if both specs are omitted, since the device is
        then inferred from your first reset()/add() call instead.
    """

    def __init__(
        self,
        k: int,
        obs_spec: Optional[Union[ActionOrObs, "_gymnasium.Space"]] = None,
        action_spec: Optional[Union[ActionOrObs, "_gymnasium.Space"]] = None,
        dummy_value: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.k = k
        self._dummy_value = dummy_value
        self._idx = RingBufferIndex(k)

        self._obs_buffer: Optional[ActionOrObs] = None
        self._action_buffer: Optional[ActionOrObs] = None

        # If a slot-0 placeholder couldn't be filled yet at reset() time
        # (because the action structure wasn't known), this holds the
        # buffer position it needs to be back-filled into, once known.
        self._pending_dummy_pos: Optional[int] = None

        if obs_spec is not None:
            if _gymnasium is not None and isinstance(obs_spec, _gymnasium.Space):
                obs_spec = _space_to_tensor_template(obs_spec, device)
            self._obs_buffer = _alloc(obs_spec, 2 * k)
        if action_spec is not None:
            if _gymnasium is not None and isinstance(action_spec, _gymnasium.Space):
                action_spec = _space_to_tensor_template(action_spec, device)
            self._action_buffer = _alloc(action_spec, 2 * k)

    # -- setup ---------------------------------------------------------------

    def _fill_dummy_action(self, pos: int) -> None:
        """Write the synthesized placeholder action into `pos` (and its
        mirrored slot pos+k). Requires the action buffer to be allocated."""
        buf = self._action_buffer
        if isinstance(buf, TensorDict):
            # Build a single-time-step (batch_size=[]) TensorDict, not a
            # (2k, ...)-shaped one -- buf[0] gives us that shape/dtype
            # template directly, we just need to overwrite its values.
            fill = buf[0].apply(lambda t: torch.full_like(t, self._dummy_value))
        else:
            fill = torch.full(buf.shape[1:], self._dummy_value, dtype=buf.dtype, device=buf.device)
        buf[pos] = fill
        buf[pos + self.k] = fill

    def _write_obs(self, obs: ActionOrObs, pos: int) -> None:
        if self._obs_buffer is None:
            self._obs_buffer = _alloc(obs, 2 * self.k)
        else:
            _check_structure(self._obs_buffer, obs, "obs")
        self._obs_buffer[pos] = obs
        self._obs_buffer[pos + self.k] = obs

    def _write_action(self, action: ActionOrObs, pos: int) -> None:
        if self._action_buffer is None:
            self._action_buffer = _alloc(action, 2 * self.k)
            if self._pending_dummy_pos is not None:
                self._fill_dummy_action(self._pending_dummy_pos)
                self._pending_dummy_pos = None
        else:
            _check_structure(self._action_buffer, action, "action")
        self._action_buffer[pos] = action
        self._action_buffer[pos + self.k] = action

    # -- core API --------------------------------------------------------------

    def reset(self, obs: ActionOrObs) -> None:
        """Start a new episode with initial observation `obs` (already
        encoded, if that's your use case).

        Resets the write pointers (no reallocation) and writes a
        (placeholder action, obs) pair into slot 0. If the action
        structure is already known (from `action_spec` at construction,
        or a previous episode's add() calls), the placeholder is filled
        immediately, so get_context() right after reset() -- before
        you've taken any action -- is already valid. Otherwise it's
        filled retroactively on the first subsequent add() call.
        """
        self._write_obs(obs, pos=0)
        self._idx.mark_seeded()

        if self._action_buffer is not None:
            self._fill_dummy_action(0)
            self._pending_dummy_pos = None
        else:
            self._pending_dummy_pos = 0

    def add(self, action: ActionOrObs, obs: ActionOrObs) -> None:
        """Append one (action, obs) step, where `action` is the action
        that led to `obs`. Must be preceded by reset()."""
        if not self.has_been_reset:
            raise RuntimeError("add() called before reset(); call reset(obs) first.")
        p = self._idx.write_pos()
        self._write_action(action, p)
        self._write_obs(obs, p)
        self._idx.mark_added()

    def get_context(self, copy: bool = False) -> Optional[Tuple[Optional[ActionOrObs], ActionOrObs]]:
        """Return the current context window as (action_ctx, obs_ctx),
        each of shape (n, *leaf_shape) where n = min(steps_added, k),
        ordered oldest -> newest. action_ctx[0] is the internally
        synthesized placeholder.

        Returns None if reset() has not been called yet.

        action_ctx is None if no action structure is known yet (no
        action_spec given, and no add() call has happened yet).

        The returned views share memory with the internal buffer by
        default (copy=False) -- see the module docstring's caveat: this
        is NOT enforced read-only the way the numpy version is, since
        PyTorch has no equivalent of numpy's writeable flag. Pass
        copy=True for an actual safety guarantee (uses .clone()).
        """
        if not self.has_been_reset:
            return None

        s, e = self._idx.read_range()
        obs_ctx = self._obs_buffer[s:e]
        if copy:
            obs_ctx = obs_ctx.clone()

        if self._action_buffer is None:
            action_ctx = None
        else:
            action_ctx = self._action_buffer[s:e]
            if copy:
                action_ctx = action_ctx.clone()

        return action_ctx, obs_ctx

    def __len__(self) -> int:
        return len(self._idx)

    @property
    def has_been_reset(self) -> bool:
        """Whether reset() has been called at least once."""
        return self._idx.has_been_reset
