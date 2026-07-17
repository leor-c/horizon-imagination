"""
RollingContextBuffer: a fixed-size, allocation-free rolling context window over
(action, obs) pairs, using the "action-in" convention: at each step you
provide the action that *led to* the observation (action --> obs). This
matches how an autoregressive world model consumes history: to predict
obs[t+1], it conditions on the action about to be taken, and past
(action, obs) pairs are aligned so action[t] is "what produced obs[t]".

At episode start, there is no preceding action for the initial (reset)
observation. This is handled entirely internally: the buffer synthesizes
a placeholder action for slot 0 and fills it with a fixed dummy value
(zeros by default). The caller never sees or passes a dummy action --
the public API is just `reset(obs)` and `add(action, obs)`.

Usage
-----
    buf = RollingContextBuffer(k=8, action_spec=np.zeros(4, dtype=np.float32))
    buf.reset(obs)                  # seeds slot 0, placeholder action synthesized internally
    action = policy(buf.get_context())
    obs = env.step(action)
    buf.add(action, obs)
    ...

`action_spec` is optional but recommended: it's a one-time template (any
tree with the right shape/dtype per leaf; values are ignored) that lets
the buffer know the action structure *before* the first real action
exists, so `get_context()` right after `reset()` -- i.e. before you've
taken any action yet -- already has a valid placeholder in it. If you
omit `action_spec`, the buffer instead infers the structure lazily from
the first call to `add()`, and retroactively fills slot 0's placeholder
at that point; until then, `get_context()`'s action tree will simply be
empty.

`action_spec` also accepts a gymnasium `Space` directly (Box, Discrete,
MultiDiscrete, MultiBinary, or Dict of these), e.g.:

    buf = RollingContextBuffer(k=8, action_spec=env.action_space)

It's converted to a template Tree automatically. gymnasium is an
optional dependency -- only imported if you actually pass a Space.

Design notes
------------
- Supports both simple `np.ndarray` inputs and nested dicts of `np.ndarray`
  (a lightweight "pytree") for both `action` and `obs`.
- `add()` is O(1) amortized (two array writes per leaf, no allocation).
- `get_context()` is O(1): returns a read-only *view* into the underlying
  buffer, never a copy. This relies on a "double buffer" trick: each leaf
  is stored in a (2k, *shape) array, and every write is mirrored to both
  `buf[j]` and `buf[j + k]`. This guarantees the most recent k entries are
  always contiguous in memory, regardless of wraparound.
- `reset(obs)` starts a new episode: resets the write pointers (no
  reallocation -- old data is simply overwritten as new add() calls
  land) and writes the (placeholder action, obs) pair into slot 0.

Important caveat (by design, not a bug): the array returned by
`get_context()` is a *view*, not a snapshot. If you call `add()` again
after obtaining a context, the previously returned view no longer
represents "the context at the time you called it". If you need a
stable snapshot (e.g. to run an async forward pass while new frames
keep arriving), pass `get_context(copy=True)`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .ring_buffer_index import RingBufferIndex

try:
    import gymnasium as _gymnasium
except ImportError:  # gymnasium is optional; only needed if you pass a Space
    _gymnasium = None

# A "leaf" is a plain np.ndarray. A "tree" is either a leaf or a dict
# mapping str -> tree. This is a minimal pytree, enough for the common
# case of nested named arrays (e.g. {"rgb": arr, "meta": {"speed": arr}}).
Tree = Union[np.ndarray, Dict[str, "Tree"]]
Path = Tuple[str, ...]


def _space_to_template(space: "_gymnasium.Space") -> Tree:
    """Convert a gymnasium Space into a template Tree (shape/dtype only,
    values are zeros) usable as `action_spec`.

    Supported: Box, Discrete, MultiDiscrete, MultiBinary, and Dict (of
    any combination of the above, nested arbitrarily deep). Other space
    types (Tuple, Text, Sequence, Graph, ...) don't map cleanly onto this
    buffer's dict-of-arrays tree structure -- convert them to a plain
    Tree yourself and pass that as action_spec instead.
    """
    if isinstance(space, _gymnasium.spaces.Dict):
        return {key: _space_to_template(sub) for key, sub in space.spaces.items()}
    if hasattr(space, "shape") and hasattr(space, "dtype"):
        # Covers Box, Discrete, MultiDiscrete, MultiBinary: each exposes
        # a fixed .shape and .dtype describing a single sample. Discrete's
        # shape is () -- a 0-d array -- so actions for it should be passed
        # as e.g. np.array(action_int), not a bare python int.
        return np.zeros(space.shape, dtype=space.dtype)
    raise TypeError(
        f"Unsupported gymnasium space type {type(space).__name__}; "
        "only Box, Discrete, MultiDiscrete, MultiBinary, and Dict are "
        "supported. Convert it to a template Tree manually and pass that "
        "as action_spec instead."
    )


def _flatten(tree: Tree, prefix: Path = ()) -> List[Tuple[Path, np.ndarray]]:
    """Flatten a nested dict-of-arrays into a sorted list of (path, array).

    Sorting keys at each level makes the flattened order deterministic
    and independent of dict insertion order, so structure comparisons
    across calls are reliable.
    """
    if isinstance(tree, dict):
        items: List[Tuple[Path, np.ndarray]] = []
        for key in sorted(tree.keys()):
            items.extend(_flatten(tree[key], prefix + (key,)))
        return items
    if isinstance(tree, np.ndarray):
        return [(prefix, tree)]
    raise TypeError(
        f"Unsupported leaf type {type(tree)} at path {prefix}; "
        "expected np.ndarray or dict."
    )


def _unflatten(pairs: List[Tuple[Path, Any]]) -> Tree:
    """Inverse of _flatten. Rebuilds the nested dict structure."""
    if len(pairs) == 1 and pairs[0][0] == ():
        return pairs[0][1]
    root: Dict[str, Any] = {}
    for path, value in pairs:
        node = root
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return root


class RollingContextBuffer:
    """Fixed-size rolling context window over (action, obs) pairs, using
    the action-in convention: action[t] is the action that produced obs[t].

    The placeholder action needed at episode start (slot 0, where there's
    no real preceding action) is handled internally -- it never appears
    in the public API.

    Parameters
    ----------
    k : int
        Context window size (number of most recent steps retained).
    action_spec : Tree or gymnasium.Space, optional
        A template action (any np.ndarray or nested dict of np.ndarray),
        or a gymnasium Space (Box, Discrete, MultiDiscrete, MultiBinary,
        or Dict of these) -- used only for its shape/dtype, establishing
        the action structure up front. Recommended: without it, the
        action structure -- and therefore the episode-start placeholder
        -- isn't known until the first add() call.
    dummy_value : float, default 0.0
        Fill value used for the synthesized placeholder action.
    """

    def __init__(
        self,
        k: int,
        action_spec: Optional[Union[Tree, "_gymnasium.Space"]] = None,
        dummy_value: float = 0.0,
    ):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self._dummy_value = dummy_value

        self._action_buffers: Dict[Path, np.ndarray] = {}
        self._obs_buffers: Dict[Path, np.ndarray] = {}
        self._action_paths: List[Path] = []
        self._obs_paths: List[Path] = []
        self._action_initialized = False
        self._obs_initialized = False

        # If a slot-0 placeholder couldn't be filled yet at reset() time
        # (because the action structure wasn't known), this holds the
        # buffer position it needs to be back-filled into, once known.
        self._pending_dummy_pos: Optional[int] = None

        self._idx = RingBufferIndex(k)

        if action_spec is not None:
            if _gymnasium is not None and isinstance(action_spec, _gymnasium.Space):
                action_spec = _space_to_template(action_spec)
            self._init_action_buffers(action_spec)

    # -- setup ---------------------------------------------------------------

    def _init_action_buffers(self, action: Tree) -> None:
        flat_action = _flatten(action)
        self._action_paths = [p for p, _ in flat_action]
        for path, arr in flat_action:
            self._action_buffers[path] = np.empty(
                (2 * self.k,) + arr.shape, dtype=arr.dtype
            )
        self._action_initialized = True

    def _init_obs_buffers(self, obs: Tree) -> None:
        flat_obs = _flatten(obs)
        self._obs_paths = [p for p, _ in flat_obs]
        for path, arr in flat_obs:
            self._obs_buffers[path] = np.empty(
                (2 * self.k,) + arr.shape, dtype=arr.dtype
            )
        self._obs_initialized = True

    def _check_structure(
        self, flat: List[Tuple[Path, np.ndarray]], expected_paths: List[Path],
        buffers: Dict[Path, np.ndarray], kind: str,
    ) -> None:
        got_paths = [p for p, _ in flat]
        if got_paths != expected_paths:
            raise ValueError(
                f"{kind} structure changed between calls.\n"
                f"expected paths: {expected_paths}\n"
                f"got paths:      {got_paths}"
            )
        for path, arr in flat:
            expected_shape = buffers[path].shape[1:]
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{kind} leaf at {path} has shape {arr.shape}, "
                    f"expected {expected_shape}"
                )

    def _fill_dummy_action(self, pos: int) -> None:
        """Write the synthesized placeholder action into `pos` (and its
        mirrored slot pos+k). Requires action buffers to be initialized."""
        for path, buf in self._action_buffers.items():
            fill = np.full(buf.shape[1:], self._dummy_value, dtype=buf.dtype)
            buf[pos] = fill
            buf[pos + self.k] = fill

    def _write_obs(self, obs: Tree, pos: int) -> None:
        flat_obs = _flatten(obs)
        if not self._obs_initialized:
            self._init_obs_buffers(obs)
        else:
            self._check_structure(flat_obs, self._obs_paths, self._obs_buffers, "obs")
        for path, arr in flat_obs:
            buf = self._obs_buffers[path]
            buf[pos] = arr
            buf[pos + self.k] = arr

    def _write_action(self, action: Tree, pos: int) -> None:
        flat_action = _flatten(action)
        if not self._action_initialized:
            self._init_action_buffers(action)
            if self._pending_dummy_pos is not None:
                self._fill_dummy_action(self._pending_dummy_pos)
                self._pending_dummy_pos = None
        else:
            self._check_structure(
                flat_action, self._action_paths, self._action_buffers, "action"
            )
        for path, arr in flat_action:
            buf = self._action_buffers[path]
            buf[pos] = arr
            buf[pos + self.k] = arr

    # -- core API --------------------------------------------------------------

    def reset(self, obs: Tree) -> None:
        """Start a new episode with initial observation `obs`.

        Resets the write pointers (no reallocation) and writes a
        (placeholder action, obs) pair into slot 0. If the action
        structure is already known (either from `action_spec` at
        construction, or from a previous episode's add() calls), the
        placeholder is filled immediately, so get_context() right after
        reset() -- before you've taken any action -- is already valid.
        Otherwise the placeholder is filled retroactively on the first
        subsequent add() call.
        """
        self._write_obs(obs, pos=0)
        self._idx.mark_seeded()

        if self._action_initialized:
            self._fill_dummy_action(0)
            self._pending_dummy_pos = None
        else:
            self._pending_dummy_pos = 0

    def add(self, action: Tree, obs: Tree) -> None:
        """Append one (action, obs) step, where `action` is the action
        that led to `obs`. O(1) amortized, no allocation after the first
        call establishing each structure. Must be preceded by reset()."""
        if not self.has_been_reset:
            raise RuntimeError("add() called before reset(); call reset(obs) first.")
        p = self._idx.write_pos()
        self._write_action(action, p)
        self._write_obs(obs, p)
        self._idx.mark_added()

    def get_context(self, copy: bool = False) -> Optional[Tuple[Tree, Tree]]:
        """Return the current context window as (action_tree, obs_tree),
        each leaf of shape (n, *leaf_shape) where n = min(steps_added, k),
        ordered oldest -> newest. action[i] is the action that produced
        obs[i] (action[0] is the internally-synthesized placeholder).

        Returns None if reset() has not been called yet -- there is no
        context to return, and the shape of "empty" would otherwise be
        ambiguous (it'd depend on whether action_spec was given at
        construction). Once reset() has been called at least once, this
        always returns a real (action_tree, obs_tree) pair, never None.

        The returned arrays are read-only views by default (zero-copy).
        Pass copy=True if you need a stable snapshot that survives
        subsequent add() calls.
        """
        if not self.has_been_reset:
            return None

        s, e = self._idx.read_range()

        action_pairs = []
        for path in self._action_paths:
            view = self._action_buffers[path][s:e]
            if copy:
                view = view.copy()
            else:
                view = view.view()
                view.flags.writeable = False
            action_pairs.append((path, view))

        obs_pairs = []
        for path in self._obs_paths:
            view = self._obs_buffers[path][s:e]
            if copy:
                view = view.copy()
            else:
                view = view.view()
                view.flags.writeable = False
            obs_pairs.append((path, view))

        action_out = _unflatten(action_pairs) if action_pairs else {}
        obs_out = _unflatten(obs_pairs) if obs_pairs else {}
        return action_out, obs_out

    def __len__(self) -> int:
        return len(self._idx)

    @property
    def has_been_reset(self) -> bool:
        """Whether reset() has been called at least once."""
        return self._idx.has_been_reset
