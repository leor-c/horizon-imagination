"""Shared ring-buffer pointer bookkeeping for rolling-context buffers.

Encapsulates the "2k mirrored-write" indexing scheme used by both
RollingContextBuffer (numpy) and TensorDictRollingContextBuffer (torch):
each of the k most-recent entries lives in a (2k, ...)-shaped underlying
buffer, and every write is mirrored to both `buf[j]` and `buf[j + k]`, so
a contiguous k-window is always available via a plain slice, with no
copy, regardless of wraparound.

This class owns only the index math (j, num_entries) -- not the actual
data storage, which differs by backend (numpy array vs. TensorDict).
Callers are responsible for performing the actual mirrored writes at the
positions this class hands back; this class only tracks where those
writes should go and what the current valid window is.
"""

from typing import Tuple


class RingBufferIndex:
    """Pointer/count bookkeeping for a (2k, ...) mirrored ring buffer.

    Usage from a buffer implementation:
        idx = RingBufferIndex(k)

        # on reset(): write your (placeholder_action, obs) pair at
        # position 0 (and mirrored position k), then:
        idx.mark_seeded()

        # on add(): write your (action, obs) pair at idx.write_pos()
        # (and mirrored position idx.write_pos() + k), then:
        idx.mark_added()

        # to read the current window:
        s, e = idx.read_range()  # slice your (2k, ...) buffer[s:e]
    """

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.j = 0  # next write slot, in [0, k)
        self.num_entries = 0  # number of valid entries so far, capped at k

    @property
    def has_been_reset(self) -> bool:
        """Whether mark_seeded() has been called at least once.
        num_entries is only ever 0 at construction, and 1+ from the
        moment mark_seeded() first runs -- it never returns to 0
        afterward -- so num_entries == 0 is an exact proxy for "never
        seeded"."""
        return self.num_entries != 0

    def write_pos(self) -> int:
        """Position to write the next add()-time entry into, before
        calling mark_added(). Always 0 for the reset()-time entry
        (mark_seeded() doesn't consult this)."""
        return self.j

    def mark_seeded(self) -> None:
        """Call once, immediately after writing the reset()-time entry
        at position 0 (and mirrored position k)."""
        self.j = 1 % self.k
        self.num_entries = 1

    def mark_added(self) -> None:
        """Call immediately after writing a regular add()-time entry at
        write_pos() (and mirrored position write_pos() + k)."""
        self.j = (self.j + 1) % self.k
        self.num_entries = min(self.num_entries + 1, self.k)

    def read_range(self) -> Tuple[int, int]:
        """Return (start, end) into the (2k, ...) buffer covering the
        current contiguous k-window (or fewer entries, if not yet
        full), oldest -> newest."""
        n = self.num_entries
        if n < self.k:
            return 0, n
        return self.j, self.j + self.k

    def __len__(self) -> int:
        return self.num_entries
