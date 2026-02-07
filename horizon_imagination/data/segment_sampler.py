from typing import Any, Literal
from torchrl.data.replay_buffers.samplers import (
    Sampler,
)
import torch


def _sample_indices(
    max_index: int, 
    sample_size: int,
    staleness_alpha,
    staleness_beta, 
    uniform_prob, 
    device, 
    rng
):
    uniform_part = torch.randint(
        max_index, (sample_size,), device=device, generator=rng
    )
    staleness_part = torch.distributions.Beta(
        torch.ones(sample_size, device=device) * staleness_alpha,
        torch.ones(sample_size, device=device) * staleness_beta,
    ).sample() * max_index - 1e-8
    staleness_part = torch.floor(staleness_part).long()

    sample = torch.where(
        torch.rand(sample_size, device=device) < uniform_prob, 
        uniform_part, 
        staleness_part,
    )
    return sample


class SegmentSampler(Sampler):
    """
    Solution:
    get the total num of transitions
    --> sample `num_slices` indices uniformly between 0 and num_transitions
    --> define the left / right window bound as idx -/+ window_size
    --> clip the above so that it does not "overflow" to previous / next episode
    using binary search on the list of indices of first obs of each episode.

    ==> return a torch.arange() of this window.

    ==> depending on side, we can switch padding side after initial padding (split_...).
    """

    def __init__(
        self,
        *,
        num_segments: int = None,
        segment_len: int = None,
        min_length: int = 1,
        traj_key: str = None,
        pad_direction: Literal["prefix", "suffix"] = "suffix",
        uniform_prob: float = 0.5,
        staleness_alpha: float = 3,
        staleness_beta: float = 1,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.segment_len = segment_len
        self.traj_key = traj_key
        self.pad_direction = pad_direction
        self.min_length = min_length if min_length is not None else 1

        self.uniform_prob = uniform_prob
        self.staleness_alpha = staleness_alpha
        self.staleness_beta = staleness_beta


    def sample(self, storage, batch_size):
        num_segments, segment_len = self.infer_2d_batch_size(batch_size=batch_size)

        episode_key = "episode"
        episode_id = storage[:][episode_key]

        # Compute index of first step of each episode + episode lengths:
        _, first_step_idxs, episode_lengths = torch.unique_consecutive(
            episode_id, return_inverse=True, return_counts=True
        )
        first_step_idxs, counts = torch.unique_consecutive(
            first_step_idxs, return_counts=True
        )
        first_step_idxs[1:] = first_step_idxs[1:] + torch.cumsum(counts[:-1] - 1, 0)

        # compute indices of valid start points to ensure min length requirement:
        valid_episodes = episode_lengths >= self.min_length
        valid_first_idxs = first_step_idxs[valid_episodes]
        valid_lengths = episode_lengths[valid_episodes] - (self.min_length - 1)

        total_valid = valid_lengths.sum()
        valid_cum_len = torch.cumsum(valid_lengths, 0)

        if self.pad_direction == "suffix":
            starts, ends = self._suffix_pad_starts_ends(
                total_valid=total_valid,
                valid_cum_len=valid_cum_len,
                valid_first_idxs=valid_first_idxs,
                episode_lengths=episode_lengths,
                valid_episodes=valid_episodes,
                num_segments=num_segments,
                segment_len=segment_len,
                device=episode_id.device
            )
        else:
            assert self.pad_direction == "prefix"
            starts, ends = self._prefix_pad_starts_ends(
                total_valid=total_valid,
                valid_cum_len=valid_cum_len,
                valid_first_idxs=valid_first_idxs,
                num_segments=num_segments,
                segment_len=segment_len,
                device=episode_id.device
            )

        index = torch.cat([torch.arange(s, e) for s, e in zip(starts, ends)])
        sampled_lengths = ends - starts

        return index, {"lengths": sampled_lengths}

    def infer_2d_batch_size(self, batch_size):
        num_segments = self.num_segments
        segment_len = self.segment_len

        if self.num_segments is None:
            assert self.segment_len is not None
            num_segments = batch_size // self.segment_len

        if self.segment_len is None:
            assert self.num_segments is not None
            segment_len = batch_size // self.num_segments

        return num_segments, segment_len
    
    def _sample_indices(self, max_index: int, sample_size: int, device):
        return _sample_indices(
            max_index=max_index,
            sample_size=sample_size,
            staleness_alpha=self.staleness_alpha,
            staleness_beta=self.staleness_beta,
            uniform_prob=self.uniform_prob,
            device=device,
            rng=self._rng,
        )

    def _suffix_pad_starts_ends(
        self,
        total_valid,
        valid_cum_len,
        valid_first_idxs,
        episode_lengths,
        valid_episodes,
        num_segments: int,
        segment_len: int,
        device,
    ):
        """
        Sample the left side of the segment,
        then set the right side as right <--- left + length
        Apply corrections - avoid "overflow" to next episodes,
        maintain minimum length requirement.
        """
        # starts = torch.randint(
        #     total_valid, (num_segments,), device=device, generator=self._rng
        # )
        starts = self._sample_indices(total_valid, num_segments, device)
        starts_idx = torch.searchsorted(valid_cum_len, starts, right=True)
        starts_shift = torch.where(
            starts_idx > 0, starts - valid_cum_len[starts_idx - 1], starts
        )
        starts = valid_first_idxs[starts_idx] + starts_shift

        filtered_lengths = episode_lengths[valid_episodes]
        ends = torch.minimum(
            starts + segment_len,
            valid_first_idxs[starts_idx] + filtered_lengths[starts_idx],
        )

        return starts, ends

    def _prefix_pad_starts_ends(
        self,
        total_valid,
        valid_cum_len,
        valid_first_idxs,
        num_segments: int,
        segment_len: int,
        device,
    ):
        """
        Sample the right side of the segment,
        then set the left side as left <--- right - length
        Apply corrections - avoid "underflow" to previous episodes,
        maintain minimum length requirement.
        """
        # ends = torch.randint(
        #     total_valid, (num_segments,), device=device, generator=self._rng
        # )
        ends = self._sample_indices(total_valid, num_segments, device)
        ends_idx = torch.searchsorted(valid_cum_len, ends, right=True)
        shift = torch.where(ends_idx > 0, ends - valid_cum_len[ends_idx - 1], ends)
        ends = valid_first_idxs[ends_idx] + shift + self.min_length

        starts = torch.maximum(ends - segment_len, valid_first_idxs[ends_idx])

        return starts, ends

    def _empty(self):
        pass

    def dumps(self, path):
        # no op
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return
