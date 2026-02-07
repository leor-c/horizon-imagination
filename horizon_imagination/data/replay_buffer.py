from typing import Literal
import torch
from tensordict.tensordict import TensorDict
from torchrl.data import (
    ReplayBuffer,
    TensorDictReplayBuffer,
    LazyMemmapStorage,
    LazyTensorStorage,
    RandomSampler,
    Storage
)
from torchrl.data.replay_buffers.replay_buffers import (
    is_tensor_collection,
    is_tensorclass,
    _to_torch,
    expand_as_right,
)
from collections import deque
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from horizon_imagination.utilities.types import ObsKey, Modality
from horizon_imagination.data.utils.split_and_pad import split_and_pad
from horizon_imagination.data.segment_sampler import SegmentSampler, _sample_indices
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass


class StalenessUniformSampler(RandomSampler):
    def __init__(
            self,
            uniform_prob: float = 0.5,
            staleness_alpha: float = 3,
            staleness_beta: float = 1,
        ):
        super().__init__()
        self.uniform_prob = uniform_prob
        self.staleness_alpha = staleness_alpha
        self.staleness_beta = staleness_beta

    def sample(self, storage: Storage, batch_size: int) -> tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError("Got Empty Storage")
        # index = storage._rand_given_ndim(batch_size)
        # a method to return random indices given the storage ndim
        if storage.ndim == 1:
            # index = torch.randint(
            #     0,
            #     len(storage),
            #     (batch_size,),
            #     generator=storage._rng,
            #     device=getattr(storage, "device", None),
            # )
            index = _sample_indices(
                max_index=len(storage),
                sample_size=batch_size,
                staleness_alpha=self.staleness_alpha,
                staleness_beta=self.staleness_beta,
                uniform_prob=self.uniform_prob,
                device=getattr(storage, "device", None),
                rng=storage._rng,
            )
        else:
            raise RuntimeError(
                f"Random number generation is not implemented for storage of type {type(self)} with ndim {self.ndim}. "
                f"Please report this exception as well as the use case (incl. buffer construction) on github."
            )
        return index, {}


class SegmentTDReplayBuffer(TensorDictReplayBuffer):
    def sample(
        self,
        batch_size=None,
        return_info=False,
        include_info=None,
    ):
        """
        Modified the `sample` method of `TensorDictReplayBuffer`.
        Changes:
        - handles info differently - allow `lengths` tensor for non uniform lengths segments.
        - automatically applies split and pad - designed to work together with
        `SegmentSampler` (also custom class in this project)
        """
        data, info = ReplayBuffer.sample(self, batch_size, return_info=True)
        is_tc = is_tensor_collection(data)
        if is_tc and not is_tensorclass(data) and include_info in (True, None):
            is_locked = data.is_locked
            if is_locked:
                data.unlock_()
            for key, val in info.items():
                if key == "index" and isinstance(val, tuple):
                    val = torch.stack(val, -1)
                elif key == "lengths":
                    continue
                try:
                    val = _to_torch(val, data.device)
                    if val.ndim < data.ndim:
                        val = expand_as_right(val, data)
                    data.set(key, val)
                except RuntimeError:
                    raise RuntimeError(
                        "Failed to set the metadata (e.g., indices or weights) in the sampled tensordict within TensorDictReplayBuffer.sample. "
                        "This is probably caused by a shape mismatch (one of the transforms has probably modified "
                        "the shape of the output tensordict). "
                        "You can always recover these items from the `sample` method from a regular ReplayBuffer "
                        "instance with the 'return_info' flag set to True."
                    )
            if "lengths" in info:
                assert isinstance(self._sampler, SegmentSampler)
                data = split_and_pad(
                    data,
                    lengths=info['lengths'],
                    trajectory_key="episode",
                    pad_direction='suffix'  # self._sampler.pad_direction,
                )
            if is_locked:
                data.lock_()
        elif not is_tc and include_info in (True, None):
            raise RuntimeError("Cannot include info in non-tensordict data")
        if return_info:
            return data, info
        return data


def get_replay_buffer_storage(
    max_size: int,
    store_on_disk: bool,
    data_path: Path = None,
    device=None,
):
    if device is None:
        device = "cpu"

    if store_on_disk:
        assert data_path is not None
        return LazyMemmapStorage(
            max_size=max_size, scratch_dir=data_path, device=device, existsok=True
        )
    else:
        return LazyTensorStorage(max_size=max_size, device=device)


def get_replay_buffer(rb_storage, rb_sampler, batch_size=None, prefetch=2):
    return TensorDictReplayBuffer(
        storage=rb_storage,
        sampler=rb_sampler,
        batch_size=batch_size,
        prefetch=prefetch,
    )


def get_segment_replay_buffer(
    rb_storage,
    rb_sampler: SegmentSampler,
    batch_size=None,
    prefetch=2,
):
    assert isinstance(rb_sampler, SegmentSampler)
    return SegmentTDReplayBuffer(
        storage=rb_storage,
        sampler=rb_sampler,
        batch_size=batch_size,
        prefetch=prefetch,
    )


class ReplayBufferIterator:
    def __init__(self, rb, filter_fn):
        self.rb = rb
        if filter_fn is None:
            filter_fn = lambda x: x
        self.filter_fn = filter_fn

    def __iter__(self):
        return self

    def __next__(self):
        next_batch = self.rb.sample().to('cuda')
        return self.filter_fn(next_batch)


class ReplayBufferTrajectoryIterator:
    def __init__(
        self,
        rb: SegmentTDReplayBuffer,
        segments_per_batch: int,
        steps_per_segment: int,
        filter_fn=None,
    ):
        self.rb = rb
        self.segments_per_batch = segments_per_batch
        self.steps_per_segment = steps_per_segment

        if filter_fn is None:
            filter_fn = lambda x: x
        self.filter_fn = filter_fn

    def __iter__(self):
        return self

    def __next__(self):
        batch_size = self.segments_per_batch * self.steps_per_segment
        if len(self.rb) == 0:
            return None
        next_batch = self.rb.sample(batch_size=batch_size).to('cuda')
        return self.filter_fn(next_batch)
    

class EpochDataIterator(Configurable):
    @dataclass
    class Config(BaseConfig):
        rb_storage: Storage
        tokenizer_steps: int
        world_model_steps: int
        controller_steps: int
        tokenizer_batch_size: int
        wm_segment_length: int
        wm_min_segment_length: int
        wm_batch_size: int
        c_segment_length: int
        c_min_segment_length: int
        c_batch_size: int
        prefetch: int = 2
        staleness_alpha: float = 3
        staleness_beta: float = 1
        uniform_prob: float = 0.7

    def __init__(self, config: Config):
        self.config = config

        self.tokenizer_rb = get_replay_buffer(
            config.rb_storage,
            rb_sampler=StalenessUniformSampler(
                staleness_alpha=config.staleness_alpha,
                staleness_beta=config.staleness_beta,
                uniform_prob=config.uniform_prob,
            ),
            batch_size=config.tokenizer_batch_size,
            prefetch=config.prefetch,
        )

        wm_sampler = SegmentSampler(
            segment_len=config.wm_segment_length, 
            min_length=config.wm_min_segment_length, 
            traj_key='episode', 
            pad_direction='suffix',
            staleness_alpha=config.staleness_alpha,
            staleness_beta=config.staleness_beta,
            uniform_prob=config.uniform_prob,
        )
        self.wm_rb = get_segment_replay_buffer(
            config.rb_storage, 
            wm_sampler,
            batch_size=config.wm_batch_size * config.wm_segment_length,
            prefetch=config.prefetch
        )

        c_sampler = SegmentSampler(
            segment_len=config.c_segment_length, 
            min_length=config.c_min_segment_length, 
            traj_key='episode', 
            pad_direction='prefix',
            staleness_alpha=config.staleness_alpha,
            staleness_beta=config.staleness_beta,
            uniform_prob=config.uniform_prob,
        )
        self.controller_rb = get_segment_replay_buffer(
            config.rb_storage,
            c_sampler,
            batch_size=config.c_batch_size * config.c_segment_length,
            prefetch=config.prefetch,
        )

    # def __len__(self):
    #     return self.config.tokenizer_steps + self.config.world_model_steps + self.config.controller_steps

    def __iter__(self):
        buffer = deque([])

        if len(self.tokenizer_rb) == 0:
            return

        def prefetch_tok():
            batch = self.tokenizer_rb.sample()['observation']
            batch = batch['image|features'].to('cuda')
            buffer.append(batch)

        def prefetch_wm():
            buffer.append(self.wm_rb.sample().to('cuda'))

        def prefetch_c():
            buffer.append(self.controller_rb.sample().to('cuda'))

        for _ in range(self.config.tokenizer_steps):
                    prefetch_tok()

        for _ in range(self.config.world_model_steps):
                    prefetch_wm()

        for _ in range(self.config.controller_steps):
                    prefetch_c()

        for i in range(self.config.tokenizer_steps):
            yield buffer.popleft(), 0

        for i in range(self.config.world_model_steps):
            yield buffer.popleft(), 1

        for i in range(self.config.controller_steps):
            yield buffer.popleft(), 2



