from typing import Literal

import episodata as ed
from episodata.utils import batch_to_tensordict
from collections import deque
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.utilities.obs_codec import get_rgb_tensors
from horizon_imagination.utilities.types import vector_keys


def infinite_loader(loader):
    while True:
        yield from loader


class EpochDataIterator(Configurable):
    @dataclass
    class Config(BaseConfig):
        replay_buffer: ed.Dataset
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
        read_chunk_size: int = 4096
        staleness_alpha: float = 3
        staleness_beta: float = 1
        uniform_prob: float = 0.7

    def __init__(self, config: Config):
        self.config = config

    def __iter__(self):
        buffer = deque([])

        # segment_stream() batches sampling+reads internally (see
        # episodata.SegmentStream / read_chunk_size) instead of the
        # DataLoader-per-item pattern this used to use, which is what makes
        # this fast on a zarr-backed replay buffer.
        tok_stream = self.config.replay_buffer.segment_stream(
            sequence_length=1,
            fields=self.config.replay_buffer.schema.field_keys(role='observation'),
            batch_size=self.config.tokenizer_batch_size,
            read_chunk_size=self.config.read_chunk_size,
            # sampler=  TODO: implement the staleness sampler for identical behavior to the existing version
        )
        if len(tok_stream.segments) == 0:
            return
        tok_iter = iter(tok_stream)

        wm_stream = self.config.replay_buffer.segment_stream(
            sequence_length=self.config.wm_segment_length,
            batch_size=self.config.wm_batch_size,
            read_chunk_size=self.config.read_chunk_size,
        )
        wm_iter = iter(wm_stream)

        c_stream = self.config.replay_buffer.segment_stream(
            sequence_length=self.config.c_segment_length,
            pad='prefix',
            batch_size=self.config.c_batch_size,
            read_chunk_size=self.config.read_chunk_size,
        )
        c_iter = iter(c_stream)

        def prefetch_tok():
            batch = batch_to_tensordict(next(tok_iter), device='cuda', include_all_observations=True)
            batch = batch['observation']
            # The obs encoders consume raw observations, one entry per key: RGB images
            # (decoded from their YCbCr components) and raw vectors, without the
            # singleton time dim of the length-1 segments.
            obs = {k: v[:, 0] for k, v in get_rgb_tensors(batch).items()}
            obs.update({k: batch[k][:, 0] for k in vector_keys(batch.keys())})
            buffer.append(obs)

        def prefetch_wm():
            batch = batch_to_tensordict(next(wm_iter), device='cuda', alignment="action_out")
            buffer.append(batch)

        def prefetch_c():
            batch = batch_to_tensordict(next(c_iter), device='cuda', alignment="action_out")
            buffer.append(batch)

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



