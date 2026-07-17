from typing import Literal
import torch
from torch.utils.data import DataLoader

import episodata as ed
from episodata.utils import batch_to_tensordict
from collections import deque
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass


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
        prefetch: int = 2
        staleness_alpha: float = 3
        staleness_beta: float = 1
        uniform_prob: float = 0.7

    def __init__(self, config: Config):
        self.config = config

    def __iter__(self):
        buffer = deque([])

        segments_dataset = self.config.replay_buffer.segments(sequence_length=1, fields=['image|features'])
        if len(segments_dataset) == 0:
            return
        tokenizer_loader = DataLoader(
            dataset=segments_dataset,
            batch_size=self.config.tokenizer_batch_size,
            shuffle=True,
            collate_fn=segments_dataset.collate,
            # sampler=  TODO: implement the staleness sampler for identical behavior to the existing version
        )
        tokenizer_iter = infinite_loader(tokenizer_loader)

        wm_segments = self.config.replay_buffer.segments(sequence_length=self.config.wm_segment_length)
        wm_loader = DataLoader(
            dataset=wm_segments,
            batch_size=self.config.wm_batch_size,
            shuffle=True,
            collate_fn=wm_segments.collate,
            # sampler=  TODO: implement the staleness sampler for identical behavior to the existing version
        )
        wm_iter = infinite_loader(wm_loader)

        c_segments = self.config.replay_buffer.segments(sequence_length=self.config.c_segment_length, pad='prefix')
        c_loader = DataLoader(
            dataset=c_segments,
            batch_size=self.config.c_batch_size,
            shuffle=True,
            collate_fn=c_segments.collate,
            # sampler=  TODO: implement the staleness sampler for identical behavior to the existing version
        )
        c_iter = infinite_loader(c_loader)

        if len(tokenizer_loader) == 0:
            return

        def prefetch_tok():
            batch = batch_to_tensordict(next(tokenizer_iter), device='cuda', include_all_observations=True)
            batch = batch['observation']
            batch = batch['image|features'][:, 0]
            buffer.append(batch)

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



