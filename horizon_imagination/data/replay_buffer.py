import time
from typing import Literal
import torch
from torch.utils.data import DataLoader

import episodata as ed
from episodata.utils import batch_to_tensordict
from collections import deque
from loguru import logger
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

        # NOTE(diagnostics): temporary timing instrumentation to localize the
        # between-epoch stall on the zarr backend — remove once root-caused.
        # persistent_workers is deliberately NOT used here: this DataLoader
        # is rebuilt from scratch every epoch (required, since the replay
        # buffer keeps growing with newly collected online episodes), and
        # persistent workers are forked once and never see later refresh()
        # calls, so they'd keep sampling a stale, frozen index — silently
        # invisible to newly collected data. num_workers>0 without
        # persistent_workers still parallelizes reads within one epoch's
        # burst, just re-forked fresh each epoch.
        t0 = time.perf_counter()
        num_workers = self.config.prefetch
        worker_kwargs = dict(num_workers=num_workers)

        segments_dataset = self.config.replay_buffer.segments(sequence_length=1, fields=['image|features'])
        if len(segments_dataset) == 0:
            return
        tokenizer_loader = DataLoader(
            dataset=segments_dataset,
            batch_size=self.config.tokenizer_batch_size,
            shuffle=True,
            collate_fn=segments_dataset.collate,
            # sampler=  TODO: implement the staleness sampler for identical behavior to the existing version
            **worker_kwargs,
        )
        tokenizer_iter = infinite_loader(tokenizer_loader)

        wm_segments = self.config.replay_buffer.segments(sequence_length=self.config.wm_segment_length)
        wm_loader = DataLoader(
            dataset=wm_segments,
            batch_size=self.config.wm_batch_size,
            shuffle=True,
            collate_fn=wm_segments.collate,
            # sampler=  TODO: implement the staleness sampler for identical behavior to the existing version
            **worker_kwargs,
        )
        wm_iter = infinite_loader(wm_loader)

        c_segments = self.config.replay_buffer.segments(sequence_length=self.config.c_segment_length, pad='prefix')
        c_loader = DataLoader(
            dataset=c_segments,
            batch_size=self.config.c_batch_size,
            shuffle=True,
            collate_fn=c_segments.collate,
            # sampler=  TODO: implement the staleness sampler for identical behavior to the existing version
            **worker_kwargs,
        )
        c_iter = infinite_loader(c_loader)

        if len(tokenizer_loader) == 0:
            return

        t_setup = time.perf_counter()
        logger.info(
            f"[EpochDataIterator] dataset/loader setup: {t_setup - t0:.3f}s "
            f"(tokenizer segments={len(segments_dataset)}, wm segments={len(wm_segments)}, "
            f"c segments={len(c_segments)}, num_workers={num_workers})"
        )

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

        t_first_tok = None
        for i in range(self.config.tokenizer_steps):
            prefetch_tok()
            if i == 0:
                t_first_tok = time.perf_counter()
        t_tok = time.perf_counter()
        logger.info(
            f"[EpochDataIterator] tokenizer prefetch: {t_tok - t_setup:.3f}s total over "
            f"{self.config.tokenizer_steps} batches "
            f"(first batch: {(t_first_tok - t_setup) if t_first_tok else 0:.3f}s, "
            f"rest: {(t_tok - t_first_tok) if t_first_tok else 0:.3f}s)"
        )

        t_first_wm = None
        for i in range(self.config.world_model_steps):
            prefetch_wm()
            if i == 0:
                t_first_wm = time.perf_counter()
        t_wm = time.perf_counter()
        logger.info(
            f"[EpochDataIterator] world_model prefetch: {t_wm - t_tok:.3f}s total over "
            f"{self.config.world_model_steps} batches "
            f"(first batch: {(t_first_wm - t_tok) if t_first_wm else 0:.3f}s, "
            f"rest: {(t_wm - t_first_wm) if t_first_wm else 0:.3f}s)"
        )

        t_first_c = None
        for i in range(self.config.controller_steps):
            prefetch_c()
            if i == 0:
                t_first_c = time.perf_counter()
        t_c = time.perf_counter()
        logger.info(
            f"[EpochDataIterator] controller prefetch: {t_c - t_wm:.3f}s total over "
            f"{self.config.controller_steps} batches "
            f"(first batch: {(t_first_c - t_wm) if t_first_c else 0:.3f}s, "
            f"rest: {(t_c - t_first_c) if t_first_c else 0:.3f}s)"
        )
        logger.info(f"[EpochDataIterator] TOTAL between-epoch setup+prefetch: {t_c - t0:.3f}s")

        for i in range(self.config.tokenizer_steps):
            yield buffer.popleft(), 0

        for i in range(self.config.world_model_steps):
            yield buffer.popleft(), 1

        for i in range(self.config.controller_steps):
            yield buffer.popleft(), 2



