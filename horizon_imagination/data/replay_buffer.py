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
        #
        # persistent_workers=True is safe here even though it's usually
        # dangerous with a growing dataset: tokenizer_loader/wm_loader/
        # c_loader are brand new local objects built fresh every __iter__
        # call (every epoch), each wrapping a `segments()` snapshot taken
        # at that same moment, so nothing ever survives across epochs to
        # go stale. What persistent_workers actually fixes: *_steps_per_epoch
        # (e.g. tokenizer_steps=300) is typically several times larger than
        # one loader pass (segments // batch_size, e.g. ~57 for 1800
        # segments / batch 32), so infinite_loader's `while True: yield from
        # loader` restarts the same DataLoader ~5x within a single epoch.
        # Without persistent_workers, torch spawns a fresh worker pool on
        # every one of those restarts — paying full worker-startup cost
        # (forking a process with a loaded model + CUDA context + wandb
        # threads is not cheap) repeatedly per epoch, which was silently
        # eating the read-parallelism gain num_workers was supposed to buy.
        t0 = time.perf_counter()
        num_workers = self.config.prefetch
        worker_kwargs = dict(num_workers=num_workers, persistent_workers=num_workers > 0)

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

        def _wraps(steps, loader):
            n = len(loader)
            return f"{steps / n:.2f}x" if n else "n/a"

        logger.info(
            f"[EpochDataIterator] dataset/loader setup: {t_setup - t0:.3f}s "
            f"(tokenizer segments={len(segments_dataset)} batches/pass={len(tokenizer_loader)} "
            f"wraps={_wraps(self.config.tokenizer_steps, tokenizer_loader)}; "
            f"wm segments={len(wm_segments)} batches/pass={len(wm_loader)} "
            f"wraps={_wraps(self.config.world_model_steps, wm_loader)}; "
            f"c segments={len(c_segments)} batches/pass={len(c_loader)} "
            f"wraps={_wraps(self.config.controller_steps, c_loader)}; "
            f"num_workers={num_workers})"
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



