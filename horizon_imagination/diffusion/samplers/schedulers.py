from typing import Union, Callable, Literal
import torch
from torch import Tensor
from horizon_imagination.diffusion.samplers.base import SamplerScheduler, dataclass


def expand_to_shape(x: Tensor, shape):
    if x.dim() < len(shape):
        x = x.reshape(*x.shape, *[1 for _ in shape[x.dim():]])
    x = x.expand(*shape)
    return x


class UniformSamplerScheduler(SamplerScheduler):

    def __init__(self, config: SamplerScheduler.Config):
        super().__init__(config=config)
        self.device = config.device
        self.dtype = config.dtype

        self._schedule = None
        self.dt = None
        self.step = None
        self.batch_size = None

    @property
    def schedule(self):
        return self._schedule

    def reset(self, num_steps: int, batch_size: Union[int, tuple[int, ...]]) -> None:
        assert num_steps >= 1
        # We want num_steps excluding 1 (no need to denoise at 1):
        self._schedule = torch.linspace(0, 1, num_steps+1, device=self.device)
        self.dt = self._schedule[1] - self._schedule[0]
        self.step = 0
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        t = self.schedule[self.step]
        if torch.all(t == torch.ones_like(t)):
            raise StopIteration()
        
        self.step += 1
        return expand_to_shape(t.to(dtype=self.dtype), self.batch_size), self.dt
    

class DiffusionForcingSamplerScheduler(SamplerScheduler):
    @dataclass
    class Config(SamplerScheduler.Config):
        diffusion_time_scheduler: SamplerScheduler.Config = None

    def __init__(self, config: Config):
        """
        :param diffusion_time_scheduler: determines the values of diffusion time
        used for denoising each temporal step. E.g., uniform 0, 0.1, 0.2,..., 1.
        This can also be non-uniform, as in the linear-quadratic above.
        """
        super().__init__(config=config)
        self.diffusion_time_scheduler = config.diffusion_time_scheduler.make_instance()
        self.horizon = None
        self.device = config.device
        self.dtype = config.dtype
        self._schedule = None
        self._index_schedule = None
        self.batch_shape = None
        self.step = None

    def _build_index_schedule(self, num_denoising_levels: int, horizon: int) -> Tensor:
        diffusion_indices = torch.arange(num_denoising_levels, device=self.device)

        padded = torch.cat([
            diffusion_indices[0].expand(horizon - 1),
            diffusion_indices,
            diffusion_indices[-1].expand(horizon - 1)
        ])

        # Create the matrix by stacking shifted slices
        index_schedule = torch.stack([
            padded[i:i + num_denoising_levels + horizon - 1] for i in reversed(range(horizon))
        ], dim=1)

        return index_schedule

    @property
    def schedule(self):
        return self._schedule

    def reset(self, num_steps, batch_size):
        assert len(batch_size) == 2, f"Got {batch_size}"
        self.batch_shape = batch_size
        self.horizon = batch_size[1]
        self.step = 0

        num_denoising_levels = num_steps + 1 - self.horizon + 1
        self.diffusion_time_scheduler.reset(num_steps=num_denoising_levels - 1, batch_size=batch_size)
        diffusion_schedule = self.diffusion_time_scheduler.schedule
        assert diffusion_schedule.dim() == 1, f"Got {diffusion_schedule.shape}"
        
        self._index_schedule = self._build_index_schedule(num_denoising_levels, self.horizon)
        self._schedule = diffusion_schedule[self._index_schedule]

    def __call__(self, *args, **kwargs):
        t = self.schedule[self.step]
        if torch.all(t == torch.ones_like(t)):
            raise StopIteration()
        
        dt = self.schedule[self.step + 1] - t
        self.step += 1
        
        t = (t[None, :]).expand(self.batch_shape).to(dtype=self.dtype)
        dt = dt[None, :].expand(self.batch_shape).to(dtype=self.dtype)
        return t, dt
    

class HorizonSamplerScheduler(SamplerScheduler):
    @dataclass
    class Config(SamplerScheduler.Config):
        decay_horizon: int = 4
        time_transform: Callable = None  # a monotonically increasing function f: [0, 1] --> [0, 1]

    def __init__(self, config):
        super().__init__(config)
        self.horizon = None
        self.device = config.device
        self.dtype = config.dtype
        self._schedule = None
        self.batch_shape = None
        self.step = None

        time_transform = config.time_transform
        self.time_transform = time_transform

    @property
    def schedule(self):
        return self._schedule

    def reset(self, num_steps, batch_size):
        assert len(batch_size) == 2, f"Got {batch_size}"
        self.batch_shape = batch_size
        self.horizon = batch_size[1]
        self.step = 0

        self._schedule = self._build_schedule(budget=num_steps)

    def _build_schedule(self, budget: int):
        assert self.horizon is not None
        decay_horizon = self.config.decay_horizon
        slope = -1 / decay_horizon
        bias = 1 - slope * (self.horizon - 1)

        seq_time_idx = torch.arange(self.horizon, device=self.device, dtype=torch.float32).unsqueeze(0)
        denoising_idx = torch.arange(budget + 1, device=self.device, dtype=torch.float32).unsqueeze(1)

        schedule = ((slope * seq_time_idx) + ((bias * denoising_idx) / budget)).clamp(0, 1).to(dtype=self.dtype)

        if self.time_transform is not None:
            schedule = self.time_transform(schedule).clamp(0, 1)
        
        return schedule
    
    def __call__(self, *args, **kwargs):
        t = self.schedule[self.step]
        if torch.all(t == torch.ones_like(t)):
            raise StopIteration()
        
        dt = self.schedule[self.step + 1] - t
        self.step += 1
        
        t = (t[None, :]).expand(self.batch_shape).to(dtype=self.dtype)
        dt = dt[None, :].expand(self.batch_shape).to(dtype=self.dtype)
        return t, dt
