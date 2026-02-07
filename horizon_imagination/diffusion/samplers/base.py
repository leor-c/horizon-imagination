from abc import ABC, abstractmethod
from typing import Union
import torch
import torch.nn as nn
from torch import Tensor
from tensordict.tensordict import TensorDict

from horizon_imagination.diffusion.base import DenoiserBase
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass


class SamplerScheduler(ABC, Configurable):
    @dataclass
    class Config(BaseConfig):
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    @abstractmethod
    def reset(self, num_steps: int, batch_size: Union[int, tuple[int, ...]]) -> None:
        """
        Resets the scheduler state.
        :param num_steps: the total number of diffusion steps.
        :param batch_size: the number of examples in the batch.
        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        """
        return (current time value, dt) where dt = next_time_value - current_time_value 
        The first time value is usually zero.
        Last call should return last time value (usually 1) and dt = 0.
        """
        pass

    @property
    @abstractmethod
    def schedule(self):
        pass


class DiffusionSamplerBase(ABC):
    def __init__(
            self, 
            denoiser: DenoiserBase, 
            scheduler: SamplerScheduler, 
            *args, 
            **kwargs
        ):
        super().__init__()
        self.denoiser = denoiser
        self.scheduler = scheduler

    @abstractmethod
    def sample(self, x, num_steps: int, *args, **kwargs):
        pass



