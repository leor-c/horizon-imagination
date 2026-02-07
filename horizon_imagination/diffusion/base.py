from abc import ABC, abstractmethod
from typing import Union, Any

import torch.nn as nn
from torch import Tensor
from tensordict.tensordict import TensorDict

from horizon_imagination.diffusion.time_sampler import TimeSamplerBase


class DenoiserBase(ABC, nn.Module):
    @abstractmethod
    def denoise(
        self, 
        x: Union[Tensor, TensorDict], 
        t: Tensor, 
        state: Any = None, 
        *args, 
        **kwargs
    ) -> tuple[Union[Tensor, TensorDict], Any]:
        pass

    def forward(
            self, 
            x: Union[Tensor, TensorDict], 
            t: Tensor, 
            state: Any = None, 
            **kwargs
        ) -> tuple[Union[Tensor, TensorDict], Any]:
        return self.denoise(
            x=x, 
            t=t, 
            state=state, 
            **kwargs
        )


class DiffusionBase(ABC):
    def __init__(self, time_sampler: TimeSamplerBase, *args, **kwargs):
        super().__init__()
        self.time_sampler = time_sampler

    @abstractmethod
    def training_step(
        self, 
        x_clean: Tensor, 
        time_dims: int, 
        denoiser: DenoiserBase, 
        denoiser_kwargs: dict,
        **kwargs,
    ) -> Tensor:
        """
        Compute the loss for the training step.
        :param x_clean: a batch of samples from the data distribution.
        :param time_dims: number of time dimensions. An independent time sample will be
        generated for each element within the first `time_dims` dimensions of x_clean.
        :param denoiser: A denoiser model inheriting from DenoiserBase.
        :param denoiser_kwargs: Keyword arguments passed to the denoiser model. Any additional
        conditioning signals should be passed here.
        :return: objective loss (Tensor)
        """
        pass

    @abstractmethod
    def get_noisy_samples(self, x_clean: Tensor, time_steps: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """
        Get x_t, the noisy samples. Different methods use different ways to combine the data
        and the noise.
        :param x_clean:
        :param time_steps:
        :return: tuple: (x_t, noise). noise is x_1 or x_0 depending on the notation.
        """
        pass
