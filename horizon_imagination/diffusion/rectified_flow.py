import torch
import torch.nn as nn
from torch import Tensor
from tensordict.tensordict import TensorDict

from horizon_imagination.diffusion.base import DiffusionBase, TimeSamplerBase, DenoiserBase
from horizon_imagination.diffusion.noise import NoiseBase, UniformNoise, GaussianNoise
from horizon_imagination.utilities import MaskedMSELoss


def _preprocess_time_steps(time_steps: Tensor, target_shape):
    if time_steps.dim() < len(target_shape):
        assert time_steps.shape == target_shape[:time_steps.dim()], \
            f"{time_steps.shape} != {target_shape[:time_steps.dim()]}"
        new_dims = len(target_shape) - time_steps.dim()
        time_steps = time_steps.view(*time_steps.shape, *([1] * new_dims)).expand(*target_shape)
        time_steps
    else:
        assert time_steps.shape == target_shape, f"{time_steps.shape} != {target_shape}"

    return time_steps


class RectifiedFlow(DiffusionBase):
    def __init__(self, time_sampler: TimeSamplerBase, sigma_min: float = 1e-5, *args, **kwargs):
        super().__init__(time_sampler, *args, **kwargs)
        self.sigma_min = sigma_min
        self.noise = UniformNoise(-1, 1)

    def training_step(
            self, 
            x_clean: TensorDict, 
            time_dims: int, 
            denoiser: DenoiserBase, 
            denoiser_kwargs: dict,
            mask: Tensor = None,
            **kwargs,
        ) -> Tensor:
        t = self.time_sampler.sample(x_clean.shape[:time_dims], device=x_clean.device, dtype=x_clean.dtype)
        x_t, noise = self.get_noisy_samples(x_clean, t)
        
        denoiser_outputs, state = denoiser(x_t, t, **denoiser_kwargs)

        v_t = x_clean - noise

        loss_fn = MaskedMSELoss()
        loss = torch.sum(torch.stack([
            loss_fn(denoiser_outputs[k].float(), v_t[k].float(), mask=mask) 
            for k in x_clean.keys()
        ]))

        return loss, x_t

    def get_noisy_samples(self, x_clean: Tensor, time_steps: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        """
        Implement simple linear combination x_t = t * x_1 + (1-t) * eps.
        :param x_clean:
        :param time_steps: a tensor with a shape that is a prefix of the shape of x_clean.
        Every missing dimension w.r.t x_clean will be broadcasted. Values must be in [0, 1].
        :param kwargs:
        :return: tuple: (x_t, noise)
        """
        time_steps = _preprocess_time_steps(time_steps, x_clean.shape)

        noise = self.noise.sample_like(x_clean)

        t = time_steps
        x_t = t * x_clean + (1 - t) * noise

        return x_t, noise
