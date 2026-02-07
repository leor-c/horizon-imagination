import torch
from tensordict.tensordict import TensorDict
from horizon_imagination.diffusion.samplers.base import DiffusionSamplerBase


class EulerSampler(DiffusionSamplerBase):
    def sample(self, x: TensorDict, num_steps: int, denoiser_kwargs: dict = None, *args, **kwargs):
        if denoiser_kwargs is None:
            denoiser_kwargs = {}

        self.scheduler.reset(num_steps=num_steps, batch_size=x.batch_size)
        for _ in range(num_steps):
            t, dt = self.scheduler()
            # TODO: can be more efficient to only pass elements where dt > 0 
            v_t = self.denoiser(x, t, **denoiser_kwargs)
            x = x + v_t * dt
        return x
