from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn.functional as F


class TimeSamplerBase(ABC):
    @abstractmethod
    def sample(self, shape: tuple[int, ...], device=None, dtype=None) -> Tensor:
        """
        Sample time values for training the diffusion model.
        :return:
        """
        pass

    def sample_like(self, other: Tensor) -> Tensor:
        return self.sample(other.shape, device=other.device, dtype=other.dtype)


class UniformTimeSampler(TimeSamplerBase):
    def sample(self, shape: tuple[int, ...], device=None, dtype=None) -> Tensor:
        return torch.rand(*shape, device=device, dtype=dtype)


class LogitNormalTimeSampler(TimeSamplerBase):
    """
    Implementing the logit-normal sampler proposed in "Scaling Rectified Flow Transformers
    for High-Resolution Image Synthesis" https://arxiv.org/pdf/2403.03206
    """
    def sample(self, shape: tuple[int, ...], device=None, dtype=None) -> Tensor:
        return F.sigmoid(torch.randn(shape, device=device, dtype=dtype))


class BetaTimeSampler(TimeSamplerBase):
    def __init__(self, alpha: float = 1, beta: float = 1):
        """
        The Beta distribution generalizes the uniform and logit-normal samplers.
        Values of alpha = beta = 1 results in a uniform distribution.
        Values of alpha = beta = pi approximates the logit-normal sampler.
        In addition, one can define non symetric distributions over [0,1]
        by using alpha > beta or alpha < beta.
        """
        super().__init__()
        assert alpha > 0 and beta > 0, f"alpha and beta must be positive!"
        self.alpha = alpha
        self.beta = beta
        self.d = torch.distributions.Beta(alpha, beta)

    def sample(self, shape, device=None, dtype=None):
        ones = torch.ones(*shape, device=device, dtype=dtype)
        beta_dist = torch.distributions.Beta(ones * self.alpha, ones * self.beta)
        return beta_dist.sample()
    

class HybridTimeSampler(TimeSamplerBase):
    """
    Combines in each batch a fraction of samples with
    uniform time, and a fraction where the context is clean.
    The context length is also randomly chosen between 1 and 
    horizon / 2.
    This strategy should help at inference, where clean context is available.
    Also, this would allow us to use the same model for the sequential baseline.
    """
    def __init__(self, base_time_sampler: TimeSamplerBase, prob_clean_context: float = 0.2):
        super().__init__()
        self.prob_clean_context = prob_clean_context
        self.base_time_sampler = base_time_sampler

    def sample(self, shape, device=None, dtype=None):
        sample = self.base_time_sampler.sample(shape=shape, device=device, dtype=dtype)

        # select samples to clean their context:
        clean_ctx_samples = torch.rand(sample.shape[0], device=device) < self.prob_clean_context
        clean_ctx_samples = clean_ctx_samples[:, None].expand(*sample.shape[:2])

        # select context lengths:
        ctx_len = torch.randint(1, int(sample.shape[1]*0.7)+1, (sample.shape[0], 1), device=device)
        ctx_len = ctx_len.expand(*sample.shape[:2])
        indices = torch.arange(sample.shape[1], device=device)[None, :].expand(*sample.shape[:2])
        mask = indices <= ctx_len

        mask = torch.logical_and(clean_ctx_samples, mask)
        sample[mask] = 1.0  # fully clean

        return sample
    

class HorizonTimeSampler(TimeSamplerBase):
    def __init__(self, max_horizon: float = 32):
        """
        :param max_horizon: upper bound on the effective number of frames generated
        in parallel. The slope would be sampled s.t. it satisfies this constraint.
        This should be equal to the horizon (num of frames) generated during
        training.
        """
        super().__init__()
        self.max_width = max_horizon
        self.min_width = 1

    def sample(self, shape, device=None, dtype=None):
        assert len(shape) == 2, f"got {shape}"
        slopes = self.min_width + torch.rand(shape[0], 1, device=device) * (self.max_width - self.min_width)
        bias = torch.rand(shape[0], 1, device=device) * (1 + (self.max_width - 1) / slopes)
        times = (-1/slopes) * torch.arange(shape[1], device=device).unsqueeze(0) + bias
        times = times.clamp(0, 1)

        times[torch.where(times < 1e-6)] = torch.rand_like(times[torch.where(times < 1e-6)])

        return times
