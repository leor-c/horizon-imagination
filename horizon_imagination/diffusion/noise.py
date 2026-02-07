from abc import ABC, abstractmethod
import torch


class NoiseBase(ABC):
    @abstractmethod
    def sample(self, shape, device=None, dtype=None):
        pass
    
    @abstractmethod
    def sample_like(self, x):
        pass


class GaussianNoise(NoiseBase):
    def sample(self, shape, device=None, dtype=None):
        return torch.randn(*shape, device=device, dtype=dtype)
    
    def sample_like(self, x):
        return torch.randn_like(x)
    

class UniformNoise(NoiseBase):
    def __init__(self, low: float = 0, high: float = 1):
        super().__init__()
        self.low = low
        self.high = high
    
    def _to_interval(self, u):
        return u *(self.high - self.low) + self.low

    def sample(self, shape, device=None, dtype=None):
        return self._to_interval(torch.rand(*shape, device=device, dtype=dtype))
    
    def sample_like(self, x):
        return self._to_interval(torch.rand_like(x))
