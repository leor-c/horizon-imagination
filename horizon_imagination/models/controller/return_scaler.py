import torch
from torch import Tensor


class EMAScaler:
    def __init__(self, decay: float = 0.001, quantile: float = 0.95):
        self.decay = decay
        self.quantile = quantile
        self._estimate_high = None
        self._estimate_low = None

    @property
    def estimate_high(self):
        return self._estimate_high

    @property
    def estimate_low(self):
        return self._estimate_low

    @property
    def scale(self):
        return self.estimate_high - self.estimate_low

    def update(self, values: Tensor):
        high = torch.quantile(values, self.quantile)
        low = torch.quantile(values, 1 - self.quantile)

        if self._estimate_high is None or self._estimate_low is None:
            self._estimate_high = high
            self._estimate_low = low
        else:
            self._estimate_high = self.decay * high + (1 - self.decay) * self._estimate_high
            self._estimate_low = self.decay * low + (1 - self.decay) * self._estimate_low


class BufferScaler:
    def __init__(
            self,
            quantile: float = 0.975,
            window_size_limit: int = 500
    ):
        self.quantile = quantile
        self.window_size_limit = window_size_limit
        self._buffer = None
        self._estimate_high = None
        self._estimate_low = None

    @property
    def estimate_high(self):
        return self._estimate_high

    @property
    def estimate_low(self):
        return self._estimate_low

    @property
    def scale(self):
        return self.estimate_high - self.estimate_low

    def update(self, values: Tensor):
        values = values.detach().clone().unsqueeze(0)
        if self._buffer is None:
            self._buffer = values
        else:
            self._buffer = torch.cat((self._buffer[-self.window_size_limit:], values), dim=0)

        high = torch.quantile(self._buffer, self.quantile)
        low = torch.quantile(self._buffer, 1 - self.quantile)

        self._estimate_high = high
        self._estimate_low = low
