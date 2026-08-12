import torch
import torch.nn as nn
from torch import Tensor


class EMAScaler(nn.Module):
    """
    An EMA of a symmetric quantile band, used to normalize advantages by the spread of
    the lambda-returns.

    An `nn.Module` holding buffers rather than plain attributes, so that the estimate
    is carried by `state_dict()` and survives a checkpoint. It takes a while to warm up
    -- with the default decay of 0.005 the time constant is ~200 updates -- so a run
    resumed with a cold scaler spends its first epochs normalizing advantages by
    whatever spread a single batch happened to show.

    Instances held by `LatentReconstructionResidual` live in a plain dict, outside any
    `nn.Module`, and so are never moved by `.to(device)`. That is why the first update
    *assigns* to `.data` (adopting the incoming device and dtype) rather than copying
    into a buffer that may still be on the CPU.
    """

    def __init__(self, decay: float = 0.001, quantile: float = 0.95):
        super().__init__()
        self.decay = decay
        self.quantile = quantile
        self.register_buffer('_estimate_high', torch.zeros(()))
        self.register_buffer('_estimate_low', torch.zeros(()))
        self.register_buffer('_initialized', torch.zeros((), dtype=torch.bool))

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)

    @property
    def estimate_high(self):
        return self._estimate_high if self.initialized else None

    @property
    def estimate_low(self):
        return self._estimate_low if self.initialized else None

    @property
    def scale(self):
        assert self.initialized, "EMAScaler.scale read before the first update()."
        return self._estimate_high - self._estimate_low

    @torch.no_grad()
    def update(self, values: Tensor):
        high = torch.quantile(values, self.quantile)
        low = torch.quantile(values, 1 - self.quantile)

        if not self.initialized:
            self._estimate_high.data = high.detach().clone()
            self._estimate_low.data = low.detach().clone()
            self._initialized.data = torch.ones_like(self._initialized)
            return

        # lerp_(end, w) == (1 - w) * self + w * end -- the same EMA as before.
        self._estimate_high.lerp_(high.to(self._estimate_high), self.decay)
        self._estimate_low.lerp_(low.to(self._estimate_low), self.decay)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """
        Accept checkpoints written before the scaler carried state.

        Those files have none of these keys, and `load_from_checkpoint` loads with
        `strict=True`, so without this a mid-run checkpoint could no longer be resumed.
        Dropping the keys from `missing_keys` leaves the scaler uninitialized, which is
        exactly what it was on a fresh start -- it re-estimates on the next update.
        """
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        for name in ('_estimate_high', '_estimate_low', '_initialized'):
            key = prefix + name
            if key in missing_keys:
                missing_keys.remove(key)


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
