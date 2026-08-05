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


class RunningReturnStdScaler:
    """
    Normalizes by a running estimate of the standard deviation of discounted returns, as
    proposed in Random Network Distillation (Burda et al., 2018) for scaling intrinsic
    rewards: rather than normalizing the *reward* itself (which would remove the signal of
    "how novel is this transition"), it tracks the variance of the discounted sum of rewards
    and only rescales by that -- no centering, since intrinsic rewards should stay positive.

    update() is called once per rollout with the raw per-step reward, shape (B, T); the
    discounted return is computed forward (within each rollout) and its running mean/var are
    updated via Welford's parallel-batch formula.
    """
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-4):
        self.gamma = gamma
        self._mean = None
        self._var = None
        self._count = epsilon

    @property
    def scale(self):
        if self._var is None:
            return torch.tensor(1.0)
        return torch.sqrt(self._var)

    def update(self, rewards: Tensor):
        B, T = rewards.shape
        returns = torch.empty_like(rewards)
        running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
        for t in range(T):
            running = running * self.gamma + rewards[:, t]
            returns[:, t] = running
        batch = returns.flatten()

        batch_mean = batch.mean()
        batch_var = batch.var(unbiased=False)
        batch_count = batch.numel()

        if self._mean is None:
            self._mean = batch_mean
            self._var = batch_var
            self._count = batch_count
        else:
            delta = batch_mean - self._mean
            tot_count = self._count + batch_count

            new_mean = self._mean + delta * batch_count / tot_count
            m_a = self._var * self._count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta ** 2 * self._count * batch_count / tot_count

            self._mean = new_mean
            self._var = m2 / tot_count
            self._count = tot_count


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
