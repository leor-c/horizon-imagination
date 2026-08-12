import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sym_log(x: Tensor, order: float = 1) -> Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x) ** (1 / order))


def sym_exp(x: Tensor, order: float = 1) -> Tensor:
    return torch.sign(x) * ((torch.exp(torch.abs(x)) - 1) ** order)
    

class RegressionHead(nn.Module):

    def __init__(
            self, 
            in_features: int, 
            sym_log_normalize: bool = False,
            sym_exp_order: float = 2,
            device=None, 
            dtype=None,
            *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sym_log_normalize = sym_log_normalize
        self.sym_exp_order = sym_exp_order
        self.linear = nn.Linear(in_features, 1, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        out_raw = self.linear(x).squeeze(-1)
        out = out_raw
        if self.sym_log_normalize:
            out = sym_exp(out_raw, order=self.sym_exp_order)

        return out, out_raw

    def compute_loss(self, x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
        assert x.numel() == target.numel(), f"{x.shape} != {target.shape}"
        if self.sym_log_normalize:
            target = sym_log(target, order=self.sym_exp_order)

        loss = F.mse_loss(x.flatten(), target.flatten().to(dtype=x.dtype), reduction=reduction)
        return loss


class TwoHotRegressionHead(nn.Module):
    """
    Discrete regression: a distribution over fixed bins in symlog space, trained by
    cross-entropy against a two-hot target, read out as its expectation.

    Replaces scalar MSE for value prediction. Regressing a scalar forces one output to
    cover returns spanning several orders of magnitude, and the squared error is
    dominated by whichever targets are currently largest. Predicting *which bin* instead
    makes the loss scale-free -- every target contributes a comparable gradient -- and
    lets the head represent a multi-modal belief rather than being dragged to the mean of
    one. DreamerV3 and TD-MPC2 arrived at this independently.

    Two-hot encoding splits a target between its two neighbouring bins in proportion to
    its distance from each, so the expectation of the target distribution is exactly the
    target: the discretization biases nothing, it only limits resolution.

    A useful side effect: the readout is a convex combination of fixed bin centres, so
    the predicted value is bounded by `sym_exp(v_max)` by construction. A free scalar
    head has no such bound, and `sym_exp` turns a drifting output into an exponentially
    large value.

    Bins are uniform in symlog space, which spends resolution where the values are: fine
    near zero, coarse in the tails.
    """

    def __init__(
            self,
            in_features: int,
            num_bins: int = 129,
            v_min: float = -20.0,
            v_max: float = 20.0,
            sym_exp_order: float = 1,
            device=None,
            dtype=None,
            *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert num_bins >= 2, f"Need at least two bins, got {num_bins}."
        assert v_min < v_max, f"Empty bin range [{v_min}, {v_max}]."
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.sym_exp_order = sym_exp_order

        self.linear = nn.Linear(in_features, num_bins, device=device, dtype=dtype)
        # A buffer, not a constant: it must follow the module across devices and dtypes.
        self.register_buffer(
            'bins', torch.linspace(v_min, v_max, num_bins, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """`(value, logits)` -- logits keep the trailing bin axis for `compute_loss`."""
        logits = self.linear(x)
        expected_symlog = (logits.softmax(dim=-1) * self.bins).sum(dim=-1)

        return sym_exp(expected_symlog, order=self.sym_exp_order), logits

    def two_hot(self, symlog_target: Tensor) -> Tensor:
        """(...) in symlog space -> (..., num_bins) summing to 1 along the last axis."""
        y = symlog_target.clamp(self.v_min, self.v_max)

        upper = torch.bucketize(y, self.bins, right=True).clamp(1, self.num_bins - 1)
        lower = upper - 1
        lower_edge, upper_edge = self.bins[lower], self.bins[upper]
        # Bins are uniformly spaced, so the denominator is never zero.
        w_upper = (y - lower_edge) / (upper_edge - lower_edge)

        target = torch.zeros(*y.shape, self.num_bins, device=y.device, dtype=y.dtype)
        target.scatter_(-1, upper.unsqueeze(-1), w_upper.unsqueeze(-1))
        target.scatter_add_(-1, lower.unsqueeze(-1), (1.0 - w_upper).unsqueeze(-1))

        return target

    def compute_loss(self, x: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
        """Cross-entropy between the predicted bin distribution and the two-hot target."""
        assert x.shape[:-1] == target.shape, \
            f"logits {x.shape} do not carry a bin axis over targets {target.shape}"

        two_hot = self.two_hot(sym_log(target.to(dtype=x.dtype), order=self.sym_exp_order))
        loss = -(two_hot * x.log_softmax(dim=-1)).sum(dim=-1)

        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
        return loss
