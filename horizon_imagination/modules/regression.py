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
    

