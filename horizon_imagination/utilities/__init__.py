from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor
from loguru import logger
import contextlib

from horizon_imagination.utilities.types import *


@dataclass
class AdamWConfig:
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


class MaskedMSELoss:
    def __init__(self):
        self.mse_loss = nn.MSELoss()

    def __call__(self, x: Tensor, target: Tensor, mask: Tensor = None):
        if mask is None: 
            return self.mse_loss(x, target)
        
        if mask.dim() < x.dim():
            mask = mask.reshape(*mask.shape, *([1]*(x.dim()-mask.dim())))
            
        return self.mse_loss(x*mask, target*mask)
    

def shift_fwd(x: Tensor) -> Tensor:
    """
    Returns a shifted copy of the given Tensor one step forward, zeroing the first step.
    This is used for shifting actions, rewards, etc one step forward
    to maintain causality.
    """
    assert x.dim() >= 2
    shifted = torch.zeros_like(x)
    shifted[:, 1:] = x[:, :-1]
    return shifted


@contextlib.contextmanager
def cuda_mem_delta(tag=""):
    torch.cuda.synchronize()
    start_alloc = torch.cuda.memory_allocated()
    start_rsrv  = torch.cuda.memory_reserved()
    yield
    torch.cuda.synchronize()
    end_alloc = torch.cuda.memory_allocated()
    end_rsrv  = torch.cuda.memory_reserved()
    logger.info(f"[{tag}] Δallocated: {(end_alloc-start_alloc)/1e6:.1f} MB  "
          f"Δreserved: {(end_rsrv-start_rsrv)/1e6:.1f} MB")
