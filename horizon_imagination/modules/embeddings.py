import math

import torch
import torch.nn as nn
from torch import Tensor


class NoiseLevelEmbedding(nn.Module):
    """
    Embeds a per-frame scalar noise level (diffusion time `t`) into a `dim`-wide
    vector: a standard log-scaled sinusoidal (Fourier) embedding followed by a
    2-layer MLP projection, mirroring the timestep-conditioning pattern used by
    the world model's DiT denoiser.
    """

    def __init__(self, dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(dim, dim, device=device, dtype=dtype),
        )

    def _sinusoidal_embedding(self, noise_level: Tensor) -> Tensor:
        assert noise_level.ndim == 2, f"Expected (B, T), got {noise_level.shape}"
        in_dtype = noise_level.dtype
        t = noise_level.flatten().float()
        half_dim = self.dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=t.device)
        exponent = exponent / half_dim
        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb.to(dtype=in_dtype).view(*noise_level.shape, self.dim)

    def forward(self, noise_level: Tensor) -> Tensor:
        return self.mlp(self._sinusoidal_embedding(noise_level))
