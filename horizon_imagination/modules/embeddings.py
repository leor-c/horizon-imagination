import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def action_dim(action_space: gym.Space) -> int:
    """The width of the action vector a `Box` action space produces."""
    assert isinstance(action_space, gym.spaces.Box), f"Got {action_space}"
    return int(np.prod(action_space.shape))


def build_action_embedder(
        action_space: gym.Space,
        embed_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
) -> nn.Module:
    """
    Build the module that maps a batch of actions to `embed_dim`-wide vectors.

    Discrete actions, shape (B, T), are looked up in an embedding table; continuous
    (`Box`) actions, shape (B, T, A), are projected linearly. Either way the output
    is (B, T, embed_dim), which is what every call site expects.
    """
    if isinstance(action_space, gym.spaces.Discrete):
        embedder = nn.Embedding(action_space.n, embed_dim, device=device, dtype=dtype)

        std = 1.0 / math.sqrt(embed_dim)
        torch.nn.init.trunc_normal_(embedder.weight, std=std, a=-3 * std, b=3 * std)

        return embedder

    if isinstance(action_space, gym.spaces.Box):
        return nn.Linear(action_dim(action_space), embed_dim, device=device, dtype=dtype)

    # TODO: support more action modalities
    raise NotImplementedError(f"Currently action space {action_space} is not supported.")


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
