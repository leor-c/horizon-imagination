from math import sqrt

import torch
import torch.nn as nn
from einops import rearrange

from horizon_imagination.modules.transform.base import BaseTransform
from horizon_imagination.utilities.config import Configurable, dataclass, BaseConfig


class VectorTokenTransform(BaseTransform, nn.Module, Configurable):
    """
    Maps a vector latent to DiT tokens and back -- the ``ImagePatcherTransform``
    analogue for the vector modality.

    ``(B, T, latent_dim, P, 1) <-> (B, T, P, model_channels)``: one token per latent
    "patch", so a key with ``P > 1`` contributes ``P`` tokens per frame with no
    change to the surrounding plumbing.
    """

    @dataclass
    class Config(BaseConfig):
        latent_dim: int = 32
        out_channels: int = 512
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.fwd_proj = nn.Linear(
            config.latent_dim, config.out_channels, bias=False,
            device=config.device, dtype=config.dtype,
        )
        std = 1.0 / sqrt(config.latent_dim)
        torch.nn.init.trunc_normal_(self.fwd_proj.weight, std=std, a=-3 * std, b=3 * std)

        self.bwd_proj = nn.Linear(
            config.out_channels, config.latent_dim, bias=False,
            device=config.device, dtype=config.dtype,
        )
        std = 1.0 / sqrt(config.out_channels)
        torch.nn.init.trunc_normal_(self.bwd_proj.weight, std=std, a=-3 * std, b=3 * std)

    def transform(self, x, *args, **kwargs):
        assert x.dim() == 5 and x.shape[-1] == 1, f"Got {x.shape}"
        return self.fwd_proj(rearrange(x, 'b t d p 1 -> b t p d'))

    def inverse(self, z, *args, **kwargs):
        assert z.dim() == 4, f"Got {z.shape}"
        return rearrange(self.bwd_proj(z), 'b t p d -> b t d p 1')


class VectorLatentToVecTransform(BaseTransform, nn.Module, Configurable):
    """
    Flattens a ``(latent_dim, P, 1)`` vector latent into the sequence models' shared
    latent dim -- the ``ImageLatentToVecTransform`` analogue for the vector modality.
    """

    @dataclass
    class Config(BaseConfig):
        latent_dim: int = 32
        num_tokens: int = 1
        hidden_dim: int = 256
        out_dim: int = 512
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        device, dtype = config.device, config.dtype

        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim * config.num_tokens, config.hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.out_dim, device=device, dtype=dtype),
            nn.SiLU(),
        )

    def transform(self, x, *args, **kwargs):
        assert x.dim() >= 4, f"Got {x.shape}"
        shape = x.shape
        x = x.reshape(*shape[:-3], self.config.latent_dim * self.config.num_tokens)
        return self.mlp(x)

    def inverse(self, z, *args, **kwargs):
        return
