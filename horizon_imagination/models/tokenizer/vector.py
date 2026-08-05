"""Stage-1 tokenizer for continuous-vector observations.

The image tokenizer maps a frame to a bounded spatial latent; this is its analogue
for vector observations (proprioception, joint states, ...): a per-key MLP
autoencoder that maps the raw vector into the shared latent space and bounds it with
a tanh, with a decoder mapping back.

Latents are shaped ``(latent_dim, P, 1)`` -- i.e. the same rank as image latents
``(C, H, W)`` -- so that every observation leaf stays rank-5 ``(B, T, C, H, W)``
downstream. That is what keeps the diffusion machinery (noise sampling, the Euler
update's ``(B, T)`` broadcast, the masked MSE reduction) modality-agnostic.

``P`` is the number of tokens the key occupies. The encoder is written as a 1-D
patch embedder: the raw vector is split into ``P`` chunks of ``chunk_size``
features, and a single MLP (shared across chunks of the same key) embeds each chunk.
With the default ``chunk_size = obs_dim`` there is exactly one chunk, i.e. one token
per vector key; smaller chunks give several tokens per key, the 1-D analogue of
image patches.
"""
from math import ceil

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from horizon_imagination.utilities import AdamWConfig
from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.utilities.types import ObsKey


@dataclass
class VectorKeySpec(BaseConfig):
    """The raw form of one vector observation key."""
    obs_dim: int
    low: np.ndarray = None
    high: np.ndarray = None

    @classmethod
    def from_box(cls, space):
        low, high = np.asarray(space.low), np.asarray(space.high)
        bounded = np.isfinite(low).all() and np.isfinite(high).all()
        return cls(
            obs_dim=int(np.prod(space.shape)),
            low=low if bounded else None,
            high=high if bounded else None,
        )


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_hidden_layers: int, device, dtype):
    layers = []
    dim = in_dim
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(dim, hidden_dim, device=device, dtype=dtype), nn.SiLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim, device=device, dtype=dtype))
    return nn.Sequential(*layers)


class VectorKeyAutoencoder(nn.Module):
    """The autoencoder of a single vector key. See the module docstring."""

    def __init__(
        self,
        spec: VectorKeySpec,
        latent_dim: int,
        chunk_size: int = None,
        hidden_dim: int = 256,
        num_hidden_layers: int = 1,
        stats_momentum: float = 0.01,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.obs_dim = spec.obs_dim
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size if chunk_size is not None else spec.obs_dim
        self.num_tokens = ceil(self.obs_dim / self.chunk_size)
        self.padding = self.num_tokens * self.chunk_size - self.obs_dim
        self.stats_momentum = stats_momentum

        if spec.low is not None and spec.high is not None:
            # A bounded Box gives an exact normalization; no statistics needed.
            center = torch.as_tensor((spec.high + spec.low) / 2, dtype=torch.float32)
            scale = torch.as_tensor((spec.high - spec.low) / 2, dtype=torch.float32)
            self.has_fixed_stats = True
        else:
            center = torch.zeros(self.obs_dim)
            scale = torch.ones(self.obs_dim)
            self.has_fixed_stats = False
        self.register_buffer('center', center.reshape(-1).to(device=device))
        self.register_buffer('scale', scale.reshape(-1).clamp_min(1e-6).to(device=device))

        self.encoder = _mlp(self.chunk_size, hidden_dim, latent_dim, num_hidden_layers, device, dtype)
        self.decoder = _mlp(latent_dim, hidden_dim, self.chunk_size, num_hidden_layers, device, dtype)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.latent_dim, self.num_tokens, 1)

    @torch.no_grad()
    def update_stats(self, x: Tensor) -> None:
        """EMA of the observation mean / std, used when the Box is unbounded.

        Only called from ``training_step`` -- updating it inside ``encode`` would
        make the latent space drift under the world model and the controller.
        """
        if self.has_fixed_stats:
            return
        x = x.reshape(-1, self.obs_dim).float()
        m = self.stats_momentum
        self.center.mul_(1 - m).add_(m * x.mean(dim=0))
        self.scale.mul_(1 - m).add_(m * x.std(dim=0).clamp_min(1e-6))

    def normalize(self, x: Tensor) -> Tensor:
        return (x.float() - self.center) / self.scale

    def denormalize(self, x: Tensor) -> Tensor:
        return x * self.scale + self.center

    def encode_normalized(self, x_norm: Tensor) -> Tensor:
        """(..., obs_dim) normalized -> (..., latent_dim, P, 1)."""
        if self.padding:
            x_norm = torch.nn.functional.pad(x_norm, (0, self.padding))
        chunks = rearrange(x_norm, '... (p c) -> ... p c', c=self.chunk_size)
        z = torch.tanh(self.encoder(chunks))  # (..., P, latent_dim)
        return rearrange(z, '... p d -> ... d p 1')

    def decode_normalized(self, z: Tensor) -> Tensor:
        """(..., latent_dim, P, 1) -> (..., obs_dim) normalized."""
        assert z.shape[-3:] == self.latent_shape, f"got {tuple(z.shape[-3:])}, expected {self.latent_shape}"
        chunks = rearrange(z, '... d p 1 -> ... p d')
        x_norm = self.decoder(chunks)
        x_norm = rearrange(x_norm, '... p c -> ... (p c)')
        if self.padding:
            x_norm = x_norm[..., :self.obs_dim]
        return x_norm

    def encode(self, x: Tensor) -> Tensor:
        return self.encode_normalized(self.normalize(x))

    def decode(self, z: Tensor) -> Tensor:
        return self.denormalize(self.decode_normalized(z))

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


class VectorAutoencoder(L.LightningModule, Configurable):
    """A collection of per-key vector autoencoders, mirroring ``CosmosImageTokenizer``."""

    @dataclass
    class Config(BaseConfig):
        obs_specs: dict[ObsKey, VectorKeySpec]
        latent_dim: int = 32
        # Number of raw features per token. None -> the whole vector is one token.
        chunk_size: int = None
        per_key_chunk_size: dict[ObsKey, int] = None
        hidden_dim: int = 256
        num_hidden_layers: int = 1
        optimizer_cfg: AdamWConfig = None
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        per_key_chunk_size = config.per_key_chunk_size or {}

        self.autoencoders = nn.ModuleDict({
            str(key): VectorKeyAutoencoder(
                spec=spec,
                latent_dim=config.latent_dim,
                chunk_size=per_key_chunk_size.get(key, config.chunk_size),
                hidden_dim=config.hidden_dim,
                num_hidden_layers=config.num_hidden_layers,
                device=config.device,
                dtype=config.dtype,
            )
            for key, spec in config.obs_specs.items()
        })

    @property
    def keys(self) -> list[ObsKey]:
        return [ObsKey(k) for k in self.autoencoders.keys()]

    def latent_shape(self, key: ObsKey) -> tuple[int, int, int]:
        return self.autoencoders[str(key)].latent_shape

    def num_tokens(self, key: ObsKey) -> int:
        return self.autoencoders[str(key)].num_tokens

    def encode(self, x: Tensor, key: ObsKey) -> Tensor:
        return self.autoencoders[str(key)].encode(x)

    def decode(self, z: Tensor, key: ObsKey) -> Tensor:
        return self.autoencoders[str(key)].decode(z)

    def forward(self, x: Tensor, key: ObsKey) -> Tensor:
        return self.autoencoders[str(key)].forward(x)

    def training_step(self, batch: dict[ObsKey, Tensor], batch_idx, log_dict_fn=None):
        losses = {}
        for key, x in batch.items():
            ae: VectorKeyAutoencoder = self.autoencoders[str(key)]
            ae.update_stats(x)
            x_norm = ae.normalize(x)
            recon_norm = ae.decode_normalized(ae.encode_normalized(x_norm))
            losses[f'vector_tokenizer/{ObsKey(key).name}_loss'] = torch.nn.functional.mse_loss(recon_norm, x_norm)

        if log_dict_fn is None:
            log_dict_fn = self.log_dict
        log_dict_fn(losses, prog_bar=True, on_epoch=True, on_step=False)

        return sum(losses.values())

    def configure_optimizers(self):
        optim_cfg = self.config.optimizer_cfg or AdamWConfig()
        return torch.optim.AdamW(
            self.parameters(),
            lr=optim_cfg.learning_rate,
            betas=optim_cfg.betas,
            eps=optim_cfg.eps,
            weight_decay=optim_cfg.weight_decay,
        )
