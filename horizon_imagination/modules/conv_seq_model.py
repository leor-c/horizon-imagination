import torch
import torch.nn as nn
from torch import Tensor
from tensordict.tensordict import TensorDict
import gymnasium as gym

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.models.tokenizer.cosmos.modules.layers3d import (
    CausalConv3d, CausalResnetBlockFactorized3d,
)
from horizon_imagination.models.tokenizer.cosmos.modules.utils import nonlinearity
from horizon_imagination.utilities.types import Modality, canonical_obs_keys


def _broadcast_to_grid(x: Tensor, h: int, w: int) -> Tensor:
    """(B, T, D, P, 1) vector latent -> (B, T, D*P, h, w)."""
    x = x.flatten(start_dim=2)
    return x[..., None, None].expand(*x.shape, h, w)


class ConvSeqModel(nn.Module, Configurable):
    """
    A causal, fixed-receptive-field 3D-conv alternative to `LightweightSeqModel`.

    Operates directly on the tokenizer's spatial latent grid (B, T, C, H, W),
    reusing the tokenizer's factorized causal ResNet block (`CausalResnetBlockFactorized3d`)
    so temporal mixing happens jointly with spatial processing, instead of pooling each
    frame to a vector before any temporal modeling (as `LightweightSeqModel` does).

    Effective receptive field ("k" recent frames a prediction can depend on) is
    `1 + 4 * num_blocks`: each `CausalResnetBlockFactorized3d` contains two temporal
    sub-convs (kernel size 3), each extending the backward context by 2 frames.
    Pick `num_blocks` from a target k accordingly.
    """

    @dataclass
    class Config(BaseConfig):
        action_space: gym.Space = None  # only required if ignore_actions=False
        in_channels: int = 6  # tokenizer latent channels
        latent_spatial_shape: tuple = (8, 8)  # tokenizer latent (H, W)
        base_channels: int = 256
        cnn_out_channels: int = 64
        num_blocks: int = 2
        latent_dim: int = 512  # output vector dim; also read generically by RewardDoneModel
        dropout: float = 0.0
        ignore_actions: bool = True
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        device, dtype = config.device, config.dtype

        in_channels = config.in_channels
        self.action_emb = None
        if not config.ignore_actions:
            assert isinstance(config.action_space, gym.spaces.Discrete)
            self.action_emb = nn.Embedding(
                num_embeddings=config.action_space.n,
                embedding_dim=config.in_channels,
                device=device,
                dtype=dtype,
            )
            in_channels = in_channels * 2

        self.in_proj = CausalConv3d(
            in_channels, config.base_channels, kernel_size=1, padding=0,
            device=device, dtype=dtype,
        )
        self.blocks = nn.ModuleList([
            CausalResnetBlockFactorized3d(
                in_channels=config.base_channels,
                out_channels=config.base_channels,
                dropout=config.dropout,
                num_groups=1,
            ).to(device=device, dtype=dtype)
            for _ in range(config.num_blocks)
        ])
        self.out_proj = CausalConv3d(
            config.base_channels, config.cnn_out_channels, kernel_size=1, padding=0,
            device=device, dtype=dtype,
        )

        h, w = config.latent_spatial_shape
        flat_dim = config.cnn_out_channels * h * w
        self.head_proj = nn.Sequential(
            nn.Linear(flat_dim, config.latent_dim, device=device, dtype=dtype),
            nn.SiLU(),
        )

        self.receptive_field = 1 + 4 * config.num_blocks

    def _conv_forward(self, x: Tensor) -> Tensor:
        # x: (B, T, C, H, W) -> (B, T, latent_dim)
        b, t = x.shape[:2]
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        x = nonlinearity(x)
        x = self.out_proj(x)
        x = x.permute(0, 2, 1, 3, 4).reshape(b, t, -1)  # (B, T, C*H*W)
        return self.head_proj(x)

    def forward(
        self,
        actions: Tensor,
        obs: TensorDict,
        state: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        h, w = self.config.latent_spatial_shape

        # Every observation key becomes a channel block of the same spatial grid:
        # image latents already are (C, H, W); vector latents (D, P, 1) are flattened
        # to D*P channels and broadcast over the grid, like the action embedding below.
        x = torch.cat([
            obs[key] if key.modality == Modality.image else _broadcast_to_grid(obs[key], h, w)
            for key in canonical_obs_keys(obs.keys())
        ], dim=2)  # (B, T, C, H, W)

        if not self.config.ignore_actions:
            assert actions.dim() == 2, f"Got {actions.shape}"
            act = self.action_emb(actions)  # (B, T, C)
            act = act[..., None, None].expand(*act.shape, h, w)
            x = torch.cat([x, act], dim=2)

        t_new = x.shape[1]
        x_in = torch.cat([state, x], dim=1) if state is not None else x

        out = self._conv_forward(x_in)[:, -t_new:]

        new_state = x_in[:, -(self.receptive_field - 1):] if self.receptive_field > 1 else x_in[:, :0]
        return out, new_state
