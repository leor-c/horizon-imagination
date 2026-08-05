from typing import Union
from collections import OrderedDict
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tensordict.tensordict import TensorDict
from einops import rearrange

import gymnasium as gym
from loguru import logger

from horizon_imagination.diffusion import DenoiserBase
from horizon_imagination.utilities.config import Configurable, BaseConfig
from horizon_imagination.utilities.types import ObsKey, Modality, canonical_obs_keys
from horizon_imagination.modules.transform import (
    BaseTransform, PerModalityTransform, ImagePatcherTransform, VectorTokenTransform
)

from horizon_imagination.models.world_model.dit import DiT, KVCache


class DiscreteActionEmbedder(nn.Module):
    def __init__(self, num_actions: int, embed_dim: int, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_actions = num_actions
        self.embed_dim = embed_dim

        self.linear = nn.Linear(num_actions, embed_dim, bias=False, device=device, dtype=dtype)
        self._uniform = None

    def forward(self, actions: Tensor, t: Tensor) -> Tensor:
        assert torch.all(0 <= t) and torch.all(t <= 1)
        one_hots = F.one_hot(actions, num_classes=self.num_actions)
        # uniform = torch.ones_like(one_hots) / self.num_actions

        assert t.shape == actions.shape, f"got {t.shape}, {actions.shape}"
        # noised_actions = t.unsqueeze(-1) * one_hots + (1 - t.unsqueeze(-1)) * uniform
        noised_actions = one_hots

        return self.linear(noised_actions.to(self.linear.weight.dtype))


def build_token_position_table(
        token_grids: OrderedDict[ObsKey, tuple[int, int]]
    ) -> tuple[Tensor, OrderedDict[ObsKey, int]]:
    """
    Lay out the tokens of one frame on the RoPE (h, w) position grid.

    Every observation key gets its own rectangular band of positions, so that tokens
    of different keys are never given the same position: image key `n` takes an
    `H' x W'` band stacked below the previous image keys, and each vector key takes a
    single row of `P` positions below all the image bands.

    A single image key therefore reproduces exactly the row-major meshgrid the RoPE
    embedding used to hardcode, which keeps embeddings (and checkpoints) unchanged.

    :param token_grids: per key, the (rows, cols) of its token band, in canonical order.
    :return: the (K, 2) position table and the per-key token count.
    """
    positions = []
    num_tokens = OrderedDict()
    row_offset = 0
    for key, (rows, cols) in token_grids.items():
        h_idx, w_idx = torch.meshgrid(
            torch.arange(rows) + row_offset, torch.arange(cols), indexing='ij'
        )
        positions.append(torch.stack([h_idx.reshape(-1), w_idx.reshape(-1)], dim=-1))
        num_tokens[key] = rows * cols
        row_offset += rows

    return torch.cat(positions, dim=0), num_tokens


class VideoDiTDenoiser(DenoiserBase, Configurable):
    @dataclass
    class Config(BaseConfig):
        dit_cfg: DiT.Config
        spatial_patch_size: int
        img_latent_channels: int
        action_space: gym.Space
        # The latent shape (C, H, W) of each observation key. Image keys contribute
        # (H/patch)*(W/patch) tokens per frame, vector keys -- whose latents are
        # (latent_dim, P, 1) -- contribute P.
        obs_spec: dict[ObsKey, tuple]
        vector_latent_dim: int = 32

        @property
        def device(self):
            return self.dit_cfg.device

        @property
        def dtype(self):
            return self.dit_cfg.dtype

    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        self.dit = config.dit_cfg.make_instance()

        self.action_embedder = self._build_action_embedder()
        self.image_patcher: ImagePatcherTransform = ImagePatcherTransform.Config(
            spatial_patch_size=config.spatial_patch_size,
            in_channels=config.img_latent_channels,
            out_channels=config.dit_cfg.model_channels,
        ).make_instance()

        self.obs_keys = canonical_obs_keys(config.obs_spec.keys())
        self.obs_spec = OrderedDict((k, tuple(config.obs_spec[k])) for k in self.obs_keys)

        # Vector keys each get their own token projection; image keys all share
        # `self.image_patcher` above (they share a tokenizer, hence a latent space)
        # and are told apart by their bands in the position table.
        self.vector_token_transforms = nn.ModuleDict({
            str(key): VectorTokenTransform.Config(
                latent_dim=config.vector_latent_dim,
                out_channels=config.dit_cfg.model_channels,
            ).make_instance()
            for key in self.obs_keys if key.modality == Modality.vector
        })

        token_positions, num_tokens = build_token_position_table(
            OrderedDict((k, self._token_grid(k)) for k in self.obs_keys)
        )
        # Not persistent: it is fully derived from the config, and keeping it out of
        # the state dict keeps checkpoints loadable across observation specs.
        self.register_buffer('token_positions', token_positions, persistent=False)
        self.num_tokens = num_tokens
        self._assert_position_capacity()

    def _token_grid(self, key: ObsKey) -> tuple[int, int]:
        """The (rows, cols) of the token band a key occupies in a frame."""
        channels, height, width = self.obs_spec[key]
        if key.modality == Modality.image:
            patch = self.config.spatial_patch_size
            assert height % patch == 0 and width % patch == 0, \
                f"Latent size {(height, width)} of '{key}' is not divisible by the patch size {patch}."
            return height // patch, width // patch

        assert key.modality == Modality.vector, f"Unsupported modality of key '{key}'."
        assert channels == self.config.vector_latent_dim and width == 1, \
            f"Expected a (vector_latent_dim, P, 1) latent for '{key}', got {self.obs_spec[key]}."
        return 1, height  # a single row of P token positions

    def _assert_position_capacity(self):
        patch = self.config.spatial_patch_size
        max_h = self.config.dit_cfg.max_img_h // patch
        max_w = self.config.dit_cfg.max_img_w // patch
        needed_h = int(self.token_positions[:, 0].max()) + 1
        needed_w = int(self.token_positions[:, 1].max()) + 1
        assert needed_h <= max_h and needed_w <= max_w, (
            f"The observation needs a {needed_h}x{needed_w} position grid, but the DiT only has "
            f"{max_h}x{max_w}. Raise max_img_h / max_img_w (they are given in pre-patch units, "
            f"i.e. multiples of spatial_patch_size={patch})."
        )

    def _build_action_embedder(self):
        action_space = self.config.action_space
        if isinstance(action_space, gym.spaces.Discrete):
            action_embedder = nn.Embedding(
                action_space.n,
                self.embed_dim,
                device=self.config.dit_cfg.device,
            )

            std = 1.0 / math.sqrt(self.embed_dim)
            torch.nn.init.trunc_normal_(action_embedder.weight, std=std, a=-3 * std, b=3 * std)

            return action_embedder
        else:
            # TODO: support more action modalities
            raise NotImplementedError(f"Currently action space {action_space} is not supported.")

    @property
    def embed_dim(self):
        return self.config.dit_cfg.model_channels

    def denoise(
            self,
            x: TensorDict[ObsKey, Tensor],
            t: Tensor,
            actions: Tensor = None,
            kv_cache: KVCache = None,
            **kwargs
    ) -> tuple[TensorDict[ObsKey, Tensor], KVCache]:
        # Assume x is encoded to "latent" form, but not flattened yet.
        # latent form is the output of the stage-1 encoders (tokenizer / vector AE).
        assert isinstance(x, TensorDict), f"Got {type(x)}"
        assert set(map(str, x.keys())) == set(map(str, self.obs_keys)), \
            f"Got observation keys {sorted(map(str, x.keys()))}, expected {sorted(map(str, self.obs_keys))}"

        action_embeddings = self.action_embedder(actions)
        assert action_embeddings.dim() == 3, f"Got shape {action_embeddings.shape}"
        # No action before the first observation of a fresh sequence.
        # When continuing from KV-cache, token 0 already has a valid predecessor.
        if kv_cache is None or len(kv_cache) == 0:
            action_embeddings[:, 0] = 0

        c = action_embeddings

        # Encode every observation key to tokens and lay them out along a single
        # token axis, in canonical order (which is what the position table assumes):
        z = torch.cat([self._to_tokens(key, x[key]) for key in self.obs_keys], dim=2)

        outputs, new_kv_cache = self.dit(z, t, c, self.token_positions, kv_cache=kv_cache)

        outputs = torch.split(outputs, [self.num_tokens[k] for k in self.obs_keys], dim=2)
        outputs = TensorDict(
            {
                key: self._from_tokens(key, out)
                for key, out in zip(self.obs_keys, outputs)
            },
            batch_size=x.batch_size,
            device=x.device,
        )

        return outputs, new_kv_cache

    def _to_tokens(self, key: ObsKey, x: Tensor) -> Tensor:
        """(B, T, C, H, W) latent -> (B, T, tokens, model_channels)."""
        if key.modality == Modality.image:
            return rearrange(self.image_patcher.transform(x), 'b t h w d -> b t (h w) d')
        return self.vector_token_transforms[str(key)].transform(x)

    def _from_tokens(self, key: ObsKey, z: Tensor) -> Tensor:
        """(B, T, tokens, model_channels) -> (B, T, C, H, W) latent."""
        if key.modality == Modality.image:
            rows, cols = self._token_grid(key)
            return self.image_patcher.inverse(rearrange(z, 'b t (h w) d -> b t h w d', h=rows, w=cols))
        return self.vector_token_transforms[str(key)].inverse(z)
