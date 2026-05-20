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
from horizon_imagination.utilities.types import ObsKey, Modality
from horizon_imagination.modules.transform import (
    BaseTransform, PerModalityTransform, MultiModalCatAndFlatten, ImagePatcherTransform
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
        uniform = torch.ones_like(one_hots) / self.num_actions

        assert t.shape == actions.shape, f"got {t.shape}, {actions.shape}"
        # noised_actions = t.unsqueeze(-1) * one_hots + (1 - t.unsqueeze(-1)) * uniform
        noised_actions = one_hots

        return self.linear(noised_actions.to(self.linear.weight.dtype))


class VideoDiTDenoiser(DenoiserBase, Configurable):
    @dataclass
    class Config(BaseConfig):
        dit_cfg: DiT.Config
        spatial_patch_size: int
        img_latent_channels: int
        # The length (in tokens / embeddings) of each obs item:
        # obs_items_lengths: OrderedDict[ObsKey, int]
        # per_modality_transforms: PerModalityTransform
        action_space: gym.Space

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
        # latent form is the output of the tokenizer.
        assert isinstance(x, TensorDict), f"Got {type(x)}"
        img_key = ObsKey.from_parts(Modality.image, 'features')
        assert img_key in x

        action_embeddings = self.action_embedder(actions)
        assert action_embeddings.dim() == 3, f"Got shape {action_embeddings.shape}"
        # No action before the first observation of a fresh sequence.
        # When continuing from KV-cache, token 0 already has a valid predecessor.
        if kv_cache is None or len(kv_cache) == 0:
            action_embeddings[:, 0] = 0

        c = action_embeddings

        # Compute model outputs:
        z = self.image_patcher.transform(x[img_key])
        outputs, new_kv_cache = self.dit(z, t, c, kv_cache=kv_cache)
        outputs = self.image_patcher.inverse(outputs)
        outputs = TensorDict({img_key: outputs}, batch_size=x.batch_size, device=x.device)

        return outputs, new_kv_cache
