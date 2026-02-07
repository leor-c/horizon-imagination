from typing import Literal
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from horizon_imagination.modules.transform.base import BaseTransform
from horizon_imagination.utilities.config import Configurable, dataclass, BaseConfig
from horizon_imagination.models.tokenizer.cosmos.modules.layers2d import ResnetBlock, Downsample, Normalize


class PatchEmbed(nn.Module):
    # This class was taken from Nvidia Cosmos 
    # https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/diffusion/module/blocks.py
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size,
        temporal_patch_size,
        in_channels=3,
        out_channels=768,
        bias=True,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b (t r) c (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, 
                out_channels, 
                bias=bias,
                device=device,
                dtype=dtype
            ),
        )
        self.out = nn.Identity()

    def forward(self, x):
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, T, C, H, W) where
            B is the batch size,            
            T is the temporal dimension,
            C is the number of channels,
            H is the height, and
            W is the width of the input.

        Returns:w=w, h=h
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return self.out(x)
    

class ImageLatentToVecTransform(BaseTransform, nn.Module, Configurable):
    """
    Class for transforming the latent of an image (frame) to 1D vector.
    Intended for the controller and the reward & done models.
    """
    @dataclass
    class Config(BaseConfig):
        in_channels: int = 6
        latent_spatial_shape: tuple = (8, 8)
        cnn_base_channels: int = 256
        cnn_out_channels: int = 64
        num_blocks: int = 2
        normalize: bool = False
        out_dim: int = 512
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.cnn = self._build_cnn()

    def _build_cnn(self):
        device=self.config.device
        dtype=self.config.dtype
        
        layers = [
            # In conv:
            nn.Conv2d(
                in_channels=self.config.in_channels,
                out_channels=self.config.cnn_base_channels,
                kernel_size=1,
                device=device,
                dtype=dtype
            )
        ]
        num_blocks = self.config.num_blocks

        for i in range(num_blocks):
            block = ResnetBlock(
                in_channels=self.config.cnn_base_channels,
                out_channels=self.config.cnn_base_channels,
                dropout=0,
                normalize=self.config.normalize,
            ).to(device=device, dtype=dtype)
            layers.append(block)
        if self.config.normalize:
            layers.append(Normalize(self.config.cnn_base_channels))
        layers.append(
            # Out conv:
            nn.Sequential(
                nn.SiLU(),
                nn.Conv2d(
                    in_channels=self.config.cnn_base_channels,
                    out_channels=self.config.cnn_out_channels,
                    kernel_size=1,
                    device=device,
                    dtype=dtype
                )
            )
        )
        # assume input shape (b c h w)
        layers.append(nn.Flatten(start_dim=1))

        in_features = np.prod(self.config.latent_spatial_shape) * self.config.cnn_out_channels
        if self.config.normalize:
            layers.append(nn.LayerNorm(in_features, device=device))
        dense = nn.Sequential(
            nn.Linear(in_features, self.config.out_dim, device=device),
            nn.SiLU(),
        )

        layers.append(dense)

        return nn.Sequential(*layers)
    
    def transform(self, x, *args, **kwargs):
        assert x.dim() >= 4, f"Got {x.shape}"
        shape = x.shape
        x = rearrange(x, "... c h w -> (...) c h w")
        x = self.cnn(x)
        x = x.reshape(*shape[:-3], self.config.out_dim)

        return x
    
    def inverse(self, z, *args, **kwargs):
        return 
