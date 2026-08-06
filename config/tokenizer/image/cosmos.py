from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch

from horizon_imagination.models.tokenizer.cosmos import (
    CosmosImageTokenizer, ContinuousImageTokenizerConfig, OptimizerConfig,
) 


def get_cosmos_tokenizer_online_config(dtype, resolution: int = 64):
    if resolution == 64:
        channels_mult = (2, 4, 4)
        spatial_compression = 8
    elif resolution == 128:
        channels_mult = (2, 4, 4, 4)
        spatial_compression = 16
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    network_cfg = ContinuousImageTokenizerConfig(
        attn_resolutions=tuple([16]),
        patch_size=2,  # 2
        patch_method="rearrange",
        decoder_output_mode="resize",
        channels_mult=channels_mult,
        spatial_compression=spatial_compression,
        resolution=resolution,
        channels=64,
        latent_channels=16,
        # formulation='VAE'
    )
    optimizer_cfg = OptimizerConfig()
    optimizer_cfg.learning_rate = 2e-4
    optimizer_cfg.weight_decay = 0.05
    optimizer_cfg.betas = (0.9, 0.95)

    cosmos_tokenizer_cfg = CosmosImageTokenizer.Config(
        network_cfg=network_cfg,
        optimizer_cfg=optimizer_cfg,
        precision=dtype,
        autocast_dtype=torch.bfloat16,
        torch_compile=True,
        # The upstream Cosmos code notes long-run CUDA-graph instability for
        # the perceptual loss, so use Inductor fusion without CUDA graphs.
        torch_compile_mode="default",
    )

    return cosmos_tokenizer_cfg
