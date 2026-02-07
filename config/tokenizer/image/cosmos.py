from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from horizon_imagination.models.tokenizer.cosmos import (
    CosmosImageTokenizer, ContinuousImageTokenizerConfig, OptimizerConfig,
) 


def get_cosmos_tokenizer_online_config(dtype):
    network_cfg = ContinuousImageTokenizerConfig(
        attn_resolutions=tuple([16]),
        patch_size=1,  # 2
        spatial_compression=8,
        resolution=64,  # 128
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
    )

    return cosmos_tokenizer_cfg
