from collections import OrderedDict
from typing import Literal

import gymnasium as gym

from horizon_imagination.models.world_model import (
    RectifiedFlowWorldModel, RewardDoneModel, VideoDiTDenoiser, DiT
)
from horizon_imagination.diffusion.samplers import (
    HorizonSamplerScheduler, UniformSamplerScheduler
)
from horizon_imagination.diffusion import HybridTimeSampler, UniformTimeSampler
from horizon_imagination.modules.lightweight_seq_model import LightweightSeqModel
from horizon_imagination.utilities.types import ObsKey, Modality
from horizon_imagination.utilities import AdamWConfig
from horizon_imagination.modules.transform import (
    PerModalityTransform, ImageToLatentTransform, 
    ImageLatentToVecTransform
)
from config.tokenizer.image.cosmos import (
    CosmosImageTokenizer
)


def get_world_model_online_config(
        action_space: gym.spaces.Discrete,
        tokenizer_channels: int,
        image_tokenizer: CosmosImageTokenizer,
        baseline: Literal['hi', 'ar', 'naive'],
        decay_horizon: float,
        device,
        dtype,
) -> RectifiedFlowWorldModel.Config:
    num_heads = 8
    head_dim = 64
    num_layers = 12
    embed_dim = num_heads * head_dim
    patch_spatial = 2
    patch_temporal = 1

    # Denoiser Network:
    denoiser_cfg = VideoDiTDenoiser.Config(
        dit_cfg=DiT.Config(
            max_img_h=8,
            max_img_w=8,
            expected_max_frames=128,
            in_channels=tokenizer_channels,
            out_channels=tokenizer_channels,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            model_channels=embed_dim,
            num_blocks=num_layers,
            num_heads=num_heads,
            device=device,
            ln_eps=1e-5,
        ),
        action_space=action_space,
    )

    # Tokenizer:
    tokenizer_transform = PerModalityTransform(fixed_per_modality_transforms={
        Modality.image: ImageToLatentTransform(
            image_tokenizer=image_tokenizer.to(device=device, dtype=dtype)
        )
    })

    # Reward & Termination model:
    reward_done_cfg = RewardDoneModel.Config(
        backbone_config=LightweightSeqModel.Config(
            action_space=action_space,
            obs_vectorize_transforms=PerModalityTransform(
                learned_per_modality_transforms={
                    Modality.image: ImageLatentToVecTransform.Config(
                        in_channels=tokenizer_channels,
                        cnn_base_channels=256,
                        cnn_out_channels=64,
                        num_blocks=1,
                        normalize=True,
                        out_dim=512,
                        device=device,
                        dtype=dtype
                    ).make_instance()
                }
            ),
            latent_dim=512,
            num_layers=1,
            ignore_actions=True,
            device=device,
            dtype=dtype
        ),
        hl_gauss_num_bins=129
    )

    if baseline == 'ar':
        decay_horizon = 1
    elif baseline == 'hi':
        decay_horizon = decay_horizon
    else:
        raise ValueError(f"baseline '{baseline}' not supported.")

    sampler_scheduler_cfg = HorizonSamplerScheduler.Config(
        device=device,
        dtype=dtype,
        decay_horizon=decay_horizon,
        time_transform=None,
    )

    time_sampler = HybridTimeSampler(
        base_time_sampler=UniformTimeSampler(),
        prob_clean_context=0.2,
    )

    wm_cfg = RectifiedFlowWorldModel.Config(
        denoiser_config=denoiser_cfg, 
        sampler_scheduler=sampler_scheduler_cfg,
        time_sampler=time_sampler,
        obs_transform=tokenizer_transform,
        reward_done_model=reward_done_cfg,
        denoiser_optim=AdamWConfig(
            learning_rate=2e-4,
            weight_decay=0.01,
            betas=(0.9, 0.99),
            eps=1e-6,
        ),
        reward_optim=AdamWConfig(
            learning_rate=2e-4,
            weight_decay=0.01,
            betas=(0.9, 0.99),
            eps=1e-6,
        )
    )
    return wm_cfg
