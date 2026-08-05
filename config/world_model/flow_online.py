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
from horizon_imagination.modules.conv_seq_model import ConvSeqModel
from horizon_imagination.models.tokenizer import VectorAutoencoder
from horizon_imagination.utilities.types import ObsKey, Modality, image_keys, vector_keys
from horizon_imagination.utilities import AdamWConfig
from horizon_imagination.modules.transform import (
    PerModalityTransform, ImageToLatentTransform,
    ImageLatentToVecTransform, VectorToLatentTransform, VectorLatentToVecTransform
)
from config.tokenizer.image.cosmos import (
    CosmosImageTokenizer
)


def get_world_model_online_config(
        obs_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        tokenizer_channels: int,
        latent_spatial_shape: tuple[int, int],
        image_tokenizer: CosmosImageTokenizer,
        vector_autoencoder: VectorAutoencoder,
        baseline: Literal['hi', 'ar', 'naive'],
        decay_horizon: float,
        device,
        dtype,
        reward_done_backbone: Literal['lstm', 'conv'] = 'lstm',
) -> RectifiedFlowWorldModel.Config:
    num_heads = 8
    head_dim = 64
    num_layers = 12
    embed_dim = num_heads * head_dim
    patch_spatial = 2
    patch_temporal = 1

    img_keys = image_keys(obs_space.spaces.keys())
    vec_keys = vector_keys(obs_space.spaces.keys())
    assert bool(vec_keys) == (vector_autoencoder is not None), \
        "A vector autoencoder is required exactly when the observation has vector keys."
    vector_dim = vector_autoencoder.config.latent_dim if vec_keys else 0

    latent_h, latent_w = latent_spatial_shape
    obs_spec = OrderedDict(
        [(k, (tokenizer_channels, latent_h, latent_w)) for k in img_keys] +
        [(k, vector_autoencoder.latent_shape(k)) for k in vec_keys]
    )

    # RoPE position grid: the image keys' patch grids are stacked vertically, then one
    # row per vector key (of as many columns as that key has tokens). max_img_* are in
    # pre-patch units.
    position_rows = len(img_keys) * (latent_h // patch_spatial) + len(vec_keys)
    position_cols = max(
        [latent_w // patch_spatial] + [vector_autoencoder.num_tokens(k) for k in vec_keys]
    )

    # Denoiser Network:
    denoiser_cfg = VideoDiTDenoiser.Config(
        dit_cfg=DiT.Config(
            max_img_h=position_rows * patch_spatial,
            max_img_w=position_cols * patch_spatial,
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
        spatial_patch_size=patch_spatial,
        img_latent_channels=tokenizer_channels,
        obs_spec=obs_spec,
        vector_latent_dim=vector_dim,
    )

    # Stage-1 encoders (raw observation -> latent):
    tokenizer_transform = PerModalityTransform(
        fixed_per_modality_transforms={
            Modality.image: ImageToLatentTransform(
                image_tokenizer=image_tokenizer.to(device=device, dtype=dtype)
            )
        },
        fixed_per_key_transforms={
            k: VectorToLatentTransform(vector_autoencoder, k) for k in vec_keys
        },
    )

    # Reward & Termination model:
    if reward_done_backbone == 'lstm':
        backbone_cfg = LightweightSeqModel.Config(
            action_space=action_space,
            obs_vectorize_transforms=PerModalityTransform(
                learned_per_modality_transforms={
                    Modality.image: ImageLatentToVecTransform.Config(
                        in_channels=tokenizer_channels,
                        latent_spatial_shape=latent_spatial_shape,
                        cnn_base_channels=256,
                        cnn_out_channels=64,
                        num_blocks=1,
                        normalize=True,
                        out_dim=512,
                        device=device,
                        dtype=dtype
                    ).make_instance()
                },
                learned_per_key_transforms={
                    k: VectorLatentToVecTransform.Config(
                        latent_dim=vector_dim,
                        num_tokens=vector_autoencoder.num_tokens(k),
                        out_dim=512,
                        device=device,
                        dtype=dtype
                    ).make_instance()
                    for k in vec_keys
                },
            ),
            latent_dim=512,
            num_layers=1,
            ignore_actions=True,
            device=device,
            dtype=dtype
        )
    elif reward_done_backbone == 'conv':
        backbone_cfg = ConvSeqModel.Config(
            action_space=action_space,
            # image latents keep their channels; vector latents (D, P, 1) are
            # flattened to D*P channels and broadcast over the spatial grid.
            in_channels=sum(
                c if key.modality == Modality.image else c * h * w
                for key, (c, h, w) in obs_spec.items()
            ),
            latent_spatial_shape=latent_spatial_shape,
            base_channels=256,
            cnn_out_channels=64,
            num_blocks=2,
            latent_dim=512,
            ignore_actions=True,
            device=device,
            dtype=dtype
        )
    else:
        raise ValueError(f"reward_done_backbone '{reward_done_backbone}' not supported.")

    reward_done_cfg = RewardDoneModel.Config(
        backbone_config=backbone_cfg,
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
