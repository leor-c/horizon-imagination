from typing import Literal
import gymnasium as gym

from horizon_imagination.utilities.types import Modality, image_keys, vector_keys
from horizon_imagination.modules.transform import (
    PerModalityTransform, ImageLatentToVecTransform, VectorLatentToVecTransform
)
from horizon_imagination.models.controller import (
    Controller, ActorCritic, DiscreteActorHead, GaussianActorHead, CriticHead
)
from horizon_imagination.modules.lightweight_seq_model import LightweightSeqModel
from horizon_imagination.modules.embeddings import action_dim
from horizon_imagination.utilities import AdamWConfig


def _get_actor_critic_cfg(
        env_name: str,
        action_space: gym.Space,
        tokenizer_channels: int,
        latent_spatial_shape: tuple[int, int],
        vector_autoencoder,
        image_obs_keys: list,
        vector_obs_keys: list,
        device,
        dtype
    ):
    # Config values only:
    latent_dim = 512
    shared_backbone = False
    use_clean_diffused_actors = True
    ignore_actions = True  # TODO: support (previous) action inputs!
    cnn_base_channels = 256
    cnn_out_channels = 64
    num_blocks = 1
    num_lstm_layers = 1

    # init config instance:
    backbone_cfg = LightweightSeqModel.Config(
        action_space=action_space,
        obs_vectorize_transforms=PerModalityTransform(
                learned_per_modality_transforms={
                    Modality.image: ImageLatentToVecTransform.Config(
                        in_channels=tokenizer_channels,
                        latent_spatial_shape=latent_spatial_shape,
                        out_dim=latent_dim,
                        cnn_base_channels=cnn_base_channels,
                        cnn_out_channels=cnn_out_channels,
                        num_blocks=num_blocks,
                        normalize=False,
                        device=device,
                        dtype=dtype
                    ).make_instance()
                } if image_obs_keys else {},
                learned_per_key_transforms={
                    k: VectorLatentToVecTransform.Config(
                        latent_dim=vector_autoencoder.config.latent_dim,
                        num_tokens=vector_autoencoder.num_tokens(k),
                        out_dim=latent_dim,
                        device=device,
                        dtype=dtype
                    ).make_instance()
                    for k in vector_obs_keys
                },
            ),
            latent_dim=latent_dim,
            num_layers=num_lstm_layers,
            ignore_actions=ignore_actions,  
            device=device,
            dtype=dtype
    )

    ac_cfg = ActorCritic.Config(
        backbone=backbone_cfg,
        shared_backbone=shared_backbone,
        use_clean_diffused_actors=use_clean_diffused_actors,
        actor=_get_actor_head_cfg(env_name, action_space, latent_dim),
        critic=CriticHead.Config(latent_dim=latent_dim),
    )
    return ac_cfg


def _get_actor_head_cfg(env_name: str, action_space: gym.Space, latent_dim: int):
    if isinstance(action_space, gym.spaces.Box):
        return GaussianActorHead.Config(
            latent_dim=latent_dim,
            action_dim=action_dim(action_space),
        )

    assert isinstance(action_space, gym.spaces.Discrete), f"Got {action_space}"

    actor_bias = None
    if 'speleo' in env_name.lower():
        # nop, forward, jump, mouse x+, mouse x-, mouse y+, mouse y-
        actor_bias = (0, 1, 0, 0, 0, 0, 0)
    elif 'choptree' in env_name.lower():
        # nop, forward, jump, dig (used to chop), mouse x+, mouse x-, mouse y+, mouse y-
        actor_bias = (0, 0, 0, 1, 0, 0, 0, 0)

    return DiscreteActorHead.Config(
        latent_dim=latent_dim,
        actor_bias=actor_bias,
        num_actions=action_space.n
    )


def get_controller_config(
        env_name: str,
        obs_space: gym.spaces.Dict,
        action_space: gym.Space,
        tokenizer_channels: int,
        latent_spatial_shape: tuple[int, int],
        world_model,
        vector_autoencoder,
        imagination_horizon: int,
        budget: int,
        device,
        dtype,
        baseline: Literal['hi', 'ar', 'naive'] = 'hi',
        intrinsic_reward_weight: float = 0.0,
        pure_exploration: bool = False,
        intrinsic_reward_truncated_roundtrip: bool = False,
    ):
    controller_context_length = 20
    imagination_batch_size = 30
    num_denoising_steps = budget
    context_noise_level = 0
    gae_gamma = 0.99
    gae_lambda = 0.95
    entropy_weight = 0.001
    return_scaler_decay = 0.005

    # init config instance:
    ac_cfg = _get_actor_critic_cfg(
        env_name=env_name,
        action_space=action_space,
        tokenizer_channels=tokenizer_channels,
        latent_spatial_shape=latent_spatial_shape,
        vector_autoencoder=vector_autoencoder,
        image_obs_keys=image_keys(obs_space.spaces.keys()),
        vector_obs_keys=vector_keys(obs_space.spaces.keys()),
        device=device,
        dtype=dtype
    )

    controller_cfg = Controller.Config(
        actor_critic=ac_cfg,
        world_model=world_model,
        optim=AdamWConfig(
            learning_rate=2e-4,
            weight_decay=0.01,
            betas=(0.9, 0.99)
        ),
        controller_context_length=controller_context_length,
        imagination_batch_size=imagination_batch_size,
        imagination_horizon=imagination_horizon,
        num_denoising_steps=num_denoising_steps,
        context_noise_level=context_noise_level,
        gae_gamma=gae_gamma,
        gae_lambda=gae_lambda,
        entropy_weight=entropy_weight,
        return_scaler_decay=return_scaler_decay,
        baseline=baseline,
        intrinsic_reward_weight=intrinsic_reward_weight,
        pure_exploration=pure_exploration,
        intrinsic_reward_truncated_roundtrip=intrinsic_reward_truncated_roundtrip,
    )

    return controller_cfg
