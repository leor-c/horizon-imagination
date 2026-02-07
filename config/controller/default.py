from typing import Literal
import gymnasium as gym

from horizon_imagination.utilities.types import Modality
from horizon_imagination.modules.transform import PerModalityTransform, ImageLatentToVecTransform
from horizon_imagination.models.controller import Controller, ActorCritic, DiscreteActorHead, CriticHead
from horizon_imagination.modules.lightweight_seq_model import LightweightSeqModel
from horizon_imagination.utilities import AdamWConfig


def _get_actor_critic_cfg(
        env_name: str,
        action_space: gym.spaces.Discrete, 
        tokenizer_channels: int,
        device, 
        dtype
    ):
    # Config values only:
    latent_dim = 512
    shared_backbone = False
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
                        out_dim=latent_dim,
                        cnn_base_channels=cnn_base_channels,
                        cnn_out_channels=cnn_out_channels,
                        num_blocks=num_blocks,
                        normalize=False,
                        device=device,
                        dtype=dtype
                    ).make_instance()
                }
            ),
            latent_dim=latent_dim,
            num_layers=num_lstm_layers,
            ignore_actions=ignore_actions,  
            device=device,
            dtype=dtype
    )

    actor_bias = None
    if 'speleo' in env_name.lower():
        # nop, forward, jump, mouse x+, mouse x-, mouse y+, mouse y-
        actor_bias = (0, 1, 0, 0, 0, 0, 0)
    elif 'choptree' in env_name.lower():
        # nop, forward, jump, dig (used to chop), mouse x+, mouse x-, mouse y+, mouse y-
        actor_bias = (0, 0, 0, 1, 0, 0, 0, 0)

    ac_cfg = ActorCritic.Config(
        backbone=backbone_cfg,
        shared_backbone=shared_backbone,
        actor=DiscreteActorHead.Config(
            latent_dim=latent_dim,
            actor_bias=actor_bias,
            num_actions=action_space.n
        ),
        critic=CriticHead.Config(latent_dim=latent_dim),
    )
    return ac_cfg


def get_controller_config(
        env_name: str,
        action_space: gym.spaces.Discrete, 
        tokenizer_channels: int,
        world_model,
        imagination_horizon: int,
        budget: int,
        device, 
        dtype,
        baseline: Literal['hi', 'ar', 'naive'] = 'hi',
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
    )

    return controller_cfg
