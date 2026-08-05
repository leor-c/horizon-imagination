from typing import Optional, Literal
import gymnasium as gym
import torch
from pathlib import Path

import episodata as ed
from episodata.utils import schema_from_gym_spaces

from horizon_imagination.agent import Agent
from horizon_imagination.models.tokenizer import ObsEncoderStack, VectorAutoencoder, VectorKeySpec
from horizon_imagination.utilities.config import BaseConfig
from horizon_imagination.utilities.types import image_keys, vector_keys
from horizon_imagination.utilities.obs_codec import y_key, cbcr_key
from config.tokenizer.image.cosmos import get_cosmos_tokenizer_online_config, CosmosImageTokenizer
from config.world_model.flow_online import get_world_model_online_config, RectifiedFlowWorldModel
from config.controller.default import get_controller_config


def init_component(
        component_cfg: BaseConfig, 
        load_pretrained: bool, 
        weights_path: Optional[Path], 
    ):
    if load_pretrained:
        assert weights_path is not None
        return component_cfg.load_from_checkpoint(weights_path, config=component_cfg)
    else:
        return component_cfg.make_instance()


def get_agent_online_config(
        env: gym.Env,
        env_name: str,
        replay_buf_data_path: Path = None,
        test_env: gym.Env = None,
        baseline: Literal['hi', 'ar', 'naive'] = 'hi',
        decay_horizon: float = 4,
        budget: int = 32,
        resolution: int = 64,
    ):
    # Config values only (for readability):
    device = torch.device('cuda')
    # dtype = torch.bfloat16
    dtype = None

    replay_buf_max_size = 1_000_000
    replay_buf_store_on_disk = (replay_buf_data_path is not None)
    replay_buf_device = 'cpu'  # if replay_buf_store_on_disk else device

    load_pretrained_tokenizer = False
    tokenizer_weights_path = None
    load_pretrained_wm = False
    world_model_weights_path = None
    load_pretrained_controller = False
    controller_weights_path = None

    num_epochs = 500
    if env_name == 'Craftium/SmallRoom-v0':
        num_epochs = 150

    tokenizer_batch_size = 32
    tokenizer_steps_per_epoch = 300
    tokenizer_train_from_epoch = 10
    tokenizer_max_grad_norm = 1

    world_model_batch_size = 8
    world_model_horizon = 32
    world_model_steps_per_epoch = 300
    world_model_train_from_epoch = 25
    world_model_min_segment_length = 4
    world_model_max_grad_norm = 1

    imagination_horizon = world_model_horizon
    controller_steps_per_epoch = 50
    controller_train_from_epoch = 40
    controller_max_grad_norm = 1

    collection_steps_per_epoch = 200
    controller_test_frequency = 50

    # Vector observations: latent width per token, and how many raw features each
    # token covers (None -> the whole vector is a single token).
    vector_latent_dim = 32
    vector_chunk_size = None

    read_chunk_size = 4096

    # Init config instance:
    observation_space: gym.spaces.Dict = env.observation_space
    action_space: gym.spaces.Discrete = env.action_space

    tokenizer_cfg = get_cosmos_tokenizer_online_config(dtype=dtype, resolution=resolution)
    tokenizer_channels = tokenizer_cfg.latent_channels
    latent_size = resolution // tokenizer_cfg.network_cfg.spatial_compression
    latent_spatial_shape = (latent_size, latent_size)

    tokenizer: CosmosImageTokenizer = init_component(
        tokenizer_cfg,
        load_pretrained_tokenizer,
        tokenizer_weights_path
    )

    vec_keys = vector_keys(observation_space.spaces.keys())
    vector_autoencoder: VectorAutoencoder = None
    if vec_keys:
        vector_autoencoder = VectorAutoencoder.Config(
            obs_specs={k: VectorKeySpec.from_box(observation_space.spaces[k]) for k in vec_keys},
            latent_dim=vector_latent_dim,
            chunk_size=vector_chunk_size,
            device=device,
            dtype=dtype,
        ).make_instance()

    obs_encoder = ObsEncoderStack.Config(
        image_tokenizer=tokenizer,
        vector_autoencoder=vector_autoencoder,
    ).make_instance()

    wm_cfg = get_world_model_online_config(
        obs_space=observation_space,
        action_space=action_space,
        tokenizer_channels=tokenizer_channels,
        latent_spatial_shape=latent_spatial_shape,
        image_tokenizer=tokenizer,
        vector_autoencoder=vector_autoencoder,
        baseline=baseline,
        decay_horizon=decay_horizon,
        device=device,
        dtype=dtype
    )
    world_model: RectifiedFlowWorldModel = init_component(
        wm_cfg,
        load_pretrained_wm,
        world_model_weights_path
    )

    if baseline == 'ar':
        assert (budget >= imagination_horizon) and (budget % imagination_horizon) == 0

    controller_cfg = get_controller_config(
        env_name=env_name,
        obs_space=observation_space,
        action_space=action_space,
        tokenizer_channels=tokenizer_channels,
        latent_spatial_shape=latent_spatial_shape,
        world_model=world_model,
        vector_autoencoder=vector_autoencoder,
        imagination_horizon=imagination_horizon,
        budget=budget,
        device=device,
        dtype=dtype,
        baseline=baseline,
    )
    controller = init_component(
        controller_cfg,
        load_pretrained_controller,
        controller_weights_path
    )

    # Images are stored as YCbCr components (4:2:0), not RGB -- one pair of fields
    # per image key:
    schema = schema_from_gym_spaces(env.observation_space, env.action_space)
    for key in image_keys(observation_space.spaces.keys()):
        img_field = schema.fields.pop(str(key))
        h, w = img_field.shape[-2:]
        for component_key, shape in [
            (y_key(key), (1, h, w)),
            (cbcr_key(key), (2, h // 2, w // 2)),
        ]:
            schema.fields[str(component_key)] = ed.FieldSpec(
                key=str(component_key), shape=shape, dtype='uint8',
                role='observation', low=0, high=255, layout='CHW',
            )
    replay_buffer = ed.Dataset.create(schema=schema, path=replay_buf_data_path)

    agent_cfg = Agent.Config(
        obs_space=observation_space,
        action_space=action_space,
        env=env,
        replay_buffer=replay_buffer,
        obs_encoder=obs_encoder,
        world_model=world_model,
        controller=controller,
        training=Agent.Config.OnlineTrainingConfig(
            num_epochs=num_epochs,
            tokenizer_batch_size=tokenizer_batch_size,
            tokenizer_steps_per_epoch=tokenizer_steps_per_epoch,
            tokenizer_train_from_epoch=tokenizer_train_from_epoch,
            tokenizer_max_grad_norm=tokenizer_max_grad_norm,
            world_model_batch_size=world_model_batch_size,
            world_model_horizon=world_model_horizon,
            world_model_steps_per_epoch=world_model_steps_per_epoch,
            world_model_train_from_epoch=world_model_train_from_epoch,
            world_model_min_segment_length=world_model_min_segment_length,
            world_model_max_grad_norm=world_model_max_grad_norm,
            controller_steps_per_epoch=controller_steps_per_epoch,
            controller_train_from_epoch=controller_train_from_epoch,
            controller_max_grad_norm=controller_max_grad_norm,
            collection_steps_per_epoch=collection_steps_per_epoch,
            controller_test_frequency=controller_test_frequency,
        ),
        read_chunk_size=read_chunk_size,
        test_env=test_env,
    )

    return agent_cfg


def get_agent_offline_config(
        env: gym.Env, 
        env_name: str,
        replay_buf_data_path: Path = None, 
        test_env: gym.Env = None,
        baseline: Literal['hi', 'ar', 'naive'] = 'hi',
    ):
    cfg = get_agent_online_config(
        env=env, 
        env_name=env_name,
        replay_buf_data_path=replay_buf_data_path, 
        test_env=test_env,
        baseline=baseline,
    )

    tokenizer_batch_size = 32
    tokenizer_steps_per_epoch = 3000
    tokenizer_num_epochs = 10

    world_model_batch_size = 8
    # Note: overriding the horizon value requires updating the controller,
    # ==> see online config.
    world_model_horizon = cfg.training.world_model_horizon
    world_model_steps_per_epoch = 3000
    world_model_min_segment_length = 4
    world_model_num_epochs = 100

    controller_steps_per_epoch = 100
    controller_num_epochs = 500

    training_cfg = Agent.Config.OfflineTrainingConfig(
        tokenizer_batch_size=tokenizer_batch_size,
        tokenizer_steps_per_epoch=tokenizer_steps_per_epoch,
        tokenizer_num_epochs=tokenizer_num_epochs,
        tokenizer_max_grad_norm=cfg.training.tokenizer_max_grad_norm,
        world_model_batch_size=world_model_batch_size,
        world_model_steps_per_epoch=world_model_steps_per_epoch,
        world_model_horizon=world_model_horizon,
        world_model_min_segment_length=world_model_min_segment_length,
        world_model_num_epochs=world_model_num_epochs,
        world_model_max_grad_norm=cfg.training.world_model_max_grad_norm,
        controller_steps_per_epoch=controller_steps_per_epoch,
        controller_num_epochs=controller_num_epochs,
        controller_max_grad_norm=cfg.training.controller_max_grad_norm,
    )

    cfg.training = training_cfg

    cfg.replay_buffer_storage.loads(replay_buf_data_path)

    return cfg
