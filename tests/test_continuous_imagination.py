"""Integration test of the continuous-action path through imagination.

Builds the world model and controller for a `Box` action space and runs one controller
training step on a synthetic batch, which exercises the pieces a `Box` env touches that
unit tests cannot: the DiT's action embedder, the stable continuous action producer
driving the denoising loop, and the actor/critic losses over a Gaussian policy.

The replay buffer is deliberately left out (the batch is synthesized directly), so this
runs without a collection loop; `test_dict_obs_end_to_end.py` covers collection.
"""
import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.obs_codec import y_key, cbcr_key
from horizon_imagination.utilities.types import vector_keys

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

RESOLUTION = 64
ACTION_DIM = 3
B, T = 2, 5
HORIZON = 4


def _obs_space():
    return gym.spaces.Dict({
        'image|rgb': gym.spaces.Box(0, 255, (3, RESOLUTION, RESOLUTION), np.uint8),
        'vector|proprio': gym.spaces.Box(-1.0, 1.0, (7,), np.float32),
    })


def _build_controller(action_space):
    """The `get_agent_online_config` pipeline, minus the replay buffer."""
    from config.tokenizer.image.cosmos import get_cosmos_tokenizer_online_config
    from config.world_model.flow_online import get_world_model_online_config
    from config.controller.default import get_controller_config
    from horizon_imagination.models.tokenizer import VectorAutoencoder, VectorKeySpec

    device, dtype = torch.device('cuda'), None
    obs_space = _obs_space()

    tokenizer_cfg = get_cosmos_tokenizer_online_config(dtype=dtype, resolution=RESOLUTION)
    tokenizer = tokenizer_cfg.make_instance().to(device)
    latent_size = RESOLUTION // tokenizer_cfg.network_cfg.spatial_compression

    vector_autoencoder = VectorAutoencoder.Config(
        obs_specs={
            k: VectorKeySpec.from_box(obs_space.spaces[k])
            for k in vector_keys(obs_space.spaces.keys())
        },
        latent_dim=32,
        chunk_size=None,
        device=device,
        dtype=dtype,
    ).make_instance()

    shared = dict(
        obs_space=obs_space,
        action_space=action_space,
        tokenizer_channels=tokenizer_cfg.latent_channels,
        latent_spatial_shape=(latent_size, latent_size),
        vector_autoencoder=vector_autoencoder,
        device=device,
        dtype=dtype,
    )
    world_model = get_world_model_online_config(
        image_tokenizer=tokenizer, baseline='hi', decay_horizon=4, **shared
    ).make_instance().to(device)

    controller_cfg = get_controller_config(
        env_name='Synthetic/Continuous-v0',
        world_model=world_model,
        imagination_horizon=HORIZON,
        budget=2,
        **shared,
    )
    controller_cfg.imagination_batch_size = B
    controller_cfg.imagination_horizon = HORIZON
    controller_cfg.num_denoising_steps = 2
    controller_cfg.controller_context_length = T

    return controller_cfg.make_instance().to('cuda')


def _synthetic_batch(action_space):
    """What `EpochDataIterator`'s controller stream yields: YCbCr obs + actions + mask."""
    rgb = 'image|rgb'
    if isinstance(action_space, gym.spaces.Discrete):
        action = torch.randint(0, action_space.n, (B, T), device='cuda')
    else:
        action = torch.rand(B, T, ACTION_DIM, device='cuda') * 2 - 1

    half = RESOLUTION // 2
    all_obs = TensorDict(
        {
            str(y_key(rgb)): torch.randint(
                0, 255, (B, T, 1, RESOLUTION, RESOLUTION), dtype=torch.uint8, device='cuda'
            ),
            str(cbcr_key(rgb)): torch.randint(
                0, 255, (B, T, 2, half, half), dtype=torch.uint8, device='cuda'
            ),
            'vector|proprio': torch.rand(B, T, 7, device='cuda') * 2 - 1,
        },
        batch_size=(B, T),
        device='cuda',
    )
    return TensorDict(
        {
            'all_observations': all_obs,
            'action': action,
            'mask': torch.ones(B, T, dtype=torch.bool, device='cuda'),
        },
        batch_size=(B, T),
        device='cuda',
    )


@pytest.mark.parametrize("action_space", [
    gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32),
    gym.spaces.Discrete(4),
], ids=['box', 'discrete'])
def test_controller_training_step(action_space):
    from horizon_imagination.models.controller.actor_critic import (
        GaussianActorHead, DiscreteActorHead,
    )

    controller = _build_controller(action_space)
    expected_head = (
        GaussianActorHead if isinstance(action_space, gym.spaces.Box) else DiscreteActorHead
    )
    assert isinstance(controller.actor_critic.actor, expected_head)

    logged = {}
    loss = controller.training_step(
        _synthetic_batch(action_space), 0, lambda d, **k: logged.update(d)
    )

    assert torch.isfinite(loss), f"Got {loss}"
    loss.backward()

    # The policy head must actually receive gradient -- the REINFORCE term is silently
    # dead if the action reaches `log_prob` still carrying its reparameterization graph.
    head_grad = controller.actor_critic.actor.head.weight.grad
    assert head_grad is not None and head_grad.abs().max() > 0

    if isinstance(action_space, gym.spaces.Box):
        assert 'actor_critic/avg_action_drift' in logged
        assert float(logged['actor_critic/avg_action_drift']) > 0
    else:
        assert 'actor_critic/avg_num_action_changes' in logged
    assert 'actor_critic/avg_action_change_time' in logged


def test_stable_producer_keeps_actions_coupled_across_denoising_steps():
    """
    Over one imagination, consecutive denoising steps must move the action by far less
    than independent samples from the same policy would -- the property the fixed `eps`
    buys, and the reason this producer exists.
    """
    from horizon_imagination.models.world_model.action_producer import (
        StablePolicyActionProducer, NaivePolicyActionProducer,
    )

    action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
    controller = _build_controller(action_space)
    batch = _synthetic_batch(action_space)

    def mean_step_distance(producer_cls):
        torch.manual_seed(0)
        world_model = controller.config.world_model
        context_obs = world_model.get_obs_from_batch(batch)
        controller.actor_critic.reset(batch['action'], context_obs, batch['mask'])
        wm_context = batch[:, -1:]

        with torch.no_grad():
            segment = world_model.imagine(
                policy=producer_cls(controller.actor_critic),
                batch_size=B,
                horizon=HORIZON,
                obs_shape={k: v.shape for k, v in context_obs.items()},
                denoising_steps=4,
                context=wm_context,
                context_noise_level=0.0,
            )
        actions = torch.stack(segment['action'], dim=0)
        return torch.linalg.vector_norm(actions[1:] - actions[:-1], dim=-1).mean()

    stable = mean_step_distance(StablePolicyActionProducer)
    naive = mean_step_distance(NaivePolicyActionProducer)

    assert stable < naive / 2, f"stable={stable}, naive={naive}"
