import gymnasium as gym
import numpy as np
import pytest
import torch

from horizon_imagination.models.tokenizer.vector import VectorAutoencoder, VectorKeySpec
from horizon_imagination.utilities.types import ObsKey, Modality


PROPRIO = ObsKey.from_parts(Modality.vector, 'proprio')
GOAL = ObsKey.from_parts(Modality.vector, 'goal')


def make_ae(specs, latent_dim=8, chunk_size=None, per_key_chunk_size=None):
    return VectorAutoencoder.Config(
        obs_specs=specs,
        latent_dim=latent_dim,
        hidden_dim=32,
        chunk_size=chunk_size,
        per_key_chunk_size=per_key_chunk_size,
    ).make_instance()


# chunk_size is what turns a vector key into several tokens; P = 1 is the default,
# but every shape must be derived from P so that P > 1 stays a config change.
@pytest.mark.parametrize("obs_dim,chunk_size,expected_tokens", [
    (7, None, 1),
    (8, 2, 4),
    (7, 2, 4),  # padded tail
    (7, 1, 7),  # per-feature tokens
])
def test_latent_shape_and_roundtrip_shapes(obs_dim, chunk_size, expected_tokens):
    ae = make_ae({PROPRIO: VectorKeySpec(obs_dim=obs_dim)}, latent_dim=8, chunk_size=chunk_size)

    assert ae.num_tokens(PROPRIO) == expected_tokens
    assert ae.latent_shape(PROPRIO) == (8, expected_tokens, 1)

    x = torch.randn(3, 5, obs_dim)
    z = ae.encode(x, PROPRIO)
    assert z.shape == (3, 5, 8, expected_tokens, 1)  # leading dims preserved
    assert ae.decode(z, PROPRIO).shape == x.shape


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_latents_are_tanh_bounded(chunk_size):
    ae = make_ae({PROPRIO: VectorKeySpec(obs_dim=7)}, chunk_size=chunk_size)

    # Even far-out-of-distribution inputs stay inside the shared latent range:
    assert ae.encode(torch.randn(64, 7) * 100, PROPRIO).abs().max() <= 1.0
    assert ae.encode(torch.randn(64, 7), PROPRIO).abs().max() < 1.0


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_reconstruction_improves_with_training(chunk_size):
    torch.manual_seed(0)
    ae = make_ae({PROPRIO: VectorKeySpec(obs_dim=7)}, latent_dim=16, chunk_size=chunk_size)
    optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-2)

    # A low-dimensional manifold, which an autoencoder should be able to fit:
    latents = torch.randn(256, 2)
    x = torch.cat([latents, latents ** 2, latents[:, :1] * latents[:, 1:], torch.zeros(256, 2)], dim=-1)
    log = lambda *args, **kwargs: None

    initial = float(ae.training_step({PROPRIO: x}, 0, log_dict_fn=log).detach())
    for step in range(300):
        optimizer.zero_grad()
        loss = ae.training_step({PROPRIO: x}, step, log_dict_fn=log)
        loss.backward()
        optimizer.step()

    assert float(loss.detach()) < initial / 2


def test_keys_are_independent_and_may_differ_in_size():
    ae = make_ae(
        {PROPRIO: VectorKeySpec(obs_dim=7), GOAL: VectorKeySpec(obs_dim=3)},
        latent_dim=8,
        per_key_chunk_size={GOAL: 1},
    )
    assert ae.num_tokens(PROPRIO) == 1 and ae.num_tokens(GOAL) == 3

    loss = ae.training_step(
        {PROPRIO: torch.randn(4, 7), GOAL: torch.randn(4, 3)}, 0, log_dict_fn=lambda *a, **k: None
    )
    assert torch.isfinite(loss)


def test_bounded_box_gives_a_fixed_normalization():
    space = gym.spaces.Box(low=-2.0, high=4.0, shape=(3,), dtype=np.float32)
    ae = make_ae({PROPRIO: VectorKeySpec.from_box(space)})
    key_ae = ae.autoencoders[str(PROPRIO)]

    assert key_ae.has_fixed_stats
    assert torch.allclose(key_ae.normalize(torch.full((1, 3), 4.0)), torch.ones(1, 3))
    assert torch.allclose(key_ae.normalize(torch.full((1, 3), -2.0)), -torch.ones(1, 3))

    # An unbounded Box falls back to running statistics, updated only while training:
    unbounded = make_ae({PROPRIO: VectorKeySpec.from_box(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32))})
    unbounded_ae = unbounded.autoencoders[str(PROPRIO)]
    assert not unbounded_ae.has_fixed_stats

    before = unbounded_ae.center.clone()
    unbounded.encode(torch.full((8, 3), 10.0), PROPRIO)
    assert torch.equal(unbounded_ae.center, before)

    unbounded.training_step({PROPRIO: torch.full((8, 3), 10.0)}, 0, log_dict_fn=lambda *a, **k: None)
    assert not torch.equal(unbounded_ae.center, before)
