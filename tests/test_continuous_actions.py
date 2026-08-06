"""Unit tests for the continuous (Box) action path.

Covers the Gaussian policy head, the stable continuous action producer and the
distribution helpers the actor losses use. CPU-only and fast -- the end-to-end
continuous run lives in `test_dict_obs_end_to_end.py`, which needs CUDA.
"""
import gymnasium as gym
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch.distributions import Categorical, Independent, Normal

from horizon_imagination.models.controller.actor_critic import GaussianActorHead
from horizon_imagination.models.controller.controller import (
    Controller, _detach_dist, _slice_dist, _cat_dists,
)
from horizon_imagination.models.world_model.action_producer import (
    StableContinuousActionProducer, StableDiscreteActionProducer, make_stable_action_producer,
)
from horizon_imagination.modules.embeddings import build_action_embedder, action_dim


def _normal(mean, std=None):
    std = torch.ones_like(mean) if std is None else std
    return Independent(Normal(mean, std), 1)


class TestGaussianActorHead:
    def _head(self, latent_dim=8, adim=3, **kwargs):
        return GaussianActorHead.Config(
            latent_dim=latent_dim, action_dim=adim, **kwargs
        ).make_instance()

    def test_log_prob_and_entropy_are_per_step_scalars(self):
        """The (B, T) shapes the actor-critic losses assume, as for a Categorical."""
        head = self._head()
        dist = head(torch.randn(2, 5, 8))

        assert dist.batch_shape == (2, 5)
        action = dist.sample()
        assert action.shape == (2, 5, 3)
        assert dist.log_prob(action).shape == (2, 5)
        assert dist.entropy().shape == (2, 5)

    def test_initial_std_follows_init_log_std(self):
        head = self._head(init_log_std=-1.0)
        dist = head(torch.randn(4, 3, 8))

        assert torch.allclose(
            dist.base_dist.scale, torch.full_like(dist.base_dist.scale, np.exp(-1.0)),
            atol=0.05,
        )

    def test_log_std_is_clamped(self):
        head = self._head(log_std_min=-0.5, log_std_max=0.5)
        # Blow up the head so the raw log_std would land far outside the range:
        with torch.no_grad():
            head.head.weight *= 1e3
            head.head.bias *= 1e3

        scale = head(torch.randn(4, 3, 8)).base_dist.scale
        assert torch.all(scale >= np.exp(-0.5) - 1e-5)
        assert torch.all(scale <= np.exp(0.5) + 1e-5)

    def test_actor_bias_is_rejected(self):
        with pytest.raises(AssertionError, match="actor_bias"):
            self._head(actor_bias=(0.0, 1.0, 0.0))


class TestStableContinuousActionProducer:
    def test_reuses_one_noise_draw_across_calls(self):
        """
        The whole point of the stable producer: with `eps` held fixed, moving the
        policy's mean by `d` moves the action by exactly `d`, instead of resampling
        somewhere unrelated.
        """
        producer = StableContinuousActionProducer()
        mean = torch.randn(2, 4, 3)

        a0, _ = producer(_normal(mean))
        shift = torch.randn(2, 4, 3)
        a1, _ = producer(_normal(mean + shift))

        assert torch.allclose(a1 - a0, shift, atol=1e-6)

    def test_action_tracks_the_std(self):
        producer = StableContinuousActionProducer()
        mean = torch.zeros(2, 4, 3)

        a0, _ = producer(_normal(mean, std=torch.ones_like(mean)))
        a1, _ = producer(_normal(mean, std=torch.full_like(mean, 2.0)))

        assert torch.allclose(a1, 2 * a0, atol=1e-6)

    def test_log_prob_matches_the_distribution(self):
        producer = StableContinuousActionProducer()
        mean, std = torch.randn(2, 4, 3), torch.rand(2, 4, 3) + 0.1
        dist = _normal(mean, std)

        action, log_pi = producer(dist)

        assert action.shape == (2, 4, 3)
        assert log_pi.shape == (2, 4)
        assert torch.allclose(log_pi, Normal(mean, std).log_prob(action).sum(-1), atol=1e-6)

    def test_action_is_detached_so_reinforce_keeps_its_gradient(self):
        """
        Regression guard for the reparameterization trap: if the returned action kept
        its graph, `log_prob(a)` would reduce to `-eps^2/2 - log(std) - c`, whose
        gradient w.r.t. the mean is exactly zero -- the policy mean would never learn.
        """
        mean = torch.randn(2, 4, 3, requires_grad=True)
        std = torch.rand(2, 4, 3) + 0.1

        action, log_pi = StableContinuousActionProducer()(_normal(mean, std))

        assert not action.requires_grad
        log_pi.sum().backward()
        assert mean.grad is not None and mean.grad.abs().max() > 0

    def test_a_live_action_would_zero_the_mean_gradient(self):
        """The failure mode the test above guards against, shown explicitly."""
        mean = torch.randn(2, 4, 3, requires_grad=True)
        std = torch.rand(2, 4, 3) + 0.1
        eps = torch.randn(2, 4, 3)

        live_action = mean + std * eps  # NOT detached
        _normal(mean, std).log_prob(live_action).sum().backward()

        assert torch.allclose(mean.grad, torch.zeros_like(mean.grad), atol=1e-6)

    def test_dispatch_by_distribution(self):
        assert isinstance(
            make_stable_action_producer(_normal(torch.zeros(2, 3, 4))),
            StableContinuousActionProducer,
        )
        assert isinstance(
            make_stable_action_producer(Categorical(logits=torch.zeros(2, 3, 4))),
            StableDiscreteActionProducer,
        )


class TestDistributionHelpers:
    @staticmethod
    def _make(family, b=2, t=5, a=3):
        if family == "categorical":
            return Categorical(logits=torch.randn(b, t, a, requires_grad=True))
        return _normal(torch.randn(b, t, a, requires_grad=True), torch.rand(b, t, a) + 0.1)

    @pytest.mark.parametrize("family", ["categorical", "normal"])
    def test_detach_drops_the_graph_but_not_the_values(self, family):
        dist = self._make(family)
        detached = _detach_dist(dist)

        sample = dist.sample()
        assert torch.allclose(detached.log_prob(sample), dist.log_prob(sample))
        assert not detached.log_prob(sample).requires_grad

    @pytest.mark.parametrize("family", ["categorical", "normal"])
    def test_slice_and_cat_along_batch_dims(self, family):
        dist = self._make(family, b=2, t=5)

        assert _slice_dist(dist, (slice(None), slice(None, -1))).batch_shape == (2, 4)
        assert _cat_dists([dist, dist], dim=1).batch_shape == (2, 10)

        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[0, 1] = mask[1, 3] = True
        assert _slice_dist(dist, torch.where(mask)).batch_shape == (2,)


def test_kl_distillation_matches_the_intended_soft_cross_entropy_gradient():
    """
    The discrete distillation term is now KL(clean || noisy). Against the *intended*
    per-step soft-target cross-entropy it differs only by the constant H(clean), so the
    gradient w.r.t. the noisy logits is identical.
    """
    clean_logits = torch.randn(6, 4)
    noisy_logits = torch.randn(6, 4, requires_grad=True)

    kl = torch.distributions.kl_divergence(
        Categorical(logits=clean_logits), Categorical(logits=noisy_logits)
    ).mean()
    (kl_grad,) = torch.autograd.grad(kl, noisy_logits)

    ce = torch.nn.functional.cross_entropy(
        noisy_logits, torch.softmax(clean_logits, dim=-1)
    )
    (ce_grad,) = torch.autograd.grad(ce, noisy_logits)

    assert torch.allclose(kl_grad, ce_grad, atol=1e-6)


class TestActionForEnv:
    """
    `Controller._clip_action` / `._action_for_env` are the only place the Box bounds are
    enforced, so they are checked directly -- building a Controller needs a GPU.
    """

    class _Fake:
        def __init__(self, action_space):
            self.action_space = action_space

        clip = Controller._clip_action
        for_env = Controller._action_for_env

    def test_box_action_is_clipped_and_converted(self):
        space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        fake = self._Fake(space)

        action = torch.tensor([[[5.0, -5.0, 0.25]]])
        for_env = fake.for_env(fake.clip(action))

        assert isinstance(for_env, np.ndarray) and for_env.dtype == np.float32
        np.testing.assert_allclose(for_env, [1.0, -1.0, 0.25])
        assert space.contains(for_env)

    def test_single_dimension_box_stays_an_array(self):
        """`.item()` used to scalarize any 1-element action, breaking a Box(1,) env."""
        fake = self._Fake(gym.spaces.Box(-1.0, 1.0, (1,), np.float32))

        for_env = fake.for_env(fake.clip(torch.tensor([[[0.5]]])))

        assert isinstance(for_env, np.ndarray) and for_env.shape == (1,)

    def test_discrete_action_becomes_a_python_int(self):
        fake = self._Fake(gym.spaces.Discrete(4))
        action = torch.tensor([[2]])

        assert fake.clip(action) is action
        assert fake.for_env(action) == 2
        assert isinstance(fake.for_env(action), int)


class TestActionEmbedder:
    def test_box_actions_are_projected_to_the_embedding_width(self):
        space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        embedder = build_action_embedder(space, embed_dim=16)

        assert embedder(torch.randn(2, 5, 3)).shape == (2, 5, 16)
        assert action_dim(space) == 3

    def test_discrete_actions_are_looked_up(self):
        embedder = build_action_embedder(gym.spaces.Discrete(7), embed_dim=16)

        assert embedder(torch.randint(0, 7, (2, 5))).shape == (2, 5, 16)

    def test_unsupported_space(self):
        with pytest.raises(NotImplementedError):
            build_action_embedder(gym.spaces.MultiBinary(3), embed_dim=16)
