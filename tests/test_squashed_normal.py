"""Unit tests for `SquashedNormal`, the bounded continuous policy.

The tests that matter most here are the entropy ones: an unbounded Gaussian's entropy
grows without limit, so a fixed entropy bonus fed a constant upward gradient into
`log_std` that nothing balanced, and sigma ran to its clamp. Squashing is what turns
that into a bounded objective with a finite maximizer.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from torch.distributions import Independent, Normal, kl_divergence

from horizon_imagination.modules.distributions import (
    SquashedNormal, sample_with_log_prob, tanh_log_jacobian,
)


def _dist(b=2, t=5, a=3, scale=None, requires_grad=False):
    loc = torch.randn(b, t, a, requires_grad=requires_grad)
    scale = torch.rand(b, t, a) + 0.1 if scale is None else torch.full((b, t, a), scale)
    return SquashedNormal(loc, scale)


class TestShapeContract:
    def test_batch_and_event_shapes(self):
        """(B, T) log-probs and entropies -- the contract the actor losses assume."""
        dist = _dist(b=2, t=5, a=3)

        assert dist.batch_shape == (2, 5)
        assert dist.event_shape == (3,)
        assert dist.rsample().shape == (2, 5, 3)
        assert dist.log_prob(dist.sample()).shape == (2, 5)
        assert dist.entropy().shape == (2, 5)

    def test_masked_indexing_collapses_batch_dims_only(self):
        """What `_slice_dist(dist, torch.where(mask))` does inside the actor loss."""
        dist = _dist(b=2, t=5, a=3)
        sliced = SquashedNormal(dist.loc[:, :2], dist.scale[:, :2])

        assert sliced.batch_shape == (2, 2)
        assert sliced.event_shape == (3,)


class TestLogProb:
    def test_support_is_the_open_box(self):
        assert torch.all(_dist(scale=5.0).sample().abs() <= 1.0)

    def test_both_log_prob_paths_agree(self):
        dist = _dist()
        u = dist.loc + dist.scale * torch.randn_like(dist.loc)

        assert torch.allclose(
            dist.log_prob_from_pre_tanh(u), dist.log_prob(torch.tanh(u)), atol=1e-4
        )

    def test_pre_tanh_path_survives_saturation(self):
        """
        `atanh(tanh(u))` loses all precision past |u| ~ 9 in fp32 and the naive
        `log(1 - a^2)` underflows to -inf; the pre-tanh path is what the hot path uses.
        """
        dist = SquashedNormal(torch.zeros(1, 1, 2), torch.ones(1, 1, 2))
        u = torch.full((1, 1, 2), 20.0)

        assert torch.isfinite(dist.log_prob_from_pre_tanh(u)).all()
        assert torch.isfinite(tanh_log_jacobian(u)).all()
        assert torch.isinf(torch.log(1 - torch.tanh(u).pow(2))).all()  # the naive form

    def test_density_integrates_to_one(self):
        """Catches a wrong sign or a missing factor in the tanh Jacobian."""
        dist = SquashedNormal(torch.tensor([[[0.3]]]), torch.tensor([[[0.8]]]))
        a = torch.linspace(-1 + 1e-6, 1 - 1e-6, 200_001).reshape(-1, 1, 1)

        density = dist.log_prob(a).exp().squeeze()
        integral = torch.trapz(density, a.squeeze())

        assert integral.item() == pytest.approx(1.0, abs=1e-3)


class TestEntropy:
    """The regression tests for the failure this class was written to fix."""

    def test_entropy_is_bounded_above(self):
        """
        Bounded by `A * log 2`, the entropy of the uniform on `(-1, 1)^A`, no matter how
        wide the underlying Gaussian gets. The unbounded Gaussian has no such ceiling.
        """
        a = 3
        for scale in (0.5, 1.0, 5.0, 50.0, 500.0):
            dist = _dist(b=64, t=64, a=a, scale=scale)
            assert dist.entropy().mean().item() < a * math.log(2) + 0.05

    def test_entropy_falls_once_tanh_saturates(self):
        """
        The restoring force. Past sigma ~ 1 a wider policy is *less* entropic, so the
        entropy bonus stops pushing sigma up -- which is exactly what it did before.
        """
        a = 3
        entropies = [
            _dist(b=64, t=64, a=a, scale=s).entropy().mean().item()
            for s in (0.5, 1.0, 3.0, 10.0)
        ]

        assert entropies[1] > entropies[0]        # still rising below sigma ~ 1
        assert entropies[2] < entropies[1]        # falling past it
        assert entropies[3] < entropies[2]

    @pytest.mark.parametrize("log_scale_value,expected_sign", [(-1.0, 1.0), (2.0, -1.0)])
    def test_entropy_gradient_reverses_sign(self, log_scale_value, expected_sign):
        """
        The mechanism, stated as a gradient. For an unbounded Gaussian
        `d H / d log_sigma == +1` everywhere -- a constant push outward. Squashed, the
        gradient flips sign once tanh saturates, so the bonus has a finite maximizer.
        """
        loc = torch.zeros(256, 256, 2)
        log_scale = torch.full((256, 256, 2), log_scale_value, requires_grad=True)

        SquashedNormal(loc, log_scale.exp()).entropy().mean().backward()
        grad = log_scale.grad.mean().item()

        assert math.copysign(1.0, grad) == expected_sign

        # The contrast: the unbounded Gaussian's gradient is +1 per dim, always.
        log_scale_ref = torch.full((8, 8, 2), log_scale_value, requires_grad=True)
        Independent(Normal(torch.zeros(8, 8, 2), log_scale_ref.exp()), 1).entropy().sum().backward()
        assert torch.allclose(log_scale_ref.grad, torch.ones_like(log_scale_ref), atol=1e-5)

    def test_entropy_estimator_is_unbiased(self):
        """Agrees with the plain `-log_prob(rsample())` estimator it Rao-Blackwellizes."""
        dist = _dist(b=256, t=256, a=3, scale=0.7)

        naive = -dist.log_prob(dist.rsample())
        assert dist.entropy().mean().item() == pytest.approx(naive.mean().item(), abs=0.02)


class TestKL:
    def test_matches_the_base_normal_kl(self):
        """Exact, since tanh is a diffeomorphism -- the Jacobian terms cancel."""
        p, q = _dist(), _dist()
        expected = kl_divergence(
            Independent(Normal(p.loc, p.scale), 1), Independent(Normal(q.loc, q.scale), 1)
        )

        kl = kl_divergence(p, q)
        assert kl.shape == (2, 5)
        assert torch.allclose(kl, expected)

    def test_is_zero_for_identical_distributions(self):
        p = _dist()
        assert torch.allclose(kl_divergence(p, p), torch.zeros_like(kl_divergence(p, p)))


class TestSampleWithLogProb:
    def test_action_is_detached_but_log_prob_carries_gradient(self):
        loc = torch.randn(2, 4, 3, requires_grad=True)
        dist = SquashedNormal(loc, torch.rand(2, 4, 3) + 0.1)

        action, log_pi = sample_with_log_prob(dist)

        assert not action.requires_grad
        assert torch.all(action.abs() < 1.0)
        assert log_pi.shape == (2, 4)

        log_pi.sum().backward()
        assert loc.grad is not None and loc.grad.abs().max() > 0

    def test_falls_back_for_other_families(self):
        from torch.distributions import Categorical

        dist = Categorical(logits=torch.randn(2, 4, 5))
        action, log_pi = sample_with_log_prob(dist)

        assert action.shape == (2, 4)
        assert log_pi.shape == (2, 4)
