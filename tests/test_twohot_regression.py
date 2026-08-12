"""Unit tests for `TwoHotRegressionHead`, the distributional value head.

The property that makes two-hot a drop-in for scalar regression is that the encoding is
*unbiased*: the expectation of the two-hot target equals the target itself, so the
discretization costs resolution but never accuracy. Most of these tests pin that.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from horizon_imagination.modules.regression import (
    TwoHotRegressionHead, sym_log, sym_exp,
)


def _head(num_bins=129, v_min=-20.0, v_max=20.0, in_features=8):
    return TwoHotRegressionHead(
        in_features=in_features, num_bins=num_bins, v_min=v_min, v_max=v_max
    )


class TestTwoHotEncoding:
    def test_it_is_a_distribution(self):
        head = _head()
        enc = head.two_hot(torch.randn(4, 5) * 3)

        assert enc.shape == (4, 5, 129)
        assert torch.allclose(enc.sum(-1), torch.ones(4, 5), atol=1e-5)
        assert torch.all(enc >= 0)

    def test_at_most_two_bins_are_occupied(self):
        head = _head()
        enc = head.two_hot(torch.randn(200) * 5)

        assert int((enc > 1e-8).sum(-1).max()) <= 2

    def test_expectation_recovers_the_target(self):
        """Unbiasedness -- the whole reason this can replace a scalar regression."""
        head = _head()
        y = torch.linspace(-19.5, 19.5, 500)

        recovered = (head.two_hot(y) * head.bins).sum(-1)
        assert torch.allclose(recovered, y, atol=1e-4)

    def test_a_target_on_a_bin_centre_is_one_hot(self):
        head = _head(num_bins=5, v_min=-2.0, v_max=2.0)   # bins at -2,-1,0,1,2
        enc = head.two_hot(torch.tensor([-2.0, 0.0, 2.0]))

        assert torch.allclose(enc[0], torch.tensor([1.0, 0, 0, 0, 0]), atol=1e-6)
        assert torch.allclose(enc[1], torch.tensor([0, 0, 1.0, 0, 0]), atol=1e-6)
        assert torch.allclose(enc[2], torch.tensor([0, 0, 0, 0, 1.0]), atol=1e-6)

    def test_a_midpoint_splits_evenly(self):
        head = _head(num_bins=5, v_min=-2.0, v_max=2.0)
        enc = head.two_hot(torch.tensor([0.5]))

        assert torch.allclose(enc[0], torch.tensor([0, 0, 0.5, 0.5, 0]), atol=1e-6)

    def test_out_of_range_targets_saturate_rather_than_wrap(self):
        head = _head(num_bins=5, v_min=-2.0, v_max=2.0)
        enc = head.two_hot(torch.tensor([-99.0, 99.0]))

        assert torch.allclose(enc[0], torch.tensor([1.0, 0, 0, 0, 0]), atol=1e-6)
        assert torch.allclose(enc[1], torch.tensor([0, 0, 0, 0, 1.0]), atol=1e-6)


class TestForwardAndLoss:
    def test_shapes_follow_the_scalar_head_contract(self):
        """`(value, logits)`, with value shaped like the input's batch dims."""
        head = _head()
        value, logits = head(torch.randn(3, 7, 8))

        assert value.shape == (3, 7)
        assert logits.shape == (3, 7, 129)

    def test_masked_indexing_keeps_the_bin_axis(self):
        """How the controller slices logits: a 2-D mask over a (B, T, K) tensor."""
        head = _head()
        _, logits = head(torch.randn(3, 7, 8))
        mask = torch.zeros(3, 7, dtype=torch.bool)
        mask[0, 1] = mask[2, 4] = True

        assert logits[torch.where(mask)].shape == (2, 129)

    def test_loss_is_minimised_at_the_target(self):
        head = _head(num_bins=41, v_min=-4.0, v_max=4.0)
        target = torch.tensor([2.0])
        symlog_target = sym_log(target)

        perfect = torch.log(head.two_hot(symlog_target).clamp_min(1e-12))
        wrong = torch.log(head.two_hot(symlog_target + 2.0).clamp_min(1e-12))

        assert head.compute_loss(perfect, target) < head.compute_loss(wrong, target)

    def test_loss_rejects_a_missing_bin_axis(self):
        head = _head()
        with pytest.raises(AssertionError, match="bin axis"):
            head.compute_loss(torch.randn(4, 5), torch.randn(4, 5))

    def test_it_trains_towards_a_constant_target(self):
        torch.manual_seed(0)
        head = _head(num_bins=41, v_min=-4.0, v_max=4.0, in_features=4)
        opt = torch.optim.Adam(head.parameters(), lr=0.05)

        x = torch.randn(64, 4)
        target = torch.full((64,), 7.0)
        for _ in range(300):
            opt.zero_grad()
            _, logits = head(x)
            head.compute_loss(logits, target).backward()
            opt.step()

        value, _ = head(x)
        assert value.mean().item() == pytest.approx(7.0, rel=0.05)


class TestRewardModelShapePaths:
    """
    `RewardDoneModel.training_step` feeds the head either a masked 2-D tensor or, with
    no mask, a 3-D one flattened to 2-D -- and `forward` feeds it 3-D during imagination.
    All three have to work, since the reward head is upstream of every lambda-return.
    """

    def test_masked_two_dimensional_path(self):
        head = _head()
        x = torch.randn(11, 8)                  # x[torch.where(mask)] -> (n, latent)
        value, logits = head(x)

        assert value.shape == (11,)
        assert logits.shape == (11, 129)
        assert torch.isfinite(head.compute_loss(logits, torch.randn(11)))

    def test_unmasked_flattened_path(self):
        head = _head()
        x = torch.randn(3, 5, 8).flatten(0, -2)  # (B, T, latent) -> (B*T, latent)
        value, logits = head(x)

        assert value.shape == (15,)
        assert torch.isfinite(head.compute_loss(logits, torch.randn(3, 5).flatten()))

    def test_imagination_forward_path_keeps_batch_and_time(self):
        head = _head()
        value, logits = head(torch.randn(4, 6, 8))

        assert value.shape == (4, 6)            # what `imagine` stores as traj reward
        assert logits.shape == (4, 6, 129)

    def test_l1_diagnostic_still_lines_up(self):
        """`reward_l1` compares the scalar readout against flattened targets."""
        head = _head()
        value, _ = head(torch.randn(3, 5, 8))
        target = torch.randn(3, 5)

        assert torch.isfinite(
            torch.nn.functional.l1_loss(value.flatten(), target.flatten())
        )


class TestBoundedReadout:
    def test_value_cannot_exceed_the_bin_range(self):
        """
        The property a free scalar head lacks: the readout is a convex combination of
        fixed bin centres, so no amount of logit drift produces an exploding `sym_exp`.
        """
        head = _head(num_bins=41, v_min=-4.0, v_max=4.0, in_features=4)
        with torch.no_grad():                    # drive the head far out of range
            head.linear.weight *= 1e4
            head.linear.bias *= 1e4

        value, _ = head(torch.randn(256, 4) * 100)

        assert torch.isfinite(value).all()
        assert value.abs().max().item() <= sym_exp(torch.tensor(4.0)).item() + 1e-3

    def test_the_bin_buffer_follows_dtype(self):
        head = _head().to(torch.float64)
        assert head.bins.dtype == torch.float64

        value, _ = head(torch.randn(2, 8, dtype=torch.float64))
        assert value.dtype == torch.float64
