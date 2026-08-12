"""Unit tests for `EMAScaler`, the advantage-normalization statistic.

The scaler takes ~200 updates to warm up at the default decay, so losing it on resume
costs real training time -- these tests pin that it now rides along in `state_dict()`,
and, just as importantly, that checkpoints written *before* it did still load.
"""
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from horizon_imagination.models.controller.return_scaler import EMAScaler


def _values(low=-1.0, high=1.0, n=10_000):
    return torch.linspace(low, high, n)


class TestEstimator:
    def test_first_update_sets_the_estimate_directly(self):
        """No annealing up from zero -- otherwise the first advantages are scaled by ~0."""
        scaler = EMAScaler(decay=0.005, quantile=0.95)
        assert not scaler.initialized
        assert scaler.estimate_high is None

        scaler.update(_values(-4.0, 4.0))

        assert scaler.initialized
        assert scaler.estimate_high.item() == pytest.approx(3.6, abs=0.05)
        assert scaler.estimate_low.item() == pytest.approx(-3.6, abs=0.05)
        assert scaler.scale.item() == pytest.approx(7.2, abs=0.1)

    def test_subsequent_updates_follow_the_documented_ema(self):
        decay = 0.1
        scaler = EMAScaler(decay=decay, quantile=0.95)
        scaler.update(_values(-1.0, 1.0))
        before = scaler.estimate_high.clone()

        scaler.update(_values(-10.0, 10.0))
        new_high = torch.quantile(_values(-10.0, 10.0), 0.95)

        expected = (1 - decay) * before + decay * new_high
        assert scaler.estimate_high.item() == pytest.approx(expected.item(), abs=1e-5)

    def test_scale_before_any_update_fails_loudly(self):
        with pytest.raises(AssertionError, match="before the first update"):
            _ = EMAScaler().scale

    def test_update_takes_no_gradient(self):
        scaler = EMAScaler()
        scaler.update(_values().requires_grad_(True))
        assert not scaler.scale.requires_grad


class TestCheckpointing:
    def test_estimate_survives_a_state_dict_round_trip(self):
        scaler = EMAScaler(decay=0.005)
        for _ in range(5):
            scaler.update(_values(-3.0, 3.0))

        restored = EMAScaler(decay=0.005)
        restored.load_state_dict(scaler.state_dict())

        assert restored.initialized
        assert restored.scale.item() == pytest.approx(scaler.scale.item(), abs=1e-6)

    def test_it_appears_in_a_parent_module_state_dict(self):
        """What makes it ride along in the controller checkpoint."""
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.return_scaler = EMAScaler()

        parent = Parent()
        parent.return_scaler.update(_values())

        keys = parent.state_dict().keys()
        assert 'return_scaler._estimate_high' in keys
        assert 'return_scaler._estimate_low' in keys
        assert 'return_scaler._initialized' in keys

    def test_it_adds_no_parameters(self):
        """Buffers, not parameters -- `configure_optimizers` must be unaffected."""
        scaler = EMAScaler()
        scaler.update(_values())
        assert list(scaler.parameters()) == []

    def test_old_checkpoints_still_load_strictly(self):
        """
        The compatibility path that matters: a checkpoint written before the scaler
        carried state has none of these keys, and `load_from_checkpoint` uses
        `strict=True`. It must still load, leaving the scaler uninitialized.
        """
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.return_scaler = EMAScaler()

        old_checkpoint = {
            k: v for k, v in Parent().state_dict().items()
            if not k.startswith('return_scaler.')
        }

        parent = Parent()
        parent.load_state_dict(old_checkpoint, strict=True)  # must not raise

        assert not parent.return_scaler.initialized
        parent.return_scaler.update(_values(-2.0, 2.0))
        assert parent.return_scaler.scale.item() == pytest.approx(3.6, abs=0.1)

    def test_strictness_is_not_weakened_for_anything_else(self):
        """
        The override forgives only its own three keys. A genuinely missing weight
        elsewhere in the model must still fail the load, or the compatibility shim
        would be hiding real checkpoint corruption.
        """
        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)
                self.return_scaler = EMAScaler()

        broken = {k: v for k, v in Parent().state_dict().items() if k != 'linear.weight'}

        with pytest.raises(RuntimeError, match="linear.weight"):
            Parent().load_state_dict(broken, strict=True)

    def test_unexpected_keys_are_still_reported(self):
        scaler = EMAScaler()
        state = dict(scaler.state_dict())
        state['_not_a_real_buffer'] = torch.zeros(())

        with pytest.raises(RuntimeError, match="Unexpected key"):
            scaler.load_state_dict(state, strict=True)


class TestDeviceAndDtype:
    def test_first_update_adopts_the_incoming_dtype(self):
        """
        Instances inside `LatentReconstructionResidual` are never `.to(device)`d, so the
        buffers must take the device/dtype of the values rather than assume the default.
        """
        scaler = EMAScaler()
        scaler.update(_values().to(torch.float64))

        assert scaler.estimate_high.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_first_update_adopts_the_incoming_device(self):
        scaler = EMAScaler()  # deliberately never moved to the GPU
        scaler.update(_values().cuda())

        assert scaler.estimate_high.is_cuda
        scaler.update(_values(-2.0, 2.0).cuda())  # the in-place EMA path must agree
        assert scaler.estimate_high.is_cuda
