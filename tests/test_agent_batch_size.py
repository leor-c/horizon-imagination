"""Regression test for Agent.training_step's batch-size lookup.

``EpochDataIterator`` yields a plain ``dict[ObsKey, Tensor]`` for the obs-encoder component
but TensorDicts for the world model and controller. ``Agent.training_step`` used
``batch.shape[0]`` unconditionally, which raised ``AttributeError: 'dict' object has no
attribute 'shape'`` as soon as the tokenizer began training (``tokenizer_train_from_epoch``).

The existing end-to-end test misses this because it calls
``agent.components[idx].training_step(...)`` directly, bypassing ``Agent.training_step``.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from tensordict.tensordict import TensorDict

from horizon_imagination.agent import batch_size_of


def test_plain_dict_batch_as_produced_for_the_obs_encoders():
    # Shape of EpochDataIterator.prefetch_tok's output: raw images and vectors, no time dim.
    batch = {
        'image|rgb': torch.zeros(7, 3, 64, 64, dtype=torch.uint8),
        'vector|proprio': torch.zeros(7, 5),
    }
    assert batch_size_of(batch) == 7


def test_tensordict_batch_as_produced_for_the_world_model_and_controller():
    batch = TensorDict(
        {'action': torch.zeros(7, 20, dtype=torch.long)},
        batch_size=(7, 20),
    )
    assert batch_size_of(batch) == 7


def test_single_key_dict():
    assert batch_size_of({'image|features': torch.zeros(3, 3, 64, 64)}) == 3
