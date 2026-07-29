import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")

from episodata import Dataset, DatasetSchema, FieldSpec

from horizon_imagination.data.replay_buffer import EpochDataIterator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _make_dataset(path):
    schema = DatasetSchema(
        fields=[
            FieldSpec("image|features", (4, 4, 3), "uint8", role="observation"),
            FieldSpec("action", (), "int64", role="action"),
            FieldSpec("reward", (), "float32", role="reward"),
        ]
    )
    dataset = Dataset.create(schema, path=path, backend="zarr")
    rng = np.random.default_rng(0)
    for length in (30, 25, 20):
        dataset.add_episode(
            {
                "initial_observation": {
                    "image|features": rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
                },
                "observations": {
                    "image|features": rng.integers(
                        0, 255, size=(length, 4, 4, 3), dtype=np.uint8
                    )
                },
                "actions": {"action": rng.integers(0, 4, size=(length,)).astype(np.int64)},
                "rewards": rng.standard_normal(length).astype(np.float32),
                "terminated": True,
            }
        )
    dataset.flush()
    return dataset


def test_epoch_data_iterator_yields_expected_batches(tmp_path):
    dataset = _make_dataset(str(tmp_path / "ds"))

    config = EpochDataIterator.Config(
        replay_buffer=dataset,
        tokenizer_steps=5,
        world_model_steps=3,
        controller_steps=2,
        tokenizer_batch_size=4,
        wm_segment_length=6,
        wm_min_segment_length=2,
        wm_batch_size=2,
        c_segment_length=4,
        c_min_segment_length=1,
        c_batch_size=2,
        read_chunk_size=64,
    )
    iterator = config.make_instance()

    items = list(iterator)
    assert len(items) == 5 + 3 + 2

    phases = [phase for _, phase in items]
    assert phases == [0] * 5 + [1] * 3 + [2] * 2

    tok_batches = [batch for batch, phase in items if phase == 0]
    for batch in tok_batches:
        # prefetch_tok extracts observation['image|features'][:, 0] directly
        assert batch.shape == (4, 4, 4, 3)
        assert batch.device.type == "cuda"

    # alignment="action_out" replaces observation/next_observation with the
    # combined all_observations entry (see batch_to_tensordict docstring).
    # The container's own .device is None (entries carry their own).
    wm_batches = [batch for batch, phase in items if phase == 1]
    for batch in wm_batches:
        assert batch["all_observations"]["image|features"].shape[0] == 2
        assert batch["reward"].device.type == "cuda"

    c_batches = [batch for batch, phase in items if phase == 2]
    for batch in c_batches:
        assert batch["all_observations"]["image|features"].shape[0] == 2
        assert batch["reward"].device.type == "cuda"


def test_epoch_data_iterator_zero_steps_yields_nothing(tmp_path):
    dataset = _make_dataset(str(tmp_path / "ds"))

    config = EpochDataIterator.Config(
        replay_buffer=dataset,
        tokenizer_steps=0,
        world_model_steps=0,
        controller_steps=0,
        tokenizer_batch_size=4,
        wm_segment_length=6,
        wm_min_segment_length=2,
        wm_batch_size=2,
        c_segment_length=4,
        c_min_segment_length=1,
        c_batch_size=2,
        read_chunk_size=64,
    )
    iterator = config.make_instance()
    assert list(iterator) == []


def test_epoch_data_iterator_empty_dataset_returns_early(tmp_path):
    schema = DatasetSchema(
        fields=[
            FieldSpec("image|features", (4, 4, 3), "uint8", role="observation"),
            FieldSpec("action", (), "int64", role="action"),
            FieldSpec("reward", (), "float32", role="reward"),
        ]
    )
    dataset = Dataset.create(schema, path=str(tmp_path / "ds"), backend="zarr")

    config = EpochDataIterator.Config(
        replay_buffer=dataset,
        tokenizer_steps=5,
        world_model_steps=0,
        controller_steps=0,
        tokenizer_batch_size=4,
        wm_segment_length=6,
        wm_min_segment_length=2,
        wm_batch_size=2,
        c_segment_length=4,
        c_min_segment_length=1,
        c_batch_size=2,
        read_chunk_size=64,
    )
    iterator = config.make_instance()
    assert list(iterator) == []
