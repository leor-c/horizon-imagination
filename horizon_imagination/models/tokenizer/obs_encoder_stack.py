"""Stage-1 observation encoders, presented to the agent as a single component.

The agent trains its components through an index-based curriculum
(``Agent.components`` / ``EpochDataIterator``'s yield indices). Dict observations
add a second stage-1 encoder -- the vector autoencoder next to the image tokenizer
-- but they are trained on the same data and with the same schedule, so they are
bundled here rather than taking a separate curriculum slot.
"""
import lightning as L
import torch
import torch.nn as nn
from torch import Tensor

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.utilities.types import ObsKey, image_keys, vector_keys
from horizon_imagination.models.tokenizer.cosmos import CosmosImageTokenizer
from horizon_imagination.models.tokenizer.vector import VectorAutoencoder


class ObsEncoderStack(L.LightningModule, Configurable):
    @dataclass
    class Config(BaseConfig):
        # Each is None when the observation has no key of that modality. At least one
        # of them is always present.
        image_tokenizer: CosmosImageTokenizer = None
        vector_autoencoder: VectorAutoencoder = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        assert config.image_tokenizer is not None or config.vector_autoencoder is not None, \
            "ObsEncoderStack needs at least one stage-1 encoder."
        self.image = config.image_tokenizer
        self.vector = config.vector_autoencoder

    # -- image tokenizer passthrough (single-image call sites) ----------------
    def encode(self, x: Tensor):
        return self.image.encode(x)

    def decode(self, z: Tensor):
        return self.image.decode(z)

    def forward(self, x: Tensor):
        return self.image.forward(x)

    def training_step(self, batch: dict[ObsKey, Tensor], batch_idx, log_dict_fn=None):
        """
        :param batch: raw observations, one entry per key -- RGB images (B, C, H, W)
        and vectors (B, obs_dim), as produced by ``EpochDataIterator``.
        """
        loss = 0

        img_keys = image_keys(batch.keys())
        if img_keys:
            assert self.image is not None, f"No image tokenizer for keys {img_keys}"
            # One shared image tokenizer across all image keys: the keys are
            # concatenated on the batch dim, so each contributes equally.
            images = torch.cat([batch[k] for k in img_keys], dim=0)
            loss = loss + self.image.training_step(images, batch_idx, log_dict_fn=log_dict_fn)

        vec_keys = vector_keys(batch.keys())
        if vec_keys:
            assert self.vector is not None, f"No vector autoencoder for keys {vec_keys}"
            loss = loss + self.vector.training_step(
                {k: batch[k] for k in vec_keys}, batch_idx, log_dict_fn=log_dict_fn
            )

        return loss

    def configure_optimizers(self):
        if self.image is None:
            return self.vector.configure_optimizers()

        optimizer = self.image.configure_optimizers()
        if self.vector is not None:
            vector_optimizer = self.vector.configure_optimizers()
            for group in vector_optimizer.param_groups:
                optimizer.add_param_group(group)
        return optimizer
