import torch
from torch import Tensor

from horizon_imagination.modules.transform.base import BaseTransform
from horizon_imagination.utilities.types import ObsKey


class VectorToLatentTransform(BaseTransform):
    """
    The ``ImageToLatentTransform`` analogue for vector observations: encodes a raw
    vector observation into its ``(latent_dim, P, 1)`` latent with the (frozen,
    separately trained) vector autoencoder.

    One instance per observation key, since the autoencoder holds a separate module
    per key -- ``PerModalityTransform`` dispatches by key and does not pass it on.
    """

    def __init__(self, vector_autoencoder, key: ObsKey):
        super().__init__()
        self.vector_autoencoder = vector_autoencoder
        self.key = ObsKey(key)

    @torch.no_grad()
    def transform(self, x: Tensor, *args, **kwargs) -> Tensor:
        self.vector_autoencoder.eval()
        return self.vector_autoencoder.encode(x, self.key)

    @torch.no_grad()
    def inverse(self, z: Tensor, *args, **kwargs) -> Tensor:
        self.vector_autoencoder.eval()
        return self.vector_autoencoder.decode(z, self.key)
