import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from horizon_imagination.modules.transform.base import BaseTransform
from horizon_imagination.models.tokenizer.cosmos import DiscreteImageTokenizer
from horizon_imagination.utilities.config import Configurable, dataclass, BaseConfig


class ImageLatentToCodeTransform(BaseTransform, Configurable):
    """
    Assume input shape is (batch, temportal, channels, H, W)
    Target format: (batch, temportal, z_channels, H, W)

    Uses the post_conv layer of the discrete FSQ tokenizer to translate
    the quantized latents to codes using the layer learned by the tokenizer,
    which contains richer information. We and others observed this significantly
    improves performance.
    """
    @dataclass
    class Config(BaseConfig):
        tokenizer: DiscreteImageTokenizer

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        assert isinstance(config.tokenizer, DiscreteImageTokenizer), f"Got {config.tokenizer}"

    def transform(self, x: Tensor, *args, **kwargs):
        # Assumes input shape (batch, temporal, channels, h, w)
        assert x.dim() == 5, f"Got {x.dim()} ({x.shape})"
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t ... -> (b t) ...")
        x = self.config.tokenizer.post_quant_conv(x)
        x = rearrange(x, "(b t) ... -> b t ...", b=b, t=t)
        return x

    def inverse(self, z, *args, **kwargs):
        assert z.dim() == 5, f"Got {z.dim()} ({z.shape})"
        b, t, c, h, w = z.shape
        z = rearrange(z, "b t ... -> (b t) ...")
        z = self.config.tokenizer.quant_conv(z)
        z = rearrange(z, "(b t) ... -> b t ...", b=b, t=t)
        return z
