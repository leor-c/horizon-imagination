import torch
import torch.nn as nn
from einops import rearrange
from horizon_imagination.modules.transform import BaseTransform


class ImageToLatentTransform(BaseTransform):
    def __init__(self, image_tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = image_tokenizer

    @torch.no_grad()
    def transform(self, x, *args, **kwargs):
        self.tokenizer.eval()
        shape = x.shape
        x = rearrange(x, '... c h w -> (...) c h w')
        z = self.tokenizer.encode(x)
        z = z.reshape(*shape[:-3], *z.shape[-3:])
        return z
    
    @torch.no_grad()
    def inverse(self, z, *args, **kwargs):
        self.tokenizer.eval()
        shape = z.shape
        z = rearrange(z, '... c h w -> (...) c h w')
        x_hat = self.tokenizer.decode(z)
        x_hat = x_hat.reshape(*shape[:-3], *x_hat.shape[-3:])
        return x_hat
