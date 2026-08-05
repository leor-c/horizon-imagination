import torch
import torch.nn as nn
from einops import rearrange
from horizon_imagination.modules.transform import BaseTransform


class ImageToLatentTransform(BaseTransform):
    def __init__(self, image_tokenizer: nn.Module):
        super().__init__()
        self.tokenizer = image_tokenizer
        self._oom_encode_chunk_limit = None

    @staticmethod
    def _is_cuda_oom_error(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return "out of memory" in msg and "cuda" in msg

    @torch.no_grad()
    def transform(self, x, *args, **kwargs):
        self.tokenizer.eval()
        shape = x.shape
        x = rearrange(x, '... c h w -> (...) c h w')
        batch_size = x.shape[0]

        if self._oom_encode_chunk_limit is None:
            chunk_size = batch_size
        else:
            chunk_size = min(batch_size, self._oom_encode_chunk_limit)

        had_oom = False
        while True:
            try:
                if chunk_size == batch_size:
                    z = self.tokenizer.encode(x)
                else:
                    z = torch.cat(
                        [self.tokenizer.encode(x[start:start + chunk_size]) for start in range(0, batch_size, chunk_size)],
                        dim=0
                    )
                if had_oom:
                    self._oom_encode_chunk_limit = chunk_size
                break
            except RuntimeError as exc:
                if not self._is_cuda_oom_error(exc):
                    raise
                had_oom = True
                if chunk_size == 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_size = max(chunk_size // 2, 1)

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
