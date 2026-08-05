import torch
import torch.nn as nn
from einops import rearrange
from horizon_imagination.modules.transform import BaseTransform


# Chunking by default (rather than only after an OOM) keeps the controller's imagination
# round-trip off the retry path, where an OOM mid-step wastes a forward pass and leaves the
# allocator fragmented. Measured over a 960-frame round-trip at resolution 64: chunking both
# directions takes peak memory from 15.2 GiB to 2.2 GiB and is marginally *faster*, so this
# is not a memory-for-speed trade. Batches smaller than the chunk size are unaffected.
DEFAULT_CHUNK_SIZE = 128


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _chunked_with_oom_retry(fn, x, chunk_size: int):
    """
    Applies ``fn`` over ``x``'s leading dim in chunks, halving the chunk size and retrying on
    CUDA OOM. Returns ``(result, calibrated_chunk_size)``, where the second element is the
    chunk size that finally worked if an OOM was hit along the way, else ``None`` -- callers
    memoize it so the next call starts at a size known to fit.
    """
    batch_size = x.shape[0]
    chunk_size = batch_size if chunk_size is None else min(batch_size, chunk_size)

    had_oom = False
    while True:
        try:
            if chunk_size >= batch_size:
                out = fn(x)
            else:
                out = torch.cat(
                    [fn(x[start:start + chunk_size]) for start in range(0, batch_size, chunk_size)],
                    dim=0
                )
            return out, (chunk_size if had_oom else None)
        except RuntimeError as exc:
            if not _is_cuda_oom_error(exc):
                raise
            had_oom = True
            if chunk_size == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk_size = max(chunk_size // 2, 1)


class ImageToLatentTransform(BaseTransform):
    def __init__(self, image_tokenizer: nn.Module, chunk_size: int = DEFAULT_CHUNK_SIZE):
        super().__init__()
        self.tokenizer = image_tokenizer
        # Separate limits per direction, since the OOM autotuner calibrates each one against
        # its own peak: encode and decode do not have the same memory profile.
        self._oom_encode_chunk_limit = chunk_size
        self._oom_decode_chunk_limit = chunk_size

    @staticmethod
    def _is_cuda_oom_error(exc: RuntimeError) -> bool:
        return _is_cuda_oom_error(exc)

    @torch.no_grad()
    def transform(self, x, *args, **kwargs):
        self.tokenizer.eval()
        shape = x.shape
        x = rearrange(x, '... c h w -> (...) c h w')

        z, calibrated = _chunked_with_oom_retry(
            self.tokenizer.encode, x, self._oom_encode_chunk_limit
        )
        if calibrated is not None:
            self._oom_encode_chunk_limit = calibrated

        z = z.reshape(*shape[:-3], *z.shape[-3:])
        return z

    @torch.no_grad()
    def _truncated_roundtrip_chunk(self, z):
        """z -> deepest shared feature map -> z, skipping the highest-resolution levels.

        The encoder and decoder are level-symmetric, but their channel widths only coincide
        at *some* depths, so this is one specific cut rather than a free choice of depth. For
        channels_mult=(2,4,4) the decoder emits 256ch at 32x32 while the encoder carries only
        128ch there; the deepest matching point is the decoder's first upsample output, which
        equals the encoder's last downsample input. The asserts below re-check that, so a
        change to channels_mult fails loudly instead of silently comparing mismatched tensors.

        Cheaper *and* cleaner than the full round-trip: it never reaches pixel space, so it
        also avoids the uint8-quantization and 4:2:0 chroma noise floor that
        _postprocess_images/_preprocess_images bake into the full cycle. It measures a weaker
        property though -- self-consistency under a partial cycle, not manifold membership.
        """
        from horizon_imagination.models.tokenizer.cosmos.networks import ContinuousImageTokenizer
        from horizon_imagination.models.tokenizer.cosmos.modules.utils import nonlinearity

        network = self.tokenizer.network
        assert isinstance(network, ContinuousImageTokenizer), \
            f"The truncated round-trip assumes the AE latent boundary; got {type(network)}"
        dec, enc = network.decoder, network.encoder
        assert dec.num_resolutions == enc.num_resolutions

        i = dec.num_resolutions - 1

        # Decode: z -> (block_in, 2*z_res, 2*z_res)
        h = network.post_quant_conv(z)
        h = dec.conv_in(h)
        h = dec.mid.block_1(h)
        h = dec.mid.attn_1(h)
        h = dec.mid.block_2(h)
        for i_block in range(dec.num_res_blocks + 1):
            h = dec.up[i].block[i_block](h)
            if len(dec.up[i].attn) > 0:
                h = dec.up[i].attn[i_block](h)
        h = dec.up[i].upsample(h)

        # Encode back from that same feature map.
        expected_ch = enc.down[i].block[0].in_channels
        assert h.shape[1] == expected_ch, (
            f"encoder/decoder widths do not meet at this depth ({h.shape[1]} vs {expected_ch}); "
            f"recompute the cut for channels_mult={enc.in_ch_mult[1:]}"
        )
        for i_block in range(enc.num_res_blocks):
            h = enc.down[i].block[i_block](h)
            if len(enc.down[i].attn) > 0:
                h = enc.down[i].attn[i_block](h)
        h = enc.down[i].downsample(h)
        h = enc.mid.block_1(h)
        h = enc.mid.attn_1(h)
        h = enc.mid.block_2(h)
        h = enc.norm_out(h)
        h = nonlinearity(h)
        h = enc.conv_out(h)
        # Matches CosmosImageTokenizer.encode's latent boundary (AE formulation).
        return torch.tanh(network.quant_conv(h))

    @torch.no_grad()
    def roundtrip(self, z, truncated: bool = False, *args, **kwargs):
        if not truncated:
            return super().roundtrip(z, *args, **kwargs)

        self.tokenizer.eval()
        shape = z.shape
        z = rearrange(z, '... c h w -> (...) c h w')
        precision = getattr(self.tokenizer, 'precision', None)
        if precision is not None:
            z = z.to(dtype=precision)

        z_hat, calibrated = _chunked_with_oom_retry(
            self._truncated_roundtrip_chunk, z, self._oom_decode_chunk_limit
        )
        if calibrated is not None:
            self._oom_decode_chunk_limit = calibrated
        return z_hat.reshape(*shape[:-3], *z_hat.shape[-3:])

    @torch.no_grad()
    def inverse(self, z, *args, **kwargs):
        self.tokenizer.eval()
        shape = z.shape
        z = rearrange(z, '... c h w -> (...) c h w')
        # `_preprocess_images` casts to the tokenizer's precision on the forward path; the
        # decode path has no equivalent, so float32 latents would hit e.g. bf16 conv weights.
        precision = getattr(self.tokenizer, 'precision', None)
        if precision is not None:
            z = z.to(dtype=precision)

        x_hat, calibrated = _chunked_with_oom_retry(
            self.tokenizer.decode, z, self._oom_decode_chunk_limit
        )
        if calibrated is not None:
            self._oom_decode_chunk_limit = calibrated

        x_hat = x_hat.reshape(*shape[:-3], *x_hat.shape[-3:])
        return x_hat
