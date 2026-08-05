"""The (B, T, K, D) token layout of the DiT.

The key property is backwards compatibility: for a purely-image observation the
token position table is the row-major meshgrid of the patch grid, which must
reproduce the RoPE embeddings of the old (B, T, H, W, D) implementation *bit for
bit*, so that existing checkpoints keep their positional encoding.
"""
import pytest
import torch
from einops import rearrange, repeat

from horizon_imagination.models.world_model.dit import (
    MiniTrainDIT, VideoRopePosition3DEmb, _make_block_causal_mask,
)


HEAD_DIM = 16
MODEL_CHANNELS = 64
NUM_HEADS = 4


def meshgrid_positions(h: int, w: int) -> torch.Tensor:
    i, j = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return torch.stack([i.reshape(-1), j.reshape(-1)], dim=-1)


def reference_rope_embeddings(rope: VideoRopePosition3DEmb, T: int, H: int, W: int, offset: int = 0):
    """The pre-refactor implementation, over a full (T, H, W) grid of tokens."""
    h_spatial_freqs = 1.0 / (10000.0 ** rope.dim_spatial_range)
    w_spatial_freqs = 1.0 / (10000.0 ** rope.dim_spatial_range)
    temporal_freqs = 1.0 / (10000.0 ** rope.dim_temporal_range)

    half_emb_h = torch.outer(rope.seq[:H], h_spatial_freqs)
    half_emb_w = torch.outer(rope.seq[:W], w_spatial_freqs)
    half_emb_t = torch.outer(rope.seq[offset:offset + T], temporal_freqs)

    em_T_H_W_D = torch.cat(
        [
            repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
            repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
            repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
        ] * 2,
        dim=-1,
    )
    return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()


@pytest.mark.parametrize("T,H,W,offset", [(4, 4, 4, 0), (2, 4, 4, 3), (3, 2, 4, 0)])
def test_rope_matches_pre_refactor_implementation(T, H, W, offset):
    rope = VideoRopePosition3DEmb(head_dim=HEAD_DIM, len_h=8, len_w=8, len_t=16)
    positions = meshgrid_positions(H, W)

    got = rope.generate_embeddings(torch.Size((1, T, H * W, MODEL_CHANNELS)), positions, offset=offset)
    expected = reference_rope_embeddings(rope, T, H, W, offset=offset)

    assert got.shape == expected.shape
    assert torch.equal(got, expected)


def test_rope_rejects_positions_beyond_the_grid():
    rope = VideoRopePosition3DEmb(head_dim=HEAD_DIM, len_h=4, len_w=4, len_t=16)
    positions = torch.tensor([[0, 0], [4, 0]])  # row 4 is out of a 4-row grid
    with pytest.raises(AssertionError, match="exceed the maximum"):
        rope.generate_embeddings(torch.Size((1, 2, 2, MODEL_CHANNELS)), positions)


@pytest.mark.parametrize("tokens_per_frame,num_frames", [(4, 3), (1, 5), (7, 2)])
def test_block_causal_mask_is_block_lower_triangular(tokens_per_frame, num_frames):
    seq_len = tokens_per_frame * num_frames
    mask = _make_block_causal_mask(tokens_per_frame, seq_len, seq_len, torch.device('cpu'))[0, 0]

    frame_of = torch.arange(seq_len) // tokens_per_frame
    expected = frame_of[:, None] >= frame_of[None, :]
    assert torch.equal(mask, expected)


def test_forward_is_shape_preserving_for_an_arbitrary_token_count():
    torch.manual_seed(0)
    model = MiniTrainDIT(
        max_img_h=16, max_img_w=8, max_frames=16,
        in_channels=6, out_channels=6, patch_spatial=2, patch_temporal=1,
        model_channels=MODEL_CHANNELS, num_blocks=2, num_heads=NUM_HEADS, ln_eps=1e-5,
    ).eval()

    b, t, k = 2, 3, 19  # not a rectangular grid: 16 image patches + 3 vector tokens
    positions = torch.cat([meshgrid_positions(4, 4), torch.tensor([[4, 0], [4, 1], [4, 2]])])
    x = torch.randn(b, t, k, MODEL_CHANNELS)

    with torch.no_grad():
        out, kv = model(x, torch.rand(b, t), positions, torch.randn(b, t, MODEL_CHANNELS))

    assert out.shape == (b, t, k, MODEL_CHANNELS)
    assert torch.isfinite(out).all()
    # The KV cache is stored per frame:
    assert kv.layers_kv_caches[0].keys.shape[:3] == (b, t, k)


def test_extra_per_block_abs_pos_emb_is_rejected():
    with pytest.raises(AssertionError, match="not supported"):
        MiniTrainDIT(
            max_img_h=8, max_img_w=8, max_frames=8,
            in_channels=6, out_channels=6, patch_spatial=2, patch_temporal=1,
            model_channels=MODEL_CHANNELS, num_blocks=1, num_heads=NUM_HEADS,
            extra_per_block_abs_pos_emb=True, ln_eps=1e-5,
        )
