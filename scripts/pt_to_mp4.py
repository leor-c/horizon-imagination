import torch
import torch.nn.functional as F
import imageio
import numpy as np
import os
from pathlib import Path
from einops import rearrange
import click

def save_sequences_as_mp4(pt_file:Path, indices, out_dir="videos", fps=5, resize=None):
    """
    Load sequences from a .pt file and save selected ones as mp4 videos.

    Args:
        pt_file (str): Path to .pt file containing tensor of shape (N, T, H, W, 3).
        indices (list[int]): List of sequence indices to export.
        out_dir (str): Directory to save videos.
        fps (int): Frames per second for output video.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load the tensor
    data = torch.load(pt_file)
    if isinstance(data, dict):  # in case it's wrapped in dict
        # try to grab the tensor inside
        for v in data.values():
            if isinstance(v, torch.Tensor):
                data = v
                break

    assert data.ndim == 5, f"Expected 5D tensor (N,T,H,W,3), got {data.shape}"

    data = data.cpu()

    for idx in indices:
        if idx < 0 or idx >= data.shape[0]:
            print(f"Skipping invalid index {idx}")
            continue

        sequence = data[idx]  # shape (T, 3, H, W)

        # Resize if requested
        if resize is not None:
            # interpolate expects (N,C,H,W) so batch over T
            sequence = F.interpolate(sequence.float(), size=resize, mode="bilinear", align_corners=False)

        sequence = rearrange(sequence, 't c h w -> t h w c').numpy()

        # Ensure uint8 in [0,255] for video writing
        if sequence.dtype != np.uint8:
            sequence = np.clip(sequence, 0, 1) if sequence.max() <= 1 else np.clip(sequence, 0, 255)
            sequence = (sequence * 255).astype(np.uint8) if sequence.max() <= 1 else sequence.astype(np.uint8)

        out_path = os.path.join(out_dir, f"{pt_file.stem}_seq_{idx}.mp4")
        imageio.mimwrite(out_path, sequence, fps=fps, codec="libx264", quality=8)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    pt_file = Path('path_to_results')       
    indices = [0, 1, 5, 42]             # replace with your desired indices
    save_sequences_as_mp4(pt_file, indices, out_dir="videos", fps=5, resize=(256, 256))
