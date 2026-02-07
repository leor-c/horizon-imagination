import imageio
import numpy as np
import matplotlib.pyplot as plt

def extract_row_baselines(
    video_path,
    row_idx=0,                 # which example row (0-based, 0..N-1)
    baseline_indices=(0, 1),   # which baselines (columns) to extract
    N=8,                       # number of rows in the grid
    M=5,                       # number of baselines (columns)
    max_frames=None,           # number of frames to use (None = all)
    every_nth=1,               # subsample frames
    frames_range=(0, np.inf),
    output_path="output.png"
):
    reader = imageio.get_reader(video_path)
    frames = {i: [] for i in baseline_indices}

    for f_idx, frame in enumerate(reader):
        if max_frames is not None and f_idx >= max_frames:
            break
        if (f_idx % every_nth != 0) or (f_idx < frames_range[0] or f_idx > frames_range[1]):
            continue

        H, W, _ = frame.shape
        cell_h = H // N
        cell_w = W // M

        for idx in baseline_indices:
            y0 = row_idx * cell_h
            y1 = (row_idx + 1) * cell_h
            x0 = idx * cell_w
            x1 = (idx + 1) * cell_w
            crop = frame[y0:y1, x0:x1, :]
            frames[idx].append(crop)

    reader.close()

    # Concatenate frames horizontally for each baseline
    rows = []
    for idx in baseline_indices:
        row = np.concatenate(frames[idx], axis=1)  # concat along width
        rows.append(row)

    # Stack baselines vertically
    output_img = np.concatenate(rows, axis=0)

    plt.imsave(output_path, output_img)
    print(f"Saved to {output_path} with shape {output_img.shape}")

if __name__ == '__main__':
    # Example usage:
    extract_row_baselines(
        "chop-tree.mp4",
        row_idx=7,                # pick the 3rd example row
        baseline_indices=(0, 3, 4),  # pick baseline 0 and 3
        N=8,
        M=5,
        max_frames=None,
        every_nth=1,
        frames_range=(3, 11),
        output_path="baselines_row_chop-tree.png"
    )
