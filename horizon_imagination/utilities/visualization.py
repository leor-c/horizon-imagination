from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np


def make_border(img, width: int = 3, color = (0, 0, 255)):
    img[:, :, :width, :, 0] = color[0]
    img[:, :, :width, :, 1] = color[1]
    img[:, :, :width, :, 2] = color[2]

    img[:, :, -width:, :, 0] = color[0]
    img[:, :, -width:, :, 1] = color[1]
    img[:, :, -width:, :, 2] = color[2]

    img[:, :, :, :width, 0] = color[0]
    img[:, :, :, :width, 1] = color[1]
    img[:, :, :, :width, 2] = color[2]

    img[:, :, :, -width:, 0] = color[0]
    img[:, :, :, -width:, 1] = color[1]
    img[:, :, :, -width:, 2] = color[2]

    return img


def plot_image_sequences(sequences, labels):
    """
    Plots two sequences of N images in two rows using matplotlib.
    
    Parameters:
        seq1 (list or array): Sequence of N images (1st row).
        seq2 (list or array): Sequence of N images (2nd row).
    """
    K = len(sequences)
    N = len(sequences[0])
    for p in sequences:
        assert len(p) == N, "Sequences must have the same length"
    
    
    fig, axes = plt.subplots(K, N, figsize=(N * 2, 2*K))

    def turn_off_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    for i in range(N):
        for j in range(K):
            axes[j, i].imshow(sequences[j][i])
            turn_off_axis(axes[j, i])
            # axes[j, i].axis('off')

    # Add labels to the left of each row
    for row, label in zip(list(range(K)), labels):
        axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=40, va='center')
    
    plt.tight_layout()
    plt.savefig('imagination.png')


def generate_video(sequences, labels, output_path="output.mp4", fps=5, scale=4):
    """
    sequences: List[List[np.ndarray]] - list of image sequences, each a list of frames
    labels: List[str] - list of labels for each sequence
    output_path: str - path to save the MP4 video
    fps: int - frames per second
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio
    num_sequences = len(sequences)
    batch_size, num_frames = sequences[0].shape[:2]
    
    # Sanity checks
    assert all(seq.shape[:2] == (batch_size, num_frames) for seq in sequences), \
    f"All sequences must have the same number of frames. got {seq.shape[:2]} != ({batch_size}, {num_frames}) "
    font = ImageFont.load_default(size=14 * 2)
    frames = []
    
    for t in range(num_frames):
        frames_grid = []
        for b in range(batch_size):
            frame_row = []

            for i, seq in enumerate(sequences):
                img = Image.fromarray(seq[b, t])
                w, h = img.size
                img = img.resize((w * scale, h * scale), Image.NEAREST)
                draw = ImageDraw.Draw(img)
                draw.text((5, 5), labels[i], font=font, fill=(255, 255, 255))
                frame_row.append(img)
            frames_grid.append(frame_row)

        # Concatenate horizontally
        total_width = sum(img.width for img in frame_row)
        height = frame_row[0].height * batch_size
        combined = Image.new('RGB', (total_width, height))
        
        y_offset = 0
        for frame_row in frames_grid:
            x_offset = 0
            for img in frame_row:
                combined.paste(img, (x_offset, y_offset))
                x_offset += img.width
            y_offset += img.height

        frames.append(np.array(combined))

    if output_path is None:
        return frames

    # Write to MP4 using imageio
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def to_img(tokenizer, img_tensor):
    shape = img_tensor.shape
    res = tokenizer.decode(img_tensor.flatten(0, 1))
    res = rearrange(res, '(b t) c h w -> b t h w c', b=shape[0], t=shape[1])
    res = res.cpu().numpy()
    return res
