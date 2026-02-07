
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from pathlib import Path
from horizon_imagination.diffusion.samplers.schedulers import HorizonSamplerScheduler, DiffusionForcingSamplerScheduler, UniformSamplerScheduler


plt.rcParams.update({
    "font.size": 16,          # base font size
    "axes.labelsize": 18,     # x and y labels
    "axes.titlesize": 18,     # title
    "xtick.labelsize": 12,    # x-axis tick labels
    "ytick.labelsize": 12,    # y-axis tick labels
    "legend.fontsize": 18,    # legend
    "figure.titlesize": 20    # overall figure title
})



def plot_schedule_matrix(M, title = None, filename = None, cell_size=0.5):
    nrows, ncols = M.shape

    # fig, ax = plt.subplots()
    figsize = (cell_size * ncols, cell_size * nrows)
    fig, ax = plt.subplots(figsize=figsize)

    # Create grid coordinates
    x = np.arange(ncols+1)
    y = np.arange(nrows+1)

    # Plot with gaps (control via linewidth)
    c = ax.pcolormesh(x-0.5, y-0.5, M, cmap="plasma", edgecolors="white", linewidth=2)

    # Set axis limits to fit the cells exactly
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))

    ax.set_aspect("equal")  # ensures square cells
    # plt.colorbar(c, ax=ax)
    fig.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    plt.xlabel('Frame Index')
    plt.ylabel('Denoising Step')
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
    plt.close()


def get_pyramidal_schedule(num_frames: int, budget: int):
    s = DiffusionForcingSamplerScheduler.Config(
        device='cpu',
        dtype=None,
        diffusion_time_scheduler=UniformSamplerScheduler.Config(device='cpu'),
    ).make_instance()

    s.reset(num_steps=budget, batch_size=(1, num_frames))
    M = s.schedule

    return M


def plot_pyramidal_schedule(num_frames: int, budget: int):
    M = get_pyramidal_schedule(num_frames=num_frames, budget=budget)
    file_format = 'pdf'
    plot_schedule_matrix(M, 'Pyramidal Schedule', filename=Path(f'results/schedule/pyramidal-{num_frames}-{budget}.{file_format}'))


def get_horizon_schedule(num_frames: int, budget: int, decay_horizon: int):
    s = HorizonSamplerScheduler.Config(
        device='cpu',
        dtype=None,
        decay_horizon=decay_horizon,
    ).make_instance()

    s.reset(num_steps=budget, batch_size=(1, num_frames))
    M = s.schedule
    return M


def plot_horizon_schedule(num_frames: int, budget: int, decay_horizon: int):
    M = get_horizon_schedule(num_frames=num_frames, budget=budget, decay_horizon=decay_horizon)

    file_format = 'pdf'
    plot_schedule_matrix(M, 'Horizon Schedule', filename=Path(f'results/schedule/horizon-{num_frames}-{budget}-{decay_horizon}.{file_format}'))


def plot_comparison():
    # Example random data (replace with your real matrices)
    M_horizon_10 = get_horizon_schedule(num_frames=8, budget=10, decay_horizon=3).T  # (rows=frames, cols=steps)
    M_horizon_20 = get_horizon_schedule(num_frames=8, budget=20, decay_horizon=3).T
    M_pyramid_10 = get_pyramidal_schedule(num_frames=8, budget=10).T
    M_pyramid_20 = get_pyramidal_schedule(num_frames=8, budget=20).T

    
    matrices = [
        [M_horizon_10, M_horizon_20],
        [M_pyramid_10, M_pyramid_20]
    ]

    row_labels = ["Horizon schedule", "Pyramidal schedule"]
    col_labels = ["budget=10", "budget=20"]

    # fig, axes = plt.subplots(
    #     2, 2, 
    #     figsize=(14.6, 8), #constrained_layout=True,
    #     gridspec_kw={'width_ratios': [1, 2], 'height_ratios': [1, 1]}
    # )

    fig = plt.figure(figsize=(12.5, 6.3))
    ax1 = fig.add_axes([-0.04, 0.5, 0.55, 0.4])  # [x, y, w, h]
    ax2 = fig.add_axes([0.135, 0.5, 1.05, 0.4])
    ax3 = fig.add_axes([-0.04, 0.05, 0.55, 0.4])
    ax4 = fig.add_axes([0.135, 0.05, 1.05, 0.4])
    cbar_ax = fig.add_axes([0.95, 0.2, 0.015, 0.6])  
    axes = [[ax1, ax2], [ax3, ax4]]

    vmin, vmax = 0.0, 1.0
    cmap = "coolwarm"
    cw_cmap = plt.get_cmap("coolwarm")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "lighter_coolwarm",
        cw_cmap(np.linspace(0.1, 0.9, 256))
    )

    for i in range(2):
        for j in range(2):
            ax = axes[i][j]
            M = matrices[i][j]

            # Use pcolormesh to enforce square cells with borders
            nrows, ncols = M.shape
            x = np.arange(ncols + 1)
            y = np.arange(nrows + 1)
            c = ax.pcolormesh(x-0.5, y-0.5, M, cmap=cmap, vmin=vmin, vmax=vmax,
                            edgecolors="white", linewidth=2)
            
            ax.set_xlim(-0.5, ncols - 0.5)
            ax.set_ylim(-0.5, nrows - 0.5)
            # ax.set_xticks(np.arange(ncols))
            # ax.set_yticks(np.arange(nrows))
            

            ax.set_aspect("equal")  # square cells
            ax.invert_yaxis()

            # Titles and labels
            if i == 0:
                
                ax.xaxis.tick_top()
                # ax.xaxis.set_label_position("top")
                ax.set_xticks(np.arange(ncols))
                ax.tick_params(axis="x", top=True, bottom=False, labelbottom=False, colors="#757575")
            else:
                ax.set_xlabel(col_labels[j])
                # ax.set_xlabel("Denoising steps")
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(row_labels[i], labelpad=50)
                ax.set_yticks(np.arange(nrows))
                ax.tick_params(axis="y", top=True, bottom=False, labelbottom=False, colors="#757575")
            else:
                ax.set_yticks([])

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)


    # Shared colorbar (smaller and aligned)
    cbar = fig.colorbar(c, cax=cbar_ax, orientation="vertical", fraction=0.025, pad=0.04)

    # Add an invisible axis covering the whole figure
    ax_invisible = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax_invisible.set_axis_off()

    # Draw arrows in figure coordinates
    ax_invisible.annotate(
        "", xy=(0.065, 0.76), xytext=(0.065, 0.965),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#757575")
    )

    ax_invisible.annotate(
        "", xy=(0.2, 0.96), xytext=(0.063, 0.96),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#757575")
    )
    ax_invisible.text(0.04, 0.85, "Frame index", va="center", rotation=90, fontsize=14, color="#757575")
    ax_invisible.text(0.16, 0.975, "Denoising steps", ha="center", fontsize=14, color="#757575")

    # schedule width annotation:
    rect = Rectangle(
        (4.5, 1.5), 1, 4,
        linewidth=2, edgecolor="black", facecolor="none", linestyle="--"  # outline only
    )
    ax3.add_patch(rect)
    rect2 = Rectangle(
        (4.5, 1.5), 1, 4,
        linewidth=2, edgecolor="black", facecolor="none", linestyle="--"  # outline only
    )
    ax1.add_patch(rect2)

    rect = Rectangle(
        (9.5, -0.5), 1, 8,
        linewidth=2, edgecolor="black", facecolor="none", linestyle="--"  # outline only
    )
    ax4.add_patch(rect)
    rect2 = Rectangle(
        (9.5, 1.5), 1, 4,
        linewidth=2, edgecolor="black", facecolor="none", linestyle="--"  # outline only
    )
    ax2.add_patch(rect2)
    
    plt.savefig("schedules_squares.pdf")
    plt.close()


def main():
    # plot_horizon_schedule(num_frames=8, budget=10, decay_horizon=3)
    # plot_pyramidal_schedule(num_frames=8, budget=10)
    # plot_horizon_schedule(num_frames=8, budget=20, decay_horizon=3)
    # plot_pyramidal_schedule(num_frames=8, budget=20)
    plot_comparison()


if __name__ == '__main__':
    main()
