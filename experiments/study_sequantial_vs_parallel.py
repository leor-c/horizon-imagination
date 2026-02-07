from typing import Literal
from pathlib import Path
from dataclasses import dataclass
from einops import rearrange
import torch
from torch import Tensor
import json
import numpy as np
from loguru import logger
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re

plt.rcParams.update({
    "font.size": 18,          # base font size
    "axes.labelsize": 22,     # x and y labels
    "axes.titlesize": 22,     # title
    "xtick.labelsize": 18,    # x-axis tick labels
    "ytick.labelsize": 18,    # y-axis tick labels
    "legend.fontsize": 18,    # legend
    "figure.titlesize": 20    # overall figure title
})

from horizon_imagination.models.world_model import RectifiedFlowWorldModel
from horizon_imagination.models.world_model.action_producer import FixedActionProducer
from horizon_imagination.diffusion.samplers import HorizonSamplerScheduler
from horizon_imagination.utilities import ObsKey, Modality, shift_fwd
from horizon_imagination.utilities.fvd import calculate_fvd


def get_replay_buffer_iter(data_dir, batch_size, horizon, min_segment_length, device):
    from horizon_imagination.data import (
        get_replay_buffer_storage, get_segment_replay_buffer,
        ReplayBufferTrajectoryIterator,
        SegmentSampler
    )
    from torchrl.data import LazyMemmapStorage
    from pathlib import Path
    
    rb_storage = LazyMemmapStorage(1_000_000, existsok=True)
    rb_storage.loads(data_dir)
    
    # rb_storage = get_replay_buffer_storage(max_size=200_000, store_on_disk=False, device=device)
    rb = get_segment_replay_buffer(
        rb_storage, 
        SegmentSampler(
            segment_len=horizon,
            traj_key='episode',
            min_length=min_segment_length,
            pad_direction='suffix',
        ), 
        prefetch=2,
        batch_size=batch_size*horizon
    )
    
    rb_iter = ReplayBufferTrajectoryIterator(
        rb, 
        segments_per_batch=batch_size,
        steps_per_segment=horizon,
    )
    return rb_iter


IMG_KEY = ObsKey.from_parts(Modality.image, "features")


@dataclass
class ExperimentConfig:
    data_path: Path
    outputs_dir: Path
    model_path: Path
    benchmark: str
    env_name: str
    budgets: tuple[int, ...]
    slope_widths: tuple[int, ...]
    segment_length: int
    context_length: int
    sample_size: int
    batch_size: int
    include_legend_in_plots: bool = True
    include_ylabel_in_plots: bool = True

    @property
    def horizon(self) -> int:
        return self.segment_length - self.context_length


def load_data_iter(config: ExperimentConfig):
    return get_replay_buffer_iter(
        data_dir=config.data_path,
        batch_size=config.batch_size,
        horizon=config.horizon,
        min_segment_length=config.segment_length,
        device='cpu'
    )


def load_world_model(config: ExperimentConfig) -> RectifiedFlowWorldModel:
    return _load_world_model_from_agent(config)


def _load_world_model_from_agent(config: ExperimentConfig) -> RectifiedFlowWorldModel:
    from config.agent import get_agent_online_config
    from horizon_imagination.envs import make_env
    from horizon_imagination.agent import Agent

    env = make_env(benchmark=config.benchmark, env_name=config.env_name, portal_env_backend='mm')[0]
    # env = make_craftium_env()
    cfg = get_agent_online_config(env=env, env_name=config.env_name, baseline='hi')
    agent = Agent.load_from_checkpoint(config.model_path, config=cfg)
    return agent.world_model


def to_img(tokenizer, img_tensor):
    shape = img_tensor.shape
    res = tokenizer.decode(img_tensor.flatten(0, 1))
    res = rearrange(res, "(b t) c h w -> b t c h w", b=shape[0], t=shape[1])
    # res = res.float() / 255  # FVD
    return res


def generate_single(
    world_model: RectifiedFlowWorldModel,
    budget: int,
    slope_width: int,
    batch,
    context_length: int,
    action_producer: Literal['fixed'] = 'fixed',
):
    if batch is None:
        return

    context_noise_level = 0

    segment_length = batch.shape[1]
    horizon = segment_length - context_length

    batch_size = batch.shape[0]
    tokenizer = world_model.obs_transform.fixed_transforms[Modality.image].tokenizer
    tokenizer.eval()
    tokenizer_channels = tokenizer.config.latent_channels

    actions = batch["action"][:, context_length:]
    if action_producer == 'fixed':
        action_producer = FixedActionProducer(actions=actions)
    else:
        raise ValueError(f"Incompatible action producer type '{action_producer}'")

    context = batch[:, :context_length]

    # set delay:
    assert isinstance(
        world_model.config.sampler_scheduler, HorizonSamplerScheduler.Config
    )
    world_model.config.sampler_scheduler.decay_horizon = slope_width

    segment = world_model.imagine(
        policy=action_producer,
        batch_size=batch_size,
        horizon=horizon,
        obs_shape={IMG_KEY: (1, 1, tokenizer_channels, 8, 8)},
        denoising_steps=budget,
        context=context,
        context_noise_level=context_noise_level,
    )
    obs_hat = segment["observation"]
    obs_hat = to_img(tokenizer, obs_hat[IMG_KEY])

    return obs_hat


def accumulate_and_store(tensor: Tensor, file_path: Path) -> None:
    """
    Appends a PyTorch tensor to an existing tensor file, or creates a new file if it doesn't exist.

    Args:
        tensor (torch.Tensor): The tensor to accumulate.
        file_path (Path): Path to the file where the tensor should be stored.
    """
    if file_path.exists():
        try:
            existing_tensor = torch.load(file_path, weights_only=True)
            if not isinstance(existing_tensor, torch.Tensor):
                raise ValueError("File does not contain a tensor.")
            combined_tensor = torch.cat([existing_tensor, tensor], dim=0)
        except Exception as e:
            raise RuntimeError(f"Failed to load existing tensor: {e}")
    else:
        combined_tensor = tensor

    torch.save(combined_tensor, file_path)


@torch.no_grad()
def generate_outputs(config: ExperimentConfig):
    if config.outputs_dir.exists():
        logger.info('Outputs directory exists, skipping outputs generation...')
        return

    # load trained model:
    world_model = load_world_model(config)
    tokenizer = world_model.obs_transform.fixed_transforms[Modality.image].tokenizer
    tokenizer.eval()

    # iterate over batches:
    data_iter = load_data_iter(config)

    if not config.outputs_dir.exists():
        config.outputs_dir.mkdir(parents=True, exist_ok=True)

    # for each batch, iterate over budgets and baselines (delay values):
    batch_size = config.batch_size
    num_iter = config.sample_size // batch_size
    for i in tqdm(range(num_iter)):
        batch = next(data_iter)

        ground_truth = batch["observation"][IMG_KEY].cpu()
        # Reconstructions (tokenizer enc-dec outputs):
        rec = tokenizer.forward(batch["observation"][IMG_KEY].flatten(0, 1))
        rec = rearrange(rec, "(b t) c h w -> b t c h w", b=batch_size).cpu()

        # save GT & reconst.:
        accumulate_and_store(ground_truth, config.outputs_dir / 'gt.pt')
        accumulate_and_store(rec, config.outputs_dir / 'gt_rec.pt')

        for budget in tqdm(config.budgets, leave=False):
            for slope_width in tqdm(config.slope_widths, leave=False):
                # if (budget, delay) combination exceeds `budget`, skip:
                # if budget - slope_width * (config.horizon - 1) + 1 < 1:
                    # logger.info(f"Config (budget={budget}, delay={delay}) exceeds budget. skipping...")
                    # continue

                # generate trajectory segment:
                obs_hat = generate_single(
                    world_model=world_model,
                    action_producer='fixed',
                    budget=budget,
                    slope_width=slope_width,
                    batch=batch,
                    context_length=config.context_length
                )
                obs_hat = torch.cat([rec[:, :config.context_length], obs_hat.cpu()], axis=1)

                # store in outputs dir for later evaluation:
                accumulate_and_store(obs_hat, config.outputs_dir / f'budget-{budget}_slope-{slope_width}.pt')


def evaluate_outputs(config: ExperimentConfig):
    """
    Evaluate the generation quality of all outputs produced
    by `generate_outputs`.
    Produce a single .json output file for each metric used. 
    """
    if (config.outputs_dir / 'mse.json').exists():
        logger.info('results files alreadt exist, skipping outputs evaluation...')
        return

    def _preprocess(x):
        return x.float() / 255  # map images to [0, 1]
    
    def extract_budget_delay(f_stem: str):
        budget, slope_width = tuple([int(s.split(sep='-')[1]) for s in f_stem.split(sep='_')])
            
        return budget, slope_width
    
    gt_rec = _preprocess(torch.load(config.outputs_dir / 'gt_rec.pt', weights_only=True))

    mse_scores = {}
    fvd_scores = {}

    for file in tqdm(config.outputs_dir.glob('*.pt')):
        if 'budget' in file.name:
            budget, slope_width = extract_budget_delay(file.stem)
            gen = _preprocess(torch.load(file, weights_only=True))
            assert gen.ndim == 5, f"got {gen.ndim} ({gen.shape})"
            assert gt_rec.shape == gen.shape, f"Got shapes {gt_rec.shape}, {gen.shape} ({file.name})"

            step_mse = (gt_rec - gen) ** 2
            step_mse = step_mse.mean(dim=(0, 2, 3, 4)).tolist()
            mse_scores[f'budget-{budget},slope-{slope_width}'] = step_mse

            step_fvd = calculate_fvd(gen, gt_rec, device='cuda', only_final=True)['value']
            fvd_scores[f'budget-{budget},slope-{slope_width}'] = step_fvd
    
    with open(config.outputs_dir / 'mse.json', 'w') as f:
        json.dump(mse_scores, f, indent=2)

    with open(config.outputs_dir / 'fvd.json', 'w') as f:
        json.dump(fvd_scores, f, indent=2)


def plot_results(config: ExperimentConfig):
    for metric_name in ['mse', 'fvd']:
        # _plot_metric_results(config, metric_name)
        _plot_metric_vs_budget(config, metric_name)


def _plot_metric_results(config, metric_name: Literal['mse', 'fvd']):
    def parse_key(key):
        """Extract budget and delay from key."""
        match = re.match(r"budget-(\d+(?:\.\d+)?),slope-(\d+(?:\.\d+)?)", key)
        if match:
            budget = float(match.group(1))
            slope_width = float(match.group(2))
            return budget, slope_width
        else:
            raise ValueError(f"Invalid key format: {key}")
        
    # Load the JSON data
    with open(config.outputs_dir / f'{metric_name}.json', 'r') as f:
        data = json.load(f)

    # Organize data by budget
    budget_map = {}
    for key, values in data.items():
        budget, slope_width = parse_key(key)
        # last_value = values[-1]
        aggregate_value = np.mean(values)
        if budget not in budget_map:
            budget_map[budget] = []
        budget_map[budget].append((slope_width, aggregate_value))

    # Sort budgets for color mapping
    sorted_budgets = sorted(budget_map.keys())
    norm = mcolors.Normalize(vmin=min(sorted_budgets), vmax=max(sorted_budgets))
    cmap = plt.get_cmap('plasma')  # from dark blue to bright red

    # Plot each budget's curve
    fig, ax = plt.subplots(figsize=(6, 6))
    markers = ['.', '+', 'x', '1', '2', '3', '4', 'D', 'd', '|', '_']
    m1 = 7
    m2 = 'o'
    # marker_opts = [
    #     {'marker': '.'},
    #     {'marker': m1, 'fillstyle': 'bottom'},
    #     {'marker': m2, 'fillstyle': 'bottom'},
    #     {'marker': m1, 'fillstyle': 'top'},
    #     {'marker': m2, 'fillstyle': 'top'},
    #     {'marker': m1, 'fillstyle': 'left'},
    #     # {'marker': m2, 'fillstyle': 'left'},
    #     {'marker': m1, 'fillstyle': 'right'},
    #     {'marker': m2, 'fillstyle': 'right'},
    # ]
    marker_opts = [
        {'marker': '.'},
        {'marker': m1, },
        {'marker': m2, 'fillstyle': 'none'},
        {'marker': m1, },
        {'marker': m2, 'fillstyle': 'none'},
        {'marker': m1, },
        # {'marker': m2, 'fillstyle': 'left'},
        {'marker': m1, },
        {'marker': m2, 'fillstyle': 'none'},
    ]
    for i, budget in enumerate(sorted_budgets):
        slope_width_agg_values = sorted(budget_map[budget], key=lambda x: x[0])
        slope_widths = [x[0] for x in slope_width_agg_values]
        agg_values = [x[1] for x in slope_width_agg_values]
        color = cmap(norm(budget))
        ax.plot(
            slope_widths, agg_values, label=f'Budget {budget}', 
            color=color, 
            # marker=markers[i], 
            markersize=10,
            # markerfacecolor='tab:blue',
            markerfacecolor=color,
            # markerfacecoloralt='lightsteelblue',
            # markerfacecoloralt='white',
            # markeredgecolor='black',
            # alpha=0.7,
            **(marker_opts[i])
        )

    ax.set_xlabel("Effective Horizon (Slope Width)")
    ax.set_ylabel(f"{metric_name.upper()}")
    # ax.set_xscale('log')
    if metric_name.lower() in ['mse']:
        ax.set_yscale('log')
    ax.set_title("Performance by Budget and Effective Horizon")
    ax.grid(True)
    ax.legend(title="Budget", loc="best")

    fig.tight_layout()
    plt.savefig(config.outputs_dir / f'results-{metric_name}.png')


def _plot_metric_vs_budget(config: ExperimentConfig, metric_name: Literal['mse', 'fvd']):
    def parse_key(key):
        """Extract budget and delay from key."""
        match = re.match(r"budget-(\d+(?:\.\d+)?),slope-(\d+(?:\.\d+)?)", key)
        if match:
            budget = float(match.group(1))
            slope_width = float(match.group(2))
            return budget, slope_width
        else:
            raise ValueError(f"Invalid key format: {key}")
        
    # Load the JSON data
    with open(config.outputs_dir / f'{metric_name}.json', 'r') as f:
        data = json.load(f)

    # Organize data by budget
    degree_of_sequentiality_map = {}
    for key, values in data.items():
        budget, slope_width = parse_key(key)
        if int(budget) in [48, 96]:
            continue
        d_o_s = slope_width 

        # last_value = values[-1]
        last_value = np.mean(values)
        if d_o_s not in degree_of_sequentiality_map:
            degree_of_sequentiality_map[d_o_s] = []
        degree_of_sequentiality_map[d_o_s].append((budget, last_value))

    # Sort budgets for color mapping
    sorted_dos = sorted(degree_of_sequentiality_map.keys())
    norm = mcolors.LogNorm(vmin=min(sorted_dos), vmax=max(sorted_dos))
    cmap = plt.get_cmap('plasma')  # from dark blue to bright red

    # Plot each budget's curve
    fig, ax = plt.subplots(figsize=(6, 6))
    m1 = 's'
    m2 = 'D'
    # markers = [m1, 's', 's', 's', "s", "s", '4', 'D', '+', '|', '_']
    markers = [m1] + ([m2] * 10)
    
    marker_opts = [
        {'marker': '.'},
        {'marker': m1, },
        {'marker': m2, 'fillstyle': 'none'},
        {'marker': m1, },
        {'marker': m2, 'fillstyle': 'none'},
        {'marker': m1, },
        # {'marker': m2, 'fillstyle': 'left'},
        {'marker': m1, },
        {'marker': m2, 'fillstyle': 'none'},
    ]
    # Vertical dashed line at x=1.5
    ax.axvline(
        x=config.horizon,
        color="dimgrey",
        linestyle="--",   # dashed
        linewidth=2,
        zorder=2          # ensure it stays behind other artists
    )
    for i, dos in enumerate(sorted_dos):
        budget_last_values = sorted(degree_of_sequentiality_map[dos], key=lambda x: x[0])
        budgets = [x[0] for x in budget_last_values]
        last_values = [x[1] for x in budget_last_values]
        color = cmap(norm(dos))
        ax.plot(
            budgets, last_values, label=f'{int(dos)}' + (' (AR)' if dos == 1 else ''), 
            color=color, 
            marker=markers[i], 
            markersize=6,
            # markerfacecolor='tab:blue',
            markerfacecolor=color,
            # markerfacecoloralt='lightsteelblue',
            # markerfacecoloralt='white',
            # markeredgecolor='black',
            # alpha=0.7,
            # **(marker_opts[i])
        )

    ax.set_xlabel(r"Denoising Budget ($B$)")
    if metric_name.lower() == 'mse':
        ylabel = 'Mean Squared Error'
    elif metric_name.lower() == 'fvd':
        ylabel = 'FVD'
    if config.include_ylabel_in_plots:
        ax.set_ylabel(f"{ylabel}")
    ax.set_xscale('log', base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(v)) for v in budgets])
    # if metric_name.lower() in ['mse']:
    #     ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    # ax.ticklabel_format(style='plain', axis='y')
    ax.set_title(f"{config.env_name}")
    # ax.grid(True)
    # ax.grid(color='lightgrey', linestyle='-', linewidth=0.7)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # Ensures grid is drawn below the plot element
    # for spine in ["top", "right"]:
    #     ax.spines[spine].set_visible(False)

    if config.include_legend_in_plots:
        ax.legend(title="Denoising schedule\ndecay horizon " + r"$\nu$", loc="best")

    fig.tight_layout()
    plt.savefig(config.outputs_dir / f'{config.env_name.replace('/', '-').lower()}_{metric_name}-vs-budget.png')
    plt.savefig(config.outputs_dir / f'{config.env_name.replace('/', '-').lower()}_{metric_name}-vs-budget.pdf')


def plot_legend(config: ExperimentConfig):
    from matplotlib import rcParams

    rcParams.update({
        "text.usetex": True,
        # "font.family": "serif",
        "text.latex.preamble": r"\usepackage{bm}"
    })

    # Legend values
    values = config.slope_widths

    # Normalization + colormap
    norm = mcolors.LogNorm(vmin=min(values), vmax=max(values))
    cmap = plt.get_cmap("plasma")

    # Create dummy handles (no axes needed)
    fig = plt.figure()
    m1 = 's'
    m2 = 'D'
    handles = [plt.Line2D([], [], color=cmap(norm(d)), marker=(m1 if d==1 else m2), markersize=6, lw=2) for d in values]
    labels  = [str(d) if d>1 else f'{d} (AR)' for d in values]
    # Legend directly on the figure
    fig.legend(handles, labels, title="Decay horizon " + r"$\nu$", loc="center", frameon=False, ncol=len(values))

    fig.savefig(config.outputs_dir / "legend.pdf", bbox_inches="tight")


def run_experiment(config: ExperimentConfig):
    generate_outputs(config=config)
    evaluate_outputs(config=config)
    plot_results(config)



def get_config():
    C = 1
    H = 32
    config = ExperimentConfig(
        data_path=Path('your_data_path'),
        outputs_dir=Path('your_output_path'),
        model_path=Path('your_model_checkpoint_path'),
        benchmark='ale',
        env_name='ALE/CrazyClimber-v5',
        budgets=(H // 4, H // 2, H, 2*H, 4*H, 8*H),
        slope_widths=(1, 2, 4, 8, 16, 32),
        segment_length=H + C,
        context_length=C,
        sample_size=512,
        batch_size=32,
        include_legend_in_plots=False,
        include_ylabel_in_plots=False,
    )
    return config


def main():
    config = get_config()
    run_experiment(config)
    plot_legend(config)


if __name__ == "__main__":
    main()
