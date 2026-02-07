import csv
import itertools
from enum import Enum
from pathlib import Path
import json
import wandb
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

plt.rcParams.update({
    "font.size": 18,          # base font size
    "axes.labelsize": 22,     # x and y labels
    "axes.titlesize": 22,     # title
    "xtick.labelsize": 18,    # x-axis tick labels
    "ytick.labelsize": 18,    # y-axis tick labels
    "legend.fontsize": 18,    # legend
    "figure.titlesize": 20    # overall figure title
})


class Baseline(Enum):
    autoregressive = 'ar'
    hi_horizon_4_budget_16 = 'hi_horizon-4_budget-16'
    hi_horizon_4_budget_32 = 'hi_horizon-4_budget-32'


WANDB_PROJECT = "HorizonImagination"




games = [
    'ALE/Boxing-v5',
    'ALE/CrazyClimber-v5',
    'ALE/Gopher-v5',
    'ALE/KungFuMaster-v5',
    'Craftium/ChopTree-v0',
    'Craftium/Room-v0',
    'Craftium/Speleo-v0',
    'Craftium/SmallRoom-v0',
]

baselines = [
    Baseline.autoregressive.value,
    Baseline.hi_horizon_4_budget_16.value,
    Baseline.hi_horizon_4_budget_32.value,
]

stable_ablation_baselines = [
    Baseline.hi_horizon_4_budget_16.value,
    Baseline.naive_horizon_4_budget_16.value,
]

seeds = [0, 1, 2, 3, 4]

run_id = ['']


def generate_results_ids_csv(baselines_list, filename: str = 'results_ids.csv'):
    dst = Path('results') / filename

    if dst.exists():
        print('File already exists! exiting to avoid overriding...')
        return

    # Put all lists in a list of lists
    lists = [games, baselines_list, seeds, run_id]
    headers = ['game', 'baseline', 'seed', 'run_id']

    # filename = "results/results_ids.csv"

    combinations = itertools.product(*lists)


    with open(dst, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        for combo in combinations:
            writer.writerow(combo)

    print(f"CSV file '{dst}' created.")







def nested_dict():
    return defaultdict(nested_dict)


def fetch_run_data(run_id):
    """Fetch avg_episode_return history for a run."""
    api = wandb.Api()
    run = api.run(f"{WANDB_PROJECT}/{run_id}")
    
    avg_return_key = "experience_stats/avg_episode_return"
    history = run.scan_history(keys=["epoch", avg_return_key])

    x, y = [], []
    for row in history:
        if avg_return_key in row:
            x.append(row["epoch"])
            y.append(row[avg_return_key])
    return {"epoch": x, "avg_episode_return": y}


def fetch_results_from_wandb(csv_file: Path, output_dir: Path):
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Process rows grouped by game
    games = {}
    for row in rows:
        game = row["game"]
        games.setdefault(game, []).append(row)

    for game, game_rows in games.items():
        safe_name = game.replace("/", "_")
        out_path = output_dir / f"{safe_name}.json"

        if out_path.exists():
            with open(out_path) as f:
                game_data = json.load(f)
        else:
            game_data = {}

        for row in game_rows:
            run_id = row["run_id"]
            baseline = row["baseline"]
            seed = row["seed"]

            if baseline not in game_data:
                game_data[baseline] = {}
            if seed not in game_data[baseline]:
                print(f"Fetching data for run {run_id} (game={game}, baseline={baseline}, seed={seed})")
                if len(run_id) > 0:
                    game_data[baseline][seed] = fetch_run_data(run_id)
            else:
                print(f"Skipping existing entry: game={game}, baseline={baseline}, seed={seed}")

        with open(out_path, "w") as f:
            json.dump(game_data, f, indent=2)
        print(f"Saved {out_path}")



def interpolate_to_epochs(x, y, epoch_max):
    epochs = np.arange(1, epoch_max + 1)
    # clip extrapolation by holding end values
    y_interp = np.interp(epochs, x, y, left=y[0], right=y[-1])
    return epochs, y_interp


my_palette = [
    'black',  # AR
    "#D55E00",  # vermilion
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
]
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=my_palette)


def _get_baseline_label(baseline: str):
    if baseline == 'ar':
        return r'$(\nu=1, B=32)$ (AR)'
    else:
        assert baseline.startswith('hi') or baseline.startswith('naive')
        _, horizon, budget = baseline.split('_')
        horizon = int(horizon.split('-')[1])
        budget = int(budget.split('-')[1])
        return rf'$(\nu={horizon}, B={budget})$'
    

def apply_ema(x, alpha: float = 0.1):
    # smoothing factor (0 < alpha <= 1); smaller = smoother
    ema = np.zeros_like(x)

    ema[0] = x[0]
    for i in range(1, len(x)):
        ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]
    
    return ema



def apply_sma(x, window_size: int = 5):
    """
    Apply simple moving average (SMA) smoothing.

    Args:
        x (array-like): Input 1D array of values.
        window_size (int): Size of the moving average window.

    Returns:
        np.ndarray: Smoothed values with the same length as x.
    """
    x = np.asarray(x, dtype=float)
    sma = np.zeros_like(x)

    cumsum = np.cumsum(np.insert(x, 0, 0))
    for i in range(len(x)):
        start = max(0, i - window_size + 1)
        sma[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)

    return sma



def apply_csma(x, window_size: int = 5):
    """
    Apply centered simple moving average (SMA) smoothing.

    Args:
        x (array-like): Input 1D array of values.
        window_size (int): Size of the moving average window (odd preferred).

    Returns:
        np.ndarray: Smoothed values with the same length as x.
    """
    x = np.asarray(x, dtype=float)
    sma = np.zeros_like(x)

    half = window_size // 2
    for i in range(len(x)):
        start = max(0, i - half)
        end = min(len(x), i + half + 1)
        sma[i] = np.mean(x[start:end])

    return sma


def process_game(json_path, output_dir: Path, baselines_to_plot: list[str]):
    game = json_path.stem
    with open(json_path) as f:
        data = json.load(f)

    plt.figure(figsize=(6, 6))
    baseline_colors = {
        'ar': "#7E7E7E",
        'hi_horizon-4_budget-16': '#E69F00',
        'hi_horizon-4_budget-32': '#56B4E9',
        'naive_horizon-4_budget-16': '#7E7E7E',
    }
    linestyles = {
        'ar': "dashdot",
        'hi_horizon-4_budget-16': 'solid',
        'hi_horizon-4_budget-32': 'solid',
        'naive_horizon-4_budget-16': 'solid',
    }

    human_scores = {
        'ale_boxing-v5': 12.1,
        'ale_crazyclimber-v5': 35829.4,
        'ale_gopher-v5': 2412.5,
        'ale_kungfumaster-v5': 22736.3,
    }

    for baseline, seeds in data.items():
        if baseline not in baselines_to_plot:
            continue

        seed_curves = []
        max_epoch = 0

        # find largest epoch across seeds
        for seed, scores in seeds.items():
            if scores.get("epoch"):
                max_epoch = max(max_epoch, max(scores["epoch"]))

        if max_epoch == 0:
            print(f"Skipping {baseline} in {game}, no data.")
            continue

        # interpolate all seeds
        for seed, scores in seeds.items():
            epochs = scores.get("epoch", [])
            returns = scores.get("avg_episode_return", [])
            if not epochs or not returns:
                continue
            epochs = np.array(epochs)
            returns = np.array(returns)
            _, y_interp = interpolate_to_epochs(epochs, returns, max_epoch)
            seed_curves.append(y_interp)

        if not seed_curves:
            continue
        
        freq = 5
        smoothing_fn = apply_csma
        smoothing_kwargs = {'window_size': 15}
        # smoothing_kwargs = {'alpha': 0.1}
        seed_curves = np.stack(seed_curves, axis=0)  # [num_seeds, max_epoch]
        mean = smoothing_fn(seed_curves.mean(axis=0), **smoothing_kwargs)[::freq]
        std = smoothing_fn(seed_curves.std(axis=0), **smoothing_kwargs)[::freq]

        epochs = np.arange(1, max_epoch + 1)[::freq]
        steps_per_epoch = 200
        env_steps = epochs * steps_per_epoch
        
        plt.plot(
            env_steps, 
            mean, 
            label=_get_baseline_label(baseline), 
            color=baseline_colors[baseline], 
            lw=4 if '16' in baseline else 2,
            linestyle=linestyles[baseline],
        )
        plt.fill_between(env_steps, mean - std, mean + std, alpha=0.15, color=baseline_colors[baseline])

    # plt.grid()
    ax = plt.gca()
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)  # Ensures grid is drawn below the plot element

    if game.lower() in human_scores:
        ax.axhline(y=human_scores[game.lower()], color="black", linestyle="--", zorder=0)

    plt.xlabel("Environment Steps")
    plt.ylabel("Episode return")
    plt.title(game.replace('_', '/'))
    # plt.legend(bbox_to_anchor=(1,1))
    
    # formatter = EngFormatter(unit="")
    class TightEngFormatter(EngFormatter):
        def __call__(self, x, pos=None):
            # Call the parent to get the formatted string
            s = super().__call__(x, pos)
            # Remove the space before the prefix (e.g. "100 k" → "100k")
            return s.replace(" ", "")

    formatter = TightEngFormatter(unit="")
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    plt.tight_layout()

    out_path = output_dir / f"{game}.pdf"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")



def process_and_plot_results(inputs_dir: Path, outputs_dir: Path, baselines_to_plot: list[str]):
    for json_path in inputs_dir.glob("*.json"):
        process_game(json_path, output_dir=outputs_dir, baselines_to_plot=baselines_to_plot)


def plot_legend(outputs_dir: Path, baselines):
    from matplotlib import rc
    rc('text', usetex=True)
    from matplotlib import rcParams

    rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{bm}"
    })
    labels_dict = {
        'ar': rf'$(\nu=1, B=32)$ (AR)', 
        'hi_horizon-4_budget-16': r'$(\nu=4, \bm{B=16})$', 
        'hi_horizon-4_budget-32': rf'$(\nu=4, B=32)$',
        'naive_horizon-4_budget-16': r'Naive $(\nu=4, B=16)$',
    }
    if 'naive' in [b.split('_')[0] for b in baselines]:
        labels_dict['hi_horizon-4_budget-16'] = r'Stable $(\nu=4, B=16)$'

    baseline_colors = {
        'ar': '#7E7E7E', 
        'hi_horizon-4_budget-16': '#E69F00',
        'hi_horizon-4_budget-32': '#56B4E9',
        'naive_horizon-4_budget-16': '#7E7E7E',
    }
    linestyles = {
        'ar': "dashdot",
        'hi_horizon-4_budget-16': 'solid',
        'hi_horizon-4_budget-32': 'solid',
        'naive_horizon-4_budget-16': 'solid',
    }

    fig = plt.figure()
    handles = [
        plt.Line2D([], [], color=baseline_colors[b], lw=4 if '16' in b else 2, linestyle=linestyles[b]) 
        for b in baselines
    ]
    labels  = [labels_dict[b] for b in baselines]
    fig.legend(handles, labels, loc="center", frameon=False, ncol=len(baselines))

    fig.savefig(outputs_dir / "legend.pdf", bbox_inches="tight")



def main():
    # Input CSV and output folder
    CSV_FILE = "results_ids.csv"
    OUTPUT_DIR = Path("results") / 'scores'
    OUTPUT_DIR.mkdir(exist_ok=True)
    ABLATION_CSV_FILE = "stable_ablation_results_ids.csv"
    ABLATION_OUTPUT_DIR = Path("results") / 'ablation_scores'
    ABLATION_OUTPUT_DIR.mkdir(exist_ok=True)

    plot_version = 'ablation'
    if plot_version == 'ablation':
        csv_file = ABLATION_CSV_FILE
        output_dir = ABLATION_OUTPUT_DIR
        baselines_version = stable_ablation_baselines
        baselines_to_plot = [
            'hi_horizon-4_budget-16',
            'naive_horizon-4_budget-16',
        ]
    elif plot_version == 'main':
        csv_file = CSV_FILE
        output_dir = OUTPUT_DIR
        baselines_version = baselines
        baselines_to_plot = [
            'ar',
            'hi_horizon-4_budget-16',
            'hi_horizon-4_budget-32'
        ]
    else:
        raise ValueError(f"Invalid value for plot_version: {plot_version}")

    generate_results_ids_csv(baselines_version, filename=csv_file)

    fetch_results_from_wandb(
        csv_file=Path('results') / csv_file,
        output_dir=output_dir
    )

    process_and_plot_results(
        inputs_dir=output_dir,
        outputs_dir=output_dir,
        baselines_to_plot=baselines_to_plot,
    )

    plot_legend(outputs_dir=output_dir, baselines=baselines_to_plot)


if __name__ == '__main__':
    main()
