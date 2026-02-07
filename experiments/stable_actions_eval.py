from typing import Literal
import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    "text.latex.preamble": r"\usepackage{bm}"
})
import torch
from einops import repeat, rearrange
from torch.distributions import Categorical, Dirichlet
from horizon_imagination.models.world_model.action_producer import (
    StableDiscreteActionProducer,
    NaivePseudoPolicyActionProducer
)


@torch.no_grad()
def estimate_bound_for_num_actions(
    num_actions, 
    num_iterations=100, 
    num_processes_to_simulate=100, 
    num_samples_per_process=1_000_000, 
    enable_permutations: bool = False
):
    """
    Sample N target categorical distributions.
    For each, simulate M action tracking processes (denoising).
    Compute mean +- std of number of action changes throughout the
    process.
    """

    dirichlet = Dirichlet(concentration=torch.ones(num_processes_to_simulate, num_actions))

    buffer = []
    for _ in tqdm(range(num_iterations), leave=False):
        initial_dist_probs = dirichlet.sample().unsqueeze(1)
        initial_dist_probs = repeat(initial_dist_probs, 'a 1 c -> a b c', b=num_samples_per_process)
        target_dist_probs = dirichlet.sample().unsqueeze(1)
        target_dist_probs = repeat(target_dist_probs, 'a 1 c -> a b c', b=num_samples_per_process)

        stable = StableDiscreteActionProducer(random_permutation=enable_permutations)
        
        stable_actions = [
            stable(Categorical(probs=initial_dist_probs))[0],
            stable(Categorical(probs=target_dist_probs))[0]
        ]

        stable_action_changes = torch.zeros(num_processes_to_simulate)
        stable_action_changes += (stable_actions[1] != stable_actions[0]).float().mean(dim=1)

        total_variations = 0.5*(initial_dist_probs - target_dist_probs).abs().sum(-1)[:, 0]  # (num_processes_to_simulate,)
        assert total_variations.numel() == stable_action_changes.numel(), f"got {total_variations.numel()} != {stable_action_changes.numel()}"
        estimated_tv = stable_action_changes / total_variations
        buffer.append(estimated_tv)
        max_changes = estimated_tv.max().item()
        maximizer = estimated_tv.argmax()

    combined = torch.cat(buffer).cpu().numpy()
    return combined


def estimate_bound(
    res_path: Path,
    num_iterations=100, 
    num_processes_to_simulate=100, 
    num_samples_per_process=1_000_000, 
    enable_permutations: bool = False
):
    action_sizes = [2, 4, 8, 16, 32, 64]
    per_size_results = {}

    for n in tqdm(action_sizes):
        per_size_results[n] = estimate_bound_for_num_actions(
            num_actions=n,
            num_iterations=num_iterations,
            num_processes_to_simulate=num_processes_to_simulate,
            num_samples_per_process=num_samples_per_process,
            enable_permutations=enable_permutations,
        )

    np.savez(res_path, **{str(k): v for k, v in per_size_results.items()})


@torch.no_grad()
def estimate_bound_of_example(p, q, seed=None):
    """
    Sample N target categorical distributions.
    For each, simulate M action tracking processes (denoising).
    Compute mean +- std of number of action changes throughout the
    process.
    """
    num_actions = 20
    num_samples_per_process = 10000000
    process_num_steps = 2

    initial_dist_probs = repeat(p, 'c -> b c', b=num_samples_per_process)
    target_dist_probs = repeat(q, 'c -> b c', b=num_samples_per_process)
    # target_dist = Categorical(probs=target_dist_probs)

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed=seed)
    stable = StableDiscreteActionProducer(generator=rng, random_permutation=False)
    
    stable_actions = [
        stable(Categorical(probs=initial_dist_probs))[0],
        stable(Categorical(probs=target_dist_probs))[0]
    ]

    stable_action_changes = torch.zeros(1)
    stable_action_changes += (stable_actions[1] != stable_actions[0]).float().mean(dim=0)

    total_variations = 0.5*(p - q).abs().sum()  # (num_processes_to_simulate,)
    max_changes = (stable_action_changes / total_variations).item()
    

    print(f"Maximum num of changes: abs: {stable_action_changes}, scaled to TV units: {max_changes}")

    return max_changes


def compute_theoretical_bound(p, q):
    
    def compute_alpha(dist: torch.Tensor):
        alpha = torch.zeros_like(dist)
        n = dist.numel()
        s = 1
        for i in range(n):
            alpha[i] = dist[i] / s
            s = (1 - alpha[i]) * s if alpha[i] < 1 else 1
        return alpha

    alpha_p = compute_alpha(p)[:-1]
    alpha_q = compute_alpha(q)[:-1]

    tv = 0.5 * (p-q).abs().sum()
    bound = (alpha_p-alpha_q).abs().sum()
    print(f"Alpha p: {alpha_p}\nAlpha q: {alpha_q}\nBound: {bound} = {bound / tv} TV(p,q)")


@torch.no_grad()
def eval_action_producer(entropy_setting: Literal['balanced', 'low', 'high'] = 'balanced'):
    """
    Sample N target categorical distributions.
    For each, simulate M action tracking processes (denoising).
    Compute mean +- std of number of action changes throughout the
    process.
    """
    num_actions = 10
    num_iterations = 1_000
    num_processes_to_simulate = 10
    num_samples_per_process = 1_000_000
    process_num_steps = 16
    assert process_num_steps > 1

    print(f"Evaluating with '{entropy_setting}' entropy setting")

    scale = 1
    if entropy_setting == 'high':
        scale = 5
    elif entropy_setting == 'low':
        scale = 0.2

    stable_buffer = []
    naive_buffer = []
    for _ in tqdm(range(num_iterations)):
        dirichlet = Dirichlet(concentration=torch.ones(num_processes_to_simulate, num_actions) * scale)

        initial_dist = Categorical(logits=torch.zeros(num_processes_to_simulate, num_samples_per_process, num_actions))
        initial_dist_probs = initial_dist.probs
        initial_dist_probs = dirichlet.sample().unsqueeze(1)
        initial_dist_probs = repeat(initial_dist_probs, 'a 1 c -> a b c', b=num_samples_per_process)
        target_dist_probs = dirichlet.sample().unsqueeze(1)

        stable = StableDiscreteActionProducer(random_permutation=True)
        
        stable_actions = []
        naive_actions = []

        for i in range(process_num_steps):
            t = (float(i) / (process_num_steps-1))
            if t < 0.5:
                dist_probs = (1-t*2) * initial_dist_probs + 2*t * target_dist_probs
            else:
                dist_probs = target_dist_probs
            dist = Categorical(probs=dist_probs)

            stable_actions.append(stable(dist)[0])
            naive_actions.append(dist.sample())

        stable_action_changes = torch.zeros(num_processes_to_simulate)
        naive_action_changes = torch.zeros(num_processes_to_simulate)
        for i in range(len(stable_actions) - 1):
            stable_action_changes += (stable_actions[i+1] != stable_actions[i]).float().mean(dim=1)
            naive_action_changes += (naive_actions[i+1] != naive_actions[i]).float().mean(dim=1)

        stable_buffer.append(stable_action_changes)
        naive_buffer.append(naive_action_changes)

    stable_action_changes = torch.cat(stable_buffer)
    naive_action_changes = torch.cat(naive_buffer)
    stable_mean = stable_action_changes.mean().item()
    stable_std = stable_action_changes.std().item()

    naive_mean = naive_action_changes.mean().item()
    naive_std = naive_action_changes.std().item()

    print(f"Stable: {stable_mean} +- {stable_std}")
    print(f"Naive: {naive_mean} +- {naive_std}")

    return (stable_mean, stable_std), (naive_mean, naive_std)


def run_analysis(results_filename: Path, entropy_settings):
    stable_res_dict = {}
    naive_res_dict = {}
    for entropy_setting in entropy_settings:
        stable_res, naive_res = eval_action_producer(entropy_setting=entropy_setting)
        stable_res_dict[entropy_setting] = stable_res
        naive_res_dict[entropy_setting] = naive_res

    # store results
    with open(results_filename, 'w') as f:
        json.dump({'stable': stable_res_dict, 'naive': naive_res_dict}, f)


def plot_results(results_filename: Path, entropy_settings: list[str]):
    with open(results_filename, 'r') as f:
        res_dict = json.load(f)

    stable_res_dict = res_dict['stable']
    naive_res_dict = res_dict['naive']

    plt.rcParams.update({
        "font.size": 18,          # base font size
        "axes.labelsize": 18,     # x and y labels
        "axes.titlesize": 18,     # title
        "xtick.labelsize": 15,    # x-axis tick labels
        "ytick.labelsize": 15,    # y-axis tick labels
        "legend.fontsize": 15,    # legend
        "figure.titlesize": 18    # overall figure title
    })

    stable_means = [stable_res_dict[s][0] for s in entropy_settings]
    stable_stds  = [stable_res_dict[s][1] for s in entropy_settings]
    naive_means  = [naive_res_dict[s][0] for s in entropy_settings]
    naive_stds   = [naive_res_dict[s][1] for s in entropy_settings]

    x = np.arange(len(entropy_settings))  # positions
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.bar(x - width/2, stable_means, width, yerr=stable_stds,
        label="Stable (ours)", capsize=5)
    ax.bar(x + width/2, naive_means, width, yerr=naive_stds,
        label="Naive", capsize=5)

    # Formatting
    ax.set_ylabel("Average number of\naction changes", fontsize=18)
    # ax.set_yticks(fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(entropy_settings)
    ax.set_xlabel('Entropy Setting', fontsize=18)
    # ax.legend()
    plt.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),   # 0.5 = center horizontally, 1.02 = a bit above the axes
        ncol=2,                       # put all labels in one row (number of columns = number of labels)
        frameon=False                 # optional: remove legend box
    )
    plt.tight_layout()
    plt.show()


def plot_boound_estimates_results(results_path: Path):
    loaded = np.load(results_path)
    data = {int(k): loaded[k] for k in loaded.files}

    keys = list(data.keys())
    values = [data[k] for k in keys]

    plt.figure(figsize=(6,4))
    plt.violinplot(values, positions=range(len(keys)), showmeans=True, showextrema=True, showmedians=False)
    plt.xticks(range(len(keys)), keys, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(r"Number of Actions ($N$)", fontsize=18)
    plt.ylabel("Number of action changes\n" + r"in $\delta(\bm{p}, \bm{q})$ units", fontsize=18)
    plt.tight_layout()
    plt.show()


def main():
    results_filename = Path('results/stable_action_changes_analysis-all-all.json')
    entropy_settings = ['low', 'balanced', 'high']
    with torch.device("cuda"):
        run_analysis(results_filename, entropy_settings)

    plot_results(results_filename, entropy_settings)

    res_path = Path('results/stable_action_changes_bound_estimates-perm.npz')
    with torch.device("cuda"):
        estimate_bound(res_path, num_iterations=1000, num_processes_to_simulate=10, enable_permutations=True)
    plot_boound_estimates_results(res_path)



        




if __name__ == '__main__':
    main()


