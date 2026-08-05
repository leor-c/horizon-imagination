from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichProgressBar, Callback
from lightning.pytorch import seed_everything
import click
from datetime import datetime

from config.agent import get_agent_online_config, Agent
from config.env.default import get_default_env_config
from horizon_imagination.envs import make_env


class GracefulShutdown(Callback):
    def on_exception(self, trainer, pl_module, err):
        # Runs on KeyboardInterrupt and any other exception
        if isinstance(pl_module, Agent):
            pl_module._close_envs()  # important! (we use portal-env)


def train_agent(
        benchmark: Literal['craftium', 'ale'], 
        offline,
        portal_env_backend: Literal['docker', 'micromamba', 'mm'],
        game: Optional[str],
        baseline: str,
        seed: Optional[int] = None,
        outputs_path: Path = None,
        discard_data: bool = False,
        decay_horizon: float = 4,
        budget: int = 16,
        intrinsic_reward_normalization: str = 'quantile_ema',
        intrinsic_reward_target_ratio: float = 0.0,
        intrinsic_reward_coeff: float = 1.0,
        pure_exploration: bool = False,
        intrinsic_reward_truncated_roundtrip: bool = False,
    ):
    if seed is not None:
        seed_everything(seed)

    if outputs_path is None:
        outputs_path = Path.cwd() / 'outputs'
    outputs_path.mkdir(exist_ok=True, parents=True)
    
    wandb_logger = WandbLogger(
        project="HorizonImagination", 
        save_dir=outputs_path / "wandb", 
        offline=offline, 
        config={
            'benchmark': benchmark,
            'game': game,
            'baseline': baseline,
            'seed': seed,
            'decay_horizon': decay_horizon,
            'budget': budget,
            'discard_data': discard_data,
            'outputs_path': outputs_path,
            'intrinsic_reward_normalization': intrinsic_reward_normalization,
            'intrinsic_reward_target_ratio': intrinsic_reward_target_ratio,
            'intrinsic_reward_coeff': intrinsic_reward_coeff,
            'pure_exploration': pure_exploration,
            'intrinsic_reward_truncated_roundtrip': intrinsic_reward_truncated_roundtrip,
        }
    )
    # wandb_logger.watch(agent)

    env_cfg = get_default_env_config(benchmark=benchmark, game=game)
    env, env_name = make_env(
        benchmark, portal_env_backend=portal_env_backend, env_name=game, resolution=env_cfg.resolution
    )
    seed_str = f'_{seed}' if seed is not None else ''
    
    if discard_data:
        data_path = None
    else:
        data_path = outputs_path / 'data' / f'run_{benchmark}_{baseline}{seed_str}_{datetime.now().strftime("%d-%m-%Y-%H-%M")}' / 'dataset'
        data_path.mkdir(exist_ok=True, parents=True)
        
    agent_cfg = get_agent_online_config(
        env=env,
        env_name=env_name,
        replay_buf_data_path=data_path,
        baseline=baseline,
        # test_env=make_env(benchmark, env_name=env_name, portal_env_backend='mm')[0],
        decay_horizon=decay_horizon,
        resolution=env_cfg.resolution,
        budget=budget,
        intrinsic_reward_normalization=intrinsic_reward_normalization,
        intrinsic_reward_target_ratio=intrinsic_reward_target_ratio,
        intrinsic_reward_coeff=intrinsic_reward_coeff,
        pure_exploration=pure_exploration,
        intrinsic_reward_truncated_roundtrip=intrinsic_reward_truncated_roundtrip,
    )

    torch.set_float32_matmul_precision('high')
    trainer = L.Trainer(
        max_epochs=agent_cfg.training.num_epochs,
        reload_dataloaders_every_n_epochs=1,
        limit_val_batches=1,
        check_val_every_n_epoch=10,
        accelerator='gpu',
        logger=wandb_logger,
        callbacks=[
            # RichProgressBar(),
            GracefulShutdown(),
        ],
        precision='32',
        num_sanity_val_steps=0,
    )

    agent = agent_cfg.make_instance()

    trainer.fit(agent)


@click.command()
@click.option(
    '-b', 
    '--benchmark', 
    type=click.Choice(['craftium', 'ale']), 
    default='ale'
)
@click.option('--offline', is_flag=True)
@click.option(
    '-p',
    '--portal-env-backend', 
    type=click.Choice(['docker', 'micromamba', 'mm']), 
    default='mm'
)
@click.option('-g', '--game', type=str)
@click.option(
    '--baseline', 
    type=click.Choice([
        'ar', 
        'hi', 
        'hi-h4-b32', 
        'hi-h4-b16', 
        'naive'
    ]), 
    default='hi'
)
@click.option('--seed', type=int, default=None)
@click.option('-o', '--outputs-path', type=Path, default=Path.cwd() / 'outputs')
@click.option('--discard-data', is_flag=True)
@click.option('--decay_horizon', type=float, default=4)
@click.option('--budget', type=int, default=32)
@click.option(
    '--intrinsic-reward-normalization',
    type=click.Choice(['quantile_ema', 'rnd_return_std']),
    default='quantile_ema',
    help="How the latent reconstruction bonus is auto-scaled. 'rnd_return_std' is the mode "
         "for sparse-reward environments, where the extrinsic band is near-degenerate."
)
@click.option(
    '--intrinsic-reward-target-ratio',
    type=float,
    default=0.0,
    help="'quantile_ema' mode: bonus scale as a fraction of the extrinsic reward's own "
         "scale. 0 disables the intrinsic reward entirely."
)
@click.option(
    '--intrinsic-reward-coeff',
    type=float,
    default=1.0,
    help="'rnd_return_std' mode: fixed coefficient on the std-normalized bonus."
)
@click.option(
    '--pure-exploration',
    is_flag=True,
    help="Train imagination on the scaled intrinsic reward alone, excluding extrinsic reward."
)
@click.option(
    '--intrinsic-reward-truncated-roundtrip',
    is_flag=True,
    help="Use the truncated latent round-trip, which never reaches pixel space. Measured at "
         "B=30/T=32: +9 percent controller-step time instead of +63, and no extra peak memory."
)
def main(
    benchmark, 
    offline: bool, 
    portal_env_backend: str, 
    game: str, 
    baseline: str, 
    seed: Optional[int],
    outputs_path: Path,
    discard_data: bool,
    decay_horizon: float,
    budget: int,
    intrinsic_reward_normalization: str,
    intrinsic_reward_target_ratio: float,
    intrinsic_reward_coeff: float,
    pure_exploration: bool,
    intrinsic_reward_truncated_roundtrip: bool,
):
    if baseline == 'hi-h4-b32':
        baseline = 'hi'
        decay_horizon = 4
        budget = 32
    elif baseline == 'hi-h4-b16':
        baseline = 'hi'
        decay_horizon = 4
        budget = 16

    train_agent(
        benchmark=benchmark, 
        offline=offline, 
        portal_env_backend=portal_env_backend, 
        game=game,
        baseline=baseline,
        seed=seed,
        outputs_path=outputs_path,
        discard_data=discard_data,
        decay_horizon=decay_horizon,
        budget=budget,
        intrinsic_reward_normalization=intrinsic_reward_normalization,
        intrinsic_reward_target_ratio=intrinsic_reward_target_ratio,
        intrinsic_reward_coeff=intrinsic_reward_coeff,
        pure_exploration=pure_exploration,
        intrinsic_reward_truncated_roundtrip=intrinsic_reward_truncated_roundtrip,
    )


if __name__ == '__main__':
    main()

