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
        }
    )
    # wandb_logger.watch(agent)

    env, env_name = make_env(benchmark, portal_env_backend=portal_env_backend, env_name=game)
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
        budget=budget,
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
    )


if __name__ == '__main__':
    main()

