from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import click
from datetime import datetime

from config.agent import get_agent_offline_config, Agent
from horizon_imagination.envs import make_env
from experiments.train_agent_online import GracefulShutdown
    

def load_pretrained_agent(agent_cfg, weights_path, load_type):
    agent: Agent = agent_cfg.make_instance()
    checkpoint = torch.load(weights_path)
    tok_weights = {k[len("tokenizer."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("tokenizer.")}
    wm_weights = {k[len("world_model."):]: v for k, v in checkpoint["state_dict"].items() if k.startswith("world_model.")}

    if load_type == 'all':
        agent = Agent.load_from_checkpoint(weights_path, config=agent_cfg)
    else: 
        agent.tokenizer.load_state_dict(tok_weights)

        if load_type == 'wm':
            agent.world_model.load_state_dict(wm_weights)
    
    return agent


def train_agent(
        benchmark: Literal['retro', 'craftium', 'ale'], 
        offline: bool,
        data_path: Path,
        weights_path: Path, 
        load_type: Literal['tok', 'wm', 'all'],
    ):
    
    wandb_logger = WandbLogger(
        project="DWM", 
        save_dir="wandb", 
        offline=offline, 
        tags=['offline-agent']
    )
    # wandb_logger.watch(agent)

    env, env_name = make_env(benchmark, portal_env_backend='mm')

    agent_cfg = get_agent_offline_config(
        env=env, 
        env_name=env_name,
        replay_buf_data_path=data_path,
        test_env=make_env(benchmark, portal_env_backend='mm')[0]
    )

    torch.set_float32_matmul_precision('high')
    trainer = L.Trainer(
        # gradient_clip_val=experiment_config.grad_clip_val,
        max_epochs=agent_cfg.training.num_epochs,
        reload_dataloaders_every_n_epochs=1,
        limit_val_batches=1,
        check_val_every_n_epoch=1,
        accelerator='gpu',
        logger=wandb_logger,
        callbacks=[
            GracefulShutdown(),
        ],
        precision='32',  # bf16-mixed = mixed precision AMP
    )

    if weights_path is not None:
        agent = load_pretrained_agent(agent_cfg, weights_path, load_type)
    else:
        agent: Agent = agent_cfg.make_instance()

    trainer.fit(agent)


def eval_agent():
    pass


@click.command()
@click.argument('data-path', type=click.Path(path_type=Path, exists=True))
@click.option(
    '-b', 
    '--benchmark', 
    type=click.Choice(['craftium', 'ale']), 
    default='ale'
)
@click.option(
    '-w', 
    '--weights-path', 
    type=click.Path(exists=True, path_type=Path), 
)
@click.option(
    '-l', 
    '--load-type', 
    type=click.Choice(['tok', 'wm', 'all']), 
    default='wm'
)
@click.option('--offline', is_flag=True)
def main(
    data_path: Path, 
    benchmark, 
    weights_path: Path, 
    load_type: Literal['tok', 'wm', 'all'],
    offline: bool, 
):
    train_agent(
        benchmark=benchmark, 
        offline=offline, 
        data_path=data_path, 
        weights_path=weights_path, 
        load_type=load_type
    )


if __name__ == '__main__':
    main()

