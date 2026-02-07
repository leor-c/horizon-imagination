from typing import Literal
import wandb
import subprocess
import os
import sys
import click
from pathlib import Path
from loguru import logger


@click.command()
@click.argument('benchmark', type=click.Choice(['ale', 'craftium']))
def create_sweep(benchmark: Literal['ale', 'craftium']):
    """
    Creates a new WandB sweep with the specified configuration.
    The command specifies how to run the training script.
    """

    grid_kwargs = {}
    if benchmark == 'ale':
        grid_kwargs['game'] = {
            'values': [
                "ALE/Boxing-v5",
                "ALE/KungFuMaster-v5",
                "ALE/CrazyClimber-v5",
                "ALE/Gopher-v5",
            ]
        }
    elif benchmark == 'craftium':
        grid_kwargs['game'] = {
            'values': [
                'Craftium/Speleo-v0',
                'Craftium/ChopTree-v0',
                'Craftium/SmallRoom-v0',
                'Craftium/Room-v0',
            ]
        }
    else:
        raise ValueError(f'benchmark "{benchmark}" not supported.')

    sweep_config = {
        'method': 'grid',
        'parameters': {
            'baseline': {
                'values': ['ar', 'hi-h4-b16', 'hi-h4-b32']  
            },
            'seed': {
                'values': [0, 1, 2, 3, 4]
            },
            **grid_kwargs
        },
        'command': [
            'python',
            'experiments/train_agent_online.py',  
            '-b', benchmark,  
            '${args}',
        ]
    }
    
    sweep_id = wandb.sweep(sweep_config, project="HorizonImagination")
    
    logger.info(f"Created new WandB sweep with ID: {sweep_id}")
    logger.info("Use this ID to start the agents on your machines.")
    logger.info(f"Example command to start agents: python {__file__} run-agents {sweep_id} --num-gpus 8")
    return sweep_id


@click.command()
@click.argument('sweep_id', type=str)
@click.option('-n', '--num-gpus', type=int, default=8)
@click.option('-b', '--benchmark', type=click.Choice(['ale', 'craftium']), default='ale')
def run_agents(sweep_id, num_gpus, benchmark):
    logger.info(f"Starting {num_gpus} WandB agents for sweep ID: {sweep_id}")

    processes = []
    portal_env_proc = None

    def cleanup_processes():
        """Terminates all running subprocesses."""
        nonlocal portal_env_proc, processes
        for proc in processes:
            if proc.poll() is None: 
                logger.info(f"Terminating subprocess with PID: {proc.pid}")
                proc.terminate()
        if portal_env_proc and portal_env_proc.poll() is None:
            logger.info("Terminating portal-env process.")
            portal_env_proc.terminate()
        
        for proc in processes:
            proc.wait(timeout=5)
        if portal_env_proc:
            portal_env_proc.wait(timeout=5)

    
    import signal
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup_processes() or sys.exit(1))

    try:
        portal_env_proc = subprocess.Popen(['portal-env', 'start', benchmark, '-b', 'mm'])

        for i in range(num_gpus):
            logger.info(f"Launching agent for GPU {i}...")

            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(i)
            env['PYTHONPATH'] = Path.cwd()

            agent_cmd = ['wandb', 'agent', sweep_id]

            proc = subprocess.Popen(agent_cmd, env=env)
            processes.append(proc)
            
        logger.info(f"All {num_gpus} agents have been launched. Waiting for them to finish...")
        
        for proc in processes:
            proc.wait()

        if portal_env_proc:
            portal_env_proc.kill()
        
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Terminating all processes.")
    finally:
        cleanup_processes()
        logger.info("All processes have been terminated.")


@click.group()
def main():
    pass


main.add_command(create_sweep)
main.add_command(run_agents)


if __name__ == '__main__':
    main()
