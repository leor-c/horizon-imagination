from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import click
import wandb
from loguru import logger
from portal_env.config import config as portal_config


Benchmark = Literal["ale", "craftium", "mujoco"]

WANDB_PROJECT = "HorizonImagination"
PORTAL_BATCHER_HOST = "127.0.0.1"
PAPER_BASELINES = ("ar", "hi-h4-b16", "hi-h4-b32")
PAPER_SEEDS = tuple(range(5))
BENCHMARK_GAMES = {
    "ale": (
        "ALE/Boxing-v5",
        "ALE/CrazyClimber-v5",
        "ALE/Gopher-v5",
        "ALE/KungFuMaster-v5",
    ),
    "craftium": (
        "Craftium/ChopTree-v0",
        "Craftium/Room-v0",
        "Craftium/SmallRoom-v0",
        "Craftium/Speleo-v0",
    ),
    "mujoco": (
        "HalfCheetah-v5",
        "Hopper-v5",
        "Walker2d-v5",
        "Ant-v5",
    ),
}


def build_sweep_config(benchmark: Benchmark) -> dict:
    """Build the fixed paper-reproduction grid for ``benchmark``.

    The named baseline values are existing presets handled by
    ``train_agent_online.py``. No model or optimizer hyperparameter is changed here.
    """
    return {
        "name": f"{benchmark}-paper-reproduction",
        "method": "grid",
        "parameters": {
            "baseline": {"values": list(PAPER_BASELINES)},
            "seed": {"values": list(PAPER_SEEDS)},
            "game": {"values": list(BENCHMARK_GAMES[benchmark])},
        },
        "program": "experiments/train_agent_online.py",
        "command": [
            "${env}",
            "${interpreter}",
            "${program}",
            "--benchmark",
            benchmark,
            "${args}",
        ],
    }


@click.command()
@click.argument("benchmark", type=click.Choice(tuple(BENCHMARK_GAMES)))
@click.option("--entity", default=None, help="Optional W&B user or team name.")
def create_sweep(benchmark: Benchmark, entity: str | None) -> str:
    """Create the fixed paper-reproduction sweep for BENCHMARK.

    For ALE this creates 60 runs: four Atari environments, five seeds, and the
    three controller configurations plotted in the paper.
    """
    sweep_id = wandb.sweep(
        sweep=build_sweep_config(benchmark),
        project=WANDB_PROJECT,
        entity=entity,
    )

    example_cmd = [
        sys.executable,
        Path(__file__).as_posix(),
        "run-agents",
        sweep_id,
        "--benchmark",
        benchmark,
        "--num-gpus",
        "8",
    ]
    if entity is not None:
        example_cmd.extend(("--entity", entity))

    logger.info(f"Created W&B sweep {sweep_id} in project {WANDB_PROJECT}.")
    logger.info(f"Start local agents with: {shlex.join(example_cmd)}")
    return sweep_id


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    return True


def _process_groups_in_session(session_id: int) -> set[int]:
    """Return groups in an isolated session, including nested child sessions."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return {session_id} if _process_group_exists(session_id) else set()

    processes: dict[int, tuple[int, int, int]] = {}
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            stat = (process_dir / "stat").read_text()
            _, separator, fields_text = stat.rpartition(")")
            if not separator:
                continue
            fields = fields_text.split()
            process_id = int(process_dir.name)
            parent_process_id = int(fields[1])
            process_group_id = int(fields[2])
            process_session_id = int(fields[3])
        except (OSError, IndexError, ValueError):
            continue
        processes[process_id] = (
            parent_process_id,
            process_group_id,
            process_session_id,
        )

    owned_process_ids = {
        process_id
        for process_id, (_, _, process_session_id) in processes.items()
        if process_session_id == session_id
    }
    while True:
        descendants = {
            process_id
            for process_id, (parent_process_id, _, _) in processes.items()
            if parent_process_id in owned_process_ids
        }
        expanded_process_ids = owned_process_ids | descendants
        if expanded_process_ids == owned_process_ids:
            break
        owned_process_ids = expanded_process_ids

    return {
        processes[process_id][1]
        for process_id in owned_process_ids
    }


def _signal_process_session(
    proc: subprocess.Popen,
    name: str,
    signum: signal.Signals,
) -> bool:
    """Signal all process groups spawned inside a child's isolated session."""
    session_id = proc.pid
    process_group_ids = _process_groups_in_session(session_id)
    if not process_group_ids:
        proc.poll()
        return False

    logger.info(
        f"Sending {signum.name} to {name} session "
        f"(SID {session_id}, PGIDs {sorted(process_group_ids)})."
    )
    signal_sent = False
    # Signal nested run groups before the W&B wrapper that waits for them.
    ordered_group_ids = sorted(
        process_group_ids,
        key=lambda process_group_id: process_group_id == session_id,
    )
    for process_group_id in ordered_group_ids:
        try:
            os.killpg(process_group_id, signum)
            signal_sent = True
        except ProcessLookupError:
            pass
    proc.poll()
    return signal_sent


def _wait_for_process_sessions(
    processes: list[tuple[subprocess.Popen, str]],
    timeout: float,
) -> list[tuple[subprocess.Popen, str]]:
    """Wait concurrently for isolated sessions and return any still alive."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = []
        for proc, name in processes:
            proc.poll()  # Reap an exited session leader before checking descendants.
            if _process_groups_in_session(proc.pid):
                remaining.append((proc, name))

        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(0.1)


def _shutdown_processes(
    agents: list[subprocess.Popen],
    portal_env_proc: subprocess.Popen | None,
    interrupted: bool,
) -> None:
    """Stop all child groups concurrently, escalating unresponsive children."""
    processes = [
        (agent, f"W&B agent for GPU {index}")
        for index, agent in enumerate(agents)
    ]
    if portal_env_proc is not None:
        processes.append((portal_env_proc, "portal-env server"))

    if not processes:
        return

    for proc, name in processes:
        first_signal = (
            signal.SIGINT
            if interrupted and name.startswith("W&B agent")
            else signal.SIGTERM
        )
        _signal_process_session(proc, name, first_signal)

    remaining = _wait_for_process_sessions(processes, timeout=5)
    if remaining:
        for proc, name in remaining:
            _signal_process_session(proc, name, signal.SIGTERM)
        remaining = _wait_for_process_sessions(remaining, timeout=5)

    if remaining:
        for proc, name in remaining:
            logger.warning(f"{name} did not terminate; killing its process session.")
            _signal_process_session(proc, name, signal.SIGKILL)
        _wait_for_process_sessions(remaining, timeout=1)

    for proc, _ in processes:
        proc.poll()


def allocate_run_counts(num_gpus: int, max_runs: int | None) -> list[int | None]:
    """Allocate an optional total run limit across the local GPU agents."""
    if max_runs is None:
        return [None] * num_gpus

    num_agents = min(num_gpus, max_runs)
    runs_per_agent, agents_with_extra_run = divmod(max_runs, num_agents)
    return [
        runs_per_agent + (agent_index < agents_with_extra_run)
        for agent_index in range(num_agents)
    ]


def build_agent_command(
    sweep_id: str,
    entity: str | None,
    run_count: int | None,
) -> list[str]:
    """Build a W&B agent command compatible with the installed CLI."""
    command = [
        sys.executable,
        "-m",
        "wandb",
        "agent",
        "--project",
        WANDB_PROJECT,
    ]
    if entity is not None:
        command.extend(("--entity", entity))
    if run_count is not None:
        command.extend(("--count", str(run_count)))
    command.append(sweep_id)
    return command


def ensure_port_available(host: str, port: int) -> None:
    """Fail before launching anything if another portal server owns the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as error:
            raise click.ClickException(
                f"Portal batcher port {host}:{port} is already in use. "
                "Stop the existing portal-env process before starting this launcher."
            ) from error


def wait_for_port(
    host: str,
    port: int,
    proc: subprocess.Popen,
    timeout: float = 30,
) -> None:
    """Wait until portal-env is accepting connections before starting W&B."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise click.ClickException(
                f"portal-env exited with status {proc.returncode} before becoming ready."
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)

    raise click.ClickException(
        f"portal-env did not listen on {host}:{port} within {timeout:g} seconds."
    )


@click.command()
@click.argument("sweep_id", type=str)
@click.option(
    "-n",
    "--num-gpus",
    type=click.IntRange(min=1),
    default=8,
    show_default=True,
    help="Launch one W&B agent per GPU, using GPU indices 0..N-1.",
)
@click.option(
    "--max-runs",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum total runs for this invocation; omit to finish the sweep.",
)
@click.option(
    "-b",
    "--benchmark",
    type=click.Choice(tuple(BENCHMARK_GAMES)),
    default="ale",
    show_default=True,
    help="Portal environment server to start.",
)
@click.option("--entity", default=None, help="W&B entity, if not encoded in SWEEP_ID.")
def run_agents(
    sweep_id: str,
    num_gpus: int,
    max_runs: int | None,
    benchmark: Benchmark,
    entity: str | None,
) -> None:
    """Run sweep agents on local GPUs, optionally for a limited batch of runs."""
    run_counts = allocate_run_counts(num_gpus, max_runs)
    logger.info(
        f"Starting {len(run_counts)} W&B agents for sweep {sweep_id}"
        + (f", limited to {max_runs} total runs." if max_runs is not None else ".")
    )

    agents: list[subprocess.Popen] = []
    portal_env_proc: subprocess.Popen | None = None
    interrupted = False
    portal_port = portal_config.env_ports[benchmark]
    try:
        ensure_port_available(PORTAL_BATCHER_HOST, portal_port)
        portal_env_proc = subprocess.Popen(
            ["portal-env", "start", benchmark, "-b", "mm"],
            start_new_session=True,
        )
        wait_for_port(
            PORTAL_BATCHER_HOST,
            portal_port,
            portal_env_proc,
        )

        workspace = str(Path.cwd())
        for gpu_index, run_count in enumerate(run_counts):
            run_limit = f" for up to {run_count} runs" if run_count is not None else ""
            logger.info(f"Launching W&B agent on GPU {gpu_index}{run_limit}.")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (workspace, env.get("PYTHONPATH")) if part
            )

            agent_cmd = build_agent_command(sweep_id, entity, run_count)
            agents.append(
                subprocess.Popen(agent_cmd, env=env, start_new_session=True)
            )

        logger.info("All agents are running; waiting for their assigned runs to complete.")
        exit_codes = [agent.wait() for agent in agents]
        failures = [code for code in exit_codes if code != 0]
        if failures:
            raise click.ClickException(
                f"{len(failures)} of {len(agents)} W&B agents exited unsuccessfully: "
                f"{exit_codes}"
            )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            "Ctrl+C received; stopping active W&B runs and the portal server."
        )
    finally:
        previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            _shutdown_processes(agents, portal_env_proc, interrupted)
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)
        logger.info("All child processes have stopped.")

    if interrupted:
        raise click.exceptions.Exit(130)


@click.group()
def main() -> None:
    """Create and execute Horizon Imagination W&B sweeps."""


main.add_command(create_sweep)
main.add_command(run_agents)


if __name__ == "__main__":
    main()
