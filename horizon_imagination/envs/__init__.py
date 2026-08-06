from typing import Literal
from horizon_imagination.envs.wrappers import (
    ModalityDictObsWrapper, ImageChannelsFirst, Float32ObsWrapper
)
import gymnasium as gym


def make_craftium_env(
        env_name: str = "Craftium/ChopTree-v0",
        agent_in_docker: bool = True,
        frameskip: int = 4,
        resolution: int = 64,
    ):
    import portal_env

    craftium_kwargs = {
        'frameskip': frameskip,
        'minetest_conf': {'time_speed': 0},  # No night, same time of day.
        'sync_mode': True,
        'fps_max': 200,
        'obs_width': resolution,
        'obs_height': resolution,
    }
    if env_name == 'Craftium/ChopTree-v0':
        craftium_kwargs['fps_max'] = 10
    if env_name == 'Craftium/SmallRoom-v0':
        craftium_kwargs['init_frames'] = 15

    env = portal_env.AgentSidePortal(
        "craftium", 
        env_args=[env_name], 
        env_kwargs=craftium_kwargs,
        agent_in_docker=agent_in_docker,
    )
    env = ImageChannelsFirst(env)
    env = ModalityDictObsWrapper(env)

    return env


def make_ale_env(
        env_name: str = "ALE/Boxing-v5",
        terminate_on_life_loss: bool = True,
        sign_rewards: bool = True,
        repeat_action_probability: float = 0.0,
        agent_in_docker: bool = True,
        resolution: int = 64,
    ):
    import portal_env
    from horizon_imagination.envs.wrappers import EpisodicLifeEnv, ResizeObsWrapper, NoopResetEnv, SignRewardWrapper
    env = portal_env.AgentSidePortal(
        "ale",
        env_args=[env_name],
        env_kwargs={"repeat_action_probability": repeat_action_probability},
        agent_in_docker=agent_in_docker
    )
    # Keep this inside the life-loss and reward-sign wrappers. It therefore records
    # raw ALE rewards and emits ``info['episode']`` only on a real game over, while
    # the agent still trains on signed rewards and treats each life as an episode.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if repeat_action_probability == 0.0:
        env = NoopResetEnv(env)
    if terminate_on_life_loss:
        env = EpisodicLifeEnv(env)
    env = ResizeObsWrapper(env, size=(resolution, resolution))
    env = ImageChannelsFirst(env)
    if sign_rewards:
        env = SignRewardWrapper(env)
    env = ModalityDictObsWrapper(env)

    return env


def make_mujoco_env(
        env_name: str = "HalfCheetah-v5",
        agent_in_docker: bool = True,
    ):
    """
    Gymnasium MuJoCo, served by portal-env's stock `mujoco` server (a bare
    `gymnasium.make`). The observation is the environment's state vector -- there is
    no image -- so none of the image preprocessing wrappers apply.
    """
    import portal_env

    env = portal_env.AgentSidePortal(
        "mujoco",
        env_args=[env_name],
        agent_in_docker=agent_in_docker,
    )
    env = Float32ObsWrapper(env)
    env = ModalityDictObsWrapper(env)

    return env


def make_env(
        benchmark: Literal['craftium', 'ale', 'mujoco'],
        portal_env_backend: Literal['docker', 'micromamba', 'mm'],
        env_name: str = None,
        resolution: int = 64,
    ) -> tuple[gym.Env, str]:
    if portal_env_backend == 'docker':
        agent_in_docker = True
    elif portal_env_backend in ['micromamba', 'mm']:
        agent_in_docker = False
    else:
        raise NotImplementedError()

    env_kwargs = {
        'agent_in_docker': agent_in_docker,
        'resolution': resolution,
    }
    
    if benchmark == 'craftium':
        if env_name is None:
            env_name = 'Craftium/ChopTree-v0'
        env_kwargs['env_name'] = env_name

        return make_craftium_env(**env_kwargs), env_name
    elif benchmark == 'ale':
        if env_name is None:
            env_name = 'ALE/Boxing-v5'
        env_kwargs['env_name'] = env_name

        return make_ale_env(**env_kwargs), env_name

    elif benchmark == 'mujoco':
        if env_name is None:
            env_name = 'HalfCheetah-v5'

        # State observations: `resolution` does not apply.
        return make_mujoco_env(env_name=env_name, agent_in_docker=agent_in_docker), env_name

    else:
        raise ValueError(f"Benchmark '{benchmark}' is not supported.")
