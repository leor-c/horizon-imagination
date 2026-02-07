from typing import Literal
from horizon_imagination.envs.wrappers import ModalityDictObsWrapper, ImageChannelsFirst, FrameSkip
import gymnasium as gym


def make_craftium_env(
        env_name: str = "Craftium/ChopTree-v0",
        agent_in_docker: bool = True,
        frameskip: int = 4,
    ):
    import portal_env

    craftium_kwargs = {
        'frameskip': frameskip, 
        'minetest_conf': {'time_speed': 0},  # No night, same time of day.
        'sync_mode': True,
        'fps_max': 200,
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
        terminate_on_life_loss: bool = False, 
        sign_rewards: bool = False,
        repeat_action_probability: float = 0.0,
        agent_in_docker: bool = True,
    ):
    import portal_env
    from horizon_imagination.envs.wrappers import EpisodicLifeEnv, ResizeObsWrapper, NoopResetEnv, SignRewardWrapper
    env = portal_env.AgentSidePortal(
        "ale", 
        env_args=[env_name], 
        env_kwargs={"repeat_action_probability": repeat_action_probability},
        agent_in_docker=agent_in_docker
    )
    if repeat_action_probability == 0.0:
        env = NoopResetEnv(env)
    if terminate_on_life_loss:
        env = EpisodicLifeEnv(env)
    env = ResizeObsWrapper(env, size=(64, 64))
    env = ImageChannelsFirst(env)
    if sign_rewards:
        env = SignRewardWrapper(env)
    env = ModalityDictObsWrapper(env)

    return env

def make_env(
        benchmark: Literal['craftium', 'ale'], 
        portal_env_backend: Literal['docker', 'micromamba', 'mm'],
        env_name: str = None,
    ) -> tuple[gym.Env, str]:
    if portal_env_backend == 'docker':
        agent_in_docker = True
    elif portal_env_backend in ['micromamba', 'mm']:
        agent_in_docker = False
    else:
        raise NotImplementedError()
    
    env_kwargs = {
        'agent_in_docker': agent_in_docker, 
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
    
    else:
        raise ValueError(f"Benchmark '{benchmark}' is not supported.")
