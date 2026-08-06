import dataclasses
from typing import Optional, Literal


@dataclasses.dataclass
class EnvConfig:
    name: str
    resolution: Optional[int] = 64


def get_default_env_config(
        benchmark: Literal['craftium', 'ale', 'mujoco'],
        game: Optional[str] = None,
) -> EnvConfig:
    resolution = 64

    if game is not None:
        return EnvConfig(game, resolution=resolution)

    if benchmark == 'craftium':
        return EnvConfig('Craftium/ChopTree-v0', resolution=resolution)
    elif benchmark == 'ale':
        return EnvConfig('ALE/Boxing-v5', resolution=resolution)
    elif benchmark == 'mujoco':
        # State observations: `resolution` is unused by the env, but still sizes the
        # image-latent constants the DiT config is derived from.
        return EnvConfig('HalfCheetah-v5', resolution=resolution)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
