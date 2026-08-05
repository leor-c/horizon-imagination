import torch.nn as nn
from tensordict.tensordict import TensorDict
from horizon_imagination.modules.transform import BaseTransform
from horizon_imagination.utilities.types import Modality, ObsKey


class PerModalityTransform(BaseTransform, nn.Module):
    """
    Applies a transform per observation key.

    A transform is looked up first by the exact key, then by the key's modality, so
    that a modality-wide default (e.g. the image tokenizer, shared by every camera)
    can be overridden for individual keys. Per-key transforms are required for the
    vector modality, where two keys may carry different raw dimensions and therefore
    cannot share the same module.
    """
    def __init__(
            self,
            learned_per_modality_transforms: dict[Modality, BaseTransform] = {},
            fixed_per_modality_transforms: dict[Modality, BaseTransform] = {},
            learned_per_key_transforms: dict[ObsKey, BaseTransform] = {},
            fixed_per_key_transforms: dict[ObsKey, BaseTransform] = {},
        ):
        super().__init__()
        learned_keys = set(learned_per_modality_transforms.keys())
        fixed_keys = set(fixed_per_modality_transforms.keys())
        assert len(learned_keys.intersection(fixed_keys)) == 0

        learned_obs_keys = set(learned_per_key_transforms.keys())
        fixed_obs_keys = set(fixed_per_key_transforms.keys())
        assert len(learned_obs_keys.intersection(fixed_obs_keys)) == 0

        self.learned_transforms = nn.ModuleDict({m.name: v for m, v in learned_per_modality_transforms.items()})
        self.fixed_transforms = fixed_per_modality_transforms
        self.per_modality_transforms = learned_per_modality_transforms | fixed_per_modality_transforms

        # nn.ModuleDict keys may not contain '.', while ObsKey is 'modality|name':
        self.learned_key_transforms = nn.ModuleDict({str(k): v for k, v in learned_per_key_transforms.items()})
        self.fixed_key_transforms = fixed_per_key_transforms
        self.per_key_transforms = learned_per_key_transforms | fixed_per_key_transforms

    def get_transform(self, key: str) -> BaseTransform:
        key = ObsKey(key)
        if key in self.per_key_transforms:
            return self.per_key_transforms[key]
        assert key.modality in self.per_modality_transforms, \
            f"No transform registered for key '{key}' or modality '{key.modality}'."
        return self.per_modality_transforms[key.modality]

    def transform(self, x: TensorDict, *args, **kwargs):
        z = TensorDict({
            k: self.get_transform(k).transform(x[k], *args, **kwargs)
            for k in x.keys()
        }, device=x.device, batch_size=x.batch_size)

        return z

    def inverse(self, z: TensorDict, *args, **kwargs) -> TensorDict:
        x = TensorDict({
            k: self.get_transform(k).inverse(z_k, *args, **kwargs)
            for k, z_k in z.items()
        }, device=z.device, batch_size=z.batch_size)
        return x

    def roundtrip(self, z: TensorDict, truncated: bool = False, *args, **kwargs) -> TensorDict:
        return TensorDict({
            k: self.get_transform(k).roundtrip(z_k, truncated=truncated, *args, **kwargs)
            for k, z_k in z.items()
        }, device=z.device, batch_size=z.batch_size)



