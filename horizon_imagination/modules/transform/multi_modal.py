import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from tensordict.tensordict import TensorDict
from horizon_imagination.modules.transform import BaseTransform
from horizon_imagination.utilities.types import Modality, ObsKey


class PerModalityTransform(BaseTransform, nn.Module):
    def __init__(
            self, 
            learned_per_modality_transforms: dict[Modality, BaseTransform] = {},
            fixed_per_modality_transforms: dict[Modality, BaseTransform] = {},
        ):
        super().__init__()
        learned_keys = set(learned_per_modality_transforms.keys())
        fixed_keys = set(fixed_per_modality_transforms.keys())
        assert len(learned_keys.intersection(fixed_keys)) == 0

        self.learned_transforms = nn.ModuleDict({m.name: v for m, v in learned_per_modality_transforms.items()})
        self.fixed_transforms = fixed_per_modality_transforms
        self.per_modality_transforms = learned_per_modality_transforms | fixed_per_modality_transforms

    def transform(self, x: TensorDict, *args, **kwargs):
        z = TensorDict({
            k: self.per_modality_transforms[ObsKey(k).modality].transform(x[k], *args, **kwargs)
            for k in x.keys()
        }, device=x.device, batch_size=x.batch_size)

        return z
    
    def inverse(self, z: TensorDict, *args, **kwargs) -> TensorDict:
        x = TensorDict({
            k: self.per_modality_transforms[ObsKey(k).modality].inverse(z_k, *args, **kwargs) 
            for k, z_k in z.items()
        }, device=z.device, batch_size=z.batch_size)
        return x


class MultiModalCatAndFlatten(BaseTransform):
    def __init__(self, temporal_dim: int = 1, concat_dim: int = 2):
        assert concat_dim == temporal_dim + 1
        super().__init__()
        self.temporal_dim = temporal_dim
        self.concat_dim = concat_dim

    def transform(self, x: TensorDict[ObsKey, Tensor], *args, **kwargs):
        x_flat = list(x.values())
        lengths = {k: z.shape[self.concat_dim] for k, z in x.items()}
        seq_len = x_flat[0].shape[self.temporal_dim]
        x_flat = torch.cat(x_flat, dim=self.concat_dim)
        x_flat = x_flat.flatten(self.temporal_dim, self.concat_dim)
        return x_flat, lengths, seq_len

    def inverse(self, z, lengths: dict[ObsKey, int], seq_len: int, *args, **kwargs):
        device = z.device
        z = rearrange(z, "b (n k) c -> b n k c", n=seq_len, k=sum(lengths.values()))
        batch_size = z.shape[:2]
        z = torch.split(z, list(lengths.values()), dim=self.concat_dim)
        x = TensorDict({
            k: z_k for k, z_k in zip(lengths.keys(), z)
        }, device=device, batch_size=batch_size)
        return x



