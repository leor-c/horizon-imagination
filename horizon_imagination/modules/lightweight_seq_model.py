import torch
import torch.nn as nn
from torch import Tensor
from tensordict.tensordict import TensorDict
import gymnasium as gym

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.modules.transform import PerModalityTransform


class LightweightSeqModel(nn.Module, Configurable):
    """
    A class for a lightwight sequence model operating in latent space.
    This type of models are appropriate for controller (actor-critic)
    and reward / termination modeling.
    """

    @dataclass
    class Config(BaseConfig):
        action_space: gym.Space
        # Assume each transform maps its modality to a 1-D vector:
        obs_vectorize_transforms: PerModalityTransform
        latent_dim: int
        num_layers: int = 1
        ignore_actions: bool = False
        device: torch.device = None
        dtype: torch.dtype = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.model = self._build_model()

        self.action_emb = self._build_action_emb()
        self.obs_transform = config.obs_vectorize_transforms

    def _build_model(self):
        return nn.LSTM(
            input_size=self.config.latent_dim,
            hidden_size=self.config.latent_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def _build_action_emb(self):
        assert isinstance(self.config.action_space, gym.spaces.Discrete)
        return nn.Embedding(
            num_embeddings=self.config.action_space.n,
            embedding_dim=self.config.latent_dim,
            device=self.config.device,
            dtype=self.config.dtype,
        )

    def forward(
        self,
        actions: Tensor,
        obs: TensorDict,
        state: tuple[Tensor, Tensor] = None,
        pad_mask: Tensor = None,
    ):
        x = self.obs_transform.transform(obs)

        if not self.config.ignore_actions:
            # If actions[i, j] = a_t  ==> obs[i, j] = o_{t+1} is the observation resulted from a_t
            assert actions.dim() == 2, f"Got {actions.shape}"
            actions = self.action_emb(actions)  # (b t d)

            x = torch.stack([actions, *list(x.values())], dim=2).flatten(1, 2)
            c = 2
        else:
            x = torch.stack([*list(x.values())], dim=-1).sum(dim=-1)
            c = 1

        if pad_mask is not None:
            # assume pad right / suffix.
            x = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths=pad_mask.sum(dim=1).cpu() * c,
                batch_first=True,
                enforce_sorted=False
            )
            
        x, state = self.model(x, state)

        if pad_mask is not None:
            x = nn.utils.rnn.pad_packed_sequence(
                x,
                batch_first=True
            )[0]

        if not self.config.ignore_actions:
            x = x[:, 1::2]

        return x, state
