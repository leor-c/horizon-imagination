import torch
import torch.nn as nn
from torch import Tensor
from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.modules.regression import RegressionHead
from horizon_imagination.modules.lightweight_seq_model import LightweightSeqModel


class RewardDoneModel(nn.Module, Configurable):
    @dataclass
    class Config(BaseConfig):
        backbone_config: LightweightSeqModel.Config
        hl_gauss_num_bins: int = 129

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.backbone = LightweightSeqModel(config.backbone_config)

        latent_dim = config.backbone_config.latent_dim
        device = config.backbone_config.device
        dtype = config.backbone_config.dtype
        self.reward_head = nn.Sequential(
            nn.Identity()
        )
        
        self.regression_head = RegressionHead(
            in_features=latent_dim,
            sym_log_normalize=True,
            sym_exp_order=1,
            device=device,
            dtype=dtype
        )

        self.done_head = nn.Sequential(
            nn.Linear(
                in_features=latent_dim, 
                out_features=2,
                device=device,
                dtype=dtype
            ),
        )

    def forward(self, actions, obs, state=None) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        x, state = self.backbone(actions, obs, state)
        reward, reward_logits = self.regression_head(self.reward_head(x))
        done_logits = self.done_head(x)
        done_probs = torch.softmax(done_logits, dim=-1)

        return reward, done_probs, state

    def training_step(self, actions, obs: TensorDict, rewards, dones, mask=None) -> tuple[Tensor, Tensor, dict]:
        x, state = self.backbone(actions, obs, None)

        if mask is not None:
            x = x[torch.where(mask)]
            rewards = rewards[torch.where(mask)]
            dones = dones[torch.where(mask)]

        reward_pred = self.reward_head(x).flatten(0, -2)
        reward_pred, reward_logits = self.regression_head(reward_pred)
        done_logits = self.done_head(x).flatten(0, -2)
        
        reward_loss = self.regression_head.compute_loss(
            reward_logits,
            rewards.flatten()
        )
        with torch.no_grad():
            reward_l1 = nn.functional.l1_loss(reward_pred.flatten(), rewards.flatten())

        done_loss = nn.functional.cross_entropy(done_logits, dones.flatten().long())

        info = {
            'reward_l1': reward_l1
        }

        return reward_loss, done_loss, info

