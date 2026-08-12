from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from tensordict.tensordict import TensorDict

from horizon_imagination.utilities.config import Configurable, BaseConfig, dataclass
from horizon_imagination.modules.regression import TwoHotRegressionHead
from horizon_imagination.modules.lightweight_seq_model import LightweightSeqModel
from horizon_imagination.modules.conv_seq_model import ConvSeqModel


class RewardDoneModel(nn.Module, Configurable):
    @dataclass
    class Config(BaseConfig):
        backbone_config: Union[LightweightSeqModel.Config, ConvSeqModel.Config]
        num_bins: int = 129
        # Bin range in *symlog* space. Deliberately the same wide default the critic
        # uses: it is generous for MuJoCo's per-step rewards (which occupy only a
        # handful of bins) but has to also cover ALE, where `SignRewardWrapper` gives
        # rewards in {-1, 0, 1}, and Craftium. Unused bins simply learn low logits.
        v_min: float = -20.0
        v_max: float = 20.0

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.backbone = config.backbone_config.make_instance()

        latent_dim = config.backbone_config.latent_dim
        device = config.backbone_config.device
        dtype = config.backbone_config.dtype
        self.reward_head = nn.Sequential(
            nn.Identity()
        )
        
        # Two-hot rather than scalar MSE, for the same reasons as the critic -- plus one
        # specific to imagination: the reward is upstream of every lambda-return, so a
        # drifting scalar head would feed `sym_exp(drift)` into the advantage and the
        # critic target alike. A distribution over fixed bins is bounded by construction
        # and cannot do that.
        self.regression_head = TwoHotRegressionHead(
            in_features=latent_dim,
            num_bins=config.num_bins,
            v_min=config.v_min,
            v_max=config.v_max,
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

