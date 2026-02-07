import logging
from typing import Any, Union

import lightning as L
from lightning.pytorch.utilities import grad_norm
import torch
from torch import Tensor

from horizon_imagination.utilities.config import Configurable, BaseConfig
from horizon_imagination.models.tokenizer.cosmos.networks import (
    ContinuousImageTokenizer, DiscreteImageTokenizer
)
from horizon_imagination.models.tokenizer.cosmos.training.configs.base.net import (
    ContinuousImageTokenizerConfig, DiscreteImageTokenizerConfig
)
from horizon_imagination.models.tokenizer.cosmos.training.configs.base.loss import ColorConfig, PerceptualConfig
from horizon_imagination.models.tokenizer.cosmos.training.losses.continuous import ColorLoss, PerceptualLoss
from horizon_imagination.models.tokenizer.cosmos.training.metrics import PSNRMetric
from horizon_imagination.models.tokenizer.cosmos.training.datasets.utils import INPUT_KEY, LATENT_KEY, MASK_KEY, RECON_KEY


class OptimizerConfig(BaseConfig):
    # Default values from the original code
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


class CosmosImageTokenizer(L.LightningModule, Configurable):
    class Config(BaseConfig):
        network_cfg: Union[ContinuousImageTokenizerConfig, DiscreteImageTokenizerConfig]
        precision: torch.dtype = None
        optimizer_cfg: OptimizerConfig

        @property
        def latent_channels(self):
            if isinstance(self.network_cfg, ContinuousImageTokenizerConfig):
                return self.network_cfg.latent_channels
            elif isinstance(self.network_cfg, DiscreteImageTokenizerConfig):
                return len(self.network_cfg.levels)
            else:
                raise NotImplementedError(f'Network config type not supported {self.network_cfg}')


    def __init__(self, config: Config):
        args, kwargs = [], {}
        super().__init__(*args, **kwargs)
        self.config = config

        if isinstance(config.network_cfg, ContinuousImageTokenizerConfig):
            self.network = ContinuousImageTokenizer(**config.network_cfg.__dict__, dtype=config.precision)
        else:
            assert isinstance(config.network_cfg, DiscreteImageTokenizerConfig)
            self.network = DiscreteImageTokenizer(**config.network_cfg.__dict__, dtype=config.precision)

        self.color_loss = ColorLoss(ColorConfig())
        self.perceptual_loss = PerceptualLoss(PerceptualConfig())
        self.precision = config.precision

        self.metrics = [PSNRMetric()]

    def forward(self, x: Tensor):
        x = self._preprocess_images(x)
        res = self.network.forward(x)
        # return res.latent, self._postprocess_images(res.reconstructions)
        return self._postprocess_images(res.reconstructions)
    
    def encode(self, x):
        x = self._preprocess_images(x)
        z = self.network.encode(x)
        if isinstance(self.network, ContinuousImageTokenizer):
            z = z[0]
        else:
            indices, z, _ = z
        return z
    
    def decode(self, z):
        if isinstance(self.network, DiscreteImageTokenizer):
            z = self.network.quantizer(z)[1]
        x_hat = self.network.decode(z)
        return self._postprocess_images(x_hat)

    def training_step(self, batch, batch_idx, log_dict_fn = None):
        assert self.network.training
        batch = self._preprocess_images(batch)
        output_dict = self.network.forward(batch)
        input_images, recon_images = batch, output_dict[RECON_KEY]

        # pass loss_mask to loss computation
        inputs = {INPUT_KEY: input_images, MASK_KEY: torch.ones_like(input_images)}

        # Compute losses:
        color_loss = self.color_loss(inputs, output_dict, batch_idx)['color']
        perceptual_loss = self.perceptual_loss(inputs, output_dict, batch_idx)

        # log:
        losses = {f"tokenizer/{k}_loss": v.mean() for k, v in perceptual_loss.items()}
        losses['tokenizer/color_loss'] = color_loss.mean()
        if log_dict_fn is None:
            log_dict_fn = self.log_dict
        log_dict_fn(losses, prog_bar=True, on_epoch=True, on_step=False)

        return sum([torch.mean(v) if (v.dim() > 0) else v for v in losses.values()])
    
    # def on_before_optimizer_step(self, optimizer):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.parameters(), norm_type=2)
    #     self.log_dict(norms)
    
    def _preprocess_images(self, x):
        assert x.dtype == torch.uint8, f"got {x.dtype}"
        # Transform values from UInt8 to float in [-1, 1]
        assert x.dim() >= 4 and x.shape[-3] in [1, 3], f"got shape {x.shape}"
        x = x.to(dtype=self.precision) / 255
        x = x * 2 - 1
        return x
    
    def _postprocess_images(self, x):
        # Reverse _preprocess_images:
        # Transform values from float in [-1, 1] to UInt8 
        x = (x.clamp(-1, 1) + 1) / 2
        x = x * 255
        x = x.to(dtype=torch.uint8)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.network.parameters(),
            lr=self.config.optimizer_cfg.learning_rate,
            betas=self.config.optimizer_cfg.betas,
            eps=self.config.optimizer_cfg.eps,
            weight_decay=self.config.optimizer_cfg.weight_decay,
        )








