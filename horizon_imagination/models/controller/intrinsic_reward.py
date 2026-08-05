"""
Intrinsic exploration reward from latent reconstruction error.

For an imagined latent ``z_t``, the residual ``|| z_t - encode(decode(z_t)) ||^2`` measures
how far the world model's imagination has strayed from the (frozen) tokenizer's manifold: a
large residual means the denoiser produced a latent the tokenizer cannot represent, which is
a proxy for "novel / poorly modeled". Rewarding it pushes the policy toward under-modeled
regions of the state space.

The round-trip reuses the world model's own ``PerModalityTransform`` -- its ``.inverse``
(latent -> raw) and ``.transform`` (raw -> latent) are already a matched, key-dispatching
pair, so this needs no new encode/decode code and automatically covers every observation key.
"""
import torch
from torch import Tensor
from tensordict.tensordict import TensorDict

from horizon_imagination.models.controller.return_scaler import EMAScaler


class LatentReconstructionResidual:
    """
    Deliberately a plain class rather than an ``nn.Module``: ``PerModalityTransform`` *is* an
    ``nn.Module``, so holding it as an attribute of one would auto-register it as a submodule
    and leak the (frozen, separately optimized) tokenizer into ``Controller.parameters()`` and
    ``Controller.state_dict()``. This mirrors how ``Controller`` holds ``self.return_scaler``.

    Each observation key is normalized independently -- by an EMA of its own quantile band --
    before the per-key residuals are averaged, so that keys whose raw reconstruction error
    differs by orders of magnitude still contribute comparably. The overall magnitude relative
    to the extrinsic reward is then set separately by the caller; the two are not redundant.
    """

    def __init__(self, obs_transform, ema_decay: float = 0.005, eps: float = 1e-8,
                 truncated: bool = False):
        self.obs_transform = obs_transform
        self.ema_decay = ema_decay
        self.eps = eps
        # Advisory: transforms without a cheaper shortcut fall back to the full round-trip.
        self.truncated = truncated
        # One scaler per observation key, created on first sight of that key.
        self.scalers: dict[str, EMAScaler] = {}

    def _scaler(self, key: str) -> EMAScaler:
        if key not in self.scalers:
            self.scalers[key] = EMAScaler(decay=self.ema_decay)
        return self.scalers[key]

    @torch.no_grad()
    def __call__(self, z: TensorDict, mask: Tensor = None) -> tuple[Tensor, dict]:
        """
        z:    clean imagined latents, batch_size (B, T), one entry per observation key.
        mask: (B, T) bool, True where the step is on-trajectory. Post-termination steps are
              off-distribution and spike the residual, so they are excluded from the EMA
              update -- otherwise they inflate the scale and shrink every advantage. The
              returned residual is *not* masked: the losses already drop those steps via
              ``make_valid_mask`` and ``compute_lambda_returns`` cuts the recursion at ends.

        Returns (residual (B, T), info dict of per-key scalars for logging).
        """
        z_hat = self.obs_transform.roundtrip(z, truncated=self.truncated)

        normalized, info = [], {}
        for key in z.keys():
            # flatten(2) rather than a fixed dim tuple: image latents are (B,T,C,H,W) and
            # vector latents (B,T,D,P,1). Mean, not sum -- every latent is tanh-bounded to
            # [-1,1], so per-key means are already on a comparable scale.
            mse = (z[key] - z_hat[key]).pow(2).flatten(2).mean(-1).float()  # (B, T)

            scaler = self._scaler(key)
            # torch.quantile rejects half/bfloat16, hence the .float() above.
            observed = mse if mask is None else mse[mask]
            if observed.numel() > 0:
                scaler.update(observed)

            if scaler.estimate_low is None:
                # No statistics yet (first call saw an all-off-trajectory batch); contribute
                # nothing rather than an arbitrarily scaled value.
                normalized.append(torch.zeros_like(mse))
            else:
                # Centering matters: once the tokenizer converges the raw MSE concentrates and
                # its quantile band collapses, so an uncentered mse/scale becomes a huge
                # near-constant per-step reward -- a survival bonus fighting the termination
                # model, carrying no exploration signal. Subtracting the low estimate keeps
                # this a roughly [0, 1] "how much worse than typical" quantity.
                norm = (mse - scaler.estimate_low) / torch.clamp_min(scaler.scale, self.eps)
                normalized.append(norm)
                info[f'recon_scale/{key}'] = scaler.scale.detach()

            info[f'recon_mse/{key}'] = mse.mean().detach()
            info[f'recon_normalized/{key}'] = normalized[-1].mean().detach()

        residual = torch.stack(normalized, dim=0).mean(dim=0)
        return residual, info
