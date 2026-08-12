"""Policy distributions that are not in ``torch.distributions``.

Lives under ``modules/`` rather than ``models/controller/`` on purpose: both
``models/controller/actor_critic.py`` and ``models/world_model/action_producer.py``
need ``SquashedNormal``, and ``models/controller/__init__.py`` imports ``controller.py``,
which imports the world model -- so a home under ``models/controller/`` would close an
import cycle through ``action_producer``.
"""
import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, constraints, kl_divergence
from torch.distributions.kl import register_kl
from torch.distributions.utils import _standard_normal


def tanh_log_jacobian(u: Tensor) -> Tensor:
    """``log(1 - tanh(u)^2)``, evaluated without cancellation for large ``|u|``.

    The naive form underflows to ``log(0) = -inf`` once ``|u|`` is past ~9 in fp32.
    This is the identity ``torch.distributions.TanhTransform`` uses internally.
    """
    return 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))


class SquashedNormal(Distribution):
    """A diagonal Gaussian pushed through ``tanh``, i.e. supported on ``(-1, 1)^A``.

    ``log_prob`` and ``entropy`` are shaped ``(B, T)`` -- the same contract a
    ``Categorical`` gives the actor-critic losses -- by taking ``batch_shape`` from
    ``loc.shape[:-1]`` and ``event_shape`` from ``loc.shape[-1:]``. Both shapes are
    *inferred* rather than stored, which is what lets ``_slice_dist`` index a
    ``(B, T, A)`` policy down to ``(n, A)`` and still get ``(n,)``-shaped log-probs.

    Why squash at all: the entropy of an *unbounded* Gaussian is
    ``sum(log sigma) + const``, which grows without limit, so an entropy bonus
    ``-w * H`` contributes a constant ``-w`` gradient to every ``log_std`` that never
    vanishes and drives sigma to whatever clamp exists. Squashing bounds the entropy
    above by ``A * log 2`` (uniform on the box) and makes it *decrease* once tanh
    starts saturating, so the bonus has a finite maximizer near sigma ~ 1.

    The bounds are enforced here, by construction, rather than by clipping at the
    environment boundary -- which is what keeps imagined actions inside the box the
    world model was trained on.
    """

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.interval(-1.0, 1.0)
    has_rsample = True

    def __init__(self, loc: Tensor, scale: Tensor, validate_args: bool = False):
        self.base_dist = Independent(Normal(loc, scale), 1)
        super().__init__(
            batch_shape=loc.shape[:-1],
            event_shape=loc.shape[-1:],
            validate_args=validate_args,
        )

    @property
    def loc(self) -> Tensor:
        return self.base_dist.base_dist.loc

    @property
    def scale(self) -> Tensor:
        return self.base_dist.base_dist.scale

    @property
    def mode(self) -> Tensor:
        return torch.tanh(self.loc)

    def expand(self, batch_shape, _instance=None):
        shape = torch.Size(batch_shape) + self.event_shape
        return SquashedNormal(
            self.loc.expand(shape), self.scale.expand(shape), validate_args=False
        )

    def rsample(self, sample_shape=torch.Size()) -> Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        return torch.tanh(self.loc + self.scale * eps)

    def log_prob_from_pre_tanh(self, u: Tensor) -> Tensor:
        """Log-density of ``tanh(u)``, given the *pre-squash* point ``u``.

        Preferred over `log_prob`: the caller that drew the sample already holds ``u``,
        so this avoids an ``atanh`` round-trip that is both lossy and unbounded near
        the box boundary.
        """
        return self.base_dist.log_prob(u) - tanh_log_jacobian(u).sum(-1)

    def log_prob(self, value: Tensor) -> Tensor:
        """Log-density at a squashed action. Lossy at the boundary -- see above."""
        limit = 1.0 - 1e-6
        return self.log_prob_from_pre_tanh(torch.atanh(value.clamp(-limit, limit)))

    def sample_with_log_prob(self) -> tuple[Tensor, Tensor]:
        """A detached action plus its log-density, which still carries policy gradient.

        The action must be a constant for the REINFORCE estimator ``-log pi(a) * adv``
        to be valid, so it is detached; ``loc``/``scale`` inside ``log_prob`` are not.
        """
        with torch.no_grad():
            eps = _standard_normal(
                self.loc.shape, dtype=self.loc.dtype, device=self.loc.device
            )
            u = self.loc + self.scale * eps
        return torch.tanh(u), self.log_prob_from_pre_tanh(u)

    def entropy(self) -> Tensor:
        """Single-sample estimate of ``H(a) = H(u) + E_u[log|da/du|]``.

        The Gaussian half is closed form, so only the tanh correction is sampled --
        this is the same estimator as ``-log_prob(rsample())`` but with the ``eps^2/2``
        noise integrated out analytically.

        The sample is deliberately *not* detached: the gradient flowing through the
        correction term is precisely what bounds the entropy bonus, and detaching it
        would restore the unbounded ``d/d log_sigma = +1`` of a bare Gaussian.
        """
        eps = _standard_normal(
            self.loc.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        u = self.loc + self.scale * eps
        return self.base_dist.entropy() + tanh_log_jacobian(u).sum(-1)


@register_kl(SquashedNormal, SquashedNormal)
def _kl_squashed_normal(p: SquashedNormal, q: SquashedNormal) -> Tensor:
    """Exact, not an approximation: ``tanh`` is a diffeomorphism and the KL divergence
    is invariant under an invertible change of variables, so the Jacobian terms cancel
    and the squashed KL equals the KL of the underlying Gaussians."""
    return kl_divergence(p.base_dist, q.base_dist)


def sample_with_log_prob(dist: Distribution) -> tuple[Tensor, Tensor]:
    """``(action, log_prob)`` for any policy family, avoiding ``atanh`` when possible."""
    if isinstance(dist, SquashedNormal):
        return dist.sample_with_log_prob()

    action = dist.sample()
    return action, dist.log_prob(action)
