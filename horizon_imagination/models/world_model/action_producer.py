from abc import ABC, abstractmethod
from tensordict.tensordict import TensorDict
from torch import Tensor
import torch
import torch.nn.functional as F


class ActionProducer(ABC):
    """
    A helper class for implementing action sampling mechanisms during
    the diffusion denoising process at imagination.
    This class wraps a policy and generates *actions* (sampled from 
    policy output distributions) given (noisy) trajectory segments.
    """

    @abstractmethod
    def __call__(self, x: TensorDict, *args, t: Tensor = None, **kwargs) -> tuple[Tensor, Tensor]:
        """
        starting from the "reset state", predict and sample *actions* given x,
        the "future" (noisy) trajectory segment.
        This method should restore the policy state after or before each method call.
        The user should not invoke 'reset' multiple times for the same context.

        :param t: diffusion time (noise level) of x, shape (B, T). Only meaningful
        (and required by the underlying noisy actor) when `is_clean` is False.

        Return: a tuple of (actions, log_probs)
        """
        pass


class FixedActionProducer(ActionProducer):
    def __init__(self, actions: Tensor):
        super().__init__()
        self.actions = actions.clone()

    def __call__(self, x: TensorDict, *args, **kwargs):
        return self.actions.clone(), torch.zeros_like(self.actions)
    

class PseudoPolicyActionProducer(ActionProducer):
    """
    To quickly test multi-step trajectory generation, we simulate a policy 
    distribution given ground-truth actions as a mixture of one hot and uniform
    distributions.
    """
    def __init__(self, actions: Tensor, num_actions: int, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.actions = actions
        self.num_actions = num_actions
        self.action_distributions = self._make_distributions()

    def _make_distributions(self):
        one_hots = F.one_hot(self.actions, num_classes=self.num_actions)
        uniform = torch.ones_like(one_hots) / self.num_actions
        p = 0.5
        d = p * one_hots + (1-p) * uniform
        return torch.distributions.Categorical(probs=d)
    

class NaivePseudoPolicyActionProducer(PseudoPolicyActionProducer):

    def __call__(self, x, *args, **kwargs):
        actions = self.action_distributions.sample()
        return actions, self.action_distributions.probs[..., actions]
    

class StableDiscreteActionProducer(ActionProducer):
    def __init__(self, generator=None, random_permutation: bool = True):
        super().__init__()
        self.generator = generator
        self.random_permutation = random_permutation
        self.permutation = None
        self.log_omega = None

    def _compute_log_corrected_distribution(self, probs, device=None):
        assert self.permutation is not None
        order = self.permutation
        num_actions = probs.shape[-1]

        log_survivals = torch.empty_like(probs, device=device)
        log_survivals[..., order[0]] = 0.0  # = log(1)

        log_q = torch.empty_like(probs, device=device)
        log_q[..., order[0]] = torch.log(probs[..., order[0]])
        for i in range(1, num_actions - 1):
            # sum of log(1 - v_j) up to j = i-1
            log_survivals[..., order[i]] = log_survivals[..., order[i - 1]] + torch.log1p(-torch.exp(log_q[..., order[i - 1]]))
            log_q[..., order[i]] = torch.log(probs[..., order[i]]) - log_survivals[..., order[i]]

        log_q[..., order[-1]] = 0  # v_n = 1 (absorbing state)

        return log_q
    
    def _sample_omega(self, shape, device=None):
        # Add extra 'sink' dim to always capture a bin:
        return torch.rand(*shape, device=device, generator=self.generator)
    
    def __call__(self, x: torch.distributions.Categorical, *args, **kwargs):
        num_actions = x.probs.shape[-1]
        device = x.probs.device
        dtype = x.probs.dtype

        if self.permutation is None:
            if self.random_permutation:
                self.permutation = torch.randperm(num_actions, device=device, generator=self.generator)
            else:
                self.permutation = torch.arange(num_actions, device=device)
            self.log_omega = torch.log(self._sample_omega(x.probs.shape, device=device))

        log_thresholds = self._compute_log_corrected_distribution(x.probs, device=device)

        order = self.permutation
                
        a = torch.argmax((self.log_omega[..., order] <= log_thresholds[..., order]).to(dtype=dtype), dim=-1)
        a = order[a]
        return a, x.log_prob(a)
    

class StableContinuousActionProducer(ActionProducer):
    """
    The continuous counterpart of `StableDiscreteActionProducer`.

    One standard-normal draw `eps` is made on the first call and reused for the rest of
    the denoising process, so the action is re-derived at every step by the
    reparameterization `a = mean + std * eps`. Holding `eps` fixed is what makes the
    sampler *stable*: as the policy distribution drifts across denoising steps the
    action follows it continuously, instead of jumping to an unrelated point of the
    action space the way an independent draw per step would.

    The returned action is detached. The actor loss is REINFORCE
    (`-log_pi(a) * advantage`), whose estimator requires the action to be a constant:
    substituting a live `a = mean + std * eps` into `log_prob` collapses it to
    `-eps^2/2 - log(std) - c`, which has *zero* gradient w.r.t. the mean. The
    reparameterization is used here for its coupling across denoising steps, not for
    pathwise gradients -- those cannot flow anyway, since the denoiser runs under
    `torch.no_grad()`.
    """

    def __init__(self, generator=None):
        super().__init__()
        self.generator = generator
        self.eps = None

    def __call__(self, x: torch.distributions.Independent, *args, **kwargs):
        mean, std = x.base_dist.loc, x.base_dist.scale

        if self.eps is None:
            self.eps = torch.randn(
                mean.shape, device=mean.device, dtype=mean.dtype, generator=self.generator
            )

        a = (mean + std * self.eps).detach()
        return a, x.log_prob(a)


class StablePseudoPolicyActionProducer(PseudoPolicyActionProducer):
    def __init__(self, actions: Tensor, num_actions, device=None):
        super().__init__(actions, num_actions, device)
        self.action_producer = StableDiscreteActionProducer()

    def __call__(self, x, *args, **kwargs):
        a, log_p_a = self.action_producer(self.action_distributions)
        return a, log_p_a
    

def make_stable_action_producer(action_dist, generator=None) -> ActionProducer:
    """Pick the stable sampler matching the policy's output distribution."""
    if isinstance(action_dist, torch.distributions.Categorical):
        return StableDiscreteActionProducer(generator=generator)
    if isinstance(action_dist, torch.distributions.Independent):
        return StableContinuousActionProducer(generator=generator)

    raise NotImplementedError(f"No stable sampler for policy distribution {type(action_dist)}.")


class StablePolicyActionProducer(ActionProducer):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic
        # Built on the first call, from the distribution the policy actually returns:
        self.action_producer = None

    def __call__(self, x, is_clean=False, *args, t: Tensor = None, **kwargs):
        # TODO: support actions - generate efficiently
        if is_clean:
            action_dist = self.actor_critic.clean_actor(prev_actions=None, obs=x, *args, **kwargs)
        else:
            action_dist, _, _ = self.actor_critic(
                prev_actions=None, obs=x, compute_critic=False, noise_level=t, *args, **kwargs
            )
        if self.action_producer is None:
            self.action_producer = make_stable_action_producer(action_dist)
        a, log_prob_a = self.action_producer(action_dist)
        return a, log_prob_a
    

class NaivePolicyActionProducer(ActionProducer):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def __call__(self, x, is_clean=False, *args, t: Tensor = None, **kwargs):
        # TODO: support actions - generate efficiently
        if is_clean:
            action_dist = self.actor_critic.clean_actor(prev_actions=None, obs=x, *args, **kwargs)
        else:
            action_dist, _, _ = self.actor_critic(
                prev_actions=None, obs=x, compute_critic=False, noise_level=t, *args, **kwargs
            )
        a = action_dist.sample()
        log_prob_a = action_dist.log_prob(a)
        return a, log_prob_a
