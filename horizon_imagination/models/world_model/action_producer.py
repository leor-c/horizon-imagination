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
    def __call__(self, x: TensorDict, *args, **kwargs) -> tuple[Tensor, Tensor]:
        """
        starting from the "reset state", predict and sample *actions* given x, 
        the "future" (noisy) trajectory segment.
        This method should restore the policy state after or before each method call.
        The user should not invoke 'reset' multiple times for the same context.

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
    

class StablePseudoPolicyActionProducer(PseudoPolicyActionProducer):
    def __init__(self, actions: Tensor, num_actions, device=None):
        super().__init__(actions, num_actions, device)
        self.action_producer = StableDiscreteActionProducer()

    def __call__(self, x, *args, **kwargs):
        a, log_p_a = self.action_producer(self.action_distributions)
        return a, log_p_a
    

class StablePolicyActionProducer(ActionProducer):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic
        self.action_producer = StableDiscreteActionProducer()

    def __call__(self, x, *args, **kwargs):
        # TODO: support actions - generate efficiently
        action_dist, _, _ = self.actor_critic(prev_actions=None, obs=x, compute_critic=False)
        a, log_prob_a = self.action_producer(action_dist)
        return a, log_prob_a
    

class NaivePolicyActionProducer(ActionProducer):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def __call__(self, x, *args, **kwargs):
        # TODO: support actions - generate efficiently
        action_dist, _, _ = self.actor_critic(prev_actions=None, obs=x, compute_critic=False)
        a = action_dist.sample()
        log_prob_a = action_dist.log_prob(a)
        return a, log_prob_a
