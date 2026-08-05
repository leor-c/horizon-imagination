from abc import ABC, abstractmethod


class BaseTransform(ABC):
    """
    This class is intended for transforming between representations.
    Some models require their inputs in some format.
    This class should be subclassed for every conversion between two 
    formats in both directions: X --> Z and Z --> X.

    Transforms may include learned modules.
    Transform objects could be learned as part of their containing
    models.  
    In that case, inherit from nn.Module as well.
    """

    @abstractmethod
    def transform(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def inverse(self, z, *args, **kwargs):
        pass

    def roundtrip(self, z, truncated: bool = False, *args, **kwargs):
        """Z --> X --> Z, used to measure how far z sits off the transform's manifold.

        Subclasses may override with a cheaper equivalent. ``truncated`` requests such a
        shortcut; it is advisory, and implementations without one simply ignore it.
        """
        return self.transform(self.inverse(z, *args, **kwargs), *args, **kwargs)
