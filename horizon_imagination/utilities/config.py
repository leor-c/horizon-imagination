from abc import ABC, abstractmethod, ABCMeta
from typing import Type, TYPE_CHECKING
from dataclasses import dataclass, is_dataclass, replace, field


def config_class(cls=None, **kwargs):
    def wrap(cls_):
        return dataclass(cls_, kw_only=True, **kwargs)

    return wrap(cls) if cls is not None else wrap


class AutoDataclassMeta(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # Set class-level property _target_cls before dataclass wraps it
        cls._target_cls = None

        cls = dataclass(cls, kw_only=True)
        return cls


class BaseConfig(metaclass=AutoDataclassMeta):
    # _target_cls: Type = None  # dynamically set by Configurable

    def make_instance(self):
        if self._target_cls is None:
            raise TypeError("No target class associated with this config.")
        return self._target_cls(self)
    
    def clone(self, **overrides):
        """
        Return a shallow copy of this config.
        You can override fields via keyword arguments.
        """
        return replace(self, **overrides)


class Configurable:
    Config: Type[BaseConfig]

    def __init_subclass__(cls):
        super().__init_subclass__()

        if not hasattr(cls, "Config"):
            raise TypeError(f"{cls.__name__} must define an inner class `Config`.")

        config_cls = cls.Config

        if not isinstance(config_cls, type) or not issubclass(config_cls, BaseConfig):
            raise TypeError(f"{cls.__name__}.Config must be a subclass of BaseConfig.")

        # Bind the config to its target class
        config_cls._target_cls = cls

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.config)})"


