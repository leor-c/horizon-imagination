from enum import Enum
import re

from torch import Tensor
import numpy as np


class Modality(Enum):
    image = 'image'
    vector = 'vector'
    token = 'token'  # integer
    token_2d = 'token_2d'


class ObsKey(str):
    """
    A class for enforcing a format on the keys of the observation dictionary
    and enabling easy fetching of the individual values within the format.
    """
    @classmethod
    def from_parts(cls, modality: Modality, name: str):
        if not isinstance(modality, Modality):
            raise ValueError(f"Expected modality to be an instance of ObsModality, got '{type(modality)}'.")
        
        if not isinstance(name, str) or not re.fullmatch(r"^[a-zA-Z0-9_]+$", name):
            raise ValueError(f"Invalid name argument: '{name}'.")

        formatted = f"{modality.value}|{name}"
        return cls(formatted)
    
    def __new__(cls, value):
        # Validate the string format
        matches = re.fullmatch(r"([a-z0-9_]+)\|[a-zA-Z0-9_]+", value)
        if not matches:
            raise ValueError(f"Invalid format for ObsModalityKey: {value}")
        try:
            Modality[matches.group(1)]
        except KeyError:
            raise ValueError(f'Invalid modality "{matches.group(1)}"')
        return str.__new__(cls, value)
    
    @property
    def modality(self):
        return Modality[self.split('|')[0]]
    
    @property
    def name(self):
        return self.split('|')[1]


MultiModalObs = dict[ObsKey, Tensor]
RawMultiModalObs = dict[ObsKey, np.ndarray]
