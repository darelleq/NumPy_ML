import numpy as np
from typing import Protocol

from ..networks import NeuralNetwork

class Optimiser(Protocol):
    __slots__ = ('lr', 'network', 't')
    lr: np.float32
    network: NeuralNetwork
    t: int

    def step(self) -> None:
        ...
