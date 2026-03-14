from collections.abc import Iterator
from typing import Protocol

from ..layers import Layer
from .._typealiases import Tensor

class NetworkTopology(Protocol):
    """ For the implementation of how to traverse the network's topological graph """
    def __iter__(self) -> Iterator[Layer]:
        ...

    def __getitem__(self, key: int, /) -> Layer:
        ...

    def __len__(self) -> int:
        ...


class NeuralNetwork(Protocol):
    topology: NetworkTopology

    def forward(self, x: Tensor, /) -> Tensor:
        ...

    def backward(self, dldu: Tensor, /) -> Tensor:
        ...

    def flush(self, cache: bool=True, grad: bool=True) -> None:
        ...

    def initialise(self) -> None:
        ...
