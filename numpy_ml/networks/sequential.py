import numpy as np
from numpy.random import PCG64
from collections import deque
from collections.abc import Sequence, MutableSequence

from ._utils import Default_Initialisers
from .neural_network import NetworkTopology
from ..layers import Layer
from ..initialisers import Initialiser
from .._typealiases import Tensor

class LinearReversibleTopology(deque[Layer]):
    """ A virtual subclass of the NetworkTopology Protocol """
    __slots__ = ()


class Sequential:
    __slots__ = ('topology', 'name', 'check_probability', '_check_scheduler')

    def __init__(
        self,
        layers: Sequence[Layer] | MutableSequence[Layer] = list(),
        *,
        name: str,
        check_probability: float=0.1
    ) -> None:
        self.topology: NetworkTopology = LinearReversibleTopology()

        self.topology.extend(layers)

        self.name: str = name

        assert 0 <= check_probability <= 1, "Probability must be within [0, 1]"
        self.check_probability: float = check_probability

        self._check_scheduler: np.random.Generator = np.random.Generator(PCG64(789))

    @property
    def n_layers(self) -> int:

        return len(self.topology)

    def flush(self, cache: bool=True, grad: bool=True) -> None:
        for _, layer in enumerate(self.topology):

            if cache is True:
                layer.cache = None

            if grad is True:
                for param in layer.parameters.values():
                    param.grad = None

    def initialise(self) -> None:
        n_layers: int = self.n_layers

        if n_layers == 0:

            return

        for i in range(n_layers):
            curr_layer: Layer = self.topology[i]

            try:
                next_layer: Layer = self.topology[i + 1]

            except IndexError:
                if curr_layer.trainable:
                    curr_layer.initialise()

                return

            if curr_layer.trainable is not True:

                continue

            default_init: type[Initialiser] = Default_Initialisers.get(next_layer.__class__)

            curr_layer.initialise(initialiser_type=default_init)

    def forward(self, x: Tensor) -> Tensor:
        self.flush()

        for depth, layer in enumerate(self.topology):
            layer_info: str = f"[Layer {depth}] {layer.__class__.__name__}"

            try:
                x = layer.forward(x)

            except Exception as e:
                msg: str = f"{layer_info} ERROR during FORWARD"

                for p in layer.parameters.values():
                    msg += f"\n\t{repr(p)}"

                print(msg)

                raise e

        U: Tensor = x

        if self._check_scheduler.random() < self.check_probability:
            if not np.all(np.isfinite(U)):
                raise ValueError("FINAL FORWARD ERROR: NaN/Inf in output")

        return U

    def backward(self, dldu: Tensor) -> Tensor:
        for depth, layer in reversed(tuple(enumerate(self.topology))):
            layer_info: str = f"[Layer {depth}] {layer.__class__.__name__}"

            try:
                dldu = layer.backward(dldu)

            except Exception as e:
                msg: str = f"{layer_info} ERROR during BACKWARD"

                for p in layer.parameters.values():
                    msg += f"\n\t{repr(p)}"

                print(msg)

                raise e

        DLDX: Tensor = dldu

        if self._check_scheduler.random() < self.check_probability:
            if not np.all(np.isfinite(DLDX)):
                raise ValueError("FINAL BACKWARD ERROR: NaN/Inf in output")

        return DLDX
