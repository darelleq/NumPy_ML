import numpy as np

from ..networks import NeuralNetwork
from .._typealiases import Tensor
from .._exceptions import CacheEmptyError

class SGD:
    """ Stochastic Gradient Descent """
    __slots__ = ('network', 'lr', 't')

    def __init__(self, network: NeuralNetwork, lr: float=0.001, *, t: int = 0) -> None:
        if lr <= 0:
            raise ValueError("Learning rate must be non-zero positive float")

        if t < 0:
            raise ValueError("SGD.t must be zero or positive int")

        self.network: NeuralNetwork = network

        self.lr: np.float32  = np.float32(lr)

        self.t: int = t

    def step(self) -> None:
        self.t += 1

        optiminfo: str = f"[SGD t={self.t}] "

        for depth, layer in enumerate(self.network.topology):

            layerinfo: str = f"[Layer {depth}, type={layer.__class__.__name__}] "

            if layer.trainable is False: # non-trainable layers have empty parameters dicts

                continue

            for param in layer.parameters.values():
                paraminfo: str = f"[Parameter {param.name}, shape={param.shape})"

                if param.grad is None:
                    raise CacheEmptyError(
                        f"{optiminfo}{layerinfo}{paraminfo} Parameter.grad is None\n"
                        f'repr(parameter): \n{repr(param)}'
                    )

                grad: Tensor = param.grad

                delta: Tensor = np.multiply(self.lr, grad, out=grad)

                param.values = np.subtract(param.values, delta, dtype=np.float32)

                param.grad = None
