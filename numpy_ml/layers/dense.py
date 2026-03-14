import numpy as np
from typing import ClassVar, Final, Annotated
import uuid

from ..initialisers import Initialiser, Standard_Normal
from ..parameter import Parameter
from .._exceptions import CacheEmptyError
from .._typealiases import Tensor, Matrix, Vector

class Dense:
    trainable: Annotated[ClassVar[bool], Final] = True
    __slots__ = ('cache', 'parameters', 'uid',
                 'in_features', 'out_features')

    def __init__(self, in_features: int, out_features: int) -> None:
        self.cache: Matrix | None = None

        self.parameters: dict[str, Parameter[Tensor]] = dict()

        self.uid: Final[uuid.UUID] = uuid.uuid4()

        self.in_features = in_features

        self.out_features = out_features


    def initialise(self, initialiser_type: type[Initialiser]=Standard_Normal) -> None:
        weights: Parameter[Matrix] = Parameter(
            name='weights',
            trainable=True,
            shape=(self.out_features, self.in_features),
            layer_id=self.uid
        )

        weights.values = initialiser_type.initialise(
            weights.values,
            fan_in=self.in_features,
            fan_out=self.out_features
        )

        self.parameters['weights'] = weights

        bias: Parameter[Vector] = Parameter(
            name='bias',
            trainable=True,
            shape=(self.out_features,),
            layer_id=self.uid
        )

        # Bias always initialised as a zeros array (so no initialiser_type.initialise)
        self.parameters['bias'] = bias

    def forward(self, z: Matrix, /) -> Matrix:
        self.cache = z.copy()

        W: Matrix = self.parameters['weights'].values

        B: Vector = self.parameters['bias'].values

        U: Matrix = np.dot(z, W.T) + B[np.newaxis, :]

        return U

    def backward(self, dldu: Matrix, /) -> Matrix:
        if self.cache is None:
            raise CacheEmptyError

        Z: Matrix = self.cache

        self.cache = None

        W: Matrix = self.parameters['weights'].values

        M: int = Z.shape[0]

        DLDZ: Matrix = np.dot(dldu, W)

        DLDW: Matrix = np.divide(np.dot(dldu.T, Z), M, dtype=np.float32)

        DLDB: Matrix = np.mean(dldu, axis=0)

        self.parameters['weights'].grad = DLDW

        self.parameters['bias'].grad = DLDB

        return DLDZ
