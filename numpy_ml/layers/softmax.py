import numpy as np
from typing import ClassVar, Final, Annotated
import uuid

from ..initialisers import Initialiser, Standard_Normal
from ..parameter import Parameter
from .._numpy_f32_constants import EPSILON
from .._exceptions import CacheEmptyError
from .._typealiases import Tensor, Matrix


class Softmax:
    trainable: Annotated[ClassVar[bool], Final] = False
    __slots__ = ('cache', 'parameters', 'uid')

    def __init__(self) -> None:
        self.cache: Matrix | None = None

        self.parameters: dict[str, Parameter[Tensor]] = dict()

        self.uid: Final[uuid.UUID] = uuid.uuid4()

    def initialise(self, initialiser_type: type[Initialiser]=Standard_Normal) -> None:
        pass

    def forward(self, z: Matrix, /) -> Matrix:
        MAXES: Matrix = np.max(z, axis=1, keepdims=True)

        EXPZ: Matrix = np.exp(z - MAXES)

        U: Matrix = np.divide(
            EXPZ,
            np.sum(EXPZ, axis=1, keepdims=True) + EPSILON(),
            dtype=np.float32
        )

        self.cache = U.copy()

        return U

    def backward(self, dldu: Matrix, /) -> Matrix:
        if self.cache is None:
            raise CacheEmptyError

        U: Matrix = self.cache

        self.cache = None

        DLDZ: Matrix = U * (dldu - np.sum(U * dldu, axis=1, keepdims=True))

        return DLDZ
