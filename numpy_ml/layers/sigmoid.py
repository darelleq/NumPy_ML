import numpy as np
from typing import ClassVar, Final, Annotated
import uuid

from ..initialisers import Initialiser, Standard_Normal
from ..parameter import Parameter
from .._numpy_f32_constants import ONE
from .._exceptions import CacheEmptyError
from .._typealiases import Tensor


class Sigmoid:
    trainable: Annotated[ClassVar[bool], Final] = False
    __slots__ = ('cache', 'parameters', 'uid')

    def __init__(self) -> None:
        self.cache: Tensor | None = None

        self.parameters: dict[str, Parameter[Tensor]] = dict()

        self.uid: Final[uuid.UUID] = uuid.uuid4()

    def initialise(self, initialiser_type: type[Initialiser]=Standard_Normal) -> None:
        pass

    def forward(self, z: Tensor, /) -> Tensor:
        U: Tensor = np.reciprocal(ONE() + np.exp(-z))

        self.cache = U.copy()

        return U

    def backward(self, dldu: Tensor, /) -> Tensor:
        if self.cache is None:
            raise CacheEmptyError

        U: Tensor = self.cache

        self.cache = None

        dfdz = np.multiply(U, ONE() - U)
        DLDZ: Tensor = np.multiply(dfdz, dldu, out=dfdz)

        return DLDZ
