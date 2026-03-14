import numpy as np
from typing import ClassVar, Final, Annotated
import uuid

from ..initialisers import Initialiser, Standard_Normal
from ..parameter import Parameter
from .._numpy_f32_constants import ZERO
from .._exceptions import CacheEmptyError
from .._typealiases import Tensor


class ReLU:
    trainable: Annotated[ClassVar[bool], Final] = False
    __slots__ = ('cache', 'parameters', 'uid')

    def __init__(self) -> None:
        self.cache: Tensor | None = None

        self.parameters: dict[str, Parameter[Tensor]] = dict()

        self.uid: Final[uuid.UUID] = uuid.uuid4()

    def initialise(self, initialiser_type: type[Initialiser]=Standard_Normal) -> None:
        pass

    def forward(self, z: Tensor, /) -> Tensor:
        self.cache = z.copy()

        U: Tensor = np.maximum(z, ZERO())

        return U

    def backward(self, dldu: Tensor, /) -> Tensor:
        if self.cache is None:
            raise CacheEmptyError

        Z: Tensor = self.cache

        self.cache = None

        DLDZ: Tensor = np.multiply(dldu, Z > 0, dtype=np.float32)

        return DLDZ
