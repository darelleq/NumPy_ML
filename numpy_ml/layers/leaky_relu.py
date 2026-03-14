import numpy as np
from typing import ClassVar, Final, Annotated
import uuid

from ..initialisers import Initialiser, Standard_Normal
from ..parameter import Parameter
from .._exceptions import CacheEmptyError
from .._typealiases import Tensor


class LeakyReLU:
    trainable: Annotated[ClassVar[bool], Final] = False
    __slots__ = ('alpha', 'cache', 'parameters', 'uid')

    def __init__(self, alpha: float=0.1) -> None:
        self.alpha: np.float32 = np.float32(alpha)

        self.cache: Tensor | None = None

        self.parameters: dict[str, Parameter[Tensor]] = dict()

        self.uid: Final[uuid.UUID] = uuid.uuid4()

    def initialise(self, initialiser_type: type[Initialiser]=Standard_Normal) -> None:
        pass

    def forward(self, z: Tensor, /) -> Tensor:
        self.cache = z.copy()

        U: Tensor = np.where(z < 0, z, self.alpha * z)

        return U

    def backward(self, dldu: Tensor, /) -> Tensor:
        if self.cache is None:
            raise CacheEmptyError

        Z: Tensor = self.cache

        self.cache = None

        DLDZ: Tensor = np.where(Z < 0, dldu, self.alpha * dldu)

        return DLDZ
