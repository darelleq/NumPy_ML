import uuid
from typing import Protocol, Annotated, Final, ClassVar

from ..parameter import Parameter
from ..initialisers import Initialiser, Standard_Normal
from .._typealiases import Tensor

class Layer(Protocol):
    trainable: Annotated[ClassVar[bool], Final]

    __slots__ = ('cache', 'parameters', 'uid')
    cache: Tensor | None
    parameters: dict[str, Parameter[Tensor]]
    uid: Final[uuid.UUID]

    def initialise(self, initialiser_type: type[Initialiser]=Standard_Normal) -> None:
        ...

    def forward(self, z: Tensor, /) -> Tensor:
        ...

    def backward(self, dldu: Tensor, /) -> Tensor:
        ...
