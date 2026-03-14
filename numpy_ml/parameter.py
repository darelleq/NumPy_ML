import numpy as np
import uuid
from typing import Final, cast

from ._typealiases import Tensor, Shape

__all__ = [
    "Parameter",
]

class Parameter[T: Tensor]:
    __slots__ = ('name', 'trainable', 'values', 'grad', 'layer_id', 'uid')

    def __init__(
        self,
        name: str,
        trainable: bool,
        layer_id: uuid.UUID,
        shape: Shape
    ) -> None:
        self.name: Final[str] = name

        self.trainable: Final[bool] = trainable

        self.layer_id: uuid.UUID = layer_id

        self.uid: uuid.UUID = uuid.uuid4()

        self.values: T = cast(T, np.zeros(shape, dtype=np.float64))

        self.grad: T | None = None

    @property
    def shape(self) -> Shape:

        return self.values.shape

    def __repr__(self) -> str:

        values_isfinite = np.all(np.isfinite(self.values))

        grad_isfinite = (self.grad is not None) and (np.all(np.isfinite(self.grad)))

        return (
            f'Parameter(name={self.name}, dtype={self.values.dtype}, shape={self.shape} '
            f'grad_type={type(self.grad)}, values_isfinite={values_isfinite}, '
            f'grad_isfinite={grad_isfinite})'
        )
