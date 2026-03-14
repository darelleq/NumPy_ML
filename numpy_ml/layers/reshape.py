import numpy as np
from itertools import cycle
import uuid
from typing import ClassVar, Final, Annotated

from .._exceptions import CacheEmptyError
from .._typealiases import Tensor, Shape

def validate_shape(shape: Shape):
    if not isinstance(shape, tuple):
        raise TypeError(f"Expected Shape tuple, got {type(shape)}")

    if not all([isinstance(i, int)] for i in shape):
        raise TypeError(f"Expected Shape tuple of ints, got {shape}")

    n_neg1: int = shape.count(-1)
    if n_neg1 > 1:
        raise ValueError(f"Reshaping Shape: {shape} can only have ONE -1, got {n_neg1}")

def ambig_shapes_match(expected: Shape, actual: Shape) -> bool:
    neg1_index: int = expected.index(-1)

    e_list: list[int] = list(expected)
    a_list: list[int] = list(actual)

    e_list.remove(-1)
    a_list.pop(neg1_index)

    return e_list == a_list

class Reshape:
    trainable: Annotated[ClassVar[bool], Final] = False
    __slots__ = ('cache', 'parameters', 'uid',
                 'from_shape', 'to_shape', 'auto_reshape',
                 '_error_check_schedule')

    def __init__(self, from_shape: Shape, to_shape: Shape) -> None:
        validate_shape(from_shape)

        validate_shape(to_shape)

        self.from_shape: Shape = from_shape

        self.to_shape: Shape = to_shape

        # auto_reshape mode is an internal state for automatic parsing of m axis
        self.auto_reshape = from_shape.count(-1) > 0 or to_shape.count(-1) > 0

        self.cache = None

        self.parameters = dict()

        self.uid = uuid.uuid4()

        # Error check every 5th iteration
        self._error_check_schedule: cycle[int] = cycle(range(0, 10))

    def forward(self, z: Tensor, /) -> Tensor:
        self.cache = z.copy() # not required -> for Layer behavioural consistency

        if self.auto_reshape is False:
            if z.shape != self.from_shape:
                raise ValueError(
                    f"Input Shape {z.shape} mismatch expected {self.from_shape}. "
                    f"Error raised as hard-defined shapes provided & auto_reshape is False"
                )
            U: Tensor = np.reshape(z, self.to_shape)

            return U

        if next(self._error_check_schedule) == 0:
            if not ambig_shapes_match(expected=self.from_shape, actual=z.shape):
                raise ValueError(
                    "Unable to automatically infer forward shape. More than one axis "
                    f"mismatches: expected={self.from_shape}, received={z.shape}"
                )

        U = np.reshape(z, self.to_shape)

        return U

    def backward(self, dldu: Tensor, /) -> Tensor:
        if self.cache is None: # not required -> for Layer behavioural consistency
            raise CacheEmptyError
        self.cache = None

        if self.auto_reshape is False:
            if dldu.shape != self.to_shape:
                raise ValueError(
                    f"Input Shape {dldu.shape} mismatch expected {self.to_shape}. "
                    f"Error raised as hard-defined shapes provided & auto_reshape is False"
                )
            DLDX: Tensor = np.reshape(dldu, self.from_shape)

            return DLDX

        if next(self._error_check_schedule) == 0:
            if not ambig_shapes_match(expected=self.to_shape, actual=dldu.shape):
                raise ValueError(
                    "Unable to automatically infer backward shape. More than one axis "
                    f"mismatches: expected={self.to_shape}, received={dldu.shape}"
                )

        DLDX =  np.reshape(dldu, self.from_shape)

        return DLDX
