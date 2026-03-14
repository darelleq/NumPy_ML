import numpy as np
from numpy.typing import NDArray
import math
import warnings
from typing import ClassVar, Any, Self
from collections.abc import Iterable

from .._typealiases import Shape

def parse_batchsize(size: int | None, m: int) -> int:
    if size is None:
        size = m
        warnings.warn("Minibatcher batching all samples into one batch as batchsize was None")

    elif size < 0:
        size = m
        warnings.warn("Minibatcher batching all samples into one batch as batchsize was negative")

    elif size > m:
        raise ValueError("Mini-batch size is greater than total number of samples")

    return size


class MiniBatcher:
    rng: ClassVar[np.random.Generator] = np.random.default_rng()
    __slots__ = ('arrays', 'batchsize', 'use_residuals', '_iter_index', '_row_indexes')

    def __init__(
        self,
        arrays: Iterable[NDArray[Any]],
        /,
        *,
        batchsize: int | None,
        use_residuals: bool = False
    ) -> None:
        array_shapes: list[Shape] = [a.shape for a in arrays]

        first_shape = array_shapes[0]

        # Arrays must have same number of samples
        if all(first_shape[0] != shape[0] for shape in array_shapes):
            raise ValueError(f"Arrays must have equal number of rows, got: {array_shapes}")

        self.arrays = list(arrays)

        M: int = self.arrays[0].shape[0]

        self.batchsize = parse_batchsize(size=batchsize, m=M)

        self.use_residuals: bool = use_residuals

        self._iter_index = 0

        self._row_indexes = np.arange(M, dtype=np.int32)

    def __len__(self) -> int:
        M: int = self.arrays[0].shape[0]

        if self.use_residuals is True:

            return math.ceil(M / self.batchsize)

        return M // self.batchsize

    def __iter__(self) -> Self:
        self._iter_index = 0

        self.rng.shuffle(self._row_indexes)

        return self

    def __next__(self) -> tuple[NDArray[Any], ...]:
        i: int = self._iter_index

        sample_indices = self._row_indexes[i * self.batchsize: (i + 1) * self.batchsize]

        if sample_indices.size == 0:
            raise StopIteration

        if sample_indices.shape[0] < self.batchsize and not self.use_residuals:
            raise StopIteration

        self._iter_index += 1

        return tuple(array[sample_indices].copy() for array in self.arrays)
