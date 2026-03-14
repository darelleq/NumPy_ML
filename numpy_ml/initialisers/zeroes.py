import numpy as np
from typing import ClassVar, cast

from .initialiser import Initialiser
from .._typealiases import Tensor
from .._numpy_f32_constants import ZERO


class Zeroes:
    rng: ClassVar[np.random.Generator] = Initialiser.rng

    @classmethod
    def initialise[T: Tensor](
        cls,
        tensor: T,
        /,
        fan_in: int | None=None,
        fan_out: int | None=None
    ) -> T:

        return cast(
            T,
            np.full(shape=tensor.shape, fill_value=ZERO(), dtype=np.float32)
        )
