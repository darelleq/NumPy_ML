import numpy as np
import math
from typing import ClassVar, cast

from .initialiser import Initialiser
from .._typealiases import Tensor


class Kaiming_Normal:
    rng: ClassVar[np.random.Generator] = Initialiser.rng

    @classmethod
    def initialise[T: Tensor](
        cls,
        tensor: T,
        /,
        fan_in: int | None=None,
        fan_out: int | None=None
    ) -> T:
        if fan_in is None:
            raise ValueError("fan_in is required")

        if fan_in <= 0:
            raise ValueError("fan_in must be a positive non-zero int")

        scale: float = math.sqrt(2 / fan_in)

        return cast(
            T,
            np.astype(cls.rng.normal(scale=scale, size=tensor.shape), np.float32)
        )
