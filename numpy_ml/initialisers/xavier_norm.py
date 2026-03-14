import numpy as np
import math
from typing import ClassVar, cast

from .initialiser import Initialiser
from .._typealiases import Tensor, Shape

class Xavier_Normal:
    rng: ClassVar[np.random.Generator] = Initialiser.rng

    @classmethod
    def initialise[T: Tensor](
        cls,
        tensor: T,
        /,
        fan_in: int | None=None,
        fan_out: int | None=None
    ) -> T:
        if fan_in is None or fan_out is None:
            raise ValueError("Both fan_in and fan_out are required")

        if fan_in <= 0 or fan_out <= 0:
            raise ValueError("fan_in and fan_out must be positive non-zero int")

        scale: float = math.sqrt(2 / (fan_in + fan_out))

        shape: Shape = tensor.shape

        return cast(
            T,
            np.astype(cls.rng.normal(scale=scale, size=shape), np.float32)
        )
