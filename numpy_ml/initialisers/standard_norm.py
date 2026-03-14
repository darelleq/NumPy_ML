import numpy as np
from typing import ClassVar, cast

from .initialiser import Initialiser
from .._typealiases import Tensor


class Standard_Normal:
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
            cls.rng.standard_normal(size=tensor.shape, dtype=np.float32)
        )
