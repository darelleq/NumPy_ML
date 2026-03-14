import numpy as np
from typing import ClassVar, Protocol

from .._typealiases import Tensor


class Initialiser(Protocol):
    rng: ClassVar[np.random.Generator] = np.random.default_rng()

    @classmethod
    def initialise[T: Tensor](
        cls,
        tensor: T,
        /,
        fan_in: int | None=None,
        fan_out: int | None=None
    ) -> T:
        ...
