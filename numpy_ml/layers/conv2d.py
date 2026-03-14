import numpy as np
from numba import njit, prange
from typing import ClassVar, Final, Annotated
import uuid

from ._utils_convpool import init_empty_2d_padded_output, pad_2d_input, remove_2d_pads
from ..initialisers import Initialiser, Standard_Normal
from ..parameter import Parameter
from .._typealiases import (
    Tensor,
    R4Tensor,
    R3Tensor,
    Vector,
    PadMode,
)
from .._exceptions import CacheEmptyError

@njit(parallel=True)
def f(
    z_padded: R4Tensor,
    weights: R4Tensor,
    u: R4Tensor,
    strides: tuple[int, int]
) -> R4Tensor:
    OUT_CHANNELS, _, KERN_HEIGHT, KERN_WIDTH = weights.shape

    M, _, U_HEIGHT, U_WIDTH = u.shape

    for out_channel in prange(OUT_CHANNELS):
        kern: R3Tensor = weights[out_channel]

        for h in range(U_HEIGHT):
            h_start: int = h * strides[0]
            h_slice = slice(h_start, h_start + KERN_HEIGHT)

            for w in range(U_WIDTH):
                w_start: int = w * strides[1]
                w_slice = slice(w_start, w_start + KERN_WIDTH)

                for m in range(M):
                    rf: R3Tensor = z_padded[m, :, h_slice, w_slice]
                    # cross-correlation instead of true convolution
                    u[m, out_channel, h, w] = np.sum(rf * kern)

    return u

@njit(parallel=True)
def chain_rule(
    dldu: R4Tensor,
    z_padded: R4Tensor,
    weights: R4Tensor,
    strides: tuple[int, int]
) -> tuple[R4Tensor, R4Tensor, Vector]:
    OUT_CHANNELS, _, KERN_HEIGHT, KERN_WIDTH = weights.shape

    M, _, DLDU_HEIGHT, DLDU_WIDTH = dldu.shape

    dudz_padded: R4Tensor = np.zeros_like(z_padded, np.float32)

    dldweights: R4Tensor = np.zeros_like(weights, np.float32)

    dldbias: Vector = np.zeros((OUT_CHANNELS,), np.float32)

    # Decided not to prange as accumulating derivative iteratively => memory safety
    for out_channel in prange(OUT_CHANNELS):
        kern: R3Tensor = weights[out_channel]

        for h in range(DLDU_HEIGHT):
            h_start: int = h * strides[0]
            h_slice = slice(h_start, h_start + KERN_HEIGHT)

            for w in range(DLDU_WIDTH):
                w_start: int = w * strides[1]
                w_slice = slice(w_start, w_start + KERN_WIDTH)

                for m in range(M):
                    dldu_rf: np.float32 = dldu[m, out_channel, h, w]
                    zpad_rf = z_padded[m, :, h_slice, w_slice]

                    # accumulate grads
                    dudz_padded[m, :, h_slice, w_slice] += kern * dldu_rf
                    dldweights[out_channel] += zpad_rf * dldu_rf
                    dldbias[out_channel] += dldu_rf

    return dudz_padded, dldweights, dldbias


class Conv2D:
    trainable: Annotated[ClassVar[bool], Final] = True
    __slots__ = ('cache', 'parameters', 'uid',
                 'out_channels', 'in_channels', 'kern_height', 'kern_width',
                 'strides', 'mode')

    def __init__(
        self,
        out_channels: int,
        in_channels: int,
        kern_height: int,
        kern_width: int,
        strides: tuple[int, int]=(1, 1),
        mode: PadMode='valid'
    ) -> None:
        self.cache: R4Tensor | None = None

        self.parameters: dict[str, Parameter[Tensor]] = dict()

        self.uid: Final[uuid.UUID] = uuid.uuid4()

        self.out_channels: Final[int] = out_channels

        self.in_channels: Final[int] = in_channels

        self.kern_height: Final[int] = kern_height

        self.kern_width: Final[int] = kern_width

        self.strides: Final[tuple[int, int]] = strides

        self.mode: Final[PadMode] = mode

    def initialise(self, initialiser_type: type[Initialiser]=Standard_Normal) -> None:
        fan_in: int =  self.in_channels * self.kern_height * self.kern_width
        fan_out: int = self.out_channels * self.kern_height * self.kern_width

        # weights initialised by intialiser_type
        w: Parameter[R4Tensor] = Parameter(
            name='weights',
            trainable=True,
            shape=(self.out_channels, self.in_channels, self.kern_height, self.kern_width),
            layer_id=self.uid
        )
        w.values = initialiser_type.initialise(w.values, fan_in=fan_in, fan_out=fan_out)

        # b.values always initialised with zeros independent of initialiser
        b: Parameter[Vector] = Parameter(
            name='bias',
            trainable=True,
            shape=(self.out_channels,),
            layer_id=self.uid
        )

        self.parameters = dict()
        self.parameters['weights'] = w
        self.parameters['bias'] = b

    def forward(self, z: R4Tensor, /) -> R4Tensor:
        self.cache = z.copy()

        W: R4Tensor = self.parameters['weights'].values

        B: Vector = self.parameters['bias'].values

        u: R4Tensor = init_empty_2d_padded_output(z, W.shape, self.strides, self.mode)

        Z_PADDED: R4Tensor = pad_2d_input(z, u, W, self.strides)

        u = f(Z_PADDED, W, u, self.strides)

        u += B[np.newaxis, :, np.newaxis, np.newaxis]

        return u

    def backward(self, dldu: R4Tensor, /) -> R4Tensor:
        if self.cache is None:
            raise CacheEmptyError

        Z: R4Tensor = self.cache

        self.cache = None

        W: R4Tensor = self.parameters['weights'].values

        Z_PADDED: R4Tensor = pad_2d_input(Z, dldu, W, self.strides)

        dudz_padded, dldweights, dldbias = chain_rule(dldu, Z_PADDED, W, self.strides)

        dudz: R4Tensor = remove_2d_pads(Z.shape, dudz_padded)

        self.parameters['weights'].grad = dldweights

        self.parameters['bias'].grad = dldbias

        return dudz
