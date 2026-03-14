import numpy as np
import math
from functools import cache

from .._numpy_f32_constants import ZERO
from .._typealiases import (
    R4Tensor,
    Shape4D,
    PadMode
)

def init_empty_2d_padded_output(
    x: R4Tensor,
    kern_dims: Shape4D,
    strides: tuple[int, int],
    mode: PadMode
) -> R4Tensor:
    m, in_channels, x_height, x_width = x.shape

    out_channels, _, kern_height, kern_width = kern_dims

    u_height: int

    u_width: int

    match mode:
        case 'valid':
            u_height = max(0, math.ceil((x_height - kern_height + 1) / strides[0]))
            u_width = max(0, math.ceil((x_width - kern_width + 1) / strides[1]))

        case 'same':
            u_height = max(0, math.ceil(x_height / strides[0]))
            u_width = max(0, math.ceil(x_width / strides[1]))

        case 'full':
            u_height = max(0, math.ceil((x_height + kern_height - 1) / strides[0]))
            u_width = max(0, math.ceil((x_width + kern_width - 1) / strides[1]))

    return np.zeros(shape=(m, out_channels, u_height, u_width), dtype=np.float32)

@cache
def _calc_axis_pads(
    input_len: int,
    kern_len: int,
    output_len: int,
    stride: int,
) -> tuple[int,int]:
    """ Given the mode, calculates the padding size required on either sides of an axis.

    Returns a tuple whose first element is the pad length for the "first edge" of the
    axis, and the second for the "last edge".
        e.g.
            somelist = [1, 2, 3, 4]
            padwidth output of (2, 1)

            somelist_padded = [0, 0, 1, 2, 3, 4, 0]

    Assumes well-validated args are always valid.
    """
    ptotal: int = max(0, (output_len - 1) * stride + kern_len - input_len)

    p1: int = ptotal // 2

    p2: int = ptotal - p1

    return (p1, p2)

def pad_2d_input(x: R4Tensor, u: R4Tensor, kern: R4Tensor, strides: tuple[int, int]) -> R4Tensor:
    *_, x_height, x_width = x.shape

    *_, u_height, u_width = u.shape

    *_, kern_height, kern_width = kern.shape

    m_pads: tuple[int, int] =  (0, 0)

    in_channels_pads: tuple[int, int] = (0, 0)

    u_height_pad: tuple[int, int] = _calc_axis_pads(x_height, kern_height, u_height, strides[0])

    u_width_pad: tuple[int, int] = _calc_axis_pads(x_width, kern_width, u_width, strides[1])

    # Returns a new array (i.e. not a copy)
    return np.pad(
        x,
        (m_pads, in_channels_pads, u_height_pad, u_width_pad),
        mode='constant',
        constant_values=ZERO()
    )

def remove_2d_pads(
    orig_arr_shape: Shape4D,
    arr_padded: R4Tensor
) -> R4Tensor:
    *_, x_height, x_width = orig_arr_shape

    *_, u_height, u_width = arr_padded.shape

    height_diff: int = max(0, u_height - x_height)
    ph1: None | int = None if height_diff == 0 else height_diff // 2
    ph2: None | int = None if (ph1 is None) else ph1 - height_diff

    width_diff: int = max(0, u_width - x_width)
    pw1: None | int = None if width_diff == 0 else width_diff // 2
    pw2: None | int = None if (pw1 is None) else pw1 - width_diff

    return arr_padded[:, :, ph1:ph2, pw1:pw2]
