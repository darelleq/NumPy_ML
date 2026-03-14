import numpy as np
from numpy.typing import NDArray
from typing import Any

from .shapes import (
    Shape0D,
    Shape1D,
    Shape2D,
    Shape3D,
    Shape4D,
    Shape5D
)
from ._numpy_datatypes import (
    F32,
    F64,
    Int32,
    Int64,
    UInt8,
    UInt16,
    NPBool,
)

# Standard Tensor types are float32
type Tensor = np.ndarray[Any, F32] | NDArray[np.float32]
type Vector = np.ndarray[Shape1D, F32]
type Matrix = np.ndarray[Shape2D, F32]
type R3Tensor = np.ndarray[Shape3D, F32]
type R4Tensor = np.ndarray[Shape4D, F32]
type R5Tensor = np.ndarray[Shape5D, F32]

type F64_Tensor = np.ndarray[Any, F64]
type F64_Vector = np.ndarray[Shape1D, F64]
type F64_Matrix = np.ndarray[Shape2D, F64]
type F64_R3Tensor = np.ndarray[Shape3D, F64]
type F64_R4Tensor = np.ndarray[Shape4D, F64]
type F64_R5Tensor = np.ndarray[Shape5D, F64]

# Int Tensors
type Int32_Tensor = np.ndarray[Any, Int32]
type Int32_Vector = np.ndarray[Shape1D, Int32]
type Int32_Matrix = np.ndarray[Shape2D, Int32]
type Int32_R3Tensor = np.ndarray[Shape3D, Int32]
type Int32_R4Tensor = np.ndarray[Shape4D, Int32]
type Int32_R5Tensor = np.ndarray[Shape5D, Int32]

type Int64_Tensor = np.ndarray[Any, Int64]
type Int64_Vector = np.ndarray[Shape1D, Int64]
type Int64_Matrix = np.ndarray[Shape2D, Int64]
type Int64_R3Tensor = np.ndarray[Shape3D, Int64]
type Int64_R4Tensor = np.ndarray[Shape4D, Int64]
type Int64_R5Tensor = np.ndarray[Shape5D, Int64]

# UInt Tensors
type UInt8_Tensor = np.ndarray[Any, UInt8]
type UInt8_Vector = np.ndarray[Shape1D, UInt8]
type UInt8_Matrix = np.ndarray[Shape2D, UInt8]
type UInt8_R3Tensor = np.ndarray[Shape3D, UInt8]
type UInt8_R4Tensor = np.ndarray[Shape4D, UInt8]
type UInt8_R5Tensor = np.ndarray[Shape5D, UInt8]

type UInt16_Tensor = np.ndarray[Any, UInt16]
type UInt16_Vector = np.ndarray[Shape1D, UInt16]
type UInt16_Matrix = np.ndarray[Shape2D, UInt16]
type UInt16_R3Tensor = np.ndarray[Shape3D, UInt16]
type UInt16_R4Tensor = np.ndarray[Shape4D, UInt16]
type UInt16_R5Tensor = np.ndarray[Shape5D, UInt16]

# Bool_ Tensors
type Mask_Tensor = np.ndarray[Any, NPBool]
type Mask_Vector = np.ndarray[Shape1D, NPBool]
type Mask_Matrix = np.ndarray[Shape2D, NPBool]
type Mask_R3Tensor = np.ndarray[Shape3D, NPBool]
type Mask_R4Tensor = np.ndarray[Shape4D, NPBool]
type Mask_R5Tensor = np.ndarray[Shape5D, NPBool]

# Special case -- added for consistency but UNUSED
#   - Type aliases for scalars wrapped within an array
#   - Different from naked scalars
type ScalarArray = np.ndarray[Shape0D, F32]
type F64_ScalarArray = np.ndarray[Shape0D, F64]
type Int32_ScalarArray = np.ndarray[Shape0D, Int32]
type Int64_ScalarArray = np.ndarray[Shape0D, Int64]
type UInt8_ScalarArray = np.ndarray[Shape0D, UInt8]
type UInt16_ScalarArray = np.ndarray[Shape0D, UInt16]
type Mask_ScalarArray = np.ndarray[Shape0D, NPBool]
