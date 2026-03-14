# Not all will be used OR exported: check __init__.py
# These are to annotate ndarrays types
# only useful within the scope of the _typealias module (hence module is private)
import numpy as np

type Floating = np.dtype[np.floating]
type F16 = np.dtype[np.float16]
type F32 = np.dtype[np.float32]
type F64 = np.dtype[np.float64]

type Integer = np.dtype[np.integer]
type Int8 = np.dtype[np.int8]
type Int16 = np.dtype[np.int16]
type Int32 = np.dtype[np.int32]
type Int64 = np.dtype[np.int64]

# Shouldn't really be used -- I will assume Int64
type IntP = np.dtype[np.intp]

type UInt8 = np.dtype[np.uint8]
type UInt16 = np.dtype[np.uint16]
type UInt32 = np.dtype[np.uint32]
type UInt64 = np.dtype[np.uint64]

type NPBool = np.dtype[np.bool_]
type NPObject = np.dtype[np.object_]
#
