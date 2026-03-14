"""
EXPLANATION OF MODULE:

    When these classes get created, the object at that memory address is replaced with
    respective value as a numpy.float32 object, thanks to how i've implemented __new__()
    This is a slightly hacky way for me to achieve more readable code and safe code.

    Example:
        >>> class ONE:
        ...     def __new__(cls) -> np.float32:
        ...         return np.float32(1)

        >>> abc = numpy.float32(1)

        >>> foo = ONE()

        >>> type(foo)
        <class 'numpy.float32'>

        >>> isinstance(foo, (type, ONE))
        False

        >>> foo == abc == 1
        np.True_

        >>> foo is abc
        False

Obviously, I've only implmented a few numbers (all of which are float32) that I will use
(or wanted to use) in this project.
"""
import numpy as np

__all__ = [
    "EPSILON",
    "ONE",
    "ZERO",
    "TWO",
    "SIX",
    "TEN",
]

class EPSILON:
    """ A constant of value np.float32(1e-7) """
    def __new__(cls) -> np.float32: return np.float32(1e-7)
    def __class_getitem__(cls) -> np.float32: ...


class ONE:
    """ A constant of value np.float32(ONE) """
    def __new__(cls) -> np.float32: return np.float32(1)
    def __class_getitem__(cls) -> np.float32: ...


class ZERO:
    """ A constant of value np.float32(ZERO) """
    def __new__(cls) -> np.float32: return np.float32(0)
    def __class_getitem__(cls) -> np.float32: ...


class TWO:
    """ A constant of value np.float32(TWO) """
    def __new__(cls) -> np.float32: return np.float32(2)
    def __class_getitem__(cls) -> np.float32: ...


class SIX:
    """ A constant of value np.float32(SIX) """
    def __new__(cls) -> np.float32: return np.float32(6)
    def __class_getitem__(cls) -> np.float32: ...


class TEN:
    """ A constant of value np.float32(TEN) """
    def __new__(cls) -> np.float32: return np.float32(10)
    def __class_getitem__(cls) -> np.float32: ...
