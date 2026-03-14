"""
This module contains commonly re-used custom exceptions.

NOT ALL exceptions are included here.
"""

class CacheEmptyError(ValueError):
    def __init__(self, message: str = "Cache empty when expected not to be") -> None:
        super().__init__(message)


class GradIsNoneError(ValueError):
    def __init__(self, message: str = "Parameter.grad is None when expected not to be") -> None:
        super().__init__(message)


class NonfiniteError(ValueError):
    def __init__(self, message: str = "np.all(np.isinstance(array) check was False") -> None:
        super().__init__(message)
