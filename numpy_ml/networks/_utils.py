from typing import Final, NoReturn
from string import ascii_letters

from ..layers import Layer
from ..initialisers import (
    Initialiser,
    Xavier_Normal,
    Kaiming_Normal,
    Standard_Normal
)

class Default_Initialisers:
    ReLU: Final[type[Initialiser]] = Kaiming_Normal
    LeakyReLU: Final[type[Initialiser]] = Kaiming_Normal
    Sigmoid: Final[type[Initialiser]] = Xavier_Normal
    Softmax: Final[type[Initialiser]] = Xavier_Normal
    Default: Final[type[Initialiser]] = Standard_Normal
    Dense: Final[type[Initialiser]] = Standard_Normal
    Conv2D: Final[type[Initialiser]] = Standard_Normal
    MaxPool2D: Final[type[Initialiser]] = Standard_Normal
    AvgPool2D: Final[type[Initialiser]] = Standard_Normal
    BatchNorm1D: Final[type[Initialiser]] = Standard_Normal
    BatchNorm2D: Final[type[Initialiser]] = Standard_Normal

    def __init__(self) -> NoReturn:
        raise Exception(f"{self.__class__.__name__} utility class not for instantiation")

    @classmethod
    def get(cls, key: type[Layer] | str) -> type[Initialiser]:
        if isinstance(key, type):
            key = key.__name__

        try:
            ini_type: type[Initialiser] = getattr(Default_Initialisers, key)

        except AttributeError as e:
            valid_names: list[str] = list(vars(cls))
            for i in range(len(valid_names)):
                if valid_names[i][0] not in ascii_letters:
                    valid_names.pop(i)
            e.args = e.args + (f'valid keys are: {valid_names}',)
            raise e

        return ini_type
