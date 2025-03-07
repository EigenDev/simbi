from typing import Generator, Callable, Sequence, Union, Any
from numpy.typing import NDArray
import numpy as np

GeneratorTuple = tuple[
    Generator[Sequence[float], None, None],  # gas state
    Generator[float, None, None],  # Bx
    Generator[float, None, None],  # By
    Generator[float, None, None],  # Bz
]
SingleGenerator = Generator[Sequence[float], None, None]
PureHydroStateGenerator = Callable[[], SingleGenerator]
MHDStateGenerator = Callable[[], GeneratorTuple]

InitialStateType = Union[PureHydroStateGenerator, MHDStateGenerator]

PrimitiveStateFunc = (
    Callable[[], SingleGenerator]
    | Callable[
        [],
        GeneratorTuple,
    ]
)

StateGenerator = SingleGenerator | GeneratorTuple

FloatOrArray = Union[float, NDArray[np.floating[Any]]]
