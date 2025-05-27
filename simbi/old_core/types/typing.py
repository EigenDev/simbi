from typing import Generator, Callable, Sequence, Union, Any
from numpy.typing import NDArray
import numpy as np

GasStateGenerator = Generator[Sequence[float], None, None]
StaggereBFieldGenerator = Generator[float, None, None]

PureHydroStateGenerator = Callable[[], GasStateGenerator]
MHDStateGenerators = tuple[
    Callable[[], GasStateGenerator],
    Callable[[], StaggereBFieldGenerator],
    Callable[[], StaggereBFieldGenerator],
    Callable[[], StaggereBFieldGenerator],
]


GeneratorTuple = tuple[
    GasStateGenerator,  # gas state
    StaggereBFieldGenerator,  # Bx
    StaggereBFieldGenerator,  # By
    StaggereBFieldGenerator,  # Bz
]
InitialStateType = Union[PureHydroStateGenerator, MHDStateGenerators]

PrimitiveStateFunc = (
    Callable[[], GasStateGenerator]
    | Callable[
        [],
        GeneratorTuple,
    ]
)

StateGenerator = GasStateGenerator | GeneratorTuple

FloatOrArray = Union[float, NDArray[np.floating[Any]]]

ExpressionDict = dict[str, object]
