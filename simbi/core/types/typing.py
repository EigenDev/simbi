"""
Type definitions for simbi.

This module provides type annotations for the simbi framework.
"""

from typing import Callable, Generator, Sequence, Union, TypeVar, Any

# Type for a generator that yields gas state tuples
GasStateGenerator = Generator[Sequence[float], None, None]

# Function that returns a gas state generator
GasStateFunction = Callable[[], GasStateGenerator]

# Type for staggered B-field generators
StaggeredBFieldGenerator = Generator[float, None, None]

# Function that returns a B-field generator
BFieldFunction = Callable[[], StaggeredBFieldGenerator]

# Type for MHD state generators (gas state + B-fields)
MHDStateGenerators = tuple[
    GasStateFunction,
    BFieldFunction,
    BFieldFunction,
    BFieldFunction,
]

# Initial state can be either a pure hydro generator or MHD generators
InitialStateType = Union[GasStateFunction, MHDStateGenerators]

ExpressionDict = dict[str, object]
