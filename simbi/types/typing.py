# =============================================================================
# simbi/types/typing.py
#
# type aliases for generator-based initial conditions.
# the rust backend (cpu_ext, pyo3) consumes python generators directly,
# so these types define the expected signatures:
#   - GasStateGenerator: yields (rho, v1, v2, v3, p) tuples
#   - BFieldGenerator: yields individual b-field component values
#   - MHDStateGenerators: tuple of gas + 3 b-field generator factories
#
# usage:
#   def my_initial_state() -> GasStateGenerator:
#       for x in grid:
#           yield (rho, vx, vy, vz, p)
# =============================================================================

from typing import Callable, Generator, Sequence, Union

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
