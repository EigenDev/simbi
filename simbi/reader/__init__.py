# =============================================================================
# simbi/reader/__init__.py
#
# simulation data loading with lazy field evaluation.
# primary interface: read_simulation(filename) -> SimData
#
# SimData provides:
#   - direct access to primitive fields (rho, p, v1, etc.)
#   - lazy computation of derived fields (W, energy, mach, etc.)
#   - automatic dependency resolution
# =============================================================================
from functools import partial

from ..functional.result import Result
from ..types import ProcessedData
from .io import open_file, read_raw_data
from .lazy import SimData
from .parsing import parse_data


def load_simulation(filename: str, unpad: bool) -> Result[SimData]:
    """
    Load simulation data with lazy field evaluation.

    Returns Result[SimData] where SimData provides:
    - Direct access to primitive fields (rho, p, v1, etc.)
    - Lazy computation of derived fields (W, energy, mach, etc.)
    - Automatic dependency resolution
    """
    partial_parse = partial(parse_data, unpad=unpad)
    return (
        open_file(filename)
        .and_then(read_raw_data)
        .and_then(partial_parse)
        .map(SimData)
    )


def read_simulation(filename: str, unpad: bool = True) -> SimData:
    """
    Convenience function that loads simulation or raises exception.

    Equivalent to load_simulation(filename).value but throws on error.
    """
    result = load_simulation(filename, unpad)
    if result.error:
        raise result.error

    if result.value is not None:
        return result.value
    else:
        raise ValueError("No simulation data found in file")


__all__ = [
    "load_simulation",
    "read_simulation",
    "SimData",
    "ProcessedData",
    "Result",
]
