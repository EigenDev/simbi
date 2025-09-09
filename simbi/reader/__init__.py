from .io import open_file, read_raw_data
from .parsing import parse_data
from .lazy import SimData
from ..core.types import ProcessedData
from ..functional.result import Result
from functools import partial


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
        open_file(filename).and_then(read_raw_data).and_then(partial_parse).map(SimData)
    )


def load_simulation_data(filename: str) -> Result[ProcessedData]:
    """
    Load parsed simulation data without lazy evaluation.

    Returns structured data with metadata, mesh, bodies, etc.
    """
    return open_file(filename).and_then(read_raw_data).and_then(parse_data)


# Convenience function for quick access (throws on error)
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
    "load_simulation_data",
    "read_simulation",
    "SimData",
    "ProcessedData",
    "Result",
]
