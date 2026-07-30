# =============================================================================
# simbi/reader/__init__.py
#
# clean checkpoint reader using io.
#
# usage:
#   from simbi.reader import read_checkpoint
#   checkpoint = read_checkpoint("file.h5").unwrap()
#
#   # or use compatibility wrapper
#   from simbi.reader import read_simulation
#   data = read_simulation("file.h5")
# =============================================================================

from ..functional import Err, Ok, Result
from .adapter import SimData
from .census import Census, CensusError, census_names, read_census
from .io import (
    Checkpoint,
    Domain,
    FieldData,
    HydroFields,
    LevelData,
    MeshGeometry,
    Metadata,
    PartitionData,
    get_base_fields,
    read_checkpoint,
)
from .logging import logger


def read_simulation(filename: str, unpad: bool = True) -> SimData:
    """
    compatibility wrapper for old read_simulation api.

    loads a checkpoint file and returns a SimData adapter that provides
    the legacy interface (.metadata, .get_field, etc).

    args:
        filename: path to checkpoint file
        unpad: ignored (kept for api compatibility)

    returns:
        SimData adapter wrapping Checkpoint

    example:
        data = read_simulation("checkpoint.h5")
        rho = data.get_field("rho")
        time = data.metadata.time
    """
    checkpoint = read_checkpoint(filename).unwrap()
    return SimData(checkpoint)


__all__ = [
    # main api
    "read_checkpoint",
    "read_census",
    "census_names",
    "Census",
    "CensusError",
    "read_simulation",
    "get_base_fields",
    # types
    "Checkpoint",
    "LevelData",
    "PartitionData",
    "HydroFields",
    "FieldData",
    "Domain",
    "MeshGeometry",
    "Metadata",
    # functional
    "Result",
    "Ok",
    "Err",
    # logging
    "logger",
]
