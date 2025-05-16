import numpy as np
from dataclasses import dataclass
from typing import Sequence, Any, TypedDict
from numpy.typing import NDArray
from pathlib import Path

from simbi.functional.reader import LazySimulationReader
from ..functional.maybe import Maybe
from ..physics import calculate_state_vector, StateVector
from ..tools.utility import read_file


class MHDFields(TypedDict):
    """Magnetic field components"""

    b1: NDArray[np.floating[Any]]
    b2: NDArray[np.floating[Any]]
    b3: NDArray[np.floating[Any]]


class SimulationFields(TypedDict):
    """Complete simulation state"""

    velocity: Sequence[NDArray[np.floating[Any]]]
    bfields: Sequence[NDArray[np.floating[Any]]]
    rho: NDArray[np.floating[Any]]
    pressure: NDArray[np.floating[Any]]
    chi: NDArray[np.floating[Any]]


@dataclass(frozen=True)
class CheckpointData:
    """Immutable checkpoint data container"""

    state: StateVector
    mesh: dict[str, NDArray[np.floating[Any]]]
    metadata: dict[str, Any]
    staggered_bfields: Sequence[NDArray[np.floating[Any]]]
    immersed_bodies: dict[str, Any]


def load_checkpoint(filepath: Path | str) -> Maybe[CheckpointData]:
    """Load checkpoint with pure functional approach"""

    def extract_fields(
        reader: LazySimulationReader,
    ) -> Maybe[CheckpointData]:
        fields, metadata, mesh, immersed_bodies = reader

        return (
            Maybe.of(fields)
            .map(
                lambda x: SimulationFields(
                    velocity=[x[f"v{i}"] for i in range(1, metadata["dimensions"] + 1)],
                    bfields=(
                        [x[f"b{i}_mean"] for i in range(1, metadata["dimensions"] + 1)]
                        if "mhd" in metadata["regime"]
                        else []
                    ),
                    rho=x["rho"],
                    pressure=x["p"],
                    chi=x["chi"],
                )
            )
            .map(
                lambda x: CheckpointData(
                    state=calculate_state_vector(
                        adiabatic_index=metadata["adiabatic_index"],
                        rho=x["rho"],
                        velocity=x["velocity"],
                        pressure=x["pressure"],
                        chi=x["chi"],
                        regime=metadata["regime"],
                        bfields=x["bfields"],
                    ),
                    staggered_bfields=(
                        [fields[f"b{i}"] for i in range(1, 4)] if x["bfields"] else []
                    ),
                    mesh=mesh,
                    metadata=metadata,
                    immersed_bodies=immersed_bodies,
                )
            )
        )

    return (
        Maybe.of(filepath)
        .map(lambda p: read_file(str(p), unpad=False))
        .bind(extract_fields)
    )
