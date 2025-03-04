import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from numpy.typing import NDArray
from pathlib import Path
from ..functional.maybe import Maybe
from ..physics import calculate_state_vector, StateVector
from ..tools.utility import read_file


@dataclass(frozen=True)
class CheckpointData:
    """Immutable checkpoint data container"""

    state: StateVector
    time: float
    mesh: dict[str, np.ndarray]
    checkpoint_idx: int
    staggered_bfields: Optional[Tuple[np.ndarray, ...]] = None


def load_checkpoint(filepath: Path) -> Maybe[CheckpointData]:
    """Load checkpoint with pure functional approach"""

    print("Loading checkpoint")

    def extract_fields(
        data: tuple[
            dict[str, NDArray[np.float64]],
            dict[str, Any],
            dict[str, NDArray[np.float64]],
        ]
    ) -> Maybe[CheckpointData]:
        fields, setup, mesh = data
        dim = setup["dimensions"]

        return (
            Maybe.of((fields, setup))
            .map(
                lambda x: {
                    "velocity": [x[0][f"v{i}"] for i in range(1, dim + 1)],
                    "bfields": (
                        [x[0][f"b{i}"] for i in range(1, dim + 1)]
                        if "mhd" in x[1]["regime"]
                        else None
                    ),
                    "rho": x[0]["rho"],
                    "pressure": x[0]["p"],
                    "setup": x[1],
                    "mesh": mesh,
                }
            )
            .map(
                lambda x: CheckpointData(
                    state=calculate_state_vector(
                        adiabatic_index=x["setup"]["adiabatic_index"],
                        rho=x["rho"],
                        velocity=x["velocity"],
                        pressure=x["pressure"],
                        chi=x["chi"],
                        regime=x["setup"]["regime"],
                        bfields=x.get("bfields", None),
                    ),
                    staggered_bfields=(
                        [fields[f"b{i}stag"] for i in range(1, 4)]
                        if x["bfields"]
                        else None
                    ),
                    mesh=x["mesh"],
                    setup=x["setup"],
                )
            )
        )

    return Maybe.of(filepath).map(lambda p: read_file(str(p))).bind(extract_fields)
