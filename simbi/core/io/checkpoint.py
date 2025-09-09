"""
Checkpoint loading functionality for new SimulationState.

This module adapts the existing checkpoint loading functionality to work with
the new SimulationState structure.
"""

import numpy as np

from simbi.reader import read_simulation
from simbi.reader.lazy import SimData


from ..config.base_config import SimbiBaseConfig
from ..simulation.state_init import SimulationState
from ...functional.maybe import Maybe


def load_checkpoint_to_state(default_config: SimbiBaseConfig) -> Maybe[SimulationState]:
    """
    Load a checkpoint file into a SimulationState object using existing checkpoint loader.

    Args:
        filepath: Path to the checkpoint file
        config: Optional config to use (otherwise config is derived from checkpoint)

    Returns:
        Maybe[SimulationState] object with data from checkpoint
    """

    def extract_fields(
        data: SimData,
    ) -> Maybe[SimulationState]:
        fields = data.fields
        metadata = data.metadata
        immersed_bodies = data.bodies
        config = SimbiBaseConfig.from_checkpoint_and_default(
            default_config, metadata, immersed_bodies
        )

        return Maybe.of(fields).map(
            lambda x: SimulationState(
                primitive_state=np.array(
                    [
                        x["rho"],
                        *[x[f"v{i}"] for i in range(1, metadata.dimensions + 1)],
                        *(
                            [x[f"b{i}_mean"] for i in range(1, metadata.dimensions + 1)]
                            if "mhd" in metadata.regime
                            else []
                        ),
                        x["p"],
                        x["chi"],
                    ],
                    dtype=np.float64,
                ),
                conserved_state=np.array(
                    [
                        x["D"],
                        *[x[f"m{i}"] for i in range(1, metadata.dimensions + 1)],
                        x["energy"],
                        *(
                            [x[f"b{i}_mean"] for i in range(1, metadata.dimensions + 1)]
                            if "mhd" in metadata.regime
                            else []
                        ),
                        x["chi_dens"],
                    ],
                    dtype=np.float64,
                ),
                staggered_bfields=(
                    [fields[f"b{i}"] for i in range(1, 4)]
                    if "mhd" in metadata.regime
                    else []
                ),
                config=config,
            )
        )

    return (
        Maybe.of(default_config.checkpoint_file)
        .map(lambda p: read_simulation(p or "", unpad=False))
        .bind(extract_fields)
    )
