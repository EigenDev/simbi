import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Any
from ..config.settings import MeshSettings, IOSettings, GridSettings, SimulationSettings
from ..config.initialization import InitializationConfig
from ...functional.maybe import Maybe
from ...io.checkpoint import load_checkpoint
from ...physics import construct_conserved_state


@dataclass(frozen=True)
class SimulationBundle:
    """Complete simulation state and configuration"""

    mesh_config: MeshSettings
    grid_config: GridSettings
    io_config: IOSettings
    sim_config: SimulationSettings
    state: NDArray[np.float64]
    staggered_bfields: Optional[NDArray[np.float64]] = None


def try_checkpoint_initialization(
    config: InitializationConfig,
) -> Maybe[SimulationBundle]:
    """Try to initialize from checkpoint"""
    if config.checkpoint_file is None:
        return Maybe(None)

    return (
        Maybe.of(config.checkpoint_file)
        .bind_with_context(load_checkpoint, "Failed to load checkpoint")
        .map_with_context(
            lambda chkpt: SimulationBundle(
                mesh_config=MeshSettings.from_dict(chkpt.setup),
                grid_config=GridSettings.from_dict(
                    chkpt.setup, spatial_order=chkpt.setup["spatial_order"]
                ),
                io_config=IOSettings.from_dict(chkpt.setup),
                sim_state=SimulationSettings.from_dict(chkpt.setup),
                state=chkpt.state.to_numpy(),
                bfield=chkpt.staggered_bfields,
            ),
            "Failed to create bundle from checkpoint",
        )
    )


def try_fresh_initialization(
    config: InitializationConfig, settings: dict[str, dict[str, Any]]
) -> Maybe[SimulationBundle]:
    """Initialize from initial conditions"""
    return (
        Maybe.of(config)
        .bind(
            lambda c: construct_conserved_state(
                settings["sim_state"]["regime"],
                settings["sim_state"]["adiabatic_index"],
                settings["sim_state"]["is_mhd"],
                c.evaluate(
                    pad_width=1 + (settings["sim_state"]["spatial_order"] == "plm"),
                    nvars=settings["sim_state"]["nvars"],
                ).unwrap(),
            )
        )
        .map(
            lambda init: SimulationBundle(
                mesh_config=MeshSettings.from_dict(settings["mesh"]),
                grid_config=GridSettings.from_dict(
                    settings["grid"],
                    spatial_order=settings["sim_state"]["spatial_order"],
                ),
                io_config=IOSettings.from_dict(settings["io"]),
                sim_config=SimulationSettings.from_dict(settings["sim_state"]),
                state=init.state.to_numpy(),
                staggered_bfields=init.staggered_bfields,
            )
        )
    )


def initialize_simulation(
    config: InitializationConfig, settings: dict[str, dict[str, Any]]
) -> Maybe[SimulationBundle]:
    """Initialize simulation using chain of responsibility"""
    return (
        Maybe.of(config)
        .bind(try_checkpoint_initialization)
        .or_else(try_fresh_initialization(config, settings))
    ).unwrap()
