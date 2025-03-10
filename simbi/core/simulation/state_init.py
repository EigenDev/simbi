import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional, Any, Sequence
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
    state: NDArray[np.floating[Any]]
    staggered_bfields: Sequence[NDArray[np.floating[Any]]]

    def update_from_cli_args(self, cli_args: dict[str, Any]) -> "SimulationBundle":
        """Update simulation bundle with new configuration"""
        # check if all the cli_args keys are None, if so, return self
        if all(value is None for value in cli_args.values()):
            return self

        return SimulationBundle(
            mesh_config=MeshSettings.update_from(self.mesh_config, cli_args),
            grid_config=GridSettings.update_from(self.grid_config, cli_args),
            io_config=IOSettings.update_from(self.io_config, cli_args),
            sim_config=SimulationSettings.update_from(self.sim_config, cli_args),
            state=self.state,
            staggered_bfields=self.staggered_bfields,
        )


def try_checkpoint_initialization(
    config: InitializationConfig,
) -> Maybe[SimulationBundle]:
    """Try to initialize from checkpoint"""
    if config.checkpoint_file is None:
        return Maybe(None)

    return (
        Maybe.of(config.checkpoint_file)
        .bind(load_checkpoint)
        .map(
            lambda chkpt: SimulationBundle(
                mesh_config=MeshSettings.from_dict(chkpt.setup),
                grid_config=GridSettings.from_dict(
                    chkpt.setup, spatial_order=chkpt.setup["spatial_order"]
                ),
                io_config=IOSettings.from_dict(chkpt.setup),
                sim_config=SimulationSettings.from_dict(chkpt.setup),
                state=chkpt.state.to_numpy(),
                staggered_bfields=chkpt.staggered_bfields,
            )
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
    )
