import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Any, Sequence

from simbi.core.config.bodies import (
    GravitationalSystemConfig,
    BinaryComponentConfig,
    BinaryConfig,
    ImmersedBodyConfig,
)
from simbi.functional.reader import BodyCapability, has_capability
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
    setup: tuple[InitializationConfig, dict[str, Any]],
) -> Maybe[SimulationBundle]:
    """Try to initialize from checkpoint"""
    config = setup[0]
    if config.checkpoint_file is None:
        return Maybe(None)

    # a user might have settings in their problem
    # class that are not present inside the checkpoint
    # metdata. If this happens, we use the defaults
    # given inside the problem

    def overwrite_if_needed(
        settings: dict[str, Any], metadata: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        meta_set = {key for key in metadata.keys()}
        overwrite = settings.copy()
        overwrite.update({k: metadata[k] for k in overwrite.keys() if k in meta_set})
        non_intersecting = set(settings.keys()) - meta_set
        if "bounds" in non_intersecting:
            if metadata["effective_dimensions"] == 3:
                overwrite["bounds"] = (
                    (float(metadata["x1min"]), float(metadata["x1max"])),
                    (float(metadata["x2min"]), float(metadata["x2max"])),
                    (float(metadata["x3min"]), float(metadata["x3max"])),
                )
            elif metadata["effective_dimensions"] == 2:
                overwrite["bounds"] = (
                    (float(metadata["x1min"]), float(metadata["x1max"])),
                    (float(metadata["x2min"]), float(metadata["x2max"])),
                )
            else:
                overwrite["bounds"] = (
                    (float(metadata["x1min"]), float(metadata["x1max"])),
                )
        elif "resolution" in non_intersecting:
            nghosts = 2 * (1 + (metadata["spatial_order"] == "plm"))
            overwrite["resolution"] = (
                int(metadata["nx"] - nghosts),
                int(metadata["ny"] - nghosts),
                int(metadata["nz"] - nghosts),
            )
        elif "checkpoint_index" in non_intersecting:
            try:
                overwrite["checkpoint_index"] = int(metadata["checkpoint_index"])
            except KeyError:  # for legacy reasons
                overwrite["checkpoint_index"] = int(metadata["checkpoint_idx"])
        elif "default_start_time" in non_intersecting:
            overwrite["default_start_time"] = metadata["time"]
            overwrite["isothermal"] = bool(metadata["adiabatic_index"] == 1.0)

        if "immersed_bodies" in kwargs:
            if metadata["system_config"] is None:
                overwrite["immersed_bodies"] = []
                for key, body in kwargs["immersed_bodies"].items():
                    overwrite["immersed_bodies"].append(
                        ImmersedBodyConfig(
                            body_type=body["type"],
                            mass=body["mass"],
                            velocity=body["velocity"],
                            position=body["position"],
                            radius=body["radius"],
                            specifics=body.get("specifics", None),
                        )
                    )
            else:
                # check if there are two bodies that are gravitational,
                # if so, this is likely a binary system
                if len(kwargs["immersed_bodies"]) == 2:
                    body1 = kwargs["immersed_bodies"]["body_0"]
                    body2 = kwargs["immersed_bodies"]["body_1"]
                    system_config = metadata["system_config"]

                    overwrite["body_system"] = GravitationalSystemConfig(
                        prescribed_motion=system_config["prescribed_motion"],
                        reference_frame=system_config["reference_frame"],
                        system_type="binary",
                        binary_config=BinaryConfig(
                            semi_major=system_config["semi_major"],
                            eccentricity=system_config["eccentricity"],
                            mass_ratio=system_config["mass_ratio"],
                            total_mass=body1["mass"] + body2["mass"],
                            components=[
                                BinaryComponentConfig(
                                    mass=body1["mass"],
                                    radius=body1["radius"],
                                    is_an_accretor=has_capability(
                                        body1["type"], BodyCapability.ACCRETION
                                    ),
                                    softening_length=body1["softening_length"],
                                    two_way_coupling=False,
                                    accretion_efficiency=body1["accretion_efficiency"],
                                    accretion_radius=body1["accretion_radius"],
                                    total_accreted_mass=body1["total_accreted_mass"],
                                    position=body1["position"],
                                    velocity=body1["velocity"],
                                    force=body1["force"],
                                ),
                                BinaryComponentConfig(
                                    mass=body2["mass"],
                                    radius=body2["radius"],
                                    is_an_accretor=has_capability(
                                        body2["type"], BodyCapability.ACCRETION
                                    ),
                                    softening_length=body2["softening_length"],
                                    two_way_coupling=False,
                                    accretion_efficiency=body2["accretion_efficiency"],
                                    accretion_radius=body2["accretion_radius"],
                                    total_accreted_mass=body2["total_accreted_mass"],
                                    position=body2["position"],
                                    velocity=body2["velocity"],
                                    force=body2["force"],
                                ),
                            ],
                        ),
                    )
        return overwrite

    settings = setup[1]

    res = (
        Maybe.of(config.checkpoint_file)
        .bind(load_checkpoint)
        .map(
            lambda chkpt: SimulationBundle(
                mesh_config=MeshSettings.from_dict(
                    overwrite_if_needed(settings["mesh"], chkpt.metadata)
                ),
                grid_config=GridSettings.from_dict(
                    overwrite_if_needed(settings["grid"], chkpt.metadata),
                    spatial_order=chkpt.metadata["spatial_order"],
                ),
                io_config=IOSettings.from_dict(
                    overwrite_if_needed(settings["io"], chkpt.metadata)
                ),
                sim_config=SimulationSettings.from_dict(
                    overwrite_if_needed(
                        settings["sim_state"],
                        chkpt.metadata,
                        immersed_bodies=chkpt.immersed_bodies,
                    )
                ),
                state=chkpt.state.to_numpy(),
                staggered_bfields=chkpt.staggered_bfields,
            )
        )
    )

    if res.is_error():
        raise ValueError("Error loading checkpoint") from res.error

    return res


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
        Maybe.of((config, settings))
        .bind(try_checkpoint_initialization)
        .or_else(try_fresh_initialization(config, settings))
    )
