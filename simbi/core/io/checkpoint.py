"""
Checkpoint loading functionality for new SimulationState.

This module adapts the existing checkpoint loading functionality to work with
the new SimulationState structure.
"""

from pathlib import Path
from typing import Optional, Union, Any
from astropy.table.np_utils import Sequence
import numpy as np

from ..types.bodies import (
    ImmersedBodyConfig,
    GravitationalSystemConfig,
    BinaryConfig,
    BinaryComponentConfig,
)
from ...functional.reader import read_file, LazySimulationReader
from ..config.base_config import SimbiBaseConfig
from ..simulation.state_init import SimulationState
from ...functional.maybe import Maybe


def load_immersed_bodies_or_body_system(
    metadata: dict[str, Any],
) -> Union[list[ImmersedBodyConfig], GravitationalSystemConfig, None]:
    """Loads inidividual immersed bodies, but if a system config is present,
    we load the system config instead."""
    from ..types.bodies import BodyCapability, has_capability

    bodies = metadata.get("immersed_bodies", {})
    if not bodies:
        return None

    if "system_config" in metadata:
        body1 = metadata["immersed_bodies"]["body_0"]
        body2 = metadata["immersed_bodies"]["body_1"]
        system_config = metadata["system_config"]
        return GravitationalSystemConfig(
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
                            body1["capability"], BodyCapability.ACCRETION
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
                            body2["capability"], BodyCapability.ACCRETION
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

    return [
        ImmersedBodyConfig(
            capability=body["capability"],
            mass=body["mass"],
            radius=body["radius"],
            position=tuple(body["position"]),
            velocity=tuple(body["velocity"]),
            force=tuple(body["force"]),
            two_way_coupling=bool(body["two_way_coupling"]),
            specifics={
                k: v
                for k, v in body.items()
                if k
                not in {
                    "capability",
                    "mass",
                    "radius",
                    "position",
                    "velocity",
                    "force",
                    "two_way_coupling",
                }
            },
        )
        for body in metadata["immersed_bodies"].items()
    ]


def load_checkpoint_to_state(
    filepath: Union[str, Path], config: Optional[SimbiBaseConfig] = None
) -> Maybe[SimulationState]:
    """
    Load a checkpoint file into a SimulationState object using existing checkpoint loader.

    Args:
        filepath: Path to the checkpoint file
        config: Optional config to use (otherwise config is derived from checkpoint)

    Returns:
        Maybe[SimulationState] object with data from checkpoint
    """
    from ..types.input import Solver, Regime, TimeStepping, CoordSystem, SpatialOrder

    def extract_fields(
        reader: LazySimulationReader,
    ) -> Maybe[SimulationState]:
        fields, metadata, mesh, immersed_bodies = reader
        ib_config = load_immersed_bodies_or_body_system(metadata)

        return Maybe.of(fields).map(
            lambda x: SimulationState(
                primitive_state=np.array(
                    [
                        x["rho"],
                        *[x[f"v{i}"] for i in range(1, metadata["dimensions"] + 1)],
                        *(
                            [
                                x[f"b{i}_mean"]
                                for i in range(1, metadata["dimensions"] + 1)
                            ]
                            if "mhd" in metadata["regime"]
                            else []
                        ),
                        x["pressure"],
                        x["chi"],
                    ],
                    dtype=np.float64,
                ),
                conserved_state=np.array(
                    [
                        x["D"],
                        *[x[f"m{i}"] for i in range(1, metadata["dimensions"] + 1)],
                        *(
                            [
                                x[f"b{i}_mean"]
                                for i in range(1, metadata["dimensions"] + 1)
                            ]
                            if "mhd" in metadata["regime"]
                            else []
                        ),
                        x["energy"],
                        x["chi_dens"],
                    ],
                    dtype=np.float64,
                ),
                staggered_bfields=(
                    [fields[f"b{i}"] for i in range(1, 4)]
                    if "mhd" in metadata["regime"]
                    else []
                ),
                config=SimbiBaseConfig(
                    resolution=tuple((metadata["nz"], metadata["ny"], metadata["nx"])),
                    bounds=tuple(
                        (
                            (metadata["x1min"], metadata["x1max"]),
                            (metadata["x2min"], metadata["x2max"]),
                            (metadata["x3min"], metadata["x3max"]),
                        ),
                    ),
                    regime=Regime(metadata["regime"]),
                    coord_system=CoordSystem(metadata["coord_system"]),
                    adiabatic_index=float(metadata["adiabatic_index"]),
                    spatial_order=SpatialOrder(metadata["spatial_order"]),
                    # ambient_sound_speed=float(metadata.get("ambient_sound_speed", 0.0)),
                    start_time=float(metadata.get("time", 0.0)),
                    checkpoint_index=int(metadata.get("checkpoint_index", 0)),
                    x1_spacing=metadata["x1_spaxing"],
                    x2_spacing=metadata["x2_spacing"],
                    x3_spacing=metadata["x3_spacing"],
                    solver=Solver(metadata["solver"]),
                    cfl_number=float(metadata["cfl"]),
                    checkpoint_interval=float(metadata["checkpoint_interval"]),
                    temporal_order=TimeStepping(metadata["temporal_order"]),
                    data_directory=metadata["data_directory"],
                    # immersed_bodies=ib_config
                    # if isinstance(ib_config, Sequence)
                    # else [],
                    # body_system=ib_config
                    # if isinstance(ib_config, GravitationalSystemConfig)
                    # else None,
                ),
            )
        )

    return (
        Maybe.of(filepath)
        .map(lambda p: read_file(str(p), unpad=False))
        .bind(extract_fields)
    )
