"""
Checkpoint loading functionality for new SimulationState.

This module adapts the existing checkpoint loading functionality to work with
the new SimulationState structure.
"""

from typing import Any

import numpy as np

from ...core.types.bodies import (
    Body,
    BodyCapability,
    GravitationalSystemConfig,
    ImmersedBodyConfig,
)
from ...functional.maybe import Maybe
from ...reader import SimData, read_simulation
from ..config.base_config import SimbiBaseConfig
from ..simulation.state_init import SimulationState


def has_capability(
    body_capability: BodyCapability, capability: BodyCapability
) -> bool:
    return bool(body_capability & capability)


def to_system(
    bodies_group: dict[str, Any] | None, bodies: dict[str, Body] | None
) -> list[ImmersedBodyConfig] | GravitationalSystemConfig | None:
    """Parse a body system configuration from a collection of bodies."""
    from ...core.types.bodies import BinaryComponentConfig, BinaryConfig

    if not bodies:
        return None

    if not bodies_group:
        return None

    system_name = bodies_group.get("system_name", "individual")
    if "binary" in system_name:
        binary_params = bodies_group["binary_params"]
        body1 = bodies.get("body_0")
        body2 = bodies.get("body_1")
        if not body1 or not body2:
            raise ValueError("Both bodies must be present for a binary system.")
        if body1.accretion is None or body2.accretion is None:
            raise ValueError(
                "Both bodies in a binary must have accretion info."
            )
        if body1.gravitational is None or body2.gravitational is None:
            raise ValueError(
                "Both bodies in a binary must have gravitational info."
            )

        return GravitationalSystemConfig(
            prescribed_motion=True,
            reference_frame=bodies_group["reference_frame"],
            system_type="binary",
            binary_config=BinaryConfig(
                semi_major=binary_params["semi_major"],
                eccentricity=binary_params["eccentricity"],
                mass_ratio=body2.mass / body1.mass if body1.mass != 0 else 0.0,
                total_mass=body1.mass + body2.mass,
                components=[
                    BinaryComponentConfig(
                        mass=body1.mass,
                        radius=body1.radius,
                        is_an_accretor=has_capability(
                            body1.capabilities, BodyCapability.ACCRETION
                        ),
                        softening_length=body1.gravitational.softening_length,
                        two_way_coupling=False,
                        sink_rate=body1.accretion.sink_rate,
                        sink_delta=body1.accretion.sink_delta,
                        accretion_radius=body1.accretion.accretion_radius,
                        total_accreted_mass=body1.accretion.total_accreted_mass,
                        position=body1.position,
                        velocity=body1.velocity,
                    ),
                    BinaryComponentConfig(
                        mass=body2.mass,
                        radius=body2.radius,
                        is_an_accretor=has_capability(
                            body2.capabilities, BodyCapability.ACCRETION
                        ),
                        softening_length=body2.gravitational.softening_length,
                        two_way_coupling=False,
                        sink_rate=body2.accretion.sink_rate,
                        sink_delta=body2.accretion.sink_delta,
                        accretion_radius=body2.accretion.accretion_radius,
                        total_accreted_mass=body2.accretion.total_accreted_mass,
                        position=body2.position,
                        velocity=body2.velocity,
                    ),
                ],
            ),
        )
    else:
        # Individual bodies
        x = [
            ImmersedBodyConfig(
                capability=body.capabilities,
                mass=body.mass,
                radius=body.radius,
                position=body.position,
                velocity=body.velocity,
                two_way_coupling=False,
                force=(0.0, 0.0, 0.0),
                gravitational=body.gravitational,
                accretion=body.accretion,
                elastic=body.elastic,
                deformable=body.deformable,
                rigid=body.rigid,
            )
            for body in bodies.values()
        ]
        return x


def load_checkpoint_to_state(
    default_config: SimbiBaseConfig,
) -> Maybe[SimulationState]:
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
        body_system = to_system(metadata.system_info, data.bodies)
        config = SimbiBaseConfig.from_checkpoint_and_default(
            default_config, metadata, body_system
        )

        return Maybe.of(fields).map(
            lambda x: SimulationState(
                primitive_state=np.array(
                    [
                        x["rho"],
                        *[
                            x[f"v{i}"]
                            for i in range(1, metadata.dimensions + 1)
                        ],
                        *(
                            [
                                x[f"b{i}_mean"]
                                for i in range(1, metadata.dimensions + 1)
                            ]
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
                        *[
                            x[f"m{i}"]
                            for i in range(1, metadata.dimensions + 1)
                        ],
                        x["energy"],
                        *(
                            [
                                x[f"b{i}_mean"]
                                for i in range(1, metadata.dimensions + 1)
                            ]
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
