from ..types.bodies import (
    ImmersedBodyConfig,
    GravitationalSystemConfig,
    BinaryConfig,
    BinaryComponentConfig,
)
from typing import Any, Union


def load_immersed_bodies_or_body_system(
    metadata: dict[str, Any], immersed_bodies: dict[str, Any]
) -> Union[list[ImmersedBodyConfig], GravitationalSystemConfig, None]:
    """Loads inidividual immersed bodies, but if a system config is present,
    we load the system config instead."""
    from ..types.bodies import BodyCapability, has_capability

    if not immersed_bodies:
        return None

    if "system_config" in metadata:
        body1 = immersed_bodies["body_0"]
        body2 = immersed_bodies["body_1"]
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
