# =============================================================================
# checkpoint.py
#
# checkpoint loading and config merging for simbi simulations.
# the actual state data is loaded by C++ - python just handles metadata
# and config merging with checkpoint_safe field validation.
#
# usage:
#   problem = SodProblem(end_time=10.0)  # user wants to extend runtime
#   merged = merge_with_checkpoint(problem, Path("data/checkpoint.h5"))
# =============================================================================
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from simbi.reader import read_simulation
from simbi.types.bodies import (
    BinaryComponentConfig,
    BinaryConfig,
    Body,
    BodyCapability,
    BodySystemConfig,
    GravitationalSystemConfig,
    ImmersedBodyConfig,
)
from simbi.types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Metadata,
    Reconstruction,
    Regime,
    Solver,
    TimeStepping,
)

if TYPE_CHECKING:
    from .problem import SimbiProblem


def _has_capability(
    body_capability: BodyCapability, capability: BodyCapability
) -> bool:
    """check if body has a specific capability."""
    return bool(body_capability & capability)


def _bodies_to_system(
    bodies_group: dict[str, Any] | None,
    bodies: dict[str, Body] | None,
) -> list[ImmersedBodyConfig] | GravitationalSystemConfig | None:
    """
    parse body system configuration from checkpoint data.

    handles binary systems and individual immersed bodies.
    """
    if not bodies or not bodies_group:
        return None

    system_name = bodies_group.get("system_name", "individual")

    if "binary" in system_name:
        binary_params = bodies_group["binary_params"]
        body1 = bodies.get("body_0")
        body2 = bodies.get("body_1")

        if not body1 or not body2:
            raise ValueError("binary system requires both bodies")
        if body1.accretion is None or body2.accretion is None:
            raise ValueError("binary bodies must have accretion info")
        if body1.gravitational is None or body2.gravitational is None:
            raise ValueError("binary bodies must have gravitational info")

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
                        is_an_accretor=_has_capability(
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
                        is_an_accretor=_has_capability(
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

    # individual bodies
    return [
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


def load_checkpoint_metadata(
    checkpoint_path: Path,
) -> tuple[Metadata, Optional[BodySystemConfig]]:
    """
    load metadata from checkpoint file.

    returns:
        tuple of (metadata, body_system_config)
    """
    data = read_simulation(str(checkpoint_path), unpad=False)
    body_system = _bodies_to_system(
        data.metadata.system_info, data.body_collection
    )
    return data.metadata, body_system


def metadata_to_config_dict(metadata: Metadata) -> dict[str, Any]:
    """
    convert checkpoint metadata to config dictionary format.

    these are the fields that define the physics state and must be
    preserved from checkpoint.
    """
    halo = metadata.halo_radius
    resolution = tuple(s - 2 * halo for s in metadata.resolution[::-1])

    return {
        "resolution": resolution,
        "start_time": float(metadata.time),
        "adiabatic_index": float(metadata.adiabatic_index),
        "coord_system": CoordSystem(metadata.coord_system),
        "regime": Regime(metadata.regime),
        "solver": Solver(metadata.solver),
        "reconstruction": Reconstruction(metadata.reconstruction),
        "timestepping": TimeStepping(metadata.timestepping),
        "plm_theta": float(metadata.plm_theta),
        "cfl_number": float(metadata.cfl_number),
        "checkpoint_index": int(metadata.checkpoint_index),
        "checkpoint_interval": float(metadata.checkpoint_interval),
        "x1_spacing": CellSpacing(metadata.x1_spacing),
        "x2_spacing": CellSpacing(metadata.x2_spacing),
        "x3_spacing": CellSpacing(metadata.x3_spacing),
        "boundary_conditions": [
            BoundaryCondition(b) for b in metadata.boundary_conditions
        ],
    }


def merge_with_checkpoint(
    problem: SimbiProblem,
    checkpoint_path: Path,
) -> SimbiProblem:
    """
    merge user problem config with checkpoint metadata.

    - checkpoint_safe=True fields: user value is kept
    - checkpoint_safe=False fields: checkpoint value is used

    raises ValueError if user tries to override an immutable field
    with a different value.

    args:
        problem: user-provided problem configuration
        checkpoint_path: path to checkpoint file

    returns:
        new problem instance with merged configuration
    """
    metadata, body_system = load_checkpoint_metadata(checkpoint_path)
    checkpoint_config = metadata_to_config_dict(metadata)

    # get field classifications
    immutable_fields = problem.get_checkpoint_immutable_fields()
    safe_fields = problem.get_checkpoint_safe_fields()

    # build merged config
    merged_data: dict[str, Any] = {}

    for field_name, field_info in problem.model_fields.items():
        user_value = getattr(problem, field_name)

        if field_name in checkpoint_config:
            checkpoint_value = checkpoint_config[field_name]

            if field_name in immutable_fields:
                # must use checkpoint value
                merged_data[field_name] = checkpoint_value
            else:
                # user can override
                merged_data[field_name] = user_value
        else:
            # field not in checkpoint, use user value
            merged_data[field_name] = user_value

    # special handling for end_time: allow extending
    if "end_time" in safe_fields:
        user_end = getattr(problem, "end_time")
        checkpoint_start = checkpoint_config.get("start_time", 0.0)
        merged_data["end_time"] = max(user_end, checkpoint_start)

    # create new instance with merged config
    return type(problem)(**merged_data)


def validate_checkpoint_compatibility(
    problem: SimbiProblem,
    checkpoint_path: Path,
) -> list[str]:
    """
    check if problem config is compatible with checkpoint.

    returns list of error messages (empty if compatible).
    """
    errors = []
    metadata, _ = load_checkpoint_metadata(checkpoint_path)
    checkpoint_config = metadata_to_config_dict(metadata)
    immutable_fields = problem.get_checkpoint_immutable_fields()

    for field_name in immutable_fields:
        if field_name not in checkpoint_config:
            continue

        user_value = getattr(problem, field_name)
        checkpoint_value = checkpoint_config[field_name]

        # skip if user didn't explicitly set (using default)
        if user_value == problem.model_fields[field_name].default:
            continue

        if user_value != checkpoint_value:
            errors.append(
                f"{field_name}: user={user_value}, checkpoint={checkpoint_value} "
                f"(field is checkpoint_safe=False)"
            )

    return errors
