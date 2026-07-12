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
from typing import TYPE_CHECKING, Any, Optional, get_args

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
) -> tuple[Metadata, Optional[BodySystemConfig], tuple[int, ...]]:
    """
    load metadata from checkpoint file.

    returns:
        tuple of (metadata, body_system_config, mesh_shape)
    """
    data = read_simulation(str(checkpoint_path), unpad=False)

    # extract system info from body_collection if present
    system_info = None
    if data.body_collection:
        system_info = {
            "system_name": data.body_collection.system_name,
            "reference_frame": data.body_collection.reference_frame,
            "binary_params": data.body_collection.binary_params
            if hasattr(data.body_collection, "binary_params")
            else None,
        }

    body_system = _bodies_to_system(
        system_info,
        {
            f"body_{i}": body
            for i, body in enumerate(data.body_collection.bodies)
        }
        if data.body_collection
        else None,
    )

    # also need mesh shape for resolution calculation
    mesh_shape = data.mesh.shape if hasattr(data, "mesh") else (1, 1, 1)

    return data.metadata, body_system, mesh_shape


def metadata_to_config_dict(
    metadata: Metadata, mesh_shape: tuple[int, ...]
) -> dict[str, Any]:
    """
    convert checkpoint metadata to config dictionary format.

    these are the fields that define the physics state and must be
    preserved from checkpoint.

    args:
        metadata: checkpoint metadata
        mesh_shape: mesh shape from data.mesh.shape (includes ghost zones)
    """
    # mesh_shape is in STORAGE order (x_n, ..., x2, x1) — REVERSED, matching the bounds/spacing
    # `[::-1]` the parser applies. un-reverse to the forward (x1, x2, x3) the config expects, and pad
    # a lower-dimensional run (2D / 2.5D) up to the 3-tuple resolution field with trailing 1s.
    resolution = tuple(int(n) for n in reversed(tuple(mesh_shape)))
    resolution = resolution + (1,) * (3 - len(resolution))

    # the rust backend tags relativistic MHD by its kernel prefix "rmhd"; the frontend Regime enum
    # names it "srmhd". normalize so a checkpoint written by the backend restarts cleanly.
    _regime_aliases = {"rmhd": "srmhd"}
    regime_str = _regime_aliases.get(str(metadata.regime), str(metadata.regime))

    config = {
        "resolution": resolution,
        "start_time": float(metadata.time),
        "adiabatic_index": float(metadata.gamma),
        "coord_system": CoordSystem(metadata.coord_system),
        "regime": Regime(regime_str),
        "solver": Solver(metadata.solver),
        "reconstruction": Reconstruction(metadata.reconstruction),
        "timestepping": TimeStepping(metadata.timestepping),
        "plm_theta": float(metadata.plm_theta),
        "cfl_number": float(metadata.cfl),
        "checkpoint_index": int(metadata.checkpoint_index),
        "checkpoint_interval": float(metadata.checkpoint_interval),
        "x1_spacing": CellSpacing(metadata.x1_spacing),
        "x2_spacing": CellSpacing(metadata.x2_spacing),
        "x3_spacing": CellSpacing(metadata.x3_spacing),
        "boundary_conditions": [
            BoundaryCondition(b) for b in metadata.boundary_conditions
        ],
    }

    # amr fields if present
    if metadata.level_dts:
        config["refinement_level_dts"] = list(metadata.level_dts)
    if metadata.level_substeps:
        config["refinement_level_substeps"] = list(metadata.level_substeps)
    if metadata.subcycling_mode and metadata.subcycling_mode != "none":
        from simbi.types import SubCycleMode

        config["refinement_subcycling_mode"] = SubCycleMode(
            metadata.subcycling_mode
        )

    return config


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
    metadata, body_system, mesh_shape = load_checkpoint_metadata(
        checkpoint_path
    )
    checkpoint_config = metadata_to_config_dict(metadata, mesh_shape)

    # get field classifications
    immutable_fields = problem.get_checkpoint_immutable_fields()
    safe_fields = problem.get_checkpoint_safe_fields()

    # build merged config
    merged_data: dict[str, Any] = {}

    def normalize_value(val):
        """normalize value for pydantic validation."""
        if isinstance(val, Path):
            return str(val)
        elif hasattr(val, "value"):  # enum
            return val
        elif hasattr(val, "dtype"):  # numpy scalar
            return val.item()
        return val

    def coerce_to_field_type(value, field_info):
        """
        coerce checkpoint value to match field's expected type.
        uses pydantic validation to try each union member.
        """
        from pydantic import TypeAdapter, ValidationError

        annotation = field_info.annotation
        args = get_args(annotation)

        # filter out NoneType from optional fields
        union_members = [a for a in args if a is not type(None)] if args else []

        if len(union_members) > 1:
            # union type - try each member with pydantic validation
            for member_type in union_members:
                try:
                    adapter = TypeAdapter(member_type)
                    return adapter.validate_python(value)
                except (ValidationError, TypeError):
                    continue

            # failed - if single-element sequence, try unwrapping
            if isinstance(value, (list, tuple)) and len(value) == 1:
                for member_type in union_members:
                    try:
                        adapter = TypeAdapter(member_type)
                        return adapter.validate_python(value[0])
                    except (ValidationError, TypeError):
                        continue

            return value

        # not a union - try direct validation
        try:
            adapter = TypeAdapter(annotation)
            return adapter.validate_python(value)
        except (ValidationError, TypeError):
            # validation failed - try unwrapping sequences
            if isinstance(value, (list, tuple)):
                # if all elements are identical, extract first
                if len(value) > 0 and all(v == value[0] for v in value):
                    try:
                        adapter = TypeAdapter(annotation)
                        return adapter.validate_python(value[0])
                    except (ValidationError, TypeError):
                        pass
                # single-element sequence
                elif len(value) == 1:
                    try:
                        adapter = TypeAdapter(annotation)
                        return adapter.validate_python(value[0])
                    except (ValidationError, TypeError):
                        pass
            return value

    for field_name, field_info in type(problem).model_fields.items():
        user_value = normalize_value(getattr(problem, field_name))

        if field_name in checkpoint_config:
            checkpoint_value = checkpoint_config[field_name]

            if field_name in immutable_fields:
                # must use checkpoint value, but coerce to field's expected type
                merged_data[field_name] = coerce_to_field_type(
                    checkpoint_value, field_info
                )
            else:
                # user can override
                merged_data[field_name] = user_value
        else:
            # field not in checkpoint, use user value
            merged_data[field_name] = user_value

    # RESUME at the checkpoint's physical time, not the config's start_time. start_time is the
    # sim clock (sim.time): a restart must continue from where the checkpoint left off, regardless
    # of what the config (or its validators) set start_time to. the LOG-checkpoint anchor is a
    # SEPARATE field (checkpoint_log_anchor) so the cadence stays fixed across the restart.
    if "start_time" in checkpoint_config:
        merged_data["start_time"] = checkpoint_config["start_time"]

    # special handling for end_time: allow extending
    if "end_time" in safe_fields:
        user_end = getattr(problem, "end_time")
        checkpoint_start = checkpoint_config.get("start_time", 0.0)
        merged_data["end_time"] = max(user_end, checkpoint_start)

    # normalize types for pydantic validation
    # convert all Path objects to strings
    for key, value in merged_data.items():
        if isinstance(value, Path):
            merged_data[key] = str(value)

    # convert numpy integers to python ints
    for key, value in merged_data.items():
        if hasattr(value, "dtype"):  # numpy scalar
            merged_data[key] = value.item()
        elif isinstance(value, (list, tuple)):
            # check if list contains numpy types
            if value and hasattr(value[0], "dtype"):
                merged_data[key] = [
                    v.item() if hasattr(v, "dtype") else v for v in value
                ]

    # boundary_conditions special handling - convert single string to list
    # checkpoint has list, but some problem classes may expect single value or vice versa
    if "boundary_conditions" in merged_data:
        bcs = merged_data["boundary_conditions"]
        # if it's already a list of strings, keep it
        # pydantic will handle Union[BoundaryCondition, Sequence[BoundaryCondition]]
        pass

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
    metadata, _, mesh_shape = load_checkpoint_metadata(checkpoint_path)
    checkpoint_config = metadata_to_config_dict(metadata, mesh_shape)
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
