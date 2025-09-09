from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from simbi.core.types.bodies import (
    BodyCapability,
    GravitationalSystemConfig,
    ImmersedBodyConfig,
)

from ..core.types import (
    AccretionProperties,
    BaseBody,
    Body,
    BodyDiagnostics,
    DeformableProperties,
    ElasticProperties,
    GravitationalProperties,
    MeshConfig,
    Metadata,
    ProcessedData,
    RawHDF5,
    RigidProperties,
)
from ..functional.result import Result

Array = NDArray[np.floating]


def has_capability(body_capability: BodyCapability, capability: BodyCapability) -> bool:
    return bool(body_capability & capability)


def unpad_field(
    field_data: Array, field_name: str, mesh: MeshConfig, metadata: Metadata
) -> Array:
    """Remove ghost zones from field data"""
    if not _is_gas_variable(field_name):
        return field_data

    padwidth = mesh.halo_radius
    if padwidth == 0:
        return field_data

    # Calculate effective dimensions (non-unity dimensions)
    inactive_dimensions = sum(1 for x in mesh.shape if x == 1)
    effective_dim = metadata.dimensions - inactive_dimensions

    # Remove inactive dimensions that have shape 1 + 2*padwidth
    data = field_data
    if any(x == 1 + 2 * padwidth for x in data.shape):
        slices: list[slice] = list(
            slice(padwidth, -padwidth) if x == 1 + 2 * padwidth else slice(None)
            for x in data.shape
        )
        data = data[tuple(slices)]

    # Create padding specification for effective dimensions
    npad = tuple((padwidth, padwidth) for _ in range(effective_dim))

    # If we're in 3D but have fewer effective dimensions, pad the tuple
    if metadata.dimensions == 3:
        npad = ((0, 0),) * (3 - effective_dim) + npad

    # Remove padding
    if padwidth > 0:
        slices = []
        for pad_start, pad_end in npad:
            end_slice = None if pad_end == 0 else -pad_end
            slices.append(slice(pad_start, end_slice))
        data = data[tuple(slices)]

    # Remove singleton dimensions
    if any(s == 1 for s in data.shape):
        data = data.reshape(tuple(s for s in data.shape if s != 1))

    return data


def _is_gas_variable(name: str) -> bool:
    """Check if field is a gas variable that needs unpadding"""
    return name in ["rho", "p", "v1", "v2", "v3", "chi"]


def preprocess_fields(
    raw_fields: dict[str, Array], mesh: MeshConfig, metadata: Metadata, unpad: bool
) -> dict[str, Array]:
    """Conditionally apply unpadding to all fields"""
    return {
        name: unpad_field(data, name, mesh, metadata) if unpad else data
        for name, data in raw_fields.items()
    }


def parse_bodies(
    bodies_group: dict[str, Any] | None,
    body_diagnostics: BodyDiagnostics | None = None,
) -> dict[str, Body] | None:
    """Parse body collection from HDF5 group data"""
    if not bodies_group or not body_diagnostics:
        return None

    parsed_bodies: dict[str, Body] = {}
    body_count = bodies_group.get("body_count", 0)

    for i in range(body_count):
        body_key = f"body_{i}"
        if body_key not in bodies_group:
            continue

        body_data = bodies_group[body_key]
        capabilities = BodyCapability(int(body_data["capabilities"]))

        # Create base body
        base = BaseBody(
            mass=float(body_data["mass"]),
            radius=float(body_data["radius"]),
            position=tuple(body_data["position"]),
            velocity=tuple(body_data["velocity"]),
            capabilities=capabilities,
        )

        # Build capability-specific properties
        gravitational = None
        if (
            has_capability(capabilities, BodyCapability.GRAVITATIONAL)
            and "softening_length" in body_data
        ):
            gravitational = GravitationalProperties(
                softening_length=float(body_data["softening_length"])
            )

        accretion = None
        if (
            has_capability(capabilities, BodyCapability.ACCRETION)
            and "sink_rate" in body_data
        ):
            accretion = AccretionProperties(
                sink_rate=float(body_data["sink_rate"]),
                accretion_radius=float(body_data["accretion_radius"]),
                total_accreted_mass=float(body_diagnostics.cumulative_mass_delta[i]),
                accretion_rate=float(body_diagnostics.accretion_rate[i]),
            )

        rigid = None
        if (
            has_capability(capabilities, BodyCapability.RIGID)
            and "inertia" in body_data
        ):
            rigid = RigidProperties(
                inertia=float(body_data["inertia"]),
                apply_no_slip=bool(body_data["apply_no_slip"]),
            )

        deformable = None
        if (
            has_capability(capabilities, BodyCapability.DEFORMABLE)
            and "yield_stress" in body_data
        ):
            deformable = DeformableProperties(
                yield_stress=float(body_data["yield_stress"]),
                plastic_strain=float(body_data["plastic_strain"]),
            )

        elastic = None
        if (
            has_capability(capabilities, BodyCapability.ELASTIC)
            and "elastic_modulus" in body_data
        ):
            elastic = ElasticProperties(
                elastic_modulus=float(body_data["elastic_modulus"]),
                poisson_ratio=float(body_data["poisson_ratio"]),
            )

        # Create the fused body
        body = Body(
            **asdict(base),
            gravitational=gravitational,
            accretion=accretion,
            rigid=rigid,
            deformable=deformable,
            elastic=elastic,
        )

        parsed_bodies[body_key] = body

    return parsed_bodies


def parse_body_system(
    bodies_group: dict[str, Any] | None, bodies: dict[str, Body]
) -> list[ImmersedBodyConfig] | GravitationalSystemConfig | None:
    """Parse a body system configuration from a collection of bodies."""
    from ..core.types.bodies import BinaryComponentConfig, BinaryConfig

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
            raise ValueError("Both bodies in a binary must have accretion info.")
        if body1.gravitational is None or body2.gravitational is None:
            raise ValueError("Both bodies in a binary must have gravitational info.")

        return GravitationalSystemConfig(
            prescribed_motion=True,
            reference_frame="inertial",
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
        return [
            ImmersedBodyConfig(
                capability=body.capabilities,
                mass=body.mass,
                radius=body.radius,
                position=body.position,
                velocity=body.velocity,
                two_way_coupling=False,
                force=(0.0, 0.0, 0.0),
                specifics={k: v for k, v in asdict(body).items() if v is not None},
            )
            for body in bodies.values()
        ]


def parse_diagnostics(groups: dict[str, Any]) -> BodyDiagnostics | None:
    """Parse body diagnostics from HDF5 groups"""
    if "diagnostics" not in groups:
        return None

    diag_data = groups["diagnostics"].get("body_diagnostics", {})
    if not diag_data:
        return None

    return BodyDiagnostics(
        force_components={
            key: np.asarray(value)
            for key, value in diag_data.items()
            if key.startswith("force_")
        },
        torque_components={
            key: np.asarray(value)
            for key, value in diag_data.items()
            if key.startswith("torque_")
        },
        cumulative_mass_delta=np.asarray(diag_data.get("cumulative_mass_delta", [])),
        accretion_rate=np.asarray(diag_data.get("accretion_rate", [])),
    )


def parse_data(raw: RawHDF5, unpad: bool = True) -> Result[ProcessedData]:
    try:
        # Parse metadata
        attrs = raw.attributes
        metadata = Metadata(
            time=float(attrs["time"]),
            dt=float(attrs["dt"]),
            iteration=int(attrs["iteration"]),
            dimensions=int(attrs["dimensions"]),
            regime=str(attrs["regime"]),
            adiabatic_index=float(attrs["adiabatic_index"]),
            is_mhd="mhd" in str(attrs["regime"]),
            coord_system=str(attrs["coord_system"]),
            boundary_conditions=tuple(str(attrs["boundary_conditions"]).split(",")),
            resolution=tuple(int(x) for x in str(attrs["resolution"]).split(",")),
            cfl_number=float(attrs["cfl_number"]),
            end_time=float(attrs["end_time"]),
            reconstruction=str(attrs["reconstruction"]),
            timestepping=str(attrs["timestepping"]),
            x1_spacing=str(attrs["x1_spacing"]),
            x2_spacing=str(attrs["x2_spacing"]),
            x3_spacing=str(attrs["x3_spacing"]),
            plm_theta=float(attrs["plm_theta"]),
            checkpoint_index=int(attrs["checkpoint_index"]),
            solver=str(attrs["solver"]),
            checkpoint_interval=int(attrs["checkpoint_interval"]),
            halo_radius=int(attrs["halo_radius"]),
        )

        # Parse mesh config
        mesh_data: dict[str, Any] = raw.groups.get("mesh_config", {})
        mesh = MeshConfig(
            shape=tuple(mesh_data["shape"]),
            bounds_min=tuple(mesh_data["bounds_min"]),
            bounds_max=tuple(mesh_data["bounds_max"]),
            halo_radius=int(mesh_data["halo_radius"]),
            spacing_types=tuple(mesh_data["spacing_types"].split(",")),
        )

        fields = preprocess_fields(raw.fields, mesh, metadata, unpad)

        # Bodies (if present)
        diagnostics = parse_diagnostics(raw.groups)
        raw_bodies = parse_bodies(raw.groups.get("bodies"), diagnostics)
        if raw_bodies is None:
            bodies = None
        else:
            bodies = parse_body_system(raw.groups.get("bodies"), raw_bodies)
        return Result.ok(
            ProcessedData(fields=fields, metadata=metadata, mesh=mesh, bodies=bodies)
        )
    except Exception as e:
        return Result.err(e)
