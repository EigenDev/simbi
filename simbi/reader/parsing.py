from dataclasses import asdict
from typing import Any

from simbi.core.types.bodies import BodyCapability
from ..core.types import (
    RawHDF5,
    ProcessedData,
    Metadata,
    MeshConfig,
    BodyDiagnostics,
    Body,
    GravitationalBody,
    AccretionBody,
    RigidBody,
    DeformableBody,
    ElasticBody,
    BaseBody,
)
from ..functional.result import Result
from numpy.typing import NDArray
import numpy as np

Array = NDArray[np.floating]


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
    raw_fields: dict[str, Array], mesh: MeshConfig, metadata: Metadata
) -> dict[str, Array]:
    """Apply unpadding to all fields"""
    return {
        name: unpad_field(data, name, mesh, metadata)
        for name, data in raw_fields.items()
    }


def parse_bodies(
    bodies_group: dict[str, Any] | None, body_diagnostics: BodyDiagnostics | None = None
) -> dict[str, Body] | None:
    """Parse body collection from HDF5 group data"""
    if not bodies_group:
        return None

    if not body_diagnostics:
        return None

    parsed_bodies: dict[str, Any] = {}

    # Get body count from attributes
    body_count = bodies_group.get("body_count", 0)

    for i in range(body_count):
        body_key = f"body_{i}"
        if body_key in bodies_group:
            body_data = bodies_group[body_key]

            bod = BaseBody(
                mass=float(body_data["mass"]),
                radius=float(body_data["radius"]),
                position=tuple(body_data["position"]),
                velocity=tuple(body_data["velocity"]),
                capabilities=BodyCapability(int(body_data["capabilities"])),
            )

            if "softening_length" in body_data:
                bod = GravitationalBody(
                    **asdict(bod), softening_length=float(body_data["softening_length"])
                )

            if "accretion_efficiency" in body_data:
                bod = AccretionBody(
                    **asdict(bod),
                    accretion_efficiency=float(body_data["accretion_efficiency"]),
                    accretion_radius=float(body_data["accretion_radius"]),
                    total_accreted_mass=float(body_diagnostics.accreted_mass[i]),
                    accretion_rate=float(body_diagnostics.accretion_rate[i]),
                )

            if "inertia" in body_data:
                bod = RigidBody(
                    **asdict(bod),
                    inertia=float(body_data["inertia"]),
                    apply_no_slip=bool(body_data["apply_no_slip"]),
                )

            if "yield_stress" in body_data:
                bod = DeformableBody(
                    **asdict(bod),
                    yield_stress=float(body_data["yield_stress"]),
                    plastic_strain=float(body_data["plastic_strain"]),
                )

            if "elastic_modulus" in body_data:
                bod = ElasticBody(
                    **asdict(bod),
                    elastic_modulus=float(body_data["elastic_modulus"]),
                    poisson_ratio=float(body_data["poisson_ratio"]),
                )

            parsed_bodies[body_key] = bod

    return parsed_bodies


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
        total_mass=np.asarray(diag_data.get("total_mass", [])),
        accreted_mass=np.asarray(diag_data.get("accreted_mass", [])),
        accretion_rate=np.asarray(diag_data.get("accretion_rate", [])),
    )


def parse_data(raw: RawHDF5) -> Result[ProcessedData]:
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

        fields = preprocess_fields(raw.fields, mesh, metadata)

        # Bodies (if present)
        diagnostics = parse_diagnostics(raw.groups)
        bodies = parse_bodies(raw.groups.get("bodies"), diagnostics)
        return Result.ok(
            ProcessedData(fields=fields, metadata=metadata, mesh=mesh, bodies=bodies)
        )
    except Exception as e:
        return Result.err(e)
