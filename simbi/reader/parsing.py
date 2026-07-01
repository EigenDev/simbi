# =============================================================================
# parsing.py
#
# parses raw hdf5 data into structured ProcessedData.
# handles metadata extraction, field mapping, and body reconstruction.
# =============================================================================
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from simbi.types.bodies import BodyCapability
from simbi.types.input import normalize_regime

from ..functional.result import Result
from ..types import (
    AccretionProperties,
    BaseBody,
    Body,
    BodyDiagnostics,
    DeformableProperties,
    ElasticProperties,
    GravitationalProperties,
    HierarchyData,
    LevelData,
    MeshConfig,
    Metadata,
    ProcessedData,
    RawHDF5,
    RigidProperties,
)

Array = NDArray[np.floating]


def has_capability(
    body_capability: BodyCapability, capability: BodyCapability
) -> bool:
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

    # calculate effective dimensions (non-unity dimensions)
    inactive_dimensions = sum(1 for x in mesh.shape if x == 1)
    effective_dim = metadata.dimensions - inactive_dimensions

    # remove inactive dimensions that have shape 1 + 2*padwidth
    data = field_data
    if any(x == 1 + 2 * padwidth for x in data.shape):
        slices: list[slice] = list(
            slice(padwidth, -padwidth) if x == 1 + 2 * padwidth else slice(None)
            for x in data.shape
        )
        data = data[tuple(slices)]

    # create padding specification for effective dimensions
    npad = tuple((padwidth, padwidth) for _ in range(effective_dim))

    # if we're in 3D but have fewer effective dimensions, pad the tuple
    if metadata.dimensions == 3:
        npad = ((0, 0),) * (3 - effective_dim) + npad

    # remove padding
    if padwidth > 0:
        slices = []
        for pad_start, pad_end in npad:
            end_slice = None if pad_end == 0 else -pad_end
            slices.append(slice(pad_start, end_slice))
        data = data[tuple(slices)]

    # remove singleton dimensions
    if any(s == 1 for s in data.shape):
        data = data.reshape(tuple(s for s in data.shape if s != 1))

    return data


def _is_gas_variable(name: str) -> bool:
    """Check if field is a gas variable that needs unpadding"""
    return name in ["rho", "p", "v1", "v2", "v3", "chi"]


def preprocess_fields(
    raw_fields: dict[str, Array],
    mesh: MeshConfig,
    metadata: Metadata,
    unpad: bool,
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

        # create base body
        base = BaseBody(
            mass=float(body_data["mass"]),
            radius=float(body_data["radius"]),
            position=tuple(body_data["position"]),
            velocity=tuple(body_data["velocity"]),
            capabilities=capabilities,
        )

        # build capability-specific properties
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
                total_accreted_mass=float(
                    body_diagnostics.cumulative_mass_delta[i]
                ),
                accretion_rate=float(body_diagnostics.accretion_rate[i]),
                sink_delta=float(body_data.get("sink_delta", 1.0)),
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

        # create the fused body
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


def parse_diagnostics(groups: dict[str, Any]) -> BodyDiagnostics | None:
    """Parse body diagnostics from HDF5 groups"""
    if "bodies" not in groups:
        return None

    if "diagnostics" not in groups["bodies"]:
        return None

    diag_data = groups["bodies"]["diagnostics"]
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
        cumulative_mass_delta=np.asarray(
            [
                value
                for key, value in diag_data.items()
                if key.startswith("cumulative_mass_delta")
            ]
        ),
        accretion_rate=np.asarray(
            [
                value
                for key, value in diag_data.items()
                if key.startswith("accretion_rate")
            ]
        ),
    )


def parse_hierarchy(groups: dict[str, Any]) -> Optional[HierarchyData]:
    """Parse refinement hierarchy information from HDF5 groups"""
    if "hierarchy" not in groups:
        return None

    hierarchy = groups["hierarchy"]
    num_levels = int(hierarchy.get("num_levels", 1))

    if num_levels == 1:
        return None

    ref_ratios = []
    if "refinement_ratios" in hierarchy:
        ref_ratios = [int(r) for r in hierarchy["refinement_ratios"]]

    return HierarchyData(
        num_levels=num_levels,
        levels=[],
        ref_ratios=ref_ratios,
    )


def parse_level_data(
    level_id: int,
    level_group: dict[str, Any],
    metadata: Metadata,
    unpad: bool = True,
) -> LevelData:
    """Parse single level data from HDF5 group"""
    # parse mesh config for this level
    mesh_data = level_group.get("mesh", {})
    mesh = MeshConfig(
        shape=tuple(mesh_data["shape"]),
        bounds_min=tuple(mesh_data["bounds_min"])[::-1],
        bounds_max=tuple(mesh_data["bounds_max"])[::-1],
        halo_radius=int(mesh_data["halo_radius"]),
        spacing_types=tuple(mesh_data["spacing_types"].split(",")[::-1]),
    )

    # get fields for this level
    fields: dict[str, Array] = {}
    for key in ["rho", "p", "v1", "v2", "v3", "chi", "b1", "b2", "b3"]:
        if key in level_group:
            field_data = np.asarray(level_group[key])
            if unpad:
                field_data = unpad_field(field_data, key, mesh, metadata)
            fields[key] = field_data

    # get refinement ratio if not finest level
    ref_ratio = None
    if f"level_{level_id}" in level_group:
        next_level = level_group[f"level_{level_id}"]
        if "ref_ratio" in next_level:
            ref_ratio = int(next_level["ref_ratio"])

    return LevelData(
        level_id=level_id, mesh=mesh, fields=fields, ref_ratio=ref_ratio
    )


def get_attr(attrs: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get attribute by trying multiple possible keys"""
    for key in keys:
        if key in attrs:
            return attrs[key]
    return default


def parse_mesh_config_v2(
    groups: dict[str, Any], attrs: dict[str, Any]
) -> MeshConfig:
    """Parse mesh config from v2 format"""
    mesh_data = groups.get("mesh_config", {})

    # try to get shape from mesh_config or from geometry info
    shape = mesh_data.get("global_cells", mesh_data.get("shape"))
    if shape is None:
        # fall back to resolution from metadata
        res = attrs.get("resolution", (1, 1, 1))
        if isinstance(res, tuple):
            shape = res
        else:
            shape = tuple(res)

    # get bounds from geometry.dims if available
    bounds_min = []
    bounds_max = []

    if "geometry" in mesh_data:
        geo = mesh_data["geometry"]
        # filter out all keys that start with dim_x
        dims = {
            key: value for key, value in geo.items() if key.startswith("dim_")
        }
        for dim in dims.values():
            bounds_min.append(dim.get("start", 0.0))
            bounds_max.append(dim.get("end", 1.0))

    # we prepend default bounds for inactive dimensions
    while len(bounds_min) < 3:
        bounds_min.insert(0, 0.0)
        bounds_max.insert(0, 1.0)

    halo_radius = int(mesh_data.get("halo_width", attrs.get("halo_radius", 0)))

    # spacing types from metadata
    spacing_types = (
        str(attrs.get("x1_spacing", "linear")),
        str(attrs.get("x2_spacing", "linear")),
        str(attrs.get("x3_spacing", "linear")),
    )

    return MeshConfig(
        shape=tuple(int(s) for s in shape),
        bounds_min=tuple(bounds_min)[::-1],
        bounds_max=tuple(bounds_max)[::-1],
        halo_radius=halo_radius,
        spacing_types=spacing_types,
    )


def parse_metadata_v2(attrs: dict[str, Any]) -> Metadata:
    """Parse metadata from v2 format attributes"""
    # handle resolution - could be tuple or need parsing
    resolution = attrs.get("resolution", (1, 1, 1))
    if isinstance(resolution, str):
        resolution = tuple(int(x) for x in resolution.split(","))
    elif not isinstance(resolution, tuple):
        resolution = tuple(int(x) for x in resolution)

    # handle boundary conditions
    bcs = attrs.get("boundary_conditions", ())
    if isinstance(bcs, str):
        bcs = tuple(bcs.split(","))

    return Metadata(
        time=float(attrs.get("time", 0.0)),
        dt=float(attrs.get("dt", 0.0)),
        iteration=int(attrs.get("iteration", 0)),
        dimensions=int(attrs.get("dimensions", 1)),
        regime=normalize_regime(str(attrs.get("regime", "newtonian"))),
        adiabatic_index=float(attrs.get("gamma", 1.4)),
        is_mhd=bool(attrs.get("is_mhd", False)),
        coord_system=str(attrs.get("coord_system", "cartesian")),
        boundary_conditions=bcs,
        resolution=resolution,
        cfl_number=float(attrs.get("cfl", 0.4)),
        end_time=float(attrs.get("tend", 1.0)),
        reconstruction=str(attrs.get("reconstruction", "plm")),
        timestepping=str(attrs.get("timestepping", "rk2")),
        x1_spacing=str(attrs.get("x1_spacing", "linear")),
        x2_spacing=str(attrs.get("x2_spacing", "linear")),
        x3_spacing=str(attrs.get("x3_spacing", "linear")),
        plm_theta=float(attrs.get("plm_theta", 1.5)),
        checkpoint_index=int(attrs.get("checkpoint_index", 0)),
        solver=str(attrs.get("solver", "hllc")),
        checkpoint_interval=float(attrs.get("checkpoint_interval", 0.1)),
        halo_radius=int(attrs.get("halo_radius", 2)),
        system_info=None,
    )


def parse_data(
    raw: RawHDF5, unpad: bool = True
) -> Result[ProcessedData, Exception]:
    """Parse raw HDF5 data into structured format"""
    print("Parsing raw data into structured ProcessedData...")
    try:
        attrs = raw.attributes

        # detect format by checking for v2 attribute names
        is_v2 = "gamma" in attrs or "tend" in attrs

        if is_v2:
            metadata = parse_metadata_v2(attrs)
            mesh = parse_mesh_config_v2(raw.groups, attrs)
        else:
            # legacy v1 format parsing
            metadata = Metadata(
                time=float(attrs["time"]),
                dt=float(attrs["dt"]),
                iteration=int(attrs["iteration"]),
                dimensions=int(attrs["dimensions"]),
                regime=normalize_regime(str(attrs["regime"])),
                adiabatic_index=float(attrs["adiabatic_index"]),
                is_mhd="mhd" in str(attrs["regime"]),
                coord_system=str(attrs["coord_system"]),
                boundary_conditions=tuple(
                    str(attrs["boundary_conditions"]).split(",")
                ),
                resolution=tuple(
                    int(x) for x in str(attrs["resolution"]).split(",")
                ),
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
                system_info=raw.groups.get("bodies"),
            )

            mesh_data: dict[str, Any] = raw.groups.get("mesh_config", {})
            mesh = MeshConfig(
                shape=tuple(mesh_data["shape"]),
                bounds_min=tuple(mesh_data["bounds_min"]),
                bounds_max=tuple(mesh_data["bounds_max"]),
                halo_radius=int(mesh_data["halo_radius"]),
                spacing_types=tuple(mesh_data["spacing_types"].split(",")),
            )

        # get base level fields
        fields = preprocess_fields(raw.fields, mesh, metadata, unpad)

        # parse refinement hierarchy if present
        hierarchy = parse_hierarchy(raw.groups)

        # parse additional levels if we have refinement data
        levels = None
        if hierarchy:
            levels = []
            level_id = 1
            while f"level_{level_id}" in raw.groups:
                level_group = raw.groups[f"level_{level_id}"]
                level = parse_level_data(level_id, level_group, metadata, unpad)
                levels.append(level)
                level_id += 1

        # bodies (if present)
        diagnostics = parse_diagnostics(raw.groups)
        bodies = parse_bodies(raw.groups.get("bodies"), diagnostics)

        return Result.ok(
            ProcessedData(
                fields=fields,
                metadata=metadata,
                mesh=mesh,
                hierarchy=hierarchy,
                levels=levels,
                bodies=bodies,
            )
        )
    except Exception as e:
        return Result.err(e)
