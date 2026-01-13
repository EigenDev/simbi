"""
Clean functional HDF5 reader for SIMBI v2.0 checkpoints.

design principles:
- composable parsers (small functions, one responsibility)
- result types (explicit error handling)
- immutable data (frozen dataclasses)
- type safety (full type hints)
- no legacy v1 support

structure mirrors C++ serialization exactly:
/metadata -> Metadata
/level_i/mesh -> MeshGeometry
/level_i/partition_j -> PartitionData
  /hydro/primitives -> primitive fields
  /hydro/magnetic -> face-centered B-fields
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from simbi.functional import Err, Ok, Result
from simbi.types.bodies import Body
from simbi.types.input import Metadata

# =============================================================================
# core types
# =============================================================================


@dataclass(frozen=True)
class Domain:
    """computational domain with ghost zones."""

    start: tuple[int, ...]
    fin: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, ...]:
        """domain extent (inclusive of ghosts)."""
        return tuple(f - s for s, f in zip(self.start, self.fin))

    @property
    def ndim(self) -> int:
        """number of dimensions."""
        return len(self.start)

    def interior(self, halo: int) -> "Domain":
        """remove ghost zones."""
        return Domain(
            start=tuple(s + halo for s in self.start),
            fin=tuple(f - halo for f in self.fin),
        )

    def __repr__(self) -> str:
        ranges = ":".join(f"{s}:{f}" for s, f in zip(self.start, self.fin))
        return f"Domain({ranges})"


@dataclass(frozen=True)
class FieldData:
    """field with computational domain (face or cell-centered)."""

    data: NDArray
    domain: Domain
    name: str

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def interior(self, halo: int) -> "FieldData":
        """extract interior (remove ghost zones)."""
        slices = tuple(
            slice(halo, -halo if halo > 0 else None) for _ in range(self.ndim)
        )
        return FieldData(
            data=self.data[slices],
            domain=self.domain.interior(halo),
            name=self.name,
        )


@dataclass(frozen=True)
class HydroFields:
    """hydrodynamic fields for one partition."""

    primitives: dict[str, FieldData]  # rho, p, v1, v2, v3, chi
    magnetic: Optional[dict[str, FieldData]]  # b1, b2, b3 (face-centered)

    @property
    def has_magnetic(self) -> bool:
        return self.magnetic is not None and len(self.magnetic) > 0


@dataclass(frozen=True)
class MeshGeometry:
    """mesh coordinate configuration."""

    dims: tuple[tuple[float, float], ...]  # [(x1min, x1max), ...]
    global_cells: tuple[int, ...]  # [nx3, nx2, nx1] in storage order
    spacing_types: tuple[str, ...]  # ["linear", "log", ...]
    metric: str  # "cartesian", "spherical", etc.
    halo_radius: int
    coordinate_system: str = "physical"  # "physical" or "comoving"
    scale_factor_a: float = 1.0  # a(t) for moving mesh
    scale_factor_adot: float = 0.0  # da/dt for moving mesh

    @property
    def ndim(self) -> int:
        return len(self.global_cells)

    @property
    def shape(self) -> tuple[int, ...]:
        """alias for global_cells for backward compatibility."""
        return self.global_cells

    @property
    def bounds_min(self) -> tuple[float, ...]:
        return tuple(d[0] for d in self.dims)

    @property
    def bounds_max(self) -> tuple[float, ...]:
        return tuple(d[1] for d in self.dims)


@dataclass(frozen=True)
class PartitionData:
    """data for one partition."""

    owned_domain: Domain
    hydro: HydroFields
    device_id: int
    partition_id: int


@dataclass(frozen=True)
class LevelData:
    """complete data for one refinement level."""

    level_id: int
    mesh: MeshGeometry
    partitions: list[PartitionData]

    @property
    def num_partitions(self) -> int:
        return len(self.partitions)


@dataclass(frozen=True)
class BodyCollection:
    """collection of bodies in the simulation."""

    count: int
    system_name: str
    reference_frame: str
    bodies: list[Body]
    binary_params: Optional[dict] = None


@dataclass(frozen=True)
class Checkpoint:
    """complete checkpoint state."""

    metadata: Metadata
    levels: list[LevelData]
    bodies: Optional[BodyCollection] = None

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def has_refinement(self) -> bool:
        return self.num_levels > 1

    def base_level(self) -> LevelData:
        """convenience accessor for level 0."""
        return self.levels[0]


# =============================================================================
# atomic parsers
# =============================================================================


def read_domain(group: h5py.Group) -> Result[Domain, str]:
    """read domain/start and domain/fin datasets."""
    try:
        if "domain" not in group:
            return Err("no domain subgroup found")

        dom = group["domain"]
        start = tuple(dom["start"][()])
        fin = tuple(dom["fin"][()])
        return Ok(Domain(start=start, fin=fin))
    except Exception as e:
        return Err(f"failed to read domain: {e}")


def read_scalar_field(
    parent: h5py.Group, name: str, py_name: Optional[str] = None
) -> Result[FieldData, str]:
    """read a scalar field dataset (primitives or similar)."""
    try:
        if name not in parent:
            return Err(f"field '{name}' not found")

        data = np.asarray(parent[name][()])

        # infer domain from shape (primitives don't have explicit domains)
        domain = Domain(start=tuple([0] * data.ndim), fin=tuple(data.shape))

        return Ok(FieldData(data=data, domain=domain, name=py_name or name))
    except Exception as e:
        return Err(f"failed to read field '{name}': {e}")


def read_face_field(
    mag_group: h5py.Group, cpp_name: str, py_name: str
) -> Result[FieldData, str]:
    """read face-centered magnetic field (has domain and data subgroups)."""
    try:
        if cpp_name not in mag_group:
            return Err(f"magnetic field '{cpp_name}' not found")

        b_group = mag_group[cpp_name]

        # read domain
        domain_result = read_domain(b_group)
        if domain_result.is_err():
            return Err(
                f"failed to read domain for {cpp_name}: {domain_result.error}"
            )

        # read data
        if "data" not in b_group:
            return Err(f"no data dataset in {cpp_name}")

        data = np.asarray(b_group["data"][()])

        return Ok(
            FieldData(data=data, domain=domain_result.value, name=py_name)
        )
    except Exception as e:
        return Err(f"failed to read magnetic field '{cpp_name}': {e}")


def read_primitives(
    hydro_group: h5py.Group,
) -> Result[dict[str, FieldData], str]:
    """read all primitive fields (cell-centered)."""
    if "primitives" not in hydro_group:
        return Err("no primitives group found")

    prim_group = hydro_group["primitives"]

    # C++ names → Python names
    field_map = {
        "rho": "rho",
        "pre": "p",
        "chi": "chi",
        "v1": "v1",
        "v2": "v2",
        "v3": "v3",
    }

    fields = {}
    for cpp_name, py_name in field_map.items():
        if cpp_name in prim_group:
            field_result = read_scalar_field(prim_group, cpp_name, py_name)
            if field_result.is_ok():
                fields[py_name] = field_result.value

    return Ok(fields) if fields else Err("no primitive fields found")


def read_magnetic_fields(
    hydro_group: h5py.Group,
) -> Result[Optional[dict[str, FieldData]], str]:
    """read face-centered magnetic fields."""
    if "magnetic" not in hydro_group:
        return Ok(None)  # not MHD, no error

    mag_group = hydro_group["magnetic"]

    # read B1, B2, B3 (face-centered with explicit domains)
    field_map = {"B1": "b1", "B2": "b2", "B3": "b3"}

    fields = {}
    for cpp_name, py_name in field_map.items():
        if cpp_name in mag_group:
            field_result = read_face_field(mag_group, cpp_name, py_name)
            if field_result.is_ok():
                fields[py_name] = field_result.value

    return Ok(fields) if fields else Ok(None)


def read_partition(
    part_group: h5py.Group, partition_id: int
) -> Result[PartitionData, str]:
    """read complete partition data."""
    # read owned domain
    try:
        owned_start = tuple(part_group["owned_start"][()])
        owned_fin = tuple(part_group["owned_fin"][()])
        owned_domain = Domain(start=owned_start, fin=owned_fin)
    except Exception as e:
        return Err(f"failed to read owned domain: {e}")

    # read hydro
    if "hydro" not in part_group:
        return Err("no hydro group in partition")

    hydro_group = part_group["hydro"]

    prims_result = read_primitives(hydro_group)
    if prims_result.is_err():
        return Err(f"failed to read primitives: {prims_result.error}")

    mag_result = read_magnetic_fields(hydro_group)
    if mag_result.is_err():
        return Err(f"failed to read magnetic fields: {mag_result.error}")

    hydro = HydroFields(
        primitives=prims_result.value, magnetic=mag_result.value
    )

    device_id = int(part_group.attrs.get("device_id", 0))

    return Ok(
        PartitionData(
            owned_domain=owned_domain,
            hydro=hydro,
            device_id=device_id,
            partition_id=partition_id,
        )
    )


def read_mesh_geometry(mesh_group: h5py.Group, level_group: h5py.Group = None) -> Result[MeshGeometry, str]:
    """read mesh configuration."""
    try:
        global_cells = tuple(mesh_group["global_cells"][()])
        halo_radius = int(mesh_group.attrs.get("halo_width", 2))

        # read geometry
        if "geometry" not in mesh_group:
            return Err("no geometry subgroup")

        geo = mesh_group["geometry"]
        metric_val = geo.attrs.get("metric", "cartesian")
        metric = (
            metric_val.decode("utf-8")
            if isinstance(metric_val, bytes)
            else str(metric_val)
        )

        # read dimensions
        rank = len(global_cells)
        dims = []
        spacing_types = []

        for dd in range(rank):
            dim_group = geo[f"dim_{dd}"]
            start = float(dim_group.attrs["start"])
            end = float(dim_group.attrs["end"])
            dims.append((start, end))

            spacing_type_val = dim_group.attrs.get("type", "linear")
            spacing_types.append(
                spacing_type_val.decode("utf-8")
                if isinstance(spacing_type_val, bytes)
                else str(spacing_type_val)
            )

        # read motion state from level group if present
        coordinate_system = "physical"
        scale_factor_a = 1.0
        scale_factor_adot = 0.0

        if level_group is not None:
            coord_sys_val = level_group.attrs.get("coordinate_system")
            if coord_sys_val:
                coordinate_system = (
                    coord_sys_val.decode("utf-8")
                    if isinstance(coord_sys_val, bytes)
                    else str(coord_sys_val)
                )
            scale_factor_a = float(level_group.attrs.get("scale_factor_a", 1.0))
            scale_factor_adot = float(level_group.attrs.get("scale_factor_adot", 0.0))

        return Ok(
            MeshGeometry(
                dims=tuple(dims),
                global_cells=global_cells,
                spacing_types=tuple(spacing_types),
                metric=metric,
                halo_radius=halo_radius,
                coordinate_system=coordinate_system,
                scale_factor_a=scale_factor_a,
                scale_factor_adot=scale_factor_adot,
            )
        )
    except Exception as e:
        return Err(f"failed to read mesh geometry: {e}")


def read_level(
    level_group: h5py.Group, level_id: int
) -> Result[LevelData, str]:
    """read complete level data."""
    # read mesh
    if "mesh" not in level_group:
        return Err(f"no mesh in level_{level_id}")

    mesh_result = read_mesh_geometry(level_group["mesh"], level_group)
    if mesh_result.is_err():
        return Err(
            f"failed to read mesh for level_{level_id}: {mesh_result.error}"
        )

    # read all partitions
    partitions = []
    partition_id = 0
    while f"partition_{partition_id}" in level_group:
        part_result = read_partition(
            level_group[f"partition_{partition_id}"], partition_id
        )
        if part_result.is_ok():
            partitions.append(part_result.value)
        else:
            # log warning but continue
            pass
        partition_id += 1

    if not partitions:
        return Err(f"no valid partitions in level_{level_id}")

    return Ok(
        LevelData(
            level_id=level_id,
            mesh=mesh_result.value,
            partitions=partitions,
        )
    )


def read_metadata(meta_group: h5py.Group) -> Result[Metadata, str]:
    """read simulation metadata."""
    try:
        attrs = meta_group.attrs

        # helper to decode bytes to str
        def decode_str(val):
            return val.decode("utf-8") if isinstance(val, bytes) else str(val)

        # read boundary conditions if present
        bcs = ()
        if "boundary_conditions" in meta_group:
            bc_group = meta_group["boundary_conditions"]
            num_bcs = len(
                [k for k in bc_group.attrs.keys() if k.startswith("bc_")]
            )
            bcs = tuple(
                decode_str(bc_group.attrs[f"bc_{i}"]) for i in range(num_bcs)
            )

        return Ok(
            Metadata(
                # time
                time=float(attrs["time"]),
                dt=float(attrs["dt"]),
                tend=float(attrs["tend"]),
                iteration=int(attrs["iteration"]),
                checkpoint_index=int(attrs["checkpoint_index"]),
                # physics
                gamma=float(attrs["gamma"]),
                cfl=float(attrs["cfl"]),
                plm_theta=float(attrs["plm_theta"]),
                viscosity=float(attrs.get("viscosity", 0.0)),
                # domain
                dimensions=int(attrs["dimensions"]),
                coord_system=decode_str(attrs["coord_system"]),
                halo_radius=int(attrs["halo_radius"]),
                # flags
                is_mhd=bool(attrs["is_mhd"]),
                is_relativistic=bool(attrs.get("is_relativistic", False)),
                # enums
                regime=decode_str(attrs["regime"]),
                solver=decode_str(attrs["solver"]),
                reconstruction=decode_str(attrs["reconstruction"]),
                timestepping=decode_str(attrs["timestepping"]),
                # optional
                checkpoint_interval=float(
                    attrs.get("checkpoint_interval", 0.0)
                ),
                x1_spacing=decode_str(attrs.get("x1_spacing", "linear")),
                x2_spacing=decode_str(attrs.get("x2_spacing", "linear")),
                x3_spacing=decode_str(attrs.get("x3_spacing", "linear")),
                boundary_conditions=bcs,
            )
        )
    except Exception as e:
        return Err(f"failed to read metadata: {e}")


def read_bodies(bodies_group: h5py.Group) -> Result[BodyCollection, str]:
    """read body collection."""
    from simbi.types.bodies import (
        AccretionProperties,
        Body,
        BodyCapability,
        GravitationalProperties,
        RigidProperties,
    )

    try:
        count = int(bodies_group.attrs["count"])
        system_name = (
            bodies_group.attrs["system_name"].decode("utf-8")
            if isinstance(bodies_group.attrs["system_name"], bytes)
            else str(bodies_group.attrs["system_name"])
        )
        reference_frame = (
            bodies_group.attrs["reference_frame"].decode("utf-8")
            if isinstance(bodies_group.attrs["reference_frame"], bytes)
            else str(bodies_group.attrs["reference_frame"])
        )

        # read binary params if present
        binary_params = None
        if "binary_params" in bodies_group:
            bp = bodies_group["binary_params"]
            binary_params = {
                "total_mass": float(bp.attrs["total_mass"]),
                "semi_major": float(bp.attrs["semi_major"]),
                "eccentricity": float(bp.attrs["eccentricity"]),
                "mass_ratio": float(bp.attrs["mass_ratio"]),
                "orbital_period": float(bp.attrs["orbital_period"]),
                "is_circular_orbit": bool(bp.attrs["is_circular_orbit"]),
                "prescribed_motion": bool(bp.attrs["prescribed_motion"]),
            }

        # read individual bodies
        bodies = []
        for ii in range(count):
            body_key = f"body_{ii}"
            if body_key not in bodies_group:
                continue

            bg = bodies_group[body_key]

            # core properties
            mass = float(bg.attrs["mass"])
            radius = float(bg.attrs["radius"])
            capabilities = BodyCapability(int(bg.attrs["capabilities"]))
            position = tuple(bg["position"][()])
            velocity = tuple(bg["velocity"][()])
            force = tuple(bg["force"][()])
            torque = tuple(bg["torque"][()])

            # capability-specific data
            gravitational = None
            if "gravitational" in bg:
                grav_g = bg["gravitational"]
                gravitational = GravitationalProperties(
                    softening_length=float(grav_g.attrs["softening_length"])
                )

            accretion = None
            if "accretion" in bg:
                accr_g = bg["accretion"]
                accretion = AccretionProperties(
                    sink_rate=float(accr_g.attrs["sink_rate"]),
                    accretion_radius=float(accr_g.attrs["accretion_radius"]),
                    total_accreted_mass=float(
                        accr_g.attrs["total_accreted_mass"]
                    ),
                    accretion_rate=float(accr_g.attrs["accretion_rate"]),
                    sink_delta=float(accr_g.attrs["sink_delta"]),
                )

            rigid = None
            if "rigid" in bg:
                rigid_g = bg["rigid"]
                rigid = RigidProperties(
                    inertia=float(rigid_g.attrs["inertia"]),
                    apply_no_slip=bool(rigid_g.attrs["apply_no_slip"]),
                )

            body = Body(
                mass=mass,
                radius=radius,
                position=position,
                velocity=velocity,
                force=force,
                torque=torque,
                capabilities=capabilities,
                gravitational=gravitational,
                accretion=accretion,
                rigid=rigid,
            )
            bodies.append(body)

        return Ok(
            BodyCollection(
                count=count,
                system_name=system_name,
                reference_frame=reference_frame,
                bodies=bodies,
                binary_params=binary_params,
            )
        )
    except Exception as e:
        return Err(f"failed to read bodies: {e}")


# =============================================================================
# top-level reader
# =============================================================================


def read_checkpoint(filename: str) -> Result[Checkpoint, str]:
    """read complete checkpoint file (v2.0 only)."""
    try:
        with h5py.File(filename, "r") as f:
            # check version
            version = f.attrs.get("format_version", "unknown")
            # handle bytes or string
            version_str = (
                version.decode("utf-8")
                if isinstance(version, bytes)
                else str(version)
            )
            if not version_str.startswith("2"):
                return Err(
                    f"unsupported format version: {version_str}. "
                    "only v2.0 is supported. use legacy reader for v1.0."
                )

            # read metadata
            if "metadata" not in f:
                return Err("no metadata group")

            meta_result = read_metadata(f["metadata"])
            if meta_result.is_err():
                return Err(f"failed to read metadata: {meta_result.error}")

            # read all levels
            levels = []
            level_id = 0
            while f"level_{level_id}" in f:
                level_result = read_level(f[f"level_{level_id}"], level_id)
                if level_result.is_ok():
                    levels.append(level_result.value)
                else:
                    # log warning but try next level
                    pass
                level_id += 1

            if not levels:
                return Err("no valid levels found")

            # read bodies if present
            bodies = None
            if "bodies" in f:
                bodies_result = read_bodies(f["bodies"])
                if bodies_result.is_ok():
                    bodies = bodies_result.value

            return Ok(
                Checkpoint(
                    metadata=meta_result.value,
                    levels=levels,
                    bodies=bodies,
                )
            )

    except FileNotFoundError:
        return Err(f"file not found: {filename}")
    except Exception as e:
        return Err(f"failed to read checkpoint '{filename}': {e}")


# =============================================================================
# convenience accessors
# =============================================================================


def get_base_fields(
    checkpoint: Checkpoint, unpad: bool = True
) -> dict[str, NDArray]:
    """
    extract base-level primitive fields as simple dict[str, array].

    useful for visualization and analysis that doesn't need partition info.
    assumes single partition at level 0.
    """
    base = checkpoint.base_level()
    if base.num_partitions != 1:
        raise ValueError(
            f"get_base_fields requires single partition, "
            f"got {base.num_partitions}"
        )

    partition = base.partitions[0]
    halo = base.mesh.halo_radius if unpad else 0

    fields = {}
    for name, field in partition.hydro.primitives.items():
        if unpad:
            fields[name] = field.interior(halo).data
        else:
            fields[name] = field.data

    # add face-centered magnetic fields if present
    if partition.hydro.has_magnetic:
        for name, field in partition.hydro.magnetic.items():
            if unpad:
                fields[name] = field.interior(halo).data
            else:
                fields[name] = field.data

    return fields
