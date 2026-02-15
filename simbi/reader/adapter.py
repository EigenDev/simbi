# =============================================================================
# adapter.py
#
# adapter to make io.Checkpoint compatible with visualization pipeline.
# provides the interface viz expects without the old lazy evaluation system.
# =============================================================================

import numpy as np
from numpy.typing import NDArray

from simbi.reader.computation import create_computation_pipeline

from .io import Checkpoint, MeshGeometry, get_base_fields


class MeshAdapter:
    """adapter to add coordinate arrays to MeshGeometry."""

    def __init__(self, mesh: MeshGeometry, owned_domain=None):
        self._mesh = mesh
        self._owned_domain = owned_domain
        self._coords_cache = {}

    def _generate_coords(self, axis: int) -> NDArray:
        """generate vertex coordinates for an axis (physical coordinates)."""
        if axis in self._coords_cache:
            return self._coords_cache[axis]

        # get comoving bounds and hypothetical full-domain cell count
        comoving_xmin, comoving_xmax = self._mesh.dims[axis]
        global_cells = self._mesh.global_cells[axis]
        spacing_type = self._mesh.spacing_types[axis]

        # if we have owned_domain, compute actual patch bounds
        if self._owned_domain is not None:
            # owned_domain indices in global coordinate system
            start_idx = self._owned_domain[0][axis]  # owned_start
            end_idx = self._owned_domain[1][axis]  # owned_fin

            # compute actual number of cells in this patch
            ncells = end_idx - start_idx

            # compute patch comoving bounds from global coordinates
            dx = (comoving_xmax - comoving_xmin) / global_cells
            xmin = comoving_xmin + start_idx * dx
            xmax = comoving_xmin + end_idx * dx
        else:
            # base level: use comoving bounds and global cells
            xmin, xmax = comoving_xmin, comoving_xmax
            ncells = global_cells

        # generate comoving coordinates based on spacing type
        if spacing_type == "linear":
            coords_comoving = np.linspace(xmin, xmax, ncells + 1)
        elif spacing_type == "log":
            coords_comoving = np.logspace(
                np.log10(xmin), np.log10(xmax), ncells + 1
            )
        else:
            # default to linear
            coords_comoving = np.linspace(xmin, xmax, ncells + 1)

        # apply scale factor for moving mesh: r_phys = a(t) * r_comoving
        coords_physical = coords_comoving * self._mesh.scale_factor_a

        self._coords_cache[axis] = coords_physical
        return coords_physical

    @property
    def x1v(self) -> NDArray:
        """x1 vertex coordinates (fastest varying, rightmost in storage)."""
        # x1 is the last dimension in storage order (nz, ny, nx)
        axis = self._mesh.ndim - 1
        return self._generate_coords(axis)

    @property
    def x2v(self) -> NDArray:
        """x2 vertex coordinates (middle dimension)."""
        if self._mesh.ndim < 2:
            return np.array([0.0, 1.0])
        # x2 is the second-to-last dimension
        axis = self._mesh.ndim - 2
        return self._generate_coords(axis)

    @property
    def x3v(self) -> NDArray:
        """x3 vertex coordinates (slowest varying, leftmost in storage)."""
        if self._mesh.ndim < 3:
            return np.array([0.0, 1.0])
        # x3 is the first dimension in storage order
        return self._generate_coords(0)

    @property
    def shape(self):
        """pass through shape."""
        return self._mesh.shape

    def __getattr__(self, name):
        """forward other attributes to underlying mesh."""
        return getattr(self._mesh, name)


class SimData:
    """
    adapter wrapping Checkpoint for visualization compatibility.

    provides dict-like access to fields and metadata properties
    that the viz pipeline expects.

    additions:
    - caching of derived field evaluations per (field, level)
    - fast membership checks via available_fields()
    - listing of available derived fields
    """

    def __init__(self, checkpoint: Checkpoint):
        self._checkpoint = checkpoint
        self._base_fields = None  # lazy load on first access
        self._mesh_adapters = {}  # cache mesh adapters per level

        # derived fields pipeline (objects exposing evaluate(level) -> array)
        self._derived_fields = create_computation_pipeline(checkpoint)

        # cache derived field names for quick membership tests
        # pipeline values may be callables/objects; keys give names
        self._derived_names = set(self._derived_fields.keys())

        # cache for evaluated derived fields: (field_name, level) -> ndarray
        self._derived_cache: dict[tuple[str, int], NDArray] = {}

    @property
    def checkpoint(self) -> Checkpoint:
        """access underlying Checkpoint for advanced usage."""
        return self._checkpoint

    @property
    def metadata(self):
        """access metadata."""
        return self._checkpoint.metadata

    @property
    def mesh(self):
        """access base level mesh with coordinate arrays."""
        if 0 not in self._mesh_adapters:
            self._mesh_adapters[0] = MeshAdapter(
                self._checkpoint.base_level().mesh
            )
        return self._mesh_adapters[0]

    @property
    def num_levels(self) -> int:
        """number of refinement levels."""
        return self._checkpoint.num_levels

    def has_refinement(self) -> bool:
        """check if AMR is present."""
        return self._checkpoint.has_refinement

    @property
    def body_collection(self):
        """access bodies (if present)."""
        return self._checkpoint.bodies

    def hierarchy(self):
        """access AMR hierarchy info."""
        return self._checkpoint.hierarchy

    def list_derived_fields(self) -> list[str]:
        """return sorted list of available derived field names."""
        return sorted(self._derived_names)

    def available_fields(self, level: int = 0) -> set[str]:
        """
        return set of available field names at a given level.

        includes base primitive fields (unpad) and derived field names.
        """
        if level >= self.num_levels:
            raise ValueError(
                f"level {level} doesn't exist (num_levels={self.num_levels})"
            )

        # base primitives (single-partition assumption)
        level_data = self._checkpoint.levels[level]
        if level_data.num_partitions != 1:
            # fall back to derived names only if partitioning is complex
            return set(self._derived_names)

        partition = level_data.partitions[0]
        halo = level_data.mesh.halo_radius

        base_names = set(partition.hydro.primitives.keys())

        # also expose face-centered magnetic names if present
        if partition.hydro.magnetic is not None:
            base_names.update(partition.hydro.magnetic.keys())

        return base_names.union(self._derived_names)

    def get_field(
        self, field_name: str, level: int = 0, crop_to_owned: bool = False
    ) -> NDArray:
        """
        get a field by name from a specific level.

        handles both primitive fields and derived fields like b1_mean.
        stitches together multi-partition data.

        Uses caching for derived field evaluations.
        """
        if level >= self.num_levels:
            raise ValueError(
                f"level {level} doesn't exist (num_levels={self.num_levels})"
            )

        level_data = self._checkpoint.levels[level]
        halo = level_data.mesh.halo_radius

        # single partition case
        if level_data.num_partitions == 1:
            partition = level_data.partitions[0]
            owned = partition.owned_domain
            global_shape = level_data.mesh.global_cells

            # check if this is a refined level (owned != full domain)
            is_refined_subset = not all(
                owned.start[ii] == 0 and owned.fin[ii] == global_shape[ii]
                for ii in range(owned.ndim)
            )

            # get field data
            field_data = None
            if field_name in partition.hydro.primitives:
                field = partition.hydro.primitives[field_name]
                field_data = field.interior(halo).data
            elif (
                partition.hydro.magnetic is not None
                and field_name in partition.hydro.magnetic
            ):
                field_data = partition.hydro.magnetic[field_name].data
            elif field_name in self._derived_names:
                # derived field handling
                cache_key = (field_name, level)
                if cache_key in self._derived_cache:
                    field_data = self._derived_cache[cache_key]
                else:
                    derived_obj = self._derived_fields[field_name]
                    val = derived_obj.evaluate(level)
                    field_data = np.asarray(val)
                    self._derived_cache[cache_key] = field_data
            else:
                raise KeyError(f"field '{field_name}' not found")

            # C++ serialization now writes mesh config that matches the data
            # (refined levels have mesh.global_cells = owned region size)
            # so no embedding/cropping logic needed - just return the data
            return field_data

        # multi-partition stitching
        return self._stitch_partitions(field_name, level)

    def _stitch_partitions(self, field_name: str, level: int) -> NDArray:
        """
        stitch together multi-partition field data for a given level.

        returns full mesh-sized array with refined regions populated
        and unrefined regions filled with zeros (or ambient values).
        """
        level_data = self._checkpoint.levels[level]
        halo = level_data.mesh.halo_radius
        global_shape = level_data.mesh.global_cells

        # allocate output array for FULL mesh domain
        result = np.zeros(global_shape, dtype=np.float64)

        # check if field exists in any partition
        found_field = False
        for partition in level_data.partitions:
            if field_name in partition.hydro.primitives:
                found_field = True
                break
            if (
                partition.hydro.magnetic is not None
                and field_name in partition.hydro.magnetic
            ):
                found_field = True
                break

        if not found_field:
            # try derived fields
            if field_name in self._derived_names:
                # derived fields not yet supported for multi-partition
                raise NotImplementedError(
                    f"derived field '{field_name}' not supported "
                    f"for multi-partition levels"
                )
            raise KeyError(f"field '{field_name}' not found in any partition")

        # stitch each partition into global array
        for partition in level_data.partitions:
            # get owned domain (without ghost zones)
            owned = partition.owned_domain

            # create slice for global array
            slices = tuple(
                slice(owned.start[ii], owned.fin[ii])
                for ii in range(owned.ndim)
            )

            # get field data from this partition
            field_data = None
            if field_name in partition.hydro.primitives:
                field = partition.hydro.primitives[field_name]
                field_data = field.interior(halo).data
            elif (
                partition.hydro.magnetic is not None
                and field_name in partition.hydro.magnetic
            ):
                field_data = partition.hydro.magnetic[field_name].data

            if field_data is not None:
                result[slices] = field_data

        return result

    def get_owned_domain(self, level: int):
        """get the owned domain for a given level (for cropping refined regions)."""
        if level >= self.num_levels:
            raise ValueError(f"level {level} doesn't exist")
        level_data = self._checkpoint.levels[level]
        if level_data.num_partitions != 1:
            raise NotImplementedError(
                "get_owned_domain not supported for multi-partition levels"
            )
        return level_data.partitions[0].owned_domain

    def level_mesh(self, level: int, crop_to_owned: bool = False):
        """
        get mesh for a specific level with coordinate arrays.

        for refined levels, computes actual patch physical bounds from
        owned_domain indices and global coordinate system.
        """
        if level >= self.num_levels:
            raise ValueError(f"level {level} doesn't exist")

        if level not in self._mesh_adapters:
            level_data = self._checkpoint.levels[level]

            # for refined levels, pass owned_domain for coordinate computation
            owned_domain = None
            if level > 0 and level_data.partitions:
                # get owned domain from first partition
                part = level_data.partitions[0]
                owned_domain = (part.owned_domain.start, part.owned_domain.fin)

            self._mesh_adapters[level] = MeshAdapter(
                level_data.mesh, owned_domain=owned_domain
            )

        return self._mesh_adapters[level]

    def __getitem__(self, field_name: str) -> NDArray:
        """dict-like access to base level fields."""
        if self._base_fields is None:
            # load and cache base-level primitives (unpad)
            self._base_fields = get_base_fields(self._checkpoint, unpad=True)

        # try base fields first
        if field_name in self._base_fields:
            return self._base_fields[field_name]

        # try derived fields
        return self.get_field(field_name, level=0)

    def __contains__(self, field_name: str) -> bool:
        """check if field exists."""
        # faster path: check cached base fields and derived names
        if self._base_fields is not None and field_name in self._base_fields:
            return True
        if field_name in self._derived_names:
            return True

        # fall back to attempting to enumerate available fields for level 0
        try:
            return field_name in self.available_fields(level=0)
        except Exception:
            return False
