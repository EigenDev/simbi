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

    def __init__(self, mesh: MeshGeometry):
        self._mesh = mesh
        self._coords_cache = {}

    def _generate_coords(self, axis: int) -> NDArray:
        """generate vertex coordinates for an axis."""
        if axis in self._coords_cache:
            return self._coords_cache[axis]

        # get bounds and number of cells
        xmin, xmax = self._mesh.dims[axis]
        ncells = self._mesh.global_cells[axis]
        spacing_type = self._mesh.spacing_types[axis]

        # generate coordinates based on spacing type
        if spacing_type == "linear":
            coords = np.linspace(xmin, xmax, ncells + 1)
        elif spacing_type == "log":
            coords = np.logspace(np.log10(xmin), np.log10(xmax), ncells + 1)
        else:
            # default to linear
            coords = np.linspace(xmin, xmax, ncells + 1)

        self._coords_cache[axis] = coords
        return coords

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
    def bodies(self):
        """access bodies (if present)."""
        return self._checkpoint.bodies

    def hierarchy(self):
        """access AMR hierarchy info."""
        # TODO: implement once hierarchy parsing is added to io
        return None

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

    def get_field(self, field_name: str, level: int = 0) -> NDArray:
        """
        get a field by name from a specific level.

        handles both primitive fields and derived fields like b1_mean.

        Uses caching for derived field evaluations.
        """
        if level >= self.num_levels:
            raise ValueError(
                f"level {level} doesn't exist (num_levels={self.num_levels})"
            )

        # quick membership test without constructing base dict
        level_data = self._checkpoint.levels[level]

        # assume single partition for now
        # TODO: handle multi-partition by stitching
        if level_data.num_partitions != 1:
            raise NotImplementedError(
                f"multi-partition support not yet implemented "
                f"(level {level} has {level_data.num_partitions} partitions)"
            )

        partition = level_data.partitions[0]
        halo = level_data.mesh.halo_radius

        # handle primitive fields
        if field_name in partition.hydro.primitives:
            field = partition.hydro.primitives[field_name]
            return field.interior(halo).data

        # handle face-centered magnetic fields - return raw face data
        # averaging to cell centers is handled by viz pipeline
        if (
            partition.hydro.magnetic is not None
            and field_name in partition.hydro.magnetic
        ):
            return partition.hydro.magnetic[field_name].data

        # derived field handling with caching
        if field_name not in self._derived_names:
            raise KeyError(f"field '{field_name}' not found")

        cache_key = (field_name, level)
        if cache_key in self._derived_cache:
            return self._derived_cache[cache_key]

        # evaluate and cache
        derived_obj = self._derived_fields[field_name]
        val = derived_obj.evaluate(level)
        # ensure numpy array and cache
        arr = np.asarray(val)
        self._derived_cache[cache_key] = arr
        return arr

    def level_mesh(self, level: int):
        """get mesh for a specific level with coordinate arrays."""
        if level >= self.num_levels:
            raise ValueError(f"level {level} doesn't exist")
        if level not in self._mesh_adapters:
            self._mesh_adapters[level] = MeshAdapter(
                self._checkpoint.levels[level].mesh
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
