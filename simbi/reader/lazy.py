from typing import Any, Optional

from ..core.types import (
    Array,
    HierarchyData,
    MeshConfig,
    Metadata,
    ProcessedData,
)
from ..core.types.bodies import Body
from .computation import create_computation_pipeline


class FieldAccessor:
    def __init__(self, sim_data: "SimData", level: int):
        self._sim_data = sim_data
        self._level = level

    def __getitem__(self, key: str) -> Array:
        return self._sim_data.get_field(key, self._level)

    def __contains__(self, key: str) -> bool:
        try:
            self._sim_data.get_field(key, self._level)
            return True
        except KeyError:
            return False


class SimData:
    def __init__(self, data: ProcessedData):
        self._data = data
        self._computed_cache: dict[str, Array] = {}
        self._pipeline = create_computation_pipeline(data)
        self._computing: set[str] = set()  # Prevent circular dependencies

        # FMR-specific initialization
        self._level_caches: dict[int, dict[str, Array]] = {
            i: {} for i in range(self.num_levels)
        }

    @property
    def metadata(self) -> Metadata:
        return self._data.metadata

    @property
    def mesh(self) -> MeshConfig:
        """Get base level mesh"""
        return self._data.mesh

    def level_mesh(self, level: int) -> MeshConfig:
        """Get mesh for specific level"""
        if level == 0:
            return self._data.mesh
        if not self.has_refinement():
            raise ValueError("No refinement levels available")
        if not self._data.levels or level > len(self._data.levels):
            raise ValueError(f"Invalid level {level}")
        return self._data.levels[level - 1].mesh

    @property
    def bodies(self) -> dict[str, Body] | None:
        return self._data.bodies

    @property
    def fields(self) -> FieldAccessor:
        """Get base level field accessor"""
        return FieldAccessor(self, 0)

    def level_fields(self, level: int) -> FieldAccessor:
        """Get field accessor for specific level"""
        return FieldAccessor(self, level)

    @property
    def num_levels(self) -> int:
        """Get number of refinement levels"""
        return self._data.num_levels

    def has_refinement(self) -> bool:
        """Check if simulation has refinement levels"""
        return self._data.has_fmr

    def hierarchy(self) -> Optional[HierarchyData]:
        """Get hierarchy information if available"""
        return self._data.hierarchy

    def get_field(self, key: str, level: int = 0) -> Array:
        """Get field data for specific level"""
        if level == 0:
            return self._get_base_level_field(key)
        return self._get_refined_level_field(key, level)

    def __getitem__(self, key: str) -> Array:
        """Default field access is from base level"""
        return self._get_base_level_field(key)

    def _get_base_level_field(self, key: str) -> Array:
        """Get field from base level with computation support"""
        if key in self._computing:
            raise ValueError(f"Circular dependency detected for field '{key}'")

        # Check primitive fields first
        if key in self._data.fields:
            return self._data.fields[key]

        # Check computed cache
        if key in self._computed_cache:
            return self._computed_cache[key]

        # Compute if in pipeline
        if key in self._pipeline:
            self._computing.add(key)
            try:
                field_dict = FieldDict(self)
                result = self._pipeline[key](field_dict)
                self._computed_cache[key] = result
                return result
            finally:
                self._computing.discard(key)

        raise KeyError(f"Field '{key}' not found")

    def _get_refined_level_field(self, key: str, level: int) -> Array:
        """Get field from refined level with computation support"""
        if not self.has_refinement():
            raise ValueError("No refinement levels available")

        if not self._data.levels or level > len(self._data.levels):
            raise ValueError(f"Invalid level {level}")

        if key in self._computing:
            raise ValueError(f"Circular dependency detected for field '{key}'")

        # Check primitive fields first
        level_data = self._data.levels[level - 1]
        if key in level_data.fields:
            return level_data.fields[key]

        # Check level-specific cache
        if key in self._level_caches[level - 1]:
            return self._level_caches[level - 1][key]

        # Compute if in pipeline
        if key in self._pipeline:
            self._computing.add(key)
            try:
                field_dict = FieldDict(self, level)
                result = self._pipeline[key](field_dict)
                self._level_caches[level - 1][key] = result
                return result
            finally:
                self._computing.discard(key)

        raise KeyError(f"Field '{key}' not found at level {level}")


class FieldDict(dict[str, Array]):
    """Dict-like wrapper that routes field access through SimData"""

    def __init__(self, lazy_fields: SimData, level: int = 0):
        super().__init__()
        self._lazy_fields = lazy_fields
        self._level = level

    def __getitem__(self, key: str) -> Array:
        return self._lazy_fields.get_field(key, self._level)

    def get(self, key: str, default: Any = None) -> Array | Any:
        try:
            return self._lazy_fields.get_field(key, self._level)
        except KeyError as e:
            if default is not None:
                return default
            raise e
