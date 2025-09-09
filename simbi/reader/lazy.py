from typing import Any

from ..core.types.bodies import Body
from ..core.types import Array, MeshConfig, Metadata, ProcessedData
from .computation import create_computation_pipeline


class FieldAccessor:
    def __init__(self, sim_data: "SimData"):
        self._sim_data = sim_data

    def __getitem__(self, key: str) -> Array:
        return self._sim_data[key]

    def __contains__(self, key: str) -> bool:
        try:
            self._sim_data[key]
            return True
        except KeyError:
            return False


class SimData:
    def __init__(self, data: ProcessedData):
        self._data = data
        self._computed_cache: dict[str, Array] = {}
        self._pipeline = create_computation_pipeline(data)
        self._computing: set[str] = set()  # Prevent circular dependencies

    @property
    def metadata(self) -> Metadata:
        return self._data.metadata

    @property
    def mesh(self) -> MeshConfig:
        return self._data.mesh

    @property
    def bodies(self) -> dict[str, Body] | None:
        return self._data.bodies

    @property
    def fields(self) -> FieldAccessor:
        return FieldAccessor(self)

    def __getitem__(self, key: str) -> Array:
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
                # Create a dict-like view that routes through this SimData instance
                field_dict = FieldDict(self)
                result = self._pipeline[key](field_dict)
                self._computed_cache[key] = result
                return result
            finally:
                self._computing.discard(key)

        raise KeyError(f"Field '{key}' not found")

    def unpack(
        self,
    ) -> tuple[FieldAccessor, Metadata, MeshConfig, dict[str, Body] | None]:
        return (FieldAccessor(self), self.metadata, self.mesh, self.bodies)


class FieldDict(dict[str, Array]):
    """Dict-like wrapper that routes field access through SimData"""

    def __init__(self, lazy_fields: SimData):
        super().__init__()
        self._lazy_fields = lazy_fields

    def __getitem__(self, key: str) -> Array:
        return self._lazy_fields[key]

    def get(self, key: str, default: Any = None) -> Array | Any:
        try:
            return self._lazy_fields[key]
        except KeyError as e:
            if default is not None:
                return default
            raise e
