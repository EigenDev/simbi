from pathlib import Path
from typing import Any, Protocol

from pydantic.fields import computed_field


class StateInterface(Protocol):
    """What BaseProblemConfig needs to know about SimulationState"""

    @computed_field
    @property
    def dimensionality(self) -> int: ...

    @computed_field
    @property
    def nvars(self) -> int: ...

    @computed_field
    @property
    def is_mhd(self) -> bool: ...

    @computed_field
    @property
    def is_relativistic(self) -> bool: ...

    @computed_field
    @property
    def isothermal(self) -> bool: ...


class ProblemInterface(Protocol):
    """What SimulationState needs to know about BaseProblemConfig"""

    plm_theta: float
    end_time: float
    checkpoint_interval: float
    cfl_number: float
    data_directory: Path

    def validate_physics(self) -> None: ...
    def initial_primitive_state(self) -> dict[str, Any]: ...
    def build_state(self) -> None: ...
