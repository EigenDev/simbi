from dataclasses import dataclass
from enum import Enum, auto
from typing import Generator, Protocol, Sequence

# Types for generator functions
PrimitiveState = tuple[float, ...]
StateGenerator = Generator[PrimitiveState, None, None]


class StateComponent(Enum):
    """Components of primitive state"""

    DENSITY = auto()
    VELOCITY_X = auto()
    VELOCITY_Y = auto()
    VELOCITY_Z = auto()
    PRESSURE = auto()
    BFIELD_X = auto()
    BFIELD_Y = auto()
    BFIELD_Z = auto()
    PASSIVE_SCALAR = auto()


@dataclass(frozen=True)
class StateLayout:
    """Describes the layout of primitive variables"""

    components: Sequence[StateComponent]

    @classmethod
    def for_hydro(cls, dims: int) -> "StateLayout":
        """Create layout for hydrodynamics"""
        return cls(
            components=[
                StateComponent.DENSITY,
                *(
                    [StateComponent.VELOCITY_X]
                    + [StateComponent.VELOCITY_Y] * (dims > 1)
                    + [StateComponent.VELOCITY_Z] * (dims > 2)
                ),
                StateComponent.PRESSURE,
                StateComponent.PASSIVE_SCALAR,
            ]
        )

    @classmethod
    def for_mhd(cls, dims: int) -> "StateLayout":
        """Create layout for magnetohydrodynamics"""
        return cls(
            components=[
                StateComponent.DENSITY,
                *(
                    [StateComponent.VELOCITY_X]
                    + [StateComponent.VELOCITY_Y] * (dims > 1)
                    + [StateComponent.VELOCITY_Z] * (dims > 2)
                ),
                StateComponent.PRESSURE,
                StateComponent.BFIELD_X,
                StateComponent.BFIELD_Y,
                StateComponent.BFIELD_Z,
                StateComponent.PASSIVE_SCALAR,
            ]
        )


class InitialCondition(Protocol):
    """Protocol for initial condition generators"""

    def generate_primitives(self) -> StateGenerator: ...
    def generate_magnetic(
        self,
    ) -> tuple[StateGenerator, StateGenerator, StateGenerator] | None: ...


@dataclass(frozen=True)
class GridPoint:
    """Physical location in the grid"""

    x: float
    y: float = 0.0
    z: float = 0.0

    @property
    def coords(self) -> tuple[float, ...]:
        """Get coordinates as tuple"""
        return (self.x, self.y, self.z)
