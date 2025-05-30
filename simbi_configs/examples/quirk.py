import math
import random
from dataclasses import dataclass
from typing import Iterator, Any

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    CellSpacing,
    Solver,
)
from simbi.core.types.typing import GasStateGenerator, InitialStateType
from pydantic import computed_field
from pathlib import Path

from simbi.old_core.types.constants import BoundaryCondition

PERTURBATION_SCALE = 0.5e-3  # Scale for random perturbations


@dataclass
class QuirkState:
    """State class for Quirk's problem with custom operations"""

    rho: float
    vx: float
    vy: float
    p: float

    def __iter__(self) -> Iterator[float]:
        """Iterator to yield state components"""
        yield self.rho
        yield self.vx
        yield self.vy
        yield self.p

    def __add__(self, other: "QuirkState") -> "QuirkState":
        """Addition operation for adding perturbations"""
        return QuirkState(
            self.rho + other.rho,
            self.vx + other.vx,
            self.vy + other.vy,
            self.p + other.p,
        )


class Quirk(SimbiBaseConfig):
    """
    Quirk's problem in Newtonian Fluid from Quirk (1994),
    "A contribution to the great Riemann solver debate"
    This problem is a shock tube which is designed to exacerbate
    the carbuncle instability and odd-even decoupling in numerical schemes.
    Turning the `use_quirk_smoothing` flag on will apply a smoothing
    technique to mitigate these issues, as described in Quirk's paper.
    """

    # Configuration parameters
    resolution: tuple[int, int] = SimbiField((2400, 20), description="Grid resolution")

    mach_mode: str = SimbiField(
        "low", description="Mach number regime", choices=["low", "high"]
    )

    # Domain boundaries
    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 2400.0), (0.0, 20.0)], description="Domain boundaries"
    )

    # Required fields from SimbiBaseConfig
    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")

    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    # Optional customizations with non-default values
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    boundary_conditions: list[BoundaryCondition] = SimbiField(
        [BoundaryCondition.REFLECTING], description="Boundary conditions"
    )

    solver: Solver = SimbiField(Solver.HLLC, description="Numerical solver")

    use_quirk_smoothing: bool = SimbiField(True, description="Enable Quirk smoothing")

    end_time: float = SimbiField(
        0.0, description="Simulation end time (calculated based on mach_mode)"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        # Initialize problem states
        self._problem_states = {
            "low": (
                QuirkState(
                    216.0 / 41.0, (35.0 / 36.0) * math.sqrt(35), 0.0, 251.0 / 6.0
                ),
                QuirkState(1.0, 0.0, 0.0, 1.0),
            ),
            "high": (
                QuirkState(160.0 / 27.0, (133.0 / 8.0) * math.sqrt(1.4), 0.0, 466.5),
                QuirkState(1.0, 0.0, 0.0, 1.0),
            ),
        }

        # Set end_time based on mach_mode if not provided
        if self.end_time == 0.0:
            self.end_time = 330 if self.mach_mode == "low" else 100

    @computed_field
    @property
    def data_directory_setter(self) -> None:
        """Compute output data directory based on configuration"""
        smoothing_dir = "smoothing" if self.use_quirk_smoothing else "raw"
        self.data_directory = Path(f"data/quirk/{smoothing_dir}/{self.mach_mode}_mach")

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Quirk problem.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            state = self._problem_states[self.mach_mode]
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / nx

            for j in range(ny):
                for i in range(nx):
                    xi = xmin + (i + 0.5) * dx  # Cell center

                    # Add random perturbations to either left or right state
                    if xi <= 5:
                        # Left state with perturbation
                        perturb = QuirkState(
                            *[
                                PERTURBATION_SCALE * random.randint(-1, 1)
                                for _ in range(4)
                            ]
                        )
                        yield tuple(state[0] + perturb)
                    else:
                        # Right state with perturbation
                        perturb = QuirkState(
                            *[
                                PERTURBATION_SCALE * random.randint(-1, 1)
                                for _ in range(4)
                            ]
                        )
                        yield tuple(state[1])

        return gas_state
