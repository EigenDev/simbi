# =============================================================================
# quirk.py
#
# quirk's problem from quirk (1994), "a contribution to the great riemann
# solver debate". designed to exacerbate carbuncle instability.
# =============================================================================
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

PERTURBATION_SCALE = 0.5e-3


@dataclass
class QuirkState:
    """state class for quirk's problem with custom operations."""

    rho: float
    vx: float
    vy: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.vx
        yield self.vy
        yield self.p

    def __add__(self, other: "QuirkState") -> "QuirkState":
        return QuirkState(
            self.rho + other.rho,
            self.vx + other.vx,
            self.vy + other.vy,
            self.p + other.p,
        )


class Quirk(SimbiProblem):
    """quirk's problem - carbuncle instability test."""

    # physics
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic index"
    )
    mach_mode: str = ProblemParam(
        "low", cli=True, description="mach number regime (low or high)"
    )

    # domain
    resolution: tuple[int, int] = ProblemParam(
        (2400, 20), cli=True, description="grid resolution"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(0.0, 2400.0), (0.0, 20.0)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(
        Regime.NEWTONIAN, description="physics regime"
    )
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # numerics
    boundary_conditions: list[BoundaryCondition] = ProblemParam(
        [BoundaryCondition.REFLECTING], description="boundary conditions"
    )
    solver: Solver = ProblemParam(Solver.HLLC, description="numerical solver")
    use_quirk_smoothing: bool = ProblemParam(
        True, cli=True, description="enable quirk smoothing"
    )

    # simulation control
    end_time: float = ProblemParam(
        0.0,
        cli=True,
        checkpoint_safe=True,
        description="simulation end time (auto if 0)",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        object.__setattr__(
            self,
            "_problem_states",
            {
                "low": (
                    QuirkState(
                        216.0 / 41.0,
                        (35.0 / 36.0) * math.sqrt(35),
                        0.0,
                        251.0 / 6.0,
                    ),
                    QuirkState(1.0, 0.0, 0.0, 1.0),
                ),
                "high": (
                    QuirkState(
                        160.0 / 27.0, (133.0 / 8.0) * math.sqrt(1.4), 0.0, 466.5
                    ),
                    QuirkState(1.0, 0.0, 0.0, 1.0),
                ),
            },
        )

        # set end_time based on mach_mode if not provided
        if self.end_time == 0.0:
            computed_end = 330.0 if self.mach_mode == "low" else 100.0
            object.__setattr__(self, "end_time", computed_end)

    @computed_field
    @property
    def data_directory(self) -> Path:
        """compute output data directory based on configuration."""
        smoothing_dir = "smoothing" if self.use_quirk_smoothing else "raw"
        return Path(f"data/quirk/{smoothing_dir}/{self.mach_mode}_mach")

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for quirk problem."""

        def gas_state() -> GasStateGenerator:
            state = self._problem_states[self.mach_mode]
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / nx

            for jj in range(ny):
                for ii in range(nx):
                    xi = xmin + (ii + 0.5) * dx

                    if xi <= 5:
                        perturb = QuirkState(
                            *[
                                PERTURBATION_SCALE * random.randint(-1, 1)
                                for _ in range(4)
                            ]
                        )
                        yield tuple(state[0] + perturb)
                    else:
                        yield tuple(state[1])

        return gas_state
