# =============================================================================
# orszag_tang.py
#
# the orszag-tang vortex test case.
# =============================================================================
import math
from functools import partial
from typing import Any

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)

# domain constants
XMIN = 0.0
XMAX = 1.0


class OrszagTang(SimbiProblem):
    """the orszag-tang vortex test case."""

    # physics
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic index"
    )
    v0: float = ProblemParam(0.5, cli=True, description="velocity scale")
    b0: float = ProblemParam(1.0, cli=True, description="magnetic field scale")

    # domain
    resolution: tuple[int, int, int] = ProblemParam(
        (256, 256, 1), cli=True, description="grid resolution"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(XMIN, XMAX), (XMIN, XMAX)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(Regime.SRMHD, description="physics regime")

    # numerics
    solver: Solver = ProblemParam(Solver.HLLE, description="numerical solver")
    boundary_conditions: list[BoundaryCondition] = ProblemParam(
        [BoundaryCondition.PERIODIC], description="boundary conditions"
    )
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # simulation control
    start_time: float = ProblemParam(0.0, description="simulation start time")
    end_time: float = ProblemParam(
        0.0,
        cli=True,
        checkpoint_safe=True,
        description="simulation end time (auto if 0)",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        cs = (self.adiabatic_index - 1.0) / self.adiabatic_index
        object.__setattr__(self, "_cs", cs)

        if self.end_time == 0.0:
            computed_end = (XMAX - XMIN) / cs
            object.__setattr__(self, "end_time", computed_end)

    @computed_field
    @property
    def cs(self) -> float:
        """sound speed parameter."""
        return self._cs

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for orszag-tang vortex."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]

            p0 = float(self.adiabatic_index)
            rho0 = float(self.adiabatic_index) ** 2
            v0 = self.v0

            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj

            for kk in range(nk):
                for jj in range(nj):
                    y = ybounds[0] + (jj + 0.5) * dy
                    for ii in range(ni):
                        x = xbounds[0] + (ii + 0.5) * dx

                        vx = -v0 * math.sin(2.0 * math.pi * y)
                        vy = +v0 * math.sin(2.0 * math.pi * x)

                        yield (rho0, vx, vy, 0.0, p0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            xbounds = self.bounds[0]
            ybounds = self.bounds[1]

            dx = (xbounds[1] - xbounds[0]) / ni
            dy = (ybounds[1] - ybounds[0]) / nj
            b0 = self.b0

            for kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    y = ybounds[0] + jj * dy
                    for ii in range(ni + (bn == "bx")):
                        x = xbounds[0] + ii * dx

                        if bn == "bx":
                            yield -b0 * math.sin(2.0 * math.pi * y)
                        elif bn == "by":
                            yield +b0 * math.sin(4.0 * math.pi * x)
                        else:
                            yield 0.0

        bx_gen = partial(b_field, "bx")
        by_gen = partial(b_field, "by")
        bz_gen = partial(b_field, "bz")

        return (gas_state, bx_gen, by_gen, bz_gen)
