# =============================================================================
# rt.py
#
# rayleigh-taylor instability in newtonian fluid.
# heavier fluid on top of lighter fluid with gravity.
# =============================================================================
import math
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import (
    ExpressionDict,
    GasStateGenerator,
    InitialStateType,
)


class RayleighTaylor(SimbiProblem):
    """rayleigh-taylor instability in newtonian fluid."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(7.0 / 5.0, description="adiabatic index")
    ]
    rhoU: Annotated[float, ProblemParam(2.0, description="upper layer density")]
    rhoD: Annotated[float, ProblemParam(1.0, description="lower layer density")]
    p0: Annotated[float, ProblemParam(2.5, description="reference pressure")]
    g0: Annotated[
        float,
        ProblemParam(0.1, cli=True, description="gravitational acceleration"),
    ]
    vamp: Annotated[
        float, ProblemParam(0.01, description="velocity perturbation amplitude")
    ]

    # domain
    resolution: Annotated[
        tuple[int, int],
        ProblemParam(
            (200, 600), cli=True, description="grid resolution (x, y)"
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(-0.25, 0.25), (-0.75, 0.75)], description="domain boundaries"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]

    # numerics
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.PERIODIC, BoundaryCondition.REFLECTING],
            description="boundary conditions [x, y]",
        ),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/rt_config"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            10.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    @property
    def ymidpoint(self) -> float:
        """calculate middle of y domain."""
        return 0.5 * (self.bounds[1][0] + self.bounds[1][1])

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for rayleigh-taylor instability."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]

            xextent = xmax - xmin
            yextent = ymax - ymin
            dx = xextent / nx
            dy = yextent / ny

            ymid = self.ymidpoint

            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx

                    if y <= ymid:
                        rho = self.rhoD
                    else:
                        rho = self.rhoU

                    p = self.p0 - self.g0 * rho * y

                    vy = (
                        self.vamp
                        * 0.25
                        * (1.0 + math.cos(4.0 * math.pi * x))
                        * (1.0 + math.cos(3.0 * math.pi * y))
                    )

                    yield (rho, 0.0, vy, p)

        return gas_state

    @computed_field
    @property
    def source_expressions(self) -> list[ExpressionDict]:
        """uniform downward gravity a = (0, -g0) as a `force` source — the rust
        backend lifts it to the newtonian momentum + energy overlays (S_mom=rho*a,
        S_nrg=rho*a.v). emitted in the rust `SourceConfig` wire format."""
        graph = expr.ExprGraph()

        x_comp = expr.constant(0.0, graph)
        y_comp = expr.constant(-self.g0, graph)

        compiled = graph.compile([x_comp, y_comp])
        return [compiled.serialize_source(expr.SourceKind.FORCE, dim=2)]
