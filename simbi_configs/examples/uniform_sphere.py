# =============================================================================
# uniform_sphere.py
#
# uniform density and pressure in 1d spherical coordinates with homologous expansion.
# tests mesh motion implementation.
# =============================================================================
from typing import Annotated, Callable, Optional, Sequence

from pydantic import model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class UniformSphere(SimbiProblem):
    """uniform gas with homologous expansion in 1d spherical coordinates."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    rho_uniform: Annotated[
        float, ProblemParam(1.0, cli=True, description="uniform density")
    ]
    p_uniform: Annotated[
        float, ProblemParam(1.0, cli=True, description="uniform pressure")
    ]
    expansion_rate: Annotated[
        float,
        ProblemParam(0.1, cli=True, description="hubble parameter H = a_dot/a"),
    ]

    # domain
    r_inner: Annotated[
        float, ProblemParam(0.1, cli=True, description="inner radius")
    ]
    r_outer: Annotated[
        float, ProblemParam(1.0, cli=True, description="outer radius")
    ]
    nr: Annotated[int, ProblemParam(100, cli=True, description="radial zones")]

    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]

    # numerics
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="radial grid spacing"),
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.REFLECTING, BoundaryCondition.OUTFLOW],
            description="boundary conditions",
        ),
    ]
    solver: Annotated[
        Solver,
        ProblemParam(Solver.HLLC, cli=True, description="numerical solver"),
    ]

    # simulation control
    start_time: Annotated[
        float, ProblemParam(0.0, description="simulation start time")
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    resolution: Annotated[
        tuple[int],
        ProblemParam(None, description="grid resolution"),
    ]

    bounds: Annotated[
        Sequence[Sequence[float]],
        ProblemParam(None, description="domain boundaries"),
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "UniformSphere":
        """compute derived parameters."""
        if self.bounds is None:
            self.bounds = [(self.r_inner, self.r_outer)]
        if self.resolution is None:
            self.resolution = (self.nr,)
        return self

    @property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        """a(t) = 1 + H * t for linear expansion."""
        return lambda t: 1.0 + self.expansion_rate * t

    @property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
        """a_dot(t) = H for constant expansion velocity."""
        return lambda t: self.expansion_rate

    def initial_primitive_state(self) -> InitialStateType:
        """uniform density and pressure everywhere."""

        def gas_state() -> GasStateGenerator:
            (nr,) = self.resolution
            for ii in range(nr):
                yield (self.rho_uniform, 0.0, self.p_uniform, 0.0)

        return gas_state
