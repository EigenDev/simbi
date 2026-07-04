# =============================================================================
# locally_isothermal_wave.py
#
# a LOCALLY isothermal 1d test: the sound speed varies in space, cs^2(x), and is
# HELD fixed for the run (the position-dependent "temperature"). the user supplies
# density rho(x) and an initial pressure p(x); the backend derives cs^2(x)=p/rho
# once and freezes it. primitive is (rho, v, p_local) when locally_isothermal.
# =============================================================================
import math
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class LocallyIsothermalWave(SimbiProblem):
    """a locally-isothermal 1d run with a sinusoidal cs^2(x) profile."""

    sound_speed: Annotated[
        float,
        ProblemParam(
            1.0, description="reference cs (unused; cs^2(x) is derived per cell)"
        ),
    ]

    resolution: Annotated[
        int, ProblemParam(256, cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.PERIODIC], description="boundary conditions"
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            0.1, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    @computed_field
    @property
    def locally_isothermal(self) -> bool:
        """cs^2 varies in space — derived from the initial pressure, then held."""
        return True

    def initial_primitive_state(self) -> InitialStateType:
        """rho(x), at rest, plus a per-cell initial pressure p(x) = cs^2(x) rho."""

        def cs2_of(x: float) -> float:
            return 0.5 + 0.3 * math.sin(2.0 * math.pi * x)

        def rho_of(x: float) -> float:
            return 1.0 + 0.05 * math.cos(2.0 * math.pi * x)

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / nx
            for ii in range(nx):
                x = xmin + (ii + 0.5) * dx
                rho = rho_of(x)
                # locally isothermal: third component is the initial pressure
                # p(x) = cs^2(x) * rho(x); the backend freezes cs^2(x) = p/rho.
                yield (rho, 0.0, cs2_of(x) * rho)

        return gas_state
