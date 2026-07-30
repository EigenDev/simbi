# =============================================================================
# homologous_expansion.py
#
# free homologous expansion on a moving (ALE) spherical mesh: cold-ish gas
# coasting outward with v = a_dot * r on a grid whose scale factor a(t) = 1 +
# a_dot * t (a_ddot = 0). the self-similar solution stays uniform in the comoving
# frame; the physical density scales as rho0 / a^3. exercises the mesh-motion path
# (the scale-factor callables a(t)/adot(t) feed sim.motion).
# =============================================================================
from typing import Annotated, Callable, Optional

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class HomologousExpansion(SimbiProblem):
    """free homologous expansion on a moving spherical mesh."""

    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    adot: Annotated[
        float, ProblemParam(0.5, cli=True, description="expansion rate a_dot")
    ]

    resolution: Annotated[
        int, ProblemParam(256, cli=True, description="radial cells")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.1, 1.0)], description="radial bounds"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW], description="boundary conditions"
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(0.2, cli=True, checkpoint_safe=True, description="end"),
    ]

    # ---- moving mesh: a(t) = 1 + a_dot t, linear (a_ddot = 0) ----
    @property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        rate = self.adot
        return lambda t: 1.0 + rate * t

    @property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
        rate = self.adot
        return lambda _t: rate

    def initial_primitive_state(self) -> InitialStateType:
        """uniform comoving rho/p, homologous radial velocity v = a_dot * r."""

        def gas_state() -> GasStateGenerator:
            nr = self.resolution
            rmin, rmax = self.bounds[0]
            dr = (rmax - rmin) / nr
            adot = self.adot
            for ii in range(nr):
                r = rmin + (ii + 0.5) * dr
                yield (1.0, adot * r, 1.0)

        return gas_state
