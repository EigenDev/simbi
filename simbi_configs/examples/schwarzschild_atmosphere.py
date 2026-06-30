# =============================================================================
# schwarzschild_atmosphere.py
#
# a relativistic gas on a 1d radial schwarzschild background — the FIRST curved-
# spacetime run. proves the lapse-densitized + GR-wavespeed `_schw` kernels select
# and run end-to-end (spacetime selector -> Schwarzschild metric -> the schwarzschild
# mass scalar binds). the domain sits OUTSIDE the horizon r = 2M.
#
# without the geodesic gravity source (B4) the gas does not yet accrete; this is the
# kernel-plumbing smoke test on the path to the michel accretion oracle (B.6).
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime, Spacetime
from simbi.types.typing import GasStateGenerator, InitialStateType


class SchwarzschildAtmosphere(SimbiProblem):
    """relativistic gas on a 1d radial schwarzschild background."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]
    # the curved spacetime: select the schwarzschild metric + its mass M (G = c = 1).
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.SCHWARZSCHILD, description="background spacetime"),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]

    # domain — radial, strictly OUTSIDE the horizon r = 2M
    resolution: Annotated[
        int, ProblemParam(256, cli=True, description="radial grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(2.5, 20.0)], description="radial domain bounds (r > 2M)"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.SRHD, description="physics regime")
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            1.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """a uniform static gas outside the horizon (rho, v_r, pre)."""

        def gas_state() -> GasStateGenerator:
            nr = self.resolution
            rmin, rmax = self.bounds[0]
            dr = (rmax - rmin) / nr
            for ii in range(nr):
                _ = rmin + (ii + 0.5) * dr
                yield (1.0, 0.0, 1e-2)

        return gas_state
