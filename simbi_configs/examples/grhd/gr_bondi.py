# =============================================================================
# gr_bondi.py
#
# spherical (bondi) accretion onto a schwarzschild black hole, started from a
# uniform gas at rest. the schwarzschild geodesic source pulls the gas inward; the
# inflow accelerates, and (once the flow develops) passes through a sonic surface
# and exits supersonically through the inner boundary toward the hole.
#
# scales (G = c = 1): the central mass M sets the gravitational radius r_g = M
# (horizon at 2M). the ambient sound speed c_inf = sqrt(gamma p / (rho h)) sets the
# bondi radius r_bondi ~ M / c_inf^2, where gravity overtakes pressure and the flow
# goes transonic; the domain spans from outside the horizon to beyond r_bondi.
#
# the inflow becomes ultra-relativistic (V -> 1, W -> infinity) as r -> 2M, which is
# a coordinate singularity of schwarzschild coordinates: the inner boundary is held
# away from 2M, and through-horizon flow requires horizon-penetrating coordinates.
# radial zones are log-spaced (geometric-mean cell centers): the bondi flow spans many
# decades in r, with the finest zones near the inner boundary.
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Spacetime,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrBondi(SimbiProblem):
    """schwarzschild spherical accretion developing from a uniform gas at rest."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.SCHWARZSCHILD_KS, description="background spacetime"
        ),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]
    rho_ambient: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="ambient rest-mass density"),
    ]
    p_ambient: Annotated[
        float,
        ProblemParam(
            1.0e-2,
            cli=True,
            description="ambient pressure (sets c_inf, r_bondi)",
        ),
    ]

    # domain — radial, from outside the horizon to beyond the bondi radius
    resolution: Annotated[
        int, ProblemParam(512, cli=True, description="radial grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(3.0, 100.0)], description="radial domain bounds (r > 2M)"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.RHD, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LOG, description="log-spaced radial zones"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW],
            description="boundary conditions (inner, outer)",
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            200.0,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """a uniform gas at rest (rho, v_r=0, pre)."""
        cs = (self.adiabatic_index * self.p_ambient / self.rho_ambient) ** 0.5
        r_bondi = self.schwarzschild_mass / cs**2
        print(
            f"ambient sound speed c_inf = {cs:.3e}, bondi radius r_bondi ~ {r_bondi:.3e}"
        )

        def gas_state() -> GasStateGenerator:
            for _ in range(self.resolution):
                yield (self.rho_ambient, 0.0, self.p_ambient)

        return gas_state
