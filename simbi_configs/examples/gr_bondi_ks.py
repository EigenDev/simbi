# =============================================================================
# gr_bondi_ks.py
#
# spherical (bondi) accretion onto a schwarzschild black hole in INGOING KERR-SCHILD
# (eddington-finkelstein) coordinates — the SAME physical spacetime as gr_bondi.py,
# but in a HORIZON-PENETRATING chart that is regular across r = 2M. started from a
# uniform gas at rest.
#
# the payoff of the KS chart: the inner boundary sits BELOW the horizon (r < 2M),
# where the transport velocity tilde v^r = v^r - beta^r/alpha is negative for EVERY
# subluminal fluid — nothing can escape, so zero-gradient outflow is unconditionally
# causal and the gas simply crosses r = 2M and leaves through the excised interior.
# no coordinate singularity, no subsonic-inner-boundary fragility (contrast the
# schwarzschild-coordinate gr_bondi.py, whose boundary must be held OUTSIDE 2M).
#
# radial zones are log-spaced (geometric-mean cell centers): the finest zones sit
# near the inner boundary where the flow is ultra-relativistic.
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


class GrBondiKS(SimbiProblem):
    """kerr-schild spherical accretion crossing the horizon, from a uniform gas at rest."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(
            Spacetime.KERR_SCHILD, description="background spacetime (ingoing kerr-schild)"
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

    # domain — radial, from INSIDE the horizon (r < 2M) to beyond the bondi radius
    resolution: Annotated[
        int, ProblemParam(512, cli=True, description="radial grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(1.5, 100.0)], description="radial domain bounds (inner boundary BELOW 2M)"
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
