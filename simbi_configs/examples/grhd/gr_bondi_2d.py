# =============================================================================
# gr_bondi_2d.py
#
# 2D axisymmetric spherical (bondi) accretion onto a schwarzschild black hole,
# started from a uniform gas at rest — the 2D sibling of gr_bondi.py. the flow is
# purely radial (the schwarzschild geodesic source has no angular component under
# spherical symmetry), so every theta column develops the SAME 1D transonic bondi
# inflow and the solution stays theta-INDEPENDENT. it exercises the multi-component
# Valencia covariant momentum: the angular conserved S_theta = rho h W^2 gamma_{theta
# theta} v^theta (gamma_{theta theta} = r^2) is allocated, stored, fluxed along the
# theta sweep, and recovered by the 2D metric-aware c2p — while staying ~0 for this
# radial flow. the grid is an EQUATORIAL theta wedge (away from the poles, where the
# cot(theta) angular geometric source is singular). radial zones are log-spaced.
# =============================================================================

import math
from typing import Annotated

from pydantic import model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Spacetime,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class GrBondi2D(SimbiProblem):
    """2D axisymmetric schwarzschild spherical accretion from a uniform gas at rest."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.SCHWARZSCHILD, description="background spacetime"),
    ]
    schwarzschild_mass: Annotated[
        float, ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)")
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, cli=True, description="ambient rest-mass density")
    ]
    p_ambient: Annotated[
        float,
        ProblemParam(1.0e-2, cli=True, description="ambient pressure (sets c_inf)"),
    ]

    # domain — (radial, polar): r from outside the horizon to beyond r_bondi; an
    # equatorial theta wedge (poles excluded — cot(theta) angular source is singular there).
    nr: Annotated[int, ProblemParam(256, cli=True, description="radial resolution")]
    npolar: Annotated[
        int, ProblemParam(16, cli=True, description="polar (theta) resolution")
    ]
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((0, 0), description="grid resolution (nr, npolar) — computed"),
    ]
    theta_halfwidth: Annotated[
        float,
        ProblemParam(0.3, description="half-width of the equatorial theta wedge (rad)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 0.0), (0.0, 0.0)], description="domain bounds — computed"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RHD, description="physics regime")]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LOG, description="log-spaced radial zones"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,     # r inner (toward the hole)
                BoundaryCondition.OUTFLOW,     # r outer (ambient)
                BoundaryCondition.REFLECTING,  # theta lo
                BoundaryCondition.REFLECTING,  # theta hi
            ],
            description="boundary conditions (r inner, r outer, theta lo, theta hi)",
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            200.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrBondi2D":
        """derive the (nr, npolar) resolution + the (radial, equatorial-theta-wedge) bounds."""
        self.resolution = (self.nr, self.npolar)
        theta_c = math.pi / 2.0
        self.bounds = [
            (3.0, 100.0),
            (theta_c - self.theta_halfwidth, theta_c + self.theta_halfwidth),
        ]
        return self

    def initial_primitive_state(self) -> InitialStateType:
        """a uniform gas at rest (rho, v_r=0, v_theta=0, pre) over the (theta, r) grid."""
        cs = (self.adiabatic_index * self.p_ambient / self.rho_ambient) ** 0.5
        r_bondi = self.schwarzschild_mass / cs**2
        print(f"ambient sound speed c_inf = {cs:.3e}, bondi radius r_bondi ~ {r_bondi:.3e}")

        def gas_state() -> GasStateGenerator:
            nr, npolar = self.resolution
            for _jj in range(npolar):
                for _ii in range(nr):
                    yield (self.rho_ambient, 0.0, 0.0, self.p_ambient)

        return gas_state
