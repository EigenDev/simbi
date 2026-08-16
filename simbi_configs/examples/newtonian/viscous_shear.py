# =============================================================================
# viscous_shear.py
#
# viscous decay of a sinusoidal shear layer in adiabatic newtonian hydro: the
# isolated, exactly-solvable probe for the navier-stokes momentum flux div(tau)
# and the viscous heating div(tau.v) that feeds the total-energy channel.
#
# setup: uniform rho and p, a single transverse shear v_y = V0 sin(2 pi x) that
# varies only in x (periodic, no pressure gradient, no advection). the linearized
# momentum equation is the heat equation d_t v_y = nu d_xx v_y, so the mode decays
# exactly as
#     v_y(x, t) = V0 exp(-nu (2 pi)^2 t) sin(2 pi x),
# and the peak speed halves after t_half = ln 2 / (nu (2 pi)^2). the kinetic
# energy the shear loses does not vanish: the dissipation Phi = tau : grad v >= 0
# converts it into gas internal energy, so with periodic walls the total energy is
# conserved to machine precision while thermal energy rises monotonically. that
# split is the whole point of the adiabatic (energy-carrying) viscosity.
#
# diagnostics: track max|v_y| against the exp(-nu (2 pi)^2 t) law, and check that
# the internal-energy gain equals the kinetic-energy loss.
#
# usage:
#  simbi run viscous_shear.py --nu 1.0e-3            # slow decay, gentle heating
#  simbi run viscous_shear.py --nu 1.0e-2 --v0 0.1   # faster; larger heat fraction
# =============================================================================
import math
from typing import Annotated

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

XMIN = 0.0
XMAX = 1.0


class ViscousShear(SimbiProblem):
    """viscous decay of a sinusoidal shear + the heat it dumps into the gas."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    v0: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            description="shear amplitude V0. kept subsonic (cs ~ 1.29) so the "
            "layer decays viscously rather than steepening into sound waves",
        ),
    ]
    rho0: Annotated[float, ProblemParam(1.0, description="uniform density")]
    p0: Annotated[float, ProblemParam(1.0, description="uniform pressure")]
    nu: Annotated[
        float,
        ProblemParam(
            1.0e-3,
            cli=True,
            description="constant kinematic viscosity. the mode decays as "
            "exp(-nu (2 pi)^2 t); 0 = inviscid (numerical-viscosity-only)",
        ),
    ]

    # the shear is 1d (varies only in x), but the cartesian viscous operator is
    # baked for d >= 2 only, so it runs in a square 2d box whose y-direction is
    # uniform (every row identical). the natural diagnostic is a 1d slice v_y(x);
    # a square domain just keeps a 2d image from being a degenerate sliver.
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((128, 128, 1), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(XMIN, XMAX), (XMIN, XMAX)], description="domain boundaries"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime (adiabatic)")
    ]

    # numerics
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.PERIODIC], description="boundary conditions"
        ),
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1"),
    ]

    # control (a few e-folds of the nu=1e-3 mode: nu (2 pi)^2 ~ 0.039, so t=20
    # is ~0.8 e-folds; bump --end-time or --nu to see deeper decay)
    start_time: Annotated[
        float, ProblemParam(0.0, description="simulation start time")
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            20.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    @computed_field
    @property
    def viscosity(self) -> float:
        # the backend reads `viscosity`; expose it as the cli nu.
        return self.nu

    def initial_primitive_state(self) -> InitialStateType:
        """uniform rho/p with a single-mode transverse shear v_y = V0 sin(2 pi x)."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            xbounds = self.bounds[0]
            dx = (xbounds[1] - xbounds[0]) / ni

            for _kk in range(nk):
                for _jj in range(nj):
                    for ii in range(ni):
                        x = xbounds[0] + (ii + 0.5) * dx
                        vy = self.v0 * math.sin(2.0 * math.pi * x)
                        # 2d newtonian hydro state is (rho, vx, vy, p): no vz slot.
                        yield (self.rho0, 0.0, vy, self.p0)

        return gas_state
