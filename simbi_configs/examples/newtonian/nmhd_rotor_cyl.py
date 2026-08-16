# =============================================================================
# nmhd_rotor_cyl.py
#
# a rotating magnetized disk on a genuine cylindrical (r, phi) grid — the
# curvilinear sibling of nmhd_rotor.py. exercises the cylindrical r-phi
# constrained-transport path (out-of-plane B_z is the cell-centered field).
# planar_cylindrical => the (r, phi) disk plane.
# =============================================================================
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)

R_IN = 0.1
R_OUT = 1.0
TWO_PI = 6.283185307179586
R0 = 0.3       # rotor core radius
V0 = 1.0       # core angular speed scale


class NewtonianRotorCyl(SimbiProblem):
    """rotating magnetized disk on a cylindrical (r, phi) mesh."""

    adiabatic_index: Annotated[float, ProblemParam(1.4, description="gamma")]
    p0: Annotated[float, ProblemParam(1.0, description="ambient pressure")]
    bz: Annotated[
        float, ProblemParam(0.5, cli=True, description="uniform out-of-plane B_z")
    ]

    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((128, 128, 1), cli=True, description="(n_r, n_phi, 1)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(R_IN, R_OUT), (0.0, TWO_PI)], description="(r, phi) bounds"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(
            CoordSystem.PLANAR_CYLINDRICAL, description="(r, phi) disk plane"
        ),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NMHD, description="physics regime")
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLD, description="numerical solver")
    ]
    # phi is periodic; r is outflow.
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW, BoundaryCondition.PERIODIC],
            description="boundary conditions [r, phi]",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(0.1, cli=True, checkpoint_safe=True, description="end time"),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """solid-body core spinning in a vertical B_z; velocity is (v_r, v_phi, v_z)."""

        def gas_state() -> GasStateGenerator:
            nr, nphi, _ = self.resolution
            (rmin, rmax), _ = self.bounds[0], self.bounds[1]
            dr = (rmax - rmin) / nr
            for _kk in range(1):
                for _jj in range(nphi):
                    for ii in range(nr):
                        r = rmin + (ii + 0.5) * dr
                        if r < R0:
                            rho = 10.0
                            v_phi = V0 * r / R0   # solid-body rotation
                        else:
                            rho = 1.0
                            v_phi = 0.0
                        # (rho, v_r, v_phi, v_z, p)
                        yield (rho, 0.0, v_phi, 0.0, self.p0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            nr, nphi, nk = self.resolution
            for kk in range(nk + (bn == "bz")):
                for jj in range(nphi + (bn == "by")):
                    for ii in range(nr + (bn == "bx")):
                        # only the out-of-plane B_z is nonzero (cell-centered for r-phi).
                        yield self.bz if bn == "bz" else 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
