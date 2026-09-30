# =============================================================================
# nmhd_rotor_cyl.py
#
# a rotating magnetized disk on a genuine cylindrical (r, phi) grid — the
# curvilinear sibling of nmhd_rotor.py. exercises the cylindrical r-phi
# constrained-transport path (out-of-plane B_z is the cell-centered field).
# planar_cylindrical => the (r, phi) disk plane. an optional in-plane field
# B_r = b_r R_IN / r, B_phi = b_phi (divergence-free, zero by default) gives the
# edge EMF E_z = v_phi B_r - v_r B_phi something to act on.
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
    br: Annotated[
        float, ProblemParam(0.0, cli=True, description="radial field b_r at R_IN, B_r = b_r R_IN / r")
    ]
    bphi: Annotated[
        float, ProblemParam(0.0, cli=True, description="uniform azimuthal field B_phi")
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
            (rmin, rmax) = self.bounds[0]
            dr = (rmax - rmin) / nr
            for kk in range(nk + (bn == "bz")):
                for jj in range(nphi + (bn == "by")):
                    for ii in range(nr + (bn == "bx")):
                        if bn == "bx":
                            # (1/r) d_r (r B_r) = 0 on every radial face
                            yield self.br * R_IN / (rmin + ii * dr)
                        elif bn == "by":
                            yield self.bphi
                        else:
                            # the out-of-plane B_z is cell-centered for r-phi
                            yield self.bz

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
