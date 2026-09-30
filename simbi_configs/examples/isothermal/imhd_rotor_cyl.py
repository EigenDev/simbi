# =============================================================================
# imhd_rotor_cyl.py
#
# isothermal MHD on the cylindrical (r, phi) disk plane: a dense core of
# radius 0.3 in solid-body rotation threaded by a uniform out-of-plane B_z,
# with a density ripple 0.1 cos(phi) across the periodic seam. outflow in r,
# periodic in phi. the in-plane faces carry the divergence-free field
# B_r = b_r R_IN / r, B_phi = b_phi, so the rotation winds it and the edge EMF
# E_z = v_phi B_r - v_r B_phi acts from the first step; the out-of-plane B_z is
# the cell-centered field of the r-phi plane.
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator

R_IN = 0.1
R_OUT = 1.0
R0 = 0.3


class IsothermalRotorCyl(SimbiProblem):
    """a rotating isothermal magnetized core on the cylindrical (r, phi) plane."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    bz: Annotated[float, ProblemParam(0.5, cli=True, description="uniform out-of-plane B_z")]
    br: Annotated[float, ProblemParam(0.1, cli=True, description="radial field b_r at R_IN, B_r = b_r R_IN / r")]
    bphi: Annotated[float, ProblemParam(0.05, cli=True, description="uniform azimuthal field B_phi")]
    resolution: Annotated[tuple[int, int, int], ProblemParam((128, 128, 1), cli=True, description="(n_r, n_phi, 1)")]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(R_IN, R_OUT), (0.0, 2.0 * math.pi)], description="(r, phi) bounds")
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.PLANAR_CYLINDRICAL, description="(r, phi) disk plane")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.IMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW, BoundaryCondition.PERIODIC], description="boundary conditions [r, phi]"),
    ]
    end_time: Annotated[float, ProblemParam(0.1, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, v_r, v_phi, v_z) = (10 (1 + 0.1 cos(phi)), 0, r / R0, 0) inside r < R0 and
        (1 + 0.1 cos(phi), 0, 0, 0) outside; B_r = b_r R_IN / r on the radial faces, B_phi = b_phi
        on the azimuthal faces, B_z uniform"""

        def gas_state() -> GasStateGenerator:
            nr, nphi, _ = self.resolution
            (r_lo, r_hi), (p_lo, p_hi) = self.bounds
            dr, dphi = (r_hi - r_lo) / nr, (p_hi - p_lo) / nphi
            for jj in range(nphi):
                ripple = 1.0 + 0.1 * math.cos(p_lo + (jj + 0.5) * dphi)
                for ii in range(nr):
                    r = r_lo + (ii + 0.5) * dr
                    core = r < R0
                    yield ((10.0 if core else 1.0) * ripple, 0.0, r / R0 if core else 0.0, 0.0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            nr, nphi, nk = self.resolution
            (r_lo, r_hi), _ = self.bounds
            dr = (r_hi - r_lo) / nr
            for _kk in range(nk + (bn == "bz")):
                for _jj in range(nphi + (bn == "by")):
                    for ii in range(nr + (bn == "bx")):
                        if bn == "bx":
                            # (1/r) d_r (r B_r) = 0 on every radial face
                            yield self.br * R_IN / (r_lo + ii * dr)
                        elif bn == "by":
                            yield self.bphi
                        else:
                            yield self.bz

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
