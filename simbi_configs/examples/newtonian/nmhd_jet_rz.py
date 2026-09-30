# =============================================================================
# nmhd_jet_rz.py
#
# newtonian ideal MHD on the axisymmetric cylindrical (r, z) plane: a dense
# core of radius 0.3 moving along z at v_z = 0.5 and swirling at
# v_phi = 0.3 r / R0, inside a still ambient gas, periodic in z with outflow in
# r over r in [0.1, 1]. the in-plane faces carry the divergence-free field
# B_r = b_r R_IN / r and a uniform B_z = b_z, so the shear at the core's edge
# drives the edge EMF E_phi = v_z B_r - v_r B_z; the out-of-plane B_phi = b_phi
# is the cell-centered field of the (r, z) plane.
# =============================================================================
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import GasStateGenerator, InitialStateType, StaggeredBFieldGenerator

R_IN = 0.1
R_OUT = 1.0
R0 = 0.3


class NewtonianJetRz(SimbiProblem):
    """a swirling magnetized jet core on the cylindrical (r, z) plane."""

    adiabatic_index: Annotated[float, ProblemParam(1.4, description="adiabatic index")]
    br: Annotated[float, ProblemParam(0.1, cli=True, description="radial field b_r at R_IN, B_r = b_r R_IN / r")]
    bz: Annotated[float, ProblemParam(0.05, cli=True, description="uniform axial field B_z")]
    bphi: Annotated[float, ProblemParam(0.2, cli=True, description="uniform azimuthal field B_phi")]
    resolution: Annotated[tuple[int, int, int], ProblemParam((128, 128, 1), cli=True, description="(n_r, n_z, 1)")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(R_IN, R_OUT), (0.0, 1.0)], description="(r, z) bounds")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="the (r, z) plane")]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW, BoundaryCondition.PERIODIC], description="boundary conditions [r, z]"),
    ]
    end_time: Annotated[float, ProblemParam(0.1, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, v_r, v_z, v_phi, p) = (10, 0, 0.5, 0.3 r / R0, 1) inside r < R0 and (1, 0, 0, 0, 1)
        outside; B_r = b_r R_IN / r on the radial faces, B_z = b_z on the axial faces, B_phi = b_phi"""

        def gas_state() -> GasStateGenerator:
            nr, nz, _ = self.resolution
            (r_lo, r_hi), _ = self.bounds
            dr = (r_hi - r_lo) / nr
            for _jj in range(nz):
                for ii in range(nr):
                    r = r_lo + (ii + 0.5) * dr
                    core = r < R0
                    yield (10.0 if core else 1.0, 0.0, 0.5 if core else 0.0, 0.3 * r / R0 if core else 0.0, 1.0)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            nr, nz, nk = self.resolution
            (r_lo, r_hi), _ = self.bounds
            dr = (r_hi - r_lo) / nr
            for _kk in range(nk + (bn == "bz")):
                for _jj in range(nz + (bn == "by")):
                    for ii in range(nr + (bn == "bx")):
                        if bn == "bx":
                            # (1/r) d_r (r B_r) = 0 on every radial face
                            yield self.br * R_IN / (r_lo + ii * dr)
                        elif bn == "by":
                            yield self.bz
                        else:
                            yield self.bphi

        return (gas_state, partial(b_field, "bx"), partial(b_field, "by"), partial(b_field, "bz"))
