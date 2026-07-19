# =============================================================================
# gr_cylindrical_rz_field_loop.py
#
# a poloidal magnetic field loop on the 2.5D cylindrical kerr-schild (R, z) plane
# the constrained-transport probe for the axisymmetric poloidal
# chart. the (R, z) spatial metric is NON-DIAGONAL (gamma_Rz = 2H R z / r^2, r =
# sqrt(R^2 + z^2)), sqrt(det gamma) = R sqrt(1 + 2M/r), and the shift is nonzero on
# BOTH poloidal axes (beta^R, beta^z). a compact poloidal loop (B_R, B_z) is seeded
# div-free through the metric-weighted discrete curl of a localized toroidal A_phi,
# and the full curved-CT machinery must PRESERVE the w-weighted div(B) as the gas
# free-falls under the covariant geodesic source. the out-of-plane toroidal field
# B_phi starts at zero (a pure poloidal loop).
#
# what it certifies: the two-component (R, z) shift corner EMF + the R sqrt(1+2M/r)
# densitized curl hold div(B) at machine precision and run stably (no floors, no
# crash) in the cylindrical chart. HLLE gas flux + contact / UCT-HLL CT.
#
# usage:
#   simbi run gr_cylindrical_rz_field_loop.py --ct-method uct
#   (the gate: simbi/simulation/tests/test_cylindrical_grmhd.py)
# =============================================================================

import math
from functools import partial
from typing import Annotated

from pydantic import model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CoordSystem,
    CtMethod,
    Regime,
    Solver,
    Spacetime,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)


class GrCylindricalRzFieldLoop(SimbiProblem):
    """poloidal field loop on the cylindrical kerr-schild (R, z) plane — the GR-CT probe."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.KERR_SCHILD, description="cylindrical kerr-schild background"),
    ]
    schwarzschild_mass: Annotated[
        float, ProblemParam(1.0, cli=True, description="black-hole mass M")
    ]
    amplitude: Annotated[
        float, ProblemParam(1.0e-3, cli=True, description="vector-potential amplitude A0")
    ]
    loop_radius: Annotated[
        float, ProblemParam(1.5, cli=True, description="loop radius R (coordinate)")
    ]
    rho_ambient: Annotated[float, ProblemParam(1.0, description="uniform density")]
    p_ambient: Annotated[float, ProblemParam(0.1, description="uniform pressure")]

    nr: Annotated[int, ProblemParam(80, cli=True, description="R resolution")]
    nz: Annotated[int, ProblemParam(80, cli=True, description="z resolution")]
    resolution: Annotated[
        tuple[int, int], ProblemParam((0, 0), description="grid resolution — computed")
    ]
    r_center: Annotated[float, ProblemParam(7.0, description="loop center R")]
    z_center: Annotated[float, ProblemParam(0.0, description="loop center z")]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="cylindrical (R, z) 2.5D")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, cli=True, description="solver")]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.UCT, cli=True, description="CT edge-EMF method")
    ]
    # an off-axis (R > 0), z-symmetric patch: r = sqrt(R^2 + z^2) stays outside r = 2M.
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(4.0, 10.0), (-3.0, 3.0)], description="(R, z) domain — off-axis"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.OUTFLOW] * 4, description="outflow on all four edges"
        ),
    ]
    # the poloidal (R, z) free-fall converges and piles up at the inner R boundary (unlike the disk,
    # which infalls only radially at fixed z). the sharp UCT-HLL EMF does not diffuse the pileup, so
    # |B| there grows quickly past t ~ 0.7 (contact's dissipation smears it); t = 0.7 keeps the loop
    # clean at BOTH CT methods. the w-weighted div(B) stays machine-zero throughout regardless.
    end_time: Annotated[
        float, ProblemParam(0.7, cli=True, checkpoint_safe=True, description="end time")
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrCylindricalRzFieldLoop":
        self.resolution = (self.nr, self.nz)
        return self

    def _sqrtg(self, big_r: float, z: float) -> float:
        # sqrt(det gamma) = R sqrt(1 + 2M/r), r = sqrt(R^2 + z^2), for cylindrical kerr-schild.
        r = math.hypot(big_r, z)
        return big_r * math.sqrt(1.0 + 2.0 * self.schwarzschild_mass / r)

    def _potential(self, big_r: float, z: float) -> float:
        """A_phi(R, z): a compact conical bump so B = curl(A_phi phihat) is a poloidal loop."""
        s = math.hypot(big_r - self.r_center, z - self.z_center)
        return self.amplitude * (self.loop_radius - s) if s < self.loop_radius else 0.0

    def radial_faces(self) -> list[float]:
        (rmin, rmax) = self.bounds[0]
        dr = (rmax - rmin) / self.nr
        return [rmin + ii * dr for ii in range(self.nr + 1)]

    def z_faces(self) -> list[float]:
        (zmin, zmax) = self.bounds[1]
        dz = (zmax - zmin) / self.nz
        return [zmin + jj * dz for jj in range(self.nz + 1)]

    def initial_primitive_state(self) -> InitialStateType:
        nr, nz = self.nr, self.nz
        rf = self.radial_faces()
        zf = self.z_faces()
        dr = rf[1] - rf[0]
        dz = zf[1] - zf[0]
        r_c = [0.5 * (rf[i] + rf[i + 1]) for i in range(nr)]
        z_c = [0.5 * (zf[j] + zf[j + 1]) for j in range(nz)]

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest: the 2.5D swirl 5-tuple (rho, v_R, v_phi, v_z, pre).
            for _jj in range(nz):
                for _ii in range(nr):
                    yield (self.rho_ambient, 0.0, 0.0, 0.0, self.p_ambient)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            a = self._potential
            if bn == "b1":
                # B_R on R-faces: (nr+1) x nz. densitized curl B_R = d_z A_phi / (sqrt(g) dz):
                # (A(R_f, z_hi) - A(R_f, z_lo)) / (sqrt(g)(R_f, z_c) dz).
                for jj in range(nz):
                    for ii in range(nr + 1):
                        w = self._sqrtg(rf[ii], z_c[jj]) * dz
                        yield (a(rf[ii], zf[jj + 1]) - a(rf[ii], zf[jj])) / w
            elif bn == "b2":
                # B_z on z-faces: nr x (nz+1). densitized curl B_z = -d_R A_phi / (sqrt(g) dR):
                # -(A(R_hi, z_f) - A(R_lo, z_f)) / (sqrt(g)(R_c, z_f) dR).
                for jj in range(nz + 1):
                    for ii in range(nr):
                        w = self._sqrtg(r_c[ii], zf[jj]) * dr
                        yield -(a(rf[ii + 1], zf[jj]) - a(rf[ii], zf[jj])) / w
            else:
                # the out-of-plane toroidal B_phi: zero (a pure poloidal loop).
                for _jj in range(nz):
                    for _ii in range(nr):
                        yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
