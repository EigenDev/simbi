# =============================================================================
# gr_disk_field_loop.py
#
# an in-plane magnetic field loop on the equatorial (R, phi) cylindrical kerr-schild
# DISK (design 45 GRMHD) — the constrained-transport probe for the razor-thin disk
# chart. on the equator (z = 0, r = R) the kerr-schild off-diagonal vanishes so the
# metric is DIAGONAL (gamma = diag(1 + 2M/R, R^2), alpha = 1/sqrt(1 + 2M/R), beta^R =
# 2M/(R + 2M), beta^phi = 0), sqrt(det gamma) = R sqrt(1 + 2M/R). a compact in-plane
# loop (B_R, B_phi) is seeded div-free through the metric-weighted discrete curl of a
# localized vertical A_z, and the curved-CT machinery (out-of-plane corner EMF E_z)
# must PRESERVE the w-weighted div(B) as the gas free-falls radially. the vertical
# field B_z starts at zero.
#
# what it certifies: the chart-generic densitized curl + corner EMF hold div(B) at
# machine precision and run stably (no floors, no crash) in the diagonal disk chart.
# HLLE gas flux + contact / UCT-HLL CT.
#
# usage:
#   simbi run gr_disk_field_loop.py --ct-method uct
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


class GrDiskFieldLoop(SimbiProblem):
    """in-plane field loop on the equatorial (R, phi) kerr-schild disk — the GR-CT probe."""

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
        float, ProblemParam(1.5, cli=True, description="loop radius (coordinate)")
    ]
    rho_ambient: Annotated[float, ProblemParam(1.0, description="uniform density")]
    p_ambient: Annotated[float, ProblemParam(0.1, description="uniform pressure")]

    nr: Annotated[int, ProblemParam(80, cli=True, description="R resolution")]
    nphi: Annotated[int, ProblemParam(64, cli=True, description="phi resolution")]
    resolution: Annotated[
        tuple[int, int], ProblemParam((0, 0), description="grid resolution — computed")
    ]
    r_center: Annotated[float, ProblemParam(8.0, description="loop center R")]
    phi_center: Annotated[float, ProblemParam(math.pi, description="loop center phi")]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.PLANAR_CYLINDRICAL, description="the (R, phi) equatorial disk"),
    ]
    regime: Annotated[Regime, ProblemParam(Regime.SRMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, cli=True, description="solver")]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.UCT, cli=True, description="CT edge-EMF method")
    ]
    # an off-hole full annulus: R outside r = 2M; phi wraps [0, 2pi).
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(4.0, 12.0), (0.0, 2.0 * math.pi)], description="(R, phi) domain"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.PERIODIC,
                BoundaryCondition.PERIODIC,
            ],
            description="R outflow, phi periodic",
        ),
    ]
    end_time: Annotated[
        float, ProblemParam(3.0, cli=True, checkpoint_safe=True, description="end time")
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrDiskFieldLoop":
        self.resolution = (self.nr, self.nphi)
        return self

    def _sqrtg(self, big_r: float) -> float:
        # sqrt(det gamma) = R sqrt(1 + 2M/R) on the equator (r = R); phi-independent.
        return big_r * math.sqrt(1.0 + 2.0 * self.schwarzschild_mass / big_r)

    def _potential(self, big_r: float, phi: float) -> float:
        """A_z(R, phi): a compact conical bump so B = curl(A_z zhat) is an in-plane loop.
        the phi lever arm uses r_center so the loop is near-isotropic in coordinate space."""
        s = math.hypot(big_r - self.r_center, self.r_center * (phi - self.phi_center))
        return self.amplitude * (self.loop_radius - s) if s < self.loop_radius else 0.0

    def radial_faces(self) -> list[float]:
        (rmin, rmax) = self.bounds[0]
        dr = (rmax - rmin) / self.nr
        return [rmin + ii * dr for ii in range(self.nr + 1)]

    def phi_faces(self) -> list[float]:
        (pmin, pmax) = self.bounds[1]
        dphi = (pmax - pmin) / self.nphi
        return [pmin + jj * dphi for jj in range(self.nphi + 1)]

    def initial_primitive_state(self) -> InitialStateType:
        nr, nphi = self.nr, self.nphi
        rf = self.radial_faces()
        pf = self.phi_faces()
        dr = rf[1] - rf[0]
        dphi = pf[1] - pf[0]
        r_c = [0.5 * (rf[i] + rf[i + 1]) for i in range(nr)]

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest: the MHD 5-tuple (rho, v_R, v_phi, v_z, pre).
            for _jj in range(nphi):
                for _ii in range(nr):
                    yield (self.rho_ambient, 0.0, 0.0, 0.0, self.p_ambient)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            a = self._potential
            if bn == "b1":
                # B_R on R-faces: (nr+1) x nphi. densitized curl B_R = d_phi A_z / (sqrt(g) dphi):
                # (A(R_f, phi_hi) - A(R_f, phi_lo)) / (sqrt(g)(R_f) dphi).
                for jj in range(nphi):
                    for ii in range(nr + 1):
                        w = self._sqrtg(rf[ii]) * dphi
                        yield (a(rf[ii], pf[jj + 1]) - a(rf[ii], pf[jj])) / w
            elif bn == "b2":
                # B_phi on phi-faces: nr x (nphi+1). densitized curl B_phi = -d_R A_z / (sqrt(g) dR):
                # -(A(R_hi, phi_f) - A(R_lo, phi_f)) / (sqrt(g)(R_c) dR).
                for jj in range(nphi + 1):
                    for ii in range(nr):
                        w = self._sqrtg(r_c[ii]) * dr
                        yield -(a(rf[ii + 1], pf[jj]) - a(rf[ii], pf[jj])) / w
            else:
                # the out-of-plane vertical B_z: zero (a pure in-plane loop).
                for _jj in range(nphi):
                    for _ii in range(nr):
                        yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
