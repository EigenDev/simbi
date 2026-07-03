# =============================================================================
# gr_kerr_field_loop.py
#
# a poloidal field loop advected radially through a SPINNING kerr (r, theta)
# wedge (ingoing kerr-schild coords) — the GRMHD phase-C smoke/dispatch gate
# (design 44 phase C). it is the gr_field_loop probe lifted to the kerr metric:
# a weak passive loop, seeded div-free via the METRIC-WEIGHTED discrete curl of a
# localized A_phi with the KERR sqrt(gamma) = Sigma sin(theta) sqrt(1 + 2Mr/Sigma)
# (so the w-weighted divergence is machine zero by construction on the kerr grid),
# carried inward through a uniform inflow. it exercises the full spinning-kerr
# RMHD kernel path — the tetrad HLLD on the NON-DIAGONAL gamma_{r phi}, the
# moving-interface radial shift, the covariant EM-stress source with the swirl
# (3-component) momentum, the metric-aware c2p, and the kerr-wired contact CT.
#
# usage:
#   simbi run gr_kerr_field_loop.py --kerr-spin 0.9 --ct-method contact
# =============================================================================

import math
from functools import partial
from typing import Annotated

from pydantic import model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
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


class GrKerrFieldLoop(SimbiProblem):
    """advected poloidal field loop on a spinning-kerr wedge — the GRMHD phase-C probe."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]
    spacetime: Annotated[
        Spacetime, ProblemParam(Spacetime.KERR, description="background spacetime")
    ]
    schwarzschild_mass: Annotated[
        float, ProblemParam(1.0, cli=True, description="black-hole mass M")
    ]
    kerr_spin: Annotated[
        float, ProblemParam(0.9, cli=True, description="dimensionless spin a (|a| < M)")
    ]
    amplitude: Annotated[
        float, ProblemParam(1.0e-3, cli=True, description="vector-potential amplitude A0")
    ]
    loop_radius: Annotated[
        float, ProblemParam(1.5, cli=True, description="loop radius R (coordinate)")
    ]
    inflow: Annotated[
        float, ProblemParam(0.2, cli=True, description="radial inflow speed |v^r|")
    ]
    rho_ambient: Annotated[float, ProblemParam(1.0, description="uniform density")]
    p_ambient: Annotated[float, ProblemParam(0.1, description="uniform pressure")]

    nr: Annotated[int, ProblemParam(128, cli=True, description="radial resolution")]
    npolar: Annotated[int, ProblemParam(64, cli=True, description="polar resolution")]
    resolution: Annotated[
        tuple[int, int], ProblemParam((0, 0), description="grid resolution — computed")
    ]
    r_center: Annotated[float, ProblemParam(8.0, description="loop center radius")]
    theta_halfwidth: Annotated[
        float, ProblemParam(0.4, description="polar wedge half-width (rad)")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 0.0), (0.0, 0.0)], description="domain bounds — computed"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.SPHERICAL, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.SRMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, cli=True, description="solver")]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.CONTACT, cli=True, description="CT edge-EMF method")
    ]
    x1_spacing: Annotated[
        CellSpacing, ProblemParam(CellSpacing.LINEAR, description="radial spacing")
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.REFLECTING,
                BoundaryCondition.REFLECTING,
            ],
            description="outflow radial; reflecting theta walls",
        ),
    ]
    end_time: Annotated[
        float, ProblemParam(6.0, cli=True, checkpoint_safe=True, description="end time")
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrKerrFieldLoop":
        self.resolution = (self.nr, self.npolar)
        theta_c = math.pi / 2.0
        # domain OUTSIDE the outer horizon r_+ = M + sqrt(M^2 - a^2).
        r_plus = self.schwarzschild_mass + math.sqrt(
            max(self.schwarzschild_mass**2 - self.kerr_spin**2, 0.0)
        )
        (r_lo, r_hi) = (max(4.0, 1.05 * r_plus), 16.0)
        self.bounds = [
            (r_lo, r_hi),
            (theta_c - self.theta_halfwidth, theta_c + self.theta_halfwidth),
        ]
        return self

    def _sqrtg(self, r: float, th: float) -> float:
        # kerr sqrt(gamma) = Sigma sin(theta) sqrt(1 + 2Mr/Sigma) (ingoing kerr-schild).
        mm, a = self.schwarzschild_mass, self.kerr_spin
        sigma = r * r + a * a * math.cos(th) ** 2
        b = 2.0 * mm * r / sigma
        return sigma * math.sin(th) * math.sqrt(1.0 + b)

    def _azimuthal_potential(self, r: float, th: float) -> float:
        """A_phi as a compact conical bump so B = curl(A_phi) is a localized poloidal loop."""
        r_c, th_c = self.r_center, math.pi / 2.0
        s = math.hypot(r - r_c, r_c * (th - th_c))
        return self.amplitude * (self.loop_radius - s) if s < self.loop_radius else 0.0

    def radial_faces(self) -> list[float]:
        (rmin, rmax) = self.bounds[0]
        dr = (rmax - rmin) / self.nr
        return [rmin + ii * dr for ii in range(self.nr + 1)]

    def theta_faces(self) -> list[float]:
        (tmin, tmax) = self.bounds[1]
        dth = (tmax - tmin) / self.npolar
        return [tmin + jj * dth for jj in range(self.npolar + 1)]

    def initial_primitive_state(self) -> InitialStateType:
        nr, npolar = self.nr, self.npolar
        rf = self.radial_faces()
        tf = self.theta_faces()
        dr = rf[1] - rf[0]
        dth = tf[1] - tf[0]
        r_c = [0.5 * (rf[i] + rf[i + 1]) for i in range(nr)]
        th_c = [0.5 * (tf[j] + tf[j + 1]) for j in range(npolar)]
        v_in = -abs(self.inflow)

        def gas_state() -> GasStateGenerator:
            for _jj in range(npolar):
                for _ii in range(nr):
                    yield (self.rho_ambient, v_in, 0.0, 0.0, self.p_ambient)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            a = self._azimuthal_potential
            if bn == "b1":
                # B_r on r-faces: metric-weighted curl B_r = dA/dtheta / (sqrt(g)(r_f, th_c) dth).
                for jj in range(npolar):
                    for ii in range(nr + 1):
                        w = self._sqrtg(rf[ii], th_c[jj]) * dth
                        yield (a(rf[ii], tf[jj + 1]) - a(rf[ii], tf[jj])) / w
            elif bn == "b2":
                # B_theta on theta-faces: B_theta = -dA/dr / (sqrt(g)(r_c, th_f) dr).
                for jj in range(npolar + 1):
                    for ii in range(nr):
                        w = self._sqrtg(r_c[ii], tf[jj]) * dr
                        yield -(a(rf[ii + 1], tf[jj]) - a(rf[ii], tf[jj])) / w
            else:
                for _jj in range(npolar):
                    for _ii in range(nr):
                        yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
