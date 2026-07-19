# =============================================================================
# gr_cartesian_field_loop.py
#
# a poloidal magnetic field loop on a CARTESIAN kerr-schild (x, y) patch off the
# origin — the constrained-transport probe for the non-spherical
# GR chart. the cartesian kerr-schild spatial metric is NON-DIAGONAL
# (gamma_ij = delta_ij + 2M x_i x_j / r^3, r = sqrt(x^2 + y^2)), so sqrt(det gamma)
# = sqrt(1 + 2M/r) and the CT densitizes the corner EMF with it. a compact loop is
# seeded div-free through the METRIC-WEIGHTED discrete curl of a localized A_z (so
# the w-weighted divergence w = sqrt(gamma)(face) x coordinate length is machine
# zero by construction), and the full curved-CT machinery must PRESERVE it as the
# gas free-falls under the covariant geodesic source.
#
# what it certifies: the chart-generic densitized curl + the two-component-shift
# corner EMF hold the w-weighted div(B) at machine precision and run stably (no
# floors, no crash) on a genuinely 2D field in the cartesian chart. the gas flux is
# the fast-magnetosonic HLLE fan (the diagonal-metric HLLD wrapper does not apply to
# the non-diagonal cartesian metric — a tetrad follow-on).
#
# usage:
#   simbi run gr_cartesian_field_loop.py --ct-method uct
#   simbi run gr_cartesian_field_loop.py --ct-method contact
#   (the gate: simbi/simulation/tests/test_cartesian_grmhd.py)
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


class GrCartesianFieldLoop(SimbiProblem):
    """poloidal field loop on the cartesian kerr-schild (x, y) patch — the GR-CT probe."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.KERR_SCHILD, description="cartesian kerr-schild background"),
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

    nx: Annotated[int, ProblemParam(96, cli=True, description="x resolution")]
    ny: Annotated[int, ProblemParam(96, cli=True, description="y resolution")]
    resolution: Annotated[
        tuple[int, int], ProblemParam((0, 0), description="grid resolution — computed")
    ]
    x_center: Annotated[float, ProblemParam(8.0, description="loop center x")]
    y_center: Annotated[float, ProblemParam(0.0, description="loop center y")]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.RMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, cli=True, description="solver")]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.UCT, cli=True, description="CT edge-EMF method")
    ]
    # an off-origin square patch: r spans 4 (at (4,0)) to ~12.6, all outside r = 2M.
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(4.0, 12.0), (-4.0, 4.0)], description="(x, y) domain — off-origin"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.OUTFLOW] * 4, description="outflow on all four edges"
        ),
    ]
    end_time: Annotated[
        float, ProblemParam(4.0, cli=True, checkpoint_safe=True, description="end time")
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrCartesianFieldLoop":
        self.resolution = (self.nx, self.ny)
        return self

    def _sqrtg(self, x: float, y: float) -> float:
        # sqrt(det gamma) = sqrt(1 + 2M/r) for the cartesian kerr-schild metric.
        r = math.hypot(x, y)
        return math.sqrt(1.0 + 2.0 * self.schwarzschild_mass / r)

    def _potential(self, x: float, y: float) -> float:
        """A_z(x, y): a compact conical bump so B = curl(A_z zhat) is a localized loop."""
        s = math.hypot(x - self.x_center, y - self.y_center)
        return self.amplitude * (self.loop_radius - s) if s < self.loop_radius else 0.0

    def x_faces(self) -> list[float]:
        (xmin, xmax) = self.bounds[0]
        dx = (xmax - xmin) / self.nx
        return [xmin + ii * dx for ii in range(self.nx + 1)]

    def y_faces(self) -> list[float]:
        (ymin, ymax) = self.bounds[1]
        dy = (ymax - ymin) / self.ny
        return [ymin + jj * dy for jj in range(self.ny + 1)]

    def initial_primitive_state(self) -> InitialStateType:
        nx, ny = self.nx, self.ny
        xf = self.x_faces()
        yf = self.y_faces()
        dx = xf[1] - xf[0]
        dy = yf[1] - yf[0]
        x_c = [0.5 * (xf[i] + xf[i + 1]) for i in range(nx)]
        y_c = [0.5 * (yf[j] + yf[j + 1]) for j in range(ny)]

        def gas_state() -> GasStateGenerator:
            # uniform gas at rest: (rho, v_x, v_y, v_z, pre).
            for _jj in range(ny):
                for _ii in range(nx):
                    yield (self.rho_ambient, 0.0, 0.0, 0.0, self.p_ambient)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            a = self._potential
            if bn == "b1":
                # B_x on x-faces: (nx+1) x ny. densitized curl B_x = d_y A_z / (sqrt(g) dy):
                # (A(x_f, y_hi) - A(x_f, y_lo)) / (sqrt(g)(x_f, y_c) dy).
                for jj in range(ny):
                    for ii in range(nx + 1):
                        w = self._sqrtg(xf[ii], y_c[jj]) * dy
                        yield (a(xf[ii], yf[jj + 1]) - a(xf[ii], yf[jj])) / w
            elif bn == "b2":
                # B_y on y-faces: nx x (ny+1). densitized curl B_y = -d_x A_z / (sqrt(g) dx):
                # -(A(x_hi, y_f) - A(x_lo, y_f)) / (sqrt(g)(x_c, y_f) dx).
                for jj in range(ny + 1):
                    for ii in range(nx):
                        w = self._sqrtg(x_c[ii], yf[jj]) * dx
                        yield -(a(xf[ii + 1], yf[jj]) - a(xf[ii], yf[jj])) / w
            else:
                for _jj in range(ny):
                    for _ii in range(nx):
                        yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
