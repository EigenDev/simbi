# =============================================================================
# field_loop.py
#
# the magnetic field-loop advection test (Gardiner & Stone 2005, section 8.6) —
# THE canonical constrained-transport / checkerboard probe. a weak, passive
# magnetic loop is advected diagonally across a periodic domain at a constant
# velocity. the exact solution is the loop translating UNDISTORTED with its
# magnetic energy conserved.
#
# the diagnostic is the in-plane current density J_z = (curl B)_z: a CT scheme
# with INSUFFICIENT edge-EMF dissipation (arithmetic / contact) seeds a
# grid-scale "checkerboard" odd-even mode INSIDE the loop, visible as salt-and-
# pepper noise in J_z; the upwind schemes (UCT-HLL / UCT-HLLD) damp it and keep
# J_z smooth. compare --ct-method contact vs uct (+ --solver hlld).
#
# usage:
#  simbi run field_loop.py --ct-method contact          # checkerboard-prone
#  simbi run field_loop.py --ct-method uct --solver hlle # suppressed (HLL EMF)
#
# verified (NMHD, 128x64, t=2): grid-scale fraction of J_z is 6.7% (contact) vs
# 0.33% (UCT-HLL) / 0.37% (UCT-HLLD) — a ~20x checkerboard suppression; UCT-HLLD
# retains ~27% more field than UCT-HLL (less diffusive). all stable, gas uniform.
#
# NOTE the velocity MUST be supersonic (default |v|=sqrt5, M&DZ v=(2,1)) — the test
# is designed so numerical diffusion is set by cell-crossing;
# subsonic advection smears the loop. caveat: the RELATIVISTIC gas HLLD flux is
# fragile on this beta>>1 passive loop (use --regime nmhd, or --solver hllc for srmhd).
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)


class FieldLoop(SimbiProblem):
    """advected magnetic field loop — the CT checkerboard test."""

    # physics
    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="adiabatic index")]
    amplitude: Annotated[
        float, ProblemParam(1.0e-3, cli=True, description="vector-potential amplitude A0 (weak/passive)")
    ]
    loop_radius: Annotated[
        float, ProblemParam(0.3, cli=True, description="field-loop radius R")
    ]
    speed: Annotated[
        float,
        ProblemParam(
            math.sqrt(5.0),  # Gardiner-Stone / M&DZ: v = (2,1) => |v| = sqrt(5), SUPERSONIC (cs~1.29).
            cli=True,
            description="advection |v| (sqrt5 = paper's supersonic v=(2,1); use <1 subluminal for SRMHD)",
        ),
    ]

    # domain (2:1 box so the diagonal advection is non-trivial; periodic)
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((128, 64, 1), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-1.0, 1.0), (-0.5, 0.5)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    # NMHD by default: the field loop is the classical Gardiner-Stone test, and the classical gas
    # HLLD is robust on the weak (passive) field. --regime srmhd runs the relativistic version (note:
    # the relativistic gas HLLD flux is fragile at beta >> 1 — use --solver hllc there).
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, cli=True, description="physics regime")]

    # numerics
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, cli=True, description="numerical solver")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.PERIODIC], description="boundary conditions"),
    ]
    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LINEAR, description="x1 spacing")]

    # control
    start_time: Annotated[float, ProblemParam(0.0, description="start time")]
    end_time: Annotated[
        float, ProblemParam(2.0, cli=True, checkpoint_safe=True, description="end time")
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """uniform rho/p, constant diagonal velocity, div-free loop B from A_z."""
        # diagonal advection (Gardiner-Stone direction 2:1), scaled to |v| = speed.
        norm = math.sqrt(5.0)
        vx = self.speed * 2.0 / norm
        vy = self.speed * 1.0 / norm

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            for _ in range(ni * nj * nk):
                yield (1.0, vx, vy, 0.0, 1.0)  # rho, vx, vy, vz, p

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            ni, nj, nk = self.resolution
            (x0, x1), (y0, y1) = self.bounds[0], self.bounds[1]
            dx = (x1 - x0) / ni
            dy = (y1 - y0) / nj
            a0, rad = self.amplitude, self.loop_radius

            # vector potential at a CORNER (x, y): A_z = A0 (R - r) inside the loop, 0 outside.
            # the staggered B = discrete curl of A_z is EXACTLY divergence-free.
            def az(x: float, y: float) -> float:
                r = math.hypot(x, y)
                return a0 * (rad - r) if r < rad else 0.0

            for _kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        if bn == "bx":
                            # bx on the x-face: B_x = dA_z/dy across the two corners in y.
                            x = x0 + ii * dx
                            yield (az(x, y0 + (jj + 1) * dy) - az(x, y0 + jj * dy)) / dy
                        elif bn == "by":
                            # by on the y-face: B_y = -dA_z/dx across the two corners in x.
                            y = y0 + jj * dy
                            yield -(az(x0 + (ii + 1) * dx, y) - az(x0 + ii * dx, y)) / dx
                        else:
                            yield 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
