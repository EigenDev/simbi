# =============================================================================
# gr_bondi_ks_magnetized.py
#
# magnetized through-horizon bondi accretion on the ingoing kerr-schild chart —
# the `_ks` GRMHD gate: uniform gas at rest threaded by the divergence-free
# radial monopole sqrt(gamma) B^r = const, with the inner boundary BELOW r = 2M.
# the gas develops transonic accretion and crosses the horizon; the radial field
# aligned with the radial flow exerts zero lorentz force, so the flow matches the
# unmagnetized kerr-schild bondi while the SHIFTED riemann fan's magnetic rows are
# fully engaged. the load-bearing term: the true mag-row flux is
# (alpha v^n - beta^n) B^i - (alpha v^i - beta^i) B^n, so the fan's uniform
# `-(beta^n/alpha) U` subtraction must be undone on the B^r row by the induction
# TRANSPOSE `+(beta^r/alpha) B^n` — without it the radial field advects with the
# shift and B^r drifts. B^r staying bitwise static IS the transpose-term gate.
#
# the field: sqrt(gamma) = r^2 sqrt(1 + b) (per unit sin(theta), b = 2M/r), so
# the contravariant B^r = C / (r^2 sqrt(1 + b)); `b_ref` sets B^r at r = 2M.
#
# usage:
#   simbi run gr_bondi_ks_magnetized.py [--b-ref 0.5] [--resolution 256]
#   (the gate: simbi/simulation/tests/test_kerr_schild_bondi_magnetized.py)
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
    Spacetime,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)


class GrBondiKsMagnetized(SimbiProblem):
    """uniform gas at rest + a radial monopole, horizon-penetrating grid."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.KERR_SCHILD, description="background spacetime"),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, description="initial uniform rest density")
    ]
    p_ambient: Annotated[
        float, ProblemParam(1.0e-2, description="initial uniform pressure")
    ]
    b_ref: Annotated[
        float,
        ProblemParam(
            0.5, cli=True, description="contravariant B^r at r = 2M"
        ),
    ]
    resolution: Annotated[
        int, ProblemParam(256, cli=True, description="radial grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(1.5, 100.0)],
            description="radial domain; the inner edge sits BELOW the horizon",
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.SPHERICAL, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.SRMHD, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LOG, description="log-spaced radial zones"),
    ]
    boundary_conditions: Annotated[
        list[str],
        ProblemParam(
            [BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW],
            description="through-horizon inflow below; ambient reservoir above",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            10.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def radial_faces(self) -> list[float]:
        (rmin, rmax) = self.bounds[0]
        n = self.resolution
        q = (rmax / rmin) ** (1.0 / n)
        return [rmin * q**ii for ii in range(n + 1)]

    def monopole(self, r: float) -> float:
        """the divergence-free contravariant B^r(r) = C / (r^2 sqrt(1 + 2M/r)),
        normalized so B^r(2M) = b_ref."""
        mm = self.schwarzschild_mass
        r_h = 2.0 * mm
        cc = self.b_ref * r_h * r_h * math.sqrt(2.0)
        return cc / (r * r * math.sqrt(1.0 + 2.0 * mm / r))

    def initial_primitive_state(self) -> InitialStateType:
        n = self.resolution
        faces = self.radial_faces()

        def gas_state() -> GasStateGenerator:
            for _ in range(n):
                yield (self.rho_ambient, 0.0, 0.0, 0.0, self.p_ambient)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            if bn == "b1":
                for r in faces:
                    yield self.monopole(r)
            else:
                for _ in range(n):
                    yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
