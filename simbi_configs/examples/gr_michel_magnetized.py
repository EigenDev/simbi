# =============================================================================
# gr_michel_magnetized.py
#
# the magnetized michel monopole — the GRMHD wiring gate (design 44 phase A): the
# exact michel (1972) transonic hydro profile threaded by a radial monopole field
# sqrt(gamma) B^r = const on the schwarzschild grid. a radial field aligned with a
# radial flow exerts ZERO lorentz force (E = -v x B = 0, J = 0), so the stationary
# solution is EXACTLY the unmagnetized michel hydro profile — while the magnetic
# terms in U, F, the covariant source, and the KKC recovery are all fully engaged
# and must cancel: any wrong magnetic term breaks a known hold. the induction
# equation is trivially static in 1D (the radial B row's flux is identically
# zero), so B^r must not change AT ALL.
#
# the field: div(B) = (1/sqrt(gamma)) d_r (sqrt(gamma) B^r) = 0 with sqrt(gamma) =
# r^2/sqrt(f) (per unit sin(theta)) gives the contravariant B^r = C sqrt(f)/r^2;
# `b_ref` sets B^r at the INNER boundary radius (the strongest-field point, where
# the magnetization sigma = b^2/rho peaks).
#
# usage:
#   simbi run gr_michel_magnetized.py [--b-ref 0.5] [--resolution 256]
#   (the gate: simbi/simulation/tests/test_schwarzschild_michel_magnetized.py)
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
    Regime,
    Spacetime,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)
from simbi_configs.examples.gr_michel import MichelSolution


class GrMichelMagnetized(SimbiProblem):
    """the michel profile with a force-free radial monopole field, held on the grid."""

    adiabatic_index: Annotated[
        float, ProblemParam(4.0 / 3.0, description="adiabatic index gamma")
    ]
    spacetime: Annotated[
        Spacetime,
        ProblemParam(Spacetime.SCHWARZSCHILD, description="background spacetime"),
    ]
    schwarzschild_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="black-hole mass M (G=c=1)"),
    ]
    rho_ambient: Annotated[
        float, ProblemParam(1.0, description="rest density at the outer boundary")
    ]
    p_ambient: Annotated[
        float, ProblemParam(1.0e-4, description="pressure at the outer boundary")
    ]
    b_ref: Annotated[
        float,
        ProblemParam(
            0.5,
            cli=True,
            description="contravariant B^r at the inner boundary radius",
        ),
    ]
    resolution: Annotated[
        int, ProblemParam(256, cli=True, description="radial grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(3.0, 100.0)], description="radial domain bounds (r > 2M)"),
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
            description="outflow at the horizon side, ambient at the outer edge",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            10.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def michel_solution(self) -> MichelSolution:
        return MichelSolution(
            mass=self.schwarzschild_mass,
            gamma=self.adiabatic_index,
            rho_inf=self.rho_ambient,
            p_inf=self.p_ambient,
        )

    def radial_faces(self) -> list[float]:
        (rmin, rmax) = self.bounds[0]
        n = self.resolution
        q = (rmax / rmin) ** (1.0 / n)
        return [rmin * q**ii for ii in range(n + 1)]

    def cell_centroids(self) -> list[float]:
        """volume-weighted cell centroids of the log-spaced radial grid — the same
        radii the backend evaluates the metric at when storing the conserved state."""
        faces = self.radial_faces()
        return [
            0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)
            for rl, rh in zip(faces[:-1], faces[1:])
        ]

    def monopole(self, r: float) -> float:
        """the divergence-free contravariant B^r(r) = C sqrt(f)/r^2, normalized so
        B^r(r_in) = b_ref."""
        mm = self.schwarzschild_mass
        r_in = self.bounds[0][0]
        f_in = 1.0 - 2.0 * mm / r_in
        cc = self.b_ref * r_in * r_in / math.sqrt(f_in)
        return cc * math.sqrt(1.0 - 2.0 * mm / r) / (r * r)

    def initial_primitive_state(self) -> InitialStateType:
        sol = self.michel_solution()
        centroids = self.cell_centroids()
        faces = self.radial_faces()

        def gas_state() -> GasStateGenerator:
            for r in centroids:
                rho, v1, pre = sol.primitive(r)
                yield (rho, v1, 0.0, 0.0, pre)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            if bn == "b1":
                for r in faces:
                    yield self.monopole(r)
            else:
                # transverse faces are ungridded in 1D: cell-count fields, zero.
                for _ in centroids:
                    yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
