# =============================================================================
# gr_michel_magnetized_2d.py
#
# the 2D (r, theta) magnetized michel monopole — the curved-CT instrument (design
# 44 phase B): the exact michel transonic hydro profile on an equatorial wedge,
# threaded by the theta-uniform radial monopole B^r = C sqrt(f)/r^2. the state is
# theta-uniform and purely radial, so the out-of-plane EMF
# E_phi = v_theta B_r - v_r B_theta vanishes POINTWISE at every gather point of
# the contact assembly — the staggered field must stay BITWISE static through the
# FULL curved-CT machinery (densitized corner EMF -> curl -> interpolation), and
# the hydro must hold the michel profile exactly as the 1D gate does. any wrong
# metric factor in the EMF/curl/interpolation chain breaks a known answer.
#
# reflecting theta walls are exact for this state (v_theta = 0, no theta
# gradients); the radial boundaries are outflow as in 1D.
#
# usage:
#   simbi run gr_michel_magnetized_2d.py [--b-ref 0.5] [--nr 128] [--npolar 16]
#   (the gate: simbi/simulation/tests/test_schwarzschild_michel_magnetized_2d.py)
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


class GrMichelMagnetized2D(SimbiProblem):
    """the michel profile + a force-free monopole on the (r, theta) wedge."""

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
    nr: Annotated[
        int, ProblemParam(128, cli=True, description="radial resolution")
    ]
    npolar: Annotated[
        int, ProblemParam(16, cli=True, description="polar (theta) resolution")
    ]
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((0, 0), description="grid resolution (nr, npolar) — computed"),
    ]
    theta_halfwidth: Annotated[
        float,
        ProblemParam(
            0.3, description="half-width of the equatorial theta wedge (rad)"
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(0.0, 0.0), (0.0, 0.0)], description="domain bounds — computed"
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
            [
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.OUTFLOW,
                BoundaryCondition.REFLECTING,
                BoundaryCondition.REFLECTING,
            ],
            description="outflow radial faces; reflecting theta walls (exact here)",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            10.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "GrMichelMagnetized2D":
        self.resolution = (self.nr, self.npolar)
        theta_c = math.pi / 2.0
        self.bounds = [
            (3.0, 100.0),
            (theta_c - self.theta_halfwidth, theta_c + self.theta_halfwidth),
        ]
        return self

    def michel_solution(self) -> MichelSolution:
        return MichelSolution(
            mass=self.schwarzschild_mass,
            gamma=self.adiabatic_index,
            rho_inf=self.rho_ambient,
            p_inf=self.p_ambient,
        )

    def radial_faces(self) -> list[float]:
        (rmin, rmax) = self.bounds[0]
        q = (rmax / rmin) ** (1.0 / self.nr)
        return [rmin * q**ii for ii in range(self.nr + 1)]

    def cell_centroids(self) -> list[float]:
        faces = self.radial_faces()
        return [
            0.75 * (rh**4 - rl**4) / (rh**3 - rl**3)
            for rl, rh in zip(faces[:-1], faces[1:])
        ]

    def monopole(self, r: float) -> float:
        mm = self.schwarzschild_mass
        r_in = self.bounds[0][0]
        f_in = 1.0 - 2.0 * mm / r_in
        cc = self.b_ref * r_in * r_in / math.sqrt(f_in)
        return cc * math.sqrt(1.0 - 2.0 * mm / r) / (r * r)

    def initial_primitive_state(self) -> InitialStateType:
        sol = self.michel_solution()
        centroids = self.cell_centroids()
        faces = self.radial_faces()
        npolar = self.npolar

        def gas_state() -> GasStateGenerator:
            for _jj in range(npolar):
                for r in centroids:
                    rho, v1, pre = sol.primitive(r)
                    yield (rho, v1, 0.0, 0.0, pre)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            if bn == "b1":
                # r-faces: (nr + 1) x npolar, axis-0-fastest; theta-uniform monopole.
                for _jj in range(npolar):
                    for r in faces:
                        yield self.monopole(r)
            elif bn == "b2":
                # theta-faces: nr x (npolar + 1); zero.
                for _jj in range(npolar + 1):
                    for _ii in range(self.nr):
                        yield 0.0
            else:
                # out-of-plane: cell-count field; zero.
                for _jj in range(npolar):
                    for _ii in range(self.nr):
                        yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
