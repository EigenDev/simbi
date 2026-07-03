# =============================================================================
# flat_srmhd_monopole.py
#
# the FLAT (Minkowski) SRMHD analog of the magnetized-michel monopole — the
# control that isolates INHERENT HLLD-RMHD fragility from any GR-generalization
# effect. a theta-uniform radial monopole B^r = C/r^2 (divergence-free on the
# flat spherical (r, theta) wedge, since div B = (1/r^2) d_r(r^2 B^r) = 0) in a
# uniform gas with a radial inflow. E_phi = v_theta B_r - v_r B_theta = 0
# pointwise, so B must stay static and no theta-momentum may grow — under ANY
# consistent scheme. if flat HLLD develops the same resolution-growing
# theta-momentum mode the schwarzschild monopole does, the fragility is inherent
# to HLLD-RMHD (not the curved-metric generalization).
#
# usage:
#   simbi run flat_srmhd_monopole.py --solver hlld --ct-method uct
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


class FlatSrmhdMonopole(SimbiProblem):
    """flat (Minkowski) SRMHD radial monopole on a spherical wedge — the HLLD control."""

    adiabatic_index: Annotated[float, ProblemParam(4.0 / 3.0, description="adiabatic index")]
    spacetime: Annotated[
        Spacetime, ProblemParam(Spacetime.MINKOWSKI, description="flat background")
    ]
    b_ref: Annotated[
        float, ProblemParam(0.5, cli=True, description="contravariant B^r at the inner radius")
    ]
    inflow: Annotated[
        float, ProblemParam(0.5, cli=True, description="radial inflow |v^r| (supersonic for the stiff test)")
    ]
    rho_ambient: Annotated[float, ProblemParam(1.0, description="uniform density")]
    p_ambient: Annotated[float, ProblemParam(1.0e-2, description="uniform pressure (low beta near r_in)")]

    nr: Annotated[int, ProblemParam(128, cli=True, description="radial resolution")]
    npolar: Annotated[int, ProblemParam(16, cli=True, description="polar resolution")]
    resolution: Annotated[
        tuple[int, int], ProblemParam((0, 0), description="grid resolution — computed")
    ]
    theta_halfwidth: Annotated[float, ProblemParam(0.3, description="polar wedge half-width")]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 0.0), (0.0, 0.0)], description="domain bounds — computed"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.SPHERICAL, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.SRMHD, description="physics regime")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, cli=True, description="solver")]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.UCT, cli=True, description="CT edge-EMF method")
    ]
    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LOG, description="radial spacing")]
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
        float, ProblemParam(10.0, cli=True, checkpoint_safe=True, description="end time")
    ]

    @model_validator(mode="after")
    def compute_defaults(self) -> "FlatSrmhdMonopole":
        self.resolution = (self.nr, self.npolar)
        theta_c = math.pi / 2.0
        self.bounds = [
            (3.0, 100.0),
            (theta_c - self.theta_halfwidth, theta_c + self.theta_halfwidth),
        ]
        return self

    def radial_faces(self) -> list[float]:
        (rmin, rmax) = self.bounds[0]
        q = (rmax / rmin) ** (1.0 / self.nr)
        return [rmin * q**ii for ii in range(self.nr + 1)]

    def monopole(self, r: float) -> float:
        # B^r = C / r^2, normalized so B^r(r_in) = b_ref (div-free on flat spherical).
        r_in = self.bounds[0][0]
        return self.b_ref * r_in * r_in / (r * r)

    def initial_primitive_state(self) -> InitialStateType:
        nr, npolar = self.nr, self.npolar
        faces = self.radial_faces()
        v_in = -abs(self.inflow)

        def gas_state() -> GasStateGenerator:
            for _jj in range(npolar):
                for _ii in range(nr):
                    yield (self.rho_ambient, v_in, 0.0, 0.0, self.p_ambient)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            if bn == "b1":
                for _jj in range(npolar):
                    for r in faces:
                        yield self.monopole(r)
            elif bn == "b2":
                for _jj in range(npolar + 1):
                    for _ii in range(nr):
                        yield 0.0
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
