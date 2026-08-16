# =============================================================================
# refined_atmosphere.py
#
# a stratified atmosphere held exactly across a coarse-fine interface.
#
# a hydrostatic atmosphere solves the continuum equations, not the discrete ones, so any
# scheme leaves a residual at truncation order and gas seeded on the exact profile starts
# moving. refinement makes it worse: the two grids reduce the same exact solution to
# different face values, the flux register differences them, and the difference acts on the
# coarse cells at the interface as a force. that force is lower order than either grid, so
# it caps the convergence rate of any stratified problem.
#
# declaring the atmosphere as the run's stationary target removes it. the backend measures
# the target's discrete imbalance once per level and adds it back at every stage, and the
# flux registers difference deviations from the target rather than the state, which holds
# the atmosphere to roundoff on every level while conserving mass exactly.
#
# the potential is declared once. the equilibrium is the closed-form inversion of hydrostatic
# balance against it, and the gravity source is its gradient, so the profile and the force it
# balances cannot disagree — which is the mistake that otherwise produces a smooth, plausible,
# slowly-collapsing atmosphere.
#
# run it and watch max|v|: without the declaration the gas is moving at ~1e-4 within a few
# hundred steps, with it the atmosphere stands still.
# =============================================================================
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.typing import (
    ExpressionDict,
    GasStateGenerator,
    InitialStateType,
)


class RefinedAtmosphere(SimbiProblem):
    """isentropic atmosphere in a point-mass potential, refined and well-balanced."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    gm: Annotated[
        float,
        ProblemParam(100.0, cli=True, description="gravitating mass GM"),
    ]
    entropy: Annotated[
        float, ProblemParam(1.0, description="entropy constant K in p = K rho^gamma")
    ]
    offset: Annotated[
        float,
        ProblemParam(
            1.0,
            description="the point mass sits this far left of x = 0, so the gas at x "
            "feels a bare point mass at radius x + offset and the domain "
            "covers r in [1, 2] with no singularity",
        ),
    ]

    # domain
    resolution: Annotated[
        tuple[int], ProblemParam((128,), cli=True, description="root grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]

    # refinement: one nested box in the middle, which is where the interface the
    # well-balancing is aimed at lives.
    refinement_enabled: Annotated[
        bool, ProblemParam(True, description="enable mesh refinement")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(2, description="coarse + 1 fine")
    ]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam([[0.3, 0.7]], description="one fine box: x in [0.3, 0.7]"),
    ]
    refinement_ratios: Annotated[
        list[int], ProblemParam([2], description="refinement ratio per level")
    ]

    # numerics. a reflecting wall exerts no work on gas at rest, so the atmosphere is a
    # fixed point of the boundary as well as of the interior.
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.REFLECTING], description="boundary conditions [x]"
        ),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]

    # seed every level from the declared target rather than from a pointwise sample of the
    # profile: cells covered by the fine level carry the restriction of the fine target,
    # which is what the hierarchy's own restriction reproduces every parent step.
    seed_from_equilibrium: Annotated[
        bool,
        ProblemParam(
            True, cli=True, description="start the run exactly on the declared target"
        ),
    ]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/refined_atmosphere"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            1.0, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def _potential(self, graph: expr.ExprGraph) -> expr.Expr:
        """phi = -GM/r with r = x + offset: a bare point mass sitting off the left edge."""
        radius = expr.variable("x1", graph) + expr.constant(self.offset, graph)
        return -expr.constant(self.gm, graph) / radius

    def _atmosphere(self, graph: expr.ExprGraph) -> expr.Equilibrium:
        """the isentropic atmosphere in balance against that potential, normalized to
        rho = 1 at the outer edge."""
        return expr.isentropic_atmosphere(
            self._potential(graph),
            gamma=self.adiabatic_index,
            k_entropy=self.entropy,
            dim=1,
            reference_density=1.0,
            reference_point=[self.bounds[0][1]],
        )

    @computed_field
    @property
    def equilibrium_expressions(self) -> ExpressionDict:
        """the atmosphere, declared as the state the scheme must hold exactly."""
        graph = expr.ExprGraph()
        atmosphere = self._atmosphere(graph)
        return graph.compile(atmosphere.primitives).serialize_equilibrium(dim=1)

    @computed_field
    @property
    def source_expressions(self) -> list[ExpressionDict]:
        """the gravity the atmosphere is in balance against, as the gradient of the SAME
        potential the equilibrium was derived from."""
        graph = expr.ExprGraph()
        atmosphere = self._atmosphere(graph)
        compiled = graph.compile(atmosphere.acceleration)
        return [compiled.serialize_source(expr.SourceKind.FORCE, dim=1)]

    def initial_primitive_state(self) -> InitialStateType:
        """the atmosphere itself, sampled pointwise.

        `seed_from_equilibrium` overwrites this with the hierarchy-consistent form of the
        same state; what this generator supplies is the shape of the initial condition and
        the place to add a perturbation on top of the equilibrium.
        """

        def gas_state() -> GasStateGenerator:
            graph = expr.ExprGraph()
            compiled = graph.compile(self._atmosphere(graph).primitives)
            (ncells,) = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / ncells
            for ii in range(ncells):
                x = xmin + (ii + 0.5) * dx
                rho, velocity, pressure = compiled.evaluate(x1=x)
                yield (rho, velocity, pressure)

        return gas_state
