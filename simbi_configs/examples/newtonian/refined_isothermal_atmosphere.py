# =============================================================================
# refined_isothermal_atmosphere.py
#
# an isothermal atmosphere held exactly across a coarse-fine interface.
#
# the energy-free counterpart of the isentropic case: with `p = cs^2 rho` the equation of
# state supplies the pressure from the density and the regime stores none, so the declared
# target carries a density and a velocity and nothing else. hydrostatic balance is then
# `grad(ln rho) = -grad phi / cs^2`, which integrates to an exponential in the potential
# rather than a power of it — a genuinely different profile, not the gamma -> 1 limit of the
# adiabatic one.
#
# the declared sound speed must be the one the run is configured with. an atmosphere built
# for a different cs is not a steady state of this run, and the backend's refinement check
# rejects it rather than quietly holding the wrong state still.
#
# the potential is declared once: the equilibrium is the closed-form inversion of hydrostatic
# balance against it, and the gravity source is its gradient.
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


class RefinedIsothermalAtmosphere(SimbiProblem):
    """isothermal atmosphere in a point-mass potential, refined and well-balanced."""

    # physics. the potential depth in units of cs^2 sets the density contrast: the potential
    # difference across the domain is GM/2, so the contrast is exp(GM / (2 cs^2)).
    gm: Annotated[
        float, ProblemParam(4.0, cli=True, description="gravitating mass GM")
    ]
    sound_speed: Annotated[
        float, ProblemParam(1.0, description="constant isothermal sound speed")
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
        Regime,
        ProblemParam(Regime.ISOTHERMAL, description="isothermal: p = cs^2 rho"),
    ]

    # refinement: one nested box in the middle, where the interface lives.
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

    # a reflecting wall exerts no work on gas at rest, so the atmosphere is a fixed point of
    # the boundary as well as of the interior.
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.REFLECTING], description="boundary conditions [x]"
        ),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]
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
            Path("data/refined_isothermal_atmosphere"),
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

    def _potential(self) -> expr.Expr:
        """phi = -GM/r with r = x + offset: a bare point mass sitting off the left edge."""
        (x,) = expr.coords(1)
        return -self.gm / (x + self.offset)

    def _atmosphere(self) -> expr.Equilibrium:
        """rho = exp(-(phi - phi_ref)/cs^2), normalized to rho = 1 at the outer edge."""
        return expr.isothermal_atmosphere(
            self._potential(),
            sound_speed=self.sound_speed,
            dim=1,
            reference_density=1.0,
            reference_point=[self.bounds[0][1]],
        )

    @computed_field
    @property
    def equilibrium_expressions(self) -> ExpressionDict:
        """the atmosphere, declared as the state the scheme must hold exactly. two
        components — density and velocity — because an isothermal regime stores no
        pressure."""
        return expr.equilibrium(self._atmosphere().primitives, dim=1)

    @computed_field
    @property
    def source_expressions(self) -> list[ExpressionDict]:
        """the gravity the atmosphere is in balance against, as the gradient of the SAME
        potential the equilibrium was derived from."""
        return [expr.force(self._atmosphere().acceleration, dim=1)]

    def initial_primitive_state(self) -> InitialStateType:
        """the atmosphere itself, sampled pointwise; `seed_from_equilibrium` replaces it
        with the hierarchy-consistent form of the same state."""

        def gas_state() -> GasStateGenerator:
            primitives = self._atmosphere().primitives
            compiled = primitives[0].graph.compile(primitives)
            (ncells,) = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / ncells
            for ii in range(ncells):
                x = xmin + (ii + 0.5) * dx
                density, velocity = compiled.evaluate(x1=x)
                yield (density, velocity)

        return gas_state
