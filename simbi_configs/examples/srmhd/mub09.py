# =============================================================================
# mub09.py
#
# mignone, ugliano, & bodo (2009), 1d srmhd test problems.
# =============================================================================
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Iterator, cast

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    MHDStateGenerators,
)


@dataclass(frozen=True)
class MHDState:
    rho: float
    v1: float
    v2: float
    v3: float
    p: float
    b1: float
    b2: float
    b3: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.v2
        yield self.v3
        yield self.p


class MUB09(SimbiProblem):
    """mignone, ugliano, & bodo (2009), 1d srmhd test problems."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    problem: Annotated[
        str,
        ProblemParam(
            "contact",
            cli=True,
            description="problem type (contact, rotational, st-1..st-4)",
        ),
    ]

    # domain
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((100, 1, 1), cli=True, description="grid resolution"),
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
        Regime, ProblemParam(Regime.RMHD, description="physics regime")
    ]

    # numerics
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLD, description="numerical solver")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            0.4,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time",
        ),
    ]

    @property
    def problem_states(self) -> dict[str, tuple[MHDState, MHDState]]:
        return {
            "contact": (
                MHDState(10.0, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
                MHDState(1.00, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
            ),
            "rotational": (
                MHDState(1.0, 0.4, -0.3, 0.5, 1.0, 2.4, 1.0, -1.6),
                MHDState(
                    1.0,
                    0.377347,
                    -0.482389,
                    0.424190,
                    1.0,
                    2.4,
                    -0.1,
                    -2.178213,
                ),
            ),
            "st-1": (
                MHDState(1.000, 0.0, 0.0, 0.0, 1.0, 0.5, +1.0, 0.0),
                MHDState(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0),
            ),
            "st-2": (
                MHDState(1.08, +0.40, +0.3, 0.2, 0.95, 2.0, +0.3, 0.3),
                MHDState(1.00, -0.45, -0.2, 0.2, 1.00, 2.0, -0.7, 0.5),
            ),
            "st-3": (
                MHDState(1.0, +0.999, 0.0, 0.0, 0.1, 10.0, +7.0, +7.0),
                MHDState(1.0, -0.999, 0.0, 0.0, 0.1, 10.0, -7.0, -7.0),
            ),
            "st-4": (
                MHDState(1.0, 0.0, 0.3, 0.4, 5.0, 1.0, 6.0, 2.0),
                MHDState(0.9, 0.0, 0.0, 0.0, 5.3, 1.0, 5.0, 2.0),
            ),
        }

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for srmhd shock tube."""

        def gas_state() -> GasStateGenerator:
            ni, nj, nk = self.resolution
            state = self.problem_states[self.problem]
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / ni

            for kk in range(nk):
                for jj in range(nj):
                    for ii in range(ni):
                        xi = xmin + (ii + 0.5) * dx
                        if xi < 0.5 * xextent:
                            yield tuple(state[0])
                        else:
                            yield tuple(state[1])

        def bfield(bn: str) -> GasStateGenerator:
            state = self.problem_states[self.problem]
            ni, nj, nk = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / ni

            for kk in range(nk + (bn == "b3")):
                for jj in range(nj + (bn == "b2")):
                    for ii in range(ni + (bn == "b1")):
                        xi = xmin + ii * dx
                        if xi < 0.5 * xextent:
                            yield getattr(state[0], bn)
                        else:
                            yield getattr(state[1], bn)

        bx_gen = partial(bfield, "b1")
        by_gen = partial(bfield, "b2")
        bz_gen = partial(bfield, "b3")

        return cast(MHDStateGenerators, (gas_state, bx_gen, by_gen, bz_gen))
