# =============================================================================
# magnetic_shock_tube.py
#
# mignone & bodo (2006), relativistic mhd test problems in 1d mesh.
# =============================================================================
from dataclasses import dataclass
from functools import partial
from typing import Any, Sequence, cast

from pydantic import model_validator

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    MHDStateGenerators,
    StaggeredBFieldGenerator,
)


@dataclass(frozen=True)
class ShockTubeState:
    """left and right states for shock tube problems."""

    rho: float
    vx: float
    vy: float
    vz: float
    p: float
    bx: float
    by: float
    bz: float


@dataclass(frozen=True)
class MHDProblemState:
    """complete state for mhd shock tube problems."""

    left: ShockTubeState
    right: ShockTubeState

    @classmethod
    def create_state(
        cls, left_vals: Sequence[float], right_vals: Sequence[float]
    ) -> "MHDProblemState":
        return cls(
            left=ShockTubeState(*left_vals), right=ShockTubeState(*right_vals)
        )


class MagneticShockTube(SimbiProblem):
    """mignone & bodo (2006), relativistic mhd test problems in 1d mesh."""

    # physics
    problem: int = ProblemParam(1, cli=True, description="problem number (1-4)")
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic index"
    )

    # domain
    resolution: tuple[int, int, int] = ProblemParam(
        (1600, 1, 1), cli=True, description="grid resolution"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(0.0, 1.0)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(Regime.SRMHD, description="physics regime")

    # numerics
    solver: Solver = ProblemParam(
        Solver.HLLE, description="solver type for mhd"
    )
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # simulation control
    end_time: float = ProblemParam(
        0.4, cli=True, checkpoint_safe=True, description="simulation end time"
    )

    @model_validator(mode="after")
    def set_adiabatic_index_by_problem(self) -> "MagneticShockTube":
        """set adiabatic index based on problem number."""
        if self.problem == 1:
            object.__setattr__(self, "adiabatic_index", 2.0)
        else:
            object.__setattr__(self, "adiabatic_index", 5.0 / 3.0)
        return self

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(
            self,
            "_problem_states",
            {
                1: MHDProblemState.create_state(
                    (1.000, 0.0, 0.0, 0.0, 1.0, 0.5, +1.0, 0.0),
                    (0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0),
                ),
                2: MHDProblemState.create_state(
                    (1.0, 0.0, 0.0, 0.0, 30.0, 5.0, 6.0, 6.0),
                    (1.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.7, 0.7),
                ),
                3: MHDProblemState.create_state(
                    (1.0, 0.0, 0.0, 0.0, 1e3, 10.0, 7.0, 7.0),
                    (1.0, 0.0, 0.0, 0.0, 0.1, 10.0, 0.7, 0.7),
                ),
                4: MHDProblemState.create_state(
                    (1.0, +0.999, 0.0, 0.0, 0.1, 10.0, +7.0, +7.0),
                    (1.0, -0.999, 0.0, 0.0, 0.1, 10.0, -7.0, -7.0),
                ),
            },
        )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for mhd shock tube."""

        def gas_state() -> GasStateGenerator:
            state = self._problem_states[self.problem]
            ni, nj, nk = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / ni

            for kk in range(nk):
                for jj in range(nj):
                    for ii in range(ni):
                        xi = self.bounds[0][0] + ii * dx
                        if xi < 0.5:
                            yield (
                                state.left.rho,
                                state.left.vx,
                                state.left.vy,
                                state.left.vz,
                                state.left.p,
                            )
                        else:
                            yield (
                                state.right.rho,
                                state.right.vx,
                                state.right.vy,
                                state.right.vz,
                                state.right.p,
                            )

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            state = self._problem_states[self.problem]
            ni, nj, nk = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / ni

            for kk in range(nk + (bn == "bz")):
                for jj in range(nj + (bn == "by")):
                    for ii in range(ni + (bn == "bx")):
                        xi = self.bounds[0][0] + ii * dx
                        if xi < 0.5:
                            yield getattr(state.left, bn)
                        else:
                            yield getattr(state.right, bn)

        bx_gen = partial(b_field, "bx")
        by_gen = partial(b_field, "by")
        bz_gen = partial(b_field, "bz")

        return cast(MHDStateGenerators, (gas_state, bx_gen, by_gen, bz_gen))
