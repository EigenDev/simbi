# =============================================================================
# k99.py
#
# komissarov (1999), 1d srmhd test problems.
# =============================================================================
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime, Solver
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    MHDStateGenerators,
    StaggeredBFieldGenerator,
)

# constants
XMIN = -2.0
XMAX = +2.0
XMEMBRANE = 0.5 * (XMIN + XMAX)


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

    @staticmethod
    def beta(u: Sequence[float]) -> NDArray[np.float64]:
        """calculate relativistic beta from 3-velocity."""
        unp: NDArray[np.float64] = np.asarray(u)
        gamma: float = (1.0 + unp.dot(unp)) ** (0.5)
        return unp / gamma

    @classmethod
    def create_state(
        cls, left_vals: Sequence[float], right_vals: Sequence[float]
    ) -> "MHDProblemState":
        return cls(
            left=ShockTubeState(*left_vals), right=ShockTubeState(*right_vals)
        )


class StaggeredMHDState(NamedTuple):
    """container that stores face-centered magnetic fields
    and cell-centered gas variables."""

    gas_vars: tuple[float, ...]
    staggered_bfields: list[list[float]]


class K99(SimbiProblem):
    """komissarov (1999), 1d srmhd test problems."""

    # physics
    adiabatic_index: float = ProblemParam(
        4.0 / 3.0, description="adiabatic index"
    )
    problem: str = ProblemParam(
        "fast-shock",
        cli=True,
        description="problem type (fast-shock, slow-shock, fast-rarefaction, etc.)",
    )

    # domain
    resolution: tuple[int, int, int] = ProblemParam(
        (100, 1, 1), cli=True, description="grid resolution"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(XMIN, XMAX)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(Regime.SRMHD, description="physics regime")

    # numerics
    solver: Solver = ProblemParam(
        Solver.HLLD, description="solver type for mhd"
    )
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # simulation control
    end_time: float = ProblemParam(
        1.0, cli=True, checkpoint_safe=True, description="simulation end time"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(
            self,
            "_problem_states",
            {
                "fast-shock": MHDProblemState.create_state(
                    (
                        1.000,
                        *MHDProblemState.beta([25.0, 0.0, 0.0]),
                        1.000,
                        20.0,
                        25.02,
                        0.0,
                    ),
                    (
                        25.48,
                        *MHDProblemState.beta([1.091, 0.3923, 0.0]),
                        367.5,
                        20.0,
                        49.00,
                        0.0,
                    ),
                ),
                "slow-shock": MHDProblemState.create_state(
                    (
                        1.000,
                        *MHDProblemState.beta([1.5300, 0.0, 0.0]),
                        10.00,
                        10.0,
                        18.28,
                        0.0,
                    ),
                    (
                        3.323,
                        *MHDProblemState.beta([0.9571, -0.6822, 0.0]),
                        55.36,
                        10.0,
                        14.49,
                        0.0,
                    ),
                ),
                "fast-rarefaction": MHDProblemState.create_state(
                    (
                        0.100,
                        *MHDProblemState.beta([-2.000, 0.0, 0.0]),
                        1.00,
                        2.0,
                        0.000,
                        0.0,
                    ),
                    (
                        0.562,
                        *MHDProblemState.beta([-0.212, -0.590, 0.0]),
                        10.0,
                        2.0,
                        4.710,
                        0.0,
                    ),
                ),
                "slow-rarefaction": MHDProblemState.create_state(
                    (
                        1.78e-3,
                        *MHDProblemState.beta([-0.765, -1.386, 0.0]),
                        0.1,
                        1.0,
                        1.022,
                        0.0,
                    ),
                    (
                        0.01000,
                        *MHDProblemState.beta([+0.0, 0.0, 0.0]),
                        1.0,
                        1.0,
                        0.000,
                        0.0,
                    ),
                ),
                "alfven": MHDProblemState.create_state(
                    (
                        1.0,
                        *MHDProblemState.beta([0.0, 0.0, 0.0]),
                        1.0,
                        3.0,
                        3.0000,
                        0.0,
                    ),
                    (
                        1.0,
                        *MHDProblemState.beta([3.70, 5.76, 0.0]),
                        1.0,
                        3.0,
                        -6.857,
                        0.0,
                    ),
                ),
                "compound": MHDProblemState.create_state(
                    (
                        1.0,
                        *MHDProblemState.beta([0.0, 0.0, 0.0]),
                        1.0,
                        3.0,
                        +3.000,
                        0.0,
                    ),
                    (
                        1.0,
                        *MHDProblemState.beta([3.70, 5.76, 0.0]),
                        1.0,
                        3.0,
                        -6.857,
                        0.0,
                    ),
                ),
                "st-1": MHDProblemState.create_state(
                    (1.0, 0.0, 0.0, 0.0, 1e3, 1.0, 0.0, 0.0),
                    (0.1, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
                ),
                "st-2": MHDProblemState.create_state(
                    (1.0, 0.0, 0.0, 0.0, 30.0, 0.0, 20.0, 0.0),
                    (0.1, 0.0, 0.0, 0.0, 1.00, 0.0, 0.00, 0.0),
                ),
                "collision": MHDProblemState.create_state(
                    (
                        1.0,
                        *MHDProblemState.beta([+5.0, 0.0, 0.0]),
                        1.0,
                        10.0,
                        +10.0,
                        0.0,
                    ),
                    (
                        1.0,
                        *MHDProblemState.beta([-5.0, 0.0, 0.0]),
                        1.0,
                        10.0,
                        -10.0,
                        0.0,
                    ),
                ),
            },
        )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for mhd shock tube."""

        def gas_state() -> GasStateGenerator:
            state = self._problem_states[self.problem]
            nx = self.resolution[0]
            dx = (self.bounds[0][1] - self.bounds[0][0]) / nx

            for ii in range(nx):
                xi = self.bounds[0][0] + ii * dx
                if xi < XMEMBRANE:
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

        def bfield_generator(field_name: str) -> StaggeredBFieldGenerator:
            state = self._problem_states[self.problem]
            ni, nj, nk = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / ni

            nx = ni + (field_name == "bx")
            ny = nj + (field_name == "by")
            nz = nk + (field_name == "bz")

            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        xi = self.bounds[0][0] + ii * dx
                        if xi < XMEMBRANE:
                            yield getattr(state.left, field_name)
                        else:
                            yield getattr(state.right, field_name)

        bx_gen = partial(bfield_generator, "bx")
        by_gen = partial(bfield_generator, "by")
        bz_gen = partial(bfield_generator, "bz")

        return cast(MHDStateGenerators, (gas_state, bx_gen, by_gen, bz_gen))
