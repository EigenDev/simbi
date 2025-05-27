from dataclasses import dataclass
from typing import Sequence, NamedTuple, Any, cast
from functools import partial
import numpy as np
from numpy.typing import NDArray

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, CellSpacing, Solver
from simbi.core.types.typing import (
    InitialStateType,
    GasStateGenerator,
    StaggeredBFieldGenerator,
    MHDStateGenerators,
)

# Constants
XMIN = -2.0
XMAX = +2.0
XMEMBRANE = 0.5 * (XMIN + XMAX)


@dataclass(frozen=True)
class ShockTubeState:
    """Left and right states for shock tube problems"""

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
    """Complete state for MHD shock tube problems"""

    left: ShockTubeState
    right: ShockTubeState

    @staticmethod
    def beta(u: Sequence[float]) -> NDArray[np.float64]:
        """Calculate relativistic beta from 3-velocity"""
        unp: NDArray[np.float64] = np.asarray(u)
        gamma: float = (1.0 + unp.dot(unp)) ** (0.5)
        return unp / gamma

    @classmethod
    def create_state(
        cls, left_vals: Sequence[float], right_vals: Sequence[float]
    ) -> "MHDProblemState":
        """Create problem state from raw values"""
        return cls(left=ShockTubeState(*left_vals), right=ShockTubeState(*right_vals))


class StaggeredMHDState(NamedTuple):
    """Container that stores face-centered magnetic fields
    and cell-centered gas variables"""

    gas_vars: tuple[float, ...]
    staggered_bfields: list[list[float]]


class MagneticShockTube(SimbiBaseConfig):
    """Komissarov (1999), 1D SRMHD test problems."""

    # Configuration parameters with choices
    problem: str = SimbiField(
        "fast-shock",
        description="Problem type from Komissarov (1999)",
        choices=[
            "fast-shock",
            "slow-shock",
            "fast-rarefaction",
            "slow-rarefaction",
            "alfven",
            "compound",
            "st-1",
            "st-2",
            "collision",
        ],
    )

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int, int] = SimbiField(
        (100, 1, 1), description="Grid resolution"
    )

    bounds: list[tuple[float, float]] = SimbiField(
        [(XMIN, XMAX)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRMHD, description="Physics regime")

    adiabatic_index: float = SimbiField(4.0 / 3.0, description="Adiabatic index")
    solver: Solver = SimbiField(
        Solver.HLLD, description="Solver type for MHD equations"
    )

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Store problem states as a private attribute
        self._problem_states = {
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
                (1.0, *MHDProblemState.beta([0.0, 0.0, 0.0]), 1.0, 3.0, 3.0000, 0.0),
                (1.0, *MHDProblemState.beta([3.70, 5.76, 0.0]), 1.0, 3.0, -6.857, 0.0),
            ),
            "compound": MHDProblemState.create_state(
                (1.0, *MHDProblemState.beta([0.0, 0.0, 0.0]), 1.0, 3.0, +3.000, 0.0),
                (1.0, *MHDProblemState.beta([3.70, 5.76, 0.0]), 1.0, 3.0, -6.857, 0.0),
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
                (1.0, *MHDProblemState.beta([+5.0, 0.0, 0.0]), 1.0, 10.0, +10.0, 0.0),
                (1.0, *MHDProblemState.beta([-5.0, 0.0, 0.0]), 1.0, 10.0, -10.0, 0.0),
            ),
        }

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for MHD shock tube.

        Returns:
            Tuple of generator functions (gas_state, bx, by, bz)
        """

        # For MHD, we need to return a tuple of 4 generators
        def gas_state() -> GasStateGenerator:
            """Generate gas state variables"""
            state = self._problem_states[self.problem]
            nx = self.resolution[0]
            dx = (self.bounds[0][1] - self.bounds[0][0]) / nx

            for i in range(nx):
                xi = self.bounds[0][0] + i * dx
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
            """Generate B-field component values"""
            state = self._problem_states[self.problem]
            ni, nj, nk = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / ni

            # Adjust grid size based on which field component
            nx = ni + (field_name == "bx")
            ny = nj + (field_name == "by")
            nz = nk + (field_name == "bz")

            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        xi = self.bounds[0][0] + i * dx
                        if xi < XMEMBRANE:
                            yield getattr(state.left, field_name)
                        else:
                            yield getattr(state.right, field_name)

        # Create partial functions for each B-field component
        bx_gen = partial(bfield_generator, "bx")
        by_gen = partial(bfield_generator, "by")
        bz_gen = partial(bfield_generator, "bz")

        # Return tuple of all generator functions
        return cast(MHDStateGenerators, (gas_state, bx_gen, by_gen, bz_gen))
