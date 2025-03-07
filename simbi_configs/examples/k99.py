from dataclasses import dataclass
from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator, NamedTuple, Any
from functools import partial
from numpy.typing import NDArray
import numpy as np

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
    def beta(u: Sequence[float]) -> NDArray[np.floating[Any]]:
        """Calculate relativistic beta from 3-velocity"""
        unp: NDArray[np.floating[Any]] = np.asarray(u)
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


class MagneticShockTube(BaseConfig):
    """Komissarov (1999), 1D SRMHD test problems."""

    class config:
        nzones = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
        problem = DynamicArg(
            "problem",
            "contact",
            help="problem number from Komissarov (1999)",
            var_type=str,
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

    def __init__(self) -> None:
        super().__init__()
        self.problem_states = {
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

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state"""

        def _gas_state() -> Generator[tuple[float, ...], None, None]:
            state = self.problem_states[str(self.config.problem)]
            nx = self.resolution[0]
            dx = (self.bounds[1] - self.bounds[0]) / nx

            for i in range(nx):
                xi = self.bounds[0] + i * dx
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

        def _bfield(bn: str) -> Generator[float, None, None]:
            state = self.problem_states[str(self.config.problem)]
            ni, nj, nk = self.resolution
            dx = (self.bounds[1] - self.bounds[0]) / ni
            for k in range(nk + (bn == "bz")):
                for j in range(nj + (bn == "by")):
                    for i in range(ni + (bn == "bx")):
                        xi = self.bounds[0] + i * dx
                        if xi < XMEMBRANE:
                            yield getattr(state.left, bn)
                        else:
                            yield getattr(state.right, bn)

        _bx = partial(_bfield, bn="bx")
        _by = partial(_bfield, bn="by")
        _bz = partial(_bfield, bn="bz")
        return (
            _gas_state,
            _bx,
            _by,
            _bz,
        )

    @simbi_property
    def bounds(self) -> Sequence[float]:
        return (XMIN, XMAX)

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg | int]:
        return (self.config.nzones, 1, 1)

    @simbi_property
    def adiabatic_index(self) -> float:
        return 4.0 / 3.0

    @simbi_property
    def regime(self) -> str:
        return "srmhd"
