from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator
from dataclasses import dataclass
from functools import partial


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

    @classmethod
    def create_state(
        cls, left_vals: Sequence[float], right_vals: Sequence[float]
    ) -> "MHDProblemState":
        """Create problem state from raw values"""
        return cls(left=ShockTubeState(*left_vals), right=ShockTubeState(*right_vals))


class MagneticShockTube(BaseConfig):
    """
    Mignone & Bodo (2006), Relativistic MHD Test Problems in 1D Mesh
    """

    class config:
        nzones = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
        problem = DynamicArg(
            "problem",
            1,
            help="problem number from Mignone & Bodo (2006)",
            var_type=int,
            choices=[1, 2, 3, 4],
        )

    def __init__(self) -> None:
        super().__init__()
        self.problem_states = {
            1: MHDProblemState.create_state(
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.5, +1.0, 0.0),
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
        }

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        # defined as (rho, v1, v2, v3, pg, b1, b2, b3)
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            state = self.problem_states[int(self.config.problem)]
            ni, nj, nk = self.resolution
            dx = (self.bounds[1] - self.bounds[0]) / ni
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        xi = self.bounds[0] + i * dx
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

        def b_field(bn: str) -> Generator[float, None, None]:
            state = self.problem_states[int(self.config.problem)]
            ni, nj, nk = self.resolution
            dx = (self.bounds[1] - self.bounds[0]) / ni
            for k in range(nk + (bn == "bz")):
                for j in range(nj + (bn == "by")):
                    for i in range(ni + (bn == "bx")):
                        xi = self.bounds[0] + i * dx
                        if xi < 0.5:
                            yield getattr(state.left, bn)
                        else:
                            yield getattr(state.right, bn)

        bx = partial(b_field, "bx")
        by = partial(b_field, "by")
        bz = partial(b_field, "bz")
        return gas_state, bx, by, bz

    @simbi_property
    def bounds(self) -> Sequence[float]:
        return (0.0, 1.0)

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
        if self.config.problem == 1:
            return 2.0
        else:
            return 5.0 / 3.0

    @simbi_property
    def regime(self) -> str:
        return "srmhd"
