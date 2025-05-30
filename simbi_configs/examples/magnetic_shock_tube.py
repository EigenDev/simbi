from typing import Sequence, cast, Any
from dataclasses import dataclass
from functools import partial

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import (
    CoordSystem,
    Regime,
    CellSpacing,
    Solver,
)
from simbi.core.types.typing import (
    InitialStateType,
    GasStateGenerator,
    StaggeredBFieldGenerator,
    MHDStateGenerators,
)
from pydantic import model_validator


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


class MagneticShockTube(SimbiBaseConfig):
    """
    Mignone & Bodo (2006), Relativistic MHD Test Problems in 1D Mesh
    """

    problem: int = SimbiField(
        1, description="Problem number from Mignone & Bodo (2006)", choices=[1, 2, 3, 4]
    )

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int, int] = SimbiField(
        (100, 1, 1), description="Grid resolution"
    )

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRMHD, description="Physics regime")
    solver: Solver = SimbiField(
        Solver.HLLD, description="Solver type for MHD equations"
    )

    # Adiabatic index depends on problem, so we provide a default here
    # and use a validator to update it
    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    @model_validator(mode="after")
    def set_adiabatic_index_by_problem(self) -> "MagneticShockTube":
        """Set adiabatic index based on problem number."""
        # Set adiabatic index based on problem
        if self.problem == 1:
            self.adiabatic_index = 2.0
        else:
            self.adiabatic_index = 5.0 / 3.0

        return self

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Store problem states as a private attribute
        self._problem_states = {
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

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for MHD shock tube.

        Returns:
            Tuple of generator functions (gas_state, bx, by, bz)
        """

        # For MHD, we need to return a tuple of 4 generators
        def gas_state() -> GasStateGenerator:
            """Generate gas state variables"""
            state = self._problem_states[self.problem]
            ni, nj, nk = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / ni

            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        xi = self.bounds[0][0] + i * dx
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
            """Generate B-field component values"""
            state = self._problem_states[self.problem]
            ni, nj, nk = self.resolution
            dx = (self.bounds[0][1] - self.bounds[0][0]) / ni

            # Adjust dimensions based on which field component
            for k in range(nk + (bn == "bz")):
                for j in range(nj + (bn == "by")):
                    for i in range(ni + (bn == "bx")):
                        xi = self.bounds[0][0] + i * dx
                        if xi < 0.5:
                            yield getattr(state.left, bn)
                        else:
                            yield getattr(state.right, bn)

        # Create generator functions for B-field components
        bx_gen = partial(b_field, "bx")
        by_gen = partial(b_field, "by")
        bz_gen = partial(b_field, "bz")

        # Return tuple of all generator functions
        return cast(MHDStateGenerators, (gas_state, bx_gen, by_gen, bz_gen))
