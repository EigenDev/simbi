from functools import partial
from dataclasses import dataclass
from typing import Iterator, Any, cast

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, CellSpacing, Solver
from simbi.core.types.typing import (
    InitialStateType,
    GasStateGenerator,
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


class MagneticShockTube(SimbiBaseConfig):
    """
    Mignone, Ugliano, & Bodo (2009), 1D SRMHD test problems.
    """

    # Configuration parameters
    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    problem: str = SimbiField(
        "contact",
        description="Problem type from Mignone, Ugliano, & Bodo (2009)",
        choices=[
            "contact",
            "rotational",
            "st-1",
            "st-2",
            "st-3",
            "st-4",
        ],
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
    solver: Solver = SimbiField(Solver.HLLD, description="Numerical solver")

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Initialize problem states dictionary
        self._problem_states = {
            "contact": (
                MHDState(10.0, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
                MHDState(1.00, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
            ),
            "rotational": (
                MHDState(1.0, 0.4, -0.3, 0.5, 1.0, 2.4, 1.0, -1.6),
                MHDState(1.0, 0.377347, -0.482389, 0.424190, 1.0, 2.4, -0.1, -2.178213),
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
        """Generate initial primitive state for SRMHD shock tube.

        Returns:
            Tuple of generator functions for gas state and B-fields
        """

        def gas_state() -> GasStateGenerator:
            """Generate gas state variables"""
            ni, nj, nk = self.resolution
            state = self._problem_states[self.problem]
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / ni

            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        xi = xmin + (i + 0.5) * dx  # Cell center
                        if xi < 0.5 * xextent:
                            yield tuple(state[0])  # Left state
                        else:
                            yield tuple(state[1])  # Right state

        def bfield(bn: str) -> GasStateGenerator:
            """Generate B-field component values"""
            state = self._problem_states[self.problem]
            ni, nj, nk = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / ni

            # Adjust dimensions based on which field component
            for k in range(nk + (bn == "b3")):
                for j in range(nj + (bn == "b2")):
                    for i in range(ni + (bn == "b1")):
                        xi = xmin + i * dx
                        if xi < 0.5 * xextent:
                            yield getattr(state[0], bn)  # Left state
                        else:
                            yield getattr(state[1], bn)  # Right state

        # Create partial functions for each B-field component
        bx_gen = partial(bfield, "b1")
        by_gen = partial(bfield, "b2")
        bz_gen = partial(bfield, "b3")

        # Return tuple of generator functions
        return cast(MHDStateGenerators, (gas_state, bx_gen, by_gen, bz_gen))
