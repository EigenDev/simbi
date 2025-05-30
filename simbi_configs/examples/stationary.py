from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, CellSpacing, Solver
from simbi.core.types.typing import GasStateGenerator, InitialStateType
from pathlib import Path


class StationaryWaveHLL(SimbiBaseConfig):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLL solver
    """

    # Required fields from SimbiBaseConfig
    resolution: int = SimbiField(400, description="Grid resolution")

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")

    adiabatic_index: float = SimbiField(5.0 / 3.0, description="Adiabatic index")

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    solver: Solver = SimbiField(Solver.HLLE, description="Numerical solver")

    data_directory: Path = SimbiField(
        Path("data/stationary/hlle"), description="Output data directory"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for stationary wave.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            for i in range(nx):
                x = xmin + (i + 0.5) * dx  # Cell center

                # Different density on left half, same velocity and pressure throughout
                if x < 0.5 * xextent:
                    yield (1.4, 0.0, 1.0)  # Left state: (rho, v, p)
                else:
                    yield (1.0, 0.0, 1.0)  # Right state: (rho, v, p)

        return gas_state


class StationaryWaveHLLC(StationaryWaveHLL):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLLC solver
    """

    # Override only the solver and data directory
    solver: Solver = SimbiField(
        Solver.HLLC, description="HLLC numerical solver (Toro et al. 1992)"
    )

    data_directory: Path = SimbiField(
        Path("data/stationary/hllc"),
        description="Output data directory for HLLC solver",
    )
