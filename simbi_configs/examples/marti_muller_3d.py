from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime
from simbi.core.types.typing import GasStateGenerator, InitialStateType


class MartiMuller3D(SimbiBaseConfig):
    """
    Marti & Muller (2003), Relativistic Shock Tube Problem on 3D Mesh
    """

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int, int] = SimbiField(
        (100, 100, 100), description="Grid resolution"
    )

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRHD, description="Physics regime")

    adiabatic_index: float = SimbiField(4.0 / 3.0, description="Adiabatic index")

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for 3D Marti & Muller shock tube.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        xi = xmin + (i + 0.5) * dx  # Cell center
                        if xi <= 0.5 * xextent:
                            yield (
                                10.0,
                                0.0,
                                0.0,
                                0.0,
                                13.33,
                            )  # Left state: (rho, vx, vy, vz, p)
                        else:
                            yield (
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                                1e-10,
                            )  # Right state: (rho, vx, vy, vz, p)

        return gas_state
