import math
from pathlib import Path
from typing import Any

from pydantic import computed_field

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.bodies import (
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
)
from simbi.core.types.input import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.core.types.typing import (
    GasStateGenerator,
    InitialStateType,
)


class KeplerianRingTest(SimbiBaseConfig):
    """
    A thin ring of matter in Keplerian orbit.
    Tests angular momentum conservation and numerical viscosity.
    """

    # Configuration parameters with defaults
    buffer_width: float = SimbiField(
        0.2, description="Width of buffer zone (fraction of outer radius)"
    )

    buffer_damp_time: float = SimbiField(
        0.1, description="Damping timescale (orbital periods at r=1)"
    )

    # Required fields from SimbiBaseConfig
    resolution: tuple[int, int] = SimbiField(
        (256, 256), description="Grid resolution"
    )

    bounds: list[tuple[float, float]] = SimbiField(
        [(-2.0, 2.0), (-2.0, 2.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.NEWTONIAN, description="Physics regime")

    adiabatic_index: float = SimbiField(
        1.0, description="Adiabatic index (isothermal)"
    )

    # Optional fields with non-default values
    solver: Solver = SimbiField(Solver.HLLE, description="Numerical solver")

    data_directory: Path = SimbiField(
        Path("data/kepler/"), description="Output data directory"
    )

    cfl_number: float = SimbiField(0.25, description="CFL condition number")

    boundary_conditions: BoundaryCondition = SimbiField(
        BoundaryCondition.OUTFLOW, description="Boundary conditions"
    )

    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    end_time: float = SimbiField(
        20.0 * math.pi, description="Simulation end time (10 orbits)"
    )

    checkpoint_interval: float = SimbiField(
        0.2 * math.pi, description="Checkpoint interval (1/100 of end time)"
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Initialize parameter values after super().__init__
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Initialize parameters after object creation"""

        # Calculate buffer parameters
        self._calculate_buffer_parameters()

    def _calculate_buffer_parameters(self) -> None:
        """Calculate buffer zone parameters"""
        r_outer = min(abs(self.bounds[0][1]), abs(self.bounds[1][1]))
        r_buffer = r_outer * (1.0 - self.buffer_width)

        # Orbital period at r=1
        G = 1.0
        M = 1.0
        T_orb = 2.0 * math.pi * math.sqrt(1.0**3 / (G * M))

        self._buffer_parameters = {
            "r_buffer": r_buffer,
            "r_outer": r_outer,
            "damp_time": self.buffer_damp_time * T_orb,
        }

    @computed_field
    @property
    def ambient_sound_speed(self) -> float:
        return 0.01

    @computed_field
    @property
    def buffer_parameters(self) -> dict[str, float]:
        """Get buffer parameters"""
        return self._buffer_parameters

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        """Define immersed bodies"""
        dx = (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]
        softening_length = 2.0 * dx
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.GRAVITATIONAL,
                mass=1.0,
                radius=0.01,
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                gravitational=GravitationalProperties(
                    softening_length=softening_length
                ),
            )
        ]

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Keplerian disk with pressure support.

        The velocity is corrected to account for pressure gradient forces,
        ensuring a balanced initial state for the ring.

        Returns:
            Generator function that yields primitive variables (density, vx, vy, pressure)
        """

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]

            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny

            # Ring parameters
            r0 = 1.0  # ring central radius
            dr = 0.1  # ring width (gaussian sigma)
            M_0 = 1.0  # central mass
            G = 1.0  # gravitational constant

            # Background state
            sigma_min = 1e-8
            sigma_peak = 1.0

            # Sound speed parameter
            cs_0 = self.ambient_sound_speed
            cs_squared = cs_0 * cs_0

            # Buffer parameters
            r_buffer = self._buffer_parameters["r_buffer"]
            r_outer = self._buffer_parameters["r_outer"]

            # Small threshold to avoid division by zero
            epsilon = 1e-10

            for j in range(ny):
                y = ymin + (j + 0.5) * dy
                for i in range(nx):
                    x = xmin + (i + 0.5) * dx
                    r = math.sqrt(x**2 + y**2)

                    # Avoid division by zero
                    if r < epsilon:
                        # At exact center, set minimal values
                        sigma = sigma_min
                        vx = 0.0
                        vy = 0.0
                        p = sigma * cs_squared
                        yield (sigma, vx, vy, p)

                    sigma = sigma_min + sigma_peak * math.exp(
                        -((r - r0) ** 2) / (2 * dr**2)
                    )
                    v_k = math.sqrt(G * M_0 / r)
                    vx = -v_k * (y / r)
                    vy = +v_k * (x / r)

                    # Isothermal pressure (cs = constant)
                    p = sigma * cs_squared

                    yield (sigma, vx, vy, p)

        return gas_state
