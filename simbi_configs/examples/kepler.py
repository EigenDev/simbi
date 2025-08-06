import math

import simbi.expression as expr
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import (
    CoordSystem,
    Regime,
    CellSpacing,
    Solver,
    BoundaryCondition,
)
from simbi.core.types.typing import GasStateGenerator, InitialStateType, ExpressionDict
from simbi.core.types.bodies import ImmersedBodyConfig, BodyCapability
from pydantic import computed_field
from pathlib import Path
from typing import Any


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
    resolution: tuple[int, int] = SimbiField((256, 256), description="Grid resolution")

    bounds: list[tuple[float, float]] = SimbiField(
        [(-2.0, 2.0), (-2.0, 2.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")

    adiabatic_index: float = SimbiField(1.0, description="Adiabatic index (isothermal)")

    # Optional fields with non-default values
    solver: Solver = SimbiField(Solver.HLLE, description="Numerical solver")

    data_directory: Path = SimbiField(
        Path("data/kepler/"), description="Output data directory"
    )

    cfl_number: float = SimbiField(0.05, description="CFL condition number")

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
        return 0.01  # Isothermal sound speed (constant)

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
                specifics={
                    "softening_length": softening_length,
                },
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
                        continue

                    # Calculate density profile
                    if r <= r_buffer:
                        # Normal ring profile inside buffer
                        sigma = sigma_min + sigma_peak * math.exp(
                            -((r - r0) ** 2) / (2 * dr**2)
                        )

                        # Calculate base Keplerian velocity
                        v_k_base = math.sqrt(G * M_0 / r)

                        # Calculate pressure gradient correction
                        # For a Gaussian ring with isothermal equation of state:
                        # The correction term is approximately: cs^2*(r-r0)/dr^2
                        # if (
                        #     sigma > sigma_min * 1.1
                        # ):  # Only apply correction where density is significant
                        #     # Pressure gradient term (negative inside r0, positive outside)
                        #     pressure_correction = cs_squared * (r - r0) / (dr * dr)

                        #     # Total velocity squared with pressure correction
                        #     v_k_squared = v_k_base * v_k_base - r * pressure_correction

                        #     v_k = math.sqrt(v_k_squared)
                        # else:
                        # Use standard Keplerian for very low density regions
                        v_k = v_k_base
                    else:
                        # Smooth transition in buffer zone
                        s = (r - r_buffer) / (r_outer - r_buffer)
                        damp = s * s

                        # Calculate ring density with buffer transition
                        sigma_ring = sigma_min + sigma_peak * math.exp(
                            -((r - r0) ** 2) / (2 * dr**2)
                        )
                        sigma = sigma_min + (sigma_ring - sigma_min) * (1.0 - damp)

                        # Keplerian velocity with buffer dampening
                        v_k = math.sqrt(G * M_0 / r) * (1.0 - damp)

                    vx = -v_k * (y / r)
                    vy = +v_k * (x / r)

                    # Isothermal pressure (cs = constant)
                    p = sigma * cs_squared

                    yield (sigma, vx, vy, p)

        return gas_state

    def apply_buffer_damping(
        self,
        x1: "expr.Expr",
        x2: "expr.Expr",
        t: "expr.Expr",
        dt: "expr.Expr",
        current: list["expr.Expr"],
        r_buffer: "expr.Expr",
        r_outer: "expr.Expr",
        tau: "expr.Expr",
    ) -> list["expr.Expr"]:
        """Calculate buffer damping source terms for fluid variables."""
        # Radius calculation
        r = expr.sqrt(x1 * x1 + x2 * x2) + 1e-10  # Avoid division by zero

        # Zero source terms for regions inside buffer
        zero = expr.constant(0.0, r.graph)
        condition = r <= r_buffer

        # Calculate damping strength (0 inside buffer, increasing outward)
        s = (r - r_buffer) / (r_outer - r_buffer)
        alpha = s * s  # quadratic smoothing

        # Calculate target state (background state)
        rho_bg = expr.constant(1e-8, r.graph)  # Background density
        vk = expr.sqrt(1.0 / r)  # Keplerian velocity
        vx_bg = -vk * (x2 / r)  # Background velocity x
        vy_bg = vk * (x1 / r)  # Background velocity y

        # Calculate damping rate
        damping_rate = -alpha / tau

        # Density source term (rate of change)
        rho_source = damping_rate * (current[0] - rho_bg)

        # Velocity components for momentum sources
        vx = current[1] / current[0]  # mx/rho
        vy = current[2] / current[0]  # my/rho

        # Apply damping to velocities
        vx_source = damping_rate * (vx - vx_bg)
        vy_source = damping_rate * (vy - vy_bg)

        # Convert to momentum source terms
        mx_source = current[0] * vx_source
        my_source = current[0] * vy_source

        # Energy source term
        # For isothermal gas, the internal energy target is e_int = rho*cs^2/(gamma-1)
        gamma = expr.constant(self.adiabatic_index, r.graph)
        cs_squared = expr.constant(self.ambient_sound_speed**2, r.graph)

        # For gamma=1 (isothermal), we need to handle specially to avoid division by zero
        # Use a small epsilon in the denominator for safety
        e_int_bg = (
            rho_bg
            * cs_squared
            / expr.max_expr(
                gamma - expr.constant(1.0, r.graph), expr.constant(1e-10, r.graph)
            )
        )

        e_source = damping_rate * (current[3] - e_int_bg)

        rho_source_final = expr.if_then_else(condition, zero, rho_source)
        mx_source_final = expr.if_then_else(condition, zero, mx_source)
        my_source_final = expr.if_then_else(condition, zero, my_source)
        e_source_final = expr.if_then_else(condition, zero, e_source)

        return [rho_source_final, mx_source_final, my_source_final, e_source_final]

    def _create_boundary_expression(self) -> ExpressionDict:
        """Create boundary expressions for buffer damping"""
        import simbi.expression as expr

        graph = expr.ExprGraph()
        x1 = expr.variable("x1", graph)
        x2 = expr.variable("x2", graph)
        t = expr.variable("t", graph)
        dt = expr.variable("dt", graph)

        # current state values
        current = [
            expr.parameter(0, graph),  # density
            expr.parameter(1, graph),  # momentum x
            expr.parameter(2, graph),  # momentum y
            expr.parameter(3, graph),  # energy
        ]

        # buffer parameters as constants
        r_buffer = expr.constant(self._buffer_parameters["r_buffer"], graph)
        r_outer = expr.constant(self._buffer_parameters["r_outer"], graph)
        tau = expr.constant(self._buffer_parameters["damp_time"], graph)

        source_terms = self.apply_buffer_damping(
            x1, x2, t, dt, current, r_buffer, r_outer, tau
        )

        compiled_exp = graph.compile(source_terms)
        return compiled_exp.serialize()

    @computed_field
    @property
    def hydro_source_expressions(self) -> ExpressionDict:
        """Buffer damping expressions"""
        return self._create_boundary_expression()
