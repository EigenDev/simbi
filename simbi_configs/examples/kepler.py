import math
import simbi.expression as expr
from simbi import BaseConfig, DynamicArg, simbi_property, ImmersedBodyConfig
from typing import Any, Sequence, Generator
from simbi.typing import InitialStateType, ExpressionDict


class KeplerianRingTest(BaseConfig):
    """
    A thin ring of matter in Keplerian orbit.
    Tests angular momentum conservation and numerical viscosity.
    """

    class config:
        npts = DynamicArg(
            "npts", 256, help="Number of zones in x and y dimensions", var_type=int
        )
        buffer_width = DynamicArg(
            "buffer_width",
            0.2,
            help="Width of buffer zone (fraction of outer radius)",
            var_type=float,
        )
        buffer_damp_time = DynamicArg(
            "buffer_damp_time",
            0.1,
            help="Damping timescale (orbital periods at r=1)",
            var_type=float,
        )

    # Larger domain to see orbital motion
    xmin = -2.0
    xmax = 2.0
    ymin = -2.0
    ymax = 2.0

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            dx = (self.xmax - self.xmin) / self.config.npts
            dy = (self.ymax - self.ymin) / self.config.npts

            # Ring parameters
            r0 = 1.0  # ring central radius
            dr = 0.1  # ring width (gaussian sigma)
            M_0 = 1.0  # central mass
            G = 1.0  # gravitational constant

            # Background state
            sigma_min = 1e-8
            sigma_peak = 1.0

            # Sound speed parameter (small to minimize pressure support)
            h_r = 0.01  # H/R aspect ratio
            cs_0 = h_r * math.sqrt(G * M_0 / r0)

            buffer_params = self.buffer_parameters
            r_buffer = buffer_params["r_buffer"]
            r_outer = buffer_params["r_outer"]

            for j in range(self.config.npts):
                y = self.ymin + (j + 0.5) * dy
                for i in range(self.config.npts):
                    x = self.xmin + (i + 0.5) * dx
                    r = math.sqrt(x**2 + y**2)

                    # Base state
                    if r <= r_buffer:
                        # Normal ring profile inside buffer
                        sigma = sigma_min + sigma_peak * math.exp(
                            -((r - r0) ** 2) / (2 * dr**2)
                        )
                        v_k = math.sqrt(G * M_0 / r) if r > 0 else 0.0
                    else:
                        # Smooth transition in buffer zone
                        s = (r - r_buffer) / (r_outer - r_buffer)
                        damp = s * s

                        sigma_ring = sigma_min + sigma_peak * math.exp(
                            -((r - r0) ** 2) / (2 * dr**2)
                        )

                        # Transition to background state
                        sigma = sigma_min + (sigma_ring - sigma_min) * (1.0 - damp)
                        v_k = math.sqrt(G * M_0 / r) * (1.0 - damp)

                    vx = -v_k * (y / r) if r > 0 else 0.0
                    vy = v_k * (x / r) if r > 0 else 0.0

                    # Isothermal pressure (cs = constant)
                    p = sigma * cs_0**2

                    yield (sigma, vx, vy, p)

        return gas_state

    @simbi_property
    def buffer_parameters(self) -> dict[str, float]:
        """Buffer zone parameters"""
        r_outer = min(abs(self.xmax), abs(self.ymax))
        r_buffer = r_outer * (1.0 - self.config.buffer_width)

        # Orbital period at r=1
        G = 1.0
        M = 1.0
        T_orb = 2.0 * math.pi * math.sqrt(1.0**3 / (G * M))

        return {
            "r_buffer": r_buffer,
            "r_outer": r_outer,
            "damp_time": self.config.buffer_damp_time * T_orb,
        }

    @simbi_property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [
            ImmersedBodyConfig(
                body_type="gravitational",
                mass=1.0,
                velocity=[0.0, 0.0],
                position=[0.0, 0.0],
                radius=0.01,
                specifics={"softening_length": 0.05 * 0.01, "two_way_coupling": False},
            )
        ]

    @simbi_property
    def default_end_time(self) -> float:
        return 10.0 * 2.0 * math.pi

    @simbi_property
    def checkpoint_interval(self) -> float:
        return self.default_end_time / 100  # 100 snapshots

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[Any]:
        return (self.config.npts, self.config.npts)

    @simbi_property
    def adiabatic_index(self) -> float:
        return 1.0

    @simbi_property
    def ambient_sound_speed(self) -> float:
        # Must define this since we're isothermal
        h_r = 0.01  # H/R aspect ratio
        G = 1.0
        M = 1.0
        r0 = 1.0
        v_k = math.sqrt(G * M / r0)
        return h_r * v_k

    @simbi_property
    def regime(self) -> str:
        return "classical"

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def data_directory(self) -> str:
        return "data/kepler/"

    @simbi_property
    def boundary_conditions(self) -> str:
        return "dynamic"

    def apply_buffer_damping(
        self,
        x1: expr.Expr,
        x2: expr.Expr,
        t: expr.Expr,
        dt: expr.Expr,  # Add timestep as parameter
        inner: list[expr.Expr],
        r_buffer: expr.Expr,
        r_outer: expr.Expr,
        tau: expr.Expr,
    ) -> list[expr.Expr]:
        """Calculate buffer damping source terms for fluid variables."""
        # Radius calculation
        r = expr.sqrt(x1 * x1 + x2 * x2)

        if r <= r_buffer:
            return inner

        # calculate damping strength (0 inside buffer, increasing outward)
        s = (r - r_buffer) / (r_outer - r_buffer)
        # quadratic smoothing
        alpha = s * s

        # Calculate target state (background state)
        rho_bg = expr.constant(1e-8, r.graph)  # Background density
        vk = expr.sqrt(1.0 / r)  # Keplerian velocity
        vx_bg = -vk * (x2 / r)  # Background velocity x
        vy_bg = vk * (x1 / r)  # Background velocity y

        # calculate relaxation terms
        # form: -(current - target) * (dt/tau) * alpha
        damping_factor = -(dt / tau) * alpha

        rho_damping = damping_factor * (inner[0] - rho_bg)
        if rho_damping <= 0.0:
            rho_damping = inner[0]

        # momentum
        vx = inner[1] / inner[0]  # mx/rho
        vy = inner[2] / inner[0]  # my/rho

        # damp velocities
        vx_damping = damping_factor * (vx - vx_bg)
        vy_damping = damping_factor * (vy - vy_bg)

        mx_damping = inner[0] * vx_damping  # rho * velocity_damping
        my_damping = inner[0] * vy_damping  # rho * velocity_damping

        # energy damping
        # for isothermal gas, this gets ignored
        gamma = expr.constant(self.adiabatic_index, r.graph)
        e_int_bg = (
            rho_bg
            * expr.constant(0.01, r.graph) ** 2
            / (gamma - expr.constant(1.0, r.graph))
        )
        e_damping = damping_factor * (inner[3] - e_int_bg)

        return [rho_damping, mx_damping, my_damping, e_damping]

    @simbi_property
    def bx1_inner_expressions(self) -> ExpressionDict:
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
        r_buffer = expr.constant(self.buffer_parameters["r_buffer"], graph)
        r_outer = expr.constant(self.buffer_parameters["r_outer"], graph)
        tau = expr.constant(self.buffer_parameters["damp_time"], graph)

        source_terms = self.apply_buffer_damping(
            x1, x2, t, dt, current, r_buffer, r_outer, tau
        )

        compiled_exp = graph.compile(source_terms)
        return compiled_exp.serialize()

    @simbi_property
    def bx1_outer_expressions(self) -> ExpressionDict:
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
        r_buffer = expr.constant(self.buffer_parameters["r_buffer"], graph)
        r_outer = expr.constant(self.buffer_parameters["r_outer"], graph)
        tau = expr.constant(self.buffer_parameters["damp_time"], graph)

        source_terms = self.apply_buffer_damping(
            x1, x2, t, dt, current, r_buffer, r_outer, tau
        )

        compiled_exp = graph.compile(source_terms)
        return compiled_exp.serialize()

    @simbi_property
    def bx2_inner_expressions(self) -> ExpressionDict:
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
        r_buffer = expr.constant(self.buffer_parameters["r_buffer"], graph)
        r_outer = expr.constant(self.buffer_parameters["r_outer"], graph)
        tau = expr.constant(self.buffer_parameters["damp_time"], graph)

        source_terms = self.apply_buffer_damping(
            x1, x2, t, dt, current, r_buffer, r_outer, tau
        )

        compiled_exp = graph.compile(source_terms)
        return compiled_exp.serialize()

    @simbi_property
    def bx2_outer_expressions(self) -> ExpressionDict:
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
        r_buffer = expr.constant(self.buffer_parameters["r_buffer"], graph)
        r_outer = expr.constant(self.buffer_parameters["r_outer"], graph)
        tau = expr.constant(self.buffer_parameters["damp_time"], graph)

        source_terms = self.apply_buffer_damping(
            x1, x2, t, dt, current, r_buffer, r_outer, tau
        )

        compiled_exp = graph.compile(source_terms)
        return compiled_exp.serialize()
