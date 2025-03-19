import math
from simbi import BaseConfig, DynamicArg, simbi_property, simbi_class_property
from typing import Any, Sequence, Generator
from simbi.typing import InitialStateType


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
    def immersed_bodies(self) -> list[dict[str, Any]]:
        return [
            {
                "body_type": "gravitational",
                "mass": 1.0,
                "velocity": [0.0, 0.0],
                "position": [0.0, 0.0],
                "radius": 0.01,
                "softening": 0.05,
            }
        ]

    @simbi_property
    def default_end_time(self) -> float:
        return 10.0 * 2.0 * math.pi

    @simbi_property
    def checkpoint_interval(self) -> float:
        return self.default_end_time / 100  # 50 snapshots

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
    def sound_speed(self) -> float:
        # Must define this since we're isothermal
        h_r = 0.01  # H/R aspect ratio
        G = 1.0
        M = 1.0
        r0 = 1.0
        v_k = math.sqrt(G * M / r0)
        return h_r * v_k  # Return cs, not cs^2

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

    @simbi_property
    def boundary_sources(self) -> str:
        buffer_params = self.buffer_parameters
        return rf"""
    #include <cmath>
    #include <iostream>
    extern "C" {{
        // Helper function for buffer damping
        inline void apply_buffer_damping(
            double x1, double x2, double t,
            double* result,
            const double r_buffer,
            const double r_outer,
            const double tau
        ) {{
            const double r = std::sqrt(x1*x1 + x2*x2);

            // First set initial values (since result array is zero-initialized)
            const double rho = 1e-8;  // Background density
            const double cs = {self.sound_speed};
            const double press = rho * cs*cs;

            // Default to background values
            result[0] = rho;
            result[1] = 0.0;  // vx
            result[2] = 0.0;  // vy
            result[3] = press;

            if (r <= r_buffer) return;

            const double s = (r - r_buffer)/(r_outer - r_buffer);
            const double alpha = s*s;

            // Keplerian velocity at this radius
            const double G = 1.0;
            const double M = 1.0;
            const double v_k = std::sqrt(G*M/r);
            const double vx = -v_k * (x2/r);
            const double vy = v_k * (x1/r);

            // Apply damping towards background state
            result[0] += -alpha*(result[0] - rho)/tau;
            result[1] += -alpha*(result[1] - rho*vx)/tau;
            result[2] += -alpha*(result[2] - rho*vy)/tau;
            result[3] += -alpha*(result[3] - press)/tau;
        }}

        // x1 boundaries (left/right)
        void bx1_inner_source(double x1, double x2, double t, double result[]) {{
            apply_buffer_damping(x1, x2, t, result,
                            {buffer_params["r_buffer"]},
                            {buffer_params["r_outer"]},
                            {buffer_params["damp_time"]});
        }}

        void bx1_outer_source(double x1, double x2, double t, double result[]) {{
            apply_buffer_damping(x1, x2, t, result,
                            {buffer_params["r_buffer"]},
                            {buffer_params["r_outer"]},
                            {buffer_params["damp_time"]});
        }}

        // x2 boundaries (bottom/top)
        void bx2_inner_source(double x1, double x2, double t, double result[]) {{
            apply_buffer_damping(x1, x2, t, result,
                            {buffer_params["r_buffer"]},
                            {buffer_params["r_outer"]},
                            {buffer_params["damp_time"]});
        }}

        void bx2_outer_source(double x1, double x2, double t, double result[]) {{
            apply_buffer_damping(x1, x2, t, result,
                            {buffer_params["r_buffer"]},
                            {buffer_params["r_outer"]},
                            {buffer_params["damp_time"]});
        }}
    }}
    """
