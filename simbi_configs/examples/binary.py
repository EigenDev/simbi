import math
from simbi import BaseConfig, DynamicArg, simbi_property, BinaryConfig, GravitationalSystemConfig
from typing import Any, Sequence, Generator
from simbi.typing import InitialStateType


class BinaryOrbitTest(BaseConfig):
    """Simple test of binary orbital dynamics without gas."""

    class config:
        npts = DynamicArg(
            "npts", 256, help="Grid resolution (minimal for test)", var_type=int
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

    # Small domain since we just need space for the binary
    xmin = -2.0
    xmax = +2.0
    ymin = -2.0
    ymax = +2.0

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            # Minimal background gas
            for j in range(self.config.npts):
                for i in range(self.config.npts):
                    yield (1, 0.0, 0.0, 1e-8)

        return gas_state

    @simbi_property
    def gravitational_system(self) -> GravitationalSystemConfig:
        """Define gravitational system configuration."""
        return GravitationalSystemConfig(
            prescribed_motion=True, # move on analytic orbits (i.e., no live binary)
            reference_frame="center_of_mass",
            system_type="binary",
            binary_config=BinaryConfig(
                semi_major=0.2,
                eccentricity=0.0,
                mass_ratio=1.0,
                total_mass=1.0,
            ),
        )

    @simbi_property
    def default_end_time(self) -> float:
        # Run for 10 orbits
        M = self.gravitational_system["binary_config"]["total_mass"]
        a = self.gravitational_system["binary_config"]["semi_major"]
        T_orbit = 2.0 * math.pi * math.sqrt(a**3 / M)
        return 10.0 * T_orbit

    @simbi_property
    def checkpoint_interval(self) -> float:
        # Save 100 snapshots per orbit
        return self.default_end_time / 100

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))

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
        G = 1.0
        M_tot = 1.0
        a = 0.2
        v_k = math.sqrt(G * M_tot / a)

        return 5.0 * v_k

    @simbi_property
    def regime(self) -> str:
        return "classical"

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def data_directory(self) -> str:
        return "data/binary_test/"

    @simbi_property
    def boundary_conditions(self) -> str:
        return "outflow"

    @simbi_property
    def buffer_parameters(self) -> dict[str, float]:
        """Buffer zone parameters"""
        r_outer = min(abs(self.xmax), abs(self.ymax))
        r_buffer = r_outer * (1.0 - self.config.buffer_width)

        # Orbital period at r=1
        a_ref = 1
        M = self.gravitational_system["binary_config"]["total_mass"]
        T_orb = 2.0 * math.pi * math.sqrt(a_ref**3 / M)

        return {
            "r_buffer": r_buffer,
            "r_outer": r_outer,
            "damp_time": self.config.buffer_damp_time * T_orb,
        }

    # @simbi_property
    # def boundary_sources(self) -> str:
    #     buffer_params = self.buffer_parameters
    #     return rf"""
    # #include <cmath>
    # #include <iostream>
    # extern "C" {{
    #     // Helper function for buffer damping
    #     inline void apply_buffer_damping(
    #         double x1, double x2, double t,
    #         double* result,
    #         const double r_buffer,
    #         const double r_outer,
    #         const double tau
    #     ) {{
    #         const double r = std::sqrt(x1*x1 + x2*x2);

    #         // First set initial values (since result array is zero-initialized)
    #         const double rho = 1e-8;  // Background density
    #         const double cs = {self.sound_speed};
    #         const double press = rho * cs*cs;

    #         // Default to background values
    #         result[0] = rho;
    #         result[1] = 0.0;  // vx
    #         result[2] = 0.0;  // vy
    #         result[3] = press;

    #         if (r <= r_buffer) return;

    #         const double s = (r - r_buffer)/(r_outer - r_buffer);
    #         const double alpha = s*s;

    #         // Keplerian velocity at this radius
    #         const double G = 1.0;
    #         const double M = 1.0;
    #         const double v_k = std::sqrt(G*M/r);
    #         const double vx = -v_k * (x2/r);
    #         const double vy = v_k * (x1/r);

    #         // Apply damping towards background state
    #         result[0] += -alpha*(result[0] - rho)/tau;
    #         result[1] += -alpha*(result[1] - rho*vx)/tau;
    #         result[2] += -alpha*(result[2] - rho*vy)/tau;
    #         result[3] += -alpha*(result[3] - press)/tau;
    #     }}

    #     // x1 boundaries (left/right)
    #     void bx1_inner_source(double x1, double x2, double t, double result[]) {{
    #         apply_buffer_damping(x1, x2, t, result,
    #                         {buffer_params["r_buffer"]},
    #                         {buffer_params["r_outer"]},
    #                         {buffer_params["damp_time"]});
    #     }}

    #     void bx1_outer_source(double x1, double x2, double t, double result[]) {{
    #         apply_buffer_damping(x1, x2, t, result,
    #                         {buffer_params["r_buffer"]},
    #                         {buffer_params["r_outer"]},
    #                         {buffer_params["damp_time"]});
    #     }}

    #     // x2 boundaries (bottom/top)
    #     void bx2_inner_source(double x1, double x2, double t, double result[]) {{
    #         apply_buffer_damping(x1, x2, t, result,
    #                         {buffer_params["r_buffer"]},
    #                         {buffer_params["r_outer"]},
    #                         {buffer_params["damp_time"]});
    #     }}

    #     void bx2_outer_source(double x1, double x2, double t, double result[]) {{
    #         apply_buffer_damping(x1, x2, t, result,
    #                         {buffer_params["r_buffer"]},
    #                         {buffer_params["r_outer"]},
    #                         {buffer_params["damp_time"]});
    #     }}
    # }}
    # """
