import math
from simbi import BaseConfig, DynamicArg, simbi_property
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

            for j in range(self.config.npts):
                y = self.ymin + (j + 0.5) * dy
                for i in range(self.config.npts):
                    x = self.xmin + (i + 0.5) * dx
                    r = math.sqrt(x**2 + y**2)

                    # Gaussian ring profile
                    sigma = sigma_min + sigma_peak * math.exp(
                        -((r - r0) ** 2) / (2 * dr**2)
                    )

                    # Exact Keplerian velocity (no pressure correction needed due to low h/r)
                    v_k = math.sqrt(G * M_0 / r) if r > 0 else 0.0
                    vx = -v_k * (y / r) if r > 0 else 0.0
                    vy = v_k * (x / r) if r > 0 else 0.0

                    # Isothermal pressure (cs = constant)
                    p = sigma * cs_0**2

                    yield (sigma, vx, vy, p)

        return gas_state

    @simbi_property
    def immersed_bodies(self) -> list[dict[str, Any]]:
        return [
            {
                "body_type": "gravitational",
                "mass": 1.0,
                "velocity": [0.0, 0.0],
                "position": [0.0, 0.0],
                "grav_strength": 1.0,
                "radius": 0.01,
                "softening": 0.05,
            }
        ]

    @simbi_property
    def default_end_time(self) -> float:
        return 10.0 * 2.0 * math.pi

    @simbi_property
    def checkpoint_interval(self) -> float:
        return self.default_end_time / 50  # 50 snapshots

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
    def boundary_conditions(self) -> str:
        return "outflow"

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def data_directory(self) -> str:
        return "data/kepler/"
