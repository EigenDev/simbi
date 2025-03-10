import math
from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Any, Sequence, Generator
from simbi.typing import InitialStateType


class KeplerProblem(BaseConfig):
    """Keplerian Disk Test Problem"""

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
            def taper_funtion(r: float, m: float) -> float:
                """Taper function for density profile"""
                return 1.0 - math.exp(-((r) ** m))

            dx = (self.xmax - self.xmin) / self.config.npts
            dy = (self.ymax - self.ymin) / self.config.npts

            # Fundamental scales (everything in code units)
            R_0 = 1.0  # Reference radius
            M_0 = self.immersed_bodies[0]["mass"]  # Central mass
            G = 1.0  # Gravitational constant
            GM = G * M_0  # Gravitational parameter
            T_0 = math.sqrt(R_0**3 / (G * M_0))  # Reference time

            # Derived scales
            v_0 = R_0 / T_0  # Reference velocity
            sigma_0 = M_0 / R_0**2  # Reference surface density

            # Non-dimensional parameters
            h_r = 0.1  # H/R aspect ratio
            r_in = 0.2  # Inner radius in units of R_0
            gamma = self.adiabatic_index
            P = 1.5  # surface density power index

            # Sound speed in units of v_0
            cs_0 = h_r * v_0

            # Non-dimensional pressure constant
            K = cs_0**2 * sigma_0 ** (1 - gamma) / gamma

            for j in range(self.config.npts):
                y = self.ymin + (j + 0.5) * dy
                for i in range(self.config.npts):
                    x = self.xmin + (i + 0.5) * dx
                    r = math.sqrt(x**2 + y**2)

                    if r >= r_in:  # Only check inner boundary
                        # Power law density profile
                        sigma = sigma_0 * (r) ** (-P)  # Changed power law

                        # Local Keplerian velocity
                        v_k = math.sqrt(GM / r)

                        # Pressure profile for stability
                        p = K * sigma**gamma
                        cs2 = gamma * p / sigma
                        pressure_support = cs2 / (v_k**2)
                        v_k *= math.sqrt(1.0 - pressure_support)
                    else:
                        # Inner cavity only
                        m_dropoff = 4.0
                        sigma = sigma_0 * r ** (-P) * taper_funtion(r / r_in, m_dropoff)
                        v_k = math.sqrt(GM / r)  # Base Keplerian
                        v_k *= math.exp(-((r_in / r) ** (m_dropoff)))
                        p = K * sigma**gamma

                    vx = -v_k * (y / r)
                    vy = v_k * (x / r)
                    yield (sigma, vx, vy, p)

        return gas_state

    @simbi_property
    def immersed_bodies(self) -> list[dict[str, Any]]:
        # Basic scales
        R_0 = 1.0
        M_0 = 10.0
        G = 1.0
        T_0 = math.sqrt(R_0**3 / (G * M_0))

        # Dimensionless gravitational strength
        grav = G * T_0**2 / R_0**3  # Should equal 1.0 with our choice of units

        return [
            {
                "body_type": "gravitational",
                "mass": M_0,  # Already dimensionless in our units
                "velocity": [0.0, 0.0],
                "position": [0.0, 0.0],
                "grav_strength": grav,  # This ensures G=1 in code units
                "radius": 0.1 * R_0,
                "softening": 0.05 * R_0,
            }
        ]

    @simbi_property
    def default_end_time(self) -> float:
        # Run for about 2 orbital periods at r=1
        G = self.immersed_bodies[0]["grav_strength"]
        M = self.immersed_bodies[0]["mass"]
        T = 2.0 * 2 * math.pi * math.sqrt(1.0**3 / (G * M))
        return T

    @simbi_property
    def checkpoint_interval(self) -> float:
        return self.tend / 50  # 50 snapshots

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
        return 5.0 / 3.0

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
