import math
from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Any, Sequence, Generator
from simbi.typing import InitialStateType


class BinaryOrbitTest(BaseConfig):
    """Simple test of binary orbital dynamics without gas."""

    class config:
        npts = DynamicArg(
            "npts", 256, help="Grid resolution (minimal for test)", var_type=int
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
                    yield (1.0, 0.0, 0.0, 1e-8)

        return gas_state

    @simbi_property
    def immersed_bodies(self) -> list[dict[str, Any]]:
        # Binary parameters in natural units
        G = 1.0
        M_tot = 1.0
        a = 0.2
        radius = 0.01
        softening = 0.001

        # Compute exact Keplerian velocity
        r2 = a * a + (softening * radius) * (softening * radius)
        v_k = math.sqrt(G * M_tot / math.sqrt(r2))

        return [
            {
                "body_type": "gravitational",
                "mass": 0.5 * M_tot,
                "velocity": [0.0, -v_k],
                "position": [-0.5 * a, 0.0],
                "grav_strength": G,
                "radius": radius,
                "softening": softening,
                "two_way_coupling": False,
            },
            {
                "body_type": "gravitational",
                "mass": 0.5 * M_tot,
                "velocity": [0.0, v_k],
                "position": [0.5 * a, 0.0],
                "grav_strength": G,
                "radius": radius,
                "softening": softening,
                "two_way_coupling": False,
            },
        ]

    @simbi_property
    def default_end_time(self) -> float:
        # Run for 10 orbits
        G = 1.0
        M = 1.0
        a = 0.2
        radius = 0.01
        softening = 0.001
        r2 = a * a + (softening * radius) * (softening * radius)
        T_orbit = 2.0 * math.pi * math.pow(r2, 0.75) / math.sqrt(G * M)
        return 10.0 * T_orbit

    @simbi_property
    def checkpoint_interval(self) -> float:
        # Save 100 snapshots per orbit
        return self.default_end_time / 1000

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
