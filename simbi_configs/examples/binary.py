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
        M_tot = self.gravitational_system["binary_config"]["total_mass"]
        a = self.gravitational_system["binary_config"]["semi_major"]
        v_k = math.sqrt(M_tot / a)

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
