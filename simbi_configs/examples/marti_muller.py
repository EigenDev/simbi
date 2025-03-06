from simbi import BaseConfig, DynamicArg, simbi_property, simbi_class_property
from typing import Sequence, Callable, Generator


class MartiMuller(BaseConfig):
    """
    Marti & Muller (2003), Relativistic  Shock Tube Problem on 1D Mesh
    """

    class config:
        nzones = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type=float
        )

    @simbi_property
    def initial_primitive_state(self) -> Generator[tuple[float, ...], None, None]:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni = self.resolution
            xextent = self.bounds[1] - self.bounds[0]
            dx = xextent / ni
            for i in range(ni):
                xi = self.bounds[0] + i * dx
                if xi <= 0.5 * xextent:
                    yield (10.0, 0.0, 13.33)
                else:
                    yield (1.0, 0.0, 1e-10)

        return gas_state

    @simbi_property
    def bounds(self) -> Sequence[float]:
        return (0.0, 1.0)

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> DynamicArg:
        return self.config.nzones

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.config.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srhd"

    # -------------------- Uncomment if one wants the mesh to move


#     @simbi_property
#     def boundary_conditions(self) -> Sequence[str]:
#         return ["outflow", "dynamic"]

#     @simbi_classproperty
#     def scale_factor(cls) -> Callable[[float], float]:
#         return lambda t: 1

#     @simbi_classproperty
#     def scale_factor_derivative(cls) -> Callable[[float], float]:
#         return lambda t: 0.5

#     @simbi_classproperty
#     def boundary_sources(self) -> str:
#         return f"""
#     #include <cmath>
# extern "C" {{
#     void bx1_outer_source(double x, double t, double arr[]){{
#         double rho_ambient = 0.1;
#         double v_ambient   = 0.0;
#         double lorentz     = 1.0 / std::sqrt(1.0 - v_ambient * v_ambient);
#         double pressure    = 1.e-10;
#         double enthalpy    = 1.0 + {self.adiabatic_index} * pressure / rho_ambient / ({self.adiabatic_index} - 1.0);
#         double d = rho_ambient * lorentz;
#         double m = d * lorentz * v_ambient * enthalpy;
#         double e = d * lorentz * enthalpy - pressure - d;
#         arr[0] = d;   // density
#         arr[1] = m;   // x1-momentum
#         arr[2] = e;   // energy
#         arr[3] = 0.0; // scalar concentration
#     }}
# }}
#         """
