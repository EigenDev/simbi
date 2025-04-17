from simbi import BaseConfig, DynamicArg, simbi_property, serialize_expressions, Expr
from simbi.typing import InitialStateType, ExpressionDict
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
    def initial_primitive_state(self) -> InitialStateType:
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

    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ["outflow", "dynamic"]

    @simbi_property
    def scale_factor(cls) -> Callable[[float], float]:
        return lambda t: 1

    @simbi_property
    def scale_factor_derivative(cls) -> Callable[[float], float]:
        return lambda t: 0.5

    @simbi_property
    def bx1_outer_expressions(self) -> ExpressionDict:
        x = Expr.x1()
        t = Expr.t()
        gamma = float(self.config.adiabatic_index)

        rho_ambient = 0.1
        v_ambient = 0.0
        pressure = 1e-10

        # build expressions
        lorentz = 1.0 / (1.0 - v_ambient * v_ambient) ** 0.5
        enthalpy = 1.0 + gamma * pressure / rho_ambient / (gamma - 1.0)

        # final expressions for each component
        density = Expr(rho_ambient * lorentz)
        momentum = Expr(rho_ambient * lorentz * lorentz * enthalpy * v_ambient)
        energy = Expr(
            rho_ambient * lorentz * lorentz * enthalpy
            - pressure
            - rho_ambient * lorentz
        )
        scalar = Expr(0.0)

        # serialize into our format
        return serialize_expressions([density, momentum, energy, scalar])
