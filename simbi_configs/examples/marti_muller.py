import simbi.expression as expr
from simbi import BaseConfig, DynamicArg, simbi_property
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

    # @simbi_property
    # def boundary_conditions(self) -> Sequence[str]:
    #     return ["outflow", "dynamic"]

    # @simbi_property
    # def scale_factor(cls) -> Callable[[float], float]:
    #     return lambda t: 1

    # @simbi_property
    # def scale_factor_derivative(cls) -> Callable[[float], float]:
    #     return lambda t: 0.5

    # @simbi_property
    # def bx1_outer_expressions(self) -> ExpressionDict:
    #     graph = expr.ExprGraph()

    #     x = expr.variable("x1", graph)
    #     t = expr.variable("t", graph)

    #     # const values
    #     gamma = expr.constant(float(self.config.adiabatic_index), graph)
    #     rho_ambient = expr.constant(0.1, graph)
    #     v_ambient = expr.constant(0.0, graph)
    #     pressure = expr.constant(1e-10, graph)

    #     # build expressions
    #     v_squared = v_ambient * v_ambient
    #     one = expr.constant(1.0, graph)
    #     lorentz = one / (one - v_squared) ** 0.5

    #     gamma_minus_1 = gamma - one
    #     enthalpy = one + gamma * pressure / rho_ambient / gamma_minus_1

    #     # final expressions for each component
    #     density = rho_ambient * lorentz
    #     momentum = rho_ambient * lorentz * lorentz * enthalpy * v_ambient
    #     energy = (
    #         rho_ambient * lorentz * lorentz * enthalpy
    #         - pressure
    #         - rho_ambient * lorentz
    #     )
    #     scalar = expr.constant(0.0, graph)

    #     # compile the expressions
    #     compiled_expr = graph.compile([density, momentum, energy, scalar])

    #     return compiled_expr.serialize()
