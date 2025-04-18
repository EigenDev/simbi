import math
from simbi.core.types.typing import ExpressionDict
import simbi.expression as expr
from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Any, Generator


class RayleighTaylor(BaseConfig):
    """
    Rayleigh Taylor problem in Newtonian Fluid
    """

    class config:
        xnpts = DynamicArg(
            "xnpts", 200, help="Number of zones in x dimensions", var_type=int
        )
        ynpts = DynamicArg(
            "ynpts", 600, help="Number of zones in y dimensions", var_type=int
        )

    xmin = -0.25
    xmax = 0.25
    ymin = -0.75
    ymax = 0.75
    rhoU = 2.0
    p0 = 2.5
    rhoD = 1.0
    g0 = 0.1
    vamp = 0.01
    ymidpoint = (ymax + ymin) * 0.5

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj = self.resolution
            xextent = self.xmax - self.xmin
            yextent = self.ymax - self.ymin
            dx = xextent / ni
            dy = yextent / nj
            for j in range(nj):
                y = self.ymin + j * dy
                for i in range(ni):
                    x = self.xmin + i * dx
                    if y <= self.ymidpoint:
                        rho = self.rhoD
                    else:
                        rho = self.rhoU
                    p = self.p0 - self.g0 * rho * y
                    vy = (
                        self.vamp
                        * 0.25
                        * (1 + math.cos(4.0 * math.pi * x))
                        * (1.0 + math.cos(3.0 * math.pi * y))
                    )
                    yield rho, 0.0, vy, p

        return gas_state

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[Any]:
        return (self.config.xnpts, self.config.ynpts)

    @simbi_property
    def adiabatic_index(self) -> float:
        return 7.0 / 5.0

    @simbi_property
    def gravity_source_expressions(self) -> ExpressionDict:
        graph = expr.ExprGraph()
        x = expr.variable("x", graph)
        y = expr.variable("y", graph)
        t = expr.variable("t", graph)

        x_comp = expr.constant(0.0, graph)
        y_comp = expr.constant(-self.g0, graph) - t
        compiled_expr = graph.compile([x_comp, y_comp])
        return compiled_expr.serialize()

    @simbi_property
    def regime(self) -> str:
        return "classical"

    @simbi_property
    def boundary_conditions(self) -> list[str]:
        return ["periodic", "reflecting"]

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def data_directory(self) -> str:
        return "data/rt_config"
