import math
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.core.types.typing import GasStateGenerator, InitialStateType, ExpressionDict
from pydantic import computed_field
from pathlib import Path
import simbi.expression as expr


class RayleighTaylor(SimbiBaseConfig):
    """
    Rayleigh Taylor problem in Newtonian Fluid
    """

    # Configuration parameters for domain and resolution
    resolution: tuple[int, int] = SimbiField(
        (200, 600), description="Grid resolution (x, y)"
    )

    bounds: list[tuple[float, float]] = SimbiField(
        [(-0.25, 0.25), (-0.75, 0.75)], description="Domain boundaries"
    )

    # Physical parameters as fields
    rhoU: float = SimbiField(2.0, description="Upper layer density")
    rhoD: float = SimbiField(1.0, description="Lower layer density")
    p0: float = SimbiField(2.5, description="Reference pressure")
    g0: float = SimbiField(0.1, description="Gravitational acceleration")
    vamp: float = SimbiField(0.01, description="Velocity perturbation amplitude")

    # Required fields from SimbiBaseConfig
    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.CLASSICAL, description="Physics regime")

    adiabatic_index: float = SimbiField(7.0 / 5.0, description="Adiabatic index")

    # Optional customizations
    boundary_conditions: list[BoundaryCondition] = SimbiField(
        [BoundaryCondition.PERIODIC, BoundaryCondition.REFLECTING],
        description="Boundary conditions [x, y]",
    )

    solver: Solver = SimbiField(Solver.HLLC, description="Numerical solver")

    data_directory: Path = SimbiField(
        Path("data/rt_config"), description="Output data directory"
    )

    @computed_field
    @property
    def ymidpoint(self) -> float:
        """Calculate middle of y domain"""
        return 0.5 * (self.bounds[1][0] + self.bounds[1][1])

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Rayleigh-Taylor instability.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]

            xextent = xmax - xmin
            yextent = ymax - ymin
            dx = xextent / nx
            dy = yextent / ny

            ymid = self.ymidpoint

            for j in range(ny):
                y = ymin + (j + 0.5) * dy  # Cell center
                for i in range(nx):
                    x = xmin + (i + 0.5) * dx  # Cell center

                    # Density stratification - heavier fluid on top
                    if y <= ymid:
                        rho = self.rhoD  # Lower density
                    else:
                        rho = self.rhoU  # Upper (heavier) density

                    # Hydrostatic pressure profile
                    p = self.p0 - self.g0 * rho * y

                    # Velocity perturbation
                    vy = (
                        self.vamp
                        * 0.25
                        * (1 + math.cos(4.0 * math.pi * x))
                        * (1.0 + math.cos(3.0 * math.pi * y))
                    )

                    yield (rho, 0.0, vy, p)

        return gas_state

    @computed_field
    @property
    def gravity_source_expressions(self) -> ExpressionDict:
        """Define gravity source terms"""
        graph = expr.ExprGraph()

        # Constant gravity in the negative y direction
        x_comp = expr.constant(0.0, graph)
        y_comp = expr.constant(-self.g0, graph)

        compiled_expr = graph.compile([x_comp, y_comp])
        return compiled_expr.serialize()
