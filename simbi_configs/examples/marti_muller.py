import simbi.expression as expr
from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, CellSpacing
from simbi.core.types.typing import GasStateGenerator, InitialStateType, ExpressionDict
from pydantic import computed_field
from typing import Callable


class MartiMuller(SimbiBaseConfig):
    """
    Marti & Muller (2003), Relativistic Shock Tube Problem on 1D Mesh
    """

    # Required fields from SimbiBaseConfig
    resolution: int = SimbiField(1000, description="Grid resolution")

    bounds: list[tuple[float, float]] = SimbiField(
        [(0.0, 1.0)], description="Domain boundaries"
    )

    coord_system: CoordSystem = SimbiField(
        CoordSystem.CARTESIAN, description="Coordinate system"
    )

    regime: Regime = SimbiField(Regime.SRHD, description="Physics regime")

    adiabatic_index: float = SimbiField(4.0 / 3.0, description="Adiabatic index")

    # Optional customizations
    x1_spacing: CellSpacing = SimbiField(
        CellSpacing.LINEAR, description="Grid spacing in x1 direction"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """Generate initial primitive state for Marti & Muller shock tube.

        Returns:
            Generator function that yields primitive variables
        """

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            for i in range(nx):
                xi = xmin + (i + 0.5) * dx  # Cell center
                if xi <= 0.5 * xextent:
                    yield (10.0, 0.0, 13.33)  # Left state: (rho, vx, p)
                else:
                    yield (1.0, 0.0, 1e-10)  # Right state: (rho, vx, p)

        return gas_state

    # -------------------- Uncomment if one wants the mesh to move
    """
    # Boundary conditions for moving mesh
    boundary_conditions: list[str] = SimbiField(
        ["outflow", "dynamic"], description="Boundary conditions"
    )

    # Scale factor and derivative for mesh motion
    @computed_field
    @property
    def scale_factor(self) -> Callable[[float], float]:
        return lambda t: 1

    @computed_field
    @property
    def scale_factor_derivative(self) -> Callable[[float], float]:
        return lambda t: 0.5

    @computed_field
    @property
    def bx1_outer_expressions(self) -> ExpressionDict:
        graph = expr.ExprGraph()

        # const values
        gamma = expr.constant(float(self.adiabatic_index), graph)
        rho_ambient = expr.constant(0.1, graph)
        v_ambient = expr.constant(0.0, graph)
        pressure = expr.constant(1e-10, graph)

        # build expressions
        v_squared = v_ambient * v_ambient
        one = expr.constant(1.0, graph)
        lorentz = one / (one - v_squared) ** 0.5

        gamma_minus_1 = gamma - one
        enthalpy = one + gamma * pressure / rho_ambient / gamma_minus_1

        # final expressions for each component
        density = rho_ambient * lorentz
        momentum = rho_ambient * lorentz * lorentz * enthalpy * v_ambient
        energy = (
            rho_ambient * lorentz * lorentz * enthalpy
            - pressure
            - rho_ambient * lorentz
        )
        scalar = expr.constant(0.0, graph)

        # compile the expressions
        compiled_expr = graph.compile([density, momentum, energy, scalar])

        return compiled_expr.serialize()
    """
