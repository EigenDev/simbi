# =============================================================================
# isothermal_sod.py
#
# a globally-isothermal shock tube (p = cs^2 rho, no energy equation). the
# isothermal closure is a regime — the primitive is (rho, v) with no
# pressure slot. exercises the IsoNewtonian path end-to-end.
# =============================================================================
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class IsothermalSod(SimbiProblem):
    """isothermal shock tube on a 1d mesh."""

    # physics — isothermal: constant sound speed, no adiabatic index.
    sound_speed: Annotated[
        float, ProblemParam(1.0, cli=True, description="isothermal sound speed")
    ]

    # domain
    resolution: Annotated[
        int, ProblemParam(400, cli=True, description="grid resolution")
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(0.0, 1.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW], description="boundary conditions"
        ),
    ]

    # simulation control
    end_time: Annotated[
        float,
        ProblemParam(
            0.2, cli=True, checkpoint_safe=True, description="simulation end time"
        ),
    ]

    def initial_primitive_state(self) -> InitialStateType:
        """isothermal sod: density jump, at rest. primitive is (rho, v)."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / nx
            for ii in range(nx):
                xi = xmin + (ii + 0.5) * dx
                rho = 1.0 if xi < 0.5 else 0.125
                yield (rho, 0.0)

        return gas_state
