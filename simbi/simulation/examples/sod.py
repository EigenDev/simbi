# =============================================================================
# sod.py
#
# sod shock tube problem on the SimbiProblem api.
# demonstrates the minimal interface for defining physics problems.
#
# usage:
#   simbi run sod --end-time 0.2 --resolution 1000
#   simbi run sod --checkpoint data/checkpoint.h5 --end-time 1.0
# =============================================================================
from simbi.simulation import (
    CellSpacing,
    CoordSystem,
    GasStateGenerator,
    InitialStateType,
    ProblemParam,
    Regime,
    SimbiProblem,
)


class SodProblem(SimbiProblem):
    """
    sod's shock tube problem in 1d newtonian hydrodynamics.

    classic test problem with a discontinuity at x=0.5:
    - left state:  (rho, v, p) = (1.0, 0.0, 1.0)
    - right state: (rho, v, p) = (0.125, 0.0, 0.1)
    """

    # problem-specific fields with cli exposure
    resolution: int = ProblemParam(
        1000, cli=True, description="grid resolution"
    )

    # required fields with problem-specific defaults
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic gas index"
    )
    bounds: tuple[tuple[float, float]] = ProblemParam(
        ((0.0, 1.0),), description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(
        Regime.NEWTONIAN, description="physics regime"
    )

    # optional customization
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for sod shock tube."""

        def gas_state() -> GasStateGenerator:
            nx = self.resolution
            x_min, x_max = self.bounds[0]
            dx = (x_max - x_min) / nx

            for ii in range(nx):
                x = x_min + (ii + 0.5) * dx
                if x < 0.5:
                    yield (1.0, 0.0, 1.0)  # rho, v, p
                else:
                    yield (0.125, 0.0, 0.1)

        return gas_state
