# =============================================================================
# ram61.py
#
# shock with non-zero transverse velocity on both sides in 2d.
# adapted from zhang and macfadyen (2006) section 6.1.
# note: hard test, may fail.
# =============================================================================
from simbi import ProblemParam, SimbiProblem
from simbi.types import CellSpacing, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class Ram61(SimbiProblem):
    """shock with transverse velocity on both sides, zhang & macfadyen (2006) 6.1."""

    # physics
    adiabatic_index: float = ProblemParam(
        5.0 / 3.0, description="adiabatic index"
    )

    # domain
    resolution: tuple[int, int] = ProblemParam(
        (400, 400), cli=True, description="grid resolution"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(0.0, 1.0), (0.0, 1.0)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(Regime.SRHD, description="physics regime")
    x1_spacing: CellSpacing = ProblemParam(
        CellSpacing.LINEAR, description="grid spacing in x1 direction"
    )

    # simulation control
    end_time: float = ProblemParam(
        0.4, cli=True, checkpoint_safe=True, description="simulation end time"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for ram61 shock."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            dx = (xmax - xmin) / nx

            for jj in range(ny):
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx

                    if ii < nx // 2:
                        yield (1.0, 0.0, 0.90, 1e3)
                    else:
                        yield (1.0, 0.0, 0.90, 1e-2)

        return gas_state
