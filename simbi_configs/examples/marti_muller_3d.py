# =============================================================================
# marti_muller_3d.py
#
# marti & muller (2003), relativistic shock tube problem on 3d mesh.
# =============================================================================
from simbi import ProblemParam, SimbiProblem
from simbi.types import CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType


class MartiMuller3D(SimbiProblem):
    """marti & muller (2003), relativistic shock tube problem on 3d mesh."""

    # physics
    adiabatic_index: float = ProblemParam(
        4.0 / 3.0, description="adiabatic index"
    )

    # domain
    resolution: tuple[int, int, int] = ProblemParam(
        (100, 100, 100), cli=True, description="grid resolution"
    )
    bounds: list[tuple[float, float]] = ProblemParam(
        [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="domain boundaries"
    )
    coord_system: CoordSystem = ProblemParam(
        CoordSystem.CARTESIAN, description="coordinate system"
    )
    regime: Regime = ProblemParam(Regime.SRHD, description="physics regime")

    # simulation control
    end_time: float = ProblemParam(
        0.4, cli=True, checkpoint_safe=True, description="simulation end time"
    )

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for 3d marti & muller shock tube."""

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            xmin, xmax = self.bounds[0]
            xextent = xmax - xmin
            dx = xextent / nx

            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        xi = xmin + (ii + 0.5) * dx
                        if xi <= 0.5 * xextent:
                            yield (10.0, 0.0, 0.0, 0.0, 13.33)
                        else:
                            yield (1.0, 0.0, 0.0, 0.0, 1e-10)

        return gas_state
