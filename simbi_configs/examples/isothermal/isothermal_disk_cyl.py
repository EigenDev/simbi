# =============================================================================
# isothermal_disk_cyl.py
#
# isothermal hydro on the cylindrical (r, phi) disk plane: a dense core of
# radius 0.3 in solid-body rotation inside a still ambient gas, with a
# density ripple 0.1 cos(phi) so the azimuth carries structure across the
# periodic seam. outflow in r, periodic in phi.
# =============================================================================
import math
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime
from simbi.types.typing import GasStateGenerator, InitialStateType

R_IN = 0.1
R_OUT = 1.0
R0 = 0.3


class IsothermalDiskCyl(SimbiProblem):
    """a rotating isothermal core on the cylindrical (r, phi) plane."""

    sound_speed: Annotated[float, ProblemParam(1.0, cli=True, description="isothermal sound speed")]
    resolution: Annotated[tuple[int, int], ProblemParam((128, 128), cli=True, description="(n_r, n_phi)")]
    bounds: Annotated[
        list[tuple[float, float]], ProblemParam([(R_IN, R_OUT), (0.0, 2.0 * math.pi)], description="(r, phi) bounds")
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.PLANAR_CYLINDRICAL, description="(r, phi) disk plane")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.ISOTHERMAL, description="physics regime")]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW, BoundaryCondition.PERIODIC], description="boundary conditions [r, phi]"),
    ]
    end_time: Annotated[float, ProblemParam(0.1, cli=True, checkpoint_safe=True, description="simulation end time")]

    def initial_primitive_state(self) -> InitialStateType:
        """(rho, v_r, v_phi) = (10 (1 + 0.1 cos(phi)), 0, r / R0) inside r < R0 and
        (1 + 0.1 cos(phi), 0, 0) outside"""

        def gas_state() -> GasStateGenerator:
            nr, nphi = self.resolution
            (r_lo, r_hi), (p_lo, p_hi) = self.bounds
            dr, dphi = (r_hi - r_lo) / nr, (p_hi - p_lo) / nphi
            for jj in range(nphi):
                ripple = 1.0 + 0.1 * math.cos(p_lo + (jj + 0.5) * dphi)
                for ii in range(nr):
                    r = r_lo + (ii + 0.5) * dr
                    core = r < R0
                    yield ((10.0 if core else 1.0) * ripple, 0.0, r / R0 if core else 0.0)

        return gas_state
