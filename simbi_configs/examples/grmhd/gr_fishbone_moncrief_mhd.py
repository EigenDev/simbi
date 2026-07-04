# =============================================================================
# gr_fishbone_moncrief_mhd.py
#
# the MAGNETIZED Fishbone-Moncrief torus (design 44 phase C, the MRI target): the
# GrFishboneMoncrief hydro torus threaded with a WEAK poloidal seed field, run on
# the spinning-kerr RMHD kernel path (tetrad HLLD flux + UCT-HLLD sharp CT +
# frame-dragging w-reconstruction + B^phi dragging ghost). the field is seeded
# div-free from a single azimuthal vector potential A_phi(r, theta) = max(rho/
# rho_max - rho_cut, 0) — concentric poloidal loops confined to the dense torus
# interior — via the METRIC-WEIGHTED discrete curl (so the w-weighted div(B) is
# machine zero on the kerr grid). the amplitude is set by the target minimum
# plasma beta = p_gas/(b^2/2): the standard MRI initial condition. the weak field
# is magneto-rotationally UNSTABLE inside the differentially-rotating torus; the
# fastest-growing mode has lambda_MRI ~ 2 pi v_A / Omega, and the turbulence it
# drives transports angular momentum outward and accretes the torus onto the hole.
#
# usage:
#   simbi run gr_fishbone_moncrief_mhd.py --kerr-spin 0.9 --target-beta 100
# =============================================================================

import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam
from simbi.types import CtMethod, Regime, Solver
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)

from simbi_configs.examples.grmhd.gr_fishbone_moncrief import GrFishboneMoncrief


class GrFishboneMoncriefMhd(GrFishboneMoncrief):
    """the magnetized FM torus — the weak-field MRI initial condition on a kerr background."""

    # a FAT torus (r_max ~ 12 M, the classic Gammie 2003 / HARM MRI torus) — well-resolved on the
    # log grid, unlike the thin kappa ~ 1.01 gate torus.
    kappa: Annotated[
        float, ProblemParam(1.3, cli=True, description="FM angular-momentum parameter (fat torus)")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.SRMHD, description="physics regime (RMHD)")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLD, cli=True, description="Riemann solver")]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.UCT, cli=True, description="CT edge-EMF (UCT-HLLD)")
    ]
    target_beta: Annotated[
        float, ProblemParam(100.0, cli=True, description="minimum plasma beta p_gas/(b^2/2)")
    ]
    rho_cut: Annotated[
        float, ProblemParam(0.2, cli=True, description="A_phi = max(rho/rho_max - rho_cut, 0)")
    ]

    def _sqrtg(self, r: float, th: float) -> float:
        mm, a = self.schwarzschild_mass, self.kerr_spin
        s = math.sin(th)
        if a != 0.0:
            sigma = r * r + a * a * math.cos(th) ** 2
            return sigma * s * math.sqrt(1.0 + 2.0 * mm * r / sigma)
        return r * r * s / math.sqrt(1.0 - 2.0 * mm / r)

    def _gamma_poloidal(self, r: float, th: float) -> tuple[float, float]:
        # (gamma_rr, gamma_theta_theta) for the poloidal |B|^2 (B^phi = 0 at seed).
        mm, a = self.schwarzschild_mass, self.kerr_spin
        if a != 0.0:
            sigma = r * r + a * a * math.cos(th) ** 2
            return 1.0 + 2.0 * mm * r / sigma, sigma
        return 1.0 / (1.0 - 2.0 * mm / r), r * r

    def initial_primitive_state(self) -> InitialStateType:
        gas_state = super().initial_primitive_state()
        torus = self.torus()
        nr, npolar = self.resolution
        (rmin, rmax) = self.bounds[0]
        (tmin, tmax) = self.bounds[1]
        q = (rmax / rmin) ** (1.0 / nr)
        dth = (tmax - tmin) / npolar
        rf = [rmin * q**ii for ii in range(nr + 1)]
        tf = [tmin + jj * dth for jj in range(npolar + 1)]
        r_c = [0.5 * (rf[i] + rf[i + 1]) for i in range(nr)]
        th_c = [0.5 * (tf[j] + tf[j + 1]) for j in range(npolar)]

        # A_phi = max(rho/rho_max - rho_cut, 0) sampled at every FACE CORNER, ONCE against a
        # single torus. torus construction runs a 4096-point scan + a 200-step bisection for
        # r_max, so re-deriving the potential per gather point (curl stencils x beta scan x the
        # three staggered generators = O(nr*npolar) evaluations) dominates the ic cost. the curl,
        # beta normalization, and generators all index this grid instead.
        def aphi_corner(r: float, th: float) -> float:
            state = torus.primitive(r, th)
            if state is None:
                return 0.0
            return max(state[0] / self.rho_torus_max - self.rho_cut, 0.0)

        aphi = [[aphi_corner(rf[i], tf[j]) for j in range(npolar + 1)] for i in range(nr + 1)]
        # cell-centered torus primitive, sampled once for the beta scan (rho + pressure).
        cell = [[torus.primitive(r_c[i], th_c[j]) for j in range(npolar)] for i in range(nr)]

        # unit-amplitude poloidal field from the metric-weighted curl of A_phi (arithmetic face
        # centers, matching the CT curl weights so the w-weighted div(B) is machine zero).
        def br_unit(i: int, j: int) -> float:
            w = self._sqrtg(rf[i], th_c[j]) * dth
            return (aphi[i][j + 1] - aphi[i][j]) / w

        def bth_unit(i: int, j: int) -> float:
            w = self._sqrtg(r_c[i], tf[j]) * (rf[i + 1] - rf[i])
            return -(aphi[i + 1][j] - aphi[i][j]) / w

        # beta normalization: the minimum plasma beta over the DENSE CORE (rho > 0.5 of the actual
        # grid-peak density). the cut excludes the low-pressure torus surface, where beta is small
        # for any field and would otherwise pin the amplitude to a truncation-noise edge cell (the
        # standard FM-MRI recipe). the threshold adapts to the resolved peak, not the nominal rho_max.
        rho_peak = max((cell[ii][jj] or (0.0,))[0] for jj in range(npolar) for ii in range(nr))
        beta_min = math.inf
        for jj in range(npolar):
            for ii in range(nr):
                state = cell[ii][jj]
                if state is None or state[0] < 0.5 * rho_peak:
                    continue
                bcell_r = 0.5 * (br_unit(ii, jj) + br_unit(ii + 1, jj))
                bcell_th = 0.5 * (bth_unit(ii, jj) + bth_unit(ii, jj + 1))
                g_rr, g_thth = self._gamma_poloidal(r_c[ii], th_c[jj])
                bsq = g_rr * bcell_r * bcell_r + g_thth * bcell_th * bcell_th
                if bsq <= 0.0:
                    continue
                beta_min = min(beta_min, state[3] / (0.5 * bsq))
        a0 = math.sqrt(beta_min / self.target_beta) if math.isfinite(beta_min) else 0.0
        print(f"fm-mhd: r_max={torus.r_max:.3f} A0={a0:.4e} (min plasma beta -> {self.target_beta})")

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            if bn == "b1":
                for jj in range(npolar):
                    for ii in range(nr + 1):
                        yield a0 * br_unit(ii, jj)
            elif bn == "b2":
                for jj in range(npolar + 1):
                    for ii in range(nr):
                        yield a0 * bth_unit(ii, jj)
            else:
                for _jj in range(npolar):
                    for _ii in range(nr):
                        yield 0.0

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
