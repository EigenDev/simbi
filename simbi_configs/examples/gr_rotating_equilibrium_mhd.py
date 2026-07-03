# =============================================================================
# gr_rotating_equilibrium_mhd.py
#
# the RMHD (B = 0) rotating-equilibrium accuracy gate (design 44 phase C): the
# GrRotatingEquilibrium constant-l orbit (Fishbone-Moncrief 1976, at general
# spin, dynamic boundaries pinned to the analytic state) run on the RELATIVISTIC-
# MHD kernel path with a vanishing magnetic field. at B = 0 the RMHD equations
# reduce to GRHD, so the SAME analytic equilibrium is stationary — but the state
# now flows through the full spinning-kerr RMHD flux (the tetrad HLLD, the shift,
# and the v^phi FRAME-DRAGGING w-reconstruction), c2p, and covariant EM-stress
# source. it is the accuracy oracle for those pieces: the one-step S_phi residual
# must sit at truncation and CONVERGE under refinement (a raw v^phi
# reconstruction generates S_phi off the dragging manifold and fails to converge).
#
# usage:
#   simbi run gr_rotating_equilibrium_mhd.py --kerr-spin 0.9
# =============================================================================

from functools import partial
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime, Solver, CtMethod
from simbi.types.typing import InitialStateType, StaggeredBFieldGenerator

from simbi_configs.examples.gr_rotating_equilibrium import GrRotatingEquilibrium


class GrRotatingEquilibriumMhd(GrRotatingEquilibrium):
    """the RMHD (B=0) rotating-equilibrium gate — the frame-dragging accuracy oracle."""

    regime: Annotated[
        Regime, ProblemParam(Regime.SRMHD, description="physics regime (RMHD; B = 0)")
    ]
    solver: Annotated[Solver, ProblemParam(Solver.HLLE, cli=True, description="solver")]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.CONTACT, cli=True, description="CT edge-EMF method")
    ]

    def _boundary_prescription(self) -> dict:
        """the analytic equilibrium prescription EXTENDED with B = 0: the MHD driven boundary
        requires the full 8-component prim [rho, v_0..v_2, pre, B_0..B_2]. reuses the FM constant-l
        gas expressions (identical to the hydro base) and pins the three magnetic components to zero."""
        import simbi.expression as expr
        import math

        eq = self.equilibrium()
        gm = self.adiabatic_index
        gm1 = gm - 1.0
        mm = self.schwarzschild_mass
        a = self.kerr_spin
        l = eq.fm.ell

        g = expr.ExprGraph()
        r = expr.variable("r", g)
        th = expr.variable("theta", g)
        st = expr.sin(th)
        ct = expr.cos(th)
        sigma = r * r + (a * a) * ct * ct
        delta = r * r - (2.0 * mm) * r + a * a
        big_a = (r * r + a * a) * (r * r + a * a) - (a * a) * delta * st * st
        xx = (4.0 * l * l) * sigma * sigma * delta / (big_a * st * (big_a * st))
        sq = expr.sqrt(1.0 + xx)
        lnh = (
            0.5 * expr.log((1.0 + sq) / (sigma * delta / big_a))
            - 0.5 * sq
            - (2.0 * a * mm * l) * r / big_a
            - eq.cc
        )
        h = expr.exp(lnh)
        rho = ((h - 1.0) * (gm1 / (gm * eq.kk))) ** (1.0 / gm1)
        pre = eq.kk * rho**gm
        u_lnrf = math.copysign(1.0, l) * expr.sqrt(0.5 * (sq - 1.0))
        e_nu = expr.sqrt(sigma * delta / big_a)
        e_psi = expr.sqrt(big_a / sigma) * st
        omega = (2.0 * a * mm) * r / big_a
        u_t = expr.sqrt(1.0 + u_lnrf * u_lnrf) / e_nu
        u_p = omega * u_t + u_lnrf / e_psi
        zero = expr.constant(0.0, g)
        if self.kerr_spin != 0.0:
            b = (2.0 * mm) * r / sigma
            sq_b = expr.sqrt(1.0 + b)
            v_r = b / sq_b
            vphi = u_p * sq_b / u_t
        else:
            v_r = zero
            vphi = u_p / (e_nu * u_t)

        compiled = g.compile([rho, v_r, zero, vphi, pre, zero, zero, zero])
        return compiled.serialize_boundary(dim=3)

    def initial_primitive_state(self) -> InitialStateType:
        gas_state = super().initial_primitive_state()
        nr, npolar = self.resolution

        def b_zero(bn: str) -> StaggeredBFieldGenerator:
            # vanishing field on the staggered faces: (nr+1) x npolar for B_r, nr x (npolar+1) for
            # B_theta, nr x npolar for B_phi. B = 0 -> RMHD reduces to GRHD, equilibrium unchanged.
            if bn == "b1":
                for _jj in range(npolar):
                    for _ii in range(nr + 1):
                        yield 0.0
            elif bn == "b2":
                for _jj in range(npolar + 1):
                    for _ii in range(nr):
                        yield 0.0
            else:
                for _jj in range(npolar):
                    for _ii in range(nr):
                        yield 0.0

        return (
            gas_state,
            partial(b_zero, "b1"),
            partial(b_zero, "b2"),
            partial(b_zero, "b3"),
        )
