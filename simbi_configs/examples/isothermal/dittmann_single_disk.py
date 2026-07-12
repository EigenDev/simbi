# =============================================================================
# dittmann_single_disk.py
#
# the single-object accretion disk of dittmann & ryan (2021), section 3: a
# geometrically thin (mach ~ 10), locally isothermal disk around ONE accretor,
# the sharpest test of a sink prescription. a standard sink removes the full
# local angular momentum and its steady-state surface density depends on the
# sink rate even outside r_s (their figure 1); a torque-free sink removes mass
# but not angular momentum and the profile is rate-independent.
#
# the accretor's surface stack is selected by AccretionProperties.torque_free_xi
# xi = 0 is the standard drain, xi = 1 the torque-free sink.
# sweep it (and, once c_drain is config-exposed, the sink rate) to reproduce the
# figure-1 collapse.
#
# usage:
#  simbi run dittmann_single_disk --torque-free-xi 0.0   # standard sink
#  simbi run dittmann_single_disk --torque-free-xi 1.0   # torque-free sink
# =============================================================================
import math
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime, Solver
from simbi.types.bodies import (
    AccretionProperties,
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
)
from simbi.types.typing import ExpressionDict, GasStateGenerator, InitialStateType


class DittmannSingleDisk(SimbiProblem):
    """single-object thin accretion disk (dittmann & ryan 2021, section 3)."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(1.0, description="adiabatic index (isothermal)")
    ]
    mach: Annotated[
        float,
        ProblemParam(
            10.0, cli=True, description="disk mach number v_kep / c_s (H/r = 1/mach)"
        ),
    ]
    central_mass: Annotated[
        float, ProblemParam(1.0, description="central accretor mass (G = M = 1)")
    ]
    sink_radius: Annotated[
        float,
        ProblemParam(
            0.05, cli=True, description="accretion mask radius r_s (code units)"
        ),
    ]
    torque_free_xi: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            description="torque-free strength: 0 = standard sink (books full "
            "angular momentum), 1 = torque-free (mass drains, angular momentum "
            "stays). the dittmann dial.",
        ),
    ]
    nu: Annotated[
        float,
        ProblemParam(
            0.01,
            cli=True,
            description="constant kinematic viscosity. dittmann "
            "section 3 uses 0.01; the viscous inflow drives accretion "
            "(v_r = -3 nu / 2r). 0 = inviscid (numerical-viscosity-only).",
        ),
    ]
    alpha: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            description="Shakura-Sunyaev alpha: nu(r) = alpha "
            "cs^2 / Omega_k(r) about the central mass. >0 TAKES PRECEDENCE over "
            "the constant nu. 0 = use the constant nu instead.",
        ),
    ]

    # buffer zone: a sponge that relaxes the outer annulus toward the initial
    # disk equilibrium, so the flow reaches a steady state instead of draining
    # out through the outflow boundary (dittmann uses a dirichlet outer BC).
    use_buffer: Annotated[
        bool,
        ProblemParam(
            True, cli=True, description="enable the outer-boundary sponge"
        ),
    ]
    buffer_fraction: Annotated[
        float,
        ProblemParam(
            0.85,
            description="sponge inner edge as a fraction of the domain half-width",
        ),
    ]
    buffer_damp_orbits: Annotated[
        float,
        ProblemParam(
            0.1,
            description="sponge damping timescale in orbits at r = 1 (T = 2 pi)",
        ),
    ]

    # disk profile (dittmann eq. 21: a cavity filled by accreting gas)
    cavity_radius: Annotated[
        float,
        ProblemParam(0.15, description="cavity scale r_e in Sigma = exp(-(r_e/r)^xi)"),
    ]
    cavity_index: Annotated[
        float, ProblemParam(10.0, description="cavity steepness xi_cav")
    ]

    # domain
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((512, 512), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-2.5, 2.5), (-2.5, 2.5)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime,
        ProblemParam(Regime.ISOTHERMAL, description="physics regime (thin disk)"),
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1"),
    ]

    # numerics
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLE, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(BoundaryCondition.OUTFLOW, description="boundary conditions"),
    ]
    cfl_number: Annotated[
        float, ProblemParam(0.4, description="cfl condition number")
    ]

    # simulation control (times in orbital periods at r = 1, T = 2 pi)
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/dittmann_single_disk/"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            20.0 * math.pi,
            cli=True,
            checkpoint_safe=True,
            description="end time (10 orbits at r = 1)",
        ),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            0.5 * math.pi,
            cli=True,
            checkpoint_safe=True,
            description="checkpoint interval (0.25 orbit)",
        ),
    ]
    diagnostic_interval: Annotated[
        float,
        ProblemParam(
            0.1 * math.pi,
            cli=True,
            checkpoint_safe=True,
            description="body-diagnostics cadence; writes the accretor's "
            "torque/force/accreted-mass to diagnostics.dat",
        ),
    ]

    @property
    def ambient_sound_speed(self) -> float:
        # c_s = v_kep(r=1) / mach, with v_kep(1) = sqrt(G M) = 1.
        return 1.0 / self.mach

    @computed_field
    @property
    def viscosity(self) -> float:
        # the backend reads `viscosity`; expose it as the CLI nu.
        return self.nu

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        dx = (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]
        softening = 2.0 * dx
        r_acc = max(self.sink_radius, 2.0 * dx)
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
                mass=self.central_mass,
                radius=0.0,
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                gravitational=GravitationalProperties(softening_length=softening),
                accretion=AccretionProperties(
                    accretion_radius=r_acc,
                    torque_free_xi=self.torque_free_xi,
                ),
            )
        ]

    @property
    def buffer_parameters(self) -> dict[str, float]:
        """outer-sponge geometry + timescale."""
        domain_half = abs(self.bounds[0][1])
        buffer_width = (1.0 - self.buffer_fraction) * domain_half
        buffer_radius = domain_half - buffer_width
        t_orbit = 2.0 * math.pi
        return {
            "buffer_radius": buffer_radius,
            "buffer_width": buffer_width,
            "damp_time": self.buffer_damp_orbits * t_orbit,
        }

    def buffer_sponge_terms(
        self, x1: expr.Expr, x2: expr.Expr
    ) -> list[expr.Expr]:
        """the iso sponge outputs [kappa, den_ref, mom_ref_x, mom_ref_y]: relax
        the conserved state toward the outer disk equilibrium (Sigma -> 1 in
        sub-keplerian rotation, well outside the cavity), kappa ramping
        0 -> 1/damp_time across the buffer annulus. the rust `sponge` kind forms
        S_U = kappa (U_ref - U) for density and momentum (iso: no energy)."""
        bp = self.buffer_parameters
        buffer_radius = expr.constant(bp["buffer_radius"], x1.graph)
        buffer_width = expr.constant(bp["buffer_width"], x1.graph)
        damp_time = expr.constant(bp["damp_time"], x1.graph)

        r = expr.sqrt(x1 * x1 + x2 * x2) + 1e-10
        radial = (r - buffer_radius) / buffer_width
        ramp = expr.max_expr(expr.constant(0.0, x1.graph), radial)
        ramp = expr.min_expr(ramp, expr.constant(1.0, x1.graph))
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)  # smooth cubic
        kappa = ramp / damp_time

        # the outer disk reference: Sigma = 1 (outside the cavity), sub-keplerian
        # v_phi^2 = v_kep^2 - cs^2, tangential in the plane.
        gm = expr.constant(self.central_mass, x1.graph)
        cs2 = expr.constant(self.ambient_sound_speed ** 2, x1.graph)
        den_ref = expr.constant(1.0, x1.graph)
        v_phi = expr.sqrt(
            expr.max_expr(gm / r - cs2, expr.constant(0.0, x1.graph))
        )
        r_inv = 1.0 / r
        mom_x = -1.0 * den_ref * v_phi * x2 * r_inv
        mom_y = den_ref * v_phi * x1 * r_inv
        return [kappa, den_ref, mom_x, mom_y]

    @computed_field
    @property
    def hydro_source_expressions(self) -> ExpressionDict:
        """the outer sponge as a rust `sponge` source (or none when disabled)."""
        if not self.use_buffer:
            return {}
        graph = expr.ExprGraph()
        x1 = expr.variable("x1", graph)
        x2 = expr.variable("x2", graph)
        outputs = self.buffer_sponge_terms(x1, x2)
        return graph.compile(outputs).serialize_source(
            expr.SourceKind.SPONGE, dim=2, params=[0.0]
        )

    def initial_primitive_state(self) -> InitialStateType:
        """cavity surface density + keplerian rotation with pressure correction."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]
            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny

            gm = self.central_mass  # G = 1
            cs = self.ambient_sound_speed
            cs2 = cs * cs
            r_e = self.cavity_radius
            xi_cav = self.cavity_index
            sigma_floor = 1e-6
            eps = 1e-10

            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx
                    r = math.sqrt(x * x + y * y)

                    if r < eps:
                        yield (sigma_floor, 0.0, 0.0, sigma_floor * cs2)
                        continue

                    # cavity profile (dittmann eq. 21): evacuated for r << r_e,
                    # -> 1 for r >> r_e.
                    sigma = sigma_floor + math.exp(-((r_e / r) ** xi_cav))

                    # keplerian angular velocity with the pressure correction of
                    # a locally isothermal disk: v_phi^2 = v_kep^2 - cs^2 (the
                    # thin-disk sub-keplerian rotation).
                    v_kep2 = gm / r
                    v_phi = math.sqrt(max(v_kep2 - cs2, 0.0))
                    vx = -v_phi * (y / r)
                    vy = +v_phi * (x / r)

                    yield (sigma, vx, vy, sigma * cs2)

        return gas_state

    def summary(self) -> list[tuple[str, str, str]]:
        dx = (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]
        r_acc = max(self.sink_radius, 2.0 * dx)
        prescription = "torque-free" if self.torque_free_xi > 0.0 else "standard"
        return [
            ("disk", "mach", f"{self.mach:.1f}"),
            ("disk", "sound speed", f"{self.ambient_sound_speed:.4f}"),
            ("sink", "prescription", f"{prescription} (xi = {self.torque_free_xi})"),
            ("sink", "r_acc", f"{r_acc:.4f}"),
            ("sink", "cells across r_acc", f"{r_acc / dx:.1f}"),
        ]
