# =============================================================================
# bondi.py
#
# spherical bondi accretion test case.
# classical spherical bondi accretion onto a single point mass.
# tests boundary conditions and should reproduce analytical bondi solution.
#
# this implementation uses a "telescoping" grid strategy:
# 1. the domain size is set by domain_radius * R_B
# 2. base_resolution controls the coarsest level (memory constraint)
# 3. target_zones_per_bondi controls finest resolution (physics constraint)
# 4. fmr levels are auto-calculated to bridge the gap
# =============================================================================
import math
import os
from pathlib import Path
from typing import Annotated

import numpy as np

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CoordSystem,
    Regime,
    Solver,
    SubCycleMode,
)
from simbi.types.bodies import (
    AccretionProperties,
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
)
from simbi.types.typing import (
    ExpressionDict,
    GasStateGenerator,
    InitialStateType,
)


class SphericalBondiTest(SimbiProblem):
    """spherical bondi accretion test case."""

    # =========================================================================
    # physical parameters
    # =========================================================================
    central_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="central black hole mass", group="physics"),
    ]
    ambient_density: Annotated[
        float,
        ProblemParam(
            1.0, cli=True, description="ambient gas density at infinity", group="physics"
        ),
    ]
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, cli=True, description="adiabatic index", group="physics")
    ]

    # =========================================================================
    # computational parameters (telescoping grid)
    # =========================================================================
    domain_radius: Annotated[
        float,
        ProblemParam(
            5.0, cli=True, description="domain radius in units of bondi radius", group="grid"
        ),
    ]
    base_resolution: Annotated[
        int,
        ProblemParam(
            128,
            cli=True,
            description="resolution per axis at coarsest level (L0)", group="grid",
        ),
    ]
    target_zones_per_bondi: Annotated[
        int,
        ProblemParam(
            64,
            cli=True,
            description="target zones per bondi radius at finest level", group="grid",
        ),
    ]

    # =========================================================================
    # initial conditions
    # =========================================================================
    use_bondi_initial_conditions: Annotated[
        bool,
        ProblemParam(
            False,
            cli=True,
            description="start with bondi flow vs uniform static gas", group="initial conditions",
        ),
    ]

    # =========================================================================
    # buffer zone parameters
    # =========================================================================
    use_buffer: Annotated[
        bool,
        ProblemParam(
            True,
            cli=True,
            description="enable buffer zone damping to background solution",
            group="buffer zone",
        ),
    ]
    buffer_time_fraction: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            description="damping timescale as fraction of bondi time",
            group="buffer zone",
        ),
    ]
    domain_fraction: Annotated[
        float,
        ProblemParam(
            0.8,
            cli=True,
            description="inner edge of buffer zone as fraction of domain radius",
            group="buffer zone",
        ),
    ]

    # =========================================================================
    # accretion parameters (the well-posed uniform-scaling DRAIN, docs/ideas/accretor.md)
    #
    # the drain scales EVERY conserved component inside the mask by exp(-drain_rate*dt),
    # drain_rate = chi * min(sink_rate, cs/dx): the intensive gas state is invariant (no
    # acoustic injection), positivity is unconditional, and the accretion rate is EMERGENT
    # (a functional of the solved flow, never a target). the emergent rate is INSENSITIVE to
    # sink_rate once it saturates the sound-crossing cap cs/dx (accretor.md §6, the plateau) --
    # the sonic surface regulates it, so long as r_acc sits inside the sonic radius r_s.
    # =========================================================================
    sink_rate: Annotated[
        float,
        ProblemParam(
            1.0e6,
            cli=True,
            description="drain rate dial (1/time); saturates at the sound-crossing "
            "rate cs/dx -> the fast drain (accretor.md C_drain=1). 0 disables accretion",
            group="accretion",
        ),
    ]
    r_acc_scale: Annotated[
        float,
        ProblemParam(
            5.0,
            cli=True,
            description="mask radius r_acc in units of finest cell size (keep r_acc < r_s)", group="accretion",
        ),
    ]

    # =========================================================================
    # output control
    # =========================================================================
    snapshots_per_bondi_time: Annotated[
        int,
        ProblemParam(
            10, cli=True, description="snapshots per bondi accretion timescale", group="output"
        ),
    ]
    total_bondi_times: Annotated[
        float,
        ProblemParam(
            20.0,
            cli=True,
            description="total simulation time in bondi timescales", group="output",
        ),
    ]

    # =========================================================================
    # computed fields (auto-filled by setup)
    # =========================================================================
    resolution: Annotated[
        tuple[int, int, int] | None,
        ProblemParam(
            None,
            cli=True,
            checkpoint_safe=False,
            description="grid resolution (calculated)",
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(
            None,
            cli=True,
            checkpoint_safe=False,
            description="domain boundaries (calculated)",
        ),
    ]
    end_time: Annotated[
        float | None,
        ProblemParam(
            None,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time (calculated)",
        ),
    ]
    checkpoint_interval: Annotated[
        float | None,
        ProblemParam(
            None,
            cli=True,
            checkpoint_safe=True,
            description="checkpoint interval (calculated)",
        ),
    ]
    data_directory: Annotated[
        Path | None,
        ProblemParam(
            None,
            cli=True,
            checkpoint_safe=True,
            description="output directory (calculated)",
        ),
    ]

    # =========================================================================
    # base configuration
    # =========================================================================
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLE, description="riemann solver")
    ]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(
            BoundaryCondition.OUTFLOW, description="boundary conditions"
        ),
    ]
    diagnostic_per_bondi: Annotated[
        float,
        ProblemParam(0.01, gt=0.0, cli=True, description="diagnostic cadence", group="output"),
    ] = 100

    # =========================================================================
    # refinement parameters (auto-filled by setup)
    # =========================================================================
    refinement_enabled: Annotated[
        bool, ProblemParam(False, cli=True, description="enable fmr")
    ]
    refinement_max_levels: Annotated[
        int, ProblemParam(0, description="max refinement levels (calculated)")
    ]
    refinement_regions: Annotated[
        list[list[float]] | None,
        ProblemParam(None, description="refinement regions (calculated)"),
    ]
    refinement_ratios: Annotated[
        list[int] | None,
        ProblemParam(None, description="refinement ratios (calculated)"),
    ]
    refinement_subcycling_mode: Annotated[
        SubCycleMode,
        ProblemParam(SubCycleMode.ADAPTIVE, description="fmr subcycling mode"),
    ]

    # =========================================================================
    # setup hook
    # =========================================================================
    def setup(self) -> None:
        """
        compute grid hierarchy from physics parameters.

        logic:
        1. domain size from domain_radius * R_B
        2. base grid from base_resolution (memory constraint)
        3. compute refinement levels to achieve target_zones_per_bondi
        4. generate telescoping refinement boxes centered on accretor
        """
        super().setup()

        # gamma = 1 IS the isothermal equation of state: the adiabatic branch
        # is degenerate there (sound speed gamma(gamma-1)e = 0 -> zero wave
        # speed -> one unbounded dt). select the genuine isothermal regime,
        # whose sound speed is `ambient_sound_speed`.
        if self.adiabatic_index == 1.0:
            self.regime = Regime.ISOTHERMAL

        R_B = self.bondi_radius
        box_radius = self.domain_radius * R_B

        # 1. domain bounds
        if self.bounds is None:
            self.bounds = [
                (-box_radius, box_radius),
                (-box_radius, box_radius),
                (-box_radius, box_radius),
            ]

        # 2. base resolution (L0)
        if self.resolution is None:
            n_base = self.base_resolution
            self.resolution = (n_base, n_base, n_base)

        # 3. compute cell sizes
        dx_coarse = (2.0 * box_radius) / self.base_resolution
        dx_target = R_B / self.target_zones_per_bondi

        # 4. calculate refinement levels needed
        total_ratio = dx_coarse / dx_target

        self.diagnostic_interval = self.bondi_time / float(
            self.diagnostic_per_bondi
        )

        if total_ratio <= 1.0:
            num_levels = 0
        else:
            num_levels = int(math.ceil(math.log2(total_ratio)))

        # 5. configure fmr hierarchy
        if self.refinement_enabled and self.refinement_regions is None:
            self.refinement_max_levels = num_levels + 1
            self.refinement_ratios = [np.uint64(2)] * num_levels

            # telescoping regions centered on the accretor: finest box covers
            # ~2 R_B, each coarser level doubles, with a HALVING CAP — each
            # region is at most half its parent's radius, so nesting margins
            # are guaranteed (a quarter of the parent box per side). without
            # the cap, deep telescopes stack domain-clamped levels against the
            # boundary with sub-cell margins and the hierarchy's coverage
            # check rejects them.
            regions = []
            r_prev = box_radius
            for ii in range(num_levels):
                levels_from_fine = (num_levels - 1) - ii
                region_radius = min(
                    2.0 * R_B * (2.0**levels_from_fine), 0.5 * r_prev
                )
                regions.append(
                    [
                        -region_radius,
                        region_radius,
                        -region_radius,
                        region_radius,
                        -region_radius,
                        region_radius,
                    ]
                )
                r_prev = region_radius
            self.refinement_regions = regions

        # 6. runtime control
        if self.end_time is None:
            self.end_time = self.total_bondi_times * self.bondi_time

        if self.checkpoint_interval is None:
            self.checkpoint_interval = self.bondi_time / float(
                self.snapshots_per_bondi_time
            )

        # 7. output directory
        if self.data_directory is None:
            scratch = os.getenv("SCRATCH", "data")
            initial_type = (
                "bondi_ic"
                if self.use_bondi_initial_conditions
                else "uniform_ic"
            )
            if self.adiabatic_index == 1.0:
                eos = "isothermal"
            elif self.adiabatic_index == 5.0 / 3.0:
                eos = "monoatomic"
            elif self.adiabatic_index == 7.0 / 5.0:
                eos = "diatomic"
            elif self.adiabatic_index == 4.0 / 3.0:
                eos = "relativistic"
            else:
                eos = "polytropic"
            ref = (
                f"fmr{self.refinement_max_levels}"
                if self.refinement_enabled
                else "uniform"
            )

            racc_tag = f"racc{self.r_acc_scale:g}dx"
            dirname = (
                f"{scratch}/bondi_test/{initial_type}/{eos}/{ref}/{racc_tag}"
            )
            os.makedirs(dirname, exist_ok=True)
            self.data_directory = Path(dirname)

    # =========================================================================
    # physics properties
    # =========================================================================
    @property
    def bondi_radius(self) -> float:
        """bondi radius: R_B = GM/cs²"""
        return self.central_mass / self.ambient_sound_speed**2

    @property
    def bondi_time(self) -> float:
        """bondi accretion timescale: t_B = R_B/cs"""
        return self.bondi_radius / self.ambient_sound_speed

    @property
    def ambient_sound_speed(self) -> float:
        """ambient sound speed (set to unity for this test)."""
        return 1.0

    def accretion_coefficient(self) -> float:
        """accretion coefficient λ based on adiabatic index."""
        if self.adiabatic_index == 1.0:
            return 1.12
        elif self.adiabatic_index == 5.0 / 3.0:
            return 0.25
        else:
            gamma = self.adiabatic_index
            return float(
                0.25
                * (2.0 / (5.0 - 3.0 * gamma))
                ** ((5.0 - 3.0 * gamma) / (2.0 * (gamma - 1.0)))
            )

    @property
    def bondi_accretion_rate(self) -> float:
        """classical bondi accretion rate: λ * 4π * ρ_∞ * R_B² * cs"""
        lambda_bondi = self.accretion_coefficient()
        return (
            lambda_bondi
            * 4.0
            * math.pi
            * self.ambient_density
            * self.bondi_radius**2
            * self.ambient_sound_speed
        )

    @property
    def buffer_parameters(self) -> dict[str, float]:
        """buffer damping parameters for outer boundary."""
        if not self.use_buffer:
            return {}

        domain_size = self.domain_radius * self.bondi_radius
        buffer_width = (1.0 - self.domain_fraction) * domain_size
        buffer_radius = domain_size - buffer_width
        damp_time = self.buffer_time_fraction * self.bondi_time

        return {
            "buffer_radius": buffer_radius,
            "buffer_width": buffer_width,
            "damp_time": damp_time,
        }

    # =========================================================================
    # buffer damping implementation
    # =========================================================================
    def buffer_sponge_terms(
        self,
        x1: expr.Expr,
        x2: expr.Expr,
        x3: expr.Expr,
    ) -> list[expr.Expr]:
        """the buffer-zone full-state SPONGE outputs [kappa, den_ref, mom_ref_x, mom_ref_y,
        mom_ref_z, nrg_ref] for the rust `sponge` source kind: relax the whole CONSERVED state
        (density, momentum, AND energy) toward the FAR-FIELD analytical Bondi solution (the ambient
        reservoir) in the outer buffer zone, with kappa ramping 0 -> 1/damp_time from buffer_radius
        outward. density is included -- the physical far field is rho_inf, which the velocity-only
        relax could not hold and the inner free-fall profile gets wrong (it decays to zero)."""
        buffer_params = self.buffer_parameters
        buffer_radius = expr.constant(buffer_params["buffer_radius"], x1.graph)
        buffer_width = expr.constant(buffer_params["buffer_width"], x1.graph)
        damp_time = expr.constant(buffer_params["damp_time"], x1.graph)

        r = expr.sqrt(x1 * x1 + x2 * x2 + x3 * x3) + 1e-10

        # smooth cubic ramp: kappa = 0 inside buffer_radius, -> 1/damp_time over buffer_width outward.
        radial_param = (r - buffer_radius) / buffer_width
        damp_factor = expr.max_expr(expr.constant(0.0, x1.graph), radial_param)
        damp_factor = expr.min_expr(damp_factor, expr.constant(1.0, x1.graph))
        damp_factor = damp_factor * damp_factor * (3.0 - 2.0 * damp_factor)
        kappa = damp_factor / damp_time

        # the FAR-FIELD ASYMPTOTIC Bondi reference (subsonic, OUTSIDE R_B -- where the outer buffer
        # sits). the O(1/r_norm) density (1 + 0.5/r_norm) and velocity (1 - 0.5/r_norm) corrections
        # CANCEL in rho*v, so the mass rate Mdot = 4*pi*r^2*rho*v is constant (the steady-state Bondi
        # flux) to O(1/r_norm) -- consistent with the far-field asymptotic rate.
        # EoS-generic: cs_eq carries the gamma-dependent enthalpy correction and p = rho*cs^2/gamma.
        # convention: r_norm uses R_B = 2GM/cs^2 (the profile), self.bondi_radius is R_B = GM/cs^2.
        cs = self.ambient_sound_speed
        rho_inf = self.ambient_density
        gamma = self.adiabatic_index
        lammy = self.accretion_coefficient()
        r_norm = r / (2.0 * self.bondi_radius)
        rho_ref = rho_inf * (1.0 + 0.5 / r_norm)
        v_r = -0.25 * lammy * cs * r_norm ** (-2.0) * (1.0 - 0.5 / r_norm)
        cs_eq = cs * (1.0 + 0.25 * (gamma - 1.0) / r_norm)
        p_ref = rho_ref * cs_eq * cs_eq / gamma

        # cartesian reference momentum (rho_ref * v_r * x_hat) and total energy density. |v|^2 = v_r^2
        # since the reference flow is purely radial; nrg = p/(gamma-1) + (1/2)*rho*|v|^2.
        r_inv = 1.0 / r
        mom_x = rho_ref * v_r * x1 * r_inv
        mom_y = rho_ref * v_r * x2 * r_inv
        mom_z = rho_ref * v_r * x3 * r_inv
        inv_gm1 = 1.0 / (gamma - 1.0) if gamma != 1.0 else 0.0
        nrg_ref = p_ref * inv_gm1 + 0.5 * rho_ref * v_r * v_r

        # kappa (output 0) is the only masked channel; it is already zoned by the cubic ramp, so no
        # separate region mask is needed. rust `sponge` forms S_U = kappa*(U_ref - U) for each of den,
        # mom, nrg -- the density channel is what the old velocity-only relax lacked.
        # the ISOTHERMAL regime has no energy equation: its sponge spec takes
        # [kappa, den_ref, mom_ref_*] only (5 outputs, no nrg_ref).
        if self.adiabatic_index == 1.0:
            return [kappa, rho_ref, mom_x, mom_y, mom_z]
        return [kappa, rho_ref, mom_x, mom_y, mom_z, nrg_ref]

    @property
    def source_expressions(self) -> list[ExpressionDict]:
        """the outer buffer-zone full-state sponge as a rust `sponge` source (or no source when
        disabled). params=[inv_gm1] = 1/(gamma-1) lets the energy channel reconstruct the conserved
        total energy from pressure. the well-posed (gamma < 5/3) Bondi test is regulated by the sonic
        surface, so the sponge is an outer-boundary nicety, not load-bearing."""
        if not self.use_buffer:
            return []
        graph = expr.ExprGraph()
        x1 = expr.variable("x1", graph)
        x2 = expr.variable("x2", graph)
        x3 = expr.variable("x3", graph)
        outputs = self.buffer_sponge_terms(x1, x2, x3)
        gamma = self.adiabatic_index
        inv_gm1 = 1.0 / (gamma - 1.0) if gamma != 1.0 else 0.0
        return [
            graph.compile(outputs).serialize_source(
                expr.SourceKind.SPONGE, dim=3, params=[inv_gm1]
            )
        ]

    # =========================================================================
    # analytical solution
    # =========================================================================
    def bondi_solution(self, r: float) -> tuple[float, float, float]:
        """analytical bondi solution at radius r: (density, radial_velocity, pressure)"""
        if r <= 0:
            return (
                self.ambient_density,
                0.0,
                self.ambient_density * self.ambient_sound_speed**2,
            )

        xi = r / self.bondi_radius

        if xi > 1.0:
            rho_ratio = (xi / 2.0) ** (-1.5)
            v_r = -self.ambient_sound_speed * math.sqrt(2.0 / xi)
        else:
            rho_ratio = 1.0 / xi**1.5
            v_r = -self.ambient_sound_speed

        rho = self.ambient_density * rho_ratio

        if self.adiabatic_index == 1.0:
            pressure = self.ambient_density * self.ambient_sound_speed**2
        else:
            pressure = (
                self.ambient_density
                * self.ambient_sound_speed**2
                * rho_ratio ** (self.adiabatic_index)
            )

        return (rho, v_r, pressure)

    # =========================================================================
    # initial conditions
    # =========================================================================
    def initial_primitive_state(self) -> InitialStateType:
        """initial conditions: uniform static gas or bondi solution."""

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]
            zmin, zmax = self.bounds[2]

            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny
            dz = (zmax - zmin) / nz

            for kk in range(nz):
                z = zmin + (kk + 0.5) * dz
                for jj in range(ny):
                    y = ymin + (jj + 0.5) * dy
                    for ii in range(nx):
                        x = xmin + (ii + 0.5) * dx

                        r = math.sqrt(x * x + y * y + z * z)

                        if self.use_bondi_initial_conditions and r > 0:
                            rho, v_r, pressure = self.bondi_solution(r)
                            vx = v_r * x / r
                            vy = v_r * y / r
                            vz = v_r * z / r
                        else:
                            # static initial conditions with zero velocity
                            vx = vy = vz = 0.0

                            if self.adiabatic_index == 1.0:
                                # isothermal: uniform density, uniform pressure
                                rho = self.ambient_density
                                pressure = (
                                    self.ambient_density
                                    * self.ambient_sound_speed**2
                                )
                            else:
                                # adiabatic: hydrostatic equilibrium with point mass
                                # dP/dr = -rho * GM/r^2
                                # for polytropic EOS: P = K * rho^gamma
                                # solution: rho = rho_inf * (1 + (g-1)*GM/(cs^2*r))^(1/(g-1))
                                gamma = self.adiabatic_index
                                cs2 = self.ambient_sound_speed**2
                                r_safe = max(r, 1e-10 * self.bondi_radius)

                                # dimensionless potential depth
                                phi_factor = (
                                    (gamma - 1.0)
                                    * self.central_mass
                                    / (cs2 * r_safe)
                                )
                                rho = self.ambient_density * (
                                    1.0 + phi_factor
                                ) ** (1.0 / (gamma - 1.0))

                                # polytropic pressure: P/P_inf = (rho/rho_inf)^gamma
                                p_inf = self.ambient_density * cs2 / gamma
                                rho_ratio = rho / self.ambient_density
                                pressure = p_inf * (rho_ratio**gamma)

                        yield (rho, vx, vy, vz, pressure)

        return gas_state

    # =========================================================================
    # immersed bodies
    # =========================================================================
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        """single point mass with softened gravity + the well-posed uniform-scaling drain."""
        # compute finest cell size
        dx_coarse = (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]
        dx_fine = dx_coarse

        if self.refinement_enabled and self.refinement_ratios:
            for ratio in self.refinement_ratios:
                dx_fine /= ratio

        # mask radius (drain) + gravitational softening at the finest level.
        softening = 0.01 * self.bondi_radius
        r_acc = self.r_acc_scale * dx_fine

        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION
                | BodyCapability.GRAVITATIONAL,
                mass=self.central_mass,
                radius=0.0,
                position=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                gravitational=GravitationalProperties(
                    softening_length=softening
                ),
                # the drain: mask radius r_acc + the rate dial sink_rate (saturating at the
                # sound-crossing rate). the emergent Mdot validates against 4 pi lambda_c(gamma).
                accretion=AccretionProperties(
                    accretion_radius=r_acc,
                    sink_rate=self.sink_rate,
                ),
            )
        ]

    # =========================================================================
    # summary (derived quantities for the dashboard's problem-setup panel)
    # =========================================================================
    def summary(self) -> list[tuple[str, str, str]]:
        """the derived quantities: the bondi scales, the expected rate, and
        the grid facts the declared dials imply."""
        if self.bounds is None or self.resolution is None:
            return []
        rows = [
            ("derived", "bondi radius", f"{self.bondi_radius:.4f}"),
            ("derived", "bondi time", f"{self.bondi_time:.4f}"),
            ("derived", "expected mdot", f"{self.bondi_accretion_rate:.6f}"),
        ]
        dx_coarse = (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]
        dx_fine = dx_coarse / (2 ** max(0, self.refinement_max_levels - 1))
        rows.append(("derived", "coarse dx", f"{dx_coarse:.4f}"))
        if self.refinement_enabled:
            rows.append(
                (
                    "derived",
                    "finest dx",
                    f"{dx_fine:.5f} ({self.bondi_radius / dx_fine:.1f} zones/R_B)",
                )
            )
        for key, value in self.buffer_parameters.items():
            rows.append(("buffer (derived)", key.replace("_", " "), f"{value:.3f}"))
        return rows
