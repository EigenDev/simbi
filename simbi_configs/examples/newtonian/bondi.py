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
from simbi.functional import bondi as bondi_shared
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
    # sponge zone parameters
    # =========================================================================
    use_sponge: Annotated[
        bool,
        ProblemParam(
            True,
            cli=True,
            description="enable the outer sponge zone relaxing to the background",
            group="sponge zone",
            deprecated_names=["use_buffer"],
        ),
    ]
    sponge_time_fraction: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            description="sponge damping timescale as a fraction of the bondi time",
            group="sponge zone",
            deprecated_names=["buffer_time_fraction"],
        ),
    ]
    domain_fraction: Annotated[
        float,
        ProblemParam(
            0.8,
            cli=True,
            description="inner edge of the sponge zone as a fraction of the domain radius",
            group="sponge zone",
        ),
    ]

    # =========================================================================
    # accretion (the well-posed uniform-scaling drain, docs/ideas/accretor.md)
    #
    # the drain scales every conserved component inside the mask by exp(-dt/tau) with
    # tau = c_drain * dx / c_s, the local sound-crossing time: the intensive gas state is invariant
    # (no acoustic injection), positivity is unconditional, and the accretion rate is emergent -- a
    # functional of the solved flow, never a target. the sonic surface regulates it, so long as
    # r_acc sits inside the sonic radius r_s.
    #
    # there is no rate dial. the penalization surface owns accretion and c_drain is fixed at 1
    # (accretor.md C_drain = 1, the fast drain), which is the saturated / plateau case this setup
    # wants. the dial lives on the substrate kernel set and is not exposed to the configuration
    # layer.
    # =========================================================================
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
        ProblemParam(
            SubCycleMode.STANDARD,
            description="fmr subcycling: level l advances 2^l times per root step, the only "
            "implemented schedule (an adaptive per-level substep count is not)",
        ),
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

        # gamma = 1 is the isothermal equation of state: the adiabatic branch
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
        num_levels = bondi_shared.refinement_levels(dx_coarse, dx_target)

        self.diagnostic_interval = self.bondi_time / float(
            self.diagnostic_per_bondi
        )

        # 5. configure fmr hierarchy: telescoping halving-cap boxes centered on
        # the accretor, finest covering ~2 R_B (one text with the science config).
        if self.refinement_enabled and self.refinement_regions is None:
            self.refinement_max_levels = num_levels + 1
            self.refinement_ratios = [np.uint64(2)] * num_levels
            self.refinement_regions = bondi_shared.telescoping_regions(
                num_levels, finest_radius=2.0 * R_B, box_radius=box_radius
            )

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
        """bondi radius: R_B = GM/cs^2"""
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
        """bondi coefficient lambda_c(gamma) -- the shared transcription."""
        return bondi_shared.accretion_coefficient(self.adiabatic_index)

    @property
    def bondi_accretion_rate(self) -> float:
        """classical bondi accretion rate: lambda * 4 pi * rho_inf * R_B^2 * cs"""
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
    def sponge_parameters(self) -> dict[str, float]:
        """sponge damping parameters for outer boundary."""
        if not self.use_sponge:
            return {}

        domain_size = self.domain_radius * self.bondi_radius
        buffer_width = (1.0 - self.domain_fraction) * domain_size
        buffer_radius = domain_size - buffer_width
        damp_time = self.sponge_time_fraction * self.bondi_time

        return {
            "buffer_radius": buffer_radius,
            "buffer_width": buffer_width,
            "damp_time": damp_time,
        }

    # =========================================================================
    # sponge damping implementation
    # =========================================================================
    def sponge_terms(
        self,
        x1: expr.Expr,
        x2: expr.Expr,
        x3: expr.Expr,
    ) -> list[expr.Expr]:
        """the outer sponge zone as the rust `sponge` source's outputs
        [kappa, rho_ref, vel_ref_*, pre_ref] (the isothermal regime stops before
        pre_ref), relaxing toward the far-field asymptotic bondi state. one
        shared text: `simbi.functional.bondi.far_field_sponge_outputs`."""
        buffer_params = self.sponge_parameters
        return bondi_shared.far_field_sponge_outputs(
            x1,
            x2,
            x3,
            onset_radius=buffer_params["buffer_radius"],
            width=buffer_params["buffer_width"],
            damp_time=buffer_params["damp_time"],
            bondi_radius=self.bondi_radius,
            density=self.ambient_density,
            sound_speed=self.ambient_sound_speed,
            gamma=self.adiabatic_index,
        )

    @property
    def source_expressions(self) -> list[ExpressionDict]:
        """the outer sponge zone full-state sponge as a rust `sponge` source (or no source when
        disabled). the reference travels as primitives and the regime
        converts it, so no closure parameter rides the source. the well-posed (gamma < 5/3) Bondi test is regulated by the sonic
        surface, so the sponge is an outer-boundary nicety, not load-bearing."""
        if not self.use_sponge:
            return []
        outputs = self.sponge_terms(*expr.coords(3))
        return [expr.sponge(outputs, dim=3)]

    # =========================================================================
    # analytical solution
    # =========================================================================
    def bondi_solution(self, r: float) -> tuple[float, float, float]:
        """analytical bondi profile at radius r: (density, radial velocity,
        pressure). one shared text: `simbi.functional.bondi.bondi_profile`."""
        return bondi_shared.bondi_profile(
            r,
            bondi_radius=self.bondi_radius,
            density=self.ambient_density,
            sound_speed=self.ambient_sound_speed,
            gamma=self.adiabatic_index,
        )

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
                # the drain: the mask radius is the only parameter. the surface drains at the local
                # sound-crossing rate, and the emergent Mdot validates against 4 pi lambda_c(gamma).
                accretion=AccretionProperties(accretion_radius=r_acc),
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
        for key, value in self.sponge_parameters.items():
            rows.append(("sponge (derived)", key.replace("_", " "), f"{value:.3f}"))
        return rows
