# =============================================================================
# magnetized_sink.py
#
# magnetized accretion onto a resistive sink: the bondi problem (radial inflow
# onto a central gravitating accretor) carrying a weak uniform background field,
# with the central body given a localized ohmic resistivity (`MagneticSpec::
# Resistive`). the accretion flow drags the frozen-in field inward; near the sink
# the converging streamlines compress it (flux freezing would pile |B| up without
# bound at the point mass), and the body's resistivity reconnects/annihilates that
# accumulating flux -> a STEADY central field set by the advection-diffusion
# balance v_in |B| ~ eta |B| / L, so the field-annihilation region has size
# L ~ eta / v_in. larger eta => a broader, weaker central field.
#
# this replaces the earlier static field-loop-in-a-periodic-box demo, whose
# signal was contaminated by boundary wrap and the loop's own edge + center
# artifacts. here the interesting physics sits DEEP in the interior around the
# accretor while the outer boundary is a plain outflow far away, so real physics
# is cleanly separable from boundary effects, and the flow reaches a steady state.
#
# the field is kept weak (high plasma beta = 2 p / B^2 >> 1) so it is essentially
# passive: the gas follows the ordinary gamma = 5/3 bondi inflow and the field is
# a tracer of the flow the resistive sink acts on. 3D cartesian (a genuine point
# mass; a 2d sink would represent an infinite current-carrying cylinder).
#
# stability note: the resistivity REGULATES the field pileup, so the run is stable
# only once eta is large enough to diffuse flux out of the sink faster than the
# inflow drags it in (advection-diffusion balance). at very small eta the frozen-in
# field piles up near-singularly at the point mass and the step stiffens; use the
# small-vs-large eta contrast (not eta = 0) as the physical control.
#
# usage:
#  simbi run magnetized_sink.py --eta 1.0e-2   # broad annihilation region (stable)
#  simbi run magnetized_sink.py --eta 3.0e-2   # broader, weaker central field
#  simbi run magnetized_sink.py --resolution 96,96,96   # finer (cpu cost ~ n^3)
# =============================================================================
import math
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CellSpacing, CoordSystem, Regime, Solver
from simbi.types.bodies import (
    AccretionProperties,
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
    MagneticProperties,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)

# the drain rate is not a parameter: the immersed-boundary penalization surface drains at the local
# sound-crossing rate c_s/(c_drain*dx), with the convergence coefficient fixed at 1. there the
# removed mass is set by the flow reaching the sink -- the physical plateau -- rather than by a dial.


class MagnetizedBondiSink(SimbiProblem):
    """bondi accretion of a weakly magnetized gas onto a resistive point mass."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(5.0 / 3.0, description="adiabatic index")
    ]
    central_mass: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="central point-mass GM (cs = 1 units)"),
    ]
    ambient_density: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="ambient gas density at infinity"),
    ]
    beta: Annotated[
        float,
        ProblemParam(
            1.0e3,
            cli=True,
            description="ambient plasma beta = 2 p_inf / B0^2; sets the uniform "
            "background field strength. large = weak, passive field",
        ),
    ]
    eta: Annotated[
        float,
        ProblemParam(
            1.0e-2,
            cli=True,
            description="the body's ohmic resistivity. the central field-"
            "annihilation region has size ~ eta / v_infall; too small and the "
            "frozen-in flux piles up near-singularly at the point mass",
        ),
    ]

    # domain: a few bondi radii across, the accretor at the center. R_B = GM/cs^2
    # = central_mass here (cs = 1), so the default box spans +/- 4 R_B.
    domain_radius: Annotated[
        float,
        ProblemParam(
            4.0, cli=True, description="half-box size in units of the bondi radius"
        ),
    ]
    # cpu-affordable 3D default (~262k cells); scale up with --resolution on a gpu.
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((64, 64, 64), cli=True, description="grid resolution"),
    ]
    # filled by setup() from the bondi scales (base-class fields).
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(
            None, checkpoint_safe=False, description="domain boundaries (derived)"
        ),
    ]
    end_time: Annotated[
        float | None,
        ProblemParam(
            None, checkpoint_safe=True, description="simulation end time (derived)"
        ),
    ]
    checkpoint_interval: Annotated[
        float | None,
        ProblemParam(
            None, checkpoint_safe=True, description="checkpoint interval (derived)"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NMHD, description="physics regime")
    ]

    # numerics: hlld for the mhd waves; outflow far field (no periodic wrap)
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLD, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW], description="outflow on every face"
        ),
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1"),
    ]

    # accretion sink: only the mask RADIUS is a config knob. the drain RATE is not
    # exposed -- it is pinned to the saturated (sound-crossing) value so the removed
    # mass is flow-limited. thin-disk / point-mass accretion has no
    # analytic attractor (unlike bondi), so a tunable rate would be a numerical
    # boundary condition posing as a physical parameter; the sink is a hole.
    r_acc_scale: Annotated[
        float,
        ProblemParam(
            4.0,
            cli=True,
            description="sink mask radius in units of the cell size",
        ),
    ]

    # control (a handful of bondi times t_B = R_B / cs lets the inflow settle)
    total_bondi_times: Annotated[
        float,
        ProblemParam(
            8.0, cli=True, description="run length in bondi times R_B/cs"
        ),
    ]
    snapshots_per_bondi_time: Annotated[
        int, ProblemParam(4, cli=True, description="checkpoints per bondi time")
    ]
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/ibm/magnetized_sink/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]

    def setup(self) -> None:
        """derive the domain size and run control from the bondi scales."""
        super().setup()
        if self.bounds is None:
            rb = self.domain_radius * self.bondi_radius
            self.bounds = [(-rb, rb), (-rb, rb), (-rb, rb)]
        if self.end_time is None:
            self.end_time = self.total_bondi_times * self.bondi_time
        if self.checkpoint_interval is None:
            self.checkpoint_interval = self.bondi_time / float(
                self.snapshots_per_bondi_time
            )

    # ambient sound speed cs = 1 (the unit of speed); R_B = GM / cs^2 = GM.
    @property
    def ambient_sound_speed(self) -> float:
        return 1.0

    @property
    def bondi_radius(self) -> float:
        return self.central_mass / self.ambient_sound_speed**2

    @property
    def bondi_time(self) -> float:
        return self.bondi_radius / self.ambient_sound_speed

    @property
    def ambient_pressure(self) -> float:
        # p_inf = rho_inf cs^2 / gamma so that cs^2 = gamma p / rho holds.
        return (
            self.ambient_density
            * self.ambient_sound_speed**2
            / self.adiabatic_index
        )

    @property
    def b0(self) -> float:
        # uniform background field from the target plasma beta = 2 p_inf / B0^2.
        return math.sqrt(2.0 * self.ambient_pressure / self.beta)

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        """a gravitating, accreting, resistive point mass at the origin."""
        nx = self.resolution[0]
        dx = (self.bounds[0][1] - self.bounds[0][0]) / nx
        r_acc = self.r_acc_scale * dx
        softening = 0.05 * self.bondi_radius

        body = ImmersedBodyConfig(
            capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
            mass=self.central_mass,
            # the mask-gate scale for BOTH the drain and the magnetic indicator chi.
            radius=r_acc,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            gravitational=GravitationalProperties(softening_length=softening),
            accretion=AccretionProperties(
                accretion_radius=r_acc
            ),
        )
        # the localized ohmic coupling that annihilates the field threading the sink.
        return [replace(body, magnetic=MagneticProperties(resistivity=self.eta))]

    # -- initial state ------------------------------------------------------

    def initial_primitive_state(self) -> InitialStateType:
        """uniform static gas (gravity drives the inflow) threaded by a uniform
        background field B0 x_hat; the discrete uniform field is trivially div-free."""

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            rho = self.ambient_density
            pre = self.ambient_pressure
            for _ in range(nx * ny * nz):
                yield (rho, 0.0, 0.0, 0.0, pre)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            nx, ny, nz = self.resolution
            b0 = self.b0
            for _kk in range(nz + (bn == "bz")):
                for _jj in range(ny + (bn == "by")):
                    for _ii in range(nx + (bn == "bx")):
                        # a uniform field along x: bx = B0 on every x-face, by = bz = 0.
                        yield b0 if bn == "bx" else 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
