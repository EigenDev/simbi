# =============================================================================
# magnetic_slip_sink.py
#
# magnetized accretion onto a point mass with the force-selective magnetic slip
# (`MagneticSpec::Slip`): the bondi problem (radial inflow onto a central
# gravitating accretor) carrying a weak uniform background field, with the sink's
# shell letting the field slip relative to the draining gas. the inflow drags the
# frozen-in field toward the sink and compresses it there; flux freezing alone
# would pile |B| up without bound at the point mass. in the slip shell the field
# is transported relative to the matter by the Lorentz-driven slip velocity, so
# the compressed (non-force-free) part of the field is released back outward and
# its magnetic energy heats the gas, while a force-free field is left exactly
# untouched. the model carries no resistivity dial: the slip coefficient
#
#   a_B = ell_B^2 / ((|B|^2 + B_0^2) D_B tau_rho),   ell_B = slip_length_ratio w
#
# closes on the sink's own drain time tau_rho through the magnetic Damkohler
# number D_B = diffusivity_ratio, so the field decouples on a multiple of the
# accretion time. the operator is solenoidal (it enters through the constrained-
# transport curl only) and dissipative, and the drain, the slip, and the ideal-MHD
# step advance together in the palindromic D-M-H-M-D coupled step.
#
# parameters (code units, magnetic energy |B|^2 / 2):
#   diffusivity_ratio    D_B > 0: tau_B / tau_rho. near 1 releases flux on the
#                        accretion time; larger holds the field in longer.
#   shell_cells          the slip shell's mollification width w in cell sizes;
#                        the shell must be resolved (two or more cells).
#   slip_length_ratio    ell_B / w > 0; 1 is the sharp-interface scaling, kept free
#                        so a convergence study can test it.
#   regularization       B_0 / B_ambient > 0: bounds the slip speed at magnetic nulls.
#   placement            shell center in shell widths relative to the mass surface:
#                        negative inside, 0 centered on it, positive outside.
#
# the field is kept weak (plasma beta = 2 p / B^2 >> 1) so the gas follows the
# ordinary gamma = 5/3 bondi inflow and the field traces the flow the sink acts on.
# 3D cartesian, adiabatic Newtonian MHD, cpu: the slip solve is an implicit
# midpoint on the host.
#
# usage:
#  simbi run magnetic_slip_sink.py                          # D_B = 2, a two-cell shell
#  simbi run magnetic_slip_sink.py --diffusivity-ratio 8    # field held in longer
#  simbi run magnetic_slip_sink.py --placement -0.5         # shell half a width inside
#  simbi run magnetic_slip_sink.py --resolution 96,96,96    # finer (cpu cost ~ n^3)
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
    MagneticSlipProperties,
)
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)


class MagneticSlipBondiSink(SimbiProblem):
    """bondi accretion of a weakly magnetized gas onto a point mass with a magnetic-slip shell."""

    # the thermal closure the slip operator runs under: an adiabatic gas keeps the released
    # magnetic energy as heat (Newtonian MHD); an isothermal gas exports it to the cooling bath
    # at once (isothermal MHD), the body booking it as `exported_slip_heat`.
    thermal_closure: Annotated[
        str,
        ProblemParam(
            "adiabatic",
            cli=True,
            description="thermal closure: 'adiabatic' (regime nmhd, heat retained) or "
            "'isothermal' (regime imhd, heat exported)",
        ),
    ]
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

    # the magnetic-slip shell
    diffusivity_ratio: Annotated[
        float,
        ProblemParam(
            2.0,
            cli=True,
            description="magnetic Damkohler number D_B = tau_B / tau_rho: the field "
            "decouples from the draining gas this many drain times",
        ),
    ]
    shell_cells: Annotated[
        float,
        ProblemParam(
            2.0,
            cli=True,
            description="slip-shell mollification width in cell sizes (resolve it: >= 2)",
        ),
    ]
    slip_length_ratio: Annotated[
        float,
        ProblemParam(
            1.0,
            cli=True,
            description="transport length over shell width, ell_B / w (1 = sharp interface)",
        ),
    ]
    regularization: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            description="null regularization B_0 as a fraction of the ambient field",
        ),
    ]
    placement: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            description="shell center in shell widths from the mass surface "
            "(negative inside, positive outside)",
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
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((64, 64, 64), cli=True, description="grid resolution"),
    ]
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

    # the sink: only the mask radius is a knob. the drain rate is pinned to the
    # local signal-crossing value so the removed mass is flow-limited, and the slip
    # shell rides on that drain's timescale.
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
            Path("data/ibm/magnetic_slip_sink/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]

    def setup(self) -> None:
        """derive the domain size and run control from the bondi scales, and the regime from the
        thermal closure."""
        if self.thermal_closure not in ("adiabatic", "isothermal"):
            raise ValueError(
                f"thermal_closure={self.thermal_closure!r}; choose 'adiabatic' or 'isothermal'"
            )
        if self.isothermal_closure:
            self.regime = Regime.IMHD
            self.sound_speed = self.ambient_sound_speed
        else:
            self.regime = Regime.NMHD
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
    def isothermal_closure(self) -> bool:
        return self.thermal_closure == "isothermal"

    @property
    def ambient_pressure(self) -> float:
        # p_inf = rho_inf cs^2 / gamma so that cs^2 = gamma p / rho holds on the adiabatic gas;
        # p = rho cs^2 on the isothermal one.
        gamma = 1.0 if self.isothermal_closure else self.adiabatic_index
        return self.ambient_density * self.ambient_sound_speed**2 / gamma

    @property
    def b0(self) -> float:
        # uniform background field from the target plasma beta = 2 p_inf / B0^2.
        return math.sqrt(2.0 * self.ambient_pressure / self.beta)

    @property
    def cell_size(self) -> float:
        """the finest level's cell size: the sink and its shell are sized on the level that
        owns them."""
        dx = (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]
        if self.refinement_enabled:
            for ratio in self.refinement_ratios:
                dx /= float(ratio)
        return dx

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        """a gravitating, accreting point mass at the origin wearing a magnetic-slip shell."""
        dx = self.cell_size
        r_acc = self.r_acc_scale * dx
        softening = 0.05 * self.bondi_radius

        body = ImmersedBodyConfig(
            capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
            mass=self.central_mass,
            # the mask-gate scale for both the drain and the slip shell.
            radius=r_acc,
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            gravitational=GravitationalProperties(softening_length=softening),
            accretion=AccretionProperties(accretion_radius=r_acc),
        )
        slip = MagneticSlipProperties(
            diffusivity_ratio=self.diffusivity_ratio,
            shell_width=self.shell_cells * dx,
            field_regularization=self.regularization * self.b0,
            slip_length_ratio=self.slip_length_ratio,
            placement=self.placement,
        )
        return [replace(body, magnetic=slip)]

    # -- initial state ------------------------------------------------------

    def initial_primitive_state(self) -> InitialStateType:
        """uniform static gas (gravity drives the inflow) threaded by a uniform
        background field B0 x_hat; the discrete uniform field is trivially div-free."""

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            rho = self.ambient_density
            pre = self.ambient_pressure
            # the isothermal primitive carries no pressure slot: p = cs^2 rho is the closure.
            state = (rho, 0.0, 0.0, 0.0) if self.isothermal_closure else (rho, 0.0, 0.0, 0.0, pre)
            for _ in range(nx * ny * nz):
                yield state

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
