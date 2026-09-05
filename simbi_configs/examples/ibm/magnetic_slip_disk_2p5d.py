# =============================================================================
# magnetic_slip_disk_2p5d.py
#
# a vertically invariant 2.5D model of gas with a vertical magnetic field falling
# onto a point mass through a cylindrical sink region wearing the magnetic slip
# (`MagneticSpec::Slip`). the x-y plane carries all three vector components: the
# in-plane field B_x, B_y lives on staggered faces and evolves by constrained
# transport, and the vertical field B_z lives at cell centers and evolves in flux
# form. a uniform B_z threads the plane at the start; the inflow drags that flux
# toward the sink and compresses it, and the slip shell releases the compressed
# part relative to the draining gas while a force-free field is left untouched.
# both channels take part: the in-plane current from the vertical field's
# gradients (d_y B_z, -d_x B_z) and the vertical current from the in-plane field,
# so the released flux and its heat carry the vertical channel, which dominates
# for a vertically magnetized disk. the coefficient
#
#   a_B = ell_B^2 / ((|B|^2 + B_0^2) D_B tau_rho),   ell_B = slip_length_ratio w
#
# closes on the sink's own drain time tau_rho through the magnetic Damkohler number
# D_B = diffusivity_ratio. the removal region is a cylinder along the missing axis
# and the prescribed gravity is that of a planar point mass, so the model is a
# vertically invariant plane with cylindrical sink regions, the reduced form a
# circumbinary calculation builds on.
#
# the ideal-MHD transport is the UCT edge EMF with the two-wave HLLE solver, a
# choice made for regularity rather than sharpness. the default Contact EMF
# upwinds its corner terms by the sign of the mass flux, which switches wherever
# that flux crosses zero and holds the in-plane field at first order in time on
# flows with such crossings. under UCT the choice of solver still matters once a
# vertical field is present: HLLD's fan has degenerate branches where the normal
# field crosses zero under a rotated tangential field, and on a shear flow with
# vertical flux its in-plane convergence is irregular; that regime is exactly
# this model's. UCT with HLLE carries neither switch and reads second order in
# every field on that fixture, at the cost of more diffusion. HLLD remains
# available through --solver hlld as an opt-in sharp solver until its B_n -> 0
# branch is regularized and gated on its own.
#
# parameters (code units, magnetic energy |B|^2 / 2):
#   diffusivity_ratio    D_B > 0: tau_B / tau_rho.
#   shell_cells          the slip shell's mollification width w in cell sizes.
#   slip_length_ratio    ell_B / w > 0; 1 is the sharp-interface scaling.
#   regularization       B_0 / B_ambient > 0: bounds the slip speed at nulls.
#   placement            shell center in shell widths from the mass surface.
#   beta                 ambient plasma beta = 2 p_inf / B0^2 of the vertical field.
#
# usage:
#  simbi run magnetic_slip_disk_2p5d.py
#  simbi run magnetic_slip_disk_2p5d.py --beta 10 --diffusivity-ratio 4
#  simbi run magnetic_slip_disk_2p5d.py --resolution 256,256,1
#  simbi run magnetic_slip_disk_2p5d.py --solver hlld        # sharper, see the caveat above
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
from simbi.types.input import CtMethod
from simbi.types.typing import (
    GasStateGenerator,
    InitialStateType,
    StaggeredBFieldGenerator,
)


class MagneticSlipDisk2p5d(SimbiProblem):
    """vertically magnetized gas falling onto a point mass through a cylindrical slip sink, in the plane."""

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
            1.0e2,
            cli=True,
            description="ambient plasma beta = 2 p_inf / B0^2 of the uniform vertical field",
        ),
    ]
    diffusivity_ratio: Annotated[
        float,
        ProblemParam(2.0, cli=True, description="magnetic Damkohler number D_B = tau_B / tau_rho"),
    ]
    shell_cells: Annotated[
        float,
        ProblemParam(2.0, cli=True, description="slip-shell mollification width in cell sizes"),
    ]
    slip_length_ratio: Annotated[
        float,
        ProblemParam(1.0, cli=True, description="transport length over shell width, ell_B / w"),
    ]
    regularization: Annotated[
        float,
        ProblemParam(0.1, cli=True, description="null regularization B_0 as a fraction of the ambient field"),
    ]
    placement: Annotated[
        float,
        ProblemParam(0.0, cli=True, description="shell center in shell widths from the mass surface"),
    ]
    domain_radius: Annotated[
        float,
        ProblemParam(4.0, cli=True, description="half-box size in units of the bondi radius"),
    ]
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((128, 128, 1), cli=True, description="grid resolution (x, y, 1)"),
    ]
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(None, checkpoint_safe=False, description="domain boundaries (derived)"),
    ]
    end_time: Annotated[
        float | None,
        ProblemParam(None, checkpoint_safe=True, description="simulation end time (derived)"),
    ]
    checkpoint_interval: Annotated[
        float | None,
        ProblemParam(None, checkpoint_safe=True, description="checkpoint interval (derived)"),
    ]
    coord_system: Annotated[
        CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="coordinate system")
    ]
    regime: Annotated[Regime, ProblemParam(Regime.NMHD, description="physics regime")]
    solver: Annotated[
        Solver,
        ProblemParam(Solver.HLLE, description="riemann solver (hlle for regularity; hlld opt-in)"),
    ]
    ct_method: Annotated[
        CtMethod, ProblemParam(CtMethod.UCT, description="constrained-transport edge EMF")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW], description="outflow on every side"),
    ]
    x1_spacing: Annotated[
        CellSpacing, ProblemParam(CellSpacing.LINEAR, description="grid spacing in x1")
    ]
    r_acc_scale: Annotated[
        float,
        ProblemParam(4.0, cli=True, description="sink mask radius in units of the cell size"),
    ]
    total_bondi_times: Annotated[
        float, ProblemParam(8.0, cli=True, description="run length in bondi times R_B/cs")
    ]
    snapshots_per_bondi_time: Annotated[
        int, ProblemParam(4, cli=True, description="checkpoints per bondi time")
    ]
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/ibm/magnetic_slip_disk_2p5d/"),
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
            self.bounds = [(-rb, rb), (-rb, rb)]
        if self.end_time is None:
            self.end_time = self.total_bondi_times * self.bondi_time
        if self.checkpoint_interval is None:
            self.checkpoint_interval = self.bondi_time / float(self.snapshots_per_bondi_time)

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
        return self.ambient_density * self.ambient_sound_speed**2 / self.adiabatic_index

    @property
    def b0(self) -> float:
        # the uniform vertical field from the target plasma beta = 2 p_inf / B0^2.
        return math.sqrt(2.0 * self.ambient_pressure / self.beta)

    @property
    def cell_size(self) -> float:
        return (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        """a gravitating, accreting point mass at the origin wearing a magnetic-slip shell."""
        dx = self.cell_size
        r_acc = self.r_acc_scale * dx
        softening = 0.05 * self.bondi_radius
        body = ImmersedBodyConfig(
            capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
            mass=self.central_mass,
            radius=r_acc,
            position=(0.0, 0.0),
            velocity=(0.0, 0.0),
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

    def initial_primitive_state(self) -> InitialStateType:
        """uniform static gas threaded by a uniform vertical field B0 z_hat."""

        def gas_state() -> GasStateGenerator:
            nx, ny, _ = self.resolution
            rho = self.ambient_density
            pre = self.ambient_pressure
            for _ in range(nx * ny):
                yield (rho, 0.0, 0.0, 0.0, pre)

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            nx, ny, _ = self.resolution
            b0 = self.b0
            for _kk in range(1 + (bn == "bz")):
                for _jj in range(ny + (bn == "by")):
                    for _ii in range(nx + (bn == "bx")):
                        yield b0 if bn == "bz" else 0.0

        return (
            gas_state,
            partial(b_field, "bx"),
            partial(b_field, "by"),
            partial(b_field, "bz"),
        )
