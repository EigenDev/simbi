# =============================================================================
# spinning_rotor.py
#
# a rigid cross-shaped rotor spinning in initially-still gas: the immersed wall
# rotates its own mask by R(omega t) and its no-slip surface drags the gas at
# omega x r, so a rotating flow develops around the blades — a stirred tank. the
# rotor is a CSG union of two crossed boxes (a plus sign), spun about z at a
# prescribed rate.
#
# showcases: DYNAMIC rotation of an arbitrary-shape immersed wall (the mask + its
# SDF-gradient normal track the spin) and the omega x r surface velocity that
# spins the fluid up. the reaction torque is written to diagnostics.dat.
#
# usage:
#  simbi run spinning_rotor --omega 4.0
#  simbi run spinning_rotor --omega -6.0 --no-slip False  # free-slip, reversed
# =============================================================================
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Shape, Solver
from simbi.types.bodies import BodyCapability, ImmersedBodyConfig, RigidProperties
from simbi.types.typing import GasStateGenerator, InitialStateType

RHO_0 = 1.0
PRE_0 = 1.0


class SpinningRotor(SimbiProblem):
    """a rigid cross rotor stirring initially-still gas (prescribed spin)."""

    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="ratio of specific heats")
    ]
    omega: Annotated[
        float,
        ProblemParam(4.0, cli=True, description="rotor spin rate about z (rad/time)"),
    ]
    blade_length: Annotated[
        float, ProblemParam(0.6, cli=True, description="rotor blade half-length")
    ]
    blade_width: Annotated[
        float, ProblemParam(0.12, description="rotor blade half-width")
    ]
    no_slip: Annotated[
        bool,
        ProblemParam(True, cli=True, description="no-slip (True) vs free-slip wall"),
    ]

    resolution: Annotated[
        tuple[int, int],
        ProblemParam((384, 384), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-2.0, 2.0), (-2.0, 2.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime (adiabatic)")
    ]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="solver")]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(BoundaryCondition.OUTFLOW, description="boundary conditions"),
    ]
    cfl_number: Annotated[float, ProblemParam(0.3, description="cfl number")]

    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/ibm/spinning_rotor/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(8.0, cli=True, checkpoint_safe=True, description="end time"),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(0.2, cli=True, checkpoint_safe=True, description="checkpoint interval"),
    ]
    diagnostic_interval: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            checkpoint_safe=True,
            description="body-diagnostics cadence; writes the rotor's reaction "
            "torque to diagnostics.dat",
        ),
    ]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # a plus-sign rotor: two crossed boxes. the z half-extent spans the plane.
        length, width = self.blade_length, self.blade_width
        blade_x = Shape.box((0.0, 0.0, 0.0), (length, width, 1.0))
        blade_y = Shape.box((0.0, 0.0, 0.0), (width, length, 1.0))
        rotor = blade_x.union(blade_y)
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.RIGID,
                mass=0.0,  # a driven rotor on a fixed axle (prescribed spin)
                radius=2.0 * length,  # the mask-gate scale; the CSG is the geometry
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                rigid=RigidProperties(
                    inertia=1.0,
                    apply_no_slip=self.no_slip,
                    k_eta_n=50.0,
                    k_eta_t=50.0,
                    shape=rotor,
                    omega=self.omega,  # prescribed spin about z (default spin_axis)
                ),
            )
        ]

    def initial_primitive_state(self) -> InitialStateType:
        """uniform gas at rest; the spinning rotor stirs it up."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            for _jj in range(ny):
                for _ii in range(nx):
                    yield (RHO_0, 0.0, 0.0, PRE_0)

        return gas_state
