# =============================================================================
# tumbling_body.py
#
# a free, asymmetric rigid body cast into a 3D wind: it tumbles and drifts. the
# body is a flat elongated block (a card) with anisotropic principal moments, so
# Euler's gyroscopic term makes it precess/nutate; the gas reaction torque drives
# its spin and the drag pushes it downstream. everything is two-way: the flow
# moves the body (mass dv = drag) and turns it (I domega = torque), while the
# body's rotating mask + omega x r surface act back on the gas.
#
# showcases: full rigid-body rotation of an immersed wall — a runtime orientation
# matrix + angular-velocity vector, anisotropic-inertia gyroscopic tumbling, and
# two-way force- and torque-driven motion. the body's position, velocity,
# orientation-driven torque, and drag are written to diagnostics.dat.
#
# usage:
#  simbi run tumbling_body
#  simbi run tumbling_body --mach 1.2 --spin 6.0
# =============================================================================
import math
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Shape, Solver
from simbi.types.bodies import BodyCapability, ImmersedBodyConfig, RigidProperties
from simbi.types.typing import ExpressionDict, GasStateGenerator, InitialStateType

RHO_INF = 1.0
PRE_INF = 1.0


class TumblingBody(SimbiProblem):
    """a free anisotropic body tumbling + drifting in a 3D wind (two-way)."""

    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="ratio of specific heats")
    ]
    mach: Annotated[
        float, ProblemParam(0.9, cli=True, description="freestream Mach number")
    ]
    spin: Annotated[
        float,
        ProblemParam(
            5.0, cli=True, description="initial spin rate about a tilted axis (rad/time)"
        ),
    ]
    body_mass: Annotated[
        float, ProblemParam(6.0, cli=True, description="body mass (sets the drift rate)")
    ]

    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((160, 96, 96), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(-2.0, 6.0), (-2.0, 2.0), (-2.0, 2.0)], description="domain boundaries"
        ),
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
        list[BoundaryCondition],
        ProblemParam(
            [
                BoundaryCondition.DYNAMIC,  # x-inner: driven freestream inflow
                BoundaryCondition.OUTFLOW,  # x-outer
                BoundaryCondition.OUTFLOW,  # y-inner
                BoundaryCondition.OUTFLOW,  # y-outer
                BoundaryCondition.OUTFLOW,  # z-inner
                BoundaryCondition.OUTFLOW,  # z-outer
            ],
            description="left face is a driven inflow; the rest are outflow",
        ),
    ]
    cfl_number: Annotated[float, ProblemParam(0.3, description="cfl number")]

    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/ibm/tumbling_body/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(10.0, cli=True, checkpoint_safe=True, description="end time"),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(0.2, cli=True, checkpoint_safe=True, description="checkpoint interval"),
    ]
    diagnostic_interval: Annotated[
        float,
        ProblemParam(
            0.05,
            cli=True,
            checkpoint_safe=True,
            description="body-diagnostics cadence; writes the body's position, "
            "velocity, and reaction force/torque to diagnostics.dat",
        ),
    ]

    def freestream_velocity(self) -> float:
        return self.mach * math.sqrt(self.adiabatic_index * PRE_INF / RHO_INF)

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # a flat elongated card: long in x, medium in y, thinnest in z. the unequal
        # principal moments make it precess/nutate (Euler's gyroscopic term), and its
        # asymmetry lets the flow torque tumble it. the z half-extent 0.15 spans ~7
        # cells (dz = 4/96 = 0.0417): the penalization needs several cells across the
        # thinnest solid dimension, or the wall leaks and FOFC freezes the interior.
        card = Shape.box((0.0, 0.0, 0.0), (0.45, 0.22, 0.15))
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.RIGID,
                mass=self.body_mass,
                radius=1.0,  # the mask-gate scale; the CSG defines the geometry
                position=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                two_way_coupling=True,  # the flow moves + turns the body, and back
                rigid=RigidProperties(
                    inertia=1.0,
                    apply_no_slip=True,
                    k_eta_n=50.0,
                    k_eta_t=50.0,
                    shape=card,
                    omega=self.spin,
                    # spin cast about a tilted axis (not a principal axis), so the
                    # anisotropic moments drive a torque-free precession from the start.
                    spin_axis=(0.3, 1.0, 0.2),
                    # anisotropic principal moments (I1 < I2 < I3): the card tumbles.
                    inertia_principal=(1.0, 3.0, 3.8),
                ),
            )
        ]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> ExpressionDict:
        g = expr.ExprGraph()
        rho = expr.constant(RHO_INF, g)
        vx = expr.constant(self.freestream_velocity(), g)
        vy = expr.constant(0.0, g)
        vz = expr.constant(0.0, g)
        pre = expr.constant(PRE_INF, g)
        return g.compile([rho, vx, vy, vz, pre]).serialize_boundary(dim=3)

    def initial_primitive_state(self) -> InitialStateType:
        """a uniform 3D wind fills the domain; the body tumbles + drifts through it."""

        def gas_state() -> GasStateGenerator:
            nx, ny, nz = self.resolution
            v = self.freestream_velocity()
            for _kk in range(nz):
                for _jj in range(ny):
                    for _ii in range(nx):
                        yield (RHO_INF, v, 0.0, 0.0, PRE_INF)

        return gas_state
