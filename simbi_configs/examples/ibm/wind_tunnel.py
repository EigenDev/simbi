# =============================================================================
# wind_tunnel.py
#
# supersonic flow past a RIGID immersed obstacle of arbitrary (CSG) shape: a
# classic wind tunnel. a Mach-2 freestream is driven in through the left face
# and meets a rounded bluff body (a box fused with a spherical nose), forming a
# detached bow shock and a low-pressure wake. the obstacle is a no-slip wall
# whose mask + surface normal come from the signed-distance CSG, penalized by
# the design-50 porous stack with the drain channel sealed (porosity 0).
#
# showcases: arbitrary-shape rigid boundaries (Shape CSG), the no-slip wall, and
# a driven-boundary freestream. the body's drag is written to diagnostics.dat.
#
# usage:
#  simbi run wind_tunnel --mach 2.0                 # detached bow shock
#  simbi run wind_tunnel --mach 0.6 --no-slip False # subsonic, free-slip wall
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


class WindTunnel(SimbiProblem):
    """supersonic flow past a rigid CSG obstacle -> a bow shock."""

    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="ratio of specific heats")
    ]
    mach: Annotated[
        float,
        ProblemParam(2.0, cli=True, description="freestream Mach number v_inf / c_s"),
    ]
    obstacle_size: Annotated[
        float,
        ProblemParam(0.4, cli=True, description="obstacle half-size (code units)"),
    ]
    no_slip: Annotated[
        bool,
        ProblemParam(
            True,
            cli=True,
            description="no-slip wall (True) vs free-slip (False, the tangential "
            "channel is switched off)",
        ),
    ]

    # domain: a tunnel, longer than it is tall, obstacle in the upstream third
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((640, 256), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-2.0, 6.0), (-2.0, 2.0)], description="domain boundaries"),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime (adiabatic)")
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLC, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [
                BoundaryCondition.DYNAMIC,  # x-inner: driven freestream inflow
                BoundaryCondition.OUTFLOW,  # x-outer: downstream outflow
                BoundaryCondition.OUTFLOW,  # y-inner
                BoundaryCondition.OUTFLOW,  # y-outer
            ],
            description="left face is a driven inflow; the rest are outflow",
        ),
    ]
    cfl_number: Annotated[float, ProblemParam(0.3, description="cfl number")]

    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/ibm/wind_tunnel/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(12.0, cli=True, checkpoint_safe=True, description="end time"),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(0.25, cli=True, checkpoint_safe=True, description="checkpoint interval"),
    ]
    diagnostic_interval: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            checkpoint_safe=True,
            description="body-diagnostics cadence; writes the obstacle's drag "
            "force to diagnostics.dat",
        ),
    ]

    def freestream_velocity(self) -> float:
        # v_inf = mach * c_s, c_s = sqrt(gamma p / rho) at the freestream state.
        return self.mach * math.sqrt(self.adiabatic_index * PRE_INF / RHO_INF)

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # a rounded bluff body: a rectangular block fused with a spherical nose on
        # its upstream (-x) face. the z half-extent spans the 2D plane.
        s = self.obstacle_size
        shape = Shape.box((0.0, 0.0, 0.0), (s, s, 1.0)).union(
            Shape.sphere((-s, 0.0, 0.0), 0.9 * s)
        )
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.RIGID,
                mass=0.0,  # a fixed obstacle (not two-way)
                radius=2.0 * s,  # the mask-gate scale; the CSG defines the geometry
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                rigid=RigidProperties(
                    inertia=1.0,
                    apply_no_slip=self.no_slip,
                    k_eta_n=50.0,
                    k_eta_t=50.0,
                    shape=shape,
                ),
            )
        ]

    @computed_field
    @property
    def bx1_inner_expressions(self) -> ExpressionDict:
        # drive the left face with the uniform supersonic freestream.
        g = expr.ExprGraph()
        rho = expr.constant(RHO_INF, g)
        vx = expr.constant(self.freestream_velocity(), g)
        vy = expr.constant(0.0, g)
        pre = expr.constant(PRE_INF, g)
        return g.compile([rho, vx, vy, pre]).serialize_boundary(dim=2)

    def initial_primitive_state(self) -> InitialStateType:
        """a uniform freestream fills the tunnel; the obstacle carves the shock."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            v = self.freestream_velocity()
            for _jj in range(ny):
                for _ii in range(nx):
                    yield (RHO_INF, v, 0.0, PRE_INF)

        return gas_state
