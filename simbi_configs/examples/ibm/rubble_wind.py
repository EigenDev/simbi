# =============================================================================
# rubble_wind.py
#
# a bonded rubble pile in a supersonic wind: a disk-shaped cluster of rigid
# spherical fragments, packed on a lattice and joined by breakable elastic
# bonds, meets a Mach-2 freestream. drag loads the bond network until the
# weakest links part; freed fragments tumble downstream, collide through the
# soft-sphere contact law, and the cluster progressively ablates.
#
# showcases: the bonded-fragment assembly (BondedAssembly.pack over a CSG
# shape), breakable bonds under aerodynamic load, fragment-fragment contact
# with coulomb friction, and per-fragment drag receipts in diagnostics.dat.
#
# usage:
#  simbi run rubble_wind                       # mach 2, bonds part in the wind
#  simbi run rubble_wind --bond-strength 1e30  # unbreakable: a flexible cluster
#  simbi run rubble_wind --mach 0.5            # subsonic, gentler loading
# =============================================================================
import math
from pathlib import Path
from typing import Annotated, Optional

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Shape, Solver
from simbi.types.bodies import BondedAssembly, BondMaterial, ContactMaterial
from simbi.types.typing import ExpressionDict, GasStateGenerator, InitialStateType

RHO_INF = 1.0
PRE_INF = 1.0


class RubbleWind(SimbiProblem):
    """a breakable bonded fragment cluster ablating in a supersonic wind."""

    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="ratio of specific heats")
    ]
    mach: Annotated[
        float,
        ProblemParam(2.0, cli=True, description="freestream Mach number v_inf / c_s"),
    ]
    cluster_radius: Annotated[
        float,
        ProblemParam(0.45, cli=True, description="rubble-disk radius (code units)"),
    ]
    fragment_spacing: Annotated[
        float,
        ProblemParam(
            0.15,
            cli=True,
            description="lattice spacing; fragment radius is half of it",
        ),
    ]
    fragment_mass: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            description="mass per fragment. added-mass stability: keep it well "
            "above the displaced gas mass rho pi r^2 (~0.018 at the default "
            "spacing) or the explicit two-way coupling overshoots and dt collapses",
        ),
    ]
    bond_strength: Annotated[
        float,
        ProblemParam(
            0.5,
            cli=True,
            description="bond tensile/shear strength (stress units); the wind "
            "parts bonds whose tension exceeds it",
        ),
    ]

    resolution: Annotated[
        tuple[int, int],
        ProblemParam((384, 192), cli=True, description="grid resolution"),
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
            Path("data/ibm/rubble_wind/"),
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
        ProblemParam(
            0.25, cli=True, checkpoint_safe=True, description="checkpoint interval"
        ),
    ]
    diagnostic_interval: Annotated[
        float,
        ProblemParam(
            0.1,
            cli=True,
            checkpoint_safe=True,
            description="body-diagnostics cadence; per-fragment drag in diagnostics.dat",
        ),
    ]

    def freestream_velocity(self) -> float:
        # v_inf = mach * c_s, c_s = sqrt(gamma p / rho) at the freestream state.
        return self.mach * math.sqrt(self.adiabatic_index * PRE_INF / RHO_INF)

    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        # a disk of touching fragments; every lattice neighbor (axis + diagonal)
        # carries a breakable bond, and unbonded overlaps repel with friction.
        disk = Shape.sphere((0.0, 0.0, 0.0), self.cluster_radius)
        r = self.cluster_radius
        return BondedAssembly.pack(
            disk,
            bounds=[(-r, r), (-r, r)],
            spacing=self.fragment_spacing,
            fragment_mass=self.fragment_mass,
            bond_material=BondMaterial(
                k_n=200.0,
                k_t=100.0,
                gamma=0.5,
                sigma_t=self.bond_strength,
                tau_s=self.bond_strength,
            ),
            contact=ContactMaterial(k_n=400.0, k_t=200.0, gamma_n=0.5, mu=0.4),
        )

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
        """a uniform freestream fills the tunnel; the cluster carves the shock."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            v = self.freestream_velocity()
            for _jj in range(ny):
                for _ii in range(nx):
                    yield (RHO_INF, v, 0.0, PRE_INF)

        return gas_state
