# =============================================================================
# roche_passage.py
#
# tidal disruption of a self-gravitating rubble pile on a parabolic encounter:
# a jittered cluster of rigid fragments, held together by its own gravity,
# weak cohesive bonds, and frictional contact, falls past a softened point
# mass on a parabolic orbit with pericenter q. the tidal balance sets the
# outcome: for a pile of size a and mass m about a primary M, the roche-type
# critical distance is
#
#     d_roche = a (2 M / m)^{1/3}
#
# (~3.1 code units at the defaults). a pericenter well inside shreds the pile
# into a tidal stream; well outside, the pile flexes and survives. the ambient
# gas is tenuous (drag is decorative at these masses) but rides along fully
# coupled, so debris-gas interaction strengthens with `--rho-ambient`.
#
# showcases: fragment self-gravity + a gravitational source body in one pair
# sum, cohesion vs tides, and the disruption bracket from the pericenter dial.
#
# usage:
#  simbi run roche_passage                    # q = 1.5 ~ 0.5 d_roche: disrupts
#  simbi run roche_passage --pericenter 5.0   # ~ 1.6 d_roche: survives
#  simbi run roche_passage --bond-strength 50 # cohesion rescues a deep passage
# =============================================================================
import math
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Optional

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Shape, Solver
from simbi.types.bodies import (
    BodyCapability,
    BondedAssembly,
    BondMaterial,
    ContactMaterial,
    GravitationalProperties,
    ImmersedBodyConfig,
    MutualGravity,
)
from simbi.types.typing import GasStateGenerator, InitialStateType

M_PRIMARY = 100.0
SOFTENING = 1.0
RHO_AMB_DEFAULT = 1.0e-3
PRE_AMB = 1.0e-3
R_START = 6.0


class RochePassage(SimbiProblem):
    """rubble-pile tidal disruption bracketing the roche distance."""

    adiabatic_index: Annotated[
        float, ProblemParam(1.4, description="ratio of specific heats")
    ]
    pericenter: Annotated[
        float,
        ProblemParam(
            1.5,
            cli=True,
            description="parabolic pericenter q; d_roche ~ 3 at the defaults, "
            "so q = 1.5 disrupts and q = 5 survives",
        ),
    ]
    pile_radius: Annotated[
        float, ProblemParam(0.55, cli=True, description="rubble-pile radius")
    ]
    fragment_spacing: Annotated[
        float,
        ProblemParam(0.2, cli=True, description="packing lattice spacing"),
    ]
    fragment_mass: Annotated[
        float,
        ProblemParam(
            0.05,
            cli=True,
            description="mass per fragment; keep well above the displaced gas "
            "mass rho pi r^2 (~3e-5 at the tenuous ambient) for added-mass "
            "stability",
        ),
    ]
    bond_strength: Annotated[
        float,
        ProblemParam(
            0.5,
            cli=True,
            description="cohesive bond strength; small = gravity-dominated "
            "rubble, large = strength-dominated body",
        ),
    ]
    rho_ambient: Annotated[
        float,
        ProblemParam(
            RHO_AMB_DEFAULT,
            cli=True,
            description="ambient gas density; raise it to make debris-gas drag "
            "dynamically relevant",
        ),
    ]

    resolution: Annotated[
        tuple[int, int],
        ProblemParam((384, 384), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam([(-8.0, 8.0), (-8.0, 8.0)], description="domain boundaries"),
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
            [BoundaryCondition.OUTFLOW] * 4,
            description="outflow on every face",
        ),
    ]
    cfl_number: Annotated[float, ProblemParam(0.3, description="cfl number")]

    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/ibm/roche_passage/"),
            cli=True,
            checkpoint_safe=True,
            description="output directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            4.0,
            cli=True,
            checkpoint_safe=True,
            description="end time (~ pericenter passage + dispersal)",
        ),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            0.1, cli=True, checkpoint_safe=True, description="checkpoint interval"
        ),
    ]
    diagnostic_interval: Annotated[
        float,
        ProblemParam(
            0.05,
            cli=True,
            checkpoint_safe=True,
            description="per-fragment state cadence in diagnostics.dat",
        ),
    ]

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        # the primary: a fixed softened point mass. it pulls the gas through
        # the baked source fan and the fragments through the mutual-gravity
        # pair sum (one loop, no double count: fragments never enter the fan).
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.GRAVITATIONAL,
                mass=M_PRIMARY,
                radius=0.2,
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                gravitational=GravitationalProperties(softening_length=SOFTENING),
            )
        ]

    @computed_field
    @property
    def bonded_assembly(self) -> Optional[BondedAssembly]:
        # pack the pile at the origin, then translate it to the orbit start
        # and give every fragment the parabolic center-of-mass velocity:
        # v^2 = 2 G M / r0 with angular momentum L = sqrt(2 G M q).
        disk = Shape.sphere((0.0, 0.0, 0.0), self.pile_radius)
        r = self.pile_radius
        pile = BondedAssembly.pack(
            disk,
            bounds=[(-r, r), (-r, r)],
            spacing=self.fragment_spacing,
            fragment_mass=self.fragment_mass,
            bond_material=BondMaterial(
                k_n=500.0,
                k_t=250.0,
                gamma=1.0,
                sigma_t=self.bond_strength,
                tau_s=self.bond_strength,
            ),
            jitter=0.12,
            contact=ContactMaterial(k_n=2000.0, k_t=1000.0, gamma_n=1.0, mu=0.5),
            gravity=MutualGravity(g=1.0, softening=0.02),
        )
        v2 = 2.0 * M_PRIMARY / R_START
        v_t = math.sqrt(2.0 * M_PRIMARY * self.pericenter) / R_START
        v_r = -math.sqrt(max(v2 - v_t * v_t, 0.0))
        return replace(
            pile,
            positions=[[p[0] + R_START, p[1]] for p in pile.positions],
            velocities=[[v_r, v_t] for _ in pile.positions],
        )

    def initial_primitive_state(self) -> InitialStateType:
        """a tenuous uniform ambient; the primary carves its own infall."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            for _jj in range(ny):
                for _ii in range(nx):
                    yield (self.rho_ambient, 0.0, 0.0, PRE_AMB)

        return gas_state
