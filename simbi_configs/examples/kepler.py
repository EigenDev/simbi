# =============================================================================
# kepler.py
#
# thin ring of matter in keplerian orbit.
# tests angular momentum conservation and numerical viscosity.
# =============================================================================
import math
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

from simbi import ProblemParam, SimbiProblem
from simbi.types import (
    BoundaryCondition,
    CellSpacing,
    CoordSystem,
    Regime,
    Solver,
)
from simbi.types.bodies import (
    BodyCapability,
    GravitationalProperties,
    ImmersedBodyConfig,
)
from simbi.types.typing import GasStateGenerator, InitialStateType


class KeplerianRingTest(SimbiProblem):
    """thin ring of matter in keplerian orbit."""

    # physics
    adiabatic_index: Annotated[
        float, ProblemParam(1.0, description="adiabatic index (isothermal)")
    ]
    buffer_width: Annotated[
        float,
        ProblemParam(
            0.2, description="width of buffer zone (fraction of outer radius)"
        ),
    ]
    buffer_damp_time: Annotated[
        float,
        ProblemParam(
            0.1, description="damping timescale (orbital periods at r=1)"
        ),
    ]

    # domain
    resolution: Annotated[
        tuple[int, int],
        ProblemParam((256, 256), cli=True, description="grid resolution"),
    ]
    bounds: Annotated[
        list[tuple[float, float]],
        ProblemParam(
            [(-2.0, 2.0), (-2.0, 2.0)], description="domain boundaries"
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="coordinate system"),
    ]
    regime: Annotated[
        Regime, ProblemParam(Regime.NEWTONIAN, description="physics regime")
    ]
    x1_spacing: Annotated[
        CellSpacing,
        ProblemParam(
            CellSpacing.LINEAR, description="grid spacing in x1 direction"
        ),
    ]

    # numerics
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLE, description="numerical solver")
    ]
    boundary_conditions: Annotated[
        BoundaryCondition,
        ProblemParam(
            BoundaryCondition.OUTFLOW, description="boundary conditions"
        ),
    ]
    cfl_number: Annotated[
        float, ProblemParam(0.25, description="cfl condition number")
    ]

    # simulation control
    data_directory: Annotated[
        Path,
        ProblemParam(
            Path("data/kepler/"),
            cli=True,
            checkpoint_safe=True,
            description="output data directory",
        ),
    ]
    end_time: Annotated[
        float,
        ProblemParam(
            20.0 * math.pi,
            cli=True,
            checkpoint_safe=True,
            description="simulation end time (10 orbits)",
        ),
    ]
    checkpoint_interval: Annotated[
        float,
        ProblemParam(
            0.2 * math.pi,
            cli=True,
            checkpoint_safe=True,
            description="checkpoint interval",
        ),
    ]

    @property
    def ambient_sound_speed(self) -> float:
        return 0.01

    @property
    def buffer_parameters(self) -> dict[str, float]:
        r_outer = min(abs(self.bounds[0][1]), abs(self.bounds[1][1]))
        r_buffer = r_outer * (1.0 - self.buffer_width)

        G = 1.0
        M = 1.0
        T_orb = 2.0 * math.pi * math.sqrt(1.0**3 / (G * M))

        return {
            "r_buffer": r_buffer,
            "r_outer": r_outer,
            "damp_time": self.buffer_damp_time * T_orb,
        }

    @computed_field
    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        dx = (self.bounds[0][1] - self.bounds[0][0]) / self.resolution[0]
        softening_length = 2.0 * dx
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.GRAVITATIONAL,
                mass=1.0,
                radius=0.01,
                position=(0.0, 0.0),
                velocity=(0.0, 0.0),
                gravitational=GravitationalProperties(
                    softening_length=softening_length
                ),
            )
        ]

    def initial_primitive_state(self) -> InitialStateType:
        """generate initial primitive state for keplerian disk."""

        def gas_state() -> GasStateGenerator:
            nx, ny = self.resolution
            xmin, xmax = self.bounds[0]
            ymin, ymax = self.bounds[1]

            dx = (xmax - xmin) / nx
            dy = (ymax - ymin) / ny

            r0 = 1.0
            dr = 0.1
            M_0 = 1.0
            G = 1.0

            sigma_min = 1e-8
            sigma_peak = 1.0

            cs_0 = self.ambient_sound_speed
            cs_squared = cs_0 * cs_0

            epsilon = 1e-10

            for jj in range(ny):
                y = ymin + (jj + 0.5) * dy
                for ii in range(nx):
                    x = xmin + (ii + 0.5) * dx
                    r = math.sqrt(x**2 + y**2)

                    if r < epsilon:
                        sigma = sigma_min
                        vx = 0.0
                        vy = 0.0
                        p = sigma * cs_squared
                        yield (sigma, vx, vy, p)
                        continue

                    sigma = sigma_min + sigma_peak * math.exp(
                        -((r - r0) ** 2) / (2 * dr**2)
                    )
                    v_k = math.sqrt(G * M_0 / r)
                    vx = -v_k * (y / r)
                    vy = +v_k * (x / r)

                    p = sigma * cs_squared

                    yield (sigma, vx, vy, p)

        return gas_state
