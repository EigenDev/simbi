# =============================================================================
# magnetized_accretor.py
#
# uniform gas accreting onto a fixed, softened newtonian point mass in 3d.
# the plain drain removes gas and leaves the magnetic field on the grid.
# supported closures are gamma-law nmhd and constant-sound-speed imhd.
# the sink radius is fixed in bondi units when resolution changes; its smooth
# mask transition remains grid-dependent. outflow boundaries are finite-box
# boundaries, not an imposed reservoir at infinity.
#
# defaults define a diagnostic experiment, not a converged accretion solution:
# beta=1, r_acc/r_b=0.25, box half-width=4 r_b, duration=1 t_b.
# checkpoints carry fields and body history, not a complete conservation ledger.
# use an external process-group wall-time bound in addition to the step cap.
#
# usage (from the repository root):
#  .venv/bin/python -m simbi_configs.examples.ibm.magnetized_accretor --max-steps 3000
#  .venv/bin/python -m simbi_configs.examples.ibm.magnetized_accretor --regime imhd
# =============================================================================
import argparse
import math
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
)
from simbi.types.typing import InitialStateType


class MagnetizedAccretor(SimbiProblem):
    """fixed-mass, flux-retaining accretor with a uniform ambient state."""

    central_mass: Annotated[
        float,
        ProblemParam(
            1.0, gt=0.0, allow_inf_nan=False, cli=True, description="GM in code units"
        ),
    ]
    ambient_density: Annotated[
        float,
        ProblemParam(
            1.0, gt=0.0, allow_inf_nan=False, cli=True, description="ambient density"
        ),
    ]
    sound_speed: Annotated[
        float,
        ProblemParam(
            1.0,
            gt=0.0,
            allow_inf_nan=False,
            cli=True,
            description="ambient acoustic speed; adiabatic for nmhd, isothermal for imhd",
        ),
    ]
    adiabatic_index: Annotated[
        float,
        ProblemParam(
            5.0 / 3.0,
            gt=1.0,
            le=2.0,
            cli=True,
            description="gamma-law index; unused by imhd",
        ),
    ]
    beta: Annotated[
        float,
        ProblemParam(
            1.0,
            gt=0.0,
            allow_inf_nan=False,
            cli=True,
            description="ambient plasma beta = 2 p / B^2",
        ),
    ]
    domain_radius: Annotated[
        float,
        ProblemParam(
            4.0,
            gt=0.0,
            allow_inf_nan=False,
            cli=True,
            description="box half-width in bondi radii",
        ),
    ]
    accretion_radius_ratio: Annotated[
        float,
        ProblemParam(
            0.25,
            gt=0.0,
            allow_inf_nan=False,
            cli=True,
            description="sink radius / bondi radius, independent of resolution",
        ),
    ]
    softening_ratio: Annotated[
        float,
        ProblemParam(
            0.05,
            gt=0.0,
            allow_inf_nan=False,
            cli=True,
            description="plummer softening length / bondi radius",
        ),
    ]
    total_bondi_times: Annotated[
        float,
        ProblemParam(
            1.0,
            gt=0.0,
            allow_inf_nan=False,
            cli=True,
            description="physical end time / bondi time",
        ),
    ]
    snapshots_per_bondi_time: Annotated[
        int,
        ProblemParam(
            10, gt=0, cli=True, description="numbered checkpoints per bondi time"
        ),
    ]
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam(
            (128, 128, 128), cli=True, description="uniform cubic grid resolution"
        ),
    ]
    bounds: Annotated[
        list[tuple[float, float]] | None,
        ProblemParam(None, description="derived cartesian bounds"),
    ]
    end_time: Annotated[
        float | None,
        ProblemParam(
            None,
            checkpoint_safe=True,
            description="physical end time; defaults to total_bondi_times times t_b",
        ),
    ]
    checkpoint_interval: Annotated[
        float | None,
        ProblemParam(
            None, checkpoint_safe=True, description="physical checkpoint interval"
        ),
    ]
    data_directory: Annotated[
        Path | None,
        ProblemParam(
            None,
            cli=True,
            checkpoint_safe=True,
            description="output directory; use a unique path for each experiment",
        ),
    ]
    regime: Annotated[
        Regime,
        ProblemParam(
            Regime.NMHD,
            cli=True,
            choices=[Regime.NMHD, Regime.IMHD],
            description="nmhd (gamma-law) or imhd (isothermal)",
        ),
    ]
    coord_system: Annotated[
        CoordSystem,
        ProblemParam(CoordSystem.CARTESIAN, description="cartesian coordinates"),
    ]
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam(
            [BoundaryCondition.OUTFLOW], description="outflow on all six faces"
        ),
    ]
    solver: Annotated[
        Solver, ProblemParam(Solver.HLLE, cli=True, description="riemann solver")
    ]
    cfl_number: Annotated[float, ProblemParam(0.3, gt=0.0, le=1.0, cli=True)]
    x1_spacing: Annotated[CellSpacing, ProblemParam(CellSpacing.LINEAR)]

    def setup(self) -> None:
        super().setup()
        if self.regime not in (Regime.NMHD, Regime.IMHD):
            raise ValueError("magnetized_accretor supports nmhd and imhd only")
        if self.locally_isothermal:
            raise ValueError("magnetized_accretor uses a constant ambient sound speed")
        if self.refinement_enabled:
            raise ValueError("magnetized_accretor defines a single uniform grid")
        if self.coord_system != CoordSystem.CARTESIAN or any(
            spacing != CellSpacing.LINEAR
            for spacing in (self.x1_spacing, self.x2_spacing, self.x3_spacing)
        ):
            raise ValueError("magnetized_accretor requires linear cartesian axes")
        if len(set(self.resolution)) != 1 or self.resolution[0] < 16:
            raise ValueError(
                "magnetized_accretor requires a cubic grid with at least 16 cells per axis"
            )
        half_width = self.domain_radius * self.bondi_radius
        expected_bounds = [(-half_width, half_width)] * 3
        if self.bounds is None:
            self.bounds = expected_bounds
        elif [tuple(pair) for pair in self.bounds] != expected_bounds:
            raise ValueError("bounds must match domain_radius times the bondi radius")
        if not 2.0 * self.cell_width <= self.sink_radius < half_width:
            raise ValueError(
                "sink radius must span at least two cells and lie inside the box"
            )
        if self.softening_ratio > self.accretion_radius_ratio:
            raise ValueError("softening_ratio must not exceed accretion_radius_ratio")
        if self.end_time is None:
            self.end_time = self.total_bondi_times * self.bondi_time
        if self.checkpoint_interval is None:
            self.checkpoint_interval = self.bondi_time / self.snapshots_per_bondi_time
        if self.data_directory is None:
            self.data_directory = Path(
                f"data/ibm/magnetized_accretor/{self.regime.value}/"
                f"beta_{self.beta:g}/n{self.resolution[0]}"
            )

    @property
    def bondi_radius(self) -> float:
        return self.central_mass / self.sound_speed**2

    @property
    def bondi_time(self) -> float:
        return self.bondi_radius / self.sound_speed

    @property
    def cell_width(self) -> float:
        return 2.0 * self.domain_radius * self.bondi_radius / self.resolution[0]

    @property
    def sink_radius(self) -> float:
        return self.accretion_radius_ratio * self.bondi_radius

    @property
    def ambient_pressure(self) -> float:
        pressure = self.ambient_density * self.sound_speed**2
        return pressure if self.isothermal else pressure / self.adiabatic_index

    @property
    def field_strength(self) -> float:
        return math.sqrt(2.0 * self.ambient_pressure / self.beta)

    @property
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        return [
            ImmersedBodyConfig(
                capability=BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL,
                mass=self.central_mass,
                radius=self.sink_radius,
                position=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                two_way_coupling=False,
                gravitational=GravitationalProperties(
                    softening_length=self.softening_ratio * self.bondi_radius,
                    softening_kind="plummer",
                ),
                accretion=AccretionProperties(accretion_radius=self.sink_radius),
            )
        ]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state():
            state = (self.ambient_density, 0.0, 0.0, 0.0)
            if not self.isothermal:
                state += (self.ambient_pressure,)
            for _ in range(math.prod(self.resolution)):
                yield state

        def magnetic_faces(axis):
            count = math.prod(
                size + (ii == axis) for ii, size in enumerate(self.resolution)
            )
            for _ in range(count):
                yield self.field_strength if axis == 0 else 0.0

        return (
            gas_state,
            partial(magnetic_faces, 0),
            partial(magnetic_faces, 1),
            partial(magnetic_faces, 2),
        )

    def summary(self) -> list[tuple[str, str, str]]:
        return [
            ("accretor", "bondi radius", f"{self.bondi_radius:g}"),
            ("accretor", "bondi time", f"{self.bondi_time:g}"),
            (
                "accretor",
                "sink radius / cell width",
                f"{self.sink_radius / self.cell_width:g}",
            ),
            ("accretor", "ambient beta", f"{self.beta:g}"),
            ("accretor", "magnetic coupling", "plain drain; no slip or resistivity"),
        ]


def main(argv=None) -> None:
    """run on cpu with an absolute root-iteration cap; wall time is bounded externally."""
    from simbi.simulation import runner

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-steps", type=int, default=3000)
    args, problem_args = parser.parse_known_args(argv)
    if "--help" in problem_args or "-h" in problem_args:
        help_parser = argparse.ArgumentParser(parents=[parser])
        MagnetizedAccretor.setup_cli(help_parser)
        help_parser.print_help()
        return
    if args.max_steps <= 0:
        parser.error(
            "--max-steps must be positive; it is an absolute iteration cap on restart"
        )
    problem = MagnetizedAccretor.from_cli(problem_args)
    result = runner.run(problem, compute_mode="cpu", max_steps=args.max_steps)
    if result is not None:
        print(guard_report(result))


def guard_report(result) -> str:
    """the accepted guard totals of one run segment: cells the recovery flagged troubled and
    cells the correcting select froze, over the states that survived into the solution. a
    rejected attempt's acts are excluded, and rejections themselves are not counted."""
    guards = result.diagnostics.guards
    return (
        "accepted guard totals for this run segment: "
        f"troubled cells {guards.troubled_cells.total}, "
        f"frozen cells {guards.frozen_cells.total} "
        "(rejected attempts excluded and not counted)"
    )


if __name__ == "__main__":
    main()
