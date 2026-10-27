# =============================================================================
# test_parity_coverage_refinement.py
#
# single-device settings on refined grids, each run through the tiled and
# launched drivers on the session's backend and placement (parity_harness) and
# held to the exact contract against the single device on that backend.
# the table (parity_table.install) gates every checkpoint of two host tiles,
# two launched workers on an even cut and three on an uneven cut, a launch
# resumed onto three uneven workers, and a stage rejected on one worker in the
# last stage of the last substep of the deepest level, for:
# - the refined 3D blast (one fine box at 24^3) with a constant viscosity, a
#   declared initial perturbation, and a passive scalar painted on the low x
#   side of the blast
# - the 3D isentropic atmosphere in a point mass's gravity under the balanced
#   reconstruction, one fine box and two nested boxes, the cuts clear of every
#   balanced restriction band, flat and under the hllc+ flux
# - the 2D and 3D isentropic atmospheres held as a declared stationary target
#   under one fine box, the 3D one also under the hllc+ flux
# every entry names the same fixture without its setting as a control, whose
# single-device run differs from the entry's.
# a census and a tracer population leave the gas untouched, so their gates
# compare the recorded group itself, at every checkpoint and across a rejected
# stage: a radial density census on the refined bondi sink, also across a
# resume, and a discrete tracer population in the refined 3D blast, whose
# checkpoint holds tracers owned on the fine level and resumes on no driver.
# entries outside the gpu admission run under SIMBI_LAUNCH_EXPERIMENTAL, the
# explicit route for a combination whose gates are being established; the
# refined tracer population stays refused on gpu device memory, which the
# harness records as a missing implementation.
# each table entry is a module of one problem class, written beside the test.
#
# run: pytest -m simulation simbi/simulation/tests/test_parity_coverage_refinement.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest

from simbi.reader.census import read_census
from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.parity_table import (
    REJECTED_STEP,
    STEPS,
    Feature,
    final_of,
    install,
    launch,
    same_run,
    single,
    tiles,
    workers_of,
)
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels
from simbi.simulation.tests.test_tiled_census import _census_groups
from simbi.simulation.tests.test_tiled_tracers import _population

pytestmark = pytest.mark.simulation

ONE_LEVEL = f"reject_level:1:{REJECTED_STEP}:1:1:1"
TWO_LEVELS = f"reject_level:1:{REJECTED_STEP}:2:3:1"
# the fine box spans root cells 6..18 of every 24-cell root below
EVEN, UNEVEN = [[12], [], []], [[7, 15], [], []]
DYE_LOW, DYE_HIGH = 0.25, 1.0

BLAST = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.newtonian.refined_blast_3d import RefinedBlast3D


class {name}(RefinedBlast3D):
    """the 3D blast under one fine box at 24^3{what}"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((24, 24, 24), cli=True, description="grid")]
{body}
'''
VISCOSITY = '''
    @computed_field
    @property
    def viscosity(self) -> float:
        return 0.01
'''
# a density, velocity and pressure delta varying along all three axes
PERTURBATION = '''
    @computed_field
    @property
    def perturbation_expressions(self) -> dict:
        x1, x2, x3 = expr.coords(3)
        wave = expr.sin(2.0 * math.pi * x1) * expr.cos(2.0 * math.pi * x2) * expr.cos(2.0 * math.pi * x3)
        return x1._graph.compile([0.05 * wave, 0.02 * wave, -0.02 * wave, 0.02 * wave, 0.01 * wave]).serialize_equilibrium(dim=3)
'''
# the dye boundary at x = -1/24 lies inside the blast, whose outflow carries the undyed gas across it
DYE = f'''
    def passive_scalar(self):
        def dye():
            nx, ny, nz = self.resolution
            for _kk in range(nz):
                for _jj in range(ny):
                    for ii in range(nx):
                        yield {DYE_HIGH!r} if ii < nx // 2 - 1 else {DYE_LOW!r}

        return dye()
'''
# about one tracer per root cell, so the few cells the blast moves gas through within six steps
# hold tracers
TRACERS = '''
    @computed_field
    @property
    def n_tracers(self) -> int:
        return 16000
'''
BLAST_CONTROL = (BLAST.format(name="BlastControl", what="", body=""), "BlastControl")

# an isentropic atmosphere in the potential of a point mass below the cube's base, at rest, with
# reflecting walls; the balanced face pair holds it near rest
BALANCED = '''
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig

GM = 100.0


class {name}(SimbiProblem):
    """the isentropic atmosphere above a point mass at (0.5, 0.5, -1) in the unit cube at 24^3,
    under the balanced reconstruction and {what}"""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int, int, int], ProblemParam((24, 24, 24), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="cube")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    solver: Annotated[Solver, ProblemParam(Solver.{solver}, description="solver")]
    wb_reconstruction: Annotated[bool, ProblemParam(True, description="balanced")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
{refinement}
    def _atmosphere(self) -> expr.Equilibrium:
        x, y, z = expr.coords(3)
        r = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z + 1.0) ** 2) ** 0.5
        return expr.isentropic_atmosphere(-GM / r, gamma=self.adiabatic_index, k_entropy=1.0, dim=3, reference_density=1.0,
                                          reference_point=[0.5, 0.5, 1.0])

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=GM, radius=0.0, position=(0.5, 0.5, -1.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            nx, ny, nz = self.resolution
            for kk in range(nz):
                for jj in range(ny):
                    for ii in range(nx):
                        yield tuple(sample(x1=(ii + 0.5) / nx, x2=(jj + 0.5) / ny, x3=(kk + 0.5) / nz))

        return gas
'''
ONE_BOX = '''    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]
'''
TWO_BOXES = '''    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[
        list[list[float]],
        ProblemParam([[0.25, 0.75, 0.25, 0.75, 0.25, 0.75], [0.375, 0.625, 0.375, 0.625, 0.375, 0.625]], description="boxes"),
    ]
    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description="ratios")]
'''
# the level-1 box spans root cells 6..18, its balanced bands 6..10 and 14..18; the level-2 box spans
# level-1 cells 18..30, its bands 18..22 and 26..30 (root cells 9..11 and 13..15). root cuts at 4
# and 12 fall outside every band
BALANCED_EVEN, BALANCED_UNEVEN = [[12], [], []], [[4, 12], [], []]

# an isentropic atmosphere stratified along the last axis, uniform across the others, its
# discrete imbalance probed once and added back at every stage
TARGET = '''
from typing import Annotated

import numpy as np
from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver


class {name}(SimbiProblem):
    """an isentropic atmosphere in the gravity of a point mass below the unit box, held as the
    run's declared target under one fine box over the central half{what}"""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int, ...], ProblemParam({resolution!r}, cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam({bounds!r}, description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    solver: Annotated[Solver, ProblemParam(Solver.{solver}, description="solver")]
    seed_from_equilibrium: Annotated[bool, ProblemParam(True, description="seeded")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([{box!r}], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]

    def _atmosphere(self) -> expr.Equilibrium:
        height = expr.coords({dim})[-1]
        return expr.isentropic_atmosphere(-100.0 / (height + 1.0), gamma=self.adiabatic_index, k_entropy=1.0, dim={dim},
                                          reference_density=1.0, reference_point={reference!r})

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return expr.equilibrium(self._atmosphere().primitives, dim={dim})

    @computed_field
    @property
    def source_expressions(self) -> list[dict]:
        return [expr.force(self._atmosphere().acceleration, dim={dim})]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            shape = self.resolution
            for flat in range(int(np.prod(shape))):
                index = np.unravel_index(flat, shape[::-1])[::-1]
                yield tuple(sample(**{{f"x{{ax + 1}}": (index[ax] + 0.5) / shape[ax] for ax in range(len(shape))}}))

        return gas
'''


def _target(name: str, dim: int, solver: str = "HLLC", what: str = "") -> str:
    return TARGET.format(name=name, dim=dim, solver=solver, what=what, resolution=(32, 32) if dim == 2 else (24, 24, 24),
                         bounds=[(0.0, 1.0)] * dim, box=[0.25, 0.75] * dim, reference=[1.0] * dim)


def _control(key: str, base: str, body: str, imports: str = "") -> tuple[str, str]:
    """the entry's fixture with its setting reverted: a subclass of the feature's own class, from the
    feature module written beside the test"""
    module = f"feature_{key.replace('-', '_')}"
    return (f"{imports}from {module} import {base}\n\n\nclass Control({base}):\n{body}\n", "Control")


UNBALANCED = '    wb_reconstruction: Annotated[bool, ProblemParam(False, description="plain")]\n'
HLLC = '    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="hllc")]\n'
FIELD_IMPORTS = "from typing import Annotated\n\nfrom simbi import ProblemParam\nfrom simbi.types import Solver\n"
WITHOUT_TARGET = (
    '    seed_from_equilibrium: Annotated[bool, ProblemParam(False, description="unseeded")]\n\n'
    "    @computed_field\n    @property\n    def equilibrium_expressions(self) -> dict:\n        return {}\n"
)
TARGET_IMPORTS = "from typing import Annotated\n\nfrom pydantic import computed_field\n\nfrom simbi import ProblemParam\n"


def _dye_mixed(mine: Path, control: Path | None) -> None:
    """the blast carries undyed gas across the dye boundary on the fine level: a concentration
    strictly inside the painted range proves the dye is transported, beyond a dataset written
    alongside the gas"""
    (chi,) = [x for name, x in _all_levels(mine).items() if name.startswith("level_1/") and name.endswith("primitives/chi")]
    mixed = (chi > DYE_LOW + 1e-3) & (chi < DYE_HIGH - 1e-3)
    assert mixed.any(), "the fine dye holds only its painted values; the dye transport is never exercised"


FEATURES = {
    "refined-blast-viscosity-3d": Feature(BLAST.format(name="ViscousBlast", what=" and a constant viscosity", body=VISCOSITY), "ViscousBlast",
                                          (24, 24, 24), EVEN, UNEVEN, ONE_LEVEL, opt_in=True, control=BLAST_CONTROL),
    "refined-blast-perturbation-3d": Feature(BLAST.format(name="PerturbedBlast", what=" and a declared perturbation", body=PERTURBATION),
                                             "PerturbedBlast", (24, 24, 24), EVEN, UNEVEN, ONE_LEVEL, opt_in=True, control=BLAST_CONTROL),
    "refined-blast-dye-3d": Feature(BLAST.format(name="DyedBlast", what=" carrying a dye", body=DYE), "DyedBlast", (24, 24, 24),
                                    EVEN, UNEVEN, ONE_LEVEL, opt_in=True, control=BLAST_CONTROL, premise=_dye_mixed),
    "balanced-refined-3d": Feature(BALANCED.format(name="BalancedRefined", what="one fine box", solver="HLLC", refinement=ONE_BOX),
                                   "BalancedRefined", (24, 24, 24), BALANCED_EVEN, BALANCED_UNEVEN, ONE_LEVEL, opt_in=True,
                                   control=_control("balanced-refined-3d", "BalancedRefined", UNBALANCED, FIELD_IMPORTS)),
    "balanced-nested-3d": Feature(BALANCED.format(name="BalancedNested", what="two nested boxes", solver="HLLC", refinement=TWO_BOXES),
                                  "BalancedNested", (24, 24, 24), BALANCED_EVEN, BALANCED_UNEVEN, TWO_LEVELS, opt_in=True,
                                  control=_control("balanced-nested-3d", "BalancedNested", UNBALANCED, FIELD_IMPORTS)),
    "hllc-plus-balanced-refined-3d": Feature(BALANCED.format(name="HllcPlusBalanced", what="one fine box with the hllc+ flux", solver="HLLC_PLUS",
                                                             refinement=ONE_BOX),
                                             "HllcPlusBalanced", (24, 24, 24), BALANCED_EVEN, BALANCED_UNEVEN, ONE_LEVEL, opt_in=True,
                                             control=_control("hllc-plus-balanced-refined-3d", "HllcPlusBalanced", HLLC, FIELD_IMPORTS)),
    # the fine box spans root cells 8..24 of the 32-cell root
    "target-refined-2d": Feature(_target("TargetRefined2d", 2), "TargetRefined2d", (32, 32), [[16], []], [[10, 21], []], ONE_LEVEL, opt_in=True,
                                 control=_control("target-refined-2d", "TargetRefined2d", WITHOUT_TARGET, TARGET_IMPORTS)),
    "target-refined-3d": Feature(_target("TargetRefined3d", 3), "TargetRefined3d", (24, 24, 24), EVEN, UNEVEN, ONE_LEVEL, opt_in=True,
                                 control=_control("target-refined-3d", "TargetRefined3d", WITHOUT_TARGET, TARGET_IMPORTS)),
    "hllc-plus-target-refined-3d": Feature(_target("HllcPlusTarget", 3, "HLLC_PLUS", " with the hllc+ flux"), "HllcPlusTarget", (24, 24, 24),
                                           EVEN, UNEVEN, ONE_LEVEL, opt_in=True,
                                           control=_control("hllc-plus-target-refined-3d", "HllcPlusTarget", HLLC, FIELD_IMPORTS)),
}


install(globals(), FEATURES)


# ------------------------------------------------------------------------------------------------
# recorded groups: a census and a tracer population, compared as recorded
# ------------------------------------------------------------------------------------------------
CENSUSED_SINK = '''
from typing import Annotated

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


def _shells():
    g = expr.ExprGraph()
    x, y, z = expr.variable("x", g), expr.variable("y", g), expr.variable("z", g)
    mass = expr.density(g) * expr.cell_volume(g)
    r = expr.sqrt(x * x + y * y + z * z)
    return expr.Census(
        name="shells",
        axes=[expr.BinAxis("r", r, [0.0, 0.5, 1.0, 1.5, 2.5, 4.0, 9.0])],
        values={"mass": mass, "mass_x": mass * x},
    ).serialize()


class CensusedSink(SphericalBondiTest):
    """the bondi sink at the origin of a 24-cell root on [-5, 5]^3 under one fine box over
    [-2.5, 2.5]^3, its gas mass binned in radius at every root step"""

    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam(False, description="no sponge source")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]] | None, ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5]], description="box")]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2], description="ratios")]

    @property
    def census_expressions(self):
        return [_shells()]
'''


@dataclass(frozen=True)
class Recorded:
    feature: Feature
    # the recorded group of one checkpoint, by dataset
    record: Callable[[Path], dict[str, np.ndarray]]
    # a check of the single device's checkpoints that the group records a changing quantity
    exercised: Callable[[dict[str, Path]], None]
    # a check that the first segment's record and the resumed one's make the uninterrupted record
    continues: Callable[[Path, Path, Path], None]


def _census_exercised(files: dict[str, Path]) -> None:
    shells = read_census(str(final_of(files)), "shells")
    assert len(shells.time) == STEPS, f"the per-step census holds {len(shells.time)} rows"
    assert not np.array_equal(shells.values[0], shells.values[-1]), "the census never changed; the comparison is vacuous"


def _census_continues(whole: Path, first: Path, resumed: Path) -> None:
    want, head, tail = (read_census(str(p), "shells") for p in (whole, first, resumed))
    assert len(head.time) > 0 and len(tail.time) > 0, "a segment recorded no census row"
    np.testing.assert_array_equal(np.concatenate([head.time, tail.time]), want.time, err_msg="census sample times")
    np.testing.assert_array_equal(np.concatenate([head.values, tail.values]), want.values, err_msg="census samples")


def _tracers_exercised(files: dict[str, Path]) -> None:
    start = _population(files[min(files, key=lambda name: _iteration(files[name]))], "tracers")
    end = _population(final_of(files), "tracers")
    assert len(end["id"]) > 0, "the blast seeded no tracer"
    with h5py.File(final_of(files), "r") as f:
        assert len(end["id"]) == int(f["tracers"].attrs["n_tracers"])
    assert not np.array_equal(start["owner"], end["owner"]), "no tracer changed cell; the comparison is vacuous"


def _tracers_continue(whole: Path, first: Path, resumed: Path) -> None:
    ps, qs = _population(resumed, "tracers"), _population(whole, "tracers")
    assert ps.keys() == qs.keys(), f"tracer fields {sorted(set(ps) ^ set(qs))} differ"
    for name, p in ps.items():
        np.testing.assert_array_equal(p, qs[name], err_msg=f"resumed tracers/{name}")


def _iteration(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"])


RECORDED = {
    "census-sink-refined-3d": Recorded(Feature(CENSUSED_SINK, "CensusedSink", None, EVEN, UNEVEN, ONE_LEVEL, opt_in=True),
                                       _census_groups, _census_exercised, _census_continues),
    # the refined tracer population is refused on gpu device memory, with or without the opt-in. a
    # checkpoint holding tracers owned on a refined level resumes on no driver, so the entry has
    # no restart
    "tracers-blast-refined-3d": Recorded(Feature(BLAST.format(name="TracedBlast", what=" carrying discrete tracers", body=TRACERS),
                                                 "TracedBlast", (24, 24, 24), EVEN, UNEVEN, ONE_LEVEL, opt_in=True, restart=False),
                                         lambda path: _population(path, "tracers"), _tracers_exercised, _tracers_continue),
}


@pytest.fixture
def recorded(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Recorded, Path]:
    r = RECORDED[request.param]
    script = tmp_path / f"recorded_{request.param.replace('-', '_')}.py"
    script.write_text(r.feature.source)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return r, script


def _same_record(record: Callable[[Path], dict[str, np.ndarray]], got: Path, want: Path, label: str) -> None:
    xs, ys = record(got), record(want)
    assert ys, f"{label}: the reference wrote no recorded group"
    assert xs.keys() == ys.keys(), f"{label}: recorded entries {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _same_records(r: Recorded, got: dict[str, Path], want: dict[str, Path], label: str) -> None:
    """every checkpoint's recorded group, exactly"""
    if sorted(got) != sorted(want):
        raise ph.NumericalDisagreement(f"{label}: checkpoints {sorted(got)} against {sorted(want)}")
    for name in sorted(got):
        ph.disagreement(_same_record, r.record, got[name], want[name], f"{label}: {name}")


@pytest.mark.parametrize("recorded", list(RECORDED), indirect=True)
def test_tiles_and_launches_record_the_single_device_group_at_every_checkpoint(tmp_path: Path, recorded) -> None:
    r, script = recorded
    f = r.feature
    reference = single(f, script, tmp_path / "single")
    assert len(reference) >= 4, f"{len(reference)} checkpoints; the cadence is not exercised"
    r.exercised(reference)
    tiled = tiles(f, script, tmp_path / "tiles", f.even)
    same_run(tiled, reference, tmp_path / "tiles", tmp_path / "single", "two host tiles")
    _same_records(r, tiled, reference, "two host tiles")
    for label, cuts in (("two workers", f.even), ("three uneven workers", f.uneven)):
        out = tmp_path / label.replace(" ", "-")
        run = launch(f, script, out, cuts)
        assert run.reports() == {w: (STEPS, 0) for w in range(workers_of(cuts))}, f"{label}: (steps, rejections) per worker {run.reports()}"
        same_run(run.files, reference, out, tmp_path / "single", label)
        _same_records(r, run.files, reference, label)


@pytest.mark.parametrize("recorded", [k for k, r in RECORDED.items() if r.feature.restart], indirect=True)
def test_a_recorded_group_resumed_onto_uneven_workers_continues_the_single_device(tmp_path: Path, recorded) -> None:
    r, script = recorded
    f = r.feature
    whole = final_of(single(f, script, tmp_path / "single"))
    first = launch(f, script, tmp_path / "first", f.even, steps=STEPS // 2)
    assert _iteration(first.final) == STEPS // 2
    resumed = launch(f, script, tmp_path / "resumed", f.uneven, checkpoint=first.final)
    assert resumed.reports() == {w: (STEPS - STEPS // 2, 0) for w in range(workers_of(f.uneven))}, (
        f"(steps, rejections) per worker {resumed.reports()}"
    )
    resumed_single = final_of(single(f, script, tmp_path / "single-resumed", checkpoint=first.final))
    ph.disagreement(_same_record, r.record, resumed.final, resumed_single, "resumed onto three uneven workers against the resumed single device")
    ph.disagreement(r.continues, whole, first.final, resumed.final)


@pytest.mark.parametrize("recorded", list(RECORDED), indirect=True)
def test_a_rejected_stage_records_as_the_capped_clean_launch(tmp_path: Path, recorded) -> None:
    r, script = recorded
    f = r.feature
    steps = REJECTED_STEP + 1
    rejected = launch(f, script, tmp_path / "rejected", f.even, steps=steps, inject=f.rejection)
    assert rejected.reports() == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {rejected.reports()}: a worker kept out of the rollback"
    clean = launch(f, script, tmp_path / "clean", f.even, steps=steps)
    with h5py.File(rejected.final, "r") as fr, h5py.File(clean.final, "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    capped = launch(f, script, tmp_path / "capped", f.even, steps=steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert capped.reports() == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    same_run({"final": rejected.final}, {"final": capped.final}, tmp_path / "rejected", tmp_path / "capped", "rejected stage against the capped clean launch")
    ph.disagreement(_same_record, r.record, rejected.final, capped.final, "rejected stage against the capped clean launch")
