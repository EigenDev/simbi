# =============================================================================
# test_tiled_balanced_reconstruction.py
#
# the balanced reconstruction on tiles. an isentropic atmosphere at rest in
# the potential of a point mass below the domain, with reflecting walls: the
# balanced face pair holds it near rest where the plain reconstruction drives
# a spurious flow, so the balance changes the run. on the 2D grid two host
# tiles cut across the stratification, four host tiles cut on both axes and a
# two-by-two launch publish the single grid's state bit for bit; on the 3D
# grid two and four host tiles and a four-worker launch do too, and so do the
# cartesian, spherical and cylindrical lines and the (r, theta) shell around a
# central mass, each on two host-tile layouts and a launch. under one
# refined box the coarse-fine transfer chains each fine ghost from the patch
# cell nearest it and the restriction band rewrites the four covered cells
# inside each seam from the uncovered cell beyond it: refined host tiles and a
# refined launch publish the single grid's state whether the cuts keep every
# band on one tile or cross a band once or twice, the far tiles resuming the
# band's chains across each cut. a cut on a patch face matches on host tiles
# and launch, fresh and after a rejected step, the tile across it refluxing and
# recovering the band's uncovered cell before the band chains from it. a chart
# outside that scope (the (r, z) plane) and a cut on a level-2 patch face, whose
# band starts across a cut of level 1, are refused by name before any output.
# every arm runs as a fresh process on the session's backend and placement
# (parity_harness), the tiled ones as host tiles; the refusal on host tiles
# involves no launch and runs under the local placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_balanced_reconstruction.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

STEPS = 8
FIXTURE = '''
import math
from pathlib import Path
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam, SimbiProblem
from simbi.types import BoundaryCondition, CoordSystem, Regime, Solver
from simbi.types.bodies import BodyCapability, GravitationalProperties, ImmersedBodyConfig

GM = 100.0
CENTER = (0.5, -1.0)


class BalancedAtmosphere(SimbiProblem):
    """an isentropic atmosphere in the potential of a point mass below the box, at rest"""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int, int], ProblemParam((24, 32), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0)], description="box")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CARTESIAN, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="solver")]
    wb_reconstruction: Annotated[bool, ProblemParam(True, description="balanced")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    gpus: Annotated[int, ProblemParam(1, cli=True, description="devices")]
    data_directory: Annotated[Path, ProblemParam(Path("data/balanced_atmosphere"), cli=True, checkpoint_safe=True, description="out")]

    def _atmosphere(self) -> expr.Equilibrium:
        x, y = expr.coords(2)
        r = ((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2) ** 0.5
        return expr.isentropic_atmosphere(-GM / r, gamma=self.adiabatic_index, k_entropy=1.0, dim=2, reference_density=1.0, reference_point=[0.5, 1.0])

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=GM, radius=0.0, position=(CENTER[0], CENTER[1], 0.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            nx, ny = self.resolution
            for jj in range(ny):
                for ii in range(nx):
                    yield tuple(sample(x1=(ii + 0.5) / nx, x2=(jj + 0.5) / ny))

        return gas


class RefinedBalancedAtmosphere(BalancedAtmosphere):
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75]], description="box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratios")]


class NestedBalancedAtmosphere(RefinedBalancedAtmosphere):
    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.25, 0.75, 0.25, 0.75], [0.375, 0.625, 0.375, 0.625]], description="boxes")]
    refinement_ratios: Annotated[list[int], ProblemParam([2, 2], description="ratios")]


class BalancedAtmosphere3d(BalancedAtmosphere):
    """the same atmosphere in a cube, the point mass below its base"""

    resolution: Annotated[tuple[int, int, int], ProblemParam((16, 16, 16), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], description="cube")]

    def _atmosphere(self) -> expr.Equilibrium:
        x, y, z = expr.coords(3)
        r = ((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z + 1.0) ** 2) ** 0.5
        return expr.isentropic_atmosphere(-GM / r, gamma=self.adiabatic_index, k_entropy=1.0, dim=3, reference_density=1.0, reference_point=[0.5, 0.5, 1.0])

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


class BalancedColumn(BalancedAtmosphere):
    """the atmosphere along one axis, the point mass below its base"""

    resolution: Annotated[tuple[int], ProblemParam((32,), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(0.0, 1.0)], description="column")]

    def _atmosphere(self) -> expr.Equilibrium:
        (x,) = expr.coords(1)
        return expr.isentropic_atmosphere(-GM / (x + 1.0), gamma=self.adiabatic_index, k_entropy=1.0, dim=1, reference_density=1.0, reference_point=[1.0])

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=GM, radius=0.0, position=(-1.0, 0.0, 0.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]

    def initial_primitive_state(self):
        def gas():
            sample = self._atmosphere().sampler()
            (nx,) = self.resolution
            for ii in range(nx):
                yield tuple(sample(x1=(ii + 0.5) / nx))

        return gas


CENTRAL_GM = 10.0



class SphericalAtmosphere(SimbiProblem):
    """an isentropic atmosphere at rest around a central point mass on the radial line"""

    adiabatic_index: Annotated[float, ProblemParam(5.0 / 3.0, description="gamma")]
    resolution: Annotated[tuple[int], ProblemParam((64,), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(1.0, 2.0)], description="r")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.SPHERICAL, description="chart")]
    regime: Annotated[Regime, ProblemParam(Regime.NEWTONIAN, description="regime")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.REFLECTING], description="walls")]
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="solver")]
    wb_reconstruction: Annotated[bool, ProblemParam(True, description="balanced")]
    end_time: Annotated[float, ProblemParam(1.0, cli=True, checkpoint_safe=True, description="t")]
    data_directory: Annotated[Path, ProblemParam(Path("data/sph_atm"), cli=True, checkpoint_safe=True, description="out")]

    @property
    def immersed_bodies(self):
        return [ImmersedBodyConfig(capability=BodyCapability.GRAVITATIONAL, mass=CENTRAL_GM, radius=0.0, position=(0.0, 0.0, 0.0),
                                   velocity=(0.0, 0.0, 0.0), gravitational=GravitationalProperties(softening_length=0.05))]

    def initial_primitive_state(self):
        g = self.adiabatic_index
        k = g / (g - 1.0)
        cst = k - CENTRAL_GM / 2.0  # rho = 1 at r = 2

        def gas():
            (nr,) = self.resolution
            for ii in range(nr):
                r = 1.0 + (ii + 0.5) / nr
                rho = ((cst + CENTRAL_GM / r) / k) ** (1.0 / (g - 1.0))
                yield (rho, 0.0, rho**g)

        return gas


class CylindricalPlaneAtmosphere(SphericalAtmosphere):
    """the radial atmosphere on the (r, z) plane, a chart the tiled balance has no gate on"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 16), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(1.0, 2.0), (0.0, 1.0)], description="(r, z)")]
    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="chart")]

    def initial_primitive_state(self):
        g = self.adiabatic_index
        k = g / (g - 1.0)
        cst = k - CENTRAL_GM / 2.0

        def gas():
            nr, nz = self.resolution
            for _jj in range(nz):
                for ii in range(nr):
                    r = 1.0 + (ii + 0.5) / nr
                    rho = ((cst + CENTRAL_GM / r) / k) ** (1.0 / (g - 1.0))
                    yield (rho, 0.0, 0.0, rho**g)

        return gas


class CylindricalAtmosphere(SphericalAtmosphere):
    """the same atmosphere on the cylindrical radial line, the mass on the axis"""

    coord_system: Annotated[CoordSystem, ProblemParam(CoordSystem.CYLINDRICAL, description="chart")]


class ShellAtmosphere(SphericalAtmosphere):
    """the radial atmosphere on the (r, theta) shell, uniform in theta"""

    resolution: Annotated[tuple[int, int], ProblemParam((32, 16), cli=True, description="grid")]
    bounds: Annotated[list[tuple[float, float]], ProblemParam([(1.0, 2.0), (0.0, math.pi)], description="(r, theta)")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam(
        [BoundaryCondition.REFLECTING, BoundaryCondition.REFLECTING, BoundaryCondition.AXIS, BoundaryCondition.AXIS], description="walls")]

    def initial_primitive_state(self):
        g = self.adiabatic_index
        k = g / (g - 1.0)
        cst = k - CENTRAL_GM / 2.0

        def gas():
            nr, nt = self.resolution
            for _jj in range(nt):
                for ii in range(nr):
                    r = 1.0 + (ii + 0.5) / nr
                    rho = ((cst + CENTRAL_GM / r) / k) ** (1.0 / (g - 1.0))
                    yield (rho, 0.0, 0.0, rho**g)

        return gas
'''
# (class, resolution, momentum dataset along the stratification, host tile cuts, launch cuts)
GRIDS = {
    "2d": ("BalancedAtmosphere", (24, 32), "m2", [[[], [13]], [[11], [17]]], [[11], [17]]),
    "3d": ("BalancedAtmosphere3d", (16, 16, 16), "m3", [[[], [], [7]], [[9], [], [7]]], [[9], [], [7]]),
    "cartesian-line": ("BalancedColumn", (32,), "m1", [[[16]], [[10, 21]]], [[11, 22]]),
    "spherical-line": ("SphericalAtmosphere", (64,), "m1", [[[32]], [[20, 41]]], [[21, 43]]),
    "cylindrical-line": ("CylindricalAtmosphere", (64,), "m1", [[[32]], [[20, 41]]], [[21, 43]]),
    "spherical-shell": ("ShellAtmosphere", (32, 16), "m1", [[[16], []], [[10], [8]]], [[11], [8]]),
}


@pytest.fixture
def fixture(tmp_path: Path) -> Path:
    path = tmp_path / "balanced_atmosphere.py"
    path.write_text(FIXTURE)
    return path


def _tiles(cuts: list[list[int]]) -> int:
    return int(np.prod([len(c) + 1 for c in cuts]))


def _run(fixture: Path, cls: str, out: Path, gpus: int = 1, extra: str = "", steps: int = STEPS, resume: Path | None = None,
         check: bool = True) -> subprocess.CompletedProcess:
    """`cls` on the session's backend, one device or `gpus` host tiles, capped at the absolute root
    iteration `steps` and resumed from `resume` when given, in a fresh interpreter"""
    resume_arg = f", checkpoint_file={str(resume)!r}" if resume else ""
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(fixture.parent)!r})\n"
        "from simbi.simulation import runner\n"
        f"from balanced_atmosphere import {cls}\n"
        f"runner.run({cls}(gpus={gpus}, data_directory=sys.argv[1], checkpoint_interval=1e9{extra}{resume_arg}), "
        f"compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    return ph.python(ph.snippet_args(snippet, str(out)), f"{cls} on {gpus} device(s){extra} to step {steps}", devices=gpus, check=check)


def _final(fixture: Path, cls: str, out: Path, gpus: int = 1, extra: str = "", steps: int = STEPS, resume: Path | None = None) -> Path:
    """the final checkpoint of `_run`"""
    _run(fixture, cls, out, gpus, extra, steps, resume)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _momentum(path: Path, name: str) -> float:
    with h5py.File(path, "r") as f:
        return float(np.abs(f[f"level_0/conserved/{name}"][()]).max())


def _launch_script(fixture: Path, cls: str) -> Path:
    # the launch evolves the one problem its script defines
    script = fixture.parent / f"launched_{cls.lower()}.py"
    script.write_text(f"from balanced_atmosphere import {cls} as Base\n\n\nclass Launched{cls}(Base):\n    pass\n")
    return script


def _launch(fixture: Path, cls: str, out: Path, cuts=None, shape=None, steps: int = STEPS, resolution: tuple[int, ...] | None = (24, 32),
            checkpoint: Path | None = None, inject: str | None = None, expect_ok: bool = True) -> ph.Launch:
    """`simbi launch` of `cls` on the session's backend and placement without the experimental
    opt-in, capped at the absolute root iteration `steps`; with `expect_ok` every worker exits zero"""
    workers = _tiles(cuts) if cuts is not None else int(np.prod(shape))
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e9"]
    if resolution is not None:
        args += ["--resolution", ",".join(str(r) for r in resolution)]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    env = ph.environment(inject)
    env["PYTHONPATH"] = os.pathsep.join([str(fixture.parent), os.getcwd(), os.environ.get("PYTHONPATH", "")])
    return ph.launch(_launch_script(fixture, cls), out, workers, cuts=cuts, shape=shape, args=args, env=env, timeout=600, expect_ok=expect_ok)


def _rejections(run: ph.Launch) -> set[int]:
    """every worker's rejection count"""
    return {r for _, r in run.reports().values()}


@pytest.mark.parametrize("grid", list(GRIDS))
def test_the_balanced_reconstruction_on_tiles_matches_the_single_grid(tmp_path: Path, fixture: Path, grid: str) -> None:
    cls, resolution, momentum, host_cuts, launch_cuts = GRIDS[grid]
    want = _final(fixture, cls, tmp_path / "single")
    plain = _final(fixture, cls, tmp_path / "plain", extra=", wb_reconstruction=False")
    # the 2D atmosphere measured over eight steps: 11.7 plain against 0.053 balanced
    assert _momentum(want, momentum) < 0.1 * _momentum(plain, momentum), f"{grid}: the balance leaves the spurious flow in place; the gate is vacuous"
    for cuts in host_cuts:
        gpus = _tiles(cuts)
        got = _final(fixture, cls, tmp_path / f"tiles{gpus}", gpus, f", decompose={cuts!r}")
        ph.same_checkpoint(got, want, f"{grid} on {gpus} host tiles cut at {cuts}")
    got = _launch(fixture, cls, tmp_path / "launched", cuts=launch_cuts, resolution=resolution).final
    ph.same_checkpoint(got, want, f"{grid} launched on {_tiles(launch_cuts)} workers")


def test_the_balanced_reconstruction_on_refined_tiles_matches_the_single_grid(tmp_path: Path, fixture: Path) -> None:
    # the patch spans root cells 6..18 along x and 8..24 along y; each seam's balanced band covers
    # the four covered cells inside it, and the cuts keep every band on one tile
    want = _final(fixture, "RefinedBalancedAtmosphere", tmp_path / "single")
    plain = _final(fixture, "RefinedBalancedAtmosphere", tmp_path / "plain", extra=", wb_reconstruction=False")
    with h5py.File(want, "r") as a, h5py.File(plain, "r") as b:
        assert not np.array_equal(a["level_1/conserved/den"][()], b["level_1/conserved/den"][()]), "the balance left the refined run unchanged; the gate is vacuous"
    # the patch spans root cells 6..18 in x and 8..24 in y, its bands 6..10, 14..18 and 8..12,
    # 20..24: x = 8 and x = 16 cut inside an x band, with y = 10 the bands intersect at a corner
    # cut on both axes, and 7, 9 and 15, 17 cut each x band twice around a two-cell tile
    for cuts in [[[12], []], [[12], [16]], [[8], []], [[16], []], [[8], [10]], [[7, 9, 15, 17], []]]:
        gpus = _tiles(cuts)
        got = _final(fixture, "RefinedBalancedAtmosphere", tmp_path / f"tiles{gpus}_{cuts[0]}_{cuts[1]}", gpus, f", decompose={cuts!r}")
        ph.same_checkpoint(got, want, f"refined host tiles cut at {cuts}")
    # the launched workers carry a band's chains across each cut inside it at the seam's band
    # points, the cut nearest the seam first: x = 8 inside the x band, with y = 10 the bands
    # intersect at a corner cut on both axes, and 7, 9 and 15, 17 cut each x band twice
    for cuts in [[[12], [16]], [[8], []], [[8], [10]], [[7, 9, 15, 17], []]]:
        got = _launch(fixture, "RefinedBalancedAtmosphere", tmp_path / f"launched_{cuts[0]}_{cuts[1]}", cuts=cuts).final
        ph.same_checkpoint(got, want, f"refined launch on {_tiles(cuts)} workers cut at {cuts}")


def test_a_balanced_patch_face_on_a_cut_matches_on_host_tiles_and_launch(tmp_path: Path, fixture: Path) -> None:
    # the patch spans root cells 6..18 in x and 8..24 in y: a cut at x = 6 lies on the low x face and
    # x = 18 with y = 24 on the high faces of both axes. the fine ghosts past each face prolong
    # through the balanced transfer from a parent stencil gathered across the cut, and the tile
    # across it refluxes and recovers the band's uncovered cell before the band chains from it; on
    # launch the start cell reaches the patch's worker at the band start point
    want = _final(fixture, "RefinedBalancedAtmosphere", tmp_path / "single")
    for cuts in [[[6], []], [[18], [24]]]:
        gpus = _tiles(cuts)
        got = _final(fixture, "RefinedBalancedAtmosphere", tmp_path / f"tiles{gpus}_{cuts[0]}_{cuts[1]}", gpus, f", decompose={cuts!r}")
        ph.same_checkpoint(got, want, f"refined host tiles cut on a patch face at {cuts}")
        got = _launch(fixture, "RefinedBalancedAtmosphere", tmp_path / f"launched_{cuts[0]}_{cuts[1]}", cuts=cuts).final
        ph.same_checkpoint(got, want, f"refined launch on {gpus} workers cut on a patch face at {cuts}")


def test_the_balanced_reconstruction_on_nested_refined_tiles_matches_the_single_grid(tmp_path: Path, fixture: Path) -> None:
    # level 1 spans root cells 6..18 x 8..24 and level 2 root cells 9..15 x 12..20; each level's
    # bands keep cuts 4 parent cells clear of its faces, so the root cuts x = 12 and y = 16 cross
    # both boxes with every band on one tile
    want = _final(fixture, "NestedBalancedAtmosphere", tmp_path / "single")
    plain = _final(fixture, "NestedBalancedAtmosphere", tmp_path / "plain", extra=", wb_reconstruction=False")
    with h5py.File(want, "r") as a, h5py.File(plain, "r") as b:
        assert not np.array_equal(a["level_2/conserved/den"][()], b["level_2/conserved/den"][()]), "the balance left level 2 unchanged; the gate is vacuous"
    # root cut x = 10 falls on level-1 cell 20, inside level 2's band 18..22
    for cuts in [[[12], []], [[12], [16]], [[10], []]]:
        gpus = _tiles(cuts)
        got = _final(fixture, "NestedBalancedAtmosphere", tmp_path / f"tiles{gpus}_{cuts[0]}_{cuts[1]}", gpus, f", decompose={cuts!r}")
        ph.same_checkpoint(got, want, f"nested host tiles cut at {cuts}")
    for cuts in [[[12], [16]], [[10], []]]:
        got = _launch(fixture, "NestedBalancedAtmosphere", tmp_path / f"launched_{cuts[0]}_{cuts[1]}", cuts=cuts).final
        ph.same_checkpoint(got, want, f"nested launch on {_tiles(cuts)} workers cut at {cuts}")


def test_a_rejection_after_a_band_exchange_replays_the_band_on_launch(tmp_path: Path, fixture: Path) -> None:
    # every level-1 substep restricts level 2 into level 1 and, with the root cut x = 10 inside
    # level 2's band, opens the band's point. worker 1 rejects a stage of the second level-1
    # substep of step 2, after the first substep's band equilibria crossed the cut; the replayed
    # step exchanges its own. the same rejection with the cut at x = 12, clear of every band,
    # exchanges no equilibrium, and the tiles equal the single hierarchy on both layouts: the two
    # runs publish one checkpoint only if the replay resumes every chain from the equilibria of
    # its own attempt
    step = 2
    published = {}
    for cuts in ([[10], []], [[12], []]):
        run = _launch(fixture, "NestedBalancedAtmosphere", tmp_path / f"rejecting_{cuts[0][0]}", cuts=cuts, steps=step + 1,
                      inject=f"reject_level:1:{step}:1:1:1")
        counts = _rejections(run)
        assert counts == {1}, f"cut at {cuts}: the rejection was not counted once on every worker: {counts}"
        published[cuts[0][0]] = run.final
    clean = _launch(fixture, "NestedBalancedAtmosphere", tmp_path / "clean", cuts=[[10], []], steps=step + 1).final
    with h5py.File(published[10], "r") as fr, h5py.File(clean, "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    ph.same_checkpoint(published[10], published[12], "the rejecting runs cut inside and clear of level 2's band")
    # the clean control: the run whose timestep at the rejected step is capped to the one the
    # replay accepted, the replay entering from the rejected attempt's entry state
    capped = _launch(fixture, "NestedBalancedAtmosphere", tmp_path / "capped", cuts=[[10], []], steps=step + 1,
                     inject=f"cap:{step}:{int(np.float64(accepted).view(np.uint64))}")
    assert _rejections(capped) == {0}, "the capped control rejected a step"
    ph.same_checkpoint(published[10], capped.final, "the run replaying a step after its band exchange against the capped clean run")


@pytest.mark.parametrize("cut", [12, 6], ids=["clear", "on-face"])
def test_a_rejected_balanced_refined_step_replays_to_the_capped_clean_run_on_launch(tmp_path: Path, fixture: Path, cut: int) -> None:
    # the root step's band rewrite leaves the root's band pressure off the recovery of its
    # conserved state; worker 1 rejects a stage of the second fine substep of step 2, and the
    # replay enters from the rejected attempt's entry, that pressure included. a cut at x = 6 lies
    # on the patch's low x face: the replay refluxes, recovers and carries the band's start again
    step = 2
    cuts = [[cut], []]
    rejecting = _launch(fixture, "RefinedBalancedAtmosphere", tmp_path / "rejecting", cuts=cuts, steps=step + 1, inject=f"reject_level:1:{step}:1:1:1")
    counts = _rejections(rejecting)
    assert counts == {1}, f"the rejection was not counted once on every worker: {counts}"
    clean = _launch(fixture, "RefinedBalancedAtmosphere", tmp_path / "clean", cuts=cuts, steps=step + 1).final
    with h5py.File(rejecting.final, "r") as fr, h5py.File(clean, "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    capped = _launch(fixture, "RefinedBalancedAtmosphere", tmp_path / "capped", cuts=cuts, steps=step + 1,
                     inject=f"cap:{step}:{int(np.float64(accepted).view(np.uint64))}")
    assert _rejections(capped) == {0}, "the capped control rejected a step"
    ph.same_checkpoint(rejecting.final, capped.final, "the balanced refined run replaying a rejected step against the capped clean run")


@pytest.mark.parametrize("cls", ["RefinedBalancedAtmosphere", "NestedBalancedAtmosphere"])
def test_a_resumed_balanced_refined_run_continues_the_uninterrupted_one(tmp_path: Path, fixture: Path, cls: str) -> None:
    # the band rewrite leaves each band's stored pressure off the recovery of its conserved state:
    # the checkpoint stores that pressure, and a resumed run continues from it on every driver.
    # the body history covers the resumed segment alone and stays out of the comparison
    total, half = 4, 2
    for label, gpus, extra in (("single grid", 1, ""), ("host tiles", 2, ", decompose=[[12], []]")):
        whole = _final(fixture, cls, tmp_path / f"w{gpus}", gpus, extra, total)
        first = _final(fixture, cls, tmp_path / f"h{gpus}", gpus, extra, half)
        resumed = _final(fixture, cls, tmp_path / f"r{gpus}", gpus, extra, total, first)
        ph.same_checkpoint(resumed, whole, f"{cls} resumed on the {label}", exclude=("body_diagnostics",))
    cuts = [[12], []]
    whole = _launch(fixture, cls, tmp_path / "lw", cuts=cuts, steps=total).final
    first = _launch(fixture, cls, tmp_path / "lh", cuts=cuts, steps=half).final
    resumed = _launch(fixture, cls, tmp_path / "lr", cuts=cuts, steps=total, checkpoint=first).final
    ph.same_checkpoint(resumed, whole, f"{cls} resumed on two launched workers", exclude=("body_diagnostics",))


@ph.local_only
@pytest.mark.parametrize(
    ("cls", "decompose", "reason"),
    [
        ("CylindricalPlaneAtmosphere", None, "wb_reconstruction on the 2-dimensional (r, z) cylindrical grid with several tiles is outside the gated tiled runs"),
        # root cut x = 9 falls on level-1 cell 18, level 2's low x face: its band starts across a cut of level 1
        ("NestedBalancedAtmosphere", [[9], []], "crosses the 4-cell balanced restriction band inside the level-2 patch face at level-1 cell 18"),
    ],
)
def test_the_balanced_reconstruction_outside_the_gated_scope_is_refused_on_host_tiles(tmp_path: Path, fixture: Path, cls: str, decompose, reason: str) -> None:
    gpus = 2 if decompose is None else _tiles(decompose)
    done = _run(fixture, cls, tmp_path / "out", gpus, "" if decompose is None else f", decompose={decompose!r}", check=False)
    assert done.returncode != 0, f"{cls} ran on host tiles"
    assert reason in " ".join(done.stderr.split()), done.stderr[-1500:]
    assert not list((tmp_path / "out").glob("*.h5")), "output was written before the refusal"


def test_an_ungated_balanced_chart_is_refused_on_launch(tmp_path: Path, fixture: Path) -> None:
    out = tmp_path / "out"
    done = _launch(fixture, "CylindricalPlaneAtmosphere", out, shape=(2, 1), steps=1, resolution=None, expect_ok=False)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    assert "wb_reconstruction on the 2-dimensional (r, z) cylindrical grid with several tiles is outside the gated tiled runs" in " ".join(done.stderr.split()), done.stderr
    assert not (out.exists() and list(out.glob("*.h5"))), "a checkpoint was published"
