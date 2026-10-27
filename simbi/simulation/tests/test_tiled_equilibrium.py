# =============================================================================
# test_tiled_equilibrium.py
#
# a declared stationary target on host tiles and launched workers. the
# isentropic atmosphere in a point mass's gravity, adiabatic and isothermal, on
# one unrefined grid: its discrete imbalance is read off one probe stage of the
# decomposed pipeline and added back at every stage, so two tiles and three
# uneven tiles publish the single grid's state bit for bit, fresh and resumed
# from a checkpoint written on two tiles; two and three uneven launched workers
# likewise, resumed onto three. the target changes the run (the same atmosphere
# without it lands elsewhere), and every decomposition probes at the single
# grid's own timestep, bit for bit. a perturbation declared over the seeded
# target is laid over it after the probe, on tiles and on launch, as the
# single grid lays it. with one refined box, adiabatic and isothermal, two
# tiles and three uneven tiles cut through the box publish every level of the
# single hierarchy bit for bit, fresh and resumed onto three tiles, and probe
# every level at the single hierarchy's own timestep. the same refined
# atmosphere without a target, its gravity evaluated at cell centers of a
# spacing no power of two divides, runs on two tiles as on one hierarchy, and
# the target changes the run. two and three uneven launched refined workers
# hold the target likewise, fresh and resumed onto three, probing every level
# at the single hierarchy's timesteps. every arm runs as a fresh process on the
# session's backend and placement (parity_harness), the tiled ones as host
# tiles; the arms on host tiles alone involve no launch and run under the local
# placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_equilibrium.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 8
CELLS = 96
VARIANTS = '''
import math
from typing import Annotated

from pydantic import computed_field

import simbi.expression as expr
from simbi import ProblemParam
from simbi_configs.examples.newtonian.refined_atmosphere import RefinedAtmosphere
from simbi_configs.examples.newtonian.refined_isothermal_atmosphere import RefinedIsothermalAtmosphere


class Atmosphere(RefinedAtmosphere):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class AtmosphereWithoutTarget(Atmosphere):
    seed_from_equilibrium: Annotated[bool, ProblemParam(False, description="unseeded")]

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return {}


class IsothermalAtmosphere(RefinedIsothermalAtmosphere):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class RefinedTarget(RefinedAtmosphere):
    pass


class RefinedIsothermalTarget(RefinedIsothermalAtmosphere):
    pass


class DyedRefinedTarget(RefinedAtmosphere):
    """the refined target carrying a dye graded across the grid"""

    def passive_scalar(self):
        def dye():
            (n,) = self.resolution
            for ii in range(n):
                yield 0.25 + 0.5 * (ii + 0.5) / n

        return dye()


class RefinedWithoutTarget(RefinedAtmosphere):
    seed_from_equilibrium: Annotated[bool, ProblemParam(False, description="unseeded")]

    @computed_field
    @property
    def equilibrium_expressions(self) -> dict:
        return {}


class PerturbedAtmosphere(Atmosphere):
    @computed_field
    @property
    def perturbation_expressions(self) -> dict:
        x1 = expr.coords(1)[0]
        return x1._graph.compile([0.01 * expr.sin(6.0 * x1), 0.0 * x1, 0.0 * x1]).serialize_equilibrium(dim=1)
'''
# three uneven tiles of the 96 cells
UNEVEN = ", decompose=[[30, 70]]"


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _run_raw(problem: str, out: Path, gpus: int, steps: int, extra: str = "", checkpoint: Path | None = None,
             check: bool = True) -> subprocess.CompletedProcess:
    """`problem` on the session's backend, one device or `gpus` host tiles, in a fresh interpreter"""
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution=({CELLS},), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}{resume}), "
        f"compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    return ph.python(ph.snippet_args(snippet, str(out)), f"{problem} on {gpus} device(s){extra} to step {steps}", devices=gpus, check=check)


PROBE = re.compile(r"SIMBI equilibrium probe: level=0 dt_bits=([0-9a-f]{16})")
LEVEL_PROBES = re.compile(r"SIMBI equilibrium probe: level=(\d+) dt_bits=([0-9a-f]{16})")


def _run(problem: str, out: Path, gpus: int, steps: int = STEPS, extra: str = "", checkpoint: Path | None = None) -> Path:
    path, _ = _run_probed(problem, out, gpus, steps, extra, checkpoint)
    return path


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _run_probed(problem: str, out: Path, gpus: int, steps: int = STEPS, extra: str = "", checkpoint: Path | None = None) -> tuple[Path, set[str]]:
    """the final checkpoint and the probe timesteps the run reported, as f64 bits"""
    done = _run_raw(problem, out, gpus, steps, extra, checkpoint)
    return _final(out), set(PROBE.findall(done.stderr))


def _iteration(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"])


def _compare(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert "level_0/conserved/den" in ys, f"{label}: the reference publishes no density; the comparison is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _assert_same(got: Path, want: Path, label: str) -> None:
    """every cell and face dataset of every level, halos trimmed, bit for bit"""
    ph.disagreement(_compare, got, want, label)


@ph.local_only
@pytest.mark.parametrize("problem", ["Atmosphere", "IsothermalAtmosphere"])
def test_tiles_hold_the_declared_target_as_the_single_grid(tmp_path: Path, variants: Path, problem: str) -> None:
    want, probe = _run_probed(problem, tmp_path / "single", 1)
    assert len(probe) == 1, f"{problem}: the single grid reported probe steps {probe}"
    if problem == "Atmosphere":
        bare = _all_levels(_run("AtmosphereWithoutTarget", tmp_path / "bare", 1))
        assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], bare["level_0/conserved/den"]), "the target left the run unchanged"
    # the tiles' probe step, the smallest of their cfl candidates, is the single grid's own
    two, two_probe = _run_probed(problem, tmp_path / "two", 2)
    assert two_probe == probe, f"{problem}: two tiles probed at {two_probe}, the single grid at {probe}"
    _assert_same(two, want, f"{problem}: two tiles")
    uneven, uneven_probe = _run_probed(problem, tmp_path / "uneven", 3, extra=UNEVEN)
    assert uneven_probe == probe, f"{problem}: three uneven tiles probed at {uneven_probe}, the single grid at {probe}"
    _assert_same(uneven, want, f"{problem}: three uneven tiles")


@ph.local_only
def test_tiles_resume_against_the_declared_target(tmp_path: Path, variants: Path) -> None:
    want = _run("Atmosphere", tmp_path / "single", 1)
    first = _run("Atmosphere", tmp_path / "first", 2, steps=STEPS // 2)
    resumed = _run("Atmosphere", tmp_path / "resumed", 3, steps=STEPS, extra=UNEVEN, checkpoint=first)
    assert _iteration(resumed) == STEPS
    _assert_same(resumed, want, "resumed onto three uneven tiles")


@ph.local_only
def test_tiles_lay_a_perturbation_over_the_seeded_target(tmp_path: Path, variants: Path) -> None:
    # the single grid seeds the target and lays the perturbation over it; the tiles lay it after
    # their probe, over the same seeded target
    want = _run("PerturbedAtmosphere", tmp_path / "single", 1)
    calm = _all_levels(_run("Atmosphere", tmp_path / "calm", 1))
    assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], calm["level_0/conserved/den"]), "the perturbation left the run unchanged"
    _assert_same(_run("PerturbedAtmosphere", tmp_path / "two", 2), want, "perturbed: two tiles")
    _assert_same(_run("PerturbedAtmosphere", tmp_path / "uneven", 3, extra=UNEVEN), want, "perturbed: three uneven tiles")


def _launch_script(variants: Path, problem: str) -> Path:
    # the launch evolves the one problem its script defines
    script = variants.parent / f"launched_{problem.lower()}.py"
    script.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    return script


def _launch(variants: Path, problem: str, out: Path, steps: int, cuts=None, shape=None, checkpoint: Path | None = None,
            resolution: bool = True, expect_ok: bool = True) -> ph.Launch:
    """`simbi launch` of `problem` on the session's backend and placement without the experimental
    opt-in, capped at the absolute root iteration `steps`"""
    workers = int(np.prod([len(c) + 1 for c in cuts])) if cuts is not None else int(np.prod(shape))
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e9"]
    if resolution:
        args += ["--resolution", str(CELLS)]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return ph.launch(_launch_script(variants, problem), out, workers, cuts=cuts, shape=shape, args=args, env=ph.environment(),
                     expect_ok=expect_ok)


def _launch_probed(variants: Path, problem: str, out: Path, steps: int, cuts=None, shape=None, checkpoint: Path | None = None,
                   levels: bool = False) -> tuple[Path, set]:
    """the launch's final checkpoint and the probe timesteps its workers reported, by level when `levels`"""
    run = _launch(variants, problem, out, steps, cuts=cuts, shape=shape, checkpoint=checkpoint)
    return run.final, set(LEVEL_PROBES.findall(run.stderr)) if levels else set(PROBE.findall(run.stderr))


@pytest.mark.parametrize("problem", ["Atmosphere", "IsothermalAtmosphere"])
def test_launched_workers_hold_the_declared_target_as_the_single_grid(tmp_path: Path, variants: Path, problem: str) -> None:
    want, probe = _run_probed(problem, tmp_path / "single", 1)
    got, got_probe = _launch_probed(variants, problem, tmp_path / "two", STEPS, shape=(2,))
    assert got_probe == probe, f"{problem}: two workers probed at {got_probe}, the single grid at {probe}"
    _assert_same(got, want, f"{problem}: two workers")
    got, got_probe = _launch_probed(variants, problem, tmp_path / "three", STEPS, cuts=[[30, 70]])
    assert got_probe == probe, f"{problem}: three uneven workers probed at {got_probe}, the single grid at {probe}"
    _assert_same(got, want, f"{problem}: three uneven workers")
    # resumed from a two-worker checkpoint: the probe reads the build-time target before the
    # restart restores the state, as the single grid does
    first, _ = _launch_probed(variants, problem, tmp_path / "first", STEPS // 2, shape=(2,))
    resumed, _ = _launch_probed(variants, problem, tmp_path / "resumed", STEPS, cuts=[[30, 70]], checkpoint=first)
    assert _iteration(resumed) == STEPS
    _assert_same(resumed, want, f"{problem}: resumed onto three uneven workers")


def test_launched_workers_lay_a_perturbation_over_the_seeded_target(tmp_path: Path, variants: Path) -> None:
    want = _run("PerturbedAtmosphere", tmp_path / "single", 1)
    got, _ = _launch_probed(variants, "PerturbedAtmosphere", tmp_path / "two", STEPS, shape=(2,))
    _assert_same(got, want, "perturbed: two workers")
    got, _ = _launch_probed(variants, "PerturbedAtmosphere", tmp_path / "three", STEPS, cuts=[[30, 70]])
    _assert_same(got, want, "perturbed: three uneven workers")


# three uneven tiles of the 96 cells, the box (cells 29 to 67) cut at 48 and clear of the cut at 20
REFINED_UNEVEN = ", decompose=[[20, 48]]"


def _level_probes(problem: str, out: Path, gpus: int, steps: int = STEPS, extra: str = "", checkpoint: Path | None = None) -> tuple[Path, set[tuple[str, str]]]:
    """the final checkpoint and the (level, timestep bits) every probe reported"""
    done = _run_raw(problem, out, gpus, steps, extra, checkpoint)
    return _final(out), set(LEVEL_PROBES.findall(done.stderr))


@ph.local_only
@pytest.mark.parametrize("problem", ["RefinedTarget", "RefinedIsothermalTarget"])
def test_refined_tiles_hold_the_declared_target_as_the_single_hierarchy(tmp_path: Path, variants: Path, problem: str) -> None:
    want, probes = _level_probes(problem, tmp_path / "single", 1)
    assert {level for level, _ in probes} == {"0", "1"}, f"{problem}: the single hierarchy probed levels {probes}"
    assert any(name.startswith("level_1/") for name in _all_levels(want)), f"{problem}: no fine level was published"
    two, two_probes = _level_probes(problem, tmp_path / "two", 2)
    assert two_probes == probes, f"{problem}: two tiles probed at {two_probes}, the single hierarchy at {probes}"
    _assert_same(two, want, f"{problem}: two refined tiles")
    uneven, uneven_probes = _level_probes(problem, tmp_path / "uneven", 3, extra=REFINED_UNEVEN)
    assert uneven_probes == probes, f"{problem}: three uneven tiles probed at {uneven_probes}, the single hierarchy at {probes}"
    _assert_same(uneven, want, f"{problem}: three uneven refined tiles")
    first, _ = _level_probes(problem, tmp_path / "first", 2, steps=STEPS // 2)
    # the step cap is absolute on the root iteration: the resumed run stops where the uninterrupted one does
    resumed, _ = _level_probes(problem, tmp_path / "resumed", 3, steps=STEPS, extra=REFINED_UNEVEN, checkpoint=first)
    _assert_same(resumed, want, f"{problem}: resumed onto three uneven refined tiles")


@ph.local_only
def test_refined_tiles_place_their_cells_where_the_single_hierarchy_does(tmp_path: Path, variants: Path) -> None:
    # the 96-cell spacing is no power of two, so a tile placing its cells from its own origin
    # lands them an ulp away from the single hierarchy's, and the gravity read at them follows
    want, _ = _level_probes("RefinedWithoutTarget", tmp_path / "single", 1)
    held, _ = _level_probes("RefinedTarget", tmp_path / "held", 1)
    assert not np.array_equal(_all_levels(want)["level_0/conserved/den"], _all_levels(held)["level_0/conserved/den"]), (
        "the target left the refined run unchanged"
    )
    two, _ = _level_probes("RefinedWithoutTarget", tmp_path / "two", 2)
    _assert_same(two, want, "the untargeted refined atmosphere on two tiles")


@pytest.mark.parametrize("problem", ["RefinedTarget", "RefinedIsothermalTarget"])
def test_launched_refined_workers_hold_the_declared_target_as_the_single_hierarchy(tmp_path: Path, variants: Path, problem: str) -> None:
    want, probes = _level_probes(problem, tmp_path / "single", 1)
    assert {level for level, _ in probes} == {"0", "1"}, f"{problem}: the single hierarchy probed levels {probes}"
    got, got_probes = _launch_probed(variants, problem, tmp_path / "two", STEPS, shape=(2,), levels=True)
    assert got_probes == probes, f"{problem}: two refined workers probed at {got_probes}, the single hierarchy at {probes}"
    _assert_same(got, want, f"{problem}: two refined workers")
    got, got_probes = _launch_probed(variants, problem, tmp_path / "three", STEPS, cuts=[[20, 48]], levels=True)
    assert got_probes == probes, f"{problem}: three uneven refined workers probed at {got_probes}, the single hierarchy at {probes}"
    _assert_same(got, want, f"{problem}: three uneven refined workers")
    first, _ = _launch_probed(variants, problem, tmp_path / "first", STEPS // 2, shape=(2,), levels=True)
    # the step cap is absolute on the root iteration: the resumed launch stops where the uninterrupted one does
    resumed, _ = _launch_probed(variants, problem, tmp_path / "resumed", STEPS, cuts=[[20, 48]], checkpoint=first, levels=True)
    _assert_same(resumed, want, f"{problem}: resumed onto three uneven refined workers")


def test_a_dyed_target_is_refused_by_name_on_every_driver(tmp_path: Path, variants: Path) -> None:
    # the stationary target declares the gas alone: a passive scalar beside it has no target
    # concentration, and every driver refuses the pair by name before any output
    reason = "the passive scalar carries no declared target concentration"
    for label, gpus in (("one device", 1), ("two host tiles", 2)):
        done = _run_raw("DyedRefinedTarget", tmp_path / label, gpus, STEPS, check=False)
        assert done.returncode != 0, f"{label}: the run succeeded"
        assert reason in " ".join(done.stderr.split()), f"{label}: refusal not named:\n{done.stderr[-2000:]}"
        assert not list((tmp_path / label).glob("*.h5")), f"{label}: a refused run published a checkpoint"
    out = tmp_path / "launched"
    done = _launch(variants, "DyedRefinedTarget", out, 1, shape=(2,), resolution=False, expect_ok=False)
    assert done.returncode != 0, "the launch succeeded"
    assert reason in " ".join(done.stderr.split()), f"launch: refusal not named:\n{done.stderr[-2000:]}"
    assert not list(out.glob("*.h5")), "a refused launch published a checkpoint"
