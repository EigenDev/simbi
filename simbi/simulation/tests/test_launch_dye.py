# =============================================================================
# test_launch_dye.py
#
# the passive scalar (dye) on every driver. the refined blast painted with a
# dye of two values, with and without its refined box, and an isothermal
# channel painted half and half:
# - the single grid seeds the fine level with its parent's dye: the covered
#   coarse dye is the restriction of the fine dye at the start, and the
#   concentration stays inside the painted range on every level after a step;
# - host tiles and two launched workers publish the single grid's dye and gas,
#   every level bit for bit;
# - a launch resumed onto three unevenly cut workers from its own checkpoint
#   reaches the uninterrupted single grid bit for bit;
# - a launch whose second worker rejects a late stage rolls the dye back with
#   the gas: its final file equals one worker's under the same rejection bit for
#   bit, and differs from the clean run.
# every arm runs on the session's backend and placement (parity_harness), its
# single-device references and host tiles as fresh processes; the arm on the
# single grid alone involves no launch and runs under the local placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_dye.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 8
LOW, HIGH = 0.25, 1.0
VARIANTS = f'''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.isothermal.isothermal_channel import IsothermalChannel
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast


class DyedBlast(RefinedBlast):
    def passive_scalar(self):
        def dye():
            nx, ny, nz = self.resolution
            for _kk in range(nz):
                for _jj in range(ny):
                    for ii in range(nx):
                        yield {HIGH!r} if ii < nx // 2 else {LOW!r}

        return dye()


class FlatDyedBlast(DyedBlast):
    refinement_enabled: Annotated[bool, ProblemParam(False, description="flat")]


class DyedChannel(IsothermalChannel):
    def passive_scalar(self):
        def dye():
            ni, nj = self.resolution
            for _jj in range(nj):
                for ii in range(ni):
                    yield {HIGH!r} if ii < ni // 2 else {LOW!r}

        return dye()
'''
ARMS = {
    "refined": ("DyedBlast", (32, 32, 1)),
    "flat": ("FlatDyedBlast", (32, 32, 1)),
    "isothermal": ("DyedChannel", (32, 32)),
}


@pytest.fixture
def variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "variants.py").write_text(VARIANTS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return tmp_path / "variants.py"


def _script(variants: Path, problem: str) -> Path:
    """a launch reads the one problem class its script defines"""
    path = variants.parent / f"{problem.lower()}.py"
    path.write_text(f"from variants import {problem} as Base\n\n\nclass Launched{problem}(Base):\n    pass\n")
    return path


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _single(problem: str, resolution: tuple, out: Path, gpus: int = 1, steps: int = STEPS) -> Path:
    """`problem` on the session's backend, one device or `gpus` host tiles; its final checkpoint"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), "
        f"compute_mode=sys.argv[2], max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{problem} on {gpus} device(s)", devices=gpus)
    return _final(out)


def _launch_run(script: Path, resolution: tuple, out: Path, workers: int, steps: int, cuts: list | None = None, shape: tuple | None = None,
                checkpoint: Path | None = None, inject: str | None = None) -> ph.Launch:
    """`script` launched on `workers` workers partitioned by `cuts` (else `shape`) on the session's
    backend and placement, with no experimental opt-in; every worker exits zero"""
    args = ["--max-steps", str(steps), "--resolution", ",".join(str(r) for r in resolution), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return ph.launch(script, out, workers=workers, cuts=cuts, shape=shape, args=args, env=ph.environment(inject))


def _launch(script: Path, resolution: tuple, out: Path, workers: int, steps: int, cuts: list | None = None, shape: tuple | None = None,
            checkpoint: Path | None = None) -> Path:
    return _launch_run(script, resolution, out, workers, steps, cuts, shape, checkpoint).final


def _rejections(run: ph.Launch) -> set[int]:
    """the rejection counts the workers report"""
    return {rejections for _, rejections in run.reports().values()}


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert any(name.endswith("/chi") for name in ys), f"{label}: the reference carries no dye; the gate is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _concentrations(path: Path) -> dict[str, np.ndarray]:
    return {name: data for name, data in _all_levels(path).items() if name.endswith("primitives/chi")}


@ph.local_only
def test_the_fine_level_starts_with_its_parents_dye(tmp_path: Path, variants: Path) -> None:
    out = tmp_path / "single"
    _single("DyedBlast", (32, 32, 1), out, steps=1)
    start = next(p for p in out.glob("*.h5") if "final" not in p.name)
    with h5py.File(start, "r") as f:
        levels = sorted(k for k in f if k.startswith("level_"))
        assert len(levels) == 2, f"the blast publishes {levels}; the fine level is the premise"
        ng0 = int(f["level_0/mesh"].attrs["halo_width"])
        ng1 = int(f["level_1/mesh"].attrs["halo_width"])
        fine = f["level_1/conserved/chi"][()][ng1:-ng1, ng1:-ng1]
        coarse = f["level_0/conserved/chi"][()][ng0:-ng0, ng0:-ng0]
    assert fine.max() > 0.0, "the fine level starts undyed"
    # the prolongation conserves the parent's dye: each covered coarse cell holds the mean of its
    # four children, to the roundoff of a four-term sum (4 u relative, u = 2^-53 rounded up to
    # 4e-16)
    restricted = 0.25 * (fine[0::2, 0::2] + fine[1::2, 0::2] + fine[0::2, 1::2] + fine[1::2, 1::2])
    matches = [
        (j, i)
        for j in range(coarse.shape[0] - restricted.shape[0] + 1)
        for i in range(coarse.shape[1] - restricted.shape[1] + 1)
        if np.allclose(coarse[j:j + restricted.shape[0], i:i + restricted.shape[1]], restricted, rtol=4e-16 * 4, atol=0.0)
    ]
    assert matches, "no coarse window holds the restriction of the fine dye"
    # the donor-cell dye update is a convex combination of upwind concentrations under the step's
    # cfl, so chi stays inside the painted range up to the rounding of the recovery D_chi / rho, a
    # few u relative per step
    for name, chi in _concentrations(_final(out)).items():
        assert chi.min() >= LOW * (1 - 1e-14) and chi.max() <= HIGH * (1 + 1e-14), f"{name} left [{LOW}, {HIGH}]: [{chi.min()}, {chi.max()}]"


@pytest.mark.parametrize("arm", list(ARMS))
def test_tiles_and_launched_workers_carry_the_single_grid_dye(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution = ARMS[arm]
    want = _single(problem, resolution, tmp_path / "single")
    ph.disagreement(_assert_same, _single(problem, resolution, tmp_path / "tiles", gpus=2), want, f"{arm}: two host tiles")
    launched = _launch(_script(variants, problem), resolution, tmp_path / "launched", 2, STEPS, shape=(2, 1))
    ph.disagreement(_assert_same, launched, want, f"{arm}: two launched workers")


@pytest.mark.parametrize("arm", ["refined", "flat"])
def test_a_dyed_launch_resumed_onto_three_workers_reaches_the_single_grid(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution = ARMS[arm]
    want = _single(problem, resolution, tmp_path / "single")
    script = _script(variants, problem)
    first = _launch(script, resolution, tmp_path / "first", 2, STEPS // 2, shape=(2, 1))
    # the step cap is an absolute root iteration: the resume from STEPS // 2 runs to STEPS
    resumed = _launch(script, resolution, tmp_path / "resumed", 3, STEPS, cuts=[[10, 21], []], checkpoint=first)
    ph.disagreement(_assert_same, resumed, want, f"{arm}: resumed onto three workers")


def test_a_late_rejection_rolls_the_dye_back(tmp_path: Path, variants: Path) -> None:
    """the replay after a rejection runs on a reduced step, so the reference is one worker under
    the same rejection; the clean run shows the rejection changed the run"""
    problem, resolution = ARMS["flat"]
    script = _script(variants, problem)
    clean = _launch_run(script, resolution, tmp_path / "clean", 1, STEPS, shape=(1, 1))
    assert _rejections(clean) == {0}, "the clean launch rejected a step"
    # the last stage (rk2) of step 3, on the one worker and on worker 1 of two
    reference = _launch_run(script, resolution, tmp_path / "one", 1, STEPS, shape=(1, 1), inject="reject:0:3:1")
    assert _rejections(reference) == {1}, "the one-worker run did not count its rejection"
    rejected = _launch_run(script, resolution, tmp_path / "two", 2, STEPS, shape=(2, 1), inject="reject:1:3:1")
    assert _rejections(rejected) == {1}, f"the injection never fired: {rejected.reports()}"
    xs, ys = _all_levels(clean.final), _all_levels(reference.final)
    assert not np.array_equal(xs["level_0/conserved/chi"], ys["level_0/conserved/chi"]), "the rejection left the dye unchanged; the gate is vacuous"
    ph.disagreement(_assert_same, rejected.final, reference.final, "two workers rejecting a late stage on worker 1")
