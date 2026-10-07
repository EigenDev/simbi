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
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_dye.py
# =============================================================================

import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

STEPS = 8
LOW, HIGH = 0.25, 1.0
REJECTIONS = re.compile(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)")
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
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from variants import {problem}\n"
        f"runner.run({problem}(resolution={resolution!r}, gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    env = dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1")
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"{problem} on {gpus} device(s) failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch_run(script: Path, resolution: tuple, out: Path, layout: Path, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> tuple[Path, str]:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in resolution), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out), done.stderr


def _launch(script: Path, resolution: tuple, out: Path, layout: Path, steps: int, checkpoint: Path | None = None) -> Path:
    return _launch_run(script, resolution, out, layout, steps, checkpoint)[0]


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert any(name.endswith("/chi") for name in ys), f"{label}: the reference carries no dye; the gate is vacuous"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _concentrations(path: Path) -> dict[str, np.ndarray]:
    return {name: data for name, data in _all_levels(path).items() if name.endswith("primitives/chi")}


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
    _assert_same(_single(problem, resolution, tmp_path / "tiles", gpus=2), want, f"{arm}: two host tiles")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch(_script(variants, problem), resolution, tmp_path / "launched", two, STEPS), want, f"{arm}: two launched workers")


@pytest.mark.parametrize("arm", ["refined", "flat"])
def test_a_dyed_launch_resumed_onto_three_workers_reaches_the_single_grid(tmp_path: Path, variants: Path, arm: str) -> None:
    problem, resolution = ARMS[arm]
    want = _single(problem, resolution, tmp_path / "single")
    script = _script(variants, problem)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    first = _launch(script, resolution, tmp_path / "first", two, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    # the step cap is an absolute root iteration: the resume from STEPS // 2 runs to STEPS
    resumed = _launch(script, resolution, tmp_path / "resumed", three, STEPS, checkpoint=first)
    _assert_same(resumed, want, f"{arm}: resumed onto three workers")


def test_a_late_rejection_rolls_the_dye_back(tmp_path: Path, variants: Path) -> None:
    """the replay after a rejection runs on a reduced step, so the reference is one worker under
    the same rejection; the clean run shows the rejection changed the run"""
    problem, resolution = ARMS["flat"]
    script = _script(variants, problem)
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    clean, clean_log = _launch_run(script, resolution, tmp_path / "clean", one, STEPS)
    assert {int(r) for r in REJECTIONS.findall(clean_log)} == {0}, "the clean launch rejected a step"
    # the last stage (rk2) of step 3, on the one worker and on worker 1 of two
    reference, reference_log = _launch_run(script, resolution, tmp_path / "one", one, STEPS, inject="reject:0:3:1")
    assert {int(r) for r in REJECTIONS.findall(reference_log)} == {1}, "the one-worker run did not count its rejection"
    rejected, log = _launch_run(script, resolution, tmp_path / "two", two, STEPS, inject="reject:1:3:1")
    assert {int(r) for r in REJECTIONS.findall(log)} == {1}, f"the injection never fired: {REJECTIONS.findall(log)}"
    xs, ys = _all_levels(clean), _all_levels(reference)
    assert not np.array_equal(xs["level_0/conserved/chi"], ys["level_0/conserved/chi"]), "the rejection left the dye unchanged; the gate is vacuous"
    _assert_same(rejected, reference, "two workers rejecting a late stage on worker 1")
