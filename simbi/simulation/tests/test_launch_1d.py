# =============================================================================
# test_launch_1d.py
#
# one-dimensional grids on launched workers: the newtonian and isothermal sod
# shock tubes and the relativistic marti-muller blast. each launched on two
# workers, and resumed onto three uneven workers from a two-worker checkpoint,
# publishes the single grid's state bit for bit; a troubled cell flagged
# against a cut corrects alike on one worker and on three.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_1d.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

pytestmark = pytest.mark.simulation

CELLS = 128
STEPS = 8
# (module, class); every case runs on CELLS cells
CASES = {
    "newtonian": ("simbi_configs.examples.newtonian.sod", "SodProblem"),
    "isothermal": ("simbi_configs.examples.isothermal.isothermal_sod", "IsothermalSod"),
    "relativistic": ("simbi_configs.examples.srhd.marti_muller", "MartiMuller"),
}
# uneven, one cut at the initial discontinuity (cell 64), where a flagged cell's first-order
# redo differs from its high-order update
THREE = [[64, 100]]


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _single(module: str, cls: str, out: Path) -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {cls}\n"
        f"runner.run({cls}(resolution={CELLS}, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode='cpu', max_steps={STEPS})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"{cls} failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(module: str, out: Path, layout: Path, steps: int, checkpoint: Path | None = None, inject: str | None = None) -> Path:
    script = module.replace(".", "/") + ".py"
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", str(CELLS), "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


def _assert_same(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("case", list(CASES))
def test_a_1d_launch_matches_the_single_grid_fresh_resumed_and_corrected(tmp_path: Path, case: str) -> None:
    module, cls = CASES[case]
    want = _single(module, cls, tmp_path / "single")
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2,))
    _assert_same(_launch(module, tmp_path / "two", two, STEPS), want, f"{case}: two workers")
    first = _launch(module, tmp_path / "first", two, STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=THREE)
    _assert_same(_launch(module, tmp_path / "resumed", three, STEPS - STEPS // 2, checkpoint=first), want, f"{case}: resumed onto three")
    # the troubled cell beside the cut at the discontinuity, in the first stage of step 3
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1,))
    inject = "trouble:3:0:63:0"
    clean = _launch(module, tmp_path / "clean", one, STEPS)
    flagged = _launch(module, tmp_path / "flagged", one, STEPS, inject=inject)
    xs, ys = _all_levels(clean), _all_levels(flagged)
    assert any(not np.array_equal(xs[n], ys[n]) for n in xs), "the flag left the run unchanged; the correction arm is vacuous"
    _assert_same(_launch(module, tmp_path / "corrected", three, STEPS, inject=inject), flagged, f"{case}: corrected on three")
