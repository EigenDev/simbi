# =============================================================================
# test_launch_1d.py
#
# one-dimensional grids on launched workers: the newtonian and isothermal sod
# shock tubes and the relativistic marti-muller blast. each launched on two
# workers, and resumed onto three uneven workers from a two-worker checkpoint,
# publishes the single grid's state bit for bit; a troubled cell flagged
# against a cut corrects alike on one worker and on three. a sod tube with a
# fine box over the discontinuity publishes the single hierarchy's levels bit
# for bit on two workers and resumed onto three uneven ones. the single-grid
# references and the launches run on the session's backend and placement
# (parity_harness).
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_1d.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================

import os
from pathlib import Path

import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
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
    """the single grid of `cls` on the session's backend; its final checkpoint"""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {cls}\n"
        f"runner.run({cls}(resolution={CELLS}, checkpoint_interval=1e9, data_directory=sys.argv[1]), compute_mode=sys.argv[2], max_steps={STEPS})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{cls} on one device", timeout=600)
    return _final(out)


def _launch(script: str | Path, out: Path, steps: int, workers: int, cuts=None, shape=None, checkpoint: Path | None = None,
            inject: str | None = None) -> Path:
    """the launch of `script` on the session's backend and placement without the experimental
    opt-in; its final checkpoint"""
    args = ["--max-steps", str(steps), "--resolution", str(CELLS), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    ph.launch(script, out, workers, cuts=cuts, shape=shape, args=args, env=ph.environment(inject), timeout=600)
    return _final(out)


def _module_script(module: str) -> str:
    return module.replace(".", "/") + ".py"


def _assert_same_levels(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def _assert_same(got: Path, want: Path, label: str) -> None:
    """every level equal bit for bit, a difference raised as a NumericalDisagreement"""
    ph.disagreement(_assert_same_levels, got, want, f"{label}, {ph.BACKEND}")


@pytest.mark.parametrize("case", list(CASES))
def test_a_1d_launch_matches_the_single_grid_fresh_resumed_and_corrected(tmp_path: Path, case: str) -> None:
    module, cls = CASES[case]
    want = _single(module, cls, tmp_path / "single")
    script = _module_script(module)
    two = dict(workers=2, shape=(2,))
    _assert_same(_launch(script, tmp_path / "two", STEPS, **two), want, f"{case}: two workers")
    first = _launch(script, tmp_path / "first", STEPS // 2, **two)
    three = dict(workers=3, cuts=THREE)
    # the cap is absolute: the resume from root iteration STEPS // 2 runs to STEPS
    _assert_same(_launch(script, tmp_path / "resumed", STEPS, **three, checkpoint=first), want, f"{case}: resumed onto three")
    # the troubled cell beside the cut at the discontinuity, in the first stage of step 3
    one = dict(workers=1, shape=(1,))
    inject = "trouble:3:0:63:0"
    clean = _launch(script, tmp_path / "clean", STEPS, **one)
    flagged = _launch(script, tmp_path / "flagged", STEPS, **one, inject=inject)
    xs, ys = _all_levels(clean), _all_levels(flagged)
    assert any(not np.array_equal(xs[n], ys[n]) for n in xs), "the flag left the run unchanged; the correction arm is vacuous"
    _assert_same(_launch(script, tmp_path / "corrected", STEPS, **three, inject=inject), flagged, f"{case}: corrected on three")


REFINED = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.sod import SodProblem


class RefinedSod(SodProblem):
    """the sod tube with one fine box over the initial discontinuity"""

    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(2, description="coarse + 1 fine")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([[0.375, 0.625]], description="fine box")]
    refinement_ratios: Annotated[list[int], ProblemParam([2], description="ratio")]
'''


def test_a_refined_1d_launch_matches_the_single_hierarchy_fresh_and_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "refined_variants.py").write_text(REFINED)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    want = _single("refined_variants", "RefinedSod", tmp_path / "single")
    assert any(name.startswith("level_1/") for name in _all_levels(want)), "the single run published no fine level; the refined arm is vacuous"
    script = tmp_path / "launched_refined_sod.py"
    script.write_text("from refined_variants import RefinedSod as Base\n\n\nclass LaunchedRefinedSod(Base):\n    pass\n")
    two = dict(workers=2, shape=(2,))
    _assert_same(_launch(script, tmp_path / "two", STEPS, **two), want, "refined sod: two workers")
    first = _launch(script, tmp_path / "first", STEPS // 2, **two)
    three = dict(workers=3, cuts=THREE)
    # the cap is absolute: the resume from root iteration STEPS // 2 runs to STEPS
    _assert_same(_launch(script, tmp_path / "resumed", STEPS, **three, checkpoint=first), want, "refined sod: resumed onto three uneven workers")
