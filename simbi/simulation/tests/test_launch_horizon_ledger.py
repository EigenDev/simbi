# =============================================================================
# test_launch_horizon_ledger.py
#
# an excised kerr-schild box asked for its horizon ledger (`horizon_ledger`) on
# launched workers. the run carries the horizon body, whose ledger the root
# books once per accepted root step: every worker merges its tiles' exact shell
# partials, the workers' partials merge into the whole grid's, and every tile
# books the rates rounded once. these gates establish the ledger's accounting
# across drivers; no immersed body acts on relativistic gas (a declared body is
# refused, test_relativistic_bodies). so, against the single grid (or hierarchy):
# - two workers and four uneven workers cut through the shell publish the same
#   grid, bodies and horizon ledger bit for bit, unrefined and refined;
# - a run resumed onto another partition continues the ledger and the grid as
#   the uninterrupted run does;
# - a step rejected on one worker books nothing of the rejected attempt: the
#   run equals the clean run capped at the accepted timestep;
# - a refined patch across the diagnostic shell is refused by name before any
#   output.
# the ledger is nonzero in every run compared. the single grid and the launches
# run on the session's backend and placement (parity_harness); a launched
# worker on gpu device memory refuses the horizon body, which the harness
# records as an admission refusal.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_horizon_ledger.py
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

STEPS = 4
RESOLUTION = (24, 24, 24)
FIELDS = "domain_radius=0.4, excision_radius=1.6"

BASE = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.grhd.gr_bondi_cartesian import GrBondiCartesian


class WithLedger(GrBondiCartesian):
    """the excised kerr-schild bondi box booking its horizon ledger"""

    horizon_ledger: Annotated[bool, ProblemParam(True, description="ledger")]


class RefinedWithLedger(WithLedger):
    """a patch over the corner box from r = 5.2 M, outside the diagnostic shell's band"""

    def setup(self) -> None:
        super().setup()
        ll = self.bounds[0][1]
        self.refinement_enabled = True
        self.refinement_max_levels = 2
        self.refinement_ratios = [2]
        self.refinement_regions = [[0.5 * ll, 5.0 * ll / 6.0] * 3]


class RefinedAcrossShell(RefinedWithLedger):
    """the patch over the corner box from r = 3.5 M, across the diagnostic shell at r = 3 M"""

    def setup(self) -> None:
        super().setup()
        ll = self.bounds[0][1]
        self.refinement_regions = [[ll / 3.0, 0.75 * ll] * 3]
'''

CASES = {"flat": "WithLedger", "refined": "RefinedWithLedger", "across": "RefinedAcrossShell"}


@pytest.fixture
def scripts(tmp_path: Path) -> dict[str, Path]:
    (tmp_path / "gr_ledger_base.py").write_text(BASE)
    out = {}
    for case, cls in CASES.items():
        path = tmp_path / f"launched_{case}.py"
        path.write_text(f"from gr_ledger_base import {cls} as Base\n\n\nclass Launched{cls}(Base):\n    pass\n")
        out[case] = path
    return out


def _env(tmp_path: Path, inject: str | None = None) -> dict[str, str]:
    """this process's environment without the experimental opt-in, the fixtures importable, and the
    named fault injected"""
    env = ph.environment(inject)
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")])
    return env


def _final(out: Path) -> Path:
    (final,) = sorted(out.glob("*final*.h5"))
    return final


def _reference(tmp_path: Path, cls: str, out: Path, steps: int = STEPS) -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from gr_ledger_base import {cls}\n"
        f"runner.run({cls}(resolution={RESOLUTION!r}, {FIELDS}, data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out)), "the reference", env=_env(tmp_path))
    return _final(out)


def _launch(tmp_path: Path, script: Path, out: Path, workers: int, cuts, steps: int = STEPS, checkpoint: Path | None = None,
            inject: str | None = None, expect_ok: bool = True) -> ph.Launch:
    """the launch on the session's backend and placement; with `expect_ok` every worker exits zero"""
    args = ["--max-steps", str(steps), "--resolution", ",".join(str(r) for r in RESOLUTION), "--domain-radius", "0.4",
            "--excision-radius", "1.6", "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return ph.launch(script, out, workers, cuts=cuts, args=args, env=_env(tmp_path, inject), expect_ok=expect_ok)


def _launched(tmp_path: Path, script: Path, out: Path, workers: int, cuts, **kwargs) -> Path:
    return _launch(tmp_path, script, out, workers, cuts, **kwargs).final


def _ledger(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as f:
        return float(f["bodies/total_accreted_mass"][-1]), float(f["bodies/total_accreted_energy"][-1])


def _assert_same(got: Path, want: Path, label: str) -> None:
    """every level dataset and the bodies group bit for bit, a difference raised as a
    NumericalDisagreement"""
    ph.disagreement(_assert_same_run, got, want, label)


def _assert_same_run(got: Path, want: Path, label: str) -> None:
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    with h5py.File(got, "r") as a, h5py.File(want, "r") as b:
        for name in b["bodies"]:
            np.testing.assert_array_equal(a["bodies"][name][()], b["bodies"][name][()], err_msg=f"{label}: bodies/{name}")


@pytest.mark.parametrize("case", ["flat", "refined"])
def test_an_excised_box_with_its_ledger_launches_as_the_single_grid(tmp_path: Path, scripts: dict[str, Path], case: str) -> None:
    want = _reference(tmp_path, CASES[case], tmp_path / "single")
    mass, energy = _ledger(want)
    assert mass != 0.0 and energy != 0.0, f"the reference booked ({mass}, {energy}); its ledger is empty"
    # x = 12 cuts the shell at the box center; 9 and 15 cut it twice around a six-cell tile, y = 11 once more
    for tag, workers, cuts in (("two", 2, [[12], [], []]), ("six", 6, [[9, 15], [11], []])):
        got = _launched(tmp_path, scripts[case], tmp_path / tag, workers, cuts)
        _assert_same(got, want, f"{case} on {workers} workers")


@pytest.mark.parametrize("case", ["flat", "refined"])
def test_the_launched_horizon_ledger_continues_across_a_restart(tmp_path: Path, scripts: dict[str, Path], case: str) -> None:
    whole = _reference(tmp_path, CASES[case], tmp_path / "single")
    first = _launched(tmp_path, scripts[case], tmp_path / "first", 2, [[12], [], []], steps=STEPS // 2)
    assert _ledger(first)[0] != 0.0, "the first segment booked nothing; the restart carries no ledger"
    resumed = _launched(tmp_path, scripts[case], tmp_path / "resumed", 3, [[9, 15], [], []], checkpoint=first)
    _assert_same(resumed, whole, f"{case} resumed onto three workers")


@pytest.mark.parametrize(("case", "inject"), [("flat", "reject:1:2:1"), ("refined", "reject_level:1:2:1:0:1")])
def test_a_rejection_books_nothing_of_the_rejected_attempt(tmp_path: Path, scripts: dict[str, Path], case: str, inject: str) -> None:
    two = [[12], [], []]
    run = _launch(tmp_path, scripts[case], tmp_path / "rejecting", 2, two, inject=inject)
    counts = {r for _, r in run.reports().values()}
    assert counts == {1}, f"the rejection was not counted once on every worker: {counts}"
    rejecting = run.final
    with h5py.File(rejecting, "r") as f:
        accepted = float(f["body_diagnostics/dt"][2])
    capped = _launched(tmp_path, scripts[case], tmp_path / "capped", 2, two, inject=f"cap:2:{int(np.float64(accepted).view(np.uint64))}")
    assert _ledger(capped)[0] != 0.0, "the capped control booked nothing"
    _assert_same(rejecting, capped, f"{case} rejection against the capped control")


def test_a_refined_patch_across_the_shell_is_refused_on_launch_by_name(tmp_path: Path, scripts: dict[str, Path]) -> None:
    done = _launch(tmp_path, scripts["across"], tmp_path / "out", 2, [[12], [], []], expect_ok=False)
    assert done.returncode != 0, "the launch across the shell ran"
    assert "meets the horizon ledger's diagnostic shell" in " ".join((done.stdout + done.stderr).split()), done.stderr[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a refused launch published a checkpoint"
