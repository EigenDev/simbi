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
# the ledger is nonzero in every run compared.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_horizon_ledger.py
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

STEPS = 4
RESOLUTION = (24, 24, 24)
FIELDS = "domain_radius=0.4, excision_radius=1.6"
REJECTIONS = re.compile(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)")

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
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    return env


def _final(out: Path) -> Path:
    (final,) = sorted(out.glob("*final*.h5"))
    return final


def _reference(tmp_path: Path, cls: str, out: Path, steps: int = STEPS) -> Path:
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from gr_ledger_base import {cls}\n"
        f"runner.run({cls}(resolution={RESOLUTION!r}, {FIELDS}, data_directory=sys.argv[1], checkpoint_interval=1e9), compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=900, env=_env(tmp_path))
    assert done.returncode == 0, f"the reference failed:\n{done.stderr[-3000:]}"
    return _final(out)


def _launch(tmp_path: Path, script: Path, out: Path, layout: Path, steps: int = STEPS, checkpoint: Path | None = None,
            inject: str | None = None) -> tuple[subprocess.CompletedProcess, set[int]]:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--domain-radius", "0.4", "--excision-radius", "1.6",
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=_env(tmp_path, inject))
    return done, {int(r) for r in REJECTIONS.findall(done.stderr)}


def _launched(tmp_path: Path, script: Path, out: Path, layout: Path, **kwargs) -> Path:
    done, _ = _launch(tmp_path, script, out, layout, **kwargs)
    assert done.returncode == 0, f"the launch failed:\nstdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-3000:]}"
    return _final(out)


def _ledger(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as f:
        return float(f["bodies/total_accreted_mass"][-1]), float(f["bodies/total_accreted_energy"][-1])


def _assert_same(got: Path, want: Path, label: str) -> None:
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
    for tag, workers, kwargs in (("two", 2, {"cuts": [[12], [], []]}), ("six", 6, {"cuts": [[9, 15], [11], []]})):
        layout = _layout(tmp_path / f"{tag}.toml", workers=workers, **kwargs)
        got = _launched(tmp_path, scripts[case], tmp_path / tag, layout)
        _assert_same(got, want, f"{case} on {workers} workers")


@pytest.mark.parametrize("case", ["flat", "refined"])
def test_the_launched_horizon_ledger_continues_across_a_restart(tmp_path: Path, scripts: dict[str, Path], case: str) -> None:
    whole = _reference(tmp_path, CASES[case], tmp_path / "single")
    two = _layout(tmp_path / "two.toml", workers=2, cuts=[[12], [], []])
    first = _launched(tmp_path, scripts[case], tmp_path / "first", two, steps=STEPS // 2)
    assert _ledger(first)[0] != 0.0, "the first segment booked nothing; the restart carries no ledger"
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[9, 15], [], []])
    resumed = _launched(tmp_path, scripts[case], tmp_path / "resumed", three, checkpoint=first)
    _assert_same(resumed, whole, f"{case} resumed onto three workers")


@pytest.mark.parametrize(("case", "inject"), [("flat", "reject:1:2:1"), ("refined", "reject_level:1:2:1:0:1")])
def test_a_rejection_books_nothing_of_the_rejected_attempt(tmp_path: Path, scripts: dict[str, Path], case: str, inject: str) -> None:
    two = _layout(tmp_path / "two.toml", workers=2, cuts=[[12], [], []])
    done, counts = _launch(tmp_path, scripts[case], tmp_path / "rejecting", two, inject=inject)
    assert done.returncode == 0, f"the rejecting launch failed:\n{done.stderr[-3000:]}"
    assert counts == {1}, f"the rejection was not counted once on every worker: {counts}"
    rejecting = _final(tmp_path / "rejecting")
    with h5py.File(rejecting, "r") as f:
        accepted = float(f["body_diagnostics/dt"][2])
    capped = _launched(tmp_path, scripts[case], tmp_path / "capped", two, inject=f"cap:2:{int(np.float64(accepted).view(np.uint64))}")
    assert _ledger(capped)[0] != 0.0, "the capped control booked nothing"
    _assert_same(rejecting, capped, f"{case} rejection against the capped control")


def test_a_refined_patch_across_the_shell_is_refused_on_launch_by_name(tmp_path: Path, scripts: dict[str, Path]) -> None:
    two = _layout(tmp_path / "two.toml", workers=2, cuts=[[12], [], []])
    done, _ = _launch(tmp_path, scripts["across"], tmp_path / "out", two)
    assert done.returncode != 0, "the launch across the shell ran"
    assert "meets the horizon ledger's diagnostic shell" in " ".join((done.stdout + done.stderr).split()), done.stderr[-2000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a refused launch published a checkpoint"
