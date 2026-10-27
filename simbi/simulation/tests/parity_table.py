# =============================================================================
# parity_table.py
#
# the table-driven distributed parity gates, shared by every family module. a
# family module holds a table of Feature entries (one single-device problem
# class per entry, written beside the test as a module) and calls
# install(globals(), FEATURES), which defines in that module, on the session's
# backend and placement (parity_harness):
# - test_tiles_and_launches_equal_one_device_at_every_checkpoint: two host
#   tiles, two workers on an even cut and three on an uneven cut equal the
#   single device at every checkpoint: every level, the bodies group, the body
#   history and the body diagnostics file
# - test_a_launch_resumed_onto_uneven_workers_equals_one_device (entries with
#   restart): half the steps on two workers, the rest on three uneven workers,
#   against the single device resumed from the same checkpoint (every dataset)
#   and the uninterrupted single device (every level and the bodies state)
# - test_a_rejected_stage_replays_as_the_capped_clean_launch (entries with a
#   rejection): a stage rejected on one worker replays as the clean launch
#   capped at the accepted timestep
# - test_the_setting_changes_the_single_device_run (entries with a control or a
#   premise): the single device with the setting differs from the control (the
#   same fixture without it), and the entry's premise check holds, so a gate
#   whose setting never reaches the run reports it
#
# usage:
#   from simbi.simulation.tests.parity_table import Feature, install
#   FEATURES = {"name": Feature(SOURCE, "Problem", resolution, even, uneven, rejection, control=(CONTROL, "Control"))}
#   install(globals(), FEATURES)
# =============================================================================
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch_bodies import _assert_same, _body_groups
from simbi.simulation.tests.test_launch_refined_numerics import _all_levels

STEPS = 6
REJECTED_STEP = 2


@dataclass(frozen=True)
class Feature:
    source: str
    problem: str
    resolution: tuple | None
    even: list
    uneven: list
    # the injected rejection (worker, step, [level, substep,] stage), or None where the configuration
    # has no rejection replay to gate
    rejection: str | None
    opt_in: bool = False
    restart: bool = True
    # (source, problem): the same fixture without the setting under test
    control: tuple[str, str] | None = None
    # a check of the single device's final checkpoint (and the control's, when the entry has one)
    premise: Callable[[Path, Path | None], None] | None = None


def workers_of(cuts: list) -> int:
    return int(np.prod([len(axis) + 1 for axis in cuts]))


def run_snippet(f: Feature, problem: str, module: str, fields: str, steps: int, checkpoint: Path | None) -> str:
    if f.resolution is not None:
        fields += f", resolution={f.resolution!r}"
    if checkpoint is not None:
        fields += f", checkpoint_file={str(checkpoint)!r}"
    return (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {module} import {problem}\n"
        f"p = {problem}(data_directory=sys.argv[1]{fields})\n"
        "p.checkpoint_interval = 1e-9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )


def files_of(out: Path) -> dict[str, Path]:
    files = {p.name: p for p in out.glob("*.h5")}
    if not any("final" in name for name in files):
        raise ph.HarnessFailure(f"no final checkpoint in {out}")
    return files


def final_of(files: dict[str, Path]) -> Path:
    return [p for name, p in files.items() if "final" in name][0]


def single(f: Feature, script: Path, out: Path, steps: int = STEPS, checkpoint: Path | None = None,
           problem: str | None = None) -> dict[str, Path]:
    """the single device on the session's backend, a checkpoint at every step"""
    snippet = run_snippet(f, problem or f.problem, script.stem, ", gpus=1", steps, checkpoint)
    ph.python(ph.snippet_args(snippet, str(out)), "single device")
    return files_of(out)


def tiles(f: Feature, script: Path, out: Path, cuts: list) -> dict[str, Path]:
    """in-process host tiles cut at `cuts`, a checkpoint at every step"""
    n = workers_of(cuts)
    snippet = run_snippet(f, f.problem, script.stem, f", gpus={n}, decompose={cuts!r}", STEPS, None)
    ph.python(ph.snippet_args(snippet, str(out)), "host tiles", devices=n)
    return files_of(out)


def launch(f: Feature, script: Path, out: Path, cuts: list, steps: int = STEPS, checkpoint: Path | None = None,
           inject: str | None = None) -> ph.Launch:
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e-9"]
    if f.resolution is not None:
        args += ["--resolution", ",".join(str(r) for r in f.resolution)]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    return ph.launch(script, out, workers=workers_of(cuts), cuts=cuts, args=args, env=ph.environment(inject, opt_in=f.opt_in))


def assert_same_levels_and_bodies(got: Path, want: Path, label: str) -> None:
    if _body_groups(want):
        _assert_same(got, want, label)
        return
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def assert_same_levels_and_state(got: Path, want: Path, label: str) -> None:
    """every level, and the bodies group when the reference carries one"""
    xs, ys = _all_levels(got), _all_levels(want)
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")
    bx = {k: v for k, v in _body_groups(got).items() if k.startswith("bodies/")}
    by = {k: v for k, v in _body_groups(want).items() if k.startswith("bodies/")}
    assert bx.keys() == by.keys(), f"{label}: body datasets {sorted(set(bx) ^ set(by))} differ"
    for name, x in bx.items():
        np.testing.assert_array_equal(x, by[name], err_msg=f"{label}: {name}")


def same_run(got: dict[str, Path], want: dict[str, Path], got_dir: Path, want_dir: Path, label: str) -> None:
    """every checkpoint's levels and bodies, and the body diagnostics file, exactly"""
    if sorted(got) != sorted(want):
        raise ph.NumericalDisagreement(f"{label}: checkpoints {sorted(got)} against {sorted(want)}")
    for name in sorted(got):
        ph.disagreement(assert_same_levels_and_bodies, got[name], want[name], f"{label}: {name}")
    a, b = got_dir / "diagnostics.dat", want_dir / "diagnostics.dat"
    if b.exists() != a.exists() or (b.exists() and a.read_text() != b.read_text()):
        raise ph.NumericalDisagreement(f"{label}: the body diagnostics files differ")


def differs(got: Path, want: Path) -> bool:
    """whether two final checkpoints differ in any level or body dataset"""
    xs, ys = _all_levels(got), _all_levels(want)
    if xs.keys() != ys.keys():
        return True
    if any(not np.array_equal(x, ys[k]) for k, x in xs.items()):
        return True
    bx, by = _body_groups(got), _body_groups(want)
    return bx.keys() != by.keys() or any(not np.array_equal(x, by[k]) for k, x in bx.items())


def install(namespace: dict, features: dict[str, Feature]) -> None:
    """define the table's fixture and gates in the family module whose globals are `namespace`"""

    @pytest.fixture
    def feature(request, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Feature, Path]:
        f = features[request.param]
        module = f"feature_{request.param.replace('-', '_')}"
        script = tmp_path / f"{module}.py"
        script.write_text(f.source)
        if f.control is not None:
            (tmp_path / f"{module}_control.py").write_text(f.control[0])
        monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
        return f, script

    @pytest.mark.parametrize("feature", list(features), indirect=True)
    def test_tiles_and_launches_equal_one_device_at_every_checkpoint(tmp_path: Path, feature) -> None:
        f, script = feature
        reference = single(f, script, tmp_path / "single")
        assert len(reference) >= 4, f"{len(reference)} checkpoints; the cadence is not exercised"
        tiled = tiles(f, script, tmp_path / "tiles", f.even)
        same_run(tiled, reference, tmp_path / "tiles", tmp_path / "single", "two host tiles")
        for label, cuts in (("two workers", f.even), ("three uneven workers", f.uneven)):
            out = tmp_path / label.replace(" ", "-")
            run = launch(f, script, out, cuts)
            assert run.reports() == {w: (STEPS, 0) for w in range(workers_of(cuts))}, f"{label}: (steps, rejections) per worker {run.reports()}"
            same_run(run.files, reference, out, tmp_path / "single", label)

    @pytest.mark.parametrize("feature", [k for k, f in features.items() if f.restart], indirect=True)
    def test_a_launch_resumed_onto_uneven_workers_equals_one_device(tmp_path: Path, feature) -> None:
        f, script = feature
        reference = single(f, script, tmp_path / "single")
        first = launch(f, script, tmp_path / "first", f.even, steps=STEPS // 2)
        with h5py.File(first.final, "r") as h:
            assert int(h["level_0"].attrs["iteration"]) == STEPS // 2
        resumed = launch(f, script, tmp_path / "resumed", f.uneven, checkpoint=first.final)
        assert resumed.reports() == {w: (STEPS - STEPS // 2, 0) for w in range(workers_of(f.uneven))}, (
            f"(steps, rejections) per worker {resumed.reports()}"
        )
        resumed_single = single(f, script, tmp_path / "single-resumed", checkpoint=first.final)
        same_run({"final": resumed.final}, {"final": final_of(resumed_single)}, tmp_path / "resumed", tmp_path / "single-resumed",
                 "resumed onto three uneven workers against the resumed single device")
        ph.disagreement(assert_same_levels_and_state, resumed.final, final_of(reference),
                        "resumed onto three uneven workers against the uninterrupted single device")

    @pytest.mark.parametrize("feature", [k for k, f in features.items() if f.rejection is not None], indirect=True)
    def test_a_rejected_stage_replays_as_the_capped_clean_launch(tmp_path: Path, feature) -> None:
        f, script = feature
        steps = REJECTED_STEP + 1
        rejected = launch(f, script, tmp_path / "rejected", f.even, steps=steps, inject=f.rejection)
        assert rejected.reports() == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {rejected.reports()}: a worker kept out of the rollback"
        clean = launch(f, script, tmp_path / "clean", f.even, steps=steps)
        assert clean.reports() == {0: (steps, 0), 1: (steps, 0)}
        with h5py.File(rejected.final, "r") as fr, h5py.File(clean.final, "r") as fc:
            accepted = float(fr["level_0"].attrs["dt"])
            assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
        capped = launch(f, script, tmp_path / "capped", f.even, steps=steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
        assert capped.reports() == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
        with h5py.File(capped.final, "r") as h:
            assert float(h["level_0"].attrs["dt"]) == accepted, "the capped control's step took another timestep"
        same_run({"final": rejected.final}, {"final": capped.final}, tmp_path / "rejected", tmp_path / "capped",
                 "rejected stage against the capped clean launch")

    @pytest.mark.parametrize("feature", [k for k, f in features.items() if f.control is not None or f.premise is not None], indirect=True)
    def test_the_setting_changes_the_single_device_run(tmp_path: Path, feature) -> None:
        f, script = feature
        mine = final_of(single(f, script, tmp_path / "single"))
        control = None
        if f.control is not None:
            control_script = script.with_name(f"{script.stem}_control.py")
            control = final_of(single(f, control_script, tmp_path / "control", problem=f.control[1]))
            assert differs(mine, control), "the setting leaves the single-device run unchanged; the gate is vacuous"
        if f.premise is not None:
            f.premise(mine, control)

    namespace.update({
        "feature": feature,
        "test_tiles_and_launches_equal_one_device_at_every_checkpoint": test_tiles_and_launches_equal_one_device_at_every_checkpoint,
        "test_a_launch_resumed_onto_uneven_workers_equals_one_device": test_a_launch_resumed_onto_uneven_workers_equals_one_device,
        "test_a_rejected_stage_replays_as_the_capped_clean_launch": test_a_rejected_stage_replays_as_the_capped_clean_launch,
        "test_the_setting_changes_the_single_device_run": test_the_setting_changes_the_single_device_run,
    })
