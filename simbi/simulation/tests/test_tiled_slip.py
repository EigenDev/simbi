# =============================================================================
# test_tiled_slip.py
#
# a magnetic slip sink on host tiles steps the single grid's palindrome
# D M H M D across the tiles: the drain and the slip act on the finest level,
# the slip solve runs in lockstep over the finest level's tiles with every inner
# product and the heat merged exactly, and H is the collective root step with
# the drain withheld. so, on the 2.5D disk unrefined and refined under both
# closures, with the cuts through the sink and its slip shell:
# - two and three uneven host tiles publish every dataset of every level, the
#   body state and the body history of the single grid bit for bit;
# - a run resumed onto another partition continues the uninterrupted one, its
#   body history the one a single hierarchy resumed from the same file records;
# - the slip acts (the field departs from a magnetically transparent sink).
# on launched workers the unrefined disk under both closures steps as the single
# grid on two and three uneven workers, through the versioned fabric whose
# repeating slip points carry every pass of the solve; a run resumed onto three
# workers continues it, and a stage rejected on one worker replays as the run
# capped at the accepted timestep. the refined disk under both closures steps as
# the single hierarchy on two workers and continues it resumed onto three; on a
# 64^2 root with a patch clear of one worker's tiles, that worker holds no tile
# of the finest level and still joins every exchange and sum of the solve.
#
# run: pytest -m simulation simbi/simulation/tests/test_tiled_slip.py
# =============================================================================
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi.simulation.tests.test_launch import _layout
from simbi_configs.examples.ibm.magnetic_slip_disk_2p5d import MagneticSlipDisk2p5d

pytestmark = pytest.mark.simulation

RES = 32
STEPS = 6


class TransparentDisk(MagneticSlipDisk2p5d):
    """the same disk with no magnetic coupling"""

    @property
    def immersed_bodies(self):
        return [replace(b, magnetic=None) for b in super().immersed_bodies]


@pytest.fixture
def host_tiles(monkeypatch):
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")


def _disk(cls, directory: Path, closure: str, refined: bool = True, fraction: float = 0.875, res: int = RES, **kw):
    # the finest level covers the central seven eighths; the sink of two fine cells with a one-cell
    # shell sits at the center, where the cut at root cell 16 passes
    probe = cls(resolution=(res, res, 1), data_directory=directory)
    probe.setup()
    half = fraction * probe.domain_radius * probe.bondi_radius
    refinement = dict(refinement_enabled=True, refinement_max_levels=2, refinement_ratios=[2], refinement_regions=[[-half, half, -half, half]]) if refined else {}
    return cls(resolution=(res, res, 1), r_acc_scale=2.0, shell_cells=1.0, thermal_closure=closure, data_directory=directory,
               checkpoint_interval=1e9, **refinement, **kw)


def _run(cls, out: Path, closure: str, refined: bool, gpus: int = 1, steps: int = STEPS, fraction: float = 0.875, res: int = RES, **kw) -> Path:
    runner.run(_disk(cls, out, closure, refined, fraction, res, gpus=gpus, **kw), compute_mode="cpu", max_steps=steps)
    (final,) = sorted(out.glob("*final*.h5"))
    return final


def _datasets(path: Path, history: bool = True) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for group in [k for k in f.keys() if k.startswith("level_")] + ["bodies"] + (["body_diagnostics"] if history else []):
            if group in f:
                f[group].visititems(lambda n, obj, g=group: out.__setitem__(f"{g}/{n}", obj[()])
                                    if isinstance(obj, h5py.Dataset) and not n.startswith("mesh") else None)
    return out


def _assert_same(got: Path, want: Path, label: str, history: bool = True) -> None:
    xs, ys = _datasets(got, history), _datasets(want, history)
    assert any(n.startswith("bodies/") for n in ys), f"{label}: the reference carries no bodies"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


ARMS = [(closure, refined) for refined in (False, True) for closure in ("adiabatic", "isothermal")]


@pytest.mark.parametrize(("closure", "refined"), ARMS)
def test_a_slip_sink_on_host_tiles_steps_as_the_single_grid(tmp_path: Path, host_tiles, closure: str, refined: bool) -> None:
    want = _run(MagneticSlipDisk2p5d, tmp_path / "single", closure, refined)
    level = "level_1" if refined else "level_0"
    assert level + "/partition_0/hydro/primitives/b3" in _datasets(want), f"{closure}: the reference lacks {level}"
    transparent = _datasets(_run(TransparentDisk, tmp_path / "transparent", closure, refined))
    assert not np.array_equal(_datasets(want)[f"{level}/partition_0/hydro/primitives/b3"], transparent[f"{level}/partition_0/hydro/primitives/b3"]), \
        f"{closure}: the slip left the field unchanged; the gate is vacuous"
    label = f"{closure}{' refined' if refined else ''}"
    _assert_same(_run(MagneticSlipDisk2p5d, tmp_path / "two", closure, refined, gpus=2), want, f"{label} on two host tiles")
    _assert_same(_run(MagneticSlipDisk2p5d, tmp_path / "three", closure, refined, gpus=3, decompose=[[12, 17], []]), want, f"{label} on three uneven host tiles")


@pytest.mark.parametrize(("closure", "refined"), ARMS)
def test_a_slip_sink_resumed_onto_other_tiles_continues_the_single_grid(tmp_path: Path, host_tiles, closure: str, refined: bool) -> None:
    want = _run(MagneticSlipDisk2p5d, tmp_path / "single", closure, refined)
    first = _run(MagneticSlipDisk2p5d, tmp_path / "first", closure, refined, gpus=2, steps=STEPS // 2)
    resumed = _run(MagneticSlipDisk2p5d, tmp_path / "resumed", closure, refined, gpus=3, decompose=[[12, 17], []], checkpoint_file=str(first))
    # every level and the body state continue the uninterrupted run; the body history a resumed run
    # records covers its own steps, as a resumed single grid's does
    _assert_same(resumed, want, f"{closure} resumed onto three uneven host tiles", history=False)
    single_resumed = _run(MagneticSlipDisk2p5d, tmp_path / "single_resumed", closure, refined, checkpoint_file=str(first))
    _assert_same(resumed, single_resumed, f"{closure} resumed onto three uneven host tiles against one device resumed")


import re
import subprocess as _subprocess

REJECTIONS = re.compile(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)")
LAUNCH_SCRIPT = "from simbi_configs.examples.ibm.magnetic_slip_disk_2p5d import MagneticSlipDisk2p5d as Base\n\n\nclass LaunchedDisk(Base):\n    pass\n"


def _launch_disk(tmp_path: Path, out: Path, layout: Path, closure: str, steps: int = STEPS, checkpoint: Path | None = None,
                 inject: str | None = None) -> tuple[Path, set[int]]:
    script = tmp_path / "launched_disk.py"
    script.write_text(LAUNCH_SCRIPT)
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", f"{RES},{RES},1", "--r-acc-scale", "2.0", "--shell-cells", "1.0", "--thermal-closure", closure,
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    done = _subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"the launch failed:\nstdout:\n{done.stdout[-1500:]}\nstderr:\n{done.stderr[-3000:]}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final, {int(r) for r in REJECTIONS.findall(done.stderr)}


@pytest.mark.parametrize("closure", ["adiabatic", "isothermal"])
def test_an_unrefined_slip_sink_on_launched_workers_steps_as_the_single_grid(tmp_path: Path, closure: str) -> None:
    want = _run(MagneticSlipDisk2p5d, tmp_path / "single", closure, False)
    for tag, workers, kwargs in (("two", 2, {"shape": (2, 1)}), ("three", 3, {"cuts": [[12, 17], []]})):
        got, _ = _launch_disk(tmp_path, tmp_path / tag, _layout(tmp_path / f"{tag}.toml", workers=workers, **kwargs), closure)
        _assert_same(got, want, f"{closure} on {workers} launched workers")


@pytest.mark.parametrize("closure", ["adiabatic", "isothermal"])
def test_an_unrefined_slip_sink_resumed_onto_other_workers_continues_the_single_grid(tmp_path: Path, closure: str) -> None:
    want = _run(MagneticSlipDisk2p5d, tmp_path / "single", closure, False)
    first, _ = _launch_disk(tmp_path, tmp_path / "first", _layout(tmp_path / "two.toml", workers=2, shape=(2, 1)), closure, steps=STEPS // 2)
    resumed, _ = _launch_disk(tmp_path, tmp_path / "resumed", _layout(tmp_path / "three.toml", workers=3, cuts=[[12, 17], []]), closure, checkpoint=first)
    _assert_same(resumed, want, f"{closure} resumed onto three launched workers", history=False)
    single_resumed = _run(MagneticSlipDisk2p5d, tmp_path / "single_resumed", closure, False, checkpoint_file=str(first))
    _assert_same(resumed, single_resumed, f"{closure} resumed onto three launched workers against one device resumed")


def test_a_slip_step_rejected_on_one_worker_replays_as_the_capped_run(tmp_path: Path) -> None:
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    rejecting, counts = _launch_disk(tmp_path, tmp_path / "rejecting", two, "adiabatic", inject="reject:1:2:1")
    assert counts == {1}, f"the rejection was not counted once on every worker: {counts}"
    with h5py.File(rejecting, "r") as f:
        accepted = float(f["body_diagnostics/dt"][2])
    capped, _ = _launch_disk(tmp_path, tmp_path / "capped", two, "adiabatic", inject=f"cap:2:{int(np.float64(accepted).view(np.uint64))}")
    _assert_same(rejecting, capped, "a rejected slip step against the capped control")


def _refined_script(tmp_path: Path, fraction: float, res: int = RES) -> Path:
    probe = _disk(MagneticSlipDisk2p5d, tmp_path, "adiabatic", fraction=fraction, res=res)
    half = probe.refinement_regions[0][1]
    script = tmp_path / f"refined_disk_{int(round(fraction * 1000))}.py"
    script.write_text(
        "from typing import Annotated\n\n"
        "from simbi import ProblemParam\n"
        "from simbi_configs.examples.ibm.magnetic_slip_disk_2p5d import MagneticSlipDisk2p5d\n\n\n"
        "class RefinedDisk(MagneticSlipDisk2p5d):\n"
        "    refinement_enabled: Annotated[bool, ProblemParam(True, description='refined')]\n"
        "    refinement_max_levels: Annotated[int, ProblemParam(2, description='two levels')]\n"
        "    refinement_ratios: Annotated[list[int], ProblemParam([2], description='ratio')]\n"
        f"    refinement_regions: Annotated[list[list[float]], ProblemParam([[{-half!r}, {half!r}, {-half!r}, {half!r}]], description='patch')]\n"
    )
    return script


def _launch_refined(tmp_path: Path, script: Path, out: Path, layout: Path, closure: str, steps: int = STEPS, checkpoint: Path | None = None,
                    res: int = RES) -> Path:
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", f"{res},{res},1", "--r-acc-scale", "2.0", "--shell-cells", "1.0", "--thermal-closure", closure,
           "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = _subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    assert done.returncode == 0, f"the launch failed:\nstdout:\n{done.stdout[-1500:]}\nstderr:\n{done.stderr[-3000:]}"
    (final,) = sorted(out.glob("*final*.h5"))
    return final


@pytest.mark.parametrize("closure", ["adiabatic", "isothermal"])
def test_a_refined_slip_sink_on_launched_workers_steps_as_the_single_hierarchy(tmp_path: Path, closure: str) -> None:
    want = _run(MagneticSlipDisk2p5d, tmp_path / "single", closure, True)
    script = _refined_script(tmp_path, 0.875)
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    _assert_same(_launch_refined(tmp_path, script, tmp_path / "two", two, closure), want, f"refined {closure} on two launched workers")
    first = _launch_refined(tmp_path, script, tmp_path / "first", two, closure, steps=STEPS // 2)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[12, 17], []])
    resumed = _launch_refined(tmp_path, script, tmp_path / "resumed", three, closure, checkpoint=first)
    _assert_same(resumed, want, f"refined {closure} resumed onto three launched workers", history=False)
    single_resumed = _run(MagneticSlipDisk2p5d, tmp_path / "single_resumed", closure, True, checkpoint_file=str(first))
    _assert_same(resumed, single_resumed, f"refined {closure} resumed against one device resumed")


def test_a_worker_holding_no_tile_of_the_finest_level_joins_the_slip_solve(tmp_path: Path) -> None:
    # on a 64^2 root the patch covers root cells 16..48 along x; the cuts at x = 12 and 32 leave
    # worker 0 the tile [0, 12), clear of the patch, and split the patch between the other two
    want = _run(MagneticSlipDisk2p5d, tmp_path / "single", "adiabatic", True, fraction=0.5, res=64)
    with h5py.File(want, "r") as f:
        assert "level_1" in f, "the reference has no finest level"
    script = _refined_script(tmp_path, 0.5, res=64)
    three = _layout(tmp_path / "three.toml", workers=3, cuts=[[12, 32], []])
    got = _launch_refined(tmp_path, script, tmp_path / "three", three, "adiabatic", res=64)
    _assert_same(got, want, "refined disk with a worker holding no finest tile")
