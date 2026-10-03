# =============================================================================
# test_launch_isothermal.py
#
# the isothermal closures through `simbi launch`: isothermal hydro on the
# shear layer and isothermal MHD on the orszag-tang vortex, both globally
# isothermal, against the single-grid run, resumed onto a changed partition,
# and with the correction fired against the cut; a locally isothermal closure
# with refinement is refused by name, since the refined worker levels seed no
# closure field. the flat locally isothermal launch is gated in
# test_locally_isothermal_tiles.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_isothermal.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi_configs.examples.isothermal.imhd_orszag_tang import IsothermalOrszagTang
from simbi_configs.examples.isothermal.isothermal_kh import IsothermalKelvinHelmholtz
from simbi_configs.examples.isothermal.refined_locally_iso import RefinedLocallyIso

from simbi.simulation.tests.test_launch import (
    EXIT,
    MHD_STEPS,
    STEPS,
    _assert_same_state,
    _interior,
    _layout,
    _max_div_b,
    bounded_processes,  # noqa: F401  the autouse process guard
)

pytestmark = pytest.mark.simulation
# -----------------------------------------------------------------------------
# the isothermal closures through the public path: isothermal hydro on the shear layer,
# isothermal MHD on the orszag-tang vortex, both globally isothermal; a locally isothermal
# closure with refinement is refused by name, since the refined worker levels seed no closure
# field.
# -----------------------------------------------------------------------------


def _script_launch(script: str, resolution: tuple[int, ...], out: Path, layout: Path, steps: int, workers: int, checkpoint: Path | None = None, env: dict[str, str] | None = None, flags: list[str] | None = None) -> Path:
    cmd = [
        sys.executable, "-m", "simbi.cli", "launch", script,
        "--layout", str(layout), "--max-steps", str(steps),
        "--resolution", ",".join(str(r) for r in resolution),
        "--data-directory", str(out), "--checkpoint-interval", "1e9",
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    cmd += flags or []
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ, **(env or {})))
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _script_reference(module: str, cls: str, resolution: tuple[int, ...], out: Path, steps: int, fields: str = "") -> Path:
    """the single-grid run of `cls`; `fields` are extra constructor arguments, `nu=0.05`"""
    extra = f", {fields}" if fields else ""
    snippet = f"""
import sys
from simbi.simulation import runner
from {module} import {cls}
runner.run({cls}(resolution={resolution!r}, data_directory=sys.argv[1], checkpoint_interval=1e9{extra}), compute_mode="cpu", max_steps=int(sys.argv[2]))
"""
    done = subprocess.run([sys.executable, "-c", snippet, str(out), str(steps)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


ISO_KH_SCRIPT = str(Path(IsothermalKelvinHelmholtz.__module__.replace(".", "/") + ".py"))
IMHD_OT_SCRIPT = str(Path(IsothermalOrszagTang.__module__.replace(".", "/") + ".py"))
# a two-dimensional locally isothermal problem on a refined mesh
LOCAL_ISO_SCRIPT = str(Path(RefinedLocallyIso.__module__.replace(".", "/") + ".py"))


def test_an_isothermal_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _script_reference(IsothermalKelvinHelmholtz.__module__, "IsothermalKelvinHelmholtz", (32, 32), tmp_path / "single", STEPS)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched = _script_launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "two", layout, STEPS, workers=2)
    _assert_same_state(launched, reference, "two isothermal workers")


def test_an_isothermal_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path) -> None:
    reference = _script_reference(IsothermalKelvinHelmholtz.__module__, "IsothermalKelvinHelmholtz", (32, 32), tmp_path / "single", STEPS)
    layout2 = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    partial = _script_launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "first", layout2, STEPS // 2, workers=2)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    layout3 = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _script_launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "three", layout3, STEPS - written_at, workers=3, checkpoint=partial)
    _assert_same_state(resumed, reference, "isothermal, resumed on three workers")


def test_an_isothermal_launch_with_a_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:15:10"}
    clean = _script_launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "clean", one, STEPS, workers=1)
    reference = _script_launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "one", one, STEPS, workers=1, env=inject)
    launched = _script_launch(ISO_KH_SCRIPT, (32, 32), tmp_path / "two", two, STEPS, workers=2, env=inject)
    with h5py.File(clean, "r") as fc, h5py.File(reference, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the flag left the one-worker run unchanged; the gate is vacuous"
    _assert_same_state(launched, reference, "two isothermal workers, trouble against the cut")


def test_an_isothermal_magnetized_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _script_reference(IsothermalOrszagTang.__module__, "IsothermalOrszagTang", (32, 32, 1), tmp_path / "single", MHD_STEPS)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched = _script_launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "two", layout, MHD_STEPS, workers=2, env={"SIMBI_FABRIC_AUDIT": "all"})
    with h5py.File(reference, "r") as f:
        assert "level_0/partition_0/hydro/magnetic/B1/data" in f
    _assert_same_state(launched, reference, "two isothermal magnetized workers")
    assert _max_div_b(launched) < 1e-10


def test_an_isothermal_magnetized_launch_with_a_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    inject = {"SIMBI_FABRIC_INJECT": "trouble:3:0:15:10", "SIMBI_FABRIC_AUDIT": "all"}
    clean = _script_launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "clean", one, MHD_STEPS, workers=1)
    reference = _script_launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "one", one, MHD_STEPS, workers=1, env=inject)
    launched = _script_launch(IMHD_OT_SCRIPT, (32, 32, 1), tmp_path / "two", two, MHD_STEPS, workers=2, env=inject)
    with h5py.File(clean, "r") as fc, h5py.File(reference, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the flag left the one-worker run unchanged; the gate is vacuous"
    _assert_same_state(launched, reference, "two isothermal magnetized workers, trouble against the cut")
    assert _max_div_b(launched) < 1e-10


def test_a_refined_locally_isothermal_launch_resumed_from_a_checkpoint_is_refused_by_name(tmp_path: Path) -> None:
    # a resumed launch passes the same front door before any tile is built or seeded
    reference = _script_reference(RefinedLocallyIso.__module__, "RefinedLocallyIso", (16, 16, 1), tmp_path / "single", 1)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", LOCAL_ISO_SCRIPT, "--layout", str(layout), "--max-steps", "1", "--resolution", "16,16,1",
           "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9", "--checkpoint", str(reference)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode != 0, f"the resumed launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "a locally isothermal closure with mesh refinement is outside the distributed first release" in flat, done.stderr
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


def test_a_refined_locally_isothermal_launch_is_refused_by_name(tmp_path: Path) -> None:
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [sys.executable, "-m", "simbi.cli", "launch", LOCAL_ISO_SCRIPT, "--layout", str(layout), "--max-steps", "1", "--resolution", "16,16,1", "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    flat = " ".join(done.stderr.split())
    assert "a locally isothermal closure with mesh refinement is outside the distributed first release" in flat, done.stderr
