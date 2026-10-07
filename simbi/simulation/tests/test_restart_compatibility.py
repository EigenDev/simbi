# =============================================================================
# test_restart_compatibility.py
#
# a checkpoint resumes only onto the grid it was written on. every loader, the
# single grid's and the partitioned one host tiles and launched workers read
# through, compares each level's cells and comoving region with the run's and
# refuses a mismatch by name before any output:
# - a blast written on [-0.5, 0.5]^2 resumed with bounds [-2, 2]^2 (same cell
#   count, four times the spacing);
# - a refined blast whose fine region moved by two root cells;
# - a three-level checkpoint resumed by a two-level refined run on host tiles
#   (the extra level would be dropped).
# the same resume onto the grid it was written on is admitted on every driver.
#
# run: pytest -m simulation simbi/simulation/tests/test_restart_compatibility.py
# =============================================================================

import os
import subprocess
import sys
from pathlib import Path

import pytest

from simbi.simulation.tests.test_launch import _layout, bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

MODULE = "simbi_configs.examples.newtonian.refined_blast"
SCRIPT = MODULE.replace(".", "/") + ".py"
FLAT = "refinement_enabled=False"
NESTED = ("refinement_max_levels=3, refinement_regions=[[-0.25, 0.25, -0.25, 0.25], [-0.125, 0.125, -0.125, 0.125]], "
          "refinement_ratios=[2, 2]")


def _run(out: Path, gpus: int, fields: str, steps: int = 2, checkpoint: Path | None = None) -> subprocess.CompletedProcess:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {MODULE} import RefinedBlast\n"
        f"runner.run(RefinedBlast(resolution=(16, 16, 1), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1], {fields}{resume}), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))


def _written(out: Path, fields: str) -> Path:
    done = _run(out, 1, fields)
    assert done.returncode == 0, f"the checkpoint run failed:\n{done.stderr[-2000:]}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _assert_refused(done: subprocess.CompletedProcess, out: Path, reason: str, label: str) -> None:
    assert done.returncode != 0, f"{label}: the resume succeeded:\n{done.stdout[-1500:]}"
    flat = " ".join((done.stdout + done.stderr).split())
    assert reason in flat, f"{label}: refusal not named:\n{flat[-2500:]}"
    assert not list(out.glob("*final*.h5")), f"{label}: a refused resume published a final checkpoint"


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
def test_a_checkpoint_resumed_onto_other_bounds_is_refused(tmp_path: Path, gpus: int) -> None:
    written = _written(tmp_path / "first", FLAT)
    # the checkpoint's two iterations and one more on the single grid, two more on host tiles
    steps = 3 if gpus == 1 else 4
    done = _run(tmp_path / "resumed", gpus, f"{FLAT}, bounds=[(-2.0, 2.0), (-2.0, 2.0)]", steps=steps, checkpoint=written)
    _assert_refused(done, tmp_path / "resumed", "level 0 axis 0 spans", f"other bounds on {gpus} device(s)")


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
def test_a_checkpoint_resumed_onto_a_moved_fine_region_is_refused(tmp_path: Path, gpus: int) -> None:
    written = _written(tmp_path / "first", "refinement_regions=[[-0.25, 0.25, -0.25, 0.25]]")
    # the checkpoint's two iterations and one more on the single grid, two more on host tiles
    steps = 3 if gpus == 1 else 4
    # two root cells (dx = 1/16) to the left: the same fine cell count over another region
    done = _run(tmp_path / "resumed", gpus, "refinement_regions=[[-0.375, 0.125, -0.25, 0.25]]", steps=steps, checkpoint=written)
    _assert_refused(done, tmp_path / "resumed", "level 1 axis 0 spans", f"moved fine region on {gpus} device(s)")


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
def test_a_deeper_checkpoint_is_refused_rather_than_truncated(tmp_path: Path, gpus: int) -> None:
    written = _written(tmp_path / "first", NESTED)
    # the checkpoint's two iterations and one more on the single grid, two more on host tiles
    steps = 3 if gpus == 1 else 4
    done = _run(tmp_path / "resumed", gpus, "refinement_regions=[[-0.25, 0.25, -0.25, 0.25]]", steps=steps, checkpoint=written)
    reason = "carries 3 refinement level(s) but this run builds 2" if gpus == 1 else "the checkpoint holds 3 level(s), this run refines to 2"
    _assert_refused(done, tmp_path / "resumed", reason, f"deeper checkpoint on {gpus} device(s)")


def test_a_launch_resumed_onto_other_bounds_is_refused(tmp_path: Path) -> None:
    written = _written(tmp_path / "first", FLAT)
    script = tmp_path / "flat_blast.py"
    script.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        f"from {MODULE} import RefinedBlast\n\n\n"
        "class WideBlast(RefinedBlast):\n"
        "    refinement_enabled: Annotated[bool, ProblemParam(False, description='flat')]\n"
        "    bounds: Annotated[list, ProblemParam([(-2.0, 2.0), (-2.0, 2.0)], description='box')]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    # the checkpoint's two iterations and two more
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "4",
           "--resolution", "16,16,1", "--data-directory", str(tmp_path / "launched"), "--checkpoint-interval", "1e9",
           "--checkpoint", str(written)]
    env = dict(os.environ)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    _assert_refused(done, tmp_path / "launched", "level 0 axis 0 spans", "launch onto other bounds")


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
def test_a_checkpoint_resumes_onto_its_own_grid(tmp_path: Path, gpus: int) -> None:
    written = _written(tmp_path / "first", "refinement_regions=[[-0.25, 0.25, -0.25, 0.25]]")
    # the checkpoint's two iterations and one more
    done = _run(tmp_path / "resumed", gpus, "refinement_regions=[[-0.25, 0.25, -0.25, 0.25]]", steps=3, checkpoint=written)
    assert done.returncode == 0, f"the resume onto the written grid failed on {gpus} device(s):\n{done.stderr[-2000:]}"
    assert list((tmp_path / "resumed").glob("*final*.h5")), "the resume published no final checkpoint"


ISOTHERMAL = "simbi_configs.examples.isothermal.isothermal_channel"


def _iso(out: Path, gpus: int, extra: str, steps: int, checkpoint: Path | None = None) -> subprocess.CompletedProcess:
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from {ISOTHERMAL} import IsothermalChannel\n"
        f"runner.run(IsothermalChannel(resolution=(32, 16), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1]{extra}{resume}), "
        f"compute_mode='cpu', max_steps={steps})\n"
    )
    return subprocess.run([sys.executable, "-c", snippet, str(out)], capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))


def _final(out: Path) -> Path:
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
def test_a_resumed_isothermal_run_keeps_its_recorded_sound_speed(tmp_path: Path, gpus: int) -> None:
    # the file records cs = 1. a class whose default is cs = 3 resumes at the recorded closure; an
    # explicit cs = 3 on the resume is refused by name, since it cannot be honored
    import h5py

    assert _iso(tmp_path / "first", 1, "", 2).returncode == 0
    first = _final(tmp_path / "first")
    with h5py.File(first, "r") as f:
        assert float(f["metadata"].attrs["sound_speed"]) == 1.0
    # the checkpoint's two iterations and two more
    steps = 4
    script = tmp_path / "hot_channel.py"
    script.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        f"from {ISOTHERMAL} import IsothermalChannel\n\n\n"
        "class HotChannel(IsothermalChannel):\n"
        "    sound_speed: Annotated[float, ProblemParam(3.0, cli=True, description='cs')]\n"
    )
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from simbi.simulation import runner\n"
        "from hot_channel import HotChannel\n"
        f"runner.run(HotChannel(resolution=(32, 16), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1], checkpoint_file={str(first)!r}), compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(tmp_path / "resumed")], capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    assert done.returncode == 0, f"the resume failed:\n{done.stderr[-2000:]}"
    with h5py.File(_final(tmp_path / "resumed"), "r") as f:
        assert float(f["metadata"].attrs["sound_speed"]) == 1.0, "the resume changed the recorded closure"
    explicit = _iso(tmp_path / "explicit", gpus, ", sound_speed=3.0", steps, checkpoint=first)
    _assert_refused(explicit, tmp_path / "explicit", "'sound_speed' cannot be changed on restart", f"explicit cs on {gpus} device(s)")


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
@pytest.mark.parametrize(("written", "resumed", "reason"), [
    ("", "", None),
    ("", "TRACED", "records no tracer population; this run configures a discrete one"),
], ids=["untraced-to-untraced", "untraced-to-traced"])
def test_a_resume_continues_the_tracer_population_its_checkpoint_records(tmp_path: Path, gpus: int, written: str, resumed: str, reason: str | None) -> None:
    script = tmp_path / "traced_channel.py"
    script.write_text(
        "from pydantic import computed_field\n"
        f"from {ISOTHERMAL} import IsothermalChannel\n\n\n"
        "class Traced(IsothermalChannel):\n"
        "    @computed_field\n"
        "    @property\n"
        "    def n_tracers(self) -> int:\n"
        "        return 200\n"
    )
    assert _iso(tmp_path / "first", 1, "", 2).returncode == 0
    first = _final(tmp_path / "first")
    # the checkpoint's two iterations and two more on the single grid, four more on host tiles
    steps = 4 if gpus == 1 else 6
    if resumed == "TRACED":
        snippet = (
            "import sys\n"
            f"sys.path.insert(0, {str(tmp_path)!r})\n"
            "from simbi.simulation import runner\n"
            "from traced_channel import Traced\n"
            f"runner.run(Traced(resolution=(32, 16), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1], checkpoint_file={str(first)!r}), compute_mode='cpu', max_steps={steps})\n"
        )
        done = subprocess.run([sys.executable, "-c", snippet, str(tmp_path / "resumed")], capture_output=True, text=True, timeout=600,
                              env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
        _assert_refused(done, tmp_path / "resumed", reason, f"a traced resume of an untraced file on {gpus} device(s)")
    else:
        # the checkpoint's two iterations and two more
        done = _iso(tmp_path / "resumed", gpus, "", 4, checkpoint=first)
        assert done.returncode == 0, f"the untraced resume failed:\n{done.stderr[-2000:]}"


@pytest.mark.parametrize("gpus", [1, 2], ids=["single", "tiles"])
def test_a_dyed_resume_of_an_undyed_checkpoint_is_refused(tmp_path: Path, gpus: int) -> None:
    script = tmp_path / "dyed_blast.py"
    script.write_text(
        f"from {MODULE} import RefinedBlast\n\n\n"
        "class DyedBlast(RefinedBlast):\n"
        "    def passive_scalar(self):\n"
        "        return (0.5 for _ in range(16 * 16))\n"
    )
    written = _written(tmp_path / "first", FLAT)
    # the checkpoint's two iterations and two more on the single grid, four more on host tiles
    steps = 4 if gpus == 1 else 6
    snippet = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "from simbi.simulation import runner\n"
        "from dyed_blast import DyedBlast\n"
        f"runner.run(DyedBlast(resolution=(16, 16, 1), gpus={gpus}, checkpoint_interval=1e9, data_directory=sys.argv[1], {FLAT}, checkpoint_file={str(written)!r}), compute_mode='cpu', max_steps={steps})\n"
    )
    done = subprocess.run([sys.executable, "-c", snippet, str(tmp_path / "resumed")], capture_output=True, text=True, timeout=600,
                          env=dict(os.environ, SYMBI_GPU_OVERSUBSCRIBE="1"))
    _assert_refused(done, tmp_path / "resumed", "records no passive scalar (conserved/chi)", f"dyed resume of an undyed file on {gpus} device(s)")
