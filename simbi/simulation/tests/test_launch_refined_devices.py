# =============================================================================
# test_launch_refined_devices.py
#
# a refined 3D newtonian launch on two workers against the single-device
# hierarchy, on the cpu and (with SIMBI_LAUNCH_GPU_TESTS) on one gpu per worker.
# two cuts make the coarse-fine traffic cross the workers: a cut on the fine
# box's low x face (the flux-register contributions travel to the coarse cells'
# owner, the parent prim stencil to the fine tile's), and a cut through the box
# (the fine level's halos travel between the workers). each launch reports, by
# plan role, the transfers its compiled plans schedule to the other worker
# (summed over the phase table's points, a property of the plans); the gate
# asserts that the role its cut is built to exercise schedules remote transfers,
# and the bitwise equality with the single device is the runtime evidence.
# - evolution, priming and checkpoints: every published checkpoint, the opening
#   one included, equals the single device's, whole file, both levels
# - restart: half the steps cut through the box, the rest resumed on the cut on
#   the box face, against the uninterrupted single device
# - rollback: a rejection in the last stage of the last fine substep of one
#   step, on the worker holding the box, rolls both workers back; stopped after
#   that step it equals the clean launch capped at the accepted timestep
# - the public admission on the gpu: the gated scope (two workers, plm of the
#   minmod-MC family at any validated theta, hlle, rk2, outflow, one refined
#   region) runs without SIMBI_LAUNCH_EXPERIMENTAL, scalar parameters varied;
#   each untested structure is refused by name without it; a tracer population
#   on device memory and two workers masked onto one gpu are refused even with
#   it
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_refined_devices.py
# =============================================================================
import os
import re
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import launcher
from simbi.simulation.tests.checkpoint_compare import assert_same_checkpoint
from simbi.simulation.tests.test_launch import EXIT, MODES, _bound_devices, _layout, bounded_processes, needs_gpu  # noqa: F401  the autouse process guard
from simbi_configs.examples.newtonian.refined_blast_3d import RefinedBlast3D

pytestmark = pytest.mark.simulation

SCRIPT = str(Path(RefinedBlast3D.__module__.replace(".", "/") + ".py"))
RESOLUTION = (24, 24, 24)
STEPS = 8
# the box spans root cells 6..18 on every axis: a cut at 6 lies on its low x face, one at 12
# halves it
GEOMETRIES = {
    "cut-on-box-face": ([[6], [], []], ("FluxContribution", "PrimStencil")),
    "cut-through-box": ([[12], [], []], ("Fine",)),
}
REMOTE = re.compile(r"SIMBI launch refined plans: worker=(\d+)/\d+ remote_sends=(\S+)")
REPORT = re.compile(r"SIMBI launch: worker=(\d+)/\d+ steps=(\d+) rejections=(\d+) refined=1")
REFERENCE_SNIPPET = f"""
import sys
from simbi.simulation import runner
from {RefinedBlast3D.__module__} import RefinedBlast3D
out, mode, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
runner.run(RefinedBlast3D(resolution={RESOLUTION!r}, data_directory=out, checkpoint_interval=1e-9), compute_mode=mode, max_steps=steps)
"""


def _mode_env(mode: str) -> dict[str, str]:
    """the environment of a launch: no injected fault and no experimental opt-in; the gated
    refined scope runs on either device without it"""
    env = dict(os.environ)
    env.pop("SIMBI_FABRIC_INJECT", None)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    return env


def _reference(out: Path, mode: str, steps: int = STEPS) -> dict[str, Path]:
    """the single-device hierarchy, a checkpoint at every step, by file name"""
    done = subprocess.run([sys.executable, "-c", REFERENCE_SNIPPET, str(out), mode, str(steps)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    files = {p.name: p for p in out.glob("*.h5")}
    assert any("final" in name for name in files), f"no final checkpoint in {out}"
    return files


def _launch(out: Path, layout: Path, mode: str, steps: int = STEPS, checkpoint: Path | None = None, inject: str | None = None) -> tuple[dict[str, Path], str]:
    """the two-worker launch, a checkpoint at every step: its files by name and its stderr"""
    cmd = [sys.executable, "-m", "simbi.cli", "launch", SCRIPT, "--layout", str(layout), "--max-steps", str(steps),
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(out), "--checkpoint-interval", "1e-9"]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    if mode == "gpu":
        cmd.append("--gpu")
    env = _mode_env(mode)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == [0, 1] and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = {p.name: p for p in out.glob("*.h5")}
    assert any("final" in name for name in files), f"no final checkpoint in {out}"
    if mode == "gpu":
        bound = _bound_devices(done.stderr)
        assert sorted(bound) == [0, 1] and bound[0] != bound[1], f"device bindings {bound}"
    return files, done.stderr


def _remote_sends(stderr: str) -> dict[int, dict[str, int]]:
    out = {}
    for worker, roles in REMOTE.findall(stderr):
        out[int(worker)] = {role: int(n) for role, n in (item.split(":") for item in roles.split(","))}
    assert sorted(out) == [0, 1], f"plan reports from workers {sorted(out)}"
    return out


def _reports(stderr: str) -> dict[int, tuple[int, int]]:
    return {int(w): (int(n), int(r)) for w, n, r in REPORT.findall(stderr)}


def _final(files: dict[str, Path]) -> Path:
    finals = [p for name, p in files.items() if "final" in name]
    assert len(finals) == 1, f"final checkpoints {finals}"
    return finals[0]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_a_refined_launch_equals_one_device_at_every_checkpoint(tmp_path: Path, geometry: str, mode: str) -> None:
    cuts, roles = GEOMETRIES[geometry]
    reference = _reference(tmp_path / "single", mode)
    layout = _layout(tmp_path / "two.toml", workers=2, cuts=cuts)
    launched, stderr = _launch(tmp_path / "two", layout, mode)
    remote = _remote_sends(stderr)
    for role in roles:
        assert any(remote[w].get(role, 0) > 0 for w in remote), f"{geometry}: the plans schedule no remote {role} transfer ({remote}); the gate is vacuous"
    assert all(n == STEPS for n, _ in _reports(stderr).values()), f"steps per worker {_reports(stderr)}"
    assert sorted(launched) == sorted(reference), f"checkpoints {sorted(launched)} against {sorted(reference)}"
    assert len(launched) >= 4, f"{len(launched)} checkpoints; the cadence is not exercised"
    with h5py.File(_final(launched), "r") as f:
        assert "level_1" in f, "the launch published no refined level"
        assert int(f["level_0"].attrs["iteration"]) == STEPS
    for name in sorted(launched):
        assert_same_checkpoint(launched[name], reference[name], f"{geometry}, {mode}: {name}")


@pytest.mark.parametrize("mode", MODES)
def test_a_refined_launch_resumed_onto_another_partition_equals_one_device(tmp_path: Path, mode: str) -> None:
    reference = _reference(tmp_path / "single", mode)
    through = _layout(tmp_path / "through.toml", workers=2, cuts=GEOMETRIES["cut-through-box"][0])
    face = _layout(tmp_path / "face.toml", workers=2, cuts=GEOMETRIES["cut-on-box-face"][0])
    first, _ = _launch(tmp_path / "first", through, mode, steps=STEPS // 2)
    partial = _final(first)
    with h5py.File(partial, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == STEPS // 2
    resumed, stderr = _launch(tmp_path / "resumed", face, mode, checkpoint=partial)
    assert all(n == STEPS - STEPS // 2 for n, _ in _reports(stderr).values()), f"steps per worker {_reports(stderr)}"
    assert_same_checkpoint(_final(resumed), _final(reference), f"{mode}: resumed onto the cut on the box face")


REJECTED_STEP = 2
# worker 1 holds the box under the cut on its face; the fine level takes two substeps per root step
# under rk2, so substep 1 stage 1 is the last stage of the last substep of the step
LEVEL_REJECTION = f"reject_level:1:{REJECTED_STEP}:1:1:1"


@pytest.mark.parametrize("mode", MODES)
def test_a_fine_substep_rejection_rolls_both_workers_back_as_the_capped_clean_launch(tmp_path: Path, mode: str) -> None:
    face = _layout(tmp_path / "face.toml", workers=2, cuts=GEOMETRIES["cut-on-box-face"][0])
    steps = REJECTED_STEP + 1
    rejected, stderr = _launch(tmp_path / "rejected", face, mode, steps=steps, inject=LEVEL_REJECTION)
    reports = _reports(stderr)
    assert reports == {0: (steps, 1), 1: (steps, 1)}, f"(steps, rejections) per worker {reports}: a worker kept out of the rollback"
    clean, clean_err = _launch(tmp_path / "clean", face, mode, steps=steps)
    assert _reports(clean_err) == {0: (steps, 0), 1: (steps, 0)}
    with h5py.File(_final(rejected), "r") as fr, h5py.File(_final(clean), "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
        assert "level_1" in fr
    capped, capped_err = _launch(tmp_path / "capped", face, mode, steps=steps, inject=f"cap:{REJECTED_STEP}:{int(np.float64(accepted).view(np.uint64))}")
    assert _reports(capped_err) == {0: (steps, 0), 1: (steps, 0)}, "the capped control rejected a step"
    with h5py.File(_final(capped), "r") as f:
        assert float(f["level_0"].attrs["dt"]) == accepted, "the capped control's step took another timestep"
    assert_same_checkpoint(_final(rejected), _final(capped), f"{mode}: fine-substep rejection against the capped clean launch")


# subclass lines of the admission variants
VARIANT_IMPORTS = (
    "from typing import Annotated\n"
    "from pydantic import computed_field\n"
    "from simbi import ProblemParam\n"
    "from simbi.types import BoundaryCondition, Reconstruction, Solver, TimeStepping\n"
    "from simbi.types.input import Limiter\n"
    f"from {RefinedBlast3D.__module__} import RefinedBlast3D\n\n\n"
)
ADMITTED = {
    "theta-1.2-cfl-0.3-gamma-1.4": [
        "plm_theta: Annotated[float, ProblemParam(1.2, description='t')]",
        "cfl_number: Annotated[float, ProblemParam(0.3, description='c')]",
        "adiabatic_index: Annotated[float, ProblemParam(1.4, description='g')]",
    ],
    "theta-2.0": ["plm_theta: Annotated[float, ProblemParam(2.0, description='t')]"],
}
UNTESTED = {
    "hllc": (["solver: Annotated[Solver, ProblemParam(Solver.HLLC, description='s')]"], 2, "the hllc solver"),
    "ppm": (["reconstruction: Annotated[Reconstruction, ProblemParam(Reconstruction.PPM, description='r')]"], 2, "ppm reconstruction"),
    "rk3": (["timestepping: Annotated[TimeStepping, ProblemParam(TimeStepping.RK3, description='t')]"], 2, "Rk3 time stepping"),
    "van-leer": (["limiter: Annotated[Limiter, ProblemParam(Limiter.VAN_LEER, description='l')]"], 2, "the van leer limiter"),
    "reflecting": (["boundary_conditions: Annotated[list, ProblemParam([BoundaryCondition.REFLECTING], description='b')]"], 2, "a boundary other than outflow"),
    "three-workers": ([], 3, "3 workers"),
}
TRACERS = ["@computed_field", "@property", "def n_tracers(self) -> int:", "    return 64"]


def _variant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, lines: list[str]) -> str:
    stem = "variant_" + name.replace("-", "_").replace(".", "_")
    body = "\n".join(f"    {line}" for line in lines) or "    pass"
    (tmp_path / f"{stem}.py").write_text(VARIANT_IMPORTS + f"class Variant(RefinedBlast3D):\n{body}\n")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    return str(tmp_path / f"{stem}.py")


def _gpu_launch(script: str, tmp_path: Path, workers: int, env: dict[str, str]) -> subprocess.CompletedProcess:
    cuts = [[12], [], []] if workers == 2 else [[6, 12], [], []]
    layout = _layout(tmp_path / f"{workers}.toml", workers=workers, cuts=cuts)
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", "1", "--gpu",
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)


@needs_gpu
@pytest.mark.parametrize("name", list(ADMITTED))
def test_the_gated_refined_scope_launches_on_gpus_without_the_opt_in(tmp_path: Path, monkeypatch, name: str) -> None:
    """scalar parameters of the tested family vary freely: the launch runs a step on two gpus"""
    done = _gpu_launch(_variant(tmp_path, monkeypatch, name, ADMITTED[name]), tmp_path, 2, _mode_env("gpu"))
    assert done.returncode == 0, f"the gated scope was refused:\n{done.stderr[-3000:]}"
    assert "experimental combination" not in done.stderr, "the gated scope ran as experimental"
    bound = _bound_devices(done.stderr)
    assert sorted(bound) == [0, 1] and bound[0] != bound[1], f"device bindings {bound}"
    assert {int(w): (int(n), int(r)) for w, n, r in REPORT.findall(done.stderr)} == {0: (1, 0), 1: (1, 0)}


@needs_gpu
@pytest.mark.parametrize("name", list(UNTESTED))
def test_an_untested_refined_structure_is_refused_by_name_on_gpus(tmp_path: Path, monkeypatch, name: str) -> None:
    """each structure outside the gated scope runs on the cpu launch and is refused by name on the
    gpu without the experimental opt-in, before any checkpoint"""
    lines, workers, prop = UNTESTED[name]
    # the launcher refuses more workers than visible devices before any worker reaches the
    # admission; the arm needs one visible gpu per worker
    masks = [m for m in launcher.GPU_MASKS if os.environ.get(m)]
    visible = len([e for e in os.environ[masks[0]].split(",") if e.strip()]) if masks else workers
    assert visible >= workers, f"{masks[0]} exposes {visible} gpus; the {workers}-worker admission is unreachable"
    done = _gpu_launch(_variant(tmp_path, monkeypatch, name, lines), tmp_path, workers, _mode_env("gpu"))
    assert done.returncode != 0, f"the launch ran:\n{done.stdout}"
    want = f"mesh refinement on gpu device memory with {prop} is installed on the launched path without a distributed gate"
    assert want in " ".join(done.stderr.split()), done.stderr[-3000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


@pytest.mark.parametrize("name", list(UNTESTED))
def test_an_untested_refined_structure_runs_on_the_cpu_launch(tmp_path: Path, monkeypatch, name: str) -> None:
    """the gpu refusals are the device scope's: each structure is a valid refined cpu launch"""
    lines, workers, _ = UNTESTED[name]
    script = _variant(tmp_path, monkeypatch, name, lines)
    cuts = [[12], [], []] if workers == 2 else [[6, 12], [], []]
    layout = _layout(tmp_path / f"{workers}.toml", workers=workers, cuts=cuts)
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", "1",
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=_mode_env("cpu"))
    assert done.returncode == 0, f"{name} is no valid refined cpu launch:\n{done.stderr[-3000:]}"


@needs_gpu
def test_a_refined_tracer_population_is_refused_on_gpus_even_with_the_opt_in(tmp_path: Path, monkeypatch) -> None:
    env = dict(_mode_env("gpu"), SIMBI_LAUNCH_EXPERIMENTAL="1")
    done = _gpu_launch(_variant(tmp_path, monkeypatch, "tracers", TRACERS), tmp_path, 2, env)
    assert done.returncode != 0, f"the launch ran:\n{done.stdout}"
    assert "a tracer population on device memory with mesh refinement is outside the distributed first release" in " ".join(done.stderr.split()), done.stderr[-3000:]
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"


@needs_gpu
def test_two_refined_gpu_workers_masked_onto_one_gpu_are_refused_together(tmp_path: Path) -> None:
    present = [m for m in launcher.GPU_MASKS if m in os.environ]
    first = os.environ[present[0]].split(",")[0].strip() if present else "0"
    # with the experimental opt-in: the binding agreement refuses a shared gpu all the same
    env = dict(_mode_env("gpu"), SIMBI_LAUNCH_EXPERIMENTAL="1")
    env.update({m: f"{first},{first}" for m in (present or ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])})
    layout = _layout(tmp_path / "two.toml", workers=2, cuts=GEOMETRIES["cut-through-box"][0])
    cmd = [sys.executable, "-m", "simbi.cli", "launch", SCRIPT, "--layout", str(layout), "--max-steps", "1", "--gpu",
           "--resolution", ",".join(str(r) for r in RESOLUTION), "--data-directory", str(tmp_path / "out"), "--checkpoint-interval", "1e9"]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    bound = _bound_devices(done.stderr)
    assert sorted(bound) == [0, 1] and bound[0] == bound[1], f"the mask did not put both workers on one gpu: {bound}"
    assert " ".join(done.stderr.split()).count("workers 0 and 1 bound one physical gpu") == 2, done.stderr
    assert not list((tmp_path / "out").glob("*.h5")), "a checkpoint was published"
