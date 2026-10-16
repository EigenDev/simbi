# =============================================================================
# test_launch_refined.py
#
# mesh refinement through `simbi launch`: a root grid with one fine box,
# launched across workers whose cuts cross the box, against the single-process
# refined run, both levels compared bit for bit; resumed onto a changed
# partition from the launch's own two-level checkpoint; a hydro patch face on
# a cut under the production halo, its fine ghosts prolonged from parent cells
# gathered past the root halo, equal bit for bit fresh and resumed, and a
# magnetized one refused by name. each arm runs the refined blast in newtonian hydro and the
# magnetized blast in 2.5D newtonian and special-relativistic MHD, the latter
# two under the Contact and the UCT edge EMF, their staggered faces on both
# levels compared with the cells; a
# resumed launch names no scheme flag, so the recorded edge EMF continues. the
# same fresh run and repartitioned restart with the workers started as a
# scheduler starts them: rank and size from the environment, every worker
# bound on all interfaces and advertising an address this host reaches, each
# exiting zero, the files carrying the single-process run's build, config and
# numerics, every staggered face present, and no rendezvous or staging file
# left behind. dispatch and addressing on one host only: cross-node routing,
# shared-filesystem visibility and node failure need real nodes.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_refined.py
# =============================================================================

import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

from simbi_configs.examples.newtonian.refined_blast import RefinedBlast
from simbi_configs.examples.newtonian.refined_mhd_blast import RefinedMhdBlast
from simbi_configs.examples.srmhd.refined_rmhd_blast import RefinedRmhdBlast

from simbi.simulation.tests.test_launch import (
    EXIT,
    _lan_address,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference

pytestmark = pytest.mark.simulation

# the fine box covers root cells 8..24 on both axes of the 32 x 32 root
RESOLUTION = (32, 32, 1)
STEPS = 8
# the checkpoint attributes that record the build, the configuration and the numerics a file
# was written under; a launched file carries the single-process run's values
PROVENANCE = (
    "build_source_id", "config_source", "config_sha256", "regime", "solver", "reconstruction",
    "eos", "ct_method", "timestepping", "cfl", "plm_theta", "gamma", "boundary_conditions",
)


class Problem:
    """a refined problem, its launch flags and the matching single-process constructor fields"""

    def __init__(self, cls: type, flags: list[str], fields: str, ct_method: str) -> None:
        self.cls = cls
        # the edge EMF the checkpoint records, empty for a run without a staggered field
        self.ct_method = ct_method
        self.script = str(Path(cls.__module__.replace(".", "/") + ".py"))
        self.flags = flags
        self.fields = fields

    def reference(self, out: Path, steps: int) -> Path:
        return _script_reference(self.cls.__module__, self.cls.__name__, RESOLUTION, out, steps, self.fields)

    def launch(self, out: Path, layout: Path, steps: int, workers: int, checkpoint: Path | None = None) -> Path:
        # a resumed launch names no scheme flag: the checkpoint's recorded numerics continue
        flags = [] if checkpoint is not None else self.flags
        return _script_launch(self.script, RESOLUTION, out, layout, steps, workers, checkpoint=checkpoint, flags=flags)


PROBLEMS = [
    pytest.param(Problem(RefinedBlast, [], "", ""), id="hydro"),
    pytest.param(Problem(RefinedMhdBlast, [], "", "contact"), id="mhd-contact"),
    pytest.param(Problem(RefinedMhdBlast, ["--ct-method", "uct"], "ct_method='uct'", "uct"), id="mhd-uct"),
    pytest.param(Problem(RefinedRmhdBlast, [], "", "contact"), id="rmhd-contact"),
    pytest.param(Problem(RefinedRmhdBlast, ["--ct-method", "uct"], "ct_method='uct'", "uct"), id="rmhd-uct"),
]


def _levels(f: h5py.File) -> dict[str, np.ndarray]:
    """every cell and face dataset of both levels trimmed to its level's interior"""
    out = {}
    for level_name in ("level_0", "level_1"):
        assert level_name in f, f"the checkpoint carries no {level_name}"
        level = f[level_name]
        ng = int(level["mesh"].attrs["halo_width"])
        names: list[str] = []
        level.visit(lambda name, names=names, level=level: names.append(name) if isinstance(level[name], h5py.Dataset) else None)
        for name in names:
            if name.startswith("mesh/") or name.startswith("partition_0/owned") or "/domain/" in name:
                continue
            data = level[name][()]
            if data.ndim < 2:
                continue
            # the staggered faces are stored over the interior face range, every face compared;
            # the cell fields carry the halo, trimmed away
            if "/magnetic/" in name:
                out[f"{level_name}/{name}"] = data
            else:
                out[f"{level_name}/{name}"] = data[tuple(slice(ng, -ng) for _ in range(data.ndim))]
    return out


def test_every_staggered_face_is_compared() -> None:
    # a face dataset keeps its full interior face range: n + 1 faces along its own axis
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "levels.h5"
        with h5py.File(path, "w") as f:
            for level_name, ng in (("level_0", 2), ("level_1", 3)):
                level = f.create_group(level_name)
                level.create_group("mesh").attrs["halo_width"] = ng
                level.create_dataset("conserved/den", data=np.zeros((32 + 2 * ng, 32 + 2 * ng)))
                level.create_dataset("partition_0/hydro/magnetic/B1/data", data=np.zeros((32, 33)))
        with h5py.File(path, "r") as f:
            levels = _levels(f)
    for level_name in ("level_0", "level_1"):
        assert levels[f"{level_name}/conserved/den"].shape == (32, 32)
        assert levels[f"{level_name}/partition_0/hydro/magnetic/B1/data"].shape == (32, 33)


def _assert_same_levels(a: Path, b: Path, label: str) -> None:
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for level in ("level_0", "level_1"):
            for key in ("time", "iteration"):
                assert fa[level].attrs[key] == fb[level].attrs[key], f"{label}: {level} attr {key}"
        xs, ys = _levels(fa), _levels(fb)
        assert xs.keys() == ys.keys(), f"{label}: dataset sets differ"
        for name, x in xs.items():
            np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


@pytest.mark.parametrize("problem", PROBLEMS)
def test_a_refined_launch_matches_the_single_process_run(tmp_path: Path, problem: Problem) -> None:
    reference = problem.reference(tmp_path / "single", STEPS)
    # a cut through the box's center, then ragged cuts on both axes
    for tag, workers, kwargs in [("two", 2, {"shape": (2, 1)}), ("four", 4, {"cuts": [[13], [18]]})]:
        layout = _layout(tmp_path / f"{tag}.toml", workers=workers, **kwargs)
        final = problem.launch(tmp_path / tag, layout, STEPS, workers)
        _assert_same_levels(final, reference, f"{workers} workers")


@pytest.mark.parametrize("problem", PROBLEMS)
def test_a_refined_launch_resumed_onto_another_partition_matches(tmp_path: Path, problem: Problem) -> None:
    reference = problem.reference(tmp_path / "single", STEPS)
    first = problem.launch(tmp_path / "first", _layout(tmp_path / "two.toml", workers=2, shape=(2, 1)), STEPS // 2, 2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # the step cap is an absolute root iteration: the resume from written_at runs to STEPS
    resumed = problem.launch(
        tmp_path / "four", _layout(tmp_path / "four.toml", workers=4, cuts=[[13], [18]]), STEPS, 4, checkpoint=first
    )
    with h5py.File(resumed, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == STEPS
    _assert_same_levels(resumed, reference, "resumed on four workers")


# one overall deadline for a scheduler-started session, from the first spawn to the last exit
SESSION_DEADLINE = 600.0
# the time a terminated worker has to exit before it is killed
TERMINATE_GRACE = 5.0


def _run_ranks(
    cmd: list[str],
    envs: list[dict[str, str]],
    logs: list[Path],
    deadline: float,
    started: list[subprocess.Popen],
    spawn=subprocess.Popen,
) -> list[int]:
    """start one process per environment through `spawn`, each writing its stderr to its own
    log file, and wait for every exit under one deadline; a spawn failure or the deadline raises.
    on every exit path each started process is terminated, killed after the grace, and reaped.
    `started` receives the processes as they start"""
    end = time.monotonic() + deadline
    try:
        for env, log in zip(envs, logs):
            with open(log, "w") as sink:
                started.append(spawn(cmd, env=env, stdout=subprocess.DEVNULL, stderr=sink))
        while any(p.poll() is None for p in started):
            if time.monotonic() > end:
                held = [int(env["SLURM_PROCID"]) for env, p in zip(envs, started) if p.poll() is None]
                raise TimeoutError(f"workers {held} still running after {deadline} s")
            time.sleep(0.05)
        return [p.returncode for p in started]
    finally:
        for p in started:
            if p.poll() is None:
                p.terminate()
        grace = time.monotonic() + TERMINATE_GRACE
        for p in started:
            while p.poll() is None and time.monotonic() < grace:
                time.sleep(0.02)
            if p.poll() is None:
                p.kill()
            p.wait()


def _scheduler_session(
    problem: Problem, out: Path, workers: int, partition: str, steps: int, checkpoint: Path | None = None
) -> tuple[list[str], list[dict[str, str]], list[Path]]:
    """the command, the per-rank environments and the per-rank log files of one process per
    worker started as a scheduler starts them: rank and size in the environment, no launcher
    parent, every worker bound on all interfaces and advertising an address this host reaches,
    the ranks cycling through the loopback address, a name that resolves to it and, when the
    host has one, its network address. the logs sit beside the output directory"""
    layout = out.parent / f"{out.name}.toml"
    layout.write_text(
        f'[execution]\nmode = "scheduler"\nworkers = {workers}\nbind = "0.0.0.0"\n\n[partition]\n{partition}\n\n'
        f'[checkpoint]\nstaging_mb = 1\ndirectory = "{out}"\n'
    )
    cmd = [
        sys.executable, "-m", "simbi.cli", "launch", problem.script, "--layout", str(layout),
        "--max-steps", str(steps), "--resolution", ",".join(str(r) for r in RESOLUTION),
        "--data-directory", str(out.parent / f"{out.name}-ignored"), "--checkpoint-interval", "1e9",
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    else:
        cmd += problem.flags
    addresses = ["127.0.0.1", "localhost"] + [a for a in [_lan_address()] if a]
    envs = []
    for rank in range(workers):
        env = dict(os.environ, SLURM_PROCID=str(rank), SLURM_NTASKS=str(workers), SIMBI_ADVERTISE=addresses[rank % len(addresses)])
        env.pop("SIMBI_LAUNCHED_WORKER", None)
        envs.append(env)
    logs = [out.parent / f"{out.name}-worker{rank}.log" for rank in range(workers)]
    return cmd, envs, logs


def _scheduler_launch(
    problem: Problem, out: Path, workers: int, partition: str, steps: int, checkpoint: Path | None = None
) -> Path:
    """a scheduler-started session run to completion: every worker exits zero and one final
    checkpoint lands in the layout's directory. returns the final checkpoint"""
    cmd, envs, logs = _scheduler_session(problem, out, workers, partition, steps, checkpoint)
    codes = _run_ranks(cmd, envs, logs, SESSION_DEADLINE, [])
    for rank, (code, log) in enumerate(zip(codes, logs)):
        assert code == 0, f"scheduler-started worker {rank} exited {code}:\n{log.read_text()[-3000:]}"
    finals = sorted(out.glob("*final*.h5"))
    assert len(finals) == 1, f"expected one final checkpoint in {out}, found {finals}"
    assert not (out.parent / f"{out.name}-ignored").exists(), "the layout's checkpoint directory was not honored"
    return finals[0]


def _assert_reaped(started: list[subprocess.Popen]) -> None:
    for rank, p in enumerate(started):
        assert p.returncode is not None, f"worker {rank} was left running"
        try:
            os.kill(p.pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError(f"worker {rank} (pid {p.pid}) survives its session")


def test_a_held_scheduler_worker_is_ended_and_reaped_at_the_deadline(tmp_path: Path) -> None:
    # rank 1 of two, started alone: it waits for a coordinator that never writes the rendezvous
    # record, so it is running when the deadline passes
    cmd, envs, logs = _scheduler_session(PROBLEMS[0].values[0], tmp_path / "held", 2, "shape = [2, 1]", 1)
    started: list[subprocess.Popen] = []
    with pytest.raises(TimeoutError, match=r"workers \[1\] still running"):
        _run_ranks(cmd, envs[1:], logs[1:], 3.0, started)
    assert len(started) == 1, "the held worker was never started; the deadline gate is vacuous"
    _assert_reaped(started)


def test_a_failed_spawn_ends_and_reaps_the_workers_already_started(tmp_path: Path) -> None:
    # the second spawn fails after the first worker is running and waiting for its peer
    cmd, envs, logs = _scheduler_session(PROBLEMS[0].values[0], tmp_path / "spawn", 2, "shape = [2, 1]", 1)
    calls = []

    # the injection reaches the worker spawns alone; every other process start is untouched
    def failing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("spawn refused")
        return subprocess.Popen(*args, **kwargs)

    started: list[subprocess.Popen] = []
    with pytest.raises(OSError, match="spawn refused"):
        _run_ranks(cmd, envs, logs, SESSION_DEADLINE, started, spawn=failing)
    assert len(started) == 1 and started[0].returncode is not None and started[0].returncode < 0, (
        "the first worker was not running when the spawn failed; the cleanup gate is vacuous"
    )
    _assert_reaped(started)


def _assert_clean_output(out: Path) -> None:
    """the session leaves its published checkpoints and its log: the coordinator retired the
    rendezvous record and no staging or partial file remains"""
    leftovers = sorted(p.name for p in out.iterdir() if not (p.suffix == ".h5" and ".chkpt." in p.name) and p.name != "simbi.log")
    assert not leftovers, f"the session left {leftovers} in {out}"


def _assert_provenance(launched: Path, reference: Path, ct_method: str, label: str) -> None:
    with h5py.File(launched, "r") as fl, h5py.File(reference, "r") as fr:
        recorded = fl["metadata"].attrs.get("ct_method")
        recorded = recorded.decode() if isinstance(recorded, bytes) else recorded
        assert recorded == ct_method, f"{label}: the checkpoint records ct_method {recorded!r}, expected {ct_method!r}"
        for key in PROVENANCE:
            assert key in fl["metadata"].attrs, f"{label}: the checkpoint records no {key}"
            assert fl["metadata"].attrs[key] == fr["metadata"].attrs[key], f"{label}: {key} differs from the single-process run"


def _assert_complete_faces(path: Path, magnetized: bool) -> None:
    """a magnetized file holds every interior face of both levels, n + 1 along the face's own
    axis; a hydro file holds none"""
    with h5py.File(path, "r") as f:
        for level_name, cells in (("level_0", 32), ("level_1", 32)):
            magnetic = f[level_name].get("partition_0/hydro/magnetic")
            if not magnetized:
                assert magnetic is None, f"a hydro file carries faces on {level_name}"
                continue
            assert magnetic is not None, f"{level_name} carries no staggered faces"
            # storage order is (x2, x1): B1 lies on the x1 faces
            assert magnetic["B1/data"].shape == (cells, cells + 1), f"{level_name} B1 {magnetic['B1/data'].shape}"
            assert magnetic["B2/data"].shape == (cells + 1, cells), f"{level_name} B2 {magnetic['B2/data'].shape}"


@pytest.mark.parametrize("problem", PROBLEMS)
def test_scheduler_started_refined_workers_match_and_resume_onto_another_partition(tmp_path: Path, problem: Problem) -> None:
    # a problem records the edge EMF of its staggered field, and none without one
    magnetized = bool(problem.ct_method)
    reference = problem.reference(tmp_path / "single", STEPS)
    fresh = _scheduler_launch(problem, tmp_path / "fresh", 4, "cuts = [[13], [18]]", STEPS)
    _assert_same_levels(fresh, reference, "scheduler-started, four workers")
    _assert_provenance(fresh, reference, problem.ct_method, "scheduler-started, four workers")
    _assert_complete_faces(fresh, magnetized)
    _assert_clean_output(tmp_path / "fresh")

    first = _scheduler_launch(problem, tmp_path / "first", 2, "shape = [2, 1]", STEPS // 2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # the step cap is an absolute root iteration: the resume from written_at runs to STEPS
    resumed = _scheduler_launch(problem, tmp_path / "resumed", 4, "cuts = [[13], [18]]", STEPS, checkpoint=first)
    with h5py.File(resumed, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == STEPS
    _assert_same_levels(resumed, reference, "scheduler-started, resumed on four workers")
    _assert_provenance(resumed, reference, problem.ct_method, "scheduler-started, resumed on four workers")
    _assert_complete_faces(resumed, magnetized)
    _assert_clean_output(tmp_path / "resumed")


def _refused(tmp_path: Path, script: str, resolution: tuple[int, ...] | None, layout: Path) -> str:
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", "1", "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if resolution is not None:
        cmd += ["--resolution", ",".join(str(r) for r in resolution)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    assert not (out.exists() and list(out.glob("*.h5"))), "a checkpoint was published"
    assert EXIT.findall(done.stdout) or done.stderr, "the launch reported nothing"
    return " ".join(done.stderr.split())


# the coarse-fine prolongation's reach in root cells past a patch face: half the fine halo (one
# cell deeper than the root's) rounded up, plus the stencil half-width of 2
PROLONG_REACH = 4


def test_a_hydro_patch_face_on_a_cut_matches_the_single_process_run(tmp_path: Path) -> None:
    # the box's low face on axis 0 at root cell 8 and its high face on axis 1 at root cell 24 lie
    # on the cuts: their fine ghosts prolong from parent cells past the root halo, gathered from
    # the workers owning them each root step, the corner from the diagonal worker; fresh and
    # resumed from a two-worker checkpoint whose cut crosses the box
    problem = PROBLEMS[0].values[0]
    reference = problem.reference(tmp_path / "single", STEPS)
    with h5py.File(reference, "r") as f:
        halo = int(f["level_0/mesh"].attrs["halo_width"])
    assert halo < PROLONG_REACH, f"the root halo ({halo}) holds the prolongation reach; the cut faces read no gathered cell"
    on_faces = _layout(tmp_path / "faces.toml", workers=4, cuts=[[8], [24]])
    _assert_same_levels(problem.launch(tmp_path / "faces", on_faces, STEPS, 4), reference, "patch faces on the cuts")
    first = problem.launch(tmp_path / "first", _layout(tmp_path / "two.toml", workers=2, shape=(2, 1)), STEPS // 2, 2)
    resumed = problem.launch(tmp_path / "resumed", on_faces, STEPS, 4, checkpoint=first)
    _assert_same_levels(resumed, reference, "patch faces on the cuts, resumed")


def test_a_magnetized_patch_face_on_a_cut_is_refused_by_name(tmp_path: Path) -> None:
    # the magnetized box's low face on axis 0 at root cell 8, on the cut: its cell and face field
    # prolong from the root halo, which plm's two ghost cells do not hold
    problem = PROBLEMS[1].values[0]
    stderr = _refused(tmp_path, problem.script, RESOLUTION, _layout(tmp_path / "two.toml", workers=2, cuts=[[8], []]))
    assert "lies on a decomposition cut" in stderr, stderr
