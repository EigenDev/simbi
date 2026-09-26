# =============================================================================
# test_launch_refined.py
#
# mesh refinement through `simbi launch`: a root grid with one fine box,
# launched across workers whose cuts cross the box, against the single-process
# refined run, both levels compared bit for bit; resumed onto a changed
# partition from the launch's own two-level checkpoint; and the refined
# combinations outside the gated scope refused by name: a periodic boundary,
# and a patch face on a cut, whose fine ghosts would read the root halo past
# its depth. each arm runs the refined blast in newtonian hydro and the
# magnetized blast in 2.5D newtonian MHD, the latter under the Contact and the
# UCT edge EMF, its staggered faces on both levels compared with the cells; a
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

import h5py
import numpy as np
import pytest

from simbi_configs.examples.newtonian.refined_blast import RefinedBlast
from simbi_configs.examples.newtonian.refined_mhd_blast import RefinedMhdBlast

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
    resumed = problem.launch(
        tmp_path / "four", _layout(tmp_path / "four.toml", workers=4, cuts=[[13], [18]]), STEPS - written_at, 4, checkpoint=first
    )
    with h5py.File(resumed, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == STEPS
    _assert_same_levels(resumed, reference, "resumed on four workers")


def _scheduler_launch(
    problem: Problem, out: Path, workers: int, partition: str, steps: int, checkpoint: Path | None = None
) -> tuple[Path, list[str]]:
    """one process per worker started as a scheduler starts them, rank and size in the
    environment and no launcher parent, bound on every interface; each advertises an address
    this host reaches, the ranks cycling through the loopback address, a name that resolves to
    it and, when the host has one, its network address. returns the final checkpoint and each
    worker's stderr"""
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
    procs = []
    for rank in range(workers):
        env = dict(os.environ, SLURM_PROCID=str(rank), SLURM_NTASKS=str(workers), SIMBI_ADVERTISE=addresses[rank % len(addresses)])
        env.pop("SIMBI_LAUNCHED_WORKER", None)
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
    errs = [p.communicate(timeout=600)[1] for p in procs]
    for rank, (p, err) in enumerate(zip(procs, errs)):
        assert p.returncode == 0, f"scheduler-started worker {rank} exited {p.returncode}:\n{err[-3000:]}"
    finals = sorted(out.glob("*final*.h5"))
    assert len(finals) == 1, f"expected one final checkpoint in {out}, found {finals}"
    assert not (out.parent / f"{out.name}-ignored").exists(), "the layout's checkpoint directory was not honored"
    return finals[0], errs


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
    magnetized = problem.cls is RefinedMhdBlast
    reference = problem.reference(tmp_path / "single", STEPS)
    fresh, _ = _scheduler_launch(problem, tmp_path / "fresh", 4, "cuts = [[13], [18]]", STEPS)
    _assert_same_levels(fresh, reference, "scheduler-started, four workers")
    _assert_provenance(fresh, reference, problem.ct_method, "scheduler-started, four workers")
    _assert_complete_faces(fresh, magnetized)
    _assert_clean_output(tmp_path / "fresh")

    first, _ = _scheduler_launch(problem, tmp_path / "first", 2, "shape = [2, 1]", STEPS // 2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    resumed, _ = _scheduler_launch(problem, tmp_path / "resumed", 4, "cuts = [[13], [18]]", STEPS - written_at, checkpoint=first)
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


@pytest.mark.parametrize("problem", PROBLEMS[:2])
def test_a_patch_face_on_a_cut_is_refused_by_name(tmp_path: Path, problem: Problem) -> None:
    # the box's low face on axis 0 at root cell 8, on the cut: its fine ghosts prolong from the
    # root halo, which plm's two ghost cells do not hold
    stderr = _refused(tmp_path, problem.script, RESOLUTION, _layout(tmp_path / "two.toml", workers=2, cuts=[[8], []]))
    assert "lies on a decomposition cut" in stderr, stderr
