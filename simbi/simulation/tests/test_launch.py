# =============================================================================
# test_launch.py
#
# the public distributed path: `simbi launch` evolves a 2D cartesian newtonian
# problem across worker processes over the fabric and writes its checkpoints
# through the coordinator. a fresh launch on two workers reaches the state the
# single-grid run reaches, interior cell for interior cell. a launch resumed
# from a two-worker checkpoint onto three unevenly cut workers reaches the
# state the uninterrupted single-grid run reaches at the same final time. both
# gates drive the public command through a subprocess, so argument parsing,
# worker startup, execution, and checkpoint publication are covered together,
# and both record the backend hash every worker reports and each worker's exit
# code.
# =============================================================================
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import launcher
from simbi.simulation.runner import _load_backend, to_execution_dict
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang
from simbi_configs.examples.newtonian.ordered_sources import OrderedSources
from simbi_configs.examples.newtonian.refined_mhd_3d import RefinedMhd3D
from simbi_configs.examples.srhd.ram44 import Ram44

from simbi.simulation.tests.checkpoint_compare import assert_same_checkpoint
from simbi.simulation.tests.process_guard import ProcessGuard, descendants
from simbi.simulation.tests.staggered_div_b import DIV_B_TOLERANCE, normalized_div_b

pytestmark = pytest.mark.simulation


@pytest.fixture(autouse=True)
def bounded_processes():
    """every gate here re-executes the command line: the descendants of this test process are
    bounded, and a gate that passes the bound fails with its processes killed."""
    # the steady-state bound is 16 descendants; each process's h5py platform probe (`uname -p`)
    # is budgeted above it during the process's first ten seconds, which covers interpreter start
    # and the imports on a loaded machine
    with ProcessGuard(limit=16, startup=10.0) as guard:
        yield guard
    assert not guard.tripped, (
        f"the test started more than {guard.limit} processes (peak {guard.peak}):\n" + "\n".join(guard.report)
    )
    assert not [p for p in descendants(os.getpid())], "the test left processes behind"

RESOLUTION = (32, 32)
STEPS = 8
SCRIPT = str(Path(KelvinHelmholtz.__module__.replace(".", "/") + ".py"))


PROVENANCE = re.compile(r"backend=([0-9a-f]{40}(?:-dirty)?) config=")
EXIT = re.compile(r"launch: worker (\d+) exit (-?\d+)")
GPU_BACKEND = _load_backend("gpu") if "SIMBI_LAUNCH_GPU_TESTS" in os.environ else None
needs_gpu = pytest.mark.skipif(GPU_BACKEND is None, reason="set SIMBI_LAUNCH_GPU_TESTS on a gpu node with the gpu backend built")


@dataclass
class Launched:
    final: Path
    backend: set[str]
    exits: dict[int, int]
    stderr: str = ""


def _backend_of(text: str) -> set[str]:
    return set(PROVENANCE.findall(text))


REFERENCE_SNIPPET = """
import sys
from simbi.simulation import runner
from simbi.simulation import launcher
from simbi.simulation.runner import _load_backend, to_execution_dict
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz
out, steps = sys.argv[1], int(sys.argv[2])
extra = dict(arg.split("=", 1) for arg in sys.argv[3:])
mode = extra.pop("compute_mode", "cpu")
runner.run(KelvinHelmholtz(resolution=({rx}, {ry}), data_directory=out, checkpoint_interval=1e9, **extra), compute_mode=mode, max_steps=steps)
"""


def _reference(out: Path, steps: int, **extra: str) -> tuple[Path, set[str]]:
    """the single-grid run in a fresh interpreter: the example seeds its perturbation once
    per process, so a reference drawn in the test process would carry a different realization
    from the launched workers, which each start a process of their own."""
    snippet = REFERENCE_SNIPPET.format(rx=RESOLUTION[0], ry=RESOLUTION[1])
    args = [f"{k}={v}" for k, v in extra.items()]
    done = subprocess.run([sys.executable, "-c", snippet, str(out), str(steps), *args], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    # a single-grid run records its backend build in the checkpoint metadata and the run log; the
    # launched workers announce theirs on stderr
    with h5py.File(files[0], "r") as f:
        backend = {str(f["metadata"].attrs["build_source_id"])}
    assert all(PROVENANCE.fullmatch(f"backend={b} config=") for b in backend), f"the reference records backend {backend}"
    return files[0], backend


def _layout(path: Path, workers: int, cuts=None, shape=None, staging_mb: int = 1, directory: Path | None = None) -> Path:
    lines = ["[execution]", f"workers = {workers}", "", "[partition]"]
    if cuts is not None:
        lines.append("cuts = [" + ", ".join("[" + ", ".join(str(c) for c in axis) + "]" for axis in cuts) + "]")
    else:
        lines.append("shape = [" + ", ".join(str(s) for s in shape) + "]")
    lines += ["", "[checkpoint]", f"staging_mb = {staging_mb}"]
    if directory is not None:
        lines.append(f'directory = "{directory}"')
    path.write_text("\n".join(lines) + "\n")
    return path


def _launch_command(out: Path, layout: Path, steps: int, checkpoint: Path | None = None, interval: float = 1e9, flags: list[str] | None = None) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "simbi.cli",
        "launch",
        SCRIPT,
        "--layout",
        str(layout),
        "--max-steps",
        str(steps),
        "--resolution",
        ",".join(str(r) for r in RESOLUTION),
        "--data-directory",
        str(out),
        "--checkpoint-interval",
        str(interval),
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    cmd += flags or []
    return cmd


def _launch(out: Path, layout: Path, steps: int, workers: int, checkpoint: Path | None = None, interval: float = 1e9, flags: list[str] | None = None, output: Path | None = None, env: dict[str, str] | None = None) -> Launched:
    cmd = _launch_command(out, layout, steps, checkpoint, interval, flags)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)), f"exit codes reported for {sorted(exits)}, expected {workers} workers"
    assert all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    backend = _backend_of(done.stderr)
    assert len(backend) == 1, f"the workers report backends {backend}"
    where = output or out
    files = sorted(where.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {where}, found {files}"
    return Launched(final=files[0], backend=backend, exits=exits, stderr=done.stderr)


def _interior(f: h5py.File) -> dict[str, np.ndarray]:
    """every level-0 cell dataset trimmed to the interior: the state a checkpoint carries."""
    level = f["level_0"]
    ng = int(level["mesh"].attrs["halo_width"])
    names: list[str] = []
    level.visit(lambda name: names.append(name) if isinstance(level[name], h5py.Dataset) else None)
    out = {}
    for name in names:
        if name.startswith("mesh/") or name.startswith("partition_0/owned") or "/domain/" in name:
            continue
        data = level[name][()]
        if "/magnetic/" in name:
            # a staggered face field is stored over its interior faces alone
            out[name] = data
        elif data.ndim >= 2:
            # a cell field is stored with its halo on every axis
            out[name] = data[tuple(slice(ng, -ng) for _ in range(data.ndim))]
    assert out, "no field datasets under level_0"
    return out


def _assert_same_state(a: Path, b: Path, label: str) -> None:
    """the whole checkpoint, every group and dataset, halos and faces included, bit for bit"""
    assert_same_checkpoint(a, b, label)


def test_a_fresh_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference, backend = _reference(tmp_path / "single", STEPS)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched = _launch(tmp_path / "two", layout, STEPS, workers=2)
    assert launched.backend == backend, f"launched {launched.backend} against reference {backend}"
    assert launched.exits == {0: 0, 1: 0}
    _assert_same_state(launched.final, reference, "two workers")


def test_a_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path) -> None:
    reference, backend = _reference(tmp_path / "single", STEPS)
    # two workers write a bounded run's final checkpoint partway through the reference run
    first = tmp_path / "first"
    layout2 = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    partial = _launch(first, layout2, STEPS // 2, workers=2)
    assert partial.backend == backend
    with h5py.File(partial.final, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    # three unevenly cut workers resume that file up to the reference's absolute step cap
    layout3 = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _launch(tmp_path / "three", layout3, STEPS, workers=3, checkpoint=partial.final)
    assert resumed.backend == backend
    assert resumed.exits == {0: 0, 1: 0, 2: 0}
    with h5py.File(resumed.final, "r") as fr, h5py.File(reference, "r") as fs:
        assert fr["level_0"].attrs["time"] == fs["level_0"].attrs["time"], "the continuation ends at the reference's final time"
        assert int(fr["level_0"].attrs["iteration"]) == STEPS
    _assert_same_state(resumed.final, reference, "resumed on three workers")


def test_trouble_on_one_side_of_a_cut_matches_the_one_worker_launch(tmp_path: Path, monkeypatch) -> None:
    """a troubled cell in the column against the cut, its neighbor across the cut clean: both
    workers take the first-order flux on the shared face, so two workers equal one bitwise and
    the corrected run differs from the clean one."""
    cell = (RESOLUTION[0] // 2 - 1, RESOLUTION[1] // 3)
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    clean = _launch(tmp_path / "clean", one, STEPS, workers=1)
    monkeypatch.setenv("SIMBI_FABRIC_INJECT", f"trouble:3:0:{cell[0]}:{cell[1]}")
    reference = _launch(tmp_path / "one", one, STEPS, workers=1)
    launched = _launch(tmp_path / "two", two, STEPS, workers=2)
    with h5py.File(clean.final, "r") as fc, h5py.File(reference.final, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), (
            "the troubled cell left the one-worker run unchanged; the gate is vacuous"
        )
    assert launched.backend == reference.backend
    _assert_same_state(launched.final, reference.final, "two workers, trouble against the cut")


def test_a_ppm_launch_matches_the_single_grid_ppm_run(tmp_path: Path) -> None:
    """the requested reconstruction reaches the workers: a ppm launch equals a ppm single-grid
    run, and differs from the plm reference."""
    reference, backend = _reference(tmp_path / "single", STEPS, reconstruction="ppm")
    plm, _ = _reference(tmp_path / "plm", STEPS)
    with h5py.File(reference, "r") as fp, h5py.File(plm, "r") as fl:
        assert fp["level_0"].attrs["time"] != fl["level_0"].attrs["time"], "ppm and plm references coincide; the gate would be vacuous"
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched = _launch(tmp_path / "two", layout, STEPS, workers=2, flags=["--reconstruction", "ppm"])
    assert launched.backend == backend
    _assert_same_state(launched.final, reference, "ppm on two workers")


def test_the_layout_checkpoint_directory_is_where_the_files_land(tmp_path: Path) -> None:
    reference, _ = _reference(tmp_path / "single", STEPS)
    directory = tmp_path / "elsewhere"
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1), directory=directory)
    launched = _launch(tmp_path / "two", layout, STEPS, workers=2, output=directory)
    assert launched.final.parent == directory
    assert not (tmp_path / "two").exists() or not list((tmp_path / "two").glob("*.h5")), "files landed in the problem's data directory"
    assert not list(directory.glob(".rendezvous-*")), "the rendezvous file was left behind"
    _assert_same_state(launched.final, reference, "checkpoint directory")


def test_a_failed_spawn_terminates_the_workers_already_started(tmp_path: Path, monkeypatch) -> None:
    """the second spawn fails: the first worker is terminated and reaped, and no rendezvous
    file remains."""
    started: list[subprocess.Popen] = []
    real_popen = subprocess.Popen

    def popen(cmd, *args, **kwargs):
        if started:
            raise OSError("no more processes")
        proc = real_popen([sys.executable, "-c", "import time; time.sleep(60)"])
        started.append(proc)
        return proc

    monkeypatch.setattr(launcher.subprocess, "Popen", popen)
    layout = launcher.Layout(workers=2, cuts=[[16], []], owner=[0, 1])
    with pytest.raises(OSError):
        launcher.spawn_workers(layout, ["launch", "x.py"], str(tmp_path / "out"))
    assert len(started) == 1
    assert started[0].poll() is not None, "the started worker is still running"
    assert not list((tmp_path / "out").glob(".rendezvous-*"))


def test_a_terminated_launcher_ends_its_workers(tmp_path: Path) -> None:
    """SIGTERM to a running launcher terminates every worker and removes the rendezvous file."""
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = _launch_command(tmp_path / "two", layout, steps=0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    pids: list[int] = []
    deadline = time.monotonic() + 60
    while len(pids) < 2 and time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        m = re.search(r"launch: worker (\d+) pid (\d+)", line)
        if m:
            pids.append(int(m.group(2)))
    assert len(pids) == 2, "the launcher did not report two worker pids"
    time.sleep(1.0)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)
    proc.stdout.close()
    assert proc.returncode != 0
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            break
        time.sleep(0.1)
    assert not alive, f"workers {alive} survived the launcher"
    assert not list((tmp_path / "two").glob(".rendezvous-*")), "the rendezvous file was left behind"


def _problem_dict(**kwargs) -> dict:
    return to_execution_dict(KelvinHelmholtz(resolution=RESOLUTION, data_directory="/tmp/digest", **kwargs))


def test_the_configuration_digest_is_canonical() -> None:
    """equal configurations hash identically whatever the insertion order, at every depth; a
    resolved numerical parameter changes the digest; an unsupported value is refused."""
    backend = _load_backend("cpu")
    base = _problem_dict()
    reordered = {k: base[k] for k in reversed(list(base))}
    for key, value in list(reordered.items()):
        if isinstance(value, dict):
            reordered[key] = {k: value[k] for k in reversed(list(value))}
    assert backend.config_digest(base) == backend.config_digest(reordered)
    assert backend.config_digest(base) == backend.config_digest(_problem_dict()), "a rebuilt configuration hashes the same"
    changed = _problem_dict(cfl_number=0.05)
    assert backend.config_digest(base) != backend.config_digest(changed)
    tampered = dict(base)
    tampered["adiabatic_index"] = base["adiabatic_index"] * (1 + 1e-12)
    assert backend.config_digest(base) != backend.config_digest(tampered), "a one-ulp-scale change is a different configuration"
    unsupported = dict(base)
    unsupported["custom_params"] = object()
    with pytest.raises(ValueError, match="unsupported type"):
        backend.config_digest(unsupported)


def test_workers_with_different_resolved_parameters_refuse_to_rendezvous(tmp_path: Path) -> None:
    """two workers of one session started by hand with different cfl numbers: the handshake
    refuses the session as a configuration disagreement and both exit nonzero."""
    out = tmp_path / "out"
    out.mkdir()
    rendezvous = out / ".rendezvous-test"
    common = [
        sys.executable, "-m", "simbi.cli", "launch", SCRIPT,
        "--max-steps", "2", "--resolution", ",".join(str(r) for r in RESOLUTION),
        "--data-directory", str(out), "--checkpoint-interval", "1e9",
        "--workers", "2", "--rendezvous", str(rendezvous), "--credential", "12345",
        "--owner", "0,1", "--cuts", "16;", "--staging-cells", "131072",
    ]
    a = subprocess.Popen([*common, "--worker", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    b = subprocess.Popen([*common, "--worker", "1", "--cfl-number", "0.05"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    _, err_a = a.communicate(timeout=120)
    _, err_b = b.communicate(timeout=120)
    assert a.returncode != 0 and b.returncode != 0, f"worker exits {a.returncode}, {b.returncode}"
    assert "build or configuration disagreement" in err_a, err_a[-2000:]
    assert not list(out.glob("*.h5")), "a worker wrote output before the session was admitted"


def _lan_address() -> str | None:
    """an address of this host that is not loopback, if it has one."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            address = probe.getsockname()[0]
    except OSError:
        return None
    return None if address.startswith("127.") else address


def test_workers_rendezvous_over_a_non_loopback_address(tmp_path: Path) -> None:
    """the layout binds every worker on all interfaces and advertises this host's address:
    the rendezvous file carries that address, and the run equals the reference."""
    address = _lan_address()
    if address is None:
        pytest.skip("this host has no non-loopback address")
    reference, _ = _reference(tmp_path / "single", STEPS)
    layout = tmp_path / "lan.toml"
    layout.write_text(
        "[execution]\nworkers = 2\nbind = \"0.0.0.0\"\n"
        f'advertise = "{address}"\n\n[partition]\nshape = [2, 1]\n\n[checkpoint]\nstaging_mb = 1\n'
        "keep_rendezvous = true\n"
    )
    out = tmp_path / "lan"
    launched = _launch(out, layout, STEPS, workers=2)
    _assert_same_state(launched.final, reference, "non-loopback rendezvous")
    kept = sorted(out.glob(".rendezvous-*.kept"))
    assert kept, "the rendezvous record was not kept for inspection"
    first = kept[0].read_text().splitlines()[0]
    assert first.startswith(address + ":"), f"the coordinator advertised {first!r}, expected {address}"


def test_scheduler_started_workers_rendezvous_through_the_restricted_file(tmp_path: Path) -> None:
    """two workers started as a scheduler would, with rank and size in the environment and no
    launcher parent: the coordinator writes its advertised address and a fresh credential into
    a rendezvous file readable by its owner alone, the other worker reads it, and the run
    equals the reference."""
    reference, _ = _reference(tmp_path / "single", STEPS)
    out = tmp_path / "sched"
    layout = tmp_path / "sched.toml"
    layout.write_text(
        "[execution]\nmode = \"scheduler\"\nworkers = 2\n\n[partition]\nshape = [2, 1]\n\n"
        f'[checkpoint]\nstaging_mb = 1\ndirectory = "{out}"\nkeep_rendezvous = true\n'
    )
    cmd = [
        sys.executable, "-m", "simbi.cli", "launch", SCRIPT, "--layout", str(layout),
        "--max-steps", str(STEPS), "--resolution", ",".join(str(r) for r in RESOLUTION),
        "--data-directory", str(tmp_path / "ignored"), "--checkpoint-interval", "1e9",
    ]
    procs = []
    for rank in range(2):
        env = dict(os.environ, SLURM_PROCID=str(rank), SLURM_NTASKS="2")
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
    errs = [p.communicate(timeout=300)[1] for p in procs]
    for rank, (p, err) in enumerate(zip(procs, errs)):
        assert p.returncode == 0, f"scheduler-started worker {rank} failed:\n{err[-3000:]}"
    finals = sorted(out.glob("*final*.h5"))
    assert len(finals) == 1, f"expected one final checkpoint in {out}, found {finals}"
    assert not list((tmp_path / "ignored").glob("*.h5")), "the layout's directory was not honored"
    kept = out / ".rendezvous.kept"
    assert kept.exists(), "the rendezvous record was not kept for inspection"
    assert (kept.stat().st_mode & 0o777) == 0o600, "the rendezvous file is not owner-only"
    lines = kept.read_text().splitlines()
    assert len(lines) == 2 and int(lines[1]) != 0, "the file carries the address and a credential"
    _assert_same_state(finals[0], reference, "scheduler-started workers")


def test_the_guard_kills_its_own_descendants_alone() -> None:
    """a bystander started outside the guarded root survives a trip; the root's descendants
    end."""
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    root = subprocess.Popen(
        [sys.executable, "-c", "import subprocess, sys, time\nps = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']) for _ in range(4)]\ntime.sleep(30)"]
    )
    try:
        with ProcessGuard(limit=2, root=root.pid) as guard:
            deadline = time.monotonic() + 10
            while not guard.tripped and time.monotonic() < deadline:
                time.sleep(0.05)
        assert guard.tripped and guard.peak >= 3
        time.sleep(0.3)
        assert not descendants(root.pid), "the guarded descendants survived"
        assert bystander.poll() is None, "the guard reached a process outside its root"
    finally:
        for p in (root, bystander):
            p.kill()
            p.wait()


# -----------------------------------------------------------------------------
# ideal newtonian MHD through the public path. the orszag-tang vortex corrects nothing by
# itself; the correction-active arm raises the troubled flag on one physical cell against
# the cut at a named step and audits the shared faces after every exchange.
# -----------------------------------------------------------------------------

MHD_SCRIPT = str(Path(NewtonianOrszagTang.__module__.replace(".", "/") + ".py"))
MHD_RESOLUTION = (32, 32)
MHD_STEPS = 8
MHD_REFERENCE_SNIPPET = """
import sys
from simbi.simulation import runner
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang
out, steps = sys.argv[1], int(sys.argv[2])
runner.run(NewtonianOrszagTang(resolution=({rx}, {ry}, 1), data_directory=out, checkpoint_interval=1e9), compute_mode="cpu", max_steps=steps)
"""


def _mhd_reference(out: Path, steps: int) -> Path:
    snippet = MHD_REFERENCE_SNIPPET.format(rx=MHD_RESOLUTION[0], ry=MHD_RESOLUTION[1])
    done = subprocess.run([sys.executable, "-c", snippet, str(out), str(steps)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _mhd_launch_command(out: Path, layout: Path, steps: int, checkpoint: Path | None = None) -> list[str]:
    cmd = [
        sys.executable, "-m", "simbi.cli", "launch", MHD_SCRIPT,
        "--layout", str(layout), "--max-steps", str(steps),
        "--resolution", f"{MHD_RESOLUTION[0]},{MHD_RESOLUTION[1]},1",
        "--data-directory", str(out), "--checkpoint-interval", "1e9",
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    return cmd


def _mhd_launch(out: Path, layout: Path, steps: int, workers: int, checkpoint: Path | None = None, env: dict[str, str] | None = None) -> Launched:
    full_env = dict(os.environ, **(env or {}))
    done = subprocess.run(_mhd_launch_command(out, layout, steps, checkpoint), capture_output=True, text=True, timeout=600, env=full_env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return Launched(final=files[0], backend=_backend_of(done.stderr), exits=exits)


def _max_div_b(path: Path) -> float:
    """the largest |div B| over the cells of a checkpoint from its staggered faces."""
    with h5py.File(path, "r") as f:
        b1 = f["level_0/partition_0/hydro/magnetic/B1/data"][()]
        b2 = f["level_0/partition_0/hydro/magnetic/B2/data"][()]
    ny, nx1 = b1.shape
    nx = nx1 - 1
    assert b2.shape == (ny + 1, nx), f"face shapes {b1.shape}, {b2.shape}"
    h = 1.0 / nx
    div = (b1[:, 1:] - b1[:, :-1]) / h + (b2[1:, :] - b2[:-1, :]) / h
    return float(abs(div).max())


def test_a_magnetized_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _mhd_reference(tmp_path / "single", MHD_STEPS)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched = _mhd_launch(tmp_path / "two", layout, MHD_STEPS, workers=2)
    with h5py.File(reference, "r") as f:
        assert "level_0/partition_0/hydro/magnetic/B1/data" in f, "the reference carries no staggered faces"
    _assert_same_state(launched.final, reference, "two magnetized workers")
    assert _max_div_b(launched.final) < 1e-10


def test_a_magnetized_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path) -> None:
    reference = _mhd_reference(tmp_path / "single", MHD_STEPS)
    layout2 = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    partial = _mhd_launch(tmp_path / "first", layout2, MHD_STEPS // 2, workers=2)
    with h5py.File(partial.final, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < MHD_STEPS
    layout3 = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _mhd_launch(tmp_path / "three", layout3, MHD_STEPS, workers=3, checkpoint=partial.final)
    with h5py.File(resumed.final, "r") as fr, h5py.File(reference, "r") as fs:
        assert fr["level_0"].attrs["time"] == fs["level_0"].attrs["time"]
        assert int(fr["level_0"].attrs["iteration"]) == MHD_STEPS
    _assert_same_state(resumed.final, reference, "magnetized, resumed on three workers")
    assert _max_div_b(resumed.final) < 1e-10


def test_a_magnetized_launch_with_a_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    """the flag raised on the cell against the cut at step 3 in the predictor, on whichever
    worker holds it, with the shared faces audited after every exchange: two workers equal one
    worker under the same flag, and the flag changed the run."""
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    clean = _mhd_launch(tmp_path / "clean", one, MHD_STEPS, workers=1)
    inject = {"SIMBI_FABRIC_INJECT": f"trouble:3:0:{MHD_RESOLUTION[0] // 2 - 1}:{MHD_RESOLUTION[1] // 3}", "SIMBI_FABRIC_AUDIT": "all"}
    reference = _mhd_launch(tmp_path / "one", one, MHD_STEPS, workers=1, env=inject)
    launched = _mhd_launch(tmp_path / "two", two, MHD_STEPS, workers=2, env=inject)
    with h5py.File(clean.final, "r") as fc, h5py.File(reference.final, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the flag left the one-worker run unchanged; the gate is vacuous"
    _assert_same_state(launched.final, reference.final, "two magnetized workers, trouble against the cut")
    assert _max_div_b(launched.final) < 1e-10


def test_a_shared_face_mismatch_exits_every_worker_and_publishes_no_checkpoint(tmp_path: Path) -> None:
    """worker 1's opening copy of a shared face moved by a unit in the last place after step
    3: the audit before the next checkpoint names it, every worker exits nonzero, and no
    checkpoint is published."""
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    out = tmp_path / "two"
    env = dict(os.environ, SIMBI_FABRIC_INJECT="face:1:3")
    done = subprocess.run(_mhd_launch_command(out, layout, MHD_STEPS), capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == [0, 1] and all(c != 0 for c in exits.values()), f"worker exit codes {exits}"
    assert "shared-face mismatch on bface0" in done.stderr, done.stderr
    # the checkpoint of the primed state, written before the march, is the only file: no
    # checkpoint of a state after the perturbation, the final one included, was published
    published = sorted(out.glob("*.h5"))
    assert len(published) == 1 and "final" not in published[0].name, f"published: {published}"
    with h5py.File(published[0], "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == 0, "a checkpoint after the perturbation was published"


@pytest.mark.parametrize(
    ("region", "reason"),
    [([0.25, 0.75, 0.25, 0.75, 0.0, 0.5], "a refined patch on the periodic face of axis 2 with mesh refinement")],
)
def test_out_of_scope_problems_are_refused_by_name(tmp_path: Path, region: list[float], reason: str) -> None:
    # the 3d vortex's fine box narrowed along z to reach the periodic z face without spanning the
    # axis, so it has a face on the domain's seam; a layout cutting z alone makes the refusal the
    # scope's
    script = tmp_path / "one_sided_vortex.py"
    script.write_text(
        "from typing import Annotated\n"
        "from simbi import ProblemParam\n"
        f"from {RefinedMhd3D.__module__} import RefinedMhd3D\n\n\n"
        "class OneSided(RefinedMhd3D):\n"
        f"    refinement_regions: Annotated[list[list[float]], ProblemParam([{region!r}], description='r')]\n"
    )
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(1, 1, 2))
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--max-steps", "1", "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.getcwd(), os.environ.get("PYTHONPATH", "")]))
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    # the traceback renderer wraps a long message across lines
    assert f"{reason} is outside the distributed first release" in " ".join(done.stderr.split()), done.stderr
    assert not (out.exists() and list(out.glob("*.h5"))), "a checkpoint was published"


def test_a_magnetized_launch_refuses_the_numerics_its_substrate_has_no_door_for(tmp_path: Path) -> None:
    """a magnetized run installs the limiter, the solver and the edge EMF; a request for a
    reconstruction, a flattening or an equation of state it cannot install is refused by
    name rather than ignored, so a comparison never appears to exercise it."""
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    # a synge equation of state is refused by the problem's own validation before the launch
    # scope sees it, so the reconstruction request is the one that reaches the scope
    for flags, reason in [(["--reconstruction", "ppm"], "ppm reconstruction is wired for the newtonian (adiabatic ideal-gas) regime only; got 'nmhd'")]:
        out = tmp_path / reason.split()[0]
        cmd = _mhd_launch_command(out, layout, 1) + flags
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        assert done.returncode != 0, f"the launch succeeded with {flags}:\n{done.stdout}"
        # the error printer wraps long messages at the terminal width
        flat = " ".join(done.stderr.split())
        assert reason in flat, done.stderr


def test_a_rejection_on_one_worker_rolls_every_worker_back(tmp_path: Path) -> None:
    """a rejection forced on worker 1 at step 5 in the corrector: two workers reach the state
    one worker reaches under the same rejection, both count it, and the rejection changed the
    run. the replay starts from the step entry on every worker, which the worker-level gate
    establishes against a clean run on the accepted timestep sequence."""
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    rejections = re.compile(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)")

    def launch(out: Path, layout: Path, workers: int, inject: str | None) -> tuple[Path, set[int]]:
        env = dict(os.environ)
        if inject is not None:
            env["SIMBI_FABRIC_INJECT"] = inject
        done = subprocess.run(_launch_command(out, layout, STEPS), capture_output=True, text=True, timeout=600, env=env)
        assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
        exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
        assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
        files = sorted(out.glob("*final*.h5"))
        assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
        return files[0], {int(r) for r in rejections.findall(done.stderr)}

    clean, clean_rejections = launch(tmp_path / "clean", one, 1, None)
    assert clean_rejections == {0}
    reference, reference_rejections = launch(tmp_path / "one", one, 1, "reject:0:5:1")
    assert reference_rejections == {1}, "the one-worker run did not count its rejection"
    launched, launched_rejections = launch(tmp_path / "two", two, 2, "reject:1:5:1")
    assert launched_rejections == {1}, "a worker did not count the rejection"
    with h5py.File(clean, "r") as fc, h5py.File(reference, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the rejection left the run unchanged; the gate is vacuous"
    _assert_same_state(launched, reference, "two workers, rejection on worker 1")


REPORT_LINE = re.compile(r"SIMBI launch: worker=(\d+)/\d+ steps=(\d+) rejections=(\d+)")
MODES = ["cpu", pytest.param("gpu", marks=needs_gpu)]


def _mode_flags(mode: str) -> list[str]:
    return ["--gpu"] if mode == "gpu" else []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("magnetized", [False, True], ids=["hydro", "mhd"])
def test_a_rejection_on_one_worker_replays_every_worker_against_the_capped_clean_launch(tmp_path: Path, magnetized: bool, mode: str) -> None:
    """a rejection forced on worker 1 at step 5 in the corrector replays the step on every
    worker from its entry: the conserved state, and on a magnetized store the cell field and the
    staggered faces. the launch stops after that step. every worker counts the one rejection, and
    the state equals the clean two-worker launch whose step 5 is capped at the timestep the
    replay accepted (no rejection there, the same accepted timestep sequence entered from a state
    no rollback touched), bit for bit with the faces, the clock and the metadata; it also equals
    one worker under the same rejection."""
    rejected_step = 5
    steps = rejected_step + 1
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))

    def launch(out: Path, layout: Path, workers: int, inject: str | None) -> tuple[Path, dict[int, int]]:
        env = dict(os.environ)
        env.pop("SIMBI_FABRIC_INJECT", None)
        if inject is not None:
            env["SIMBI_FABRIC_INJECT"] = inject
        command = _mhd_launch_command(out, layout, steps) if magnetized else _launch_command(out, layout, steps)
        done = subprocess.run(command + _mode_flags(mode), capture_output=True, text=True, timeout=600, env=env)
        assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
        exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
        assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
        reports = {int(w): (int(n), int(r)) for w, n, r in REPORT_LINE.findall(done.stderr)}
        assert sorted(reports) == list(range(workers)), f"the workers reported {reports}"
        assert all(n == steps for n, _ in reports.values()), f"the workers accepted {reports} steps, expected {steps}"
        files = sorted(out.glob("*final*.h5"))
        assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
        with h5py.File(files[0], "r") as f:
            assert int(f["level_0"].attrs["iteration"]) == steps, "the launch did not stop after the replayed step"
            if magnetized:
                assert "level_0/partition_0/hydro/magnetic/B1/data" in f, "the checkpoint carries no staggered faces"
        return files[0], {w: r for w, (_, r) in reports.items()}

    launched, launched_rejections = launch(tmp_path / "two", two, 2, f"reject:1:{rejected_step}:1")
    assert launched_rejections == {0: 1, 1: 1}, f"rejections per worker {launched_rejections}: a worker kept out of the rollback"
    clean, clean_rejections = launch(tmp_path / "clean", two, 2, None)
    assert clean_rejections == {0: 0, 1: 0}
    with h5py.File(launched, "r") as fr, h5py.File(clean, "r") as fc:
        accepted = float(fr["level_0"].attrs["dt"])
        assert accepted < float(fc["level_0"].attrs["dt"]), "the rejection left the timestep unreduced; the replay is never exercised"
    capped, capped_rejections = launch(tmp_path / "capped", two, 2, f"cap:{rejected_step}:{int(np.float64(accepted).view(np.uint64))}")
    assert capped_rejections == {0: 0, 1: 0}, "the capped control rejected a step"
    with h5py.File(capped, "r") as f:
        assert float(f["level_0"].attrs["dt"]) == accepted, "the capped control's last step took another timestep"
    _assert_same_state(launched, capped, "two workers, rejection on worker 1, against the capped clean launch")
    reference, reference_rejections = launch(tmp_path / "one", one, 1, f"reject:0:{rejected_step}:1")
    assert reference_rejections == {0: 1}, "the one-worker run did not count its rejection"
    _assert_same_state(launched, reference, "two workers, rejection on worker 1, against one worker")

# -----------------------------------------------------------------------------
# relativistic hydro through the public path, on the two-dimensional riemann problem of
# ram44. the correction-active arm raises the flag on one cell against the cut; the
# rejection arm forces one on worker 1 and compares with one worker under the same one.
# -----------------------------------------------------------------------------

RHD_SCRIPT = str(Path(Ram44.__module__.replace(".", "/") + ".py"))
RHD_RESOLUTION = (32, 32)
RHD_STEPS = 8
RHD_REFERENCE_SNIPPET = """
import sys
from simbi.simulation import runner
from simbi_configs.examples.srhd.ram44 import Ram44
out, steps = sys.argv[1], int(sys.argv[2])
runner.run(Ram44(resolution=({rx}, {ry}), data_directory=out, checkpoint_interval=1e9), compute_mode="cpu", max_steps=steps)
"""


def _rhd_reference(out: Path, steps: int) -> Path:
    snippet = RHD_REFERENCE_SNIPPET.format(rx=RHD_RESOLUTION[0], ry=RHD_RESOLUTION[1])
    done = subprocess.run([sys.executable, "-c", snippet, str(out), str(steps)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _rhd_launch_command(out: Path, layout: Path, steps: int, checkpoint: Path | None = None) -> list[str]:
    cmd = [
        sys.executable, "-m", "simbi.cli", "launch", RHD_SCRIPT,
        "--layout", str(layout), "--max-steps", str(steps),
        "--resolution", f"{RHD_RESOLUTION[0]},{RHD_RESOLUTION[1]}",
        "--data-directory", str(out), "--checkpoint-interval", "1e9",
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", str(checkpoint)]
    return cmd


def _rhd_launch(out: Path, layout: Path, steps: int, workers: int, checkpoint: Path | None = None, env: dict[str, str] | None = None) -> tuple[Path, set[int]]:
    full_env = dict(os.environ, **(env or {}))
    done = subprocess.run(_rhd_launch_command(out, layout, steps, checkpoint), capture_output=True, text=True, timeout=600, env=full_env)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)) and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    rejections = {int(r) for r in re.findall(r"SIMBI launch: worker=\d+/\d+ steps=\d+ rejections=(\d+)", done.stderr)}
    return files[0], rejections


def test_a_relativistic_launch_matches_the_single_grid_run(tmp_path: Path) -> None:
    reference = _rhd_reference(tmp_path / "single", RHD_STEPS)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched, _ = _rhd_launch(tmp_path / "two", layout, RHD_STEPS, workers=2)
    _assert_same_state(launched, reference, "two relativistic workers")


def test_a_relativistic_launch_resumed_onto_three_uneven_workers_matches(tmp_path: Path) -> None:
    reference = _rhd_reference(tmp_path / "single", RHD_STEPS)
    layout2 = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    partial, _ = _rhd_launch(tmp_path / "first", layout2, RHD_STEPS // 2, workers=2)
    with h5py.File(partial, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < RHD_STEPS
    layout3 = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed, _ = _rhd_launch(tmp_path / "three", layout3, RHD_STEPS, workers=3, checkpoint=partial)
    with h5py.File(resumed, "r") as fr, h5py.File(reference, "r") as fs:
        assert fr["level_0"].attrs["time"] == fs["level_0"].attrs["time"]
        assert int(fr["level_0"].attrs["iteration"]) == RHD_STEPS
    _assert_same_state(resumed, reference, "relativistic, resumed on three workers")


def test_a_relativistic_launch_with_a_correction_against_the_cut_matches_one_worker(tmp_path: Path) -> None:
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    clean, _ = _rhd_launch(tmp_path / "clean", one, RHD_STEPS, workers=1)
    inject = {"SIMBI_FABRIC_INJECT": f"trouble:3:0:{RHD_RESOLUTION[0] // 2 - 1}:{RHD_RESOLUTION[1] // 3}"}
    reference, _ = _rhd_launch(tmp_path / "one", one, RHD_STEPS, workers=1, env=inject)
    launched, _ = _rhd_launch(tmp_path / "two", two, RHD_STEPS, workers=2, env=inject)
    with h5py.File(clean, "r") as fc, h5py.File(reference, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the flag left the one-worker run unchanged; the gate is vacuous"
    _assert_same_state(launched, reference, "two relativistic workers, trouble against the cut")


def test_a_relativistic_rejection_on_one_worker_rolls_every_worker_back(tmp_path: Path) -> None:
    one = _layout(tmp_path / "one.toml", workers=1, shape=(1, 1))
    two = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    clean, clean_rejections = _rhd_launch(tmp_path / "clean", one, RHD_STEPS, workers=1)
    assert clean_rejections == {0}
    reference, reference_rejections = _rhd_launch(tmp_path / "one", one, RHD_STEPS, workers=1, env={"SIMBI_FABRIC_INJECT": "reject:0:5:1"})
    assert reference_rejections == {1}
    launched, launched_rejections = _rhd_launch(tmp_path / "two", two, RHD_STEPS, workers=2, env={"SIMBI_FABRIC_INJECT": "reject:1:5:1"})
    assert launched_rejections == {1}
    with h5py.File(clean, "r") as fc, h5py.File(reference, "r") as fr:
        xs, ys = _interior(fc), _interior(fr)
        assert any(not np.array_equal(xs[name], ys[name]) for name in xs), "the rejection left the run unchanged; the gate is vacuous"
    _assert_same_state(launched, reference, "two relativistic workers, rejection on worker 1")


# -----------------------------------------------------------------------------
# user sources through the public path: the ordered-sources example carries two pointwise
# sources each tile evaluates at its own global coordinates.
# -----------------------------------------------------------------------------

SOURCES_SCRIPT = str(Path(OrderedSources.__module__.replace(".", "/") + ".py"))
SOURCES_RESOLUTION = (24, 16)
SOURCES_REFERENCE_SNIPPET = """
import sys
from simbi.simulation import runner
from simbi_configs.examples.newtonian.ordered_sources import OrderedSources
out, steps = sys.argv[1], int(sys.argv[2])
runner.run(OrderedSources(resolution=({rx}, {ry}), data_directory=out, checkpoint_interval=1e9), compute_mode="cpu", max_steps=steps)
"""


def test_a_launch_with_user_sources_matches_the_single_grid_run(tmp_path: Path) -> None:
    snippet = SOURCES_REFERENCE_SNIPPET.format(rx=SOURCES_RESOLUTION[0], ry=SOURCES_RESOLUTION[1])
    done = subprocess.run([sys.executable, "-c", snippet, str(tmp_path / "single"), str(STEPS)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    references = sorted((tmp_path / "single").glob("*final*.h5"))
    assert len(references) == 1
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = [
        sys.executable, "-m", "simbi.cli", "launch", SOURCES_SCRIPT,
        "--layout", str(layout), "--max-steps", str(STEPS),
        "--resolution", f"{SOURCES_RESOLUTION[0]},{SOURCES_RESOLUTION[1]}",
        "--data-directory", str(tmp_path / "two"), "--checkpoint-interval", "1e9",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert exits == {0: 0, 1: 0}, f"worker exit codes {exits}"
    launched = sorted((tmp_path / "two").glob("*final*.h5"))
    assert len(launched) == 1
    _assert_same_state(launched[0], references[0], "two workers with user sources")



def test_a_final_checkpoint_after_a_cadence_checkpoint_of_the_same_step_is_published(tmp_path: Path) -> None:
    """an MHD launch writing a checkpoint every step up to its end time: the cadence checkpoint
    of the last step audits the shared faces, and the final checkpoint of that same step reuses
    the certification (its audit epoch is spent) in place of a second audit the fabric refuses;
    every worker exits zero and the final file equals the single-grid run. the run ends on its
    end time: the step that exhausts a step budget publishes through the final checkpoint alone."""
    end_time = 0.02
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    cmd = _mhd_launch_command(tmp_path / "two", layout, 0) + ["--end-time", repr(end_time)]
    cmd[cmd.index("--checkpoint-interval") + 1] = "1e-9"
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == [0, 1] and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    finals = sorted((tmp_path / "two").glob("*final*.h5"))
    assert len(finals) == 1, f"no final checkpoint was published: {sorted(p.name for p in (tmp_path / 'two').glob('*'))}"
    cadence = [p for p in (tmp_path / "two").glob("*.h5") if "final" not in p.name]
    with h5py.File(finals[0], "r") as f:
        last, time = int(f["level_0"].attrs["iteration"]), float(f["level_0"].attrs["time"])
    assert time == end_time and last > 1, f"the launch ended at iteration {last}, t = {time!r}"
    assert any(int(h5py.File(p, "r")["level_0"].attrs["iteration"]) == last for p in cadence), "no cadence checkpoint shares the last step; the gate is vacuous"
    snippet = MHD_REFERENCE_SNIPPET.format(rx=MHD_RESOLUTION[0], ry=MHD_RESOLUTION[1]).replace(
        "checkpoint_interval=1e9", f"checkpoint_interval=1e9, end_time={end_time!r}")
    done = subprocess.run([sys.executable, "-c", snippet, str(tmp_path / "single"), "0"], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    reference = sorted((tmp_path / "single").glob("*final*.h5"))
    assert len(reference) == 1, f"expected one final checkpoint, found {reference}"
    _assert_same_state(finals[0], reference[0], "two workers, a checkpoint every step")


DEVICE_LINE = re.compile(r"SIMBI launch: worker=(\d+) host=(\S*) pci=(\S+) device=")


def _bound_devices(stderr: str) -> dict[int, tuple[str, str]]:
    """each worker's physical device: its host and the gpu's pci address"""
    return {int(w): (host, pci) for w, host, pci in DEVICE_LINE.findall(stderr)}


@needs_gpu
def test_a_two_gpu_launch_matches_the_single_gpu_run(tmp_path: Path) -> None:
    """two workers, one gpu each, against the one-gpu single grid: the whole checkpoint bit for bit"""
    reference, backend = _reference(tmp_path / "single", STEPS, compute_mode="gpu")
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    launched = _launch(tmp_path / "two", layout, STEPS, workers=2, flags=["--gpu"])
    assert launched.backend == backend, f"launched {launched.backend} against reference {backend}"
    bound = _bound_devices(launched.stderr)
    assert sorted(bound) == [0, 1], f"the workers reported device bindings {bound}"
    assert bound[0][0] == bound[1][0], f"a local launch spanned hosts: {bound}"
    assert bound[0][1] != bound[1][1], f"both workers bound the gpu at one pci address: {bound}"
    _assert_same_state(launched.final, reference, "two gpu workers")


@needs_gpu
def test_a_two_gpu_launch_resumed_from_its_checkpoint_matches(tmp_path: Path) -> None:
    """half the steps on two gpus, a checkpoint, and the other half resumed on two gpus, against
    the uninterrupted single-gpu run"""
    reference, backend = _reference(tmp_path / "single", STEPS, compute_mode="gpu")
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    partial = _launch(tmp_path / "first", layout, STEPS // 2, workers=2, flags=["--gpu"])
    resumed = _launch(tmp_path / "second", layout, STEPS, workers=2, checkpoint=partial.final, flags=["--gpu"])
    assert resumed.backend == backend
    _assert_same_state(resumed.final, reference, "two gpu workers, resumed")


# -----------------------------------------------------------------------------
# the session's physical device bindings. a fault reports one pci address for every worker,
# on this host or on a host of each worker's own, so the agreement runs on a host build.
# -----------------------------------------------------------------------------


def test_workers_bound_to_one_gpu_of_one_host_are_refused_together(tmp_path: Path) -> None:
    """two workers reporting one pci address on one host: every worker refuses at the binding
    agreement, before any state moves, within the session's startup, and nothing is published"""
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    out = tmp_path / "two"
    env = dict(os.environ, SIMBI_FABRIC_INJECT="device:shared:0000:c1:00.0")
    start = time.monotonic()
    done = subprocess.run(_launch_command(out, layout, STEPS), capture_output=True, text=True, timeout=600, env=env)
    elapsed = time.monotonic() - start
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    bound = _bound_devices(done.stderr)
    assert sorted(bound) == [0, 1] and bound[0] == bound[1], f"the fault did not bind one gpu twice: {bound}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == [0, 1] and all(c > 0 for c in exits.values()), f"worker exit codes {exits}"
    flat = " ".join(done.stderr.split())
    assert flat.count("workers 0 and 1 bound one physical gpu") == 2, f"a worker did not refuse at the agreement:\n{done.stderr}"
    # well inside the 120 s rendezvous budget and the 600 s collective deadline: the refusal is
    # the agreement's, not a stalled peer's
    assert elapsed < 120, f"the refusal took {elapsed:.1f} s"
    assert not list(out.glob("*.h5")), "a checkpoint was published"


def test_one_pci_address_on_two_hosts_is_two_gpus(tmp_path: Path) -> None:
    """the same pci address on two hosts names two gpus: the session runs and equals the
    single-grid run"""
    reference, _ = _reference(tmp_path / "single", STEPS)
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    env = dict(os.environ, SIMBI_FABRIC_INJECT="device:hosts:0000:c1:00.0")
    launched = _launch(tmp_path / "two", layout, STEPS, workers=2, env=env)
    bound = _bound_devices(launched.stderr)
    assert sorted(bound) == [0, 1], f"device bindings {bound}"
    assert bound[0][1] == bound[1][1] and bound[0][0] != bound[1][0], f"the fault did not share a pci address across hosts: {bound}"
    _assert_same_state(launched.final, reference, "one pci address on two hosts")


@needs_gpu
def test_two_gpu_workers_masked_onto_one_gpu_are_refused_together(tmp_path: Path) -> None:
    """a parent mask naming one device twice hands both workers the same physical gpu, each as
    its logical device 0: the binding agreement refuses on every worker and nothing is published"""
    present = [m for m in launcher.GPU_MASKS if m in os.environ]
    first = os.environ[present[0]].split(",")[0].strip() if present else "0"
    env = dict(os.environ, **{m: f"{first},{first}" for m in (present or ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"])})
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    out = tmp_path / "two"
    done = subprocess.run(_launch_command(out, layout, STEPS, flags=["--gpu"]), capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    bound = _bound_devices(done.stderr)
    assert sorted(bound) == [0, 1] and bound[0] == bound[1], f"the mask did not put both workers on one gpu: {bound}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == [0, 1] and all(c > 0 for c in exits.values()), f"worker exit codes {exits}"
    flat = " ".join(done.stderr.split())
    assert flat.count("workers 0 and 1 bound one physical gpu") == 2, f"a worker did not refuse at the agreement:\n{done.stderr}"
    assert not list(out.glob("*.h5")), "a checkpoint was published"


# -----------------------------------------------------------------------------
# magnetized parity to a requested end time, every checkpoint of the run compared
# -----------------------------------------------------------------------------

MHD_END_TIME = 0.05
MHD_INTERVAL = 0.0125
AUDIT_LINE = re.compile(r"SIMBI launch: worker=(\d+)/\d+ .*?checkpoint_audits=(\d+) audit_faces=(\d+)")
MHD_RUN_SNIPPET = """
import sys
from simbi.simulation import runner
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang
out, mode, end, interval = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
runner.run(NewtonianOrszagTang(resolution=({rx}, {ry}, 1), data_directory=out, end_time=end, checkpoint_interval=interval), compute_mode=mode)
"""


@pytest.mark.parametrize("mode", MODES)
def test_a_magnetized_launch_equals_one_device_at_every_checkpoint_to_its_end_time(tmp_path: Path, mode: str) -> None:
    """orszag-tang to t = 0.05 with a checkpoint every 0.0125: two workers against one device,
    every checkpoint of the run bit for bit with its staggered faces, the shared faces audited
    after every exchange, and the divergence of the stored faces at every checkpoint of both runs
    within DIV_B_TOLERANCE of the largest field per cell width"""
    snippet = MHD_RUN_SNIPPET.format(rx=MHD_RESOLUTION[0], ry=MHD_RESOLUTION[1])
    single = tmp_path / "single"
    done = subprocess.run([sys.executable, "-c", snippet, str(single), mode, repr(MHD_END_TIME), repr(MHD_INTERVAL)], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    out = tmp_path / "two"
    cmd = _mhd_launch_command(out, layout, 0) + ["--end-time", repr(MHD_END_TIME)] + _mode_flags(mode)
    cmd[cmd.index("--checkpoint-interval") + 1] = repr(MHD_INTERVAL)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=dict(os.environ, SIMBI_FABRIC_AUDIT="all"))
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == [0, 1] and all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    launched, references = sorted(out.glob("*.h5")), sorted(single.glob("*.h5"))
    assert [p.name for p in launched] == [p.name for p in references], f"checkpoints {launched} against {references}"
    assert len(launched) >= 4, f"the run published {len(launched)} checkpoints; the cadence is not exercised"
    for got, want in zip(launched, references):
        _assert_same_state(got, want, f"two workers against one device, {got.name}")
    for path in (launched[-1], references[-1]):
        with h5py.File(path, "r") as f:
            assert "final" in path.name and float(f["level_0"].attrs["time"]) == MHD_END_TIME, f"{path} ends at t = {f['level_0'].attrs['time']!r}"
    divergence = {p.name: normalized_div_b(p) for p in launched + references}
    assert all(d <= DIV_B_TOLERANCE for d in divergence.values()), f"normalized div B {divergence}"
    audits = {int(w): (int(n), int(faces)) for w, n, faces in AUDIT_LINE.findall(done.stderr)}
    assert sorted(audits) == [0, 1] and all(n > 0 and faces > 0 for n, faces in audits.values()), f"shared-face audits {audits}"
    if mode == "gpu":
        bound = _bound_devices(done.stderr)
        assert sorted(bound) == [0, 1] and bound[0] != bound[1], f"device bindings {bound}"


# -----------------------------------------------------------------------------
# a worker lost during evolution
# -----------------------------------------------------------------------------

PID_LINE = re.compile(r"launch: worker (\d+) pid (\d+)")


def _wait_for(condition, seconds: float, what: str):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = condition()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError(f"{what} within {seconds} s")


def _published_iterations(out: Path) -> dict[Path, int]:
    """every published checkpoint by its iteration; a file the writer renamed into place is whole"""
    found = {}
    for path in out.glob("*.h5"):
        with h5py.File(path, "r") as f:
            found[path] = int(f["level_0"].attrs["iteration"])
    return found


@pytest.mark.parametrize("victim", [0, 1], ids=["coordinator", "peer"])
@pytest.mark.parametrize("mode", MODES)
def test_a_killed_worker_ends_the_session_and_keeps_the_last_complete_checkpoint(tmp_path: Path, mode: str, victim: int) -> None:
    """a worker killed by SIGKILL once a checkpoint past the second step is published: the
    survivor detects the loss itself and exits nonzero within the launcher's 5 s grace, the
    launcher exits nonzero and records both exits, no final checkpoint is published, and the last
    published checkpoint resumes"""
    layout = _layout(tmp_path / "two.toml", workers=2, shape=(2, 1))
    out = tmp_path / "two"
    cmd = _launch_command(out, layout, 1_000_000, interval=0.02, flags=["--end-time", "1e9", *_mode_flags(mode)])
    stdout, stderr = tmp_path / "launcher.out", tmp_path / "launcher.err"
    with open(stdout, "w") as so, open(stderr, "w") as se:
        proc = subprocess.Popen(cmd, stdout=so, stderr=se, text=True)
        try:
            pids = _wait_for(lambda: (lambda p: p if len(p) == 2 else None)({int(w): int(pid) for w, pid in PID_LINE.findall(stdout.read_text())}), 120, "two worker pids")
            _wait_for(lambda: any(it >= 2 for it in _published_iterations(out).values()), 300, "a checkpoint past the second step")
            os.kill(pids[victim], signal.SIGKILL)
            killed = time.monotonic()
            proc.wait(timeout=120)
            ended = time.monotonic() - killed
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    assert proc.returncode != 0, "the launcher reported success"
    exits = {int(w): int(c) for w, c in EXIT.findall(stdout.read_text())}
    survivor = 1 - victim
    assert exits.get(victim) == -signal.SIGKILL, f"worker exit codes {exits}"
    # a positive code is the survivor's own exit; the launcher's termination would be a signal
    assert exits.get(survivor, 0) > 0, f"the survivor did not end on its own: worker exit codes {exits}"
    assert ended < 5.0 + 2.0 + 5.0, f"the session took {ended:.1f} s to end after the kill"
    record = " ".join(stderr.read_text().split())
    assert f"worker {victim}" in record or "disconnected" in record.lower(), f"the survivor's record names no lost peer:\n{stderr.read_text()[-3000:]}"
    published = _published_iterations(out)
    assert published and not [p for p in published if "final" in p.name], f"published {sorted(p.name for p in published)}"
    assert not list(out.glob(".rendezvous-*")), "the rendezvous file was left behind"
    last = max(published, key=published.get)
    resumed = _launch(tmp_path / "resumed", layout, published[last] + 2, workers=2, checkpoint=last, flags=_mode_flags(mode))
    with h5py.File(resumed.final, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == published[last] + 2
