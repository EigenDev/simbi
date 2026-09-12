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

pytestmark = pytest.mark.simulation

RESOLUTION = (32, 32)
STEPS = 8
SCRIPT = str(Path(KelvinHelmholtz.__module__.replace(".", "/") + ".py"))


PROVENANCE = re.compile(r"backend=([0-9a-f]{40}(?:-dirty)?) config=")
EXIT = re.compile(r"launch: worker (\d+) exit (-?\d+)")


@dataclass
class Launched:
    final: Path
    backend: set[str]
    exits: dict[int, int]


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
runner.run(KelvinHelmholtz(resolution=({rx}, {ry}), data_directory=out, checkpoint_interval=1e9, **extra), compute_mode="cpu", max_steps=steps)
"""


def _reference(out: Path, steps: int, **extra: str) -> tuple[Path, set[str]]:
    """the single-grid run in a fresh interpreter: the example seeds its perturbation once
    per process, so a reference drawn in the test process would carry a different realization
    from the launched workers, which each start a process of their own."""
    snippet = REFERENCE_SNIPPET.format(rx=RESOLUTION[0], ry=RESOLUTION[1])
    args = [f"{k}={v}" for k, v in extra.items()]
    done = subprocess.run([sys.executable, "-c", snippet, str(out), str(steps), *args], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"reference failed:\n{done.stderr}"
    backend = _backend_of(done.stderr)
    assert len(backend) == 1, f"the reference reports backends {backend}"
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
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


def _launch(out: Path, layout: Path, steps: int, workers: int, checkpoint: Path | None = None, interval: float = 1e9, flags: list[str] | None = None, output: Path | None = None) -> Launched:
    cmd = _launch_command(out, layout, steps, checkpoint, interval, flags)
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, f"launch failed:\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
    assert sorted(exits) == list(range(workers)), f"exit codes reported for {sorted(exits)}, expected {workers} workers"
    assert all(c == 0 for c in exits.values()), f"worker exit codes {exits}"
    backend = _backend_of(done.stderr)
    assert len(backend) == 1, f"the workers report backends {backend}"
    where = output or out
    files = sorted(where.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {where}, found {files}"
    return Launched(final=files[0], backend=backend, exits=exits)


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
        if data.ndim == 2:
            out[name] = data[ng:-ng, ng:-ng]
    assert out, "no field datasets under level_0"
    return out


def _assert_same_state(a: Path, b: Path, label: str) -> None:
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for key in ("time", "iteration"):
            assert fa["level_0"].attrs[key] == fb["level_0"].attrs[key], f"{label}: level attr {key}"
        xs, ys = _interior(fa), _interior(fb)
        assert xs.keys() == ys.keys(), f"{label}: dataset sets differ"
        for name, x in xs.items():
            np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


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
    # three unevenly cut workers resume that file for the remaining steps
    layout3 = _layout(tmp_path / "three.toml", workers=3, cuts=[[10, 21], []])
    resumed = _launch(tmp_path / "three", layout3, STEPS - written_at, workers=3, checkpoint=partial.final)
    assert resumed.backend == backend
    assert resumed.exits == {0: 0, 1: 0, 2: 0}
    with h5py.File(resumed.final, "r") as fr, h5py.File(reference, "r") as fs:
        assert fr["level_0"].attrs["time"] == fs["level_0"].attrs["time"], "the continuation ends at the reference's final time"
        assert int(fr["level_0"].attrs["iteration"]) == STEPS
    _assert_same_state(resumed.final, reference, "resumed on three workers")


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
