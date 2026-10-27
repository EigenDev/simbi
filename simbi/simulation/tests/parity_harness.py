# =============================================================================
# parity_harness.py
#
# the shared backend and placement of the distributed parity gates. a session
# runs every gate on one backend and one placement, read from the environment,
# so one test id runs on the cpu or the gpu, its launch on one node or across
# nodes, and its single-device and tiled references on the same backend:
#   SIMBI_PARITY_BACKEND    cpu (default) | gpu
#   SIMBI_PARITY_PLACEMENT  local (default): `simbi launch` starts the workers
#                           scheduler: one process per worker, started by srun
#                           under SIMBI_PARITY_SRUN=1 (inside an allocation),
#                           otherwise by this harness with the scheduler's rank
#                           and size variables (the same code path on one host)
#   SIMBI_PARITY_NODES      nodes the scheduler placement spreads workers over
#   SIMBI_PARITY_CPUS_PER_TASK  srun -c for every task (default 7)
#   SIMBI_PARITY_LOGS       a directory receiving every command's output, by
#                           test id and call order
# every failure the harness raises carries its category: an admission refusal,
# a missing implementation, an execution failure, a harness failure, a timeout,
# resource exhaustion, or an infrastructure failure (wrong device placement,
# stale binary, exhausted storage, unreaped workers), which stops a batch. a
# disagreement between the distributed run and its reference is a
# NumericalDisagreement. the pytest plugin in conftest.py records each test's
# category with the session's backend, placement and source identity.
#
# usage:
#   from simbi.simulation.tests import parity_harness as ph
#   ref = ph.python(["-c", SNIPPET, str(out)], "reference")      # backend's single device
#   run = ph.launch(script, out, workers=2, cuts=[[12], [], []], args=["--max-steps", "8"])
#   ph.same_checkpoint(run.final, ref_final, "two workers")
# =============================================================================
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

BACKEND = os.environ.get("SIMBI_PARITY_BACKEND", "cpu")
PLACEMENT = os.environ.get("SIMBI_PARITY_PLACEMENT", "local")
NODES = int(os.environ.get("SIMBI_PARITY_NODES", "1"))
SRUN = os.environ.get("SIMBI_PARITY_SRUN") == "1"
CPUS_PER_TASK = os.environ.get("SIMBI_PARITY_CPUS_PER_TASK", "7")
LOGS = os.environ.get("SIMBI_PARITY_LOGS")
assert BACKEND in ("cpu", "gpu"), f"SIMBI_PARITY_BACKEND={BACKEND!r}; give cpu or gpu"
assert PLACEMENT in ("local", "scheduler"), f"SIMBI_PARITY_PLACEMENT={PLACEMENT!r}; give local or scheduler"

EXIT = re.compile(r"launch: worker (\d+) exit (-?\d+)")
REPORT = re.compile(r"SIMBI launch: worker=(\d+)/\d+ steps=(\d+) rejections=(\d+)")
REMOTE = re.compile(r"SIMBI launch refined plans: worker=(\d+)/\d+ remote_sends=(\S+)")
BOUND = re.compile(r"SIMBI launch: worker=(\d+) host=(\S*) pci=(\S+) device=")
# every launched worker announces the host it started on, on either backend
STARTUP = re.compile(r"SIMBI launch startup: worker=(\d+)/\d+ host=(\S+)")
EXPERIMENTAL = "experimental combination"


class ParityFailure(Exception):
    """a failure other than a numerical disagreement, with the category the results record"""

    category = "harness failure"


class AdmissionRefusal(ParityFailure):
    category = "admission refusal"


class MissingImplementation(ParityFailure):
    category = "missing implementation"


class ExecutionFailure(ParityFailure):
    category = "execution failure"


class HarnessFailure(ParityFailure):
    category = "harness failure"


class Timeout(ParityFailure):
    category = "timeout"


class ResourceExhaustion(ParityFailure):
    category = "resource exhaustion"


class InfrastructureFailure(ParityFailure):
    """wrong device placement, a stale binary, exhausted storage or unreaped workers: the batch stops"""

    category = "infrastructure failure"


class NumericalDisagreement(AssertionError):
    """the distributed run and its reference differ under the exact criterion"""

    category = "numerical disagreement"


# the text a command prints for each category, checked in order; a refusal names its combination
_INFRASTRUCTURE = re.compile(r"bound one physical gpu|gpu workers but \S+ makes \d+ devices visible|the gpu extension")
_CLASSIFIERS = [
    (ResourceExhaustion, ("No space left on device", "Disk quota exceeded", "out of memory", "OutOfMemory", "hipErrorOutOfMemory",
                          "CUDA_ERROR_OUT_OF_MEMORY", "Cannot allocate memory")),
    (AdmissionRefusal, ("outside the distributed first release", "without a distributed gate", "run on one device or")),
    (MissingImplementation, ("not implemented", "no device path", "unimplemented", "does not support")),
]


def classify(text: str) -> type[ParityFailure]:
    """the category of a failed command, from its output"""
    flat = " ".join(text.split())
    if _INFRASTRUCTURE.search(flat):
        return InfrastructureFailure
    for kind, needles in _CLASSIFIERS:
        if any(n in flat for n in needles):
            return kind
    return ExecutionFailure


# ------------------------------------------------------------------------------------------------
# command execution with durable logs
# ------------------------------------------------------------------------------------------------
_calls: dict[str, int] = {}
# the workers' hosts of every launch the current test ran, by test id: one set per launch
observed_launch_hosts: dict[str, list[list[str]]] = {}


def _current_test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "session (call)").rsplit(" ", 1)[0]


def _log_dir() -> Path | None:
    if not LOGS:
        return None
    test = os.environ.get("PYTEST_CURRENT_TEST", "session (call)").rsplit(" ", 1)[0]
    path = Path(LOGS) / re.sub(r"[^A-Za-z0-9_.-]+", "_", test)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record(label: str, cmd: list[str], done: subprocess.CompletedProcess | None, elapsed: float, note: str = "") -> None:
    d = _log_dir()
    if d is None:
        return
    n = _calls[str(d)] = _calls.get(str(d), 0) + 1
    stem = d / f"{n:02d}-{re.sub(r'[^A-Za-z0-9_.-]+', '_', label)}"
    lines = [f"cmd: {' '.join(cmd)}", f"elapsed: {elapsed:.1f} s", f"exit: {done.returncode if done else 'none'}"]
    if note:
        lines.append(f"note: {note}")
    stem.with_suffix(".cmd").write_text("\n".join(lines) + "\n")
    if done is not None:
        stem.with_suffix(".out").write_text(done.stdout or "")
        stem.with_suffix(".err").write_text(done.stderr or "")


def _gpu_task_flags(devices: int) -> list[str]:
    return ["--gpus-per-task=1", "--gpu-bind=closest"] if devices == 1 else [f"--gpus={devices}"]


def python(args: list[str], label: str, timeout: float = 900, env: dict[str, str] | None = None, devices: int = 1,
           check: bool = True) -> subprocess.CompletedProcess:
    """`python <args>` on the session's backend: under the srun scheduler placement as one task
    holding `devices` gpus (a single device or in-process tiles); otherwise in this host's
    interpreter. a nonzero exit raises its category when `check`"""
    cmd = [sys.executable, *args]
    if SRUN:
        task = ["srun", "-N1", "-n1", f"-c{CPUS_PER_TASK}"]
        if BACKEND == "gpu":
            task += _gpu_task_flags(devices)
        cmd = task + cmd
    environ = dict(os.environ if env is None else env)
    if devices > 1:
        # host tiles: on the cpu build the logical devices are host tiles, on the gpu build they
        # fold onto the visible gpus
        environ.setdefault("SYMBI_GPU_OVERSUBSCRIBE", "1")
    start = time.monotonic()
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=environ)
    except subprocess.TimeoutExpired as e:
        _record(label, cmd, None, time.monotonic() - start, f"timeout after {timeout} s")
        raise Timeout(f"{label}: no exit within {timeout} s") from e
    _record(label, cmd, done, time.monotonic() - start)
    if check and done.returncode != 0:
        raise classify(done.stdout + done.stderr)(f"{label} exited {done.returncode}:\n{done.stderr[-3000:]}")
    return done


def snippet_args(code: str, *argv: str) -> list[str]:
    """the interpreter arguments running `code` with `argv`"""
    return ["-c", code, *argv]


# ------------------------------------------------------------------------------------------------
# layouts and launches
# ------------------------------------------------------------------------------------------------
def write_layout(path: Path, workers: int, cuts=None, shape=None, staging_mb: int = 1, directory: Path | None = None) -> Path:
    """the session placement's layout file: a local launch, or scheduler-started workers whose
    coordinator writes the checkpoints into `directory`"""
    lines = ["[execution]", f"workers = {workers}"]
    if PLACEMENT == "scheduler":
        lines += ['mode = "scheduler"', 'bind = "0.0.0.0"' if SRUN else 'bind = "127.0.0.1"']
    lines += ["", "[partition]"]
    if cuts is not None:
        lines.append("cuts = [" + ", ".join("[" + ", ".join(str(c) for c in axis) + "]" for axis in cuts) + "]")
    else:
        lines.append("shape = [" + ", ".join(str(s) for s in shape) + "]")
    lines += ["", "[checkpoint]", f"staging_mb = {staging_mb}"]
    if directory is not None:
        lines.append(f'directory = "{directory}"')
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass
class Launch:
    out: Path
    returncode: int
    stdout: str
    stderr: str
    exits: dict[int, int]
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def final(self) -> Path:
        finals = [p for name, p in self.files.items() if "final" in name]
        if len(finals) != 1:
            raise HarnessFailure(f"final checkpoints in {self.out}: {finals}")
        return finals[0]

    def reports(self) -> dict[int, tuple[int, int]]:
        """(steps, rejections) per worker"""
        return {int(w): (int(n), int(r)) for w, n, r in REPORT.findall(self.stderr)}

    def remote_sends(self) -> dict[int, dict[str, int]]:
        """each worker's remote transfers by plan role; a nested role names its level"""
        out = {}
        for worker, roles in REMOTE.findall(self.stderr):
            out[int(worker)] = {role: int(n) for role, n in (item.rsplit(":", 1) for item in roles.split(","))}
        return out

    def bindings(self) -> dict[int, tuple[str, str]]:
        return {int(w): (h, p) for w, h, p in BOUND.findall(self.stderr)}

    def hosts(self) -> dict[int, str]:
        """the host each worker announced at startup"""
        return {int(w): h for w, h in STARTUP.findall(self.stderr)}

    @property
    def experimental(self) -> bool:
        return EXPERIMENTAL in self.stderr


def launch(script: str | Path, out: Path, workers: int, cuts=None, shape=None, args: list[str] | None = None,
           env: dict[str, str] | None = None, timeout: float = 900, staging_mb: int = 1, expect_ok: bool = True,
           label: str = "launch") -> Launch:
    """`simbi launch` of `script` on the session's backend and placement, its checkpoints in `out`.
    the layout is written beside `out`. with `expect_ok` a failed launch raises its category, and
    on the gpu every worker must hold a gpu of its own (a shared one is an infrastructure failure)"""
    out = Path(out)
    layout = write_layout(out.parent / f"{out.name}.harness-layout.toml", workers, cuts=cuts, shape=shape, staging_mb=staging_mb,
                          directory=out if PLACEMENT == "scheduler" else None)
    cmd = ["-m", "simbi.cli", "launch", str(script), "--layout", str(layout), "--data-directory", str(out), *(args or [])]
    if BACKEND == "gpu":
        cmd.append("--gpu")
    environ = dict(os.environ if env is None else env)
    start = time.monotonic()
    if PLACEMENT == "local":
        full = [sys.executable, *cmd]
        try:
            done = subprocess.run(full, capture_output=True, text=True, timeout=timeout, env=environ)
        except subprocess.TimeoutExpired as e:
            _record(label, full, None, time.monotonic() - start, f"timeout after {timeout} s")
            raise Timeout(f"{label}: no exit within {timeout} s") from e
        exits = {int(w): int(c) for w, c in EXIT.findall(done.stdout)}
        result = Launch(out, done.returncode, done.stdout, done.stderr, exits)
    elif SRUN:
        nodes = min(NODES, workers)
        full = ["srun", f"-N{nodes}", f"-n{workers}", f"-c{CPUS_PER_TASK}"]
        if BACKEND == "gpu":
            full += ["--gpus-per-task=1", "--gpu-bind=closest"]
        full += [sys.executable, *cmd]
        try:
            done = subprocess.run(full, capture_output=True, text=True, timeout=timeout, env=environ)
        except subprocess.TimeoutExpired as e:
            _record(label, full, None, time.monotonic() - start, f"timeout after {timeout} s")
            raise Timeout(f"{label}: no exit within {timeout} s") from e
        result = Launch(out, done.returncode, done.stdout, done.stderr, {w: done.returncode for w in range(workers)})
    else:
        full, result = _emulated_scheduler([sys.executable, *cmd], workers, environ, timeout, label, start)
        done = subprocess.CompletedProcess(full, result.returncode, result.stdout, result.stderr)
    _record(label, full, done, time.monotonic() - start)
    result.files = {p.name: p for p in out.glob("*.h5")} if out.exists() else {}
    hosts = result.hosts()
    if hosts:
        observed_launch_hosts.setdefault(_current_test(), []).append(sorted(set(hosts.values())))
    if expect_ok:
        if result.returncode != 0 or any(c != 0 for c in result.exits.values()):
            raise classify(result.stdout + result.stderr)(f"{label} failed (exit {result.returncode}, workers {result.exits}):\n{result.stderr[-3000:]}")
        if PLACEMENT == "local" and sorted(result.exits) != list(range(workers)):
            raise HarnessFailure(f"{label}: worker exits {result.exits} for {workers} workers")
        if BACKEND == "gpu":
            bound = result.bindings()
            if sorted(bound) != list(range(workers)) or len(set(bound.values())) != workers:
                raise InfrastructureFailure(f"{label}: device bindings {bound} for {workers} workers")
        # srun spreads the workers over the allocation's nodes: the hosts they started on must span
        # as many nodes as there are workers to place, the evidence a cross-node result rests on
        if PLACEMENT == "scheduler" and SRUN:
            hosts = result.hosts()
            if sorted(hosts) != list(range(workers)) or len(set(hosts.values())) != min(NODES, workers):
                raise InfrastructureFailure(f"{label}: workers started on hosts {hosts}, expected {min(NODES, workers)} distinct hosts")
    return result


def _emulated_scheduler(cmd: list[str], workers: int, environ: dict[str, str], timeout: float, label: str,
                        start: float) -> tuple[list[str], Launch]:
    """one process per worker with the scheduler's rank and size variables, each the leader of its
    own process group, killed with its group on a timeout"""
    procs = []
    for rank in range(workers):
        env = dict(environ, SLURM_PROCID=str(rank), SLURM_NTASKS=str(workers))
        env.pop("SLURM_JOB_ID", None)
        env.pop("SLURM_STEP_ID", None)
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True))
    outs: list[tuple[str, str]] = []
    try:
        for p in procs:
            remaining = max(1.0, timeout - (time.monotonic() - start))
            outs.append(p.communicate(timeout=remaining))
    except subprocess.TimeoutExpired as e:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
        raise Timeout(f"{label}: scheduler-started workers did not exit within {timeout} s") from e
    exits = {rank: p.returncode for rank, p in enumerate(procs)}
    stdout = "".join(o for o, _ in outs)
    stderr = "".join(e for _, e in outs)
    code = 0 if all(c == 0 for c in exits.values()) else 1
    return cmd, Launch(Path(), code, stdout, stderr, exits)


# ------------------------------------------------------------------------------------------------
# comparisons
# ------------------------------------------------------------------------------------------------
def same_checkpoint(got: Path, want: Path, label: str, **kwargs) -> None:
    """the whole-file exact comparison, a difference raised as a NumericalDisagreement"""
    from simbi.simulation.tests.checkpoint_compare import assert_same_checkpoint

    disagreement(assert_same_checkpoint, got, want, label, **kwargs)


# the words of a comparison helper's premise assertion: the reference or the run carries nothing
# to compare, which records as an assertion
_PREMISE = re.compile(r"is vacuous|publishes no|published no|wrote no|holds no dataset")


def disagreement(check, *args, **kwargs) -> None:
    """run an exact comparison helper, its assertion raised as a NumericalDisagreement; a failed
    premise (nothing to compare) stays an assertion"""
    try:
        check(*args, **kwargs)
    except NumericalDisagreement:
        raise
    except AssertionError as e:
        if _PREMISE.search(str(e)):
            raise
        raise NumericalDisagreement(str(e)) from e


# ------------------------------------------------------------------------------------------------
# session identity
# ------------------------------------------------------------------------------------------------
def source_identity() -> dict[str, str]:
    """the backend extension's build identity and the repository head"""
    from simbi.simulation.runner import _load_backend

    built = str(getattr(_load_backend(BACKEND), "BUILD_SOURCE_ID", "unknown"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"backend": BACKEND, "placement": PLACEMENT, "nodes": str(NODES), "srun": str(SRUN), "built": built, "head": head}


def stale_binary(identity: dict[str, str]) -> str | None:
    """why the backend extension is no build of the head's source tree, or None"""
    built = identity["built"]
    if built.endswith("-dirty"):
        return f"the {BACKEND} extension was built from a dirty tree ({built})"
    sha = built.split("-")[0]
    if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode != 0:
        return f"the {BACKEND} extension's commit {sha} is not in this repository"
    if subprocess.run(["git", "diff", "--quiet", sha, "HEAD", "--", "src"], capture_output=True).returncode != 0:
        return f"the {BACKEND} extension is built from {sha}, whose src/ differs from HEAD {identity['head']}"
    return None


def free_gigabytes(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------
# session selection
# ------------------------------------------------------------------------------------------------
def _marks():
    import pytest

    return (
        pytest.mark.skipif(BACKEND != "gpu", reason="a gpu device-scope gate; the session backend is cpu"),
        pytest.mark.skipif(BACKEND != "cpu", reason="a cpu gate; the session backend is gpu"),
        pytest.mark.skipif(PLACEMENT != "local", reason="runs on the local launcher or in one process alone; the session placement is scheduler"),
    )


gpu_only, cpu_only, local_only = _marks()


def environment(inject: str | None = None, opt_in: bool = False) -> dict[str, str]:
    """this process's environment without an injected fault or the experimental opt-in, then with
    the ones the gate names"""
    env = dict(os.environ)
    env.pop("SIMBI_FABRIC_INJECT", None)
    env.pop("SIMBI_LAUNCH_EXPERIMENTAL", None)
    if inject is not None:
        env["SIMBI_FABRIC_INJECT"] = inject
    if opt_in:
        env["SIMBI_LAUNCH_EXPERIMENTAL"] = "1"
    return env
