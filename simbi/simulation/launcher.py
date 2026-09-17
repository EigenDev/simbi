# =============================================================================
# simbi/simulation/launcher.py
#
# the launcher for `simbi launch`. the layout file names the worker count,
# the partition, the checkpoint directory, and the execution mode. in local
# mode the launcher derives one tile per worker and the per-axis cuts, writes
# a fresh session credential, and spawns the workers as processes running the
# same command line with the worker-role flags; it waits for every worker,
# the first failure ends the others after a grace period, and every started
# worker is terminated and reaped and the rendezvous file removed on any
# exit, including an exception during spawning and a signal. in scheduler
# mode the scheduler has already started one process per worker; each takes
# its worker id from the scheduler's environment and runs the worker role
# directly, with the coordinator writing its advertised address and the
# session credential into a restricted rendezvous file on the shared output
# directory that the others poll.
#
# usage:
#   layout = Layout.from_file("cluster.toml", resolution)
#   code = spawn_workers(layout, sys.argv[1:], data_directory)      # local mode
#   worker, workers = scheduler_identity(layout)                     # scheduler mode
# =============================================================================
from __future__ import annotations

import math
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


def even_cuts(n_cells: Sequence[int], shape: Sequence[int]) -> list[list[int]]:
    """the interior cut indices that split each axis into `shape[ax]` equal tiles."""
    cuts: list[list[int]] = []
    for n, count in zip(n_cells, shape):
        if count < 1 or n % count != 0:
            raise ValueError(f"{count} tiles do not divide {n} cells evenly")
        width = n // count
        cuts.append([width * i for i in range(1, count)])
    return cuts


# set in the environment of every process the local launcher starts
LAUNCHED_MARK = "SIMBI_LAUNCHED_WORKER"
MAX_LOCAL_WORKERS = 64

SCHEDULER_RANK = ("SLURM_PROCID", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "PMIX_RANK")
SCHEDULER_SIZE = ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE")


def advertised_host(bind: str, advertise: str | None) -> str:
    """the host peers connect to: the advertised name when given, the bind address when it
    names one interface, and this host's name when the bind address is unspecified."""
    if advertise:
        return advertise
    if bind in ("0.0.0.0", "::", ""):
        return socket.gethostname()
    return bind


@dataclass
class Layout:
    workers: int
    cuts: list[list[int]]
    checkpoint_directory: str | None = None
    staging_mb: int = 4
    owner: list[int] = field(default_factory=list)
    mode: str = "local"
    bind: str = "127.0.0.1"
    advertise: str | None = None
    # keep the rendezvous record after the session as `<file>.kept`, for inspection
    keep_rendezvous: bool = False

    @property
    def advertised(self) -> str:
        return advertised_host(self.bind, self.advertise)

    @property
    def tiles(self) -> int:
        return math.prod(len(c) + 1 for c in self.cuts)

    @property
    def staging_cells(self) -> int:
        return self.staging_mb * (1 << 20) // 8

    @classmethod
    def from_file(cls, path: str, resolution: Sequence[int]) -> "Layout":
        with open(path, "rb") as f:
            doc = tomllib.load(f)
        execution = doc.get("execution", {})
        partition = doc.get("partition", {})
        checkpoint = doc.get("checkpoint", {})
        workers = int(execution.get("workers", execution.get("nodes", 1)))
        if "shape" in partition and "cuts" in partition:
            raise ValueError("the layout names both a shape and cuts; give one")
        if "cuts" in partition:
            cuts = [[int(c) for c in axis] for axis in partition["cuts"]]
        elif "shape" in partition:
            cuts = even_cuts(list(resolution), [int(s) for s in partition["shape"]])
        else:
            raise ValueError("the layout names no partition; give [partition] shape or cuts")
        if len(cuts) != len(resolution):
            raise ValueError(f"the partition has {len(cuts)} axes; the problem has {len(resolution)}")
        mode = str(execution.get("mode", "local"))
        if mode not in ("local", "scheduler"):
            raise ValueError(f"execution mode {mode!r}; give local or scheduler")
        layout = cls(
            workers=workers,
            cuts=cuts,
            checkpoint_directory=checkpoint.get("directory"),
            staging_mb=int(checkpoint.get("staging_mb", 4)),
            mode=mode,
            bind=str(execution.get("bind", "127.0.0.1")),
            advertise=execution.get("advertise"),
            keep_rendezvous=bool(checkpoint.get("keep_rendezvous", False)),
        )
        if layout.tiles != workers:
            raise ValueError(
                f"the partition has {layout.tiles} tiles and the layout {workers} workers; "
                "the first release places one tile per worker"
            )
        layout.owner = list(range(layout.tiles))
        return layout


def worker_flags(layout: Layout, worker: int, rendezvous: Path, credential: int, output: Path) -> list[str]:
    """the worker-role flags plus the output directory, which every worker and the
    rendezvous file share."""
    return [
        "--worker",
        str(worker),
        "--workers",
        str(layout.workers),
        "--rendezvous",
        str(rendezvous),
        "--credential",
        str(credential),
        "--bind",
        layout.bind,
        "--advertise",
        layout.advertised,
        "--owner",
        ",".join(str(o) for o in layout.owner),
        "--cuts",
        ";".join(",".join(str(c) for c in axis) for axis in layout.cuts),
        "--staging-cells",
        str(layout.staging_cells),
        "--data-directory",
        str(output),
    ]


class Session:
    """the started workers of one launch and their cleanup. `finish` terminates and reaps
    every worker still running and removes the rendezvous file; it runs on the normal path,
    on an exception, and on SIGINT or SIGTERM to the launcher."""

    def __init__(self, rendezvous: Path, keep: bool = False):
        self.rendezvous = rendezvous
        self.keep = keep
        self.procs: list[subprocess.Popen] = []
        self.finished = False

    def finish(self, grace: float = 2.0) -> None:
        if self.finished:
            return
        self.finished = True
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.monotonic() + grace
        for p in self.procs:
            while p.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            if p.poll() is None:
                p.kill()
            p.wait()
        retire_rendezvous(self.rendezvous, self.keep)


def retire_rendezvous(rendezvous: Path, keep: bool) -> None:
    """remove the rendezvous file, or keep it under `.kept` for inspection."""
    if not rendezvous.exists():
        return
    if keep:
        kept = rendezvous.with_name(rendezvous.name + ".kept")
        if kept.exists():
            kept.unlink()
        rendezvous.rename(kept)
    else:
        rendezvous.unlink()


def spawn_workers(layout: Layout, argv: Sequence[str], data_directory: str, grace: float = 5.0) -> int:
    """spawn one process per worker running `argv` plus the worker flags; the exit code is
    zero when every worker exited zero. the first nonzero exit ends the rest after `grace`
    seconds. the layout's checkpoint directory, when named, is the output directory of
    every worker and of the rendezvous file."""
    # a worker process never launches: the launcher marks its children, and a marked process
    # reaching this point is a role-dispatch fault that would otherwise multiply without bound
    if os.environ.get(LAUNCHED_MARK):
        raise RuntimeError("a launched worker reached the launcher role; refusing to spawn")
    if not 1 <= layout.workers <= MAX_LOCAL_WORKERS:
        raise ValueError(f"{layout.workers} local workers; the local launcher starts at most {MAX_LOCAL_WORKERS}")
    output = Path(layout.checkpoint_directory or data_directory)
    output.mkdir(parents=True, exist_ok=True)
    rendezvous = output / f".rendezvous-{os.getpid()}"
    if rendezvous.exists():
        rendezvous.unlink()
    credential = secrets.randbits(63)
    session = Session(rendezvous, layout.keep_rendezvous)
    previous = {}

    def on_signal(signum, _frame):
        session.finish()
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, on_signal)
    try:
        for worker in range(layout.workers):
            cmd = [sys.executable, "-m", "simbi.cli", *argv, *worker_flags(layout, worker, rendezvous, credential, output)]
            proc = subprocess.Popen(cmd, env=dict(os.environ, **{LAUNCHED_MARK: "1"}))
            session.procs.append(proc)
            print(f"launch: worker {worker} pid {proc.pid}", flush=True)
        code = 0
        failed_at: float | None = None
        while True:
            codes = [p.poll() for p in session.procs]
            if all(c is not None for c in codes):
                code = next((c for c in codes if c), 0)
                break
            if any(c not in (None, 0) for c in codes):
                failed_at = failed_at or time.monotonic()
                if time.monotonic() - failed_at > grace:
                    code = next(c for c in codes if c)
                    break
            time.sleep(0.05)
    finally:
        session.finish()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    for worker, p in enumerate(session.procs):
        print(f"launch: worker {worker} exit {p.returncode}", flush=True)
    return code


def scheduler_identity(layout: Layout) -> tuple[int, int]:
    """this process's worker id and the worker count under a scheduler, from the first rank
    and size variables the scheduler set; the count must agree with the layout."""
    rank = next((os.environ[k] for k in SCHEDULER_RANK if k in os.environ), None)
    if rank is None:
        raise RuntimeError(
            "scheduler mode needs the worker id from the scheduler; none of "
            + ", ".join(SCHEDULER_RANK)
            + " is set (or pass --worker)"
        )
    size = next((os.environ[k] for k in SCHEDULER_SIZE if k in os.environ), None)
    workers = int(size) if size is not None else layout.workers
    if workers != layout.workers:
        raise RuntimeError(f"the scheduler started {workers} tasks; the layout names {layout.workers} workers")
    worker = int(rank)
    if not 0 <= worker < workers:
        raise RuntimeError(f"scheduler rank {worker} is outside {workers} workers")
    return worker, workers


def scheduler_rendezvous(layout: Layout, data_directory: str) -> Path:
    """the rendezvous file every scheduler-started worker of a session shares: a fixed name
    under the output directory, which the workers see through the shared filesystem."""
    output = Path(layout.checkpoint_directory or data_directory)
    output.mkdir(parents=True, exist_ok=True)
    return output / ".rendezvous"
