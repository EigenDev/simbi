# =============================================================================
# simbi/simulation/launcher.py
#
# the local process launcher for `simbi launch`: the layout file names the
# worker count, the partition, and the checkpoint directory, the launcher
# derives one tile per worker and the per-axis cuts, writes a fresh session
# credential, and spawns the workers as processes running the same command
# line with the worker-role flags. it waits for every worker; the first
# failure ends the others after a grace period and the launcher exits nonzero.
# every started worker is terminated and reaped and the rendezvous file
# removed on any exit, including an exception during spawning and a signal
# to the launcher. the coordinator worker announces its address in the
# rendezvous file under the output directory.
#
# usage:
#   layout = Layout.from_file("cluster.toml", resolution)
#   code = spawn_workers(layout, sys.argv[1:], data_directory)
# =============================================================================
from __future__ import annotations

import math
import os
import secrets
import signal
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


@dataclass
class Layout:
    workers: int
    cuts: list[list[int]]
    checkpoint_directory: str | None = None
    staging_mb: int = 4
    owner: list[int] = field(default_factory=list)

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
        layout = cls(
            workers=workers,
            cuts=cuts,
            checkpoint_directory=checkpoint.get("directory"),
            staging_mb=int(checkpoint.get("staging_mb", 4)),
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

    def __init__(self, rendezvous: Path):
        self.rendezvous = rendezvous
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
        if self.rendezvous.exists():
            self.rendezvous.unlink()


def spawn_workers(layout: Layout, argv: Sequence[str], data_directory: str, grace: float = 5.0) -> int:
    """spawn one process per worker running `argv` plus the worker flags; the exit code is
    zero when every worker exited zero. the first nonzero exit ends the rest after `grace`
    seconds. the layout's checkpoint directory, when named, is the output directory of
    every worker and of the rendezvous file."""
    output = Path(layout.checkpoint_directory or data_directory)
    output.mkdir(parents=True, exist_ok=True)
    rendezvous = output / f".rendezvous-{os.getpid()}"
    if rendezvous.exists():
        rendezvous.unlink()
    credential = secrets.randbits(63)
    session = Session(rendezvous)
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
            proc = subprocess.Popen(cmd)
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
