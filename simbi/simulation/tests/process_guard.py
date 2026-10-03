# =============================================================================
# process_guard.py
#
# a bound on the processes a test session starts. the guard follows the
# descendants of one root process, by parent links, and kills those
# descendants alone if their number passes a limit; every other process on the
# machine, including other simulation runs, is outside its reach. a test that
# re-executes the command line uses it so a role-dispatch fault ends at the
# limit and fails the test.
#
# usage:
#   with ProcessGuard(limit=12) as guard:
#       subprocess.run([...])
#   assert not guard.tripped
# =============================================================================
from __future__ import annotations

import os
import signal
import subprocess
import threading


def descendant_rows(root: int) -> list[tuple[int, int, str, str]]:
    """every running descendant of `root` as (pid, ppid, stat, command), from one process-table
    snapshot."""
    snapshot = subprocess.Popen(["ps", "-axo", "pid=,ppid=,stat=,command="], stdout=subprocess.PIPE, text=True)
    table, _ = snapshot.communicate()
    rows: dict[int, list[tuple[int, int, str, str]]] = {}
    for line in table.splitlines():
        parts = line.split(None, 3)
        # the snapshot process lists itself; it is this function's own child, not the root's work
        # and an ended process awaiting its parent's wait holds no resources
        if len(parts) >= 3 and int(parts[0]) != snapshot.pid and not parts[2].startswith("Z"):
            pid, ppid = int(parts[0]), int(parts[1])
            rows.setdefault(ppid, []).append((pid, ppid, parts[2], parts[3] if len(parts) > 3 else ""))
    found: list[tuple[int, int, str, str]] = []
    stack = [root]
    while stack:
        for row in rows.get(stack.pop(), []):
            found.append(row)
            stack.append(row[0])
    return found


def descendants(root: int) -> list[int]:
    """every running descendant of `root`, from one process-table snapshot."""
    return [row[0] for row in descendant_rows(root)]


class ProcessGuard:
    def __init__(self, limit: int, root: int | None = None, period: float = 0.1):
        self.limit = limit
        self.root = root if root is not None else os.getpid()
        self.period = period
        self.peak = 0
        self.tripped = False
        # the processes of the snapshot that passed the bound, one `pid ppid stat command` line each
        self.report: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self._stop.wait(self.period):
            rows = descendant_rows(self.root)
            self.peak = max(self.peak, len(rows))
            if len(rows) > self.limit:
                self.tripped = True
                # the census of the snapshot that crossed the bound, every counted process with its
                # parent, before the kill below changes the table
                self.report = [f"{pid} {ppid} {stat} {command}"[:200] for pid, ppid, stat, command in rows]
                found = [row[0] for row in rows]
                for pid in found:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def __enter__(self) -> "ProcessGuard":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
