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


def descendants(root: int) -> list[int]:
    """every running descendant of `root`, from one process-table snapshot."""
    snapshot = subprocess.Popen(["ps", "-axo", "pid=,ppid=,stat="], stdout=subprocess.PIPE, text=True)
    table, _ = snapshot.communicate()
    children: dict[int, list[int]] = {}
    for line in table.splitlines():
        parts = line.split()
        # the snapshot process lists itself; it is this function's own child, not the root's work
        # and an ended process awaiting its parent's wait holds no resources
        if len(parts) == 3 and int(parts[0]) != snapshot.pid and not parts[2].startswith("Z"):
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found: list[int] = []
    stack = [root]
    while stack:
        for child in children.get(stack.pop(), []):
            found.append(child)
            stack.append(child)
    return found


class ProcessGuard:
    def __init__(self, limit: int, root: int | None = None, period: float = 0.1):
        self.limit = limit
        self.root = root if root is not None else os.getpid()
        self.period = period
        self.peak = 0
        self.tripped = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self._stop.wait(self.period):
            found = descendants(self.root)
            self.peak = max(self.peak, len(found))
            if len(found) > self.limit:
                self.tripped = True
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
