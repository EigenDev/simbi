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
# every descendant is counted. a startup window budgets one attributed child
# per process on top of the limit: importing h5py calls platform.processor(),
# which on macOS runs `uname -p` in a child, so each launched worker briefly
# holds that child (seen as the exec'd `uname`, as `(uname)` while its exec
# completes, or as the worker's own fork before the exec, `(<image>)`). the
# child is budgeted only while it is a
# leaf and its parent is younger than the window; a second child, a child
# with children of its own, or one outliving the window counts against the
# limit.
#
# `run_reaped` is `subprocess.run` with the command as the leader of its own
# process group: on a timeout or any error every member of the group (a
# launcher's workers included) is killed and reaped before the error is
# raised, and a member that outlives a finished command is killed and named.
#
# usage:
#   with ProcessGuard(limit=12, startup=10.0) as guard:
#       subprocess.run([...])
#   assert not guard.tripped
#   done = run_reaped([...], capture_output=True, text=True, timeout=600)
# =============================================================================
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time


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


def _image(command: str) -> str:
    """the executable's file name of a process-table command"""
    return os.path.basename(command.split(" ", 1)[0]) if command else ""


def budgeted_startup_children(rows: list[tuple[int, int, str, str]], first_seen: dict[int, float], now: float, startup: float) -> set[int]:
    """the pids of `rows` budgeted as attributed startup children: per parent at most one leaf
    child that is the platform probe (`uname`, or the parent's fork before its exec), while the
    parent was first seen less than `startup` seconds before `now`."""
    by_pid = {row[0]: row for row in rows}
    parents = {row[1] for row in rows}
    budgeted: dict[int, int] = {}
    for pid, ppid, _, command in sorted(rows):
        parent = by_pid.get(ppid)
        if parent is None or pid in parents or ppid in budgeted:
            continue
        probe = _image(command) == "uname" or command in ("(uname)", f"({_image(parent[3])})")
        young = now - first_seen.get(ppid, now) < startup
        if probe and young:
            budgeted[ppid] = pid
    return set(budgeted.values())


def descendants(root: int) -> list[int]:
    """every running descendant of `root`, from one process-table snapshot."""
    return [row[0] for row in descendant_rows(root)]


class ProcessGuard:
    def __init__(self, limit: int, root: int | None = None, period: float = 0.1, startup: float = 0.0):
        self.limit = limit
        # seconds after the guard first sees a process during which its attributed startup child
        # is budgeted above the limit
        self.startup = startup
        self._first_seen: dict[int, float] = {}
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
            now = time.monotonic()
            self._first_seen = {row[0]: self._first_seen.get(row[0], now) for row in rows}
            budgeted = budgeted_startup_children(rows, self._first_seen, now, self.startup)
            self.peak = max(self.peak, len(rows))
            if len(rows) - len(budgeted) > self.limit:
                self.tripped = True
                # the census of the snapshot that crossed the bound, every counted process with its
                # parent, before the kill below changes the table
                self.report = [
                    f"{pid} {ppid} {stat} {'(budgeted startup child) ' if pid in budgeted else ''}{command}"[:200]
                    for pid, ppid, stat, command in rows
                ]
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


_RUN = subprocess.run


def group_members(pgid: int) -> list[int]:
    """the live processes of process group `pgid`"""
    table = _RUN(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True).stdout
    return [int(pid) for pid, group, stat in (row.split()[:3] for row in table.splitlines() if row.strip()) if int(group) == pgid and not stat.startswith("Z")]


def kill_group(pgid: int) -> list[int]:
    """kill every live member of process group `pgid` and return their pids; members exiting
    meanwhile are skipped"""
    members = group_members(pgid)
    for pid in members:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    deadline = time.monotonic() + 5.0
    while group_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return members


class LingeringProcesses(RuntimeError):
    """members of a finished command's process group were still running; they were killed"""


def run_reaped(args, *, input=None, capture_output=False, timeout=None, check=False, **kwargs) -> subprocess.CompletedProcess:
    """`subprocess.run` with the command leading its own process group: a timeout or any error
    kills and reaps every member of the group before it propagates, and a member outliving the
    finished command is killed and reported as `LingeringProcesses`"""
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    kwargs["start_new_session"] = True
    proc = subprocess.Popen(args, **kwargs)
    end = None if timeout is None else time.monotonic() + timeout
    lingering: list[int] = []
    try:
        # read in short slices: the leader may exit while a group member it left behind holds its
        # output pipes open, which a single blocking read would wait on until the timeout
        while True:
            try:
                out, err = proc.communicate(input, timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                input = None
                if end is not None and time.monotonic() > end:
                    kill_group(proc.pid)
                    out, err = proc.communicate()
                    raise subprocess.TimeoutExpired(proc.args, timeout, output=out, stderr=err) from None
                if proc.poll() is not None:
                    lingering = kill_group(proc.pid)
    except BaseException:
        kill_group(proc.pid)
        proc.wait()
        raise
    lingering += kill_group(proc.pid)
    if lingering:
        raise LingeringProcesses(f"{proc.args!r} exited {proc.returncode} leaving processes {lingering} in its group; they were killed")
    done = subprocess.CompletedProcess(proc.args, proc.returncode, out, err)
    if check:
        done.check_returncode()
    return done
