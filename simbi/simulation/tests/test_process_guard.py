# =============================================================================
# test_process_guard.py
#
# the process guard's startup budget. the attribution rule is checked on
# synthetic process tables (the exec'd `uname`, the pre-exec fork, one child
# per parent, leaves only, the window); live arms check that budgeted startup
# children keep a run under the bound while every process is still counted,
# and that excess processes, a second child, and a child lingering past the
# window trip the guard.
# =============================================================================
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from simbi.simulation.tests.process_guard import ProcessGuard, budgeted_startup_children, descendants

PY = "/usr/bin/Python -m simbi.cli launch"


def _rows(*rows: tuple[int, int, str]) -> list[tuple[int, int, str, str]]:
    return [(pid, ppid, "S", command) for pid, ppid, command in rows]


def test_the_exec_and_the_pre_exec_probe_are_budgeted_while_the_parent_is_young() -> None:
    rows = _rows((10, 1, PY), (11, 10, "uname -p"), (20, 1, PY), (21, 20, "(Python)"))
    seen = {10: 0.0, 11: 0.0, 20: 0.0, 21: 0.0}
    assert budgeted_startup_children(rows, seen, now=5.0, startup=10.0) == {11, 21}
    assert budgeted_startup_children(rows, seen, now=10.0, startup=10.0) == set(), "the window is not bounded"
    assert budgeted_startup_children(rows, seen, now=5.0, startup=0.0) == set(), "a zero window budgets a child"


def test_only_attributed_leaf_children_are_budgeted_one_per_parent() -> None:
    seen = {pid: 0.0 for pid in range(100)}
    two = _rows((10, 1, PY), (11, 10, "uname -p"), (12, 10, "uname -p"))
    assert len(budgeted_startup_children(two, seen, 1.0, 10.0)) == 1, "a second child of one parent was budgeted"
    other = _rows((10, 1, PY), (11, 10, "/bin/sleep 5"), (12, 10, "(sleep)"))
    assert budgeted_startup_children(other, seen, 1.0, 10.0) == set(), "an unattributed child was budgeted"
    branching = _rows((10, 1, PY), (11, 10, "(Python)"), (12, 11, "uname -p"))
    assert budgeted_startup_children(branching, seen, 1.0, 10.0) == {12}, "a child with children of its own was budgeted"


def _uname(tmp: Path) -> str:
    """a long-lived stand-in for the platform probe: `sleep` under the file name `uname`"""
    path = tmp / "uname"
    path.symlink_to("/bin/sleep")
    return str(path)


def _workers(n: int, children: list[str], hold: float) -> list[subprocess.Popen]:
    """n python processes, each starting the given child commands and holding for `hold` seconds"""
    script = (
        "import subprocess, sys, time\n"
        f"kids = [subprocess.Popen(c.split()) for c in {children!r}]\n"
        f"time.sleep({hold})\n"
        "for k in kids: k.kill(); k.wait()\n"
    )
    return [subprocess.Popen([sys.executable, "-c", script]) for _ in range(n)]


def _finish(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.wait(timeout=30)
    deadline = time.monotonic() + 10
    while descendants(os.getpid()) and time.monotonic() < deadline:
        time.sleep(0.1)


def test_budgeted_startup_children_keep_a_launch_under_the_bound_and_are_counted(tmp_path: Path) -> None:
    uname = _uname(tmp_path)
    with ProcessGuard(limit=3, startup=10.0) as guard:
        procs = _workers(3, [f"{uname} 30"], hold=1.5)
        _finish(procs)
    assert guard.peak == 6, f"the run never held three workers with their probes (peak {guard.peak}); the arm is vacuous"
    assert not guard.tripped, "\n".join(guard.report)


def test_excess_unattributed_processes_trip_the_guard() -> None:
    with ProcessGuard(limit=3, startup=10.0) as guard:
        procs = [subprocess.Popen(["/bin/sleep", "1.5"]) for _ in range(4)]
        _finish(procs)
    assert guard.tripped, f"four plain processes over a limit of three passed (peak {guard.peak})"


def test_a_second_startup_child_of_one_worker_trips_the_guard(tmp_path: Path) -> None:
    uname = _uname(tmp_path)
    # one worker over a limit of one: the budget covers exactly one probe child, the second counts
    with ProcessGuard(limit=1, startup=10.0) as guard:
        procs = _workers(1, [f"{uname} 30", f"{uname} 30"], hold=1.5)
        _finish(procs)
    assert guard.tripped, f"a worker's second probe child was budgeted (peak {guard.peak})"


def test_a_startup_child_lingering_past_the_window_trips_the_guard(tmp_path: Path) -> None:
    uname = _uname(tmp_path)
    with ProcessGuard(limit=3, startup=0.5) as guard:
        procs = _workers(3, [f"{uname} 30"], hold=3.0)
        _finish(procs)
    assert guard.tripped, f"probe children outliving the startup window were budgeted (peak {guard.peak})"
    assert any("uname" in line and "budgeted" not in line for line in guard.report), "\n".join(guard.report)
