# =============================================================================
# test_launch_roles.py
#
# the role dispatch of `simbi launch`, checked without starting a process. the
# local launcher forwards its whole command line to every worker, the layout
# flag included, so a worker's command line must reach the worker role and
# never the launcher role; a process the launcher started refuses to spawn
# even if dispatch were wrong, since a launcher reached from a worker
# multiplies without bound.
# =============================================================================
from pathlib import Path

import pytest

from simbi.cli.simbi_parser import SimbiParser
from simbi.cli.commands.launch import executor
from simbi.simulation import launcher

SCRIPT = "simbi_configs/examples/newtonian/kh.py"


def _layout(tmp_path: Path, mode: str = "local") -> Path:
    path = tmp_path / "two.toml"
    path.write_text(f'[execution]\nmode = "{mode}"\nworkers = 2\n\n[partition]\nshape = [2, 1]\n')
    return path


def _dispatch(monkeypatch, argv: list[str]) -> dict:
    """run the launch executor on `argv` with both roles replaced by recorders."""
    seen: dict = {}
    monkeypatch.setattr(executor, "spawn_workers", lambda *a, **k: seen.setdefault("launcher", a) and 0 or 0)
    monkeypatch.setattr(executor, "_run_worker_role", lambda *a, **k: seen.setdefault("worker", a))
    parser = SimbiParser()
    args, remaining = parser.parse_known_args(argv)
    args.func(args, remaining)
    return seen


def test_the_launcher_command_line_reaches_the_launcher_role(tmp_path, monkeypatch) -> None:
    seen = _dispatch(monkeypatch, ["launch", SCRIPT, "--layout", str(_layout(tmp_path)), "--resolution", "32,32"])
    assert "launcher" in seen and "worker" not in seen


def test_a_spawned_worker_command_line_reaches_the_worker_role(tmp_path, monkeypatch) -> None:
    """exactly the command line the local launcher builds for worker 1: the launcher's argv,
    layout flag included, plus the worker-role flags."""
    layout_path = _layout(tmp_path)
    layout = launcher.Layout.from_file(str(layout_path), (32, 32))
    argv = ["launch", SCRIPT, "--layout", str(layout_path), "--resolution", "32,32"]
    argv += launcher.worker_flags(layout, 1, tmp_path / ".rendezvous", 12345, tmp_path)
    seen = _dispatch(monkeypatch, argv)
    assert "worker" in seen, "a spawned worker did not reach the worker role"
    assert "launcher" not in seen, "a spawned worker reached the launcher role"
    _args, _problem, worker, workers, owner, cuts, *_ = seen["worker"]
    assert (worker, workers, owner, cuts) == (1, 2, [0, 1], [[16], []])


def test_a_scheduler_started_process_reaches_the_worker_role(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SLURM_PROCID", "1")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    argv = ["launch", SCRIPT, "--layout", str(_layout(tmp_path, "scheduler")), "--resolution", "32,32", "--data-directory", str(tmp_path)]
    seen = _dispatch(monkeypatch, argv)
    assert "worker" in seen and "launcher" not in seen
    assert seen["worker"][2:4] == (1, 2)


def test_a_launched_process_refuses_to_spawn(tmp_path, monkeypatch) -> None:
    started = []
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **k: started.append(a))
    monkeypatch.setenv(launcher.LAUNCHED_MARK, "1")
    layout = launcher.Layout(workers=2, cuts=[[16], []], owner=[0, 1])
    with pytest.raises(RuntimeError, match="refusing to spawn"):
        launcher.spawn_workers(layout, ["launch", "x.py"], str(tmp_path))
    assert not started


def test_the_launcher_marks_its_children_and_bounds_their_number(tmp_path, monkeypatch) -> None:
    class Done:
        pid, returncode = 1, 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    envs = []
    monkeypatch.delenv(launcher.LAUNCHED_MARK, raising=False)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda cmd, env=None: envs.append(env) or Done())
    layout = launcher.Layout(workers=2, cuts=[[16], []], owner=[0, 1])
    assert launcher.spawn_workers(layout, ["launch", "x.py"], str(tmp_path)) == 0
    assert len(envs) == 2 and all(e[launcher.LAUNCHED_MARK] == "1" for e in envs)
    too_many = launcher.Layout(workers=launcher.MAX_LOCAL_WORKERS + 1, cuts=[[], []], owner=[])
    with pytest.raises(ValueError, match="at most"):
        launcher.spawn_workers(too_many, ["launch", "x.py"], str(tmp_path))


def test_an_unresolvable_advertised_host_fails_by_name() -> None:
    with pytest.raises(RuntimeError, match="does not resolve"):
        launcher.require_resolvable("no-such-host.invalid")
    launcher.require_resolvable("127.0.0.1")


def test_a_scheduler_layout_refuses_one_advertise_host_for_every_node(tmp_path, monkeypatch) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('[execution]\nmode = "scheduler"\nworkers = 2\nadvertise = "node-a"\n\n[partition]\nshape = [2, 1]\n')
    with pytest.raises(ValueError, match="one advertise host"):
        launcher.Layout.from_file(str(path), (32, 32))
    monkeypatch.setenv(launcher.ADVERTISE_ENV, "10.1.2.3")
    assert launcher.advertised_host("0.0.0.0", None) == "10.1.2.3"
