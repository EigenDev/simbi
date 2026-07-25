# =============================================================================
# test_validate_problem.py
#
# dry-run validation must build the production payload and call the rust
# preflight entry point without entering the simulation runner.
# =============================================================================

from types import SimpleNamespace

from simbi.simulation import runner


def test_validate_problem_calls_rust_preflight(monkeypatch, capsys) -> None:
    problem = SimpleNamespace()
    calls: list[dict[str, object]] = []
    backend = SimpleNamespace(
        validate_simulation=lambda **kwargs: calls.append(kwargs)
    )

    monkeypatch.setattr(runner, "_validate_generator", lambda _problem: None)
    monkeypatch.setattr(runner, "to_execution_dict", lambda _problem: {"name": "probe"})
    monkeypatch.setattr(runner, "_get_iterators", lambda _problem: (iter([(1.0, 0.0, 1.0)]), []))
    monkeypatch.setattr(runner, "_check_first_tuple", lambda _problem, iterator: iterator)
    monkeypatch.setattr(runner, "_load_backend", lambda _mode: backend)

    runner.validate_problem(problem)

    assert calls == [{"sim_info": {"name": "probe"}}]
    assert "validation passed" in capsys.readouterr().out
