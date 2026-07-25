# =============================================================================
# test_validate_problem.py
#
# dry-run validation must build the production payload and call the rust
# preflight entry point without entering the simulation runner.
# =============================================================================

from types import SimpleNamespace

import pytest

import simbi.expression as expr
import simbi.libs.cpu_ext as backend
from simbi.simulation import runner
from simbi.simulation.runner import to_execution_dict
from simbi_configs.examples.newtonian.sod import SodProblem


def test_validate_problem_calls_rust_preflight(monkeypatch, capsys) -> None:
    problem = SimpleNamespace(passive_scalar=lambda: None)
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


def test_validate_problem_writes_no_output_directory(tmp_path) -> None:
    output = tmp_path / "must-not-exist"
    runner.validate_problem(SodProblem(data_directory=str(output)))
    assert not output.exists()


def test_rust_preflight_rejects_legacy_bare_source_payload() -> None:
    payload = to_execution_dict(SodProblem())
    graph = expr.ExprGraph()
    payload["hydro_source_expressions"] = graph.compile(
        [expr.constant(1.0, graph)]
    ).serialize()

    with pytest.raises(ValueError, match="missing field `kind`"):
        backend.validate_simulation(sim_info=payload)


def test_rust_preflight_rejects_legacy_bare_boundary_payload() -> None:
    payload = to_execution_dict(SodProblem())
    graph = expr.ExprGraph()
    payload["boundary_conditions"][0] = "dynamic"
    payload["bx1_inner_expressions"] = graph.compile(
        [expr.constant(1.0, graph)]
    ).serialize()

    with pytest.raises(ValueError, match="missing field `kind`"):
        backend.validate_simulation(sim_info=payload)


def test_scalar_sample_rejects_nonfinite_magnetic_field() -> None:
    with pytest.raises(ValueError, match="Bz.*finite"):
        runner._check_first_scalar("Probe", "Bz", iter([float("nan")]))


def test_scalar_sample_rejects_empty_passive_scalar() -> None:
    with pytest.raises(ValueError, match="passive scalar.*yielded nothing"):
        runner._check_first_scalar("Probe", "passive scalar", iter(()))
