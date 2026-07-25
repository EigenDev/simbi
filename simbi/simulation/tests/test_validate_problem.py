# =============================================================================
# test_validate_problem.py
#
# dry-run validation must build the production payload and call the rust
# preflight entry point without entering the simulation runner.
# =============================================================================

import argparse
from types import SimpleNamespace
from typing import Annotated

import pytest

import simbi.expression as expr
import simbi.libs.cpu_ext as backend
from simbi import ProblemParam
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


def test_rust_preflight_rejects_bare_source_payload() -> None:
    payload = to_execution_dict(SodProblem())
    payload["source_expressions"] = [{"outputs": [0], "nodes": []}]

    with pytest.raises(ValueError, match="missing field `kind`"):
        backend.validate_simulation(sim_info=payload)


def test_rust_preflight_rejects_bare_boundary_payload() -> None:
    payload = to_execution_dict(SodProblem())
    payload["boundary_conditions"][0] = "dynamic"
    payload["bx1_inner_expressions"] = {"outputs": [0], "nodes": []}

    with pytest.raises(ValueError, match="missing field `kind`"):
        backend.validate_simulation(sim_info=payload)


def test_scalar_sample_rejects_nonfinite_magnetic_field() -> None:
    with pytest.raises(ValueError, match="Bz.*finite"):
        runner._check_first_scalar("Probe", "Bz", iter([float("nan")]))


def test_scalar_sample_rejects_empty_passive_scalar() -> None:
    with pytest.raises(ValueError, match="passive scalar.*yielded nothing"):
        runner._check_first_scalar("Probe", "passive scalar", iter(()))


def test_execution_dict_preserves_source_collection_order() -> None:
    class MultipleSources(SodProblem):
        @property
        def source_expressions(self):
            return [
                {
                    "kind": "raw",
                    "dim": 1,
                    "target": target,
                    "outputs": [0],
                    "nodes": [{"op": "CONSTANT", "value": value}],
                }
                for target, value in (("den", 1.0), ("mom", 2.0), ("nrg", 3.0))
            ]

    payload = to_execution_dict(MultipleSources())
    assert [source["target"] for source in payload["source_expressions"]] == [
        "den",
        "mom",
        "nrg",
    ]


def test_compiled_expression_exposes_only_typed_serializers() -> None:
    graph = expr.ExprGraph()
    compiled = graph.compile([expr.constant(1.0, graph)])
    assert not hasattr(compiled, "serialize")


def test_execution_dict_rejects_bare_boundary_serialize() -> None:
    class BareBoundary(SodProblem):
        @property
        def bx1_inner_expressions(self):
            return {"outputs": [0], "nodes": []}

    with pytest.raises(ValueError, match="bx1_inner_expressions.*missing `kind`"):
        to_execution_dict(BareBoundary())


@pytest.mark.parametrize(
    ("kind", "count", "expected"),
    [
        ("force", 1, 2),
        ("cooling", 2, 1),
        ("relax", 2, 3),
        ("sponge", 4, 5),
        ("inject", 3, 4),
    ],
)
def test_execution_dict_rejects_source_arity(
    kind: str, count: int, expected: int
) -> None:
    payload = {
        "source_expressions": [{
            "kind": kind,
            "dim": 2,
            "outputs": list(range(count)),
        }],
        "isothermal": False,
    }
    with pytest.raises(ValueError, match=rf"expected {expected}"):
        runner._validate_expression_payloads(payload)


def test_execution_dict_rejects_boundary_arity() -> None:
    payload = {
        "bx1_inner_expressions": {
            "kind": "dirichlet",
            "dim": 2,
            "outputs": [0, 1, 2],
        },
        "isothermal": False,
        "is_mhd": False,
    }
    with pytest.raises(ValueError, match="expected 4 primitive"):
        runner._validate_expression_payloads(payload)


def test_percent_in_cli_description_formats_cleanly() -> None:
    class PercentDescription(SodProblem):
        pad: Annotated[
            float,
            ProblemParam(0.01, cli=True, description="add a 1% pad"),
        ]

    parser = argparse.ArgumentParser()
    PercentDescription.setup_cli(parser)
    assert "1% pad" in parser.format_help()


@pytest.mark.parametrize("target", [None, "temperature", ""])
def test_raw_source_requires_a_conserved_target(target) -> None:
    payload = {
        "source_expressions": [{
            "kind": "raw",
            "dim": 2,
            "target": target,
            "outputs": [0],
        }],
    }
    with pytest.raises(ValueError, match="raw.*target"):
        runner._validate_expression_payloads(payload)


def test_isothermal_raw_source_rejects_energy_target() -> None:
    payload = {
        "source_expressions": [{
            "kind": "raw",
            "dim": 2,
            "target": "nrg",
            "outputs": [0],
        }],
        "isothermal": True,
    }
    with pytest.raises(ValueError, match="energy equation"):
        runner._validate_expression_payloads(payload)


@pytest.mark.parametrize("kind", ["force", "cooling", "relax", "sponge"])
def test_relativistic_regime_rejects_newtonian_source_laws(kind: str) -> None:
    counts = {"force": 2, "cooling": 1, "relax": 3, "sponge": 5}
    payload = {
        "source_expressions": [{
            "kind": kind,
            "dim": 2,
            "outputs": list(range(counts[kind])),
        }],
        "is_relativistic": True,
    }
    with pytest.raises(ValueError, match="relativistic"):
        runner._validate_expression_payloads(payload)


@pytest.mark.parametrize(
    ("kind", "target", "count"),
    [("raw", "nrg", 1), ("inject", None, 4)],
)
def test_relativistic_regime_accepts_conserved_sources(
    kind: str, target: str | None, count: int
) -> None:
    source = {
        "kind": kind,
        "dim": 2,
        "outputs": list(range(count)),
    }
    if target is not None:
        source["target"] = target
    runner._validate_expression_payloads(
        {
            "source_expressions": [source],
            "is_relativistic": True,
        }
    )


def test_isothermal_raw_density_source_is_valid() -> None:
    runner._validate_expression_payloads(
        {
            "source_expressions": [{
                "kind": "raw",
                "dim": 2,
                "target": "den",
                "outputs": [0],
            }],
            "isothermal": True,
        }
    )
