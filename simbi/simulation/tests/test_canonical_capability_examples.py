# =============================================================================
# test_canonical_capability_examples.py
#
# import, payload, rust-preflight, and bounded-evolution gates for the canonical
# examples of the modern source, table, geometric-grid, and decomposition APIs.
# =============================================================================

from pathlib import Path

import pytest

import simbi.libs.cpu_ext as backend
from simbi.simulation.runner import run, to_execution_dict
from simbi_configs.examples.newtonian.decomposed_tabulated_geometric import (
    DecomposedTabulatedGeometric,
)
from simbi_configs.examples.newtonian.geometric_boundaries import (
    GeometricBoundaries,
)
from simbi_configs.examples.newtonian.ordered_sources import OrderedSources
from simbi_configs.examples.newtonian.rotating_sponge import RotatingSponge
from simbi_configs.examples.newtonian.tabulated_source_1d import TabulatedSource1D
from simbi_configs.examples.newtonian.tabulated_source_2d import TabulatedSource2D


EXAMPLE_FACTORIES = [
    OrderedSources,
    RotatingSponge,
    TabulatedSource1D,
    TabulatedSource2D,
    GeometricBoundaries,
    DecomposedTabulatedGeometric,
]


def test_ordered_source_example_preserves_channel_order():
    payload = to_execution_dict(OrderedSources())

    assert [source["target"] for source in payload["source_expressions"]] == [
        "den",
        "mom",
        "nrg",
    ]


def test_rotating_sponge_example_composes_two_source_kinds():
    payload = to_execution_dict(RotatingSponge())

    assert [source["kind"] for source in payload["source_expressions"]] == [
        "rotating_frame",
        "sponge",
    ]


def test_geometric_example_concentrates_at_either_boundary():
    lower = GeometricBoundaries(cluster_upper=False)
    upper = GeometricBoundaries(cluster_upper=True)

    assert lower.x1_spacing_ratio > 1.0
    assert upper.x1_spacing_ratio < 1.0


def test_integration_example_enables_decomposition():
    problem = DecomposedTabulatedGeometric(gpus=2)
    payload = to_execution_dict(problem)

    assert problem.gpus == 2
    assert problem.x1_spacing_ratio != 1.0
    assert payload["source_expressions"][0]["kind"] == "raw"


@pytest.mark.parametrize("factory", EXAMPLE_FACTORIES)
def test_canonical_example_passes_rust_preflight(factory):
    backend.validate_simulation(sim_info=to_execution_dict(factory()))


@pytest.mark.parametrize("factory", EXAMPLE_FACTORIES)
def test_canonical_example_advances_one_production_step(factory, tmp_path: Path):
    problem = factory(data_directory=tmp_path / factory.__name__)

    run(problem, compute_mode="cpu", max_steps=1)
