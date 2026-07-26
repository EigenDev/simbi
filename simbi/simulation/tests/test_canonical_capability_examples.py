# =============================================================================
# test_canonical_capability_examples.py
#
# import, payload, rust-preflight, and bounded-evolution gates for the canonical
# examples of the modern source, table, geometric-grid, decomposition, and
# mass-transport tracer APIs.
# =============================================================================

from itertools import islice
from pathlib import Path

import h5py
import numpy as np
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
from simbi_configs.examples.newtonian.traced_kh import TracedKelvinHelmholtz


EXAMPLE_FACTORIES = [
    OrderedSources,
    RotatingSponge,
    TabulatedSource1D,
    TabulatedSource2D,
    GeometricBoundaries,
    DecomposedTabulatedGeometric,
    TracedKelvinHelmholtz,
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
    assert problem.x2_spacing_ratio != 1.0
    assert payload["source_expressions"][0]["kind"] == "raw"


def test_traced_kh_initial_state_is_repeatable():
    problem = TracedKelvinHelmholtz()

    first = list(islice(problem.initial_primitive_state()(), 1024))
    second = list(islice(problem.initial_primitive_state()(), 1024))

    assert first == second
    assert problem.n_tracers == problem.tracers
    cohorts = np.fromiter(problem.tracer_cohort(), dtype=np.uint16)
    assert cohorts.size == np.prod(problem.resolution)
    np.testing.assert_array_equal(np.unique(cohorts), [0, 1])
    assert np.count_nonzero(cohorts == 0) == np.count_nonzero(cohorts == 1)


@pytest.mark.parametrize("factory", EXAMPLE_FACTORIES)
def test_canonical_example_passes_rust_preflight(factory):
    backend.validate_simulation(sim_info=to_execution_dict(factory()))


@pytest.mark.parametrize("factory", EXAMPLE_FACTORIES)
def test_canonical_example_advances_one_production_step(factory, tmp_path: Path):
    problem = factory(data_directory=tmp_path / factory.__name__)

    run(problem, compute_mode="cpu", max_steps=1)
    if factory is TracedKelvinHelmholtz:
        checkpoints = list(problem.data_directory.glob("*final*.h5"))
        assert checkpoints
        with h5py.File(checkpoints[0]) as checkpoint:
            cohorts = np.asarray(checkpoint["tracers/cohort"], dtype=np.uint16)
        assert cohorts.size == problem.n_tracers
        np.testing.assert_array_equal(np.unique(cohorts), [0, 1])
