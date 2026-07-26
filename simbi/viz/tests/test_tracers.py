# =============================================================================
# test_tracers.py
#
# ownership-native tracer loading and tracer-only rendering. exact uint64
# identities and owners never pass through floating point, deterministic spawn
# state remains available, and standalone plots use checkpoint geometry.
# =============================================================================

import argparse
from types import SimpleNamespace

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from simbi.viz import api
from simbi.viz.cli import setup_parser
from simbi.viz.config import PlotConfig, VisualizationConfig
from simbi.viz.tracers import load_tracers, tracer_concentration


def test_load_tracers_preserves_exact_ownership_and_spawn_state(tmp_path):
    path = tmp_path / "tracers.h5"
    ids = np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64)
    owners = np.array([2**56 + 7, 2**62], dtype=np.uint64)
    with h5py.File(path, "w") as checkpoint:
        group = checkpoint.create_group("tracers")
        group.attrs["run_seed"] = np.uint64(2**64 - 3)
        group.attrs["next_id"] = np.uint64(2**63 + 9)
        group.attrs["injection_remainder"] = 0.125
        group.create_dataset("position", data=np.array([[0.25], [0.75]]))
        group.create_dataset("id", data=ids)
        group.create_dataset("owner", data=owners)
        group.create_dataset("escaped", data=np.array([0.0, 0.0]))
        group.create_dataset("crossed_sink", data=np.array([0.0, 1.0]))
        group.create_dataset("crossing_time", data=np.array([0.0, 0.5]))
        group.create_dataset("weight", data=np.array([0.01]))

    cloud = load_tracers(str(path))

    assert cloud is not None
    np.testing.assert_array_equal(cloud.id, ids)
    np.testing.assert_array_equal(cloud.owner, owners)
    assert cloud.id.dtype == np.dtype("uint64")
    assert cloud.owner.dtype == np.dtype("uint64")
    assert cloud.run_seed == 2**64 - 3
    assert cloud.next_id == 2**63 + 9
    assert cloud.injection_remainder == 0.125


def test_tracers_only_cli_flag_is_available():
    parser = argparse.ArgumentParser()
    setup_parser(parser)

    args = parser.parse_args(["checkpoint.h5", "--tracers-only"])

    assert args.tracers_only
    assert args.tracer_render == "concentration"


def test_tracer_concentration_is_mass_per_area_and_excludes_reservoirs():
    cloud = SimpleNamespace(
        position=np.array(
            [[0.5, 0.5], [0.5, 0.5], [1.5, 0.5], [1.5, 1.5]]
        ),
        owner=np.array([0, 0, 1, 1 << 62], dtype=np.uint64),
        weight=2.0,
    )
    edges = np.array([0.0, 1.0, 2.0])

    concentration = tracer_concentration(
        cloud,
        edges,
        edges,
        smoothing=0.0,
    )

    np.testing.assert_array_equal(concentration, [[4.0, 2.0], [0.0, 0.0]])
    assert np.sum(concentration) == 3 * cloud.weight


def test_tracers_only_plot_uses_checkpoint_bounds(monkeypatch):
    mesh = SimpleNamespace(
        ndim=2,
        x1v=np.array([-2.0, 3.0]),
        x2v=np.array([-1.0, 4.0]),
        x3v=np.array([0.0, 1.0]),
    )
    sim_data = SimpleNamespace(
        metadata=SimpleNamespace(coord_system="cartesian"),
        mesh=mesh,
    )
    cloud = type("Cloud", (), {"__len__": lambda self: 2})()

    monkeypatch.setattr(api, "load_data", lambda _path: sim_data)
    monkeypatch.setattr("simbi.viz.tracers.load_tracers", lambda _path: cloud)

    def scatter(ax, _path, **_kwargs):
        return ax.scatter([0.0, 1.0], [1.0, 2.0])

    monkeypatch.setattr("simbi.viz.tracers.overlay_tracers", scatter)
    config = VisualizationConfig(
        plot=PlotConfig(plot_type="multidim", fields=["rho"], ndim=2)
    )

    figure = api.plot_tracers(
        config,
        "checkpoint.h5",
        show=False,
        tracer_render="scatter",
    )
    ax = figure.axes["main"]

    np.testing.assert_allclose(ax.get_xlim(), [-2.0, 3.0])
    np.testing.assert_allclose(ax.get_ylim(), [-1.0, 4.0])
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert len(ax.collections) == 1
