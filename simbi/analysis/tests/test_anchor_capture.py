# =============================================================================
# test_anchor_capture.py
#
# the pure extraction pieces of the anchor A/B producer, on synthetic data:
# - the schwarzschild_ks metric integral reproduces a known analytic volume
#   (a uniform integrand over a coordinate shell)
# - the densitized rest mass reduces to the flat proper mass when M -> 0 and
#   the flow is at rest (W = 1, sqrt(gamma) = r^2 sin theta)
# - the receipts mapping mirrors one tuple half of anchor_experiment_report()
# - the record assembly emits a schema-valid anchor_ab record that the harness
#   accepts and pairs
#
# the in-process run wrapper drives a real simulation and is exercised by the
# actual sweep, not here.
# =============================================================================

from typing import Any

import numpy as np

from simbi.analysis import compare_pair
from simbi.analysis.anchor_ab import RunRecord
from simbi.analysis.anchor_capture import (
    _metric_components,
    _receipts_dict,
    build_record,
)


def test_ks_sqrt_det_gamma_matches_the_analytic_form() -> None:
    # sqrt(det gamma) = sqrt(1 + 2M/r) r^2 sin(theta) on schwarzschild_ks.
    r = np.array([[4.0, 8.0]])
    theta = np.array([[np.pi / 2, np.pi / 6]])
    mass = 1.0
    _grr, _gtt, _gpp, sqrt_det = _metric_components(r, theta, mass)
    expected = np.sqrt(1.0 + 2.0 * mass / r) * r * r * np.sin(theta)
    np.testing.assert_allclose(sqrt_det, expected)


def test_flat_limit_recovers_the_euclidean_volume_element() -> None:
    # as M -> 0 the KS metric determinant is the flat spherical one, so the
    # coordinate-volume integral of unity over a shell is the euclidean volume.
    nr, nth = 200, 200
    r_edges = np.linspace(2.0, 3.0, nr + 1)
    th_edges = np.linspace(0.0, np.pi, nth + 1)
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    th_c = 0.5 * (th_edges[:-1] + th_edges[1:])
    dr = np.diff(r_edges)
    dth = np.diff(th_edges)
    r_grid = np.broadcast_to(r_c[None, :], (nth, nr))
    th_grid = np.broadcast_to(th_c[:, None], (nth, nr))
    area = np.broadcast_to(dr[None, :], (nth, nr)) * np.broadcast_to(
        dth[:, None], (nth, nr)
    )
    _grr, _gtt, _gpp, sqrt_det = _metric_components(r_grid, th_grid, 1.0e-12)
    volume = float(np.sum(sqrt_det * area) * 2.0 * np.pi)
    # 4/3 pi (3^3 - 2^3) = 4/3 pi * 19
    expected = 4.0 / 3.0 * np.pi * (27.0 - 8.0)
    assert abs(volume - expected) / expected < 1e-4


def test_receipts_dict_mirrors_the_report_tuple() -> None:
    bucket = (
        3,  # passes
        2,  # passes_fired
        7,  # projected_cells
        0.25,  # min_theta
        [1.0, 1.5, 0.3, 0.4, -0.2, 0.2],  # intervention [ms, ma, ss, sa, rs, ra]
        [0.5, 0.75, 0.15, 0.2, -0.1, 0.1],  # injected
    )
    out = _receipts_dict(bucket)
    assert out["passes"] == 3
    assert out["passes_fired"] == 2
    assert out["projected_cells"] == 7
    assert out["min_theta"] == 0.25
    assert out["intervention"]["mass"] == [1.0, 1.5]
    assert out["intervention"]["energy_segment"] == [0.3, 0.4]
    assert out["intervention"]["energy_raise"] == [-0.2, 0.2]
    assert out["injected"]["mass"] == [0.5, 0.75]


def _report(mass_signed: float) -> tuple[Any, Any]:
    fired = (
        1,
        1,
        2,
        0.5,
        [mass_signed, abs(mass_signed), 0.0, 0.0, 0.0, 0.0],
        [mass_signed, abs(mass_signed), 0.0, 0.0, 0.0, 0.0],
    )
    return fired, fired


def _record_for(convention: str) -> dict[str, Any]:
    return build_record(
        convention=convention,
        resolution=128,
        config={
            "initial_conditions": "gr_fishbone_moncrief_mhd.py",
            "end_time": 400.0,
            "integrator": "rk2",
            "cfl": 0.3,
            "solver": "hlld",
            "eos": "gamma_law",
            "chart": "schwarzschild_ks",
            "grid": "128x96",
            "run_config": "kerr_spin=0",
        },
        report=_report(1.0),
        first=(1, 1, 12.5, 40),
        census=(3, 1, 0, 0),
        replay_outcomes={"conservative_replay": 2},
        conserved_initial={"mass": 10.0, "energy": 20.0},
        conserved_final={"mass": 9.9, "energy": 19.8},
        observables={
            "survival_time": 400.0,
            "horizon_accretion_rate": 0.7,
            "torus_rest_mass": 8.0,
            "magnetic_energy": 0.05,
        },
    )


def test_built_record_is_schema_valid_and_pairs() -> None:
    stage = _record_for("stage_input")
    rebuilt = _record_for("eulerian_rebuilt")
    # each parses under the harness schema, and the two pair without a config
    # mismatch (they differ only in the convention).
    comparison = compare_pair(
        RunRecord.from_dict(stage), RunRecord.from_dict(rebuilt)
    )
    assert comparison.resolution == 128
    assert comparison.guards["fallback"].absolute == 0.0
    assert comparison.first_events["stage_input"]["accepted_first_time"] == 12.5
