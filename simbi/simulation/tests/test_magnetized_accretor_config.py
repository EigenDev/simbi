# =============================================================================
# test_magnetized_accretor_config.py
#
# configuration-only checks for the plain magnetic accretor: physical scales,
# closure-specific primitive layouts, staggered uniform fields, body coupling,
# the bounded cpu entry point, and the guard totals the entry point prints from
# the run result. no simulation is evolved.
# =============================================================================
import math

import pytest

from simbi.types import Regime
from simbi.types.bodies import BodyCapability
from simbi_configs.examples.ibm.magnetized_accretor import MagnetizedAccretor, main


def test_default_scales_and_plain_fixed_mass_body():
    problem = MagnetizedAccretor()
    assert problem.bondi_radius == 1.0
    assert problem.bondi_time == 1.0
    assert problem.bounds == [(-4.0, 4.0)] * 3
    assert problem.sink_radius == 0.25
    assert problem.sink_radius / problem.cell_width == 4.0
    assert problem.end_time == 1.0
    assert problem.checkpoint_interval == 0.1
    body = problem.immersed_bodies[0]
    assert body.capability == BodyCapability.ACCRETION | BodyCapability.GRAVITATIONAL
    assert body.mass == 1.0
    assert not body.two_way_coupling
    assert body.magnetic is None
    assert body.accretion.accretion_radius == 0.25
    assert body.accretion.porosity is None
    assert body.accretion.torque_free_xi is None
    assert body.gravitational.softening_length == 0.05


@pytest.mark.parametrize("regime", [Regime.NMHD, Regime.IMHD])
def test_initial_gas_and_faces_match_the_selected_closure(regime):
    problem = MagnetizedAccretor(
        regime=regime,
        resolution=(16, 16, 16),
        domain_radius=1.0,
        ambient_density=2.0,
        sound_speed=3.0,
        beta=4.0,
    )
    gas, bx, by, bz = problem.initial_primitive_state()
    states = list(gas())
    pressure = 18.0 if regime == Regime.IMHD else 18.0 / (5.0 / 3.0)
    expected = (2.0, 0.0, 0.0, 0.0)
    if regime == Regime.NMHD:
        expected += (pressure,)
    assert len(states) == 16**3
    assert all(state == expected for state in states)
    for axis, faces in enumerate((bx, by, bz)):
        values = list(faces())
        assert len(values) == 17 * 16**2
        assert all(
            value == (math.sqrt(2.0 * pressure / 4.0) if axis == 0 else 0.0)
            for value in values
        )
    assert 2.0 * pressure / problem.field_strength**2 == pytest.approx(4.0)


def test_changing_resolution_keeps_the_physical_sink_and_softening():
    coarse = MagnetizedAccretor(resolution=(64, 64, 64))
    fine = MagnetizedAccretor(resolution=(128, 128, 128))
    assert coarse.immersed_bodies == fine.immersed_bodies
    assert coarse.cell_width == 2.0 * fine.cell_width
    assert coarse.data_directory != fine.data_directory


def test_dimensional_scales_and_cli_closure():
    problem = MagnetizedAccretor.from_cli(
        [
            "--central-mass",
            "8",
            "--sound-speed",
            "2",
            "--regime",
            "imhd",
            "--total-bondi-times",
            "0.5",
            "--beta",
            "10",
        ]
    )
    assert problem.regime == Regime.IMHD
    assert problem.bondi_radius == 2.0
    assert problem.bondi_time == 1.0
    assert problem.bounds == [(-8.0, 8.0)] * 3
    assert problem.sink_radius == 0.5
    assert problem.end_time == 0.5
    assert problem.ambient_pressure == 4.0


@pytest.mark.parametrize("regime", [Regime.NMHD, Regime.IMHD])
def test_execution_payload_preserves_the_closure_and_body(regime):
    from simbi.simulation.runner import to_execution_dict
    from simbi.types.bodies import body_payload

    problem = MagnetizedAccretor(regime=regime, sound_speed=2.0)
    payload = to_execution_dict(problem)
    assert payload["regime"] == regime.value
    assert payload["isothermal"] == (regime == Regime.IMHD)
    assert payload["ambient_sound_speed"] == 2.0
    assert payload["end_time"] == problem.bondi_time
    expected = body_payload(problem.body_system, problem.immersed_bodies)
    for key, value in expected.items():
        assert payload[key] == value


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"beta": 0.0}, "beta"),
        ({"beta": float("nan")}, "beta"),
        ({"regime": Regime.NEWTONIAN}, "nmhd and imhd"),
        ({"resolution": (32, 64, 64)}, "cubic grid"),
        ({"resolution": (16, 16, 16)}, "at least two cells"),
        ({"accretion_radius_ratio": 5.0}, "inside the box"),
        ({"softening_ratio": 0.5}, "softening_ratio"),
        ({"bounds": [(-1.0, 1.0)] * 3}, "bounds must match"),
    ],
)
def test_invalid_or_unsupported_setup_is_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        MagnetizedAccretor(**changes)


def test_entry_point_passes_a_step_bound_without_running_a_simulation(monkeypatch):
    from simbi.simulation import runner

    calls = []
    monkeypatch.setattr(
        runner, "run", lambda problem, **kwargs: calls.append((problem, kwargs))
    )
    main(["--max-steps", "25", "--beta", "2", "--regime", "imhd"])
    problem, kwargs = calls.pop()
    assert problem.beta == 2.0
    assert problem.regime == Regime.IMHD
    assert kwargs == {"compute_mode": "cpu", "max_steps": 25}
    with pytest.raises(SystemExit):
        main(["--max-steps", "0"])
    assert not calls


def test_entry_point_prints_the_accepted_guard_totals_of_the_run_result(monkeypatch, capsys):
    from pathlib import Path

    from simbi.simulation import (
        CellCount,
        GuardDiagnostics,
        Injection,
        ProjectionDiagnostics,
        RunDiagnostics,
        RunResult,
        runner,
    )

    zero = Injection(signed=0.0, gross=0.0)
    projection = ProjectionDiagnostics(passes_fired=0, projected_cells=0, min_theta=1.0, injected_den=zero, injected_nrg=zero)
    guards = GuardDiagnostics(troubled_cells=CellCount(total=7, inside_horizon=0), frozen_cells=CellCount(total=2, inside_horizon=0))
    result = RunResult(data_directory=Path("x"), diagnostics=RunDiagnostics(projection=projection, guards=guards))
    monkeypatch.setattr(runner, "run", lambda problem, **kwargs: result)
    main(["--max-steps", "25"])
    out = capsys.readouterr().out
    assert "troubled cells 7" in out and "frozen cells 2" in out, out
