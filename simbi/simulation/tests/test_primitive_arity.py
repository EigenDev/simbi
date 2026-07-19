# =============================================================================
# test_primitive_arity.py
#
# the per-cell primitive tuple is read POSITIONALLY by the backend, so a
# too-long tuple silently shifts a trailing field (e.g. pressure) into an
# ignored slot with no error. these tests pin the width where the regime
# determines it exactly (cartesian hydro, energy mhd) and confirm the ambiguous
# regimes (isothermal's optional pressure, curvilinear/relativistic velocity
# dof) are left to a lower-bound check so a legitimate config is never rejected.
#
# the reproduced defect: a 2d cartesian newtonian config that yields the mhd
# 5-tuple (rho, vx, vy, vz, p) where the hydro 4-tuple (rho, vx, vy, p) is expected —
# the reader takes p from the vz slot and drops the real pressure, so the gas
# runs pressureless (cold, wildly supersonic) with no error.
# =============================================================================
import pytest

from simbi.simulation.runner import _check_first_tuple
from simbi_configs.examples.isothermal.kepler import KeplerianRingTest
from simbi_configs.examples.newtonian.viscous_shear import ViscousShear
from simbi_configs.examples.srmhd.rmhd_orszag_tang import OrszagTang


def test_cartesian_hydro_arity_is_exact() -> None:
    # 2d cartesian newtonian: rho + 2 velocities + pressure.
    prob = ViscousShear()
    assert prob.expected_primitive_arity() == (4, "(rho, v1, v2, p)")


def test_energy_mhd_arity_is_exact_five() -> None:
    # mhd carries the full 3-velocity on every chart: rho + 3 velocities + p.
    prob = OrszagTang()
    assert prob.expected_primitive_arity() == (5, "(rho, v1, v2, v3, p)")


def test_hydro_dimensionality_sets_the_velocity_count() -> None:
    # the exact width tracks the spatial dimensionality for cartesian hydro.
    assert ViscousShear.from_cli(["--resolution", "64"]).expected_primitive_arity()[0] == 3
    assert ViscousShear.from_cli(["--resolution", "64,64"]).expected_primitive_arity()[0] == 4
    assert ViscousShear.from_cli(["--resolution", "16,16,16"]).expected_primitive_arity()[0] == 5


def test_isothermal_width_is_undetermined() -> None:
    # an isothermal run may or may not pass an explicit p/cs field.
    assert KeplerianRingTest().expected_primitive_arity() is None


def test_overlong_tuple_is_rejected_with_a_pressure_hint() -> None:
    # the defect: the mhd 5-tuple in a 2d cartesian hydro run drops pressure.
    prob = ViscousShear()
    with pytest.raises(ValueError) as exc:
        _check_first_tuple(prob, iter([(1.0, 0.0, 0.1, 0.0, 1.0)]))
    msg = str(exc.value)
    assert "EXACTLY 4" in msg
    assert "pressure" in msg


def test_correct_tuple_passes_and_replays() -> None:
    # the right 4-tuple is accepted and the peeked first item is replayed intact.
    prob = ViscousShear()
    out = _check_first_tuple(prob, iter([(1.0, 0.0, 0.1, 1.0)]))
    assert next(out) == (1.0, 0.0, 0.1, 1.0)


def test_undetermined_regime_tolerates_a_longer_tuple() -> None:
    # isothermal returns None -> the exact check is skipped and a longer tuple
    # (rho, vx, vy, p) is accepted.
    prob = KeplerianRingTest()
    out = _check_first_tuple(prob, iter([(1.0, 0.0, 0.0, 1.0)]))
    assert next(out) == (1.0, 0.0, 0.0, 1.0)


def test_undetermined_floor_scales_with_dimensionality() -> None:
    # only the trailing pressure is optional; velocities never are. a 2d
    # isothermal tuple missing v2 must be rejected by the lower-bound check.
    prob = KeplerianRingTest()
    with pytest.raises(ValueError, match=">= 3"):
        _check_first_tuple(prob, iter([(1.0, 0.0)]))
