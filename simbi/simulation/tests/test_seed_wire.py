# =============================================================================
# test_seed_wire.py
#
# the python -> rust velocity-seed wire. `seed_modes` rows are
# [kx, ky, kz, ex, ey, ez, amp, phase, r_cut] read positionally by the backend,
# and the seed is defined only on a 3d cartesian newtonian grid — so the runner
# validation must reject every payload the backend would refuse (or silently
# drop), with the reason.
# =============================================================================
import math

import pytest

from simbi.simulation.runner import _validate_seed_payload


def valid_payload() -> dict:
    return {
        "seed_modes": [
            [6.0, 2.0, -1.5, 0.0, 0.3, 0.9, 0.02, 0.7, 0.0],
            [3.0, -8.0, 5.0, 0.8, 0.1, -0.2, 0.05, 2.1, 0.6],
        ],
        "seed_taper": [0.5, 0.4],
        "dimensionality": 3,
        "coord_system": "cartesian",
        "is_mhd": False,
        "is_relativistic": False,
    }


def test_valid_payload_passes() -> None:
    _validate_seed_payload(valid_payload())


def test_empty_seed_ignores_everything_else() -> None:
    _validate_seed_payload({"seed_modes": [], "dimensionality": 1})


def test_mhd_is_rejected() -> None:
    bad = valid_payload() | {"is_mhd": True}
    with pytest.raises(ValueError, match="div\\(B\\)"):
        _validate_seed_payload(bad)


def test_relativistic_is_rejected() -> None:
    bad = valid_payload() | {"is_relativistic": True}
    with pytest.raises(ValueError, match="newtonian"):
        _validate_seed_payload(bad)


def test_lower_dimensionality_is_rejected() -> None:
    bad = valid_payload() | {"dimensionality": 2}
    with pytest.raises(ValueError, match="3d"):
        _validate_seed_payload(bad)


def test_curvilinear_grid_is_rejected() -> None:
    bad = valid_payload() | {"coord_system": "spherical"}
    with pytest.raises(ValueError, match="cartesian"):
        _validate_seed_payload(bad)


def test_missing_taper_is_rejected() -> None:
    bad = valid_payload() | {"seed_taper": []}
    with pytest.raises(ValueError, match="seed_taper"):
        _validate_seed_payload(bad)


def test_nonpositive_taper_is_rejected() -> None:
    bad = valid_payload() | {"seed_taper": [0.5, 0.0]}
    with pytest.raises(ValueError, match="positive"):
        _validate_seed_payload(bad)


def test_short_row_is_rejected() -> None:
    bad = valid_payload()
    bad["seed_modes"] = [bad["seed_modes"][0][:8]]
    with pytest.raises(ValueError, match="9 finite entries"):
        _validate_seed_payload(bad)


def test_nonfinite_row_is_rejected() -> None:
    bad = valid_payload()
    bad["seed_modes"][1][6] = math.nan
    with pytest.raises(ValueError, match="9 finite entries"):
        _validate_seed_payload(bad)


def test_zero_wavevector_is_rejected() -> None:
    bad = valid_payload()
    bad["seed_modes"][0][0:3] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="zero wavevector"):
        _validate_seed_payload(bad)
