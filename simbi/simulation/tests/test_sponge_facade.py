# =============================================================================
# test_sponge_facade.py
#
# the sponge-parameters compatibility facade: `sponge_parameters` is the canonical
# override point, `buffer_parameters` is the deprecated historical name kept as the
# one serialized computed field for a compatibility window. the two bridge both
# ways, so a config overriding either name works, and exactly one
# `buffer_parameters` key is serialized (the wire is unchanged).
#
# the legacy subclass here mirrors the ignored porous science config's pattern
# (`@computed_field @property def buffer_parameters`) so the gate proves that file
# still dispatches identically without editing it.
# =============================================================================
import warnings

import pytest
from pydantic import computed_field

from simbi import SimbiProblem
from simbi.types import CoordSystem, Regime


class _Base(SimbiProblem):
    regime: Regime = Regime.NEWTONIAN
    coord_system: CoordSystem = CoordSystem.CARTESIAN
    resolution: tuple[int, int] = (16, 16)
    bounds: list = [(0.0, 1.0), (0.0, 1.0)]
    adiabatic_index: float = 5.0 / 3.0

    def initial_primitive_state(self):
        return lambda: (1.0, 0.0, 0.0, 1.0)


class _NewStyle(_Base):
    @property
    def sponge_parameters(self) -> dict[str, float]:
        return {"kappa": 1.0, "buffer_radius": 2.0}


class _LegacyStyle(_Base):
    # the historical pattern the ignored porous config uses, unchanged.
    @computed_field
    @property
    def buffer_parameters(self) -> dict[str, float]:
        return {"kappa": 3.0, "buffer_radius": 4.0}


class _BothStyle(_Base):
    @property
    def sponge_parameters(self) -> dict[str, float]:
        return {"kappa": 1.0}

    @computed_field
    @property
    def buffer_parameters(self) -> dict[str, float]:
        return {"kappa": 3.0}


def _dump(problem) -> dict:
    # model_dump on a SimbiProblem emits an unrelated pydantic serialization
    # warning for a uint64 field; that is orthogonal to the facade.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return problem.model_dump()


def test_new_override_feeds_the_serialized_legacy_key():
    p = _NewStyle()
    assert p.sponge_parameters == {"kappa": 1.0, "buffer_radius": 2.0}
    # the serialized computed field carries the canonical value under the historical key.
    assert _dump(p)["buffer_parameters"] == {"kappa": 1.0, "buffer_radius": 2.0}
    assert p.buffer_parameters == {"kappa": 1.0, "buffer_radius": 2.0}


def test_legacy_override_is_read_by_the_canonical_name():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p = _LegacyStyle()
    # accessing the canonical name returns the legacy override's value.
    assert p.sponge_parameters == {"kappa": 3.0, "buffer_radius": 4.0}
    assert _dump(p)["buffer_parameters"] == {"kappa": 3.0, "buffer_radius": 4.0}


def test_exactly_one_key_is_serialized():
    dumped = _dump(_NewStyle())
    assert "buffer_parameters" in dumped
    assert "sponge_parameters" not in dumped


def test_a_legacy_override_does_not_warn():
    # a legacy method override is a structural compat bridge, not a user input:
    # it works silently. only deprecated input keys and cli flags warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _LegacyStyle()


def test_a_new_override_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        _NewStyle()  # must not raise — no deprecation for the canonical name


def test_a_legacy_input_key_warns_once():
    from simbi_configs.examples.newtonian.bondi import SphericalBondiTest

    with pytest.warns(DeprecationWarning, match="use_buffer"):
        SphericalBondiTest(use_buffer=False)


def test_overriding_both_fails_loudly():
    with pytest.raises(ValueError, match="both"):
        _BothStyle()


# ---- the migrated tracked config: canonical == legacy input, and the cli ------


def _bondi(**kwargs):
    from simbi_configs.examples.newtonian.bondi import SphericalBondiTest

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p = SphericalBondiTest(**kwargs)
        p.setup()
    return p


def test_canonical_and_legacy_inputs_serialize_identically():
    canonical = _bondi(use_sponge=False, sponge_time_fraction=0.2)
    legacy = _bondi(use_buffer=False, buffer_time_fraction=0.2)
    assert _dump(canonical) == _dump(legacy)
    # the computed run directory (its path tokens) is identical too.
    assert canonical.data_directory == legacy.data_directory


def test_only_the_legacy_wire_key_is_serialized():
    dumped = _dump(_bondi())
    assert "buffer_parameters" in dumped
    assert "sponge_parameters" not in dumped


def test_cli_accepts_a_deprecated_flag_with_one_warning():
    from simbi_configs.examples.newtonian.bondi import SphericalBondiTest

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always", DeprecationWarning)
        p = SphericalBondiTest.from_cli(["--buffer-time-fraction", "0.25"])
    assert p.sponge_time_fraction == 0.25
    legacy_warnings = [w for w in rec if "buffer-time-fraction" in str(w.message)]
    assert len(legacy_warnings) == 1


def test_cli_rejects_mixing_canonical_and_legacy_flags():
    from simbi_configs.examples.newtonian.bondi import SphericalBondiTest

    with pytest.raises(Exception, match="not both"):
        SphericalBondiTest.from_cli(
            ["--sponge-time-fraction", "0.1", "--buffer-time-fraction", "0.2"]
        )
