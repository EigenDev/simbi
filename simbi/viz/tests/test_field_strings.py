# =============================================================================
# test_field_strings.py
#
# verifies that stored, derived, and tracer visualization fields have
# presentation labels instead of falling back to raw field identifiers.
#
# usage:
#  pytest -q simbi/viz/tests/test_field_strings.py
# =============================================================================
from simbi.viz.utility import FIELD_MAP, get_field_str, get_tracer_field_str


def test_all_reader_fields_have_visualization_strings() -> None:
    stored_fields = {
        "rho",
        "p",
        "chi",
        "v1",
        "v2",
        "v3",
        "b1",
        "b2",
        "b3",
    }
    derived_fields = {
        "W",
        "D",
        "v",
        "u",
        "energy",
        "enthalpy",
        "enthalpy_density",
        "sigma",
        "ptot",
        "pmag",
        "emag",
        "mach",
        "chi_dens",
        "j",
        "mass_flux",
        "j_spec",
        "Sigma",
        "vorticity",
        "vorticity_magnitude",
        "q_criterion",
        "okubo_weiss",
        "div_v",
        "schlieren",
        "u1",
        "u2",
        "u3",
        "m1",
        "m2",
        "m3",
        "b1_mean",
        "b2_mean",
        "b3_mean",
        "bmu1",
        "bmu2",
        "bmu3",
    }

    assert stored_fields | derived_fields <= FIELD_MAP.keys()


def test_tracer_field_strings_use_math_labels_and_cohort() -> None:
    assert get_field_str("tracer_concentration").startswith("$")
    label = get_tracer_field_str("tracer_cohort_ratio", cohort=7)
    assert "c=7" in label
    assert label.startswith("$")
