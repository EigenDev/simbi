# =============================================================================
# test_cli_resolution_and_errors.py
#
# two contracts of the CLI input path:
#  - a short resolution input is padded to the field's declared tuple arity with
#    singleton trailing axes, so a 2d run on a 3-component (mhd) config needs
#    only `--resolution nx,ny`; the unused nz=1 is filled in.
#  - bad cli input surfaces as a clean, traceback-free `ConfigError`: an invalid
#    enum names the valid choices, and a malformed number does not echo the whole
#    model-input dict.
# =============================================================================
from typing import Annotated

import pytest

from simbi.simulation.param import ProblemParam
from simbi.simulation.problem import ConfigError, SimbiProblem
from simbi.types.input import Solver
from simbi_configs.examples.srmhd.rmhd_orszag_tang import OrszagTang


def test_short_resolution_pads_to_field_arity() -> None:
    # OrszagTang declares resolution as tuple[int, int, int]; a 2-value cli input
    # must pad to (nx, ny, 1); an unpadded 2-value input would fail the required 3-tuple field.
    prob = OrszagTang.from_cli(["--resolution", "128,256"])
    assert prob.resolution == (128, 256, 1)
    assert prob.dimensionality == 2


def test_full_resolution_is_unchanged() -> None:
    # an exactly-sized input is passed through verbatim (no spurious padding).
    prob = OrszagTang.from_cli(["--resolution", "64,64,4"])
    assert prob.resolution == (64, 64, 4)
    assert prob.dimensionality == 3


def test_overlong_resolution_raises_config_error() -> None:
    # too many axes is a real error — surfaced clean, with no padding or truncation.
    with pytest.raises(ConfigError) as exc:
        OrszagTang.from_cli(["--resolution", "64,64,64,64"])
    assert "resolution" in str(exc.value)
    assert "Traceback" not in str(exc.value)


def test_non_numeric_resolution_raises_config_error() -> None:
    with pytest.raises(ConfigError) as exc:
        OrszagTang.from_cli(["--resolution", "abc,64"])
    msg = str(exc.value)
    assert "resolution" in msg
    # the whole model-input dict must not be echoed into the message.
    assert "cfl_number" not in msg


def test_bad_enum_raises_config_error_with_choices() -> None:
    # a bad enum value must surface as a ConfigError, never a raw KeyError('bogus').
    with pytest.raises(ConfigError) as exc:
        OrszagTang.from_cli(["--resolution", "64,64", "--solver", "bogus"])
    msg = str(exc.value)
    assert "solver" in msg
    # the message must enumerate the valid choices.
    for choice in (Solver.HLLE.name.lower(), Solver.HLLD.name.lower()):
        assert choice in msg


def test_tuple_field_arity_resolution() -> None:
    # the arity helper underpins the padding: fixed tuple -> its length;
    # a variadic/non-tuple field -> None (no padding).
    assert OrszagTang._tuple_field_arity("resolution") == 3


class _VariadicRes(SimbiProblem):
    resolution: Annotated[
        tuple[int, ...], ProblemParam((256,), description="variadic resolution")
    ]


def test_variadic_tuple_field_has_no_fixed_arity() -> None:
    # a `tuple[int, ...]` field must not be padded (arity is None).
    assert _VariadicRes._tuple_field_arity("resolution") is None
