# =============================================================================
# test_viscosity_alpha_alias.py
#
# the alpha viscosity's one wire name is `viscosity_alpha`. the deprecated
# alias `shakura_sunyaev_alpha` is read into it with a deprecation warning and
# leaves the wire. the base problem declares no `viscosity_alpha`, so a
# problem carrying it sets it, by its declaration or by passing it, and a
# value other than the alias's is refused by name, a zero included; an equal
# one runs as the one value.
#
# run: pytest simbi/simulation/tests/test_viscosity_alpha_alias.py
# =============================================================================
import warnings
from typing import Annotated

import pytest
from pydantic import computed_field

from simbi import ProblemParam
from simbi.simulation.runner import to_execution_dict
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast


def _problem(alias: float, canonical: float | None, passed: float | None = None):
    fields = {}
    if canonical is not None:
        fields["viscosity_alpha"] = Annotated[float, ProblemParam(canonical, description="alpha")]

    class Disk(RefinedBlast):
        __annotations__ = fields

        @computed_field
        @property
        def shakura_sunyaev_alpha(self) -> float:
            return alias

    kwargs = {} if passed is None else {"viscosity_alpha": passed}
    return Disk(resolution=(16, 16, 1), **kwargs)


def test_the_alias_alone_is_read_as_viscosity_alpha_with_a_warning() -> None:
    with pytest.warns(DeprecationWarning, match="shakura_sunyaev_alpha is a deprecated alias of viscosity_alpha"):
        wire = to_execution_dict(_problem(0.1, None))
    assert wire["viscosity_alpha"] == 0.1
    assert "shakura_sunyaev_alpha" not in wire, "the alias rode the wire beside its value"


def test_a_different_viscosity_alpha_beside_the_alias_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match="sets viscosity_alpha = 0.2 and its deprecated alias shakura_sunyaev_alpha = 0.1"):
        to_execution_dict(_problem(0.1, 0.2))


def test_an_equal_viscosity_alpha_beside_the_alias_runs_as_the_one_value() -> None:
    with pytest.warns(DeprecationWarning):
        wire = to_execution_dict(_problem(0.1, 0.1))
    assert wire["viscosity_alpha"] == 0.1


def test_viscosity_alpha_alone_carries_no_warning_and_no_alias() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        wire = to_execution_dict(_problem(0.0, 0.3))
    assert wire["viscosity_alpha"] == 0.3
    assert "shakura_sunyaev_alpha" not in wire


def test_an_explicit_zero_viscosity_alpha_beside_the_alias_is_refused_by_name() -> None:
    # the run asks for no alpha viscosity under the new name and for alpha = 0.1 under the old one
    with pytest.raises(ValueError, match="sets viscosity_alpha = 0.0 and its deprecated alias shakura_sunyaev_alpha = 0.1"):
        to_execution_dict(_problem(0.1, 0.0, passed=0.0))


def test_a_declared_zero_viscosity_alpha_beside_the_alias_is_refused_by_name() -> None:
    # the class declares viscosity_alpha = 0 and overrides the old name with 0.1
    with pytest.raises(ValueError, match="sets viscosity_alpha = 0.0 and its deprecated alias shakura_sunyaev_alpha = 0.1"):
        to_execution_dict(_problem(0.1, 0.0))
