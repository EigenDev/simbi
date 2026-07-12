# =============================================================================
# test_expression_variable_aliases.py
#
# coordinate variable aliases must map to the right axis op. "phi" is the
# azimuth (x3): in 3d spherical (r, theta, phi) a phi-dependent source that
# serialized to VARIABLE_X2 would silently read theta instead.
# =============================================================================

import pytest

from simbi.expression.dag_expression import variable


def _op_of(name: str) -> str:
    expr = variable(name)
    exprs = expr.graph.compile([expr]).serialize()["expressions"]
    assert len(exprs) == 1
    return exprs[0]["op"]


def test_phi_serializes_to_x3() -> None:
    assert _op_of("phi") == "VARIABLE_X3"


def test_theta_serializes_to_x2() -> None:
    assert _op_of("theta") == "VARIABLE_X2"


@pytest.mark.parametrize(
    "name, op",
    [
        ("x1", "VARIABLE_X1"),
        ("r", "VARIABLE_X1"),
        ("x2", "VARIABLE_X2"),
        ("x3", "VARIABLE_X3"),
        ("z", "VARIABLE_X3"),
    ],
)
def test_axis_aliases_are_unambiguous(name: str, op: str) -> None:
    assert _op_of(name) == op
