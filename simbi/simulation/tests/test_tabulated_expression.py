# =============================================================================
# test_tabulated_expression.py
#
# immutable one-dimensional table validation, interpolation, bounds behavior,
# and source-wire lowering through the ordinary expression dag.
# =============================================================================

import pytest

import simbi.expression as expr


def compiled_table(bounds: expr.TableBounds | str):
    graph = expr.ExprGraph()
    radius = expr.variable("x1", graph)
    table = expr.tabulated_1d(
        radius,
        [1.0, 2.0, 4.0],
        [10.0, 20.0, 0.0],
        bounds=bounds,
    )
    return graph.compile([table])


@pytest.mark.parametrize(
    ("coordinate", "expected"),
    [
        (1.0, 10.0),
        (1.5, 15.0),
        (2.0, 20.0),
        (3.0, 10.0),
        (4.0, 0.0),
    ],
)
def test_tabulated_1d_interpolates_knots_and_interiors(
    coordinate: float, expected: float
) -> None:
    assert compiled_table("clamp").evaluate(x1=coordinate) == [expected]


def test_tabulated_1d_clamps_outside_samples() -> None:
    table = compiled_table(expr.TableBounds.CLAMP)
    assert table.evaluate(x1=-5.0) == [10.0]
    assert table.evaluate(x1=9.0) == [0.0]


def test_tabulated_1d_zeroes_outside_samples() -> None:
    table = compiled_table(expr.TableBounds.ZERO)
    assert table.evaluate(x1=-5.0) == [0.0]
    assert table.evaluate(x1=9.0) == [0.0]


@pytest.mark.parametrize(
    ("coordinates", "values", "message"),
    [
        ([1.0], [2.0], "at least two"),
        ([1.0, 2.0], [3.0], "lengths differ"),
        ([1.0, 1.0], [2.0, 3.0], "strictly increasing"),
        ([1.0, float("nan")], [2.0, 3.0], "finite"),
    ],
)
def test_tabulated_1d_rejects_invalid_samples(
    coordinates, values, message: str
) -> None:
    graph = expr.ExprGraph()
    with pytest.raises(ValueError, match=message):
        expr.tabulated_1d(
            expr.variable("x1", graph),
            coordinates,
            values,
            bounds="clamp",
        )


def test_tabulated_1d_rejects_implicit_bounds_policy() -> None:
    graph = expr.ExprGraph()
    with pytest.raises(ValueError, match="clamp.*zero"):
        expr.tabulated_1d(
            expr.variable("x1", graph),
            [0.0, 1.0],
            [0.0, 1.0],
            bounds="extrapolate",
        )


def test_tabulated_1d_serializes_to_backend_supported_nodes() -> None:
    payload = compiled_table("clamp").serialize_source(
        expr.SourceKind.RAW,
        dim=1,
        target=expr.ConservedField.ENERGY,
    )
    ops = {node["op"] for node in payload["nodes"]}
    assert ops <= {
        "CONSTANT",
        "VARIABLE_X1",
        "SUBTRACT",
        "DIVIDE",
        "MULTIPLY",
        "ADD",
        "LT",
        "GT",
        "IF_THEN_ELSE",
    }
