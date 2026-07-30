# =============================================================================
# test_tabulated_2d_expression.py
#
# immutable bilinear table interpolation, bounds behavior, validation, and
# serialization through the ordinary expression wire.
# =============================================================================

import pytest

import simbi.expression as expr


def compiled_table(bounds: expr.TableBounds | str):
    graph = expr.ExprGraph()
    x = expr.variable("x1", graph)
    y = expr.variable("x2", graph)
    table = expr.tabulated_2d(
        x,
        y,
        [0.0, 1.0, 3.0],
        [-1.0, 2.0, 4.0],
        [
            [-2.0, -1.0, 1.0],
            [4.0, 5.0, 7.0],
            [8.0, 9.0, 11.0],
        ],
        bounds=bounds,
    )
    return graph.compile([table])


@pytest.mark.parametrize(
    ("x", "y"),
    [
        (0.0, -1.0),
        (1.0, 2.0),
        (3.0, 4.0),
        (0.5, 0.5),
        (2.0, 3.0),
    ],
)
def test_tabulated_2d_reproduces_linear_field(x: float, y: float) -> None:
    assert compiled_table("clamp").evaluate(x1=x, x2=y) == [x + 2.0 * y]


def test_tabulated_2d_clamps_each_axis() -> None:
    table = compiled_table(expr.TableBounds.CLAMP)
    assert table.evaluate(x1=-5.0, x2=0.5) == [1.0]
    assert table.evaluate(x1=2.0, x2=8.0) == [10.0]


def test_tabulated_2d_zeroes_if_either_axis_is_outside() -> None:
    table = compiled_table(expr.TableBounds.ZERO)
    assert table.evaluate(x1=-5.0, x2=0.5) == [0.0]
    assert table.evaluate(x1=2.0, x2=8.0) == [0.0]


def test_tabulated_2d_rejects_ragged_values() -> None:
    graph = expr.ExprGraph()
    with pytest.raises(ValueError, match="must have shape"):
        expr.tabulated_2d(
            expr.variable("x1", graph),
            expr.variable("x2", graph),
            [0.0, 1.0],
            [0.0, 1.0],
            [[0.0, 1.0], [2.0]],
            bounds="clamp",
        )


def test_tabulated_2d_serializes_to_backend_supported_nodes() -> None:
    payload = compiled_table("clamp").serialize_source(
        expr.SourceKind.RAW,
        dim=2,
        target=expr.ConservedField.ENERGY,
    )
    assert {node["op"] for node in payload["nodes"]} <= {
        "CONSTANT",
        "VARIABLE_X1",
        "VARIABLE_X2",
        "SUBTRACT",
        "DIVIDE",
        "MULTIPLY",
        "ADD",
        "LT",
        "GT",
        "IF_THEN_ELSE",
    }
