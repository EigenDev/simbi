# =============================================================================
# test_tabulated_3d_expression.py
#
# immutable trilinear table interpolation, bounds behavior, validation, and
# serialization through the ordinary expression wire.
# =============================================================================

import pytest

import simbi.expression as expr


def compiled_table(bounds: expr.TableBounds | str):
    graph = expr.ExprGraph()
    x = expr.variable("x1", graph)
    y = expr.variable("x2", graph)
    z = expr.variable("x3", graph)
    xs = [0.0, 1.0, 3.0]
    ys = [-1.0, 2.0, 4.0]
    zs = [-2.0, 0.0, 5.0]
    values = [
        [[x_value + 2.0 * y_value - 0.5 * z_value for x_value in xs] for y_value in ys]
        for z_value in zs
    ]
    table = expr.tabulated_3d(
        x,
        y,
        z,
        xs,
        ys,
        zs,
        values,
        bounds=bounds,
    )
    return graph.compile([table])


@pytest.mark.parametrize(
    ("x", "y", "z"),
    [
        (0.0, -1.0, -2.0),
        (1.0, 2.0, 0.0),
        (3.0, 4.0, 5.0),
        (0.5, 0.5, -1.0),
        (2.0, 3.0, 2.5),
    ],
)
def test_tabulated_3d_reproduces_linear_field(x: float, y: float, z: float) -> None:
    expected = x + 2.0 * y - 0.5 * z
    assert compiled_table("clamp").evaluate(x1=x, x2=y, x3=z) == [expected]


def test_tabulated_3d_clamps_each_axis() -> None:
    table = compiled_table(expr.TableBounds.CLAMP)
    assert table.evaluate(x1=-5.0, x2=0.5, x3=-1.0) == [1.5]
    assert table.evaluate(x1=2.0, x2=8.0, x3=9.0) == [7.5]


@pytest.mark.parametrize(
    ("x", "y", "z"),
    [
        (-5.0, 0.5, -1.0),
        (2.0, 8.0, -1.0),
        (2.0, 3.0, 9.0),
    ],
)
def test_tabulated_3d_zeroes_if_any_axis_is_outside(
    x: float, y: float, z: float
) -> None:
    assert compiled_table("zero").evaluate(x1=x, x2=y, x3=z) == [0.0]


def test_tabulated_3d_rejects_ragged_values() -> None:
    graph = expr.ExprGraph()
    with pytest.raises(ValueError, match="must have shape"):
        expr.tabulated_3d(
            expr.variable("x1", graph),
            expr.variable("x2", graph),
            expr.variable("x3", graph),
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0]]],
            bounds="clamp",
        )


def test_tabulated_3d_serializes_to_backend_supported_nodes() -> None:
    payload = compiled_table("clamp").serialize_source(
        expr.SourceKind.RAW,
        dim=3,
        target=expr.ConservedField.ENERGY,
    )
    assert {node["op"] for node in payload["nodes"]} <= {
        "CONSTANT",
        "VARIABLE_X1",
        "VARIABLE_X2",
        "VARIABLE_X3",
        "SUBTRACT",
        "DIVIDE",
        "MULTIPLY",
        "ADD",
        "LT",
        "GT",
        "IF_THEN_ELSE",
    }
