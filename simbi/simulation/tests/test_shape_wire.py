# =============================================================================
# test_shape_wire.py
#
# the immersed-body shape wire: `Shape.to_wire` must emit the json schema the
# rust `SdfExpr::from_json` (symbi-ib/src/sdf.rs) consumes. the CSG tree of a
# sphere unioned with a box is pinned byte-for-byte against the exact wire the
# rust `from_json_parses_csg_and_equals_native` test parses, so the two sides
# cannot drift.
# =============================================================================
import json

import pytest

from simbi.types.shape import Shape


def test_sphere_box_union_wire_matches_the_rust_parser() -> None:
    s = Shape.sphere((0.0, 0.0, 0.0), 1.0).union(
        Shape.box((2.0, 0.0, 0.0), (0.5, 0.5, 0.5))
    )
    assert s.to_wire() == {
        "kind": "union",
        "a": {"kind": "sphere", "center": [0.0, 0.0, 0.0], "radius": 1.0},
        "b": {
            "kind": "box",
            "center": [2.0, 0.0, 0.0],
            "half_extents": [0.5, 0.5, 0.5],
        },
    }
    # crosses the boundary as json (the config exec-dict convention).
    json.dumps(s.to_wire())


def test_translate_complement_intersect_compose() -> None:
    s = (
        Shape.sphere((0.0, 0.0, 0.0), 1.0)
        .intersect(Shape.box((0.0, 0.0, 0.0), (2.0, 2.0, 0.3)))
        .translated((1.0, 0.0, 0.0))
    )
    w = s.to_wire()
    assert w["kind"] == "translated"
    assert w["offset"] == [1.0, 0.0, 0.0]
    assert w["inner"]["kind"] == "intersect"
    hollow = Shape.sphere((0.0, 0.0, 0.0), 1.0).complement().to_wire()
    assert hollow["kind"] == "complement"


def test_degenerate_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match="radius must be > 0"):
        Shape.sphere((0.0, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="half_extents must all be > 0"):
        Shape.box((0.0, 0.0, 0.0), (0.5, -1.0, 0.5))
    with pytest.raises(ValueError, match="3 components"):
        Shape.sphere((0.0, 0.0), 1.0)
