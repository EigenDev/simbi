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


def test_rotated_z_emits_the_orientation_matrix() -> None:
    import math

    s = Shape.box((0.0, 0.0, 0.0), (0.5, 0.2, 0.3)).rotated_z(math.pi / 2)
    w = s.to_wire()
    assert w["kind"] == "rotated"
    assert w["inner"]["kind"] == "box"
    # 90 deg about z: [[0,-1,0],[1,0,0],[0,0,1]] (within float tolerance).
    rot = w["rot"]
    assert abs(rot[0][0]) < 1e-12 and abs(rot[0][1] + 1.0) < 1e-12
    assert abs(rot[1][0] - 1.0) < 1e-12 and abs(rot[1][1]) < 1e-12
    assert rot[2] == [0.0, 0.0, 1.0]
    import json

    json.dumps(w)


def test_rotated_rejects_non_3x3() -> None:
    with pytest.raises(ValueError, match="3x3 matrix"):
        Shape.sphere((0.0, 0.0, 0.0), 1.0).rotated([[1.0, 0.0], [0.0, 1.0]])


def test_signed_distance_matches_analytic() -> None:
    import math

    sphere = Shape.sphere((0.0, 0.0, 0.0), 1.0)
    assert abs(sphere.signed_distance((0.0, 0.0, 0.0)) + 1.0) < 1e-12  # center: -radius
    assert abs(sphere.signed_distance((2.0, 0.0, 0.0)) - 1.0) < 1e-12  # outside by 1
    assert sphere.contains((0.5, 0.0, 0.0)) and not sphere.contains((1.5, 0.0, 0.0))

    box = Shape.box((0.0, 0.0, 0.0), (0.5, 0.3, 0.2))
    assert abs(box.signed_distance((0.0, 0.0, 0.0)) + 0.2) < 1e-12  # inside, nearest face at z=0.2
    assert abs(box.signed_distance((0.6, 0.0, 0.0)) - 0.1) < 1e-12  # outside in x by 0.1

    cylinder = Shape.cylinder((0.0, 0.0, 1.0), radius=0.5, half_height=2.0)
    assert cylinder.signed_distance((0.0, 0.0, 1.0)) == pytest.approx(-0.5)
    assert cylinder.signed_distance((0.75, 0.0, 1.0)) == pytest.approx(0.25)
    assert cylinder.signed_distance((0.0, 0.0, 3.25)) == pytest.approx(0.25)
    assert cylinder.signed_distance((0.8, 0.0, 3.4)) == pytest.approx(0.5)

    union = Shape.sphere((0.0, 0.0, 0.0), 1.0).union(Shape.box((3.0, 0.0, 0.0), (0.5, 0.5, 0.5)))
    assert union.contains((0.0, 0.0, 0.0)) and union.contains((3.0, 0.0, 0.0))
    assert not union.contains((1.6, 0.0, 0.0))

    # a box rotated 90deg about z: its 0.5 x-extent maps onto y, the 0.2 onto x.
    rot = Shape.box((0.0, 0.0, 0.0), (0.5, 0.2, 0.3)).rotated_z(math.pi / 2)
    assert rot.contains((0.0, 0.4, 0.0))  # inside the long (now-y) extent
    assert not rot.contains((0.4, 0.0, 0.0))  # outside the short (now-x) extent

    # from_wire round-trips the evaluation.
    assert Shape.from_wire(box.to_wire()).signed_distance((0.6, 0.0, 0.0)) == box.signed_distance(
        (0.6, 0.0, 0.0)
    )


def test_degenerate_dimensions_rejected() -> None:
    with pytest.raises(ValueError, match="radius must be > 0"):
        Shape.sphere((0.0, 0.0, 0.0), 0.0)
    with pytest.raises(ValueError, match="half_extents must all be > 0"):
        Shape.box((0.0, 0.0, 0.0), (0.5, -1.0, 0.5))
    with pytest.raises(ValueError, match="radius must be > 0"):
        Shape.cylinder((0.0, 0.0, 0.0), 0.0, 1.0)
    with pytest.raises(ValueError, match="half_height must be > 0"):
        Shape.cylinder((0.0, 0.0, 0.0), 1.0, 0.0)
    with pytest.raises(ValueError, match="3 components"):
        Shape.sphere((0.0, 0.0), 1.0)
