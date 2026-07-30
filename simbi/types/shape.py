# =============================================================================
# shape.py
#
# signed-distance CSG shapes for immersed rigid boundaries. mirrors the rust
# `SdfExpr` (symbi-ib/src/sdf.rs): sphere / box primitives composed by union,
# intersect, complement, and translate. a Shape serializes to the json wire form
# `SdfExpr::from_json` reads. coordinates are the body-LOCAL frame; the backend
# translates the whole tree to the body position.
# usage:
#  s = Shape.sphere((0, 0, 0), 1.0).union(Shape.box((2, 0, 0), (0.5, 0.5, 0.5)))
#  wire = s.to_wire()
# =============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence


def _vec3(name: str, v: Sequence[float]) -> list[float]:
    t = [float(x) for x in v]
    if len(t) != 3:
        raise ValueError(f"{name} must have 3 components (x, y, z), got {len(t)}")
    return t


@dataclass(frozen=True)
class Shape:
    """an immutable signed-distance CSG node. build with the `sphere` / `box`
    factories and compose with `union` / `intersect` / `complement` /
    `translated`; `to_wire` emits the backend json."""

    # the serialized SdfExpr node — opaque; produced by the factories/combinators.
    wire: dict[str, Any]

    @staticmethod
    def sphere(center: Sequence[float], radius: float) -> "Shape":
        if radius <= 0.0:
            raise ValueError(f"sphere radius must be > 0, got {radius}")
        return Shape(
            {"kind": "sphere", "center": _vec3("center", center), "radius": float(radius)}
        )

    @staticmethod
    def box(center: Sequence[float], half_extents: Sequence[float]) -> "Shape":
        h = _vec3("half_extents", half_extents)
        if any(x <= 0.0 for x in h):
            raise ValueError(f"box half_extents must all be > 0, got {h}")
        return Shape({"kind": "box", "center": _vec3("center", center), "half_extents": h})

    @staticmethod
    def cylinder(
        center: Sequence[float], radius: float, half_height: float
    ) -> "Shape":
        """a finite cylinder aligned with the body-local z axis."""
        if radius <= 0.0:
            raise ValueError(f"cylinder radius must be > 0, got {radius}")
        if half_height <= 0.0:
            raise ValueError(
                f"cylinder half_height must be > 0, got {half_height}"
            )
        return Shape(
            {
                "kind": "cylinder",
                "center": _vec3("center", center),
                "radius": float(radius),
                "half_height": float(half_height),
            }
        )

    def union(self, other: "Shape") -> "Shape":
        """the region inside EITHER shape (min of signed distances)."""
        return Shape({"kind": "union", "a": self.wire, "b": other.wire})

    def intersect(self, other: "Shape") -> "Shape":
        """the region inside BOTH shapes (max of signed distances)."""
        return Shape({"kind": "intersect", "a": self.wire, "b": other.wire})

    def complement(self) -> "Shape":
        """inside becomes outside — the unbounded exterior (has no bounding ball)."""
        return Shape({"kind": "complement", "inner": self.wire})

    def translated(self, offset: Sequence[float]) -> "Shape":
        """shift the whole tree by `offset` in the body-local frame."""
        return Shape({"kind": "translated", "inner": self.wire, "offset": _vec3("offset", offset)})

    def rotated(self, rot: Sequence[Sequence[float]]) -> "Shape":
        """rotate the whole tree by the 3x3 row-major orientation matrix `rot` about the
        body-local origin. a world point maps into the shape's frame as R^T x."""
        m = [list(row) for row in rot]
        if len(m) != 3 or any(len(row) != 3 for row in m):
            raise ValueError(f"rotation must be a 3x3 matrix, got {[len(r) for r in m]}")
        return Shape(
            {"kind": "rotated", "inner": self.wire, "rot": [[float(x) for x in row] for row in m]}
        )

    def rotated_z(self, angle: float) -> "Shape":
        """rotate the shape by `angle` radians about the z axis — the in-plane spin of a 2D run."""
        c, s = math.cos(angle), math.sin(angle)
        return self.rotated([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_wire(self) -> dict[str, Any]:
        return self.wire

    @staticmethod
    def from_wire(wire: dict[str, Any]) -> "Shape":
        """reconstruct a Shape from its serialized wire (e.g. loaded from a checkpoint)."""
        return Shape(wire)

    def signed_distance(self, point: Sequence[float]) -> float:
        """the signed distance to the shape in its BODY-LOCAL frame: negative inside, positive
        outside, zero on the surface. mirrors the rust `SdfExpr::dist`. to evaluate at a body's pose,
        map the world point in first: `shape.signed_distance(R.T @ (x_world - position))`."""
        return _sdf_dist(self.wire, [float(point[0]), float(point[1]), float(point[2])])

    def contains(self, point: Sequence[float]) -> bool:
        """whether `point` (body-local frame) is inside the shape."""
        return self.signed_distance(point) <= 0.0


def _sdf_dist(node: dict[str, Any], x: list[float]) -> float:
    # the carrier-generic CSG distance, mirroring symbi-ib/src/sdf.rs::dist (min/max/affine + sqrt).
    kind = node["kind"]
    if kind == "sphere":
        c, r = node["center"], node["radius"]
        return math.sqrt(sum((x[a] - c[a]) ** 2 for a in range(3))) - r
    if kind == "box":
        c, h = node["center"], node["half_extents"]
        q = [abs(x[a] - c[a]) - h[a] for a in range(3)]
        outside = math.sqrt(sum(max(qi, 0.0) ** 2 for qi in q))
        return outside + min(max(q), 0.0)
    if kind == "cylinder":
        c = node["center"]
        radial = math.hypot(x[0] - c[0], x[1] - c[1]) - node["radius"]
        axial = abs(x[2] - c[2]) - node["half_height"]
        outside = math.hypot(max(radial, 0.0), max(axial, 0.0))
        return outside + min(max(radial, axial), 0.0)
    if kind == "union":
        return min(_sdf_dist(node["a"], x), _sdf_dist(node["b"], x))
    if kind == "intersect":
        return max(_sdf_dist(node["a"], x), _sdf_dist(node["b"], x))
    if kind == "complement":
        return -_sdf_dist(node["inner"], x)
    if kind == "translated":
        o = node["offset"]
        return _sdf_dist(node["inner"], [x[a] - o[a] for a in range(3)])
    if kind == "rotated":
        r = node["rot"]  # 3x3 row-major; map the point into the inner frame via R^T
        xr = [sum(r[j][i] * x[j] for j in range(3)) for i in range(3)]
        return _sdf_dist(node["inner"], xr)
    raise ValueError(f"unknown shape kind {kind!r}")


__all__ = ["Shape"]
