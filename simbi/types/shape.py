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

    def to_wire(self) -> dict[str, Any]:
        return self.wire


__all__ = ["Shape"]
