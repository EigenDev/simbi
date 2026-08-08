# =============================================================================
# panels.py
#
# angular layout for several fields on one curvilinear chart.
#
# a spherical run is usually evolved on a wedge -- one quadrant, one hemisphere
# -- and the rest of the circle carries no data. giving each field its own
# sector of that circle puts them side by side at a single epoch on a shared
# radial axis: two fields on a quarter wedge tile the upper half-plane, split
# down the polar axis.
#
# sector k is the wedge reflected about the polar axis for odd k and stepped
# out by a whole wedge every second panel, so neighbouring panels meet along a
# shared edge and a mirrored pair reads continuously across the axis.
#
# usage:
#   groups = group_by_field(fields)
#   panel = place_in_sector(field, index=1, count=len(groups))
# =============================================================================
import re
from typing import Sequence

import numpy as np

from ..types import CoordSystem, FieldData

# a 2d field is stored slowest-axis-first, (x2, x1); on a spherical chart that
# is (angle, radius), and the polar axes plots the angle
ANGULAR_AXIS = 0

# what a field's name picks up on the way to being drawn: the refinement level
# it was prepared from, and the polygon contract it was composed into
NAME_SUFFIXES = re.compile(r"(_polygons)?(_L\d+)?(_polygons)?$")


def base_field_name(name: str) -> str:
    """the quantity a field draws, without the level or contract it arrived in."""
    return NAME_SUFFIXES.sub("", name)


def group_by_field(fields: Sequence[FieldData]) -> list[list[FieldData]]:
    """fields grouped by base name, in order of first appearance.

    one group is one plotted quantity across every level it exists on, which
    is the unit that gets composed into a single artist and, on a curvilinear
    chart, occupies a single sector."""
    groups: dict[str, list[FieldData]] = {}
    for field in fields:
        groups.setdefault(base_field_name(field.name), []).append(field)
    return list(groups.values())


def is_sectorable(field: FieldData) -> bool:
    """whether this field is drawn on a chart that has a circle to divide."""
    return field.coord_system == CoordSystem.SPHERICAL and field.ndim == 2


def wedge_angle(field: FieldData) -> float:
    """the angular extent the run was evolved over."""
    angles = np.asarray(field.domain[ANGULAR_AXIS], dtype=float)
    return float(abs(angles[-1] - angles[0]))


def sector_transform(index: int, wedge: float) -> tuple[float, float]:
    """the (sign, offset) carrying a wedge angle into sector `index`.

    even sectors run in the sense the data was evolved in and odd sectors are
    reflected, so a pair meets along the polar axis rather than butting one
    field's outer edge against the other's inner edge."""
    sign = 1.0 if index % 2 == 0 else -1.0
    return sign, sign * (index // 2) * wedge


def place_in_sector(field: FieldData, index: int, count: int) -> FieldData:
    """the field with its angular vertices carried into sector `index` of `count`.

    the radial vertices and the values are untouched: this moves where the
    wedge is drawn, not what it holds."""
    wedge = wedge_angle(field)

    if count * wedge > 2.0 * np.pi + 1.0e-9:
        raise ValueError(
            f"{count} fields of {np.degrees(wedge):.1f} degrees each overrun the "
            f"circle; plot at most {int(2.0 * np.pi / wedge)} fields together on "
            "this wedge"
        )

    sign, offset = sector_transform(index, wedge)

    domain = list(field.domain)
    domain[ANGULAR_AXIS] = (
        sign * np.asarray(domain[ANGULAR_AXIS], dtype=float) + offset
    )

    return field.model_copy(update={"domain": domain})
