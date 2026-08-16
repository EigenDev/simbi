# =============================================================================
# test_panels.py
#
# several quantities drawn on one spherical chart divide its circle between
# them, one sector each: the mirrored figure that carries density on one half
# and four-velocity on the other, at one epoch, on a shared radial axis.
#
# what has to hold for that figure to mean anything:
#   - the sectors are disjoint. two fields laid on the same angles paint over
#     one another, and the survivor looks like a finished plot.
#   - the reflection touches the angles only. a sector that rescaled the radii
#     or reordered the values would put one field's shock at the other's radius.
#   - the view holds every sector, and each quantity keeps its own scale and
#     its own colorbar. a shared scale reads two quantities as one field.
# =============================================================================
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simbi.viz.api import PANEL_CMAPS, _panel_cmap, _panel_props
from simbi.viz.components.quad import QuadPlotComponent, QuadPlotProps
from simbi.viz.config import (
    FigureConfig,
    PlotConfig,
    RefinementConfig,
    VisualizationConfig,
)
from simbi.viz.formatting import (
    bar_cell,
    colorbar_side,
    find_mappables,
    is_hemispherical,
    wedge_extent,
)
from simbi.viz.pipeline.panels import (
    base_field_name,
    group_by_field,
    place_in_sector,
    sector_transform,
)
from simbi.viz.pipeline.transforms import compose_fields_for_render, prepare_figure
from simbi.viz.types import (
    CoordSystem,
    FieldData,
    PolygonData,
    RenderResult,
)

WEDGE = 0.5 * np.pi
THETA_EDGES = np.linspace(0.0, WEDGE, 9)
RADIAL_EDGES = np.geomspace(1.0, 10.0, 17)


def wedge_field(
    name: str, coord_system: CoordSystem = CoordSystem.SPHERICAL
) -> FieldData:
    values = np.arange(
        (THETA_EDGES.size - 1) * (RADIAL_EDGES.size - 1), dtype=float
    ).reshape(THETA_EDGES.size - 1, RADIAL_EDGES.size - 1)
    return FieldData(
        name=name,
        values=values,
        domain=[THETA_EDGES, RADIAL_EDGES],
        coord_system=coord_system,
        time=0.0,
    )


def config(render_mode: str = "pcolormesh") -> VisualizationConfig:
    return VisualizationConfig(
        plot=PlotConfig(plot_type="multidim", fields=["D", "u"], ndim=2),
        refinement=RefinementConfig(render_mode=render_mode),
    )


def angular_span(field: FieldData) -> tuple[float, float]:
    angles = np.asarray(field.domain[0], dtype=float)
    return float(angles.min()), float(angles.max())


# --- sector geometry --------------------------------------------------------


def test_sectors_alternate_sense_and_step_by_a_wedge() -> None:
    assert sector_transform(0, WEDGE) == (1.0, 0.0)
    assert sector_transform(1, WEDGE) == (-1.0, -0.0)
    assert sector_transform(2, WEDGE) == (1.0, WEDGE)
    assert sector_transform(3, WEDGE) == (-1.0, -WEDGE)


def test_a_mirrored_pair_tiles_the_half_plane() -> None:
    """the pair meets along the polar axis and covers the hemisphere exactly:
    a gap leaves a blank slice, an overlap paints one field over the other."""
    left = angular_span(place_in_sector(wedge_field("u"), 1, 2))
    right = angular_span(place_in_sector(wedge_field("D"), 0, 2))

    assert right == pytest.approx((0.0, WEDGE))
    assert left == pytest.approx((-WEDGE, 0.0))


def test_four_sectors_tile_the_circle() -> None:
    spans = [
        angular_span(place_in_sector(wedge_field(f"f{kk}"), kk, 4))
        for kk in range(4)
    ]
    edges = sorted(edge for span in spans for edge in span)

    assert edges == pytest.approx(
        [-2 * WEDGE, -WEDGE, -WEDGE, 0.0, 0.0, WEDGE, WEDGE, 2 * WEDGE]
    )


def test_a_sector_moves_angles_only() -> None:
    """the radial vertices and the values place the physics; only where the
    wedge is drawn changes."""
    original = wedge_field("u")
    placed = place_in_sector(original, 1, 2)

    assert np.allclose(placed.domain[1], original.domain[1])
    assert np.array_equal(placed.values, original.values)
    assert np.allclose(placed.domain[0], -np.asarray(original.domain[0]))


def test_fields_that_overrun_the_circle_are_refused() -> None:
    """three half-planes overrun the circle; wrapping them would stack fields
    on top of each other in the overlap."""
    half_plane = wedge_field("D").model_copy(
        update={"domain": [np.linspace(0.0, np.pi, 9), RADIAL_EDGES]}
    )

    with pytest.raises(ValueError, match="overrun the circle"):
        place_in_sector(half_plane, 2, 3)


# --- grouping ---------------------------------------------------------------


def test_levels_of_one_quantity_stay_together() -> None:
    """a refinement level is another view of one quantity, so it shares that
    quantity's panel: split across sectors the levels of one field would be
    drawn apart."""
    fields = [
        wedge_field("rho"),
        wedge_field("rho_L1"),
        wedge_field("pre"),
    ]

    groups = group_by_field(fields)

    assert [[f.name for f in group] for group in groups] == [
        ["rho", "rho_L1"],
        ["pre"],
    ]
    assert base_field_name("rho_L12") == "rho"


# --- composition ------------------------------------------------------------


def test_two_spherical_fields_land_in_disjoint_sectors() -> None:
    composed = compose_fields_for_render(
        [wedge_field("D"), wedge_field("u")], config()
    )

    assert len(composed) == 2
    assert angular_span(composed[0]) == pytest.approx((0.0, WEDGE))
    assert angular_span(composed[1]) == pytest.approx((-WEDGE, 0.0))


def test_two_fields_compose_to_separate_polygon_sets() -> None:
    """composition merges a level hierarchy into one artist. handed two
    quantities at once it reads the second as a refinement of the first and
    drops most of it on the floor."""
    composed = compose_fields_for_render(
        [wedge_field("D"), wedge_field("u")], config(render_mode="polygons")
    )

    assert [f.name for f in composed] == ["D", "u"]
    cells = (THETA_EDGES.size - 1) * (RADIAL_EDGES.size - 1)
    assert all(isinstance(f, PolygonData) for f in composed)
    assert all(f.values.size == cells for f in composed)


def test_a_single_field_keeps_the_angles_it_was_evolved_on() -> None:
    composed = compose_fields_for_render([wedge_field("D")], config())

    assert angular_span(composed[0]) == pytest.approx((0.0, WEDGE))


def test_a_cartesian_chart_has_no_circle_to_divide() -> None:
    composed = compose_fields_for_render(
        [
            wedge_field("D", CoordSystem.CARTESIAN),
            wedge_field("u", CoordSystem.CARTESIAN),
        ],
        config(),
    )

    assert all(
        angular_span(field) == pytest.approx((0.0, WEDGE))
        for field in composed
    )


# --- the drawn figure -------------------------------------------------------


def render_panels(fields, style: FigureConfig = FigureConfig()):
    """draw composed panels through the production figure path."""
    figure = prepare_figure(
        VisualizationConfig(
            plot=PlotConfig(plot_type="multidim", fields=["D", "u"], ndim=2),
            figure=style,
        ),
        projection="polar",
        coord_system=CoordSystem.SPHERICAL,
    )
    for index, field in enumerate(fields):
        component = QuadPlotComponent(
            QuadPlotProps(cmap=_panel_cmap("viridis", index))
        )
        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(component, field)

    figure.render()
    return figure


def test_the_view_holds_every_sector() -> None:
    """each component reports only its own wedge; a view taken from one of them
    crops the panel beside it."""
    panels = compose_fields_for_render(
        [wedge_field("D"), wedge_field("u")], config()
    )
    figure = render_panels(panels)

    assert figure.axes["main"].get_xlim() == pytest.approx((-WEDGE, WEDGE))


def test_each_panel_carries_its_own_colorbar() -> None:
    panels = compose_fields_for_render(
        [wedge_field("D"), wedge_field("u")], config()
    )
    figure = render_panels(panels)

    bars = getattr(figure.axes["main"], "_simbi_colorbars")
    assert len(bars) == 2
    # the label sits on whichever axis the bar runs along
    assert [
        bar.ax.get_xlabel() or bar.ax.get_ylabel() for bar in bars.values()
    ] == ["$D / D_0$", r"$\Gamma \beta$"]


def test_a_mirrored_pair_puts_each_scale_on_its_own_side() -> None:
    assert colorbar_side(0, 2) == "right"
    assert colorbar_side(1, 2) == "left"
    # a lone quantity keeps the conventional right-hand bar
    assert colorbar_side(0, 1) == "right"


def test_a_vector_overlay_does_not_claim_a_scale() -> None:
    """a quiver is colormapped but reads off the field beneath it; giving it a
    bar of its own would label the panel twice."""
    field = RenderResult(
        artists={"mesh": object()}, mappable=object(), colorbar_label="D"
    )
    overlay = RenderResult(artists={"quiver": object()}, is_vector=True)

    entries = find_mappables([field, overlay])

    assert [label for _, label in entries] == ["D"]


def test_panels_do_not_share_a_colormap() -> None:
    """two quantities in one colormap read as one field with a discontinuity
    down the middle."""
    chosen = [_panel_cmap("viridis", index) for index in range(4)]

    assert chosen[0] == "viridis"
    assert len(set(chosen)) == 4
    assert set(chosen) <= set(PANEL_CMAPS)


def test_a_panel_takes_the_styling_asked_of_its_own_quantity() -> None:
    """two quantities on one chart rarely share a scaling: a four-velocity that
    reaches zero takes a linear scale where its neighboring density wants a
    log one."""
    shared = QuadPlotProps(cmap="inferno", log_scale=True)
    only_u = QuadPlotProps(log_scale=False)

    props = _panel_props(
        {"quad": shared, "quad:u": only_u},
        "quad",
        QuadPlotProps,
        "u_L0",
        index=1,
        npanels=2,
    )

    # the per-field override wins, the panel default colormap still applies,
    # and every prop the override leaves out is inherited
    assert props.log_scale is False
    assert props.cmap == _panel_cmap("inferno", 1)
    assert props.alpha == shared.alpha


# --- where the bars go ------------------------------------------------------


def test_a_hemisphere_carries_its_bars_in_the_strip_below_it() -> None:
    """two quarter panels tile a half-plane, which fills the axes box edge to
    edge and only half its height. the room is underneath; a bar squeezed
    alongside is tall and thin and strands a quarter of the figure empty."""
    panels = compose_fields_for_render(
        [wedge_field("D"), wedge_field("u")], config()
    )
    ax = render_panels(panels).axes["main"]

    assert np.ptp(ax.get_xlim()) == pytest.approx(np.pi)
    assert is_hemispherical(ax)

    bars = getattr(ax, "_simbi_colorbars")
    assert all(bar.orientation == "horizontal" for bar in bars.values())

    # the strip is measured in axes fractions and the bars report figure
    # coordinates, so compare them in one frame
    _, _, wedge_bottom, _ = wedge_extent(ax)
    floor, ceiling = (
        ax.get_figure()
        .transFigure.inverted()
        .transform(ax.transAxes.transform([(0.0, 0.0), (0.0, wedge_bottom)]))[
            :, 1
        ]
    )

    for bar in bars.values():
        box = bar.ax.get_position()
        # clear of the field above and of the axes floor below, where the tick
        # labels go
        assert box.y1 <= ceiling
        assert box.y0 > floor


def test_a_narrower_wedge_keeps_its_bar_beside_the_chart() -> None:
    """a quarter wedge is centred in the box with no clear strip under it: a
    bar placed below would sit on the field."""
    panels = compose_fields_for_render([wedge_field("D")], config())
    ax = render_panels(panels).axes["main"]

    assert np.ptp(ax.get_xlim()) == pytest.approx(WEDGE)
    assert not is_hemispherical(ax)
    assert all(
        bar.orientation == "vertical"
        for bar in getattr(ax, "_simbi_colorbars").values()
    )


def test_a_full_circle_keeps_its_bars_beside_the_chart() -> None:
    quadrants = [wedge_field(name) for name in ("D", "u", "p", "chi")]
    ax = render_panels(compose_fields_for_render(quadrants, config())).axes[
        "main"
    ]

    assert np.ptp(ax.get_xlim()) == pytest.approx(2.0 * np.pi)
    assert not is_hemispherical(ax)
    assert all(
        bar.orientation == "vertical"
        for bar in getattr(ax, "_simbi_colorbars").values()
    )


def test_the_bars_read_in_the_order_the_panels_do() -> None:
    """the leftmost bar describes the leftmost panel; swapped, every reading of
    the figure is wrong and nothing looks amiss."""
    # sector 1 is the reflected panel, drawn on the left
    assert bar_cell(slot=1, total=2) == 0
    assert bar_cell(slot=0, total=2) == 1
    assert bar_cell(slot=0, total=1) == 0
