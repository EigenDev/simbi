# =============================================================================
# test_mesh_overlay.py
#
# the cell-edge overlay has to describe the mesh the solver actually used, and
# a moving mesh moves its vertices between checkpoints. two things separate a
# faithful overlay from a decorative one:
#
#   - the lines live in DATA coordinates, so on a spherical (polar) chart a
#     constant-radius edge is an arc at that radius and a constant-angle edge
#     is a ray. lines drawn in axes-fraction coordinates look plausible and
#     annotate nothing.
#   - the edges are rebuilt from each frame's vertices, and the view follows
#     them. a homologously expanding mesh leaves a frame-0 view within a few
#     checkpoints.
#
# both failures are invisible in a still frame of a static mesh, which is what
# makes them worth gating.
# =============================================================================
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from simbi.viz.components.mesh_overlay import (
    ARC_SAMPLES,
    edge_stride,
    mesh_segments,
    select_edges,
)
from simbi.viz.components.polygons import PolygonPlotComponent, PolygonPlotProps
from simbi.viz.components.quad import QuadPlotComponent, QuadPlotProps
from simbi.viz.config import FigureConfig, PlotConfig, VisualizationConfig
from simbi.viz.pipeline.transforms import _compose_polygons, prepare_figure
from simbi.viz.types import Bounds, CoordSystem, FieldData

# a wedge of a graded spherical mesh: the outermost cell is many times the
# width of the innermost, which exposes a decimated or mis-mapped overlay that
# uniform spacing would hide
THETA_EDGES = np.linspace(0.0, np.pi, 17)
RADIAL_EDGES = np.geomspace(1.0, 10.0, 33)


def spherical_field(scale_factor: float = 1.0, time: float = 0.0) -> FieldData:
    """a 2d spherical field whose radial vertices carry a moving-mesh scale
    factor, in storage order (x2, x1) = (angle, radius)."""
    values = np.random.default_rng(0).random(
        (THETA_EDGES.size - 1, RADIAL_EDGES.size - 1)
    )
    return FieldData(
        name="rho",
        values=values,
        domain=[THETA_EDGES, scale_factor * RADIAL_EDGES],
        axis_names=["x1", "x2"],
        time=time,
    )


def polar_quad(props: QuadPlotProps) -> tuple[plt.Axes, QuadPlotComponent]:
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    component = QuadPlotComponent(props)
    component.initialize(fig, ax)
    return ax, component


def render_frames(component, frames, style: FigureConfig) -> plt.Axes:
    """drive the production path: components draw, and the figure composes the
    view from what they report. the figure owns the view -- several components
    share one axes -- so a gate on it has to run through the figure."""
    config = VisualizationConfig(
        plot=PlotConfig(plot_type="multidim", fields=["rho"], ndim=2),
        figure=style,
    )
    figure = prepare_figure(
        config, projection="polar", coord_system=CoordSystem.SPHERICAL
    )
    component.initialize(figure.fig, figure.axes["main"])

    for frame in frames:
        figure._components = [(component, frame)]
        figure.render()

    return figure.axes["main"]


# --- edge geometry ----------------------------------------------------------


def test_cartesian_segments_are_one_line_per_edge() -> None:
    segments = mesh_segments([0.0, 1.0, 2.0], [0.0, 3.0], stride=1)

    assert len(segments) == 5
    assert all(seg.shape == (2, 2) for seg in segments)


def test_polar_constant_radius_edge_is_an_arc() -> None:
    """a two-point segment at constant radius is drawn as a chord across the
    wedge, cutting through cells it is meant to bound."""
    segments = mesh_segments(
        THETA_EDGES, RADIAL_EDGES, curved=True, stride=1
    )

    rays = [seg for seg in segments if seg.shape[0] == 2]
    arcs = [seg for seg in segments if seg.shape[0] == ARC_SAMPLES]

    assert len(rays) == THETA_EDGES.size
    assert len(arcs) == RADIAL_EDGES.size

    # each ray holds its angle and spans the full radial extent
    for ray in rays:
        assert ray[0, 0] == ray[1, 0]
        assert ray[:, 1].min() == pytest.approx(RADIAL_EDGES[0])
        assert ray[:, 1].max() == pytest.approx(RADIAL_EDGES[-1])

    # each arc holds its radius and sweeps the full angular extent
    for arc in arcs:
        assert np.ptp(arc[:, 1]) == 0.0
        assert arc[:, 0].min() == pytest.approx(THETA_EDGES[0])
        assert arc[:, 0].max() == pytest.approx(THETA_EDGES[-1])


def test_arc_chord_error_stays_subpixel() -> None:
    """the arc sampling sets how far the drawn edge departs from the true
    circle. at the sagitta of one sampled chord that error must stay below a
    pixel at the outermost radius, where it is largest."""
    half_angle = 0.5 * (THETA_EDGES[-1] - THETA_EDGES[0]) / (ARC_SAMPLES - 1)
    sagitta = RADIAL_EDGES[-1] * (1.0 - np.cos(half_angle))

    # a 6-inch axes at 300 dpi resolves the full diameter in ~1800 px
    pixel = 2.0 * RADIAL_EDGES[-1] / 1800.0
    assert sagitta < pixel


def test_decimation_keeps_the_domain_boundary() -> None:
    """dropping the outermost edge draws a grid that stops short of the field
    it annotates, and understates the extent of a graded mesh."""
    edges = np.geomspace(1.0, 100.0, 100)
    kept = select_edges(edges, stride=7)

    assert kept[0] == edges[0]
    assert kept[-1] == edges[-1]
    assert np.all(np.isin(kept, edges))


def test_stride_bounds_the_line_count() -> None:
    assert edge_stride(n_edges=10, max_lines=64) == 1
    assert edge_stride(n_edges=1025, max_lines=64) == 17
    assert select_edges(np.arange(1025.0), 17).size <= 64


def test_degenerate_mesh_draws_nothing() -> None:
    assert mesh_segments([1.0], [0.0, 1.0]) == []
    assert mesh_segments([0.0, 1.0], []) == []


def test_non_vertex_input_is_refused() -> None:
    with pytest.raises(ValueError, match="1d vertex arrays"):
        mesh_segments(np.zeros((2, 2)), [0.0, 1.0])


# --- the overlay on a moving mesh -------------------------------------------


def test_overlay_lives_in_data_coordinates() -> None:
    """axes-fraction lines track the viewport rather than the mesh, so they
    survive any change of vertices while describing none of it."""
    ax, component = polar_quad(QuadPlotProps(show_mesh_grid=True, mesh_stride=1))
    component.render(spherical_field(), FigureConfig())

    assert isinstance(component._mesh_edges, LineCollection)
    assert component._mesh_edges.get_transform() is ax.transData


def test_overlay_follows_the_moving_mesh() -> None:
    ax, component = polar_quad(QuadPlotProps(show_mesh_grid=True, mesh_stride=1))
    style = FigureConfig()

    component.render(spherical_field(scale_factor=1.0), style)
    component.render(spherical_field(scale_factor=3.0, time=1.0), style)

    radii = np.concatenate(
        [seg[:, 1] for seg in component._mesh_edges.get_segments()]
    )
    assert radii.max() == pytest.approx(3.0 * RADIAL_EDGES[-1])
    assert radii.min() == pytest.approx(3.0 * RADIAL_EDGES[0])


def test_view_follows_the_moving_mesh() -> None:
    ax = render_frames(
        QuadPlotComponent(QuadPlotProps()),
        [
            spherical_field(scale_factor=1.0),
            spherical_field(scale_factor=3.0, time=1.0),
        ],
        FigureConfig(),
    )

    assert ax.get_ylim() == pytest.approx(
        (3.0 * RADIAL_EDGES[0], 3.0 * RADIAL_EDGES[-1])
    )


def test_requested_limits_outrank_the_mesh() -> None:
    """a fixed frame is how an expansion is watched from outside; the mesh must
    not steal the view back."""
    ax = render_frames(
        QuadPlotComponent(QuadPlotProps()),
        [spherical_field(scale_factor=3.0)],
        FigureConfig(ylims=Bounds(min=2.0, max=5.0)),
    )

    assert ax.get_ylim() == pytest.approx((2.0, 5.0))
    # the mesh reaches well past the requested view, so the clamp is doing work
    assert 3.0 * RADIAL_EDGES[-1] > 5.0


def test_overlay_is_a_single_artist_at_any_resolution() -> None:
    """one artist per coordinate line costs thousands of artists per frame on a
    production mesh, and each one is torn down and rebuilt."""
    theta = np.linspace(0.0, np.pi, 513)
    radius = np.geomspace(1.0, 100.0, 1025)
    field = FieldData(
        name="rho",
        values=np.zeros((512, 1024)),
        domain=[theta, radius],
        time=0.0,
    )

    ax, component = polar_quad(QuadPlotProps(show_mesh_grid=True))
    component.render(field, FigureConfig())

    overlays = [
        art for art in ax.collections if isinstance(art, LineCollection)
    ]
    assert len(overlays) == 1
    assert len(overlays[0].get_segments()) <= 2 * QuadPlotProps().mesh_max_lines


def test_overlay_is_removed_when_switched_off() -> None:
    ax, component = polar_quad(QuadPlotProps(show_mesh_grid=True))
    style = FigureConfig()
    component.render(spherical_field(), style)

    component.props = QuadPlotProps(show_mesh_grid=False)
    component.render(spherical_field(time=1.0), style)

    assert component._mesh_edges is None
    assert not [
        art for art in ax.collections if isinstance(art, LineCollection)
    ]


# --- the field artist across a mesh move ------------------------------------


def test_moved_vertices_rebuild_the_field_artist() -> None:
    """a quadmesh owns its vertices: refilling one whose mesh has moved paints
    the new frame's values onto the old frame's geometry."""
    _, component = polar_quad(QuadPlotProps())
    style = FigureConfig()

    component.render(spherical_field(scale_factor=1.0), style)
    first = component._mesh
    component.render(spherical_field(scale_factor=1.0, time=1.0), style)
    assert component._mesh is first

    component.render(spherical_field(scale_factor=3.0, time=2.0), style)
    assert component._mesh is not first


def test_colorbar_is_repointed_at_the_rebuilt_artist() -> None:
    """the colorbar is bound to an artist, so once the mesh is rebuilt it
    describes a discarded one and stops tracking the data range."""
    from simbi.viz.formatting import FigureFormatter

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    component = QuadPlotComponent(QuadPlotProps())
    component.initialize(fig, ax)
    style = FigureConfig()

    result = component.render(spherical_field(scale_factor=1.0), style)
    formatter = FigureFormatter(style)
    formatter.refresh_colorbars(fig, ax, [result], None)
    colorbar = getattr(ax, "_simbi_colorbars")[0]

    moved = component.render(spherical_field(scale_factor=3.0, time=1.0), style)
    assert moved.mappable is not result.mappable

    formatter.refresh_colorbars(fig, ax, [moved], None)
    assert colorbar.mappable is moved.mappable


def test_mesh_of_a_different_size_is_accepted() -> None:
    """comparing vertex arrays of two different lengths is undefined; a level
    change or a resized mesh must rebuild rather than raise."""
    _, component = polar_quad(QuadPlotProps())
    style = FigureConfig()
    component.render(spherical_field(), style)

    coarse = FieldData(
        name="rho",
        values=np.zeros((8, 16)),
        domain=[np.linspace(0.0, np.pi, 9), np.geomspace(1.0, 10.0, 17)],
        time=1.0,
    )
    component.render(coarse, style)

    assert component._mesh.get_array().size == 8 * 16


# --- the polygon chart ------------------------------------------------------


def test_polygon_vertices_are_mapped_onto_the_polar_chart() -> None:
    """the mesh stores (radius, angle); a polar axes reads a vertex as
    (angle, radius). handing over the mesh order transposes the whole plot into
    a chart where radius is an angle."""
    field = spherical_field()
    polygons = _compose_polygons([field])

    component = PolygonPlotComponent(PolygonPlotProps())
    ax = render_frames(component, [polygons], FigureConfig())

    drawn = np.asarray(
        [path.vertices for path in component._poly_collection.get_paths()]
    )
    assert drawn[..., 0].max() <= THETA_EDGES[-1]
    assert drawn[..., 1].max() == pytest.approx(RADIAL_EDGES[-1])
    assert ax.get_ylim() == pytest.approx((RADIAL_EDGES[0], RADIAL_EDGES[-1]))


def test_polygon_vertices_are_untouched_on_a_cartesian_chart() -> None:
    field = spherical_field()
    polygons = _compose_polygons([field])

    fig, ax = plt.subplots()
    component = PolygonPlotComponent(PolygonPlotProps())
    component.initialize(fig, ax)
    component.render(polygons, FigureConfig())

    drawn = np.asarray(
        [path.vertices for path in component._poly_collection.get_paths()]
    )
    assert drawn[..., 0].max() == pytest.approx(RADIAL_EDGES[-1])
    assert drawn[..., 1].max() == pytest.approx(THETA_EDGES[-1])


# --- the colorbar on a spherical chart ---------------------------------------


def spherical_colorbar(component, data) -> tuple[plt.Axes, object]:
    from simbi.viz.formatting import FigureFormatter

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    component.initialize(fig, ax)
    style = FigureConfig()
    result = component.render(data, style)

    FigureFormatter(style).apply_figure_formatting(fig, ax, [result], data)
    return ax, getattr(ax, "_simbi_colorbars", {}).get(0)


def test_polygon_chart_carries_a_colorbar() -> None:
    """placing the bar takes the wedge's angular extent. taking it from the
    field's own domain assumes vertex arrays, where a polygon render carries
    polygon geometry -- and the failure costs the whole colorbar, position and
    bar alike."""
    ax, colorbar = spherical_colorbar(
        PolygonPlotComponent(PolygonPlotProps()),
        _compose_polygons([spherical_field()]),
    )

    assert colorbar is not None


def wedge_box(ax: plt.Axes) -> tuple[float, float, float, float]:
    """the drawn sector's bounds in figure coordinates."""
    from simbi.viz.formatting import wedge_extent

    left, right, bottom, top = wedge_extent(ax)
    pos = ax.get_position()
    return (
        pos.x0 + left * pos.width,
        pos.x0 + right * pos.width,
        pos.y0 + bottom * pos.height,
        pos.y0 + top * pos.height,
    )


@pytest.mark.parametrize("wedge", [0.5 * np.pi, np.pi])
def test_a_spherical_colorbar_never_covers_the_field(wedge: float) -> None:
    """a bar over the wedge hides the field it is describing. where there is
    room for one changes with the opening angle -- a half-plane leaves a strip
    below its flat edge that a quarter does not -- so both are checked, and
    through the figure, which is what fixes the view the placement reads."""
    angles = np.linspace(0.0, wedge, THETA_EDGES.size)
    field = spherical_field().model_copy(
        update={"domain": [angles, RADIAL_EDGES]}
    )

    ax = render_frames(
        QuadPlotComponent(QuadPlotProps()), [field], FigureConfig()
    )
    colorbar = getattr(ax, "_simbi_colorbars", {}).get(0)
    assert colorbar is not None

    # the drawn wedge really does span the requested angle, so the placement
    # under test saw the case it is named for
    assert abs(np.ptp(ax.get_xlim()) - wedge) < 1.0e-9

    left, right, bottom, top = wedge_box(ax)
    bar = colorbar.ax.get_position()
    assert not (
        bar.x0 < right and bar.x1 > left and bar.y0 < top and bar.y1 > bottom
    )
