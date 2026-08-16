# =============================================================================
# test_polygon_compose.py
#
# a level hierarchy is drawn one quadrilateral per cell, because a quadmesh is
# a single logically-rectangular lattice holding cells of one size. composing
# it has to get two things right, and a finished picture hides both:
#
#   - every leaf cell appears exactly once, carrying its own value. a cell
#     drawn twice is a coarse value painted over a fine one, and the plot looks
#     finished either way.
#   - a cell belongs to the finest level that covers its center. testing the
#     corners instead drops the coarse cells that merely abut a refined patch,
#     leaving a hairline gap around every refinement boundary.
#
# the composition is also on the animation path once per frame, so its cost is
# multiplied by the frame count.
# =============================================================================
import numpy as np
import pytest

from simbi.viz.pipeline.transforms import _compose_polygons
from simbi.viz.types import CoordSystem, FieldData, PolygonData


def level(
    name: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    shape: tuple[int, int],
    seed: int = 0,
) -> FieldData:
    """one refinement level over the given extent; `shape` is (ny, nx)."""
    ny, nx = shape
    return FieldData(
        name=name,
        values=np.random.default_rng(seed).random((ny, nx)),
        # storage order is (x2, x1) = (y, x)
        domain=[
            np.linspace(*y_range, ny + 1),
            np.linspace(*x_range, nx + 1),
        ],
        coord_system=CoordSystem.CARTESIAN,
        axis_names=["x1", "x2"],
        time=0.5,
    )


def centers(patches: np.ndarray) -> np.ndarray:
    return patches.mean(axis=1)


# --- the unigrid case -------------------------------------------------------


def test_every_cell_becomes_one_quadrilateral() -> None:
    field = level("rho", (0.0, 1.0), (0.0, 2.0), (5, 7))

    drawn = _compose_polygons([field])

    assert isinstance(drawn, PolygonData)
    assert drawn.patches.shape == (35, 4, 2)
    # values follow the cells in row-major order, the order the mesh stores them
    assert np.array_equal(drawn.values, field.values.ravel())


def test_the_corners_are_the_cell_edges() -> None:
    """a patch wound in the wrong order draws a bow tie, and one built from the
    wrong axis puts a cell at another cell's coordinates."""
    field = level("rho", (0.0, 2.0), (0.0, 1.0), (1, 2))

    patches = _compose_polygons([field]).patches

    assert np.allclose(
        patches[0], [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    assert np.allclose(
        patches[1], [[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]
    )


def test_a_non_square_level_keeps_its_axes_apart() -> None:
    """with nx != ny a swapped unpack indexes the values off the end of an axis
    or silently transposes the picture; a square level hides it."""
    field = level("rho", (0.0, 3.0), (0.0, 1.0), (4, 9))

    drawn = _compose_polygons([field])

    assert drawn.patches[..., 0].max() == pytest.approx(3.0)
    assert drawn.patches[..., 1].max() == pytest.approx(1.0)
    assert drawn.values.size == 36


# --- a level hierarchy ------------------------------------------------------


def test_a_refined_patch_replaces_the_coarse_cells_beneath_it() -> None:
    coarse = level("rho", (0.0, 1.0), (0.0, 1.0), (8, 8), seed=1)
    fine = level("rho_L1", (0.25, 0.75), (0.25, 0.75), (8, 8), seed=2)

    drawn = _compose_polygons([coarse, fine])

    covered = np.all(
        (centers(drawn.patches) > 0.25) & (centers(drawn.patches) < 0.75),
        axis=1,
    )
    # inside the refined box only the fine cells survive, and they are smaller
    widths = drawn.patches[..., 0].max(axis=1) - drawn.patches[..., 0].min(
        axis=1
    )
    assert set(np.round(widths[covered], 6)) == {round(0.5 / 8, 6)}
    assert set(np.round(widths[~covered], 6)) == {round(1.0 / 8, 6)}


def test_no_cell_is_drawn_twice() -> None:
    """an overlap is a coarse value painted over a fine one -- the plot looks
    finished, and reports the wrong number where it matters most."""
    coarse = level("rho", (0.0, 1.0), (0.0, 1.0), (10, 12), seed=1)
    fine = level("rho_L1", (0.2, 0.8), (0.2, 0.8), (12, 12), seed=2)
    finer = level("rho_L2", (0.4, 0.6), (0.4, 0.6), (8, 8), seed=3)

    drawn = _compose_polygons([coarse, fine, finer])

    seen = {tuple(np.round(point, 9)) for point in centers(drawn.patches)}
    assert len(seen) == drawn.values.size


def test_a_cell_belongs_to_the_level_that_holds_its_centre() -> None:
    """the coarse cells that merely touch the refined box must survive: judged
    by their corners they would all be dropped, leaving a gap around it."""
    coarse = level("rho", (0.0, 1.0), (0.0, 1.0), (4, 4))
    # exactly one coarse cell wide, aligned to the coarse edges
    fine = level("rho_L1", (0.25, 0.5), (0.25, 0.5), (4, 4))

    drawn = _compose_polygons([coarse, fine])

    # 16 coarse cells less the single one covered, plus the 16 fine ones
    assert drawn.values.size == 16 - 1 + 16


def test_level_bounds_run_coarsest_first() -> None:
    coarse = level("rho", (0.0, 1.0), (0.0, 2.0), (4, 4))
    fine = level("rho_L1", (0.25, 0.75), (0.5, 1.5), (4, 4))

    bounds = _compose_polygons([coarse, fine]).level_bounds

    assert bounds is not None
    assert np.allclose(bounds[0], (0.0, 1.0, 0.0, 2.0))
    assert np.allclose(bounds[1], (0.25, 0.75, 0.5, 1.5))


def test_a_unigrid_field_reports_no_level_bounds() -> None:
    """there is no refinement boundary to outline, and drawing one would put a
    box around the whole domain."""
    assert _compose_polygons([level("rho", (0, 1), (0, 1), (4, 4))]).level_bounds is None


# --- the type ---------------------------------------------------------------


def test_the_composed_set_keeps_the_quantity_name() -> None:
    """the name identifies the quantity for labels and for per-field styling;
    a suffix naming the render mode belongs to the type instead."""
    assert _compose_polygons([level("rho", (0, 1), (0, 1), (2, 2))]).name == "rho"


def test_patches_that_are_not_quadrilaterals_are_refused() -> None:
    """the shape is what separates cell corners from a coordinate array, and a
    mistake there is silent: (n, 2) reads as n cells of two corners."""
    with pytest.raises(ValueError, match=r"\(n_cells, 4, 2\)"):
        PolygonData(
            name="rho", patches=np.zeros((5, 2)), values=np.zeros(5)
        )


# --- polygons through the animation path ------------------------------------
#
# a still and a movie compose the same way, but only the movie packs the result
# back into a PlotData and matches it to the component that drew the previous
# frame. a type that a still accepts and a movie rejects fails every frame of
# a refined animation, which is the only kind of animation refined data has.


def polygon_plot_data() -> "PlotData":
    from simbi.viz.types import PlotData

    drawn = _compose_polygons([level("rho", (0.0, 1.0), (0.0, 1.0), (4, 4))])
    return PlotData(
        fields=[drawn], time=0.0, dimensions=drawn.ndim
    )


def test_a_frame_of_polygons_packs_into_plot_data() -> None:
    from simbi.viz.types import PolygonData

    assert isinstance(polygon_plot_data().fields[0], PolygonData)


def test_a_polygon_component_is_matched_to_its_frame() -> None:
    """components are re-matched to each frame by name; a polygon set that
    fails the match is handed the whole PlotData and the component rejects it."""
    import matplotlib

    matplotlib.use("Agg")
    from simbi.viz.components.polygons import (
        PolygonPlotComponent,
        PolygonPlotProps,
    )
    from simbi.viz.config import PlotConfig, VisualizationConfig
    from simbi.viz.pipeline.transforms import prepare_figure

    figure = prepare_figure(
        VisualizationConfig(
            plot=PlotConfig(plot_type="multidim", fields=["rho"], ndim=2)
        ),
        coord_system=CoordSystem.CARTESIAN,
    )
    component = PolygonPlotComponent(PolygonPlotProps())
    component.initialize(figure.fig, figure.axes["main"])

    first = polygon_plot_data()
    figure.add_component(component, first.fields[0])

    # the second frame is a different object carrying the same name
    figure.draw_frame(polygon_plot_data(), full=True)

    assert component._poly_collection is not None
    assert len(component._poly_collection.get_paths()) == 16
