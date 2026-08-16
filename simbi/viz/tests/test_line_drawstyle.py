# =============================================================================
# test_line_drawstyle.py
#
# a finite-volume solution is piecewise constant across each cell, and `drawstyle="steps"`
# is the mark that says so. the geometry is what it has to get right: the marks belong at
# the cell edges, which on a graded mesh sit apart from the cell centers.
#
# a line through centers is silently smooth -- it interpolates across a cell that may be
# twice its neighbor's width, and the change in spacing at a refinement boundary vanishes.
# that is invisible in the output, which is what makes it worth a gate.
# =============================================================================
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch

from simbi.viz.components.line import LinePlotComponent, LinePlotProps
from simbi.viz.config import FigureConfig
from simbi.viz.types import FieldData


def graded_field(ncells: int = 8) -> FieldData:
    """a logarithmically graded mesh: the widest cell is many times the narrowest, which is
    what separates edge-drawn marks from centre-drawn ones."""
    edges = np.geomspace(1.0, 100.0, ncells + 1)
    values = np.linspace(1.0, 2.0, ncells)
    return FieldData(
        name="rho", values=values, domain=[edges], spacing_types=["log"]
    )


def render(drawstyle: str, data: FieldData):
    fig, ax = plt.subplots()
    component = LinePlotComponent(LinePlotProps(drawstyle=drawstyle))
    component.initialize(fig, ax)
    result = component.render(data, FigureConfig())
    return fig, ax, component, result


def test_the_mesh_is_genuinely_graded() -> None:
    # non-vacuity: on a uniform mesh edges and centers differ by half a cell and every
    # claim below would hold trivially. the point is a mesh whose widths vary.
    edges = graded_field().domain[0]
    widths = np.diff(edges)
    assert widths.max() / widths.min() > 3.0, (
        f"the test mesh spans only {widths.max() / widths.min():.2f}x in cell width; "
        "it is not graded enough to distinguish edge-drawn marks from centre-drawn ones"
    )


def test_steps_are_drawn_at_the_cell_edges() -> None:
    data = graded_field()
    _, _, _, result = render("steps", data)
    artist = result.artists["line"]
    assert isinstance(artist, StepPatch)
    # StairData is (values, edges, baseline)
    edges = artist.get_data().edges
    np.testing.assert_allclose(
        edges,
        data.domain[0],
        rtol=0,
        atol=0,
        err_msg="the step was not drawn at the mesh's own edges",
    )


def test_a_line_is_drawn_at_the_cell_centres() -> None:
    # the default is unchanged: N centres, strictly inside the N+1 edges.
    data = graded_field()
    _, _, _, result = render("line", data)
    artist = result.artists["line"]
    assert isinstance(artist, Line2D)
    x, _ = artist.get_data()
    edges = data.domain[0]
    assert len(x) == len(edges) - 1
    assert np.all(x > edges[:-1]) and np.all(x < edges[1:])


def test_the_two_styles_disagree_on_a_graded_mesh() -> None:
    # the substantive claim: these are different places. on a graded mesh a cell center
    # sits off the midpoint of its edges, so recovering one mark from the other takes
    # more than a uniform half-cell shift.
    data = graded_field()
    _, _, _, step_result = render("steps", data)
    _, _, _, line_result = render("line", data)
    edges = step_result.artists["line"].get_data().edges
    centres, _ = line_result.artists["line"].get_data()
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    assert not np.allclose(centres, midpoints), (
        "the log mesh's centres coincide with its arithmetic midpoints, so the mesh is "
        "not being centred geometrically and the two styles carry the same geometry"
    )


@pytest.mark.parametrize("drawstyle", ["line", "steps"])
def test_animation_updates_in_place(drawstyle: str) -> None:
    # the update path differs by artist type: a StepPatch takes set_data(values=, edges=)
    # while a Line2D takes positional x, y. rendering twice must reuse the artist and
    # carry the new values, or an animation would stack artists or freeze.
    data = graded_field()
    fig, ax, component, first = render(drawstyle, data)
    moved = FieldData(
        name="rho",
        values=data.values * 2.0,
        domain=data.domain,
        spacing_types=data.spacing_types,
    )
    second = component.render(moved, FigureConfig())
    assert second.artists["line"] is first.artists["line"], "the artist was recreated"
    artist = second.artists["line"]
    values = (
        artist.get_data().values
        if isinstance(artist, StepPatch)
        else artist.get_data()[1]
    )
    np.testing.assert_allclose(values, moved.values)


@pytest.mark.parametrize("drawstyle", ["line", "steps"])
def test_cleanup_removes_the_artist(drawstyle: str) -> None:
    # `ax.lines` holds lines alone and a StepPatch lives elsewhere, so a membership test
    # there would leak the step across renders while reporting success.
    data = graded_field()
    fig, ax, component, _ = render(drawstyle, data)
    component.cleanup()
    assert len(ax.lines) == 0
    assert not [p for p in ax.patches if isinstance(p, StepPatch)]
