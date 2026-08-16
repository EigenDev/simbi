# =============================================================================
# test_loud_failures.py
#
# formatting is cosmetic enough that a failure should leave a finished render
# standing, and that is exactly what makes silence dangerous here: a plot
# missing its colorbar, its labels or its legend still looks like a plot, and
# the process exits 0. a whole spherical rendering mode shipped with its
# colorbar silently dropped because one `except: pass` sat over the placement
# call.
#
# so the rule is: a formatting step may fail, the render still finishes, and
# the failure is announced. these gates hold each step to that.
# =============================================================================
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from simbi.viz import formatting
from simbi.viz.components.quad import QuadPlotComponent, QuadPlotProps
from simbi.viz.config import FigureConfig, PlotConfig, VisualizationConfig
from simbi.viz.pipeline.transforms import prepare_figure
from simbi.viz.types import CoordSystem, FieldData

EDGES = np.linspace(0.0, 1.0, 5)


@pytest.fixture(autouse=True)
def forget_previous_warnings():
    """each failure warns once per process, so a test that ran earlier would
    otherwise mask the one under test."""
    formatting._WARNED.clear()


def rendered_figure():
    figure = prepare_figure(
        VisualizationConfig(
            plot=PlotConfig(plot_type="multidim", fields=["rho"], ndim=2)
        ),
        coord_system=CoordSystem.CARTESIAN,
    )
    component = QuadPlotComponent(QuadPlotProps())
    component.initialize(figure.fig, figure.axes["main"])
    figure.add_component(
        component,
        FieldData(
            name="rho",
            values=np.random.default_rng(0).random((4, 4)),
            domain=[EDGES, EDGES],
            coord_system=CoordSystem.CARTESIAN,
            time=0.0,
        ),
    )
    return figure


def line_figure():
    """two 1d series, which is what brings out the legend and the spine
    handling: a field render carries no series names and keeps its spines."""
    from simbi.viz.components.line import LinePlotComponent, LinePlotProps

    figure = prepare_figure(
        VisualizationConfig(
            plot=PlotConfig(plot_type="line", fields=["rho", "pre"], ndim=1)
        ),
        coord_system=CoordSystem.CARTESIAN,
    )
    for name in ("rho", "pre"):
        component = LinePlotComponent(LinePlotProps())
        component.initialize(figure.fig, figure.axes["main"])
        figure.add_component(
            component,
            FieldData(
                name=name,
                values=np.linspace(1.0, 2.0, 4),
                domain=[EDGES],
                coord_system=CoordSystem.CARTESIAN,
                time=0.0,
            ),
        )
    return figure


def breaks(*_args, **_kwargs):
    raise RuntimeError("the step failed")


@pytest.mark.parametrize(
    "build, step, expected",
    [
        (rendered_figure, "apply_axis_labels", "axis labels"),
        (rendered_figure, "apply_axis_limits", "axis limits"),
        (line_figure, "apply_legend", "legend"),
        (line_figure, "remove_spines", "spine"),
    ],
)
def test_a_failed_formatting_step_is_reported(
    monkeypatch, build, step: str, expected: str
) -> None:
    figure = build()
    monkeypatch.setattr(formatting, step, breaks)

    with pytest.warns(UserWarning, match=expected):
        figure.render()


def test_a_failed_colorbar_is_reported(monkeypatch) -> None:
    """the case that motivated the rule: every spherical polygon plot lost its
    colorbar to a swallowed exception, and looked finished without one."""
    monkeypatch.setattr(formatting, "find_mappables", breaks)
    figure = rendered_figure()

    with pytest.warns(UserWarning, match="colorbar"):
        figure.render()


def test_the_render_still_completes(monkeypatch) -> None:
    """loud and survivable: the field is drawn even when its labels fail."""
    monkeypatch.setattr(formatting, "apply_axis_labels", breaks)
    figure = rendered_figure()

    with pytest.warns(UserWarning):
        figure.render()

    assert figure.axes["main"].collections


def test_a_failure_is_reported_once_not_once_per_frame(monkeypatch) -> None:
    """a movie would otherwise emit the same warning a thousand times and bury
    everything else."""
    monkeypatch.setattr(formatting, "apply_axis_limits", breaks)

    with pytest.warns(UserWarning) as first:
        rendered_figure().render()
    with pytest.warns(UserWarning) as second:
        # a second warning here would be the same failure, reported again
        rendered_figure().render()
        formatting.warn_once("other", "a different failure")

    assert len(first) == 1
    assert [str(w.message) for w in second] == ["a different failure"]
