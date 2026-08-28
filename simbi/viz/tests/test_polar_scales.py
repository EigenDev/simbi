# =============================================================================
# test_polar_scales.py
#
# a polar field chart draws with no axis numbers by default; the show_scales
# option labels its radial and angular scale. what has to hold:
#   - the default chart stays clean: every tick label is blank.
#   - with the option on, the radial and angular labels carry text.
#   - the labels come from matplotlib's automatic locators, so a later frame
#     whose mesh expanded relabels to the new radial extent. a fixed set of
#     labels would caption a moving mesh with the first frame's radii forever.
#   - a declared characteristic length divides each labeled radius and carries
#     the unit string, so a run in code units reads in physical ones.
#   - the under-wedge colorbars drop below the band the radial labels occupy;
#     a bar in the label band prints the two scales through each other.
# =============================================================================
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from simbi.viz.components.quad import QuadPlotComponent, QuadPlotProps
from simbi.viz.config import (
    FigureConfig,
    PlotConfig,
    VisualizationConfig,
)
from simbi.viz.pipeline.transforms import prepare_figure
from simbi.viz.types import CoordSystem, FieldData, PlotData

THETA_EDGES = np.linspace(0.0, np.pi, 9)


def shell_field(name: str, r_max: float) -> FieldData:
    radial_edges = np.geomspace(1.0, r_max, 17)
    values = np.arange(
        (THETA_EDGES.size - 1) * (radial_edges.size - 1), dtype=float
    ).reshape(THETA_EDGES.size - 1, radial_edges.size - 1)
    return FieldData(
        name=name,
        values=values,
        domain=[THETA_EDGES, radial_edges],
        coord_system=CoordSystem.SPHERICAL,
        time=0.0,
    )


def polar_figure(style: FigureConfig, field: FieldData):
    """a polar figure carrying one quad component, through the production path."""
    figure = prepare_figure(
        VisualizationConfig(
            plot=PlotConfig(plot_type="multidim", fields=[field.name], ndim=2),
            figure=style,
        ),
        projection="polar",
        coord_system=CoordSystem.SPHERICAL,
    )
    component = QuadPlotComponent(QuadPlotProps())
    component.initialize(figure.fig, figure.axes["main"])
    figure.add_component(component, field)
    figure.render()
    return figure


def label_texts(ax) -> list[str]:
    ax.figure.canvas.draw()
    return [
        text.get_text()
        for text in ax.get_xticklabels() + ax.get_yticklabels()
    ]


def radial_labels(ax) -> list[str]:
    ax.figure.canvas.draw()
    return [
        text.get_text() for text in ax.get_yticklabels() if text.get_text()
    ]


def radial_label_values(ax, units: str = "") -> list[float]:
    suffix = f" {units}" if units else ""
    return [
        float(label.removesuffix(suffix).replace("\N{MINUS SIGN}", "-"))
        for label in radial_labels(ax)
    ]


def test_a_polar_chart_is_unlabeled_by_default() -> None:
    figure = polar_figure(FigureConfig(), shell_field("D", 10.0))

    assert all(label == "" for label in label_texts(figure.axes["main"]))


def test_show_scales_labels_both_axes() -> None:
    figure = polar_figure(
        FigureConfig(show_scales=True), shell_field("D", 10.0)
    )
    ax = figure.axes["main"]

    assert any(text.get_text() for text in ax.get_xticklabels())
    assert radial_label_values(ax)


def test_scales_follow_a_mesh_that_expands_between_frames() -> None:
    figure = polar_figure(
        FigureConfig(show_scales=True), shell_field("D", 10.0)
    )
    ax = figure.axes["main"]
    first_frame_max = max(radial_label_values(ax))

    grown = shell_field("D", 40.0)
    figure.draw_frame(
        PlotData(
            fields=[grown],
            time=1.0,
            dimensions=2,
            coord_system=CoordSystem.SPHERICAL,
        ),
        full=False,
    )

    assert max(radial_label_values(ax)) > first_frame_max


def test_a_characteristic_length_rescales_the_radial_labels() -> None:
    """radii labeled as r / length_scale with the unit string attached read a
    code-unit run in physical units."""
    figure = polar_figure(
        FigureConfig(show_scales=True, length_scale=2.0, length_units="pc"),
        shell_field("D", 10.0),
    )
    ax = figure.axes["main"]

    labels = radial_labels(ax)
    assert labels
    assert all(label.endswith(" pc") for label in labels)

    labeled = radial_label_values(ax, units="pc")
    drawn_ticks = [
        tick
        for tick, text in zip(ax.get_yticks(), ax.get_yticklabels())
        if text.get_text()
    ]
    assert labeled == pytest.approx([tick / 2.0 for tick in drawn_ticks])


def test_an_awkward_scale_still_labels_round_numbers() -> None:
    """the ticks are chosen in the scaled units and mapped back to radii, so
    an irrational characteristic length labels 0.6, 0.8, 1 rather than the
    full float width of nice-radius / scale."""
    figure = polar_figure(
        FigureConfig(
            show_scales=True, length_scale=0.8779, length_units=r"$\ell$"
        ),
        shell_field("D", 10.0),
    )
    ax = figure.axes["main"]

    numeric = [
        label.removesuffix(r" $\ell$") for label in radial_labels(ax)
    ]
    assert numeric
    # a round tick prints in a few characters; six significant digits is the
    # smashed-label failure
    assert all(len(part) <= 4 for part in numeric)


def test_the_radial_scale_stays_sparse() -> None:
    """a handful of labels fits the half-width the radial axis spans; the
    automatic density prints them into each other."""
    figure = polar_figure(
        FigureConfig(show_scales=True), shell_field("D", 10.0)
    )

    assert len(radial_labels(figure.axes["main"])) <= 6


def mirrored_shell_field(name: str, r_max: float) -> FieldData:
    """a hemisphere with a horizontal flat edge, the geometry that lays its
    colorbars in the strip beneath the wedge."""
    return shell_field(name, r_max).model_copy(
        update={
            "domain": [
                np.linspace(-0.5 * np.pi, 0.5 * np.pi, THETA_EDGES.size),
                shell_field(name, r_max).domain[1],
            ]
        }
    )


def underwedge_bar_top(style: FigureConfig) -> float:
    figure = polar_figure(style, mirrored_shell_field("D", 10.0))
    bars = getattr(figure.axes["main"], "_simbi_colorbars")
    return float(bars[0].ax.get_position().y1)


def test_scale_labels_push_the_underwedge_bar_down() -> None:
    with_scales = underwedge_bar_top(FigureConfig(show_scales=True))
    without = underwedge_bar_top(FigureConfig())

    assert with_scales < without


def test_blank_labels_stay_blank_as_the_mesh_expands() -> None:
    """the empty fixed formatter holds every frame of a default movie clean,
    even as the view limits move under it."""
    figure = polar_figure(FigureConfig(), shell_field("D", 10.0))
    ax = figure.axes["main"]

    figure.draw_frame(
        PlotData(
            fields=[shell_field("D", 40.0)],
            time=1.0,
            dimensions=2,
            coord_system=CoordSystem.SPHERICAL,
        ),
        full=False,
    )

    assert all(label == "" for label in label_texts(ax))
