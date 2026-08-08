# =============================================================================
# test_color_scale.py
#
# colour is read as an absolute quantity. a movie whose scale is taken from
# each frame's own extremes therefore reports a constant where the data has a
# decay: a shock fading by decades is redrawn at full brightness in every
# frame, and nothing in the picture says so.
#
# the scale is swept over the whole sequence and pinned before the first frame
# instead. what has to hold: the pinned range spans every frame, all levels of
# one quantity share it, and anything the user pinned themselves outranks it.
# =============================================================================
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from simbi.viz.api import _has_color_range, _panel_props
from simbi.viz.components.quad import QuadPlotComponent, QuadPlotProps
from simbi.viz.config import FigureConfig, PlotConfig, VisualizationConfig
from simbi.viz.pipeline import color_range as color_range_module
from simbi.viz.types import ColorRange, CoordSystem, FieldData, PlotData

EDGES = np.linspace(0.0, 1.0, 5)


def frame(name: str, values: np.ndarray) -> FieldData:
    return FieldData(
        name=name,
        values=values,
        domain=[EDGES, EDGES],
        coord_system=CoordSystem.CARTESIAN,
        time=0.0,
    )


@pytest.fixture
def sequence(monkeypatch):
    """three checkpoints of a decaying quantity, without touching a disk.

    the peak falls by two decades across the sequence, which is exactly the
    case a per-frame scale hides."""
    frames = {
        "t0": [frame("rho", np.full((4, 4), 100.0))],
        "t1": [frame("rho", np.full((4, 4), 10.0))],
        "t2": [frame("rho", np.full((4, 4), 1.0))],
    }

    monkeypatch.setattr(
        color_range_module, "load_data", lambda path: path, raising=False
    )
    import simbi.viz.pipeline.plot_data as plot_data_module
    import simbi.viz.pipeline.transforms as transforms_module

    monkeypatch.setattr(transforms_module, "load_data", lambda path: path)
    monkeypatch.setattr(
        plot_data_module,
        "create_plot_data",
        lambda data, fields, config: PlotData(
            fields=frames[data], time=0.0, dimensions=2
        ),
    )
    return list(frames)


def config() -> VisualizationConfig:
    return VisualizationConfig(
        plot=PlotConfig(plot_type="multidim", fields=["rho"], ndim=2)
    )


# --- the sweep --------------------------------------------------------------


def test_the_range_spans_every_frame(sequence) -> None:
    swept = color_range_module.sequence_color_range(
        sequence, ["rho"], config()
    )

    assert swept["rho"].min == pytest.approx(1.0)
    assert swept["rho"].max == pytest.approx(100.0)


def test_the_levels_of_one_quantity_share_a_scale(monkeypatch) -> None:
    """levels scaled apart put a seam in the colours at every refinement
    boundary, where the data is continuous."""
    import simbi.viz.pipeline.plot_data as plot_data_module
    import simbi.viz.pipeline.transforms as transforms_module

    monkeypatch.setattr(transforms_module, "load_data", lambda path: path)
    monkeypatch.setattr(
        plot_data_module,
        "create_plot_data",
        lambda data, fields, config: PlotData(
            fields=[
                frame("rho", np.full((4, 4), 2.0)),
                frame("rho_L1", np.full((4, 4), 50.0)),
            ],
            time=0.0,
            dimensions=2,
        ),
    )

    swept = color_range_module.sequence_color_range(["t0"], ["rho"], config())

    assert set(swept) == {"rho"}
    assert (swept["rho"].min, swept["rho"].max) == pytest.approx((2.0, 50.0))


def test_a_frame_of_no_finite_values_does_not_poison_the_scale(
    monkeypatch,
) -> None:
    """a nan or inf swept into the limits makes the norm reject every frame."""
    import simbi.viz.pipeline.plot_data as plot_data_module
    import simbi.viz.pipeline.transforms as transforms_module

    monkeypatch.setattr(transforms_module, "load_data", lambda path: path)
    values = np.array([[1.0, np.nan], [np.inf, 4.0]])
    monkeypatch.setattr(
        plot_data_module,
        "create_plot_data",
        lambda data, fields, config: PlotData(
            fields=[frame("rho", values)], time=0.0, dimensions=2
        ),
    )

    swept = color_range_module.sequence_color_range(["t0"], ["rho"], config())

    assert (swept["rho"].min, swept["rho"].max) == pytest.approx((1.0, 4.0))


# --- what reaches the panel -------------------------------------------------


def test_the_swept_range_reaches_the_panel() -> None:
    props = _panel_props(
        None,
        "quad",
        QuadPlotProps,
        "rho",
        index=0,
        npanels=1,
        color_ranges={"rho": ColorRange(min=1.0, max=100.0)},
    )

    assert (props.color_range.min, props.color_range.max) == (1.0, 100.0)


def test_a_requested_range_outranks_the_sweep() -> None:
    """the sweep is a default drawn from the data; an explicit stretch is a
    decision about what the reader should see."""
    asked = QuadPlotProps(color_range=ColorRange(min=0.0, max=5.0))

    props = _panel_props(
        {"quad": asked},
        "quad",
        QuadPlotProps,
        "rho",
        index=0,
        npanels=1,
        color_ranges={"rho": ColorRange(min=1.0, max=100.0)},
    )

    assert (props.color_range.min, props.color_range.max) == (0.0, 5.0)


def test_an_unset_range_is_not_mistaken_for_a_request() -> None:
    assert not _has_color_range(QuadPlotProps())
    assert _has_color_range(
        QuadPlotProps(color_range=ColorRange(min=None, max=2.0))
    )


# --- what the reader sees ---------------------------------------------------


def draw(props: QuadPlotProps, values_per_frame) -> list[tuple[float, float]]:
    """the colour limits the mesh ends up drawn with, frame by frame."""
    fig, ax = plt.subplots()
    component = QuadPlotComponent(props)
    component.initialize(fig, ax)

    limits = []
    for values in values_per_frame:
        component.render(frame("rho", values), FigureConfig())
        limits.append((component._mesh.norm.vmin, component._mesh.norm.vmax))
    return limits


def test_a_pinned_scale_holds_still_while_the_data_decays() -> None:
    decaying = [np.full((4, 4), 100.0), np.full((4, 4), 1.0)]

    limits = draw(
        QuadPlotProps(color_range=ColorRange(min=1.0, max=100.0)), decaying
    )

    assert limits[0] == limits[1] == (1.0, 100.0)


def test_an_unpinned_scale_follows_the_data_down() -> None:
    """this is the behaviour the sweep exists to replace: the two frames differ
    by two decades and are drawn with identical colours."""
    decaying = [np.full((4, 4), 100.0), np.full((4, 4), 1.0)]

    limits = draw(QuadPlotProps(), decaying)

    assert limits[0] != limits[1]
    assert limits[1][1] < limits[0][1]


# --- half-open ranges -------------------------------------------------------


def test_one_end_of_a_range_can_be_left_to_the_data() -> None:
    """clipping the top of a scale while the bottom follows the field is an
    ordinary request; refusing it turns a well-formed range into an error."""
    from simbi.viz.config_loader import load_component_props

    props = load_component_props(overrides=["quad.color_range.max=5"])

    assert props["quad"].color_range.max == 5.0
    assert props["quad"].color_range.min is None


def test_an_inverted_range_is_still_refused() -> None:
    from simbi.viz.config_loader import load_component_props

    with pytest.raises(ValueError, match="must be greater than"):
        load_component_props(
            overrides=["quad.color_range.min=9", "quad.color_range.max=2"]
        )
