"""
Line plot component for visualization.

This component is a simple renderer. It expects to be given
a single, 1D FieldData object and will render it.
"""

from typing import Any, Literal, Optional, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import StepPatch
from pydantic import ValidationInfo, field_validator

from simbi.functional import calc_any_mean
from simbi.viz.utility import get_field_str

from ..config import FigureConfig
from ..types import Array, FieldData, RenderResult
from .interface import Component, ComponentProps


class LinePlotProps(ComponentProps):
    """Properties for a *single* line plot component."""

    label: Optional[str] = None
    linewidth: float = 2.0
    marker: Optional[str] = None
    marker_size: float = 6.0
    alpha: float = 1.0
    drawstyle: Literal["line", "steps"] = "line"
    """How the samples are joined.

    `"line"` interpolates between cell CENTRES. `"steps"` draws the solution
    piecewise-constant across each cell, at its true edges.

    A finite-volume solution IS piecewise constant per cell, so `"steps"` is the
    faithful mark, and on a GRADED mesh it is the one that stays honest: a line
    through centres interpolates across a cell that may be twice its neighbour's
    width, and the jump in spacing at a refinement boundary disappears entirely.
    On a logarithmically spaced mesh it also removes the need to centre cells
    geometrically, since the marks are drawn at the edges themselves.

    `"line"` remains the default because at a few hundred cells per level the two
    are visually identical and the line is one artist rather than a patch.
    """

    @field_validator("linewidth", "marker_size", "alpha")
    @classmethod
    def validate_positive_float(cls, v: float, info: ValidationInfo) -> float:
        """Validate that numeric values are positive."""
        if v <= 0:
            field_name = info.field_name or "Value"
            raise ValueError(f"{field_name} must be positive, got {v}")
        return v


# the artist a line component owns. `steps` renders a StepPatch, which carries no
# marker and updates through a different setter than a Line2D.
LineArtist = Union[Line2D, StepPatch]


def _create_line_style(props: LinePlotProps) -> dict[str, Any]:
    """Create styling for the line from props."""
    style: dict[str, Any] = {
        "linewidth": props.linewidth,
        "alpha": props.alpha,
    }
    if props.marker:
        style["marker"] = props.marker
        style["markersize"] = props.marker_size
    return style


def _step_style(style: dict[str, Any]) -> dict[str, Any]:
    """The subset of a line style a StepPatch accepts.

    A step has no vertices to place markers on, so marker keys are dropped rather
    than passed through — matplotlib would raise on them.
    """
    return {k: v for k, v in style.items() if k not in ("marker", "markersize")}


def _update_line_data(line: LineArtist, x_data: Array, y_data: Array):
    """Updates an existing artist's data for animation.

    `x_data` is cell centres for a Line2D and cell EDGES for a StepPatch, so it
    carries one more entry in the step case.
    """
    if isinstance(line, StepPatch):
        line.set_data(values=y_data, edges=x_data)
    else:
        line.set_data(x_data, y_data)


def _update_line_style(
    line: LineArtist, style: dict[str, Any], label: Optional[str]
):
    """Updates an existing artist's style and label."""
    if label is not None:
        line.set_label(label)  # label is already formatted

    line.set_linewidth(style.get("linewidth", 2.0))
    line.set_alpha(style.get("alpha", 1.0))
    if isinstance(line, StepPatch):
        # a step carries no markers; the remaining keys do not apply.
        return
    line.set_marker(style.get("marker", ""))
    line.set_markersize(style.get("markersize", 6.0))


class LinePlotComponent(Component):
    """
    A simple renderer for a single 1D line.
    Expects 1D FieldData.
    """

    def __init__(self, props: LinePlotProps):
        self.props = props
        self._line: Optional[LineArtist] = None
        self._initialized: bool = False

    def initialize(self, fig: Figure, ax: Axes) -> None:
        """Initialize the component with figure and axes."""
        self.fig = fig
        self.ax = ax
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def update(self, props: LinePlotProps) -> None:
        """Update component properties and restyle the line if it exists."""
        self.props = props
        if self._line and self._initialized:
            style = _create_line_style(self.props)
            _update_line_style(self._line, style, self.props.label)

    def render(self, data: FieldData, style: FigureConfig) -> RenderResult:
        """
        Render the line plot with guaranteed 1D data.
        `data` is a *single* FieldData object.
        Returns a RenderResult containing the created line artist and metadata.
        """
        if not self._initialized:
            raise RuntimeError(
                "Component not initialized. Call initialize() first."
            )

        if data.ndim != 1:
            raise ValueError(
                f"LinePlotComponent received data with ndim={data.ndim}."
                " It can only render 1D FieldData."
            )

        ax = self.ax

        # domain contains vertices (edges), convert to cell centers for line plots
        x_vertices = data.domain[0]

        # use correct cell center calculation based on spacing type
        spacing_type = "linear"
        if data.spacing_types and len(data.spacing_types) > 0:
            spacing_type = data.spacing_types[0]

        # a step is drawn at the cell EDGES; a line needs their centres, taken with
        # the spacing's own mean so a logarithmic mesh centres geometrically.
        steps = self.props.drawstyle == "steps"
        x_data = x_vertices if steps else calc_any_mean(x_vertices, spacing_type)
        y_data = data.values

        line_style = _create_line_style(self.props)
        level_label = self.props.label or data.name
        if level_label and not level_label.startswith("$"):
            if "_L" in level_label:
                b = level_label.split("_")
                level_label = get_field_str(b[0]) + "$_{{{0}}}$".format(b[1])
            else:
                level_label = get_field_str(level_label)

        if self._line is None:
            # first render: create the artist
            if steps:
                # `baseline=None` leaves the step an open curve rather than closing
                # it to zero, so it reads as a solution and not as a filled histogram.
                self._line = ax.stairs(
                    y_data,
                    x_data,
                    baseline=None,
                    label=level_label,
                    **_step_style(line_style),
                )
            else:
                self._line = ax.plot(
                    x_data, y_data, label=level_label, **line_style
                )[0]
        else:
            # animation update: set new data and style
            _update_line_data(self._line, x_data, y_data)
            _update_line_style(self._line, line_style, level_label)

        # update x-axis limits for moving mesh animations
        ax.set_xlim(x_data.min(), x_data.max())

        return RenderResult(
            artists={"line": self._line}, metadata={"label": level_label}
        )

    def cleanup(self) -> None:
        """Clean up resources."""
        # a step is a patch and never appears in `ax.lines`; testing membership
        # there would silently leak it across renders.
        if self._line is not None and self._line.axes is not None:
            self._line.remove()
        self._line = None
        # DO NOT call self.ax.cla() - it would wipe other components!
