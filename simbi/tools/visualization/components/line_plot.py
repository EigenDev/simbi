from itertools import cycle
from typing import Any
from simbi.tools.utility import get_field_str
from .base import Component
from ..bridge import SimbiDataBridge
import matplotlib.lines as mlines


class LinePlotComponent(Component):
    """Line plot visualization component"""

    def setup(self) -> None:
        """Initialize line plot resources"""
        # Setup axis style
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Create bridge to access data
        self.bridge = SimbiDataBridge(self.state)

        # Set field as y-axis label if this is the only field being plotted
        if self.props.get("show_as_label", False):
            field = self.props.get("field", "rho")
            field_label = self.bridge.get_field_label(field)
            self.ax.set_ylabel(field_label)

        # For multi-file mode, initialize lines for all files
        if self.props.get("is_multi_file", False):
            self._setup_multi_file_lines()
        else:
            # Initialize single empty line
            self.line = self.ax.plot([], [], label=self.props.get("label", ""))[0]
            # Store reference in state
            self.state.plot_elements[f"{self.id}_line"] = self.line

    def _setup_multi_file_lines(self) -> None:
        """Setup lines for multiple files"""
        files = self.props.get("files", [])
        if not files:
            return

        # Create a line for each file
        self.multi_file_lines = []
        # Store the current file and data
        original_file = self.state.data

        for idx, file_path in enumerate(files):
            for field in self.props.get("fields", ["rho"]):
                # Get field label for legend
                label = get_field_str(field)

                # Create and store the line
                line = self.ax.plot(
                    [],
                    [],
                    linewidth=self.props.get("linewidth", 2),
                    label=label,
                )[0]

                self.multi_file_lines.append(line)

                # Store reference in state
                self.state.plot_elements[f"{self.id}_line_{idx}"] = line

                # Load data for this file and update the line
                data = self.bridge.load_file(file_path)
                self.state.update_data(data)

                # Update the line with data
                self._update_line_data(line)

        # Restore original file data
        if original_file:
            self.state.update_data(original_file)

    def _update_line_data(self, line):
        """Update line with current data"""
        if not self.state.data:
            return

        # Get field data and coordinates
        field = self.props.get("field", "rho")
        var = self.bridge.get_variable(field)
        mesh = self.state.data.mesh
        setup = self.state.data.setup

        # Get slice if needed
        slice_along = self.state.config["multidim"].get("slice_along")
        if slice_along:
            from ....functional.helpers import calc_any_mean

            x = calc_any_mean(mesh[f"{slice_along}v"], setup[f"{slice_along}_spacing"])
            sliced_vars, _ = self.bridge.get_slice_data(var, mesh, setup)
            var = sliced_vars[0].flatten() if sliced_vars else var
        else:
            x, _ = self.bridge.transform_coordinates(mesh, setup)
            if setup["effective_dimensions"] > 1:
                var = var[:, 0]  # Just take first column for line plot

        # Update line data
        line.set_data(x, var)

        # Track data ranges for auto-scaling
        if not hasattr(self, "x_min"):
            self.x_min = x.min()
            self.x_max = x.max()
            self.y_min = var.min()
            self.y_max = var.max()
        else:
            self.x_min = min(self.x_min, x.min())
            self.x_max = max(self.x_max, x.max())
            self.y_min = min(self.y_min, var.min())
            self.y_max = max(self.y_max, var.max())

    def _setup_slice_lines(self, slice_count: int) -> None:
        """Setup multiple lines for slicing"""
        # Keep the existing line for the first slice
        self.slice_lines = [self.line]

        # Create additional lines for other slices
        for i in range(1, slice_count):
            line = self.ax.plot(
                [], [], color=f"C{i % 10}", linestyle=self.line.get_linestyle()
            )[0]
            self.slice_lines.append(line)

            # Store reference in state
            self.state.plot_elements[f"{self.id}_slice_line_{i}"] = line

    def render(self) -> Any:
        """Render the line plot with current data"""
        if not self.state.data:
            return self.line if hasattr(self, "line") else None

        # For multi-file mode, we already loaded all files during setup
        # Just update axis limits and handle legend
        if self.props.get("is_multi_file", False):
            config = self.state.config
            if hasattr(self, "x_min") and hasattr(self, "y_min"):
                auto_scale_x = not any(config["style"]["xlims"])
                auto_scale_y = not any(config["style"]["ylims"])
                if auto_scale_x:
                    # Add margins
                    x_margin = (
                        0.05 * (self.x_max - self.x_min)
                        if self.x_max > self.x_min
                        else 0.1 * abs(self.x_max)
                    )
                    self.ax.set_xlim(self.x_min - x_margin, self.x_max + x_margin)
                else:
                    self.ax.set_xlim(config["style"]["xlims"])
                if auto_scale_y:
                    y_margin = (
                        0.05 * (self.y_max - self.y_min)
                        if self.y_max > self.y_min
                        else 0.1 * abs(self.y_max)
                    )
                    self.ax.set_ylim(self.y_min - y_margin, self.y_max + y_margin)
                else:
                    self.ax.set_ylim(config["style"]["ylims"])

            nfields = len(self.props.get("fields", []))
            # Always show legend in multi-file mode
            if nfields > 1:
                # In order to prevemt from creating a busy legend,
                # we only show which linestyle corresponds to the
                # field being plotted. The linestyles will appear
                # in the legend with the corresponding label, but
                # the color in the legend will be a simple grey
                linestyles = cycle(["-", "--", ":", "-."])
                labs = [
                    ell.get_label()
                    for _, ell in zip(range(nfields), self.multi_file_lines)
                ]
                legend_lines = [
                    mlines.Line2D(
                        [], [], color="grey", linestyle=next(linestyles), label=label
                    )
                    for label in labs
                ]
                self.ax.legend(handles=legend_lines, loc="upper right")

            # Return the first line for animation compatibility
            return (
                self.multi_file_lines[0]
                if hasattr(self, "multi_file_lines") and self.multi_file_lines
                else None
            )

        # For single file mode, handle normally
        # Get field data
        field = self.props.get("field", "rho")
        var = self.bridge.get_variable(field)

        # Get coordinate data
        mesh = self.state.data.mesh
        setup = self.state.data.setup
        slice_along = self.state.config["multidim"].get("slice_along")

        if slice_along:
            # Handle slice data (simplifying this block)
            from ....functional.helpers import calc_any_mean

            x = calc_any_mean(mesh[f"{slice_along}v"], setup[f"{slice_along}_spacing"])
            sliced_vars, _ = self.bridge.get_slice_data(var, mesh, setup)

            # Update the main line
            if sliced_vars:
                self.line.set_data(x, sliced_vars[0].flatten())

            # Update axis limits
            if self.props.get("auto_scale", True):
                self._update_axis_limits(x, sliced_vars[0].flatten())

        else:
            # Regular line plot
            x, _ = self.bridge.transform_coordinates(mesh, setup)

            # Handle dimensionality for regular line plots
            if setup["effective_dimensions"] > 1:
                var = var[:, 0]  # Just take first column for line plot

            # Update line data
            self.line.set_data(x, var)

            # Set line label if not using field as y-axis label
            if self.props.get("label") and not self.props.get("show_as_label", False):
                self.line.set_label(self.props.get("label"))

            # Update axis limits for auto scaling
            if self.props.get("auto_scale", True):
                self._update_axis_limits(x, var)

        # Show legend if needed
        if not self.props.get("show_as_label", False):
            handles, labels = self.ax.get_legend_handles_labels()
            if labels:
                self.ax.legend()

        self.format_axis()
        return self.line

    def _update_axis_limits(self, x, y):
        """Update axis limits with margin"""
        if len(y) > 0:
            y_margin = (
                0.05 * (y.max() - y.min()) if y.max() != y.min() else 0.1 * abs(y.max())
            )
            self.ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

        if len(x) > 0:
            x_margin = (
                0.05 * (x.max() - x.min()) if x.max() != x.min() else 0.1 * abs(x.max())
            )
            self.ax.set_xlim(x.min() - x_margin, x.max() + x_margin)

    def update(self, props: dict[str, Any]) -> None:
        """Update component properties"""
        super().update(props)

        # Apply any property updates to the line
        if hasattr(self, "line"):
            if "color" in props:
                self.line.set_color(props["color"])
            if "linestyle" in props:
                self.line.set_linestyle(props["linestyle"])
            if "linewidth" in props:
                self.line.set_linewidth(props["linewidth"])
            if "label" in props and not self.props.get("show_as_label", False):
                self.line.set_label(props["label"])
