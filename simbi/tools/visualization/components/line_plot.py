from typing import Any
from matplotlib.lines import Line2D
from simbi.tools.utility import get_field_str
from .base import Component
from ..bridge import SimbiDataBridge


class LinePlotComponent(Component):
    """Line plot visualization component"""

    def setup(self) -> None:
        """Initialize line plot resources"""
        # Setup axis style
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)

        # Create bridge to access data
        self.bridge = SimbiDataBridge(self.state)

        # Initialize empty line
        self.line = self.ax.plot([], [])[0]

        # Store reference in state
        self.state.plot_elements[f"{self.id}_line"] = self.line

        self.label_placed = False

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

    def render(self) -> Line2D:
        """Render the line plot with current data"""
        if not self.state.data:
            return self.line

        # Get field data
        field = self.props.get("field", "rho")
        var = self.bridge.get_variable(field)

        # Get coordinate data
        mesh = self.state.data.mesh
        setup = self.state.data.setup
        slice_along = self.state.config["multidim"]["slice_along"]
        if slice_along:
            # Get slice coordinates
            from ....functional.helpers import calc_any_mean

            x = calc_any_mean(mesh[f"{slice_along}v"], setup[f"{slice_along}_spacing"])

            # Get sliced data
            sliced_vars, sliced_labels = self.bridge.get_slice_data(
                var,
                mesh,
                setup,
                str(get_field_str(self.props["field"])),
            )

            # Ensure we habe enough lines for all slices
            if len(sliced_vars) > 1 and not hasattr(self, "slice_line"):
                self._setup_slice_lines(len(sliced_vars))

            # Update each line with its slice
            lines_to_return = []
            for i, (slice_var, slice_label) in enumerate(
                zip(sliced_vars, sliced_labels)
            ):
                if i == 0:
                    # Update primary line
                    self.line.set_data(x, slice_var.flatten())
                    if slice_label and not self.label_placed:
                        self.label_placed = True
                        self.line.set_label(slice_label)
                    lines_to_return.append(self.line)
                elif hasattr(self, "slice_lines") and i < len(self.slice_lines):
                    # Update additional slice lines
                    self.slice_lines[i].set_data(x, slice_var.flatten())
                    if slice_label and not self.label_placed:
                        self.label_placed = True
                        self.slice_lines[i].set_label(slice_label)
                    lines_to_return.append(self.slice_lines[i])
            # Return all lines that were updated
            # [TODO: Update] For now return just the first line for animation
            # Update axis limits if auto scaling
            if self.props.get("auto_scale", True):
                margin = 0.05 * (var.max() - var.min())
                self.ax.set_ylim(var.min() - margin, var.max() + margin)

                margin = 0.05 * (x.max() - x.min())
                self.ax.set_xlim(x.min() - margin, x.max() + margin)

            # Show legend if configured
            if self.state.config["style"]["legend"]:
                self.ax.legend()

            # self.format_axis()
            return lines_to_return[0]
        else:
            # Get standard coordinates
            x, _ = self.bridge.transform_coordinates(mesh, setup)

            # Handle dimensionality for regular line plots
            if var.ndim > 1:
                var = var[:, 0]  # Just take first column for line plot

            # Update line data
            self.line.set_data(x, var)

            # Update axis limits if auto scaling
            if self.props.get("auto_scale", True):
                margin = 0.05 * (var.max() - var.min())
                self.ax.set_ylim(var.min() - margin, var.max() + margin)

                margin = 0.05 * (x.max() - x.min())
                self.ax.set_xlim(x.min() - margin, x.max() + margin)

            # Update label if needed
            if not self.line.get_label():
                self.line.set_label(self.bridge.get_field_label(field))

            # Show legend if configured
            if self.state.config.get("style", {}).get("legend", True):
                self.ax.legend()

            self.format_axis()
            return self.line

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
            if "label" in props:
                self.line.set_label(props["label"])
                self.ax.legend()
