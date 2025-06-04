from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.style import Style
from rich import box
from ..functional import get_memory_usage

import time
import math
from typing import Any, Sequence

from .summary import SimulationParameterSummary

TABLE_WIDTH = 103


class RichSimulationSummary:
    """We're now gonna use Rich for beautiful terminal output"""

    def __init__(self) -> None:
        """Initialize the Rich console and styling options"""
        self.console = Console(width=TABLE_WIDTH)
        self.summary = SimulationParameterSummary()

        # style themes
        self.styles = {
            "header": Style(color="bright_cyan", bold=True),
            "subheader": Style(color="cyan", bold=True),
            "param_name": Style(color="white"),
            "param_value": Style(color="bright_white"),
            "grid_params": Style(color="bright_cyan"),
            "physics_params": Style(color="bright_cyan"),
            "boundary_params": Style(color="bright_cyan"),
            "numerical_params": Style(color="bright_cyan"),
            "runtime_params": Style(color="bright_cyan"),
            "custom_params": Style(color="bright_cyan"),
            "output_params": Style(color="bright_cyan"),
            "statistics": Style(color="bright_white", italic=True),
            "memory": Style(color="yellow"),
            "critical": Style(color="red", bold=True),
            "warning": Style(color="yellow", bold=True),
            "ok": Style(color="green"),
        }

        # box style for different parameter categories
        self.boxes = {
            "Grid": box.DOUBLE,
            "Physics": box.ROUNDED,
            "Boundary Conditions": box.HEAVY,
            "Time Configuration": box.SIMPLE_HEAVY,
            "Numerical Method": box.MINIMAL,
            "Simulation Runtime": box.SIMPLE,
            "Output": box.SQUARE,
            "Statistics": box.DOUBLE_EDGE,
        }

    def show_loading_animation(self) -> None:
        """Show a loading animation while preparing the summary"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Preparing simulation summary..."),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Preparing...", total=100)
            for i in range(101):
                progress.update(task, completed=i)
                time.sleep(0.001)  # simulated work :)

    def create_header(self) -> Panel:
        """Create a stylish header for the simulation summary"""
        title_text = Text(
            "SIMBI SIMULATION PARAMETERS",
            style=self.styles["header"],
        )
        title_text.justify = "center"

        return Panel(
            title_text,
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 2),
            width=TABLE_WIDTH,
        )

    def create_parameter_table(self, category: str, params: dict[str, Any]) -> Table:
        """Create a rich table for a specific parameter category"""
        # get appropriate box style for this category or default to ROUNDED
        box_style = self.boxes.get(category, box.ROUNDED)

        active_params = any(p for p in params.values())
        if not active_params:
            # if no parameters are active, return an empty table
            return Table(box=box_style, title=category, width=TABLE_WIDTH)

        # determine style based on category
        if "Grid" in category:
            category_style = self.styles["grid_params"]
        elif "Physics" in category:
            category_style = self.styles["physics_params"]
        elif "Boundary" in category:
            category_style = self.styles["boundary_params"]
        elif "Numerical" in category or "Time" in category:
            category_style = self.styles["numerical_params"]
        elif "Output" in category:
            category_style = self.styles["output_params"]
        elif "Runtime" in category:
            category_style = self.styles["runtime_params"]
        elif "Custom" in category:
            category_style = self.styles["custom_params"]
        else:
            category_style = self.styles["param_name"]

        table = Table(
            title=category,
            box=box_style,
            title_style=category_style,
            header_style=category_style,
            expand=False,
            show_lines=True,
            width=TABLE_WIDTH,
        )

        # add columns
        table.add_column("Parameter", style=self.styles["param_name"], justify="right")
        table.add_column("Value", style=self.styles["param_value"])
        table.add_column("Description", style="white", justify="left")

        # add rows for each parameter
        for name, value in params.items():
            if value is None:
                continue
            elif value == {}:
                continue
            elif value == []:
                continue
            elif value == 0 and name != "start_time":
                continue

            if "expression" in name:
                the_value = "user-defined"
            else:
                the_value = value
            # format value properly based on type
            formatted_value = self._format_parameter_value(the_value)

            # add description if available
            description = self._get_parameter_description(name)

            table.add_row(name, formatted_value, description)

        return table

    def _format_parameter_value(self, value: Any) -> str:
        """Format parameter values nicely based on their type"""
        if isinstance(value, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in value):
                # format numeric arrays with precision
                if all(isinstance(x, int) for x in value):
                    return str(value)
                else:
                    return str(tuple(round(x, 2) for x in value))
            elif all(isinstance(x, str) for x in value):
                # format string arrays with quotes
                # this is likely the boundary conditions
                return str(
                    tuple(tuple((x, y)) for x, y in zip(value[0::2], value[1::2]))
                )
            else:
                return str(value)
        elif isinstance(value, float):
            return f"{value:.6g}"
        elif isinstance(value, bool):
            return "[green]True[/green]" if value else "[red]False[/red]"
        else:
            return str(value)

    def _get_parameter_description(self, param_name: str) -> str:
        """Get description for a parameter (placeholder - would be better with actual descriptions)"""
        descriptions = {
            "nx": "Number of cells in x1-direction",
            "ny": "Number of cells in x2-direction",
            "nz": "Number of cells in x3-direction",
            "gamma": "Adiabatic index",
            "cfl_number": "Courant-Friedrichs-Lewy condition",
            "dt": "Timestep size",
            "start_time": "Simulation start time",
            "end_time": "Simulation end time",
            "reconstruction_method": "Spatial reconstruction scheme",
            "x1_spacing": "Grid spacing in x1-direction",
            "x2_spacing": "Grid spacing in x2-direction",
            "x3_spacing": "Grid spacing in x3-direction",
            "x1bounds": "Physical bounds in x1-direction",
            "x2bounds": "Physical bounds in x2-direction",
            "x3bounds": "Physical bounds in x3-direction",
            "dimensionality": "Number of dimensions (1, 2, or 3)",
            "coord_system": "Coordinate system (cartesian, cylindrical, spherical)",
            "regime": "Physical regime (classical, srhd, srmhd)",
            "adiabatic_index": "Adiabatic index for the gas",
            "isothermal": "Isothermal condition (True/False)",
            "is_mhd": "Magnetohydrodynamics (True/False)",
            "is_relativistic": "Relativistic regime (True/False)",
            "shakura_sunyaev_alpha": "Shakura-Sunyaev alpha parameter",
            "viscosity": "Viscosity coefficient",
            "ambient_sound_speed": "Ambient sound speed",
            "use_quirk_smoothing": "Use Quirk (1994) smoothing (True/False)",
            "use_fleischmann_limiter": "Use Fleischmann et al. (2020) low-Mach HLLC fix (True / False)",
            "solver": "Riemann solver used",
            "spatial_order": "Spatial reconstruction schemes",
            "temporal_order": "temporal integration scheme",
            "plm_theta": "PLM theta parameter",
            "boundary_conditions": "Boundary conditions (periodic, reflective, etc.)",
            "data_directory": "Directory for input/output data",
            "checkpoint_interval": "Interval for saving checkpoints",
            "checkpoint_index": "Index of the current checkpoint (if any)",
            "bx1_inner_expressions": "User-defined inner boundary conditions for inner x1 boundary",
            "bx1_outer_expressions": "User-defined outer boundary conditions for outer x1 boundary",
            "bx2_inner_expressions": "User-defined inner boundary conditions for inner x2 boundary",
            "bx2_outer_expressions": "User-defined outer boundary conditions for outer x2 boundary",
            "bx3_inner_expressions": "User-defined inner boundary conditions for inner x3 boundary",
            "bx3_outer_expressions": "User-defined outer boundary conditions for outer x3 boundary",
            "gravity_source_expressions": "Gravity source term expressions",
            "hydro_source_expressions": "Hydrodynamic source term expressions",
            "local_sound_speed_expressions": "Local sound speed expressions",
        }
        return descriptions.get(param_name, "")

    def create_statistics_panel(self, stats: dict[str, Any]) -> Panel:
        """Create a visually appealing panel for simulation statistics"""
        stats_table = Table(box=box.SIMPLE_HEAD, width=TABLE_WIDTH)

        # add memory usage with visual indicators
        stats_table.add_column("Statistic", style=self.styles["statistics"])
        stats_table.add_column("Value", style=self.styles["statistics"])
        stats_table.add_column("Visual", justify="left")

        # format memory usage with unit conversion
        memory_gb = stats.get("estimated_memory_gb", 0)
        memory_text = f"{memory_gb:.2f} GB"

        # create a visual indicator for memory usage
        memory_usage_visual = self._create_memory_usage_bar(memory_gb)

        stats_table.add_row("Estimated Memory Usage", memory_text, memory_usage_visual)

        # add cell metrics
        if "cells_per_dim" in stats:
            cells = stats["cells_per_dim"]
            stats_table.add_row(
                "Cells per dimension",
                f"X: {cells[0]}, Y: {cells[1]}, Z: {cells[2]}",
                "",
            )

        # add performance estimate if available
        if "performance_estimate" in stats:
            perf = stats["performance_estimate"]
            stats_table.add_row("Estimated Performance", f"{perf} cell updates/s", "")

        # add timestep info
        if "dt" in stats and "tmax" in stats:
            steps = math.ceil(stats["tmax"] / stats["dt"])
            stats_table.add_row("Estimated Timesteps", f"{steps:,}", "")

        return Panel(
            stats_table,
            title="Simulation Statistics",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2),
        )

    def _create_memory_usage_bar(self, memory_gb: float) -> Text:
        """Create a visual memory usage indicator"""
        # create a bar representing memory usage
        bar_length = 20

        # determine color based on memory usage
        if memory_gb < 1:
            color = "green"
        elif memory_gb < 8:
            color = "yellow"
        elif memory_gb < 32:
            color = "orange"
        else:
            color = "red"

        # scale to max 64GB for visualization purposes
        filled = min(math.ceil(memory_gb / 64 * bar_length), bar_length)
        empty = bar_length - filled

        bar = Text(f"[{color}]{'█' * filled}{'░' * empty}[/] ({memory_gb:.2f} GB)")
        return bar

    def generate_and_display(self, params: dict[str, Any]) -> None:
        """Generate and display a beautiful parameter summary using Rich"""
        # optionally show loading animation
        self.show_loading_animation()

        # clear the console for a clean display
        # self.console.clear()

        # display the header
        self.console.print(self.create_header())

        # create a layout for organizing content
        layout = Layout()
        layout.split_column(
            Layout(name="header"),
            Layout(name="parameters"),
            Layout(name="statistics"),
            Layout(name="footer"),
        )

        # get organized parameters from the original summary class
        param_categories = self.summary.define_categories()

        # extract actual parameter values from the input params
        organized_params = {}
        for category, category_info in param_categories.items():
            param_dict = {}
            for param_name in category_info.get("params", []):
                if param_name in params:
                    param_dict[param_name] = params[param_name]
            organized_params[category] = {
                "title": category_info.get("title", category),
                "params": param_dict,
            }

        # create tables for each parameter category
        tables = []
        for category, info in organized_params.items():
            if info["params"]:  # Only add if there are parameters
                table = self.create_parameter_table(info["title"], info["params"])
                tables.append(table)

        # compute statistics
        stats: dict[str, int | float | Sequence[Any]] = {}
        ni, nj, nk = params["resolution"]
        memory_bytes = float(get_memory_usage())
        stats["estimated_memory_gb"] = memory_bytes / (1024**3)
        stats["cells_per_dim"] = (ni, nj, nk)

        # create statistics panel
        stats_panel = self.create_statistics_panel(stats)

        # add all panels to the console output
        for table in tables:
            self.console.print(table)

        self.console.print(stats_panel)

        # display a footer
        footer_text = Text(
            "End of Simulation Parameters", style=self.styles["subheader"]
        )
        footer_text.justify = "center"
        footer_panel = Panel(footer_text, box=box.DOUBLE, border_style="bright_cyan")
        self.console.print(footer_panel)


# function to use as an entry point
def print_rich_simulation_parameters(params: dict[str, Any]) -> None:
    """Print a beautiful simulation parameter summary using Rich"""
    rich_summary = RichSimulationSummary()
    rich_summary.generate_and_display(params)
