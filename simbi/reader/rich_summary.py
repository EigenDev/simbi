import math
import time
from typing import Any, Sequence

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text

from .summary import SimulationParameterSummary

# table width is computed at runtime from the console size; see self.table_width
TABLE_WIDTH = None


class RichSimulationSummary:
    """we're now gonna use Rich for beautiful terminal output"""

    def __init__(self, console: Console | None = None) -> None:
        """initialize the rich console and styling options

        - console: optional rich.Console to render into (useful for tests)
        - computes self.table_width from the active console so tables and panels
          can expand responsively to the terminal size.
        """
        # use provided console or create a default one
        self.console = console or Console()
        # compute table width with a small padding and a sensible minimum
        self.table_width = max(40, self.console.size.width - 6)
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
        """show a loading animation while preparing the summary"""
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
        """create a stylish header for the simulation summary"""
        title_text = Text(
            "SIMBI SIMULATION PARAMETERS", style=self.styles["header"]
        )
        title_text.justify = "center"

        return Panel(
            title_text,
            border_style="bright_cyan",
            box=box.DOUBLE,
            padding=(1, 2),
            width=self.table_width,
        )

    def create_parameter_table(
        self, category: str, params: dict[str, Any]
    ) -> Table:
        """create a rich table for a specific parameter category"""
        # get appropriate box style for this category or default to rounded
        box_style = self.boxes.get(category, box.ROUNDED)

        active_params = any(p for p in params.values())
        if not active_params:
            # if no parameters are active, return an empty table with the configured width
            return Table(
                box=box_style,
                title=category,
                width=self.table_width,
                title_justify="center",
            )

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
            width=self.table_width,
            title_justify="center",
        )

        # add columns
        table.add_column(
            "Parameter", style=self.styles["param_name"], justify="right"
        )
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

            # emphasize important parameters so they pop in the table
            if name in (
                "regime",
                "resolution",
                "is_mhd",
                "is_relativistic",
                "isothermal",
            ):
                formatted_value = (
                    f"[bold yellow]{formatted_value}[/bold yellow]"
                )

            table.add_row(name, formatted_value, description)

        return table

    def _format_parameter_value(self, value: Any) -> str:
        """format parameter values nicely based on their type"""
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
                    tuple(
                        tuple((x, y)) for x, y in zip(value[0::2], value[1::2])
                    )
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
        """get description for a parameter (placeholder - would be better with actual descriptions)"""
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
            "x1_bounds": "Physical bounds in x1-direction",
            "x2_bounds": "Physical bounds in x2-direction",
            "x3_bounds": "Physical bounds in x3-direction",
            "dimensionality": "Number of dimensions (1, 2, or 3)",
            "coord_system": "Coordinate system (cartesian, cylindrical, spherical)",
            "regime": "Physical regime (newtonian, rhd, rmhd)",
            "adiabatic_index": "Adiabatic index for the gas",
            "isothermal": "Isothermal condition (True/False)",
            "is_mhd": "Magnetohydrodynamics (True/False)",
            "is_relativistic": "Relativistic regime (True/False)",
            "shakura_sunyaev_alpha": "Shakura-Sunyaev alpha parameter",
            "viscosity": "Viscosity coefficient",
            "resistivity": "Ohmic resistivity coefficient",
            "ambient_sound_speed": "Ambient sound speed",
            "use_quirk_smoothing": "Use Quirk (1994) smoothing (True/False)",
            "use_fleischmann_limiter": "Use Fleischmann et al. (2020) low-Mach HLLC fix (True / False)",
            "solver": "Riemann solver used",
            "reconstruction": "Spatial reconstruction schemes",
            "timestepping": "time_series integration scheme",
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
            "source_expressions": "Source term expressions",
        }
        return descriptions.get(param_name, "")

    def create_statistics_panel(self, stats: dict[str, Any]) -> Panel:
        """create a visually appealing panel for simulation statistics"""
        stats_table = Table(box=box.SIMPLE_HEAD, width=self.table_width)

        # add memory usage with visual indicators
        stats_table.add_column("Statistic", style=self.styles["statistics"])
        stats_table.add_column("Value", style=self.styles["statistics"])
        stats_table.add_column("Visual", justify="left")

        # format memory usage with unit conversion
        memory_gb = stats.get("estimated_memory_gb", 0)
        memory_text = f"{memory_gb:.2f} GB"

        # create a visual indicator for memory usage
        memory_usage_visual = self._create_memory_usage_bar(memory_gb)

        stats_table.add_row(
            "Estimated Memory Usage", memory_text, memory_usage_visual
        )

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
            stats_table.add_row(
                "Estimated Performance", f"{perf} cell updates/s", ""
            )

        # add timestep info
        if "dt" in stats and "tmax" in stats:
            steps = math.ceil(stats["tmax"] / stats["dt"])
            stats_table.add_row("Estimated Timesteps", f"{steps:,}", "")

        if "gpu_block_dims" in stats:
            gpu_dims = stats["gpu_block_dims"]
            stats_table.add_row(
                "GPU Block Dimensions",
                f"X: {gpu_dims[0]}, Y: {gpu_dims[1]}, Z: {gpu_dims[2]}",
                "",
            )

        return Panel(
            stats_table,
            title="Simulation Statistics",
            border_style="bright_blue",
            box=box.DOUBLE,
            padding=(1, 2),
            width=self.table_width,
            title_align="center",
        )

    def _create_memory_usage_bar(self, memory_gb: float) -> Text:
        """create a visual memory usage indicator"""
        # create a bar representing memory usage
        bar_length = 20

        # determine color based on memory usage
        if memory_gb < 1:
            color = "green"
        elif memory_gb < 8:
            color = "yellow"
        elif memory_gb < 32:
            color = "orange3"
        else:
            color = "red"

        # scale to max 64gb for visualization purposes
        filled = min(math.ceil(memory_gb / 64 * bar_length), bar_length)
        empty = bar_length - filled

        bar = Text(f"{'█' * filled}{'░' * empty}[/] ({memory_gb:.2f} GB)")
        bar.stylize(color)
        return bar

    def generate_and_display(self, params: dict[str, Any]) -> None:
        """generate and display a beautiful parameter summary using rich"""
        # optionally show loading animation
        # self.show_loading_animation()

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
            if info["params"]:  # only add if there are parameters
                table = self.create_parameter_table(
                    info["title"], info["params"]
                )
                tables.append(table)

        # compute statistics
        stats: dict[str, int | float | Sequence[Any]] = {}
        ni, nj, nk = params["resolution"]
        nzones = math.prod(params["resolution"])
        ncons = 1
        nprims = 1
        nfluxes = params["dimensionality"]
        nvars = params["dimensionality"] + 3  # dens, vec, edens, chi
        if params["is_mhd"]:
            nvars = 9  # dens, vec(3), B(3), edens, chi

        zbytes = 8 * nvars * nzones
        memory_bytes = (ncons + nprims + nfluxes) * zbytes
        if params["timestepping"] == "rk2":
            memory_bytes += 2 * ncons * zbytes
        stats["estimated_memory_gb"] = memory_bytes / 1024**3
        stats["cells_per_dim"] = (ni, nj, nk)
        if params.get("gpu_block_dims") is not None:
            stats["gpu_block_dims"] = tuple(params["gpu_block_dims"])

        # create statistics panel
        stats_panel = self.create_statistics_panel(stats)

        # add all panels to the console output responsively
        # when the terminal is wide, attempt a two-column presentation
        if self.console.size.width > 110 and len(tables) > 1:
            left = tables[::2]
            right = tables[1::2]
            # when printing side-by-side, tables may need to be narrower than full width
            # strict side-by-side fitting requires a smaller width passed to each table
            for l, r in zip(left, right):
                # print paired tables side-by-side
                self.console.print(l, r)
            # print leftover if odd number of tables
            if len(left) > len(right):
                self.console.print(left[-1])
        else:
            for table in tables:
                self.console.print(table)

        self.console.print(stats_panel)

        # display a footer
        footer_text = Text(
            "End of Simulation Parameters", style=self.styles["subheader"]
        )
        footer_text.justify = "center"
        footer_panel = Panel(
            footer_text,
            box=box.DOUBLE,
            border_style="bright_cyan",
            width=self.table_width,
        )
        self.console.print(footer_panel)


# function to use as an entry point
def print_rich_simulation_parameters(params: dict[str, Any]) -> None:
    """print a beautiful simulation parameter summary using rich"""
    rich_summary = RichSimulationSummary()
    rich_summary.generate_and_display(params)
