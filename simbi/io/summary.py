"""
Parameter summary and visualization utilities for simbi simulations.

This module provides tools for formatting, grouping, and displaying
simulation parameters in a visually appealing way (imo).
"""

from typing import Any, Optional
import numpy as np
from .logging import logger
from ..functional.helpers import tuple_of_tuples, to_tuple_of_pairs


class ParameterFormatter:
    """Handles formatting of different parameter types for display"""

    @staticmethod
    def format_tuple_of_tuples(param: Any) -> str:
        """Format nested tuples with uniform presentation"""
        if tuple_of_tuples(param):
            formatted = tuple(
                tuple(
                    f"{x:.3f}" if isinstance(x, float) else str(x) for x in inner_tuple
                )
                for inner_tuple in param
            )
            return str(formatted).replace("'", "").replace(" ", "")
        else:
            return str(param)

    @staticmethod
    def format_param(param: Any) -> str:
        """Format a parameter value for display based on its type"""
        if isinstance(param, (float, np.float64)):
            return f"{param:.3f}"
        elif callable(param):
            return f"user-defined {param.__name__} function"
        elif isinstance(param, (list, np.ndarray)):
            if len(param) > 6:
                return f"user-defined {param.__class__.__name__} terms"
            return to_tuple_of_pairs(  # type: ignore
                list(ParameterFormatter.format_param(p) for p in param)
            )
        elif isinstance(param, tuple):
            return ParameterFormatter.format_tuple_of_tuples(param)

        x = param.decode("utf-8") if isinstance(param, bytes) else str(param)
        if x == "":
            return "None"
        return x

    @staticmethod
    def format_group(group_name: str, values: dict[str, Any]) -> str:
        """Format a group of related parameters"""
        if group_name == "resolution":
            nx = ParameterFormatter.format_param(values.get("nx", "N/A"))
            ny = ParameterFormatter.format_param(values.get("ny", "N/A"))
            nz = ParameterFormatter.format_param(values.get("nz", "N/A"))
            return f"({nx} x {ny} x {nz})"
        elif group_name == "spacing":
            dx = ParameterFormatter.format_param(values.get("x1_spacing", "N/A"))
            dy = ParameterFormatter.format_param(values.get("x2_spacing", "N/A"))
            dz = ParameterFormatter.format_param(values.get("x3_spacing", "N/A"))
            return f"(dx1, dx2, dx3) = ({dx}, {dy}, {dz})"
        elif group_name == "domain":
            x1 = ParameterFormatter.format_param(values.get("x1bounds", "N/A"))
            x2 = ParameterFormatter.format_param(values.get("x2bounds", "N/A"))
            x3 = ParameterFormatter.format_param(values.get("x3bounds", "N/A"))
            return f"X1: {x1}, X2: {x2}, X3: {x3}"
        else:
            return ", ".join(
                f"{p}={ParameterFormatter.format_param(v)}" for p, v in values.items()
            )


class AsciiArtGenerator:
    """Generates ASCII art for parameter summary display"""

    @staticmethod
    def get_header() -> str:
        """Get the ASCII art header for the parameter summary"""
        return r"""
        .---------------------------------------------------------------.
        |                                                               |
        |   ____  _           _     _   ____                            |
        |  / ___|(_)_ __ ___ | |__ (_) |  _ \ __ _ _ __ __ _ _ __ ___   |
        |  \___ \| | '_ ` _ \| '_ \| | | |_) / _` | '__/ _` | '_ ` _ \  |
        |   ___) | | | | | | | |_) | | |  __/ (_| | | | (_| | | | | | | |
        |  |____/|_|_| |_| |_|_.__/|_| |_|   \__,_|_|  \__,_|_| |_| |_| |
        |                                                               |
        '---------------------------------------------------------------'
        """

    @staticmethod
    def get_footer() -> str:
        """Get the ASCII art footer for the parameter summary"""
        return "\n".join(
            [
                "+" + "=" * 78 + "+",
                "|" + " " * 78 + "|",
                "|" + "  End of Simulation Parameters  ".center(78, " ") + "|",
                "|" + " " * 78 + "|",
                "+" + "=" * 78 + "+",
            ]
        )

    @staticmethod
    def get_dimension_art(dimensionality: int) -> str:
        """Get dimension-specific ASCII art"""
        if dimensionality == 1:
            return r"|----|"
        elif dimensionality == 2:
            return r"|####|"
        else:  # 3D
            return r"/--\|"

    @staticmethod
    def get_physics_art(regime: str) -> str:
        """Get physics-specific ASCII art based on regime"""
        if "mhd" in regime.lower():
            return r" |~B| "  # Magnetic field
        elif "relativistic" in regime.lower():
            return r" |c^2| "  # Speed of light squared
        else:
            return r" |*| "  # Default physics


class SummaryStatistics:
    """Computes and formats summary statistics for the simulation"""

    @staticmethod
    def estimate_memory_usage(sim_state: dict[str, Any]) -> str:
        """Estimate memory usage based on grid size and variables"""
        nx = sim_state["nx"]
        ny = sim_state["ny"]
        nz = sim_state["nz"]

        nvars: int
        if "nvars" in sim_state:
            nvars = sim_state["nvars"]
        elif "is_mhd" in sim_state and sim_state["is_mhd"]:
            nvars = 9  # MHD + passive scalar
        else:
            nvars = sim_state["dimensionality"] + 3  # hydro var + passive scalar

        # alc memory (8 bytes per double)
        cells = nx * ny * nz
        bytes_per_cell = nvars * 8  # 8 bytes per double
        if "is_mhd" in sim_state and sim_state["is_mhd"]:
            narrays = (
                3 + 3 + 2
            )  # 3 staggered B fields + 3 flux arrays + 1 conserved + 1 primitive
        elif sim_state["regime"] == "srhd":
            narrays = 2 + 1  # 1 conserved + 1 primitive + 1 pressure_guess
        else:
            narrays = 1 + 1  # 1 conserved + 1 primitive
        total_bytes = cells * bytes_per_cell * narrays

        # format in appropriate units
        if total_bytes < 1024:
            return f"{total_bytes} B"
        elif total_bytes < 1024**2:
            return f"{total_bytes / 1024:.1f} KB"
        elif total_bytes < 1024**3:
            return f"{total_bytes / 1024**2:.1f} MB"
        else:
            return f"{total_bytes / 1024**3:.2f} GB"

    @staticmethod
    def compute_cell_metrics(sim_state: dict[str, Any]) -> dict[str, str]:
        """Compute metrics about grid cells"""
        metrics = {}

        spacings = []
        for i in range(3):
            bounds = sim_state[f"x{i + 1}bounds"]
            resx = sim_state[f"n{'x' if i == 0 else 'y' if i == 1 else 'z'}"]
            if resx == 1:
                spacings.append(0.0)
                continue
            if sim_state[f"x{i + 1}_spacing"] == "linear":
                dx = (bounds[1] - bounds[0]) / (resx - 1)
            else:
                dlogx = np.log10(bounds[1] / bounds[0]) / (resx - 1)
                dx = bounds[0] * (10**dlogx - 1) / (10**dlogx - 1)
            spacings.append(dx)
        dx, dy, dz = spacings

        # calc aspect ratio
        max_spacing = max(dx, dy, dz)
        if max_spacing > 0:
            aspect_x = dx / max_spacing
            aspect_y = dy / max_spacing
            aspect_z = dz / max_spacing
            metrics["aspect_ratio"] = f"{aspect_x:.1f}:{aspect_y:.1f}:{aspect_z:.1f}"

        # min/max cell size
        min_spacing = min(s for s in [dx, dy, dz] if s > 0)
        metrics["min_cell"] = f"{min_spacing:.6f}"
        metrics["max_cell"] = f"{max_spacing:.6f}"

        # CFL timestep bound estimate
        cfl = sim_state["cfl"]
        sound_speed = 1.0  # default assumption
        if "ambient_sound_speed" in sim_state:
            sound_speed = sim_state["ambient_sound_speed"]
        dt_estimate = cfl * min_spacing / sound_speed
        metrics["dt_estimate"] = f"{dt_estimate:.6f}"

        return metrics


class CategoryRenderer:
    """Renders parameter categories and their parameters"""

    def __init__(self, formatter: Optional[ParameterFormatter] = None):
        self.formatter = formatter or ParameterFormatter()

    def render_category_header(self, title: str, ascii_art: str) -> None:
        """Render a category header with ASCII art"""
        logger.info("\n" + "+" + "-" * 78 + "+")
        logger.info(f"| {ascii_art} {title}")
        logger.info("+" + "-" * 78 + "+")

    def render_parameter(
        self, name: str, value: str, description: Optional[str] = None
    ) -> None:
        """Render a single parameter with optional description"""
        line = f"| {name.ljust(28, '.')} {value}"
        logger.info(line)

        # If there's a description, add it on the next line
        if description:
            desc_line = f"|{' ' * 30}{description}"
            logger.info(desc_line)

    def render_parameter_group(
        self, group_name: str, group_values: dict[str, Any]
    ) -> None:
        """Render a group of related parameters"""
        val_str = self.formatter.format_group(group_name, group_values)
        self.render_parameter(group_name, val_str)

    def render_summary_block(self, summary_data: dict[str, str]) -> None:
        """Render a summary block with key metrics"""
        if not summary_data:
            return

        self.render_category_header("+-S-+ Simulation Overview", "| S |")

        for name, value in summary_data.items():
            self.render_parameter(name, value)


class SimulationParameterSummary:
    """Main class for generating simulation parameter summaries"""

    def __init__(self) -> None:
        self.formatter = ParameterFormatter()
        self.art_generator = AsciiArtGenerator()
        self.stats = SummaryStatistics()
        self.renderer = CategoryRenderer(self.formatter)
        self.excluded_params = ["bfield", "staggered_bfields", "bodies", "body_system"]

    def define_categories(self) -> dict[str, dict[str, Any]]:
        """Define parameter categories and their organization"""
        return {
            "Grid": {
                "title": "+----+ Grid Configuration",
                "ascii": "|####|",
                "params": [
                    "nx",
                    "ny",
                    "nz",
                    "x1_spacing",
                    "x2_spacing",
                    "x3_spacing",
                    "x1bounds",
                    "x2bounds",
                    "x3bounds",
                    "dimensionality",
                    "coord_system",
                    "effective_dimensions",
                ],
                "groups": {
                    "resolution": ["nx", "ny", "nz"],
                    "spacing": ["x1_spacing", "x2_spacing", "x3_spacing"],
                    "domain": ["x1bounds", "x2bounds", "x3bounds"],
                },
                "descriptions": {
                    "coord_system": "Coordinate system for the simulation grid"
                },
            },
            "Physics": {
                "title": "*/|\\* Physics Settings",
                "ascii": " |*| ",
                "params": [
                    "regime",
                    "adiabatic_index",
                    "gamma",
                    "isothermal",
                    "shakura_sunyaev_alpha",
                    "ambient_sound_speed",
                    "is_mhd",
                    "is_relativistic",
                    "viscosity",
                ],
                "descriptions": {
                    "adiabatic_index": "Ratio of specific heats (gamma)",
                    "viscosity": "Kinematic viscosity coefficient",
                },
            },
            "Numerics": {
                "title": "+---+ Numerical Methods",
                "ascii": "|123|",
                "params": [
                    "solver",
                    "spatial_order",
                    "temporal_order",
                    "plm_theta",
                    "cfl_number",
                    "use_quirk_smoothing",
                    "use_fleischmann_limiter",
                    "order_of_integration",
                ],
                "descriptions": {
                    "plm_theta": "Controls slope limiter (1.0=minmod, 2.0=MC)",
                    "cfl_number": "Courant-Friedrichs-Lewy stability number",
                },
            },
            "Boundaries": {
                "title": "/---\\ Boundary Conditions",
                "ascii": "|<->|",
                "params": ["boundary_conditions"],
            },
            "Output": {
                "title": " ___ Output Configuration",
                "ascii": "|_o_|",
                "params": [
                    "data_directory",
                    "checkpoint_interval",
                    "output_format",
                    "checkpoint_label",
                    "checkpoint_index",
                    "is_restart",
                ],
            },
            "Runtime": {
                "title": " ___ Simulation Runtime",
                "ascii": "|>_>|",
                "params": [
                    "start_time",
                    "end_time",
                    "time",
                    "final_time",
                ],
            },
            "Custom": {
                "title": " /~\\ Custom Expressions",
                "ascii": "|/\\|",
                "params": [
                    "bx1_inner_expressions",
                    "bx1_outer_expressions",
                    "bx2_inner_expressions",
                    "bx2_outer_expressions",
                    "bx3_inner_expressions",
                    "bx3_outer_expressions",
                    "gravity_source_expressions",
                    "hydro_source_expressions",
                    "local_sound_speed_expressions",
                ],
            },
        }

    def generate_summary_statistics(self, sim_state: dict[str, Any]) -> dict[str, str]:
        """Generate summary statistics for the simulation"""
        summary = {}

        nx = sim_state.get("nx", 1)
        ny = sim_state.get("ny", 1)
        nz = sim_state.get("nz", 1)
        total_cells = nx * ny * nz

        if total_cells > 1_000_000:
            summary["Total cells"] = f"{total_cells / 1_000_000:.2f} million"
        else:
            summary["Total cells"] = f"{total_cells:,}"

        # add memory usage estimate
        summary["Est. memory usage"] = self.stats.estimate_memory_usage(sim_state)

        x1_bounds = sim_state.get("x1bounds", (0, 1))
        x2_bounds = sim_state.get("x2bounds", (0, 1))
        x3_bounds = sim_state.get("x3bounds", (0, 1))

        x1_size = abs(x1_bounds[1] - x1_bounds[0])
        x2_size = abs(x2_bounds[1] - x2_bounds[0])
        x3_size = abs(x3_bounds[1] - x3_bounds[0])

        volume = x1_size * x2_size * x3_size
        summary["Physical domain"] = (
            f"({x1_size:.1f} × {x2_size:.1f} × {x3_size:.1f}) = {volume:.1f} cubic units"
        )

        # add cell metrics
        cell_metrics = self.stats.compute_cell_metrics(sim_state)

        if "aspect_ratio" in cell_metrics:
            summary["Cell aspect ratio"] = cell_metrics["aspect_ratio"]
        if "min_cell" in cell_metrics and "max_cell" in cell_metrics:
            summary["Min/max cell size"] = (
                f"{cell_metrics['min_cell']} / {cell_metrics['max_cell']}"
            )
        if "dt_estimate" in cell_metrics:
            summary["CFL timestep est."] = f"~{cell_metrics['dt_estimate']} time units"

        return summary

    def generate_parameter_summary(self, sim_state: dict[str, Any]) -> dict[str, Any]:
        """
        Generate and print a beautifully formatted summary of simulation parameters,
        grouped by logical categories with enhanced visual presentation.

        Parameters:
            sim_state (dict[str, Any]): Simulation state dictionary.

        Returns:
            dict[str, Any]: The unchanged simulation state.
        """
        # print header
        logger.info("\n" + "=" * 80)
        logger.info(self.art_generator.get_header())
        logger.info("=" * 80)

        # generate and display summary statistics
        summary_data = self.generate_summary_statistics(sim_state)
        self.renderer.render_summary_block(summary_data)

        # get categories
        categories = self.define_categories()
        expression_params = categories["Custom"]["params"]

        # process parameters by category
        for category, info in categories.items():
            if category == "Custom":
                # handle expressions separately at the end
                continue

            # check if category has any parameters in sim_state
            has_params = any(param in sim_state for param in info["params"])
            if not has_params:
                continue

            # get dynamic ASCII art if possible
            ascii_art = info["ascii"]
            if category == "Grid" and "dimensionality" in sim_state:
                ascii_art = self.art_generator.get_dimension_art(
                    sim_state["dimensionality"]
                )
            elif category == "Physics" and "regime" in sim_state:
                ascii_art = self.art_generator.get_physics_art(sim_state["regime"])

            # print category header
            self.renderer.render_category_header(info["title"], ascii_art)

            # process parameter groups first
            if "groups" in info:
                for group_name, group_params in info["groups"].items():
                    # check if any parameters in this group exist in sim_state
                    group_values = {}
                    for param in group_params:
                        if param in sim_state:
                            group_values[param] = sim_state[param]

                    if group_values:
                        self.renderer.render_parameter_group(group_name, group_values)

                        # remove processed parameters from the list
                        for param in group_params:
                            if param in info["params"]:
                                info["params"].remove(param)

            # process remaining individual parameters
            descriptions = info.get("descriptions", {})
            for param in info["params"]:
                if (
                    param in sim_state
                    and param not in expression_params
                    and param not in self.excluded_params
                ):
                    val_str = self.formatter.format_param(sim_state[param])
                    description = descriptions.get(param)
                    self.renderer.render_parameter(param, val_str, description)

        # handle expression parameters separately
        has_expressions = any(
            param in sim_state and sim_state[param] for param in expression_params
        )

        if has_expressions:
            info = categories["Custom"]
            self.renderer.render_category_header(info["title"], info["ascii"])

            for param in expression_params:
                if param in sim_state and sim_state[param]:
                    self.renderer.render_parameter(
                        param, "user-defined source expressions"
                    )

        # footer
        logger.info("\n" + self.art_generator.get_footer() + "\n")

        return sim_state


def print_simulation_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Print a summary of the simulation parameters"""
    try:
        # Import locally to avoid dependency issues
        from .rich_summary import print_rich_simulation_parameters

        print_rich_simulation_parameters(params)
    except ImportError:
        # Fall back to the original version if Rich is not available
        summary = SimulationParameterSummary()
        summary_string = summary.generate_parameter_summary(params)
        print(summary_string)
    return params
