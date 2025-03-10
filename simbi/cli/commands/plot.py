from argparse import ArgumentParser, Namespace
from typing import Optional
from ...tools.visualization.core.plot_parser import PlottingArgumentBuilder
from ..utils.formatter import HelpFormatter


def setup_parser(subparsers: ArgumentParser) -> None:
    """Setup plot command parser"""
    plot_parser = subparsers.add_parser(
        "plot",
        help="plots the given simbi checkpoint file",
        formatter_class=HelpFormatter,
        usage="simbi plot <checkpoints> <setup_name> [options]",
    )
    plotting_args = PlottingArgumentBuilder()
    plotting_args.add_to_subparser(plot_parser)
    plot_parser.set_defaults(func=execute)


def execute(args: Namespace, argv: Optional[list] = None) -> None:
    """Execute plot command"""
    from ...tools.visual import visualize

    config = PlottingArgumentBuilder.get_config(args)
    visualize(config)
