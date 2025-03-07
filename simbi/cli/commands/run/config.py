import ast
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Tuple, Set
from argparse import ArgumentParser, Namespace
from ....simulator import Hydro
from ....detail import bcolors
from ....core.config.base_config import BaseConfig


def _build_inheritance_graph(root: ast.Module) -> Dict[str, Set[str]]:
    """Build graph of class inheritance relationships"""
    inheritance_graph = {}

    for node in root.body:
        if isinstance(node, ast.ClassDef):
            # Get all base classes for this class
            bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
            inheritance_graph[node.name] = set(bases)

    return inheritance_graph


def _get_derived_classes(
    graph: Dict[str, Set[str]], base_class: str = "BaseConfig"
) -> Set[str]:
    """Find all classes that inherit from base_class directly or indirectly"""
    derived = set()

    def visit(class_name: str) -> None:
        # Find all classes that directly inherit from this class
        direct_children = {name for name, bases in graph.items() if class_name in bases}
        # Add them to our result set
        derived.update(direct_children)
        # Recursively visit each child
        for child in direct_children:
            visit(child)

    # Start search from base class
    visit(base_class)
    return derived


def _get_setup_classes(script: str) -> List[str]:
    """Extract all classes that inherit from BaseConfig directly or indirectly"""
    with open(script) as setup_file:
        root = ast.parse(setup_file.read())

    # Build inheritance relationships
    inheritance_graph = _build_inheritance_graph(root)

    # Find all derived classes
    setup_classes = _get_derived_classes(inheritance_graph)

    return sorted(setup_classes)  # Sort for deterministic order


def _configure_single_state(
    base_script: str,
    setup_class: str,
    parser: ArgumentParser,
    args: Namespace,
    argv: Optional[Sequence],
) -> Tuple[Hydro, Dict[str, Any], str]:
    """Configure single hydro state"""

    # Import problem class
    problem_class_t = getattr(
        importlib.import_module(f"{base_script}"), f"{setup_class}"
    )

    # check if the user has passed any non-void arguments
    run_parser = getattr(args, "active_parser")
    problem_class_t.setup_cli(parser, run_parser)
    problem_class_t.parse_args_and_update_configuration()

    problem_class: Any = problem_class_t()

    if args.peek:
        print(
            f"{bcolors.YELLOW}Printing dynamic arguments in {setup_class}{bcolors.ENDC}"
        )
        return None, {}, ""

    _setup_logging(problem_class, args)

    # Create hydro state
    state: Hydro = Hydro(problem_class)
    kwarg_dict = _build_kwargs_dict(problem_class, args)

    return (
        state,
        kwarg_dict,
        problem_class.__doc__ or f"No docstring: {setup_class}",
    )


def _setup_logging(config: BaseConfig, args: Namespace) -> None:
    """Setup logging configuration"""
    if args.log_output:
        config.log_output = True
        config.set_logdir(
            args.log_directory or args.data_directory or config.data_directory
        )
    config.trace_memory = args.trace_mem


def use_arg_or_default(arg_value, config_value):
    """Use arg value if provided, otherwise fallback to config value"""
    return arg_value if arg_value is not None else config_value


def _build_kwargs_dict(config: BaseConfig, args: Namespace) -> Dict[str, Any]:
    """Build kwargs dictionary for simulation"""
    spatial_order = args.spatial_order
    temporal_order = args.time_order
    if config.order_of_integration == "first" or args.order == "first":
        spatial_order = "pcm"
        temporal_order = "rk1"
    elif config.order_of_integration == "second" or args.order == "second":
        spatial_order = "plm"
        temporal_order = "rk2"
    elif config.order_of_integration is not None:
        raise ValueError("Order of integration must be 'first' or 'second'")

    return {
        "spatial_order": spatial_order or config.spatial_order,
        "temporal_order": temporal_order or config.temporal_order,
        "cfl": args.cfl,
        "checkpoint_interval": args.checkpoint_interval,
        "tstart": args.tstart,
        "tend": args.tend,
        "solver": args.solver,
        "plm_theta": args.plm_theta,
        "data_directory": args.data_directory,
        "quirk_smoothing": args.quirk_smoothing,
        "compute_mode": args.compute_mode,
    }


def configure_state(
    args: Namespace, argv: Optional[Sequence]
) -> tuple[List[Hydro], Dict[int, Dict[str, Any]], List[str]]:
    """Configure hydro state from setup script"""
    parser = getattr(args, "main_parser")
    script = args.setup_script
    script_dirname = Path(script).parent
    base_script = Path(script).stem
    sys.path.insert(1, f"{script_dirname}")

    setup_classes = _get_setup_classes(script)
    if not setup_classes:
        raise ValueError("Invalid simbi configuration")

    states = []
    state_docs = []
    kwargs = {}

    for idx, setup_class in enumerate(setup_classes):
        state, kwarg_dict, doc = _configure_single_state(
            base_script, setup_class, parser, args, argv
        )
        states.append(state)
        kwargs[idx] = kwarg_dict
        state_docs.append(doc)

    return states, kwargs, state_docs
