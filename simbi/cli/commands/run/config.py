import ast
import sys
import importlib
from pathlib import Path
from typing import Sequence, Any, Optional, Set
from argparse import ArgumentParser, Namespace
from ....simulator import Hydro
from ....detail import bcolors
from ....core.config.base_config import SimbiBaseConfig
from ...utils.type_checker import type_check_input


def _build_inheritance_graph(root: ast.Module) -> dict[str, Set[str]]:
    """Build graph of class inheritance relationships"""
    inheritance_graph = {}

    for node in root.body:
        if isinstance(node, ast.ClassDef):
            # Get all base classes for this class
            bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
            inheritance_graph[node.name] = set(bases)

    return inheritance_graph


def _get_derived_classes(
    graph: dict[str, Set[str]], base_class: str = "SimbiBaseConfig"
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


def _get_setup_classes(script: str) -> Sequence[str]:
    """Extract all classes that inherit from SimbiBaseConfig."""
    with open(script) as setup_file:
        root = ast.parse(setup_file.read())

    # Build inheritance relationships
    inheritance_graph = _build_inheritance_graph(root)

    # Find all derived classes of SimbiBaseConfig
    setup_classes = _get_derived_classes(inheritance_graph, "SimbiBaseConfig")

    return sorted(setup_classes)


def _configure_single_state(
    base_script: str,
    setup_class: str,
    parser: ArgumentParser,
    args: Namespace,
    argv: Optional[Sequence],
) -> tuple[Optional[Hydro], str]:
    """Configure single hydro state"""

    # Import problem class
    problem_class_t = getattr(
        importlib.import_module(f"{base_script}"), f"{setup_class}"
    )

    # Setup CLI
    run_parser = getattr(args, "active_parser")
    problem_class_t.setup_cli(run_parser)

    # Create an instance of the config class
    problem_class = problem_class_t.from_cli(parser)

    if args.info:
        print(f"{bcolors.YELLOW}Printing parameters in {setup_class}{bcolors.ENDC}")
        problem_class.cli_parser.print_help()
        return None, ""

    # Set checkpoint file if provided
    if args.checkpoint is not None:
        problem_class.checkpoint_file = args.checkpoint

    # Create hydro state
    state = Hydro(problem_class)

    return (
        state,
        problem_class.__doc__ or f"No docstring: {setup_class}",
    )


def configure_state(
    args: Namespace, argv: Optional[Sequence]
) -> tuple[Sequence[Hydro], Sequence[str]]:
    """Configure hydro state from setup script"""
    parser = getattr(args, "main_parser")
    script = args.setup_script
    script_dirname = Path(script).parent
    base_script = Path(script).stem
    sys.path.insert(1, f"{script_dirname}")

    setup_classes = _get_setup_classes(script)
    if not setup_classes:
        raise ValueError(
            "Invalid simbi configuration - no classes that extend SimbiBaseConfig found"
        )

    if args.type_check:
        type_check_input(script)

    states = []
    state_docs = []

    for idx, setup_class in enumerate(setup_classes):
        state, doc = _configure_single_state(
            base_script, setup_class, parser, args, argv
        )
        states.append(state)
        state_docs.append(doc)

    if args.info:
        sys.exit(0)

    return states, state_docs
