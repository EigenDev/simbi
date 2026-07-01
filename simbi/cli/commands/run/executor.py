# =============================================================================
# run/executor.py
#
# executes simulation configs using the new simulation module.
# discovers SimbiProblem subclasses, sets up cli, and runs.
# =============================================================================
from __future__ import annotations

import ast
import importlib
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional, Sequence

from simbi.simulation import SimbiProblem, run


def _get_problem_classes(script: str) -> list[str]:
    """
    extract all classes that inherit from SimbiProblem.
    uses ast parsing to avoid importing the module twice.
    """
    with open(script) as f:
        root = ast.parse(f.read())

    # build inheritance graph
    graph: dict[str, set[str]] = {}
    for node in root.body:
        if isinstance(node, ast.ClassDef):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            graph[node.name] = set(bases)

    # find all classes deriving from SimbiProblem (directly or indirectly)
    def is_problem_class(name: str, visited: set[str] | None = None) -> bool:
        if visited is None:
            visited = set()
        if name in visited:
            return False
        visited.add(name)

        if name == "SimbiProblem":
            return True

        bases = graph.get(name, set())
        return any(is_problem_class(b, visited) for b in bases)

    return [
        name
        for name in graph
        if is_problem_class(name) and name != "SimbiProblem"
    ]


def _load_problem_class(script: str, class_name: str) -> type[SimbiProblem]:
    """dynamically import and return a problem class."""
    script_path = Path(script).resolve()
    module_name = script_path.stem

    # add script directory to path
    sys.path.insert(0, str(script_path.parent))

    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    finally:
        # clean up sys.path
        if str(script_path.parent) in sys.path:
            sys.path.remove(str(script_path.parent))


def run_config(args: Namespace, argv: Optional[Sequence[str]] = None) -> None:
    """
    run a simulation config.

    1. discovers problem classes in the config file
    2. for each class, sets up cli params and creates instance
    3. runs the simulation
    """
    script = args.config_script
    # the active subparser (run parser) is used to register the config's cli params and,
    # when no config is given, to print the generic run help.
    active_parser = getattr(args, "active_parser", None)

    # no config supplied: `simbi run --help/--peek/--info` -> generic run help (exit 0);
    # a bare `simbi run` -> a helpful error (the config is required).
    if script is None:
        if active_parser is not None:
            active_parser.print_help()
        if not getattr(args, "info", False):
            print(
                "\nerror: a config is required.  usage: simbi run <config> [options]\n"
                "       list configs:        simbi run --configs\n"
                "       peek a config's flags: simbi run <config> --help",
                file=sys.stderr,
            )
            sys.exit(2)
        return

    problem_classes = _get_problem_classes(script)

    if not problem_classes:
        raise ValueError(
            f"no SimbiProblem subclasses found in {script}. "
            "ensure your config defines a class that inherits from SimbiProblem."
        )

    for class_name in problem_classes:
        # load the class
        problem_class = _load_problem_class(script, class_name)

        # setup cli params from the problem class
        if active_parser is not None:
            problem_class.setup_cli(active_parser)

        # show help if --info flag
        if args.info:
            print(f"\n{class_name} parameters:")
            print("=" * 60)
            if (
                hasattr(problem_class, "_cli_parser")
                and problem_class._cli_parser
            ):
                problem_class._cli_parser.print_help()
            else:
                # show field info manually
                for name, info in problem_class.model_fields.items():
                    if not name.startswith("_"):
                        default = (
                            info.default
                            if info.default is not ...
                            else "required"
                        )
                        desc = info.description or ""
                        print(f"  {name}: {default}  # {desc}")
            continue

        # create instance from cli args
        problem = problem_class.from_cli(argv, args)

        # set checkpoint if provided
        if args.checkpoint:
            # use model_copy to create new instance with checkpoint
            problem = problem.model_copy(
                update={"checkpoint_file": args.checkpoint}
            )

        # configure environment
        _configure_environment(args)

        # run the simulation. no preamble: the live dashboard is a self-contained
        # full-screen tool that owns the terminal for the duration of the run.
        run(problem, compute_mode=args.compute_mode)


def _configure_environment(args: Namespace) -> None:
    """configure environment variables for simulation."""
    if hasattr(args, "nthreads") and args.nthreads:
        os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
        os.environ["NTHREADS"] = str(args.nthreads)

    if args.compute_mode == "omp":
        os.environ["USE_OMP"] = "1"

    # gpu block dims are set by RegisterGPUBlockDimensions action
