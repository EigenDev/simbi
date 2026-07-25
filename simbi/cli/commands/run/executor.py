# =============================================================================
# run/executor.py
#
# executes simulation configs using the new simulation module.
# discovers SimbiProblem subclasses, sets up cli, and runs.
# =============================================================================
from __future__ import annotations

import importlib
import inspect
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional, Sequence

from simbi.simulation import SimbiProblem, run
from simbi.simulation.runner import validate_problem


def _discover_problem_classes(script: str) -> list[tuple[str, type[SimbiProblem]]]:
    """
    import the config module and return its SimbiProblem subclasses in source order.

    discovery imports the module (the same import the run itself performs) and tests
    the real subclass relationship, so a config may inherit from another config in an
    imported module — a config subclassing an imported base is resolved correctly,
    which a base-NAME scan of the file cannot do. only classes DEFINED in the script
    are returned: an imported base config is a SimbiProblem subclass too, but its
    __module__ names its own module, so it is excluded.
    """
    script_path = Path(script).resolve()
    module_name = script_path.stem

    sys.path.insert(0, str(script_path.parent))
    try:
        module = importlib.import_module(module_name)
    finally:
        if str(script_path.parent) in sys.path:
            sys.path.remove(str(script_path.parent))

    def source_line(cls: type) -> int:
        # a class without recoverable source (dynamically built) sorts first.
        try:
            return inspect.getsourcelines(cls)[1]
        except (OSError, TypeError):
            return 0

    found = [
        (name, obj)
        for name, obj in vars(module).items()
        if inspect.isclass(obj)
        and issubclass(obj, SimbiProblem)
        and obj is not SimbiProblem
        and obj.__module__ == module.__name__
    ]
    found.sort(key=lambda pair: source_line(pair[1]))
    return found


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

    problem_classes = _discover_problem_classes(script)

    if not problem_classes:
        from simbi.simulation.problem import ConfigError

        raise ConfigError(
            f"no SimbiProblem subclasses found in {script}. "
            "ensure your config defines a class that inherits from SimbiProblem."
        )

    for class_name, problem_class in problem_classes:
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

        if getattr(args, "validate_only", False):
            validate_problem(problem, compute_mode=args.compute_mode)
            continue

        # run the simulation. no preamble: the live dashboard is a self-contained
        # full-screen tool that owns the terminal for the duration of the run.
        run(
            problem,
            compute_mode=args.compute_mode,
            live_monitor=getattr(args, "live_monitor", False),
        )


def _configure_environment(args: Namespace) -> None:
    """configure environment variables for simulation."""
    if hasattr(args, "nthreads") and args.nthreads:
        os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
        os.environ["NTHREADS"] = str(args.nthreads)

    if args.compute_mode == "omp":
        os.environ["USE_OMP"] = "1"

    # gpu block dims are set by RegisterGPUBlockDimensions action
