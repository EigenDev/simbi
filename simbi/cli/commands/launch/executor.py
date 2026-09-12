# =============================================================================
# simbi/cli/commands/launch/executor.py
#
# the two roles of `simbi launch`. the launcher role reads the layout, decides
# the partition and the placement, and spawns one worker process per worker
# with the same command line plus the worker-role flags; it waits for every
# worker and exits nonzero if any did. the worker role constructs the problem
# exactly as `simbi run` would and evolves its tiles through the fabric.
#
# usage:
#   launch_config(args, argv)
# =============================================================================
import sys
from argparse import Namespace
from typing import Optional, Sequence

from simbi.simulation.launcher import Layout, spawn_workers
from simbi.simulation.problem import ConfigError

from ..run.executor import _discover_problem_classes


def _build_problem(args: Namespace, argv: Optional[Sequence[str]]):
    """the problem instance, built the way `simbi run` builds it."""
    script = args.config_script
    if script is None:
        raise ConfigError("a config is required.  usage: simbi launch <config> --layout <layout.toml>")
    problem_classes = _discover_problem_classes(script)
    if not problem_classes:
        raise ConfigError(f"no SimbiProblem subclasses found in {script}")
    if len(problem_classes) > 1:
        names = ", ".join(name for name, _ in problem_classes)
        raise ConfigError(f"{script} defines several problems ({names}); launch evolves one")
    class_name, problem_class = problem_classes[0]
    active_parser = getattr(args, "active_parser", None)
    if active_parser is not None:
        problem_class.setup_cli(active_parser)
    if args.info:
        print(f"\n{class_name} parameters:")
        if getattr(problem_class, "_cli_parser", None):
            problem_class._cli_parser.print_help()
        return None
    problem = problem_class.from_cli(argv, args)
    if args.checkpoint:
        problem = problem.model_copy(update={"checkpoint_file": args.checkpoint})
    return problem


def launch_config(args: Namespace, argv: Optional[Sequence[str]] = None) -> None:
    if args.worker is not None:
        from simbi.simulation import runner

        problem = _build_problem(args, argv)
        if problem is None:
            return
        owner = [int(x) for x in args.owner.split(",")]
        cuts = [[int(c) for c in axis.split(",") if c] for axis in args.cuts.split(";")] if args.cuts else []
        runner.launch_worker(
            problem,
            worker=args.worker,
            workers=args.workers,
            owner=owner,
            cuts=cuts,
            rendezvous=args.rendezvous,
            credential=args.credential,
            staging_cells=args.staging_cells,
            max_steps=args.max_steps,
        )
        return
    problem = _build_problem(args, argv)
    if problem is None:
        return
    if args.layout is None:
        raise ConfigError("a layout is required.  usage: simbi launch <config> --layout <layout.toml>")
    layout = Layout.from_file(args.layout, problem.resolution)
    code = spawn_workers(layout, sys.argv[1:], problem.data_directory)
    if code != 0:
        sys.exit(code)
