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

from pathlib import Path

from simbi.simulation.launcher import Layout, retire_rendezvous, scheduler_identity, scheduler_rendezvous, spawn_workers
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


def _run_worker_role(args: Namespace, problem, worker: int, workers: int, owner, cuts, rendezvous: str, credential, bind: str, advertise: str, staging_cells: int) -> None:
    from simbi.simulation import runner

    runner.launch_worker(
        problem,
        worker=worker,
        workers=workers,
        owner=owner,
        cuts=cuts,
        rendezvous=rendezvous,
        credential=credential,
        bind=bind,
        advertise=advertise,
        staging_cells=staging_cells,
        max_steps=args.max_steps,
    )


def launch_config(args: Namespace, argv: Optional[Sequence[str]] = None) -> None:
    if args.worker is not None and args.owner is not None:
        # the worker role as the local launcher spawns it: every parameter on the command
        # line. the launcher forwards its whole command line, the layout flag included, so
        # the role is decided by the worker-role flags alone.
        problem = _build_problem(args, argv)
        if problem is None:
            return
        owner = [int(x) for x in args.owner.split(",")]
        cuts = [[int(c) for c in axis.split(",") if c] for axis in args.cuts.split(";")] if args.cuts else []
        _run_worker_role(
            args, problem, args.worker, args.workers, owner, cuts, args.rendezvous, args.credential,
            args.bind or "127.0.0.1", args.advertise or args.bind or "127.0.0.1", args.staging_cells,
        )
        return
    problem = _build_problem(args, argv)
    if problem is None:
        return
    if args.layout is None:
        raise ConfigError("a layout is required.  usage: simbi launch <config> --layout <layout.toml>")
    layout = Layout.from_file(args.layout, problem.resolution)
    if layout.mode == "scheduler":
        # the worker role as a scheduler started it: identity from the scheduler, the session
        # credential from the coordinator's restricted rendezvous file
        if args.worker is not None:
            worker, workers = args.worker, args.workers or layout.workers
        else:
            worker, workers = scheduler_identity(layout)
        rendezvous = args.rendezvous or str(scheduler_rendezvous(layout, problem.data_directory))
        if layout.checkpoint_directory:
            problem = problem.model_copy(update={"data_directory": layout.checkpoint_directory})
        try:
            _run_worker_role(
                args, problem, worker, workers, layout.owner, layout.cuts, rendezvous, args.credential,
                layout.bind, layout.advertised, layout.staging_cells,
            )
        finally:
            # the coordinator wrote the record; it retires it once its session has ended
            if worker == 0:
                retire_rendezvous(Path(rendezvous), layout.keep_rendezvous)
        return
    code = spawn_workers(layout, sys.argv[1:], problem.data_directory)
    if code != 0:
        sys.exit(code)
