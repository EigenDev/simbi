import os
from typing import Sequence
from argparse import Namespace
from ....simulator import Hydro
from .... import logger


def run_simulation(
    states: Sequence[Hydro],
    state_docs: Sequence[str],
    args: Namespace,
) -> None:
    """Run simulation with configured states"""
    _configure_environment(args)

    for idx, sim_state in enumerate(states):
        logger.info("=" * 80)
        logger.info(state_docs[idx])
        logger.info("=" * 80)
        sim_state.simulate(compute_mode=args.compute_mode)


def _configure_environment(args: Namespace) -> None:
    """Configure environment variables"""
    if args.nthreads:
        os.environ["OMP_NUM_THREADS"] = f"{args.nthreads}"
        os.environ["NTHREADS"] = f"{args.nthreads}"

    if args.compute_mode == "omp":
        os.environ["USE_OMP"] = "1"

    for coord, block in zip(["X", "Y", "Z"], args.gpu_block_dims):
        os.environ[f"GPU{coord}BLOCK_SIZE"] = str(block)
