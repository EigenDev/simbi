import os
from typing import List, Dict
from argparse import Namespace
from ....simulator import Hydro
from .... import logger
from typing import Any

def run_simulation(
    states: List[Hydro],
    kwargs: Dict[int, Dict[str, Any]],
    state_docs: List[str],
    args: Namespace
) -> None:
    """Run simulation with configured states"""
    _configure_environment(args)
    
    for idx, sim_state in enumerate(states):
        logger.info("=" * 80)
        logger.info(state_docs[idx])
        logger.info("=" * 80)
        sim_state.simulate(**kwargs[idx])
        
def _configure_environment(args: Namespace) -> None:
    """Configure environment variables"""
    if args.nthreads:
        os.environ['OMP_NUM_THREADS'] = f'{args.nthreads}'
        os.environ['NTHREADS'] = f'{args.nthreads}'
        
    if args.compute_mode == 'omp':
        os.environ['USE_OMP'] = "1"
        
    for coord, block in zip(['X', 'Y', 'Z'], args.gpu_block_dims):
        os.environ[f'GPU{coord}BLOCK_SIZE'] = str(block)