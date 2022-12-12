import argparse
from typing import List
import os 
import ast
import sys
import importlib
from simbi import Hydro
from pathlib import Path

class CustomParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        self.print_help()
        sys.exit(2)

class print_the_version(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string, **kwargs):
        from simbi import __version__ as version
        print(f"SIMBI version {version}")
        parser.exit()
        
def parse_arguments(cli_args: List[str] = None) -> argparse.Namespace:
    parser = CustomParser(prog='simbi', usage='%(prog)s <setup_script> [options]', description="Relativistic gas dynamics module")
    parser.add_argument('setup_script', help='setup script for simulation run', type=valid_pyscript)
    parser.add_argument('--tstart',    help='start time for simulation', default=None, type=float)
    parser.add_argument('--tend',    help='end time for simulation', default=None, type=float)
    parser.add_argument('--dlogt',     help='logarithmic time bin spacing for checkpoints', default=0.0, type=float)
    parser.add_argument('--plm_theta', help='piecewise linear consturction parameter', default=1.5, type=float)
    parser.add_argument('--first_order', help='Set flag if wanting first order accuracy in solution', default=False, action='store_true')
    parser.add_argument('--cfl', help='Courant-Friedrichs-Lewy stability number', default=0.1, type=float)
    parser.add_argument('--hllc', help='flag for HLLC computation as opposed to HLLE', default=False, action='store_true')
    parser.add_argument('--chkpt', help='checkpoint file to restart computation from', default=None, type=str)
    parser.add_argument('--chkpt_interval', help='checkpoint interval spacing in simulation time units', default=0.1, type=float)
    parser.add_argument('--data_directory', help='directory to save checkpoint files', default='data/', type=str)
    parser.add_argument('--boundary_condition', help='boundary condition for inner boundary', default='outflow', choices=['reflecting', 'outflow', 'inflow', 'periodic'])
    parser.add_argument('--engine_duration', help='duration of hydrodynamic source terms', default=0.0, type=float)
    parser.add_argument('--mode', help='execution mode for computation', default='cpu', choices=['cpu', 'gpu'], dest='compute_mode')
    parser.add_argument('--quirk_smoothing', help='flag to activate Quirk (1994) smoothing at poles', default=False, action='store_true')
    parser.add_argument('--version','-V', help='print current version of simbi module', action=print_the_version)
    parser.add_argument('--nthreads', '-p', help="number of omp threads to run at", type=max_thread_count, default=None)
    parser.add_argument('--peek', help='print setup-script usage', default=False, action='store_true')
    
    # print help message if no args supplied
    return parser, parser.parse_known_args(args=None if sys.argv[1:] else ['--help'])

configs_path = Path('configs').resolve()
overideable_args = ['tstart', 'tend', 'hllc', 'boundary_condition', 'plm_theta', 'dlogt', 'data_directory', 'quirk_smoothing']
def valid_pyscript(param):
    base, ext = os.path.splitext(param)
    if ext.lower() != '.py':
        param = None
        hard_configs = [file for file in configs_path.rglob('*.py')]
        soft_paths   = [soft_path for soft_path in (Path('configs')).glob("*") if soft_path.is_symlink()]   
        soft_configs = [file for path in soft_paths for file in path.rglob('*.py')]
        for file in hard_configs + soft_configs:
            if base == Path(file).stem:
                param = file
        if not param:
            raise argparse.ArgumentTypeError('<script_file> must have a .py extension or exist in the configs directory')
    return param

def max_thread_count(param) -> int:
    import multiprocessing
    num_threads_available = multiprocessing.cpu_count()
    
    try:
        val = int(param)
    except ValueError:    
        raise argparse.ArgumentTypeError("\nMust be a integer\n")
    
    if val > num_threads_available:
        raise argparse.ArgumentTypeError(f'\nTrying to set thread count greater than available compute core(s) equal to {num_threads_available}\n')
    
    return val

def configure_state(script: str, parser: argparse.ArgumentParser, argv = None):
    """
    Configure the Hydro state based on the Config class that exists in the passed
    in setup script. Once configured, pass it back to main to be simulated 
    """
    import sys 
    script_dirname = os.path.dirname(script)
    base_script    = Path(os.path.abspath(script)).stem
    sys.path.insert(1, f'{script_dirname}')
    
    with open(script) as setup_file:
        root = ast.parse(setup_file.read())
    
    setup_classes = []
    for node in root.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if base.id == 'BaseConfig':
                    setup_classes += [node.name]
                elif base.id in setup_classes:
                    # if the setup class inherited from another setup class
                    # then we already know it is a descendant of the BaseConfig
                    setup_classes += [node.name]
    
    states = []
    state_docs = []
    kwargs = {}
    peek_only = False
    for idx, setup_class in enumerate(setup_classes):
        problem_class = getattr(importlib.import_module(f'{base_script}'), f'{setup_class}')
        static_config = problem_class
        if argv:
            static_config.parse_args(parser)
        
        if getattr(parser.parse_args(), 'peek'):
            print(f"Printing dynamic arguments present in -- {setup_class}")
            static_config.print_problem_params()
            peek_only = True
            continue
            
        # Call initializer once static vars modified
        config = static_config()
        
        if config.__doc__:
            state_docs += [f"{config.__doc__}"]
        else:
            state_docs += [f"No docstring for problem class: {setup_class}"]
        state: Hydro = Hydro.gen_from_setup(config)
        kwargs[idx] = {}
        kwargs[idx]['tstart']                   = config.default_start_time
        kwargs[idx]['tend']                     = config.default_end_time
        kwargs[idx]['hllc']                     = config.use_hllc_solver 
        kwargs[idx]['boundary_condition']       = config.boundary_condition
        kwargs[idx]['plm_theta']                = config.plm_theta
        kwargs[idx]['dlogt']                    = config.dlogt
        kwargs[idx]['data_directory']           = config.data_directory
        kwargs[idx]['linspace']                 = config.linspace 
        kwargs[idx]['sources']                  = config.sources 
        kwargs[idx]['passive_scalars']          = config.passive_scalars 
        kwargs[idx]['scale_factor']             = config.scale_factor 
        kwargs[idx]['scale_factor_derivative']  = config.scale_factor_derivative
        kwargs[idx]['edens_outer']              = config.edens_outer
        kwargs[idx]['mom_outer']                = config.mom_outer 
        kwargs[idx]['dens_outer']               = config.dens_outer 
        kwargs[idx]['quirk_smoothing']          = config.use_quirk_smoothing
        states.append(state)
    
    if peek_only:
        exit(0)
    
    return states, kwargs, state_docs 
        
def main(parser: argparse.ArgumentParser = parse_arguments()[0], args: argparse.Namespace = parse_arguments()[1][0], argv: List = parse_arguments()[1][1]):
    sim_states, kwargs, state_docs  = configure_state(args.setup_script, parser, argv)
    if args.nthreads:
        os.environ['OMP_NUM_THREADS'] = f'{args.nthreads}'
    
    global_nonsim_args = ['setup_script', 'nthreads', 'peek']
    for idx, sim_state in enumerate(sim_states):
        for arg in vars(args):
            if arg in global_nonsim_args:
                continue
            if arg == 'tend' or arg == 'tstart':
                command_line_time = getattr(args, arg)
                # override the default time args if they've been set
                if command_line_time:
                    kwargs[idx][arg] = command_line_time
                elif not kwargs[idx][arg]:
                   kwargs[idx][arg] = 1.0 if arg == 'tend' else 0.0
                continue
            if arg in overideable_args and kwargs[idx][arg]:
                continue 
            
            kwargs[idx][arg] = getattr(args, arg)
        print("="*80, flush=True)
        print(state_docs[idx], flush=True)
        print("="*80, flush=True)
        sim_state.simulate(**kwargs[idx])
    
    
if __name__ == '__main__':
    sys.exit(main(*parse_arguments()))
    