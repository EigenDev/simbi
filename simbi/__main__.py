import argparse
from typing import List
import os 
import ast
import sys
import subprocess
import importlib
from simbi import Hydro
from pathlib import Path
from .key_types import Optional, Sequence 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
class CustomParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        if 'configurations' not in message:
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
    overridable = parser.add_argument_group('override', 'overridable simuations options')
    global_args = parser.add_argument_group('globals', 'global module-specific options')
    onthefly    = parser.add_argument_group('onthefly', 'simulation otions that are given on the fly')
    overridable.add_argument('--tstart',    help='start time for simulation', default=None, type=float)
    overridable.add_argument('--tend',    help='end time for simulation', default=None, type=float)
    overridable.add_argument('--dlogt',     help='logarithmic time bin spacing for checkpoints', default=None, type=float)
    overridable.add_argument('--plm_theta', help='piecewise linear consturction parameter', default=None, type=float)
    overridable.add_argument('--first_order', help='Set flag if wanting first order accuracy in solution', default=None, action='store_true')
    overridable.add_argument('--cfl', help='Courant-Friedrichs-Lewy stability number', default=None, type=float)
    overridable.add_argument('--hllc', help='flag for HLLC computation as opposed to HLLE', default=None, action=argparse.BooleanOptionalAction)
    overridable.add_argument('--chkpt_interval', help='checkpoint interval spacing in simulation time units', default=None, type=float)
    overridable.add_argument('--data_directory', help='directory to save checkpoint files', default='data/', type=str)
    overridable.add_argument('--boundary_conditions', help='boundary condition for inner boundary', default=None, nargs="+", choices=['reflecting', 'outflow', 'inflow', 'periodic'])
    overridable.add_argument('--engine_duration', help='duration of hydrodynamic source terms', default=None, type=float)
    overridable.add_argument('--quirk_smoothing', help='flag to activate Quirk (1994) smoothing at poles', default=None, action='store_true')
    overridable.add_argument('--constant_sources', help='flag to indicate source terms provided are constant', default=None, action='store_true')
    onthefly.add_argument('--mode', help='execution mode for computation', default='cpu', choices=['cpu', 'gpu'], dest='compute_mode')
    onthefly.add_argument('--chkpt', help='checkpoint file to restart computation from', default=None, type=str)
    global_args.add_argument('--version','-V', help='print current version of simbi module', action=print_the_version)
    global_args.add_argument('--nthreads', '-p', help="number of omp threads to run at", type=max_thread_count, default=None)
    global_args.add_argument('--peek', help='print setup-script usage', default=False, action='store_true')
    global_args.add_argument('--type-check', help='flag for static type checking configration files', default=True, action=argparse.BooleanOptionalAction)
    # print help message if no args supplied
    return parser, parser.parse_known_args(args=None if sys.argv[1:] else ['--help'])

configs_src = Path(__file__).resolve().parent / 'configs'
def valid_pyscript(param):
    base, ext = os.path.splitext(param)
    if ext.lower() != '.py':
        param        = None
        pkg_configs  = [file for file in configs_src.rglob('*.py')]
        soft_paths   = [soft_path for soft_path in (Path('simbi_configs')).glob("*") if soft_path.is_symlink()]   
        soft_configs = [file for path in soft_paths for file in path.rglob('*.py')]
        soft_configs+= [file for file in Path('simbi_configs').resolve().rglob('*.py')]
        for file in pkg_configs + soft_configs:
            if base == Path(file).stem:
                param = file
        if not param:
            available_configs = sorted([Path(conf).stem for conf in pkg_configs + soft_configs])
            raise argparse.ArgumentTypeError('No configuration named {}{}{}. The only valid configurations are:\n\n{}'.format(
                bcolors.OKCYAN, base, bcolors.ENDC, '\n'.join(f'{conf}' for conf in available_configs)))
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

def configure_state(script: str, parser: argparse.ArgumentParser, argv: Optional[Sequence] = None, type_checking_active: bool = True):
    """
    Configure the Hydro state based on the Config class that exists in the passed
    in setup script. Once configured, pass it back to main to be simulated 
    """
    import sys 
    script_dirname = os.path.dirname(script)
    base_script    = Path(os.path.abspath(script)).stem
    sys.path.insert(1, f'{script_dirname}')
    
    if type_checking_active:
        print("Validating Script Type Safety...\n")
        try:
            subprocess.run(['python',  '-m',  'mypy',  f'{script}'], check = True)
        except subprocess.CalledProcessError:
            print("\nYour configuration script failed type safety checks. Please fix them or run with --no-type-check option")
            sys.exit(0)
    
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
        kwargs[idx]['first_order']              = config.first_order
        kwargs[idx]['cfl']                      = config.cfl_number
        kwargs[idx]['chkpt_interval']           = config.check_point_interval
        kwargs[idx]['tstart']                   = config.default_start_time
        kwargs[idx]['tend']                     = config.default_end_time
        kwargs[idx]['hllc']                     = config.use_hllc_solver 
        kwargs[idx]['boundary_conditions']      = config.boundary_conditions
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
        kwargs[idx]['constant_sources']         = config.constant_sources
        kwargs[idx]['object_positions']         = config.object_zones
        kwargs[idx]['boundary_sources']         = config.boundary_sources
        kwargs[idx]['engine_duration']          = config.engine_duration
        states.append(state)
    
    if peek_only:
        exit(0)
        
    return states, kwargs, state_docs 
        
def main():
    parser, (args, argv) = parse_arguments()
    sim_states, kwargs, state_docs  = configure_state(args.setup_script, parser, argv, args.type_check)
    if args.nthreads:
        os.environ['OMP_NUM_THREADS'] = f'{args.nthreads}'
    
    sim_actions = [g for g in parser._action_groups if g.title in ['override', 'onthefly']]
    sim_dicts = [{a.dest:getattr(args,a.dest,None) for a in group._group_actions} for group in sim_actions]
    overridable_args = vars(argparse.Namespace(**sim_dicts[0])).keys()
    sim_args = argparse.Namespace(**{**sim_dicts[0], **sim_dicts[1]})
    for idx, sim_state in enumerate(sim_states):
        for arg in vars(sim_args):
            if arg in overridable_args and getattr(args, arg) is None:
                continue 
            
            kwargs[idx][arg] = getattr(args, arg)
        print("="*80, flush=True)
        print(state_docs[idx], flush=True)
        print("="*80, flush=True)
        sim_state.simulate(**kwargs[idx])
    
    
if __name__ == '__main__':
    sys.exit(main())
    