import argparse
import os
import ast
import sys
import subprocess
import tracemalloc
import importlib
from pathlib import Path
from typing import Optional, Sequence
from . import Hydro, logger
from .detail import get_subparser, bcolors, max_thread_count

try:
    from rich_argparse import RichHelpFormatter
    help_formatter = RichHelpFormatter
except ImportError:
    help_formatter = argparse.HelpFormatter

class ComputeModeAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(
            option_strings,
            dest,
            nargs=0,
            default=argparse.SUPPRESS,
            **kwargs)
        
    def __call__(self, 
                 parser: argparse.ArgumentParser, 
                 namespace: argparse.Namespace, 
                 values: list, 
                 option_string: str | None = None) -> None:
        setattr(namespace, 'compute_mode', self.const)

class print_the_version(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(
            option_strings,
            dest,
            nargs=0,
            default=argparse.SUPPRESS,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        from simbi import __version__ as version
        print(f"SIMBI version {version}")
        parser.exit()
        
class CustomParser(argparse.ArgumentParser):
    command = []
    def error(self, message):
        sys.stderr.write(f'error: {message}\n')
        if "(choose from 'run', 'plot', 'afterglow', 'clone')" in message:
            self.print_help()
        elif 'configurations' not in message:
            if self.command in ['run', 'plot', 'afterglow', 'clone']:
                self.parse_args([self.command, '--help'])
            else:
                self.print_help()
        sys.exit(2)
        
    def parse_args(self, args=None, namespace=None):
        args, argv = super().parse_known_args(args, namespace)
        self.command = args.command
        if argv:
            msg = 'unrecognized arguments: {:s}'
            self.error(msg.format(' '.join(argv)))
        return args

def parse_module_arguments():
    parser = CustomParser(
        prog='simbi',
        usage='%(prog)s {run, plot, afterglow, clone} <input> [options]',
        description="Relativistic gas dynamics module",
        formatter_class=help_formatter,
        add_help=False,
        # exit_on_error=False
    )
    parser.add_argument(
        '--version', 
        action=print_the_version
    )
    main_parser = CustomParser(formatter_class=help_formatter, parents=[parser])
    subparsers = main_parser.add_subparsers(
        dest='command'
    )
    script_run = subparsers.add_parser(
        'run',
        help='runs the setup script',
        formatter_class=help_formatter,
        usage='simbi run <setup_script> [options]',
        parents=[parser],
        # exit_on_error=False
    )
    
    script_run.set_defaults(func=run)
    plot = subparsers.add_parser(
        'plot',
        help='plots the given simbi checkpoint file',
        formatter_class=help_formatter,
        usage='simbi plot <checkpoints> <setup_name> [options]',
        parents=[parser],
        # exit_on_error=False    
    )
    plot.set_defaults(func=plot_checkpoints)
    afterglow = subparsers.add_parser(
        'afterglow',
        help='compute the afterglow for given data',
        usage='simbi afterglow <files> [options]',
        formatter_class=help_formatter,
        parents=[parser],
        # exit_on_error=False
    )
    afterglow.set_defaults(func=calc_afterglow)
    gen_clone = subparsers.add_parser(
        'clone',
        help='generate a shadow clone of a setup script to build off of',
        usage='simbi clone [--name clone_name]',
        formatter_class=help_formatter,
        parents=[parser],
        # exit_on_error=False
    )
    gen_clone.set_defaults(func=generate_a_setup)
    gen_clone.add_argument(
        '--name', 
        help='name of the generated setup script',
        default='some_clone.py',
        type=str,
        dest='clone_name'
    )
    return main_parser, main_parser.parse_known_args(
        args=None if sys.argv[1:] else ['--help'])


def get_available_configs():
    with open(Path(__file__).resolve().parent / 'gitrepo_home.txt') as f:
        githome = f.read()
        
    configs_src = Path(githome).resolve() / 'simbi_configs'
    pkg_configs = [file for file in configs_src.rglob('*.py')]
    soft_paths = [
        soft_path for soft_path in (
            Path('simbi_configs')).glob("*") if soft_path.is_symlink()]
    soft_configs = [
        file for path in soft_paths for file in path.rglob('*.py')]
    soft_configs += [file for file in Path(
        'simbi_configs').resolve().rglob('*.py') if file not in pkg_configs]
    
    return pkg_configs + soft_configs

class print_available_configs(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        return super().__init__(
            option_strings,
            dest,
            nargs=0,
            default=argparse.SUPPRESS,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string, **kwargs):
        available_configs = get_available_configs()
        available_configs = sorted(
                    [Path(conf).stem for conf in available_configs])
        
        print("Available configs are:\n{}".format(
            ''.join(
            f'> {bcolors.BOLD}{conf}{bcolors.ENDC}\n' for conf in available_configs)
            )
        )
        parser.exit()
    
    
def valid_simbiscript(param):  
    base, ext = os.path.splitext(param)
    available_configs = get_available_configs()
    if ext.lower() != '.py':
        param = None
        for file in available_configs:
            if base == Path(file).stem:
                param = file
            elif base.replace('-', '_') == Path(file).stem:
                param = file
                
        if not param:
            available_configs = sorted(
                [Path(conf).stem for conf in available_configs])
            raise argparse.ArgumentTypeError(
                'No configuration named {}{}{}. The valid configurations are:\n{}'.format(
                    bcolors.OKCYAN, base, bcolors.ENDC, ''.join(
                        f'> {bcolors.BOLD}{conf.replace("_", "-")}{bcolors.ENDC}\n' for conf in available_configs)))
    return param

def parse_run_arguments(parser: argparse.ArgumentParser):
    run_parser = get_subparser(parser, 0)
    overridable = run_parser.add_argument_group(
        'override', 'overridable simuations options')
    global_args = run_parser.add_argument_group(
        'globals', 'global module-specific options')
    onthefly = run_parser.add_argument_group(
        'onthefly', 'simulation options that are given on the fly')
    run_parser.add_argument(
        'setup_script',
        help='setup script for simulation run',
        type=valid_simbiscript)
    overridable.add_argument(
        '--tstart',
        help='start time for simulation',
        default=None,
        type=float)
    overridable.add_argument(
        '--tend',
        help='end time for simulation',
        default=None,
        type=float)
    overridable.add_argument(
        '--dlogt',
        help='logarithmic time bin spacing for checkpoints',
        default=None,
        type=float)
    overridable.add_argument(
        '--plm-theta',
        help='piecewise linear construction parameter',
        default=None,
        type=float)
    overridable.add_argument(
        '--cfl',
        help='Courant-Friedrichs-Lewy stability number',
        default=None,
        type=float)
    overridable.add_argument(
        '--solver',
        help='flag for hydro solver',
        default=None,
        choices=['hllc', 'hlle', 'hlld'],
    )
    overridable.add_argument(
        '--chkpt-interval',
        help='checkpoint interval spacing in simulation time units',
        default=None,
        type=float)
    overridable.add_argument(
        '--data-directory',
        help='directory to save checkpoint files',
        default=None,
        type=str)
    overridable.add_argument(
        '--boundary-conditions',
        help='boundary condition for inner boundary',
        default=None,
        nargs="+",
        choices=[
            'reflecting',
            'outflow',
            'inflow',
            'periodic'])
    overridable.add_argument(
        '--engine-duration',
        help='duration of hydrodynamic source terms',
        default=None,
        type=float)
    overridable.add_argument(
        '--quirk-smoothing',
        help='flag to activate Quirk (1994) smoothing at poles',
        default=None,
        action=argparse.BooleanOptionalAction)
    overridable.add_argument(
        '--constant-sources',
        help='flag to indicate source terms provided are constant',
        default=None,
        action='store_true')
    overridable.add_argument(
        '--time-order',
        help='set the order of time integration',
        default=None,
        type=str,
        choices = ["rk1", "rk2"]
    )
    overridable.add_argument(
        '--spatial-order',
        help='set the order of spatial integration',
        default=None,
        type=str,
        choices = ["pcm", "plm"]
    )
    overridable.add_argument(
        '--order',
        help='order of time *and* space integrtion',
        default=None,
        type=str,
        choices=['first', 'second'])
    onthefly.add_argument(
        '--mode',
        help='execution mode for computation',
        default='cpu',
        choices=[
            'cpu',
            'omp',
            'gpu'],
        dest='compute_mode')
    onthefly.add_argument(
        '--chkpt',
        help='checkpoint file to restart computation from',
        default=None,
        type=str)
    global_args.add_argument(
        '--nthreads',
        '-p',
        help="number of omp threads to run with",
        type=max_thread_count,
        default=None)
    global_args.add_argument(
        '--peek',
        help='print setup-script usage',
        default=False,
        action='store_true')
    global_args.add_argument(
        '--type-check',
        help='flag for static type checking configration files',
        default=True,
        action=argparse.BooleanOptionalAction)
    global_args.add_argument(
        '--gpu-block-dims',
        help='gpu dim3 thread block dimensions',
        default=[],
        type=int,
        nargs='+'
    )
    global_args.add_argument(
        '--configs',
        help='print the available config files',
        action=print_available_configs
    )
    global_args.add_argument(
        '--cpu', 
        action=ComputeModeAction,
        const='cpu'
    )
    global_args.add_argument(
        '--gpu', 
        action=ComputeModeAction,  
        const='gpu'
    )
    global_args.add_argument(
        '--omp', 
        action=ComputeModeAction, 
        const='omp',
    )
    global_args.add_argument(
        '--log-output', 
        help='log the simulation params to a file',
        action='store_true', 
        default=False,
    )
    global_args.add_argument(
        '--log-directory',
        help='directory to place the log file',
        type=str,
        default=None
    )
    global_args.add_argument(
        '--trace-mem',
        help='flag to trace memory usage of python instance',
        action=argparse.BooleanOptionalAction,
        default=False
    )


    return parser, parser.parse_known_args(
        args=None if sys.argv[2:] else ['run', '--help'])

def type_check_input(file: str) -> None:
    type_checker = subprocess.run(
        [sys.executable, 
         '-m', 
         'mypy', 
         '--strict',
         '--ignore-missing-imports',
         file])
    
    if type_checker.returncode != 0:
        raise TypeError("\nYour configuration script failed type safety checks. " +
                        "Please fix them or run with --no-type-check option")

def configure_state(
        script: str,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        argv: Optional[Sequence]):
    """
    Configure the Hydro state based on the Config class that exists in the passed
    in setup script. Once configured, pass it back to main to be simulated
    """
    if args.trace_mem:
        tracemalloc.start()
        
    script_dirname = Path(script).parent
    base_script = Path(script).stem
    sys.path.insert(1, f'{script_dirname}')

    with open(Path(__file__).resolve().parent / 'gitrepo_home.txt') as f:
        githome = f.read()
        
    with open(script) as setup_file:
        root = ast.parse(setup_file.read())

    setup_classes = [
        node.name for node in root.body if isinstance(node, ast.ClassDef)
        for base in node.bases if base.id == 'BaseConfig' or base.id in setup_classes
    ]
                    
    if not setup_classes:
        raise ValueError("Invalid simbi configuration")
    
    if args.type_check and str(Path().absolute()) == githome:
        print("-"*80)
        print("Validating Config Script Type Safety...")
        type_check_input(script)
        print("-"*80)
        
    states = []
    state_docs = []
    kwargs = {}
    peek_only = False
    for idx, setup_class in enumerate(setup_classes):
        problem_class = getattr(
            importlib.import_module(f'{base_script}'),
            f'{setup_class}')
        static_config = problem_class
        
        if argv:
            static_config._parse_args(parser)
    
        if args.peek:
            print(f"{bcolors.YELLOW}Printing dynamic arguments present in -- {setup_class}{bcolors.ENDC}")
            static_config._print_problem_params()
            peek_only = True
            continue

        # Call initializer once static vars modified
        config = static_config()

        if args.log_output:
            static_config.log_output = True 
            static_config.set_logdir(args.log_directory or args.data_directory or config.data_directory)
        
        config.trace_memory = args.trace_mem
            
        state_docs.append(config.__doc__ or f"No docstring for problem class: {setup_class}")
        state: Hydro = Hydro.gen_from_setup(config)
        
        if config.order_of_integration == "first" or args.order == "first":
            config.spatial_order = "pcm"
            config.time_order  = "rk1"
        elif config.order_of_integration == "second" or args.order == "second":
            config.spatial_order = "plm"
            config.time_order  = "rk2"
        elif config.order_of_integration is not None:
            raise ValueError("Order of integration must either be first or second")
        
        # if a user defined c++ functions are present, compile them
        config._compile_source_terms()
        kwargs[idx] = {
            'spatial_order': config.spatial_order,
            'time_order': config.time_order,
            'cfl': config.cfl_number,
            'chkpt_interval': config.check_point_interval,
            'tstart': config.default_start_time,
            'tend': config.default_end_time,
            'solver': config.solver,
            'boundary_conditions': config.boundary_conditions,
            'plm_theta': config.plm_theta,
            'dlogt': config.dlogt,
            'data_directory': config.data_directory,
            'x1_cell_spacing': config.x1_cell_spacing,
            'x2_cell_spacing': config.x2_cell_spacing,
            'x3_cell_spacing': config.x3_cell_spacing,
            'passive_scalars': config.passive_scalars,
            'scale_factor': config.scale_factor,
            'scale_factor_derivative': config.scale_factor_derivative,
            'quirk_smoothing': config.use_quirk_smoothing,
            'constant_sources': config.constant_sources,
            'object_positions': config.object_zones,
            'engine_duration': config.engine_duration,
            'hdir': config.hydro_source_lib,
            'gdir': config.gravity_source_lib,
            'bdir': config.boundary_source_lib,
        }
        states.append(state)

    if peek_only:
        sys.exit(0)

    return states, kwargs, state_docs


def run(parser: argparse.ArgumentParser, *_) -> None:
    parser, (args, argv) = parse_run_arguments(parser)
    sim_states, kwargs, state_docs = configure_state(
        args.setup_script, parser, args, argv)
    
    if args.nthreads:
        os.environ['OMP_NUM_THREADS'] = f'{args.nthreads}'
        os.environ['NTHREADS'] = f'{args.nthreads}'
        
    if args.compute_mode == 'omp':
        os.environ['USE_OMP'] = "1"

    run_parser = get_subparser(parser, 0)
    sim_actions = [
        g for g in run_parser._action_groups if g.title in ['override', 'onthefly']
    ]
    
    sim_dicts = [
        {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        for group in sim_actions
    ]
    
    overridable_args = vars(argparse.Namespace(**sim_dicts[0])).keys()
    sim_args = argparse.Namespace(**{**sim_dicts[0], **sim_dicts[1]})
    
    for coord, block in zip(['X', 'Y', 'Z'], args.gpu_block_dims):
        os.environ[f'GPU{coord}BLOCK_SIZE'] = str(block)
        
    for idx, sim_state in enumerate(sim_states):
        for arg in vars(sim_args):
            if arg in overridable_args and getattr(args, arg) is None:
                continue
            
            if arg == "order":
                continue
            
            kwargs[idx][arg] = getattr(args, arg)
        
        logger.info("=" * 80)
        logger.info(state_docs[idx])
        logger.info("=" * 80)
        sim_state.simulate(**kwargs[idx])


def plot_checkpoints(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list) -> None:
    
    from .tools.plot import main
    main(parser, args, argv)

def calc_afterglow(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list) -> None:
    
    from .afterglow import radiation
    radiation.run(parser, args, argv)

def generate_a_setup(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *_) -> None:

    parser.parse_args()
    from . import clone 
    clone.generate(args.clone_name)
    
def main():
    try:
        from rich.traceback import install
        install()
    except ImportError:
        pass 
    
    parser, (args, _) = parse_module_arguments()
    args.func(parser, args, _)


if __name__ == '__main__':
    sys.exit(main())
