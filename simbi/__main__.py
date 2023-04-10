import argparse
import os
import ast
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Optional, Sequence
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
        if "(choose from 'run', 'plot', 'afterglow')" in message:
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
                        f'> {bcolors.BOLD}{conf}{bcolors.ENDC}\n' for conf in available_configs)))
    return param

def parse_run_arguments(parser: argparse.ArgumentParser):
    run_parser = get_subparser(parser, 0)
    overridable = run_parser.add_argument_group(
        'override', 'overridable simuations options')
    global_args = run_parser.add_argument_group(
        'globals', 'global module-specific options')
    onthefly = run_parser.add_argument_group(
        'onthefly', 'simulation otions that are given on the fly')
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
        help='piecewise linear consturction parameter',
        default=None,
        type=float)
    overridable.add_argument(
        '--first-order',
        help='Set flag if wanting first order accuracy in solution',
        default=None,
        action='store_true')
    overridable.add_argument(
        '--cfl',
        help='Courant-Friedrichs-Lewy stability number',
        default=None,
        type=float)
    overridable.add_argument(
        '--hllc',
        help='flag for HLLC computation as opposed to HLLE',
        default=None,
        action=argparse.BooleanOptionalAction)
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
        action='store_true')
    overridable.add_argument(
        '--constant-sources',
        help='flag to indicate source terms provided are constant',
        default=None,
        action='store_true')
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
        help="number of omp threads to run at",
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
        '--log-dir',
        help='directory to place the log file',
        type=str,
        default=None
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
        raise TypeError("\nYour configuration script failed type safety checks." +
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
    script_dirname = Path(script).parent
    base_script = Path(script).stem
    sys.path.insert(1, f'{script_dirname}')

    with open(Path(__file__).resolve().parent / 'gitrepo_home.txt') as f:
        githome = f.read()
        
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
            print(f"Printing dynamic arguments present in -- {setup_class}")
            static_config.print_problem_params()
            peek_only = True
            continue

        # Call initializer once static vars modified
        config = static_config()

        if args.log_output:
            static_config.log_output = True 
            static_config.set_logdir(args.data_directory or config.data_directory)
                
        if config.__doc__:
            state_docs += [f"{config.__doc__}"]
        else:
            state_docs += [f"No docstring for problem class: {setup_class}"]
        state: Hydro = Hydro.gen_from_setup(config)
        kwargs[idx] = {}
        kwargs[idx]['first_order'] = config.first_order
        kwargs[idx]['cfl'] = config.cfl_number
        kwargs[idx]['chkpt_interval'] = config.check_point_interval
        kwargs[idx]['tstart'] = config.default_start_time
        kwargs[idx]['tend'] = config.default_end_time
        kwargs[idx]['hllc'] = config.use_hllc_solver
        kwargs[idx]['boundary_conditions'] = config.boundary_conditions
        kwargs[idx]['plm_theta'] = config.plm_theta
        kwargs[idx]['dlogt'] = config.dlogt
        kwargs[idx]['data_directory'] = config.data_directory
        kwargs[idx]['linspace'] = config.linspace
        kwargs[idx]['sources'] = config.sources
        kwargs[idx]['passive_scalars'] = config.passive_scalars
        kwargs[idx]['scale_factor'] = config.scale_factor
        kwargs[idx]['scale_factor_derivative'] = config.scale_factor_derivative
        kwargs[idx]['edens_outer'] = config.edens_outer
        kwargs[idx]['mom_outer'] = config.mom_outer
        kwargs[idx]['dens_outer'] = config.dens_outer
        kwargs[idx]['quirk_smoothing'] = config.use_quirk_smoothing
        kwargs[idx]['constant_sources'] = config.constant_sources
        kwargs[idx]['object_positions'] = config.object_zones
        kwargs[idx]['boundary_sources'] = config.boundary_sources
        kwargs[idx]['engine_duration'] = config.engine_duration
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
        os.environ['NTHREADS']        = f'{args.nthreads}'
        
    if args.compute_mode == 'omp':
        os.environ['USE_OMP'] = "1"

    run_parser = get_subparser(parser, 0)
    sim_actions = [
            g for g in run_parser._action_groups if g.title in [
                'override', 'onthefly']
        ]
    
    sim_dicts = [{
        a.dest: getattr(args, a.dest, None)
        for a in group._group_actions
        } for group in sim_actions
    ]
    
    overridable_args = vars(argparse.Namespace(**sim_dicts[0])).keys()
    sim_args = argparse.Namespace(**{**sim_dicts[0], **sim_dicts[1]})
    
    for coord, block in zip(['X','Y','Z'], args.gpu_block_dims):
        os.environ[f'GPU{coord}BLOCK_SIZE'] = str(block)
        
    for idx, sim_state in enumerate(sim_states):
        for arg in vars(sim_args):
            if arg in overridable_args and getattr(args, arg) is None:
                continue

            kwargs[idx][arg] = getattr(args, arg)
        logger.info("=" * 80) # type: ignore
        logger.info(state_docs[idx]) # type: ignore
        logger.info("=" * 80) #type: ignore
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
