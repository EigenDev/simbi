import ast
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Tuple
from argparse import ArgumentParser, Namespace
from ....simulator import Hydro
from ....detail import bcolors
from ....detail.base_config import BaseConfig

def _get_setup_classes(script: str) -> List[str]:
    """Extract setup classes from script"""
    with open(script) as setup_file:
        root = ast.parse(setup_file.read())

    setup_classes = [
        node.name for node in root.body 
        if isinstance(node, ast.ClassDef)
        for base in node.bases 
        if base.id == 'BaseConfig' or base.id in setup_classes
    ]
    return setup_classes

def _configure_single_state(
    base_script: str,
    setup_class: str,
    parser: ArgumentParser,
    args: Namespace,
    argv: Optional[Sequence]
) -> Tuple[Hydro, Dict[str, Any], str]:
    """Configure single hydro state"""
    
    # Import problem class
    problem_class = getattr(
        importlib.import_module(f'{base_script}'),
        f'{setup_class}'
    )
    static_config = problem_class

    # Handle peek mode
    if argv:
        run_parser = getattr(args, "active_parser")
        static_config._parse_args(parser, run_parser)
    if args.peek:
        print(f"{bcolors.YELLOW}Printing dynamic arguments in {setup_class}{bcolors.ENDC}")
        static_config._print_problem_params()
        return None, {}, ""

    # Initialize configuration
    config = static_config()
    _setup_logging(config, args)
    
    # Create hydro state
    state: Hydro = Hydro.gen_from_setup(config)
    kwarg_dict = _build_kwargs_dict(config, args)
    
    return state, kwarg_dict, config.__doc__ or f"No docstring: {setup_class}"

def _setup_logging(config: BaseConfig, args: Namespace) -> None:
    """Setup logging configuration"""
    if args.log_output:
        config.log_output = True
        config.set_logdir(
            args.log_directory or 
            args.data_directory or 
            config.data_directory
        )
    config.trace_memory = args.trace_mem

def use_arg_or_default(arg_value, config_value):
    """Use arg value if provided, otherwise fallback to config value"""
    return arg_value if arg_value is not None else config_value

def _build_kwargs_dict(config: BaseConfig, args: Namespace) -> Dict[str, Any]:
    """Build kwargs dictionary for simulation"""
    if config.order_of_integration == "first" or args.order == "first":
        config.spatial_order = "pcm"
        config.time_order = "rk1"
    elif config.order_of_integration == "second" or args.order == "second":
        config.spatial_order = "plm"
        config.time_order = "rk2"
    elif config.order_of_integration is not None:
        raise ValueError("Order must be first or second")

    # Compile user defined functions if present
    config._compile_source_terms()
    return {
        'spatial_order': config.spatial_order,
        'time_order': config.time_order,
        'cfl': use_arg_or_default(args.cfl, config.cfl_number),
        'checkpoint_interval': args.checkpoint_interval or config.check_point_interval,
        'tstart': use_arg_or_default(args.tstart, config.default_start_time),
        'tend': use_arg_or_default(args.tend, config.default_end_time),
        'solver': use_arg_or_default(args.solver, config.solver),
        'boundary_conditions': config.boundary_conditions,
        'plm_theta': use_arg_or_default(args.plm_theta, config.plm_theta),
        'dlogt': config.dlogt,
        'data_directory': use_arg_or_default(args.data_directory, config.data_directory),
        'x1_cell_spacing': config.x1_cell_spacing,
        'x2_cell_spacing': config.x2_cell_spacing,
        'x3_cell_spacing': config.x3_cell_spacing,
        'passive_scalars': config.passive_scalars,
        'scale_factor': config.scale_factor,
        'scale_factor_derivative': config.scale_factor_derivative,
        'quirk_smoothing': use_arg_or_default(args.quirk_smoothing, config.use_quirk_smoothing),
        'constant_sources': config.constant_sources,
        'object_positions': config.object_zones,
        'engine_duration': use_arg_or_default(args.engine_duration, config.engine_duration),
        'hdir': config.hydro_source_lib,
        'gdir': config.gravity_source_lib,
        'bdir': config.boundary_source_lib,
    }



def configure_state(
    args: Namespace,
    argv: Optional[Sequence]
) -> tuple[List[Hydro], Dict[int, Dict[str, Any]], List[str]]:
    """Configure hydro state from setup script"""
    parser = getattr(args, 'main_parser')
    script = args.setup_script
    script_dirname = Path(script).parent
    base_script = Path(script).stem
    sys.path.insert(1, f'{script_dirname}')
    
    setup_classes = _get_setup_classes(script)
    if not setup_classes:
        raise ValueError("Invalid simbi configuration")

    states = []
    state_docs = []
    kwargs = {}

    for idx, setup_class in enumerate(setup_classes):
        state, kwarg_dict, doc = _configure_single_state(
            base_script, setup_class, parser, args, argv
        )
        states.append(state)
        kwargs[idx] = kwarg_dict
        state_docs.append(doc)

    return states, kwargs, state_docs