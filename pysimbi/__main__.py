import argparse
import os 
import ast
import sys
import importlib
from pysimbi import Hydro
from pathlib import Path

overideable_args = ['tstart', 'tend', 'hllc', 'boundary_condition', 'plm_theta', 'dlogt', 'data_directory']
def valid_pyscript(param):
    base, ext = os.path.splitext(param)
    if ext.lower() != '.py':
        raise argparse.ArgumentTypeError('File must have a .py extension')
    return param

def configure_state(script: str, parser: argparse.ArgumentParser, argv = None):
    """
    Configure the Hydro state based on the Config class that exists in the passed
    in setup script. Once configured, pass it back to main to be simulated 
    """
    import sys 
    script_dirname = os.path.dirname(script)
    base_script    = Path(os.path.abspath(script)).stem
    sys.path.insert(0, f'{script_dirname}')
    
    with open(script) as setup_file:
        root = ast.parse(setup_file.read())
    
    setup_class = []
    for node in root.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if base.id == 'BaseConfig':
                    setup_class += [node.name]
        
    if len(setup_class) == 1:
        setup_class = setup_class[0]
    
    problem_class = getattr(importlib.import_module(f'{base_script}'), f'{setup_class}')
    config = problem_class()
    if argv:
        config.parse_args(parser)
    print("="*80, flush=True)
    print(f"{config.__doc__:<40}", flush=True)
    print("="*80, flush=True)
    state: Hydro = Hydro.gen_from_setup(config)
    kwargs = {}
    kwargs['tstart']                   = config.start_time
    kwargs['tend']                     = config.end_time
    kwargs['hllc']                     = config.use_hllc_solver 
    kwargs['boundary_condition']       = config.boundary_condition
    kwargs['plm_theta']                = config.plm_theta
    kwargs['dlogt']                    = config.dlogt
    kwargs['data_directory']           = config.data_directory
    kwargs['linspace']                 = state.linspace 
    kwargs['sources']                  = state.sources 
    kwargs['scalars']                  = state.scalars 
    kwargs['scale_factor']             = state.scale_factor 
    kwargs['scale_factor_derivative']  = state.scale_factor_derivative
    kwargs['edens_outer']              = state.edens_outer
    kwargs['mom_outer']                = state.mom_outer 
    kwargs['dens_outer']               = state.dens_outer 
    return state, kwargs 

def main():
    parser = argparse.ArgumentParser("Primitive parameters for PySimbi simulation configuration")
    parser.add_argument('setup_script', help='setup script for simulation run', type=valid_pyscript)
    parser.add_argument('--tstart',    help='start time for simulation', default=0.0, type=float)
    parser.add_argument('--tend',    help='end time for simulation', default=1.0, type=float)
    parser.add_argument('--dlogt',     help='logarithmic time bin spacing for checkpoints', default=0.0, type=float)
    parser.add_argument('--plm_theta', help='piecewise linear consturction parameter', default=1.5, type=float)
    parser.add_argument('--first_order', help='Set flag if wanting first order accuracy in solution', default=False, action='store_true')
    parser.add_argument('--cfl', help='Courant-Friedrichs-Lewy stability number', default=0.1, type=float)
    parser.add_argument('--hllc', help='flag for HLLC computation as opposed to HLLE', default=False, action='store_true')
    parser.add_argument('--chkpt', help='checkpoint file to restart computation from', default=None, type=str)
    parser.add_argument('--chkpt_interval', help='checkpoint interval spacing in simulation time units', default=0.1, type=float)
    parser.add_argument('--data_directory', help='directory to save checpoint files', default='data/', type=str)
    parser.add_argument('--boundary_condition', help='boundary condition for inner boundary', default='outflow', choices=['reflecting', 'outflow', 'inflow', 'periodic'])
    parser.add_argument('--engine_duration', help='duration of hydrodynamic source terms', default=0.0, type=float)
    parser.add_argument('--mode', help='execution mode for computation', default='cpu', choices=['cpu', 'gpu'], dest='compute_mode')
    parser.add_argument('--quirk_smoothing', help='flag to activate Quirk (1994) smoothing at poles', default=False, action='store_true')
    
    # print help message if no args supplied
    args, argv = parser.parse_known_args(args=None if sys.argv[1:] else ['--help'])

    sim_state, kwargs = configure_state(args.setup_script, parser, argv)
    for arg in vars(args):
        if arg == 'setup_script':
            continue
        if arg in overideable_args and kwargs[arg]:
            continue 
        
        kwargs[arg] = getattr(args, arg)
    sim_state.simulate(**kwargs)
    
    
if __name__ == '__main__':
    main()
    