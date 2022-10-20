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
    for idx, setup_class in enumerate(setup_classes):
        problem_class = getattr(importlib.import_module(f'{base_script}'), f'{setup_class}')
        config = problem_class()
        if argv:
            config.parse_args(parser)
            
        
        if config.__doc__:
            state_docs += [f"{config.__doc__}"]
        else:
            state_docs += [f"Not docstring for problem class: {setup_class}"]
        state: Hydro = Hydro.gen_from_setup(config)
        kwargs[idx] = {}
        kwargs[idx]['tstart']                   = config.start_time
        kwargs[idx]['tend']                     = config.end_time
        kwargs[idx]['hllc']                     = config.use_hllc_solver 
        kwargs[idx]['boundary_condition']       = config.boundary_condition
        kwargs[idx]['plm_theta']                = config.plm_theta
        kwargs[idx]['dlogt']                    = config.dlogt
        kwargs[idx]['data_directory']           = config.data_directory
        kwargs[idx]['linspace']                 = state.linspace 
        kwargs[idx]['sources']                  = state.sources 
        kwargs[idx]['scalars']                  = state.scalars 
        kwargs[idx]['scale_factor']             = state.scale_factor 
        kwargs[idx]['scale_factor_derivative']  = state.scale_factor_derivative
        kwargs[idx]['edens_outer']              = state.edens_outer
        kwargs[idx]['mom_outer']                = state.mom_outer 
        kwargs[idx]['dens_outer']               = state.dens_outer 
        states.append(state)
        
    return states, kwargs, state_docs 

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

    sim_states, kwargs, state_docs  = configure_state(args.setup_script, parser, argv)
    for idx, sim_state in enumerate(sim_states):
        for arg in vars(args):
            if arg == 'setup_script':
                continue
            if arg in overideable_args and kwargs[idx][arg]:
                continue 
            
            kwargs[idx][arg] = getattr(args, arg)
        print("="*80, flush=True)
        print(state_docs[idx], flush=True)
        print("="*80, flush=True)
        sim_state.simulate(**kwargs[idx])
    
    
if __name__ == '__main__':
    main()
    