from .simbi import Hydro
from .config import BaseConfig 
from .free_arg import DynamicArg
from .helpers import print_problem_params, compute_num_polar_zones
from pathlib import Path

script_path = Path(__file__, '..').resolve()
with open(script_path.joinpath('VERSION')) as vfile:
    __version__ = vfile.readline()