from .simbi import Hydro
from .config import BaseConfig 
from .helpers import print_problem_params
from pathlib import Path

script_path = Path(__file__, '..').resolve()
with open(script_path.joinpath('VERSION')) as vfile:
    __version__ = vfile.readline()