import sys
from pathlib import Path

setup_clone = """# Auto-generate setup script template to quickly get up
# and running with the code! All @simbi_property types 
# that have a default type can be removed completely if 
# one does not plan on changing any of these fields

from simbi import (
    BaseConfig, 
    DynamicArg, 
    simbi_classproperty, 
    simbi_property
)

class {setup_name}(BaseConfig):
    \""" Some Hydro Problem
    A more descriptive doc string of what you're solving
    \"""
    x = DynamicArg('x', value=1.0, help='a dummy dynamic arg', var_type=float)

    def __init__(self) -> None:
        \"""
        initiate the problem here
        \"""
        pass

    @simbi_property
    def initial_state(self) -> Union[
                                Sequence[
                                    Union[NDArray[np.float64], 
                                    Sequence[float]]], 
                                    NDArray[np.float64]
                               ]:
        raise NotImplementedError()

    @simbi_property
    def coord_system(self) -> str:
        raise NotImplementedError()

    @simbi_property
    def regime(self) -> str:
        raise NotImplementedError()

    @simbi_property
    def resolution(self) -> Union[int,
                                  Sequence[Union[int,
                                                 DynamicArg]],
                                  NDArray[numpy_int],
                                  DynamicArg,
                                  Sequence[Sequence[Union[int,
                                                          DynamicArg]]]]:
        raise NotImplementedError()

    @simbi_property
    def bounds(self) -> Union[Sequence[Union[float, DynamicArg]],
                                Sequence[Sequence[Union[float, DynamicArg]]]]:
        raise NotImplementedError()

    @simbi_property
    def adiabatic_index(self) -> Union[float, DynamicArg]:
        raise NotImplementedError()

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def sources(
            self) -> Optional[Union[Sequence[NDArray[np.float64]], NDArray[np.float64]]]:
        return None

    @simbi_property
    def passive_scalars(
            self) -> Optional[Union[Sequence[float], NDArray[np.float64]]]:
        return None

    @simbi_classproperty
    def scale_factor(cls) -> Optional[Callable[[float], float]]:
        return None

    @simbi_classproperty
    def scale_factor_derivative(cls) -> Optional[Callable[[float], float]]:
        return None

    @simbi_property
    def default_start_time(self) -> Union[DynamicArg, float]:
        return 0.0

    @simbi_property
    def default_end_time(self) -> Union[DynamicArg, float]:
        return 1.0

    @simbi_property
    def olver(self) -> str:
        return 'hllc'

    @simbi_property
    def boundary_conditions(
            self) -> Union[Sequence[str], str, NDArray[numpy_string]]:
        return 'outflow'

    @simbi_property
    def plm_theta(self) -> float:
        return 1.5

    @simbi_property
    def data_directory(self) -> str:
        return 'data/'

    @simbi_property
    def dlogt(self) -> float:
        return 0.0

    @simbi_property
    def use_quirk_smoothing(self) -> bool:
        return False

    @simbi_property
    def x1(self) -> Optional[NDArray[np.float64]]:
        return None

    @simbi_property
    def x2(self) -> Optional[NDArray[np.float64]]:
        return None

    @simbi_property
    def x3(self) -> Optional[NDArray[np.float64]]:
        return None

    @simbi_property
    def object_zones(self) -> Optional[Union[NDArray[Any], Sequence[Any]]]:
        return None

    @simbi_property
    def cfl_number(self) -> float:
        return 0.1

    @simbi_property
    def order_of_integration(self) -> str:
        return "second"

    @simbi_property
    def checkpoint_interval(self) -> float:
        return 0.1

    @simbi_property
    def engine_duration(self) -> float:
        return 0.0
"""


def pascalcase(name: str) -> str:
    return "".join(x for x in name.title() if not x.isspace())


def generate(name: str):
    with open(Path(__file__).resolve().parent / "gitrepo_home.txt") as f:
        githome = f.read()

    if not name.endswith(".py"):
        name += ".py"

    name = name.replace(" ", "_").replace("-", "_")
    setup_name = str(Path(name.replace("_", " ")).stem)
    file = Path(githome).resolve() / "simbi_configs" / name
    if file.is_file():
        raise ValueError(f"{file} already exists")

    print(f"generating {file} file...")
    with open(file, "w") as f:
        f.write(setup_clone.format(setup_name=pascalcase(setup_name)))
