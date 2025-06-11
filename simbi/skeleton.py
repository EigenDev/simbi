from pathlib import Path

setup_skeleton = """# Auto-generated skeleton for a hydro problem configuration.

from simbi.core.config.base_config import SimbiBaseConfig
from simbi.core.config.fields import SimbiField
from simbi.core.types.input import CoordSystem, Regime, Solver
from simbi.core.types.typing import GasStateGenerator, InitialStateType
from typing import Sequence
from pathlib import Path

class {setup_name}(SimbiBaseConfig):
    \""" Some Hydro Problem
    A more descriptive doc string of what you're solving
    \"""
    adiabtic_index: float = SimbiField(1.7, help='adiabatic index')
    coord_system: CoordSystem = SimbiField(
        value=CoordSystem.CARTESIAN,
        help='Coordinate system used for the simulation'
    )
    regime: Regime = SimbiField(
        value=Regime.CLASSICAL,
        help='Regime of the simulation, e.g. CLASSICAL, SRHD, RMHD.'
    )
    solver: Solver = SimbiField(
        value=Solver.HLLC,
        help='Solver used for the simulation, e.g. HLLE, HLLC, HLLD (MHD only).'
    )
    resolution: Sequence[int] = SimbiField(
        value=(100, 100, 100),
        help='Resolution of the simulation grid in each dimension.'
    )
    bounds: Sequence[float] = SimbiField(
        value=[(-1.0, 1.0),(-1.0, 1.0), (-1.0, 1.0)],
        help='Bounds of the simulation domain in each dimension.'
    )
    start_time: float = SimbiField(
        value=0.0,
        help='Start time of the simulation.'
    )
    end_time: float = SimbiField(
        value=1.0,
        help='End time of the simulation.'
    )
    boundary_conditions: Sequence[str] = SimbiField(
        value=['outflow', 'outflow', 'outflow'],
        help='Boundary conditions for the simulation.'
    )
    plm_theta: float = SimbiField(
        value=1.5,
        help='PLM theta parameter for the piecewise linear method.'
    )
    cfl_number: float = SimbiField(
        value=0.1,
        help='CFL number for the simulation, used to control time step size.'
    )
    checkpoint_interval: float = SimbiField(
        value=0.1,
        help='Interval at which to save checkpoints during the simulation.'
    )
    data_directory: Path = SimbiField(
        value=Path('data/'),
        help='Directory where simulation data will be stored.'
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        pass

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> GasStateGenerator:
            '''Initial gas state generator function.'''
            # Implement the logic to generate the initial gas state
            raise NotImplementedError("Implement the initial gas state logic here.")
        raise NotImplementedError()
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
        f.write(setup_skeleton.format(setup_name=pascalcase(setup_name)))
