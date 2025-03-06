import math
import random
from simbi import BaseConfig, DynamicArg, simbi_property
from typing import Any, Sequence, Generator, Iterator
from dataclasses import dataclass


@dataclass
class QuirkState:
    rho: float
    vx: float
    vy: float
    p: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.vx
        yield self.vy
        yield self.p

    def __add__(self, other: "QuirkState") -> "QuirkState":
        return QuirkState(
            self.rho + other.rho,
            self.vx + other.vx,
            self.vy + other.vy,
            self.p + other.p,
        )


class Quirk(BaseConfig):
    """
    Quirk's problem in Newtonian Fluid
    """

    class config:
        nxpts = DynamicArg(
            "nxpts", 2400, help="Number of zones in x direction", var_type=int
        )
        nypts = DynamicArg(
            "nypts", 20, help="Number of zones in y direction", var_type=int
        )
        mach_mode = DynamicArg(
            "mach_mode",
            "low",
            help="Low Mach number problem",
            var_type=str,
            choices=["low", "high"],
        )

    xmin = +0.0
    xmax = +2400.0
    ymin = +0.0
    ymax = +20.0

    def __init__(self) -> None:
        self.problem_states = {
            "low": (
                QuirkState(
                    216.0 / 41.0, (35.0 / 36.0) * math.sqrt(35), 0.0, 251.0 / 6.0
                ),
                QuirkState(1.0, 0.0, 0.0, 1.0),
            ),
            "high": (
                QuirkState(160.0 / 27.0, (133.0 / 8.0) * math.sqrt(1.4), 0.0, 466.5),
                QuirkState(1.0, 0.0, 0.0, 1.0),
            ),
        }

    @simbi_property
    def initial_primitive_state(self) -> Generator[tuple[float, ...], None, None]:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            state = self.problem_states[self.config.mach_mode.value]
            ni, nj = self.resolution
            dx = (self.xmax - self.xmin) / ni
            for j in range(nj):
                for i in range(ni):
                    xi = self.xmin + i * dx
                    if xi <= 5:
                        yield tuple(
                            state[0] + QuirkState(*[random.random() for _ in range(4)])
                        )
                    else:
                        yield tuple(
                            state[1] + QuirkState(*[random.random() for _ in range(4)])
                        )

        return gas_state

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[Any]:
        return (self.config.nxpts, self.config.nypts)

    @simbi_property
    def adiabatic_index(self) -> float:
        return 5.0 / 3.0

    @simbi_property
    def regime(self) -> str:
        return "classical"

    @simbi_property
    def boundary_conditions(self) -> str:
        return "reflecting"

    @simbi_property
    def solver(self) -> str:
        return "hllc"

    @simbi_property
    def use_quirk_smoothing(self) -> bool:
        return True

    @simbi_property
    def data_directory(self) -> str:
        return f"data/quirk/{f'smoothing' if self.use_quirk_smoothing else 'raw'}/{f'{self.config.mach_mode}_mach'}"

    @simbi_property
    def default_end_time(self) -> float:
        return 330 if self.config.mach_mode == "low" else 100
