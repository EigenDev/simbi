from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.typing import InitialStateType
from typing import Sequence, Generator, Iterator, Any
from dataclasses import dataclass
from functools import partial


@dataclass(frozen=True)
class MHDState:
    rho: float
    v1: float
    v2: float
    v3: float
    p: float
    b1: float
    b2: float
    b3: float

    def __iter__(self) -> Iterator[float]:
        yield self.rho
        yield self.v1
        yield self.v2
        yield self.v3
        yield self.p


class MagneticShockTube(BaseConfig):
    """
    Mignone, Ugliano, & Bodo (2009), 1D SRMHD test problems.
    """

    class config:
        nzones = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
        adiabatic_index = DynamicArg(
            "ad-gamma", (5 / 3), help="Adiabatic gas index", var_type=float
        )
        problem = DynamicArg(
            "problem",
            "contact",
            help="problem number from Mignone & Bodo (2006)",
            var_type=str,
            choices=[
                "contact",
                "rotational",
                "st-1",
                "st-2",
                "st-3",
                "st-4",
            ],
        )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.problem_states = {
            "contact": (
                MHDState(10.0, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
                MHDState(1.00, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
            ),
            "rotational": (
                MHDState(1.0, 0.4, -0.3, 0.5, 1.0, 2.4, 1.0, -1.6),
                MHDState(1.0, 0.377347, -0.482389, 0.424190, 1.0, 2.4, -0.1, -2.178213),
            ),
            "st-1": (
                MHDState(1.000, 0.0, 0.0, 0.0, 1.0, 0.5, +1.0, 0.0),
                MHDState(0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0),
            ),
            "st-2": (
                MHDState(1.08, +0.40, +0.3, 0.2, 0.95, 2.0, +0.3, 0.3),
                MHDState(1.00, -0.45, -0.2, 0.2, 1.00, 2.0, -0.7, 0.5),
            ),
            "st-3": (
                MHDState(1.0, +0.999, 0.0, 0.0, 0.1, 10.0, +7.0, +7.0),
                MHDState(1.0, -0.999, 0.0, 0.0, 0.1, 10.0, -7.0, -7.0),
            ),
            "st-4": (
                MHDState(1.0, 0.0, 0.3, 0.4, 5.0, 1.0, 6.0, 2.0),
                MHDState(0.9, 0.0, 0.0, 0.0, 5.3, 1.0, 5.0, 2.0),
            ),
        }

    @simbi_property
    def initial_primitive_state(self) -> InitialStateType:
        def gas_state() -> Generator[tuple[float, ...], None, None]:
            ni, nj, nk = self.resolution
            state = self.problem_states[str(self.config.problem)]
            xextent = self.bounds[1] - self.bounds[0]
            dx = xextent / ni
            for k in range(nk):
                for j in range(nj):
                    for i in range(ni):
                        xi = self.bounds[0] + i * dx
                        if xi < 0.5:
                            yield tuple(state[0])
                        else:
                            yield tuple(state[1])

        def bfield(bn: str) -> Generator[float, None, None]:
            state = self.problem_states[str(self.config.problem)]
            ni, nj, nk = self.resolution
            xextent = self.bounds[1] - self.bounds[0]
            dx = xextent / ni
            for k in range(nk + (bn == "b3")):
                for j in range(nj + (bn == "b2")):
                    for i in range(ni + (bn == "b1")):
                        xi = self.bounds[0] + i * dx
                        if xi < 0.5 * xextent:
                            yield getattr(state[0], bn)
                        else:
                            yield getattr(state[1], bn)

        bx = partial(bfield, "b1")
        by = partial(bfield, "b2")
        bz = partial(bfield, "b3")
        return gas_state, bx, by, bz

    @simbi_property
    def bounds(self) -> Sequence[float]:
        return (0.0, 1.0)

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> tuple[DynamicArg, int, int]:
        return (self.config.nzones, 1, 1)

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.config.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srmhd"
