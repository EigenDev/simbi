from enum import Enum
from typing import Any


class ExtendedEnum(Enum):
    @classmethod
    def list(cls: Any) -> list[Any]:
        return list(map(lambda c: c.value, cls))

    def encode(self) -> bytes:
        return bytes(self.value.encode("utf-8"))


class CoordSystem(str, ExtendedEnum):
    CARTESIAN = "cartesian"
    SPHERICAL = "spherical"
    CYLINDRICAL = "cylindrical"
    PLANAR_CYLINDRICAL = "planar_cylindrical"
    AXIS_CYLINDRICAL = "axis_cylindrical"


class Regime(str, ExtendedEnum):
    CLASSICAL = "classical"
    SRHD = "srhd"
    SRMHD = "srmhd"


class BoundaryCondition(str, ExtendedEnum):
    OUTFLOW = "outflow"
    REFLECTING = "reflecting"
    DYNAMIC = "dynamic"
    PERIODIC = "periodic"


class CellSpacing(str, ExtendedEnum):
    LINEAR = "linear"
    LOG = "log"


class TimeStepping(str, ExtendedEnum):
    RK1 = "rk1"
    RK2 = "rk2"


class SpatialOrder(str, ExtendedEnum):
    PCM = "pcm"
    PLM = "plm"


class Solver(str, ExtendedEnum):
    HLLE = "hlle"
    HLLC = "hllc"
    HLLD = "hlld"


class BodyType(str, ExtendedEnum):
    RIGID = "rigid"
    ELASTIC = "elastic"
    VISCUOUS = "viscous"
    SINK = "sink"
    SOURCE = "source"
    GRAVITATIONAL = "gravitational"
    GRAVITATIONAL_SINK = "gravitational_sink"
    # TODO: Implement these later
    POROUS = "porous"
    PASSIVE = "passive"
