import numpy as np
from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *

class RayleighTaylor(BaseConfig):
    """
    Rayleigh Taylor problem in Newtonian Fluid
    """
    xnpts = DynamicArg("xnpts", 200, help="Number of zones in x dimensions", var_type=int)
    ynpts = DynamicArg("ynpts", 600, help="Number of zones in y dimensions", var_type=int)
    
    xmin = -0.25
    xmax =  0.25
    ymin = -0.75
    ymax =  0.75
    rhoU =  2.0
    p0   =  2.5
    rhoD =  1.0
    g0   =  0.1
    vamp = 0.01
    ymidpoint = (ymax + ymin) * 0.5
    
    def __init__(self) -> None:
        x = np.linspace(self.xmin, self.xmax, self.xnpts.value)
        y = np.linspace(self.ymin, self.ymax, self.ynpts.value)
        xx, yy = np.meshgrid(x, y)

        self.rho = np.zeros_like(xx)
        self.rho[np.where(yy <= self.ymidpoint)] = self.rhoD
        self.rho[np.where(yy >  self.ymidpoint)] = self.rhoU

        # Seed the RT instability with velocity perturbation
        self.vy = self.vamp * 0.25 * (1 + np.cos(4.0 * np.pi * xx)) * (1.0 + np.cos(3.0 * np.pi * yy))
        self.vx = np.zeros_like(self.vy)
        self.p  = self.p0 - self.g0 * self.rho * yy
        
        self.gravityx = np.zeros_like(self.rho)
        self.gravityy = - self.g0 * np.ones_like(self.rho)
    
    @simbi_property
    def initial_state(self) -> Sequence[NDArray[numpy_float]]:
        return (self.rho, self.vx, self.vy, self.p)
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[Any]:
        return (self.xnpts, self.ynpts)
    
    @simbi_property
    def gamma(self) -> float:
        return (7.0 / 5.0)
    
    @simbi_property
    def gravity_sources(self) -> Sequence[NDArray[numpy_float]]:
        return (self.gravityx, self.gravityy)
    
    @simbi_property
    def regime(self) -> str:
        return "classical"
    
    @simbi_property
    def boundary_conditions(self) -> list[str]:
        return ["periodic", "reflecting"]
    
    @simbi_property
    def solver(self) -> str:
        return 'hllc'
    
    @simbi_property 
    def data_directory(self) -> str:
        return "data/rt_config"