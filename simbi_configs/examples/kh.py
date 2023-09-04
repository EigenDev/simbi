import numpy as np
from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *
SEED = 12345
class KelvinHelmholtz(BaseConfig):
    """
    Kelvin Helmholtz problem in Newtonian Fluid
    """
    npts = DynamicArg("npts", 256, help="Number of zones in x and y dimensions", var_type=int)
    
    xmin = -0.5
    xmax =  0.5
    ymin = -0.5
    ymax =  0.5
    rhoL =  2.0
    vxT  =  0.5
    pL   =  2.5
    rhoR =  1.0
    vxB  = -0.5
    pR   =  2.5
    
    def __init__(self) -> None:
        x = np.linspace(self.xmin, self.xmax, self.npts.value)
        y = np.linspace(self.ymin, self.ymax, self.npts.value)

        self.rho = np.zeros(shape=(self.npts.value, self.npts.value))
        self.rho[np.where(np.abs(y) < 0.25)] = self.rhoL 
        self.rho[np.where(np.abs(y) > 0.25)] = self.rhoR

        self.vx = np.zeros(shape=(self.npts.value, self.npts.value))
        self.vx[np.where(np.abs(y) > 0.25)]  = self.vxT
        self.vx[np.where(np.abs(y) < 0.25)]  = self.vxB

        self.vy = np.zeros_like(self.vx)

        self.p = np.zeros(shape=(self.npts.value, self.npts.value))
        self.p[np.where(np.abs(y) > 0.25)] = self.pL 
        self.p[np.where(np.abs(y) < 0.25)] = self.pR

        # Seed the KH instability with random velocities
        rng     = np.random.default_rng(SEED)
        sin_arr = 0.01 * np.sin(2 * np.pi * x)
        vx_rand = rng.choice(sin_arr, size=self.vx.shape)
        vy_rand = rng.choice(sin_arr, size=self.vy.shape)
        
        self.vx += vx_rand
        self.vy += vy_rand
            
    @simbi_property
    def initial_state(self) -> Sequence[NDArray[numpy_float]]:
        return (self.rho, self.vx, self.vy, self.p)
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))

    @simbi_property
    def linspace(self) -> bool:
        return True
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[Any]:
        return (self.npts, self.npts)
    
    @simbi_property
    def gamma(self) -> float:
        return (5.0 / 3.0)
    
    @simbi_property
    def regime(self) -> str:
        return "classical"
    
    @simbi_property
    def boundary_conditions(self) -> str:
        return "periodic"
    
    @simbi_property
    def solver(self) -> str:
        return 'hllc'
    
    @simbi_property 
    def data_directory(self) -> str:
        return "data/kh_config"