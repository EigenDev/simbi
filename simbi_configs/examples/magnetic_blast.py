import numpy as np
from simbi import (
    BaseConfig, 
    simbi_property, 
    DynamicArg, 
    compute_num_polar_zones
)
from simbi.key_types import *

XMIN = -6.0
XMAX = 6.0
PEXP = 1.0
RHOEXP = 0.1
XEXP = 0.08
XSTOP = 1.0
def find_nearest(arr: NDArray[numpy_float], val: float) -> Tuple[Any, Any]:
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]

class thermalBomb(BaseConfig):
    """The Thermal Bomb 
    Launch a relativistic blast wave on a 2D Spherical Logarithmic mesh with variable zones per decade in radius
    """
        
    # Dynamic Args to be fed to argparse 
    e0            = DynamicArg("e0", 1.0,             help='energy scale',  var_type=float)                        
    rho0          = DynamicArg("rho0", 1.e-4,         help='density scale', var_type=float) 
    p0            = DynamicArg("p0", 3.e-5,           help='pressure scale', var_type=float)          
    b0            = DynamicArg("b0", 0.1,             help='magnetic field scale', var_type=float)           
    k             = DynamicArg("k", 0.0,              help='density power law k', var_type=float) 
    nzones        = DynamicArg("nzones", 256,         help='number of zones in x and y', var_type=int)
    ad_gamma      = DynamicArg("ad-gamma", 4.0 / 3.0, help="Adiabtic gas index", var_type=float)
    
    def __init__(self) -> None:
        nzones = int(self.nzones)
        self.x1          = np.linspace(XMIN, XMAX, nzones)
        self.x2          = np.linspace(XMIN, XMAX, nzones)
        self.rho        = np.ones((1, nzones, nzones), float) * self.rho0
        self.p          = np.ones_like(self.rho) * self.p0
        self.v1         = np.zeros_like(self.rho)
        self.v2         = self.v1.copy()
        self.v3        = self.v1.copy()
        self.bvec      = np.array([np.ones_like(self.rho) * self.b0, np.zeros_like(self.rho), np.zeros_like(self.rho)])
        
        xx, yy            = np.meshgrid(self.x1, self.x2)
        exp_reg           = xx**2 + yy**2 < 0.08
        pslope            = (PEXP - self.p0) / (XSTOP - XEXP)
        rhoslope         = (RHOEXP - self.rho0) / (XSTOP - XEXP)
        self.p[:,exp_reg]   = PEXP
        self.rho[:,exp_reg] = RHOEXP
        self.p[:,(xx > XEXP) & (xx < XSTOP)] = PEXP - pslope * (xx[(xx > XEXP) & (xx < XSTOP)] - XEXP)
        self.rho[:,(xx > XEXP) & (xx < XSTOP)] = RHOEXP - rhoslope * (xx[(xx > XEXP) & (xx < XSTOP)] - XEXP)
        
        
           
    @simbi_property
    def initial_state(self) -> Sequence[NDArray[numpy_float]]:
        return (self.rho, self.v1, self.v2, self.v3, self.p, self.bvec[0], self.bvec[1], self.bvec[2])
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[Any]]:
        return ((XMIN, XMAX), (XMIN, XMAX), (0, 1))

    @simbi_property
    def x1_cell_spacing(self) -> str:
        return "linear"
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[int | DynamicArg]:
        return (self.nzones, self.nzones, 1)
    
    @simbi_property
    def gamma(self) -> DynamicArg:
        return self.ad_gamma
    
    @simbi_property
    def regime(self) -> str:
        return "srmhd"
    
    @simbi_property
    def default_start_time(self) -> float:
        return 0.0
    
    @simbi_property
    def default_end_time(self) -> float:
        return 1.0
    
    @simbi_property
    def solver(self) -> str:
        return 'hllc'
    
    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ["outflow", "outflow", "outflow"]