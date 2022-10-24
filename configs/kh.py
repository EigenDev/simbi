import numpy as np
from pysimbi import BaseConfig, DynamicArg
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
    
    def __init__(self):
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
            
    @property
    def initial_state(self):
        return (self.rho, self.p, self.vx, self.vy)
    
    @property
    def geometry(self):
        return ((self.xmin, self.xmax), (self.ymin, self.ymax))

    @property
    def linspace(self):
        return True
    
    @property
    def coord_system(self):
        return "cartesian"

    @property
    def dimensions(self):
        return (self.npts.value, self.npts.value)
    
    @property
    def gamma(self):
        return (5.0 / 3.0)
    
    @property
    def regime(self):
        return "classical"
    
    @property
    def boundary_condition(self):
        return "periodic"
    
    @property
    def use_hllc_solver(self):
        return True
    
    @property 
    def data_directory(self):
        return "data/kh_config"