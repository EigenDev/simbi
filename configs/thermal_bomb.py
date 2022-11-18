import numpy as np
from pysimbi import BaseConfig, compute_num_polar_zones, DynamicArg

RHO_AMB = 1.0
P_AMB   = RHO_AMB * 1e-10
NU      = 3.0 

def find_nearest(arr: np.ndarray, val: float):
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]

class thermalBomb(BaseConfig):
    """The Thermal Bomb 
    Launch a relativistic blast wave on a 2D Spherical Logarithmic mesh with variable zones per decade in radius
    """
        
    # Dynamic Args to be fed to argparse 
    e0            = DynamicArg("e0", 10.0,             help='energy scale',  var_type=float)                        
    rho0          = DynamicArg("rho0", 1.0,           help='density scale', var_type=float)                      
    rinit         = DynamicArg("rinit", 0.1,          help='intial grid radius', var_type=float)
    rend          = DynamicArg("rend", 1.0,           help='radial extent', var_type=float)
    k             = DynamicArg("k", 0.0,              help='density power law k', var_type=float) 
    full_sphere   = DynamicArg("full_sphere", False,  help='flag for full_sphere computation',  var_type=bool, action='store_true') 
    zpd           = DynamicArg("zpd", 1024,            help='number of radial zones per decade', var_type=int)
    ad_gamma      = DynamicArg("ad_gamma", 4.0 / 3.0, help="Adiabtic gas index", var_type=float)
    
    def __init__(self):
        ndec             = np.log10(self.rend / self.rinit)
        self.nr          = round(self.zpd * ndec)
        r                = np.geomspace(self.rinit.value, self.rend.value, self.nr)
        self.theta_min   = 0
        self.theta_max   = np.pi if self.full_sphere else 0.5 * np.pi
        self.npolar      = compute_num_polar_zones(self.rinit, self.rend, self.nr, theta_bounds=(self.theta_min, self.theta_max))
        dr               = self.rinit * 1.5 
        
        p_zones = find_nearest(r, dr)[0]
        p_c     = (self.ad_gamma - 1.)*(3*self.e0/((NU + 1)*np.pi*dr ** NU))
        
        self.rho            = np.ones((self.npolar , self.nr), float) * r ** (- self.k)
        self.p              = P_AMB * self.rho 
        self.p[:, :p_zones] = p_c
        self.vx             = np.zeros_like(self.p)
        self.vy             = self.vx.copy()
           
    @property
    def initial_state(self):
        return (self.rho, self.vx, self.vy, self.p)
    
    @property
    def geometry(self):
        return ((self.rinit.value, self.rend.value), (self.theta_min, self.theta_max))

    @property
    def linspace(self):
        return False
    
    @property
    def coord_system(self):
        return "spherical"

    @property
    def resolution(self):
        return (self.nr, self.npolar)
    
    @property
    def gamma(self):
        return self.ad_gamma.value
    
    @property
    def regime(self):
        return "relativistic"
    
    @property
    def start_time(self):
        return 0.0
    
    @property
    def end_time(self):
        return 1.0
    
    @property
    def use_hllc_solver(self):
        return True