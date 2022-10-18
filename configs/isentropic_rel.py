from pysimbi import BaseConfig, DynamicArg
import numpy as np 
import argparse 

ALPHA_MAX = 2.0 
ALPHA_MIN = 1e-3

def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < ALPHA_MIN or f >= ALPHA_MAX:
        raise argparse.ArgumentTypeError("Argument must be < " + str(ALPHA_MAX) + " and > " + str(ALPHA_MIN))
    return f

def func(x):
        return np.sin(2*np.pi*x)

def rho(alpha, x):
    return 1.0 + alpha*func(x)

def cs(gamma, rho, pressure):
    h = 1.0 + gamma * pressure / (rho * (gamma - 1.0))
    return np.sqrt(gamma*pressure/(rho * h))

def pressure(p_ref:float, gamma:float, rho:float, rho_ref: float):
    return p_ref*(rho/rho_ref)**gamma

def velocity(gamma, rho, rho_ref, pressure, p_ref):
    return 2/(gamma - 1.)*(cs(gamma, rho, pressure) - cs(gamma, rho_ref, p_ref))

class IsentropicRelWave(BaseConfig):
    """
    Relativistic Isentropic Pulse in 1D, Entropy conserving
    """
    nzones    = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type = float)
    alpha     = DynamicArg("alpha", 0.5, help = "Wave amplitude", var_type=range_limited_float_type)
    
    rho_ref = 1.0
    p_ref   = 1.0
    K       = p_ref*rho_ref**(-ad_gamma)
    x       = np.linspace(0, 1, nzones.default, dtype=float)
    density = rho(alpha, x)
    pre     = pressure(p_ref, ad_gamma, density, rho_ref)
    beta    = velocity(ad_gamma, density, rho_ref, pre, p_ref)
    
    @property
    def initial_state(self):
        return (self.density, self.pre, self.beta)
    
    @property
    def geometry(self):
        return (0.0, 1.0)

    @property
    def linspace(self):
        return True
    
    @property
    def coord_system(self):
        return "cartesian"

    @property
    def dimensions(self):
        return self.nzones.default 
    
    @property
    def gamma(self):
        return self.ad_gamma.default 
    
    @property
    def regime(self):
        return "relativistic"