from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import * 
import numpy as np 

class Ram45(BaseConfig):
    """
    1D shock-heating problem in planar geometry
    This setup was adapted from Zhang and MacFadyen (2006) section 4.5 pg. 9
    """
    nzones    = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad_gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        return (np.ones(self.nzones.value), np.ones(self.nzones.value)*(1.0 - 8e-9), np.ones(self.nzones.value)*1e-6)
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return (0.0, 1.0)

    @simbi_property
    def linspace(self) -> bool:
        return True
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg]:
        return (self.nzones,)
    
    @simbi_property
    def gamma(self) -> DynamicArg:
        return self.ad_gamma 
    
    @simbi_property
    def regime(self) -> str:
        return "relativistic"
    
    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ['outflow', 'reflecting']
    
    @simbi_property
    def default_end_time(self) -> float:
        return 2.0