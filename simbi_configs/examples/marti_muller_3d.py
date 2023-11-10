from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *

class MartiMuller3D(BaseConfig):
    """
    Marti & Muller (2003), Relativistic  Shock Tube Problem in 3D Fluid
    """
    nzones    = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        return ((10.0, 0.0, 0.0, 0.0, 13.33), (1.0, 0.0, 0.0, 0.0, 1e-10))
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0, 0.5),
                (0.0, 1.0),
                (0.0, 1.0))

    @simbi_property
    def x1_cell_spacing(self) -> str:
        return "linear"
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg]:
        return (self.nzones, self.nzones, self.nzones) 
    
    @simbi_property
    def gamma(self) -> DynamicArg:
        return self.ad_gamma 
    
    @simbi_property
    def regime(self) -> str:
        return "relativistic"