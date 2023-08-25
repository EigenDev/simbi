from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import * 

class SodProblemQuad(BaseConfig):
    """
    Sod's Shock Tube Problem in 2D Newtonian Fluid with 4 Partitions
    This setup was adapted from Zhang and MacFadyen (2006) section 4.8 pg. 11
    """
    nzones    = DynamicArg("nzones", 256, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        return ((0.5, 0.0, 0.0, 1.0),  # Southwest 
                (0.1, 0.0, 0.90, 1.0), # Southeast
                (0.1, 0.90, 0.0, 1.0), # Northwest
                (0.1, 0.0, 0.0, 0.01)) # Northeast
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0, 0.5), (0.0, 1.0, 0.5))

    @simbi_property
    def linspace(self) -> bool:
        return True
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg]:
        return (self.nzones, self.nzones) 
    
    @simbi_property
    def gamma(self) -> DynamicArg:
        return self.ad_gamma 
    
    @simbi_property
    def regime(self) -> str:
        return "relativistic"
    
    @simbi_property
    def default_end_time(self) -> float:
        return 0.4 