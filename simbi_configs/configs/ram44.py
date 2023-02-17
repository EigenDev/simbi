from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import * 

class Ram44(BaseConfig):
    """
    Shock with non-zero transverse velocity on one side in 2D with 1 Partition
    This setup was adapted from Zhang and MacFadyen (2006) section 4.4
    """
    nzones    = DynamicArg("nzones", 400, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad_gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        return ((1.0, 0.0, 0.0, 1e3),   # Left 
                (1.0, 0.0, 0.99, 1e-2)) # Right
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0, 0.5), (0.0, 1.0))

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