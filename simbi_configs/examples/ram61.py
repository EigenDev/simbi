from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import * 

class Ram61(BaseConfig):
    """ (Hard Test: Fails)
    Shock with non-zero transverse velocity on both sides in 2D with 1 Partition
    This setup was adapted from Zhang and MacFadyen (2006) section 6.1
    """
    nzones    = DynamicArg("nzones", 400, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        return ((1.0, 0.0, 0.90, 1e+3), # Left 
                (1.0, 0.0, 0.90, 1e-2)) # Right
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0, 0.5), (0.0, 1.0))

    @simbi_property
    def x1_cell_spacing(self) -> str:
        return "linear"
    
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
        return "srhd"
    
    @simbi_property
    def default_end_time(self) -> float:
        return 0.4 