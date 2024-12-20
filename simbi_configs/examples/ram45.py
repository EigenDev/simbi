from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import * 

class Ram45(BaseConfig):
    """
    1D shock-heating problem in planar geometry
    This setup was adapted from Zhang and MacFadyen (2006) section 4.5
    """
    nzones    = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Any]:
        return (1.0, (1.0 - 1.e-5), 1e-6)
    
    @simbi_property
    def geometry(self) -> Sequence[float]:
        return (0.0, 1.0)

    @simbi_property
    def x1_cell_spacing(self) -> str:
        return "linear"
    
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
        return "srhd"
    
    @simbi_property
    def boundary_conditions(self) -> Sequence[str]:
        return ['outflow', 'reflecting']
    
    @simbi_property
    def default_end_time(self) -> float:
        return 2.0