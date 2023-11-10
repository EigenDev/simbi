from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *

class StationaryWaveHLL(BaseConfig):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLL solver
    """
    nzones    = DynamicArg("nzones", 400, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        return ((1.4, 0.0, 1.0), (1.0, 0.0, 1.0))
    
    @simbi_property
    def geometry(self) -> Sequence[float]:
        return (0.0, 1.0, 0.5)

    @simbi_property
    def x1_cell_spacing(self) -> str:
        return "linear"
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> DynamicArg:
        return self.nzones 
    
    @simbi_property
    def gamma(self) -> DynamicArg:
        return self.ad_gamma 
    
    @simbi_property
    def regime(self) -> str:
        return "classical"
    
    @simbi_property
    def solver(self) -> str:
        return 'hlle' 
    
    @simbi_property
    def data_directory(self) -> str:
        return 'data/stationary/hlle'
    
class StationaryWaveHLLC(StationaryWaveHLL):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLLC Toro et al. (1992) solver
    """
    @simbi_property
    def solver(self) -> str:
        return 'hllc' 
    
    @simbi_property
    def data_directory(self) -> str:
        return 'data/stationary/hllc'
    