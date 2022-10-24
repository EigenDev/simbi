from pysimbi import BaseConfig, DynamicArg

class StationaryWaveHLL(BaseConfig):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLL solver
    """
    nzones    = DynamicArg("nzones", 400, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad_gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @property
    def initial_state(self):
        return ((1.4, 1.0, 0.0), (1.0, 1.0, 0.0))
    
    @property
    def geometry(self):
        return (0.0, 1.0, 0.5)

    @property
    def linspace(self):
        return True
    
    @property
    def coord_system(self):
        return "cartesian"

    @property
    def dimensions(self):
        return self.nzones.value 
    
    @property
    def gamma(self):
        return self.ad_gamma.value 
    
    @property
    def regime(self):
        return "classical"
    
    @property
    def use_hllc_solver(self):
        return False 
    
    @property
    def data_directory(self):
        return 'data/stationary/hll'
    
class StationaryWaveHLLC(StationaryWaveHLL):
    """
    Stationary Wave Test Problems in 1D Newtonian Fluid using HLLC Toro et al. (1992) solver
    """
    @property
    def use_hllc_solver(self):
        return True 
    
    @property
    def data_directory(self):
        return 'data/stationary/hllc'
    