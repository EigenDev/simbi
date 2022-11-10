from pysimbi import BaseConfig, DynamicArg

class SodProblem(BaseConfig):
    """
    Sod's Shock Tube Problem in 1D Newtonian Fluid
    """
    nzones    = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad_gamma", 5.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @property
    def initial_state(self):
        return ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
    
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
    def resolution(self):
        return self.nzones.value 
    
    @property
    def gamma(self):
        return self.ad_gamma.value 
    
    @property
    def regime(self):
        return "classical"