from pysimbi import BaseConfig, DynamicArg

class MartiMuller(BaseConfig):
    """
    Marti & Muller (2003), Relativistic  Shock Tube Problem in 1D Fluid
    """
    nzones    = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type = float)
    @property
    def initial_state(self):
        return ((10.0, 13.33, 0.0), (0.1, 1e-10, 0.0))
    
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
        return "relativistic"