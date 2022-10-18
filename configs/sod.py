from pysimbi import BaseConfig, DynamicArg

class SodProblem(BaseConfig):
    """
    Sod's Shock Tube Problem in 1D Newtonian Fluid
    """
    nzones = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    @property
    def initial_state(self):
        return ((1.0, 1.0, 0.0), (0.125, 0.1, 0.0))
    
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
        if isinstance(self.nzones, DynamicArg):
            return self.nzones.default 
        return self.nzones
    
    @property
    def gamma(self):
        return 1.4 
    
    @property
    def regime(self):
        return "classical"