from pysimbi import BaseConfig, DynamicArg
class MignoneBodo(BaseConfig):
    """
    Mignone & Bodo (2005), Relativistic Test Problems in 1D Fluid
    """
    nzones    = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad_gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type = float)
    problem   = DynamicArg("problem", 1, help = "test problem to compute", var_type=int, choices=[1,2])
    @property
    def initial_state(self):
        if self.problem == 1:
            return ((1.0, 10.0, -0.6), (10.0, 20.0, 0.5))
        return ((1.0, 1.0, 0.9), (1.0, 10.0, 0.5))
    
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
        return "relativistic"