from pysimbi import BaseConfig 

class SodProblem(BaseConfig):
    """
    Sod's Shock Tube Problem in 1D Newtonian Fluid
    """
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
        return 1000
    
    @property
    def gamma(self):
        return 1.4 
    
    @property
    def regime(self):
        return "classical"