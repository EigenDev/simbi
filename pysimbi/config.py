class BaseConfig:
    @property
    def initial_state(self):
        raise NotImplementedError("Your subclass need to implement the initial_state property")
    
    @property
    def coord_system(self):
        raise NotImplementedError("Your subclass needs to implement the coord_system property")
    
    @property
    def regime(self):
        raise NotImplementedError("Your subclass needs to implement the regime property")
        
    @property
    def dimensions(self):
        raise NotImplementedError("Your subclass needs to implement the dimensions property")
    
    @property
    def geometry(self):
        raise NotImplementedError("Your subclass needs to implement the geometry property")
    
    @property
    def linspace(self):
        return False
    
    @property
    def sources(self):
       return None
    
    @property
    def scalars(self):
        return None 
    
    @property
    def passive_scalars(self):
        return None
    
    @property
    def scale_factor(self):
        return None 
    
    @property
    def scale_factor_derivative(self):
       return None
    
    @property
    def edens_outer(self):
        return None 
    
    @property
    def mom_outer(self):
        return None

    @property
    def dens_outer(self):
       return None