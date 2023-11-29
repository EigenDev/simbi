from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *
class MagneticShockTube(BaseConfig):
    """
    Mignone & Bodo (2006), Relativistic MHD Test Problems in 1D Fluid
    """
    nzones    = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 2, help="Adiabatic gas index", var_type = float)
    problem   = DynamicArg("problem", 1, help='problem number from Mignone & Bodo (2006)', var_type=int, choices=[1,2,3,4])
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        # defined as (rho, v1, v2, v3, pg, b1, b2, b3)
        if self.problem == 1:
            return ((1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0), (0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0))
        elif self.problem == 2:
            return ((1.0, 0.0, 0.0, 0.0, 30.0, 5.0, 6.0, 6.0), (1.0, 0.0, 0.0, 0.0, 1.0, 5.0, 0.7, 0.7))
        elif self.problem == 3:
            return ((1.0, 0.0, 0.0, 0.0, 1e3, 10.0, 7.0, 7.0), (1.0, 0.0, 0.0, 0.0, 0.1, 10.0, 0.7, 0.7))
        else:
            return ((1.0, 0.999, 0.0, 0.0, 0.1, 10.0, 7.0, 7.0), (1.0, -0.999, 0.0, 0.0, 0.1, 10.0, -7.0, -7.0))
    
    @simbi_property
    def geometry(self) -> Sequence[float]:
        return ((0.0, 1.0, 0.5))

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
        return "srmhd"