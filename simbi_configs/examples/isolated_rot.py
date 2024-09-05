from simbi import BaseConfig, DynamicArg, simbi_property
from simbi.key_types import *
class MagneticShockTube(BaseConfig):
    """
    Mignone, Ugliano, & Bodo (2009), Relativistic Isolated Rotational Wac=ve Test
    """
    nzones    = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 2, help="Adiabatic gas index", var_type = float)
    problem   = DynamicArg("problem", 1, help='problem number from Mignone & Bodo (2006)', var_type=int, choices=[1,2,3,4])
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        # defined as (rho, v1, v2, v3, pg, b1, b2, b3)
        return ((1.0, 0.4, -0.3, 0.5, 1.0, 2.4, 1.0, -1.6), 
                (1.0, 0.377347, -0.482389, 0.424190, 1.0, 2.4, -0.1, -2.178213))
    
    @simbi_property
    def geometry(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0, 0.5), (0.0, 1.0), (0.0, 1.0))

    @simbi_property
    def x1_cell_spacing(self) -> str:
        return "linear"
    
    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg | int]:
        return (self.nzones, 1, 1) 
    
    @simbi_property
    def gamma(self) -> DynamicArg:
        return self.ad_gamma 
    
    @simbi_property
    def regime(self) -> str:
        return "srmhd"