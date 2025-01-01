from simbi import BaseConfig, DynamicArg, simbi_property, simbi_classproperty
from simbi.key_types import *

class MartiMuller(BaseConfig):
    """
    Marti & Muller (2003), Relativistic  Shock Tube Problem in 1D Fluid
    """
    nzones    = DynamicArg("nzones", 1000, help="number of grid zones", var_type=int)
    ad_gamma  = DynamicArg("ad-gamma", 4.0 / 3.0, help="Adiabatic gas index", var_type = float)
    
    @simbi_property
    def initial_state(self) -> Sequence[Sequence[float]]:
        return ((10.0, 0.0, 13.33), (1.0, 0.0, 1e-10))
    
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
        return "srhd"
    
    #-------------------- Uncomment if one wants the mesh to move
#     @simbi_property
#     def boundary_conditions(self) -> Sequence[str]:
#         return ["outflow", "dynamic"]
    
#     @simbi_classproperty
#     def scale_factor(cls) -> Callable[[float], float]:
#         return lambda t: 1 
    
#     @simbi_classproperty
#     def scale_factor_derivative(cls) -> Callable[[float], float]:
#         return lambda t: 0.5
    
#     @simbi_classproperty
#     def boundary_sources(self) -> str:
#         return f"""
# extern "C" {{
#     void bx1_outer_source(double x, double t, double arr[]){{
#         double rho_ambient = 0.1;
#         double v_ambient   = 0.0;
#         double pressure    = 1.e-10;
#         double enthalpy    = 1.0 + {self.ad_gamma} * pressure / rho_ambient / ({self.ad_gamma} - 1.0);
#         arr[0] = rho_ambient;                     // density
#         arr[1] = 0.0;                             // x1-momentum
#         arr[2] = rho_ambient * enthalpy - pressure - rho_ambient; // energy
#         arr[3] = 0.0;                             // scalar concentration
#     }}
# }}
#         """
    