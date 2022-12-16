from typing import Callable, Union, Tuple, final
from .dynarg import DynamicArg


class_props = [
    'boundary_condition', 'coord_system', 'data_directory', 
    'dens_outer', 'resolution', 'dlogt', 'dynamic_args', 
    'edens_outer', 'end_time', 'find_dynamic_args', 'gamma', 
    'geometry', 'initial_state', 'linspace', 'mom_outer', 
    'parse_args', 'passive_scalars', 'plm_theta', 
    'regime', 'rho_ref', 'scale_factor',  'scale_factor_derivative', 
    'sources', 'start_time', 'use_hllc_solver']

__all__ = ['BaseConfig']
    
class BaseConfig:
    dynamic_args = None 
    
    @property
    def initial_state(self) -> tuple:
        raise NotImplementedError("Your subclass need to implement the initial_state property")
    
    @property
    def coord_system(self) -> str:
        raise NotImplementedError("Your subclass needs to implement the coord_system property")
    
    @property
    def regime(self) -> str:
        raise NotImplementedError("Your subclass needs to implement the regime property")
        
    @property
    def resolution(self) -> tuple:
        raise NotImplementedError("Your subclass needs to implement the resolution property")
    
    @property
    def geometry(self) -> tuple:
        raise NotImplementedError("Your subclass needs to implement the geometry property")
    
    @property
    def gamma(self) -> float:
        raise NotImplementedError("Your subclass needs to implement the gamma property")
    
    @property
    def linspace(self) -> bool:
        return False
    
    @property
    def sources(self) -> tuple:
       return None
    
    @property
    def passive_scalars(self) -> tuple:
        return 0.0
    
    @property
    def scale_factor(self) -> Callable:
        return None 
    
    @property
    def scale_factor_derivative(self) -> Callable:
       return None
    
    @property
    def edens_outer(self) -> Callable:
        return None 
    
    @property
    def mom_outer(self) -> Union[Callable, Tuple[Callable]]:
        return None
    
    @property
    def dens_outer(self) -> Callable:
       return None
   
    @property
    def default_start_time(self) -> float:
       return None
   
    @property
    def default_end_time(self) -> float:
       return None
   
    @property
    def use_hllc_solver(self) -> bool:
       return None
   
    @property
    def boundary_condition(self) -> str:
       return None 
   
    @property
    def plm_theta(self) -> float:
        return None 
    
    @property
    def data_directory(self) -> str:
        return None
    
    @property 
    def dlogt(self) -> float:
        return None 
    
    @property
    def use_quirk_smoothing(self) -> bool:
        return False
    
    @property
    def constant_sources(self):
        return False 
    
    @property
    def x1(self):
        return None 
    
    @property
    def x2(self):
        return None 
    
    @property
    def x3(self):
        return None
    
    @final
    @classmethod
    def find_dynamic_args(cls) -> None:
        """
        Find all derived class member's members defined as DynamicArg class instances 
        """
        members = [attr for attr in dir(cls) if attr not in class_props and not callable(getattr(cls, attr)) and not attr.startswith("__")]
        cls.dynamic_args = [getattr(cls, member) for member in members if isinstance(getattr(cls,member), DynamicArg)]
        for arg in cls.dynamic_args:
            if arg.name in class_props:
                raise ValueError(f"Your dynamic argument name ({arg.name}) is a reserved class property name. Please choose a different name")
        
    @final
    @classmethod
    def parse_args(cls, parser) -> None:
        """
        Parse extra problem-specific args from command line
        """
        if not cls.dynamic_args:
            cls.find_dynamic_args()
        for member in cls.dynamic_args:
            try:
                if type(member.value) == bool:
                    parser.add_argument(
                        f'--{member.name}',
                        help    = member.help,
                        action  = member.action,
                        default = member.value,
                    )
                else:
                    parser.add_argument(
                        f'--{member.name}',
                        help    = member.help,
                        action  = member.action,
                        type    = member.var_type,
                        choices = member.choices,
                        default = member.value,
                    )
            except:
                # ignore duplicate arguments if inheriting from another problem setup 
                pass
                    
        args = parser.parse_args()
        # Update dynamic var attributes to reflect new values passed from cli
        for var in cls.dynamic_args:
            if var.name in vars(args):
                var.value = vars(args)[var.name]
                setattr(cls, var.name, DynamicArg(name=var.name, 
                                                  help=var.help, var_type=var.var_type, 
                                                  choices=var.choices, action = var.action, value=var.value))

    @final
    @classmethod
    def print_problem_params(cls):
        """
        Read from problem params and print to stdout
        """
        import math
        def order_of_mag(val: float) -> int:
            if val == 0:
                return 0
            return int(math.floor(math.log10(val)))
        
        if not cls.dynamic_args:
            cls.find_dynamic_args()
            
        print("\nProblem Parameters:", flush=True)
        print("="*80, flush=True)
        for member in cls.dynamic_args:
            val = member.value
            if (isinstance(val, float)):
                if order_of_mag(val) < -3 or order_of_mag(val) > 3:
                    print(f"{member.name:.<30} {val:<15.2e} {member.help}", flush = True)
                    continue
                val = round(val, 3)
            val = str(val)
            print(f"{member.name:.<30} {val:<15} {member.help}", flush = True)
    
    @final
    def __del__(self):
        """
        Print problem params on class destruction
        """
        self.print_problem_params()