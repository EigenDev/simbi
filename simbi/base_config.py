import argparse
import abc
from .dynarg import DynamicArg
from .key_types import *
from typing import ParamSpec, TypeVar
class_props = [
    'boundary_conditions', 'coord_system', 'data_directory', 
    'dens_outer', 'resolution', 'dlogt', 'dynamic_args', 
    'edens_outer', 'end_time', 'find_dynamic_args', 'gamma', 
    'geometry', 'initial_state', 'linspace', 'mom_outer', 
    'parse_args', 'passive_scalars', 'plm_theta', 
    'regime', 'rho_ref', 'scale_factor',  'scale_factor_derivative', 
    'sources', 'start_time', 'use_hllc_solver']

P = ParamSpec('P')
T = TypeVar('T')

def simbi_property(func: Callable[P, T]) -> Callable[P, T]:
    """ Do an implicit type conversion 
    Type converts the simbi_property object if a DynamicArg 
    is given as the return value
    """
    @property # type: ignore
    def wrapper(*args: P.args, **kwds: P.kwargs) -> T:
        result = func(*args, **kwds)
        if isinstance(result, Iterable) and not isinstance(result, str):
            if all(isinstance(val, Iterable) for val in result)and not any(isinstance(val, str) for val in result):
                transform = lambda x: x.var_type(x.value) if isinstance(x, DynamicArg) else x
                cleaned_result = tuple(tuple(map(transform, i)) for i in result)
                return cast(T, cleaned_result)
            return cast(T, [res if not isinstance(res, DynamicArg) else res.var_type(res.value) for res in result])
        else:
            if isinstance(result, DynamicArg):
                return cast(T, result.var_type(result.value))
        return cast(T, result)
    return wrapper


__all__ = ['BaseConfig', 'simbi_property']
class BaseConfig(metaclass=abc.ABCMeta):
    dynamic_args: ListOrNone = None
    
    @simbi_property
    @abc.abstractmethod
    def initial_state(self) -> Union[Sequence[Union[NDArray[numpy_float], Sequence[float]]], NDArray[numpy_float]]:
        pass
    
    @simbi_property
    @abc.abstractmethod
    def coord_system(self) -> str:
        pass
    
    @simbi_property
    @abc.abstractmethod
    def regime(self) -> str:
        pass
        
    @simbi_property
    @abc.abstractmethod
    def resolution(self) -> Union[int, Sequence[Union[int, DynamicArg]], NDArray[numpy_int], DynamicArg, Sequence[Sequence[Union[int, DynamicArg]]]]:
        pass
    
    @simbi_property
    @abc.abstractmethod
    def geometry(self) -> Union[Sequence[Union[float, DynamicArg]], Sequence[Sequence[Union[float, DynamicArg]]]]:
        pass
    
    @simbi_property
    @abc.abstractmethod
    def gamma(self) -> Union[float, DynamicArg]:
        pass
    
    @simbi_property
    def linspace(self) -> bool:
        return False
    
    @simbi_property
    def sources(self) -> Optional[Union[Sequence[NDArray[numpy_float]], NDArray[numpy_float]]]:
        return None
    
    @simbi_property
    def passive_scalars(self) -> Optional[Union[Sequence[float], NDArray[numpy_float]]]:
        return None
    
    @simbi_property
    def scale_factor(self) -> Optional[Callable[[float], float]]:
        return None 
    
    @simbi_property
    def scale_factor_derivative(self) -> Optional[Callable[[float], float]]:
       return None
    
    @simbi_property
    def edens_outer(self) -> Optional[Union[Callable[[float], float], Callable[[float, float], float], Callable[[float, float, float], float]]]:
        return None 
    
    @simbi_property
    def mom_outer(self) ->  Optional[Union[Callable[[float], float], Sequence[Union[Callable[[float, float], float], Callable[[float, float, float], float]]]]]:
        return None
    
    @simbi_property
    def dens_outer(self) ->  Optional[Union[Callable[[float], float], Callable[[float, float], float], Callable[[float, float, float], float]]]:
       return None
   
    @simbi_property
    def default_start_time(self) -> Optional[Union[DynamicArg, float]]:
       return None
   
    @simbi_property
    def default_end_time(self) -> Optional[Union[DynamicArg, float]]:
       return None
   
    @simbi_property
    def use_hllc_solver(self) -> bool:
       return True
   
    @simbi_property
    def boundary_conditions(self) -> Optional[Union[Sequence[str], str, NDArray[numpy_string]]]:
       return None
   
    @simbi_property
    def plm_theta(self) -> FloatOrNone:
        return None 
    
    @simbi_property
    def data_directory(self) -> StrOrNone:
        return None
    
    @simbi_property 
    def dlogt(self) -> FloatOrNone:
        return None 
    
    @simbi_property
    def use_quirk_smoothing(self) -> bool:
        return False
    
    @simbi_property
    def constant_sources(self) -> bool:
        return False 
    
    @simbi_property
    def x1(self) -> ArrayOrNone:
        return None 
    
    @simbi_property
    def x2(self) -> ArrayOrNone:
        return None 
    
    @simbi_property
    def x3(self) -> ArrayOrNone:
        return None
    
    @simbi_property
    def object_zones(self) -> Optional[Union[NDArray[Any], Sequence[Any]]]:
        return None
    
    @simbi_property
    def boundary_sources(self) -> Optional[Union[NDArray[Any], Sequence[Any]]]:
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
    def parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """
        Parse extra problem-specific args from command line
        """
        if cls.dynamic_args:
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
        else:
            cls.find_dynamic_args()
            cls.parse_args(parser)
        

    @final
    @classmethod
    def print_problem_params(cls) -> None:
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
        if cls.dynamic_args:
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
    def __del__(self) -> None:
        """
        Print problem params on class destruction
        """
        self.print_problem_params()