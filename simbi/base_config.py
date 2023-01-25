import argparse
import abc
from .dynarg import DynamicArg
from .key_types import *
from ._detail import get_subparser
from typing import ParamSpec, TypeVar, Generic

class_props = [
    'boundary_conditions', 'coord_system', 'data_directory', 
    'dens_outer', 'resolution', 'dlogt', 'dynamic_args', 
    'edens_outer', 'end_time', 'find_dynamic_args', 'gamma', 
    'geometry', 'initial_state', 'linspace', 'mom_outer', 
    'parse_args', 'passive_scalars', 'plm_theta', 
    'regime', 'rho_ref', 'scale_factor',  'scale_factor_derivative', 
    'sources', 'start_time', 'use_hllc_solver', 'cfl_number',
    'boundary_sources', 'object_zones', 'x1', 'x2', 'x3']

T = TypeVar('T', covariant=True)
P = ParamSpec('P')


class simbi_classproperty(property):
    def __get__(self, owner_self: Any, owner_cls: Optional[Any] = ..., /) -> Any:
        if not self.fget: return self 
        return self.fget(owner_cls)

class simbi_property(Generic[T]):
    def __init__(self, fget: Callable[P, T]) -> None:
        self._name = ''
        self.fget  = fget 
        self.__doc__ = fget.__doc__ 
        
    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = name
        
    def __get__(self, obj: Any, objtype: Optional[Any],/) -> T:
        if self.fget is None:
            raise ValueError("Property has not getter")
        return self.type_converter(self.fget(obj))
    
    @staticmethod
    def type_converter(input_obj: Any) -> T:
        if isinstance(input_obj, str):
            return input_obj
        if isinstance(input_obj, DynamicArg):
            return input_obj.var_type(input_obj.value)
        else:
            try:
                if any(isinstance(x, str) for x in input_obj):
                    return input_obj
                
                if any(isinstance(a, Sequence) for a in input_obj):
                    transform = lambda x: x.var_type(x.value) if isinstance(x, DynamicArg) else x
                    return  cast(T, tuple(tuple(map(transform, i)) for i in input_obj))
                elif any(isinstance(x, DynamicArg) for x in input_obj):
                    return tuple(res if not isinstance(res, DynamicArg) else res.var_type(res.value) for res in input_obj)
                else:
                    return input_obj
            except TypeError:
                return input_obj

def err_message(name: str) -> str:
    return f"Configuration must include a {name} simbi_property"

__all__ = ['BaseConfig', 'simbi_property', 'simbi_classproperty']
class BaseConfig(metaclass=abc.ABCMeta):
    dynamic_args: ListOrNone = None
    
    @simbi_property
    @abc.abstractmethod
    def initial_state(self) -> Union[Sequence[Union[NDArray[numpy_float], Sequence[float]]], NDArray[numpy_float]]:
        raise NotImplementedError(err_message('initial_state'))
    
    @simbi_property
    @abc.abstractmethod
    def coord_system(self) -> str:
        raise NotImplementedError(err_message('coord_system'))
    
    @simbi_property
    @abc.abstractmethod
    def regime(self) -> str:
        raise NotImplementedError(err_message('regime'))
        
    @simbi_property
    @abc.abstractmethod
    def resolution(self) -> Union[int, Sequence[Union[int, DynamicArg]], NDArray[numpy_int], DynamicArg, Sequence[Sequence[Union[int, DynamicArg]]]]:
        raise NotImplementedError(err_message('resolution'))
    
    @simbi_property
    @abc.abstractmethod
    def geometry(self) -> Union[Sequence[Union[float, DynamicArg]], Sequence[Sequence[Union[float, DynamicArg]]]]:
        raise NotImplementedError(err_message('geometry'))
    
    @simbi_property
    @abc.abstractmethod
    def gamma(self) -> Union[float, DynamicArg]:
        raise NotImplementedError(err_message('gamma'))
    
    @simbi_property
    def linspace(self) -> bool:
        return False
    
    @simbi_property
    def sources(self) -> Optional[Union[Sequence[NDArray[numpy_float]], NDArray[numpy_float]]]:
        return None
    
    @simbi_property
    def passive_scalars(self) -> Optional[Union[Sequence[float], NDArray[numpy_float]]]:
        return None
    
    @simbi_classproperty
    def scale_factor(cls) -> Optional[Callable[[float], float]]:
        return None 
    
    @simbi_classproperty
    def scale_factor_derivative(cls) -> Optional[Callable[[float], float]]:
       return None
    
    @simbi_classproperty
    def edens_outer(cls) -> Optional[Union[Callable[[float], float], Callable[[float, float], float], Callable[[float, float, float], float]]]:
        return None 
    
    @simbi_classproperty
    def mom_outer(cls) ->  Optional[Union[Callable[[float], float], Sequence[Union[Callable[[float, float], float], Callable[[float, float, float], float]]]]]:
        return None
    
    @simbi_classproperty
    def dens_outer(cls) ->  Optional[Union[Callable[[float], float], Callable[[float, float], float], Callable[[float, float, float], float]]]:
       return None
   
    @simbi_property
    def default_start_time(self) -> Union[DynamicArg, float]:
       return 0.0
   
    @simbi_property
    def default_end_time(self) -> Union[DynamicArg, float]:
       return 1.0
   
    @simbi_property
    def use_hllc_solver(self) -> bool:
       return True
   
    @simbi_property
    def boundary_conditions(self) -> Union[Sequence[str], str, NDArray[numpy_string]]:
       return 'outflow'
   
    @simbi_property
    def plm_theta(self) -> float:
        return 1.5
    
    @simbi_property
    def data_directory(self) -> str:
        return 'data/'
    
    @simbi_property 
    def dlogt(self) -> float:
        return 0.0 
    
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
    
    @simbi_property
    def cfl_number(self) -> float:
        return 0.1
    
    @simbi_property
    def first_order(self) -> bool:
        return False
    
    @simbi_property
    def check_point_interval(self)-> float:
        return 0.1 
    
    @simbi_property
    def engine_duration(self) -> float:
        return 0.0
    
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
        run_parser = get_subparser(parser, 0)
        if cls.dynamic_args:
            for member in cls.dynamic_args:
                try:
                    if type(member.value) == bool:
                        run_parser.add_argument(
                            f'--{member.name}',
                            help    = member.help,
                            action  = member.action,
                            default = member.value,
                        )
                    else:
                        run_parser.add_argument(
                            f'--{member.name}',
                            help    = member.help,
                            action  = member.action,
                            type    = member.var_type,
                            choices = member.choices,
                            default = member.value,
                        )
                except argparse.ArgumentError:
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
                    if abs(order_of_mag(val)) > 3:
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
        