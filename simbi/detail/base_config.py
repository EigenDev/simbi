import argparse
import abc
import logging
from .dynarg import DynamicArg
from ..key_types import *
from . import get_subparser, bcolors
from typing import ParamSpec, TypeVar, Generic

__all__ = ['BaseConfig', 'simbi_property', 'simbi_classproperty']

T = TypeVar('T', covariant=True)
P = ParamSpec('P')
    
class simbi_classproperty:
    registry: dict[str, Any] = {}
    def __init__(self, fget: Optional[Callable[...,Any]] = None):
        self.fget = fget
        if self.fget:
            simbi_classproperty.registry[self.fget.__name__] = self.fget

    def __get__(self, owner_self: Any, owner_cls: Optional[Any] = ..., /) -> Any:
        if not self.fget:
            return self 
        return self.fget(owner_cls)

class simbi_property(Generic[T]):
    registry: dict[str, Any] = {}
    def __init__(self, fget: Callable[P, T]) -> None:
        self._name = ''
        self.fget  = fget 
        self.__doc__ = fget.__doc__ 
        simbi_property.registry[self.fget.__name__] = 'singelton'
        
    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = name
        
    def __get__(self, obj: Any, objtype: Optional[Any],/) -> Union[T, 'simbi_property[Any]']:
        if obj is None:
            return self 
        if self.fget is None:
            raise ValueError("Property has not getter")
        return cast(T, self.type_converter(self.fget(obj))) 
    
    @staticmethod
    def type_converter(input_obj: Any) -> Any:
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

def class_register(cls: Any) -> Any:
    for prop in dir(cls):        
        if prop in list(simbi_property.registry.keys()) +  list(simbi_classproperty.registry.keys()):
            cls.base_properties.update(
                {prop: 'singelton'}
            )
    return cls 

@class_register
class BaseConfig(metaclass=abc.ABCMeta):
    dynamic_args: ListOrNone = None
    base_properties: dict[str, Any] = {}
    log_output = False 
    log_directory: str = ""
    
    def __init_subclass__(cls: Any, *args: Any, **kwargs: Any) -> None:
        """Check Child Behavior
        To save from defining do-nothing properties, raise an error when a user
        tries to define a read-only property in their subclass which does not 
        exist already in the base configuration. 
         
        Args:
            cls (Any): The subclasses instance

        Raises:
            TypeError: Error if Child tries to create a @property or @simbi_property getter 
            that is not predefined in the base configuration.
        """
        super().__init_subclass__(*args, **kwargs)
        for prop in dir(cls):
            if prop.startswith('_'):
                continue 
            
            if isinstance(getattr(cls.__mro__[0], prop), simbi_property):
                if prop not in BaseConfig.base_properties.keys():
                    bullet_list = ''.join(f'>{s}\n ' for s in BaseConfig.base_properties.keys())  
                    raise TypeError(f"simbi_property {bcolors.OKCYAN}{prop}{bcolors.ENDC} defined in {cls.__name__} " +   
                        f"does not exist in BaseConfig. The available simbi properties are:\n {bullet_list}")  


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
    def _find_dynamic_args(cls) -> None:
        """
        Find all derived class member's members defined as DynamicArg class instances 
        """
        members = [attr for attr in dir(cls) if attr not in simbi_property.registry.keys() and not callable(getattr(cls, attr)) and not attr.startswith("__")]
        cls.dynamic_args = [getattr(cls, member) for member in members if isinstance(getattr(cls,member), DynamicArg)]
        for arg in cls.dynamic_args:
            if arg.name in simbi_property.registry.keys():
                raise ValueError(f"Your dynamic argument name ({arg.name}) is a reserved class property name. Please choose a different name")
    
    @classmethod
    def set_logdir(cls, value: str) -> None:
        setattr(cls, 'log_directory', value)
    @final
    @classmethod
    def _parse_args(cls, parser: argparse.ArgumentParser) -> None:
        """
        Parse extra problem-specific args from command line
        """
        run_parser = get_subparser(parser, 0)
        if cls.dynamic_args:
            problem_args = run_parser.add_argument_group(f'{cls.__name__}', f'simulation options specific to {cls.__name__} config')
            for member in cls.dynamic_args:
                try:
                    if type(member.value) == bool:
                        problem_args.add_argument(
                            f'--{member.name}',
                            help    = member.help,
                            action  = member.action,
                            default = member.value,
                        )
                    else:
                        problem_args.add_argument(
                            f'--{member.name}',
                            help    = member.help,
                            action  = member.action,
                            type    = member.var_type,
                            choices = member.choices,
                            default = member.value,
                        )
                
                except argparse.ArgumentError as e:
                    # ignore duplicate arguments if inheriting from another problem setup 
                    pass
                
            args = parser.parse_args()
            
            # Update dynamic var attributes to reflect new values passed from cli
            for var in cls.dynamic_args:
                var.name = var.name.replace('-', '_')
                if var.name in vars(args):
                    var.value = vars(args)[var.name]
                    setattr(cls, var.name, DynamicArg(name=var.name, 
                                                    help=var.help, var_type=var.var_type, 
                                                    choices=var.choices, action = var.action, value=var.value))
        else:
            cls._find_dynamic_args()
            cls._parse_args(parser)
        

    @final
    @classmethod
    def _print_problem_params(cls) -> None:
        from .slogger import logger, SimbiFormatter
        """
        Read from problem params and print to stdout
        """
        import math
        def order_of_mag(val: float) -> int:
            if val == 0:
                return 0
            return int(math.floor(math.log10(val)))
        
        if not cls.dynamic_args:
            cls._find_dynamic_args()
        
        if cls.log_output:
            from datetime import datetime
            from pathlib import Path
            timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
            Path(cls.log_directory).mkdir(parents=True, exist_ok=True)
            logfile = Path(cls.log_directory) / f"simbilog_{timestr}.log"
            logger.debug(f"Writing log file: {logfile}")
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(SimbiFormatter())
            logger.addHandler(file_handler)
            
        logger.info("\nProblem Parameters:")
        logger.info("="*80)
        if cls.dynamic_args:
            for member in cls.dynamic_args:
                val = member.value
                if (isinstance(val, float)):
                    if abs(order_of_mag(val)) > 3:
                        logger.info(f"{member.name:.<30} {val:<15.2e} {member.help}")
                        continue
                    val = round(val, 3)
                val = str(val)
                logger.info(f"{member.name:.<30} {val:<15} {member.help}") 
    
    @final
    def __del__(self) -> None:
        """
        Print problem params on class destruction
        """
        self._print_problem_params()
        