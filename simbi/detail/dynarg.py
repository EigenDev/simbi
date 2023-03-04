import math
from ..key_types import *

__all__ = ['DynamicArg']
class DynamicArg:
    def __init__(self, 
                 name:     str, 
                 value:    Any, 
                 help:     str, 
                 var_type: Callable[...,Any],
                 choices:  Optional[list[Any]] = None, 
                 action:   Optional[str] = 'store') -> None:
        self.name     = name
        self.value    = var_type(value)
        self.help     = help 
        self.var_type = var_type
        self.choices  = choices
        self.action   = action 
        
    def __add__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return self.value + operand.value
        return self.value + operand 
    
    def __radd__(self, operand: Any) -> Any:
        return self.__add__(operand)
    
    def __iadd__(self, operand: Any) -> Any:
        return self.__add__(operand) 
    
    def __mul__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return self.value * operand.value
        return self.value * operand 
    
    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any) -> Any:
        sanitized_inputs = []
        for arg in inputs:
            if isinstance(arg, DynamicArg):
                sanitized_inputs += [arg.value]
            else:
                sanitized_inputs += [arg]
        return getattr(ufunc, method)(*sanitized_inputs, **kwargs)
    
    def __rmul__(self, operand: Any) -> Any:
        return self.__mul__(operand)
    
    def __imul__(self, operand: Any) -> Any:
        return self.__mul__(operand) 
    
    def __sub__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return self.value - operand.value 
        return self.value - operand 
    
    def __isub__(self, operand: Any) -> Any:
        return self.__sub__(operand) 
    
    def __rsub__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return operand.value - self.value
        return operand - self.value
    
    def __truediv__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return self.value / operand.value
        return self.value / operand 
    
    def __rtruediv__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return operand.value / self.value
        return operand / self.value
    
    def __floordiv__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return self.value // operand.value
        return self.value // operand  
    
    def __rfloordiv__(self, operand: Any) -> Any:
        if isinstance(operand, DynamicArg):
            return operand.value // self.value
        return operand // self.value  
    
    def __abs__(self) ->Any:
        return abs(self.value)
    
    def __eq__(self, other: Any) -> Any:
        if isinstance(other, DynamicArg):
            return self.value == other.value
        return self.value == other 
    
    def __ne__(self, other: Any) -> Any:
        if isinstance(other, DynamicArg):
            return self.value != other.value
        return self.value != other 
    
    def __pow__(self, power: Any) -> Any:
        if isinstance(power, DynamicArg):
            return self.value ** power.value
        return self.value ** power 
    
    def __lt__(self, other: Any) -> Any:
        if isinstance(other, DynamicArg):
            return self.value < other.value
        return self.value < other 
    
    def __le__(self, other: Any) -> Any:
        if isinstance(other, DynamicArg):
            return self.value <= other.value
        return self.value <= other 
    
    def _ge__(self, other: Any) -> Any:
        if isinstance(other, DynamicArg):
            return self.value >= other.value
        return self.value >= other 
    
    def __gt__(self, other: Any) -> Any:
        if isinstance(other, DynamicArg):
            return self.value > other.value
        return self.value > other 
    
    def __neg__(self) -> Any:
        return  self.value * (-1.0)
    
    def __bool__(self) -> Any:
        if isinstance(self.value, bool):
            return self.value
        return self.value != None
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __float__(self) -> float:
        return float(self.value)
    
    def __int__(self) -> int:
        return int(self.value)
    
    def log10(self) -> float:
        return math.log10(self.value)
    
    def exp(self) -> float:
        return math.exp(self.value)
    
    def sqrt(self) -> float:
        return math.sqrt(self.value)