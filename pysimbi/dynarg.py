from typing import Any
class DynamicArg:
    def __init__(self, 
                 name: str, 
                 value: Any, 
                 help: str, 
                 var_type: type,
                 choices: list = None, 
                 action: str = 'store') -> None:
        self.name     = name
        self.value    = value
        self.help     = help 
        self.var_type = var_type
        self.choices  = choices
        self.action   = action 
        
    def __add__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.value + operand.value
        return self.value + operand 
    
    def __radd__(self, operand: Any):
        return self.__add__(operand)
    
    def __iadd__(self, operand: Any):
        return self.__add__(operand) 
    
    def __mul__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.value * operand.value
        return self.value * operand 
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return getattr(ufunc, method)(self.value, **kwargs)
    
    def __rmul__(self, operand: Any):
        return self.__mul__(operand)
    
    def __imul__(self, operand: Any):
        return self.__mul__(operand) 
    
    def __sub__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.value - operand.value 
        return self.value - operand 
    
    def __isub__(self, operand: Any):
        return self.__sub__(operand) 
    
    def __rsub__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return operand.value - self.value
        return operand - self.value
    
    def __truediv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.value / operand.value
        return self.value / operand 
    
    def __rtruediv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return operand.value / self.value
        return operand / self.value
    
    def __floordiv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.value // operand.value
        return self.value // operand  
    
    def __rfloordiv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return operand.value // self.value
        return operand // self.value  
    
    def __abs__(self):
        return abs(self.value)
    
    def __eq__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.value == other.value
        return self.value == other 
    
    def __ne__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.value != other.value
        return self.value != other 
    
    def __pow__(self, power: Any):
        return self.value ** power 
    
    def __lt__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.value < other.value
        return self.value < other 
    
    def __le__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.value <= other.value
        return self.value <= other 
    
    def _ge__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.value >= other.value
        return self.value >= other 
    
    def __gt__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.value > other.value
        return self.value > other 
    
    def __neg__(self):
        return  self.value * (-1.0)
    
    def __bool__(self):
        if isinstance(self.value, bool):
            return self.value
        return self.value != None
    
    def __str__(self):
        return str(self.value)