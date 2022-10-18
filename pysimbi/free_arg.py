from typing import Any
class DynamicArg:
    def __init__(self, 
                 name: str, 
                 default: Any, 
                 help: str, 
                 var_type: type,
                 choices: list = None, 
                 action: str = 'store') -> None:
        self.name     = name
        self.default  = default
        self.help     = help 
        self.var_type = var_type
        self.choices  = choices
        self.action   = action 
        
    def __add__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.default + operand.default
        return self.default + operand 
    
    def __radd__(self, operand: Any):
        return self.__add__(operand)
    
    def __iadd__(self, operand: Any):
        return self.__add__(operand) 
    
    def __mul__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.default * operand.default
        return self.default * operand 
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return ufunc.__call__(inputs[0], self.default)
    
    def __rmul__(self, operand: Any):
        return self.__mul__(operand)
    
    def __imul__(self, operand: Any):
        return self.__mul__(operand) 
    
    def __sub__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.default - operand.default 
        return self.default - operand 
    
    def __isub__(self, operand: Any):
        return self.__sub__(operand) 
    
    def __rsub__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return operand.default - self.default
        return operand - self.default
    
    def __truediv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.default / operand.default
        return self.default / operand 
    
    def __rtruediv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return operand.default / self.default
        return operand / self.default
    
    def __floordiv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return self.default // operand.default
        return self.default // operand  
    
    def __rfloordiv__(self, operand: Any):
        if isinstance(operand, DynamicArg):
            return operand.default // self.default
        return operand // self.default  
    
    def __abs__(self):
        return abs(self.default)
    
    def __eq__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.default == other.default
        return self.default == other 
    
    def __ne__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.default != other.default
        return self.default != other 
    
    def __pow__(self, power: Any):
        return self.default ** power 
    
    def __lt__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.default < other.default
        return self.default < other 
    
    def __le__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.default <= other.default
        return self.default <= other 
    
    def _ge__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.default >= other.default
        return self.default >= other 
    
    def __gt__(self, other: Any):
        if isinstance(other, DynamicArg):
            return self.default > other.default
        return self.default > other 
    
    def __neg__(self):
        return  self.default * (-1.0)
    
    def __bool__(self):
        if isinstance(self.default, bool):
            return self.default
        return self.default != None
    
    def __str__(self):
        return str(self.default)