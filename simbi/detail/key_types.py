from numpy.typing import NDArray
from typing import Optional, Union, Sequence, Callable, final, Tuple, Any, cast, Iterable, Type
from numpy import float64 as numpy_float, string_ as numpy_string, int64 as numpy_int

FloatOrArray = Union[float, NDArray[Any]]
FloatOrNone  = Optional[float] 
IntOrNone    = Optional[int] 
ListOrNone   = Optional[list] 
ArrayOrNone  = Optional[NDArray[Any]] 
StrOrNone    = Optional[str] 
BoolOrNone   = Optional[bool]
SequenceOrNone = Optional[Sequence[Any]]
CallableOrNone = Optional[Callable[...,Any]]

__all__ = ['Optional', 'Union', 'Sequence', 'Callable', 'final', 'Tuple', 'Any', 'NDArray', 'FloatOrArray',
           'FloatOrNone', 'IntOrNone', 'ListOrNone', 'ArrayOrNone', 'StrOrNone', 'Type', 
           'BoolOrNone', 'SequenceOrNone', 'CallableOrNone', 'cast', 'Iterable', 'numpy_float', 'numpy_int', 'numpy_string']