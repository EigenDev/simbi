from numpy.typing import NDArray
from typing import Optional, Union, Sequence, Callable, final, Tuple, Any, cast


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
           'FloatOrNone', 'IntOrNone', 'ListOrNone', 'ArrayOrNone', 'StrOrNone', 
           'BoolOrNone', 'SequenceOrNone', 'CallableOrNone', 'cast']