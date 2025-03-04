from typing import TypeVar, Callable
from functools import reduce

T = TypeVar("T")
U = TypeVar("U")


def compose(*functions: Callable[[U], T]) -> Callable[[T], U]:
    """Compose multiple functions from right to left"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions)


def pipe(value: T, *functions: Callable[[T], T]) -> T:
    """Pipe a value through a series of functions"""
    return reduce(lambda v, f: f(v), functions, value)
