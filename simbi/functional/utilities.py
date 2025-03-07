from typing import TypeVar, Callable, Any, Iterable
from functools import reduce

# Type variable for pipe() input/output
T = TypeVar("T")


def compose(*functions: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Compose multiple functions from right to left.

    Given functions f(x) and g(x), compose(f, g)(x) = f(g(x))
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions)


def pipe(value: T, *functions: Callable[[T], T]) -> T:
    """Pipe a value through a series of functions"""
    return reduce(lambda v, f: f(v), functions, value)


def for_each(func: Callable[[T], Any], items: Iterable[T]) -> None:
    """Apply a function to each item in an iterable"""
    for item in items:
        func(item)
