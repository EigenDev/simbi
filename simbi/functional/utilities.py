# =============================================================================
# utilities.py
#
# functional programming combinators and utilities.
# provides compose, pipe, curry, and other fp primitives.
#
# usage:
#   result = pipe(data, parse, validate, transform)
#   process = compose(transform, validate, parse)
# =============================================================================
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache, partial, reduce
from typing import Any, Callable, Hashable, Iterable, TypeVar

# transform type - a function that transforms one type to another
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
K = TypeVar("K", bound=Hashable)
Transform = Callable[[T], U]


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


def map_with_index(func: Callable[[int, T], U], items: Iterable[T]) -> list[U]:
    """Apply a function to each item and its index in an iterable"""
    return [func(i, item) for i, item in enumerate(items)]


def partition(
    predicate: Callable[[T], bool], items: Iterable[T]
) -> tuple[list[T], list[T]]:
    """Split items into two lists based on predicate"""
    passes: list[T] = []
    fails: list[T] = []
    for item in items:
        (passes if predicate(item) else fails).append(item)
    return passes, fails


def reduce_with_index(
    func: Callable[[U, T, int], U], items: Iterable[T], initial: U
) -> U:
    """Reduce with index information"""
    result = initial
    for i, item in enumerate(items):
        result = func(result, item, i)
    return result


def curry(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Callable[..., Any]:
    """Partially apply arguments to a function"""
    return partial(func, *args, **kwargs)


def zip_with(
    func: Callable[[T, U], V], xs: Iterable[T], ys: Iterable[U]
) -> list[V]:
    """Combine two iterables using a function"""
    return [func(x, y) for x, y in zip(xs, ys)]


def group_by(
    key_func: Callable[[T], K], items: Iterable[T]
) -> dict[K, list[T]]:
    """Group items by a key function"""
    groups = defaultdict(list)
    for item in items:
        groups[key_func(item)].append(item)
    return dict(groups)


def memoize(func: Callable[..., Any]) -> Callable[..., Any]:
    """Cache function results"""
    return lru_cache(maxsize=None)(func)


def update_with(obj: T, updates: dict[str, Any]) -> T:
    """Create a new object with updates applied"""
    result = deepcopy(obj)
    for k, v in updates.items():
        setattr(result, k, v)
    return result
