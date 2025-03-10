from typing import TypeVar, Generic, Callable, Optional, Any, Union
from dataclasses import dataclass
from functools import wraps

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Maybe(Generic[T]):
    """Option type for handling computations that may fail"""

    _value: Optional[T]
    _error: Optional[Exception] = None

    @staticmethod
    def of(value: T) -> "Maybe[T]":
        """Create a Maybe with a value"""
        return Maybe(value)

    def map(self, f: Callable[[T], U]) -> "Maybe[U]":
        """Apply function if value exists, propagate error otherwise"""
        if self._error:
            return Maybe(None, self._error)
        if self._value is None:
            return Maybe(None, ValueError("Cannot map over None value"))
        try:
            return Maybe(f(self._value))
        except Exception as e:
            return Maybe(None, e)

    def bind(self, f: Callable[[T], "Maybe[U]"]) -> "Maybe[U]":
        """Chain Maybe operations, preserving error chain"""
        if self._error:
            return Maybe(None, self._error)
        if self._value is None:
            return Maybe(None, ValueError("Cannot bind over None value"))
        try:
            result = f(self._value)
            if not isinstance(result, Maybe):
                raise TypeError(f"Function must return Maybe, got {type(result)}")
            return result
        except Exception as e:
            return Maybe(None, e)

    def unwrap(self) -> T:
        """Get value or raise error with full chain"""
        if self._error:
            raise self._error
        if self._value is None:
            raise ValueError("Cannot unwrap None value")
        return self._value

    def or_else(self, default: "Maybe[T]") -> "Maybe[T]":
        """Get value or return default"""
        return self if self._value is not None else default

    @property
    def error(self) -> Optional[Exception]:
        return self._error

    def is_error(self) -> bool:
        return self._error is not None
