from typing import TypeVar, Generic, Callable, Optional, Any, Union
from dataclasses import dataclass
from functools import wraps

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Maybe(Generic[T]):
    """Option type for handling computations that may fail"""

    _value: Optional[T]
    _error: Optional[Union[str, Exception]] = None

    @staticmethod
    def of(value: T) -> "Maybe[T]":
        """Create a Maybe with a value"""
        return Maybe(value)

    @staticmethod
    def save_failure(error: Union[str, Exception]) -> "Maybe[Any]":
        """Create a Maybe with an error"""
        if isinstance(error, str):
            return Maybe(None, ValueError(error))
        return Maybe(None, error)

    def map(self, f: Callable[[T], U]) -> "Maybe[U]":
        """Apply function if value exists, propagate error otherwise"""
        if self._error:
            return Maybe(None, self._error)
        if self._value is None:
            return Maybe(None, ValueError("Cannot map over None value"))
        try:
            return Maybe.of(f(self._value))
        except Exception as e:
            return Maybe.save_failure(e)

    def bind(self, f: Callable[[T], "Maybe[U]"]) -> "Maybe[U]":
        """Chain Maybe operations, propagating errors"""
        if self._error:
            return Maybe(None, self._error)

        if self._value is None:
            return Maybe(None, ValueError("Cannot bind over None value"))

        try:
            result: Maybe[U] = f(self._value)
            if not isinstance(result, Maybe):
                raise TypeError(f"Function must return Maybe, got {type(result)}")
            return result
        except Exception as e:
            return Maybe.save_failure(e)

    def map_with_context(self, f: Callable[[T], U], context: str) -> "Maybe[U]":
        """Apply function if value exists, propagate error with context"""
        if self._error:
            return Maybe(None, self._error)

        if self._value is None:
            return Maybe(None, ValueError(f"{context}: Cannot map over None value"))

        try:
            return Maybe.of(f(self._value))
        except Exception as e:
            return Maybe.save_failure(RuntimeError(f"{context}: {str(e)}"))

    def bind_with_context(
        self, f: Callable[[T], "Maybe[U]"], context: str
    ) -> "Maybe[U]":
        """Chain Maybe operations with error context"""
        if self._error:
            return Maybe(None, self._error)

        if self._value is None:
            return Maybe(None, ValueError(f"{context}: Cannot bind over None value"))

        try:
            result: Maybe[U] = f(self._value)
            if not isinstance(result, Maybe):
                raise TypeError(f"Function must return Maybe, got {type(result)}")
            return result
        except Exception as e:
            return Maybe.save_failure(RuntimeError(f"{context}: {str(e)}"))

    def unwrap_or_raise(self, error: Exception) -> T:
        """Get value or raise appropriate error"""
        if self._value is None:
            raise error
        return self._value

    def unwrap_o_else(self, default: T) -> T:
        """Get value or return default"""
        return self._value if self._value is not None else default

    def or_else(self, default: "Maybe[T]") -> "Maybe[T]":
        """Get value or return default"""
        return self if self._value is not None else default

    def unwrap(self) -> T:
        """Get value or raise ValueError"""
        if self._value is None:
            if self._error is not None:
                if isinstance(self._error, Exception):
                    raise self._error
                raise ValueError(str(self._error))
            raise ValueError("Cannot unwrap None value")
        return self._value

    @property
    def error(self) -> Optional[Exception | str]:
        """Access the stored error"""
        return self._error
