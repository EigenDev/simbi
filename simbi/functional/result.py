# =============================================================================
# result.py
#
# result monad for error handling without exceptions.
# represents either success (Ok) or failure (Err), with safe chaining.
#
# usage:
#   Ok(value).map(transform).unwrap()
#   Err("error").unwrap_or(default)
# =============================================================================
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")

# sentinel for missing value
_MISSING = object()


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """
    result type representing either success (Ok) or failure (Err).

    never construct directly - use Ok(value) or Err(error) helpers.
    """

    _value: T | object = _MISSING
    _error: E | object = _MISSING
    _is_ok: bool = field(default=False, init=False)

    def __post_init__(self):
        # determine state based on which value was provided
        is_ok = self._value is not _MISSING
        is_err = self._error is not _MISSING

        # enforce exactly one of value or error
        if not (is_ok ^ is_err):
            raise ValueError("Result must have exactly one of value or error")

        # the dataclass is frozen, so field assignment goes through object.__setattr__
        object.__setattr__(self, "_is_ok", is_ok)

    @property
    def value(self) -> T:
        """access value (raises if Err)."""
        if not self._is_ok:
            raise RuntimeError(f"called .value on Err: {self._error}")
        return self._value  # type: ignore

    @property
    def error(self) -> E:
        """access error (raises if Ok)."""
        if self._is_ok:
            raise RuntimeError("called .error on Ok")
        return self._error  # type: ignore

    def is_ok(self) -> bool:
        """check if Result is Ok."""
        return self._is_ok

    def is_err(self) -> bool:
        """check if Result is Err."""
        return not self._is_ok

    def map(self, f: Callable[[T], U]) -> "Result[U, E]":
        """apply function to Ok value, propagate Err."""
        if self.is_err():
            return Result(_error=self._error)
        try:
            return Result(_value=f(self._value))  # type: ignore
        except Exception as e:
            # if f raises, convert to Err
            return Result(_error=e)  # type: ignore

    def and_then(self, f: Callable[[T], "Result[U, E]"]) -> "Result[U, E]":
        """flat_map / bind: chain Result-returning functions."""
        if self.is_err():
            return Result(_error=self._error)
        try:
            return f(self._value)  # type: ignore
        except Exception as e:
            return Result(_error=e)  # type: ignore

    def unwrap(self) -> T:
        """extract value or raise error."""
        if self.is_err():
            if isinstance(self._error, Exception):
                raise self._error
            raise RuntimeError(f"Result error: {self._error}")
        return self._value  # type: ignore

    def unwrap_or(self, default: T) -> T:
        """extract value or return default."""
        if self.is_err():
            return default
        return self._value  # type: ignore

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """extract value or compute default from error."""
        if self.is_err():
            return f(self._error)  # type: ignore
        return self._value  # type: ignore


def Ok(value: T) -> Result[T, Any]:
    """create success Result."""
    return Result(_value=value)


def Err(error: E) -> Result[Any, E]:
    """create failure Result."""
    return Result(_error=error)
