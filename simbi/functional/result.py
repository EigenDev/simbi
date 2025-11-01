from dataclasses import dataclass
from typing import Callable, Generic, Self, TypeVar, Any

T = TypeVar("T")
U = TypeVar("U")


@dataclass(frozen=True)
class Result(Generic[T]):
    value: T | None = None
    error: Exception | None = None

    @classmethod
    def ok(cls, value: T) -> Self:
        return cls(value=value)

    @classmethod
    def err(cls, error: Exception) -> Self:
        return cls(error=error)

    def and_then(
        self, f: Callable[[T, *tuple[Any, ...]], "Result[U]"]
    ) -> "Result[U]":
        if self.error is not None:
            return Result.err(self.error)
        else:
            if self.value is None:
                return Result.err(ValueError("No value present"))
            return f(self.value)

    def map(self, f: Callable[[T], U]) -> "Result[U]":
        try:
            if self.error is not None:
                return Result.err(self.error)
            if self.value is None:
                return Result.err(ValueError("No value present"))
            return Result.ok(f(self.value))
        except Exception as e:
            return Result.err(e)

    @property
    def is_ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> T:
        if self.error is not None:
            raise self.error
        if self.value is None:
            raise ValueError("No value present")
        return self.value
