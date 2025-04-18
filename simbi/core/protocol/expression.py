"""
File to house expression tree logic to be serialized
into a form that both the CPU and GPU can understand.
"""

from typing import Any, Optional, Union, Tuple, TypeVar


# type variable for numeric types
T = TypeVar("T", bound=Union[int, float])


__all__ = ["Expr", "serialize_expressions"]


class Expr:
    """Expression builder for user-defined functions in simulations."""

    def __init__(
        self,
        op: Union[str, int, float],
        *args: "Expr",
        value: Optional[float] = None,
        param_idx: Optional[int] = None,
    ) -> None:
        """Initialize an expression node.

        Args:
            op: Operation type
            args: Child expressions
            value: Constant value if op is CONSTANT
            param_idx: Parameter index if op is PARAMETER
        """
        self.value: Optional[float]
        self.op: str
        self.args: Tuple["Expr", ...]
        self.param_idx: Optional[int]

        if isinstance(op, (int, float)):
            self.op = "CONSTANT"
            self.value = float(op)
        else:
            self.op = op
            self.value = value
        self.args = args
        self.param_idx = param_idx

    # arithmetic operators
    def __add__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("ADD", self, other)

    def __sub__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("SUBTRACT", self, other)

    def __mul__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("MULTIPLY", self, other)

    def __rmul__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("MULTIPLY", other, self)

    def __truediv__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("DIVIDE", self, other)

    def __pow__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("POW", self, other)

    def __neg__(self) -> "Expr":
        return Expr("NEG", self)

    # Comparison operators
    def __lt__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("LT", self, other)

    def __gt__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("GT", self, other)

    def __eq__(self, other: Union["Expr", float, int]) -> "Expr":  # type: ignore
        other = self._ensure_expr(other)
        return Expr("EQ", self, other)

    def __le__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("LE", self, other)

    def __ge__(self, other: Union["Expr", float, int]) -> "Expr":
        other = self._ensure_expr(other)
        return Expr("GE", self, other)

    # helper to convert literals to expressions
    @staticmethod
    def _ensure_expr(value: Union["Expr", float, int, str]) -> "Expr":
        """Ensure value is an expression.

        Args:
            value: Value to convert to expression

        Returns:
            Expression object
        """
        if isinstance(value, Expr):
            return value
        return Expr("CONSTANT", value=float(value))

    # math functions
    @staticmethod
    def sqrt(x: Union["Expr", float, int]) -> "Expr":
        """Square root function.

        Args:
            x: Input expression

        Returns:
            Square root expression
        """
        x = Expr._ensure_expr(x)
        return Expr("SQRT", x)

    @staticmethod
    def sin(x: Union["Expr", float, int]) -> "Expr":
        """Sine function.

        Args:
            x: Input expression (radians)

        Returns:
            Sine expression
        """
        x = Expr._ensure_expr(x)
        return Expr("SIN", x)

    @staticmethod
    def cos(x: Union["Expr", float, int]) -> "Expr":
        """Cosine function.

        Args:
            x: Input expression (radians)

        Returns:
            Cosine expression
        """
        x = Expr._ensure_expr(x)
        return Expr("COS", x)

    @staticmethod
    def tan(x: Union["Expr", float, int]) -> "Expr":
        """Tangent function.

        Args:
            x: Input expression (radians)

        Returns:
            Tangent expression
        """
        x = Expr._ensure_expr(x)
        return Expr("TAN", x)

    @staticmethod
    def exp(x: Union["Expr", float, int]) -> "Expr":
        """Exponential function.

        Args:
            x: Input expression

        Returns:
            Exponential expression
        """
        x = Expr._ensure_expr(x)
        return Expr("EXP", x)

    @staticmethod
    def log(x: Union["Expr", float, int]) -> "Expr":
        """Natural logarithm.

        Args:
            x: Input expression

        Returns:
            Logarithm expression
        """
        x = Expr._ensure_expr(x)
        return Expr("LOG", x)

    @staticmethod
    def abs(x: Union["Expr", float, int]) -> "Expr":
        """Absolute value function.

        Args:
            x: Input expression

        Returns:
            Absolute value expression
        """
        x = Expr._ensure_expr(x)
        return Expr("ABS", x)

    @staticmethod
    def log10(x: Union["Expr", float, int]) -> "Expr":
        """Base-10 logarithm.

        Args:
            x: Input expression

        Returns:
            Base-10 logarithm expression
        """
        x = Expr._ensure_expr(x)
        return Expr("LOG10", x)

    @staticmethod
    def asin(x: Union["Expr", float, int]) -> "Expr":
        """Arc sine function.

        Args:
            x: Input expression

        Returns:
            Arc sine expression
        """
        x = Expr._ensure_expr(x)
        return Expr("ASIN", x)

    @staticmethod
    def acos(x: Union["Expr", float, int]) -> "Expr":
        """Arc cosine function.

        Args:
            x: Input expression

        Returns:
            Arc cosine expression
        """
        x = Expr._ensure_expr(x)
        return Expr("ACOS", x)

    @staticmethod
    def max(x: Union["Expr", float, int], y: Union["Expr", float, int]) -> "Expr":
        """Maximum of two values.

        Args:
            x: First expression
            y: Second expression

        Returns:
            Maximum expression
        """
        x = Expr._ensure_expr(x)
        y = Expr._ensure_expr(y)
        return Expr("MAX", x, y)

    @staticmethod
    def floor(x: Union["Expr", float, int]) -> "Expr":
        """Floor function.

        Args:
            x: Input expression

        Returns:
            Floor expression
        """
        x = Expr._ensure_expr(x)
        return Expr("FLOOR", x)

    @staticmethod
    def ceil(x: Union["Expr", float, int]) -> "Expr":
        """Ceiling function.

        Args:
            x: Input expression

        Returns:
            Ceiling expression
        """
        x = Expr._ensure_expr(x)
        return Expr("CEIL", x)

    @staticmethod
    def round(x: Union["Expr", float, int]) -> "Expr":
        """Round function.

        Args:
            x: Input expression

        Returns:
            Rounded expression
        """
        x = Expr._ensure_expr(x)
        return Expr("ROUND", x)

    @staticmethod
    def min(x: Union["Expr", float, int], y: Union["Expr", float, int]) -> "Expr":
        """Minimum of two values.

        Args:
            x: First expression
            y: Second expression

        Returns:
            Minimum expression
        """
        x = Expr._ensure_expr(x)
        y = Expr._ensure_expr(y)
        return Expr("MIN", x, y)

    @staticmethod
    def if_then_else(
        condition: Union["Expr", bool],
        then_expr: Union["Expr", float, int],
        else_expr: Union["Expr", float, int],
    ) -> "Expr":
        """Conditional expression.

        Args:
            condition: Boolean expression
            then_expr: Expression if condition is true
            else_expr: Expression if condition is false

        Returns:
            Conditional expression
        """
        condition = Expr._ensure_expr(condition)
        then_expr = Expr._ensure_expr(then_expr)
        else_expr = Expr._ensure_expr(else_expr)
        return Expr("IF_THEN_ELSE", condition, then_expr, else_expr)

    # Logical operators
    def __and__(self, other: Union["Expr", bool, float, int]) -> "Expr":
        """Logical AND.

        Args:
            other: Second expression

        Returns:
            AND expression
        """
        other = self._ensure_expr(other)
        return Expr("AND", self, other)

    def __or__(self, other: Union["Expr", bool, float, int]) -> "Expr":
        """Logical OR.

        Args:
            other: Second expression

        Returns:
            OR expression
        """
        other = self._ensure_expr(other)
        return Expr("OR", self, other)

    def __invert__(self) -> "Expr":
        """Logical NOT.

        Returns:
            NOT expression
        """
        return Expr("NOT", self)

    # Variables and parameters
    @staticmethod
    def x1() -> "Expr":
        """X1 coordinate variable (alternative name).

        Returns:
            X1 coordinate expression
        """
        return Expr("VARIABLE_X1")

    @staticmethod
    def x2() -> "Expr":
        """X2 coordinate variable (alternative name).

        Returns:
            X2 coordinate expression
        """
        return Expr("VARIABLE_X2")

    @staticmethod
    def x3() -> "Expr":
        """X3 coordinate variable (alternative name).

        Returns:
            X3 coordinate expression
        """
        return Expr("VARIABLE_X3")

    @staticmethod
    def t() -> "Expr":
        """Time variable.

        Returns:
            Time expression
        """
        return Expr("VARIABLE_T")

    @staticmethod
    def dt() -> "Expr":
        """Time step variable.

        Returns:
            Time step expression
        """
        return Expr("VARIABLE_DT")

    @staticmethod
    def param(idx: int) -> "Expr":
        """Parameter reference.

        Args:
            idx: Parameter index

        Returns:
            Parameter expression
        """
        return Expr("PARAMETER", param_idx=idx)

    def serialize(
        self,
        expressions: Optional[list[dict[str, Any]]] = None,
        expr_map: Optional[dict[int, int]] = None,
    ) -> int:
        """Serialize expression tree to a format C++ can understand.

        Args:
            expressions: Optional list to store expression nodes
            expr_map: Optional map to track processed expressions

        Returns:
            Index of the root node
        """
        if expressions is None:
            expressions = []
            expr_map = {}

        # Use object id for better performance
        expr_id = id(self)
        if expr_map and expr_id in expr_map:
            return expr_map[expr_id]

        # Create new expression record
        idx = len(expressions)
        if expr_map is not None:
            expr_map[expr_id] = idx

        if self.op == "CONSTANT":
            expressions.append({"op": self.op, "value": self.value})
        elif self.op.startswith("VARIABLE_"):
            expressions.append({"op": self.op})
        elif self.op == "PARAMETER":
            expressions.append({"op": self.op, "param_idx": self.param_idx})
        elif self.op == "IF_THEN_ELSE":
            # Special case for ternary operator
            cond_idx = self.args[0].serialize(expressions, expr_map)
            then_idx = self.args[1].serialize(expressions, expr_map)
            else_idx = self.args[2].serialize(expressions, expr_map)
            expressions.append(
                {
                    "op": self.op,
                    "condition": cond_idx,
                    "then": then_idx,
                    "else": else_idx,
                }
            )
        else:
            # Handle binary/unary operations
            left_idx = self.args[0].serialize(expressions, expr_map)

            if len(self.args) > 1:
                right_idx = self.args[1].serialize(expressions, expr_map)
                expressions.append(
                    {"op": self.op, "left": left_idx, "right": right_idx}
                )
            else:
                expressions.append(
                    {
                        "op": self.op,
                        "left": left_idx,
                        "right": -1,  # Not used for unary ops
                    }
                )

        return idx

    @staticmethod
    def vector(components: list[Union["Expr", float, int]]) -> list["Expr"]:
        """Create a vector of expressions.

        Args:
            components: List of component expressions

        Returns:
            List of expressions representing a vector
        """
        return [Expr._ensure_expr(c) for c in components]


def serialize_expressions(
    expressions: list[Expr],
    parameters: Optional[list[Union[float, int, Expr]]] = None,
) -> dict[str, object]:
    """Serialize expressions for boundary conditions.

    Args:
        expressions: List of expressions for each output component
        output_indices: Indices of the root nodes
        parameters: Optional list of parameters

    Returns:
        Dictionary with serialized expressions and metadata
    """
    # create serialization container
    serialized_exprs: list[dict[str, Any]] = []
    expr_map: dict[int, int] = {}

    # serialize each expression
    # and get their root indices
    output_indices = []
    for expr in expressions:
        root_idx = expr.serialize(serialized_exprs, expr_map)
        output_indices.append(root_idx)

    # Create the final structure
    result = {"expressions": serialized_exprs, "output_indices": output_indices}

    # Add parameters if provided
    if parameters:
        result["parameters"] = parameters

    return result
