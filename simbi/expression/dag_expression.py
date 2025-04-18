from __future__ import annotations
from typing import Any, Callable, Optional, Set, TypeVar, Union
import math

# type defs for clarity
NodeId = int
OpType = str
NodeAttrs = dict[str, Any]
NodeDef = tuple[OpType, tuple[NodeId, ...], NodeAttrs]
T = TypeVar("T")
GraphInputs = dict[str, float]

__all__ = [
    "ExprGraph",
    "Expr",
    "constant",
    "variable",
    "parameter",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "log",
    "log10",
    "asin",
    "acos",
    "atan",
    "exp",
    "max_expr",
    "min_expr",
    "if_then_else",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_not",
    "bitwise_left_shift",
    "bitwise_right_shift",
]

X1_ALIASES = ["x", "r", "x1"]
X2_ALIASES = ["y", "theta", "x2", "phi"]
X3_ALIASES = ["z", "phi", "x3"]


class ExprGraph:
    """Immutable directed acyclic graph of expressions."""

    def __init__(self) -> None:
        self._nodes: dict[NodeId, NodeDef] = {}
        self._next_id: int = 0

    def add_node(self, op_type: str, *inputs: NodeId, **attrs: Any) -> NodeId:
        """Add a node to the graph, returning its unique ID."""
        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = (op_type, inputs, attrs)
        return node_id

    def get_node(self, node_id: NodeId) -> Optional[NodeDef]:
        """Get node definition by ID."""
        return self._nodes.get(node_id)

    def compile(self, outputs: list[Expr]) -> CompiledExpr:
        """Prepare the graph for evaluation with specific outputs."""
        return CompiledExpr(self, outputs)

    def nodes(self) -> dict[NodeId, NodeDef]:
        """Get all nodes in the graph (immutable copy)."""
        return dict(self._nodes)


class Expr:
    """Functional, immutable expression reference."""

    def __init__(self, graph: ExprGraph, node_id: NodeId) -> None:
        self._graph: ExprGraph = graph
        self._node_id: NodeId = node_id

    @property
    def graph(self) -> ExprGraph:
        """Get the underlying graph."""
        return self._graph

    @property
    def node_id(self) -> NodeId:
        """Get the node ID in the graph."""
        return self._node_id

    # arithmetic operators
    def __add__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph, self._graph.add_node("add", self._node_id, other_expr._node_id)
        )

    def __sub__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("subtract", self._node_id, other_expr._node_id),
        )

    def __radd__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("add", other_expr._node_id, self._node_id),
        )

    def __rsub__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("subtract", other_expr._node_id, self._node_id),
        )

    def __mul__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("multiply", self._node_id, other_expr._node_id),
        )

    def __rmul__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("multiply", other_expr._node_id, self._node_id),
        )

    def __truediv__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("divide", self._node_id, other_expr._node_id),
        )

    def __rtruediv__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("divide", other_expr._node_id, self._node_id),
        )

    def __pow__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("power", self._node_id, other_expr._node_id),
        )

    def __neg__(self) -> Expr:
        return Expr(self._graph, self._graph.add_node("negate", self._node_id))

    def __pos__(self) -> Expr:
        return self

    # comparison operators
    def __lt__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph, self._graph.add_node("lt", self._node_id, other_expr._node_id)
        )

    def __gt__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph, self._graph.add_node("gt", self._node_id, other_expr._node_id)
        )

    def __le__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph, self._graph.add_node("le", self._node_id, other_expr._node_id)
        )

    def __ge__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph, self._graph.add_node("ge", self._node_id, other_expr._node_id)
        )

    def _ensure_expr(self, value: Union[Expr, float, int]) -> Expr:
        """Convert a value to an expression in this graph."""
        if isinstance(value, Expr):
            if value._graph is self._graph:
                return value
            else:
                raise ValueError(
                    "Expressions from different graphs cannot be combined directly. "
                    "Consider creating a new graph and rebuilding both expressions."
                )

        return constant(float(value), self._graph)

    # pattern matching
    def match(
        self, patterns: dict[str, Callable[[Expr, tuple[NodeId, ...], NodeAttrs], T]]
    ) -> Optional[T]:
        """Pattern match on node type."""
        node_def = self._graph.get_node(self._node_id)
        if node_def is None:
            return None

        op, inputs, attrs = node_def
        handler = patterns.get(op, patterns.get("default"))
        return handler(self, inputs, attrs) if handler else None

    # function composition
    def pipe(self, *funcs: Callable[[Expr], Expr]) -> Expr:
        """Apply a sequence of functions to this expression."""
        result = self
        for f in funcs:
            result = f(result)
        return result


# factory functions
def constant(value: float, graph: Optional[ExprGraph] = None) -> Expr:
    """Create a constant expression."""
    g = graph or ExprGraph()
    return Expr(g, g.add_node("constant", value=float(value)))


def variable(name: str, graph: Optional[ExprGraph] = None) -> Expr:
    """Create a variable expression."""
    g = graph or ExprGraph()
    return Expr(g, g.add_node("variable", name=name))


def parameter(idx: int, graph: Optional[ExprGraph] = None) -> Expr:
    """Create a parameter expression."""
    g = graph or ExprGraph()
    return Expr(g, g.add_node("parameter", param_idx=idx))


# math functions
def sqrt(expr: Expr) -> Expr:
    """Square root function."""
    return Expr(expr._graph, expr._graph.add_node("sqrt", expr._node_id))


def sin(expr: Expr) -> Expr:
    """Sine function."""
    return Expr(expr._graph, expr._graph.add_node("sin", expr._node_id))


def cos(expr: Expr) -> Expr:
    """Cosine function."""
    return Expr(expr._graph, expr._graph.add_node("cos", expr._node_id))


def tan(expr: Expr) -> Expr:
    """Tangent functions"""
    return Expr(expr._graph, expr._graph.add_node("tan", expr._node_id))


def log(expr: Expr) -> Expr:
    """Natural log"""
    return Expr(expr._graph, expr._graph.add_node("log", expr._node_id))


def log10(expr: Expr) -> Expr:
    """Base 10 log"""
    return Expr(expr._graph, expr._graph.add_node("log10", expr._node_id))


def asin(expr: Expr) -> Expr:
    """Inverse sine function."""
    return Expr(expr._graph, expr._graph.add_node("asin", expr._node_id))


def acos(expr: Expr) -> Expr:
    """Inverse cosine function"""
    return Expr(expr._graph, expr._graph.add_node("acos", expr._node_id))


def atan(expr: Expr) -> Expr:
    """Inverse tangent function"""
    return Expr(expr._graph, expr._graph.add_node("atan", expr._node_id))


def atan2(expr1: Expr, expr2: Expr) -> Expr:
    return Expr(
        expr1._graph, expr1._graph.add_node("atan2", expr1._node_id, expr2._node_id)
    )


def sinh(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("sinh", expr._node_id))


def cosh(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("cosh", expr._node_id))


def tanh(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("tanh", expr._node_id))


def asinh(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("asinh", expr._node_id))


def acosh(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("acosh", expr._node_id))


def atanh(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("atanh", expr._node_id))


def exp(expr: Expr) -> Expr:
    """Exponential function."""
    return Expr(expr._graph, expr._graph.add_node("exp", expr._node_id))


def max_expr(expr1: Expr, expr2: Expr) -> Expr:
    """Maximum of two expressions."""
    return Expr(
        expr1._graph, expr1._graph.add_node("max", expr1._node_id, expr2._node_id)
    )


def min_expr(expr1: Expr, expr2: Expr) -> Expr:
    """Minimum of two expressions."""
    return Expr(
        expr1._graph, expr1._graph.add_node("min", expr1._node_id, expr2._node_id)
    )


def bitwise_and(expr1: Expr, expr2: Expr) -> Expr:
    return Expr(
        expr1._graph,
        expr1._graph.add_node("bitwise_and", expr1._node_id, expr2._node_id),
    )


def bitwise_or(expr1: Expr, expr2: Expr) -> Expr:
    return Expr(
        expr1._graph,
        expr1._graph.add_node("bitwise_or", expr1._node_id, expr2._node_id),
    )


def bitwise_xor(expr1: Expr, expr2: Expr) -> Expr:
    return Expr(
        expr1._graph,
        expr1._graph.add_node("bitwise_xor", expr1._node_id, expr2._node_id),
    )


def bitwise_not(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("bitwise_not", expr._node_id))


def bitwise_left_shift(expr1: Expr, expr2: Expr) -> Expr:
    return Expr(
        expr1._graph,
        expr1._graph.add_node("bitwise_left_shift", expr1._node_id, expr2._node_id),
    )


def bitwise_right_shift(expr1: Expr, expr2: Expr) -> Expr:
    return Expr(
        expr1._graph,
        expr1._graph.add_node("bitwise_right_shift", expr1._node_id, expr2._node_id),
    )


def sgn(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("sgn", expr._node_id))


def if_then_else(condition: Expr, true_case: Expr, false_case: Expr) -> Expr:
    """If-then-else expression."""
    return Expr(
        condition._graph,
        condition._graph.add_node(
            "if_then_else", condition._node_id, true_case._node_id, false_case._node_id
        ),
    )


# higher-order functions
def map_expr(f: Callable[[Expr], Expr], exprs: list[Expr]) -> list[Expr]:
    """Map a function over expressions."""
    return [f(expr) for expr in exprs]


# evaluator
class CompiledExpr:
    """Compiled expression for efficient evaluation."""

    def __init__(self, graph: ExprGraph, outputs: list[Expr]) -> None:
        self._graph: ExprGraph = graph
        self._output_ids: list[NodeId] = [out._node_id for out in outputs]
        # topologically sort nodes for evaluation
        self._eval_order: list[NodeId] = self._sort_nodes()

    def _sort_nodes(self) -> list[NodeId]:
        """Topologically sort nodes for evaluation."""
        # identify nodes needed for outputs
        needed_nodes: Set[NodeId] = set()
        to_process: list[NodeId] = list(self._output_ids)

        while to_process:
            node_id = to_process.pop()
            if node_id in needed_nodes:
                continue

            needed_nodes.add(node_id)
            node_def = self._graph.get_node(node_id)
            if node_def:
                _, inputs, _ = node_def
                to_process.extend(inputs)

        # topological sort
        result: list[NodeId] = []
        visited: Set[NodeId] = set()
        temp_visited: Set[NodeId] = set()

        def visit(node_id: NodeId) -> None:
            if node_id in visited:
                return
            if node_id in temp_visited:
                raise ValueError("Cyclic dependency detected in expression graph")

            temp_visited.add(node_id)

            node_def = self._graph.get_node(node_id)
            if node_def:
                _, inputs, _ = node_def
                for input_id in inputs:
                    visit(input_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)

        for node_id in needed_nodes:
            visit(node_id)

        return result

    def evaluate(self, **inputs: float) -> list[float]:
        """Evaluate the expression with given inputs."""
        # map from node IDs to computed values
        values: dict[NodeId, float] = {}

        # evaluate nodes in topological order
        for node_id in self._eval_order:
            node_def = self._graph.get_node(node_id)
            if not node_def:
                raise ValueError(f"Node {node_id} not found in graph")

            op, input_ids, attrs = node_def

            # handle different node types
            if op == "constant":
                values[node_id] = attrs["value"]
            elif op == "variable":
                values[node_id] = inputs.get(attrs["name"], 0.0)
            elif op == "parameter":
                values[node_id] = inputs.get(f"param_{attrs['param_idx']}", 0.0)
            # basic arithmetic
            elif op == "add":
                values[node_id] = values[input_ids[0]] + values[input_ids[1]]
            elif op == "subtract":
                values[node_id] = values[input_ids[0]] - values[input_ids[1]]
            elif op == "multiply":
                values[node_id] = values[input_ids[0]] * values[input_ids[1]]
            elif op == "divide":
                denominator = values[input_ids[1]]
                if denominator == 0.0:
                    values[node_id] = 0.0  # we handle division by zero
                else:
                    values[node_id] = values[input_ids[0]] / denominator
            # math functions
            elif op == "power":
                values[node_id] = values[input_ids[0]] ** values[input_ids[1]]
            elif op == "negate":
                values[node_id] = -values[input_ids[0]]
            elif op == "sqrt":
                val = values[input_ids[0]]
                if val < 0.0:
                    values[node_id] = 0.0  # we handle negative sqrt
                else:
                    values[node_id] = math.sqrt(val)
            elif op == "sin":
                values[node_id] = math.sin(values[input_ids[0]])
            elif op == "cos":
                values[node_id] = math.cos(values[input_ids[0]])
            elif op == "tan":
                values[node_id] = math.tan(values[input_ids[0]])
            elif op == "sgn":
                values[node_id] = math.copysign(1, values[input_ids[0]])
            elif op == "log":
                values[node_id] = math.log(values[input_ids[0]])
            elif op == "log10":
                values[node_id] = math.log10(values[input_ids[0]])
            elif op == "asin":
                values[node_id] = math.asin(values[input_ids[0]])
            elif op == "acos":
                values[node_id] = math.acos(values[input_ids[0]])
            elif op == "atan":
                values[node_id] = math.atan(values[input_ids[0]])
            elif op == "exp":
                values[node_id] = math.exp(values[input_ids[0]])
            elif op == "sinh":
                values[node_id] = math.sinh(values[input_ids[0]])
            elif op == "cosh":
                values[node_id] = math.cosh(values[input_ids[0]])
            elif op == "tanh":
                values[node_id] = math.tanh(values[input_ids[0]])
            elif op == "asinh":
                values[node_id] = math.asinh(values[input_ids[0]])
            elif op == "acosh":
                values[node_id] = math.acosh(values[input_ids[0]])
            elif op == "atanh":
                values[node_id] = math.atanh(values[input_ids[0]])
            elif op == "atan2":
                values[node_id] = math.atan2(values[input_ids[0]], values[input_ids[1]])
            # binary ops
            elif op == "max":
                values[node_id] = max(values[input_ids[0]], values[input_ids[1]])
            elif op == "min":
                values[node_id] = min(values[input_ids[0]], values[input_ids[1]])
            # bitwise ops
            elif op == "bitwise_and":
                values[node_id] = int(values[input_ids[0]]) & int(values[input_ids[1]])
            elif op == "bitwise_or":
                values[node_id] = int(values[input_ids[0]]) | int(values[input_ids[1]])
            elif op == "bitwise_xor":
                values[node_id] = int(values[input_ids[0]]) ^ int(values[input_ids[1]])
            elif op == "bitwise_not":
                values[node_id] = ~int(values[input_ids[0]])
            elif op == "bitwise_left_shift":
                values[node_id] = int(values[input_ids[0]]) << int(values[input_ids[1]])
            elif op == "bitwise_right_shift":
                values[node_id] = int(values[input_ids[0]]) >> int(values[input_ids[1]])
            elif op == "if_then_else":
                condition = values[input_ids[0]]
                if condition:
                    values[node_id] = values[input_ids[1]]
                else:
                    values[node_id] = values[input_ids[2]]

        # return output values
        return [values[out_id] for out_id in self._output_ids]

    def serialize(self) -> dict[str, object]:
        """Serialize the compiled expression for C++ evaluation."""
        expressions: list[dict[str, Any]] = []

        # map from our internal node ids to serialized indices
        node_map: dict[NodeId, int] = {}

        for node_id in self._eval_order:
            node_def = self._graph.get_node(node_id)
            if not node_def:
                continue

            op, input_ids, attrs = node_def
            node_idx = len(expressions)
            node_map[node_id] = node_idx

            # convert to serialized format
            if op == "constant":
                expressions.append({"op": "CONSTANT", "value": attrs["value"]})
            elif op == "variable":
                if attrs["name"] in X1_ALIASES:
                    expressions.append({"op": "VARIABLE_X1"})
                elif attrs["name"] in X2_ALIASES:
                    expressions.append({"op": "VARIABLE_X2"})
                elif attrs["name"] in X3_ALIASES:
                    expressions.append({"op": "VARIABLE_X3"})
                elif attrs["name"] == "t":
                    expressions.append({"op": "VARIABLE_T"})
                elif attrs["name"] == "dt":
                    expressions.append({"op": "VARIABLE_DT"})
            elif op == "parameter":
                expressions.append({"op": "PARAMETER", "param_idx": attrs["param_idx"]})
            elif op == "add":
                expressions.append(
                    {
                        "op": "ADD",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "subtract":
                expressions.append(
                    {
                        "op": "SUBTRACT",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "multiply":
                expressions.append(
                    {
                        "op": "MULTIPLY",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "divide":
                expressions.append(
                    {
                        "op": "DIVIDE",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "negate":
                expressions.append(
                    {"op": "NEG", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "sqrt":
                expressions.append(
                    {"op": "SQRT", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "sin":
                expressions.append(
                    {"op": "SIN", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "cos":
                expressions.append(
                    {"op": "COS", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "tan":
                expressions.append(
                    {"op": "TAN", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "log":
                expressions.append(
                    {"op": "LOG", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "log10":
                expressions.append(
                    {"op": "LOG10", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "asin":
                expressions.append(
                    {"op": "ASIN", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "acos":
                expressions.append(
                    {"op": "ACOS", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "atan":
                expressions.append(
                    {"op": "ATAN", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "atan2":
                expressions.append(
                    {
                        "op": "ATAN2",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "sinh":
                expressions.append(
                    {"op": "SINH", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "cosh":
                expressions.append(
                    {"op": "COSH", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "tanh":
                expressions.append(
                    {"op": "TANH", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "asinh":
                expressions.append(
                    {"op": "ASINH", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "acosh":
                expressions.append(
                    {"op": "ACOSH", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "atanh":
                expressions.append(
                    {"op": "ATANH", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "sgn":
                expressions.append(
                    {"op": "SGN", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "exp":
                expressions.append(
                    {"op": "EXP", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "power":
                expressions.append(
                    {
                        "op": "POW",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "max":
                expressions.append(
                    {
                        "op": "MAX",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "min":
                expressions.append(
                    {
                        "op": "MIN",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "bitwise_and":
                expressions.append(
                    {
                        "op": "BITWISE_AND",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "bitwise_or":
                expressions.append(
                    {
                        "op": "BITWISE_OR",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "bitwise_xor":
                expressions.append(
                    {
                        "op": "BITWISE_XOR",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "bitwise_not":
                expressions.append(
                    {"op": "BITWISE_NOT", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "bitwise_left_shift":
                expressions.append(
                    {
                        "op": "BITWISE_LEFT_SHIFT",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "bitwise_right_shift":
                expressions.append(
                    {
                        "op": "BITWISE_RIGHT_SHIFT",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "if_then_else":
                expressions.append(
                    {
                        "op": "IF_THEN_ELSE",
                        "condition": node_map[input_ids[0]],
                        "true_case": node_map[input_ids[1]],
                        "false_case": node_map[input_ids[2]],
                    }
                )
            else:
                raise ValueError(f"Unknown operation: {op}")

        # map output indices
        output_indices = [node_map[out_id] for out_id in self._output_ids]

        max_param_idx = -1
        for node_id in self._eval_order:
            node_def = self._graph.get_node(node_id)
            if node_def and node_def[0] == "parameter":
                max_param_idx = max(max_param_idx, node_def[2].get("param_idx", -1))

        return {
            "expressions": expressions,
            "output_indices": output_indices,
            "param_count": max_param_idx + 1,
        }
