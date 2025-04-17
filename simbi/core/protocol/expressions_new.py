from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
import math

# type defs for clarity
NodeId = int
OpType = str
NodeAttrs = Dict[str, Any]
NodeDef = Tuple[OpType, Tuple[NodeId, ...], NodeAttrs]
T = TypeVar("T")
GraphInputs = Dict[str, float]


class ExprGraph:
    """Immutable directed acyclic graph of expressions."""

    def __init__(self) -> None:
        self._nodes: Dict[NodeId, NodeDef] = {}
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

    def compile(self, outputs: List[Expr]) -> CompiledExpr:
        """Prepare the graph for evaluation with specific outputs."""
        return CompiledExpr(self, outputs)

    def nodes(self) -> Dict[NodeId, NodeDef]:
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

    # Arithmetic operators
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

    def __mul__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("multiply", self._node_id, other_expr._node_id),
        )

    def __truediv__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("divide", self._node_id, other_expr._node_id),
        )

    def __pow__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("power", self._node_id, other_expr._node_id),
        )

    def __neg__(self) -> Expr:
        return Expr(self._graph, self._graph.add_node("negate", self._node_id))

    # Comparison operators
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

    def _ensure_expr(self, value: Union[Expr, float, int]) -> Expr:
        """Convert a value to an expression in this graph."""
        if isinstance(value, Expr) and value._graph is self._graph:
            return value
        return constant(float(value), self._graph)

    # pattern matching
    def match(
        self, patterns: Dict[str, Callable[[Expr, Tuple[NodeId, ...], NodeAttrs], T]]
    ) -> Optional[T]:
        """Pattern match on node type."""
        node_def = self._graph.get_node(self._node_id)
        if node_def is None:
            return None

        op, inputs, attrs = node_def
        handler = patterns.get(op, patterns.get("default"))
        return handler(self, inputs, attrs) if handler else None

    # Function composition
    def pipe(self, *funcs: Callable[[Expr], Expr]) -> Expr:
        """Apply a sequence of functions to this expression."""
        result = self
        for f in funcs:
            result = f(result)
        return result


# Factory functions
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


# Math functions
def sqrt(expr: Expr) -> Expr:
    """Square root function."""
    return Expr(expr._graph, expr._graph.add_node("sqrt", expr._node_id))


def sin(expr: Expr) -> Expr:
    """Sine function."""
    return Expr(expr._graph, expr._graph.add_node("sin", expr._node_id))


def cos(expr: Expr) -> Expr:
    """Cosine function."""
    return Expr(expr._graph, expr._graph.add_node("cos", expr._node_id))


# Higher-order functions
def map_expr(f: Callable[[Expr], Expr], exprs: List[Expr]) -> List[Expr]:
    """Map a function over expressions."""
    return [f(expr) for expr in exprs]


def compose(*funcs: Callable[[T], T]) -> Callable[[T], T]:
    """Function composition."""

    def composed(x: T) -> T:
        result = x
        for f in reversed(funcs):
            result = f(result)
        return result

    return composed


# Evaluator
class CompiledExpr:
    """Compiled expression for efficient evaluation."""

    def __init__(self, graph: ExprGraph, outputs: List[Expr]) -> None:
        self._graph: ExprGraph = graph
        self._output_ids: List[NodeId] = [out._node_id for out in outputs]
        # Topologically sort nodes for evaluation
        self._eval_order: List[NodeId] = self._sort_nodes()

    def _sort_nodes(self) -> List[NodeId]:
        """Topologically sort nodes for evaluation."""
        # Identify nodes needed for outputs
        needed_nodes: Set[NodeId] = set()
        to_process: List[NodeId] = list(self._output_ids)

        while to_process:
            node_id = to_process.pop()
            if node_id in needed_nodes:
                continue

            needed_nodes.add(node_id)
            node_def = self._graph.get_node(node_id)
            if node_def:
                _, inputs, _ = node_def
                to_process.extend(inputs)

        # Topological sort
        result: List[NodeId] = []
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

    def evaluate(self, **inputs: float) -> List[float]:
        """Evaluate the expression with given inputs."""
        # Map from node IDs to computed values
        values: Dict[NodeId, float] = {}

        # Evaluate nodes in topological order
        for node_id in self._eval_order:
            node_def = self._graph.get_node(node_id)
            if not node_def:
                raise ValueError(f"Node {node_id} not found in graph")

            op, input_ids, attrs = node_def

            # Handle different node types
            if op == "constant":
                values[node_id] = attrs["value"]
            elif op == "variable":
                values[node_id] = inputs.get(attrs["name"], 0.0)
            elif op == "parameter":
                values[node_id] = inputs.get(f"param_{attrs['param_idx']}", 0.0)
            elif op == "add":
                values[node_id] = values[input_ids[0]] + values[input_ids[1]]
            elif op == "subtract":
                values[node_id] = values[input_ids[0]] - values[input_ids[1]]
            elif op == "multiply":
                values[node_id] = values[input_ids[0]] * values[input_ids[1]]
            elif op == "divide":
                denominator = values[input_ids[1]]
                if denominator == 0.0:
                    values[node_id] = 0.0  # Handle division by zero
                else:
                    values[node_id] = values[input_ids[0]] / denominator
            elif op == "power":
                values[node_id] = values[input_ids[0]] ** values[input_ids[1]]
            elif op == "negate":
                values[node_id] = -values[input_ids[0]]
            elif op == "sqrt":
                val = values[input_ids[0]]
                if val < 0.0:
                    values[node_id] = 0.0  # Handle negative sqrt
                else:
                    values[node_id] = math.sqrt(val)
            elif op == "sin":
                values[node_id] = math.sin(values[input_ids[0]])
            elif op == "cos":
                values[node_id] = math.cos(values[input_ids[0]])
            # More operations can be added here

        # Return output values
        return [values[out_id] for out_id in self._output_ids]

    def serialize(self) -> Dict[str, Any]:
        """Serialize the compiled expression for C++ evaluation."""
        expressions: List[Dict[str, Any]] = []

        # Map from our internal node ids to serialized indices
        node_map: Dict[NodeId, int] = {}

        for node_id in self._eval_order:
            node_def = self._graph.get_node(node_id)
            if not node_def:
                continue

            op, input_ids, attrs = node_def
            node_idx = len(expressions)
            node_map[node_id] = node_idx

            # Convert to serialized format
            if op == "constant":
                expressions.append({"op": "CONSTANT", "value": attrs["value"]})
            elif op == "variable":
                if attrs["name"] == "x":
                    expressions.append({"op": "VARIABLE_X1"})
                elif attrs["name"] == "y":
                    expressions.append({"op": "VARIABLE_X2"})
                elif attrs["name"] == "z":
                    expressions.append({"op": "VARIABLE_X3"})
                elif attrs["name"] == "t":
                    expressions.append({"op": "VARIABLE_T"})
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
            # Add other operations as needed

        # Map output indices
        output_indices = [node_map[out_id] for out_id in self._output_ids]

        return {"expressions": expressions, "output_indices": output_indices}
