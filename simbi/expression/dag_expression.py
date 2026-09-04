from __future__ import annotations

import enum
import math
from typing import Any, Callable, Optional, Sequence, Set, TypeVar, Union


class SourceKind(str, enum.Enum):
    """the conservation law the rust framework wraps a user source field in.
    typo-proof front for the `kind` string crossing into rust's `SourceConfig`
    (str subclass -> serializes to its `.value`). FORCE/COOLING/RELAX are the
    safe primitive-lifted constructors; RAW writes conserved components directly
    to a single slot (the regime-agnostic escape hatch). INJECT writes the whole
    conserved vector [den, mom_0..mom_{D-1}, nrg] additively from one config — a
    mass+momentum+energy deposition (jet/wind), which reaches past the one slot
    RAW targets; like RAW it supplies conserved components, so it is valid on
    relativistic/MHD. SPONGE is the full-state relaxation (buffer zone): it
    relaxes den, mom and nrg toward a reference given as PRIMITIVES
    [kappa, rho_ref, vel_ref.., pre_ref], which the regime converts through its own
    conservation law — so one wire serves newtonian, relativistic and curved
    backgrounds alike, and no closure parameter rides the source. RELAX relaxes the
    velocity alone (density-preserving drag). ROTATING_FRAME takes
    [omega, origin_x, origin_y] and applies the Newtonian Coriolis and centrifugal
    force for constant rotation about the positive z axis."""

    FORCE = "force"
    ROTATING_FRAME = "rotating_frame"
    COOLING = "cooling"
    RELAX = "relax"
    SPONGE = "sponge"
    INJECT = "inject"
    RAW = "raw"


class ReductionOp(str, enum.Enum):
    """how a census combines the cells that land in one bin.

    the accumulated object is a commutative monoid — associative and order-agnostic —
    which is what lets the reduction run in parallel, over blocks, and across restart
    segments. ADD gives moments, histograms, mass budgets and fluxes; MIN and MAX give
    per-bin extrema.

    mean, variance, dispersion and percentile are deliberately absent: each is a
    function of sums, which places it outside the monoid requirement above. register
    `m*v` and `m` and divide in the reader. an api offering variance directly would
    compute `<v^2> - <v>^2`, which
    loses most of its significant digits whenever the mean dominates the dispersion —
    register `m*(v - v_ref)^2` against a known reference instead. a product is absent
    because it overflows to zero or infinity at any realistic cell count.
    """

    ADD = "add"
    MIN = "min"
    MAX = "max"


class ConservedField(str, enum.Enum):
    """conserved slot a `kind=RAW` source targets (`target` field)."""

    DENSITY = "den"
    MOMENTUM = "mom"
    ENERGY = "nrg"


class TableBounds(str, enum.Enum):
    """out-of-bounds behavior for an immutable one-dimensional table."""

    CLAMP = "clamp"
    ZERO = "zero"

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
    "floor",
    "ceil",
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
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_xor",
    "logical_nand",
    "logical_nor",
    "logical_xnor",
    "where",
    "sgn",
]

# "phi" is the azimuth (x3): in 3d spherical (r, theta, phi) it names the third
# axis, leaving x2 (theta) to its own aliases. a 2d-polar azimuthal coordinate is
# written "x2" or "theta".
X1_ALIASES = ["x", "r", "x1"]
X2_ALIASES = ["y", "theta", "x2"]
X3_ALIASES = ["z", "phi", "x3"]

# per-cell fluid-state leaves — let a source read the local state, so the physics
# itself, beyond position and time, is in the user's hands: e.g., cooling ~ rho^2,
# velocity drag ~ -k*vel. these map to the rust `VARIABLE_RHO/VEL{1,2,3}/PRESSURE`
# ops (symbi-hydro::expr_bridge lowers them to the per-cell rho / vel_k / pre reads).
# regime validity rests with the user: an isothermal regime carries no energy, so
# `pressure` there has no `pre` field to read and is rejected at lower time.
RHO_ALIASES = ["rho", "density"]
PRE_ALIASES = ["pre", "pressure", "p"]
VEL1_ALIASES = ["vel1", "vx", "v1"]
VEL2_ALIASES = ["vel2", "vy", "v2"]
VEL3_ALIASES = ["vel3", "vz", "v3"]

# the cell's lab-frame volume measure, the natural weight for an extensive quantity in a
# binned reduction. it is the measure the finite-volume update itself uses, so a mass sum
# `rho*dV` stays correct on a curvilinear grid, where the measure is
# r^2 sin(theta) dr dtheta dphi in place of dx^3. this leaf serves as a reduction weight: a
# source term is a per-unit-volume density, so a source referencing this leaf is rejected.
DV_ALIASES = ["dv", "cell_volume", "volume"]


def read_symbol(name: str) -> str:
    """the wire symbol a variable name denotes: `x_0`/`x_1`/`x_2` for the coordinate
    aliases, `t`, `rho`, `vel_0`/`vel_1`/`vel_2`, `pre`, or `dv`. raises for a name
    outside the leaf vocabulary."""
    if name in X1_ALIASES:
        return "x_0"
    if name in X2_ALIASES:
        return "x_1"
    if name in X3_ALIASES:
        return "x_2"
    if name == "t":
        return "t"
    if name in RHO_ALIASES:
        return "rho"
    if name in VEL1_ALIASES:
        return "vel_0"
    if name in VEL2_ALIASES:
        return "vel_1"
    if name in VEL3_ALIASES:
        return "vel_2"
    if name in PRE_ALIASES:
        return "pre"
    if name in DV_ALIASES:
        return "dv"
    raise ValueError(f"unknown variable '{name}'")


class ExprGraph:
    """Immutable directed acyclic graph of expressions.

    the graph is the source-building context: every leaf a constructor mints on it
    (`variable`, `density`, `velocity`, `pressure`, `parameter`, ...) is recorded as a
    granted read or parameter. `serialize_source` emits that granted set as the
    source's vocabulary declaration, independently of the node walk that emits the
    expression itself, so the backend can hold the observed expression to the symbols
    the context granted."""

    def __init__(self) -> None:
        self._nodes: dict[NodeId, NodeDef] = {}
        self._next_id: int = 0
        self._granted_reads: set[str] = set()
        self._granted_parameters: set[int] = set()

    def add_node(self, op_type: str, *inputs: NodeId, **attrs: Any) -> NodeId:
        """Add a node to the graph, returning its unique ID."""
        node_id = self._next_id
        self._next_id += 1
        self._nodes[node_id] = (op_type, inputs, attrs)
        return node_id

    def grant_read(self, name: str) -> None:
        """record that this context granted the variable leaf `name`."""
        self._granted_reads.add(name)

    def grant_parameter(self, idx: int) -> None:
        """record that this context granted the parameter leaf `param_idx = idx`."""
        self._granted_parameters.add(int(idx))

    def granted_vocabulary(self) -> dict[str, list[Any]]:
        """the granted leaves as the wire declaration `{reads, params}`: the reads by
        wire symbol, sorted, and the parameter indices, sorted."""
        return {
            "reads": sorted({read_symbol(name) for name in self._granted_reads}),
            "params": sorted(self._granted_parameters),
        }

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
            self._graph,
            self._graph.add_node("add", self._node_id, other_expr._node_id),
        )

    def __sub__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node(
                "subtract", self._node_id, other_expr._node_id
            ),
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
            self._graph.add_node(
                "subtract", other_expr._node_id, self._node_id
            ),
        )

    def __mul__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node(
                "multiply", self._node_id, other_expr._node_id
            ),
        )

    def __rmul__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node(
                "multiply", other_expr._node_id, self._node_id
            ),
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

    def __abs__(self) -> Expr:
        """Absolute value of the expression."""
        return Expr(self._graph, self._graph.add_node("abs", self._node_id))

    def __neg__(self) -> Expr:
        return Expr(self._graph, self._graph.add_node("negate", self._node_id))

    def __pos__(self) -> Expr:
        return self

    # comparison operators
    def __lt__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("lt", self._node_id, other_expr._node_id),
        )

    def __gt__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("gt", self._node_id, other_expr._node_id),
        )

    def __le__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("le", self._node_id, other_expr._node_id),
        )

    def __ge__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("ge", self._node_id, other_expr._node_id),
        )

    def __mod__(self, other: Union[Expr, float, int]) -> Expr:
        other_expr = self._ensure_expr(other)
        return Expr(
            self._graph,
            self._graph.add_node("mod", self._node_id, other_expr._node_id),
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
        self,
        patterns: dict[str, Callable[[Expr, tuple[NodeId, ...], NodeAttrs], T]],
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

    def diff(self, var: "Expr") -> "Expr":
        """symbolic derivative d(self)/d(var), built into the same graph (forward chain rule with
        memoization on shared subexpressions). `var` is a variable() leaf (e.g., variable('t'));
        matching is by name, so every leaf of that name differentiates to 1.

        comparisons / mod raise: they are non-differentiable, and their legitimate home is a branch
        condition, distinct from a smooth a(t). abs and select differentiate per-branch, which is
        correct away from the kink / branch boundary; the backend's finite-difference cross-check
        guards those cases, so a wrong derivative there fails loudly at setup."""
        g = self._graph
        vdef = g.get_node(var._node_id)
        if vdef is None or vdef[0] != "variable":
            raise ValueError("diff: `var` must be a variable() leaf")
        vname = vdef[2]["name"]
        cache: dict[NodeId, NodeId] = {}

        def kid(nid: NodeId) -> Expr:
            return Expr(g, nid)

        def d(nid: NodeId) -> Expr:
            if nid in cache:
                return Expr(g, cache[nid])
            op, ins, attrs = g.get_node(nid)  # type: ignore[misc]
            c = [kid(i) for i in ins]
            # the conditional differentiates per branch: the condition passes
            # through untouched — comparisons are non-differentiable, so the
            # recursion stops at one — and it is handled before the eager child
            # pass.
            if op == "if_then_else":
                res = Expr(g, g.add_node(op, ins[0], d(ins[1])._node_id, d(ins[2])._node_id))
                cache[nid] = res._node_id
                return res
            dc = [d(i) for i in ins]
            two = constant(2.0, g)
            one = constant(1.0, g)
            if op in ("constant", "parameter"):
                res = constant(0.0, g)
            elif op == "variable":
                res = constant(1.0 if attrs["name"] == vname else 0.0, g)
            elif op == "add":
                res = dc[0] + dc[1]
            elif op == "subtract":
                res = dc[0] - dc[1]
            elif op == "multiply":
                res = dc[0] * c[1] + c[0] * dc[1]
            elif op == "divide":
                res = (dc[0] * c[1] - c[0] * dc[1]) / (c[1] * c[1])
            elif op == "power":
                edef = g.get_node(ins[1])
                if edef is not None and edef[0] == "constant":  # constant exponent: n*a^(n-1)*da
                    n = float(edef[2]["value"])
                    res = constant(n, g) * (c[0] ** constant(n - 1.0, g)) * dc[0]
                else:  # general: a^b * (db*ln(a) + b*da/a)
                    res = (c[0] ** c[1]) * (dc[1] * log(c[0]) + c[1] * dc[0] / c[0])
            elif op == "negate":
                res = -dc[0]
            elif op == "sqrt":
                res = dc[0] / (two * sqrt(c[0]))
            elif op == "abs":
                res = (c[0] / kid(nid)) * dc[0]
            elif op == "exp":
                res = kid(nid) * dc[0]
            elif op == "log":
                res = dc[0] / c[0]
            elif op == "log10":
                res = dc[0] / (c[0] * constant(math.log(10.0), g))
            elif op == "sin":
                res = cos(c[0]) * dc[0]
            elif op == "cos":
                res = -sin(c[0]) * dc[0]
            elif op == "tan":
                res = dc[0] / (cos(c[0]) ** two)
            elif op == "asin":
                res = dc[0] / sqrt(one - c[0] ** two)
            elif op == "acos":
                res = -dc[0] / sqrt(one - c[0] ** two)
            elif op == "atan":
                res = dc[0] / (one + c[0] ** two)
            elif op == "atan2":  # atan2(y, x): (x*dy - y*dx)/(x^2 + y^2)
                res = (c[1] * dc[0] - c[0] * dc[1]) / (c[0] * c[0] + c[1] * c[1])
            elif op == "sinh":
                res = cosh(c[0]) * dc[0]
            elif op == "cosh":
                res = sinh(c[0]) * dc[0]
            elif op == "tanh":
                res = (one - kid(nid) ** two) * dc[0]
            else:
                raise ValueError(
                    f"diff: op '{op}' is not differentiable (comparisons/mod belong in a branch "
                    f"condition, not in a smooth scale factor a(t))"
                )
            cache[nid] = res._node_id
            return res

        return d(self._node_id)


def _graph_of(where: "ExprGraph | Expr") -> ExprGraph:
    """the graph to build a leaf in, named either directly or by any expression
    already living in it.

    a state leaf is almost always wanted alongside coordinates that exist, and
    `density(x1)` says "the density at the same place as x1" without the caller
    having to hold the graph in a variable to say it."""
    return where.graph if isinstance(where, Expr) else where


# factory functions
def constant(value: float, graph: "ExprGraph | Expr") -> Expr:
    """Create a constant expression."""
    g = _graph_of(graph)
    return Expr(g, g.add_node("constant", value=float(value)))


def variable(name: str, graph: "ExprGraph | Expr") -> Expr:
    """Create a variable expression. the graph records the leaf as a granted read."""
    g = _graph_of(graph)
    g.grant_read(name)
    return Expr(g, g.add_node("variable", name=name))


def parameter(idx: int, graph: "ExprGraph | Expr") -> Expr:
    """Create a parameter expression. the graph records the index as a granted parameter."""
    g = _graph_of(graph)
    g.grant_parameter(idx)
    return Expr(g, g.add_node("parameter", param_idx=idx))


def density(graph: "ExprGraph | Expr") -> Expr:
    """the per-cell density rho (a fluid-state leaf for state-dependent sources)."""
    return variable("rho", graph)


def velocity(axis: int, graph: "ExprGraph | Expr") -> Expr:
    """the per-cell velocity component `vel[axis]` (0-indexed: 0->vx, 1->vy, 2->vz)."""
    if axis not in (0, 1, 2):
        raise ValueError(f"velocity axis must be 0, 1, or 2, got {axis}")
    return variable(f"vel{axis + 1}", graph)


def pressure(graph: "ExprGraph | Expr") -> Expr:
    """the per-cell pressure (energy-bearing regimes only; rejected on isothermal)."""
    return variable("pre", graph)


def cell_volume(graph: "ExprGraph | Expr") -> Expr:
    """the cell's lab-frame volume measure dV, the weight for an extensive quantity.

    this is the measure the finite-volume update itself uses, so `density() * cell_volume()`
    is the cell mass on a curvilinear grid as well as a cartesian one. valid in a binned
    reduction only; a source term referencing it is rejected, because a source is a
    per-unit-volume density and weighting it by the measure would make the deposited
    amount depend on the resolution.
    """
    return variable("dv", graph)


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
        expr1._graph,
        expr1._graph.add_node("atan2", expr1._node_id, expr2._node_id),
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
        expr1._graph,
        expr1._graph.add_node("max", expr1._node_id, expr2._node_id),
    )


def min_expr(expr1: Expr, expr2: Expr) -> Expr:
    """Minimum of two expressions."""
    return Expr(
        expr1._graph,
        expr1._graph.add_node("min", expr1._node_id, expr2._node_id),
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
        expr1._graph.add_node(
            "bitwise_left_shift", expr1._node_id, expr2._node_id
        ),
    )


def bitwise_right_shift(expr1: Expr, expr2: Expr) -> Expr:
    return Expr(
        expr1._graph,
        expr1._graph.add_node(
            "bitwise_right_shift", expr1._node_id, expr2._node_id
        ),
    )


def sgn(expr: Expr) -> Expr:
    return Expr(expr._graph, expr._graph.add_node("sgn", expr._node_id))


def if_then_else(
    condition: Expr,
    true_case: Union[Expr, float, int],
    false_case: Union[Expr, float, int],
) -> Expr:
    """If-then-else expression.

    either branch may be a plain number, which is lifted into the condition's
    graph. a branch that is a bare constant is the common case (a damping rate
    that is a fixed value inside a region and zero outside it)."""
    true_expr = condition._ensure_expr(true_case)
    false_expr = condition._ensure_expr(false_case)
    return Expr(
        condition._graph,
        condition._graph.add_node(
            "if_then_else",
            condition._node_id,
            true_expr._node_id,
            false_expr._node_id,
        ),
    )


def logical_and(expr1: Expr, expr2: Expr) -> Expr:
    """Logical AND operation."""
    return Expr(
        expr1._graph,
        expr1._graph.add_node("logical_and", expr1._node_id, expr2._node_id),
    )


def logical_or(expr1: Expr, expr2: Expr) -> Expr:
    """Logical OR operation."""
    return Expr(
        expr1._graph,
        expr1._graph.add_node("logical_or", expr1._node_id, expr2._node_id),
    )


def logical_not(expr: Expr) -> Expr:
    """Logical NOT operation."""
    return Expr(expr._graph, expr._graph.add_node("logical_not", expr._node_id))


def logical_xor(expr1: Expr, expr2: Expr) -> Expr:
    """Logical XOR operation."""
    return Expr(
        expr1._graph,
        expr1._graph.add_node("logical_xor", expr1._node_id, expr2._node_id),
    )


def logical_nand(expr1: Expr, expr2: Expr) -> Expr:
    """Logical NAND operation."""
    return Expr(
        expr1._graph,
        expr1._graph.add_node("logical_nand", expr1._node_id, expr2._node_id),
    )


def logical_nor(expr1: Expr, expr2: Expr) -> Expr:
    """Logical NOR operation."""
    return Expr(
        expr1._graph,
        expr1._graph.add_node("logical_nor", expr1._node_id, expr2._node_id),
    )


def logical_xnor(expr1: Expr, expr2: Expr) -> Expr:
    """Logical XNOR operation."""
    return Expr(
        expr1._graph,
        expr1._graph.add_node("logical_xnor", expr1._node_id, expr2._node_id),
    )


def where(condition: Expr, true_case: Expr, false_case: Expr) -> Expr:
    """Where function for conditional expressions."""
    return if_then_else(condition, true_case, false_case)


def tabulated_1d(
    coordinate: Expr,
    coordinates: Sequence[float],
    values: Sequence[float],
    *,
    bounds: TableBounds | str,
) -> Expr:
    """piecewise-linear immutable field evaluated at `coordinate`.

    samples must be finite, equal-length, and strictly increasing. `bounds` is
    mandatory: `clamp` returns the nearest endpoint and `zero` returns zero.
    the table lowers to the ordinary constant/comparison/select graph, so host
    and device execution use the same backend program.
    """
    xs = tuple(float(value) for value in coordinates)
    ys = tuple(float(value) for value in values)
    if len(xs) != len(ys):
        raise ValueError(
            f"tabulated_1d coordinate/value lengths differ: {len(xs)} != {len(ys)}"
        )
    if len(xs) < 2:
        raise ValueError("tabulated_1d requires at least two samples")
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise ValueError("tabulated_1d samples must be finite")
    if any(right <= left for left, right in zip(xs, xs[1:])):
        raise ValueError("tabulated_1d coordinates must be strictly increasing")
    try:
        bounds_mode = bounds if isinstance(bounds, TableBounds) else TableBounds(bounds)
    except ValueError as exc:
        raise ValueError(
            "tabulated_1d bounds must be 'clamp' or 'zero'"
        ) from exc

    return _tabulated_1d_expr_values(
        coordinate,
        xs,
        tuple(constant(value, coordinate.graph) for value in ys),
        bounds_mode,
    )


def _tabulated_1d_expr_values(
    coordinate: Expr,
    coordinates: tuple[float, ...],
    values: tuple[Expr, ...],
    bounds: TableBounds,
) -> Expr:
    """linear interpolation over expression-valued samples."""
    graph = coordinate.graph
    segments: list[Expr] = []
    for left_x, right_x, left_y, right_y in zip(
        coordinates, coordinates[1:], values, values[1:]
    ):
        x0 = constant(left_x, graph)
        fraction = (coordinate - x0) / (right_x - left_x)
        segments.append(left_y + fraction * (right_y - left_y))

    interior = segments[-1]
    for upper_x, segment in reversed(
        list(zip(coordinates[1:-1], segments[:-1]))
    ):
        interior = where(coordinate < constant(upper_x, graph), segment, interior)

    outside_low = (
        values[0] if bounds is TableBounds.CLAMP else constant(0.0, graph)
    )
    outside_high = (
        values[-1] if bounds is TableBounds.CLAMP else constant(0.0, graph)
    )
    return where(
        coordinate < constant(coordinates[0], graph),
        outside_low,
        where(
            coordinate > constant(coordinates[-1], graph),
            outside_high,
            interior,
        ),
    )


def tabulated_2d(
    coordinate_x: Expr,
    coordinate_y: Expr,
    coordinates_x: Sequence[float],
    coordinates_y: Sequence[float],
    values: Sequence[Sequence[float]],
    *,
    bounds: TableBounds | str,
) -> Expr:
    """bilinear immutable field on a rectilinear two-dimensional table."""
    if coordinate_x.graph is not coordinate_y.graph:
        raise ValueError("tabulated_2d coordinates must belong to the same graph")
    xs = tuple(float(value) for value in coordinates_x)
    ys = tuple(float(value) for value in coordinates_y)
    rows = tuple(tuple(float(value) for value in row) for row in values)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError("tabulated_2d requires at least two samples per axis")
    if len(rows) != len(ys) or any(len(row) != len(xs) for row in rows):
        raise ValueError(
            "tabulated_2d values must have shape "
            "(len(coordinates_y), len(coordinates_x))"
        )
    flat_values = tuple(value for row in rows for value in row)
    if not all(math.isfinite(value) for value in (*xs, *ys, *flat_values)):
        raise ValueError("tabulated_2d samples must be finite")
    if any(right <= left for left, right in zip(xs, xs[1:])) or any(
        right <= left for left, right in zip(ys, ys[1:])
    ):
        raise ValueError("tabulated_2d coordinates must be strictly increasing")
    try:
        bounds_mode = bounds if isinstance(bounds, TableBounds) else TableBounds(bounds)
    except ValueError as exc:
        raise ValueError(
            "tabulated_2d bounds must be 'clamp' or 'zero'"
        ) from exc

    graph = coordinate_x.graph
    row_fields = tuple(
        _tabulated_1d_expr_values(
            coordinate_x,
            xs,
            tuple(constant(value, graph) for value in row),
            bounds_mode,
        )
        for row in rows
    )
    return _tabulated_1d_expr_values(
        coordinate_y,
        ys,
        row_fields,
        bounds_mode,
    )


def tabulated_3d(
    coordinate_x: Expr,
    coordinate_y: Expr,
    coordinate_z: Expr,
    coordinates_x: Sequence[float],
    coordinates_y: Sequence[float],
    coordinates_z: Sequence[float],
    values: Sequence[Sequence[Sequence[float]]],
    *,
    bounds: TableBounds | str,
) -> Expr:
    """trilinear immutable field on a rectilinear three-dimensional table."""
    if not (
        coordinate_x.graph is coordinate_y.graph
        and coordinate_x.graph is coordinate_z.graph
    ):
        raise ValueError("tabulated_3d coordinates must belong to the same graph")
    xs = tuple(float(value) for value in coordinates_x)
    ys = tuple(float(value) for value in coordinates_y)
    zs = tuple(float(value) for value in coordinates_z)
    planes = tuple(
        tuple(tuple(float(value) for value in row) for row in plane)
        for plane in values
    )
    if len(xs) < 2 or len(ys) < 2 or len(zs) < 2:
        raise ValueError("tabulated_3d requires at least two samples per axis")
    if (
        len(planes) != len(zs)
        or any(len(plane) != len(ys) for plane in planes)
        or any(len(row) != len(xs) for plane in planes for row in plane)
    ):
        raise ValueError(
            "tabulated_3d values must have shape "
            "(len(coordinates_z), len(coordinates_y), len(coordinates_x))"
        )
    flat_values = tuple(
        value for plane in planes for row in plane for value in row
    )
    if not all(
        math.isfinite(value) for value in (*xs, *ys, *zs, *flat_values)
    ):
        raise ValueError("tabulated_3d samples must be finite")
    if (
        any(right <= left for left, right in zip(xs, xs[1:]))
        or any(right <= left for left, right in zip(ys, ys[1:]))
        or any(right <= left for left, right in zip(zs, zs[1:]))
    ):
        raise ValueError("tabulated_3d coordinates must be strictly increasing")
    try:
        bounds_mode = bounds if isinstance(bounds, TableBounds) else TableBounds(bounds)
    except ValueError as exc:
        raise ValueError(
            "tabulated_3d bounds must be 'clamp' or 'zero'"
        ) from exc

    graph = coordinate_x.graph
    plane_fields = []
    for plane in planes:
        row_fields = tuple(
            _tabulated_1d_expr_values(
                coordinate_x,
                xs,
                tuple(constant(value, graph) for value in row),
                bounds_mode,
            )
            for row in plane
        )
        plane_fields.append(
            _tabulated_1d_expr_values(
                coordinate_y,
                ys,
                row_fields,
                bounds_mode,
            )
        )
    return _tabulated_1d_expr_values(
        coordinate_z,
        zs,
        tuple(plane_fields),
        bounds_mode,
    )


def floor(expr: Expr) -> Expr:
    """Floor function."""
    return Expr(expr._graph, expr._graph.add_node("floor", expr._node_id))


def ceil(expr: Expr) -> Expr:
    """Ceiling function."""
    return Expr(expr._graph, expr._graph.add_node("ceil", expr._node_id))


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
                raise ValueError(
                    "Cyclic dependency detected in expression graph"
                )

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
                    values[node_id] = 0.0  # division by zero yields 0.0
                else:
                    values[node_id] = values[input_ids[0]] / denominator
            elif op == "le":
                values[node_id] = float(
                    values[input_ids[0]] <= values[input_ids[1]]
                )
            elif op == "lt":
                values[node_id] = float(
                    values[input_ids[0]] < values[input_ids[1]]
                )
            elif op == "ge":
                values[node_id] = float(
                    values[input_ids[0]] >= values[input_ids[1]]
                )
            elif op == "gt":
                values[node_id] = float(
                    values[input_ids[0]] > values[input_ids[1]]
                )
            elif op == "abs":
                values[node_id] = abs(values[input_ids[0]])
            # math functions
            elif op == "power":
                values[node_id] = values[input_ids[0]] ** values[input_ids[1]]
            elif op == "negate":
                values[node_id] = -values[input_ids[0]]
            elif op == "sqrt":
                val = values[input_ids[0]]
                if val < 0.0:
                    values[node_id] = 0.0  # negative argument yields 0.0
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
                values[node_id] = math.atan2(
                    values[input_ids[0]], values[input_ids[1]]
                )
            elif op == "floor":
                values[node_id] = math.floor(values[input_ids[0]])
            elif op == "ceil":
                values[node_id] = math.ceil(values[input_ids[0]])
            # binary ops
            elif op == "max":
                values[node_id] = max(
                    values[input_ids[0]], values[input_ids[1]]
                )
            elif op == "min":
                values[node_id] = min(
                    values[input_ids[0]], values[input_ids[1]]
                )
            # bitwise ops
            elif op == "bitwise_and":
                values[node_id] = int(values[input_ids[0]]) & int(
                    values[input_ids[1]]
                )
            elif op == "bitwise_or":
                values[node_id] = int(values[input_ids[0]]) | int(
                    values[input_ids[1]]
                )
            elif op == "bitwise_xor":
                values[node_id] = int(values[input_ids[0]]) ^ int(
                    values[input_ids[1]]
                )
            elif op == "bitwise_not":
                values[node_id] = ~int(values[input_ids[0]])
            elif op == "bitwise_left_shift":
                values[node_id] = int(values[input_ids[0]]) << int(
                    values[input_ids[1]]
                )
            elif op == "bitwise_right_shift":
                values[node_id] = int(values[input_ids[0]]) >> int(
                    values[input_ids[1]]
                )
            elif op == "mod":
                values[node_id] = values[input_ids[0]] % values[input_ids[1]]
            elif op == "if_then_else":
                condition = values[input_ids[0]]
                if condition:
                    values[node_id] = values[input_ids[1]]
                else:
                    values[node_id] = values[input_ids[2]]

        # return output values
        return [values[out_id] for out_id in self._output_ids]

    def _serialize_nodes(self) -> dict[str, object]:
        """serialize graph nodes for the typed wire-format methods."""
        expressions: list[dict[str, Any]] = []

        # map from internal node ids to serialized indices
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
                name = attrs["name"]
                if name in X1_ALIASES:
                    expressions.append({"op": "VARIABLE_X1"})
                elif name in X2_ALIASES:
                    expressions.append({"op": "VARIABLE_X2"})
                elif name in X3_ALIASES:
                    expressions.append({"op": "VARIABLE_X3"})
                elif name == "t":
                    expressions.append({"op": "VARIABLE_T"})
                elif name in RHO_ALIASES:
                    expressions.append({"op": "VARIABLE_RHO"})
                elif name in VEL1_ALIASES:
                    expressions.append({"op": "VARIABLE_VEL1"})
                elif name in VEL2_ALIASES:
                    expressions.append({"op": "VARIABLE_VEL2"})
                elif name in VEL3_ALIASES:
                    expressions.append({"op": "VARIABLE_VEL3"})
                elif name in PRE_ALIASES:
                    expressions.append({"op": "VARIABLE_PRESSURE"})
                elif name in DV_ALIASES:
                    expressions.append({"op": "VARIABLE_DV"})
                else:
                    raise ValueError(f"unknown variable '{name}'")
            elif op == "parameter":
                expressions.append(
                    {"op": "PARAMETER", "param_idx": attrs["param_idx"]}
                )
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
            elif op == "abs":
                expressions.append(
                    {"op": "ABS", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "lt":
                expressions.append(
                    {
                        "op": "LT",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "le":
                expressions.append(
                    {
                        "op": "LE",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "gt":
                expressions.append(
                    {
                        "op": "GT",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "ge":
                expressions.append(
                    {
                        "op": "GE",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
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
            elif op == "floor":
                expressions.append(
                    {"op": "FLOOR", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "ceil":
                expressions.append(
                    {"op": "CEIL", "left": node_map[input_ids[0]], "right": -1}
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
                    {
                        "op": "BITWISE_NOT",
                        "left": node_map[input_ids[0]],
                        "right": -1,
                    }
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
            elif op == "sgn":
                expressions.append(
                    {"op": "SGN", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "abs":
                expressions.append(
                    {"op": "ABS", "left": node_map[input_ids[0]], "right": -1}
                )
            elif op == "logical_and":
                expressions.append(
                    {
                        "op": "LOGICAL_AND",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "logical_or":
                expressions.append(
                    {
                        "op": "LOGICAL_OR",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "logical_not":
                expressions.append(
                    {
                        "op": "LOGICAL_NOT",
                        "left": node_map[input_ids[0]],
                        "right": -1,
                    }
                )
            elif op == "logical_xor":
                expressions.append(
                    {
                        "op": "LOGICAL_XOR",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "logical_nand":
                expressions.append(
                    {
                        "op": "LOGICAL_NAND",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "logical_nor":
                expressions.append(
                    {
                        "op": "LOGICAL_NOR",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "logical_xnor":
                expressions.append(
                    {
                        "op": "LOGICAL_XNOR",
                        "left": node_map[input_ids[0]],
                        "right": node_map[input_ids[1]],
                    }
                )
            elif op == "where":
                expressions.append(
                    {
                        "op": "WHERE",
                        "condition": node_map[input_ids[0]],
                        "true_case": node_map[input_ids[1]],
                        "false_case": node_map[input_ids[2]],
                    }
                )
            else:
                raise ValueError(f"Unknown operation: {op}")

        # map output indices
        output_indices = [node_map[out_id] for out_id in self._output_ids]

        # unary ops emit a `right: -1` "no operand" sentinel; the rust NodeDesc index
        # fields are Option<usize> and reject -1. drop any -1 index key so the absent
        # field deserializes to None (the correct "no operand"). the `value` key holds
        # a float, outside this set of index keys, so a legitimate -1.0 constant
        # passes through untouched.
        for node in expressions:
            for key in ("left", "right", "condition", "true_case", "false_case"):
                if node.get(key) == -1:
                    del node[key]

        max_param_idx = -1
        for node_id in self._eval_order:
            node_def = self._graph.get_node(node_id)
            if node_def and node_def[0] == "parameter":
                max_param_idx = max(
                    max_param_idx, node_def[2].get("param_idx", -1)
                )

        return {
            "expressions": expressions,
            "output_indices": output_indices,
            "param_count": max_param_idx + 1,
        }

    def serialize_source(
        self,
        kind: "SourceKind | str",
        dim: int,
        *,
        params: "list[float] | tuple[float, ...] | None" = None,
        region: int | None = None,
        target: "ConservedField | str | None" = None,
    ) -> dict[str, object]:
        """serialize to the rust `SourceConfig` wire format consumed by
        symbi-expr (`load.rs::SourceConfig::from_json`) and lowered by
        symbi-hydro's `build_user_source`:
          kind   -- 'force' | 'rotating_frame' | 'cooling' | 'relax' | 'sponge' |
                    'inject' | 'raw' (law wrap)
          dim    -- spatial dimensionality (force needs `dim` accel outputs,
                    cooling 1, relax 1+dim; sponge [kappa, rho_ref, dim*vel_ref,
                    pre_ref] = 3+dim on energy regimes, 2+dim on iso; inject
                    [den, dim*mom, nrg] = 2+dim on energy regimes, 1+dim on iso)
          params -- runtime scalar VALUES for the parameter() nodes (p0, p1, ...);
                    sponge takes no params: the closure comes from the regime
          region -- optional node index of a chi(x) mask folded into the source
          target -- for kind='raw' only: the conserved slot ('den'|'mom'|'nrg')

        the `vocabulary` field is the graph's granted leaves (`ExprGraph.granted_vocabulary`),
        captured as the leaf constructors ran; the backend holds the serialized nodes to it.
        """
        base = self._serialize_nodes()
        # normalize the enums to their canonical rust strings at the boundary.
        kind_str = kind.value if isinstance(kind, SourceKind) else str(kind)
        cfg: dict[str, object] = {
            "kind": kind_str,
            "dim": int(dim),
            "outputs": base["output_indices"],
            "params": [float(p) for p in (params or [])],
            "vocabulary": self._graph.granted_vocabulary(),
            "nodes": base["expressions"],
        }
        if region is not None:
            cfg["region"] = int(region)
        if target is not None:
            cfg["target"] = (
                target.value if isinstance(target, ConservedField) else str(target)
            )
        return cfg

    def serialize_equilibrium(
        self,
        dim: int,
        *,
        params: "Sequence[float] | None" = None,
    ) -> dict[str, object]:
        """serialize to the rust `EquilibriumConfig` wire format consumed by symbi-expr
        (`load.rs::EquilibriumConfig::from_json`) and handed to
        `Hierarchy::with_equilibrium_expression`.

        this is a state declaration: the compiled outputs are the primitive components
        themselves, in the order `[rho, v1, ..., vN, p]` — one velocity component per
        momentum degree of freedom, and the pressure omitted on an isothermal regime. the
        conservation-law wrap and conserved slot a source carries stay out of this wire
        form.

        the state declared here is the one a well-balanced scheme holds exactly: its
        discrete imbalance is measured once per level and subtracted back at every stage.
        it is therefore a genuine steady state of the equations being solved — the backend
        requires its imbalance to converge under refinement, since well-balancing a
        non-equilibrium would freeze the run in place while reporting success.

          dim    -- spatial dimensionality of the grid
          params -- runtime scalar values for the parameter() nodes (p0, p1, ...)
        """
        base = self._serialize_nodes()
        return {
            "dim": int(dim),
            "outputs": base["output_indices"],
            "params": [float(p) for p in (params or [])],
            "nodes": base["expressions"],
        }

    def serialize_census(
        self,
        name: str,
        *,
        axes: "Sequence[tuple[str, Sequence[float]]]",
        value_names: "Sequence[str]",
        op: "ReductionOp | str" = "add",
        params: "Sequence[float] | None" = None,
        sample_interval: "float | None" = None,
        accumulate: bool = False,
        cadence: object = "root_step",
    ) -> dict[str, object]:
        """serialize to the rust `CensusConfig` wire format consumed by symbi-expr and
        lowered by symbi-hydro's `build_census_expressions`.

        the compiled outputs must be the bin-axis coordinates FIRST, in `axes` order,
        then the accumulator values in `value_names` order — the order
        `Census.serialize` assembles them in. compiling them together is the point:
        a subexpression an axis and a value share (a radius, its logarithm) is written
        once and evaluated once per cell.
        """
        base = self._serialize_nodes()
        out = base["output_indices"]
        n_axes = len(axes)
        if len(out) != n_axes + len(value_names):
            raise ValueError(
                f"census '{name}': compiled {len(out)} outputs for {n_axes} axes and "
                f"{len(value_names)} values"
            )
        op_str = op.value if isinstance(op, ReductionOp) else str(op)
        return {
            "name": name,
            "axes": [
                {"name": axis_name, "expr": out[ii], "edges": [float(e) for e in edges]}
                for ii, (axis_name, edges) in enumerate(axes)
            ],
            "values": out[n_axes:],
            "value_names": [str(v) for v in value_names],
            "op": op_str,
            "params": [float(p) for p in (params or [])],
            "sample_interval": None if sample_interval is None else float(sample_interval),
            "accumulate": bool(accumulate),
            "cadence": getattr(cadence, "value", cadence),
            "nodes": base["expressions"],
        }

    def serialize_boundary(self, dim: int) -> dict[str, object]:
        """serialize a DRIVEN (Dirichlet) boundary prescription to the rust
        `SourceConfig` wire format (`kind="dirichlet"`). the compiled outputs must
        be the COMPLETE primitive state in order:
          hydro: [rho, vel_0..vel_{dim-1}, (pre)]
          mhd:   [rho, vel_0..vel_{dim-1}, (pre), B_0..B_{dim-1}]
        `dim` is the VECTOR component count (= the regime DOF: 2.5D MHD -> 3). the
        rust side splits these into the den/mom/nrg/bcell ghost-fill slots. for a
        purely toroidal injection set the in-plane B to 0 and the out-of-plane
        component to B_phi (cell-centered, div-free by axisymmetry)."""
        base = self._serialize_nodes()
        return {
            "kind": "dirichlet",
            "dim": int(dim),
            "outputs": base["output_indices"],
            "params": [],
            "nodes": base["expressions"],
        }

    def serialize_motion(self) -> dict[str, object]:
        """serialize a scale-factor motion law to the rust mesh-motion wire. the compiled outputs
        are exactly `[a(t), a_dot(t)]` in that order (a_dot is typically `a.diff(variable('t'))`).
        the two scalar-in-`t` expressions are lowered to a `BuiltSource` and evaluated every step,
        free of the conservation-law wrap `serialize_source` applies; the backend
        finite-difference-checks `a_dot` against `a` at setup before running, so an inconsistent
        derivative fails loudly."""
        base = self._serialize_nodes()
        out = base["output_indices"]
        if len(out) != 2:
            raise ValueError(f"motion expects exactly 2 outputs [a, a_dot], got {len(out)}")
        return {
            "kind": "motion",
            "dim": 1,  # unused by the motion path (a(t) is a scalar), required by the SourceConfig wire
            "outputs": out,
            "params": [],
            "nodes": base["expressions"],
        }
