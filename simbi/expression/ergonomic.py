# =============================================================================
# ergonomic.py
#
# the config-facing way to write a source term, a driven boundary, or a mesh
# motion law. it keeps the expression graph out of the caller's vocabulary:
# - coords(n) hands back the spatial variables already sharing one graph
# - each source kind has a constructor taking the outputs and the dimension
# - bare python floats are lifted into the graph wherever an output wants one
# - every output is checked to belong to one graph, and the count is checked
#   against what the named kind requires
#
# usage:
#   x, y, z = coords(3)
#   r = expr.sqrt(x * x + y * y + z * z)
#   src = sponge([kappa, rho_ref, vx, vy, vz, pre_ref], dim=3)
#   bc  = boundary([rho, vx, vy, vz, pre], dim=3)
#   g   = force([0.0, -9.8], dim=2)
# =============================================================================
from typing import Iterable, Optional, Sequence, Union

from .dag_expression import ConservedField, Expr, ExprGraph, SourceKind, constant, variable

Outputish = Union[Expr, float, int]

# the axis names the rust wire expects, in order.
_AXES = ("x1", "x2", "x3")


def coords(ndim: int = 3, *, graph: Optional[ExprGraph] = None) -> tuple[Expr, ...]:
    """the spatial coordinate variables `x1..x{ndim}`, sharing one graph.

    the shared graph is the point. building them one at a time through
    `variable(name)` with the graph omitted gives each its own graph, and
    expressions from different graphs cannot be compiled together."""
    if not 1 <= ndim <= 3:
        raise ValueError(f"coords: ndim must be 1, 2 or 3, got {ndim}")
    g = graph or ExprGraph()
    return tuple(variable(name, g) for name in _AXES[:ndim])


def _resolve(outputs: Sequence[Outputish]) -> tuple[ExprGraph, list[Expr]]:
    """the graph these outputs live in, and every output lifted into it.

    a caller may mix expressions with plain numbers, since a constant reference
    state is an ordinary thing to want. the numbers join the graph the
    expressions already share; if every output is a number, a graph is made for
    them."""
    if not outputs:
        raise ValueError("expression outputs are empty; nothing to serialize")
    graphs = [o._graph for o in outputs if isinstance(o, Expr)]
    for first, other in zip(graphs, graphs[1:]):
        if first is not other:
            raise ValueError(
                "expression outputs come from two different graphs. build the "
                "coordinates once with coords(ndim) and derive every output from "
                "them, so they share one graph"
            )
    g = graphs[0] if graphs else ExprGraph()
    return g, [o if isinstance(o, Expr) else constant(float(o), g) for o in outputs]


def _check_arity(kind: str, count: int, admissible: Iterable[int], slots: str) -> None:
    admissible = sorted(set(admissible))
    if count not in admissible:
        wanted = " or ".join(str(a) for a in admissible)
        raise ValueError(
            f"{kind} takes {wanted} outputs [{slots}], got {count}. a short list "
            "would otherwise be read from whichever output sat at that index"
        )


def source(
    kind: Union[SourceKind, str],
    outputs: Sequence[Outputish],
    *,
    dim: int,
    params: Optional[Sequence[float]] = None,
    region: Optional[Expr] = None,
    target: Union[ConservedField, str, None] = None,
) -> dict:
    """compile `outputs` and serialize them as the named source kind.

    the shared core the per-kind constructors below delegate to. `region` is an
    optional chi(x) mask restricting where the source acts, given as an
    expression rather than a node index."""
    if region is not None and not isinstance(region, Expr):
        raise TypeError("region must be an expression in the same graph as the outputs")
    combined = list(outputs) + ([region] if region is not None else [])
    g, lifted = _resolve(combined)
    region_expr = lifted.pop() if region is not None else None
    compiled = g.compile(lifted + ([region_expr] if region_expr is not None else []))
    region_index = len(lifted) if region_expr is not None else None
    return compiled.serialize_source(
        kind,
        dim,
        params=list(params) if params is not None else None,
        region=region_index,
        target=target,
    )


def force(outputs: Sequence[Outputish], *, dim: int, **kw) -> dict:
    """a body acceleration `a(x)`, one output per spatial axis."""
    _check_arity("force", len(outputs), [dim], "a_0..a_{dim-1}")
    return source(SourceKind.FORCE, outputs, dim=dim, **kw)


def cooling(outputs: Sequence[Outputish], *, dim: int, **kw) -> dict:
    """an energy sink, one output."""
    _check_arity("cooling", len(outputs), [1], "rate")
    return source(SourceKind.COOLING, outputs, dim=dim, **kw)


def velocity_relaxation(outputs: Sequence[Outputish], *, dim: int, **kw) -> dict:
    """velocity relaxation toward a target, `[rate, v_0..v_{dim-1}]`: a
    density-preserving momentum drag whose energy term is the kinetic work of the
    drag alone."""
    _check_arity("velocity_relaxation", len(outputs), [1 + dim], "rate, v_0..v_{dim-1}")
    return source(SourceKind.VELOCITY_RELAXATION, outputs, dim=dim, **kw)


def relax(outputs: Sequence[Outputish], *, dim: int, **kw) -> dict:
    """deprecated alias of `velocity_relaxation`."""
    import warnings

    warnings.warn(
        "`relax(...)` is deprecated; use `velocity_relaxation(...)`.",
        DeprecationWarning,
        stacklevel=2,
    )
    return velocity_relaxation(outputs, dim=dim, **kw)


def rotating_frame(outputs: Sequence[Outputish], *, dim: int, **kw) -> dict:
    """the frame's rotation, `[omega, origin_x, origin_y]`."""
    _check_arity("rotating_frame", len(outputs), [3], "omega, origin_x, origin_y")
    return source(SourceKind.ROTATING_FRAME, outputs, dim=dim, **kw)


def sponge(outputs: Sequence[Outputish], *, dim: int, **kw) -> dict:
    """relaxation toward a reference state given in PRIMITIVES,
    `[kappa, rho_ref, vel_ref_0..vel_ref_{dim-1}, pre_ref]`.

    the regime converts that state through its own conservation law, so one
    wire serves a newtonian gas, a relativistic one and a curved background.
    an isothermal regime evolves no energy and stops before `pre_ref`."""
    _check_arity(
        "sponge", len(outputs), [2 + dim, 3 + dim],
        "kappa, rho_ref, vel_ref_0..vel_ref_{dim-1}, (pre_ref)",
    )
    return source(SourceKind.SPONGE, outputs, dim=dim, **kw)


def inject(outputs: Sequence[Outputish], *, dim: int, **kw) -> dict:
    """a conserved-vector injection, `[den, mom_0..mom_{dim-1}, (nrg)]`."""
    _check_arity(
        "inject", len(outputs), [1 + dim, 2 + dim],
        "den, mom_0..mom_{dim-1}, (nrg)",
    )
    return source(SourceKind.INJECT, outputs, dim=dim, **kw)


def raw(
    outputs: Sequence[Outputish],
    *,
    dim: int,
    target: Union[ConservedField, str],
    **kw,
) -> dict:
    """a conserved-slot source supplied directly, for one named slot."""
    return source(SourceKind.RAW, outputs, dim=dim, target=target, **kw)


def equilibrium(
    outputs: Sequence[Outputish], *, dim: int, params: Optional[Sequence[float]] = None
) -> dict:
    """the run's stationary target, as the complete primitive state.

    the backend measures this target's discrete imbalance once per level and adds
    it back each stage, so a stratified atmosphere holds exactly rather than
    drifting at the scheme's truncation order."""
    g, lifted = _resolve(outputs)
    return g.compile(lifted).serialize_equilibrium(
        dim, params=list(params) if params is not None else None
    )


def boundary(outputs: Sequence[Outputish], *, dim: int) -> dict:
    """a driven (Dirichlet) face, prescribing the COMPLETE primitive state:
    `[rho, vel_0..vel_{dim-1}, (pre), (B_0..B_{dim-1})]`.

    `dim` is the vector component count, which is the regime's DOF, so a 2.5d
    MHD run passes 3."""
    g, lifted = _resolve(outputs)
    return g.compile(lifted).serialize_boundary(dim)
