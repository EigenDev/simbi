# =============================================================================
# census.py
#
# the user-facing registration for a binned reduction over the grid: a pointwise
# map followed by a segmented reduce, emitted as a time series alongside the body
# diagnostics.
#
# a census declares
# - a list of bin axes, each an expression plus a set of explicit edges;
# - a set of labelled accumulators, each an expression;
# - a reduce op.
# axes take an OUTER PRODUCT, so one radial axis gives shell profiles, a radial and
# an angular-momentum axis give the histogram per shell, and NO axes give a global
# reduction over the whole grid.
#
# gating needs no separate concept: "only inflowing cells" is an ordinary value
# expression, `m * v_r * (v_r < 0)`, built from the comparison operators.
#
# usage:
#  g = ExprGraph()
#  r = sqrt(variable("x", g) ** 2 + variable("y", g) ** 2)
#  m = density(g) * cell_volume(g)
#  census = Census(
#      name="shells",
#      axes=[BinAxis("r", r, log_edges)],
#      values={"volume": cell_volume(g), "mass": m, "radial_momentum": m * v_r},
#  )
#  payload = census.serialize()
# =============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .dag_expression import Expr, ReductionOp


@dataclass(frozen=True)
class BinAxis:
    """one bin axis: the coordinate to bin on, and the edges that cut it.

    edges are given explicitly rather than as a spacing rule, so log spacing, linear
    spacing and hand-chosen edges all work without a spacing enum. `n` edges give
    `n - 1` bins; bin `k` covers `[edges[k], edges[k+1])`, and the last bin is closed
    at its upper edge so a value sitting exactly on the domain boundary is counted.

    a cell outside these edges is dropped from the census and counted as such, since
    a binning that silently under-covers its domain is indistinguishable from a
    physics result.
    """

    name: str
    expr: Expr
    edges: Sequence[float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a bin axis needs a name; it labels the edges in the output")
        edges = [float(e) for e in self.edges]
        if len(edges) < 2:
            raise ValueError(
                f"bin axis '{self.name}': {len(edges)} edge(s) define no bin; "
                "at least 2 are needed"
            )
        for kk, e in enumerate(edges):
            if not math.isfinite(e):
                raise ValueError(f"bin axis '{self.name}': edge {kk} is not finite ({e})")
        for kk, (lo, hi) in enumerate(zip(edges, edges[1:])):
            if not hi > lo:
                raise ValueError(
                    f"bin axis '{self.name}': edges must strictly increase, but edge "
                    f"{kk} = {lo} is not below edge {kk + 1} = {hi}"
                )


def log_edges(lo: float, hi: float, n_bins: int) -> list[float]:
    """`n_bins + 1` logarithmically spaced edges spanning [lo, hi].

    the natural cut for shells around an accretor, where the flow's correlation time
    scales with radius, so equal ratios rather than equal widths sample each radius
    equally well in units of its own decorrelation.
    """
    if not (lo > 0.0 and hi > lo):
        raise ValueError(f"log edges need 0 < lo < hi, got lo={lo}, hi={hi}")
    if n_bins < 1:
        raise ValueError(f"log edges need at least one bin, got {n_bins}")
    step = (math.log(hi) - math.log(lo)) / n_bins
    return [math.exp(math.log(lo) + kk * step) for kk in range(n_bins + 1)]


def linear_edges(lo: float, hi: float, n_bins: int) -> list[float]:
    """`n_bins + 1` uniformly spaced edges spanning [lo, hi]."""
    if not hi > lo:
        raise ValueError(f"linear edges need lo < hi, got lo={lo}, hi={hi}")
    if n_bins < 1:
        raise ValueError(f"linear edges need at least one bin, got {n_bins}")
    step = (hi - lo) / n_bins
    return [lo + kk * step for kk in range(n_bins + 1)]


@dataclass(frozen=True)
class Census:
    """a registered binned reduction.

    `values` maps each accumulator's label to its expression; the labels travel with
    the output so a reader can name a column without re-deriving the registration
    order. `params` supplies the runtime values for any `parameter()` nodes.

    every expression — axes and values alike — must come from ONE `ExprGraph`. that
    is what lets a shared subexpression be written once and evaluated once per cell,
    and it is why the cost of a census scales with the size of its graph rather than
    with the number of accumulators registered.
    """

    name: str
    values: Mapping[str, Expr]
    axes: Sequence[BinAxis] = ()
    op: ReductionOp = ReductionOp.ADD
    params: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a census needs a name; it names its output group")
        if not self.values:
            raise ValueError(f"census '{self.name}': registers no values")
        seen: set[str] = set()
        for axis in self.axes:
            if axis.name in seen:
                raise ValueError(
                    f"census '{self.name}': two bin axes are both named '{axis.name}'"
                )
            seen.add(axis.name)

    def _graph(self) -> object:
        """the single graph every expression must belong to.

        expressions from different graphs carry unrelated node numbering, so mixing
        them would index into the wrong dag rather than fail — this is checked here
        so the error names the census instead of surfacing as wrong physics.
        """
        exprs = [axis.expr for axis in self.axes] + list(self.values.values())
        graph = exprs[0]._graph
        for expr in exprs[1:]:
            if expr._graph is not graph:
                raise ValueError(
                    f"census '{self.name}': every bin axis and value must be built from "
                    "one ExprGraph, so they share subexpressions and one node numbering"
                )
        return graph

    def serialize(self) -> dict[str, object]:
        """emit the rust `CensusConfig` wire format.

        the axes are compiled ahead of the values, which is the output order the rust
        side unpacks: bin coordinates first, then accumulators.
        """
        graph = self._graph()
        outputs = [axis.expr for axis in self.axes] + list(self.values.values())
        compiled = graph.compile(outputs)  # type: ignore[attr-defined]
        return compiled.serialize_census(
            self.name,
            axes=[(axis.name, list(axis.edges)) for axis in self.axes],
            value_names=list(self.values.keys()),
            op=self.op,
            params=self.params,
        )
