# =============================================================================
# expert.py
#
# expert / compiler-facing access to the expression graph. ordinary scientific
# configs never import this: they state physics as algebra over named quantities
# and let the framework mint and compile the graph. tooling that must inspect,
# build against, or compile the DAG directly reaches for it here, by name, so the
# escape is deliberate rather than accidental.
# =============================================================================

from .dag_expression import CompiledExpr, Expr, ExprGraph, variable


def graph_of(expression: Expr) -> ExprGraph:
    """the graph an expression lives in — for tooling that compiles or inspects
    the DAG. a scientific config never needs this."""
    return expression._graph


def compile_outputs(outputs: list[Expr]) -> CompiledExpr:
    """compile a set of expression outputs for direct evaluation."""
    return outputs[0]._graph.compile(outputs)


def coordinate(name: str, graph: ExprGraph) -> Expr:
    """a named position variable in an explicit graph — the tooling form of the
    coordinates a config gets from `coords()`."""
    return variable(name, graph)


__all__ = ["ExprGraph", "CompiledExpr", "graph_of", "compile_outputs", "coordinate"]
