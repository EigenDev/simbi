# =============================================================================
# equilibrium.py
#
# stationary target states built from a declared gravitational potential.
#
# a well-balanced run needs two things that must agree exactly: the equilibrium state it
# holds, and the source term that state is in balance against. writing them separately is
# how they drift apart — the profile balances one gravity and the run applies another, and
# the result is a smooth, plausible, wrong atmosphere. declaring the POTENTIAL once and
# deriving both from it makes disagreement unrepresentable.
#
# for a barotropic equation of state the balance integrates in closed form. with
# `p = K rho^gamma` the specific enthalpy is `h = gamma K/(gamma-1) rho^(gamma-1)`, and
# hydrostatic balance `grad p = -rho grad phi` reduces to the bernoulli invariant
#   h + phi = const,
# so no integration is needed at all:
#   rho = [(gamma-1)/(gamma K) (C - phi)]^(1/(gamma-1)),  p = K rho^gamma.
# the constant is fixed by a reference density at a reference point.
#
# the matching force is `a = -grad phi`, taken by symbolic differentiation of the same
# graph, so the two are one declaration.
#
# usage:
#  phi = -gm / sqrt(x*x + eps*eps)
#  eq  = isentropic_atmosphere(phi, gamma=5/3, k_entropy=1.0, dim=1)
#  graph.compile(eq.primitives).serialize_equilibrium(dim=1)
#  graph.compile(eq.acceleration).serialize_source("force", dim=1)
# =============================================================================
from dataclasses import dataclass
from typing import Sequence

from .dag_expression import (
    X1_ALIASES,
    X2_ALIASES,
    X3_ALIASES,
    Expr,
    constant,
    exp,
    variable,
)

__all__ = [
    "Equilibrium",
    "gradient",
    "isentropic_atmosphere",
    "isothermal_atmosphere",
]

_POSITION = ("x1", "x2", "x3")


@dataclass(frozen=True)
class Equilibrium:
    """a stationary state and the acceleration field it balances, from one potential.

    `primitives` is `[rho, v_0 .. v_{dim-1}, p]` in the order
    `CompiledExpr.serialize_equilibrium` expects; `acceleration` is `[a_0 .. a_{dim-1}]`
    in the order a `force` source expects.
    """

    primitives: list[Expr]
    acceleration: list[Expr]
    potential: Expr


def gradient(scalar: Expr, dim: int) -> list[Expr]:
    """the spatial gradient of a scalar expression, one component per grid dimension.

    differentiation is symbolic on the same graph, so a subexpression the potential and its
    gradient share is written once. the position variables are matched by name, which is why
    the potential must be written in terms of `variable("x1")` and friends rather than a
    numeric coordinate array.
    """
    if not 1 <= dim <= 3:
        raise ValueError(f"gradient needs a grid dimension in 1..3, got {dim}")
    graph = scalar._graph
    return [scalar.diff(variable(_POSITION[axis], graph)) for axis in range(dim)]


def isentropic_atmosphere(
    potential: Expr,
    *,
    gamma: float,
    k_entropy: float,
    dim: int,
    reference_density: float = 1.0,
    reference_point: Sequence[float] | None = None,
) -> Equilibrium:
    """the isentropic atmosphere in hydrostatic balance against `potential`, at rest.

    solves `h + phi = const` in closed form, with the constant fixed by requiring
    `rho = reference_density` at `reference_point` (the domain origin by default).

    the gas is isentropic by construction: `p = k_entropy rho^gamma` everywhere, one
    entropy for the whole atmosphere. a stratified-entropy atmosphere is a different
    problem — it does not reduce to a bernoulli invariant and needs the balance integrated
    along a path, which has no closed form in more than one dimension.

    `gamma` must exceed 1: at `gamma = 1` the enthalpy is logarithmic and the inversion
    below is not the right one (that is the isothermal atmosphere, an exponential).
    """
    if gamma <= 1.0:
        raise ValueError(
            f"isentropic_atmosphere needs gamma > 1, got {gamma}; an isothermal "
            "atmosphere is exponential in the potential, not a power of it"
        )
    if k_entropy <= 0.0:
        raise ValueError(f"the entropy constant must be positive, got {k_entropy}")
    if reference_density <= 0.0:
        raise ValueError(f"the reference density must be positive, got {reference_density}")

    graph = potential._graph
    point = list(reference_point) if reference_point is not None else [0.0] * dim
    if len(point) != dim:
        raise ValueError(
            f"reference_point has {len(point)} coordinates for a {dim}-dimensional grid"
        )

    # h = gamma K/(gamma-1) rho^(gamma-1), so the invariant is C = h(rho_ref) + phi(x_ref).
    enthalpy_scale = gamma * k_entropy / (gamma - 1.0)
    phi_reference = _evaluate_at(potential, point)
    invariant = enthalpy_scale * reference_density ** (gamma - 1.0) + phi_reference

    # rho = [(C - phi)/(gamma K/(gamma-1))]^(1/(gamma-1)); the bracket is the enthalpy, which
    # is positive wherever the atmosphere exists and vanishes at its outer edge.
    enthalpy = constant(invariant, graph) - potential
    rho = (enthalpy / constant(enthalpy_scale, graph)) ** (1.0 / (gamma - 1.0))
    pressure = constant(k_entropy, graph) * rho ** constant(gamma, graph)
    at_rest = [constant(0.0, graph) for _ in range(dim)]

    return Equilibrium(
        primitives=[rho, *at_rest, pressure],
        # the force the atmosphere is in balance against, from the same graph: a = -grad phi.
        acceleration=[-component for component in gradient(potential, dim)],
        potential=potential,
    )


def isothermal_atmosphere(
    potential: Expr,
    *,
    sound_speed: float,
    dim: int,
    reference_density: float = 1.0,
    reference_point: Sequence[float] | None = None,
) -> Equilibrium:
    """the isothermal atmosphere in hydrostatic balance against `potential`, at rest.

    with `p = cs^2 rho` the balance `grad p = -rho grad phi` becomes
    `grad(ln rho) = -grad phi / cs^2`, which integrates to an EXPONENTIAL rather than a
    power:
        rho = rho_ref exp(-(phi - phi_ref)/cs^2).
    that is why the isentropic construction refuses `gamma = 1` instead of taking a limit.

    the equilibrium carries no pressure component: an isothermal regime stores none, and its
    equation of state supplies `p = cs^2 rho` from the density. `sound_speed` must therefore
    be the SAME cs the run is configured with — an atmosphere built for a different one is
    not a steady state of the run, and the backend's refinement check will say so.

    a LOCALLY isothermal run, whose cs varies with position, is a different problem: the
    balance is then a linear ODE with non-constant coefficients and has no closed form.
    """
    if sound_speed <= 0.0:
        raise ValueError(f"the sound speed must be positive, got {sound_speed}")
    if reference_density <= 0.0:
        raise ValueError(f"the reference density must be positive, got {reference_density}")

    graph = potential._graph
    point = list(reference_point) if reference_point is not None else [0.0] * dim
    if len(point) != dim:
        raise ValueError(
            f"reference_point has {len(point)} coordinates for a {dim}-dimensional grid"
        )

    cs2 = sound_speed * sound_speed
    phi_reference = _evaluate_at(potential, point)
    # rho = rho_ref exp(-(phi - phi_ref)/cs^2), written with the reference folded into the
    # exponent so the whole profile is one exponential of a shifted potential.
    exponent = (constant(phi_reference, graph) - potential) / constant(cs2, graph)
    rho = constant(reference_density, graph) * exp(exponent)
    at_rest = [constant(0.0, graph) for _ in range(dim)]

    return Equilibrium(
        # no pressure slot: the isothermal regime does not store one.
        primitives=[rho, *at_rest],
        acceleration=[-component for component in gradient(potential, dim)],
        potential=potential,
    )


def _evaluate_at(scalar: Expr, point: Sequence[float]) -> float:
    """the numeric value of a position-only expression at one point, for pinning the
    integration constant.

    every spelling of an axis is bound to that axis's value, because a variable carries the
    name it was written with — a potential in terms of `r` and one in terms of `x1` are the
    same coordinate to the backend and must be to this evaluation too.
    """
    axis_aliases = (X1_ALIASES, X2_ALIASES, X3_ALIASES)
    coordinates: dict[str, float] = {}
    for aliases in axis_aliases:
        for name in aliases:
            coordinates[name] = 0.0
    for axis, value in enumerate(point):
        for name in axis_aliases[axis]:
            coordinates[name] = float(value)
    return scalar._graph.compile([scalar]).evaluate(**coordinates)[0]
