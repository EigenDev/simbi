# =============================================================================
# test_equilibrium_from_potential.py
#
# an equilibrium and the force it balances, derived from one declared potential.
#
# the failure this construction exists to prevent is a profile that balances one gravity
# while the run applies another: smooth, positive, monotone, and wrong by a constant in a
# single term. deriving both sides from the same expression graph makes that
# unrepresentable, and the test that matters is the one asserting the derived state really
# does solve `grad p = -rho grad phi` — checked symbolically, by differentiating the derived
# pressure on the same graph, so it is exact rather than a finite-difference approximation.
# =============================================================================
import math

import pytest

import simbi.expression as expr
from simbi.expression.equilibrium import gradient, isentropic_atmosphere

GAMMA = 5.0 / 3.0
K0 = 1.0
GM = 100.0
OFFSET = 1.0


def point_mass_potential(graph):
    """phi = -GM/r with r = x + OFFSET, so the domain covers r in [1, 2] with no singularity."""
    r = expr.variable("x1", graph) + expr.constant(OFFSET, graph)
    return -expr.constant(GM, graph) / r


def test_the_derived_state_satisfies_hydrostatic_balance() -> None:
    graph = expr.ExprGraph()
    phi = point_mass_potential(graph)
    atmosphere = isentropic_atmosphere(
        phi, gamma=GAMMA, k_entropy=K0, dim=1, reference_density=1.0, reference_point=[1.0]
    )
    rho, _, pressure = atmosphere.primitives

    # dp/dx + rho dphi/dx, differentiated on the graph rather than differenced on a grid:
    # an exact statement about the expressions, with no truncation error to hide behind.
    x1 = expr.variable("x1", graph)
    residual = pressure.diff(x1) + rho * phi.diff(x1)
    compiled = graph.compile([residual, pressure.diff(x1), rho])

    worst = 0.0
    for sample in (0.05 * ii for ii in range(1, 20)):
        imbalance, dpdx, density = compiled.evaluate(x1=sample)
        # relative to the size of the terms being cancelled, so this measures the
        # cancellation rather than the magnitude of either side.
        scale = max(abs(dpdx), abs(density * GM / (sample + OFFSET) ** 2))
        worst = max(worst, abs(imbalance) / scale)
    print(f"largest relative hydrostatic imbalance of the derived state: {worst:.3e}")

    # the construction is a closed-form inversion of the same balance, so what is left is
    # floating-point roundoff in the graph arithmetic, not an approximation error.
    assert worst < 1.0e-12, (
        f"the derived atmosphere violates grad p = -rho grad phi by {worst:.3e} relative; "
        "the closed-form inversion of the bernoulli invariant is wrong"
    )


def test_the_acceleration_is_the_gradient_of_the_same_potential() -> None:
    # the point of one declaration: the force cannot disagree with the state it balances.
    graph = expr.ExprGraph()
    phi = point_mass_potential(graph)
    atmosphere = isentropic_atmosphere(phi, gamma=GAMMA, k_entropy=K0, dim=1)
    compiled = graph.compile(atmosphere.acceleration)

    for sample in (0.0, 0.3, 0.7, 1.0):
        (accel,) = compiled.evaluate(x1=sample)
        exact = -GM / (sample + OFFSET) ** 2
        assert accel == pytest.approx(exact, rel=1.0e-14), (
            f"a({sample}) = {accel:.6e} against the analytic -GM/r^2 = {exact:.6e}"
        )


def test_the_derived_profile_matches_the_closed_form() -> None:
    # a golden check against the hand-written profile the same physics gives:
    # rho = [(gamma-1)/(gamma K)(GM/r + c)]^(1/(gamma-1)), normalized to 1 at the outer edge.
    graph = expr.ExprGraph()
    atmosphere = isentropic_atmosphere(
        point_mass_potential(graph),
        gamma=GAMMA,
        k_entropy=K0,
        dim=1,
        reference_density=1.0,
        reference_point=[1.0],
    )
    compiled = graph.compile(atmosphere.primitives)

    a = (GAMMA - 1.0) / (GAMMA * K0)
    c = 1.0 / a - GM / (1.0 + OFFSET)
    for sample in (0.0, 0.3, 0.7, 1.0):
        density, velocity, pressure = compiled.evaluate(x1=sample)
        expected_rho = (a * (GM / (sample + OFFSET) + c)) ** (1.0 / (GAMMA - 1.0))
        assert density == pytest.approx(expected_rho, rel=1.0e-13)
        assert pressure == pytest.approx(K0 * expected_rho**GAMMA, rel=1.0e-13)
        assert velocity == 0.0


def test_the_reference_density_is_honoured() -> None:
    graph = expr.ExprGraph()
    atmosphere = isentropic_atmosphere(
        point_mass_potential(graph),
        gamma=GAMMA,
        k_entropy=K0,
        dim=1,
        reference_density=3.0,
        reference_point=[0.5],
    )
    (density, _, _) = graph.compile(atmosphere.primitives).evaluate(x1=0.5)
    assert density == pytest.approx(3.0, rel=1.0e-13)


def test_a_multidimensional_potential_gradients_on_every_axis() -> None:
    # a spherically symmetric potential in a 2d cartesian box: each component of the derived
    # acceleration must be the corresponding partial derivative, not a repeat of the first.
    graph = expr.ExprGraph()
    x = expr.variable("x1", graph)
    y = expr.variable("x2", graph)
    softening = expr.constant(0.1, graph)
    radius = expr.sqrt(x * x + y * y + softening * softening)
    phi = -expr.constant(GM, graph) / radius

    components = graph.compile(gradient(phi, 2))
    for px, py in ((0.4, 0.9), (-0.7, 0.2)):
        gx, gy = components.evaluate(x1=px, x2=py)
        r2 = px * px + py * py + 0.01
        scale = GM / r2**1.5
        assert gx == pytest.approx(px * scale, rel=1.0e-12)
        assert gy == pytest.approx(py * scale, rel=1.0e-12)
        # and the two really are different numbers, or the check above is symmetric noise.
        assert not math.isclose(gx, gy)


def test_an_isothermal_gamma_is_refused() -> None:
    # at gamma = 1 the enthalpy is logarithmic and the atmosphere is exponential in the
    # potential, not a power of it; the closed form below does not apply.
    graph = expr.ExprGraph()
    with pytest.raises(ValueError, match="gamma > 1"):
        isentropic_atmosphere(
            point_mass_potential(graph), gamma=1.0, k_entropy=K0, dim=1
        )


def test_the_isothermal_atmosphere_is_exponential_in_the_potential() -> None:
    # p = cs^2 rho makes the balance grad(ln rho) = -grad phi / cs^2, which integrates to an
    # exponential rather than a power. checked symbolically on the same graph.
    graph = expr.ExprGraph()
    phi = point_mass_potential(graph)
    sound_speed = 2.0
    atmosphere = expr.isothermal_atmosphere(
        phi, sound_speed=sound_speed, dim=1, reference_density=1.0, reference_point=[1.0]
    )
    # no pressure component: an isothermal regime stores none.
    assert len(atmosphere.primitives) == 2
    rho = atmosphere.primitives[0]

    x1 = expr.variable("x1", graph)
    # d(cs^2 rho)/dx + rho dphi/dx, the balance the isothermal regime actually integrates.
    residual = expr.constant(sound_speed**2, graph) * rho.diff(x1) + rho * phi.diff(x1)
    compiled = graph.compile([residual, rho])

    worst = 0.0
    for sample in (0.05 * ii for ii in range(1, 20)):
        imbalance, density = compiled.evaluate(x1=sample)
        scale = density * GM / (sample + OFFSET) ** 2
        worst = max(worst, abs(imbalance) / scale)
    print(f"largest relative isothermal imbalance: {worst:.3e}")
    assert worst < 1.0e-12, (
        f"the isothermal atmosphere violates cs^2 grad rho = -rho grad phi by {worst:.3e}"
    )


def test_the_isothermal_profile_matches_the_closed_form() -> None:
    graph = expr.ExprGraph()
    sound_speed = 2.0
    atmosphere = expr.isothermal_atmosphere(
        point_mass_potential(graph),
        sound_speed=sound_speed,
        dim=1,
        reference_density=1.0,
        reference_point=[1.0],
    )
    compiled = graph.compile(atmosphere.primitives)
    phi_reference = -GM / (1.0 + OFFSET)
    for sample in (0.0, 0.3, 0.7, 1.0):
        density, velocity = compiled.evaluate(x1=sample)
        phi = -GM / (sample + OFFSET)
        expected = math.exp(-(phi - phi_reference) / sound_speed**2)
        assert density == pytest.approx(expected, rel=1.0e-13)
        assert velocity == 0.0


def test_a_nonpositive_sound_speed_is_refused() -> None:
    graph = expr.ExprGraph()
    with pytest.raises(ValueError, match="sound speed must be positive"):
        expr.isothermal_atmosphere(
            point_mass_potential(graph), sound_speed=0.0, dim=1
        )
