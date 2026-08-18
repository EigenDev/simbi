# =============================================================================
# test_bondi_machinery.py
#
# the shared bondi machinery must hold the properties the configs rely on:
# the ramp is its own region mask, the far field carries the constant bondi
# flux, the sponge wire respects the isothermal regime's five-output spec, and
# the telescoping ladder guarantees nesting margins. these functions replaced
# two to four hand-copied transcriptions that had already drifted, so every
# property here is one a copy actually violated or nearly violated.
# =============================================================================
import math

import pytest

import simbi.expression as expr
from simbi.functional.bondi import (
    accretion_coefficient,
    bondi_far_field,
    bondi_profile,
    far_field_sponge_outputs,
    refinement_levels,
    sponge_ramp,
    sponge_wire,
    telescoping_regions,
)


def _eval_at(outputs: list[expr.Expr], x: float, y: float, z: float) -> list[float]:
    graph = outputs[0].graph
    return graph.compile(outputs).evaluate(x1=x, x2=y, x3=z)


def _coords(graph: expr.ExprGraph) -> tuple[expr.Expr, expr.Expr, expr.Expr]:
    return (
        expr.variable("x1", graph),
        expr.variable("x2", graph),
        expr.variable("x3", graph),
    )


def test_accretion_coefficient_branches() -> None:
    assert accretion_coefficient(1.0) == math.e**1.5 / 4.0
    assert accretion_coefficient(5.0 / 3.0) == 0.25
    # the general branch approaches both exact edges continuously.
    assert abs(accretion_coefficient(1.00002) - math.e**1.5 / 4.0) < 1e-3
    assert abs(accretion_coefficient(5.0 / 3.0 - 2e-5) - 0.25) < 1e-3


def test_bondi_profile_interior_is_free_fall() -> None:
    # inside the bondi radius the profile is the free-fall branch: rho ~ r^-3/2
    # at sonic inflow v = -cs.
    kwargs = dict(bondi_radius=1.0, density=2.0, sound_speed=0.5, gamma=5.0 / 3.0)
    rho_a, v_a, _ = bondi_profile(0.4, **kwargs)
    rho_b, v_b, _ = bondi_profile(0.1, **kwargs)
    assert v_a == v_b == -0.5
    assert abs(rho_b / rho_a - 4.0**1.5) < 1e-12


def test_bondi_profile_isothermal_pressure_is_ambient() -> None:
    _, _, p = bondi_profile(0.3, bondi_radius=1.0, density=2.0, sound_speed=0.5, gamma=1.0)
    assert p == 2.0 * 0.25


def test_bondi_profile_center_guard() -> None:
    rho, v, p = bondi_profile(0.0, bondi_radius=1.0, density=1.0, sound_speed=1.0, gamma=1.4)
    assert (rho, v, p) == (1.0, 0.0, 1.0)


def test_sponge_ramp_is_its_own_region_mask() -> None:
    # kappa must be exactly zero inside the onset (it replaces a `where` gate)
    # and saturate at 1/damp_time beyond onset + width.
    graph = expr.ExprGraph()
    x1, x2, x3 = _coords(graph)
    kappa, _ = sponge_ramp(x1, x2, x3, onset_radius=2.0, width=1.0, damp_time=4.0)

    (inside,) = _eval_at([kappa], 1.0, 0.5, 0.0)
    assert inside == 0.0

    (saturated,) = _eval_at([kappa], 3.5, 0.0, 0.0)
    assert abs(saturated - 0.25) < 1e-14

    # the cubic smoothstep's midpoint: t = 0.5 -> 3/4 - 2/8 = 0.5 of the rate.
    (mid,) = _eval_at([kappa], 2.5, 0.0, 0.0)
    assert abs(mid - 0.125) < 1e-9


def test_far_field_carries_the_constant_bondi_flux() -> None:
    # the O(1/r_norm) density and velocity corrections cancel in rho v, so
    # 4 pi r^2 rho v must be constant to second order across the buffer.
    gamma = 5.0 / 3.0
    graph = expr.ExprGraph()
    x1, x2, x3 = _coords(graph)
    r = expr.sqrt(x1 * x1 + x2 * x2 + x3 * x3)
    rho, v_r, _ = bondi_far_field(
        r, bondi_radius=1.0, density=1.0, sound_speed=1.0, gamma=gamma
    )

    def flux_at(radius: float) -> float:
        rho_v, rv = _eval_at([rho * v_r, r], radius, 0.0, 0.0)
        return 4.0 * math.pi * rv**2 * rho_v

    # (1 + 0.5/rr)(1 - 0.5/rr) = 1 - 0.25/rr^2 exactly, so the flux departs its
    # asymptote by precisely that factor -- an identity, not an approximation.
    asymptote = -accretion_coefficient(gamma) * 4.0 * math.pi
    for radius in (10.0, 40.0, 200.0):
        rr = radius / 2.0
        expected = asymptote * (1.0 - 0.25 / rr**2)
        got = flux_at(radius)
        assert abs(got / expected - 1.0) < 1e-12, (
            f"at r = {radius} the flux is {got:.15g}, not the asymptote times "
            f"(1 - 0.25/rr^2) = {expected:.15g}; the corrections no longer cancel"
        )


def test_sponge_wire_isothermal_has_no_energy_channel() -> None:
    graph = expr.ExprGraph()
    x1, x2, x3 = _coords(graph)
    kappa, r = sponge_ramp(x1, x2, x3, onset_radius=1.0, width=1.0, damp_time=1.0)
    rho, v_r, cs_eq = bondi_far_field(
        r, bondi_radius=1.0, density=1.0, sound_speed=1.0, gamma=1.0
    )
    vx = v_r * x1 / r
    outputs = sponge_wire(kappa, rho, vx, vx, vx, cs_eq=cs_eq, gamma=1.0)
    assert len(outputs) == 5
    outputs = sponge_wire(kappa, rho, vx, vx, vx, cs_eq=cs_eq, gamma=5.0 / 3.0)
    assert len(outputs) == 6


def test_sponge_wire_carries_primitives_under_the_ideal_gas_closure() -> None:
    # the wire hands over a state, not a conserved vector: slot 5 is the reference
    # PRESSURE under p = rho cs^2/gamma, and the velocity slots are velocities. the
    # conserved reference -- including the kinetic term the energy needs -- is built
    # by the evolving regime from these, which is what lets one wire serve a newtonian
    # gas, a relativistic one and a curved background.
    gamma = 1.4
    graph = expr.ExprGraph()
    x1, x2, x3 = _coords(graph)
    rho = expr.constant(2.0, graph) + 0.0 * x1
    cs = expr.constant(0.5, graph) + 0.0 * x1
    vx = expr.constant(0.3, graph) + 0.0 * x1
    zero = expr.constant(0.0, graph) + 0.0 * x1
    outputs = sponge_wire(zero, rho, vx, zero, zero, cs_eq=cs, gamma=gamma)
    vals = _eval_at(outputs, 1.0, 0.0, 0.0)
    assert abs(vals[5] - 2.0 * 0.25 / gamma) < 1e-12
    # the velocity slot is the velocity itself, undivided by and unmultiplied by density:
    # at rho = 2 a momentum wire would read 0.6 here.
    assert abs(vals[2] - 0.3) < 1e-12


def test_composed_outputs_match_the_pieces() -> None:
    # the convenience composer must be exactly the pieces it is built from.
    gamma = 5.0 / 3.0
    kw = dict(
        onset_radius=3.0,
        width=1.0,
        damp_time=2.0,
        bondi_radius=1.0,
        density=1.5,
        sound_speed=0.8,
        gamma=gamma,
    )
    graph = expr.ExprGraph()
    x1, x2, x3 = _coords(graph)
    composed = far_field_sponge_outputs(x1, x2, x3, **kw)

    graph2 = expr.ExprGraph()
    y1, y2, y3 = _coords(graph2)
    kappa, r = sponge_ramp(
        y1, y2, y3, onset_radius=3.0, width=1.0, damp_time=2.0
    )
    rho, v_r, cs_eq = bondi_far_field(
        r, bondi_radius=1.0, density=1.5, sound_speed=0.8, gamma=gamma
    )
    manual = sponge_wire(
        kappa, rho, v_r * y1 / r, v_r * y2 / r, v_r * y3 / r, cs_eq=cs_eq, gamma=gamma
    )

    point = (3.7, 1.2, -0.4)
    for got, want in zip(_eval_at(composed, *point), _eval_at(manual, *point)):
        assert got == pytest.approx(want, abs=0.0, rel=1e-15)


def test_refinement_levels() -> None:
    assert refinement_levels(1.0, 1.0) == 0
    assert refinement_levels(1.0, 2.0) == 0
    assert refinement_levels(1.0, 0.5) == 1
    assert refinement_levels(1.0, 0.126) == 3
    assert refinement_levels(1.0, 0.125) == 3


def test_telescoping_regions_guarantee_nesting_margins() -> None:
    # each region is at most half its parent's radius; without the cap, deep
    # telescopes stack domain-clamped levels the coverage check rejects.
    for box_radius in (10.0, 200.0):
        regions = telescoping_regions(6, finest_radius=2.0, box_radius=box_radius)
        assert len(regions) == 6
        r_prev = box_radius
        for box in regions:
            radius = box[1]
            assert box == [-radius, radius] * 3
            assert radius <= 0.5 * r_prev + 1e-15
            r_prev = radius
        radii = [b[1] for b in regions]
        assert radii == sorted(radii, reverse=True)
    # in a wide domain the cap releases and the finest level covers the request;
    # in a small one the cap binds all the way down (box_radius / 2^levels), which
    # is what guarantees the margins the coverage check demands.
    assert telescoping_regions(6, finest_radius=2.0, box_radius=200.0)[-1][1] == 2.0
    assert telescoping_regions(6, finest_radius=2.0, box_radius=10.0)[-1][1] == 10.0 / 2**6


def test_telescoping_regions_ndim() -> None:
    assert len(telescoping_regions(1, finest_radius=1.0, box_radius=4.0, ndim=2)[0]) == 4
