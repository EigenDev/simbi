# =============================================================================
# test_equilibrium_expression_wire.py
#
# the python -> rust stationary-target wire. `CompiledExpr.serialize_equilibrium`
# must emit the rust `EquilibriumConfig` schema (dim/outputs/params/nodes) that
# symbi-expr's load.rs parses and `Hierarchy::with_equilibrium_expression`
# evaluates per cell centre.
#
# a target is a state, not a source: it carries no conservation law to be wrapped
# in and no conserved slot to target, so the payload has no `kind` and the runner
# must not demand one. what it does have to carry is exactly the primitive
# components the regime evolves, since a missing or extra one silently shifts
# every component after it.
# =============================================================================
import json

import pytest

import simbi.expression as expr
from simbi.simulation.runner import _validate_equilibrium_payload


def hydrostatic_payload(dim: int = 1, *, isothermal: bool = False):
    """an isentropic atmosphere in a point-mass potential, as the DAG a config emits:
    `rho = (a (GM/r + c))^(1/(gamma-1))`, `p = K rho^gamma`, `v = 0`."""
    g = expr.ExprGraph()
    gamma, k0, gm, offset = 5.0 / 3.0, 1.0, 100.0, 1.0
    a = (gamma - 1.0) / (gamma * k0)
    c = 1.0 / a - gm / (1.0 + offset)

    r = expr.variable("x1", g) + expr.constant(offset, g)
    rho = (expr.constant(a, g) * (expr.constant(gm, g) / r + expr.constant(c, g))) ** (
        1.0 / (gamma - 1.0)
    )
    outputs = [rho] + [expr.constant(0.0, g) for _ in range(dim)]
    if not isothermal:
        outputs.append(expr.constant(k0, g) * rho ** expr.constant(gamma, g))
    return g.compile(outputs).serialize_equilibrium(dim=dim)


def test_serialize_equilibrium_emits_the_state_schema() -> None:
    cfg = hydrostatic_payload(dim=1)

    assert cfg["dim"] == 1
    # density, one velocity component, pressure -- and nothing that names a
    # conservation law, because a state does not have one.
    assert len(cfg["outputs"]) == 3
    assert "kind" not in cfg
    assert "target" not in cfg
    assert cfg["params"] == []
    # nodes carry the shared op encoding the rust NodeDesc reads.
    assert {n["op"] for n in cfg["nodes"]} >= {"VARIABLE_X1", "CONSTANT", "POW"}
    # must be json-serializable (it crosses the boundary as the exec-dict).
    json.dumps(cfg)


def test_component_count_must_match_the_regime() -> None:
    # a target one component short is the failure this check exists for: the rust
    # side reads positionally, so a missing velocity slot would silently promote
    # the pressure into it and declare a moving atmosphere as the equilibrium.
    short = hydrostatic_payload(dim=2)
    short["outputs"] = short["outputs"][:-1]
    with pytest.raises(ValueError, match="expected 4 primitive components"):
        _validate_equilibrium_payload(
            {"equilibrium_expressions": short, "dimensionality": 2}
        )


def test_isothermal_carries_no_pressure_slot() -> None:
    iso = hydrostatic_payload(dim=1, isothermal=True)
    assert len(iso["outputs"]) == 2
    _validate_equilibrium_payload(
        {
            "equilibrium_expressions": iso,
            "dimensionality": 1,
            "isothermal": True,
        }
    )
    # the same payload on an energy-bearing run is one component short.
    with pytest.raises(ValueError, match="expected 3 primitive components"):
        _validate_equilibrium_payload(
            {"equilibrium_expressions": iso, "dimensionality": 1}
        )


def test_dimension_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="1-dimensional grid"):
        _validate_equilibrium_payload(
            {"equilibrium_expressions": hydrostatic_payload(dim=1), "dimensionality": 2}
        )


def test_mhd_is_refused() -> None:
    # a cell-centered primitive cannot seed the staggered face field, so a
    # magnetized target has no defined interface flux.
    with pytest.raises(ValueError, match="mhd regime"):
        _validate_equilibrium_payload(
            {
                "equilibrium_expressions": hydrostatic_payload(dim=1),
                "dimensionality": 1,
                "is_mhd": True,
            }
        )


def test_seeding_without_a_declared_target_is_refused() -> None:
    # seeding from a target that was never declared would silently do nothing.
    with pytest.raises(ValueError, match="no equilibrium_expressions"):
        _validate_equilibrium_payload(
            {"dimensionality": 1, "seed_from_equilibrium": True}
        )


def test_a_valid_payload_passes() -> None:
    _validate_equilibrium_payload(
        {
            "equilibrium_expressions": hydrostatic_payload(dim=1),
            "dimensionality": 1,
            "seed_from_equilibrium": True,
        }
    )
