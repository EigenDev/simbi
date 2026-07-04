# =============================================================================
# test_source_expression_wire.py
#
# the python -> rust source-expression wire. `CompiledExpr.serialize_source`
# must emit the rust `SourceConfig` schema (kind/dim/outputs/params/nodes) that
# symbi-expr's load.rs + symbi-hydro's build_user_source consume. the node
# encoding is shared with `serialize()`; this pins the source-level wrapper so
# the binding intake keeps working.
# =============================================================================
import json

import simbi.expression as expr
from simbi_configs.examples.newtonian.rt import RayleighTaylor


def test_serialize_source_emits_sourceconfig_schema() -> None:
    g = expr.ExprGraph()
    ax = expr.constant(0.0, g)
    ay = expr.constant(-1.0, g)
    cfg = g.compile([ax, ay]).serialize_source("force", dim=2, params=[2.0])

    assert cfg["kind"] == "force"
    assert cfg["dim"] == 2
    assert cfg["outputs"] == [0, 1]  # one accel component per dim
    assert cfg["params"] == [2.0]
    # nodes carry the shared op encoding the rust NodeDesc reads.
    assert [n["op"] for n in cfg["nodes"]] == ["CONSTANT", "CONSTANT"]
    assert cfg["nodes"][1]["value"] == -1.0
    # must be json-serializable (it crosses the boundary as the exec-dict).
    json.dumps(cfg)


def test_serialize_source_region_and_target_optional() -> None:
    g = expr.ExprGraph()
    out = expr.constant(1.0, g)
    bare = g.compile([out]).serialize_source("cooling", dim=2)
    assert "region" not in bare and "target" not in bare  # omitted when unset

    tagged = g.compile([out]).serialize_source("raw", dim=2, region=0, target="nrg")
    assert tagged["region"] == 0
    assert tagged["target"] == "nrg"


def test_fluid_state_leaves_emit_state_ops() -> None:
    # state-dependent sources: density/velocity/pressure leaves -> the rust
    # VARIABLE_RHO / VEL{1,2,3} / PRESSURE ops the bridge reads per cell.
    g = expr.ExprGraph()
    # cooling ~ C * rho^2
    rate = expr.parameter(0, g) * expr.density(g) * expr.density(g)
    nodes = g.compile([rate]).serialize_source(
        expr.SourceKind.COOLING, dim=2, params=[0.1]
    )["nodes"]
    assert "VARIABLE_RHO" in [n["op"] for n in nodes]

    g2 = expr.ExprGraph()
    nodes2 = g2.compile(
        [expr.velocity(0, g2), expr.velocity(1, g2), expr.pressure(g2)]
    ).serialize_source(expr.SourceKind.RAW, dim=2, target=expr.ConservedField.ENERGY)["nodes"]
    ops = [n["op"] for n in nodes2]
    assert ops == ["VARIABLE_VEL1", "VARIABLE_VEL2", "VARIABLE_PRESSURE"]


def test_velocity_axis_bounds() -> None:
    import pytest

    with pytest.raises(ValueError):
        expr.velocity(3)


def test_rt_config_emits_force_source() -> None:
    # the reference config: RT gravity is a `force` source with `dim` accel outputs.
    prob = RayleighTaylor(g0=0.3)
    src = prob.gravity_source_expressions
    assert src["kind"] == "force"
    assert len(src["outputs"]) == src["dim"] == 2
    # a = (0, -g0): the second output node is the -g0 constant.
    ay_node = src["nodes"][src["outputs"][1]]
    assert ay_node["op"] == "CONSTANT"
    assert ay_node["value"] == -0.3
