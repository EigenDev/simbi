# =============================================================================
# test_census_expression_wire.py
#
# the python -> rust census wire. `Census.serialize` must emit the rust
# `CensusConfig` schema (name/axes/values/value_names/op/params/nodes) that
# symbi-expr's load.rs and symbi-hydro's `build_census_expressions` consume.
#
# the registration-time rejections matter as much as the schema: a census that
# only fails once it reaches the grid has already cost a queue slot, and a
# malformed binning that DOESN'T fail is indistinguishable from a physics result.
# =============================================================================
import json

import pytest

import simbi.expression as expr


def _graph_with_radius():
    g = expr.ExprGraph()
    x, y = expr.variable("x", g), expr.variable("y", g)
    return g, expr.sqrt(x * x + y * y)


def test_serialize_emits_the_censusconfig_schema() -> None:
    g, r = _graph_with_radius()
    mass = expr.density(g) * expr.cell_volume(g)
    payload = expr.Census(
        name="shells",
        axes=[expr.BinAxis("r", r, [1.0, 2.0, 4.0])],
        values={"mass": mass},
    ).serialize()

    assert payload["name"] == "shells"
    assert payload["op"] == "add"
    assert payload["value_names"] == ["mass"]
    assert payload["params"] == []
    assert [a["name"] for a in payload["axes"]] == ["r"]
    assert payload["axes"][0]["edges"] == [1.0, 2.0, 4.0]
    # the axis coordinate and the accumulator are node indices into the shared dag.
    assert isinstance(payload["axes"][0]["expr"], int)
    assert len(payload["values"]) == 1
    # the accumulator reads the cell measure, which is what makes it extensive.
    assert "VARIABLE_DV" in [n["op"] for n in payload["nodes"]]
    # must be json-serializable: it crosses the boundary as the exec-dict.
    json.dumps(payload)


def test_a_census_with_no_axes_is_a_global_reduction() -> None:
    # total mass and energy are a census with no bins. the wire must carry that,
    # not require a dummy axis spanning the domain.
    g = expr.ExprGraph()
    payload = expr.Census(
        name="conservation",
        values={"mass": expr.density(g) * expr.cell_volume(g)},
    ).serialize()
    assert payload["axes"] == []
    assert len(payload["values"]) == 1


def test_axis_coordinates_precede_the_accumulators_in_the_outputs() -> None:
    # the rust side unpacks the compiled outputs as bin coordinates first, then
    # values. a swap here would bin on a moment and accumulate a radius.
    g, r = _graph_with_radius()
    rho = expr.density(g)
    payload = expr.Census(
        name="phase",
        axes=[
            expr.BinAxis("r", r, [1.0, 2.0]),
            expr.BinAxis("rho", rho, [0.0, 1.0, 2.0]),
        ],
        values={"m": rho * expr.cell_volume(g), "count": expr.constant(1.0, g)},
    ).serialize()

    axis_nodes = [a["expr"] for a in payload["axes"]]
    assert len(axis_nodes) == 2
    assert len(payload["values"]) == 2
    assert payload["value_names"] == ["m", "count"]
    # the two groups are disjoint node sets in the compiled output order.
    assert not set(axis_nodes) & set(payload["values"])


def test_a_shared_subexpression_is_written_once() -> None:
    # binning on the radius while accumulating a moment that also uses it must not
    # emit the radius twice: the cost of a census scales with the size of its dag.
    g, r = _graph_with_radius()
    payload = expr.Census(
        name="shells",
        axes=[expr.BinAxis("r", r, [1.0, 2.0])],
        values={"m_r": expr.density(g) * r},
    ).serialize()
    ops = [n["op"] for n in payload["nodes"]]
    assert ops.count("SQRT") == 1, f"the radius must be emitted once, got {ops}"


def test_reduction_ops_serialize_to_their_rust_strings() -> None:
    g = expr.ExprGraph()
    for op, expected in (
        (expr.ReductionOp.ADD, "add"),
        (expr.ReductionOp.MIN, "min"),
        (expr.ReductionOp.MAX, "max"),
    ):
        payload = expr.Census(
            name="x", values={"v": expr.density(g)}, op=op
        ).serialize()
        assert payload["op"] == expected


def test_edges_must_strictly_increase_and_be_finite() -> None:
    # a repeated edge makes a bin no value can land in; a decreasing one makes the
    # bin search return the wrong bin. both must be refused at registration.
    g, r = _graph_with_radius()
    with pytest.raises(ValueError, match="strictly increase"):
        expr.BinAxis("r", r, [0.0, 1.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="strictly increase"):
        expr.BinAxis("r", r, [0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="define no bin"):
        expr.BinAxis("r", r, [1.0])
    with pytest.raises(ValueError, match="not finite"):
        expr.BinAxis("r", r, [0.0, float("inf")])


def test_a_census_must_register_at_least_one_value() -> None:
    g = expr.ExprGraph()
    with pytest.raises(ValueError, match="registers no values"):
        expr.Census(name="empty", values={})


def test_duplicate_axis_names_are_refused() -> None:
    # the axis name labels its edges in the output, so a collision is an unreadable
    # result rather than a harmless duplicate.
    g, r = _graph_with_radius()
    with pytest.raises(ValueError, match="both named 'r'"):
        expr.Census(
            name="phase",
            axes=[expr.BinAxis("r", r, [1.0, 2.0]), expr.BinAxis("r", r, [1.0, 2.0])],
            values={"m": expr.density(_graph_with_radius()[0])},
        )


def test_expressions_from_different_graphs_are_refused() -> None:
    # node numbering is per-graph, so mixing graphs would index into the wrong dag
    # and produce a wrong answer rather than an error.
    g1, r = _graph_with_radius()
    g2 = expr.ExprGraph()
    census = expr.Census(
        name="mixed",
        axes=[expr.BinAxis("r", r, [1.0, 2.0])],
        values={"m": expr.density(g2)},
    )
    with pytest.raises(ValueError, match="one ExprGraph"):
        census.serialize()


def test_log_edges_span_the_range_with_equal_ratios() -> None:
    edges = expr.log_edges(1.0, 1000.0, 3)
    assert len(edges) == 4
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(1000.0)
    # equal ratios, which is what makes each shell sample its own decorrelation time
    # equally well around an accretor.
    ratios = [b / a for a, b in zip(edges, edges[1:])]
    assert all(r == pytest.approx(ratios[0]) for r in ratios)
    with pytest.raises(ValueError, match="0 < lo < hi"):
        expr.log_edges(0.0, 10.0, 3)


def test_linear_edges_span_the_range_with_equal_widths() -> None:
    edges = expr.linear_edges(-1.0, 1.0, 4)
    assert edges == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="lo < hi"):
        expr.linear_edges(1.0, 1.0, 4)


def test_gating_needs_no_filter_concept() -> None:
    # "only inflowing cells" is an ordinary value expression built from the
    # comparison operators, not a mechanism the census has to provide.
    g = expr.ExprGraph()
    v_r = expr.velocity(0, g)
    m = expr.density(g) * expr.cell_volume(g)
    payload = expr.Census(
        name="inflow",
        values={"inflow_mass": m * (v_r < expr.constant(0.0, g))},
    ).serialize()
    assert "LT" in [n["op"] for n in payload["nodes"]]
    json.dumps(payload)


# ---- the runner preflight: a malformed registration must not reach the grid -----------

def _sod_payload():
    from simbi.simulation.runner import to_execution_dict
    from simbi_configs.examples.newtonian.sod import SodProblem

    return to_execution_dict(SodProblem())


def _validate(payload):
    from simbi.simulation.runner import _validate_census_payloads

    _validate_census_payloads(payload)


def test_preflight_accepts_a_well_formed_registration() -> None:
    g = expr.ExprGraph()
    payload = _sod_payload()
    payload["census_expressions"] = [
        expr.Census(
            name="conservation",
            values={"mass": expr.density(g) * expr.cell_volume(g)},
        ).serialize()
    ]
    _validate(payload)


def test_preflight_rejects_a_bare_payload() -> None:
    # a hand-built dict carrying only a dag registers a nameless census with no
    # reduce op and no labels. it must be refused rather than reaching the grid.
    payload = _sod_payload()
    payload["census_expressions"] = [{"values": [0], "nodes": []}]
    with pytest.raises(ValueError, match="use Census"):
        _validate(payload)


def test_preflight_rejects_a_duplicate_census_name() -> None:
    # the name is the checkpoint group; two censuses sharing one would have the
    # second silently overwrite the first.
    g = expr.ExprGraph()
    one = expr.Census(name="shells", values={"m": expr.density(g)}).serialize()
    payload = _sod_payload()
    payload["census_expressions"] = [one, dict(one)]
    with pytest.raises(ValueError, match="reuses the census name"):
        _validate(payload)


def test_preflight_rejects_an_unimplemented_reduce_op() -> None:
    g = expr.ExprGraph()
    cfg = expr.Census(name="c", values={"m": expr.density(g)}).serialize()
    cfg["op"] = "mean"
    payload = _sod_payload()
    payload["census_expressions"] = [cfg]
    with pytest.raises(ValueError, match="not order-agnostic"):
        _validate(payload)


def test_preflight_rejects_mismatched_labels() -> None:
    g = expr.ExprGraph()
    cfg = expr.Census(name="c", values={"m": expr.density(g)}).serialize()
    cfg["value_names"] = ["m", "extra"]
    payload = _sod_payload()
    payload["census_expressions"] = [cfg]
    with pytest.raises(ValueError, match="against 2 labels"):
        _validate(payload)


# ---- the full crossing: python registration -> rust lowering ---------------------------

def test_a_registered_census_lowers_in_the_backend() -> None:
    # the whole wire in one step: a Census built from expressions, serialized, and
    # compiled by the rust backend. this is what proves the python emitter and the
    # rust `CensusConfig` reader agree on the schema.
    import simbi.libs.cpu_ext as backend

    g = expr.ExprGraph()
    x, y = expr.variable("x", g), expr.variable("y", g)
    r = expr.sqrt(x * x + y * y)
    m = expr.density(g) * expr.cell_volume(g)

    payload = _sod_payload()
    payload["census_expressions"] = [
        expr.Census(
            name="shells",
            axes=[expr.BinAxis("r", r, expr.log_edges(0.1, 10.0, 8))],
            values={"mass": m, "radial_momentum": m * expr.velocity(0, g)},
        ).serialize()
    ]
    backend.validate_simulation(sim_info=payload)


def test_the_backend_rejects_a_census_it_cannot_lower() -> None:
    # an expression outside the lowerable set must be refused at setup, naming the
    # registration, rather than failing at the first sample.
    import simbi.libs.cpu_ext as backend

    g = expr.ExprGraph()
    payload = _sod_payload()
    cfg = expr.Census(name="c", values={"m": expr.density(g)}).serialize()
    # MOD has no carrier primitive, so the bridge cannot lower it.
    cfg["nodes"] = [
        {"op": "CONSTANT", "value": 1.0},
        {"op": "CONSTANT", "value": 2.0},
        {"op": "MOD", "left": 0, "right": 1},
    ]
    cfg["values"] = [2]
    payload["census_expressions"] = [cfg]
    with pytest.raises(ValueError, match=r"census_expressions\[0\] lower:"):
        backend.validate_simulation(sim_info=payload)


def test_the_backend_rejects_a_duplicate_census_name() -> None:
    # the name is the checkpoint group. the rust side refuses a collision even if a
    # config assembled the list without going through the runner preflight.
    import simbi.libs.cpu_ext as backend

    g = expr.ExprGraph()
    one = expr.Census(name="shells", values={"m": expr.density(g)}).serialize()
    payload = _sod_payload()
    payload["census_expressions"] = [one, dict(one)]
    with pytest.raises(ValueError, match="reuses the census name"):
        backend.validate_simulation(sim_info=payload)
