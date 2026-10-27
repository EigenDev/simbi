# =============================================================================
# test_source_region_mask.py
#
# the wire contract of a source restricted to a region chi(x). the payload's
# `outputs` lists the source kind's own outputs, one node index each, and
# `region` names the node index of the mask, which the backend multiplies into
# the channels the kind masks. a source without a region carries no `region`
# key and the same outputs it always carried.
# =============================================================================
import pytest

import simbi.expression as expr


def _plane():
    x, y = expr.coords(2)
    return x, y, expr.where(x * x + y * y < 0.16, 1.0, 0.0)


CASES = {
    "cooling": (lambda x, y: [0.3], expr.cooling, 1),
    "force": (lambda x, y: [0.3 * x, -0.2], expr.force, 2),
    "velocity_relaxation": (lambda x, y: [2.0, 0.0, 0.0], expr.velocity_relaxation, 3),
    "inject": (lambda x, y: [0.05, 0.02, 0.0, 0.1], expr.inject, 4),
}


@pytest.mark.parametrize("kind", list(CASES))
def test_a_region_names_the_mask_node_and_leaves_the_outputs_to_the_kind(kind: str) -> None:
    outputs, build, count = CASES[kind]
    x, y, disc = _plane()
    payload = build(outputs(x, y), dim=2, region=disc)
    assert len(payload["outputs"]) == count, f"{kind}: outputs {payload['outputs']} carry the mask"
    region = payload["region"]
    assert 0 <= region < len(payload["nodes"]), f"{kind}: region {region} names no node"
    assert region not in payload["outputs"], f"{kind}: the region node is one of the kind's outputs"
    assert payload["nodes"][region]["op"] == "IF_THEN_ELSE", f"{kind}: region names {payload['nodes'][region]} in place of the mask"


@pytest.mark.parametrize("kind", list(CASES))
def test_a_source_without_a_region_serializes_its_outputs_alone(kind: str) -> None:
    outputs, build, count = CASES[kind]
    x, y, _ = _plane()
    payload = build(outputs(x, y), dim=2)
    assert "region" not in payload
    assert len(payload["outputs"]) == count


def test_a_region_shared_with_an_output_names_that_node() -> None:
    x, y, disc = _plane()
    payload = expr.cooling([disc], dim=2, region=disc)
    assert payload["outputs"] == [payload["region"]], "the mask and the rate are one node"
