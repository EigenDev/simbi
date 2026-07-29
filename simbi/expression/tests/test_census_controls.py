# =============================================================================
# test_census_controls.py
#
# the two registration controls that decide what a census COSTS rather than what it measures: the
# sample interval, which sets how often the grid is swept, and the accumulate flag, which sets
# whether the samples are stored apiece or folded into one row.
#
# both are inert defaults, which is exactly why they need gating: a control that silently failed
# to reach the wire would leave a run sampling every step and writing every sample, producing
# entirely correct output at many times the intended cost, and nothing in the numbers would say so.
# =============================================================================

import pytest

from simbi.expression import cell_volume, density, variable
from simbi.expression.census import BinAxis, Census, describe, linear_edges
from simbi.expression.dag_expression import ExprGraph


def _census(**kwargs) -> Census:
    g = ExprGraph()
    return Census(
        name="shells",
        axes=[BinAxis("r", variable("x1", g), linear_edges(1.0, 5.0, 8))],
        values={"mass": density(g) * cell_volume(g)},
        **kwargs,
    )


def test_the_controls_reach_the_wire() -> None:
    payload = _census(sample_interval=0.25, accumulate=True).serialize()
    assert payload["sample_interval"] == 0.25
    assert payload["accumulate"] is True


def test_the_defaults_are_every_step_and_a_row_per_sample() -> None:
    # the keys must be PRESENT and explicitly inert rather than absent: the rust side defaults a
    # missing key the same way, so an omission and a deliberate default would be indistinguishable
    # on the wire and a serializer that dropped the fields would go unnoticed.
    payload = _census().serialize()
    assert payload["sample_interval"] is None
    assert payload["accumulate"] is False


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.25])
def test_a_non_positive_interval_is_refused(bad: float) -> None:
    # zero means "every step", which is what omitting it already means, and a negative interval is
    # a sign the caller computed it rather than chose it. clamping either would hide the mistake
    # behind a run that costs what the user was trying to avoid.
    with pytest.raises(ValueError, match="not positive"):
        _census(sample_interval=bad)


def test_the_report_names_every_cost_fixed_at_registration() -> None:
    # what a user needs before submitting: the bin count and the accumulator count set the output
    # size, the graph size sets the per-cell work, and the cadence multiplies both by the number of
    # samples. a hundred-thousand-bin histogram sampled every step is a legitimate ask; finding out
    # from a queue slot is not.
    line = describe([_census(sample_interval=0.25, accumulate=True).serialize()])
    assert "shells" in line
    assert "r:8" in line and "8 bin(s)" in line
    assert "1 accumulator(s)" in line and "mass" in line
    assert "graph node(s)" in line
    assert "every 0.25 in time" in line
    assert "accumulated into one row" in line


def test_the_report_distinguishes_the_defaults() -> None:
    line = describe([_census().serialize()])
    assert "every step" in line
    assert "one row per sample" in line
    assert describe([]) == "no censuses registered"


def test_an_axis_free_census_reports_as_a_global_reduction() -> None:
    g = ExprGraph()
    line = describe(
        [Census(name="totals", values={"mass": density(g) * cell_volume(g)}).serialize()]
    )
    assert "global (no axes)" in line and "1 bin(s)" in line
