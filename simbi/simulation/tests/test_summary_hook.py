# =============================================================================
# test_summary_hook.py
#
# the config summary hook: SimbiProblem.summary() rows (derived quantities)
# join the declared params in the runner's custom_params collection — the
# live dashboard's problem-setup panel.
# =============================================================================

from typing import Annotated

from simbi import ProblemParam, SimbiProblem
from simbi.simulation.runner import _collect_custom_params
from simbi.types import CoordSystem, Regime

_BASE = dict(
    resolution=(8, 8),
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    coord_system=CoordSystem.CARTESIAN,
    regime=Regime.NEWTONIAN,
    adiabatic_index=1.4,
)


class _WithSummary(SimbiProblem):
    knob: Annotated[
        float, ProblemParam(2.0, cli=True, description="a dial", group="physics")
    ]

    def initial_primitive_state(self):
        return lambda: iter(())

    def summary(self) -> list[tuple[str, str, str]]:
        return [("derived", "knob squared", f"{self.knob**2:.1f}")]


def test_summary_rows_join_the_problem_setup_panel() -> None:
    rows = _collect_custom_params(_WithSummary(**_BASE))
    assert ["physics", "knob", "2"] in rows
    assert ["derived", "knob squared", "4.0"] in rows


def test_base_summary_is_empty() -> None:
    class _Plain(SimbiProblem):
        def initial_primitive_state(self):
            return lambda: iter(())

    assert _Plain(**_BASE).summary() == []
    assert all(row[0] != "derived" for row in _collect_custom_params(_Plain(**_BASE)))
