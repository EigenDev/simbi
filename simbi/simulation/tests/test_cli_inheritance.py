# =============================================================================
# test_cli_inheritance.py
#
# regression: a subclass that overrides a core knob (e.g., `solver`) ONLY to
# change its default must keep the base class's `cli=True`. otherwise the
# Annotated override replaces the base metadata wholesale, dropping the cli
# flag, so `--solver` is an unknown arg that `parse_known_args` silently
# swallows — and the run uses the wrong riemann solver. the mro-aware
# `_field_is_cli` restores the exposure (the child default/type still win).
# =============================================================================
from typing import Annotated

from simbi.simulation.param import ProblemParam
from simbi.simulation.problem import SimbiProblem
from simbi.types.input import Solver
from simbi_configs.examples.newtonian.kh import KelvinHelmholtz


class _ShadowSolver(SimbiProblem):
    # mirrors the example configs (kh, sedov, rt, ...): override `solver` only
    # to change the default, dropping `cli=True` from the Annotated metadata.
    solver: Annotated[Solver, ProblemParam(Solver.HLLC, description="solver")]


def test_overridden_core_knob_keeps_cli_exposure() -> None:
    # the base declares solver cli=True; the override drops it, but the
    # mro walk must still report the field as cli-exposed (classmethod — no
    # instantiation, so the abstract base needs no concrete subclass here).
    assert _ShadowSolver._field_is_cli("solver")


def test_non_cli_base_field_stays_hidden() -> None:
    # a field that no class in the mro marks cli=True must not be exposed.
    assert not _ShadowSolver._field_is_cli("regime")


def test_cli_solver_override_applies_to_real_config() -> None:
    # the real regression: kh overrides solver's default to HLLC and drops
    # cli=True. without the mro-aware fix, --solver is silently ignored.
    assert KelvinHelmholtz.from_cli([]).solver is Solver.HLLC
    assert KelvinHelmholtz.from_cli(["--solver", "hlle"]).solver is Solver.HLLE
    assert KelvinHelmholtz.from_cli(["--solver", "hllc"]).solver is Solver.HLLC
