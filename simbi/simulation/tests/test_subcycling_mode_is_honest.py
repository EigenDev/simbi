# =============================================================================
# test_subcycling_mode_is_honest.py
#
# a configuration may not declare a refinement subcycling schedule the backend does not implement.
#
# only the fixed-ratio schedule exists: level `l` advances `2^l` times per root step, and the root
# step is the minimum over levels of that level's own cfl limit times `2^l`, so every level lands
# inside its own cfl. `refinement_subcycling_mode` and `refinement_substeps` reach NO backend code —
# they are validated here and dropped at the binding.
#
# the failure this closes is not a crash. two production configs declared ADAPTIVE and received the
# fixed schedule, and their surrounding reasoning about timestep depth was written against the
# declaration rather than against what runs. the two differ: under the fixed schedule the ROOT is
# throttled by the finest level's requirement — measured at roughly twenty times more root steps
# than its own cfl needs on a fourteen-level gravitational ladder. bounded (the finest level
# dominates the work either way, so an ideal schedule saves order twenty percent), but not nothing.
# =============================================================================

import pytest

from simbi.simulation.examples.sod import SodProblem
from simbi.types import SubCycleMode


def _refined(**overrides):
    """a minimally refined cartesian problem — refinement must be ON for the subcycling schedule to
    be validated at all, since an unrefined run has no levels to subcycle."""
    return SodProblem(
        refinement_enabled=True,
        refinement_max_levels=2,
        refinement_regions=[[0.4, 0.6]],
        refinement_ratios=[2],
        **overrides,
    )


@pytest.mark.parametrize("mode", [SubCycleMode.ADAPTIVE, SubCycleMode.MANUAL])
def test_an_unimplemented_schedule_is_refused(mode: SubCycleMode) -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        _refined(refinement_subcycling_mode=mode)


@pytest.mark.parametrize("mode", [SubCycleMode.STANDARD, SubCycleMode.NONE])
def test_the_implemented_schedule_is_accepted(mode: SubCycleMode) -> None:
    # NONE and STANDARD both name the fixed-ratio schedule and are equivalent. the premise: the
    # refusal must be specific to the unimplemented modes, not a blanket rejection of anything set
    # explicitly.
    problem = _refined(refinement_subcycling_mode=mode)
    assert problem.refinement_subcycling_mode is mode


def test_the_default_is_an_implemented_schedule() -> None:
    # a default that was itself refused would make every refined config fail out of the box.
    assert _refined().refinement_subcycling_mode in (
        SubCycleMode.STANDARD,
        SubCycleMode.NONE,
    )


def test_no_shipped_config_declares_an_unimplemented_schedule() -> None:
    # the point of the guard is that a config cannot reason about a schedule it is not getting, so
    # the shipped configs must not do it either. a source scan rather than an import sweep: importing
    # every config runs its setup, and some build large initial conditions.
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "simbi_configs"
    if not root.is_dir():
        pytest.skip("simbi_configs is not present in this checkout")

    offenders = []
    for path in root.rglob("*.py"):
        source = path.read_text()
        if "SubCycleMode.ADAPTIVE" in source or "SubCycleMode.MANUAL" in source:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, (
        "these configs declare a subcycling schedule the backend does not implement, so they fail "
        f"validation: {offenders}"
    )
