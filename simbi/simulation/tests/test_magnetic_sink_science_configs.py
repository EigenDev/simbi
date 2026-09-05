# =============================================================================
# test_magnetic_sink_science_configs.py
#
# the magnetic-sink science configurations on a refined hierarchy: each accepts fixed mesh
# refinement, derives its telescoping levels, sizes the sink and its slip shell on the finest
# level, and places the sink's operator support (the accretion sphere, the drain mask and the
# slip shell to their f64 support, and the stencil reach) inside the innermost region, the rule
# the hierarchy enforces when the bodies attach.
# =============================================================================
import math

import pytest

# the science suite lives outside the tracked tree; its absence skips these gates.
pytest.importorskip("simbi_configs.science.projects.magnetic_sink_common")
from simbi_configs.science.projects.magnetic_bhl import MagnetizedBHL  # noqa: E402
from simbi_configs.science.projects.magnetic_binary_bondi import MagnetizedBinaryBondi  # noqa: E402
from simbi_configs.science.projects.magnetic_bondi import MagnetizedBondi  # noqa: E402


def _mask_support_widths() -> float:
    # the smooth mask chi = (1 - tanh(phi / w)) / 2 is exactly zero in f64 beyond this many
    # widths, plus one width of margin for the library tanh.
    return 0.5 * math.log(4.0 / 2.220446049250313e-16) + 1.0


def _support_radius(accretion_radius: float, shell_width: float | None, dx: float) -> float:
    widths = _mask_support_widths()
    drain = accretion_radius + widths * dx
    slip = accretion_radius + widths * shell_width if shell_width else accretion_radius
    return max(drain, slip) + 3.0 * dx


def _actual_finest_dx(problem) -> float:
    dx = (problem.bounds[0][1] - problem.bounds[0][0]) / problem.resolution[0]
    for ratio in problem.refinement_ratios or ():
        dx /= float(ratio)
    return dx


def _innermost_half_width(problem) -> float:
    inner = problem.refinement_regions[-1]
    return min((inner[2 * a + 1] - inner[2 * a]) / 2.0 for a in range(3))


@pytest.mark.parametrize("cls", [MagnetizedBondi, MagnetizedBHL])
def test_single_sink_science_configs_refine_with_the_support_inside_the_finest(cls) -> None:
    problem = cls(refinement_enabled=True)
    problem.setup()
    assert problem.refinement_max_levels >= 2, f"{cls.__name__} derived no refinement levels"
    assert len(problem.refinement_regions) == problem.refinement_max_levels - 1
    assert problem.finest_dx == pytest.approx(_actual_finest_dx(problem))
    body = problem.immersed_bodies[0]
    shell = getattr(body.magnetic, "shell_width", None)
    support = _support_radius(body.accretion.accretion_radius, shell, problem.finest_dx)
    assert support < _innermost_half_width(problem), (
        f"{cls.__name__}: the sink's operator support ({support:.4g}) reaches past the innermost "
        f"region (half-width {_innermost_half_width(problem):.4g})"
    )


def test_binary_science_config_refines_with_both_sinks_inside_the_finest() -> None:
    problem = MagnetizedBinaryBondi(refinement_enabled=True)
    problem.setup()
    assert problem.refinement_max_levels >= 2
    assert problem.finest_dx == pytest.approx(_actual_finest_dx(problem))
    system = problem.body_system
    half = _innermost_half_width(problem)
    for component in system.binary_config.components:
        shell = getattr(component.magnetic, "shell_width", None)
        support = _support_radius(component.accretion_radius, shell, problem.finest_dx)
        # the components orbit at half the separation from the center of the innermost region.
        assert 0.5 * problem.binary_separation + support < half, (
            f"a binary sink's operator support ({support:.4g}) at its orbit reaches past the "
            f"innermost region (half-width {half:.4g})"
        )


@pytest.mark.parametrize("cls", [MagnetizedBondi, MagnetizedBHL, MagnetizedBinaryBondi])
def test_science_configs_keep_the_uniform_grid_arm(cls) -> None:
    problem = cls(refinement_enabled=False)
    problem.setup()
    assert not problem.refinement_enabled
    assert problem.finest_dx == pytest.approx(
        (problem.bounds[0][1] - problem.bounds[0][0]) / problem.resolution[0]
    )
