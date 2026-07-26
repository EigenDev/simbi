# =============================================================================
# test_gr_example_layout.py
#
# regression coverage for the public gr example set: canonical science
# configurations remain runnable, numerical gates remain test fixtures, and the
# spherical fishbone-moncrief domain excludes only a narrow polar cutout.
#
# usage:
#  pytest simbi/simulation/tests/test_gr_example_layout.py
# =============================================================================
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from simbi_configs.examples.grhd.gr_fishbone_moncrief import GrFishboneMoncrief
from simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd import (
    GrFishboneMoncriefMhd,
)
from simbi_configs.examples.grmhd.gr_fishbone_moncrief_mhd_cartesian import (
    GrFishboneMoncriefMhdCartesian,
)


EXAMPLES = Path(__file__).parents[3] / "simbi_configs" / "examples"


def test_regression_only_gr_configs_are_not_public_examples() -> None:
    removed = (
        EXAMPLES / "grhd" / "gr_rotating_equilibrium.py",
        EXAMPLES / "grhd" / "gr_cylindrical_3d_ks_bh.py",
        EXAMPLES / "grhd" / "gr_disk_ks_bh.py",
        EXAMPLES / "grhd" / "schwarzschild_atmosphere.py",
    )
    assert all(not path.exists() for path in removed)


def test_spherical_fishbone_moncrief_uses_a_narrow_symmetric_cutout() -> None:
    problem = GrFishboneMoncrief()
    theta_lo, theta_hi = problem.bounds[1]
    assert theta_lo == pytest.approx(0.1)
    assert theta_hi == pytest.approx(math.pi - theta_lo)
    assert problem.resolution == (problem.nr, problem.npolar)


@pytest.mark.parametrize("theta_cut", [0.0, -0.1, math.pi / 2.0])
def test_spherical_fishbone_moncrief_rejects_invalid_cutouts(
    theta_cut: float,
) -> None:
    with pytest.raises(ValidationError, match="theta_cut"):
        GrFishboneMoncrief(theta_cut=theta_cut)


@pytest.mark.parametrize(
    "problem_type",
    [GrFishboneMoncriefMhd, GrFishboneMoncriefMhdCartesian],
)
def test_grmhd_fishbone_moncrief_defaults_are_resolved_for_a_spinning_hole(
    problem_type,
) -> None:
    problem = problem_type()
    assert problem.kerr_spin == pytest.approx(0.9)
    assert problem.kappa == pytest.approx(1.3)
