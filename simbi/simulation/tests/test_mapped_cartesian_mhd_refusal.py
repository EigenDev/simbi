# =============================================================================
# test_mapped_cartesian_mhd_refusal.py
#
# the cartesian constrained-transport curl reads one uniform width per axis, so
# a magnetized regime on a cartesian grid with a non-uniform (geometric or log)
# axis would evolve its staggered field on the uniform lattice while the gas
# moves on the mapped one. such a configuration is refused by name before the
# first step; the same problem on a uniform grid runs.
#
# run: pytest -m simulation simbi/simulation/tests/test_mapped_cartesian_mhd_refusal.py
# =============================================================================

import pytest

from simbi.simulation import runner
from simbi_configs.examples.newtonian.nmhd_orszag_tang import NewtonianOrszagTang

pytestmark = pytest.mark.simulation


def _vortex(tmp_path, **spacing) -> NewtonianOrszagTang:
    return NewtonianOrszagTang(resolution=(16, 16, 1), data_directory=str(tmp_path), checkpoint_interval=1e9, **spacing)


def test_a_geometric_cartesian_axis_with_a_magnetized_regime_is_refused(tmp_path) -> None:
    with pytest.raises(Exception, match="the nmhd regime on a non-uniform cartesian grid"):
        runner.run(_vortex(tmp_path, x1_spacing="geometric", x1_spacing_ratio=1.05), compute_mode="cpu", max_steps=1)
    assert not list(tmp_path.glob("*final*.h5")), "a refused run published a final checkpoint"


def test_the_uniform_cartesian_vortex_runs(tmp_path) -> None:
    runner.run(_vortex(tmp_path), compute_mode="cpu", max_steps=1)
    assert list(tmp_path.glob("*final*.h5")), "the uniform vortex did not complete its step"
