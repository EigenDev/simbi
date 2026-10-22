# =============================================================================
# test_restart_resolution.py
#
# what a restart takes from the checkpoint and what it keeps from the
# configuration: the chart of a two-dimensional cylindrical grid, which the
# file records as a family plus a plane, and the boundary condition of a face
# the file records by kind alone (a gradient or robin wall), whose
# coefficients live in the configuration.
#
# run: pytest simbi/simulation/tests/test_restart_resolution.py
# =============================================================================
import pytest

from simbi.simulation.checkpoint import boundaries_on_restart, chart_on_restart
from simbi.simulation.problem import ConfigError
from simbi.types.input import BoundaryCondition, CoordSystem, Neumann, Robin


def test_the_disk_plane_survives_a_restart_from_its_recorded_plane() -> None:
    assert chart_on_restart("cylindrical", "rphi", CoordSystem.PLANAR_CYLINDRICAL) == CoordSystem.PLANAR_CYLINDRICAL
    assert chart_on_restart("cylindrical", "rz", CoordSystem.CYLINDRICAL) == CoordSystem.CYLINDRICAL
    assert chart_on_restart("cylindrical", "rz", CoordSystem.AXIS_CYLINDRICAL) == CoordSystem.AXIS_CYLINDRICAL


def test_a_checkpoint_without_a_plane_keeps_the_configured_cylindrical_variant() -> None:
    assert chart_on_restart("cylindrical", "", CoordSystem.PLANAR_CYLINDRICAL) == CoordSystem.PLANAR_CYLINDRICAL
    assert chart_on_restart("cylindrical", "", CoordSystem.AXIS_CYLINDRICAL) == CoordSystem.AXIS_CYLINDRICAL


def test_a_recorded_plane_the_configuration_contradicts_selects_the_recorded_plane() -> None:
    assert chart_on_restart("cylindrical", "rphi", CoordSystem.CYLINDRICAL) == CoordSystem.PLANAR_CYLINDRICAL
    assert chart_on_restart("cylindrical", "rz", CoordSystem.PLANAR_CYLINDRICAL) == CoordSystem.CYLINDRICAL


def test_another_family_is_taken_as_recorded() -> None:
    assert chart_on_restart("spherical", "", CoordSystem.CARTESIAN) == CoordSystem.SPHERICAL
    assert chart_on_restart("cartesian", "", CoordSystem.PLANAR_CYLINDRICAL) == CoordSystem.CARTESIAN


def test_vocabulary_faces_come_from_the_checkpoint() -> None:
    faces = boundaries_on_restart(("outflow", "reflecting", "periodic", "periodic"), [BoundaryCondition.OUTFLOW])
    assert faces == [BoundaryCondition.OUTFLOW, BoundaryCondition.REFLECTING, BoundaryCondition.PERIODIC, BoundaryCondition.PERIODIC]


def test_gradient_and_robin_faces_take_the_configured_coefficients() -> None:
    wall = Neumann(rho=0.5, velocity=(0.0, 0.0), pressure=1.5)
    robin = Robin(rho=(1.0, 0.0, 2.0))
    declared = [wall, wall, robin, robin]
    faces = boundaries_on_restart(("neumann", "neumann", "robin", "robin"), declared)
    assert faces == declared


def test_a_recorded_kind_the_configuration_contradicts_is_refused() -> None:
    with pytest.raises(ConfigError, match="face 1: the checkpoint records a 'neumann' boundary"):
        boundaries_on_restart(("outflow", "neumann"), [BoundaryCondition.OUTFLOW, BoundaryCondition.OUTFLOW])


@pytest.mark.simulation
def test_a_grid_spelled_at_its_own_dimension_keeps_that_spelling_on_restart(tmp_path) -> None:
    # the checkpoint records a 2d grid in the three-axis form (n, n, 1). a problem that spells it
    # (n, n), and reads its rank from that length, keeps (n, n): both name the same grid, and the
    # three-axis form would make the problem generate a 3d state for a 2d run
    from simbi.simulation import runner
    from simbi.simulation.checkpoint import merge_with_checkpoint
    from simbi_configs.examples.newtonian.linear_wave import LinearWave

    first = tmp_path / "first"
    runner.run(LinearWave(resolution=(32, 32), data_directory=first, end_time=1e9), compute_mode="cpu", max_steps=2)
    final = next(first.glob("*final*.h5"))
    resumed = LinearWave(resolution=(32, 32), data_directory=tmp_path / "resumed", checkpoint_file=final, end_time=1e9)
    assert merge_with_checkpoint(resumed, final).resolution == (32, 32)
    runner.run(resumed, compute_mode="cpu", max_steps=4)
    assert next((tmp_path / "resumed").glob("*final*.h5"))
