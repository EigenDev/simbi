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
