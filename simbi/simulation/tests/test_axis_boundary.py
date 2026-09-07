# =============================================================================
# test_axis_boundary.py
#
# the axis boundary condition on the python side: the enum member parses from the
# cli and from checkpoint metadata, and a problem carrying it serializes the tag
# the backend reads.
# =============================================================================
from pathlib import Path

import pytest

from simbi.simulation import runner
from simbi.types.input import BoundaryCondition
from simbi_configs.examples.grhd.gr_bondi_ks import GrBondiKS


def test_axis_parses_from_its_tag():
    assert BoundaryCondition("axis") is BoundaryCondition.AXIS
    assert BoundaryCondition.AXIS.value == "axis"


def test_axis_parses_from_the_cli():
    problem = GrBondiKS.from_cli(["--boundary-conditions", "outflow,outflow,axis,reflecting"])
    assert list(problem.boundary_conditions) == [
        BoundaryCondition.OUTFLOW,
        BoundaryCondition.OUTFLOW,
        BoundaryCondition.AXIS,
        BoundaryCondition.REFLECTING,
    ]


@pytest.mark.simulation
def test_a_polar_axis_run_accepts_two_production_steps(tmp_path: Path) -> None:
    # the science suite's rotating conductor grids the full sphere with the axis on both
    # theta faces; two production steps through the backend prove the tag is admitted at the
    # pole and refused nowhere along the way.
    rotating = pytest.importorskip("simbi_configs.science.projects.rotating_conductor")
    problem = rotating.rotatingConductor(zpd=8, data_directory=tmp_path / "axis")
    assert [b.value for b in problem.boundary_conditions][2:] == ["axis", "axis"]
    result = runner.run(problem, compute_mode="cpu", validate=True, max_steps=2)
    assert result.diagnostics.guards.troubled_cells.total == 0
    assert len(list((tmp_path / "axis").glob("*final*.h5"))) == 1


@pytest.mark.simulation
def test_an_axis_face_off_the_pole_is_refused(tmp_path: Path) -> None:
    # the placement rule runs before dispatch: a cartesian chart has no axis.
    from simbi_configs.examples.newtonian.quirk import Quirk

    problem = Quirk(
        resolution=(24, 8),
        boundary_conditions=[
            BoundaryCondition.REFLECTING,
            BoundaryCondition.OUTFLOW,
            BoundaryCondition.AXIS,
            BoundaryCondition.REFLECTING,
        ],
        data_directory=tmp_path / "refused",
    )
    with pytest.raises(Exception, match="cartesian chart has no coordinate axis"):
        runner.run(problem, compute_mode="cpu", validate=True, max_steps=1)
