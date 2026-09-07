# =============================================================================
# test_axis_boundary.py
#
# the axis boundary condition on the python side: the enum member parses from the
# cli and from checkpoint metadata, and a problem carrying it serializes the tag
# the backend reads.
# =============================================================================
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
