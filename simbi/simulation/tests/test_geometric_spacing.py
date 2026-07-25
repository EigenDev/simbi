# =============================================================================
# test_geometric_spacing.py
#
# validation gates for geometrically graded x1 cell widths.
# =============================================================================

import pytest
from pydantic import ValidationError

from simbi.types import CellSpacing, MeshConfig
from simbi_configs.examples.grhd.gr_bondi_ks import GrBondiKS


def test_geometric_spacing_accepts_positive_width_ratio():
    problem = GrBondiKS(
        x1_spacing=CellSpacing.GEOMETRIC,
        x1_spacing_ratio=0.98,
    )

    assert problem.x1_spacing == CellSpacing.GEOMETRIC
    assert problem.x1_spacing_ratio == 0.98


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("inf"), float("nan")])
def test_geometric_spacing_rejects_nonpositive_or_nonfinite_ratio(ratio):
    with pytest.raises(ValidationError, match="positive and finite"):
        GrBondiKS(
            x1_spacing=CellSpacing.GEOMETRIC,
            x1_spacing_ratio=ratio,
        )


def test_width_ratio_is_rejected_for_other_spacing_kinds():
    with pytest.raises(ValidationError, match="only valid"):
        GrBondiKS(
            x1_spacing=CellSpacing.LINEAR,
            x1_spacing_ratio=0.98,
        )


def test_mesh_config_reconstructs_geometric_faces_and_centers():
    mesh = MeshConfig(
        shape=(4,),
        bounds_min=(2.0,),
        bounds_max=(5.0,),
        halo_radius=2,
        spacing_types=(CellSpacing.GEOMETRIC,),
        spacing_ratios=(0.8,),
    )

    widths = mesh.x1v[1:] - mesh.x1v[:-1]
    assert mesh.x1v[0] == pytest.approx(2.0)
    assert mesh.x1v[-1] == pytest.approx(5.0)
    assert widths[1:] / widths[:-1] == pytest.approx([0.8, 0.8, 0.8])
    assert mesh.x1c == pytest.approx(0.5 * (mesh.x1v[:-1] + mesh.x1v[1:]))
