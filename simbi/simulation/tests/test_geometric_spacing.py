# =============================================================================
# test_geometric_spacing.py
#
# validation gates for geometrically graded x1 cell widths.
# =============================================================================

from pathlib import Path

import h5py
import pytest
from pydantic import ValidationError

from simbi.simulation.checkpoint import (
    load_checkpoint_metadata,
    metadata_to_config_dict,
)
from simbi.simulation.runner import run
from simbi.types import CellSpacing, MeshConfig
from simbi_configs.examples.grhd.gr_bondi_ks import GrBondiKS
from simbi_configs.examples.srhd.marti_muller_3d import MartiMuller3D


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


@pytest.mark.parametrize("axis", [1, 2, 3])
def test_geometric_spacing_is_configurable_on_every_axis(axis):
    problem = MartiMuller3D(
        resolution=(8, 6, 4),
        **{
            f"x{axis}_spacing": CellSpacing.GEOMETRIC,
            f"x{axis}_spacing_ratio": 0.9 + 0.02 * axis,
        },
    )

    assert getattr(problem, f"x{axis}_spacing") == CellSpacing.GEOMETRIC
    assert getattr(problem, f"x{axis}_spacing_ratio") == pytest.approx(
        0.9 + 0.02 * axis
    )


@pytest.mark.parametrize("axis", [2, 3])
def test_transverse_width_ratio_is_rejected_for_linear_spacing(axis):
    with pytest.raises(ValidationError, match=f"x{axis}_spacing_ratio.*only valid"):
        MartiMuller3D(**{f"x{axis}_spacing_ratio": 0.98})


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


def test_mesh_config_reconstructs_independent_geometric_axes():
    mesh = MeshConfig(
        shape=(4, 5, 6),
        bounds_min=(-1.0, 2.0, 10.0),
        bounds_max=(1.0, 5.0, 16.0),
        halo_radius=2,
        spacing_types=(CellSpacing.GEOMETRIC,) * 3,
        spacing_ratios=(1.1, 0.9, 1.05),
    )

    for vertices, ratio in zip(
        (mesh.x1v, mesh.x2v, mesh.x3v),
        mesh.spacing_ratios,
        strict=True,
    ):
        widths = vertices[1:] - vertices[:-1]
        assert widths[1:] / widths[:-1] == pytest.approx(ratio)


@pytest.mark.parametrize(("axis", "ratio"), [(2, 1.04), (3, 0.96)])
def test_transverse_geometric_axis_evolves_and_round_trips(
    axis, ratio, tmp_path: Path
):
    output = tmp_path / f"x{axis}"
    problem = MartiMuller3D(
        resolution=(8, 6, 4),
        data_directory=output,
        **{
            f"x{axis}_spacing": CellSpacing.GEOMETRIC,
            f"x{axis}_spacing_ratio": ratio,
        },
    )

    run(problem, compute_mode="cpu", max_steps=1)

    checkpoint = next(output.glob("*.final.h5"))
    storage_slot = 3 - axis
    with h5py.File(checkpoint) as handle:
        geometry = handle[f"level_0/mesh/geometry/dim_{storage_slot}"]
        assert geometry.attrs["type"] == "geometric"
        assert geometry.attrs["ratio"] == pytest.approx(ratio)
        assert geometry.attrs["start"] == pytest.approx(0.0)
        assert geometry.attrs["end"] == pytest.approx(1.0)

    metadata, shape = load_checkpoint_metadata(checkpoint)
    restored = metadata_to_config_dict(metadata, shape)
    assert restored[f"x{axis}_spacing"] == CellSpacing.GEOMETRIC
    assert restored[f"x{axis}_spacing_ratio"] == pytest.approx(ratio)
