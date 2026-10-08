# =============================================================================
# test_refined_periodic_face.py
#
# a refined patch reaching one periodic face of an axis without spanning it is
# refused by name before any output, on the single grid and on host tiles cut
# across the other axis: its fine level would wrap onto its own opposite side.
# the same patch kept clear of the face runs.
#
# run: pytest -m simulation simbi/simulation/tests/test_refined_periodic_face.py
# =============================================================================
from pathlib import Path
from typing import Annotated

import pytest

from simbi import ProblemParam
from simbi.simulation import runner
from simbi.types import BoundaryCondition
from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

pytestmark = pytest.mark.simulation

RESOLUTION = (16, 16, 1)


class PeriodicInY(RefinedBlast):
    boundary_conditions: Annotated[
        list[BoundaryCondition],
        ProblemParam([BoundaryCondition.OUTFLOW, BoundaryCondition.PERIODIC], description="x outflow, y periodic"),
    ]


def _run(out: Path, gpus: int, region: list[float]) -> None:
    problem = PeriodicInY(resolution=RESOLUTION, gpus=gpus, data_directory=out, refinement_regions=[region])
    runner.run(problem, compute_mode="cpu", max_steps=2)


@pytest.mark.parametrize("gpus", [1, 2])
def test_a_patch_reaching_one_periodic_face_is_refused_by_name(tmp_path: Path, gpus: int, monkeypatch) -> None:
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")
    # the patch's low y face is the domain's periodic face at y = -0.5
    with pytest.raises(Exception, match="reaches the periodic face of axis 1 on one side only"):
        _run(tmp_path / "touching", gpus, [-0.25, 0.25, -0.5, 0.0])
    assert not list((tmp_path / "touching").glob("*.h5")), "a refused run published a checkpoint"
    _run(tmp_path / "clear", gpus, [-0.25, 0.25, -0.375, 0.0])
    assert list((tmp_path / "clear").glob("*final*.h5")), "the patch clear of the face did not run"
