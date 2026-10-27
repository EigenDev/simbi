# =============================================================================
# test_isothermal_face_on_cut.py
#
# refined isothermal runs whose patch face lies on a decomposition cut, through
# the tiled and launched drivers on the session's backend and placement
# (parity_table, parity_harness), held to the exact contract against the single
# device on that backend. the fine ghosts beside the cut prolong their density
# and velocity from a parent stencil gathered across the cut; the sound speed
# cs^2(x) is fixed at construction on every cell a tile allocates (a locally
# isothermal closure sliced from the whole patch, a uniform one continued
# exactly), so the closure needs no transfer of its own. the table:
# - the drifting isothermal blob in the uniform closure, the fine box's low x
#   face on the cut, and its high x face on the cut
# - the locally isothermal plane, cs^2(x) = 0.5 + 0.3 sin(2 pi x) at uniform
#   density and rest, with outflow walls, the same two faces on the cut
# - the isothermal bondi point mass and drain under two nested boxes, the even
#   cut on level 1's low x face and the uneven cuts on both levels' low x faces
# each entry's premise asserts a refined patch face lies on the even cut, and its
# even launch moves parent stencil cells between workers; the
# locally isothermal entries also assert their varying cs^2 moves the fine
# level's gas, which starts at rest at uniform density.
#
# run: pytest -m simulation simbi/simulation/tests/test_isothermal_face_on_cut.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests.parity_table import REJECTED_STEP, Feature, install, launch
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard

pytestmark = pytest.mark.simulation

BLOB = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.isothermal.refined_isothermal_blob import RefinedIsothermalBlob


class {name}(RefinedIsothermalBlob):
    """the drifting isothermal blob on the unit square, its fine box {box}"""

    refinement_regions: Annotated[list[list[float]], ProblemParam([{box}], description="the fine box")]
'''
LOCAL = '''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import BoundaryCondition
from simbi_configs.examples.isothermal.refined_locally_iso import RefinedLocallyIso


class {name}(RefinedLocallyIso):
    """the locally isothermal plane with outflow walls, its fine box {box}"""

    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="outflow walls")]
    refinement_regions: Annotated[list[list[float]], ProblemParam([{box}], description="the fine box")]
'''
SINK = '''
from typing import Annotated

from simbi import ProblemParam
from simbi_configs.examples.newtonian.bondi import SphericalBondiTest


class IsothermalNestedSink(SphericalBondiTest):
    """the bondi point mass and its drain at rest at the origin of a 24-cell root on [-5, 5]^3 in the
    isothermal closure, under two nested fine boxes over root cells 6..18 and 9..15; no sponge"""

    adiabatic_index: Annotated[float, ProblemParam(1.0, description="isothermal")]
    base_resolution: Annotated[int, ProblemParam(24, description="base")]
    use_sponge: Annotated[bool, ProblemParam(False, description="no sponge source")]
    refinement_enabled: Annotated[bool, ProblemParam(True, description="refined")]
    refinement_max_levels: Annotated[int, ProblemParam(3, description="levels")]
    refinement_regions: Annotated[
        list[list[float]] | None,
        ProblemParam([[-2.5, 2.5, -2.5, 2.5, -2.5, 2.5], [-1.25, 1.25, -1.25, 1.25, -1.25, 1.25]], description="boxes"),
    ]
    refinement_ratios: Annotated[list[int] | None, ProblemParam([2, 2], description="ratios")]
'''
LOW_FACE = "[0.5, 0.75, 0.25, 0.75]"
HIGH_FACE = "[0.25, 0.5, 0.25, 0.75]"


def _face_on_cut(x: float, level: str = "level_1", axis: str = "dim_1"):
    """the premise that `level`'s low or high face along the cut axis lies at the cut x. the mesh
    geometry lists a 2D grid's axes y first, so x is `dim_1`; the 3D boxes are cubes"""

    def premise(mine: Path, _control: Path | None) -> None:
        with h5py.File(mine, "r") as f:
            assert level in f, f"the single device published no {level}"
            dim = f[level]["mesh/geometry"][axis].attrs
            lo, hi = float(dim["start"]), float(dim["end"])
        assert x in (lo, hi), f"{level} spans x in [{lo}, {hi}], neither face on the cut at {x}"

    return premise


def _moves_the_fine_gas(x: float):
    face = _face_on_cut(x)

    def premise(mine: Path, control: Path | None) -> None:
        face(mine, control)
        with h5py.File(mine, "r") as f:
            level = f["level_1"]
            ng = int(level["mesh"].attrs["halo_width"])
            cs2 = level["iso_cs2"][ng:-ng, ng:-ng]
            momentum = level["conserved/m1"][ng:-ng, ng:-ng]
        assert cs2.max() - cs2.min() > 0.1, f"cs^2 spans [{cs2.min()}, {cs2.max()}] on the fine level; the closure is uniform there"
        assert np.abs(momentum).max() > 1e-6, "the fine gas stays at rest; the varying cs^2 never drives the fine level"

    return premise


def _sink_premise(mine: Path, control: Path | None) -> None:
    _face_on_cut(-2.5, axis="dim_0")(mine, control)
    with h5py.File(mine, "r") as f:
        assert "level_2" in f, "the single device published no second refined level"
        assert np.all(np.abs(np.asarray(f["body_diagnostics/mass_delta"][()])) > 0.0), "the sink accreted nothing at some step"


BLOB_CUTS = {"resolution": (32, 32), "even": [[16], []], "rejection": f"reject:1:{REJECTED_STEP}:1"}
LOCAL_CUTS = {"resolution": (32, 32, 1), "even": [[16], []], "rejection": f"reject:1:{REJECTED_STEP}:1"}

FEATURES = {
    "uniform-low-face": Feature(BLOB.format(name="LowFace", box=LOW_FACE), "LowFace", uneven=[[16, 26], []], **BLOB_CUTS,
                                premise=_face_on_cut(0.5)),
    "uniform-high-face": Feature(BLOB.format(name="HighFace", box=HIGH_FACE), "HighFace", uneven=[[6, 16], []], **BLOB_CUTS,
                                 premise=_face_on_cut(0.5)),
    "local-low-face": Feature(LOCAL.format(name="LocalLowFace", box=LOW_FACE), "LocalLowFace", uneven=[[16, 26], []], **LOCAL_CUTS,
                              premise=_moves_the_fine_gas(0.5)),
    "local-high-face": Feature(LOCAL.format(name="LocalHighFace", box=HIGH_FACE), "LocalHighFace", uneven=[[6, 16], []], **LOCAL_CUTS,
                               premise=_moves_the_fine_gas(0.5)),
    # root cells 6 and 9 are level 1's and level 2's low x faces; worker 1 holds all of level 2 under
    # the even cut, and level 2's last substep of a root step under rk2 is substep 3
    "nested-sink": Feature(SINK, "IsothermalNestedSink", None, [[6], [], []], [[6, 9], [], []],
                           f"reject_level:1:{REJECTED_STEP}:2:3:1", premise=_sink_premise),
}


install(globals(), FEATURES)


@pytest.mark.parametrize("feature", list(FEATURES), indirect=True)
def test_the_launch_gathers_the_parent_stencil_across_the_cut(tmp_path: Path, feature) -> None:
    """the even cut's launch plans move parent prim stencil cells between workers, so the parity
    gates above compare fine ghosts prolonged from gathered cells"""
    f, script = feature
    run = launch(f, script, tmp_path / "launched", f.even, steps=1)
    remote = run.remote_sends()
    assert any(r.get("PrimStencil", 0) > 0 for r in remote.values()), f"the plans schedule no remote PrimStencil transfer ({remote})"
