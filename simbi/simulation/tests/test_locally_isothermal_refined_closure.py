# =============================================================================
# test_locally_isothermal_refined_closure.py
#
# the sound speed a locally isothermal refined run holds on every level, checked
# against the profile it is built from rather than against another driver. the
# plane holds uniform density 1 at rest with p(x) = cs2(x) rho, cs2(x) = 0.3 +
# 0.4 x on [0, 1]^2 with outflow walls, and one fine box over [0.25, 0.75]^2:
# - the root holds cs2 = p / rho = cs2(x_c) at every interior cell, and each
#   ghost holds its nearest interior cell (the clamped continuation)
# - the fine level holds the prolongation of the root's cs2, which reproduces a
#   linear profile at the fine cell centers; each fine ghost holds its nearest
#   fine interior cell
# - cs2 is fixed in time, so every checkpoint of the run holds the same field, and
#   a run resumed from a checkpoint holds it too and ends where the uninterrupted
#   run ends
#
# run: pytest -m simulation simbi/simulation/tests/test_locally_isothermal_refined_closure.py
#      (SIMBI_PARITY_BACKEND selects the session's backend, parity_harness.py)
# =============================================================================
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation

STEPS = 6
A, B = 0.3, 0.4
# the prolonged fine value of a linear profile is the coarse value plus a slope times a quarter
# coarse width: three roundings of values below 1, so a few units of 2^-53 at most
LINEAR_ROUNDING = 4.0e-15

PLANE = f'''
from typing import Annotated

from simbi import ProblemParam
from simbi.types import BoundaryCondition, InitialStateType
from simbi_configs.examples.isothermal.refined_locally_iso import RefinedLocallyIso


class LinearClosure(RefinedLocallyIso):
    """the locally isothermal plane at rest with cs2(x) = {A} + {B} x and outflow walls"""

    resolution: Annotated[tuple, ProblemParam((32, 32, 1), description="grid")]
    boundary_conditions: Annotated[list[BoundaryCondition], ProblemParam([BoundaryCondition.OUTFLOW], description="outflow walls")]

    def initial_primitive_state(self) -> InitialStateType:
        def gas_state():
            ni, nj, nk = self.resolution
            for _kk in range(nk):
                for _jj in range(nj):
                    for ii in range(ni):
                        x = (ii + 0.5) / ni
                        yield (1.0, 0.0, 0.0, ({A} + {B} * x) * 1.0)

        return gas_state
'''


def _run(tmp_path: Path, out: Path, steps: int, checkpoint: Path | None = None) -> list[Path]:
    module = tmp_path / "linear_closure.py"
    module.write_text(PLANE)
    resume = f", checkpoint_file={str(checkpoint)!r}" if checkpoint is not None else ""
    snippet = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from simbi.simulation import runner\n"
        "from linear_closure import LinearClosure\n"
        f"p = LinearClosure(data_directory=sys.argv[1]{resume})\n"
        "p.checkpoint_interval = 1e-9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out), str(tmp_path)), f"locally isothermal plane to step {steps}")
    files = sorted(out.glob("*.h5"))
    assert any("final" in f.name for f in files), f"no final checkpoint in {out}"
    return files


def _level(path: Path, name: str) -> tuple[np.ndarray, int, np.ndarray]:
    """the level's cs2 with its halo, the halo width, and the x of every interior cell center"""
    with h5py.File(path, "r") as f:
        level = f[name]
        ng = int(level["mesh"].attrs["halo_width"])
        cs2 = np.asarray(level["iso_cs2"][()])
        # the mesh geometry lists a 2D grid's axes y first: x is dim_1 and the array's last axis
        dim = level["mesh/geometry/dim_1"].attrs
        start, end = float(dim["start"]), float(dim["end"])
    n = cs2.shape[-1] - 2 * ng
    x = start + (np.arange(n) + 0.5) * (end - start) / n
    return cs2, ng, x


def _nearest_interior(cs2: np.ndarray, ng: int) -> np.ndarray:
    """every cell's nearest interior cell, each axis clamped into the interior"""
    rows = np.clip(np.arange(cs2.shape[0]), ng, cs2.shape[0] - ng - 1)
    cols = np.clip(np.arange(cs2.shape[1]), ng, cs2.shape[1] - ng - 1)
    return cs2[np.ix_(rows, cols)]


def _assert_profile(path: Path) -> None:
    root, ng, x = _level(path, "level_0")
    interior = root[ng:-ng, ng:-ng]
    want = (A + B * x) * 1.0 / 1.0
    np.testing.assert_array_equal(interior, np.broadcast_to(want, interior.shape), err_msg=f"{path.name}: root interior cs2 against p/rho")
    np.testing.assert_array_equal(root, _nearest_interior(root, ng), err_msg=f"{path.name}: root ghosts against their nearest interior cell")

    fine, ng, x = _level(path, "level_1")
    interior = fine[ng:-ng, ng:-ng]
    want = np.broadcast_to(A + B * x, interior.shape)
    worst = np.abs(interior - want).max()
    assert worst <= LINEAR_ROUNDING, f"{path.name}: fine interior cs2 departs from the linear profile by {worst:e} (first row {interior[0]} against {want[0]})"
    np.testing.assert_array_equal(fine, _nearest_interior(fine, ng), err_msg=f"{path.name}: fine ghosts against their nearest interior cell")
    assert interior.max() - interior.min() > 0.1, f"{path.name}: the fine level holds no varying profile; the check is vacuous"


def test_every_level_holds_the_declared_profile_at_every_checkpoint(tmp_path: Path) -> None:
    files = _run(tmp_path, tmp_path / "run", STEPS)
    assert len(files) > STEPS, f"{len(files)} checkpoints; every step writes one"
    for path in files:
        _assert_profile(path)


def test_a_resumed_run_holds_the_profile_and_ends_where_the_whole_run_ends(tmp_path: Path) -> None:
    whole = _run(tmp_path, tmp_path / "whole", STEPS)
    first = _run(tmp_path, tmp_path / "first", STEPS // 2)
    half = [f for f in first if "final" in f.name][0]
    resumed = _run(tmp_path, tmp_path / "resumed", STEPS, checkpoint=half)
    for path in resumed:
        _assert_profile(path)
    final_whole = [f for f in whole if "final" in f.name][0]
    final_resumed = [f for f in resumed if "final" in f.name][0]
    with h5py.File(final_whole, "r") as fw, h5py.File(final_resumed, "r") as fr:
        assert int(fr["level_0"].attrs["iteration"]) == STEPS
        for level in ("level_0", "level_1"):
            np.testing.assert_array_equal(fr[f"{level}/iso_cs2"][()], fw[f"{level}/iso_cs2"][()], err_msg=f"resumed {level}/iso_cs2")
            for name in fw[f"{level}/conserved"]:
                np.testing.assert_array_equal(fr[f"{level}/conserved/{name}"][()], fw[f"{level}/conserved/{name}"][()],
                                              err_msg=f"resumed {level}/conserved/{name}")
    momentum = h5py.File(final_whole, "r")["level_1/conserved/m1"][()]
    assert np.abs(momentum).max() > 1e-6, "the cs2 gradient drives no flow on the fine level; the resume compares a state at rest"
