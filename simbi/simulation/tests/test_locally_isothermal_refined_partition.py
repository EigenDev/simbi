# =============================================================================
# test_locally_isothermal_refined_partition.py
#
# the locally isothermal refined plane of test_locally_isothermal_refined_closure
# (cs2(x) = 0.3 + 0.4 x, uniform density 1 at rest, outflow walls) with its fine
# box's face on the device cut at x = 0.5, the box on either side of the cut, on
# two host tiles and two launched workers:
# - every published level of every checkpoint holds the declared profile: the
#   root p / rho, the fine level the linear profile its prolongation reproduces,
#   every published ghost its nearest interior cell
# - every checkpoint equals the single device's, its time step included; the
#   first difference names the checkpoint and the dataset (iso_cs2 for the
#   closure, a conserved or primitive field for the state, dt for the CFL input)
# - a launch resumed onto three uneven workers holds the profile and ends where
#   the uninterrupted single device ends
#
# run: pytest -m simulation simbi/simulation/tests/test_locally_isothermal_refined_partition.py
#      (SIMBI_PARITY_BACKEND / SIMBI_PARITY_PLACEMENT select the session, parity_harness.py)
# =============================================================================
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph
from simbi.simulation.tests.test_launch import bounded_processes  # noqa: F401  the autouse process guard
from simbi.simulation.tests.test_locally_isothermal_refined_closure import PLANE, STEPS, _assert_profile

pytestmark = pytest.mark.simulation

BOXES = {"low-face": "[[0.5, 0.75, 0.25, 0.75]]", "high-face": "[[0.25, 0.5, 0.25, 0.75]]"}
CUT = [[16], []]
UNEVEN = [[10, 16], []]


def _module(tmp_path: Path, side: str) -> Path:
    """the box problem alone in its module (a launch evolves the one problem a module defines),
    built on the plane of the closure test"""
    (tmp_path / "linear_closure.py").write_text(PLANE)
    path = tmp_path / "linear_closure_box.py"
    path.write_text(
        "from typing import Annotated\n\nfrom simbi import ProblemParam\nfrom linear_closure import LinearClosure\n\n\n"
        "class Box(LinearClosure):\n"
        f"    refinement_regions: Annotated[list[list[float]], ProblemParam({BOXES[side]}, description='the fine box')]\n"
    )
    return path


def _run(module: Path, out: Path, gpus: int = 1, decompose=None, steps: int = STEPS, checkpoint: Path | None = None) -> dict[str, Path]:
    fields = f", gpus={gpus}" + (f", decompose={decompose!r}" if decompose else "")
    if checkpoint is not None:
        fields += f", checkpoint_file={str(checkpoint)!r}"
    snippet = (
        "import sys\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "from simbi.simulation import runner\n"
        f"from {module.stem} import Box\n"
        f"p = Box(data_directory=sys.argv[1]{fields})\n"
        "p.checkpoint_interval = 1e-9\n"
        f"runner.run(p, compute_mode={ph.BACKEND!r}, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out), str(module.parent)), f"{module.stem} on {gpus} device(s)", devices=gpus)
    return {p.name: p for p in out.glob("*.h5")}


def _launch(module: Path, out: Path, cuts: list, steps: int = STEPS, checkpoint: Path | None = None) -> dict[str, Path]:
    args = ["--max-steps", str(steps), "--checkpoint-interval", "1e-9"]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    workers = int(np.prod([len(c) + 1 for c in cuts]))
    env = ph.environment()
    env["PYTHONPATH"] = os.pathsep.join([str(module.parent), env.get("PYTHONPATH", "")])
    ph.launch(module, out, workers=workers, cuts=cuts, args=args, env=env)
    return {p.name: p for p in out.glob("*.h5")}


def _datasets(path: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        for level in sorted(k for k in f if k.startswith("level_")):
            out[f"{level}/dt"] = np.asarray(f[level].attrs.get("dt", np.nan))
            f[level].visititems(lambda name, obj, lv=level: out.__setitem__(f"{lv}/{name}", obj[()]) if isinstance(obj, h5py.Dataset) else None)
    return out


def _first_difference(got: dict[str, Path], want: dict[str, Path]) -> str | None:
    if sorted(got) != sorted(want):
        return f"checkpoints {sorted(got)} against {sorted(want)}"
    for name in sorted(want, key=lambda n: (n.endswith("final.h5"), n)):
        a, b = _datasets(got[name]), _datasets(want[name])
        if a.keys() != b.keys():
            return f"{name}: datasets {sorted(a.keys() ^ b.keys())}"
        for key in sorted(b, key=lambda k: (not k.endswith("iso_cs2"), not k.endswith("/dt"), k)):
            if not np.array_equal(a[key], b[key]):
                cells = np.argwhere(np.asarray(a[key]) != np.asarray(b[key]))
                return f"{name}: {key} differs at {len(cells)} cell(s), first {cells[:4].tolist()}"
    return None


@pytest.mark.parametrize("side", list(BOXES))
@pytest.mark.parametrize("driver", ["host tiles", "launched workers"])
def test_a_cut_patch_face_holds_the_profile_and_the_single_device_run(tmp_path: Path, side: str, driver: str) -> None:
    module = _module(tmp_path, side)
    single = _run(module, tmp_path / "single")
    for path in single.values():
        _assert_profile(path)
    got = _run(module, tmp_path / "tiles", gpus=2, decompose=CUT) if driver == "host tiles" else _launch(module, tmp_path / "launched", CUT)
    for path in got.values():
        _assert_profile(path)
    difference = _first_difference(got, single)
    assert difference is None, f"{driver}, box {side}: {difference}"


@pytest.mark.parametrize("side", list(BOXES))
def test_a_resumed_cut_patch_face_holds_the_profile_and_ends_on_the_single_device(tmp_path: Path, side: str) -> None:
    module = _module(tmp_path, side)
    whole = _run(module, tmp_path / "single")
    first = _launch(module, tmp_path / "first", CUT, steps=STEPS // 2)
    half = [p for n, p in first.items() if "final" in n][0]
    resumed = _launch(module, tmp_path / "resumed", UNEVEN, checkpoint=half)
    for path in resumed.values():
        _assert_profile(path)
    final = lambda run: {"final": [p for n, p in run.items() if "final" in n][0]}
    difference = _first_difference(final(resumed), final(whole))
    assert difference is None, f"resumed onto three uneven workers, box {side}: {difference}"
