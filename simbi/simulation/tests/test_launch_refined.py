# =============================================================================
# test_launch_refined.py
#
# mesh refinement through `simbi launch`: the refined blast, a root grid with
# one fine box, launched across workers whose cuts cross the box, against the
# single-process refined run, both levels compared bit for bit; resumed onto a
# changed partition from the launch's own two-level checkpoint; and the
# refined combinations outside the gated scope refused by name: a periodic
# boundary, and a patch face on a cut, whose fine ghosts would read the root
# halo past its depth.
#
# run: pytest -m simulation simbi/simulation/tests/test_launch_refined.py
# =============================================================================

from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

from simbi.simulation.tests.test_launch import (
    EXIT,
    _layout,
    bounded_processes,  # noqa: F401  the autouse process guard
)
from simbi.simulation.tests.test_launch_isothermal import _script_launch, _script_reference

pytestmark = pytest.mark.simulation

BLAST = str(Path(RefinedBlast.__module__.replace(".", "/") + ".py"))
MODULE = RefinedBlast.__module__
# the fine box covers root cells 8..24 on both axes of the 32 x 32 root
RESOLUTION = (32, 32, 1)
STEPS = 8


def _levels(f: h5py.File) -> dict[str, np.ndarray]:
    """every cell dataset of both levels trimmed to its level's interior"""
    out = {}
    for level_name in ("level_0", "level_1"):
        assert level_name in f, f"the checkpoint carries no {level_name}"
        level = f[level_name]
        ng = int(level["mesh"].attrs["halo_width"])
        names: list[str] = []
        level.visit(lambda name, names=names, level=level: names.append(name) if isinstance(level[name], h5py.Dataset) else None)
        for name in names:
            if name.startswith("mesh/") or name.startswith("partition_0/owned") or "/domain/" in name:
                continue
            data = level[name][()]
            if data.ndim >= 2:
                out[f"{level_name}/{name}"] = data[tuple(slice(ng, -ng) for _ in range(data.ndim))]
    return out


def _assert_same_levels(a: Path, b: Path, label: str) -> None:
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for level in ("level_0", "level_1"):
            for key in ("time", "iteration"):
                assert fa[level].attrs[key] == fb[level].attrs[key], f"{label}: {level} attr {key}"
        xs, ys = _levels(fa), _levels(fb)
        assert xs.keys() == ys.keys(), f"{label}: dataset sets differ"
        for name, x in xs.items():
            np.testing.assert_array_equal(x, ys[name], err_msg=f"{label}: {name}")


def test_a_refined_launch_matches_the_single_process_run(tmp_path: Path) -> None:
    reference = _script_reference(MODULE, "RefinedBlast", RESOLUTION, tmp_path / "single", STEPS)
    # a cut through the box's center, then ragged cuts on both axes
    for tag, workers, kwargs in [("two", 2, {"shape": (2, 1)}), ("four", 4, {"cuts": [[13], [18]]})]:
        layout = _layout(tmp_path / f"{tag}.toml", workers=workers, **kwargs)
        final = _script_launch(BLAST, RESOLUTION, tmp_path / tag, layout, STEPS, workers)
        _assert_same_levels(final, reference, f"{workers} workers")


def test_a_refined_launch_resumed_onto_another_partition_matches(tmp_path: Path) -> None:
    reference = _script_reference(MODULE, "RefinedBlast", RESOLUTION, tmp_path / "single", STEPS)
    first = _script_launch(BLAST, RESOLUTION, tmp_path / "first", _layout(tmp_path / "two.toml", workers=2, shape=(2, 1)), STEPS // 2, 2)
    with h5py.File(first, "r") as f:
        written_at = int(f["level_0"].attrs["iteration"])
    assert 0 < written_at < STEPS
    resumed = _script_launch(
        BLAST, RESOLUTION, tmp_path / "four", _layout(tmp_path / "four.toml", workers=4, cuts=[[13], [18]]),
        STEPS - written_at, 4, checkpoint=first,
    )
    with h5py.File(resumed, "r") as f:
        assert int(f["level_0"].attrs["iteration"]) == STEPS
    _assert_same_levels(resumed, reference, "resumed on four workers")


def _refused(tmp_path: Path, script: str, resolution: tuple[int, ...] | None, layout: Path) -> str:
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "simbi.cli", "launch", script, "--layout", str(layout), "--max-steps", "1", "--data-directory", str(out), "--checkpoint-interval", "1e9"]
    if resolution is not None:
        cmd += ["--resolution", ",".join(str(r) for r in resolution)]
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert done.returncode != 0, f"the launch succeeded:\n{done.stdout}"
    assert not (out.exists() and list(out.glob("*.h5"))), "a checkpoint was published"
    assert EXIT.findall(done.stdout) or done.stderr, "the launch reported nothing"
    return " ".join(done.stderr.split())


def test_a_patch_face_on_a_cut_is_refused_by_name(tmp_path: Path) -> None:
    # the box's low face on axis 0 at root cell 8, on the cut: its fine ghosts prolong from the
    # root halo, which plm's two ghost cells do not hold
    stderr = _refused(tmp_path, BLAST, RESOLUTION, _layout(tmp_path / "two.toml", workers=2, cuts=[[8], []]))
    assert "lies on a decomposition cut" in stderr, stderr
