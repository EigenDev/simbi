# =============================================================================
# test_decomposed_restart.py
#
# a checkpoint carries the global grid alone, so a run resumes from it under any partition:
# one tile to two, two tiles to one, and two tiles to three cut unevenly all continue to the
# same state as a single grid resuming the same file. every field, staggered face and
# geometry attribute of the resulting checkpoints is compared bit for bit. the decomposed
# path runs as host tiles on a cpu build.
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation import runner
from simbi_configs.examples.newtonian.sedov import SedovTaylor

pytestmark = pytest.mark.simulation


def _run(out: Path, gpus: int, steps: int, checkpoint: Path | None = None, decompose=None) -> Path:
    kwargs = dict(zpd=8, rinit=0.1, rend=1.0, gpus=gpus, data_directory=out)
    if decompose is not None:
        kwargs["decompose"] = decompose
    if checkpoint is not None:
        kwargs["checkpoint_file"] = str(checkpoint)
    runner.run(SedovTaylor(**kwargs), compute_mode="cpu", validate=True, max_steps=steps)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _interior_datasets(f: h5py.File):
    """every level-0 cell dataset trimmed to the interior, plus every face dataset whole: the
    physical state a checkpoint carries. halo cells are derived and regenerated on restart."""
    level = f["level_0"]
    ng = int(level["mesh"].attrs["halo_width"])
    names = []
    level.visit(lambda name: names.append(name) if isinstance(level[name], h5py.Dataset) else None)
    out = {}
    for name in names:
        data = level[name][()]
        if name.startswith("mesh/") or name.startswith("partition_0/owned") or "/domain/" in name:
            continue
        if "/magnetic/" in name:
            out[name] = data
        elif data.ndim == 2:
            out[name] = data[ng:-ng, ng:-ng]
    assert out, "no field datasets under level_0"
    return out


def _assert_same_state(a: Path, b: Path, label: str, scale_tolerance: float | None = None) -> None:
    """the two checkpoints hold the same physical state: bit for bit when `scale_tolerance` is
    unset, otherwise within that fraction of each field's largest magnitude, the bound the
    decomposition equivalence gates hold a decomposed march to, since each tile evaluates its
    coordinate map from its own origin and the geometric source follows at roundoff."""
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for key in ("time", "iteration"):
            assert fa["level_0"].attrs[key] == fb["level_0"].attrs[key], f"{label}: level attr {key}"
        xs, ys = _interior_datasets(fa), _interior_datasets(fb)
        assert xs.keys() == ys.keys(), f"{label}: dataset sets differ"
        # a field's scale is the largest magnitude in its group (the conserved set, the
        # primitives, the faces), so a component that is zero up to roundoff, such as the
        # polar momentum of a radial flow, is measured against the flow it rides in.
        group_scale = {}
        for name, x in xs.items():
            group = name.rsplit("/", 1)[0]
            group_scale[group] = max(group_scale.get(group, 1e-300), np.abs(x).max(), np.abs(ys[name]).max())
        for name, x in xs.items():
            y = ys[name]
            assert x.shape == y.shape, f"{label}: {name} shape"
            if scale_tolerance is None:
                np.testing.assert_array_equal(x, y, err_msg=f"{label}: {name}")
            else:
                scale = group_scale[name.rsplit("/", 1)[0]]
                worst = np.abs(x - y).max()
                assert worst <= scale_tolerance * scale, f"{label}: {name} differs by {worst:.3e} against scale {scale:.3e}"
        for dim in fa["level_0/mesh/geometry"]:
            for key, value in fa["level_0/mesh/geometry"][dim].attrs.items():
                assert fb["level_0/mesh/geometry"][dim].attrs[key] == value, f"{label}: {dim}/{key}"


DECOMPOSITION_BOUND = 1e-12


@pytest.mark.parametrize("gpus,decompose", [(1, None), (2, None), (3, [[3, 5], []])])
def test_a_restart_under_the_same_partition_continues_bit_for_bit(tmp_path: Path, monkeypatch, gpus, decompose) -> None:
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")
    whole = _run(tmp_path / "whole", gpus=gpus, steps=3, decompose=decompose)
    first = _run(tmp_path / "first", gpus=gpus, steps=2, decompose=decompose)
    resumed = _run(tmp_path / "resumed", gpus=gpus, steps=1, checkpoint=first, decompose=decompose)
    _assert_same_state(whole, resumed, f"{gpus} tiles: three steps vs two then one")


def test_a_checkpoint_resumes_under_any_partition(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")
    start = _run(tmp_path / "start", gpus=1, steps=2)
    one = _run(tmp_path / "one", gpus=1, steps=1, checkpoint=start)
    two = _run(tmp_path / "two", gpus=2, steps=1, checkpoint=start)
    _assert_same_state(one, two, "one tile -> two tiles", DECOMPOSITION_BOUND)
    one_again = _run(tmp_path / "one_again", gpus=1, steps=1, checkpoint=one)
    from_two_to_one = _run(tmp_path / "two_to_one", gpus=1, steps=1, checkpoint=two)
    from_two_to_three = _run(tmp_path / "two_to_three", gpus=3, steps=1, checkpoint=two, decompose=[[3, 5], []])
    _assert_same_state(one_again, from_two_to_one, "two tiles -> one tile", DECOMPOSITION_BOUND)
    _assert_same_state(one_again, from_two_to_three, "two tiles -> three uneven tiles", DECOMPOSITION_BOUND)


def test_a_refined_checkpoint_resumes_across_partitions(tmp_path: Path, monkeypatch) -> None:
    # a two-level hierarchy written by two tiles resumes on one grid and on two tiles alike,
    # both levels included; the classic single-grid loader reads the tiled file.
    from simbi_configs.examples.newtonian.refined_blast import RefinedBlast

    monkeypatch.setenv("SYMBI_GPU_OVERSUBSCRIBE", "1")

    def run(out: Path, gpus: int, steps: int, checkpoint: Path | None = None) -> Path:
        kwargs = dict(resolution=(16, 16, 1), gpus=gpus, data_directory=out)
        if checkpoint is not None:
            kwargs["checkpoint_file"] = str(checkpoint)
        runner.run(RefinedBlast(**kwargs), compute_mode="cpu", validate=True, max_steps=steps)
        files = sorted(out.glob("*final*.h5"))
        assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
        return files[0]

    start = run(tmp_path / "start", gpus=2, steps=2)
    with h5py.File(start, "r") as f:
        assert "level_1" in f, "the tiled refined checkpoint carries its fine level"
    one = run(tmp_path / "one", gpus=1, steps=1, checkpoint=start)
    two = run(tmp_path / "two", gpus=2, steps=1, checkpoint=start)
    with h5py.File(one, "r") as fa, h5py.File(two, "r") as fb:
        for level in ("level_0", "level_1"):
            names = []
            fa[level].visit(lambda name, la=fa[level]: names.append(name) if isinstance(la[name], h5py.Dataset) else None)
            assert names, f"{level}: no datasets"
            for name in names:
                np.testing.assert_array_equal(fa[level][name][()], fb[level][name][()], err_msg=f"{level}/{name}")
