# =============================================================================
# test_decomposed_restart.py
#
# a checkpoint carries the global grid alone, so a run resumes from it under any partition:
# one tile to two, two tiles to one, and two tiles to three cut unevenly all continue to the
# same state as a single grid resuming the same file. every field, staggered face and
# geometry attribute of the resulting checkpoints is compared bit for bit. every run is a
# fresh process on the session's backend (parity_harness), the decomposed ones as host
# tiles; host tiles alone involve no launch, so every gate runs under the local placement.
#
# run: pytest -m simulation simbi/simulation/tests/test_decomposed_restart.py
#      (SIMBI_PARITY_BACKEND selects the session, parity_harness.py)
# =============================================================================
from pathlib import Path

import h5py
import numpy as np
import pytest

from simbi.simulation.tests import parity_harness as ph

pytestmark = pytest.mark.simulation


def _fresh(problem: str, fields: list[str], out: Path, gpus: int, steps: int) -> Path:
    """`problem` (module:class under simbi_configs.examples) built from `fields` on the session's
    backend over `gpus` devices, capped at the absolute root iteration `steps`; its final
    checkpoint"""
    module, name = problem.split(":")
    snippet = (
        "import sys\n"
        "from simbi.simulation import runner\n"
        f"from simbi_configs.examples.{module} import {name}\n"
        f"p = {name}(gpus={gpus}, data_directory=sys.argv[1]{''.join(', ' + f for f in fields)})\n"
        f"runner.run(p, compute_mode=sys.argv[2], validate=True, max_steps={steps})\n"
    )
    ph.python(ph.snippet_args(snippet, str(out), ph.BACKEND), f"{name} on {gpus} device(s) to step {steps}", devices=gpus)
    files = sorted(out.glob("*final*.h5"))
    assert len(files) == 1, f"expected one final checkpoint in {out}, found {files}"
    return files[0]


def _run(out: Path, gpus: int, steps: int, checkpoint: Path | None = None, decompose=None) -> Path:
    fields = ["zpd=8", "rinit=0.1", "rend=1.0"]
    if decompose is not None:
        fields.append(f"decompose={decompose!r}")
    if checkpoint is not None:
        fields.append(f"checkpoint_file={str(checkpoint)!r}")
    return _fresh("newtonian.sedov:SedovTaylor", fields, out, gpus, steps)


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
    assert out, "level_0 holds no dataset of fields"
    return out


def _assert_same_state(a: Path, b: Path, label: str) -> None:
    """the two checkpoints hold the same state bit for bit: every tile evaluates the whole grid's
    coordinates at the whole-grid index of its cells, so a repartitioned march reproduces the
    single grid's arithmetic exactly."""
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for key in ("time", "iteration"):
            assert fa["level_0"].attrs[key] == fb["level_0"].attrs[key], f"{label}: level attr {key}"
        xs, ys = _interior_datasets(fa), _interior_datasets(fb)
        assert xs.keys() == ys.keys(), f"{label}: dataset sets differ"
        for name, x in xs.items():
            y = ys[name]
            assert x.shape == y.shape, f"{label}: {name} shape"
            np.testing.assert_array_equal(x, y, err_msg=f"{label}: {name}")
        for dim in fa["level_0/mesh/geometry"]:
            for key, value in fa["level_0/mesh/geometry"][dim].attrs.items():
                assert fb["level_0/mesh/geometry"][dim].attrs[key] == value, f"{label}: {dim}/{key}"


def _iteration(path: Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["level_0"].attrs["iteration"])


# `max_steps` caps the root iteration itself, a resumed run's checkpointed iterations included:
# resumed at the cap a run takes no step and leaves the state as the checkpoint holds it, and
# resumed with a cap one higher it takes exactly one step
@ph.local_only
@pytest.mark.parametrize("gpus", [1, 2])
def test_max_steps_caps_the_root_iteration_across_a_resume(tmp_path: Path, gpus) -> None:
    first = _run(tmp_path / "first", gpus=gpus, steps=2)
    assert _iteration(first) == 2, "premise: the first run stops at iteration two"
    at_cap = _run(tmp_path / "at_cap", gpus=gpus, steps=2, checkpoint=first)
    assert _iteration(at_cap) == 2, f"{gpus} tile(s): a run resumed at its cap took a step"
    ph.disagreement(_assert_same_state, first, at_cap, f"{gpus} tile(s): resumed at the cap")
    past = _run(tmp_path / "past", gpus=gpus, steps=3, checkpoint=first)
    assert _iteration(past) == 3, f"{gpus} tile(s): resumed with cap three, stopped at {_iteration(past)}"


@ph.local_only
@pytest.mark.parametrize("gpus,decompose", [(1, None), (2, None), (3, [[3, 5], []])])
def test_a_restart_under_the_same_partition_continues_bit_for_bit(tmp_path: Path, gpus, decompose) -> None:
    whole = _run(tmp_path / "whole", gpus=gpus, steps=3, decompose=decompose)
    first = _run(tmp_path / "first", gpus=gpus, steps=2, decompose=decompose)
    # the cap is the absolute root iteration: the checkpoint's two and one more
    resumed = _run(tmp_path / "resumed", gpus=gpus, steps=3, checkpoint=first, decompose=decompose)
    ph.disagreement(_assert_same_state, whole, resumed, f"{gpus} tiles: three steps vs two then one")


@ph.local_only
def test_a_checkpoint_resumes_under_any_partition(tmp_path: Path) -> None:
    # each resume caps the absolute root iteration one past its checkpoint's
    start = _run(tmp_path / "start", gpus=1, steps=2)
    one = _run(tmp_path / "one", gpus=1, steps=3, checkpoint=start)
    two = _run(tmp_path / "two", gpus=2, steps=3, checkpoint=start)
    ph.disagreement(_assert_same_state, one, two, "one tile -> two tiles")
    one_again = _run(tmp_path / "one_again", gpus=1, steps=4, checkpoint=one)
    from_two_to_one = _run(tmp_path / "two_to_one", gpus=1, steps=4, checkpoint=two)
    from_two_to_three = _run(tmp_path / "two_to_three", gpus=3, steps=4, checkpoint=two, decompose=[[3, 5], []])
    ph.disagreement(_assert_same_state, one_again, from_two_to_one, "two tiles -> one tile")
    ph.disagreement(_assert_same_state, one_again, from_two_to_three, "two tiles -> three uneven tiles")


def _refined(out: Path, gpus: int, steps: int, checkpoint: Path | None = None) -> Path:
    fields = ["resolution=(16, 16, 1)"] + ([f"checkpoint_file={str(checkpoint)!r}"] if checkpoint is not None else [])
    return _fresh("newtonian.refined_blast:RefinedBlast", fields, out, gpus, steps)


def _assert_same_levels(a: Path, b: Path) -> None:
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        for level in ("level_0", "level_1"):
            names = []
            fa[level].visit(lambda name, la=fa[level]: names.append(name) if isinstance(la[name], h5py.Dataset) else None)
            assert names, f"{level} holds no dataset"
            for name in names:
                np.testing.assert_array_equal(fa[level][name][()], fb[level][name][()], err_msg=f"{level}/{name}")


@ph.local_only
def test_a_refined_checkpoint_resumes_across_partitions(tmp_path: Path) -> None:
    # a two-level hierarchy written by two tiles resumes on one grid and on two tiles alike,
    # both levels included; the classic single-grid loader reads the tiled file.
    start = _refined(tmp_path / "start", gpus=2, steps=2)
    with h5py.File(start, "r") as f:
        assert "level_1" in f, "the tiled refined checkpoint carries its fine level"
    # the cap is the absolute root iteration: the checkpoint's two and one more
    one = _refined(tmp_path / "one", gpus=1, steps=3, checkpoint=start)
    two = _refined(tmp_path / "two", gpus=2, steps=3, checkpoint=start)
    ph.disagreement(_assert_same_levels, one, two)
