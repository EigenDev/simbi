# =============================================================================
# checkpoint_compare.py
#
# whole-file comparison of two checkpoints for the distributed parity gates.
# every dataset of every group is compared bit for bit, halos and staggered
# faces included, together with each level's clock (time, iteration). a
# caller leaves a group out only by naming it in `exclude`, with the reason at
# the call site. the comparison fails when the reference holds no dataset,
# when a dataset the caller expects is absent, and when the two files hold
# different dataset sets.
#
# usage:
#   assert_same_checkpoint(got, want, "two tiles", exclude=("metadata",), expect=("level_0/iso_cs2",))
# =============================================================================

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np


def _excluded(name: str, exclude: Iterable[str]) -> bool:
    return any(name == e or name.startswith(e.rstrip("/") + "/") for e in exclude)


def checkpoint_datasets(path: Path, exclude: Iterable[str] = ()) -> dict[str, np.ndarray]:
    """every dataset of the file by its path, and each level's clock as `level_k@time` and
    `level_k@iteration`; groups and datasets named in `exclude` (path prefixes) left out"""
    exclude = tuple(exclude)
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        def visit(name: str, obj) -> None:
            if isinstance(obj, h5py.Dataset) and not _excluded(name, exclude):
                out[name] = np.asarray(obj[()])

        f.visititems(visit)
        for level in (k for k in f if k.startswith("level_")):
            for clock in ("time", "iteration"):
                if clock in f[level].attrs and not _excluded(f"{level}@{clock}", exclude):
                    out[f"{level}@{clock}"] = np.asarray(f[level].attrs[clock])
    return out


def assert_same_checkpoint(
    got: Path,
    want: Path,
    label: str,
    exclude: Iterable[str] = (),
    expect: Iterable[str] = (),
) -> None:
    """the two checkpoints hold the same datasets, bit for bit, outside the declared exclusions;
    `expect` names datasets or groups the reference must carry for the comparison to mean
    anything"""
    exclude = tuple(exclude)
    xs, ys = checkpoint_datasets(got, exclude), checkpoint_datasets(want, exclude)
    assert ys, f"{label}: the reference holds no dataset outside {exclude}; the comparison is vacuous"
    missing = [e for e in expect if not any(k == e or k.startswith(e.rstrip("/") + "/") for k in ys)]
    assert not missing, f"{label}: the reference lacks {missing}; the comparison would not exercise them"
    assert xs.keys() == ys.keys(), f"{label}: datasets {sorted(set(xs) ^ set(ys))} differ"
    for name, x in xs.items():
        y = ys[name]
        assert x.shape == y.shape, f"{label}: {name} shape {x.shape} against {y.shape}"
        assert x.dtype == y.dtype, f"{label}: {name} dtype {x.dtype} against {y.dtype}"
        # bit for bit: the bytes of numeric data, so a sign of zero or a nan payload counts
        if x.dtype.kind in "fiu":
            same = np.ascontiguousarray(x).reshape(-1).view(np.uint8).tobytes() == np.ascontiguousarray(y).reshape(-1).view(np.uint8).tobytes()
        else:
            same = np.array_equal(x, y)
        assert same, f"{label}: {name} differs"
