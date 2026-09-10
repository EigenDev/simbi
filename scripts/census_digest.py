# =============================================================================
# census_digest.py
#
# collapse a directory of checkpoints into one small file holding only their census
# groups. a census is ~0.02% of a checkpoint's bytes, so a record that costs hundreds of
# gigabytes to copy whole costs tens of megabytes as a digest.
#
# the digest mirrors the checkpoint layout exactly -- census/<name>/{axis0_edges, values,
# level, time, t_start, n_samples, dropped} -- so every reader that accepts a checkpoint
# accepts a digest, as a one-element list:
#
#   read_census_series([digest], "shells")
#
# rows are deduplicated on (level, time), which is the same key the series reader unions on.
# a run segment never clears its history, so consecutive checkpoints from one segment repeat
# every earlier row; concatenating them blindly would count a sample many times over and the
# duplicates are invisible afterwards, since a repeated sample looks exactly like a real one.
#
# --append merges into an existing digest, which is what makes a queue-limited campaign
# recoverable. a census history lives in process memory and starts empty on resume, so each
# job segment holds only its own samples; a segment killed by the wall clock writes them to
# a checkpoint named `interrupted`, and that name is fixed, so the next segment overwrites
# it. when a segment is shorter than the dump interval -- many hours of integration per dump
# is routine at depth -- every segment but the one that finally reaches a dump loses its
# record that way. digesting the interrupted checkpoint after each segment preserves all of
# them, and because the merge is idempotent under (level, time) it is safe on every relaunch.
#
# dependencies are h5py and numpy alone, so this runs against a bare module environment on a
# machine that has no simbi install.
#
# usage:
#  python census_digest.py RUN_DIR [RUN_DIR ...] --out digest.h5
#  python census_digest.py RUN_DIR --census shells --out shells.h5
#  python census_digest.py run/chkpt.interrupted.h5 --out archive.h5 --append
# =============================================================================
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

ROW_KEYS = ("values", "level", "time", "t_start", "n_samples", "dropped")


def census_groups(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        return list(handle["census"]) if "census" in handle else []


def read_rows(path: Path, name: str):
    """the census group's attributes, bin edges and row arrays, or None if this file lacks it.

    the attributes carry the schema -- value_names, axis names, the monoid's `op` -- so a
    digest that drops them is unreadable even though every number survives.
    """
    with h5py.File(path, "r") as handle:
        group = handle.get(f"census/{name}")
        if group is None:
            return None
        edges = {k: np.array(group[k][:]) for k in group if k.endswith("_edges")}
        return (dict(group.attrs), edges,
                {k: np.array(group[k][:]) for k in ROW_KEYS})


def merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """concatenate row arrays and keep one row per (level, time)."""
    joined = {k: np.concatenate([p[k] for p in parts], axis=0) for k in ROW_KEYS}
    _, keep = np.unique(
        np.stack([joined["level"].astype(np.int64),
                  np.round(joined["time"], 9).view(np.int64)], axis=1),
        axis=0, return_index=True,
    )
    keep = np.sort(keep)
    return {k: v[keep] for k, v in joined.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dirs", nargs="+", help="directories of checkpoints, or checkpoint files")
    p.add_argument("--census", default=None,
                   help="census name; every census present is digested when omitted")
    p.add_argument("--glob", default="*.h5")
    p.add_argument("--out", required=True)
    p.add_argument("--append", action="store_true",
                   help="merge into an existing digest rather than replacing it. safe to run "
                        "repeatedly: the merge is idempotent under (level, time)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files: list[Path] = []
    for entry in args.dirs:
        path = Path(entry)
        files.extend(sorted(path.glob(args.glob)) if path.is_dir() else [path])

    readable: list[Path] = []
    for path in files:
        try:
            h5py.File(path, "r").close()
            readable.append(path)
        except OSError as err:
            print(f"  skipped unreadable {path.name}: {str(err).splitlines()[0][:60]}")
    if not readable:
        raise SystemExit("no readable checkpoints")

    names = [args.census] if args.census else census_groups(readable[-1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # an existing digest is read in full before the output is opened for writing, so the
    # archive is a source like any checkpoint and h5py never holds the same path twice.
    carried: dict[str, tuple] = {}
    if args.append and out.exists():
        for name in set(names) | set(census_groups(out)):
            found = read_rows(out, name)
            if found is not None:
                carried[name] = found
        names = sorted(set(names) | set(carried))

    with h5py.File(out, "w") as handle:
        for name in names:
            attrs, edges, parts = None, None, []
            if name in carried:
                attrs, edges, rows = carried[name]
                parts.append(rows)
            for path in readable:
                found = read_rows(path, name)
                if found is None:
                    continue
                attrs, edges, rows = found
                parts.append(rows)
            if not parts:
                print(f"  census '{name}' present in no file; skipped")
                continue
            before = sum(p["time"].size for p in parts)
            rows = merge(parts)
            if args.append:
                print(f"  census '{name}': {before} rows in, {before - rows['time'].size} "
                      f"already held")
            group = handle.create_group(f"census/{name}")
            group.attrs.update(attrs)
            for key, value in edges.items():
                group.create_dataset(key, data=value)
            for key, value in rows.items():
                group.create_dataset(key, data=value, compression="gzip", compression_opts=4)
            span = rows["time"].max() - rows["time"].min()
            print(f"  census '{name}': {len(parts)} sources -> {rows['time'].size} rows, "
                  f"t = {rows['time'].min():.3f} to {rows['time'].max():.3f} ({span:.2f} t_B)")

    source = sum(p.stat().st_size for p in readable)
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB from "
          f"{source / 1e9:.1f} GB of checkpoints, {out.stat().st_size / source * 100:.4f}%)")


if __name__ == "__main__":
    main()
