# =============================================================================
# diag.py
#
# cli subcommand for computing windowed time-averages of diagnostic scalars.
# reads diagnostic hdf5 files directly (fast path, no full checkpoint parse).
# prints a table of mean, std, and standard error for each window size.
#
# usage:
#   simbi diag data/.../diagnostics --field mdot --t-start 20 --windows 5 10 20
# =============================================================================
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from ..utils.formatter import HelpFormatter


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diag",
        help="compute windowed time-averages of diagnostic scalars",
        formatter_class=HelpFormatter,
        usage="simbi diag <path> [options]",
    )
    parser.add_argument(
        "path",
        type=str,
        help="diagnostics directory or glob pattern",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="mdot",
        help="diagnostic field (default: mdot)",
    )
    parser.add_argument(
        "--t-start",
        type=float,
        default=0.0,
        help="start time in orbits (default: 0)",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="end time in orbits (default: use all data)",
    )
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[5.0, 10.0, 20.0],
        help="window sizes in orbits (default: 5 10 20)",
    )
    parser.add_argument(
        "--normalize-by",
        type=float,
        default=None,
        help="divide all values by this number before computing stats",
    )
    parser.set_defaults(func=execute)


def _extract_vector_component(h5, nb, attr, idx):
    return [float(h5[f"bodies/body_{i}"].attrs[attr][idx]) for i in range(nb)]


_FIELD_EXTRACTORS = {
    "mdot": lambda h5, nb: [
        float(h5[f"bodies/body_{i}/accretion"].attrs["accretion_rate"])
        for i in range(nb)
    ],
    "maccr": lambda h5, nb: [
        float(h5[f"bodies/body_{i}/accretion"].attrs["total_accreted_mass"])
        for i in range(nb)
    ],
    "force_x": lambda h5, nb: _extract_vector_component(h5, nb, "force", 0),
    "force_y": lambda h5, nb: _extract_vector_component(h5, nb, "force", 1),
    "force_z": lambda h5, nb: _extract_vector_component(h5, nb, "force", 2),
    "torque_x": lambda h5, nb: _extract_vector_component(h5, nb, "torque", 0),
    "torque_y": lambda h5, nb: _extract_vector_component(h5, nb, "torque", 1),
    "torque_z": lambda h5, nb: _extract_vector_component(h5, nb, "torque", 2),
}


def _read_diagnostics(
    directory: Path, field: str
) -> tuple[np.ndarray, np.ndarray, Optional[float], int]:
    """read times and field values from diagnostic files.

    returns (times, values, orbital_period, n_bodies).
    values shape: (n_files, n_bodies) for per-body fields, (n_files,) otherwise.
    """
    from simbi.reader.checkpoint_utils import glob_checkpoints

    files = glob_checkpoints(str(directory))
    if not files:
        print(f"no diagnostic files found in {directory}", file=sys.stderr)
        sys.exit(1)

    extractor = _FIELD_EXTRACTORS.get(field)
    if extractor is None:
        print(f"unknown field: {field}. available: {list(_FIELD_EXTRACTORS)}", file=sys.stderr)
        sys.exit(1)

    times = []
    values = []
    orbital_period = None
    n_bodies = 0

    for f in files:
        with h5py.File(str(f), "r") as h5:
            t = float(h5["metadata"].attrs["time"])
            if t == 0.0:
                continue

            if not n_bodies and "bodies" in h5:
                n_bodies = int(h5["bodies"].attrs["count"])
                if orbital_period is None and "binary_params" in h5["bodies"]:
                    orbital_period = float(
                        h5["bodies/binary_params"].attrs["orbital_period"]
                    )

            times.append(t)
            values.append(extractor(h5, n_bodies))

    return np.array(times), np.array(values), orbital_period, n_bodies


def _windowed_stats(
    times: np.ndarray,
    values: np.ndarray,
    t_start: float,
    t_end: float,
    window_size: float,
) -> tuple[float, float, float, float, float, int]:
    """compute mean, std, standard error, and percentiles over non-overlapping windows.

    returns (grand_mean, window_std, standard_error, p10, p90, n_windows).
    """
    mask = (times >= t_start) & (times <= t_end)
    t_sel = times[mask]
    v_sel = values[mask]

    if len(v_sel) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0

    grand_mean = float(np.mean(v_sel))

    n_windows = max(1, int((t_end - t_start) / window_size))
    edges = np.linspace(t_start, t_end, n_windows + 1)

    window_means = []
    for ii in range(n_windows):
        w_mask = (t_sel >= edges[ii]) & (t_sel < edges[ii + 1])
        if np.any(w_mask):
            window_means.append(float(np.mean(v_sel[w_mask])))

    n_eff = len(window_means)
    if n_eff < 2:
        return grand_mean, 0.0, 0.0, grand_mean, grand_mean, n_eff

    wm = np.array(window_means)
    window_std = float(np.std(wm, ddof=1))
    standard_error = window_std / math.sqrt(n_eff)
    p10 = float(np.percentile(wm, 10))
    p90 = float(np.percentile(wm, 90))

    return grand_mean, window_std, standard_error, p10, p90, n_eff


def execute(args: argparse.Namespace, _: Optional[list] = None) -> None:
    directory = Path(args.path)
    field = args.field

    times, values, orbital_period, n_bodies = _read_diagnostics(directory, field)

    norm = getattr(args, "normalize_by", None)
    if norm is not None:
        values = values / norm

    if orbital_period is None or orbital_period <= 0:
        print("no orbital period detected. t-start/t-end/windows are in raw time units.")
        orbital_period_scale = 1.0
        time_unit = "time"
    else:
        orbital_period_scale = orbital_period
        time_unit = "orbits"

    times_scaled = times / orbital_period_scale
    t_start = args.t_start
    t_end = args.t_end if args.t_end is not None else float(times_scaled[-1])

    # total values (sum over bodies)
    if values.ndim == 2:
        total = values.sum(axis=1)
    else:
        total = values

    mask = (times_scaled >= t_start) & (times_scaled <= t_end)
    dt_median = float(np.median(np.diff(times[mask]))) if np.sum(mask) > 1 else 0.0

    norm_label = f" (normalized by {norm})" if norm is not None else ""
    print(f"field: {field} (total){norm_label}")
    print(f"t_start: {t_start:.2f} {time_unit}, t_end: {t_end:.2f} {time_unit}")
    print(f"samples: {int(np.sum(mask))}, dt_median: {dt_median:.6f}")
    if orbital_period is not None:
        print(f"orbital_period: {orbital_period:.6f}")
    print()

    header = (
        f"{'window [' + time_unit + ']':<16s} "
        f"{'<' + field + '>':<14s} "
        f"{'sigma':<14s} "
        f"{'sigma/sqrt(N)':<14s} "
        f"{'p10':<14s} "
        f"{'p90':<14s} "
        f"{'N_windows':<10s}"
    )
    print(header)
    print("-" * len(header))

    for w in sorted(args.windows):
        w_abs = w * orbital_period_scale
        mean, std, se, p10, p90, n_w = _windowed_stats(
            times, total, t_start * orbital_period_scale, t_end * orbital_period_scale, w_abs
        )
        print(
            f"{w:<16.1f} {mean:<14.4f} {std:<14.4f} {se:<14.4f} "
            f"{p10:<14.4f} {p90:<14.4f} {n_w:<10d}"
        )

    # per-body breakdown
    if values.ndim == 2 and n_bodies > 1:
        print()
        for jj in range(n_bodies):
            print(f"--- body {jj} ---")
            print(header)
            print("-" * len(header))
            for w in sorted(args.windows):
                w_abs = w * orbital_period_scale
                mean, std, se, p10, p90, n_w = _windowed_stats(
                    times, values[:, jj],
                    t_start * orbital_period_scale, t_end * orbital_period_scale, w_abs
                )
                print(
                    f"{w:<16.1f} {mean:<14.4f} {std:<14.4f} {se:<14.4f} "
                    f"{p10:<14.4f} {p90:<14.4f} {n_w:<10d}"
                )
