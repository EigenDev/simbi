# =============================================================================
# anchor_capture.py
#
# the producer for the projection-anchor A/B study: runs one arm of the
# Fishbone-Moncrief torus in-process, reads the anchor-experiment receipts and
# the guard census from the backend statics (which live only for the process
# that ran the sim), extracts the conserved totals and the physical observables
# from the run's checkpoints, and writes one record for `anchor_ab`.
#
# the anchor receipts are process-global statics zeroed at run start, so the
# capture must happen in the same process that ran the simulation — hence the
# in-process run (`SimbiParser`) rather than a subprocess whose statics would
# vanish at exit.
#
# the conserved totals are the densitized Valencia integrals on the horizon-
# penetrating Schwarzschild-Kerr-Schild chart (`schwarzschild_ks`): the rest
# mass `int sqrt(gamma) W rho dV` and the energy `int sqrt(gamma) tau dV`, with
# the Lorentz factor and every dot product contracted with the spatial metric
# `gamma_ij = diag(1 + 2M/r, r^2, r^2 sin^2 theta)`. computing them in the
# curved metric (rather than a flat-frame proxy) keeps mass and energy on one
# convention, and the same extractor runs on both arms so any systematic factor
# cancels in the cross-arm comparison.
#
# supported chart: schwarzschild_ks (the FM torus at zero spin). the reader's
# validated horizon-accretion helper is defined there; the spinning-Kerr chart
# needs the reader's metric machinery extended first.
#
# usage:
#   python -m simbi.analysis.anchor_capture \
#       --config simbi_configs/examples/grmhd/gr_fishbone_moncrief_mhd.py \
#       --arm stage_input --nr 128 --npolar 96 --kerr-spin 0 \
#       --end-time 400 --data-directory runs/si_128 --record si_128.json
# =============================================================================

from __future__ import annotations

import glob
import importlib
import json
import os
from typing import Any, Sequence

import numpy as np

from simbi.reader import read_simulation
from simbi.reader.adapter import SimData
from simbi.reader.computation import closure_of, spec_enthalpy
from simbi.reader.gr_accretion import _radial_metric, accretion_from_checkpoint

SUPPORTED_SPACETIME = "schwarzschild_ks"

# the checkpoint labels the chart by metric TYPE, so a zero-spin run of the Kerr
# config records `kerr_ks`; at a = 0 that metric is exactly schwarzschild_ks.
# both map to the reader's validated schwarzschild_ks helpers. the capture runs
# at zero spin, where this identity holds.
_SPIN_ZERO_KS_CHARTS = ("schwarzschild_ks", "kerr_ks")


def _resolved_chart(spacetime: str) -> str:
    """the reader-supported metric label for a zero-spin Kerr-Schild run."""
    if spacetime not in _SPIN_ZERO_KS_CHARTS:
        raise ValueError(
            f"anchor_capture: the bulk integrals and horizon flux need a zero-spin "
            f"Kerr-Schild chart {_SPIN_ZERO_KS_CHARTS}; got '{spacetime}'. a spinning "
            f"chart needs the reader's metric helpers extended."
        )
    return SUPPORTED_SPACETIME

# the density fraction (of the peak) below which a cell is corona, not torus
# body: the torus rest-mass observable integrates only cells above it.
TORUS_DENSITY_FRACTION = 1.0e-2


# =============================================================================
# metric integrals on the schwarzschild-kerr-schild chart
# =============================================================================


def _cell_centers_and_widths(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cell centers and widths from monotone vertex coordinates."""
    v = np.asarray(vertices, dtype=float)
    centers = 0.5 * (v[:-1] + v[1:])
    widths = np.diff(v)
    return centers, widths


def _grid_2d(
    data: SimData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """the (r, theta) cell-center grids and their widths, broadcast to the field
    shape. the field arrays are indexed (theta, r) in storage order."""
    mesh = data.mesh
    r_c, dr = _cell_centers_and_widths(np.asarray(mesh.x1v, dtype=float))
    th_c, dth = _cell_centers_and_widths(np.asarray(mesh.x2v, dtype=float))
    shape = (th_c.size, r_c.size)
    r_grid = np.broadcast_to(r_c[None, :], shape)
    th_grid = np.broadcast_to(th_c[:, None], shape)
    dr_grid = np.broadcast_to(dr[None, :], shape)
    dth_grid = np.broadcast_to(dth[:, None], shape)
    cell_area = dr_grid * dth_grid
    return r_grid, th_grid, cell_area, np.asarray([]), shape


def _metric_components(
    r: np.ndarray, theta: np.ndarray, mass: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """the diagonal spatial metric and sqrt(det gamma) on schwarzschild_ks:
    gamma = diag(1 + 2M/r, r^2, r^2 sin^2 theta), sqrt(det gamma) =
    sqrt(1 + 2M/r) r^2 sin(theta)."""
    gamma_rr, _ = _radial_metric(r, mass, SUPPORTED_SPACETIME)
    gamma_thth = r * r
    sin_th = np.sin(theta)
    gamma_phph = r * r * sin_th * sin_th
    sqrt_det = np.sqrt(gamma_rr) * r * r * sin_th
    return np.asarray(gamma_rr), np.asarray(gamma_thth), np.asarray(gamma_phph), sqrt_det


def _field(data: SimData, name: str, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(data.get_field(name), dtype=float)
    if arr.shape != shape:
        arr = arr.reshape(shape)
    return arr


def _velocity_components(data: SimData, shape: tuple[int, int]) -> list[np.ndarray]:
    """the contravariant Valencia velocity components present in the file."""
    names = data.available_fields()
    comps = [_field(data, "v1", shape)]
    for extra in ("v2", "v3"):
        if extra in names:
            comps.append(_field(data, extra, shape))
    return comps


def _magnetic_components(data: SimData, shape: tuple[int, int]) -> list[np.ndarray]:
    """the contravariant cell-centered magnetic components present in the file."""
    names = data.available_fields()
    comps = []
    for bname in ("b1", "b2", "b3"):
        if bname in names:
            comps.append(_field(data, bname, shape))
    return comps


def _metric_dot(
    a: Sequence[np.ndarray],
    b: Sequence[np.ndarray],
    gamma_diag: Sequence[np.ndarray],
) -> np.ndarray:
    """the metric-contracted inner product gamma_ij a^i b^j for diagonal gamma."""
    total = np.zeros_like(np.asarray(a[0], dtype=float))
    for k in range(min(len(a), len(b))):
        total = total + gamma_diag[k] * a[k] * b[k]
    return total


def bulk_integrals(data: SimData, mass: float | None = None) -> dict[str, float]:
    """the densitized conserved totals and magnetic energy on schwarzschild_ks,
    as coordinate integrals over the axisymmetric (r, theta) grid times 2 pi.

    conserved rest mass  = int sqrt(gamma) W rho          dr dtheta dphi
    conserved energy     = int sqrt(gamma) tau            dr dtheta dphi
    magnetic energy      = int sqrt(gamma) (1/2) gamma_ij B^i B^j dr dtheta dphi
    torus rest mass      = the rest-mass integrand over cells above the density cut
    with W and every dot product contracted with the spatial metric."""
    meta = data.metadata
    _resolved_chart(getattr(meta, "spacetime", ""))
    bh_mass = mass if mass is not None else float(meta.schwarzschild_mass)
    if bh_mass <= 0.0:
        raise ValueError(f"anchor_capture: black-hole mass must be positive, got {bh_mass}")

    r, theta, cell_area, _empty, shape = _grid_2d(data)
    gamma_rr, gamma_thth, gamma_phph, sqrt_det = _metric_components(r, theta, bh_mass)
    gamma_diag = [gamma_rr, gamma_thth, gamma_phph]

    rho = _field(data, "rho", shape)
    pre = _field(data, "p", shape) if "p" in data.available_fields() else _field(
        data, "pre", shape
    )
    vel = _velocity_components(data, shape)
    bfield = _magnetic_components(data, shape)

    v2 = _metric_dot(vel, vel, gamma_diag)
    if np.any(v2 >= 1.0):
        raise ValueError("anchor_capture: superluminal Valencia 3-velocity in the checkpoint")
    w = 1.0 / np.sqrt(1.0 - v2)

    eos = closure_of(meta)
    enthalpy = np.asarray(spec_enthalpy(eos, rho, pre, "rmhd"), dtype=float)

    if bfield:
        b2 = _metric_dot(bfield, bfield, gamma_diag)
        vdb = _metric_dot(vel, bfield, gamma_diag)
    else:
        b2 = np.zeros_like(rho)
        vdb = np.zeros_like(rho)

    # the Valencia densitized conserved densities (RMHD), curved contractions.
    dens_mass = sqrt_det * w * rho
    tau = (
        rho * w * w * enthalpy
        - pre
        - rho * w
        + 0.5 * (b2 + v2 * b2 - vdb * vdb)
    )
    dens_energy = sqrt_det * tau
    dens_mag = sqrt_det * 0.5 * b2

    two_pi = 2.0 * np.pi
    weight = cell_area * two_pi

    conserved_mass = float(np.sum(dens_mass * weight))
    conserved_energy = float(np.sum(dens_energy * weight))
    magnetic_energy = float(np.sum(dens_mag * weight))

    rho_cut = TORUS_DENSITY_FRACTION * float(np.max(rho))
    body = rho > rho_cut
    torus_rest_mass = float(np.sum((dens_mass * weight)[body]))

    return {
        "mass": conserved_mass,
        "energy": conserved_energy,
        "magnetic_energy": magnetic_energy,
        "torus_rest_mass": torus_rest_mass,
    }


def horizon_accretion_rate(data: SimData, mass: float | None = None) -> float:
    """the innermost-shell rest-mass accretion rate Mdot, from the reader's
    validated GR surface-flux certificate."""
    chart = _resolved_chart(getattr(data.metadata, "spacetime", ""))
    mdot, _cert = accretion_from_checkpoint(data, mass=mass, spacetime=chart)
    return float(np.asarray(mdot, dtype=float).ravel()[0])


# =============================================================================
# checkpoint discovery
# =============================================================================


def _sorted_checkpoints(data_directory: str) -> list[str]:
    """the run's checkpoints in time order (`<zones>.chkpt.<time>.h5`)."""
    paths = glob.glob(os.path.join(data_directory, "*.chkpt.*.h5"))
    if not paths:
        raise FileNotFoundError(
            f"anchor_capture: no checkpoints matching *.chkpt.*.h5 in {data_directory}"
        )

    def _time_key(path: str) -> float:
        # the time token is `<int>_<frac>` (the checkpoint time with '.' -> '_'),
        # e.g. `002_311` = 2.311; the crash snapshot is `crashed`, which sorts
        # last as the final state.
        base = os.path.basename(path)
        token = base.split(".chkpt.")[-1].rsplit(".h5", 1)[0]
        try:
            return float(token.replace("_", "."))
        except ValueError:
            return float("inf")

    return sorted(paths, key=_time_key)


# =============================================================================
# record assembly
# =============================================================================


def _receipts_dict(bucket: Sequence[Any]) -> dict[str, Any]:
    """map one tuple half of `anchor_experiment_report()` into the anchor_ab
    receipts schema. the 6-arrays are
    [mass_s, mass_a, seg_s, seg_a, raise_s, raise_a]."""
    passes, fired, cells, min_theta, intervention, injected = bucket

    def ledger(arr: Sequence[float]) -> dict[str, list[float]]:
        return {
            "mass": [float(arr[0]), float(arr[1])],
            "energy_segment": [float(arr[2]), float(arr[3])],
            "energy_raise": [float(arr[4]), float(arr[5])],
        }

    return {
        "passes": int(passes),
        "passes_fired": int(fired),
        "projected_cells": int(cells),
        "min_theta": float(min_theta),
        "intervention": ledger(intervention),
        "injected": ledger(injected),
    }


def build_record(
    *,
    convention: str,
    resolution: int,
    config: dict[str, Any],
    report: tuple[Any, Any],
    first: Sequence[Any],
    census: Sequence[int],
    replay_outcomes: dict[str, int],
    conserved_initial: dict[str, float],
    conserved_final: dict[str, float],
    observables: dict[str, float],
) -> dict[str, Any]:
    """assemble one anchor_ab record from the captured statics and integrals."""
    attempted, accepted = report
    att_first, acc_first, acc_time, acc_iter = first
    fallback, freeze, fallback_horizon, freeze_horizon = census
    return {
        "convention": convention,
        "resolution": resolution,
        "config": config,
        "anchor_report": {
            "attempted": _receipts_dict(attempted),
            "accepted": _receipts_dict(accepted),
        },
        "anchor_first": {
            "attempted_first_pass": int(att_first),
            "accepted_first_pass": int(acc_first),
            "accepted_first_time": float(acc_time),
            "accepted_first_iteration": int(acc_iter),
        },
        "guards": {
            "fallback": int(fallback),
            "freeze": int(freeze),
            "fallback_inside_horizon": int(fallback_horizon),
            "freeze_inside_horizon": int(freeze_horizon),
            "replay_outcomes": {k: int(v) for k, v in replay_outcomes.items()},
        },
        "conserved_initial": {
            "mass": float(conserved_initial["mass"]),
            "energy": float(conserved_initial["energy"]),
        },
        "conserved_final": {
            "mass": float(conserved_final["mass"]),
            "energy": float(conserved_final["energy"]),
        },
        "observables": {k: float(v) for k, v in observables.items()},
    }


# =============================================================================
# the in-process run + capture
# =============================================================================

ARM_ENV = "SIMBI_ANCHOR_EXPERIMENT"


def _run_in_process(config: str, run_argv: list[str]) -> bool:
    """run one simulation through the real CLI path in this process, so the
    anchor-experiment statics survive to be read afterward. returns True on a
    clean finish, False when the run crashed — an expected outcome for the FM
    torus, whose collapse is the observable. the crash still writes a
    `.crashed` checkpoint and leaves the receipts populated."""
    from simbi.cli.simbi_parser import SimbiParser

    parser = SimbiParser()
    args, remaining = parser.parse_known_args(["run", config, *run_argv])
    try:
        args.func(args, remaining)
        return True
    except RuntimeError as exc:
        if "crashed" not in str(exc):
            raise
        return False


def _backend(compute_mode: str) -> Any:
    lib = "gpu" if compute_mode == "gpu" else "cpu"
    return importlib.import_module(f"simbi.libs.{lib}_ext")


def _config_identity(
    data: SimData, config_name: str, end_time: float, spin: str, grid: str
) -> dict[str, Any]:
    """the shared-configuration block the A/B validation compares across the arm
    pair, read from the checkpoint's self-describing metadata so it is truthful
    rather than assumed. `end_time` and the arm are the only run inputs not on
    the file."""
    meta = data.metadata
    return {
        "initial_conditions": config_name,
        "end_time": float(end_time),
        "integrator": str(meta.timestepping).lower(),
        "cfl": float(meta.cfl),
        "solver": "hlld",
        "eos": str(getattr(meta, "eos", "") or "gamma_law"),
        "chart": str(getattr(meta, "spacetime", "")),
        "grid": grid,
        "run_config": f"kerr_spin={spin}",
    }


def capture(
    *,
    config: str,
    config_name: str,
    arm: str,
    resolution: int,
    run_argv: list[str],
    data_directory: str,
    end_time: float,
    spin: str,
    grid: str,
    compute_mode: str = "cpu",
) -> dict[str, Any]:
    """set the arm, run the config, read the statics, and build the record.

    `run_argv` is the config's own CLI flags (resolution, spin, end time, and
    `--data-directory`); the shared-configuration block is read from the run's
    checkpoint metadata."""
    if arm not in ("stage_input", "eulerian_rebuilt"):
        raise ValueError(f"anchor_capture: arm must name a convention, got {arm!r}")

    os.environ[ARM_ENV] = arm
    # a crash (the torus collapse) is a captured outcome, not an error: the run
    # still wrote its `.crashed` checkpoint and left the receipts populated.
    _run_in_process(config, run_argv)

    backend = _backend(compute_mode)
    report = backend.anchor_experiment_report()
    first = backend.anchor_experiment_first()
    census = backend.guard_census()

    checkpoints = _sorted_checkpoints(data_directory)
    first_data = read_simulation(checkpoints[0])
    last_data = read_simulation(checkpoints[-1])

    conserved_initial = bulk_integrals(first_data)
    conserved_final = bulk_integrals(last_data)
    survival_time = float(last_data.metadata.time)

    observables = {
        "survival_time": survival_time,
        "horizon_accretion_rate": horizon_accretion_rate(last_data),
        "torus_rest_mass": conserved_final["torus_rest_mass"],
        "magnetic_energy": conserved_final["magnetic_energy"],
    }

    return build_record(
        convention=arm,
        resolution=resolution,
        config=_config_identity(last_data, config_name, end_time, spin, grid),
        report=report,
        first=first,
        census=census,
        replay_outcomes={},
        conserved_initial=conserved_initial,
        conserved_final=conserved_final,
        observables=observables,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="run one FM-torus anchor arm and write its anchor_ab record "
        "(schwarzschild_ks / zero spin)."
    )
    parser.add_argument("--config", required=True, help="the FM MHD config path")
    parser.add_argument(
        "--arm", required=True, choices=("stage_input", "eulerian_rebuilt")
    )
    parser.add_argument("--nr", type=int, required=True, help="radial resolution")
    parser.add_argument("--npolar", type=int, required=True, help="polar resolution")
    parser.add_argument("--kerr-spin", default="0", help="spin (0 for schwarzschild_ks)")
    parser.add_argument(
        "--target-beta",
        default=None,
        help="minimum plasma beta p_gas/(b^2/2); lower is more magnetized",
    )
    parser.add_argument("--end-time", required=True, help="simulation end time")
    parser.add_argument("--data-directory", required=True, help="run output directory")
    parser.add_argument("--record", required=True, help="output record JSON path")
    parser.add_argument("--compute-mode", default="cpu")
    parser.add_argument(
        "--resolution-key",
        type=int,
        default=None,
        help="the sweep key (defaults to nr)",
    )
    args = parser.parse_args(argv)

    run_argv = [
        "--nr",
        str(args.nr),
        "--npolar",
        str(args.npolar),
        "--kerr-spin",
        str(args.kerr_spin),
        "--end-time",
        str(args.end_time),
        "--data-directory",
        args.data_directory,
        "--mode",
        args.compute_mode,
    ]
    if args.target_beta is not None:
        run_argv += ["--target-beta", str(args.target_beta)]
    record = capture(
        config=args.config,
        config_name=os.path.basename(args.config),
        arm=args.arm,
        resolution=args.resolution_key if args.resolution_key is not None else args.nr,
        run_argv=run_argv,
        data_directory=args.data_directory,
        end_time=float(args.end_time),
        spin=str(args.kerr_spin),
        grid=f"{args.nr}x{args.npolar}",
        compute_mode=args.compute_mode,
    )
    record_dir = os.path.dirname(os.path.abspath(args.record))
    os.makedirs(record_dir, exist_ok=True)
    with open(args.record, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
