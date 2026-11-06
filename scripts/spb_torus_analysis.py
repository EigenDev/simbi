#!/usr/bin/env python3
# =============================================================================
# spb_torus_analysis.py
#
# the diagnostics of Stone, Pringle & Begelman 1999 from a run's checkpoints, on the log radial,
# uniform polar grid of simbi_configs/examples/newtonian/spb_torus.py:
# - the angle-integrated mass rates (paper eqs. 8, 10, 11) at each cell radius,
#     Mdot_acc(r) = 2 pi r^2 int rho v_r sin(theta) dtheta, split into inflow and outflow,
#   in units of the initial torus mass per orbit at R_0 = 1. the polar integral sums the cells'
#   exact areas, cos(theta_lo) - cos(theta_hi)
# - the time series of Mdot_acc at the inner boundary's first cell and at r = 0.05 and 0.1
# - over a window of orbits: the time averages of the rates and of the equatorial profiles
#   (theta from 84 to 96 degrees) of rho, P, v_phi, v_r and c_s^2 = gamma P / rho, with
#   power-law slopes fitted between two radii, and the time-averaged Bernoulli function
#   B = v^2 / 2 + gamma P / ((gamma - 1) rho) - 1 / r and specific angular momentum
# the initial torus mass is the mass above the ambient density in the run's first checkpoint.
#
# inputs: a run directory; outputs: a markdown report on stdout and <run>/spb_analysis.npz
#
# usage:
#  python scripts/spb_torus_analysis.py data/spb_torus/A --from-orbit 3.3 [--to-orbit 4.3]
#      [--fit 0.02,0.1]
# =============================================================================
from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

ORBIT = 2.0 * math.pi
AMBIENT_DENSITY = 1.0e-4
EQUATOR = (math.radians(84.0), math.radians(96.0))


def geometry(path: Path) -> tuple[float, float]:
    """the radial extent, from the checkpoint's mesh: a log radial axis and a uniform polar axis
    over [0, pi]"""
    with h5py.File(path, "r") as f:
        polar = dict(f["level_0/mesh/geometry/dim_0"].attrs)
        radial = dict(f["level_0/mesh/geometry/dim_1"].attrs)
    assert str(radial["type"]) == "log", f"the radial axis is {radial['type']}"
    assert str(polar["type"]) == "linear" and float(polar["start"]) == 0.0 and abs(float(polar["end"]) - math.pi) < 1e-14, f"the polar axis {polar}"
    return float(radial["start"]), float(radial["end"])


def read(path: Path) -> tuple[float, dict[str, np.ndarray], dict]:
    """the time, the interior conserved fields as (theta, r) arrays and the metadata"""
    with h5py.File(path, "r") as f:
        level = f["level_0"]
        ng = int(level["mesh"].attrs["halo_width"])
        fields = {k: level[f"conserved/{k}"][()][ng:-ng, ng:-ng] for k in ("den", "m1", "m2", "m3", "nrg")}
        meta = {k: f["metadata"].attrs[k] for k in f["metadata"].attrs}
        return float(level.attrs["time"]), fields, meta


class Grid:
    """the log radial, uniform polar grid of the run"""

    def __init__(self, ni: int, nj: int, inner: float, outer: float) -> None:
        self.r_face = inner * (outer / inner) ** (np.arange(ni + 1) / ni)
        self.t_face = np.arange(nj + 1) * math.pi / nj
        a, b = self.r_face[:-1], self.r_face[1:]
        self.r = (b**4 - a**4) / 4.0 / ((b**3 - a**3) / 3.0)
        self.theta = 0.5 * (self.t_face[:-1] + self.t_face[1:])
        self.polar_weight = np.cos(self.t_face[:-1]) - np.cos(self.t_face[1:])
        self.volume = np.outer(self.polar_weight, b**3 - a**3) * 2.0 * math.pi / 3.0


def checkpoints(run: Path) -> list[tuple[float, Path]]:
    found = []
    for path in sorted(run.glob("*.chkpt.*.h5")):
        if "crashed" in path.name:
            continue
        with h5py.File(path, "r") as f:
            found.append((float(f["level_0"].attrs["time"]), path))
    # the final checkpoint repeats the numbered one written at the same time
    numbered = {t for t, path in found if ".final." not in path.name}
    found = sorted((t, path) for t, path in found if ".final." not in path.name or t not in numbered)
    times = [t for t, _ in found]
    assert len(set(times)) == len(times), f"{run}: two numbered checkpoints hold one time"
    return found


def rates(fields: dict[str, np.ndarray], grid: Grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """inflow, outflow and net mass rates at each cell radius, eqs. 10, 11, 8"""
    flux = fields["m1"] * grid.polar_weight[:, None]
    factor = 2.0 * math.pi * grid.r**2
    inflow = factor * np.minimum(flux, 0.0).sum(axis=0)
    outflow = factor * np.maximum(flux, 0.0).sum(axis=0)
    return inflow, outflow, inflow + outflow


def primitives(fields: dict[str, np.ndarray], gamma: float) -> dict[str, np.ndarray]:
    rho = fields["den"]
    v = [fields[k] / rho for k in ("m1", "m2", "m3")]
    kinetic = 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    pressure = (gamma - 1.0) * (fields["nrg"] - kinetic)
    return {"rho": rho, "vr": v[0], "vt": v[1], "vp": v[2], "p": pressure}


def slope(r: np.ndarray, values: np.ndarray, lo: float, hi: float) -> float:
    """the least-squares power of r between lo and hi"""
    keep = (r >= lo) & (r <= hi) & (np.abs(values) > 0.0)
    assert keep.sum() >= 3, f"fewer than three cells between r = {lo} and {hi}"
    return float(np.polyfit(np.log(r[keep]), np.log(np.abs(values[keep])), 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--from-orbit", type=float, required=True)
    parser.add_argument("--to-orbit", type=float, default=math.inf)
    parser.add_argument("--fit", default="0.02,0.1", help="radii bounding the power-law fits")
    args = parser.parse_args()
    found = checkpoints(args.run)
    assert found, f"{args.run}: no checkpoint"
    t0, first, meta = read(found[0][1])
    gamma = float(meta["gamma"])
    nj, ni = first["den"].shape
    with h5py.File(found[0][1], "r") as f:
        cells = f["level_0/mesh/global_cells"][()]
    # the mesh records its cells in the datasets' axis order, polar first
    assert tuple(int(c) for c in cells) == (nj, ni), f"mesh {tuple(cells)} against the fields {(nj, ni)}"
    inner, outer = geometry(found[0][1])
    grid = Grid(ni, nj, inner, outer)
    assert t0 == 0.0, f"the first checkpoint is at t = {t0}; the torus mass needs the initial state"
    torus_mass = float((np.where(first["den"] > AMBIENT_DENSITY, first["den"] - AMBIENT_DENSITY, 0.0) * grid.volume).sum())
    unit = torus_mass / ORBIT
    probes = [0, int(np.argmin(np.abs(grid.r - 0.05))), int(np.argmin(np.abs(grid.r - 0.1)))]

    series = []
    window = []
    for t, path in found:
        _, fields, _ = read(path)
        inflow, outflow, net = rates(fields, grid)
        series.append((t / ORBIT, *[net[k] / unit for k in probes]))
        if args.from_orbit <= t / ORBIT <= args.to_orbit:
            window.append((t, fields, inflow, outflow, net))
    assert window, f"no checkpoint between orbits {args.from_orbit} and {args.to_orbit}"

    print(f"# SPB diagnostics: {args.run}\n")
    print(f"grid {ni} x {nj}, R_B = {grid.r_face[0]:g}, gamma = {gamma:g}; initial torus mass M_t = {torus_mass:.6e}; rates in M_t per orbit\n")
    print(f"window: {len(window)} checkpoints, orbits {window[0][0] / ORBIT:.3f} to {window[-1][0] / ORBIT:.3f}\n")
    print("## Mdot_acc time series (sampled)\n")
    print(f"| orbit | r = {grid.r[probes[0]]:.4f} | r = {grid.r[probes[1]]:.4f} | r = {grid.r[probes[2]]:.4f} |\n|---|---|---|---|")
    stride = max(1, len(series) // 20)
    for row in series[::stride] + [series[-1]]:
        print(f"| {row[0]:.3f} | {row[1]:+.3e} | {row[2]:+.3e} | {row[3]:+.3e} |")

    inflow = np.mean([w[2] for w in window], axis=0) / unit
    outflow = np.mean([w[3] for w in window], axis=0) / unit
    net = np.mean([w[4] for w in window], axis=0) / unit
    lo, hi = (float(x) for x in args.fit.split(","))
    print("\n## time-averaged mass rates\n")
    print(f"Mdot_acc at the inner boundary {net[0]:+.3e}; slope of |Mdot_in| between r = {lo} and {hi}: {slope(grid.r, inflow, lo, hi):.2f}\n")
    print("| r | Mdot_in | Mdot_out | Mdot_acc |\n|---|---|---|---|")
    for k in np.unique(np.linspace(0, ni - 1, 12).astype(int)):
        print(f"| {grid.r[k]:.4f} | {inflow[k]:+.3e} | {outflow[k]:+.3e} | {net[k]:+.3e} |")

    equator = (grid.theta >= EQUATOR[0]) & (grid.theta <= EQUATOR[1])
    assert equator.any(), "no polar zone lies within 84 to 96 degrees"
    averaged = {k: np.mean([primitives(w[1], gamma)[k] for w in window], axis=0) for k in ("rho", "p", "vp", "vr", "vt")}
    profile = {k: (v[equator] * grid.polar_weight[equator, None]).sum(axis=0) / grid.polar_weight[equator].sum() for k, v in averaged.items()}
    profile["cs2"] = gamma * profile["p"] / profile["rho"]
    print("\n## equatorial profiles, time-averaged, 84 to 96 degrees\n")
    print(f"power of r between r = {lo} and {hi}: " + ", ".join(f"{k} {slope(grid.r, profile[k], lo, hi):.2f}" for k in ("rho", "p", "vp", "vr", "cs2")))
    print("(paper, run A: rho r^0, P r^-1, v_phi r^-1/2, v_r r^-1; c_s^2 near 1 / (3 r))\n")
    print("| r | rho | P | v_phi | v_phi / v_K | v_r | c_s^2 | c_s^2 / (1 / 3r) |\n|---|---|---|---|---|---|---|---|")
    for k in np.unique(np.linspace(0, ni - 1, 12).astype(int)):
        r = grid.r[k]
        print(f"| {r:.4f} | {profile['rho'][k]:.3e} | {profile['p'][k]:.3e} | {profile['vp'][k]:.3e} | {profile['vp'][k] * math.sqrt(r):.3f} | {profile['vr'][k]:+.3e} | {profile['cs2'][k]:.3e} | {profile['cs2'][k] * 3.0 * r:.3f} |")

    radius = grid.r[None, :]
    bernoulli = 0.5 * (averaged["vr"] ** 2 + averaged["vt"] ** 2 + averaged["vp"] ** 2) + gamma * averaged["p"] / ((gamma - 1.0) * averaged["rho"]) - 1.0 / radius
    angular = averaged["vp"] * radius * np.sin(grid.theta)[:, None]
    print(f"\ntime-averaged Bernoulli: fraction of cells with B < 0 within 30 degrees of the equator "
          f"{(bernoulli[np.abs(grid.theta - math.pi / 2) < math.radians(30)] < 0).mean():.3f}, within 30 degrees of the poles "
          f"{(bernoulli[(grid.theta < math.radians(30)) | (grid.theta > math.radians(150))] < 0).mean():.3f}")
    np.savez(args.run / "spb_analysis.npz", r=grid.r, theta=grid.theta, series=np.array(series), inflow=inflow, outflow=outflow, net=net,
             bernoulli=bernoulli, angular_momentum=angular, **{f"avg_{k}": v for k, v in averaged.items()}, torus_mass=torus_mass)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
