#!/usr/bin/env python3
# =============================================================================
# torus_receipts.py
#
# the conservation receipts of a torus run on the log radial, uniform polar (r, theta) grid of
# simbi_configs/examples/newtonian/spb_torus.py, read from its checkpoints:
# - the totals of mass, angular momentum about the axis and E + rho Phi over the grid, with
#   Phi = -1 / r at each cell's volume-weighted radial centroid (GM = 1; the point mass's compact
#   softening lies inside the inner radius) and the lever arm r sin(theta) at the volume-weighted
#   centroids, the points the kernels evaluate them at
# - the production ledger each checkpoint's metadata carries ("conservation_ledger", the accepted
#   totals as hex words of the f64 bits): outward transport through each wall, the gross mass
#   entering through inward faces, the viscous torque and stress work through the walls and the
#   viscous heating
# - the closure residuals since the initial checkpoint:
#     mass:              dM + outflow
#     angular momentum:  dL + outflow - viscous torque in
#     E + rho Phi:       dE + outflow - viscous work in
# - the mass and the mass-weighted poloidal speed sqrt(v_r^2 + v_theta^2) in the dense torus
#   (rho >= 0.1), the surface (2 rho_0 <= rho < 0.1) and the ambient gas, in units of the
#   keplerian speed at R_0 = 1
#
# inputs: run directories; output: a markdown report on stdout
#
# usage:
#  python scripts/torus_receipts.py data/N44 data/N44_outward --orbits 0.5,1,2,3
#  from torus_receipts import closure_series
# =============================================================================
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spb_torus_analysis import ORBIT, Grid, checkpoints, geometry, read  # noqa: E402

AMBIENT_DENSITY = 1.0e-4
DENSE = 0.1
WALLS = ("r_low", "r_high", "theta_low", "theta_high")


@dataclass(frozen=True)
class Ledger:
    """the accepted receipts: per wall (r low, r high, theta low, theta high) the outward mass,
    angular momentum and energy and the gross mass entering, then the viscous terms and counts"""

    mass: np.ndarray
    angular_momentum: np.ndarray
    energy: np.ndarray
    mass_in: np.ndarray
    viscous_torque_in: float
    viscous_work_in: float
    viscous_heating: float
    viscous_energy_change: float
    stages: int
    viscous_steps: int


def ledger(meta: dict, key: str = "conservation_ledger") -> Ledger | None:
    """the checkpoint's production ledger, None where the run booked none. the key
    "conservation_attempted" gives this process's attempted totals, rejected attempts included"""
    text = meta.get(key)
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode()
    values = np.array([np.uint64(int(w, 16)) for w in str(text).split()], dtype=np.uint64).view(np.float64)
    assert values.size == 39, f"a conservation ledger holds 39 values; {values.size} found"
    walls = values[:24].reshape(6, 4)[:4]
    rest = values[24:]
    return Ledger(walls[:, 0], walls[:, 1], walls[:, 2], walls[:, 3], float(rest[0]), float(rest[1]), float(rest[2]), float(rest[3]),
                  int(rest[13]), int(rest[14]))


def totals(fields: dict[str, np.ndarray], grid: Grid) -> tuple[float, float, float]:
    """mass, angular momentum about the axis and E + rho Phi over the grid"""
    radius = grid.r[None, :]
    lever = radius * np.sin(grid.theta_centroid)[:, None]
    mass = float((fields["den"] * grid.volume).sum())
    angular = float((fields["m3"] * lever * grid.volume).sum())
    energy = float(((fields["nrg"] - fields["den"] / radius) * grid.volume).sum())
    return mass, angular, energy


def regions(fields: dict[str, np.ndarray], grid: Grid) -> dict[str, tuple[float, float]]:
    """mass and mass-weighted poloidal speed of the dense torus, the surface and the ambient gas"""
    rho = fields["den"]
    speed = np.hypot(fields["m1"], fields["m2"]) / rho
    mass = rho * grid.volume
    out = {}
    for name, keep in (("dense", rho >= DENSE), ("surface", (rho >= 2.0 * AMBIENT_DENSITY) & (rho < DENSE)), ("ambient", rho < 2.0 * AMBIENT_DENSITY)):
        m = float(mass[keep].sum())
        out[name] = (m, float((mass[keep] * speed[keep]).sum() / m) if m > 0.0 else 0.0)
    return out


@dataclass(frozen=True)
class Row:
    time: float
    totals: tuple[float, float, float]
    ledger: Ledger | None
    regions: dict[str, tuple[float, float]]
    max_density: float


def closure_series(run: Path) -> tuple[Grid, list[Row]]:
    """every checkpoint's totals, ledger and region measures"""
    found = checkpoints(run)
    assert found, f"{run}: no checkpoint"
    _, first, _ = read(found[0][1])
    nj, ni = first["den"].shape
    inner, outer = geometry(found[0][1])
    grid = Grid(ni, nj, inner, outer)
    rows = []
    for t, path in found:
        _, fields, meta = read(path)
        rows.append(Row(t, totals(fields, grid), ledger(meta), regions(fields, grid), float(fields["den"].max())))
    assert rows[0].time == 0.0, f"{run}: the first checkpoint is at t = {rows[0].time}; closure needs the initial state"
    return grid, rows


def residuals(first: Row, row: Row) -> tuple[float, float, float]:
    """the closure residuals of mass, angular momentum and E + rho Phi since `first`"""
    assert row.ledger is not None, f"t = {row.time}: the checkpoint carries no conservation ledger"
    l = row.ledger
    # a ledger with no stages booked nothing (a store without host access books no receipts), so
    # its residuals would be the raw changes
    assert row.time == first.time or l.stages > 0, f"t = {row.time}: the ledger booked no stage"
    dm, dl, de = (row.totals[k] - first.totals[k] for k in range(3))
    return dm + l.mass.sum(), dl + l.angular_momentum.sum() - l.viscous_torque_in, de + l.energy.sum() - l.viscous_work_in


def nearest(rows: list[Row], time: float) -> Row:
    return min(rows, key=lambda r: abs(r.time - time))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--orbits", default="0.5,1,2,3", help="sample times in orbits at R_0")
    args = parser.parse_args()
    samples = [float(x) * ORBIT for x in args.orbits.split(",")]
    for run in args.runs:
        grid, rows = closure_series(run)
        first = rows[0]
        print(f"## {run}\n")
        print(f"grid {grid.r.size} x {grid.theta.size}; last checkpoint t = {rows[-1].time:.4f} ({rows[-1].time / ORBIT:.4f} orbits); "
              f"initial M = {first.totals[0]:.6e}, L = {first.totals[1]:.6e}, E + rho Phi = {first.totals[2]:.6e}\n")
        print("| orbit | stages | dM + out (/ M0) | dL + out - torque (/ L0) | dE + out - work (/ abs E0) | viscous heating | mass out r_in | mass out r_out | mass in r_out | L out r_in | L out r_out | E out r_in | E out r_out |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for time in samples:
            row = nearest(rows, time)
            if abs(row.time - time) > 0.02 * ORBIT:
                print(f"| {time / ORBIT:.3f} | not reached | | | | | | | | | | | |")
                continue
            rm, rl, re = residuals(first, row)
            l = row.ledger
            print(f"| {row.time / ORBIT:.3f} | {l.stages} | {rm / first.totals[0]:+.2e} | {rl / first.totals[1]:+.2e} | {re / abs(first.totals[2]):+.2e} | {l.viscous_heating:.3e} | "
                  f"{l.mass[0]:+.3e} | {l.mass[1]:+.3e} | {l.mass_in[1]:.3e} | {l.angular_momentum[0]:+.3e} | {l.angular_momentum[1]:+.3e} | {l.energy[0]:+.3e} | {l.energy[1]:+.3e} |")
        print("\n| orbit | max rho | M dense | <v_pol> dense | M surface | <v_pol> surface | M ambient | <v_pol> ambient |\n|---|---|---|---|---|---|---|---|")
        for time in [0.0, *samples]:
            row = nearest(rows, time)
            if abs(row.time - time) > 0.02 * ORBIT:
                continue
            g = row.regions
            print(f"| {row.time / ORBIT:.3f} | {row.max_density:.4f} | {g['dense'][0]:.4e} | {g['dense'][1]:.3e} | {g['surface'][0]:.4e} | {g['surface'][1]:.3e} | "
                  f"{g['ambient'][0]:.4e} | {g['ambient'][1]:.3e} |")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
