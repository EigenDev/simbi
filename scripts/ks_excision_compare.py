# =============================================================================
# ks_excision_compare.py
#
# compare the states ks_excision_assessment writes at t = 1 for two revisions of
# the code, band by band in cell-center radius: the excised region r < r_exc,
# the live cells between the excision surface and the horizon r_exc <= r < 2M,
# and the exterior r >= 2M. for each edge-EMF arm it reports
# - the x <-> y mirror error of each revision (rho against its transpose, and
#   the x faces against the transposed y faces);
# - the largest absolute and relative difference of rho, pre and the velocity,
#   and the largest absolute difference of each face component;
# - the python gate's own bounds on the rho mirror error.
# a pair whose dumps are missing, or whose end times or step counts differ, is
# reported and not compared: states are compared at one physical time only.
#
# usage:
#  python scripts/ks_excision_compare.py <dump-dir> <baseline-tag> <candidate-tag>
#  (reads <dump-dir>/<tag>_<ct>.bin for ct in contact, uct)
# =============================================================================
import sys
from pathlib import Path

import numpy as np

N = 32
HALF = 5.0
R_EXC = 1.4
R_HORIZON = 2.0
DX = 2 * HALF / N
CENTERS = (np.arange(N) + 0.5) * DX - HALF
FACES = np.arange(N + 1) * DX - HALF


def radius(axis: int | None) -> np.ndarray:
    grids = [CENTERS, CENTERS, CENTERS]
    if axis is not None:
        grids[axis] = FACES
    x, y, z = np.meshgrid(*grids, indexing="ij")
    return np.sqrt(x * x + y * y + z * z)


def bands(r: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("excised", r < R_EXC),
        ("live", (r >= R_EXC) & (r < R_HORIZON)),
        ("exterior", r >= R_HORIZON),
    ]


def load(path: Path) -> tuple[float, int, dict[str, np.ndarray]]:
    raw = np.fromfile(path)
    time, steps, raw = raw[0], int(raw[1]), raw[2:]
    cells = N**3
    fields = {}
    for ii, name in enumerate(["rho", "pre", "v1", "v2", "v3"]):
        fields[name] = raw[ii * cells : (ii + 1) * cells].reshape((N, N, N))
    offset = 5 * cells
    for axis in range(3):
        shape = [N, N, N]
        shape[axis] += 1
        size = int(np.prod(shape))
        fields[f"B{axis + 1}"] = raw[offset : offset + size].reshape(shape)
        offset += size
    assert offset == len(raw), f"{path}: {len(raw) - offset} values past the expected layout"
    return time, steps, fields


def mirror(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(d["rho"] - d["rho"].transpose(1, 0, 2)), np.abs(d["B1"] - d["B2"].transpose(1, 0, 2))


def main(directory: Path, base: str, cand: str) -> None:
    rc, rf = radius(None), [radius(axis) for axis in range(3)]
    for ct in ("contact", "uct"):
        paths = [directory / f"{tag}_{ct}.bin" for tag in (base, cand)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f"== {ct}: incomplete, no state at t = 1 for {missing}")
            continue
        (tb, sb, b), (tc, sc, c) = load(paths[0]), load(paths[1])
        print(f"== {ct}: {base} t={tb} steps={sb} | {cand} t={tc} steps={sc}")
        if tb != tc:
            print(f"   incomplete: the end times differ ({tb} vs {tc}); not compared")
            continue
        for tag, d in ((base, b), (cand, c)):
            rho, face = mirror(d)
            cells = "  ".join(f"{nm}={rho[m].max():.2e}" for nm, m in bands(rc))
            faces = "  ".join(f"{nm}={face[m].max():.2e}" for nm, m in bands(rf[0]))
            print(f"   mirror {tag:>10s}  rho: {cells} | faces Bx<->By: {faces}")
            print(
                f"   python bounds {tag:>10s}: all < 1e-4 {'ok' if rho.max() < 1e-4 else 'FAIL'} ({rho.max():.2e}), "
                f"r > 2 < 2e-5 {'ok' if rho[rc > 2].max() < 2e-5 else 'FAIL'} ({rho[rc > 2].max():.2e})"
            )
        for name in ("rho", "pre", "v1", "v2", "v3"):
            diff = np.abs(c[name] - b[name])
            rel = diff / np.maximum(np.abs(b[name]), 1e-300)
            cells = "  ".join(f"{nm}: {diff[m].max():.2e} / {rel[m].max():.2e}" for nm, m in bands(rc))
            print(f"   {cand} - {base} {name:4s} abs / rel  {cells}")
        for axis in range(3):
            name = f"B{axis + 1}"
            diff = np.abs(c[name] - b[name])
            cells = "  ".join(f"{nm}: {diff[m].max():.2e}" for nm, m in bands(rf[axis]))
            print(f"   {cand} - {base} {name:4s} abs        {cells}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: ks_excision_compare.py <dump-dir> <baseline-tag> <candidate-tag>")
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
