# =============================================================================
# low_beta_stage_reference.py
#
# the stage of low_beta_stage_replay.rs recomputed from the intended discrete
# formulas of the 2.5D newtonian MHD update, independent of the production
# kernels:
# - PLM minmod3 (theta = 1) on rho, v, p and the cell B, with the face-normal B
#   taken from the faces;
# - HLLE fluxes with davis wave speeds from the fast magnetosonic speed;
# - the gardiner-stone contact edge EMF with the smooth upwind sign;
# - the godunov update U = U_stage - dt div F (a0 = 0, ac = 1) and the B_z update;
# - the constrained-transport curl at the full dt and the face-to-cell average.
# reads the 13 x 13 stage-entry block of low_beta_stage_entry.rs, evaluates the
# center cell, compares every slot with the production values the fixture holds,
# and prints the internal energy E - |m|^2 / 2 rho - |B|^2 / 2 at entry and after.
# exits nonzero when a slot departs from production beyond round-off.
#
# usage:
#  python3 crates/symbi/tests/fixtures/low_beta_stage_reference.py
# =============================================================================

import re
import sys
from pathlib import Path

import numpy as np

GAMMA = 5.0 / 3.0
H = 1.0 / 256.0
FIXTURE = Path(__file__).with_name("low_beta_stage_entry.rs")


def constants(text: str) -> dict[str, np.ndarray | float]:
    """every `const NAME: ... = value;` of the fixture as a float or an array"""
    out: dict[str, np.ndarray | float] = {}
    for name, body in re.findall(r"const (\w+):[^=]*=\s*(.*?);", text, flags=re.S):
        numbers = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", body)]
        compact = re.sub(r"\s", "", body)
        if compact.startswith("[["):
            rows = [[float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", row)] for row in re.findall(r"\[([^\[\]]*)\]", body)]
            out[name] = np.array(rows)
        elif compact.startswith("["):
            out[name] = np.array(numbers)
        else:
            out[name] = numbers[0]
    return out


C = constants(FIXTURE.read_text())
DT = C["DT"]
N = int(C["N"])
CI = CJ = N // 2
rho, vx, vy, vz, p = C["ENTRY_RHO"], C["ENTRY_VEL0"], C["ENTRY_VEL1"], C["ENTRY_VEL2"], C["ENTRY_PRE"]
B = [C["ENTRY_BCELL0"], C["ENTRY_BCELL1"], C["ENTRY_BCELL2"]]
bface = [C["ENTRY_BFACE0"], C["ENTRY_BFACE1"]]
U_ENTRY = np.array([C[f"ENTRY_{n}"][CJ, CI] for n in ("DEN", "MOM0", "MOM1", "MOM2", "NRG")])


def minmod3(a, b, c):
    s = np.sign(a)
    same = (np.sign(b) == s) & (np.sign(c) == s) & (s != 0)
    return np.where(same, s * np.minimum(np.minimum(abs(a), abs(b)), abs(c)), 0.0)


def slope(ql, q0, qr, theta=1.0):
    a, b = q0 - ql, qr - q0
    return minmod3(theta * a, 0.5 * (a + b), theta * b)


def prim_vec(j, i):
    return np.array([rho[j, i], vx[j, i], vy[j, i], vz[j, i], p[j, i], B[0][j, i], B[1][j, i], B[2][j, i]])


def face_states(d, j, i):
    """the left and right states of the lower face of cell (i, j) along axis d"""
    step = (0, 1) if d == 0 else (1, 0)
    q = lambda k: prim_vec(j + k * step[0], i + k * step[1])
    qm2, qm1, q0, qp1 = q(-2), q(-1), q(0), q(1)
    left = qm1 + 0.5 * slope(qm2, qm1, q0)
    right = q0 - 0.5 * slope(qm1, q0, qp1)
    left[5 + d] = right[5 + d] = bface[d][j, i]
    return left, right


def physical(w, d):
    r, u, v, wz, pr, bx, by, bz = w
    vel, b = np.array([u, v, wz]), np.array([bx, by, bz])
    vn, bn = vel[d], b[d]
    b2 = b @ b
    ptot = pr + 0.5 * b2
    E = pr / (GAMMA - 1) + 0.5 * r * (vel @ vel) + 0.5 * b2
    n = np.zeros(3)
    n[d] = 1.0
    F = np.concatenate([[r * vn], r * vel * vn + ptot * n - bn * b, [(E + ptot) * vn - bn * (vel @ b)], vn * b - bn * vel])
    U = np.concatenate([[r], r * vel, [E], b])
    a2 = GAMMA * pr / r
    s = a2 + b2 / r
    cf = np.sqrt(max(0.5 * (s + np.sqrt(max(s * s - 4 * a2 * bn * bn / r, 0.0))), 0.0))
    return F, U, vn, cf


def hlle(d, j, i):
    """[den, mom0, mom1, mom2, nrg, B0, B1, B2] flux through the lower face of cell (i, j) on axis d"""
    L, R = face_states(d, j, i)
    FL, UL, vnL, cfL = physical(L, d)
    FR, UR, vnR, cfR = physical(R, d)
    bm, bp = min(vnL - cfL, vnR - cfR, 0.0), max(vnL + cfL, vnR + cfR, 0.0)
    if bp - bm > 0:
        return (bp * FL - bm * FR + bp * bm * (UR - UL)) / (bp - bm)
    return FL


F = {(d, j, i): hlle(d, j, i) for d in (0, 1) for j in range(CJ - 2, CJ + 3) for i in range(CI - 2, CI + 3)}


def contact_emf(j, i):
    """E_z at the lower-left corner of cell (i, j)"""
    ezc = lambda jj, ii: vy[jj, ii] * B[0][jj, ii] - vx[jj, ii] * B[1][jj, ii]
    ene, enw, ese, esw = ezc(j, i), ezc(j, i - 1), ezc(j - 1, i), ezc(j - 1, i - 1)
    en, es = -F[(0, j, i)][6], -F[(0, j - 1, i)][6]
    ee, ew = F[(1, j, i)][5], F[(1, j, i - 1)][5]
    fn, fs, fe, fw = F[(0, j, i)][0], F[(0, j - 1, i)][0], F[(1, j, i)][0], F[(1, j, i - 1)][0]
    eps = 32 * np.finfo(float).eps * max(abs(fn), abs(fs), abs(fe), abs(fw))
    soft = lambda f, a, b: 0.5 * ((a + b) + (f / (abs(f) + eps)) * (a - b))
    de_jl = soft(fw, 2 * (es - esw), 2 * (en - enw))
    de_jr = soft(fe, 2 * (ese - es), 2 * (ene - en))
    de_kl = soft(fs, 2 * (ew - esw), 2 * (ee - ese))
    de_kr = soft(fn, 2 * (enw - ew), 2 * (ene - ee))
    return 0.25 * (es + en + ew + ee) + 0.125 * (de_jl - de_jr + de_kl - de_kr)


EZ = {(j, i): contact_emf(j, i) for j in (CJ, CJ + 1) for i in (CI, CI + 1)}

div = lambda k: (F[(0, CJ, CI + 1)][k] - F[(0, CJ, CI)][k]) / H + (F[(1, CJ + 1, CI)][k] - F[(1, CJ, CI)][k]) / H
gas = U_ENTRY - DT * np.array([div(k) for k in range(5)])
bz = B[2][CJ, CI] - DT * div(7)
bx_lo = bface[0][CJ, CI] - DT / H * (EZ[(CJ + 1, CI)] - EZ[(CJ, CI)])
bx_hi = bface[0][CJ, CI + 1] - DT / H * (EZ[(CJ + 1, CI + 1)] - EZ[(CJ, CI + 1)])
by_lo = bface[1][CJ, CI] + DT / H * (EZ[(CJ, CI + 1)] - EZ[(CJ, CI)])
by_hi = bface[1][CJ + 1, CI] + DT / H * (EZ[(CJ + 1, CI + 1)] - EZ[(CJ + 1, CI)])
mine = np.concatenate([gas, [0.5 * (bx_lo + bx_hi), 0.5 * (by_lo + by_hi), bz]])
theirs = np.asarray(C["PRODUCTION_AFTER_CURL"])


def eint(u):
    return u[4] - 0.5 * (u[1] ** 2 + u[2] ** 2 + u[3] ** 2) / u[0] - 0.5 * (u[5] ** 2 + u[6] ** 2 + u[7] ** 2)


entry = eint(np.concatenate([U_ENTRY, [B[0][CJ, CI], B[1][CJ, CI], B[2][CJ, CI]]]))
departure = np.abs(mine - theirs) / np.abs(theirs)
print("relative departure from production per slot [den, mom0, mom1, mom2, nrg, B0, B1, B2]:", np.array2string(departure, precision=3))
print(f"internal energy: entry {entry:.6e}; after the stage {eint(mine):.6e} (production {eint(theirs):.6e})")
# the fluxes evaluate in a different operation order than the kernels, so a slot agrees to a few
# ulps (measured 4.0e-16)
if not np.all(departure < 1e-14):
    sys.exit("the intended formulas depart from the production stage")
