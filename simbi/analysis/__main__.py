# =============================================================================
# python -m simbi.analysis <checkpoint.h5> [--gamma 1.4] [--body 0]
#
# the accretor validation report (docs/ideas/accretor.md §5-6): steady-state
# detection, dt-weighted mean accretion rate + fluctuation amplitude, and the
# ratio to the analytic bondi rate f = Mdot / (4 pi lambda_c(gamma)) — rungs
# 1-2 of the ladder in one command. exits nonzero when the run is not
# quotable (never settled, or the averaging span is under 10 t_B).
# =============================================================================

import argparse
import math
import sys

from .accretion import averaged_rate, load_body_diagnostics, steady_state_time


def lambda_c(gamma: float) -> float:
    if abs(gamma - 1.0) < 1e-5:
        return math.e**1.5 / 4.0
    num = 5.0 - 3.0 * gamma
    return 0.25 * (2.0 / num) ** (num / (2.0 * gamma - 2.0))


def _find_attr(f, name):
    if name in f.attrs:
        return f.attrs[name]
    for key in f:
        g = f[key]
        if hasattr(g, "attrs") and name in g.attrs:
            return g.attrs[name]
        if hasattr(g, "keys"):
            for sub in g:
                if name in g[sub].attrs:
                    return g[sub].attrs[name]
    raise KeyError(f"attr '{name}' not found; pass --gamma")


def main() -> int:
    p = argparse.ArgumentParser(prog="simbi.analysis", description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--gamma", type=float, default=None,
                   help="override the checkpoint metadata gamma")
    p.add_argument("--body", type=int, default=0)
    p.add_argument("--window", type=float, default=5.0)
    a = p.parse_args()

    if a.gamma is None:
        import h5py

        with h5py.File(a.checkpoint, "r") as f:
            # the eos parameter travels in the checkpoint metadata; the flag
            # exists only to override a file written before it did.
            a.gamma = float(_find_attr(f, "gamma"))
    d = load_body_diagnostics(a.checkpoint)
    series = d.mdot[:, a.body]
    print(f"series: {len(d.time)} steps, t in [{d.time[0]:.3f}, {d.time[-1]:.3f}]")

    t0 = steady_state_time(d.time, series, dt=d.dt, window=a.window)
    if t0 is None:
        print("NOT SETTLED: consecutive window means never agreed to 1% — run longer")
        return 1
    mean, fluct, span = averaged_rate(d.time, d.dt, series, t0)
    mdot_b = 4.0 * math.pi * lambda_c(a.gamma)
    print(f"steady from t = {t0:.2f}; averaged over {span:.1f} t_B")
    print(f"Mdot = {mean:.6e}  (fluctuation {fluct / abs(mean):.2%})")
    print(f"analytic Mdot_B(gamma={a.gamma}) = {mdot_b:.6e}")
    print(f"f = Mdot / Mdot_B = {mean / mdot_b:.4f}")
    if span < 10.0:
        print("SHORT SPAN: averaged over < 10 t_B — extend the run before quoting")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
