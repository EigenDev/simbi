# =============================================================================
# invariants.py
#
# the CHEAP diagnostic battery for a relativistic run: the properties that are
# pathological IMMEDIATELY when the discretization is wrong, so they need tens of
# steps rather than thousands.
#
# the organizing observation is that these are properties of the STATE, not of the
# history. given a state, the timestep the scheme selects, the guards it fired, the
# admissibility of every cell and the divergence of the field are all computable
# right there. a long run adds only "and it stayed that way", which is a soak — it
# answers a different question, and it answers it at a thousand times the cost.
#
# THE TIMESTEP FLOOR is the sharpest of them. nothing propagates faster than light,
# so the largest step a hyperbolic system admits is the light-crossing step
#
#     dt_light = cfl * min_i (sqrt(gamma_ii) dx^i / alpha)
#
# which on a flat chart with c = 1 is just `cfl * dx`. the scheme's actual dt may sit
# below that for real reasons (a source characteristic rate, a stiff term), but only
# by a factor. a dt orders of magnitude below dt_light is not a cost, it is a
# constraint being imposed by something that has no business setting the timestep —
# a mis-masked source rate, a limiter firing everywhere, a metric evaluated where it
# is singular. that signature appears within a few steps and never needs a campaign
# to expose.
#
# usage:
#   health = run_and_measure(problem, directory, steps=20)
#   assert_timestep_is_not_collapsed(health)
#   assert_no_exterior_guard_activity(health)
#   assert_state_is_admissible(health)
# =============================================================================
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

# a dt this far below the light-crossing step is not a cost, it is a breakdown.
# the separation is deliberately wide: healthy relativistic runs with a stiff source
# term sit within a few orders of dt_light, while an actual collapse runs to 1e-15 of
# it. the bound sits between those populations rather than encoding today's
# efficiency, and it needs no recalibration when the resolution, domain or cfl
# change, because dt_light scales with all three.
DEFAULT_MIN_DT_FRACTION = 1.0e-8


@dataclass
class RunHealth:
    """what one short run reports about its own numerical health."""

    time: float
    dt: float
    cfl: float
    steps: int
    dt_light: float
    rho: np.ndarray
    pre: np.ndarray | None
    velocity: list[np.ndarray] = field(default_factory=list)
    bsq: np.ndarray | None = None
    guards: tuple[int, int, int, int] | None = None

    @property
    def dt_fraction(self) -> float:
        """the selected step as a fraction of the light-crossing step."""
        return self.dt / self.dt_light


def light_crossing_step(cfl: float, widths: list[float]) -> float:
    """the largest step the hyperbolic system admits, `cfl * min(dx) / c`, c = 1.

    `widths` are PHYSICAL cell widths. on a curved chart the physical width along an
    axis is `sqrt(gamma_ii) dx^i` and the coordinate light speed carries the lapse,
    so a caller on such a chart passes `sqrt(gamma_ii) dx^i / alpha` evaluated where
    the metric is most restrictive — nearest the hole.
    """
    return cfl * min(widths)


def run_and_measure(
    problem,
    directory: str,
    *,
    steps: int,
    widths: list[float],
    backend=None,
    components: int = 3,
) -> RunHealth:
    """evolve a few steps and read back what the state says about its own health.

    the caller supplies the physical cell widths because only it knows the chart.
    """
    from simbi.simulation import runner

    if backend is not None:
        backend.reset_guard_census()

    runner.run(problem, compute_mode="cpu", max_steps=steps)

    finals = glob.glob(os.path.join(directory, "*final*.h5"))
    assert finals, "the run produced no final checkpoint (it crashed)"
    with h5py.File(finals[0], "r") as h:
        meta = h["metadata"].attrs
        t, dt = float(meta["time"]), float(meta["dt"])
        cfl, n = float(meta["cfl"]), int(meta["iteration"])
        prims = h["level_0/partition_0/hydro/primitives"]
        shape = prims["rho"].shape
        res = _declared_resolution(problem, len(shape))
        sl = tuple(
            slice((s - r) // 2, (s - r) // 2 + r) for s, r in zip(shape, res)
        )
        rho = prims["rho"][sl]
        pre = prims["pre"][sl] if "pre" in prims else None
        vel = [
            prims[f"v{k}"][sl] for k in range(1, components + 1) if f"v{k}" in prims
        ]
        bsq = None
        if "b1" in prims:
            bsq = sum(prims[f"b{k}"][sl] ** 2 for k in (1, 2, 3) if f"b{k}" in prims)

    return RunHealth(
        time=t,
        dt=dt,
        cfl=cfl,
        steps=n,
        dt_light=light_crossing_step(cfl, widths),
        rho=rho,
        pre=pre,
        velocity=vel,
        bsq=bsq,
        guards=backend.guard_census() if backend is not None else None,
    )


def _declared_resolution(problem, ndim: int) -> tuple[int, ...]:
    res = getattr(problem, "resolution", None)
    if isinstance(res, int):
        res = (res,)
    res = tuple(int(n) for n in res if n)
    # the checkpoint axes are slowest-first; the config declares them fastest-first.
    return tuple(reversed(res))[:ndim] if len(res) >= ndim else res


def assert_timestep_is_not_collapsed(
    health: RunHealth,
    *,
    min_fraction: float = DEFAULT_MIN_DT_FRACTION,
    label: str = "",
) -> None:
    """the selected step must stay within reach of the light-crossing step.

    this is the cheapest signal that something other than the hyperbolic system is
    setting the timestep, and it is available after a handful of steps.

    there is deliberately NO upper bound here. `dt_light` is the step a signal moving
    at c would impose, but the cfl reduction sizes dt against the FAST MAGNETOSONIC
    speed, which is strictly sub-luminal -- so a healthy relativistic run routinely
    selects dt ABOVE cfl * dx and the ratio exceeds one. measured: 2.1 on 1D
    schwarzschild michel (where the coordinate light speed alpha^2 = 1 - 2M/r is well
    below one near the hole) and 1.26 on the 3D cartesian kerr torus (where the gas
    fast speed had fallen to 0.79c). asserting dt <= dt_light would fail both, and
    reading such a ratio as "stepping past the light cone" is a misdiagnosis -- it is
    the ordinary signature of sub-luminal waves.

    the COLLAPSE direction is the one that carries information, because nothing makes
    dt orders of magnitude SMALLER than the hyperbolic limit except a term that has no
    business setting the timestep.
    """
    tag = f"{label}: " if label else ""
    assert health.dt > 0.0, f"{tag}the timestep is not positive: {health.dt:.3e}"
    assert health.dt_fraction > min_fraction, (
        f"{tag}the timestep collapsed to {health.dt_fraction:.3e} of the "
        f"light-crossing step {health.dt_light:.4e} (bound {min_fraction:.0e}). "
        "nothing propagates faster than light, so a step this far below the "
        "hyperbolic limit is being set by a source rate, a limiter or a metric "
        "evaluated where it is singular — not by the physics."
    )


def assert_state_is_admissible(health: RunHealth, *, label: str = "") -> None:
    """every cell must lie in the physically admissible set."""
    tag = f"{label}: " if label else ""
    assert np.isfinite(health.rho).all(), f"{tag}non-finite density"
    assert health.rho.min() > 0.0, f"{tag}density went non-positive: {health.rho.min():.3e}"
    if health.pre is not None:
        assert np.isfinite(health.pre).all(), f"{tag}non-finite pressure"
        assert health.pre.min() > 0.0, (
            f"{tag}pressure went non-positive: {health.pre.min():.3e}"
        )
    if health.velocity:
        v2 = sum(v * v for v in health.velocity)
        assert np.isfinite(v2).all(), f"{tag}non-finite velocity"
        assert v2.max() < 1.0, (
            f"{tag}velocity reached or exceeded c: v^2 = {v2.max():.6f}"
        )


def assert_no_exterior_freezes(health: RunHealth, *, label: str = "") -> None:
    """no cell outside the horizon may be FROZEN.

    the two guard tiers are not the same failure. a first-order fallback is the
    scheme working as designed: the high-order candidate was inadmissible and the
    cell was recovered at first order. a FREEZE is the scheme giving up — no flux at
    any order produced an admissible state, so the cell was held at its stage input.
    outside the horizon that is a physical breakdown; inside r_+ the region is
    causally disconnected and its clamped metric is a fiction, so guards there are
    expected and exempt.

    use this for a DYNAMIC problem, where fallbacks are a legitimate cost of
    steep gradients. for a stationary solution use
    `assert_no_exterior_guard_activity`, where any limiter at all means the exact
    state is not being held.
    """
    assert health.guards is not None, "guard census was not collected"
    fb, fz, fb_h, fz_h = health.guards
    tag = f"{label}: " if label else ""
    assert fz - fz_h == 0, (
        f"{tag}{fz - fz_h} cell-steps FROZE outside the horizon (interior, causally "
        f"disconnected: {fz_h}; exterior first-order fallbacks: {fb - fb_h}). a "
        "physical cell that no flux can update admissibly is a breakdown the "
        "projection is supposed to preclude"
    )


def assert_no_exterior_guard_activity(health: RunHealth, *, label: str = "") -> None:
    """no limiter of ANY tier may fire outside the horizon.

    the criterion for a STATIONARY solution: an exact equilibrium that needs a
    limiter is not being held, so even a first-order fallback is a defect rather
    than a cost.
    """
    assert health.guards is not None, "guard census was not collected"
    fb, fz, fb_h, fz_h = health.guards
    tag = f"{label}: " if label else ""
    assert (fb - fb_h, fz - fz_h) == (0, 0), (
        f"{tag}limiters fired OUTSIDE the horizon on a stationary state: "
        f"{fb - fb_h} first-order fallbacks, {fz - fz_h} freezes "
        f"(interior, exempt: {fb_h}/{fz_h})"
    )


def assert_field_survived(health: RunHealth, *, label: str = "") -> None:
    """the magnetized premise: a gate on magnetized behavior is vacuous without a field."""
    tag = f"{label}: " if label else ""
    assert health.bsq is not None, f"{tag}the run carries no magnetic field at all"
    assert health.bsq.max() > 0.0, (
        f"{tag}the evolved state carries NO magnetic field; every magnetized "
        "assertion below degenerates to its hydrodynamic case and tests nothing"
    )
