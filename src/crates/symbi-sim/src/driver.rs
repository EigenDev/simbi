// =============================================================================
// driver.rs
//
// shared timestep-driver primitives that operate on `SimStateGeneric`.
// these are NOT specific to the single-grid integrator — both the top-level
// `evolve` loop (symbi) and the AMR hierarchy driver (symbi-amr) need them, so they
// live in the sim-state core below both drivers:
// - check_dt / check_dt_or_panic — dt-livelock guard (NaN/Inf/non-positive)
// - the SYMBI_PROFILE per-phase profiler (prof / reset_profile / report_profile)
// - stage_tag / stage_time_fractions — SSP Shu-Osher stage bookkeeping
// - evolve_bodies — immersed-body diagnostics consolidation + prescribed binary motion
//
// usage:
//  let dt = sim.timestepping...; check_dt_or_panic(dt, iter, time);
//  prof("flux", || k.flux(sim, dd));
// =============================================================================

use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

use crate::state::SimStateGeneric;

/// the loud livelock guard: a NaN/Inf/non-positive `dt` means the state went NaN and the loop
/// would spin forever without surfacing it. shared by `evolve` and the AMR driver so both paths
/// have identical protection.
pub fn check_dt_or_panic(dt: f64, iter: u64, time: f64) {
    if let Err(e) = check_dt(dt, iter, time) {
        panic!("{}", e.detail);
    }
}

/// fallible form of [`check_dt_or_panic`]: returns `Err` (same diagnostic text) on a
/// NaN/Inf/non-positive dt. used at the `evolve` loop site as `check_dt(..)?`
/// so a NaN cascade surfaces as a `Result::Err` to the caller.
pub fn check_dt(dt: f64, iter: u64, time: f64) -> symbi_xpu::Result<()> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(symbi_xpu::XpuError {
            operation: "evolve",
            code: -1,
            detail: format!(
                "evolve: invalid dt = {:e} at iter {} (time {:.4e}); state went NaN/inf — \
                 most likely cause: HLLD secant divergence, c2p NaN, or wave-speed quartic with no real roots",
                dt, iter, time,
            ),
        });
    }
    Ok(())
}

// env-gated per-phase profiler (SYMBI_PROFILE=1). accumulates main-thread wall
// time per phase; each kernel call returns when its rayon par_iter joins, so
// main-thread timing captures the phase's wall cost. used by the zone-cycle
// bench to find which phases fail to scale across cores.
static PROFILE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static PHASE_MS: std::sync::Mutex<std::collections::BTreeMap<&'static str, f64>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());
fn profiling() -> bool {
    *PROFILE_ON.get_or_init(|| std::env::var("SYMBI_PROFILE").is_ok())
}
pub fn prof<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    if !profiling() {
        return f();
    }
    let t0 = std::time::Instant::now();
    let r = f();
    *PHASE_MS.lock().unwrap().entry(name).or_insert(0.0) += t0.elapsed().as_secs_f64() * 1e3;
    r
}
/// clear the per-phase profile accumulator (call after warmup).
pub fn reset_profile() {
    PHASE_MS.lock().unwrap().clear();
}
/// drain the per-phase profile as (phase, milliseconds) pairs.
pub fn report_profile() -> Vec<(&'static str, f64)> {
    PHASE_MS.lock().unwrap().iter().map(|(k, v)| (*k, *v)).collect()
}

/// `c_{k+1} = ac*(c_k + 1)` — euler -> [1]; rk2 -> [1, 1]; rk3 -> [1, 1/2, 1].
/// consumers: the amr coarse-fine ghost time interpolation (stage k+1
/// reconstructs from that state) and the mesh-motion stage clock (a stage's
/// ENTRY time is the previous stage's exit).
pub fn stage_time_fractions(stages: &[(f64, f64)]) -> Vec<f64> {
    let mut c = 0.0;
    stages
        .iter()
        .map(|&(_a0, ac)| {
            c = ac * (c + 1.0);
            c
        })
        .collect()
}

/// map a stage index to the `post_godunov` semantic tag that the constrained-transport hook
/// consumes: 0 = single (forward-Euler), 1 = predictor (saves E_n / bcell_n), 2 = corrector
/// (time-averages the EMF). a >2-stage scheme's interior stages get tag 3 — a no-op for hydro,
/// and rejected by the RMHD CT hook (SSP-RK3 + constrained transport is unimplemented).
pub fn stage_tag(ii: usize, n: usize) -> u8 {
    if n == 1 {
        0
    } else if ii == 0 {
        1
    } else if ii == n - 1 {
        2
    } else {
        3
    }
}

/// per-step immersed-body bookkeeping: consolidate the feedback diagnostics (force / torque /
/// would-be accreted mass) recorded by the kernel `body_feedback` pass into the bodies, then
/// advance prescribed binary orbital (Keplerian) motion. DIAGNOSTICS-only for the gravitating
/// mass (fixed-potential sink); no kernel dispatch — pure host bookkeeping over the sim.
pub fn evolve_bodies<R: Regime<f64, D>, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
)
where M: Metric<f64, D> + Copy, E: Eos<f64>, S: ExecutionSpace, Mem: MemorySpace,
{
    let (dt, time) = (sim.dt, sim.time);

    let Some(im) = sim.immersed.as_mut() else { return; };
    let step_deltas = im.diagnostics.consolidate();

    // record feedback as DIAGNOSTICS + advance the prescribed binary orbit (the body's GRAVITATING
    // mass is held FIXED -- a fixed-potential sink: the fluid is removed + accretion measured, but
    // the central potential does not drift; force/torque are recorded for output only; the
    // prescribed motion does not consume them). the SAME apply the decomposed body step uses with its cross-tile sum.
    symbi_ib::apply_body_deltas(&mut im.bodies, &step_deltas, dt);

    // the per-step exchange series: Mdot(t) and F_acc(t) as functionals of the
    // solved flow — the record the steady-state detector consumes.
    im.history.push(time, dt, &step_deltas);

    im.diagnostics.reset();
}

#[cfg(test)]
mod tests {
    use super::check_dt_or_panic;

    #[test]
    fn check_dt_or_panic_accepts_positive_finite() {
        // does not panic.
        check_dt_or_panic(1e-3, 0, 0.0);
        check_dt_or_panic(1.0,  100, 0.5);
        check_dt_or_panic(f64::MIN_POSITIVE, 0, 0.0);
    }

    #[test]
    #[should_panic(expected = "invalid dt")]
    fn check_dt_or_panic_rejects_nan() {
        check_dt_or_panic(f64::NAN, 42, 1.5);
    }

    #[test]
    #[should_panic(expected = "invalid dt")]
    fn check_dt_or_panic_rejects_pos_inf() {
        check_dt_or_panic(f64::INFINITY, 42, 1.5);
    }

    #[test]
    #[should_panic(expected = "invalid dt")]
    fn check_dt_or_panic_rejects_neg_inf() {
        check_dt_or_panic(f64::NEG_INFINITY, 42, 1.5);
    }

    #[test]
    #[should_panic(expected = "invalid dt")]
    fn check_dt_or_panic_rejects_zero() {
        // dt == 0 is the proximate cause of livelock: `time += 0` never advances.
        check_dt_or_panic(0.0, 42, 1.5);
    }

    #[test]
    #[should_panic(expected = "invalid dt = -0e0")]
    fn check_dt_or_panic_rejects_negative_zero_with_diagnostic() {
        // -0.0 is the actual symptom of NaN-poisoned cfl reduction
        // (f64::max(NaN, x) returns x; reduction stays at NEG_INFINITY;
        // cfl_from_smax = c * dx / -inf = -0.0). the panic message
        // formats it as `-0e0` so future debuggers grep for the
        // canonical form.
        check_dt_or_panic(-0.0, 42, 1.5);
    }

    #[test]
    #[should_panic(expected = "invalid dt")]
    fn check_dt_or_panic_rejects_negative() {
        check_dt_or_panic(-1.0, 42, 1.5);
    }

    #[test]
    #[should_panic(expected = "iter 99")]
    fn check_dt_or_panic_message_carries_iter() {
        check_dt_or_panic(f64::NAN, 99, 2.5);
    }
}
