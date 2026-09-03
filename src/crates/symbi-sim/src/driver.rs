// =============================================================================
// driver.rs
//
// shared timestep-driver primitives that operate on `SimStateGeneric`.
// these are not specific to the single-grid integrator — both the top-level
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

use symbi_geometry::{Metric, MotionState};
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

use crate::state::{FieldStore, SimStateGeneric};

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

/// validate every local CFL candidate before reducing to the global step.
/// validating first is required because `f64::min` discards a NaN operand.
pub fn select_timestep(
    candidates: impl IntoIterator<Item = f64>,
    remaining: f64,
    iter: u64,
    time: f64,
) -> symbi_xpu::Result<f64> {
    check_dt(remaining, iter, time)?;
    let mut dt = remaining;
    let mut count = 0usize;
    for (ii, candidate) in candidates.into_iter().enumerate() {
        count += 1;
        if !candidate.is_finite() || candidate <= 0.0 {
            return Err(symbi_xpu::XpuError {
                operation: "evolve",
                code: -1,
                detail: format!(
                    "evolve: invalid CFL candidate {ii} = {candidate:e} at iter {iter} \
                     (time {time:.4e}); state went NaN/inf"
                ),
            });
        }
        dt = dt.min(candidate);
    }
    if count == 0 {
        return Err(symbi_xpu::XpuError {
            operation: "evolve",
            code: -1,
            detail: format!("evolve: no CFL candidates at iter {iter} (time {time:.4e})"),
        });
    }
    Ok(dt)
}

/// reduce a rejected explicit timestep while preserving a representable clock
/// increment. rejection is a numerical recovery action, not a state update.
pub fn retry_timestep(dt: f64, time: f64) -> symbi_xpu::Result<f64> {
    let retry = 0.5 * dt;
    if !retry.is_finite() || retry <= 0.0 || time + retry == time {
        return Err(symbi_xpu::XpuError {
            operation: "evolve",
            code: -1,
            detail: format!(
                "evolve: rejected timestep cannot be reduced further: dt={dt:e}, time={time:e}"
            ),
        });
    }
    Ok(retry)
}

/// return the simulation clock after one accepted step.
pub fn advance_clock(time: f64, iteration: u64, dt: f64) -> (f64, u64) {
    (time + dt, iteration + 1)
}

/// a multistage SSP scheme needs the step-entry state for later convex blends.
pub fn needs_step_snapshot(stages: &[(f64, f64)]) -> bool {
    stages.len() > 1
}

/// the downstream shu-osher propagation weight of stage `stage`'s output: the
/// product of the convex coefficients `ac` of every later stage. each later
/// combine `a0*u_n + ac*(u_prev + dt L)` folds the previous state in with
/// weight `ac` and the flux divergence telescopes over the interior, so a
/// conserved-quantity delta added to a stage's output reaches the accepted
/// step total scaled by exactly this factor. euler -> [1]; rk2 -> [1/2, 1];
/// rk3 -> [1/6, 2/3, 1].
pub fn downstream_injection_weight(stages: &[(f64, f64)], stage: usize) -> f64 {
    stages[stage + 1..].iter().map(|&(_a0, ac)| ac).product()
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
    PHASE_MS
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (*k, *v))
        .collect()
}

/// `c_{k+1} = ac*(c_k + 1)` — euler -> [1]; rk2 -> [1, 1]; rk3 -> [1, 1/2, 1].
/// consumers: the amr coarse-fine ghost time interpolation (stage k+1
/// reconstructs from that state) and the mesh-motion stage clock (a stage's
/// entry time is the previous stage's exit).
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageSpec {
    pub index: usize,
    pub a0: f64,
    pub ac: f64,
    pub entry: f64,
}

/// build the canonical stage schedule, including the physical time fraction
/// at which each stage reads its input state.
pub fn stage_schedule(stages: &[(f64, f64)]) -> Vec<StageSpec> {
    let exits = stage_time_fractions(stages);
    stages
        .iter()
        .enumerate()
        .map(|(ii, &(a0, ac))| StageSpec {
            index: ii,
            a0,
            ac,
            entry: if ii == 0 { 0.0 } else { exits[ii - 1] },
        })
        .collect()
}

/// set mesh motion to the stage-entry clock shared by every topology.
pub fn set_stage_motion(
    motion: &mut MotionState<f64>,
    law_value: Option<(f64, f64)>,
    dt: f64,
    a_n: f64,
    entry: f64,
) {
    if let Some((a, a_dot)) = law_value {
        motion.a = a;
        motion.a_dot = a_dot;
    } else if motion.homologous {
        motion.a = a_n + motion.a_dot * entry * dt;
    }
}

/// advance mesh motion to the accepted step time.
pub fn advance_motion(motion: &mut MotionState<f64>, law_value: Option<(f64, f64)>, dt: f64) {
    if let Some((a, a_dot)) = law_value {
        motion.a = a;
        motion.a_dot = a_dot;
    } else if motion.homologous {
        motion.a += motion.a_dot * dt;
    }
}

/// advance one field store to its accepted step time, including its motion law.
pub fn advance_state_clock<const D: usize, const DOF: usize, Mem>(
    state: &mut FieldStore<D, DOF, Mem, f64>,
    dt: f64,
) where
    Mem: MemorySpace,
{
    (state.time, state.iteration) = advance_clock(state.time, state.iteration, dt);
    let law_value = state
        .motion_law
        .as_ref()
        .map(|law| (law.a_at(state.time), law.adot_at(state.time)));
    advance_motion(&mut state.motion, law_value, dt);
}

/// return the first horizon body's index and diagnostic-shell radius.
pub fn horizon_request<const D: usize, const DOF: usize, Mem>(
    state: &FieldStore<D, DOF, Mem, f64>,
) -> Option<(usize, f64)>
where
    Mem: MemorySpace,
{
    state.immersed.as_ref().and_then(|immersed| {
        immersed
            .bodies
            .bodies()
            .iter()
            .enumerate()
            .find_map(|(index, body)| match body.kind {
                symbi_ib::BodyKind::Horizon {
                    diagnostic_radius, ..
                } => Some((index, diagnostic_radius)),
                _ => None,
            })
    })
}

/// book one accepted diagnostic-shell flux receipt onto a horizon body.
pub fn book_horizon_receipt<const D: usize, const DOF: usize, Mem>(
    state: &mut FieldStore<D, DOF, Mem, f64>,
    index: usize,
    mdot: f64,
    edot: f64,
    dt: f64,
) where
    Mem: MemorySpace,
{
    let Some(immersed) = state.immersed.as_mut() else {
        return;
    };
    if let symbi_ib::BodyKind::Horizon {
        total_accreted_mass,
        total_accreted_energy,
        mdot: mass_rate,
        edot: energy_rate,
        ..
    } = &mut immersed.bodies.get_mut(index).kind
    {
        *total_accreted_mass += mdot * dt;
        *total_accreted_energy += edot * dt;
        *mass_rate = mdot;
        *energy_rate = edot;
    }
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
/// advance prescribed binary orbital (Keplerian) motion. diagnostics-only for the gravitating
/// mass (fixed-potential sink); no kernel dispatch — pure host bookkeeping over the sim.
pub fn evolve_bodies<R: Regime<f64, D>, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
) where
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let (dt, time) = (sim.dt, sim.time);

    let Some(im) = sim.immersed.as_mut() else {
        return;
    };
    // fragments without their pair physics would feel wall forces yet never
    // move — a silently frozen cluster reads as valid output, so refuse it.
    assert!(
        im.bodies.fragment_count() == 0 || im.fragment_physics.is_some(),
        "{} fragments attached without fragment physics (attach_fragment_physics)",
        im.bodies.fragment_count(),
    );
    let step_deltas = im.diagnostics.consolidate();

    // record feedback as diagnostics + advance the prescribed binary orbit (the body's gravitating
    // mass is held fixed -- a fixed-potential sink: the fluid is removed + accretion measured, but
    // the central potential does not drift; force/torque are recorded for output only; the
    // prescribed motion does not consume them). the same apply the decomposed body step uses with its cross-tile sum.
    // only the source prefix integrates here; fragment motion belongs to the
    // bonded subcycle below.
    symbi_ib::apply_body_deltas(&mut im.bodies, &step_deltas, dt);

    // bonded fragments: the penalization receipts become frozen external
    // loads and the fragment system subcycles (bonds + contact + mutual
    // gravity + gas drag) over the step.
    if let Some(sys) = im.fragment_physics.as_mut() {
        let n_src = im.bodies.source_count();
        let mut external = vec![symbi_ib::ExternalLoad::zero(); im.bodies.len()];
        for delta in &step_deltas {
            if delta.idx >= n_src && delta.idx < external.len() {
                external[delta.idx].force = delta.force_delta;
                external[delta.idx].torque = delta.torque_delta;
            }
        }
        sys.advance(&mut im.bodies, dt, &external);
    }

    // the per-step exchange series: Mdot(t) and F_acc(t) as functionals of the
    // solved flow — the record the steady-state detector consumes.
    im.history.push(time, dt, &step_deltas);

    im.diagnostics.reset();
}

#[cfg(test)]
mod tests {
    use super::{
        advance_clock, check_dt_or_panic, downstream_injection_weight, needs_step_snapshot,
        retry_timestep, select_timestep, stage_schedule,
    };

    #[test]
    fn check_dt_or_panic_accepts_positive_finite() {
        // does not panic.
        check_dt_or_panic(1e-3, 0, 0.0);
        check_dt_or_panic(1.0, 100, 0.5);
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

    #[test]
    fn rk3_stage_schedule_uses_stage_input_times() {
        let schedule = stage_schedule(&[(0.0, 1.0), (0.75, 0.25), (1.0 / 3.0, 2.0 / 3.0)]);
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0].entry, 0.0);
        assert_eq!(schedule[1].entry, 1.0);
        assert_eq!(schedule[2].entry, 0.5);
    }

    #[test]
    fn timestep_selection_rejects_a_nan_candidate_before_reduction() {
        let err = select_timestep([0.2, f64::NAN, 0.1], 0.5, 7, 1.25)
            .expect_err("a NaN tile CFL must not be hidden by f64::min");
        assert!(err.detail.contains("candidate 1"));
        assert!(err.detail.contains("NaN"));
    }

    #[test]
    fn timestep_selection_uses_the_smallest_valid_candidate_and_remaining_time() {
        assert_eq!(select_timestep([0.2, 0.1], 0.5, 0, 0.0).unwrap(), 0.1);
        assert_eq!(select_timestep([0.2, 0.1], 0.05, 0, 0.0).unwrap(), 0.05);
    }

    #[test]
    fn rejected_timestep_is_halved_without_advancing_the_clock() {
        assert_eq!(retry_timestep(0.25, 3.0).unwrap(), 0.125);
    }

    #[test]
    fn rejected_timestep_fails_when_the_clock_cannot_represent_progress() {
        let err = retry_timestep(f64::MIN_POSITIVE, 1.0).unwrap_err();
        assert!(err.detail.contains("cannot be reduced further"));
    }

    #[test]
    fn accepted_step_advances_time_and_iteration_once() {
        let (time, iteration) = advance_clock(1.25, 7, 0.125);
        assert_eq!(time, 1.375);
        assert_eq!(iteration, 8);
    }

    #[test]
    fn only_multistage_schemes_need_a_step_snapshot() {
        assert!(!needs_step_snapshot(&[(0.0, 1.0)]));
        assert!(needs_step_snapshot(&[(0.0, 1.0), (0.5, 0.5)]));
    }

    /// the downstream propagation weights of the three SSP tables, exactly:
    /// a delta injected after a stage's combine is rescaled by every later
    /// convex coefficient on its way into the accepted step.
    #[test]
    fn injection_weights_match_the_ssp_convexity() {
        use crate::state::Timestepping;
        let w = |t: Timestepping| -> Vec<f64> {
            let stages = t.stages();
            (0..stages.len())
                .map(|s| downstream_injection_weight(stages, s))
                .collect()
        };
        assert_eq!(w(Timestepping::Euler), vec![1.0]);
        assert_eq!(w(Timestepping::Rk2), vec![0.5, 1.0]);
        assert_eq!(w(Timestepping::Rk3), vec![1.0 / 6.0, 2.0 / 3.0, 1.0]);
    }
}
