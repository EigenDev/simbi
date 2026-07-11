// =============================================================================
// evolve.rs
//
// kernel-native evolution driver. the entire timestep is sequenced through
// KernelSet trait methods — zero direct field access from the driver.
//
// all source terms (gravity, geometric) are fused into the godunov kernels.
// no separate source passes. no source fields. one kernel per operation.
//
// usage:
//   let kernels = IsoSubstrateKernelSet::<HostMemory, f64, 1>::new(cs, 0.4, &alloc);
//   evolve(&mut sim, &kernels, t_final)?;
// =============================================================================

use symbi_hydro::regime::Regime;
use symbi_hydro::eos::Eos;
use symbi_geometry::Metric;
use symbi_xpu::{ExecutionSpace, MemorySpace};
use crate::sim::state::*;
// the KernelSet trait lives at the sim<->substrate seam (docs/design/41); the driver
// is a consumer of the contract, not its home. re-exported so the
// `sim::evolve::KernelSet` path resolves for downstream callers.
pub use crate::sim::substrate_seam::KernelSet;
// shared driver primitives (dt guard, stage bookkeeping, profiler, body coupling) live in the
// sim-state core (docs/design/41) so the AMR driver shares them DRY. the public profiler
// API is re-exported at the `sim::evolve::` path for the bench examples.
use crate::sim::driver::{evolve_bodies, prof, stage_tag, stage_time_fractions};
pub use crate::sim::driver::{check_dt, report_profile, reset_profile};

// =============================================================================
// evolution driver
// =============================================================================

pub fn evolve<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    kernels: &impl KernelSet<D, DOF, Mem, f64>,
    t_final: f64,
) -> symbi_xpu::Result<()>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
{
    evolve_with_callback(sim, kernels, t_final, u64::MAX, |_| {})
}

// advance the sim by ONE step at a caller-supplied dt. `evolve` hides the per-step
// sequence inside its run-to-completion loop; the decomposition / spmd drivers need
// per-step control so a shared dt + inter-subdomain halo exchange can be interleaved
// between steps (docs/design/36). prim + cons must be current at entry (prime with
// c2p + ghost_fill once before the first call), same contract as the internal stage loop.
pub fn step_once<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    kernels: &impl KernelSet<D, DOF, Mem, f64>,
    dt: f64,
) where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    sim.dt = dt;
    step(sim, kernels);
}

pub fn evolve_with_callback<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    kernels: &impl KernelSet<D, DOF, Mem, f64>,
    t_final: f64,
    interval: u64,
    mut callback: impl FnMut(&SimStateGeneric<R, D, DOF, M, E, S, Mem>),
) -> symbi_xpu::Result<()>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
{
    // mesh motion is single-grid uniform-spacing hydro only in this pass:
    // homologous expansion on any geometry (curvilinear scales the radial
    // axis), uniform translation on cartesian axis 0 (`a` is unused there and
    // must stay 1). non-uniform maps, immersed bodies, and the mhd substrates
    // (comoving-field convention pending) are not wired.
    if sim.motion.a_dot != 0.0 || sim.motion.a != 1.0 {
        assert!(sim.geom.maps.is_none(), "mesh motion: uniform spacing only");
        if !sim.motion.homologous {
            assert!(
                sim.motion.a == 1.0,
                "mesh motion: uniform translation does not scale (keep a = 1)"
            );
            assert!(
                sim.geom.coords == symbi_geometry::Geometry::Cartesian,
                "mesh motion: uniform translation is cartesian-only"
            );
        }
        assert!(!sim.has_bodies(), "mesh motion: immersed bodies are not wired");
        assert!(
            sim.fields.mhd.is_none(),
            "mesh motion: the mhd substrates are not wired (comoving-field convention pending)"
        );
    }

    // MHD setup guardrail: constrained transport evolves the STAGGERED face B (`bface`) as the
    // divergence-free ground truth. if it was never seeded, the CT integrates garbage faces — the
    // classic first-MHD-run mistake (seeding cell-centered B via seed_cell/seed_cells alone does
    // NOT initialize the faces). fail early + actionably instead of marching to a deep c2p/dt panic.
    // one-time check at entry (zero per-step cost); every real MHD IC sets the flag via seed_face.
    if let Some(mhd) = sim.fields.mhd.as_ref() {
        assert!(
            mhd.bface_initialized.load(std::sync::atomic::Ordering::Relaxed),
            "evolve: this MHD sim's staggered face B is un-seeded (bface_initialized = false). \
             seed the face-normal B before evolve — `sim.seed_face(d, b0)` or \
             `sim.seed_face_with(d, |x| ..)` for each face axis d (these set the flag) — so the \
             constrained transport has a div-free ground truth. seed_cell/seed_cells set only the \
             CELL-centered B; for a genuinely zero-field run call `sim.seed_face(d, 0.0)` explicitly."
        );
    }

    kernels.c2p(sim);
    kernels.ghost_fill(sim);

    // every c2p step is a
    // sum-reduction over per-cell error codes; nonzero means at least one
    // cell failed inversion. on failure, panic with the decoded error code.
    // without this check, NaN cons
    // silently propagates and the runner marches to t_final with garbage,
    // forcing checkpoints with invalid state.
    let initial_err = crate::sim::hydro_ops::scan_c2p_errors(sim);
    if initial_err.is_err() {
        return Err(symbi_xpu::XpuError {
            operation: "evolve",
            code: -1,
            detail: format!(
                "c2p failed on initial conditions at iter {} (time {:.4e}): {}",
                sim.iteration, sim.time, initial_err
            ),
        });
    }

    // optional per-iter trace of a single cell to localize CPU-vs-GPU
    // numerical divergence. set SYMBI_TRACE_CELL="i,j" (2D) or "i,j,k" (3D).
    // each iter writes one line to stderr: iter,time,den,mom0,mom1[,mom2],
    // nrg,bcell0,bcell1[,bcell2]. redirect stderr to file on both backends
    // and `diff` to find the first iter that disagrees.
    let trace_coord: Option<[isize; D]> = std::env::var("SYMBI_TRACE_CELL").ok()
        .and_then(|s| {
            let parts: Vec<isize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if parts.len() == D {
                Some(std::array::from_fn(|i| parts[i]))
            } else {
                eprintln!("SYMBI_TRACE_CELL: expected {} comma-separated ints, got {:?}", D, parts);
                None
            }
        });
    if let Some(c) = trace_coord {
        eprintln!("SYMBI_TRACE: tracing cell {:?} +/- 1 neighborhood", c);
        emit_trace_neighborhood(sim, c);
    }

    let mut last_cb = sim.iteration;
    while sim.time < t_final {
        let dt = prof("cfl", || kernels.cfl(sim)).min(t_final - sim.time);
        if !(dt.is_finite() && dt > 0.0) {
            // terminal NaN/inf cascade only: name the first bad cell before the panic. one-time
            // host scan at the failure boundary — never on the happy path (see report fn).
            crate::regimes::substrate_gpu::device_sync::<Mem>();
            let _ = report_first_nonfinite_cell(sim);
        }
        check_dt(dt, sim.iteration, sim.time)?;
        sim.dt = dt;

        step(sim, kernels);

        // no per-step host-side scans: the `if !dt.is_finite()` check
        // above catches NaN/Inf cascades on the very next iteration
        // (NaN cons -> NaN wave speeds -> NaN dt -> panic). previous
        // per-cell scans of c2p_error + cons.den cost ~1.3 ms/step on
        // unified memory via page-faults, which dominated step time.
        //
        // the cfl scalar readback is the ONLY GPU->CPU roundtrip during
        // computation. all per-cell validation runs on device, or doesn't
        // run.

        // the scale factor advances for homologous expansion only; uniform
        // translation keeps a = 1 (the offset is a_dot * time, derivable).
        if sim.motion_law.is_none() && sim.motion.homologous {
            sim.motion.a += sim.motion.a_dot * sim.dt;
        }
        sim.time += sim.dt;
        // expression motion: refresh a / a_dot to the EXACT values at the new step time (for output
        // and the next step's stage-0 entry); a constant a_dot never tracks a decelerating shock.
        let tnew = sim.time;
        if let Some((a, ad)) = sim.motion_law.as_ref().map(|ml| (ml.a_at(tnew), ml.adot_at(tnew))) {
            sim.motion.a = a;
            sim.motion.a_dot = ad;
        }
        sim.iteration += 1;

        if let Some(c) = trace_coord {
            crate::regimes::substrate_gpu::device_sync::<Mem>();
            emit_trace_neighborhood(sim, c);
        }

        if sim.has_bodies() {
            // backward feedback: reduce per-body force/torque/accreted-mass from the fluid into
            // the side-car diagnostics (docs/design/19), then evolve_bodies consolidates + applies it
            // + advances the (prescribed) binary, and resets the accumulator for the next step.
            prof("body_feedback", || kernels.body_feedback(sim, sim.dt));
            prof("body_motion", || evolve_bodies(sim));
        }

        if sim.iteration - last_cb >= interval {
            last_cb = sim.iteration;
            // the callback reads fields from the host: drain the device queue
            // first (the host-read barrier; no-op on a host backend).
            crate::regimes::substrate_gpu::device_sync::<Mem>();
            callback(sim);
        }
    }
    crate::regimes::substrate_gpu::device_sync::<Mem>();
    callback(sim);
    Ok(())
}

/// failure-only NaN locator. scan the interior for the FIRST cell whose conserved state (density,
/// energy, or cell-centered B) is non-finite and report its index + physical coordinate + the
/// offending values. called ONCE, only when the cfl `dt` has already gone non-finite (the terminal
/// cascade, right before `check_dt_or_panic`) — NEVER in the happy path — so the one-time host-read
/// page-fault cost is irrelevant (the process is about to panic anyway). this is the deliberate exception to
/// the "no per-cell host scans" rule: it converts the bare "state went NaN/inf" into "cell [i,j,k]
/// at x = .. went NaN, den=.. nrg=..", which is where a no-silent-floors debug session starts.
fn report_first_nonfinite_cell<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
) -> Option<[isize; D]>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let mhd = sim.fields.mhd.as_ref();
    for c in sim.geom.interior.iter() {
        let den = *sim.fields.cons.den.view().at(c);
        let nrg = sim.fields.cons.nrg_field().map(|f| *f.view().at(c)).unwrap_or(0.0);
        let b_bad = mhd.map_or(false, |m| (0..DOF).any(|k| !(*m.bcell[k].view().at(c)).is_finite()));
        if !den.is_finite() || !nrg.is_finite() || b_bad {
            let x = sim.geom.cell_coord(c);
            eprintln!(
                "[nan-locator] iter {}: first non-finite interior cell at index {:?} (x = {:?}): \
                 den={:e} nrg={:e}{}",
                sim.iteration, c, x, den, nrg,
                if b_bad { " cell-B non-finite" } else { "" },
            );
            return Some(c);
        }
    }
    eprintln!(
        "[nan-locator] iter {}: dt went non-finite but no non-finite conserved cell was found in the \
         interior — the NaN is in a transient stage (flux / prim / ghost) or a staggered face B; \
         rerun with SYMBI_TRACE_CELL=i,j[,k] to trace a suspect cell across stages.",
        sim.iteration,
    );
    None
}

// =============================================================================
// the explicit SSP time step — ONE driver for every scheme.
//
// the integrator is `sim.timestepping.stages()`: a list of Shu-Osher convex coefficients
// `(a0, ac)`, one row per stage. each stage recomputes the spatial operator (reconstruct ->
// flux -> divergence) and applies the convex combine `cons = a0*u_n + ac*(cons - dt*div + dt*S)`
// via the one `godunov_stage` kernel. forward-Euler, SSP-RK2, and SSP-RK3 differ only in the
// table — no scheme-specific control flow. the body source is weighted by `ac` (the same convex
// coefficient that weights the flux divergence) — the SSP source treatment `ac*dt*S` per stage.
// =============================================================================

// =============================================================================
// stage pipeline (docs/design/35 R5): the per-stage kernel sequence as DATA.
//
// each `Phase` declares the field groups it READS + WRITES; `step` FOLDS over the
// list, and a debug-only assert verifies every phase's reads were produced by an
// earlier phase's writes (or were stage-entry-current) — the implicit ordering
// invariant made explicit + machine-checked. a reordered or newly-inserted phase
// that reads a stale field trips the assert instead of silently running on last
// step's data. zero hot-path cost: the pipeline is `const`, dispatch is a `match`,
// the assert is debug-only, and the calls / order / gates are byte-identical to
// the prior imperative sequence.
//
// `FieldSet` tracks only the REGIME-INDEPENDENT data flow (cons / prim / flux /
// u_stage) — every regime carries these, so the assert never false-positives.
// regime-specific scratch (the RMHD wave-speed buffers `wave_speeds` -> `flux`
// feeds; the MHD CT fields `efield` / `post_godunov` touch) is real but kept
// OUTSIDE the checked set; those orderings are fixed by the pipeline below.
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
struct FieldSet(u8);

impl FieldSet {
    const NONE:   Self = FieldSet(0);
    const CONS:   Self = FieldSet(1 << 0);
    const PRIM:   Self = FieldSet(1 << 1);
    const FLUX:   Self = FieldSet(1 << 2);
    const USTAGE: Self = FieldSet(1 << 3);
    const fn or(self, o: Self) -> Self { FieldSet(self.0 | o.0) }
    fn contains(self, need: Self) -> bool { self.0 & need.0 == need.0 }
}

#[derive(Clone, Copy)]
enum PhaseKind { SnapshotStage, WaveSpeeds, Flux, Efield, Godunov, PostGodunov, SourceApply, BodySource, C2p, GhostFill }

/// when a phase runs: unconditional, or gated by an additive source overlay / immersed bodies.
#[derive(Clone, Copy)]
enum Gate { Always, AdditiveSource, Bodies }

impl Gate {
    #[inline]
    fn active(self, additive: bool, bodies: bool) -> bool {
        match self { Gate::Always => true, Gate::AdditiveSource => additive, Gate::Bodies => bodies }
    }
}

struct Phase { name: &'static str, kind: PhaseKind, reads: FieldSet, writes: FieldSet, gate: Gate }

// the per-stage SSP pipeline, in execution order. read/write sets are the
// regime-independent data flow; stage entry is `cons | prim` (init c2p+ghost, or
// the prior stage's tail). this list IS the canonical stage order — `step` folds it.
const STAGE_PIPELINE: &[Phase] = &[
    Phase { name: "snapshot_stage", kind: PhaseKind::SnapshotStage, reads: FieldSet::CONS,                    writes: FieldSet::USTAGE, gate: Gate::AdditiveSource },
    Phase { name: "wave_speeds",    kind: PhaseKind::WaveSpeeds,    reads: FieldSet::PRIM,                    writes: FieldSet::NONE,   gate: Gate::Always },
    Phase { name: "flux",           kind: PhaseKind::Flux,          reads: FieldSet::PRIM,                    writes: FieldSet::FLUX,   gate: Gate::Always },
    Phase { name: "efield",         kind: PhaseKind::Efield,        reads: FieldSet::FLUX,                    writes: FieldSet::NONE,   gate: Gate::Always },
    Phase { name: "godunov_stage",  kind: PhaseKind::Godunov,       reads: FieldSet::CONS.or(FieldSet::FLUX), writes: FieldSet::CONS,   gate: Gate::Always },
    Phase { name: "post_godunov",   kind: PhaseKind::PostGodunov,   reads: FieldSet::CONS,                    writes: FieldSet::NONE,   gate: Gate::Always },
    Phase { name: "source_apply",   kind: PhaseKind::SourceApply,   reads: FieldSet::USTAGE,                  writes: FieldSet::CONS,   gate: Gate::AdditiveSource },
    Phase { name: "body_source",    kind: PhaseKind::BodySource,    reads: FieldSet::CONS.or(FieldSet::PRIM), writes: FieldSet::CONS,   gate: Gate::Bodies },
    Phase { name: "c2p",            kind: PhaseKind::C2p,           reads: FieldSet::CONS,                    writes: FieldSet::PRIM,   gate: Gate::Always },
    Phase { name: "ghost_fill",     kind: PhaseKind::GhostFill,     reads: FieldSet::PRIM,                    writes: FieldSet::PRIM,   gate: Gate::Always },
];

fn step<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    k: &impl KernelSet<D, DOF, Mem, f64>,
)
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let stages = sim.timestepping.stages();
    let n = stages.len();
    // snapshot u^n once for MULTI-STAGE schemes (RK2/RK3 corrector reads it with a0>0).
    // forward-Euler (n=1, a0=0) never reads u_n with non-zero weight, so the snapshot
    // write is pure bandwidth waste. RMHD additionally saves bcell -> bcell_n for the
    // CT magnetic-energy correction in the corrector — same logic applies (no corrector
    // on Euler, no read of bcell_n). regime kernel-sets need not branch internally;
    // the evolve loop is the single place to gate this.
    if n > 1 {
        prof("snapshot", || k.snapshot(sim));
    }
    let additive_source = k.has_additive_source();
    // homologous mesh motion: each stage's dispatches bind geometry / grid-velocity
    // scalars from sim.motion, so a stage must see a(t) at its shu-osher ENTRY time
    // (the time of its input state — the same clock the amr cf ghosts use). a is
    // restored afterward; the canonical step advance lives in the caller. static
    // meshes assign a_n back to itself — no behavioral change.
    let a_n = sim.motion.a;
    let frac = stage_time_fractions(stages);
    for (ii, &(a0, ac)) in stages.iter().enumerate() {
        {
            let entry = if ii == 0 { 0.0 } else { frac[ii - 1] };
            let t_entry = sim.time + entry * sim.dt;
            // expression motion: a / a_dot are EXACT functions of time -> evaluate at the stage entry
            // time (no linearization). homologous linear motion: extrapolate from a_n at constant a_dot.
            let mexpr = sim.motion_law.as_ref().map(|ml| (ml.a_at(t_entry), ml.adot_at(t_entry)));
            if let Some((a, ad)) = mexpr {
                sim.motion.a = a;
                sim.motion.a_dot = ad;
            } else if sim.motion.homologous {
                sim.motion.a = a_n + sim.motion.a_dot * (entry * sim.dt);
            }
        }
        let sim = &*sim;
        let bodies = sim.has_bodies();
        // FOLD the stage pipeline (R5). semantics preserved from the prior imperative
        // sequence, phase-by-phase:
        //  - snapshot_stage: cons BEFORE godunov overwrites it, so the additive source pass
        //    evaluates S at the stage input (the state the fused stage uses, S2 invariant).
        //  - wave_speeds: materialize per-cell speeds on the CURRENT prim (RMHD quartic ->
        //    wave_speed_l/r) so flux reads them; no-op for inline-speed regimes.
        //  - godunov/source_apply/body_source share the SSP stage weight `ac*dt` (Euler ac=1
        //    -> dt; RK2 corrector ac=0.5 -> 0.5*dt, the RK2-consistent 0.5*dt*(S^n + S*)).
        // stage entry: cons + prim are current (init c2p+ghost, or the prior stage's tail).
        //
        // at the FIRST stage of a multi-stage scheme the `snapshot` above wrote `cons -> u_n` and
        // nothing has touched cons since, so u_n ALREADY holds the stage input. flag it and skip the
        // `snapshot_stage` copy — `stage_input()` binds u_n for this stage. forward-Euler (n == 1)
        // takes no snapshot, so u_n is stale there and the copy stands.
        let stage_input_is_un = ii == 0
            && n > 1
            && sim.workspace.elide_stage_snapshot.load(std::sync::atomic::Ordering::Relaxed);
        sim.workspace
            .stage_input_is_un
            .store(stage_input_is_un, std::sync::atomic::Ordering::Relaxed);
        let mut have = FieldSet::CONS.or(FieldSet::PRIM);
        for ph in STAGE_PIPELINE {
            if !ph.gate.active(additive_source, bodies) {
                continue;
            }
            if matches!(ph.kind, PhaseKind::SnapshotStage) && stage_input_is_un {
                have = have.or(ph.writes);
                continue;
            }
            debug_assert!(
                have.contains(ph.reads),
                "R5 stage pipeline: phase '{}' reads a field not yet written this stage",
                ph.name,
            );
            match ph.kind {
                PhaseKind::SnapshotStage => prof("snapshot_stage", || k.snapshot_stage(sim)),
                PhaseKind::WaveSpeeds    => k.wave_speeds(sim),
                PhaseKind::Flux          => { for dd in 0..D { prof("flux", || k.flux(sim, dd)); } }
                PhaseKind::Efield        => prof("efield", || k.efield(sim)),
                PhaseKind::Godunov       => prof("godunov_stage", || k.godunov_stage(sim, sim.dt, a0, ac)),
                PhaseKind::PostGodunov   => prof("post_godunov", || k.post_godunov(sim, sim.dt, stage_tag(ii, n))),
                PhaseKind::SourceApply   => prof("source_apply", || k.source_apply(sim, ac * sim.dt)),
                PhaseKind::BodySource    => prof("body_source", || k.body_source(sim, ac * sim.dt)),
                PhaseKind::C2p           => prof("c2p", || k.c2p(sim)),
                PhaseKind::GhostFill     => prof("ghost_fill", || k.ghost_fill(sim)),
            }
            have = have.or(ph.writes);
        }
    }
    sim.motion.a = a_n;
}

/// the time fraction of the state AFTER each shu-osher stage: the convex
/// combine `u^{k+1} = a0*u^n + ac*(u^k + dt*L)` places it at

// emit a 3-wide-on-each-axis neighborhood of trace lines around `center`.
// for D=2 -> 9 lines; for D=3 -> 27 lines. each line is tagged with its
// offset from center so diff'd output shows which neighbor first
// diverges between CPU and GPU.
fn emit_trace_neighborhood<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    center: [isize; D],
)
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    fn rec<R, const D: usize, const DOF: usize, M, E, S, Mem>(
        sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
        center: [isize; D],
        offset: [isize; D],
        axis: usize,
    )
    where
        R: Regime<f64, D>,
        M: Metric<f64, D> + Copy,
        E: Eos<f64>,
        S: ExecutionSpace,
        Mem: MemorySpace,
    {
        if axis == D {
            let mut c = center;
            for a in 0..D { c[a] += offset[a]; }
            emit_trace_line(sim, c, &offset);
        } else {
            // SYMBI_TRACE_RADIUS env var sets the half-width; default 1 -> 3x3.
            // 2 -> 5x5 covers PLM stencil reach (radius-2 along each axis).
            let r: isize = std::env::var("SYMBI_TRACE_RADIUS").ok()
                .and_then(|s| s.parse().ok()).unwrap_or(1);
            for d in -r..=r {
                let mut o = offset;
                o[axis] = d;
                rec(sim, center, o, axis + 1);
            }
        }
    }
    rec(sim, center, [0; D], 0);
}

// emit one trace line for the SYMBI_TRACE_CELL diagnostic. format chosen
// for easy `diff` between CPU and GPU runs: fixed columns, hex floats so
// rounding differences show up exactly. coord must be inside the
// allocated domain or the line is suppressed.
fn emit_trace_line<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    coord: [isize; D],
    offset: &[isize; D],
)
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    if !sim.geom.allocated.contains(coord) {
        return;
    }
    // print as both decimal (human-readable) and bit pattern (exact compare)
    // so both value inspection and a `diff` across runs can find the first iter
    // where any bit changes.
    let fmt = |x: f64| format!("{:.16e}({:016x})", x, x.to_bits());
    // tag with offset so multiple cells per iter are distinguishable in diff
    let off_tag = match D {
        2 => format!("[{:+},{:+}]", offset[0], offset[1]),
        3 => format!("[{:+},{:+},{:+}]", offset[0], offset[1], offset[2]),
        _ => format!("{:?}", offset),
    };
    let den = *sim.fields.cons.den.view().at(coord);
    let mut parts = format!("iter={} off={} t={:.6e} den={}", sim.iteration, off_tag, sim.time, fmt(den));
    for dd in 0..D {
        parts.push_str(&format!(" mom{}={}", dd, fmt(*sim.fields.cons.mom[dd].view().at(coord))));
    }
    if let Some(nrg) = sim.fields.cons.nrg_field() {
        parts.push_str(&format!(" nrg={}", fmt(*nrg.view().at(coord))));
    }
    parts.push_str(&format!(" rho={}", fmt(*sim.fields.prim.rho.view().at(coord))));
    for dd in 0..D {
        parts.push_str(&format!(" v{}={}", dd, fmt(*sim.fields.prim.vel[dd].view().at(coord))));
    }
    if let Some(pre) = sim.fields.prim.pre_field() {
        parts.push_str(&format!(" p={}", fmt(*pre.view().at(coord))));
    }
    if let Some(mhd) = sim.fields.mhd.as_ref() {
        for dd in 0..D {
            parts.push_str(&format!(" bcell{}={}", dd, fmt(*mhd.bcell[dd].view().at(coord))));
        }
        // bface[d] at coord = value at lower-d face of cell coord
        for dd in 0..D {
            parts.push_str(&format!(" bface{}={}", dd, fmt(*mhd.bface[dd].view().at(coord))));
        }
        // bflux[dir][k] = direction-dir flux of B-component-k. value at the
        // lower-dir face of cell coord. dump all D*D for full visibility into
        // the magnetic chain (some are zero by construction in pure MHD).
        for dir in 0..D {
            for k in 0..D {
                parts.push_str(&format!(" bflux{}{}={}", dir, k, fmt(*mhd.bflux[dir][k].view().at(coord))));
            }
        }
        // efield[d] at coord. for 2D only efield[0] (= Ez at cell corner)
        // is meaningful; for 3D all three components.
        for dd in 0..D {
            parts.push_str(&format!(" ef{}={}", dd, fmt(*mhd.efield[dd].view().at(coord))));
        }
    }
    eprintln!("SYMBI_TRACE: {}", parts);
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    #[test]
    fn nan_locator_pinpoints_the_poisoned_cell() {
        use crate::prelude::*;
        let sim = SimCpu::<Newtonian, 1, Cartesian, IdealGas<f64>>::build(
            Newtonian,
            IdealGas { gamma: 1.4 },
            Cartesian,
        )
        .cells([8])
        .bounds([0.0], [1.0])
        .finish()
        .unwrap();
        sim.seed_cells(|_| Prim { rho: 1.0, vel: Tensor::new([0.0]), pre: 1.0 });
        // a clean conserved field has no non-finite cell.
        assert_eq!(super::report_first_nonfinite_cell(&sim), None);
        // poison one interior cell's conserved density; the locator returns exactly it.
        let bad = sim.geom.interior.iter().nth(3).expect("8 interior cells");
        sim.fields.cons.den.view_mut().set(bad, f64::NAN);
        assert_eq!(super::report_first_nonfinite_cell(&sim), Some(bad));
    }
}
