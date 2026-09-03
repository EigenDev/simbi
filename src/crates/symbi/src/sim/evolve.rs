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

use crate::sim::state::*;
use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};
// the KernelSet trait lives at the sim<->substrate seam; the driver only consumes
// the contract. re-exported so the `sim::evolve::KernelSet` path
// resolves for downstream callers.
pub use crate::sim::substrate_seam::KernelSet;
// shared driver primitives (dt guard, stage bookkeeping, profiler, body coupling) live in the
// sim-state core so the AMR driver shares them dry. the public profiler API is
// re-exported at the `sim::evolve::` path for the bench examples.
use crate::sim::driver::{
    advance_state_clock, book_horizon_receipt, evolve_bodies, horizon_request, needs_step_snapshot,
    prof, retry_timestep, select_timestep, set_stage_motion, stage_schedule,
};
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

// advance the sim by a single step at a caller-supplied dt. `evolve` hides the per-step
// sequence inside its run-to-completion loop; the decomposition / spmd drivers need
// per-step control so a shared dt + inter-subdomain halo exchange can be interleaved
// between steps. prim + cons must be current at entry (prime with
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
    // the per-step driver opens no projection-ledger scope; its callers advance
    // the clock and own the step boundary, so a projection here books nothing.
    sim.dt = dt;
    while step(sim, kernels) {
        kernels.restore_step(sim);
        symbi_sim::tracers::restore_transport_state(&mut sim.store);
        sim.dt = retry_timestep(sim.dt, sim.time).unwrap_or_else(|err| panic!("{}", err.detail));
    }
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
    // mesh motion is supported for single-grid hydro: homologous expansion on any geometry
    // (curvilinear scales the radial axis), and uniform translation on cartesian axis 0 (`a` is
    // unused there and must stay 1). immersed bodies and the mhd substrates need a comoving-
    // field convention still pending in the kernels.
    if sim.motion.a_dot != 0.0 || sim.motion.a != 1.0 {
        // homologous expansion multiplies every coordinate by one a(t), so a graded axis stays
        // graded with its ratios untouched -- `kernel_geom` scales the axis's face-0 start and
        // leaves its shape parameter comoving, which for a log axis maps
        // face(i) = start 10^(i s) -> a face(i). uniform translation is a different motion: it
        // shifts the axis while scaling multiplies it; a shifted graded axis takes on a shape
        // distinct from any rescaling of itself, so the comoving shape parameter applies only
        // under scaling.
        assert!(
            sim.motion.homologous || sim.geom.maps.is_none(),
            "mesh motion: uniform translation needs uniform spacing (a translated graded axis is \
             not a rescaling of itself)"
        );
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
        assert!(
            !sim.has_bodies(),
            "mesh motion: immersed bodies are not wired"
        );
        assert!(
            sim.fields.mhd.is_none(),
            "mesh motion: the mhd substrates are not wired (comoving-field convention pending)"
        );
    }

    // MHD setup guardrail: constrained transport evolves the staggered face B (`bface`) as the
    // divergence-free ground truth, so an unseeded run leaves the CT integrating garbage faces —
    // the classic first-MHD-run mistake (seeding cell-centered B via seed_cell/seed_cells alone
    // leaves the faces uninitialized). fail early + actionably here; otherwise the run marches to
    // a deep c2p/dt panic. one-time check at entry (zero per-step cost); every real MHD IC sets
    // the flag via seed_face.
    if let Some(mhd) = sim.fields.mhd.as_ref() {
        assert!(
            mhd.bface_initialized
                .load(std::sync::atomic::Ordering::Relaxed),
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
    // cell failed inversion. on failure, panic with the decoded error code:
    // this check stops NaN cons at the source, before it can spread silently
    // through the run and land in a checkpoint as invalid state.
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
    let trace_coord: Option<[isize; D]> = std::env::var("SYMBI_TRACE_CELL").ok().and_then(|s| {
        let parts: Vec<isize> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        if parts.len() == D {
            Some(std::array::from_fn(|i| parts[i]))
        } else {
            eprintln!(
                "SYMBI_TRACE_CELL: expected {} comma-separated ints, got {:?}",
                D, parts
            );
            None
        }
    });
    if let Some(c) = trace_coord {
        eprintln!("SYMBI_TRACE: tracing cell {:?} +/- 1 neighborhood", c);
        emit_trace_neighborhood(sim, c);
    }

    // the projection ledger opens its scope on the backend's tile driver, not
    // this standalone single-grid driver; a run here books no ledger evidence.
    let mut last_cb = sim.iteration;
    while sim.time < t_final {
        let cfl = prof("cfl", || kernels.cfl(sim));
        let dt = match select_timestep([cfl], t_final - sim.time, sim.iteration, sim.time) {
            Ok(dt) => dt,
            Err(err) => {
                // terminal NaN/inf cascade only: name the first bad cell before returning. one-time
                // host scan confined to the failure boundary, off the happy path entirely (see report fn).
                crate::regimes::substrate_gpu::device_sync::<Mem>();
                let _ = report_first_nonfinite_cell(sim);
                return Err(err);
            }
        };
        sim.dt = dt;
        loop {
            if step(sim, kernels) {
                kernels.restore_step(sim);
                symbi_sim::tracers::restore_transport_state(&mut sim.store);
                sim.dt = retry_timestep(sim.dt, sim.time)?;
                continue;
            }
            break;
        }
        // no per-step host-side scans: the `if !dt.is_finite()` check
        // above catches NaN/Inf cascades on the very next iteration
        // (NaN cons -> NaN wave speeds -> NaN dt -> panic). a per-cell host
        // scan of c2p_error + cons.den costs ~1.3 ms/step on unified memory
        // via page-faults, which dominates step time.
        //
        // the cfl scalar readback is the sole GPU->CPU roundtrip during
        // computation; all per-cell validation either runs on device or is
        // skipped entirely.

        // the scale factor advances for homologous expansion only; uniform
        // translation keeps a = 1 (the offset is a_dot * time, derivable).
        let dt = sim.dt;
        advance_state_clock(sim, dt);
        if sim.has_tracers() {
            let geometry = sim.geom.block_geometry(sim.physics.metric);
            let layout = symbi_sim::tracers::TransportLayout::single(&sim.geom.interior);
            symbi_sim::tracers::refresh_derived_positions_store(&mut sim.store, &geometry, layout);
        }

        if let Some(c) = trace_coord {
            crate::regimes::substrate_gpu::device_sync::<Mem>();
            emit_trace_neighborhood(sim, c);
        }

        // the constant-nu viscous transport, once per step after
        // the RK combination — the primitive velocity is current (each stage ends
        // with c2p), and the pass reads prim / writes cons.mom, so it is body-
        // independent and runs regardless of whether the sim carries immersed bodies.
        // inert when inviscid (the kernel-set gates on nu > 0).
        prof("viscous", || kernels.viscous(sim, sim.dt));

        // horizon excision, once per step after the RK combination: overwrite the
        // causally disconnected cells inside the excision sphere (within the
        // black-hole horizon on the cartesian kerr-schild chart) with a
        // zero-gradient outward primitive copy + local conserved rebuild, so the
        // next step's stencils read bounded, smooth values at the excision rim.
        // inert at zero radius (the kernel-set gates on r_exc > 0); profiled
        // inside the dispatch.
        kernels.excise(sim);

        // the GR horizon shell-flux accretion, once per step: the mass_flux / nrg_flux fields still
        // hold the last stage's flux, so a GPU Add-reduction of the boundary flux through the
        // diagnostic shell gives (mdot, edot) into the hole, booked onto the horizon body's ledger.
        if let Some((index, diagnostic_radius)) = horizon_request(sim) {
            let dt = sim.dt;
            let (mdot, edot) = prof("horizon_accretion", || {
                kernels.horizon_accretion(sim, diagnostic_radius)
            });
            book_horizon_receipt(sim, index, mdot, edot, dt);
        }

        if sim.has_bodies() {
            let accretion_density = if sim.has_tracers() {
                crate::regimes::substrate_gpu::device_sync::<Mem>();
                Some(symbi_sim::tracers::snapshot_accretion_density(sim))
            } else {
                None
            };
            // the IBM surface physics, once per step after the
            // full RK combination: applied inside the stage blend, a stage's
            // exponential removal is partially undone by the SSP convex
            // combination, but the receipt stays at its full pre-blend value —
            // the ledger then over-counts (RK2: 3/2x). post-step, receipt == removal exactly.
            prof("penalize", || kernels.penalize(sim, sim.dt));
            if let Some(density_before) = accretion_density.as_deref() {
                crate::regimes::substrate_gpu::device_sync::<Mem>();
                symbi_sim::tracers::advance_accretion_transport(sim, density_before)
                    .unwrap_or_else(|detail| panic!("tracer accretion transport: {detail}"));
                if sim.continuous_tracers.is_some() {
                    let geometry = sim.geom.block_geometry(sim.physics.metric);
                    let layout = symbi_sim::tracers::TransportLayout::single(&sim.geom.interior);
                    let crossing_time = sim.time;
                    symbi_sim::tracers::advance_continuous_accretion_transport_store(
                        &mut sim.store,
                        &geometry,
                        layout,
                        density_before,
                        crossing_time,
                    )
                    .unwrap_or_else(|detail| {
                        panic!("continuous tracer accretion transport: {detail}")
                    });
                }
            }
            // backward feedback: reduce per-body force/torque/accreted-mass from the fluid into
            // the side-car diagnostics, then evolve_bodies consolidates + applies it
            // + advances the (prescribed) binary, and resets the accumulator for the next step.
            // only bodies whose dynamics consume the reduction pay for it (two-way
            // or accreting); a one-way fixed mass skips the full-domain sweep —
            // the same gate the hierarchy driver applies.
            let needs_fb = sim
                .immersed
                .as_ref()
                .is_some_and(|im| im.bodies.needs_feedback());
            if needs_fb {
                prof("body_feedback", || kernels.body_feedback(sim, sim.dt));
            }
            prof("body_motion", || evolve_bodies(sim));
        }

        // the registered binned reductions, at the tail of the accepted step: the stage
        // sequence has finished and the last stage ended in a conserved-to-primitive
        // recovery, so the primitives every census reads are the ones belonging to the
        // state at `sim.time`. sampling mid-stage would bin a partially advanced state.
        symbi_substrate::census_sample::sample_censuses(sim);

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

/// failure-only NaN locator. scan the interior for the first cell whose conserved state (density,
/// energy, or cell-centered B) is non-finite and report its index + physical coordinate + the
/// offending values. called once, only when the cfl `dt` has already gone non-finite (the terminal
/// cascade, right before `check_dt_or_panic`), confined entirely to that failure path — so the
/// one-time host-read page-fault cost is irrelevant (the process is about to panic anyway). this
/// is the deliberate exception to the "no per-cell host scans" rule: it converts the bare "state
/// went NaN/inf" into "cell [i,j,k] at x = .. went NaN, den=.. nrg=..", which is where a
/// no-silent-floors debug session starts.
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
        let nrg = sim
            .fields
            .cons
            .nrg_field()
            .map(|f| *f.view().at(c))
            .unwrap_or(0.0);
        let b_bad = mhd.map_or(false, |m| {
            (0..DOF).any(|k| !(*m.bcell[k].view().at(c)).is_finite())
        });
        if !den.is_finite() || !nrg.is_finite() || b_bad {
            let x = sim.geom.cell_coord(c);
            eprintln!(
                "[nan-locator] iter {}: first non-finite interior cell at index {:?} (x = {:?}): \
                 den={:e} nrg={:e}{}",
                sim.iteration,
                c,
                x,
                den,
                nrg,
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
// the explicit SSP time step — a single driver for every scheme.
//
// the integrator is `sim.timestepping.stages()`: a list of Shu-Osher convex coefficients
// `(a0, ac)`, one row per stage. each stage recomputes the spatial operator (reconstruct ->
// flux -> divergence) and applies the convex combine `cons = a0*u_n + ac*(cons - dt*div + dt*S)`
// via the one `godunov_stage` kernel. forward-Euler, SSP-RK2, and SSP-RK3 differ only in the
// table — no scheme-specific control flow. the body source is weighted by `ac` (the same convex
// coefficient that weights the flux divergence) — the SSP source treatment `ac*dt*S` per stage.
// =============================================================================

// =============================================================================
// stage pipeline: the per-stage kernel sequence as data.
//
// each `Phase` declares the field groups it reads and writes; `step` folds over the
// list, and a debug-only assert verifies every phase's reads were produced by an
// earlier phase's writes (or were stage-entry-current) — the implicit ordering
// invariant made explicit + machine-checked. a reordered or newly-inserted phase
// that reads a stale field trips the assert; silently running on last step's
// data is the failure it prevents. zero hot-path cost: the pipeline is `const`, dispatch is a `match`,
// the assert is debug-only, and the calls / order / gates are byte-identical to
// a hand-written imperative sequence.
//
// `FieldSet` tracks only the regime-independent data flow (cons / prim / flux /
// u_stage) — every regime carries these, so the assert's checks hold true for every regime.
// regime-specific scratch (the RMHD wave-speed buffers `wave_speeds` -> `flux`
// feeds; the MHD CT fields `efield` / `post_godunov` touch) is real but kept
// outside the checked set; those orderings are fixed by the pipeline below.
// =============================================================================

// the stage table + fold live in symbi-sim::stage (the sequence every
// driver folds); this driver keeps only per-step scaffolding.
use symbi_sim::stage::{StageArgs, fold_stage};

fn step<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    k: &impl KernelSet<D, DOF, Mem, f64>,
) -> bool
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let stages = sim.timestepping.stages();
    let n = stages.len();
    if k.fofc_active() {
        prof("snapshot_retry", || k.snapshot_retry(sim));
    }
    // snapshot u^n once for multi-stage schemes (RK2/RK3 corrector reads it with a0>0).
    // forward-Euler (n=1, a0=0) reads u_n only at zero weight, so the snapshot
    // write is pure bandwidth waste. RMHD additionally saves bcell -> bcell_n for the
    // CT magnetic-energy correction in the corrector — same logic applies (Euler skips
    // the corrector and the bcell_n read entirely). regime kernel-sets stay branch-free
    // internally; the evolve loop is the single place to gate this.
    if needs_step_snapshot(stages) {
        prof("snapshot", || k.snapshot(sim));
    }
    if sim.has_tracers() {
        symbi_sim::tracers::snapshot_transport_state(sim);
    }
    if sim.continuous_tracers.is_some() {
        let geometry = sim.geom.block_geometry(sim.physics.metric);
        symbi_sim::tracers::begin_ito_transport_store(&mut sim.store, &geometry)
            .unwrap_or_else(|detail| panic!("ito transport initialization: {detail}"));
    }
    // homologous mesh motion: each stage's dispatches bind geometry / grid-velocity
    // scalars from sim.motion, so a stage must see a(t) at its shu-osher entry time
    // (the time of its input state — the same clock the amr cf ghosts use). a is
    // restored afterward; the canonical step advance lives in the caller. on a
    // static mesh the restore assigns a_n back to itself, a no-op.
    let motion_n = sim.motion;
    let a_n = motion_n.a;
    let mut injection_ledger = std::collections::BTreeMap::new();
    for stage in stage_schedule(stages) {
        let t_entry = sim.time + stage.entry * sim.dt;
        let law_value = sim
            .motion_law
            .as_ref()
            .map(|law| (law.a_at(t_entry), law.adot_at(t_entry)));
        let dt = sim.dt;
        set_stage_motion(&mut sim.motion, law_value, dt, a_n, stage.entry);
        // fold the stage pipeline. each phase's semantics, in execution order:
        //  - snapshot_stage: cons captured before godunov overwrites it, so the additive source pass
        //    evaluates S at the stage input (the state the fused stage uses).
        //  - wave_speeds: materialize per-cell speeds on the current prim (RMHD quartic ->
        //    wave_speed_l/r) so flux reads them; no-op for inline-speed regimes.
        //  - godunov/source_apply/body_source share the SSP stage weight `ac*dt` (Euler ac=1
        //    -> dt; RK2 corrector ac=0.5 -> 0.5*dt, the RK2-consistent 0.5*dt*(S^n + S*)).
        // stage entry: cons + prim are current (init c2p+ghost, or the prior stage's tail).
        //
        // at the first stage of a multi-stage scheme the `snapshot` above wrote `cons -> u_n` and
        // nothing has touched cons since, so u_n already holds the stage input. flag it and skip the
        // `snapshot_stage` copy — `stage_input()` binds u_n for this stage. forward-Euler (n == 1)
        // takes no snapshot, so u_n is stale there and the copy stands.
        let outcome = fold_stage(
            &*sim,
            k,
            StageArgs {
                dt: sim.dt,
                a0: stage.a0,
                ac: stage.ac,
                stage: stage.index,
                n_stages: n,
                injection_weight: symbi_sim::driver::downstream_injection_weight(
                    stages,
                    stage.index,
                ),
                allow_elision: true,
            },
            &mut |_| {},
        );
        if outcome == symbi_sim::stage::StageOutcome::RetryStep {
            sim.motion = motion_n;
            return true;
        }
        if sim.has_tracers() {
            crate::regimes::substrate_gpu::device_sync::<Mem>();
            if sim.continuous_tracers.is_some() {
                let geometry = sim.geom.block_geometry(sim.physics.metric);
                symbi_sim::tracers::accumulate_ito_transport_stage_store(
                    &mut sim.store,
                    &geometry,
                    stage.ac,
                )
                .unwrap_or_else(|detail| panic!("ito transport accumulation: {detail}"));
            }
            let mut injections = symbi_sim::tracers::boundary_injection_transfers(sim);
            injections.extend(symbi_sim::tracers::source_injection_transfers(
                sim, stage.a0, stage.ac,
            ));
            symbi_sim::tracers::fold_injection_ledger(&mut injection_ledger, injections, stage.ac);
            if sim.tracers.is_some() {
                prof("tracers", || {
                    symbi_sim::tracers::advance_stage_mass_transport(
                        sim,
                        stage.a0,
                        stage.ac,
                        stage.index,
                    )
                    .unwrap_or_else(|detail| panic!("tracer transport: {detail}"))
                });
            }
        }
    }
    if sim.tracers.is_some() {
        symbi_sim::tracers::spawn_boundary_injection(sim, injection_ledger.clone())
            .unwrap_or_else(|detail| panic!("tracer boundary injection: {detail}"));
    }
    if sim.continuous_tracers.is_some() {
        let geometry = sim.geom.block_geometry(sim.physics.metric);
        let layout = symbi_sim::tracers::TransportLayout::single(&sim.geom.interior);
        symbi_sim::tracers::materialize_ito_coefficients_store(&mut sim.store, &geometry)
            .unwrap_or_else(|detail| panic!("ito coefficient materialization: {detail}"));
        symbi_sim::tracers::fill_ito_coefficient_boundaries_host(
            sim.ito_coefficients
                .as_ref()
                .expect("ito coefficients were materialized"),
            &sim.geom,
            sim.boundaries,
        )
        .unwrap_or_else(|detail| panic!("ito coefficient boundaries: {detail}"));
        let mut tracers = sim
            .continuous_tracers
            .take()
            .expect("continuous tracers remain attached through the hydro step");
        let coefficients = sim
            .ito_coefficients
            .as_ref()
            .expect("ito coefficients were materialized");
        let (scale_start, scale_end, offset_start, offset_end) =
            symbi_sim::tracers::continuous_tracer_mesh_step(&sim.store, sim.dt);
        symbi_sim::tracers::advance_continuous_tracers(
            &mut tracers,
            coefficients,
            &sim.geom,
            scale_start,
            scale_end,
            offset_start,
            offset_end,
            sim.dt,
        )
        .unwrap_or_else(|detail| panic!("ito tracer advancement: {detail}"));
        let bounds = symbi_sim::tracers::map_continuous_tracer_bounds(
            symbi_sim::tracers::partition_physical_bounds(&sim.geom),
            scale_end,
            offset_end,
        );
        symbi_sim::tracers::apply_continuous_boundaries_host(&mut tracers, bounds, sim.boundaries)
            .unwrap_or_else(|detail| panic!("ito tracer boundaries: {detail}"));
        sim.continuous_tracers = Some(tracers);
        symbi_sim::tracers::spawn_continuous_boundary_injection_store(
            &mut sim.store,
            &geometry,
            layout,
            injection_ledger,
        )
        .unwrap_or_else(|detail| panic!("continuous tracer boundary injection: {detail}"));
    }
    sim.motion = motion_n;
    false
}

/// the time fraction of the state after each shu-osher stage: the convex
/// combine `u^{k+1} = a0*u^n + ac*(u^k + dt*L)` places it at

// emit a 3-wide-on-each-axis neighborhood of trace lines around `center`.
// for D=2 -> 9 lines; for D=3 -> 27 lines. each line is tagged with its
// offset from center so diff'd output shows which neighbor first
// diverges between CPU and GPU.
fn emit_trace_neighborhood<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    center: [isize; D],
) where
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
    ) where
        R: Regime<f64, D>,
        M: Metric<f64, D> + Copy,
        E: Eos<f64>,
        S: ExecutionSpace,
        Mem: MemorySpace,
    {
        if axis == D {
            let mut c = center;
            for a in 0..D {
                c[a] += offset[a];
            }
            emit_trace_line(sim, c, &offset);
        } else {
            // SYMBI_TRACE_RADIUS env var sets the half-width; default 1 -> 3x3.
            // 2 -> 5x5 covers PLM stencil reach (radius-2 along each axis).
            let r: isize = std::env::var("SYMBI_TRACE_RADIUS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
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
) where
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
    let mut parts = format!(
        "iter={} off={} t={:.6e} den={}",
        sim.iteration,
        off_tag,
        sim.time,
        fmt(den)
    );
    for dd in 0..D {
        parts.push_str(&format!(
            " mom{}={}",
            dd,
            fmt(*sim.fields.cons.mom[dd].view().at(coord))
        ));
    }
    if let Some(nrg) = sim.fields.cons.nrg_field() {
        parts.push_str(&format!(" nrg={}", fmt(*nrg.view().at(coord))));
    }
    parts.push_str(&format!(
        " rho={}",
        fmt(*sim.fields.prim.rho.view().at(coord))
    ));
    for dd in 0..D {
        parts.push_str(&format!(
            " v{}={}",
            dd,
            fmt(*sim.fields.prim.vel[dd].view().at(coord))
        ));
    }
    if let Some(pre) = sim.fields.prim.pre_field() {
        parts.push_str(&format!(" p={}", fmt(*pre.view().at(coord))));
    }
    if let Some(mhd) = sim.fields.mhd.as_ref() {
        for dd in 0..D {
            parts.push_str(&format!(
                " bcell{}={}",
                dd,
                fmt(*mhd.bcell[dd].view().at(coord))
            ));
        }
        // bface[d] at coord = value at lower-d face of cell coord
        for dd in 0..D {
            parts.push_str(&format!(
                " bface{}={}",
                dd,
                fmt(*mhd.bface[dd].view().at(coord))
            ));
        }
        // bflux[dir][k] = direction-dir flux of B-component-k. value at the
        // lower-dir face of cell coord. dump all D*D for full visibility into
        // the magnetic chain (some are zero by construction in pure MHD).
        for dir in 0..D {
            for k in 0..D {
                parts.push_str(&format!(
                    " bflux{}{}={}",
                    dir,
                    k,
                    fmt(*mhd.bflux[dir][k].view().at(coord))
                ));
            }
        }
        // efield[d] at coord. for 2D only efield[0] (= Ez at cell corner)
        // is meaningful; for 3D all three components.
        for dd in 0..D {
            parts.push_str(&format!(
                " ef{}={}",
                dd,
                fmt(*mhd.efield[dd].view().at(coord))
            ));
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
        sim.seed_cells(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0]), Pressure(1.0)));
        // a clean conserved field has no non-finite cell.
        assert_eq!(super::report_first_nonfinite_cell(&sim), None);
        // poison one interior cell's conserved density; the locator returns exactly it.
        let bad = sim.geom.interior.iter().nth(3).expect("8 interior cells");
        sim.fields.cons.den.view_mut().set(bad, f64::NAN);
        assert_eq!(super::report_first_nonfinite_cell(&sim), Some(bad));
    }
}
