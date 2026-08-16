// =============================================================================
// stage.rs
//
// the per-RK-stage phase sequence, in one place. every driver — the uni-grid
// evolve loop, the hierarchy's level stage (single and refined), and the
// decomposed (gpus > 1) tile loop — folds this same table through the same
// function, so a phase added here exists on every driver by construction.
// a per-driver copy of the sequence drifts silently — a frozen passive scalar or a
// dead GR accretion ledger on whichever driver the tests do not exercise — so the
// sequence lives once and the fold makes divergence unrepresentable.
//
// driver-specific structure enters only through `HookPoint` callbacks fired
// between phases with no field borrow held: the hierarchy accumulates its
// coarse/fine flux registers AfterFlux (sampling the high-order fluxes before
// fofc may splice them), removes a declared stationary target's discrete
// imbalance BeforeC2p, and re-prolongs coarse-fine ghosts BeforeGhostFill;
// the decomposed loop's halo exchange + second ghost fill happen outside the
// fold (after it, per stage), which is its documented sequence delta.
//
// usage:
//  fold_stage(sim, kernels, StageArgs { dt, a0, ac, stage, n_stages, allow_elision },
//             &mut |_hp: HookPoint| {});
// =============================================================================

use crate::driver::{prof, stage_tag};
use crate::state::FieldStore;
use crate::substrate_seam::KernelSet;
use symbi_algebra::OrderedNumeric;
use symbi_ir::algebra::Scalar;
use symbi_xpu::MemorySpace;

/// arms `workspace.stage_writes` for one stage and disarms it on drop, so the audit is scoped to
/// the stage pipeline rather than to the process. a caller that drives a substrate entry point
/// directly — a harness that seeds flux buffers by hand and then invokes one kernel — is outside
/// any stage, and "was this written earlier in the stage" has no meaning there.
#[cfg(debug_assertions)]
struct StageWriteAudit<'a, const D: usize, const DOF: usize, Mem, Sc>(
    &'a FieldStore<D, DOF, Mem, Sc>,
)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric;

#[cfg(debug_assertions)]
impl<'a, const D: usize, const DOF: usize, Mem, Sc> StageWriteAudit<'a, D, DOF, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn arm(sim: &'a FieldStore<D, DOF, Mem, Sc>) -> Self {
        *sim.workspace.stage_writes.lock().unwrap() = Some(Default::default());
        Self(sim)
    }
}

#[cfg(debug_assertions)]
impl<const D: usize, const DOF: usize, Mem, Sc> Drop for StageWriteAudit<'_, D, DOF, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    fn drop(&mut self) {
        *self.0.workspace.stage_writes.lock().unwrap() = None;
    }
}

/// the regime-independent data-flow sets a phase reads/writes; the fold
/// asserts (debug) that a phase's reads were produced earlier in the stage.
#[derive(Clone, Copy)]
pub struct FieldSet(u8);

impl FieldSet {
    pub const NONE: Self = FieldSet(0);
    pub const CONS: Self = FieldSet(1 << 0);
    pub const PRIM: Self = FieldSet(1 << 1);
    pub const FLUX: Self = FieldSet(1 << 2);
    pub const USTAGE: Self = FieldSet(1 << 3);
    pub const fn or(self, o: Self) -> Self {
        FieldSet(self.0 | o.0)
    }
    pub fn contains(self, need: Self) -> bool {
        self.0 & need.0 == need.0
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum PhaseKind {
    SnapshotStage,
    WaveSpeeds,
    Flux,
    Efield,
    Godunov,
    PostGodunov,
    SourceApply,
    BodySource,
    C2p,
    Fofc,
    ChiUpdate,
    GhostFill,
}

/// when a phase runs: unconditional, or gated by an additive source overlay /
/// immersed bodies / fofc / the passive scalar.
#[derive(Clone, Copy)]
pub enum Gate {
    Always,
    AdditiveSource,
    Bodies,
    Fofc,
    /// every consumer of the stage-input snapshot.
    StageInputConsumer,
    PassiveScalar,
}

impl Gate {
    #[inline]
    fn active(self, additive: bool, bodies: bool, fofc: bool, chi: bool) -> bool {
        match self {
            Gate::Always => true,
            Gate::AdditiveSource => additive,
            Gate::Bodies => bodies,
            Gate::Fofc => fofc,
            // the stage-input snapshot serves every source that evaluates against the state
            // the stage began from, plus the FOFC first-order redo that restarts from it: the
            // additive source pass, the immersed-body pass, and FOFC. an explicit scheme
            // evaluates the flux divergence and every source at one state, so a body run
            // without this snapshot would leave the body reading zeros and applying no force.
            Gate::StageInputConsumer => additive || bodies || fofc,
            Gate::PassiveScalar => chi,
        }
    }
}

pub struct Phase {
    pub name: &'static str,
    pub kind: PhaseKind,
    pub reads: FieldSet,
    pub writes: FieldSet,
    pub gate: Gate,
}

/// the canonical stage order. this list is the sequence — every driver folds it.
pub const STAGE_PIPELINE: &[Phase] = &[
    Phase {
        name: "snapshot_stage",
        kind: PhaseKind::SnapshotStage,
        reads: FieldSet::CONS,
        writes: FieldSet::USTAGE,
        gate: Gate::StageInputConsumer,
    },
    Phase {
        name: "wave_speeds",
        kind: PhaseKind::WaveSpeeds,
        reads: FieldSet::PRIM,
        writes: FieldSet::NONE,
        gate: Gate::Always,
    },
    Phase {
        name: "flux",
        kind: PhaseKind::Flux,
        reads: FieldSet::PRIM,
        writes: FieldSet::FLUX,
        gate: Gate::Always,
    },
    Phase {
        name: "efield",
        kind: PhaseKind::Efield,
        reads: FieldSet::FLUX,
        writes: FieldSet::NONE,
        gate: Gate::Always,
    },
    Phase {
        name: "godunov_stage",
        kind: PhaseKind::Godunov,
        reads: FieldSet::CONS.or(FieldSet::FLUX),
        writes: FieldSet::CONS,
        gate: Gate::Always,
    },
    Phase {
        name: "post_godunov",
        kind: PhaseKind::PostGodunov,
        reads: FieldSet::CONS,
        writes: FieldSet::NONE,
        gate: Gate::Always,
    },
    Phase {
        name: "source_apply",
        kind: PhaseKind::SourceApply,
        reads: FieldSet::USTAGE,
        writes: FieldSet::CONS,
        gate: Gate::AdditiveSource,
    },
    Phase {
        name: "body_source",
        kind: PhaseKind::BodySource,
        // ustage because the body contribution is evaluated at the stage input, the state this
        // stage's flux divergence was also evaluated at, and applied to the advanced cons. an
        // explicit scheme sums the flux and every source over one state; evaluating a complete
        // source operator on an already-advanced state composes them sequentially instead, which
        // is first order in dt at any Runge-Kutta order.
        reads: FieldSet::CONS.or(FieldSet::PRIM).or(FieldSet::USTAGE),
        writes: FieldSet::CONS,
        gate: Gate::Bodies,
    },
    Phase {
        name: "c2p",
        kind: PhaseKind::C2p,
        reads: FieldSet::CONS,
        writes: FieldSet::PRIM,
        gate: Gate::Always,
    },
    // first-order flux correction: redo any zone whose high-order c2p went
    // unphysical with a first-order update from the stage input; host-gated
    // internally on the failure reduction.
    Phase {
        name: "fofc",
        kind: PhaseKind::Fofc,
        reads: FieldSet::CONS.or(FieldSet::PRIM).or(FieldSet::USTAGE),
        writes: FieldSet::PRIM,
        gate: Gate::Fofc,
    },
    // the dye rides after fofc: it consumes the (possibly spliced) mass flux
    // and divides by the stage-final density.
    Phase {
        name: "chi_update",
        kind: PhaseKind::ChiUpdate,
        reads: FieldSet::CONS.or(FieldSet::FLUX).or(FieldSet::PRIM),
        writes: FieldSet::CONS.or(FieldSet::PRIM),
        gate: Gate::PassiveScalar,
    },
    Phase {
        name: "ghost_fill",
        kind: PhaseKind::GhostFill,
        reads: FieldSet::PRIM,
        writes: FieldSet::PRIM,
        gate: Gate::Always,
    },
];

/// driver-specific interleave points, fired with no field borrow held by the
/// fold itself (the callback receives nothing; drivers capture what they need
/// by shared reference — field writes go through interior mutability).
#[derive(Clone, Copy, PartialEq)]
pub enum HookPoint {
    /// after every flux direction, before efield: the hierarchy samples its
    /// coarse/fine flux registers here, on the high-order fluxes, before a
    /// fofc firing may splice them.
    AfterFlux,
    /// after chi_update, before ghost_fill: the hierarchy re-prolongs the
    /// coarse-fine ghost band at the time of the state entering the next stage.
    BeforeGhostFill,
    /// after every source, before the conserved-to-primitive recovery: the last point at which
    /// the stage-final conserved state can still be adjusted and have the primitives, the
    /// admissibility redo, and the ghost band all follow from it. a well-balanced hierarchy
    /// removes its stationary target's discrete imbalance here.
    BeforeC2p,
}

#[derive(Clone, Copy)]
pub struct StageArgs {
    pub dt: f64,
    pub a0: f64,
    pub ac: f64,
    /// zero-based stage index and the scheme's stage count.
    pub stage: usize,
    pub n_stages: usize,
    /// whether this driver may elide the stage-0 stage-input copy (the
    /// per-step snapshot already holds it). the decomposed loop passes false —
    /// it never tracks the alias.
    pub allow_elision: bool,
}

#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageOutcome {
    Accepted,
    RetryStep,
}

/// fold one RK stage of the canonical pipeline over a kernel set.
pub fn fold_stage<const D: usize, const DOF: usize, Mem, Sc, K>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    kernels: &K,
    args: StageArgs,
    hook: &mut impl FnMut(HookPoint),
) -> StageOutcome
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
    K: KernelSet<D, DOF, Mem, Sc> + ?Sized,
{
    let additive = kernels.has_additive_source();
    let fofc = kernels.fofc_active();
    let bodies = sim.has_bodies();
    let chi = sim.has_passive_scalar();

    // at the first stage of a multi-stage scheme the per-step snapshot wrote
    // `cons -> u_n` and nothing has touched cons since, so u_n already holds
    // the stage input: flag it and skip the copy — `stage_input()` binds u_n
    // for this stage. forward-Euler takes no per-step snapshot, so the copy
    // stands there.
    let stage_input_is_un = args.stage == 0
        && args.n_stages > 1
        && args.allow_elision
        && sim
            .workspace
            .elide_stage_snapshot
            .load(std::sync::atomic::Ordering::Relaxed);
    sim.workspace
        .stage_input_is_un
        .store(stage_input_is_un, std::sync::atomic::Ordering::Relaxed);
    if stage_input_is_un {
        // the elided copy: `u_n` already holds this stage's input, so the snapshot is captured
        // without the phase running.
        sim.mark_stage_input_captured();
    }

    // arm the stage-local write audit for the duration of this stage. the fluxes and per-cell wave
    // speeds carry nothing across a stage boundary — whatever a previous stage left in them
    // describes a state that no longer exists — so each stage starts from an empty ledger and every
    // read of one is checked against this stage's producers. the guard disarms on every exit path,
    // including the fofc retry.
    #[cfg(debug_assertions)]
    let _audit = StageWriteAudit::arm(sim);

    let tag = stage_tag(args.stage, args.n_stages);
    let mut have = FieldSet::CONS.or(FieldSet::PRIM);
    for ph in STAGE_PIPELINE {
        if !ph.gate.active(additive, bodies, fofc, chi) {
            continue;
        }
        if ph.kind == PhaseKind::SnapshotStage && stage_input_is_un {
            have = have.or(ph.writes);
            continue;
        }
        debug_assert!(
            have.contains(ph.reads),
            "stage pipeline: phase '{}' reads a field not yet written this stage",
            ph.name,
        );
        match ph.kind {
            PhaseKind::SnapshotStage => prof("snapshot_stage", || {
                // marked before the copy: the snapshot resolves its own destination buffer
                // through `stage_input()`, so the flag has to hold by the time it runs. this
                // phase is the statement that this stage's input is captured.
                sim.mark_stage_input_captured();
                kernels.snapshot_stage(sim);
            }),
            PhaseKind::WaveSpeeds => kernels.wave_speeds(sim),
            PhaseKind::Flux => {
                for dd in 0..D {
                    prof("flux", || kernels.flux(sim, dd));
                }
                // the interface dye flux belongs to this phase, not the dye update: the
                // coarse-fine registers sample every stored flux at the hook below, so a dye
                // flux written later would be sampled one phase stale and reflux the wrong
                // mismatch.
                prof("chi_flux", || kernels.chi_flux(sim));
                hook(HookPoint::AfterFlux);
            }
            PhaseKind::Efield => prof("efield", || kernels.efield(sim)),
            PhaseKind::Godunov => prof("godunov_stage", || {
                kernels.godunov_stage(sim, args.dt, args.a0, args.ac)
            }),
            PhaseKind::PostGodunov => {
                prof("post_godunov", || kernels.post_godunov(sim, args.dt, tag))
            }
            PhaseKind::SourceApply => prof("source_apply", || {
                kernels.source_apply(sim, args.ac * args.dt)
            }),
            PhaseKind::BodySource => prof("body_source", || {
                kernels.body_source(sim, args.ac * args.dt)
            }),
            PhaseKind::C2p => {
                hook(HookPoint::BeforeC2p);
                prof("c2p", || kernels.c2p(sim));
            }
            PhaseKind::Fofc => {
                if prof("fofc", || kernels.fofc(sim, args.dt, args.a0, args.ac, tag)) {
                    return StageOutcome::RetryStep;
                }
            }
            PhaseKind::ChiUpdate => prof("chi_update", || {
                kernels.chi_update(sim, args.dt, args.a0, args.ac)
            }),
            PhaseKind::GhostFill => {
                hook(HookPoint::BeforeGhostFill);
                prof("ghost_fill", || kernels.ghost_fill(sim));
            }
        }
        have = have.or(ph.writes);
    }
    StageOutcome::Accepted
}
