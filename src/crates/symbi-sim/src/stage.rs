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
//  fold_stage(sim, kernels,
//             StageArgs { dt, a0, ac, stage, n_stages, injection_weight, allow_elision, exit_fraction },
//             &mut |_hp: HookPoint| {});
// =============================================================================

use crate::driver::{prof, stage_tag};
use crate::state::{ConsFieldsGeneric, FieldStore};
use symbi_grid::Field;
use crate::substrate_seam::KernelSet;
use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_xpu::MemoryStorage;

/// arms `workspace.stage_writes` for one stage and disarms it on drop, so the audit is scoped to
/// the stage pipeline rather than to the process. a caller that drives a substrate entry point
/// directly — a harness that seeds flux buffers by hand and then invokes one kernel — is outside
/// any stage, and "was this written earlier in the stage" has no meaning there.
#[cfg(debug_assertions)]
struct StageWriteAudit<'a, const D: usize, const DOF: usize, Mem, Sc>(
    &'a FieldStore<D, DOF, Mem, Sc>,
)
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric;

#[cfg(debug_assertions)]
impl<'a, const D: usize, const DOF: usize, Mem, Sc> StageWriteAudit<'a, D, DOF, Mem, Sc>
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
{
    fn arm(sim: &'a FieldStore<D, DOF, Mem, Sc>) -> Self {
        *sim.workspace.stage_writes.lock().unwrap() = Some(Default::default());
        Self(sim)
    }

    /// take over the ledger a stage's first span left armed, so the second span's reads are
    /// checked against the first span's writes and the ledger disarms when the stage ends.
    fn resume(sim: &'a FieldStore<D, DOF, Mem, Sc>) -> Self {
        Self(sim)
    }
}

#[cfg(debug_assertions)]
impl<const D: usize, const DOF: usize, Mem, Sc> Drop for StageWriteAudit<'_, D, DOF, Mem, Sc>
where
    Mem: MemoryStorage,
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
    StageEmf,
    AxisEmf,
    StageCurl,
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
    // the constrained-transport update in three phases: the stage's complete edge EMF, the
    // edges on a coordinate axis reconciled to one value per ring, then the curl. a driver
    // holding a ring over several tiles folds the stage apart at the reconciliation and
    // reconciles across its tiles in between.
    Phase {
        name: "stage_emf",
        kind: PhaseKind::StageEmf,
        reads: FieldSet::CONS,
        writes: FieldSet::NONE,
        gate: Gate::Always,
    },
    Phase {
        name: "axis_emf",
        kind: PhaseKind::AxisEmf,
        reads: FieldSet::NONE,
        writes: FieldSet::NONE,
        gate: Gate::Always,
    },
    Phase {
        name: "stage_curl",
        kind: PhaseKind::StageCurl,
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
    /// after ghost_fill: the hierarchy completes the ghost cells a refined level's physical fill
    /// leaves to its coarse-fine sides, where a periodic axis meets one.
    AfterGhostFill,
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
    /// the downstream shu-osher propagation weight of this stage's output —
    /// `driver::downstream_injection_weight(stages, stage)` on the same table
    /// that supplied `(a0, ac)`. a conserved delta a fallback injects into
    /// this stage's output reaches the accepted step scaled by this factor.
    pub injection_weight: f64,
    /// whether this driver may elide the stage-0 stage-input copy (the
    /// per-step snapshot already holds it). the decomposed loop passes false —
    /// it never tracks the alias.
    pub allow_elision: bool,
    /// the time fraction of the step this stage's output stands at
    /// (`driver::stage_exit_fraction`): the ghost refill that closes the stage evaluates a
    /// time-dependent boundary there.
    pub exit_fraction: f64,
}

/// the freeze count of one level that a tiled march agrees after every correction pass, so the FOFC
/// halt reads the level-wide streak a single grid reads: the consecutive substages that ended with a
/// freeze outside any excision region anywhere on the level. it advances on every executed
/// substage, a rejected attempt's included, and starts at zero in each session. every tile's kernel
/// holds the agreed count between passes, so a tile that freezes advances it to the level-wide
/// value and makes the halt and retry decisions there. `prior` is the count the tiles held before
/// the pass, `any_froze` whether any tile of the level, on any worker, left it with a nonzero count.
pub fn agreed_freeze_streak(prior: u32, any_froze: bool) -> u32 {
    if any_froze { prior + 1 } else { 0 }
}

#[must_use]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageOutcome {
    Accepted,
    RetryStep,
}

/// the part of the canonical pipeline one fold call runs. a decomposed driver folds a stage in
/// two spans so that every tile's troubled cells are marked before any tile corrects: the
/// flags cross the cuts between the spans, and a face two tiles share takes one flux.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StageSpan {
    Whole,
    /// every phase before the correction, then the troubled-cell mark
    ThroughMark,
    /// the correction and every phase after it
    FromCorrection,
    /// every phase through the stage's complete edge EMF, before its axis rings are reconciled
    ThroughStageEmf,
    /// the curl and every phase after it before the correction, then the troubled-cell mark
    FromStageCurlThroughMark,
}

/// fold one RK stage of the canonical pipeline over a kernel set.
pub fn fold_stage<const D: usize, const DOF: usize, Mem, Sc, K>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    kernels: &K,
    args: StageArgs,
    hook: &mut impl FnMut(HookPoint),
) -> StageOutcome
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
    K: KernelSet<D, DOF, Mem, Sc> + ?Sized,
{
    fold_span(sim, kernels, args, StageSpan::Whole, hook).0
}

/// the first span of a stage folded in two: every phase up to the conserved-to-primitive
/// recovery, then the troubled-cell mark. returns the number of troubled interior cells, zero
/// for a kernel set that runs no correction. `fold_stage_from_correction` must follow.
pub fn fold_stage_through_mark<const D: usize, const DOF: usize, Mem, Sc, K>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    kernels: &K,
    args: StageArgs,
    hook: &mut impl FnMut(HookPoint),
) -> u64
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
    K: KernelSet<D, DOF, Mem, Sc> + ?Sized,
{
    fold_span(sim, kernels, args, StageSpan::ThroughMark, hook).1
}

/// the first span of a stage folded in three: every phase through the stage's complete edge EMF.
/// a driver whose tiles share an axis ring reconciles the ring across them before
/// `fold_stage_curl_through_mark` continues the stage.
pub fn fold_stage_through_emf<const D: usize, const DOF: usize, Mem, Sc, K>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    kernels: &K,
    args: StageArgs,
    hook: &mut impl FnMut(HookPoint),
) where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
    K: KernelSet<D, DOF, Mem, Sc> + ?Sized,
{
    let _ = fold_span(sim, kernels, args, StageSpan::ThroughStageEmf, hook);
}

/// the second span of a stage folded in three: the curl of the reconciled EMF and every phase
/// before the correction, then the troubled-cell mark. returns the number of troubled interior
/// cells; `fold_stage_from_correction` must follow.
pub fn fold_stage_curl_through_mark<const D: usize, const DOF: usize, Mem, Sc, K>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    kernels: &K,
    args: StageArgs,
    hook: &mut impl FnMut(HookPoint),
) -> u64
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
    K: KernelSet<D, DOF, Mem, Sc> + ?Sized,
{
    fold_span(sim, kernels, args, StageSpan::FromStageCurlThroughMark, hook).1
}

/// the second span of a stage folded in two: the correction and every phase after it.
/// `cut_trouble` says a neighboring tile's troubled cell lies in this tile's first ghost layer,
/// whose flags the driver has already written into the cut ghosts; the correction then runs
/// here even with a clean interior.
pub fn fold_stage_from_correction<const D: usize, const DOF: usize, Mem, Sc, K>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    kernels: &K,
    args: StageArgs,
    cut_trouble: bool,
    hook: &mut impl FnMut(HookPoint),
) -> StageOutcome
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
    K: KernelSet<D, DOF, Mem, Sc> + ?Sized,
{
    let mark = &sim.workspace.fofc_cut_trouble;
    mark.store(cut_trouble, std::sync::atomic::Ordering::Relaxed);
    let outcome = fold_span(sim, kernels, args, StageSpan::FromCorrection, hook).0;
    mark.store(false, std::sync::atomic::Ordering::Relaxed);
    sim.workspace
        .fofc_marked
        .store(false, std::sync::atomic::Ordering::Relaxed);
    outcome
}

/// the store's one-shot troubled cell: at the mark of the stage it names, on the interior, the
/// cell's troubled flag is raised while its state stays physical, so the correction it drives is
/// the conservative first-order redo. the slot is marked fired; returns the cells it raised.
fn raise_troubled<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>, stage: usize) -> u64
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
{
    let mut slot = sim.workspace.troubled_once.lock().expect("the troubled-cell slot");
    let Some((iteration, at_stage, cell, fired)) = *slot else { return 0 };
    if fired || (iteration, at_stage) != (sim.iteration, stage) {
        return 0;
    }
    *slot = Some((iteration, at_stage, cell, true));
    if !sim.geom.interior.contains(cell) {
        return 0;
    }
    raise_troubled_cell(sim, cell)
}

/// raises the troubled flag of `cell` with its state left physical, so the correction it drives is
/// the conservative first-order redo. the one writer of the flag outside the mark; returns 1 when
/// the flag was clear and 0 when the mark had already raised it.
pub(crate) fn raise_troubled_cell<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>, cell: [isize; D]) -> u64
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
{
    let flag = &sim.workspace.fofc_flag;
    if flag.read_one(cell).expect("the troubled flag transfer") > Sc::from_f64(0.0) {
        return 0;
    }
    flag.write_one(cell, Sc::from_f64(1.0)).expect("the troubled flag transfer");
    1
}

/// the store's scheduled unrecoverable cell for this stage, applied at `point`: a deeply
/// negative energy (the density, on a regime without an energy field) into the stage input after the
/// fluxes, at the cell wherever the store holds it, halo included, so every tile sharing the cell
/// restarts its redo from the same input; and into the stage output before the recovery, on the
/// interior, where its entry is marked fired.
fn apply_unrecoverable<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>, stage: usize, point: HookPoint)
where
    Mem: MemoryStorage,
    Sc: Scalar + OrderedNumeric,
{
    let mut slots = sim.workspace.unrecoverable_once.lock().expect("the unrecoverable-cell slots");
    let Some(slot) = slots.iter_mut().find(|(iteration, at_stage, _, fired)| !fired && (*iteration, *at_stage) == (sim.iteration, stage)) else {
        return;
    };
    let (iteration, at_stage, cell, _) = *slot;
    // the poisoned conserved: the total energy, or the density of a regime without an energy field
    fn poisoned<const D: usize, const DOF: usize, Mem: MemoryStorage, Sc: Scalar + OrderedNumeric>(
        u: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    ) -> &Field<Sc, D, Mem> {
        u.nrg_field().unwrap_or(&u.den)
    }
    let field = match point {
        HookPoint::AfterFlux => poisoned(sim.stage_input()),
        HookPoint::BeforeC2p => {
            *slot = (iteration, at_stage, cell, true);
            // the stage output carries state on the interior alone
            if !sim.geom.interior.contains(cell) {
                return;
            }
            poisoned(&sim.fields.cons)
        }
        HookPoint::BeforeGhostFill | HookPoint::AfterGhostFill => return,
    };
    field.write_one(cell, Sc::from_f64(UNRECOVERABLE_POISON)).expect("the poisoned value transfer");
}

/// the poisoned value: a thousand times the O(1) density and energy of a test flow, negative. a
/// shallow poison (-1) sits within reach of the first-order redo, whose stencil fluxes, driven by the
/// unphysical cell itself, can carry it back into the admissible set; at this depth no flux divergence
/// over one stage restores it, so the redo lands outside and the correcting select freezes the cell.
const UNRECOVERABLE_POISON: f64 = -1.0e3;

fn fold_span<const D: usize, const DOF: usize, Mem, Sc, K>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    kernels: &K,
    args: StageArgs,
    span: StageSpan,
    hook: &mut impl FnMut(HookPoint),
) -> (StageOutcome, u64)
where
    Mem: MemoryStorage,
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
    // a stage a scheduled unrecoverable cell arms takes its own snapshot, so the negative energy
    // lands in the redo's input alone and leaves the step snapshot u_n whole, on every driver
    let unrecoverable_armed = sim
        .workspace
        .unrecoverable_once
        .lock()
        .expect("the unrecoverable-cell slots")
        .iter()
        .any(|&(iteration, stage, _, _)| (iteration, stage) == (sim.iteration, args.stage));
    let stage_input_is_un = args.stage == 0
        && args.n_stages > 1
        && args.allow_elision
        && !unrecoverable_armed
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
    // the redo's normal-B input, captured before any phase runs. a stage folded in two
    // captures in its first span.
    if fofc && matches!(span, StageSpan::Whole | StageSpan::ThroughMark | StageSpan::ThroughStageEmf) {
        prof("snapshot_stage_faces", || kernels.snapshot_stage_faces(sim));
    }

    // a stage folded in two keeps one ledger: the first span leaves it armed and the second
    // takes it over, so the correction's flux reads are checked against the first span's writes.
    #[cfg(debug_assertions)]
    let audit = match span {
        StageSpan::Whole | StageSpan::ThroughMark | StageSpan::ThroughStageEmf => StageWriteAudit::arm(sim),
        StageSpan::FromCorrection | StageSpan::FromStageCurlThroughMark => StageWriteAudit::resume(sim),
    };

    let tag = stage_tag(args.stage, args.n_stages);
    let mut have = match span {
        StageSpan::Whole | StageSpan::ThroughMark | StageSpan::ThroughStageEmf => FieldSet::CONS.or(FieldSet::PRIM),
        StageSpan::FromCorrection | StageSpan::FromStageCurlThroughMark => FieldSet::CONS
            .or(FieldSet::PRIM)
            .or(FieldSet::FLUX)
            .or(FieldSet::USTAGE),
    };
    let mut reached_correction = false;
    let mut reached_curl = false;
    for ph in STAGE_PIPELINE {
        if ph.kind == PhaseKind::AxisEmf && span == StageSpan::ThroughStageEmf {
            #[cfg(debug_assertions)]
            std::mem::forget(audit);
            return (StageOutcome::Accepted, 0);
        }
        if ph.kind == PhaseKind::StageCurl {
            reached_curl = true;
        }
        if span == StageSpan::FromStageCurlThroughMark && !reached_curl {
            continue;
        }
        if ph.kind == PhaseKind::Fofc {
            reached_correction = true;
            if matches!(span, StageSpan::ThroughMark | StageSpan::FromStageCurlThroughMark) {
                let troubled = if fofc { kernels.fofc_mark(sim) + raise_troubled(sim, args.stage) } else { 0 };
                sim.workspace
                    .fofc_marked
                    .store(fofc, std::sync::atomic::Ordering::Relaxed);
                #[cfg(debug_assertions)]
                std::mem::forget(audit);
                return (StageOutcome::Accepted, troubled);
            }
        }
        if span == StageSpan::FromCorrection && !reached_correction {
            continue;
        }
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
                apply_unrecoverable(sim, args.stage, HookPoint::AfterFlux);
                hook(HookPoint::AfterFlux);
            }
            PhaseKind::Efield => prof("efield", || kernels.efield(sim)),
            PhaseKind::Godunov => prof("godunov_stage", || {
                kernels.godunov_stage(sim, args.dt, args.a0, args.ac)
            }),
            PhaseKind::StageEmf => prof("stage_emf", || kernels.stage_emf(sim, args.dt, tag)),
            PhaseKind::AxisEmf => prof("axis_emf", || kernels.axis_emf(sim)),
            PhaseKind::StageCurl => prof("stage_curl", || kernels.stage_curl(sim, args.dt, tag)),
            PhaseKind::SourceApply => prof("source_apply", || {
                kernels.source_apply(sim, args.ac * args.dt)
            }),
            PhaseKind::BodySource => prof("body_source", || {
                kernels.body_source(sim, args.ac * args.dt)
            }),
            PhaseKind::C2p => {
                apply_unrecoverable(sim, args.stage, HookPoint::BeforeC2p);
                hook(HookPoint::BeforeC2p);
                prof("c2p", || kernels.c2p(sim));
            }
            PhaseKind::Fofc => {
                // the single fold of the ladder's decision into the stage outcome; the
                // report's counts already flowed to the census as observations.
                // the redo's update passes the same composite step the first span's update passed
                let mut redo_update = || hook(HookPoint::BeforeC2p);
                let report = prof("fofc", || {
                    kernels.fofc(sim, args.dt, args.a0, args.ac, tag, &mut redo_update)
                });
                // book the projection's receipt into the sim-owned ledger — the
                // substrate returns it, the sim owns the transaction. a retried
                // pass books into the pending bucket the driver discards, so its
                // intervention stays in the attempted totals without surviving
                // into the accepted solution. only the single-grid driver opens
                // the ledger scope, so decomposed and refined runs book nothing.
                if crate::projection_ledger::scope_is_open() {
                    if let Some(receipt) = report.receipt() {
                        crate::projection_ledger::record(&receipt, args.injection_weight);
                    }
                }
                // book the guard acts into the sim-owned guard ledger. the driver
                // opens the guard scope for every run shape, so a retried pass
                // books into the pending bucket the driver discards, keeping its
                // acts in the attempted totals without surviving into the accepted
                // solution.
                if crate::guard_ledger::scope_is_open() {
                    crate::guard_ledger::record(&report.guards());
                }
                if report.decision() == crate::substrate_seam::FofcDecision::RetryStep {
                    return (StageOutcome::RetryStep, report.troubled());
                }
            }
            PhaseKind::ChiUpdate => prof("chi_update", || {
                kernels.chi_update(sim, args.dt, args.a0, args.ac)
            }),
            PhaseKind::GhostFill => {
                hook(HookPoint::BeforeGhostFill);
                // the stage carried the state to its output time; the last stage's output
                // stands at the end of the step, the clock the step advances to
                let t_out = if args.stage + 1 == args.n_stages {
                    sim.time + args.dt
                } else {
                    sim.time + args.exit_fraction * args.dt
                };
                sim.carry_state_clock(t_out);
                prof("ghost_fill", || kernels.ghost_fill(sim));
                hook(HookPoint::AfterGhostFill);
            }
        }
        if crate::state::positivity_trace_enabled() {
            sim.host_only("the positivity trace")
                .positivity_trace(&format!("stage={} phase={}", args.stage, ph.name));
        }
        have = have.or(ph.writes);
    }
    (StageOutcome::Accepted, 0)
}

// =============================================================================
// the post-step recipe: the operators every driver applies once per step after the RK
// combination, in the method's order, each declaring what it reads and writes and whether
// it leaves the cut halos stale. one executor folds it; the driver supplies the execution
// context, which places each tile's work, combines a per-tile count, and refreshes the halos
// where the recipe says an operation invalidated them.
// =============================================================================

/// a post-step operator
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepOp {
    /// the constant-nu or alpha viscous transport: reads the primitive velocity one cell into
    /// the halo and writes the interior momentum (and energy on an energy-carrying regime),
    /// so the primitives it leaves predate the conserved state until they are recovered
    Viscous,
    /// horizon excision: the fill sweeps read the rim's primitives, each writes the excised
    /// cells, and the rebuild recovers their conserved state; a neighbor's rim read needs the
    /// refreshed halo after every sweep and after the rebuild
    Excise,
}

/// whether the primitives must be recovered from the conserved state after an operation:
/// an operation that writes the conserved interior leaves the stored primitives inconsistent
/// with it, and the next reconstruction and the next checkpoint read the primitives
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recover {
    None,
    Primitives,
}

/// whether an operation leaves every tile's cut halo stale for the next reader: once after
/// it, or after every one of its passes. a refresh moves the primitives the tiles hold, so
/// it follows the recovery; an exchange of stale primitives satisfies nothing
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refresh {
    None,
    Halos,
    HalosPerPass,
}

pub struct StepPhase {
    pub name: &'static str,
    pub op: StepOp,
    pub reads: FieldSet,
    pub writes: FieldSet,
    pub recover: Recover,
    pub refresh: Refresh,
    /// the property a driver names when it refuses the operation
    pub property: &'static str,
}

impl StepPhase {
    /// whether the operation acts on `store` under `kernel` at all; a phase that does not act
    /// runs none of its recovery or refresh, so an inviscid, unexcised run is untouched
    pub fn applies<const D: usize, const DOF: usize, Mem, K>(
        &self,
        store: &FieldStore<D, DOF, Mem, f64>,
        kernel: &K,
    ) -> bool
    where
        Mem: MemoryStorage,
        K: KernelSet<D, DOF, Mem, f64> + ?Sized,
    {
        match self.op {
            StepOp::Viscous => kernel.is_viscous(),
            StepOp::Excise => kernel.excise_pass_count(store) > 0,
        }
    }
}

/// the canonical post-step order. this list is the sequence; every driver folds it.
pub const POST_STEP_PIPELINE: &[StepPhase] = &[
    StepPhase {
        name: "viscous",
        op: StepOp::Viscous,
        reads: FieldSet::PRIM,
        writes: FieldSet::CONS,
        recover: Recover::Primitives,
        refresh: Refresh::Halos,
        property: "viscous transport",
    },
    StepPhase {
        name: "excise",
        op: StepOp::Excise,
        reads: FieldSet::PRIM,
        writes: FieldSet::CONS.or(FieldSet::PRIM),
        recover: Recover::None,
        refresh: Refresh::HalosPerPass,
        property: "horizon excision",
    },
];

/// a post-step operation that left an interior cell unrecoverable: the step was accepted, so
/// the driver ends the run coherently with the phase and the cell named rather than publish
/// the state
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostStepError {
    pub phase: &'static str,
    pub detail: String,
}

impl std::fmt::Display for PostStepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the {} pass left the accepted state unrecoverable: {}", self.phase, self.detail)
    }
}

/// the refresh of the cut halos a post-step operation asks for: after a phase's operation and
/// recovery, after excision sweep `pass`, or after the excision rebuild. a driver whose
/// exchanges are named points maps each to its own point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefreshAt {
    Phase,
    ExciseSweep(usize),
    ExciseRebuild,
}

/// what a driver supplies under the post-step recipe: where each tile's work runs, how a
/// per-tile count is combined across every tile the step spans, and how the cut halos are
/// made current after an operation wrote its interior.
pub trait StepContext<const D: usize, const DOF: usize, Mem: MemoryStorage> {
    /// run `f` where tile `k` lives
    fn on_tile(&self, k: usize, f: &mut dyn FnMut());
    /// the largest of a per-tile count over every tile the step spans
    fn combine_max(&mut self, local: usize) -> usize;
    /// every tile's cut halo made current from its neighbors' interiors, for the refresh `at`
    fn refresh_halos(&mut self, stores: &[&FieldStore<D, DOF, Mem, f64>], at: RefreshAt);
}

/// the context of a step over one grid: the work runs in place, a count is its own maximum,
/// and a halo refresh is the grid's own physical fill, which the next stage runs anyway
pub struct SingleGridStep;

impl<const D: usize, const DOF: usize, Mem: MemoryStorage> StepContext<D, DOF, Mem> for SingleGridStep {
    fn on_tile(&self, _k: usize, f: &mut dyn FnMut()) {
        f()
    }
    fn combine_max(&mut self, local: usize) -> usize {
        local
    }
    fn refresh_halos(&mut self, _stores: &[&FieldStore<D, DOF, Mem, f64>], _at: RefreshAt) {}
}

/// fold the canonical post-step recipe over `stores`, `select` choosing the operators this
/// fold runs (a refined level runs the viscous transport on its finest levels alone)
pub fn fold_post_step<const D: usize, const DOF: usize, Mem, K>(
    stores: &[&FieldStore<D, DOF, Mem, f64>],
    kernels: &[&K],
    dt: f64,
    cx: &mut impl StepContext<D, DOF, Mem>,
    select: impl Fn(StepOp) -> bool,
) -> Result<(), PostStepError>
where
    Mem: MemoryStorage,
    K: KernelSet<D, DOF, Mem, f64> + ?Sized,
{
    fold_post_step_with(POST_STEP_PIPELINE, stores, kernels, dt, cx, select)
}

/// the excision at the end of an accepted stage: the post-step recipe's excision phase, the
/// same sweeps, rebuild and mask, folded through the driver's context so a tiled driver refreshes
/// its cut halos after every sweep and after the rebuild, then every tile's physical ghosts
/// refilled from the filled interior. the recovery writes its candidate primitives on a rejected
/// cell, so without this fill an excised cell rejected in this stage would carry that candidate
/// into the next stage's flux sweep; with it, every flux sweep and the first-order redo's stage
/// input read the vacuum floor across the excised region. the fill writes the live conserved and
/// primitive fields alone: the step-entry and stage-input snapshots keep their meaning, and the
/// staggered faces stay with the constrained transport. the sweep count is combined over every
/// tile the step spans, so a step with no excision region anywhere issues no call, and a tile
/// without one takes every refresh its neighbors take.
pub fn fold_stage_excision<const D: usize, const DOF: usize, Mem, K>(
    stores: &[&FieldStore<D, DOF, Mem, f64>],
    kernels: &[&K],
    cx: &mut impl StepContext<D, DOF, Mem>,
) -> Result<(), PostStepError>
where
    Mem: MemoryStorage,
    K: KernelSet<D, DOF, Mem, f64> + ?Sized,
{
    let local = (0..stores.len()).map(|k| kernels[k].excise_pass_count(stores[k])).max().unwrap_or(0);
    if cx.combine_max(local) == 0 {
        return Ok(());
    }
    fold_post_step(stores, kernels, 0.0, cx, |op| op == StepOp::Excise)?;
    for k in 0..stores.len() {
        cx.on_tile(k, &mut || prof("ghost_fill", || kernels[k].ghost_fill(stores[k])));
    }
    Ok(())
}

/// fold a post-step table over `stores`. a phase that acts on a tile runs its operation on
/// every tile, then, as the table declares, recovers every tile's interior primitives from
/// the conserved state and fills its physical ghosts, then refreshes the cut halos through
/// the context, in that order, so the halos the tiles exchange are the recovered primitives.
/// a recovery that leaves an interior cell unrecoverable ends the fold with the cell named.
pub fn fold_post_step_with<const D: usize, const DOF: usize, Mem, K>(
    table: &[StepPhase],
    stores: &[&FieldStore<D, DOF, Mem, f64>],
    kernels: &[&K],
    dt: f64,
    cx: &mut impl StepContext<D, DOF, Mem>,
    select: impl Fn(StepOp) -> bool,
) -> Result<(), PostStepError>
where
    Mem: MemoryStorage,
    K: KernelSet<D, DOF, Mem, f64> + ?Sized,
{
    let n = stores.len();
    for phase in table {
        if !select(phase.op) {
            continue;
        }
        match phase.op {
            StepOp::Viscous => {
                if !(0..n).any(|k| phase.applies(stores[k], kernels[k])) {
                    continue;
                }
                for k in 0..n {
                    cx.on_tile(k, &mut || prof("viscous", || kernels[k].viscous(stores[k], dt)));
                }
                if phase.recover == Recover::Primitives {
                    recover_primitives(phase, stores, kernels, cx)?;
                }
                if phase.refresh != Refresh::None {
                    cx.refresh_halos(stores, RefreshAt::Phase);
                }
            }
            // the sweep count is combined over every tile the step spans, so a tile without an
            // excision region takes every refresh its neighbors take
            StepOp::Excise => {
                let local = (0..n).map(|k| kernels[k].excise_pass_count(stores[k])).max().unwrap_or(0);
                let passes = cx.combine_max(local);
                if passes == 0 {
                    continue;
                }
                for pass in 0..passes {
                    for k in 0..n {
                        cx.on_tile(k, &mut || kernels[k].excise_sweep(stores[k]));
                    }
                    if phase.refresh == Refresh::HalosPerPass {
                        cx.refresh_halos(stores, RefreshAt::ExciseSweep(pass));
                    }
                }
                for k in 0..n {
                    cx.on_tile(k, &mut || kernels[k].excise_finalize(stores[k]));
                }
                if phase.refresh != Refresh::None {
                    cx.refresh_halos(stores, RefreshAt::ExciseRebuild);
                }
            }
        }
    }
    Ok(())
}

/// every tile's interior primitives recovered from its conserved state and its physical
/// ghosts filled; an interior cell the recovery cannot land ends the fold
fn recover_primitives<const D: usize, const DOF: usize, Mem, K>(
    phase: &StepPhase,
    stores: &[&FieldStore<D, DOF, Mem, f64>],
    kernels: &[&K],
    cx: &mut impl StepContext<D, DOF, Mem>,
) -> Result<(), PostStepError>
where
    Mem: MemoryStorage,
    K: KernelSet<D, DOF, Mem, f64> + ?Sized,
{
    for k in 0..stores.len() {
        cx.on_tile(k, &mut || prof("c2p", || kernels[k].c2p(stores[k])));
        let store = stores[k].host_only("the post-step recovery check");
        let code = crate::hydro_ops::scan_c2p_errors(store);
        if !code.is_ok() {
            let first = crate::hydro_ops::first_c2p_error(store)
                .map(|(coord, code)| format!("cell {coord:?} code {code}"))
                .unwrap_or_else(|| "unavailable".to_string());
            return Err(PostStepError { phase: phase.name, detail: format!("tile {k}, {first}") });
        }
        cx.on_tile(k, &mut || prof("ghost_fill", || kernels[k].ghost_fill(stores[k])));
    }
    Ok(())
}

