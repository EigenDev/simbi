// =============================================================================
// hierarchy.rs
//
// the fixed mesh-refinement hierarchy. each
// level is a complete SimStateGeneric + its KernelSet; the hierarchy adds only
// inter-level coordination: recursive berger-oliger subcycling with
// time-interpolated coarse-fine ghost prolongation, conservative restriction,
// and flux-register refluxing. the single-level engine is untouched —
//
// single-coverage cap: each level refines one box
// (`coverage: Option<Domain>`), with one FluxRegister per coarse-fine
// level-pair. the refined region is fixed at setup and stays where it was
// placed; there is a single patch per level and no berger-rigoutsos
// clustering. multi-patch adaptive refinement (a level as a disjoint cover of
// Domains) lies outside this implementation.
//
// advance_level re-sequences the SSP stage loop (sim/evolve.rs::step) so the
// register accumulation slots between flux() and the stage update.
//
// levels share absolute index space: a fine level covering coarse cells
// [cov_lo, cov_hi) lives at fine indices [2*cov_lo, 2*cov_hi), and every level
// keeps the same global physical origin — `geom.centroid` is correct on every
// level with the same formula, and no coverage-relative translation exists.
//
// two invariants pin the construction: a 1-level hierarchy reproduces the uni-grid
// evolve() bit-for-bit, and a 2-level static nesting conserves the composite-grid
// totals to machine precision.
//
// usage:
//  let mut hier = Hierarchy::with_refinement(sim, kernels, &regions, order, make)?;
//  hier.evolve(t_final)?;
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_geometry::Metric;

// the magnetic-slip coupled-step schedule spy: when armed, the palindromic driver records each
// sub-operation's name and duration, so a test can prove the composition order and durations without
// running a numerical step. off by default (a None recorder), so production carries no cost.
thread_local! {
    static SLIP_SCHEDULE: std::cell::RefCell<Option<Vec<(&'static str, f64)>>> =
        const { std::cell::RefCell::new(None) };
}

/// arm the schedule spy: subsequent coupled steps record their sub-operation trace.
pub fn slip_schedule_arm() {
    SLIP_SCHEDULE.with(|s| *s.borrow_mut() = Some(Vec::new()));
}

/// take and disarm the recorded schedule trace, if armed.
pub fn slip_schedule_take() -> Option<Vec<(&'static str, f64)>> {
    SLIP_SCHEDULE.with(|s| s.borrow_mut().take())
}

fn slip_schedule_record(op: &'static str, dt: f64) {
    SLIP_SCHEDULE.with(|s| {
        if let Some(v) = s.borrow_mut().as_mut() {
            v.push((op, dt));
        }
    });
}

/// a point of the coupled root step at which an armed failure fires once, so a retry gate can
/// reject an attempt exactly there and check that the rollback restores every level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlipFailPoint {
    /// after the opening D-M pair and its synchronization, before the ideal-MHD step.
    AfterOpening,
    /// inside the ideal-MHD step, at the finest level's fine substep of this ordinal (1-based)
    /// within the current root attempt.
    FineSubstep(usize),
    /// after the closing magnetic solve.
    ClosingSolve,
}

thread_local! {
    static SLIP_FAILURE: std::cell::RefCell<Option<SlipFailPoint>> = const { std::cell::RefCell::new(None) };
    static SLIP_FINE_SUBSTEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// arm one failure of the coupled root step at `point`; it fires once and disarms.
pub fn slip_failure_arm(point: SlipFailPoint) {
    SLIP_FAILURE.with(|f| *f.borrow_mut() = Some(point));
}

fn slip_failure_fires(point: SlipFailPoint) -> bool {
    SLIP_FAILURE.with(|f| {
        let mut armed = f.borrow_mut();
        if *armed == Some(point) {
            *armed = None;
            true
        } else {
            false
        }
    })
}
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

use symbi_grid::Field;

use super::emf_register::EmfRegister;
use super::equilibrium::{
    EquilibriumFlux, GasState, accumulate as equilibrium_accumulate, imbalance_from_stage,
    overwrite as equilibrium_overwrite, residual_components, restore_gas_state, save_gas_state,
    snapshot_flux,
};
use super::flux_register::FluxRegister;
use super::tracer_interface::{
    InterfaceFace, InterfaceTransfer, interface_faces, interface_faces_with_layout,
    interface_mass_transfers, interface_transport_kernels,
};
use super::transfer::{face_stencil_boxes, 
    ProlongOrder, ProlongSweepScratch, bcell_from_bface_region, bface_cf_halo_slabs,
    cf_ghost_slabs, clip_to_prolongable, copy_field, copy_field_region, prolong_face_field,
    prolong_field,
    prolong_prims_balanced, prolong_prims_swept, prolong_prims_targeted, restrict_band_balanced,
    restrict_bface, restrict_cell_field, restrict_cons,
};
use std::ops::ControlFlow;

use super::phase_probe;
use symbi_hydro::state::PrimFromSlots;
use symbi_sim::decomp::{
    HaloTransport, Topology, drain_devices, exchange_grid, flatten, schedule_of_tiles, unflatten,
};
use symbi_sim::collective_stage::StageCollective;
use symbi_sim::step_transaction::{RootAttempt, RootStep, StepHalt, take_root_step};
use symbi_sim::driver::{
    TimestepFailure, advance_clock, advance_state_clock, agree_timestep, book_horizon_receipt, evolve_bodies,
    horizon_request, needs_step_snapshot, prof, stage_time_fractions,
};
use symbi_sim::hydro_ops::scan_c2p_errors;
use symbi_sim::stage::{HookPoint, StageArgs, fold_stage};
use symbi_sim::state::{
    Boundaries, BoundaryType, ConsFieldsGeneric, FieldDecimation, FieldKind, FieldStore,
    PrimFieldsGeneric, SimStateGeneric, Timestepping, array_field_zeros, axis_name,
};
use symbi_sim::substrate_seam::KernelSet;

/// the refinement ratio. fixed at 2 (the transfer kernels are baked at 2 in
/// the aot registry; the builders accept any ratio when that changes).
pub const RATIO: usize = 2;

// =============================================================================
// refinement regions
// =============================================================================

/// a box in physical space that one finer level will cover (static nesting:
/// region k refines level k). snapped to coarse cell faces by rounding.
#[derive(Clone, Copy, Debug)]
pub struct RefinementRegion<const D: usize> {
    pub x_lo: [f64; D],
    pub x_hi: [f64; D],
}

// =============================================================================
// per-level data
// =============================================================================

/// one refinement level: the state, the kernel set that advances it, and the
/// inter-level metadata (present only where a finer level exists).
/// the parent faces a fine level's coarse-fine face halo prolongation reads, per component,
/// at the two time levels of the root step: `old` at the root attempt's entry, `new` after the
/// root stages. derived state: gathered once per attempt after the root stages, before the
/// first substep's prolongation, and invalid from the attempt's entry until then
pub struct FaceStencil<const D: usize, Mem: MemorySpace> {
    pub old: Vec<Field<f64, D, Mem>>,
    pub new: Vec<Field<f64, D, Mem>>,
    filled: std::cell::Cell<bool>,
}

impl<const D: usize, Mem: MemorySpace> FaceStencil<D, Mem> {
    pub fn domains(&self) -> Vec<Domain<D>> {
        self.old.iter().map(|f| f.domain().clone()).collect()
    }

    pub fn is_filled(&self) -> bool {
        self.filled.get()
    }
}

pub struct LevelData<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    pub state: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
    pub kernels: K,
    /// primitives at this level's step start, for the time-interpolated
    /// coarse-fine ghost prolongation of the finer level. None on the finest.
    pub prim_old: Option<PrimFieldsGeneric<NDIM, DOF, Mem>>,
    /// coarse scratch for the lerp-then-prolong split: `field_lerp` writes
    /// `(1-alpha)*prim_old + alpha*prim` here once per coarse cell, and the
    /// single-snapshot prolong reads it — half the time-pair kernel's gather
    /// traffic. allocated alongside prim_old (None on the finest) and reused on
    /// every call, so the step loop allocates nothing.
    pub prim_lerp: Option<PrimFieldsGeneric<NDIM, DOF, Mem>>,
    /// cell-centered B at this level's step start (mhd only), every component, the out-of-plane
    /// ones of a partially gridded chart included — the magnetic counterpart of prim_old for the
    /// finer level's ghost prolongation.
    pub bcell_old: Option<[Field<f64, NDIM, Mem>; DOF]>,
    /// staggered face B at this level's step start (mhd only) — feeds the
    /// finer level's bface transverse-halo prolongation (per-component
    /// staggered domains, cloned from this level's bface).
    pub bface_old: Option<[Field<f64, NDIM, Mem>; NDIM]>,
    /// the conserved state of a level with a coarser one at the root attempt's entry: the
    /// level's own step snapshot is rewritten by every substep, so a rejection after the
    /// first substep restores the attempt's entry from here
    pub attempt_entry: Option<ConsFieldsGeneric<NDIM, DOF, Mem>>,
    /// per axis and side, whether this level's boundary there is a decomposition cut the
    /// patch continues across: the halo beyond it comes from the level's own exchange, so
    /// the prolongation leaves it alone
    pub cut_sides: [[bool; 2]; NDIM],
    /// the whole patch this level refines, in its own index space: its interior on a single
    /// hierarchy, and past a cut the cells the neighbor tiles hold. the balanced coarse-fine
    /// transfer anchors each chain on the patch cell nearest the ghost, as the single hierarchy
    /// does, the cells past a cut read from the level's exchanged halo. `None` on the root
    pub patch_interior: Option<Domain<NDIM>>,
    /// the parent faces the coarse-fine face halo prolongation reads, gathered from the tiles
    /// that own them at both time levels of the root step: a tile's own face lattice ends at
    /// its interior along each face's normal axis, which a patch face on a cut reaches past
    pub face_stencil: Option<FaceStencil<NDIM, Mem>>,
    /// per-slab intermediates of the axis-split prolongation into this level,
    /// in `cf_ghost_slabs` order. the refinement slabs are fixed, so
    /// the shapes are too: lazily allocated on the first prolongation, reused
    /// on every call (the step loop allocates nothing). uninitialized on the
    /// root.
    pub prolong_sweep: std::sync::OnceLock<Vec<ProlongSweepScratch<NDIM, DOF, Mem>>>,
    /// this level's pressure departures from the composite-lattice equilibrium
    /// under its parent's seam bands — the balanced restriction's fine scratch.
    /// lazily allocated over this level's allocated domain on the first
    /// balanced restriction, reused on every call. unused on the root and on
    /// plain hierarchies.
    pub band_departure: std::sync::OnceLock<Field<f64, NDIM, Mem>>,
    /// the region of this level covered by the next finer level, in absolute
    /// indices of this level. None on the finest. single-coverage cap: one
    /// refined box per level (a single `Domain`,
    /// placed at setup).
    pub coverage: Option<Domain<NDIM>>,
    /// this level's numerical flux of the run's stationary target state, `F(qt)`, when the run
    /// declares one. the flux registers subtract it from both sides of the coarse-fine
    /// difference, so what is refluxed is the deviation from the target and a state sitting
    /// exactly on the target accumulates nothing. per level, because the mismatch the register
    /// would otherwise see is precisely the difference between the two grids' reconstructions of
    /// the same exact solution.
    pub flux_eq: Option<EquilibriumFlux<NDIM, DOF, Mem>>,
    /// this level's discrete imbalance of that same target, `R = div_h F_h(qt) - s_h(qt)`, per
    /// unit time and per cell. a steady state solves the continuum equations, so the scheme leaves
    /// this residual at truncation order and an atmosphere seeded on the exact hydrostatic profile
    /// starts moving. every stage adds it back, which makes the target an exact fixed point.
    pub residual_eq: Option<ConsFieldsGeneric<NDIM, DOF, Mem>>,
    /// the target's own conserved state on this level — the reference the deviation is measured
    /// from. covered cells hold the volume-weighted restriction of the finer level's target, so
    /// the restriction the run performs every parent step leaves the target where it found it.
    /// defining it by an independent evaluation of the profile per level would move it.
    pub cons_eq: Option<ConsFieldsGeneric<NDIM, DOF, Mem>>,
    /// the target primitives on this level after the same c2p and ghost-fill path
    /// used by the live state. coarse-fine transfer prolongs live departures from
    /// the parent target and decodes them against this fixed fine target, so the
    /// boundary operator never reads the fine interior solution.
    pub prim_eq: Option<PrimFieldsGeneric<NDIM, DOF, Mem>>,
}

// =============================================================================
// hierarchy
// =============================================================================

/// static-refinement hierarchy: levels[0] = coarsest, levels[n-1] = finest.
/// a fatal crash: an invalid cfl estimate (the wave speed went NaN or made the timestep
/// non-positive), a root step the retry guard or the timestep floor ended, or a panic caught
/// inside the step (the FOFC freeze-streak halt, a poisoned-cell assertion). the evolve loop
/// halts on the state the crash left and the driver snapshots a `.crashed` checkpoint + reports
/// it, so every crash reaches the user with its state on disk. a halted root step leaves the
/// step entry, the last accepted state; a panic leaves the mid-step state it fired on.
#[derive(Clone, Debug)]
pub struct CrashReport {
    pub iter: u64,
    pub time: f64,
    pub dt_cfl: f64,
    pub dt_prev: f64,
    /// the payload of a panic caught inside the step; the dt fields carry NaN in that case
    pub panic: Option<String>,
    /// the root step the retry guard or the timestep floor ended, with its evidence
    pub halt: Option<symbi_sim::step_transaction::StepHalt>,
}

/// A positive CFL estimate may increase by an arbitrary factor after a transient fast cell
/// disappears; only estimates that cannot advance time are fatal. Keep this predicate separate
/// from the final-time/user clamps so NaN cannot be hidden by `f64::min`.
#[inline]
fn fatal_cfl_dt(dt_cfl: f64) -> bool {
    dt_cfl.is_nan() || dt_cfl <= 0.0
}

/// how a builder treats a patch's real face that lies on a decomposition cut
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CutFaces {
    /// refuse the layout by name: the driver carries no register contribution across a cut
    Refuse,
    /// a contributing register face, its fine unit sent to the neighbor by the driver
    Contribute,
}

pub struct HierarchyRetrySidecars<const D: usize> {
    discrete: Vec<Option<symbi_sim::tracers::TracerSet<D>>>,
    continuous: Vec<Option<symbi_sim::tracers::ContinuousTracerSnapshot<D>>>,
    bodies: Vec<Option<symbi_ib::BodyCollection<f64, D>>>,
    motion: Vec<symbi_geometry::MotionState<f64>>,
    clocks: Vec<(f64, u64, f64)>,
    censuses: Vec<Vec<(symbi_sim::census::CensusHistory, Vec<Option<f64>>)>>,
}

pub struct Hierarchy<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    pub levels: Vec<LevelData<R, NDIM, DOF, M, E, S, Mem, K>>,
    injection_ledger: std::collections::BTreeMap<symbi_sim::mass_transport::ContainerId, f64>,
    /// each level's boundary injection of its current step, held from the step tail ahead of the
    /// finer subcycle until the continuous tracers have advanced: the injected tracers enter at
    /// the step's end, as on the single-level driver
    continuous_injection: std::collections::BTreeMap<usize, std::collections::BTreeMap<symbi_sim::mass_transport::ContainerId, f64>>,
    /// each level's density before its sinks drained it, held from the step tail ahead of the
    /// finer subcycle until the continuous tracers have advanced: they cross into the sinks from
    /// their positions at the step's end, as on the single-level driver
    continuous_accretion: std::collections::BTreeMap<usize, Vec<f64>>,
    /// coarse-fine prolongation order (one above the evolution reconstruction).
    pub prolong_order: ProlongOrder,
    /// set while the coupled root step drives the ideal-MHD hierarchy operator: the material drain
    /// belongs to the split operator D outside that operator, so no level tail penalizes.
    drain_suppressed: bool,
    /// the number of levels the whole run carries, when this hierarchy is one tile of a
    /// decomposed run and holds fewer: the finest-level physics (viscous transport, the horizon
    /// ledger, the surface stack) belongs to the run's finest level, which such a tile lacks.
    run_levels: Option<usize>,
    /// flux_registers[ll] corrects the interface between level ll and ll+1.
    pub flux_registers: Vec<FluxRegister<NDIM, DOF, Mem>>,
    /// emf_registers[ll] corrects the coarse bface at the same interface (mhd
    /// pairs only; None for pure hydro).
    pub emf_registers: Vec<Option<EmfRegister<NDIM, Mem>>>,
    tracer_interfaces: Vec<Vec<InterfaceFace<NDIM>>>,
    tracer_interface_ledgers: Vec<std::cell::RefCell<Vec<InterfaceTransfer>>>,
    tracer_root_global_cells: [usize; NDIM],
    tracer_root_offset: [usize; NDIM],
    /// the axes whose tile grid closes into a ring: the root halo exchange carries the leg across
    /// the seam, and every level's tracer layout sends mass leaving the grid to the far end.
    root_wraps: [bool; NDIM],
    /// the axes the first refined level wraps across its tiles: the root wraps them and the whole
    /// first patch spans the domain along them, so the patch has no face there and its level's
    /// halo exchange carries the leg across the seam
    fine_wraps: [bool; NDIM],
    /// set by `step_root` when the cfl dt is fatal (NaN / non-positive). `Some` halts the march at
    /// the last computed state; the driver writes a
    /// `.crashed` checkpoint and reports it, halting before advancing past t_final on garbage.
    pub crash: Option<CrashReport>,

    /// cells within this radius of the origin are excluded from the stationarity diagnostic.
    ///
    /// a declared target is a steady state of the equations the stage pipeline applies. inside a
    /// sink the drain removes mass and energy, so the target's imbalance there measures the drain
    /// rather than truncation error and reports non-convergence however exact the target is
    /// elsewhere. the well-balancing correction still lands on every cell; this radius decides
    /// which cells testify about convergence.
    ///
    /// `None` resolves to the largest accretion radius among the attached bodies, so a run with a
    /// sink gets the exclusion by default and a run with no bodies gets a zero radius that keeps
    /// every cell.
    pub equilibrium_mask_radius: Option<f64>,

    /// the previous step's raw cfl-derived dt, retained for crash diagnostics.
    /// the level's `state.dt` is the accepted dt, which the `t_final` clamp and an explicit-step
    /// rejection both shrink; comparing a fresh cfl estimate against it reports a "collapse" every
    /// time a rejected step is replayed at a smaller dt and then recovers. 0 before the first step.
    pub prev_dt_cfl: f64,
    /// a pending forced rejection at (root step, fine substep, fine stage), consumed when it
    /// fires
    forced_fine_rejection: Option<(u64, u8, usize)>,
    /// a forced rejection on any refined level: (root step, level, substep of the level within
    /// the root step, stage)
    forced_level_rejection: Option<(u64, usize, u64, usize)>,
    /// the root's staggered faces at the root attempt's entry, the old time level the parent
    /// face stencil of a neighbor's fine level gathers; held on every tile of a run whose
    /// fine levels read that stencil
    face_entry: Option<Vec<Field<f64, NDIM, Mem>>>,
    /// per refined level that parents a finer level of the run, its faces at its current
    /// substep's start: the old-time source of the finer level's parent face stencil, kept on
    /// every tile holding the level whether or not the tile holds the finer one
    level_face_entry: Vec<Option<Vec<Field<f64, NDIM, Mem>>>>,
    /// set while `prime` builds the initial state: the fine face halo is left to the first
    /// substep's prolongation, whose parent face stencil is gathered after the root stages
    priming: bool,
    /// a tile of a decomposed hierarchy: each level's tracer stage leaves the ancestry blend to
    /// the driver, which ranks the level's ids across every tile
    blend_across_tiles: bool,
    /// on such a tile, the coarse-fine interface event a level stage booked, applied by the driver
    /// once the stage's blend ranked the level's ids across every tile: (level, the stage's first
    /// receipt, the stage)
    pending_interface: Option<(usize, usize, usize)>,
    /// on such a tile, the whole domain's extent and the boundary each axis presents to a
    /// continuous tracer: the cut faces of a tile are no domain boundary
    continuous_domain: Option<([(f64, f64); NDIM], Boundaries<NDIM>)>,

    /// whether the coarse-fine ghost transfer prolongs departures from the local hydrostatic
    /// equilibrium; when off it prolongs the raw primitive state. `None` follows the kernel set:
    /// active precisely when the set reconstructs balanced (`KernelSet::hydrostatic_balance`) and
    /// a gravitating body supplies the potential. `Some(x)` forces the choice — the knob that
    /// measures the transfer as the load-bearing piece. see `cf_transfer_balanced`.
    pub balance_aware_transfer: Option<bool>,
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
    Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    /// the prime of level `ll` of a seeded target on this tile, as the single hierarchy primes it:
    /// recovery, the coarse-fine prolongation from the parent's current state, the physical fill.
    /// the caller exchanges the level's cut halos after it, before a child prolongs from them
    pub fn prime_target_level(&mut self, ll: usize) {
        self.priming = true;
        let level = &self.levels[ll];
        level.kernels.c2p(&level.state);
        if ll > 0 {
            self.prolong_cf(ll, 1.0);
        }
        self.fill_physical_ghosts(ll);
        self.priming = false;
    }

    /// install a decomposed probe's results on this tile's levels: each level's anchored target,
    /// its interface flux and its imbalance
    pub fn install_declared_target(
        &mut self,
        anchors: Vec<GasState<NDIM, DOF, Mem>>,
        fluxes: Vec<Option<EquilibriumFlux<NDIM, DOF, Mem>>>,
        residuals: Vec<Option<ConsFieldsGeneric<NDIM, DOF, Mem>>>,
    ) {
        for (((level, anchored), flux), residual) in self.levels.iter_mut().zip(anchors).zip(fluxes).zip(residuals) {
            level.flux_eq = flux;
            level.residual_eq = residual;
            let (cons, prim) = anchored.into_parts();
            level.cons_eq = Some(cons);
            level.prim_eq = Some(prim);
        }
    }

    /// seed tracers from the hierarchy's active composite mass: covered coarse
    /// cells are omitted and their finest available children supply the mass.
    pub fn seed_mass_tracers(&self, n: usize) -> symbi_sim::tracers::TracerSet<NDIM> {
        assert!(
            self.levels[0].state.geom.coords == symbi_geometry::Geometry::Cartesian,
            "refined mass tracers require cartesian geometry"
        );
        assert!(
            self.levels[0]
                .state
                .geom
                .interior
                .spaces
                .iter()
                .all(|space| space.lo == 0),
            "refined mass tracers require a root grid beginning at global index zero"
        );

        let root_cells: [usize; NDIM] =
            std::array::from_fn(|aa| self.levels[0].state.geom.interior.spaces[aa].size());
        let mut owners = Vec::new();
        let mut cells = Vec::new();
        let mut masses = Vec::new();
        for (ll, level) in self.levels.iter().enumerate() {
            let scale = 1usize
                .checked_shl(ll as u32)
                .expect("refinement level exceeds machine index width");
            let global_cells: [usize; NDIM] = std::array::from_fn(|aa| root_cells[aa] * scale);
            let geometry = level.state.geom.block_geometry(level.state.physics.metric);
            for coord in level.state.geom.interior.iter() {
                if !level.state.composite_ownership.owns_leaf(coord) {
                    continue;
                }
                let center = geometry.centroid(coord);
                let widths = level.state.geom.dx;
                let lo = std::array::from_fn(|aa| center[aa] - 0.5 * widths[aa]);
                // a level's index is already absolute on its own refined lattice, which the
                // root grid anchors at zero, so the coordinate addresses that lattice directly.
                let cell: [usize; NDIM] = std::array::from_fn(|aa| coord[aa] as usize);
                owners.push(symbi_sim::tracers::cell_container_id(
                    symbi_sim::tracers::cell_address(cell, global_cells),
                    ll as u8,
                ));
                cells.push((lo, widths));
                masses.push(*level.state.fields.cons.den.view().at(coord) * geometry.volume(coord));
            }
        }
        symbi_sim::tracers::seed_weighted_cells(&owners, &cells, &masses, n)
    }

    /// seed the active composite hierarchy once and partition the resulting
    /// globally identified population onto the levels that own its cells.
    pub fn attach_mass_tracers(&mut self, n: usize) {
        let seeded = self.seed_mass_tracers(n);
        let mut per_level: Vec<symbi_sim::tracers::TracerSet<NDIM>> = (0..self.levels.len())
            .map(|_| symbi_sim::tracers::TracerSet {
                weight: seeded.weight,
                run_seed: seeded.run_seed,
                next_id: seeded.next_id,
                ..Default::default()
            })
            .collect();
        for ii in 0..seeded.len() {
            let (level, _) = symbi_sim::tracers::cell_container_address(seeded.owner[ii])
                .expect("composite seed produced a non-cell owner");
            let target = per_level
                .get_mut(level as usize)
                .expect("composite seed owner names an absent refinement level");
            target.x.push(seeded.x[ii]);
            target.id.push(seeded.id[ii]);
            target.cohort.push(seeded.cohort[ii]);
            target.flags.push(seeded.flags[ii]);
            target.owner.push(seeded.owner[ii]);
            target.step_owner.push(seeded.step_owner[ii]);
            target.step_flags.push(seeded.step_flags[ii]);
        }
        for (level, tracers) in self.levels.iter_mut().zip(per_level) {
            level.state.tracers = Some(tracers);
        }
    }

    /// relocate continuous particles onto the finest active level containing
    /// their authoritative physical position.
    pub fn migrate_continuous_tracers_to_finest(&mut self) -> Result<usize, String> {
        let descriptors: Vec<_> = self
            .levels
            .iter()
            .enumerate()
            .map(|(level, data)| {
                (
                    level,
                    symbi_sim::tracers::partition_physical_bounds(&data.state.geom),
                    data.state.geom.maps,
                    data.state.geom.x_lo,
                    data.state.geom.dx,
                    data.state.geom.interior.clone(),
                    self.tracer_layout(level),
                )
            })
            .collect();
        let metadata = self
            .levels
            .iter()
            .filter_map(|level| level.state.continuous_tracers.as_ref())
            .max_by_key(|tracers| usize::from(tracers.len > 0))
            .map(|tracers| {
                (
                    tracers.order,
                    tracers.weight,
                    tracers.run_seed,
                    tracers.next_id,
                    tracers.injection_remainder,
                )
            })
            .ok_or_else(|| "continuous tracer population is missing".to_string())?;
        let locate = |position: [f64; NDIM]| {
            descriptors.iter().rev().find_map(
                |(level, bounds, maps, x_lo, dx, interior, layout)| {
                    if (0..NDIM)
                        .any(|dd| position[dd] < bounds[dd].0 || position[dd] >= bounds[dd].1)
                    {
                        return None;
                    }
                    let coord: [isize; NDIM] = std::array::from_fn(|dd| match maps {
                        Some(maps) => maps[dd].index_at(position[dd]),
                        None => ((position[dd] - x_lo[dd]) / dx[dd]).floor() as isize,
                    });
                    if !interior.contains(coord) {
                        return None;
                    }
                    // the level's own patch may start away from the origin, so rebase the
                    // coordinate onto the tile before offsetting it into the global lattice.
                    let cell: [usize; NDIM] = std::array::from_fn(|dd| {
                        layout.tile_offset[dd] + (coord[dd] - interior.spaces[dd].lo) as usize
                    });
                    Some((
                        *level,
                        symbi_sim::tracers::cell_container_id(
                            symbi_sim::tracers::cell_address(cell, layout.global_cells),
                            layout.level,
                        ),
                    ))
                },
            )
        };
        for level in &mut self.levels {
            if let Some(tracers) = level.state.continuous_tracers.as_mut() {
                tracers.order = metadata.0;
                tracers.weight = metadata.1;
                tracers.run_seed = metadata.2;
                tracers.next_id = metadata.3;
                tracers.injection_remainder = metadata.4;
            }
        }
        let mut moved = Vec::new();
        for (source_level, level) in self.levels.iter_mut().enumerate() {
            let Some(tracers) = level.state.continuous_tracers.as_mut() else {
                continue;
            };
            let mut ii = 0;
            while ii < tracers.len {
                let inactive = unsafe {
                    *tracers.escaped.as_ptr::<u8>().add(ii) != 0
                        || *tracers.crossed_sink.as_ptr::<u8>().add(ii) != 0
                };
                if inactive {
                    ii += 1;
                    continue;
                }
                let position: [f64; NDIM] =
                    unsafe { std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(ii)) };
                let Some((target_level, owner)) = locate(position) else {
                    ii += 1;
                    continue;
                };
                if target_level == source_level {
                    unsafe {
                        *tracers
                            .owner
                            .as_mut_ptr::<symbi_sim::mass_transport::ContainerId>()
                            .add(ii) = owner;
                    }
                    ii += 1;
                } else {
                    let mut record = tracers.swap_remove_host(ii)?;
                    record.owner = owner;
                    moved.push((target_level, record));
                }
            }
        }
        let count = moved.len();
        for (target_level, record) in moved {
            if self.levels[target_level].state.continuous_tracers.is_none() {
                let mut tracers = symbi_sim::tracers::ContinuousTracerSet::allocate(0, metadata.0)?;
                tracers.weight = metadata.1;
                tracers.run_seed = metadata.2;
                tracers.next_id = metadata.3;
                tracers.injection_remainder = metadata.4;
                self.levels[target_level].state.continuous_tracers = Some(tracers);
            }
            self.levels[target_level]
                .state
                .continuous_tracers
                .as_mut()
                .expect("continuous tracer destination was initialized")
                .push_host(record)?;
        }
        Ok(count)
    }

    pub(crate) fn tracer_cell(
        &self,
        owner: symbi_sim::mass_transport::ContainerId,
    ) -> Option<(usize, [isize; NDIM])> {
        let (level, mut linear) = symbi_sim::tracers::cell_container_address(owner)?;
        let level = level as usize;
        let scale = 1usize.checked_shl(level as u32)?;
        let global_cells: [usize; NDIM] =
            std::array::from_fn(|aa| self.tracer_root_global_cells[aa] * scale);
        let global_coord: [isize; NDIM] = std::array::from_fn(|aa| {
            let index = linear % global_cells[aa];
            linear /= global_cells[aa];
            index as isize
        });
        let coord = std::array::from_fn(|aa| {
            global_coord[aa] - (self.tracer_root_offset[aa] * scale) as isize
        });
        self.levels
            .get(level)?
            .state
            .geom
            .interior
            .contains(coord)
            .then_some((level, coord))
    }

    pub(crate) fn tracer_layout(&self, level: usize) -> symbi_sim::tracers::TransportLayout<NDIM> {
        let scale = 1usize << level;
        let interior = &self.levels[level].state.geom.interior;
        symbi_sim::tracers::TransportLayout {
            global_cells: std::array::from_fn(|aa| self.tracer_root_global_cells[aa] * scale),
            tile_offset: std::array::from_fn(|aa| {
                self.tracer_root_offset[aa] * scale + interior.spaces[aa].lo as usize
            }),
            wraps: self.root_wraps,
            level: level as u8,
        }
    }

    pub fn set_root_layout(
        &mut self,
        global_cells: [usize; NDIM],
        tile_offset: [usize; NDIM],
        wraps: [bool; NDIM],
    ) {
        self.tracer_root_global_cells = global_cells;
        self.tracer_root_offset = tile_offset;
        self.root_wraps = wraps;
        let parent_cells: [usize; NDIM] =
            std::array::from_fn(|aa| self.levels[0].state.geom.interior.spaces[aa].size());
        self.tracer_interfaces = (0..self.levels.len().saturating_sub(1))
            .map(|ll| {
                interface_faces_with_layout(
                    self.levels[ll].coverage.as_ref().unwrap(),
                    parent_cells,
                    tile_offset,
                    global_cells,
                    ll as u8,
                )
            })
            .collect();
    }

    fn tracer_owner_is_active(&self, owner: symbi_sim::mass_transport::ContainerId) -> bool {
        self.tracer_cell(owner).is_some_and(|(level, coord)| {
            self.levels[level]
                .state
                .composite_ownership
                .owns_leaf(coord)
        })
    }

    fn sync_tracer_spawn_state(&mut self, source_level: usize) {
        let Some(source) = self.levels[source_level].state.tracers.as_ref() else {
            return;
        };
        let next_id = source.next_id;
        let injection_remainder = source.injection_remainder;
        for level in &mut self.levels {
            if let Some(tracers) = level.state.tracers.as_mut() {
                tracers.next_id = next_id;
                tracers.injection_remainder = injection_remainder;
            }
        }
    }

    fn tracer_cell_mass(&self, owner: symbi_sim::mass_transport::ContainerId) -> Option<f64> {
        let (level, coord) = self.tracer_cell(owner)?;
        let state = &self.levels[level].state;
        let geometry = state.geom.block_geometry(state.physics.metric);
        Some(*state.fields.cons.den.view().at(coord) * geometry.volume(coord))
    }

    fn apply_interface_event(
        &mut self,
        transfers: &[InterfaceTransfer],
        epoch: u64,
    ) -> Result<(), String> {
        use std::collections::BTreeMap;
        use symbi_sim::mass_transport::{SamplingKey, sample_systematic};

        if transfers.is_empty() {
            return Ok(());
        }
        let mut post_mass = BTreeMap::new();
        for transfer in transfers {
            for owner in [transfer.source, transfer.destination] {
                if let std::collections::btree_map::Entry::Vacant(entry) = post_mass.entry(owner) {
                    let mass = self.tracer_cell_mass(owner).ok_or_else(|| {
                        format!("interface receipt names inactive cell {}", owner.0)
                    })?;
                    entry.insert(mass);
                }
            }
        }
        let kernels = interface_transport_kernels(transfers, &post_mass)?;
        let run_seed = self
            .levels
            .iter()
            .find_map(|level| level.state.tracers.as_ref().map(|tracers| tracers.run_seed))
            .unwrap_or(0);
        let key = SamplingKey { run_seed, epoch };
        let mut assignments = BTreeMap::new();
        for kernel in kernels {
            let ids: Vec<u64> = self
                .levels
                .iter()
                .filter_map(|level| level.state.tracers.as_ref())
                .flat_map(|tracers| {
                    tracers
                        .id
                        .iter()
                        .zip(&tracers.owner)
                        .filter_map(|(&id, &owner)| (owner == kernel.source()).then_some(id))
                })
                .collect();
            assignments.extend(sample_systematic(&kernel, &ids, key));
        }
        for level in &mut self.levels {
            if let Some(tracers) = level.state.tracers.as_mut() {
                for (ii, id) in tracers.id.iter().enumerate() {
                    if let Some(&destination) = assignments.get(id) {
                        tracers.owner[ii] = destination;
                        tracers.step_owner[ii] = destination;
                    }
                }
            }
        }

        let mut migrating = Vec::new();
        for (level_index, level) in self.levels.iter_mut().enumerate() {
            let Some(tracers) = level.state.tracers.as_mut() else {
                continue;
            };
            let mut ii = 0usize;
            while ii < tracers.len() {
                let destination_level =
                    symbi_sim::tracers::cell_container_address(tracers.owner[ii])
                        .map(|address| address.0 as usize)
                        .unwrap_or(level_index);
                if destination_level == level_index {
                    ii += 1;
                    continue;
                }
                tracers.x.swap_remove(ii);
                migrating.push((
                    destination_level,
                    tracers.id.swap_remove(ii),
                    tracers.cohort.swap_remove(ii),
                    tracers.flags.swap_remove(ii),
                    tracers.owner.swap_remove(ii),
                    tracers.step_owner.swap_remove(ii),
                    tracers.step_flags.swap_remove(ii),
                ));
            }
        }
        for (destination_level, id, cohort, flags, owner, step_owner, step_flags) in migrating {
            let (_, coord) = self
                .tracer_cell(owner)
                .ok_or_else(|| format!("interface destination {} is not active", owner.0))?;
            let state = &self.levels[destination_level].state;
            let x = state.geom.centroid(coord);
            let tracers = self.levels[destination_level]
                .state
                .tracers
                .as_mut()
                .expect("every refined level carries a tracer set");
            tracers.x.push(x);
            tracers.id.push(id);
            tracers.cohort.push(cohort);
            tracers.flags.push(flags);
            tracers.owner.push(owner);
            tracers.step_owner.push(step_owner);
            tracers.step_flags.push(step_flags);
        }
        Ok(())
    }

    /// decimate the hierarchy to a screen-sized density heatmap, compositing the
    /// nested refinement levels: each root cell descends to the finest level whose
    /// `coverage` box contains it (a single nested box per level, ratio 2), so
    /// the refined region shows its fine detail and the rest shows the coarse grid.
    /// a single-level hierarchy is just the root decimation. cost is screen-bounded.
    pub fn field_slice_composite(
        &self,
        max_dim: usize,
        index: usize,
        orient: usize,
        zoom: usize,
    ) -> Option<FieldDecimation> {
        // single grid: the root is the only level, so reuse the plain per-state decimation.
        if self.levels.len() <= 1 {
            return self.levels[0]
                .state
                .field_slice_oriented(max_dim, index, orient, zoom);
        }
        if !Mem::IS_HOST_ACCESSIBLE {
            return None;
        }
        let (kind, name) = self.levels[0].state.nth_field(index);
        let interior = &self.levels[0].state.geom.interior;
        // the display axes per orientation (the same law as field_slice_oriented):
        // (horizontal, vertical) of the picture; the remaining 3D axis holds its
        // mid-plane index. 2D shows (0, 1).
        let (ah, av) = if NDIM >= 3 {
            match orient % 3 {
                1 => (0usize, 2usize),
                2 => (1, 2),
                _ => (0, 1),
            }
        } else {
            (0, 1)
        };
        // the zoomed window: a centered span of size/2^zoom, at least 4 cells.
        let windowed = |sp: &symbi_algebra::Space| -> symbi_algebra::Space {
            if zoom == 0 {
                return sp.clone();
            }
            let size = sp.size() as isize;
            let span = (size >> zoom.min(4)).max(4.min(size));
            let mid = sp.lo + size / 2;
            symbi_algebra::Space {
                name: sp.name,
                lo: mid - span / 2,
                hi: mid - span / 2 + span,
            }
        };
        let sp0 = windowed(&interior.spaces[ah]);
        let sp1 = windowed(interior.spaces.get(av)?); // None on a 1D grid
        let (nx, ny) = (sp0.size(), sp1.size());
        if nx == 0 || ny == 0 {
            return None;
        }
        let m = max_dim.max(1);
        let sx = ((nx + m - 1) / m).max(1);
        let sy = ((ny + m - 1) / m).max(1);
        let out_w = (nx + sx - 1) / sx;
        let out_h = (ny + sy - 1) / sy;
        let (lo0, hi0, lo1, hi1) = (sp0.lo, sp0.hi, sp1.lo, sp1.hi);
        // the base root index: mid-plane on every non-display axis.
        let base: [isize; NDIM] = std::array::from_fn(|ax| {
            let s = &interior.spaces[ax];
            s.lo + (s.size() / 2) as isize
        });

        let mut data = Vec::with_capacity(out_w * out_h);
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for j in 0..out_h {
            let y0 = lo1 + (j * sy) as isize;
            let y1 = (y0 + sy as isize).min(hi1);
            for i in 0..out_w {
                let x0 = lo0 + (i * sx) as isize;
                let x1 = (x0 + sx as isize).min(hi0);
                // block-average the root-cell footprint, each root cell resolved to
                // the finest level covering it.
                let mut sum = 0.0_f64;
                let mut cnt = 0u32;
                let mut ry = y0;
                while ry < y1 {
                    let mut rx = x0;
                    while rx < x1 {
                        let mut idx = base;
                        idx[ah] = rx;
                        idx[av] = ry;
                        sum += self.sample_finest_at(idx, kind);
                        cnt += 1;
                        rx += 1;
                    }
                    ry += 1;
                }
                let v = sum / cnt.max(1) as f64;
                vmin = vmin.min(v);
                vmax = vmax.max(v);
                data.push(v as f32);
            }
        }
        Some(FieldDecimation {
            width: out_w,
            height: out_h,
            data,
            vmin,
            vmax,
            name: name.into(),
            preserve_aspect: false,
        })
    }

    /// resolve a root-level index to the finest level covering it and read the
    /// field there — the axis-general coverage descent (the display slice builds
    /// the root index per its own orientation/zoom and hands it here).
    fn sample_finest_at(&self, mut idx: [isize; NDIM], kind: FieldKind) -> f64 {
        let mut lvl = 0usize;
        while lvl + 1 < self.levels.len() {
            let cov = match &self.levels[lvl].coverage {
                Some(c) if domain_contains(c, &idx) => c,
                _ => break,
            };
            let finer = &self.levels[lvl + 1].state.geom.interior;
            idx = std::array::from_fn(|ax| finer.spaces[ax].lo + (idx[ax] - cov.spaces[ax].lo) * 2);
            lvl += 1;
        }
        self.levels[lvl].state.field_value(idx, kind)
    }

    /// seed every fine level's interior from its parent by conservative
    /// prolongation at the hierarchy's `prolong_order` (= interior reconstruction
    /// order + 1, the same order the coarse-fine ghost prolongation uses). the IC
    /// fill for a hierarchy whose coarse level was seeded but whose fine levels are
    /// still empty — a coarse-only IC, e.g., a python-driven `prim_gen`. each
    /// conserved component is prolonged coarse -> fine interior; the fine levels
    /// then refine the solution as they evolve.
    ///
    /// the conserved components are prolonged independently, and a component-wise
    /// high-order prolongation is free to break the admissibility inequality
    /// E >= |m|^2 / (2 rho): near an extremum of the momentum the
    /// non-monotone stencil overshoots `m` while `E` interpolates low, and the
    /// fine cell's internal energy E - |m|^2/(2 rho) goes negative wherever the
    /// kinetic energy density is comparable to the internal — e.g. a velocity
    /// field whose local mach approaches unity on a cold background. each level
    /// is therefore c2p-audited after its fill, and every inadmissible cell is
    /// re-seeded by piecewise-constant injection of its covering parent cell,
    /// which is unconditionally admissible (the children are copies of an
    /// admissible parent) and preserves the coarse-cell integral exactly. the
    /// audit repairs hydro components only; returns the count of re-seeded
    /// cells so callers can assert how often the fallback engaged.
    ///
    /// the seeded state is hydro-complete plus the mhd cell-centered B. the staggered
    /// fine `bface` comes from the mhd refinement path, which supplies the face
    /// prolongation a face field requires.
    pub fn seed_fine_from_coarse(&self) -> symbi_xpu::Result<usize>
    where
        R: Regime<f64, NDIM, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF>,
        M: Metric<f64, DOF>,
        <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
        <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
    {
        let mut reseeded = 0usize;
        for ll in 1..self.levels.len() {
            reseeded += self.seed_level_from_parent(ll)?;
        }
        Ok(reseeded)
    }

    /// seed level `ll` from level `ll - 1` and audit it, as [`Self::seed_fine_from_coarse`] does
    /// for every level in turn, the parent's ghosts filled for the seed and restored after it;
    /// returns the count of cells re-seeded by parent injection.
    pub fn seed_level_from_parent(&self, ll: usize) -> symbi_xpu::Result<usize>
    where
        R: Regime<f64, NDIM, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF>,
        M: Metric<f64, DOF>,
        <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
        <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
    {
        let saved = self.fill_seed_ghosts(ll - 1);
        let reseeded = self.seed_level_from_filled_parent(ll)?;
        self.restore_seed_ghosts(ll - 1, saved);
        Ok(reseeded)
    }

    /// fill level `level`'s ghost cells with the state its boundary prescribes, for the seed of
    /// the level above it: the primitives recovered, the ghost fill run, the conserved ghosts
    /// converted from the filled primitives beside a physical face and carried across the seam of
    /// a periodic axis. a decomposed driver fills before it exchanges the level's cut halos, so
    /// the exchange carries the cells where a face's ghosts meet a cut. returns the conserved
    /// ghosts the level held, which `restore_seed_ghosts` writes back after the seed
    pub fn fill_seed_ghosts(&self, level: usize) -> SeedGhosts
    where
        R: Regime<f64, NDIM, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF>,
        M: Metric<f64, DOF>,
        <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
        <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
    {
        let parent = &self.levels[level];
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        let saved = conserved_ghosts(&parent.state);
        parent.kernels.c2p(&parent.state);
        parent.kernels.ghost_fill(&parent.state);
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        parent.state.fill_conserved_ghosts();
        symbi_substrate::regimes::mhd_substrate::wrap_periodic_conserved_ghosts(&parent.state);
        SeedGhosts(saved)
    }

    /// write back the conserved ghosts `fill_seed_ghosts` saved, so a seed changes no stored state
    /// of the parent level
    pub fn restore_seed_ghosts(&self, level: usize, saved: SeedGhosts) {
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        restore_conserved_ghosts(&self.levels[level].state, &saved.0);
    }

    /// seed level `ll` by prolongation from level `ll - 1`, whose ghosts `fill_seed_ghosts` filled
    /// and whose cut halos the driver exchanged, and audit it
    pub fn seed_level_from_filled_parent(&self, ll: usize) -> symbi_xpu::Result<usize> {
        let mut reseeded = 0usize;
        {
            let (lo, hi) = self.levels.split_at(ll);
            let coarse = &lo[ll - 1].state;
            let fine = &hi[0].state;
            let region = fine.geom.interior.clone();
            let order = self.prolong_order;
            // the prolong takes a time pair (old, new) and forms (1-alpha)*old +
            // alpha*new. the IC has no second time level, so bind a disjoint zero
            // as `new` (the kernel forbids old/new aliasing) and alpha = 0, which
            // reads `old` only.
            let zero = Field::<f64, NDIM, Mem>::zeros(&coarse.geom.allocated)?;
            let (cc, fc) = (&coarse.fields.cons, &fine.fields.cons);
            prolong_field(&cc.den, &zero, &fc.den, &region, order, 0.0);
            for k in 0..DOF {
                prolong_field(&cc.mom[k], &zero, &fc.mom[k], &region, order, 0.0);
            }
            if let (Some(cn), Some(fn_nrg)) = (cc.nrg_field(), fc.nrg_field()) {
                prolong_field(cn, &zero, fn_nrg, &region, order, 0.0);
            }
            // the conserved dye rho*chi, prolonged like the mass it rides on, so the fine level
            // starts with the dye its parent holds and restricts it back unchanged
            if let (Some(cchi), Some(fchi)) = (cc.chi_field(), fc.chi_field()) {
                prolong_field(cchi, &zero, fchi, &region, order, 0.0);
            }
            // the isothermal closure's cs^2(x): prolonged over the fine interior and a band of two
            // ghost cells, the band the fine flux, drain, and viscous stencils read across the
            // coarse-fine boundary, clipped to the cells the parent's allocation can supply, then
            // continued into the remaining ghosts. the clip bites where a refined patch runs out to
            // the edge of a decomposed tile; there the continuation equals the prolongation of a
            // uniform cs^2, the globally isothermal closure the tiled drivers admit.
            if let (Some(ccs), Some(fcs)) = (coarse.fields.cs2.as_ref(), fine.fields.cs2.as_ref()) {
                let mut band = region.clone();
                for ax in 0..NDIM {
                    band = band.extend(ax, 2, 2);
                }
                let clipped = clip_to_prolongable(&band, &coarse.geom.allocated, order).unwrap_or(region.clone());
                let bit = (0..NDIM).any(|ax| {
                    (clipped.spaces[ax].lo, clipped.spaces[ax].hi) != (band.spaces[ax].lo, band.spaces[ax].hi)
                });
                if bit {
                    assert_uniform_closure(ccs, &coarse.geom.allocated, ll);
                }
                let band = clipped;
                prolong_field(ccs, &zero, fcs, &band, order, 0.0);
                fine.extend_isothermal_cs2_into_ghosts(&band);
            }
            // mhd: the staggered faces by the divergence-free prolongation per normal axis
            // (Balsara), so the fine CT starts at div(B) = 0 and coarse-fine consistent. the
            // cell-centered B is then the interpolation of the seeded faces, the relation the CT
            // corrector maintains on every step, with the total energy shifted by the change in
            // the cell magnetic energy so the prolonged thermal state is kept; the prolonged cell
            // field supplies the magnetic energy that shift starts from. the out-of-plane cell
            // component of a 2.5D grid has no face and keeps its prolonged value.
            if let (Some(cm), Some(fm)) = (coarse.fields.mhd.as_ref(), fine.fields.mhd.as_ref()) {
                for k in 0..DOF {
                    prolong_field(&cm.bcell[k], &zero, &fm.bcell[k], &region, order, 0.0);
                }
                let mut zero_faces: Vec<Field<f64, NDIM, Mem>> = Vec::with_capacity(NDIM);
                for dd in 0..NDIM {
                    zero_faces.push(Field::<f64, NDIM, Mem>::zeros(cm.bface[dd].domain())?);
                }
                for dd in 0..NDIM {
                    let face_region = fine.geom.interior.extend(dd, 0, 1);
                    prolong_face_field(dd, &cm.bface.b[..], &zero_faces[..], &fm.bface[dd], &face_region, 0.0);
                }
                bcell_from_bface_region(fm, fc.nrg_field(), &region);
                fm.bface_initialized
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }

            // admissibility audit of the freshly filled level. the audit runs
            // inside the level loop so a repaired cell is what the next level
            // prolongs from — every level prolongs from an admissible parent.
            let lvl = &self.levels[ll];
            lvl.kernels.c2p(&lvl.state);
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            if scan_c2p_errors(&lvl.state).is_err() {
                let flagged: Vec<[isize; NDIM]> = lvl
                    .state
                    .geom
                    .interior
                    .iter()
                    .filter(|coord| *lvl.state.fields.c2p_error.view().at(*coord) != 0.0)
                    .collect();
                for coord in &flagged {
                    let cell = Domain::new(std::array::from_fn(|aa| Space {
                        name: lvl.state.geom.interior.spaces[aa].name,
                        lo: coord[aa],
                        hi: coord[aa] + 1,
                    }));
                    prolong_field(&cc.den, &zero, &fc.den, &cell, ProlongOrder::Pcm, 0.0);
                    for kk in 0..DOF {
                        prolong_field(
                            &cc.mom[kk],
                            &zero,
                            &fc.mom[kk],
                            &cell,
                            ProlongOrder::Pcm,
                            0.0,
                        );
                    }
                    if let (Some(cn), Some(fn_nrg)) = (cc.nrg_field(), fc.nrg_field()) {
                        prolong_field(cn, &zero, fn_nrg, &cell, ProlongOrder::Pcm, 0.0);
                    }
                    if let (Some(cchi), Some(fchi)) = (cc.chi_field(), fc.chi_field()) {
                        prolong_field(cchi, &zero, fchi, &cell, ProlongOrder::Pcm, 0.0);
                    }
                }
                reseeded += flagged.len();
                // refresh the primitives and the error field; a cell still
                // inadmissible after parent injection is a defect of the coarse
                // state itself and is left for the caller's c2p gate to report.
                lvl.kernels.c2p(&lvl.state);
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                eprintln!(
                    "seed_fine_from_coarse: level {} re-seeded {} cell(s) by parent \
                     injection (high-order prolongation of independent conserved \
                     components broke E >= |m|^2/2rho)",
                    ll,
                    flagged.len()
                );
            }
        }
        Ok(reseeded)
    }

    /// a 1-level hierarchy: the degenerate case, which reproduces evolve()
    /// bit-for-bit.
    pub fn single(state: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>, kernels: K) -> Self {
        let tracer_root_global_cells =
            std::array::from_fn(|aa| state.geom.interior.spaces[aa].size());
        Hierarchy {
            levels: vec![LevelData {
                state,
                kernels,
                prim_old: None,
                prim_lerp: None,
                prolong_sweep: std::sync::OnceLock::new(),
                band_departure: std::sync::OnceLock::new(),
                bcell_old: None,
                bface_old: None,
                attempt_entry: None,
                cut_sides: [[false; 2]; NDIM],
                patch_interior: None,
                face_stencil: None,
                coverage: None,
                flux_eq: None,
                residual_eq: None,
                cons_eq: None,
                prim_eq: None,
            }],
            injection_ledger: std::collections::BTreeMap::new(),
            continuous_injection: std::collections::BTreeMap::new(),
            continuous_accretion: std::collections::BTreeMap::new(),
            prolong_order: ProlongOrder::Plm,
            drain_suppressed: false,
            run_levels: None,
            flux_registers: Vec::new(),
            emf_registers: Vec::new(),
            tracer_interfaces: Vec::new(),
            tracer_interface_ledgers: Vec::new(),
            tracer_root_global_cells,
            tracer_root_offset: [0; NDIM],
            root_wraps: [false; NDIM],
            fine_wraps: [false; NDIM],
            crash: None,
            equilibrium_mask_radius: None,
            prev_dt_cfl: 0.0,
            forced_fine_rejection: None,
            forced_level_rejection: None,
            face_entry: None,
            level_face_entry: Vec::new(),
            priming: false,
            blend_across_tiles: false,
            pending_interface: None,
            continuous_domain: None,
            balance_aware_transfer: None,
        }
    }

    /// build a statically nested hierarchy: region k refines level k at ratio 2.
    /// fine levels live in absolute indices (interior at 2x the covered cells)
    /// sharing the global physical origin; coarse-fine faces get
    /// BoundaryType::CoarseFine, faces flush with the parent interior inherit
    /// the parent's boundary. `make_kernels` builds each fine level's kernel
    /// set from its constructed state.
    pub fn with_refinement(
        coarse: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
        coarse_kernels: K,
        regions: &[RefinementRegion<NDIM>],
        prolong_order: ProlongOrder,
        make_kernels: impl Fn(&SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>) -> K,
    ) -> symbi_xpu::Result<Self> {
        Self::with_refinement_of(coarse, coarse_kernels, regions, regions, prolong_order, make_kernels)
    }

    /// a tile of a decomposed hierarchy: `regions` are the patches clipped to this tile's
    /// extent and `whole[i]` the unclipped patch of `regions[i]`. the fine grid covers the
    /// clipped box; the level's ownership (the covered core a stage holds at its input) is
    /// the whole patch's core clipped to the tile, so every tile and the single hierarchy
    /// hold the same cells active around the patch's real faces, wherever a cut falls.
    pub fn with_refinement_of(
        coarse: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
        coarse_kernels: K,
        regions: &[RefinementRegion<NDIM>],
        whole: &[RefinementRegion<NDIM>],
        prolong_order: ProlongOrder,
        make_kernels: impl Fn(&SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>) -> K,
    ) -> symbi_xpu::Result<Self> {
        Self::refine_levels(coarse, coarse_kernels, regions, whole, prolong_order, make_kernels, CutFaces::Refuse, None)
    }

    /// `with_refinement_of` for a driver that carries register contributions across cuts: a
    /// patch face that is the patch's real face and lies on a decomposition cut becomes a
    /// contributing register face, whose fine unit the driver sends to the tile holding the
    /// corrected coarse cell. that face's fine ghosts are prolonged from the root halo, so
    /// the prolongation stencil has to fit inside it; a root halo too shallow for it is
    /// refused with the depth it needs
    pub fn with_refinement_of_contributing(
        coarse: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
        coarse_kernels: K,
        regions: &[RefinementRegion<NDIM>],
        whole: &[RefinementRegion<NDIM>],
        prolong_order: ProlongOrder,
        make_kernels: impl Fn(&SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>) -> K,
    ) -> symbi_xpu::Result<Self> {
        Self::refine_levels(coarse, coarse_kernels, regions, whole, prolong_order, make_kernels, CutFaces::Contribute, None)
    }

    /// `with_refinement_of_contributing` for a tile placed on the whole root grid of
    /// `global_cells`, closed into a ring along the axes `wraps` names: a whole first patch
    /// spanning such an axis has no face on it, keeps its core across the seam, and its level
    /// wraps across the tiles. the tile's root layout is set as `set_root_layout` sets it
    #[allow(clippy::too_many_arguments)]
    pub fn with_refinement_placed(
        coarse: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
        coarse_kernels: K,
        regions: &[RefinementRegion<NDIM>],
        whole: &[RefinementRegion<NDIM>],
        prolong_order: ProlongOrder,
        make_kernels: impl Fn(&SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>) -> K,
        global_cells: [usize; NDIM],
        wraps: [bool; NDIM],
    ) -> symbi_xpu::Result<Self> {
        let offset: [usize; NDIM] = std::array::from_fn(|ax| coarse.geom.index_offset[ax].max(0) as usize);
        let mut h = Self::refine_levels(
            coarse,
            coarse_kernels,
            regions,
            whole,
            prolong_order,
            make_kernels,
            CutFaces::Contribute,
            Some((global_cells, wraps)),
        )?;
        h.set_root_layout(global_cells, offset, wraps);
        Ok(h)
    }

    /// a single level with receiving register faces: this tile holds the coarse cells a
    /// neighbor's fine faces correct. `slabs` names each face by `(ax, side)` of the patch it
    /// bounds and its slab in this tile's index space
    pub fn with_receiving_register(mut self, patch: &Domain<NDIM>, faces: [[bool; 2]; NDIM]) -> symbi_xpu::Result<Self> {
        assert_eq!(self.levels.len(), 1, "a receiving register lives on a tile without a fine level");
        assert!(self.flux_registers.is_empty());
        let state = &self.levels[0].state;
        let cons = &state.fields.cons;
        let n_oop = if state.fields.mhd.is_some() { DOF - NDIM } else { 0 };
        self.flux_registers.push(FluxRegister::receiving_on(patch, faces, cons.has_energy(), cons.chi_field().is_some(), n_oop)?);
        self.emf_registers.push(if state.fields.mhd.is_some() { Some(EmfRegister::receiving(patch, faces)?) } else { None });
        Ok(self)
    }

    /// apply the received register units of a tile beside a neighbor's patch in the single
    /// hierarchy's order: the EMF register to the outside faces this tile holds, cell B and
    /// the magnetic energy over the shell those faces bound, then the flux register to the
    /// gas and to the out-of-plane cell field
    pub fn reflux_received(&self) {
        assert_eq!(self.levels.len(), 1, "a tile beside a neighbor's patch holds no fine level");
        let l = &self.levels[0];
        if let Some(Some(reg)) = self.emf_registers.first() {
            let mhd = l.state.fields.mhd.as_ref().expect("an emf register lives on a magnetized level");
            let inv_dx: [f64; NDIM] = std::array::from_fn(|ax| 1.0 / l.state.geom.dx[ax]);
            reg.apply(&mhd.bface, &inv_dx);
            if reg.has_receiving() {
                self.reconcile_shell();
            }
        }
        if let Some(reg) = self.flux_registers.first() {
            if uniform_cartesian(&l.state) {
                reg.apply_uniform(&l.state.fields.cons, &l.state.geom.dx);
                if let Some(m) = l.state.fields.mhd.as_ref() {
                    if DOF > NDIM {
                        reg.apply_uniform_oop(&m.bcell, &l.state.geom.dx);
                    }
                }
            } else {
                let geo = l.state.geom.block_geometry(l.state.physics.metric);
                reg.apply(&l.state.fields.cons, &geo);
            }
        }
    }

    /// cell B and the magnetic energy over this tile's part of the shell around the patch
    fn reconcile_shell(&self) {
        let l = &self.levels[0];
        let (Some(Some(reg)), Some(mhd)) = (self.emf_registers.first(), l.state.fields.mhd.as_ref()) else { return };
        let shell = shell_around(reg.coverage(), &l.state.geom.interior);
        if shell.spaces.iter().all(|s| s.hi > s.lo) {
            bcell_from_bface_region(mhd, l.state.fields.cons.nrg_field(), &shell);
        }
    }

    /// recover the cells of a tile beside a neighbor's patch once every face around them is
    /// final: a tile that received no EMF unit takes cell B and the magnetic energy over its
    /// part of the shell here, from the corrected faces the neighbors sent it, then the
    /// primitives and the physical ghosts
    pub fn recover_beside_patch(&self) {
        assert_eq!(self.levels.len(), 1, "a tile beside a neighbor's patch holds no fine level");
        if !self.emf_registers.first().and_then(Option::as_ref).is_some_and(|r| r.has_receiving()) {
            self.reconcile_shell();
        }
        self.recover_level(0);
    }

    /// the recovery that closes a level's restriction on the single hierarchy, on a tile holding
    /// that level and none finer: the primitives from the conserved state over the whole level,
    /// then the physical ghosts. a post-step operator writing primitives directly (the excision
    /// floor) publishes their recovered values, as every tile holding part of the finer patch does
    pub fn recover_level(&self, level: usize) {
        assert_eq!(self.levels.len(), level + 1, "a tile holding a finer level recovers through its restriction");
        let l = &self.levels[level];
        prof("c2p", || l.kernels.c2p(&l.state));
        prof("ghost_fill", || self.fill_physical_ghosts(level));
    }

    /// discard every pending register unit of a rejected attempt: the contributing faces and
    /// slabs accumulated after their last send, and the landed units in the inboxes
    pub fn clear_pending_contributions(&self) {
        for reg in &self.flux_registers {
            reg.zero_contributions();
            reg.clear_inboxes();
        }
        for reg in self.emf_registers.iter().flatten() {
            reg.zero_contributions();
            reg.clear_inboxes();
        }
    }

    /// whether level `level` carries a flux register on its interface with a finer level,
    /// held here or on a neighbor tile
    pub fn has_flux_register(&self, level: usize) -> bool {
        level < self.flux_registers.len()
    }

    /// whether a register of this hierarchy sends or receives a unit across a cut
    pub fn transports_registers(&self) -> bool {
        self.flux_registers.iter().any(|r| r.has_contributing() || r.has_receiving())
            || self.emf_registers.iter().flatten().any(|r| r.has_contributing() || r.has_receiving())
    }

    /// whether the root lies on a uniform cartesian grid, where the register kernels carry
    /// a constant face area and cell volume
    pub fn root_is_uniform_cartesian(&self) -> bool {
        uniform_cartesian(&self.levels[0].state)
    }

    fn refine_levels(
        coarse: SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
        coarse_kernels: K,
        regions: &[RefinementRegion<NDIM>],
        whole: &[RefinementRegion<NDIM>],
        prolong_order: ProlongOrder,
        make_kernels: impl Fn(&SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>) -> K,
        cut_faces: CutFaces,
        layout: Option<([usize; NDIM], [bool; NDIM])>,
    ) -> symbi_xpu::Result<Self> {
        assert_eq!(regions.len(), whole.len(), "one whole patch per clipped region");
        // the axes along which the whole first patch spans a root that wraps: the patch has no
        // face there, on any tile
        let mut spans_wrap = [false; NDIM];
        // the root patch's real faces lying on a cut: contributing register faces
        let mut contributing = [[false; 2]; NDIM];
        let mut whole_coverages: Vec<Domain<NDIM>> = Vec::with_capacity(regions.len());
        let root_ng = coarse.geom.ng;
        let mut levels = vec![LevelData {
            state: coarse,
            kernels: coarse_kernels,
            prim_old: None,
            prim_lerp: None,
            prolong_sweep: std::sync::OnceLock::new(),
            band_departure: std::sync::OnceLock::new(),
            bcell_old: None,
            bface_old: None,
            attempt_entry: None,
            cut_sides: [[false; 2]; NDIM],
            patch_interior: None,
            face_stencil: None,
            coverage: None,
            flux_eq: None,
            residual_eq: None,
            cons_eq: None,
            prim_eq: None,
        }];

        for (region, whole_region) in regions.iter().zip(whole) {
            let parent = &levels.last().unwrap().state;
            whole_coverages.push(region_to_domain(whole_region, &parent.geom.x_lo, &parent.geom.dx, &parent.geom.index_offset));
            if parent.fields.mhd.is_some() {
                assert!(
                    NDIM == 2 || NDIM == 3,
                    "mhd refinement: the staggered CT substrate runs on a 2.5D or 3D grid"
                );
                assert!(
                    parent.geom.coords == symbi_geometry::Geometry::Cartesian,
                    "mhd refinement: cartesian only (the emf-reflux curl coefficients are 1/dx)"
                );
            }
            let coverage = region_to_domain(region, &parent.geom.x_lo, &parent.geom.dx, &parent.geom.index_offset);
            // a region covering its whole parent level leaves the level no leaf cell; such a run
            // wants a finer root grid, so it is refused here with the reason. a tile of the level
            // (a side on a cut) lying wholly under the region leaves leaf cells on other tiles,
            // and evolves none of its own
            let covers_parent = (0..NDIM).all(|ax| {
                coverage.spaces[ax].lo <= parent.geom.interior.spaces[ax].lo
                    && coverage.spaces[ax].hi >= parent.geom.interior.spaces[ax].hi
            });
            let whole_level = if levels.len() == 1 {
                (0..NDIM).all(|ax| parent.boundaries.lo(ax) != BoundaryType::CoarseFine && parent.boundaries.hi(ax) != BoundaryType::CoarseFine)
            } else {
                levels.last().unwrap().cut_sides.iter().flatten().all(|&cut| !cut)
            };
            if covers_parent && whole_level {
                return Err(symbi_xpu::XpuError {
                    operation: "refinement region",
                    code: 0,
                    detail: format!(
                        "region [{:?}, {:?}] covers its parent level entirely, leaving no leaf \
                         cell on it; a run resolved at that spacing everywhere wants a finer root \
                         grid, or a region strictly inside the parent",
                        region.x_lo, region.x_hi
                    ),
                });
            }
            // a fine side flush with a periodic parent face inherits the periodic boundary and
            // wraps onto the fine level's own opposite side, which is the domain's opposite side
            // only when the patch spans the whole axis
            for ax in 0..NDIM {
                let (c, p) = (&coverage.spaces[ax], &parent.geom.interior.spaces[ax]);
                let lo = c.lo == p.lo && parent.boundaries.lo(ax) == BoundaryType::Periodic;
                let hi = c.hi == p.hi && parent.boundaries.hi(ax) == BoundaryType::Periodic;
                if lo != hi {
                    return Err(symbi_xpu::XpuError {
                        operation: "refinement region",
                        code: 0,
                        detail: format!(
                            "a level-{} refinement patch reaches the periodic face of axis {ax} on one side \
                             only: its fine level would wrap onto its own opposite side; span the whole \
                             periodic axis or keep the patch clear of both faces",
                            levels.len()
                        ),
                    });
                }
            }
            // a patch face lying on a root cut: the fine faces on it live on this tile and
            // the coarse cell their reflux corrects lives on the neighbor. a driver carrying
            // register contributions across cuts takes it as a contributing face; the others
            // refuse it. the root's coarse-fine boundary is a cut by construction (a single
            // hierarchy's root has physical boundaries alone).
            let mut real_on_cut = [[false; 2]; NDIM];
            if levels.len() == 1 {
                if let Some((global_cells, wraps)) = layout {
                    let whole_face = &whole_coverages.last().unwrap().spaces;
                    spans_wrap = std::array::from_fn(|ax| {
                        let off = parent.geom.index_offset[ax];
                        wraps[ax] && whole_face[ax].lo + off == 0 && whole_face[ax].hi + off == global_cells[ax] as isize
                    });
                }
                for ax in 0..NDIM {
                    if spans_wrap[ax] {
                        continue;
                    }
                    let whole_face = &whole_coverages.last().unwrap().spaces[ax];
                    real_on_cut[ax][0] = coverage.spaces[ax].lo == parent.geom.interior.spaces[ax].lo
                        && whole_face.lo == coverage.spaces[ax].lo
                        && parent.boundaries.lo(ax) == BoundaryType::CoarseFine;
                    real_on_cut[ax][1] = coverage.spaces[ax].hi == parent.geom.interior.spaces[ax].hi
                        && whole_face.hi == coverage.spaces[ax].hi
                        && parent.boundaries.hi(ax) == BoundaryType::CoarseFine;
                    if (real_on_cut[ax][0] || real_on_cut[ax][1]) && cut_faces == CutFaces::Refuse {
                        return Err(symbi_xpu::XpuError {
                            operation: "refinement region",
                            code: 0,
                            detail: format!(
                                "a refinement patch face on axis {ax} lies on a decomposition cut: the \
                                 coarse cell its reflux corrects is on the neighbor tile, which is \
                                 outside the refined multi-device path; move the cut or the patch \
                                 face by at least one root cell"
                            ),
                        });
                    }
                }
                contributing = real_on_cut;
            } else {
                // a deeper patch face on a cut of its parent level: the parent cell its reflux
                // corrects is on the neighbor tile, and register contributions cross root cuts
                // alone
                let parent_cuts = levels.last().unwrap().cut_sides;
                let level = levels.len();
                for ax in 0..NDIM {
                    let whole_face = &whole_coverages.last().unwrap().spaces[ax];
                    let on_cut = (coverage.spaces[ax].lo == parent.geom.interior.spaces[ax].lo
                        && whole_face.lo == coverage.spaces[ax].lo
                        && parent_cuts[ax][0])
                        || (coverage.spaces[ax].hi == parent.geom.interior.spaces[ax].hi
                            && whole_face.hi == coverage.spaces[ax].hi
                            && parent_cuts[ax][1]);
                    if on_cut {
                        return Err(symbi_xpu::XpuError {
                            operation: "refinement region",
                            code: 0,
                            detail: format!(
                                "a level-{level} refinement patch face on axis {ax} lies on a decomposition \
                                 cut: the level-{} cell its reflux corrects is on the neighbor tile, and \
                                 register contributions cross root cuts alone; move the cut or the patch \
                                 face by at least one cell of level {}",
                                level - 1,
                                level - 1
                            ),
                        });
                    }
                }
            }
            // fine ghost depth: one above the root's (the prolongation stencil
            // is one order above the evolution reconstruction).
            let ng = root_ng + 1;
            validate_coverage(&coverage, parent, ng, prolong_order, real_on_cut).map_err(|detail| {
                symbi_xpu::XpuError { operation: "refinement region", code: 0, detail }
            })?;
            // a side of the fine level lying on a cut the patch continues across
            let cut_sides: [[bool; 2]; NDIM] = std::array::from_fn(|ax| {
                let whole = &whole_coverages.last().unwrap().spaces[ax];
                let (c, p) = (&coverage.spaces[ax], &parent.geom.interior.spaces[ax]);
                [
                    c.lo == p.lo && parent.boundaries.lo(ax) == BoundaryType::CoarseFine && whole.lo != c.lo,
                    c.hi == p.hi && parent.boundaries.hi(ax) == BoundaryType::CoarseFine && whole.hi != c.hi,
                ]
            });

            let n_cells: [usize; NDIM] =
                std::array::from_fn(|ax| coverage.spaces[ax].size() * RATIO);
            let interior_lo: [isize; NDIM] =
                std::array::from_fn(|ax| coverage.spaces[ax].lo * RATIO as isize);
            let dx: [f64; NDIM] = std::array::from_fn(|ax| parent.geom.dx[ax] / RATIO as f64);

            let mut boundaries = Boundaries::<NDIM>::uniform(BoundaryType::CoarseFine);
            for ax in 0..NDIM {
                if coverage.spaces[ax].lo == parent.geom.interior.spaces[ax].lo {
                    boundaries.0[ax][0] = parent.boundaries.lo(ax);
                }
                if coverage.spaces[ax].hi == parent.geom.interior.spaces[ax].hi {
                    boundaries.0[ax][1] = parent.boundaries.hi(ax);
                }
            }

            let mut fine = SimStateGeneric::new_at(
                parent.physics.regime,
                parent.physics.eos,
                parent.physics.metric,
                interior_lo,
                n_cells,
                parent.geom.x_lo,
                dx,
                ng,
                boundaries,
                parent.cfl,
                parent.timestepping,
                0,
            )?;
            // the fine level shares the parent's origin and places its cells by whole-grid fine
            // index, so a tile's fine cells sit where the single hierarchy's sit, bit for bit
            fine.geom.index_offset = std::array::from_fn(|ax| parent.geom.index_offset[ax] * RATIO as isize);
            assert!(
                parent.geom.maps.is_none(),
                "refinement: non-uniform axis maps need a per-level map split (not built)"
            );
            // the passive scalar is a run-level opt-in taken on the root, and a fine level is
            // built fresh, so the dye slots have to be carried down explicitly. the slots are what
            // hold the concentration a fine level's own cells advect.
            let fine = if parent.fields.cons.chi_field().is_some() {
                fine.with_passive_scalar()?
            } else {
                fine
            };
            // a magnetized parent carrying the rollback snapshot hands the same to its fine
            // level, so a rejection restores every level's faces
            let fine = if parent.fields.mhd.as_ref().is_some_and(|m| m.step_snapshot.is_some()) {
                fine.with_rollback_snapshot()?
            } else {
                fine
            };
            // the retry exclusion is the run's, so every level decides a step's rejection alike
            let mut fine = fine;
            fine.retry_exclusion = parent.retry_exclusion;

            let kernels = make_kernels(&fine);
            let last = levels.last_mut().unwrap();
            last.prim_old = Some(PrimFieldsGeneric::zeros_with_pressure(
                &last.state.geom.allocated,
                last.state.fields.prim.pre_field().is_some(),
            )?);
            last.prim_lerp = Some(PrimFieldsGeneric::zeros_with_pressure(
                &last.state.geom.allocated,
                last.state.fields.prim.pre_field().is_some(),
            )?);
            // the dye concentration is time-interpolated onto the fine coarse-fine ghosts like the
            // rest of the primitive state, so the parent's step-start copy has to carry it.
            if last.state.fields.prim.chi_field().is_some() {
                let alloc = last.state.geom.allocated.clone();
                last.prim_old.as_mut().unwrap().alloc_chi(&alloc)?;
                last.prim_lerp.as_mut().unwrap().alloc_chi(&alloc)?;
            }
            if let Some(pmhd) = last.state.fields.mhd.as_ref() {
                last.bcell_old = Some(array_field_zeros(&last.state.geom.allocated)?);
                // per-component staggered domains (face axis + transverse halo).
                let mut bf: Vec<Field<f64, NDIM, Mem>> = Vec::with_capacity(NDIM);
                for dd in 0..NDIM {
                    bf.push(Field::zeros(pmhd.bface[dd].domain())?);
                }
                last.bface_old = Some(bf.try_into().unwrap_or_else(|_| unreachable!()));
            }
            let patch_interior = Domain::new(std::array::from_fn(|ax| {
                let whole = &whole_coverages.last().unwrap().spaces[ax];
                Space { name: axis_name(ax), lo: whole.lo * RATIO as isize, hi: whole.hi * RATIO as isize }
            }));
            last.coverage = Some(coverage);
            let mut attempt_entry =
                ConsFieldsGeneric::zeros_with_energy(&fine.geom.allocated, fine.fields.cons.has_energy())?;
            if fine.fields.cons.chi_field().is_some() {
                attempt_entry.alloc_chi(&fine.geom.allocated)?;
            }
            levels.push(LevelData {
                state: fine,
                kernels,
                prim_old: None,
                prim_lerp: None,
                prolong_sweep: std::sync::OnceLock::new(),
                band_departure: std::sync::OnceLock::new(),
                bcell_old: None,
                bface_old: None,
                attempt_entry: Some(attempt_entry),
                cut_sides,
                patch_interior: Some(patch_interior),
                face_stencil: None,
                coverage: None,
                flux_eq: None,
                residual_eq: None,
                cons_eq: None,
                prim_eq: None,
            });
        }

        let mut flux_registers = Vec::with_capacity(levels.len().saturating_sub(1));
        let mut emf_registers = Vec::with_capacity(levels.len().saturating_sub(1));
        let root_cells: [usize; NDIM] =
            std::array::from_fn(|aa| levels[0].state.geom.interior.spaces[aa].size());
        let mut tracer_interfaces = Vec::with_capacity(levels.len().saturating_sub(1));
        for (ll, level) in levels.iter_mut().enumerate() {
            let donor_width = (level.kernels.reconstruction_reach() as usize)
                .max(prolong_order.ghost_width())
                + 1;
            // a side along which the root's first patch spans the wrapping domain continues across
            // the seam, as an interior side does
            let cut: [[bool; 2]; NDIM] = std::array::from_fn(|ax| {
                if ll == 0 && spans_wrap[ax] {
                    return [false; 2];
                }
                [
                    level.state.boundaries.lo(ax) == BoundaryType::CoarseFine,
                    level.state.boundaries.hi(ax) == BoundaryType::CoarseFine,
                ]
            });
            level.state.composite_ownership = match level.coverage.clone() {
                Some(coverage) => symbi_sim::state::CompositeOwnership::clipped(
                    coverage,
                    &whole_coverages[ll],
                    &level.state.geom.interior,
                    donor_width,
                    cut,
                ),
                None => symbi_sim::state::CompositeOwnership::new(None, &level.state.geom.interior, donor_width),
            };
        }
        for ll in 0..levels.len().saturating_sub(1) {
            // the out-of-plane magnetic components of a partially gridded chart reflux with the
            // gas; a fully gridded chart and a hydrodynamic run carry none.
            let oop_components = if levels[ll].state.fields.mhd.is_some() { DOF - NDIM } else { 0 };
            let has_energy = levels[ll].state.fields.cons.has_energy();
            let has_dye = levels[ll].state.fields.cons.chi_field().is_some();
            let coverage = levels[ll].coverage.as_ref().unwrap();
            let mut register =
                FluxRegister::new(coverage, &levels[ll].state.geom.interior, has_energy, has_dye, oop_components)?;
            if ll == 0 && contributing.iter().flatten().any(|&c| c) {
                register = register.with_contributing_faces(coverage, contributing, has_energy, has_dye, oop_components)?;
            }
            flux_registers.push(register);
            emf_registers.push(if levels[ll].state.fields.mhd.is_some() {
                let mut register = EmfRegister::new(coverage, &levels[ll].state.geom.interior)?;
                if ll == 0 && contributing.iter().flatten().any(|&c| c) {
                    register = register.with_contributing_planes(coverage, contributing)?;
                }
                Some(register)
            } else {
                None
            });
            tracer_interfaces.push(interface_faces(
                levels[ll].coverage.as_ref().unwrap(),
                root_cells,
                ll as u8,
            ));
        }
        let tracer_interface_ledgers = (0..tracer_interfaces.len())
            .map(|_| std::cell::RefCell::new(Vec::new()))
            .collect();

        Ok(Hierarchy {
            levels,
            injection_ledger: std::collections::BTreeMap::new(),
            continuous_injection: std::collections::BTreeMap::new(),
            continuous_accretion: std::collections::BTreeMap::new(),
            prolong_order,
            drain_suppressed: false,
            run_levels: None,
            flux_registers,
            emf_registers,
            tracer_interfaces,
            tracer_interface_ledgers,
            tracer_root_global_cells: root_cells,
            tracer_root_offset: [0; NDIM],
            root_wraps: [false; NDIM],
            fine_wraps: spans_wrap,
            crash: None,
            equilibrium_mask_radius: None,
            prev_dt_cfl: 0.0,
            forced_fine_rejection: None,
            forced_level_rejection: None,
            face_entry: None,
            level_face_entry: Vec::new(),
            priming: false,
            blend_across_tiles: false,
            pending_interface: None,
            continuous_domain: None,
            balance_aware_transfer: None,
        })
    }

    /// attach an immersed body collection to every level: the finest level
    /// carries the full collection (the sink and the accretion diagnostics
    /// have a single owner — the resolution truth), every coarser level a
    /// gravity-only proxy of each body (same mass / softening / motion, sink
    /// disabled), so the drain acts on the finest cells alone and the
    /// restriction then sets the covered coarse cells. body motion advances
    /// once per root step on the finest and is synced outward. the sink region
    /// lies inside the finest level — asserted every step as the bodies move.
    pub fn with_bodies(self, bodies: symbi_ib::BodyCollection<f64, NDIM>) -> Self {
        self.try_with_bodies(bodies).unwrap_or_else(|message| panic!("{message}"))
    }

    /// `with_bodies` returning the containment violation as an error instead of panicking, for
    /// a caller that reports it as a configuration fault.
    pub fn try_with_bodies(
        mut self,
        bodies: symbi_ib::BodyCollection<f64, NDIM>,
    ) -> Result<Self, String> {
        let n = self.levels.len();
        for ll in 0..n {
            let coll = if ll + 1 == n {
                bodies.clone()
            } else {
                gravity_only(&bodies)
            };
            self.levels[ll].state.attach_bodies(coll);
        }
        // a magnetic-slip sink makes the root step a transaction over every level: a rejected
        // attempt replays each level from its step entry, so each carries the rollback snapshot.
        if self.finest_has_magnetic_slip() {
            for ll in 0..self.levels.len() {
                let allocated = self.levels[ll].state.geom.allocated.clone();
                let has_energy = self.levels[ll].state.fields.cons.nrg_field().is_some();
                if let Some(mhd) = self.levels[ll].state.fields.mhd.as_mut() {
                    mhd.alloc_step_snapshot(&allocated, has_energy)
                        .expect("magnetic-slip step-entry rollback snapshot allocation");
                }
            }
        }
        if let Some(message) = self.sink_containment_violation() {
            return Err(message);
        }
        Ok(self)
    }

    /// exclude cells within `radius` of the origin from the stationarity diagnostic.
    ///
    /// overrides the default, which is the largest accretion radius among the attached bodies.
    /// pass `0.0` to consider every cell.
    pub fn with_equilibrium_mask(mut self, radius: f64) -> Self {
        self.equilibrium_mask_radius = Some(radius);
        self
    }

    /// force the coarse-fine transfer's equilibrium decomposition on or off, overriding the
    /// kernel-set default. `false` on a balanced gravitating hierarchy reproduces the seam
    /// entropy drain the decomposition removes — the measurement that identifies the transfer,
    /// over the reconstruction, as the load-bearing piece.
    pub fn balance_aware_transfer(mut self, on: bool) -> Self {
        self.balance_aware_transfer = Some(on);
        self
    }

    /// the radius the stationarity diagnostic ignores inside: the configured value, or the
    /// largest accretion radius among the bodies, or zero when neither exists.
    fn resolved_equilibrium_mask(&self) -> f64 {
        if let Some(r) = self.equilibrium_mask_radius {
            return r;
        }
        self.levels
            .last()
            .map(|level| {
                level
                    .state
                    .immersed
                    .as_ref()
                    .map(|im| {
                        im.bodies
                            .bodies()
                            .iter()
                            .filter_map(|b| b.accretion_radius())
                            .fold(0.0_f64, f64::max)
                    })
                    .unwrap_or(0.0)
            })
            .unwrap_or(0.0)
    }

    /// attach per-body immersed-boundary shapes to every level. rigid walls remain active on
    /// uncovered coarse cells when a body crosses a coarse-fine boundary, so every level must
    /// evaluate the same shape. `None` entries keep the analytic sphere.
    pub fn attach_body_shapes(&mut self, shapes: Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>>) {
        for level in &mut self.levels {
            level.state.attach_body_shapes(shapes.clone());
        }
    }

    /// every accreting body's sink sphere must lie inside the finest level's
    /// interior — a sink straddling a coarse-fine boundary corrupts the mass
    /// accounting the refluxing protects.
    fn assert_sinks_inside_finest(&self) {
        self.assert_sinks_inside_finest_clipped();
    }

    /// the sink containment invariant, clipped to this hierarchy's root: the part of the
    /// sink sphere overlapping the root interior lies inside the finest level — on a
    /// decomposed tile the sphere may span cuts (each owning tile drains its own cells),
    /// so the constraint binds exactly the tiles the sphere touches. a 1-level hierarchy
    /// is a single grid with no coarse-fine boundary to straddle; the decomposed driver
    /// separately forbids sphere overlap on an unrefined tile of a refined run (refluxing
    /// protects drains inside refined regions).
    pub fn assert_sinks_inside_finest_clipped(&self) {
        if let Some(message) = self.sink_containment_violation() {
            panic!("{message}");
        }
    }

    /// the first sink whose support sphere leaves the finest level, described; `None` when
    /// every sink is contained or the hierarchy has a single level.
    pub fn sink_containment_violation(&self) -> Option<String> {
        if self.levels.len() < 2 {
            return None;
        }
        let root = &self.levels[0].state;
        let finest = &self.levels[self.levels.len() - 1].state;
        let im = finest.immersed.as_ref()?;
        let rg = &root.geom;
        let fg = &finest.geom;
        let dx_fine = fg.dx[0];
        // a staggered-field hierarchy contains the sphere the operators act on: the drain mask
        // and the slip shell to their f64 support, plus the stencil reach, so no drained or
        // slipped cell lies within reach of a coarse-fine ghost. a hydrodynamic hierarchy keeps
        // the accretion sphere as its containment.
        let strict = finest.fields.mhd.is_some();
        let mut violation: Option<String> = None;
        im.bodies.visit_accretion(|body| {
            if violation.is_some() {
                return;
            }
            let racc: f64 = if strict {
                sink_support_radius(body, dx_fine)
            } else {
                body.accretion_radius().unwrap_or(0.0)
            };
            // sphere-box overlap against the root interior, per axis.
            let mut clip_lo = [0.0f64; NDIM];
            let mut clip_hi = [0.0f64; NDIM];
            let mut overlaps = true;
            for ax in 0..NDIM {
                let rlo = rg.x_lo[ax] + (rg.interior.spaces[ax].lo + rg.index_offset[ax]) as f64 * rg.dx[ax];
                let rhi = rg.x_lo[ax] + (rg.interior.spaces[ax].hi + rg.index_offset[ax]) as f64 * rg.dx[ax];
                let p: f64 = body.position[ax];
                clip_lo[ax] = (p - racc).max(rlo);
                clip_hi[ax] = (p + racc).min(rhi);
                overlaps &= clip_lo[ax] < clip_hi[ax];
            }
            if !overlaps {
                return;
            }
            for ax in 0..NDIM {
                let flo = fg.x_lo[ax] + (fg.interior.spaces[ax].lo + fg.index_offset[ax]) as f64 * fg.dx[ax];
                let fhi = fg.x_lo[ax] + (fg.interior.spaces[ax].hi + fg.index_offset[ax]) as f64 * fg.dx[ax];
                if !(clip_lo[ax] >= flo && clip_hi[ax] <= fhi) {
                    violation = Some(format!(
                        "refinement x decomposition: body {} sink support sphere (radius {racc:.4}: the \
                         accretion sphere, the drain mask and slip shell to their f64 support, and the \
                         stencil reach) clip [{:.4}, {:.4}] leaves this tile's finest level \
                         [{flo:.4}, {fhi:.4}] on axis {ax}",
                        body.idx,
                        clip_lo[ax],
                        clip_hi[ax]
                    ));
                    return;
                }
            }
        });
        violation
    }

    /// true when any accretion sphere overlaps this hierarchy's root interior — the
    /// decomposed driver's guard input: on a refined run, draining is confined to tiles
    /// carrying a fine patch, since reflux protection covers the fine regions alone.
    pub fn accretion_overlaps_root(&self) -> bool {
        let root = &self.levels[0].state;
        let Some(im) = root.immersed.as_ref() else {
            return false;
        };
        let rg = &root.geom;
        let mut hit = false;
        im.bodies.visit_accretion(|body| {
            let racc: f64 = body.accretion_radius().unwrap_or(0.0);
            let mut overlaps = true;
            for ax in 0..NDIM {
                let rlo = rg.x_lo[ax] + (rg.interior.spaces[ax].lo + rg.index_offset[ax]) as f64 * rg.dx[ax];
                let rhi = rg.x_lo[ax] + (rg.interior.spaces[ax].hi + rg.index_offset[ax]) as f64 * rg.dx[ax];
                let p: f64 = body.position[ax];
                overlaps &= (p - racc).max(rlo) < (p + racc).min(rhi);
            }
            hit |= overlaps;
        });
        hit
    }

    /// the local half of the decomposed body step: reduce this tile's finest-level
    /// backward feedback (force/torque/accreted mass receipts) into its diagnostics
    /// accumulator. the caller consolidates the partials across tiles and applies the
    /// identical global delta everywhere via `apply_global_body_deltas`.
    pub fn finest_body_feedback(&mut self, dt: f64) {
        // the body couples to the run's finest level, which the tiles holding it partition; a
        // tile short of that level holds none of the cells the single hierarchy reduces, and its
        // coarser gas, covered there by the finer levels, adds nothing to the reaction
        if !self.levels[0].state.has_bodies() || self.levels.len() < self.run_levels() {
            return;
        }
        let fi = self.levels.len() - 1;
        let finest = &mut self.levels[fi];
        let needs_fb = finest
            .state
            .immersed
            .as_ref()
            .is_some_and(|im| im.bodies.needs_feedback());
        if needs_fb {
            prof("body_feedback", || {
                finest.kernels.body_feedback(&finest.state, dt)
            });
        }
    }

    /// drain this tile's finest-level body-delta partials (consolidated diagnostics),
    /// resetting the accumulator. empty when the tile carries no bodies.
    pub fn take_body_deltas(&mut self) -> Vec<symbi_ib::BodyDelta<f64, NDIM>> {
        let fi = self.levels.len() - 1;
        let Some(im) = self.levels[fi].state.immersed.as_mut() else {
            return Vec::new();
        };
        let deltas = im.diagnostics.consolidate();
        im.diagnostics.reset();
        deltas
    }

    /// the global half of the decomposed body step: apply the cross-tile-summed deltas +
    /// the prescribed orbit advance to this tile's finest bodies (identical input on every
    /// tile -> identical body state, the lockstep contract), record the global per-step
    /// exchange in the history, sync the advanced positions/velocities to the coarser
    /// levels, and re-check the clipped sink containment.
    pub fn apply_global_body_deltas(
        &mut self,
        deltas: &[symbi_ib::BodyDelta<f64, NDIM>],
        dt: f64,
        time: f64,
    ) {
        if let Err(message) = self.try_apply_global_body_deltas(deltas, dt, time) {
            panic!("{message}");
        }
    }

    /// `apply_global_body_deltas` returning the clipped sink-containment violation as an error,
    /// for a caller that ends its session by name instead of panicking
    pub fn try_apply_global_body_deltas(
        &mut self,
        deltas: &[symbi_ib::BodyDelta<f64, NDIM>],
        dt: f64,
        time: f64,
    ) -> Result<(), String> {
        if !self.levels[0].state.has_bodies() {
            return Ok(());
        }
        let fi = self.levels.len() - 1;
        {
            let finest = &mut self.levels[fi].state;
            let im = finest.immersed.as_mut().unwrap();
            symbi_ib::apply_body_deltas(&mut im.bodies, deltas, dt);
            im.history.push(time, dt, deltas);
        }
        let truth: Vec<_> = {
            let bodies = &self.levels[fi].state.immersed.as_ref().unwrap().bodies;
            (0..bodies.len())
                .map(|bb| (bodies.get(bb).position, bodies.get(bb).velocity))
                .collect()
        };
        for ll in 0..fi {
            if let Some(im) = self.levels[ll].state.immersed.as_mut() {
                for (bb, (pos, vel)) in truth.iter().enumerate() {
                    let body = im.bodies.get_mut(bb);
                    body.position = *pos;
                    body.velocity = *vel;
                }
            }
        }
        match self.sink_containment_violation() {
            Some(message) => Err(message),
            None => Ok(()),
        }
    }

    /// march the hierarchy to t_final: the root's cfl sets dt, advance_level
    /// recurses through the levels, then the root advances its clock and body
    /// state (mirroring evolve_with_callback).
    pub fn evolve(
        &mut self,
        t_final: f64,
    ) -> symbi_xpu::Result<symbi_sim::run_diagnostics::RunDiagnostics> {
        self.evolve_with_callback(t_final, u64::MAX, |_| std::ops::ControlFlow::Continue(()))
    }

    /// evolve with a periodic observer: the callback fires every `interval`
    /// root iterations and once after the final step, with the device queue
    /// drained first so it may read fields from the host (the same contract
    /// as the single-level sim::evolve::evolve_with_callback).
    pub fn evolve_with_callback(
        &mut self,
        t_final: f64,
        interval: u64,
        callback: impl FnMut(&Self) -> std::ops::ControlFlow<()>,
    ) -> symbi_xpu::Result<symbi_sim::run_diagnostics::RunDiagnostics> {
        self.evolve_with_callback_impl(t_final, interval, true, callback)
    }

    /// continue a checkpoint-restored hierarchy whose primitive and staggered
    /// magnetic interiors represent the accepted step tail. physical ghost
    /// bands are reconstructed without re-running c2p or bcell recovery.
    pub fn resume_with_callback(
        &mut self,
        t_final: f64,
        interval: u64,
        callback: impl FnMut(&Self) -> std::ops::ControlFlow<()>,
    ) -> symbi_xpu::Result<symbi_sim::run_diagnostics::RunDiagnostics> {
        for ll in 0..self.levels.len() {
            self.fill_physical_ghosts(ll);
        }
        self.evolve_with_callback_impl(t_final, interval, false, callback)
    }

    fn evolve_with_callback_impl(
        &mut self,
        t_final: f64,
        interval: u64,
        prepare_initial_state: bool,
        mut callback: impl FnMut(&Self) -> std::ops::ControlFlow<()>,
    ) -> symbi_xpu::Result<symbi_sim::run_diagnostics::RunDiagnostics> {
        symbi_sim::retry_guard::assert_one_retry_exclusion(self.levels.iter().map(|l| &l.state.retry_exclusion));
        // homologous mesh motion applies on a single grid: the hierarchy's flux
        // registers and transfer operators are written for a fixed mesh. a
        // 1-level hierarchy runs without registers or transfer, so motion is
        // safe there (it reproduces the single-grid evolve); refined runs are
        // refused.
        assert!(
            self.levels.len() == 1
                || self
                    .levels
                    .iter()
                    .all(|l| { l.state.motion.a_dot == 0.0 && l.state.motion.a == 1.0 }),
            "refinement: mesh motion is uni-grid only (the registers carry no scale factor)"
        );
        if prepare_initial_state {
            self.init_levels();
        }
        // the uni-grid march is the one-tile case of the shared driver: no decomposition
        // context, so the exchange schedule is empty and the step runs the uni-grid
        // transaction. the driver owns the watchdog, the step-panic catch, and the
        // observer cadence for every shape.
        let diagnostics = evolve_tiles::<
            R,
            NDIM,
            DOF,
            M,
            E,
            S,
            Mem,
            K,
            symbi_sim::decomp::LocalCopy,
            _,
        >(
            std::slice::from_mut(self),
            None,
            t_final,
            interval,
            |ts| callback(&ts[0]),
        );
        Ok(diagnostics)
    }

    /// run the one-time IC preparation at the current time: bcell-from-bface,
    /// c2p to populate the primitive buffer from the seeded conserved state, the
    /// coarse-fine prolong, and the ghost fill. the drivers call this internally
    /// at evolve start; a caller that snapshots state at t=0 (the binding's
    /// initial-condition checkpoint) must call it first, so that snapshot carries
    /// recovered primitives and cell-centered B rather than zeroed scratch
    /// buffers. idempotent.
    pub fn prime(&mut self) {
        self.init_levels();
    }

    /// advance exactly `nsteps` root steps (the smoke/diagnostic driver).
    pub fn evolve_steps(&mut self, nsteps: u64) -> symbi_xpu::Result<()> {
        self.init_levels();
        for _ in 0..nsteps {
            self.step_root(f64::INFINITY);
        }
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        Ok(())
    }

    /// initial setup. mhd pairs first restrict the staggered B downward
    /// (finest pair first) so the invariant "coarse face = average of its fine
    /// faces" holds from step 0 — the emf reflux preserves it thereafter, and
    /// divB exactness across the interface depends on it. then coarsest-first:
    /// c2p, coarse-fine prolong (alpha = 1: the current coarse prims; prim_old
    /// is still zeroed and contributes (1-1)*0), then the physical ghost fill,
    /// with the loud c2p check. idempotent.
    fn init_levels(&mut self) {
        self.priming = true;
        self.init_levels_body();
        self.priming = false;
    }

    fn init_levels_body(&mut self) {
        for ll in (0..self.levels.len().saturating_sub(1)).rev() {
            if self.levels[ll].state.fields.mhd.is_none() {
                continue;
            }
            let (lo, hi) = self.levels.split_at(ll + 1);
            let coarse = &lo[ll];
            let fine = &hi[0];
            let cov = coarse.coverage.as_ref().unwrap();
            let cmhd = coarse.state.fields.mhd.as_ref().unwrap();
            let fmhd = fine.state.fields.mhd.as_ref().unwrap();
            for aa in 0..DOF {
                restrict_cell_field(&fmhd.bcell[aa], &cmhd.bcell[aa], cov);
            }
            restrict_bface(&fmhd.bface, &cmhd.bface, cov);
            bcell_from_bface_region(cmhd, coarse.state.fields.cons.nrg_field(), cov);
        }
        for ll in 0..self.levels.len() {
            let lvl = &self.levels[ll];
            lvl.kernels.c2p(&lvl.state);
            if ll > 0 {
                self.prolong_cf(ll, 1.0);
            }
            self.fill_physical_ghosts(ll);
            // the error scan reads c2p_error from the host mid-queue.
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let err = scan_c2p_errors(&lvl.state);
            if err.is_err() {
                let first_state = symbi_sim::hydro_ops::first_c2p_failure_state(&lvl.state)
                    .unwrap_or_else(|| "unavailable".to_string());
                panic!(
                    "hierarchy: c2p failed on initial conditions at level {} (time {:.4e}): {}; \
                     first state: {}",
                    ll, lvl.state.time, err, first_state
                );
            }
        }
    }

    /// restore every level from a checkpoint, adding levels beyond the file's own depth.
    ///
    /// this is the bootstrap ladder: a rung converges at its own resolution and the next rung
    /// resumes it with one more level, so the expensive early transient is paid once at the
    /// coarsest depth instead of once per depth. the levels the file carries are loaded; the levels
    /// beyond it are injected from their parents.
    ///
    /// the split between the two is taken from the file's own level count rather than the config:
    /// a run that guessed would either inject over converged data (replacing a level with a coarser
    /// copy of itself) or try to load a level absent from the file, and the first of those is
    /// silent.
    ///
    /// each loaded level's grid is verified against the file before its data is read. a deeper
    /// restart is only meaningful when level `i` occupies the same region regardless of the total
    /// depth, which is a property of a refinement schedule rather than of this code.
    pub fn restore_from_checkpoint(&mut self, path: &str) -> Result<usize, String> {
        let stored =
            symbi_sim::checkpoint::checkpoint_level_count(path).map_err(|e| format!("{e}"))?;
        if stored > self.levels.len() {
            return Err(format!(
                "checkpoint '{path}' carries {stored} refinement level(s) but this run builds {}. \
                 a restart may add levels to a checkpoint, never drop them: discarding a level \
                 means restricting its data onto its parent, which is a different operation and \
                 loses the resolution the checkpoint was run to obtain.",
                self.levels.len()
            ));
        }
        for (level_index, level) in self.levels.iter_mut().enumerate().take(stored) {
            symbi_sim::checkpoint::verify_checkpoint_level_geometry(
                &level.state,
                path,
                level_index,
            )
            .map_err(|e| format!("{e}"))?;
            symbi_sim::checkpoint::load_checkpoint_level(&mut level.state, path, level_index)
                .map_err(|e| format!("{e}"))?;
        }
        for level_index in stored..self.levels.len() {
            self.inject_level_from_parent(level_index)?;
            // the injected level inherits its parent's clock: it enters the run fresh and takes the
            // parent's elapsed history as its own. matching level times keep the subcycle
            // synchronized from the very first root step.
            let (time, dt, iteration, motion) = {
                let parent = &self.levels[level_index - 1].state;
                (parent.time, parent.dt, parent.iteration, parent.motion)
            };
            let fresh = &mut self.levels[level_index].state;
            fresh.time = time;
            fresh.dt = dt;
            fresh.iteration = iteration;
            fresh.motion = motion;
        }
        Ok(stored)
    }

    /// initialize `level` from its parent by piecewise-constant injection: every fine cell takes
    /// the conserved state of the coarse cell containing it.
    ///
    /// exactly conservative on cell averages — a coarse average replicated to its `RATIO^D`
    /// children preserves the integral over the coarse cell — so the new level enters the run
    /// carrying precisely the mass, momentum and energy its parent held over the same region.
    ///
    /// that exactness is the justification for injection over a higher-order
    /// prolongation. adding a level to a converged solution adds resolution, and the
    /// structure the finer cells can then represent is generated by the flow itself within a few
    /// crossing times of the new level's own (small) region, which is negligible against the time
    /// such a level is then run for. a smoother initial transient is all a higher-order operator
    /// would buy.
    ///
    /// refused for MHD: a face-centered field needs a divergence-preserving prolongation, and
    /// injecting it cell-wise seeds a `div B` that constrained transport then preserves for the
    /// rest of the run.
    pub fn inject_level_from_parent(&mut self, level: usize) -> Result<(), String> {
        if level == 0 || level >= self.levels.len() {
            return Err(format!(
                "level {level} cannot be injected from a parent: the hierarchy has {} level(s) and \
                 the root has none",
                self.levels.len()
            ));
        }
        if self.levels[level].state.fields.mhd.is_some() {
            return Err(format!(
                "level {level} carries a magnetic field: initializing it by cell-wise injection \
                 would seed a divergence the constrained transport cannot remove. a \
                 divergence-preserving prolongation of the face field is required."
            ));
        }

        let (coarser, finer) = self.levels.split_at_mut(level);
        let parent = &coarser[level - 1].state;
        let fine = &mut finer[0].state;

        // piecewise-constant prolongation is precisely parent injection: the stencil is the single
        // covering coarse cell, so a coarse average lands unchanged on each of its children and the
        // integral over the coarse cell is preserved exactly.
        //
        // routed through the same rendered kernel every other coarse-to-fine transfer uses, so a
        // device-resident hierarchy injects on the device. the source and destination buffers are
        // the same field, which collapses the time interpolation the coarse-fine ghost path needs:
        // the parent's state serves as both the old and the new level.
        let region = fine.geom.interior.clone();
        let inject = |src: &Field<f64, NDIM, Mem>, dst: &Field<f64, NDIM, Mem>| {
            crate::refinement::transfer::prolong_field(
                src,
                src,
                dst,
                &region,
                ProlongOrder::Pcm,
                0.0,
            );
        };

        inject(&parent.fields.cons.den, &fine.fields.cons.den);
        for dd in 0..DOF {
            inject(&parent.fields.cons.mom[dd], &fine.fields.cons.mom[dd]);
        }
        if let (Some(pn), Some(fnn)) =
            (parent.fields.cons.nrg_field(), fine.fields.cons.nrg_field())
        {
            inject(pn, fnn);
        }
        if let (Some(pc), Some(fc)) = (parent.fields.cons.chi_field(), fine.fields.cons.chi_field())
        {
            inject(pc, fc);
        }

        // the census, the diagnostics and the first flux evaluation all read primitives; injection
        // writes the conserved state alone, so the recovery has to run before the level is stepped.
        finer[0].kernels.c2p(&fine.store);
        Ok(())
    }

    /// the cfl-limited root dt this hierarchy would take, ahead of the t_final clamp: the min over
    /// every level of `cfl(level) * RATIO^level` (covered coarse cells are conservative averages,
    /// so a fast fine-only feature is diluted out of the root cfl; level l subcycles ratio^l times,
    /// so its limit enters scaled by ratio^l). exposed for the decomposed driver: it takes the
    /// global min of this across tiles, then drives each tile with `evolve(t + global_dt)` -- since
    /// the global dt is the min, each tile's internal `dt_cfl.min(global_dt)` collapses to
    /// global_dt, so the existing clamp alone produces a lockstep root step.
    /// the raw cfl estimate over every level: `cfl(level) * RATIO^level` (level l subcycles
    /// ratio^l times, so its limit enters scaled by ratio^l), minimized across levels. a
    /// non-finite or non-positive per-level candidate propagates out unmodified so the
    /// callers' crash heuristics see it.
    fn raw_root_cfl(&self) -> f64 {
        let mut dt = f64::INFINITY;
        for (_, candidate) in self.level_cfl_candidates() {
            if !candidate.is_finite() || candidate <= 0.0 {
                return candidate;
            }
            dt = dt.min(candidate);
        }
        dt
    }

    /// whether level `ll` evolves some cell of its own: a tile's level lying wholly under a finer
    /// level holds restrictions alone and sets no timestep
    pub fn level_evolves_cells(&self, ll: usize) -> bool {
        let state = &self.levels[ll].state;
        !state.composite_ownership.evolution_regions(&state.geom.interior).is_empty()
    }

    /// the raw timestep candidate in root units of every level evolving some cell of its own, as
    /// `(level, candidate)` in level order; a candidate that is NaN or non-positive is returned as
    /// it is, so a caller judges it before any clamp can hide it
    pub fn level_cfl_candidates(&self) -> Vec<(usize, f64)> {
        prof("cfl", || {
            (0..self.levels.len())
                .filter(|&ll| self.level_evolves_cells(ll))
                .map(|ll| {
                    let lvl = &self.levels[ll];
                    (ll, lvl.kernels.cfl(&lvl.state) * RATIO.pow(ll as u32) as f64)
                })
                .collect()
        })
    }

    /// the user clamp (max_dt > 0): pins the dt sequence across runs whose CFL estimators
    /// differ. a fatal candidate (NaN or non-positive) passes through unclamped, since
    /// `f64::min` would replace a NaN with the clamp
    pub fn clamp_dt(&self, dt_cfl: f64) -> f64 {
        let clamp = self.levels[0].state.max_dt;
        if clamp > 0.0 && !fatal_cfl_dt(dt_cfl) { dt_cfl.min(clamp) } else { dt_cfl }
    }

    pub fn root_cfl_dt(&self) -> f64 {
        // same full-grid wave-speed pass as `step_root`; instrumented under the same phase name so a
        // decomposed run attributes it identically.
        let dt_cfl = prof("cfl", || self.raw_root_cfl());
        self.clamp_dt(dt_cfl)
    }

    /// reject the fine stage `stage` of substep `substep` of root step `step` once: the
    /// attempt that reaches it requests a retry as a failed recovery would, and the retry
    /// runs clean. a test facility for the rollback of every driver
    pub fn reject_fine_once_at(&mut self, step: u64, substep: u8, stage: usize) {
        self.forced_fine_rejection = Some((step, substep, stage));
    }

    /// reject the stage `stage` of the substep `substep` of refined level `level` within root step
    /// `step` once, the substeps of the level numbered from zero across the whole root step (a
    /// level `l` takes `2^l` of them): the attempt that reaches it requests a retry as a failed
    /// recovery would, and the retry runs clean. a test facility for the rollback of every driver
    pub fn reject_level_once_at(&mut self, step: u64, level: usize, substep: u64, stage: usize) {
        self.forced_level_rejection = Some((step, level, substep, stage));
    }

    /// whether a forced fine rejection is still pending
    pub fn forced_fine_rejection_pending(&self) -> bool {
        self.forced_fine_rejection.is_some() || self.forced_level_rejection.is_some()
    }

    /// keep every finer level's conserved state at the root attempt's entry in a buffer of
    /// its own: the level's step snapshot is rewritten by each substep of the subcycle
    pub fn snapshot_attempt_entry(&self) {
        for level in self.levels.iter().skip(1) {
            if let Some(entry) = level.attempt_entry.as_ref() {
                restore_cons_region(&level.state.fields.cons, entry, &level.state.geom.allocated);
            }
        }
    }

    /// restore every finer level to the root attempt's entry after a rejection, the
    /// primitives re-derived and the physical ghosts filled as the root's own restore does
    /// the boundaries whose coarse-fine face halo a fine level prolongs: the sides on a cut
    /// the patch continues across take their halo from the fine exchange
    fn prolonged_face_boundaries(level: &LevelData<R, NDIM, DOF, M, E, S, Mem, K>) -> Boundaries<NDIM> {
        let mut prolonged = level.state.boundaries.clone();
        for ax in 0..NDIM {
            for side in 0..2 {
                if level.cut_sides[ax][side] {
                    prolonged.0[ax][side] = BoundaryType::Outflow;
                }
            }
        }
        prolonged
    }

    /// the parent faces the fine level's coarse-fine face halo reads, per component, in the
    /// parent's index space
    pub fn face_stencil_boxes(&self) -> [Option<Domain<NDIM>>; NDIM] {
        self.face_stencil_boxes_of(1)
    }

    /// the parent faces refined level `level`'s coarse-fine face halo reads, per component, in
    /// the parent's index space
    pub fn face_stencil_boxes_of(&self, level: usize) -> [Option<Domain<NDIM>>; NDIM] {
        let fine = &self.levels[level];
        face_stencil_boxes(&fine.state.geom.interior, &Self::prolonged_face_boundaries(fine))
    }

    /// the parent face stencils of the nested levels of a run of `run_levels` levels: every
    /// refined level this tile holds below the first reads its parent's faces from a stencil the
    /// driver gathers, and every level this tile holds that parents a level of the run keeps its
    /// faces at each substep's start as the old time level of the finer level's stencils. the
    /// first refined level's stencil and the root's attempt-entry faces come from
    /// `enable_face_stencil`
    pub fn enable_nested_face_stencils(&mut self, run_levels: usize) -> symbi_xpu::Result<()> {
        if self.levels[0].state.fields.mhd.is_none() {
            return Ok(());
        }
        self.level_face_entry = (0..self.levels.len()).map(|_| None).collect();
        for level in 1..self.levels.len() {
            if level + 1 < run_levels {
                let mhd = self.levels[level].state.fields.mhd.as_ref().expect("a magnetized level");
                let mut entry = Vec::with_capacity(NDIM);
                for d in 0..NDIM {
                    entry.push(Field::zeros(mhd.bface[d].domain())?);
                }
                self.level_face_entry[level] = Some(entry);
            }
        }
        for level in 2..self.levels.len() {
            let boxes = self.face_stencil_boxes_of(level);
            if boxes.iter().all(Option::is_none) {
                continue;
            }
            let mut old = Vec::with_capacity(NDIM);
            let mut new = Vec::with_capacity(NDIM);
            for b in &boxes {
                let b = b.as_ref().expect("a face halo reads every component");
                old.push(Field::zeros(b)?);
                new.push(Field::zeros(b)?);
            }
            self.levels[level].face_stencil = Some(FaceStencil { old, new, filled: std::cell::Cell::new(false) });
        }
        Ok(())
    }

    /// keep refined level `level`'s faces at its substep's start, the old time level of the next
    /// level's stencil, and invalidate that stencil until the driver gathers it after the stages
    pub fn snapshot_level_faces(&self, level: usize) {
        if let (Some(Some(entry)), Some(mhd)) =
            (self.level_face_entry.get(level), self.levels.get(level).and_then(|l| l.state.fields.mhd.as_ref()))
        {
            for d in 0..NDIM {
                copy_field(&mhd.bface[d], &entry[d]);
            }
        }
        if let Some(stencil) = self.levels.get(level + 1).and_then(|l| l.face_stencil.as_ref()) {
            stencil.filled.set(false);
        }
    }

    /// refined level `level`'s faces at its substep's start, the old-time source of a neighbor's
    /// stencil of the level below
    pub fn level_face_entry(&self, level: usize) -> Option<&[Field<f64, NDIM, Mem>]> {
        self.level_face_entry.get(level).and_then(|e| e.as_deref())
    }

    /// refined level `level`'s parent face stencil
    pub fn face_stencil_of(&self, level: usize) -> Option<&FaceStencil<NDIM, Mem>> {
        self.levels.get(level).and_then(|l| l.face_stencil.as_ref())
    }

    /// refined level `level`'s stencil holds the gathered parent faces of the current substep
    pub fn mark_face_stencil_filled_of(&self, level: usize) {
        if let Some(stencil) = self.face_stencil_of(level) {
            stencil.filled.set(true);
        }
    }

    /// read the fine level's coarse-fine face halo from a parent face stencil the driver
    /// gathers, and keep the root's faces at every attempt's entry as that stencil's old time
    /// level; a magnetized hierarchy whose fine level reads no parent face holds no stencil
    pub fn enable_face_stencil(&mut self) -> symbi_xpu::Result<()> {
        let Some(mhd) = self.levels[0].state.fields.mhd.as_ref() else { return Ok(()) };
        let mut entry = Vec::with_capacity(NDIM);
        for d in 0..NDIM {
            entry.push(Field::zeros(mhd.bface[d].domain())?);
        }
        self.face_entry = Some(entry);
        if self.levels.len() < 2 {
            return Ok(());
        }
        let boxes = self.face_stencil_boxes();
        if boxes.iter().all(Option::is_none) {
            return Ok(());
        }
        let mut old = Vec::with_capacity(NDIM);
        let mut new = Vec::with_capacity(NDIM);
        for b in &boxes {
            let b = b.as_ref().expect("a face halo reads every component");
            old.push(Field::zeros(b)?);
            new.push(Field::zeros(b)?);
        }
        self.levels[1].face_stencil = Some(FaceStencil { old, new, filled: std::cell::Cell::new(false) });
        Ok(())
    }

    /// the root's faces at the attempt's entry, the old-time source of a neighbor's stencil
    pub fn face_entry(&self) -> Option<&[Field<f64, NDIM, Mem>]> {
        self.face_entry.as_deref()
    }

    /// keep the root's faces at the attempt's entry and invalidate the fine level's stencil
    /// until the driver gathers it after the root stages
    pub fn snapshot_face_entry(&self) {
        if let (Some(entry), Some(mhd)) = (self.face_entry.as_ref(), self.levels[0].state.fields.mhd.as_ref()) {
            for d in 0..NDIM {
                copy_field(&mhd.bface[d], &entry[d]);
            }
        }
        if let Some(stencil) = self.levels.get(1).and_then(|l| l.face_stencil.as_ref()) {
            stencil.filled.set(false);
        }
    }

    /// the fine level's stencil holds the gathered parent faces of the current attempt
    pub fn mark_face_stencil_filled(&self) {
        if let Some(stencil) = self.levels.get(1).and_then(|l| l.face_stencil.as_ref()) {
            stencil.filled.set(true);
        }
    }

    pub fn face_stencil(&self) -> Option<&FaceStencil<NDIM, Mem>> {
        self.levels.get(1).and_then(|l| l.face_stencil.as_ref())
    }

    pub fn restore_attempt_entry(&self) {
        // every level stands at its step entry again, the clock its ghost refills read
        for level in &self.levels {
            level.state.reset_state_clock();
        }
        for level in self.levels.iter().skip(1) {
            if let Some(stencil) = level.face_stencil.as_ref() {
                stencil.filled.set(false);
            }
        }
        for (ll, level) in self.levels.iter().enumerate().skip(1) {
            if let Some(entry) = level.attempt_entry.as_ref() {
                restore_cons_region(entry, &level.state.fields.cons, &level.state.geom.allocated);
                level.kernels.c2p(&level.state);
                self.fill_physical_ghosts(ll);
            }
        }
    }

    pub fn snapshot_retry_sidecars(&self) -> HierarchyRetrySidecars<NDIM> {
        HierarchyRetrySidecars {
            discrete: self
                .levels
                .iter()
                .map(|level| level.state.tracers.clone())
                .collect(),
            continuous: self
                .levels
                .iter()
                .map(|level| {
                    level
                        .state
                        .continuous_tracers
                        .as_ref()
                        .map(|tracers| tracers.snapshot_retry())
                        .transpose()
                        .unwrap_or_else(|detail| {
                            panic!("continuous tracer retry snapshot: {detail}")
                        })
                })
                .collect(),
            bodies: self
                .levels
                .iter()
                .map(|level| {
                    level
                        .state
                        .immersed
                        .as_ref()
                        .map(|immersed| immersed.bodies.clone())
                })
                .collect(),
            motion: self.levels.iter().map(|level| level.state.motion).collect(),
            clocks: self
                .levels
                .iter()
                .map(|level| (level.state.time, level.state.iteration, level.state.dt))
                .collect(),
            censuses: self
                .levels
                .iter()
                .map(|level| {
                    level
                        .state
                        .censuses
                        .iter()
                        .map(|registered| {
                            (registered.history.clone(), registered.last_sample.clone())
                        })
                        .collect()
                })
                .collect(),
        }
    }

    pub fn restore_retry_sidecars(&mut self, snapshot: &HierarchyRetrySidecars<NDIM>) {
        for (ii, level) in self.levels.iter_mut().enumerate() {
            level.state.tracers.clone_from(&snapshot.discrete[ii]);
            if let (Some(tracers), Some(saved)) = (
                level.state.continuous_tracers.as_mut(),
                snapshot.continuous[ii].as_ref(),
            ) {
                tracers
                    .restore_retry(saved)
                    .unwrap_or_else(|detail| panic!("continuous tracer retry restore: {detail}"));
            }
            if let (Some(immersed), Some(bodies)) =
                (level.state.immersed.as_mut(), snapshot.bodies[ii].as_ref())
            {
                immersed.bodies.clone_from(bodies);
                // the per-step deltas the rejected attempt booked (a drain's accreted mass and
                // force, a feedback force) are discarded with it: the accumulator is empty at
                // every step entry, the accepted-root feedback having consolidated and reset it.
                immersed.diagnostics.reset();
            }
            level.state.motion = snapshot.motion[ii];
            level.state.time = snapshot.clocks[ii].0;
            level.state.iteration = snapshot.clocks[ii].1;
            level.state.dt = snapshot.clocks[ii].2;
            assert_eq!(
                level.state.censuses.len(),
                snapshot.censuses[ii].len(),
                "census registrations changed during a retryable step"
            );
            for (registered, (history, last_sample)) in
                level.state.censuses.iter_mut().zip(&snapshot.censuses[ii])
            {
                registered.history.clone_from(history);
                registered.last_sample.clone_from(last_sample);
            }
        }
        self.injection_ledger.clear();
        self.continuous_injection.clear();
        self.continuous_accretion.clear();
        for ledger in &self.tracer_interface_ledgers {
            ledger.borrow_mut().clear();
        }
    }

    /// one root step: cfl-limited dt (clamped to t_final), the recursive level
    /// advance, then the root clock + body state. every level limits the root
    /// step — covered coarse cells are conservative averages of fine data, so
    /// a fast feature resolved only on the fine level is diluted out of the
    /// root's own cfl; level l subcycles ratio^l times, so its limit enters
    /// scaled by ratio^l.
    /// the coarse-fine order pairing: the prolongation's smooth-data polynomial degree must
    /// be at least the evolution reconstruction's stencil reach (= its degree plus one: pcm
    /// evolution -> plm prolong, plm -> ppm, ppm -> quartic). the prolonged ghost averages then
    /// carry error one order above the interior truncation, so the interface layer's
    /// flux-divergence order loss leaves the interior order intact. a shallower prolongation
    /// degrades it at every refinement boundary with nothing in the output to show for it, so
    /// the pairing is checked before each root step. a single-level hierarchy is one grid with
    /// no coarse-fine boundary and carries any pairing.
    pub fn assert_cf_order_pairing(&self) {
        if self.levels.len() < 2 {
            return;
        }
        for lvl in &self.levels {
            assert!(
                lvl.kernels.reconstruction_reach() <= self.prolong_order.degree(),
                "evolution reconstruction reach {} exceeds the {:?} prolongation's \
                 degree {}: the coarse-fine ghost averages would carry a lower-order \
                 error than the interior truncation and the refinement boundary would \
                 degrade the interior order; pair this reconstruction with a \
                 higher-degree prolongation (ppm evolution -> quartic)",
                lvl.kernels.reconstruction_reach(),
                self.prolong_order,
                self.prolong_order.degree(),
            );
        }
    }

    fn step_root(&mut self, t_final: f64) {
        self.assert_cf_order_pairing();
        // the per-root-step wave-speed pass + global min reduction. instrumented because it is a
        // full-grid read of prim on every level, once per step, and sits outside the substage loop:
        // at a small domain / high step count it is a large fraction of the step that per-phase
        // timing inside the substage loop would leave unattributed.
        let Some(dt) = agree_tiles_dt(std::slice::from_mut(self), None, t_final) else {
            return;
        };
        self.step_root_with_dt(dt);
    }

    /// record a fatal timestep estimate and halt on this state: the evolve loop reports it and the
    /// driver snapshots `.crashed.h5` from the last computed state. a larger positive estimate is
    /// never itself unsafe (it means the maximum signal speed fell), so a fatal estimate is only a
    /// NaN, infinite or non-positive one.
    fn record_cfl_crash(&mut self, dt_cfl: f64) {
        let root = &self.levels[0].state;
        self.crash = Some(CrashReport {
            iter: root.iteration,
            time: root.time,
            dt_cfl,
            dt_prev: self.prev_dt_cfl,
            panic: None,
            halt: None,
        });
    }

    /// record a root step the shared transaction halted: the entry it restored is the state the
    /// crash publishes
    fn record_step_halt(&mut self, halt: &StepHalt) {
        let root = &self.levels[0].state;
        self.crash = Some(CrashReport {
            iter: root.iteration,
            time: root.time,
            dt_cfl: f64::NAN,
            dt_prev: self.prev_dt_cfl,
            panic: None,
            halt: Some(halt.clone()),
        });
    }

    /// whether a root attempt can be rejected: a level's FOFC can request a retry, a forced
    /// rejection is armed, or the magnetic-slip composite can fail its solve
    pub fn root_step_rejectable(&self) -> bool {
        self.levels.iter().any(|level| level.kernels.fofc_active()) || self.forced_fine_rejection_pending() || self.finest_has_magnetic_slip()
    }

    /// one root step at an already-selected dt: the shared step transaction over this
    /// hierarchy's attempt, then the accepted-step tail (tracer migration, census, clock and
    /// motion, body feedback). a step the retry guard or the timestep floor ends restores the
    /// step entry and records the halt in `crash`. callable with any admissible dt, which is
    /// what lets a multi-tile driver select a global minimum and drive each tile with it.
    pub fn step_root_with_dt(&mut self, candidate: f64) {
        let time = self.levels[0].state.time;
        let guard = self.levels[0].state.retry_guard;
        let mut attempt = SingleRootAttempt { hier: self, restorable: false, sidecars: None };
        let dt = match take_root_step(&mut attempt, &guard, candidate, time) {
            Ok::<_, std::convert::Infallible>(RootStep::Accepted { dt }) => dt,
            Ok(RootStep::Halted(halt)) => {
                self.record_step_halt(&halt);
                return;
            }
        };
        if self
            .levels
            .iter()
            .any(|level| level.state.continuous_tracers.is_some())
        {
            self.migrate_continuous_tracers_to_finest()
                .unwrap_or_else(|detail| panic!("continuous tracer refinement transfer: {detail}"));
        }

        // horizon excision runs once per step after the full RK combination, the same
        // point the single-grid loop applies it: the causally disconnected cells inside
        // the excision sphere take a zero-gradient fill. viscous / excise /
        // horizon-ledger / penalize run in the finest level's tail (level_step_tail),
        // in the uni-grid driver's exact order.

        let root = &mut self.levels[0];
        // homologous linear advance, taken when the run leaves the motion law untraced.
        advance_state_clock(&mut root.state, dt);
        // the census rows of the step carry the clock of the state they hold
        self.census_sample_root_step();
        self.root_body_step(dt);
    }

    /// the registered binned reductions, at the tail of the accepted step: the stage sequence has
    /// finished and its last stage ended in a conserved-to-primitive recovery, so the primitives
    /// every census reads belong to the state at this time. sampling mid-stage would bin a
    /// partially advanced state.
    ///
    /// every level contributes its leaf cells: a cell a finer level resolves is excluded here and
    /// counted there, so the refined volume enters the reduction exactly once at the finest
    /// resolution that covers it. counting a covered coarse cell as well would inflate every
    /// extensive total by the refined volume — a wrong number that looks entirely reasonable,
    /// since it is smooth, positive, and of the right order.
    pub fn census_sample_root_step(&mut self) {
        if self.levels[0].state.censuses.is_empty() {
            return;
        }
        let ops: Vec<_> = self.levels[0]
            .state
            .censuses
            .iter()
            .map(|r| r.evaluator.spec().op())
            .collect();
        // the registrations live on the root, which owns the history; every level is reduced
        // against that one list. a level carrying its own (empty) list would contribute
        // nothing while its volume stayed excluded from the parent.
        let registrations = &self.levels[0].state.store.censuses;
        let root_step = Some(symbi_sim::census::Cadence::RootStep);
        let time = self.levels[0].state.time;
        let mut totals = symbi_substrate::census_sample::level_partials(
            &self.levels[0].state,
            registrations,
            0,
            root_step,
            self.levels[0].state.composite_ownership.coverage.as_ref(),
            time,
        );
        for level in 1..self.levels.len() {
            let partial = symbi_substrate::census_sample::level_partials(
                &self.levels[level].state,
                registrations,
                0,
                root_step,
                self.levels[level]
                    .state
                    .composite_ownership
                    .coverage
                    .as_ref(),
                time,
            );
            symbi_substrate::census_sample::combine_partials(&mut totals, partial, &ops);
        }
        symbi_substrate::census_sample::record_samples(&mut self.levels[0].state, time, totals);
    }

    /// body feedback + motion at the tail of an accepted root step: the finest level owns the
    /// sink and the diagnostics; the (prescribed) motion advances there once at the root dt,
    /// then positions and velocities sync outward so every level's gravity sees the same bodies.
    pub fn root_body_step(&mut self, dt: f64) {
        if !self.levels[0].state.has_bodies() {
            return;
        }
        let fi = self.levels.len() - 1;
        let finest = &mut self.levels[fi];
        // the backward feedback reduction (force/torque/accreted-mass) is only
        // needed for bodies whose dynamics consume it — two-way-coupled or
        // accreting. a one-way fixed gravitational mass skips the entire pass.
        let needs_fb = finest
            .state
            .immersed
            .as_ref()
            .is_some_and(|im| im.bodies.needs_feedback());
        if needs_fb {
            // the backward reduction sweeps the full domain (~11 outputs per
            // body: force/torque/mass/energy) — a real per-step cost, instrumented
            // so the profile accounts for it explicitly.
            prof("body_feedback", || {
                finest.kernels.body_feedback(&finest.state, dt)
            });
        }
        finest.state.dt = dt;
        prof("body_motion", || evolve_bodies(&mut finest.state));

        let truth: Vec<_> = {
            let bodies = &self.levels[fi].state.immersed.as_ref().unwrap().bodies;
            (0..bodies.len())
                .map(|bb| (bodies.get(bb).position, bodies.get(bb).velocity))
                .collect()
        };
        for ll in 0..fi {
            if let Some(im) = self.levels[ll].state.immersed.as_mut() {
                for (bb, (pos, vel)) in truth.iter().enumerate() {
                    let body = im.bodies.get_mut(bb);
                    body.position = *pos;
                    body.velocity = *vel;
                }
            }
        }
        self.assert_sinks_inside_finest();
    }

    /// advance one level by dt, then subcycle the finer level, restrict, and
    /// reflux. `alpha0` is this level's substep start as a fraction of the
    /// parent's step; 0.0 on the root. the coarse-fine ghosts are
    /// stage-correct in time: the prolong feeding stage k reconstructs at the
    /// shu-osher stage time `alpha0 + c_k / RATIO` (see
    /// stage_time_fractions), which restores second-order temporal coupling
    /// at the interface — substep-start-frozen ghosts measurably collapse the
    /// boundary to first order. the stage loop mirrors sim/evolve.rs::step, and a
    /// 1-level hierarchy reproduces it bit-for-bit.
    fn advance_level(&mut self, level: usize, dt: f64, alpha0: f64) -> bool {
        self.level_step_begin(level, dt);
        phase_probe::sample("begin", level, &self.levels[level].state);
        let n = self.levels[level].state.timestepping.stages().len();
        for ii in 0..n {
            if self.level_stage(level, ii, dt, alpha0) {
                return true;
            }
            phase_probe::sample(phase_probe::STAGE[ii], level, &self.levels[level].state);
        }
        self.end_stage_motion(level);
        self.level_step_tail(level, dt, alpha0)
    }

    /// the palindromic coupled step for a slip-enabled draining body:
    ///   D(dt/2) M(dt/2) H(dt) M(dt/2) D(dt/2),
    /// where D is the material drain, M the implicit magnetic-slip midpoint, and H the ideal-MHD RK
    /// evolution (its stages and tail EMF, without the drain). the drain and slip each advance for a
    /// total duration dt; the magnetic solve never runs inside an RK stage. each M freezes its
    /// predictor from the state entering that half, and the rebuild points restore the primitives and
    /// magnetic/cell state each operator's successor reads. returns true on a step failure (a
    /// nonconverged magnetic solve or a stage retry request); increment 3 treats that as a hard
    /// failure, and the whole composite sits inside the caller's attempt boundary for later restore.
    /// narrow slice: one stationary slip draining body, single level, periodic 3D cartesian host,
    /// adiabatic MHD; unsupported combinations fail loudly before the first half-step.
    pub fn advance_slip_coupled_step(&mut self, level: usize, dt: f64, alpha0: f64) -> bool {
        assert_eq!(level, 0, "the coupled step is a root-step composition");
        assert!(
            self.levels.iter().all(|l| !l.state.has_tracers()),
            "magnetic-slip coupling does not support tracers yet"
        );
        let fi = self.levels.len() - 1;
        let refined = fi > 0;
        let half = 0.5 * dt;
        SLIP_FINE_SUBSTEPS.with(|c| c.set(0));

        // D(dt/2): the material drain on the finest level, then rebuild primitives/ghosts.
        slip_schedule_record("D", half);
        {
            let l = &self.levels[fi];
            l.kernels.penalize(&l.state, half);
        }
        self.slip_rebuild(fi);

        // M(dt/2): the implicit slip midpoint from the post-drain state on the finest level, then
        // rebuild magnetic/cell.
        slip_schedule_record("M", half);
        if !self.levels[fi].kernels.magnetic_slip_step(&self.levels[fi].state, half) {
            return true; // nonconverged solve -> hard failure this increment
        }
        self.slip_rebuild(fi);
        // the covered parent regions become the finest state: the split operators acted on the
        // finest level alone, with their whole support inside it, so no interface flux or edge
        // EMF is owed and the update is a pure transfer.
        if refined {
            self.sync_all_fine_to_coarse();
        }
        if slip_failure_fires(SlipFailPoint::AfterOpening) {
            return true;
        }

        // H(dt): the ideal-MHD step without the drain. on a single grid the RK stages and the tail
        // EMF, bit for bit the single-level schedule; on a refined hierarchy the complete level
        // recursion (stages, fine subcycling, stage-time prolongation, CT EMF accumulation,
        // restriction, flux and EMF reflux, level clocks) with every level tail's drain withheld.
        slip_schedule_record("H", dt);
        if refined {
            self.drain_suppressed = true;
            let failed = self.advance_level(0, dt, alpha0);
            self.drain_suppressed = false;
            if failed {
                return true;
            }
        } else {
            self.level_step_begin(level, dt);
            let n = self.levels[level].state.timestepping.stages().len();
            for ii in 0..n {
                if self.level_stage(level, ii, dt, alpha0) {
                    return true;
                }
            }
            self.level_tail_emf(level, dt);
            self.slip_rebuild(level);
        }

        // M(dt/2): a fresh predictor frozen from the accepted output of H, on the finest level.
        slip_schedule_record("M", half);
        if !self.levels[fi].kernels.magnetic_slip_step(&self.levels[fi].state, half) {
            return true;
        }
        self.slip_rebuild(fi);
        if slip_failure_fires(SlipFailPoint::ClosingSolve) {
            return true;
        }

        // D(dt/2): the closing drain on the finest level.
        slip_schedule_record("D", half);
        {
            let l = &self.levels[fi];
            l.kernels.penalize(&l.state, half);
        }
        self.slip_rebuild(fi);
        if refined {
            self.sync_all_fine_to_coarse();
        }
        false
    }

    /// the ideal-MHD step H(dt) the coupled step applies: the RK stages and the tail EMF (no drain, no
    /// magnetic slip), then the primitive/ghost rebuild its successor reads. exposed for the operator-
    /// split isolation study. returns true on a stage retry request.
    pub fn hydro_map(&mut self, level: usize, dt: f64) -> bool {
        self.level_step_begin(level, dt);
        let n = self.levels[level].state.timestepping.stages().len();
        for ii in 0..n {
            if self.level_stage(level, ii, dt, 0.0) {
                return true;
            }
        }
        self.level_tail_emf(level, dt);
        self.slip_rebuild(level);
        false
    }

    /// the complete production magnetic-slip map M(dt) as the coupled step applies it: the implicit
    /// midpoint solve and commit, then the primitive/ghost rebuild its successor reads. exposed for the
    /// M-map isolation study (per-field convergence and time reversibility). returns true on a
    /// nonconverged solve.
    pub fn magnetic_slip_map(&self, level: usize, dt: f64) -> bool {
        if !self.levels[level].kernels.magnetic_slip_step(&self.levels[level].state, dt) {
            return true;
        }
        self.slip_rebuild(level);
        false
    }

    /// restore the primitives and physical ghost bands an operator's successor reads: c2p recovers the
    /// primitive state from the conserved fields, ghost_fill refills the boundary bands.
    fn slip_rebuild(&self, level: usize) {
        let l = &self.levels[level];
        l.kernels.c2p(&l.state);
        self.fill_physical_ghosts(level);
    }

    /// one material-drain operation D(dt) at `level`: penalize followed by the primitive/ghost
    /// rebuild. exposed for the drain-semigroup pin.
    pub fn drain_and_rebuild(&self, level: usize, dt: f64) {
        self.levels[level].kernels.penalize(&self.levels[level].state, dt);
        self.slip_rebuild(level);
    }

    /// the interior conserved-density field at `level`, as a flat snapshot for state comparisons.
    pub fn density_snapshot(&self, level: usize) -> Vec<f64> {
        let l = &self.levels[level];
        l.state
            .geom
            .interior
            .iter()
            .map(|c| *l.state.fields.cons.den.at(c))
            .collect()
    }

    /// interior flat snapshots of the magnetic and thermodynamic state for the M-map isolation study:
    /// the staggered face field, the cell-centered field, the conserved total energy, and the recovered
    /// pressure. each walks the same interior order, so a per-field convergence ratio is well-defined.
    pub fn slip_state_snapshots(&self, level: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let l = &self.levels[level];
        let mhd = l.state.fields.mhd.as_ref().expect("mhd fields");
        let cells: Vec<_> = l.state.geom.interior.iter().collect();
        // the staggered field has one component per gridded axis; the cell field one per vector slot.
        let bface = (0..NDIM)
            .flat_map(|d| cells.iter().map(move |c| *mhd.bface[d].at(*c)))
            .collect();
        let bcell = (0..DOF)
            .flat_map(|d| cells.iter().map(move |c| *mhd.bcell[d].at(*c)))
            .collect();
        let nrg_field = l.state.fields.cons.nrg_field().expect("adiabatic energy");
        let nrg = cells.iter().map(|c| *nrg_field.at(*c)).collect();
        let pre_field = l.state.fields.prim.pre.as_ref().expect("pressure field");
        let pre = cells.iter().map(|c| *pre_field.at(*c)).collect();
        (bface, bcell, nrg, pre)
    }

    /// step prologue: snapshot this level's prims (for the finer level's time-interpolated ghost
    /// prolongation) + the rk u_n snapshot. `advance_level` calls begin / stage* / tail in order;
    /// splitting them out lets the decomposed root driver drive the root stages one at a time with
    /// a root halo exchange between stages (rk2-root requires the
    /// corrector to read each neighbor's stage-1 update, exactly like the single-level exchange).
    pub fn level_step_begin(&mut self, level: usize, dt: f64) {
        self.levels[level].state.motion_at_step_entry = self.levels[level].state.motion;
        let has_finer = level + 1 < self.levels.len();
        if has_finer {
            prof("refine_save_prim", || save_prim_old(&self.levels[level]));
        }
        self.levels[level].state.dt = dt;
        if self.levels[level].state.continuous_tracers.is_some() {
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            symbi_sim::tracers::begin_ito_transport_store(
                &mut self.levels[level].state.store,
                &geometry,
            )
            .unwrap_or_else(|detail| panic!("ito transport initialization: {detail}"));
        }
        let stages = self.levels[level].state.timestepping.stages();
        if needs_step_snapshot(stages) {
            let l = &self.levels[level];
            prof("snapshot", || l.kernels.snapshot(&l.state));
        }
        if self.levels[level].state.has_tracers() {
            symbi_sim::tracers::snapshot_transport_state(&mut self.levels[level].state);
            self.injection_ledger.clear();
            if has_finer {
                self.tracer_interface_ledgers[level].borrow_mut().clear();
            }
        }
    }

    /// one SSP stage `ii` of level `level` -- the body of `advance_level`'s stage loop. `alpha0` is
    /// this level's substep start as a fraction of the parent step (0 on the root). pure extraction.
    /// one stage of `level`, folded whole: the mark and the correction back to back, as a
    /// hierarchy that shares no cut runs them. returns true on a stage retry request.
    pub fn level_stage(&mut self, level: usize, ii: usize, dt: f64, alpha0: f64) -> bool {
        let mut level_stage = OneTileLevelStage { hier: self, level, ii, dt, alpha0, receipt_start: 0 };
        let Ok::<_, std::convert::Infallible>(agreed) = symbi_sim::collective_stage::collective_stage(&mut level_stage);
        agreed.retry
    }

    /// the stage's phases up to the conserved-to-primitive recovery and the troubled-cell mark.
    /// a decomposed level exchanges the marks across its cuts before
    /// `level_stage_from_correction`, so a clean tile learns of trouble against its cut and
    /// both tiles take the same corrected flux on the shared face. the receipt start is the
    /// tracer interface ledger's length as the stage opened, ahead of the transfers this
    /// stage's flux sampling appended, which the correction half's tail applies.
    pub fn level_stage_through_mark(&mut self, level: usize, ii: usize, dt: f64, alpha0: f64) -> StageMark {
        self.bind_stage_motion(level, ii, dt);
        let receipt_start = (level > 0)
            .then(|| self.tracer_interface_ledgers[level - 1].borrow().len())
            .unwrap_or(0);
        let args = self.stage_args(level, ii, dt);
        let mut hook = self.stage_hook(level, ii, dt, alpha0);
        let l = &self.levels[level];
        let troubled = symbi_sim::stage::fold_stage_through_mark(&l.state, &l.kernels, args, &mut hook);
        StageMark { troubled, receipt_start }
    }

    /// the stage's correction and every phase after it, `cut_trouble` saying whether a
    /// neighbor's mark reached this level's cut ghosts, then the tracer transport the stage
    /// carries from `receipt_start` on. returns true on a stage retry request.
    pub fn level_stage_from_correction(
        &mut self,
        level: usize,
        ii: usize,
        dt: f64,
        alpha0: f64,
        cut_trouble: bool,
        receipt_start: usize,
    ) -> bool {
        let has_coarser = level > 0;
        let stages = self.levels[level].state.timestepping.stages();
        let (a0, ac) = stages[ii];
        let args = self.stage_args(level, ii, dt);
        let outcome = {
            let mut hook = self.stage_hook(level, ii, dt, alpha0);
            let l = &self.levels[level];
            symbi_sim::stage::fold_stage_from_correction(&l.state, &l.kernels, args, cut_trouble, &mut hook)
        };
        if outcome == symbi_sim::stage::StageOutcome::RetryStep {
            return true;
        }
        {
            let l = &self.levels[level];
            symbi_sim::stage::fold_stage_excision(&[&l.state], &[&l.kernels], &mut symbi_sim::stage::SingleGridStep)
                .unwrap_or_else(|e| panic!("level {level}: {e}"));
        }
        if level == 1 {
            let substep = (alpha0 * RATIO as f64).round() as u8;
            let here = (self.levels[0].state.iteration, substep, ii);
            if self.forced_fine_rejection == Some(here) {
                self.forced_fine_rejection = None;
                return true;
            }
        }
        if let Some((step, at_level, substep, stage)) = self.forced_level_rejection {
            // the level's substeps since the root step began: its clock counts every substep it took
            let root = self.levels[0].state.iteration;
            let taken = self.levels[level].state.iteration.wrapping_sub(root * (RATIO as u64).pow(level as u32));
            if (step, at_level, substep, stage) == (root, level, taken, ii) {
                self.forced_level_rejection = None;
                return true;
            }
        }
        if self.levels[level].state.continuous_tracers.is_some() {
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            symbi_sim::tracers::accumulate_ito_transport_stage_store(
                &mut self.levels[level].state.store,
                &geometry,
                ac,
            )
            .unwrap_or_else(|detail| panic!("ito transport accumulation: {detail}"));
        }
        if self.levels[level].state.has_tracers() {
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let layout = self.tracer_layout(level);
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            let mut injections = symbi_sim::tracers::boundary_injection_transfers_store(
                &self.levels[level].state,
                &geometry,
                layout,
            );
            injections.extend(symbi_sim::tracers::source_injection_transfers_store(
                &self.levels[level].state,
                &geometry,
                layout,
                a0,
                ac,
            ));
            injections.retain(|transfer| self.tracer_owner_is_active(transfer.destination));
            symbi_sim::tracers::fold_injection_ledger(&mut self.injection_ledger, injections, ac);
            let across_tiles = self.blend_across_tiles;
            prof("tracers", || {
                let coverage = self.levels[level].coverage.clone();
                let advance = if across_tiles {
                    symbi_sim::tracers::advance_stage_mass_transport_tile_masked
                } else {
                    symbi_sim::tracers::advance_stage_mass_transport_store_masked
                };
                advance(&mut self.levels[level].state, &geometry, layout, coverage.as_ref(), a0, ac, ii)
                    .unwrap_or_else(|detail| panic!("tracer transport: {detail}"))
            });
        }
        if has_coarser && self.levels[level].state.has_tracers() {
            if self.blend_across_tiles {
                self.pending_interface = Some((level, receipt_start, ii));
            } else {
                self.apply_interface_receipts(level, receipt_start, ii);
            }
        }
        false
    }

    /// the coarse-fine interface event of `level`'s stage `ii`: the receipts the stage booked from
    /// `receipt_start` on, sampled over the records their source cells own
    fn apply_interface_receipts(&mut self, level: usize, receipt_start: usize, ii: usize) {
        let transfers = {
            let ledger = self.tracer_interface_ledgers[level - 1].borrow();
            ledger[receipt_start..].to_vec()
        };
        let epoch = self.levels[level]
            .state
            .iteration
            .wrapping_mul(64)
            .wrapping_add((level as u64) << 4)
            .wrapping_add(ii as u64);
        self.apply_interface_event(&transfers, epoch)
            .unwrap_or_else(|detail| panic!("tracer interface transport: {detail}"));
    }

    /// mark this hierarchy a tile of a decomposed run, whose level stages leave the tracer blend to
    /// the driver
    pub(crate) fn set_blend_across_tiles(&mut self) {
        self.blend_across_tiles = true;
    }

    /// the boundary and source injection this tile's level stages booked, taken for a merged spawn
    pub(crate) fn take_injection_ledger(&mut self) -> std::collections::BTreeMap<symbi_sim::mass_transport::ContainerId, f64> {
        std::mem::take(&mut self.injection_ledger)
    }

    /// the held boundary injection of `level`'s continuous tracers, taken for a merged spawn
    pub(crate) fn take_continuous_injection(&mut self, level: usize) -> std::collections::BTreeMap<symbi_sim::mass_transport::ContainerId, f64> {
        self.continuous_injection.remove(&level).unwrap_or_default()
    }

    /// the whole domain's extent and the boundary each axis presents to this tile's continuous
    /// tracers, where the tile's own faces include cuts
    pub fn set_continuous_domain(&mut self, bounds: [(f64, f64); NDIM], boundaries: Boundaries<NDIM>) {
        self.continuous_domain = Some((bounds, boundaries));
    }

    /// the cell counts of the whole root lattice the tracer containers address
    pub(crate) fn tracer_root_cells(&self) -> [usize; NDIM] {
        self.tracer_root_global_cells
    }

    /// apply the interface event a decomposed stage of `level` left pending, once the driver's
    /// blend ran; an event another level left (a stage its attempt rejected) waits for that level's
    /// next stage, which replaces it
    pub fn apply_pending_interface(&mut self, level: usize) {
        if let Some((pending, receipt_start, ii)) = self.pending_interface {
            if pending == level {
                self.pending_interface = None;
                self.apply_interface_receipts(level, receipt_start, ii);
            }
        }
    }

    /// the fold's arguments for stage `ii` of `level`
    /// bind the level's mesh motion to stage `ii`'s entry time: the traced law's a(t), or the
    /// linear advance from the step entry. a static mesh is left as it is.
    fn bind_stage_motion(&mut self, level: usize, ii: usize, dt: f64) {
        let state = &mut self.levels[level].state;
        let entry = symbi_sim::driver::stage_schedule(state.timestepping.stages())[ii].entry;
        let t_entry = state.time + entry * dt;
        let law_value = state.motion_law.as_ref().map(|law| (law.a_at(t_entry), law.adot_at(t_entry)));
        let a_n = state.motion_at_step_entry.a;
        symbi_sim::driver::set_stage_motion(&mut state.motion, law_value, dt, a_n, entry);
    }

    /// the stages moved the scale factor; the step tail starts from the step-entry value, as on
    /// every other driver
    fn end_stage_motion(&mut self, level: usize) {
        let state = &mut self.levels[level].state;
        state.motion.a = state.motion_at_step_entry.a;
    }

    fn stage_args(&self, level: usize, ii: usize, dt: f64) -> StageArgs {
        let stages = self.levels[level].state.timestepping.stages();
        let (a0, ac) = stages[ii];
        StageArgs {
            dt,
            a0,
            ac,
            stage: ii,
            n_stages: stages.len(),
            injection_weight: symbi_sim::driver::downstream_injection_weight(stages, ii),
            allow_elision: true,
            exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, ii),
        }
    }

    /// the phase sequence comes from the shared table (symbi-sim::stage); this driver
    /// contributes two structural interleaves through the hook points: flux-register sampling
    /// (on the high-order fluxes, before a fofc splice) and the coarse-fine ghost
    /// re-prolongation. everything the hooks touch is reached by shared reference (the
    /// registers and field writes go through interior mutability), so the fold's borrow of the
    /// level state stays compatible.
    fn stage_hook(&self, level: usize, ii: usize, dt: f64, alpha0: f64) -> impl FnMut(HookPoint) + '_ {
        let has_coarser = level > 0;
        let stages = self.levels[level].state.timestepping.stages();
        let weights = flux_weights(stages);
        let stage_time = stage_time_fractions(stages);
        let (_a0, ac) = stages[ii];
        let this = self;
        let l = &this.levels[level];
        move |hp: HookPoint| match hp {
            HookPoint::AfterFlux => {
                prof("refine_flux_reg", || {
                    // when the run declares a stationary target, that same accumulation runs a
                    // second time on that target's flux with the weight negated, so the register
                    // holds the coarse-fine mismatch of `F(Q) - F(qt)` rather than of `F(Q)`. a
                    // state sitting on the target then accumulates zero, while the two grids'
                    // reconstructions of the target itself — which differ, and which the plain
                    // register would apply to the coarse cells as a force — cancel.
                    if this.has_flux_register(level) {
                        let reg = &this.flux_registers[level];
                        if uniform_cartesian(&l.state) {
                            if ii == 0 {
                                reg.zero_uniform();
                            }
                            for dd in 0..NDIM {
                                reg.accumulate_coarse_uniform(
                                    &l.state.fields.flux,
                                    &l.state.geom.dx,
                                    dd,
                                    weights[ii] * dt,
                                );
                                if let Some(m) = l.state.fields.mhd.as_ref() {
                                    if DOF > NDIM {
                                        reg.accumulate_coarse_uniform_oop(&m.bflux, &l.state.geom.dx, dd, weights[ii] * dt);
                                    }
                                }
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_coarse_uniform(
                                        feq,
                                        &l.state.geom.dx,
                                        dd,
                                        -weights[ii] * dt,
                                    );
                                }
                            }
                        } else {
                            if ii == 0 {
                                reg.zero();
                            }
                            let geo = l.state.geom.block_geometry(l.state.physics.metric);
                            for dd in 0..NDIM {
                                reg.accumulate_coarse(
                                    &l.state.fields.flux,
                                    &geo,
                                    dd,
                                    weights[ii] * dt,
                                );
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_coarse(feq, &geo, dd, -weights[ii] * dt);
                                }
                            }
                        }
                    }
                    if has_coarser {
                        let reg = &this.flux_registers[level - 1];
                        if uniform_cartesian(&l.state) {
                            for dd in 0..NDIM {
                                reg.accumulate_fine_uniform(
                                    &l.state.fields.flux,
                                    &l.state.geom.dx,
                                    dd,
                                    weights[ii] * dt,
                                );
                                if let Some(m) = l.state.fields.mhd.as_ref() {
                                    if DOF > NDIM {
                                        reg.accumulate_fine_uniform_oop(&m.bflux, &l.state.geom.dx, dd, weights[ii] * dt);
                                    }
                                }
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_fine_uniform(
                                        feq,
                                        &l.state.geom.dx,
                                        dd,
                                        -weights[ii] * dt,
                                    );
                                }
                            }
                        } else {
                            let geo = l.state.geom.block_geometry(l.state.physics.metric);
                            for dd in 0..NDIM {
                                reg.accumulate_fine(
                                    &l.state.fields.flux,
                                    &geo,
                                    dd,
                                    weights[ii] * dt,
                                    RATIO,
                                );
                                if let Some(feq) = l.flux_eq.as_ref() {
                                    reg.accumulate_fine(feq, &geo, dd, -weights[ii] * dt, RATIO);
                                }
                            }
                        }
                        if l.state.has_tracers() {
                            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                            let geometry = l.state.geom.block_geometry(l.state.physics.metric);
                            let transfers = interface_mass_transfers(
                                &this.tracer_interfaces[level - 1],
                                &l.state.fields.flux,
                                &geometry,
                                weights[ii] * dt,
                            );
                            this.tracer_interface_ledgers[level - 1]
                                .borrow_mut()
                                .extend(transfers);
                        }
                    }
                });
            }
            HookPoint::BeforeC2p => {
                // only the covered shell next to a coarse-fine boundary participates in the
                // composite update: it supplies stage-time prolongation data and the coarse face
                // flux sampled by the register. the deep covered core is replaced by restriction
                // after the fine subcycle, so evolving it is both unnecessary and, for a compact
                // source resolved only on the fine grids, mathematically undefined. in particular,
                // a point-sampled softened force can drive the central cell of an under-resolved
                // covered level outside the admissible set before restriction gets a chance to
                // overwrite it.
                //
                // preserve one cell beyond both stencils at the seam. restoring the remainder
                // from this stage's input makes that inactive volume an identity update while
                // leaving every coarse value observable by reconstruction or prolongation alone.
                if let Some(inactive) = l.state.composite_ownership.inactive.as_ref() {
                    restore_cons_region(l.state.stage_input(), &l.state.fields.cons, inactive);
                    // CT fields are part of the state, not diagnostics.  A covered-core identity
                    // update must therefore restore both representations of B as well as U.
                    // `*_old` is the level-step entry field; after every earlier stage the same
                    // core was restored to it, so it is also this stage's core input.
                    if let (Some(mhd), Some(bcell_old)) =
                        (l.state.fields.mhd.as_ref(), l.bcell_old.as_ref())
                    {
                        for comp in 0..DOF {
                            copy_field_region(&bcell_old[comp], &mhd.bcell[comp], inactive);
                        }
                    }
                }
                // the core's staggered faces, restored on every tile holding a copy of them:
                // a tile without a core cell still holds the face on a cut bounding the
                // neighbor's core
                if let (Some(mhd), Some(bface_old)) =
                    (l.state.fields.mhd.as_ref(), l.bface_old.as_ref())
                {
                    for dir in 0..NDIM {
                        if let Some(face_core) = l.state.composite_ownership.inactive_faces[dir].as_ref() {
                            copy_field_region(&bface_old[dir], &mhd.bface[dir], face_core);
                        }
                    }
                }
                // the ssp recombination leaves `a0*u_n + ac*(qt - dt R) = qt - ac dt R` when the
                // stage input is the target, so adding `ac dt R` back returns the target exactly.
                // the weights are the stage's own, so this holds at every stage of every scheme.
                if let Some(res) = l.residual_eq.as_ref() {
                    prof("refine_equilibrium", || {
                        equilibrium_accumulate(
                            res,
                            &l.state.fields.cons,
                            &l.state.geom.interior,
                            ac * dt,
                        );
                    });
                }
            }
            HookPoint::BeforeGhostFill => {
                // c2p over the full allocated domain recomputed the coarse-fine
                // prim ghosts from stale cons; re-prolong them at the time of
                // the state entering the next stage before the physical fill
                // reads corners.
                if has_coarser {
                    this.prolong_cf(level, alpha0 + stage_time[ii] / RATIO as f64);
                }
            }
            HookPoint::AfterGhostFill => {
                if has_coarser {
                    symbi_substrate::regimes::mhd_substrate::wrap_periodic_corner_ghosts(&l.state);
                }
            }
        }
    }


    /// step epilogue: the tail ahead of the finer subcycle, the recursive subcycle of the finer
    /// level with its restriction and reflux into this one, and the tail after it.
    pub fn level_step_tail(&mut self, level: usize, dt: f64, alpha0: f64) -> bool {
        self.level_tail_pre(level, dt);
        if level + 1 < self.levels.len() {
            if self.level_subcycle(level, dt) {
                return true;
            }
            self.level_restrict_reflux(level, alpha0);
            phase_probe::sample("restrict", level, &self.levels[level].state);
        }
        self.level_tail_post(level, dt);
        false
    }

    /// the number of levels of the run this hierarchy belongs to: its own, or the whole run's
    /// when it is one tile of a decomposed run.
    pub fn run_levels(&self) -> usize {
        self.run_levels.unwrap_or(self.levels.len())
    }

    /// mark this tile as one of a decomposed run of `levels` levels.
    pub fn set_run_levels(&mut self, levels: usize) {
        assert!(levels >= self.levels.len(), "a tile holds {} levels of a {levels}-level run", self.levels.len());
        self.run_levels = Some(levels);
    }

    /// the step tail ahead of the finer subcycle: boundary tracer injection, emf bookkeeping, the
    /// post-step (viscous transport on the run's finest level), excision, the horizon ledger and
    /// the surface stack on the run's finest level, rigid walls on every level, and the Ito
    /// coefficients of the continuous tracers.
    pub fn level_tail_pre(&mut self, level: usize, dt: f64) {
        let has_finer = level + 1 < self.run_levels();
        if self.levels[level].state.continuous_tracers.is_some() {
            self.hold_continuous_injection(level);
        } else if self.levels[level].state.has_tracers() {
            let ledger = std::mem::take(&mut self.injection_ledger);
            let layout = self.tracer_layout(level);
            let geometry = self.levels[level]
                .state
                .geom
                .block_geometry(self.levels[level].state.physics.metric);
            symbi_sim::tracers::spawn_boundary_injection_store(
                &mut self.levels[level].state.store,
                &geometry,
                layout,
                ledger,
            )
            .unwrap_or_else(|detail| panic!("tracer boundary injection: {detail}"));
            self.sync_tracer_spawn_state(level);
        }
        self.level_tail_emf(level, dt);
        // the IBM surface physics on the finest level, once per its substep,
        // after the full RK combination (receipt == removal) and ahead of the
        // parent's restriction (so the coarse covered cells sync to the
        // drained state — restriction consistency).
        // per-step operator order matches the uni-grid driver exactly:
        // viscous -> excise -> horizon ledger -> penalize. transport first, then the
        // causally-disconnected fill, then the shell-flux booking (the flux fields still hold the
        // last stage's flux), then the IBM surface physics whose receipt must equal the removal.
        // the post-step recipe on this level: the viscous transport on the finest levels
        // alone, excision on every level
        {
            let l = &self.levels[level];
            symbi_sim::stage::fold_post_step(
                &[&l.state],
                &[&l.kernels],
                dt,
                &mut symbi_sim::stage::SingleGridStep,
                |op| match op {
                    symbi_sim::stage::StepOp::Viscous => !has_finer,
                    symbi_sim::stage::StepOp::Excise => false,
                },
            )
            .unwrap_or_else(|e| panic!("level {level}: {e}"));
        }
        // excise runs on every level: the excised (causally-disconnected) region is owned by
        // whichever level contains it, and the refinement request gate keeps every finer patch clear
        // of it, so a refined root still owns — and still excises — its singular core. gating this
        // on `!has_finer` silently skipped excision wherever the excised region sat under a refined
        // level (the root of an off-core refined run), leaving the core evolving. ordered after
        // viscous (bit-identical on the finest, where both run); a no-op at excision_radius = 0, so
        // unexcised and uni-grid runs are unchanged. the excised fill survives the finer subcycle +
        // restriction below because the excised region stays outside every covered (finer-patch)
        // region.
        self.level_tail_excise(level);
        if !has_finer {
            if let Some((index, diagnostic_radius)) = horizon_request(&self.levels[level].state) {
                let l = &self.levels[level];
                let (mdot, edot) = prof("horizon_accretion", || {
                    l.kernels.horizon_accretion(&l.state, diagnostic_radius)
                });
                book_horizon_receipt(&mut self.levels[level].state, index, mdot, edot, dt);
            }
        }
        // rigid surfaces act on every level: a wall may cross a coarse-fine boundary, and its
        // uncovered segment belongs to the coarse level. covered coarse writes are replaced by
        // restriction after the fine subcycle. accretion remains finest-only because coarse
        // proxies have their sink capability removed.
        let l = &self.levels[level];
        let has_rigid = l
            .state
            .immersed
            .as_ref()
            .is_some_and(|immersed| immersed.bodies.rigid_count() > 0);
        if l.state.has_bodies() && (!has_finer || has_rigid) && !self.drain_suppressed {
            let accretion_density = if l.state.has_tracers() {
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                Some(symbi_sim::tracers::snapshot_accretion_density(&l.state))
            } else {
                None
            };
            prof("penalize", || l.kernels.penalize(&l.state, dt));
            // a surface stack rescales the conserved state after the last stage's recovery, so
            // the stored primitives are recovered again from the penalized state: the next step's
            // reconstruction and a checkpoint at the step boundary read the drained gas.
            if l.state.immersed.as_ref().is_some_and(|im| im.bodies.penalizes()) {
                prof("c2p", || l.kernels.c2p(&l.state));
                prof("ghost_fill", || self.fill_physical_ghosts(level));
            }
            if symbi_sim::state::positivity_trace_enabled() {
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                l.state.positivity_trace(&format!("level={level} phase=penalize"));
            }
            if let Some(density_before) = accretion_density.as_deref() {
                symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
                let layout = self.tracer_layout(level);
                let geometry = self.levels[level]
                    .state
                    .geom
                    .block_geometry(self.levels[level].state.physics.metric);
                let crossing_time = self.levels[level].state.time + dt;
                symbi_sim::tracers::advance_accretion_transport_store(
                    &mut self.levels[level].state.store,
                    &geometry,
                    layout,
                    density_before,
                    crossing_time,
                )
                .unwrap_or_else(|detail| panic!("tracer accretion transport: {detail}"));
                if self.levels[level].state.continuous_tracers.is_some() {
                    self.continuous_accretion.insert(level, density_before.to_vec());
                }
            }
        }
        if self.levels[level].state.continuous_tracers.is_some() {
            self.prepare_level_ito_coefficients(level);
        }
    }

    /// the step tail after the finer subcycle and its restriction: continuous tracer transport,
    /// the per-level censuses and the level clock.
    pub fn level_tail_post(&mut self, level: usize, dt: f64) {
        if self.levels[level].state.continuous_tracers.is_some() {
            self.advance_level_continuous_tracers(level, dt);
            self.spawn_level_continuous_injection(level);
        }
        self.finish_level_tail_post(level, dt);
    }

    /// hold the boundary injection the level's stages booked until its continuous tracers have
    /// advanced, so the injected tracers enter at the step's end
    pub(crate) fn hold_continuous_injection(&mut self, level: usize) {
        let ledger = std::mem::take(&mut self.injection_ledger);
        self.continuous_injection.insert(level, ledger);
    }

    /// the step tail after the continuous tracers advanced and spawned: their crossing into the
    /// sinks, the per-level censuses and the level clock
    pub(crate) fn finish_level_tail_post(&mut self, level: usize, dt: f64) {
        if self.levels[level].state.continuous_tracers.is_some() {
            if let Some(density_before) = self.continuous_accretion.remove(&level) {
                let layout = self.tracer_layout(level);
                let geometry = self.levels[level]
                    .state
                    .geom
                    .block_geometry(self.levels[level].state.physics.metric);
                let crossing_time = self.levels[level].state.time + dt;
                symbi_sim::tracers::advance_continuous_accretion_transport_store(
                    &mut self.levels[level].state.store,
                    &geometry,
                    layout,
                    &density_before,
                    crossing_time,
                )
                .unwrap_or_else(|detail| panic!("continuous tracer accretion transport: {detail}"));
            }
        }
        // the level's rows carry the clock of the state they hold, the end of this substep; the
        // root's own clock advances at the root step's tail, by the same sum
        let end = advance_clock(self.levels[level].state.time, 0, dt).0;
        self.level_clock(level, dt);
        self.sample_per_level_censuses(level, end);
        if level + 1 == self.run_levels() {
            phase_probe::sample("tail", level, &self.levels[level].state);
        }
    }

    /// record a per-level census sample from this level's own accepted subcycle step, stamped
    /// `time`, the end of that substep.
    ///
    /// levels are time-aligned only at root-step boundaries, so this row is this level's alone: it
    /// covers this level's leaf cells at this level's clock, and carries the level index so a
    /// consumer can tell it from a root row or from another level's. summing rows across levels is
    /// therefore left to the consumer.
    ///
    /// the registrations live on the root, which owns the history; a finer level defers to it.
    fn sample_per_level_censuses(&mut self, level: usize, time: f64) {
        if self.levels[0].state.store.censuses.is_empty() {
            return;
        }
        let partial = symbi_substrate::census_sample::level_partials(
            &self.levels[level].state,
            &self.levels[0].state.store.censuses,
            level,
            Some(symbi_sim::census::Cadence::PerLevelStep),
            self.levels[level]
                .state
                .composite_ownership
                .coverage
                .as_ref(),
            time,
        );
        if partial.iter().all(|(values, _)| values.is_empty()) {
            return;
        }
        symbi_substrate::census_sample::record_samples_at_level(
            &mut self.levels[0].state,
            time,
            level,
            partial,
        );
    }

    /// derive one level's accepted-step coefficients before its finer level
    /// subcycles so fine coarse/fine ghosts can sample the parent field.
    pub(crate) fn prepare_level_ito_coefficients(&mut self, level: usize) {
        let geometry = self.levels[level]
            .state
            .geom
            .block_geometry(self.levels[level].state.physics.metric);
        symbi_sim::tracers::materialize_ito_coefficients_store(
            &mut self.levels[level].state.store,
            &geometry,
        )
        .unwrap_or_else(|detail| panic!("ito coefficient materialization: {detail}"));
        symbi_sim::tracers::fill_ito_coefficient_boundaries_host(
            self.levels[level]
                .state
                .ito_coefficients
                .as_ref()
                .expect("ito coefficients were materialized"),
            &self.levels[level].state.geom,
            self.levels[level].state.boundaries,
        )
        .unwrap_or_else(|detail| panic!("ito coefficient boundaries: {detail}"));
        if level > 0 {
            self.prolong_ito_cf(level);
        }
    }

    /// prolong accepted parent coefficients into a fine level's coarse/fine
    /// ghost slabs. the coefficients are fixed over the enclosing parent step.
    fn prolong_ito_cf(&self, level: usize) {
        let parent = &self.levels[level - 1];
        let fine = &self.levels[level];
        let parent_coefficients = parent
            .state
            .ito_coefficients
            .as_ref()
            .expect("the parent coefficients precede fine subcycling");
        let fine_coefficients = fine
            .state
            .ito_coefficients
            .as_ref()
            .expect("fine coefficients were materialized");
        let zero = Field::<f64, NDIM, Mem>::zeros(&parent.state.geom.allocated)
            .expect("ito coefficient prolongation scratch allocation failed");
        for slab in cf_ghost_slabs(
            &fine.state.geom.allocated,
            &fine.state.geom.interior,
            &fine.state.boundaries,
        )
        .iter()
        .filter_map(|slab| {
            clip_to_prolongable(slab, &parent.state.geom.allocated, self.prolong_order)
        }) {
            for dd in 0..NDIM {
                prolong_field(
                    &parent_coefficients.drift[dd],
                    &zero,
                    &fine_coefficients.drift[dd],
                    &slab,
                    self.prolong_order,
                    0.0,
                );
                prolong_field(
                    &parent_coefficients.variance[dd],
                    &zero,
                    &fine_coefficients.variance[dd],
                    &slab,
                    self.prolong_order,
                    0.0,
                );
                prolong_field(
                    &parent_coefficients.third[dd],
                    &zero,
                    &fine_coefficients.third[dd],
                    &slab,
                    self.prolong_order,
                    0.0,
                );
            }
        }
    }

    /// advance particles owned by one level over exactly that level's accepted
    /// interval. hierarchy ownership remains frozen until the root step ends.
    /// the level's held boundary injection as continuous tracers placed on the moved mesh, then
    /// the spawn bookkeeping (next id, remainder) carried to every level's population
    fn spawn_level_continuous_injection(&mut self, level: usize) {
        let ledger = self.continuous_injection.remove(&level).unwrap_or_default();
        let layout = self.tracer_layout(level);
        let geometry = self.levels[level]
            .state
            .geom
            .block_geometry(self.levels[level].state.physics.metric);
        symbi_sim::tracers::spawn_continuous_boundary_injection_store(
            &mut self.levels[level].state.store,
            &geometry,
            layout,
            ledger,
        )
        .unwrap_or_else(|detail| panic!("continuous tracer boundary injection: {detail}"));
        let source = self.levels[level]
            .state
            .continuous_tracers
            .as_ref()
            .expect("continuous tracers remain attached through the level step");
        let (next_id, injection_remainder) = (source.next_id, source.injection_remainder);
        for data in &mut self.levels {
            if let Some(tracers) = data.state.continuous_tracers.as_mut() {
                tracers.next_id = next_id;
                tracers.injection_remainder = injection_remainder;
            }
        }
    }

    pub(crate) fn advance_level_continuous_tracers(&mut self, level: usize, dt: f64) {
        let mut tracers = self.levels[level]
            .state
            .continuous_tracers
            .take()
            .expect("continuous tracers remain attached through the level step");
        let (scale_start, scale_end, offset_start, offset_end) =
            symbi_sim::tracers::continuous_tracer_mesh_step(&self.levels[level].state.store, dt);
        symbi_sim::tracers::advance_continuous_tracers(
            &mut tracers,
            self.levels[level]
                .state
                .ito_coefficients
                .as_ref()
                .expect("ito coefficients were materialized"),
            &self.levels[level].state.geom,
            scale_start,
            scale_end,
            offset_start,
            offset_end,
            dt,
        )
        .unwrap_or_else(|detail| panic!("ito tracer advancement: {detail}"));
        let (bounds, boundaries) = self.continuous_domain.unwrap_or_else(|| {
            (symbi_sim::tracers::partition_physical_bounds(&self.levels[0].state.geom), self.levels[0].state.boundaries)
        });
        let root_bounds = symbi_sim::tracers::map_continuous_tracer_bounds(bounds, scale_end, offset_end);
        symbi_sim::tracers::apply_continuous_boundaries_host(&mut tracers, root_bounds, boundaries)
        .unwrap_or_else(|detail| panic!("ito tracer boundaries: {detail}"));
        self.levels[level].state.continuous_tracers = Some(tracers);
    }

    /// emf register bookkeeping (mhd) after the stage loop: the efield buffers hold the effective
    /// per-step EMF (post_godunov wrote the rk2 time-average in place before the single curl; euler
    /// keeps the raw stage EMF) -- sample it into the coarse-side register (a finer level exists)
    /// and the fine-side register (a coarser level exists). a no-op for hydro. a pure extraction
    /// from level_step_tail so the decomposed driver can interpose a fine-level halo exchange
    /// between the root stages and the (driver-controlled) fine subcycle.
    /// the excise pass for one level, as a pure extraction from `level_step_tail` — the same reason
    /// `level_tail_emf` is extracted: the decomposed driver builds the step itself and runs this
    /// piece of the tail on its own, apart from the finest-level pieces (viscous, horizon ledger,
    /// penalize) that belong to the finest level alone.
    ///
    /// it runs on every level: the excised region is owned by whichever level contains it, and the
    /// refinement request gate keeps every finer patch clear of it, so a refined root still owns
    /// and still excises its singular core. a decomposed driver that omitted this would evolve
    /// un-excised gas inside the horizon forever — silent, since the interior is causally
    /// disconnected and reports nothing.
    ///
    /// the halo survives it: the fill writes cells inside the excised surface, and the refinement
    /// gate keeps every fine patch off that surface, so the prolongation that follows reads
    /// live cells only.
    pub fn level_tail_excise(&self, level: usize) {
        let l = &self.levels[level];
        symbi_sim::stage::fold_post_step(
            &[&l.state],
            &[&l.kernels],
            0.0,
            &mut symbi_sim::stage::SingleGridStep,
            |op| op == symbi_sim::stage::StepOp::Excise,
        )
        .unwrap_or_else(|e| panic!("level {level}: {e}"));
    }

    pub fn level_tail_emf(&self, level: usize, dt: f64) {
        let has_coarser = level > 0;
        let is_mhd = self.levels[level].state.fields.mhd.is_some();
        // the register of this level's interface with a finer one, held here or beside a
        // neighbor's patch: zeroed and sampled once per level step
        if let Some(Some(reg)) = self.emf_registers.get(level) {
            prof("refine_emf", || {
                reg.zero();
                let l = &self.levels[level];
                reg.accumulate_coarse(&l.state.fields.mhd.as_ref().unwrap().efield, dt);
            });
        }
        if has_coarser && is_mhd {
            prof("refine_emf", || {
                let reg = self.emf_registers[level - 1]
                    .as_ref()
                    .expect("mhd pair carries an emf register");
                let l = &self.levels[level];
                reg.accumulate_fine(&l.state.fields.mhd.as_ref().unwrap().efield, dt);
            });
        }
    }

    /// the monolithic finer-level subcycle: ratio substeps of level `level+1`, each prolonged from
    /// this level's time-interpolated prims. the decomposed driver replicates this loop but drives
    /// each fine substep's stages with a fine-level halo exchange between them (the fine patch may
    /// span a tile cut), so this method is the single-tile reference. pure extraction.
    fn level_subcycle(&mut self, level: usize, dt: f64) -> bool {
        let fine_dt = dt / RATIO as f64;
        let child_is_finest = level + 2 == self.levels.len();
        for sub in 0..RATIO {
            let alpha = sub as f64 / RATIO as f64;
            self.prolong_cf(level + 1, alpha);
            let f = &self.levels[level + 1];
            prof("ghost_fill", || self.fill_physical_ghosts(level + 1));
            phase_probe::sample("prolong", level + 1, &f.state);
            if self.advance_level(level + 1, fine_dt, alpha) {
                return true;
            }
            if child_is_finest {
                let ordinal = SLIP_FINE_SUBSTEPS.with(|c| {
                    c.set(c.get() + 1);
                    c.get()
                });
                if slip_failure_fires(SlipFailPoint::FineSubstep(ordinal)) {
                    return true;
                }
            }
        }
        false
    }

    /// whether the finest level carries a magnetic-slip body. the finest level is the sole owner
    /// of every sink coupling on a refined hierarchy (coarser levels hold gravity-only proxies), so
    /// the coupled-step decision reads the owner; on a single grid the finest is the root.
    pub fn finest_has_magnetic_slip(&self) -> bool {
        let fi = self.levels.len() - 1;
        self.levels[fi].kernels.has_magnetic_slip(&self.levels[fi].state)
    }

    /// the transfer-only synchronization of `level + 1` into `level`: the covered coarse cells
    /// become the conservative restriction of their fine children, the covered coarse faces the
    /// area-weighted average of the coincident fine faces (interface faces included), the cell
    /// field over the coverage and its one-cell shell the interpolation of those faces, then the
    /// coarse primitives and ghosts are rebuilt. no flux or EMF reflux is applied and no register is
    /// touched: this is the redundancy update after a split operator that acted on the finest level
    /// alone with its whole support inside the finest patch, where the coarse-fine interface saw
    /// no flux and no edge EMF. the restricted energy carries the fine magnetic energy already, and
    /// the face-to-cell interpolation leaves it alone.
    pub fn sync_fine_to_coarse(&self, level: usize) {
        let (lo, hi) = self.levels.split_at(level + 1);
        let coarse = &lo[level];
        let fine = &hi[0];
        let cov = coarse.coverage.as_ref().expect("a level with a finer child carries its coverage");
        restrict_cons(&fine.state.fields.cons, &coarse.state.fields.cons, cov);
        if let (Some(cmhd), Some(fmhd)) = (coarse.state.fields.mhd.as_ref(), fine.state.fields.mhd.as_ref()) {
            for aa in 0..DOF {
                restrict_cell_field(&fmhd.bcell[aa], &cmhd.bcell[aa], cov);
            }
            restrict_bface(&fmhd.bface, &cmhd.bface, cov);
            let shell = shell_around(cov, &coarse.state.geom.interior);
            bcell_from_bface_region(cmhd, coarse.state.fields.cons.nrg_field(), &shell);
        }
        coarse.kernels.c2p(&coarse.state);
        self.fill_physical_ghosts(level);
    }

    /// synchronize every covered parent region from the finest level down to the root.
    pub fn sync_all_fine_to_coarse(&self) {
        for level in (0..self.levels.len().saturating_sub(1)).rev() {
            self.sync_fine_to_coarse(level);
        }
    }

    /// restrict the finer level into this level's coverage + apply the flux/emf reflux + re-derive
    /// prim (and, if this level has a coarser parent, re-prolong its coarse-fine ghosts). runs after
    /// the finer-level subcycle. the flux/emf registers are tile-local (a coarse cell + the fine
    /// cells at its face are co-located), so the decomposed driver calls this per tile unchanged.
    /// a pure extraction.
    pub fn level_restrict_reflux(&mut self, level: usize, alpha0: f64) {
        let has_coarser = level > 0;
        let has_finer = level + 1 < self.levels.len();
        let is_mhd = self.levels[level].state.fields.mhd.is_some();
        if has_finer {
        prof("refine_restrict", || {
            let (lo, hi) = self.levels.split_at(level + 1);
            let coarse = &lo[level];
            let fine = &hi[0];
            let cov = coarse.coverage.as_ref().unwrap();
            restrict_cons(&fine.state.fields.cons, &coarse.state.fields.cons, cov);
            // mhd: restrict the staggered B (interface faces included),
            // reflux the outside faces from the edge-EMF mismatch, then
            // re-derive cell B + the magnetic-energy correction over the
            // coverage plus the one-cell shell whose faces just changed.
            if is_mhd {
                let cmhd = coarse.state.fields.mhd.as_ref().unwrap();
                let fmhd = fine.state.fields.mhd.as_ref().unwrap();
                for aa in 0..DOF {
                    restrict_cell_field(&fmhd.bcell[aa], &cmhd.bcell[aa], cov);
                }
                restrict_bface(&fmhd.bface, &cmhd.bface, cov);
                let inv_dx: [f64; NDIM] = std::array::from_fn(|ax| 1.0 / coarse.state.geom.dx[ax]);
                self.emf_registers[level]
                    .as_ref()
                    .unwrap()
                    .apply(&cmhd.bface, &inv_dx);
                let shell = shell_around(cov, &coarse.state.geom.interior);
                bcell_from_bface_region(cmhd, coarse.state.fields.cons.nrg_field(), &shell);
            }
        });
        }
        if self.has_flux_register(level) {
        prof("refine_reflux_apply", || {
            let l = &self.levels[level];
            if uniform_cartesian(&l.state) {
                self.flux_registers[level].apply_uniform(&l.state.fields.cons, &l.state.geom.dx);
                if let Some(m) = l.state.fields.mhd.as_ref() {
                    if DOF > NDIM {
                        self.flux_registers[level].apply_uniform_oop(&m.bcell, &l.state.geom.dx);
                    }
                }
            } else {
                let geo = l.state.geom.block_geometry(l.state.physics.metric);
                self.flux_registers[level].apply(&l.state.fields.cons, &geo);
            }
        });
        }
        let l = &self.levels[level];
        prof("c2p", || l.kernels.c2p(&l.state));
        if has_finer && self.levels[level].cons_eq.is_none() && self.cf_transfer_balanced(level + 1) {
            // SYMBI_WB_BAND=0 withholds the band rewrite so the seam carries the
            // conservative restriction alone; the identical binary then runs the
            // counterfactual arm of the seam admissibility census, attributing
            // decode abstentions between the band's anchor rewrite and the
            // turbulent density contrast.
            if wb_band_enabled() {
                prof("refine_restrict_balanced", || {
                    self.restrict_band_balanced(level)
                });
            }
        }
        if has_coarser {
            self.prolong_cf(level, alpha0 + 1.0 / RATIO as f64);
        }
        prof("ghost_fill", || self.fill_physical_ghosts(level));
    }

    /// the balanced restriction of level `level`'s seam bands from level `level + 1`:
    /// the covered coarse cells within the evolution reach of each coarse-fine seam
    /// are rewritten onto the coarse mechanical chain from the uncovered cell beyond
    /// the seam, carrying the fine departure. runs after the conservative restriction
    /// and the coarse c2p, and rebuilds the band's conserved energy, which requires
    /// the gamma-law closure the kernel set reports.
    fn restrict_band_balanced(&self, level: usize) {
        let coarse = &self.levels[level];
        let fine = &self.levels[level + 1];
        let gamma = coarse.kernels.gamma_law().unwrap_or_else(|| {
            panic!(
                "the balanced coarse-fine restriction rebuilds the seam band's energy under the \
                 gamma law; level {level}'s kernel set reports another energy closure"
            )
        });
        let departure = fine.band_departure.get_or_init(|| {
            Field::zeros(&fine.state.geom.allocated).expect("band departure scratch")
        });
        // the coarse level's own scratch holds the restricted departures; its other
        // role -- the fine encode target of the seam one level up -- writes the
        // interior edge strips, disjoint from the coverage strips written here.
        let coarse_departure = coarse.band_departure.get_or_init(|| {
            Field::zeros(&coarse.state.geom.allocated).expect("band departure scratch")
        });
        restrict_band_balanced(
            &fine.state.fields.prim,
            &fine.state.geom.interior,
            departure,
            coarse_departure,
            &coarse.state.fields.prim,
            coarse
                .state
                .fields
                .cons
                .nrg_field()
                .expect("the balanced restriction needs cons.nrg"),
            &coarse.state.geom.interior,
            coarse.coverage.as_ref().unwrap(),
            // the uncovered cell's update over one step depends on cells within
            // `reach` per stage; the band holds the cells its own stencils read
            // and two more, each of which removes a limiter-slope hop from the
            // path the conservatively restricted interior's class offset takes
            // to reach it (measured ~10x attenuation per cell on the sealed
            // class column). capped by the decode's unrolled chain.
            (coarse.kernels.reconstruction_reach() as usize + 2)
                .min(symbi_discretize::WB_CF_CHAIN_MAX as usize),
            gamma,
            &shifted_origin(&fine.state.geom),
            &fine.state.geom.dx,
            &shifted_origin(&coarse.state.geom),
            &coarse.state.geom.dx,
            &coarse.state.immersed.as_ref().unwrap().bodies,
        );
    }

    /// advance this level's clock (fine levels only; the root clock is the driver's). pure extraction.
    fn level_clock(&mut self, level: usize, dt: f64) {
        if level > 0 {
            let s = &mut self.levels[level].state;
            (s.time, s.iteration) = advance_clock(s.time, s.iteration, dt);
        }
    }

    /// whether level `level`'s coarse-fine ghost transfer rides the hydrostatic
    /// equilibrium decomposition: the explicit override when set, otherwise the level's
    /// own kernel set. active only where the decomposition is defined —
    /// a gravitating body supplies the potential and the prim set carries pressure
    /// (gamma-law; asserted at the transfer). the encode/decode are baked kernels,
    /// so host and device hierarchies take the same path.
    fn cf_transfer_balanced(&self, level: usize) -> bool {
        // SYMBI_WB_GHOST=0 withholds the balanced ghost transfer alone: the seam
        // ghosts take the plain prolongation while the in-level balanced flux,
        // the balanced body source, and the restriction band keep their own
        // switches. the ghost decode anchors on the fine interior, so a run
        // with the switch off carries seam boundary data built from the coarse
        // level only, which is the single-variable arm for attributing a seam
        // failure to that anchoring.
        if !wb_ghost_enabled() {
            return false;
        }
        let want = self
            .balance_aware_transfer
            .unwrap_or_else(|| self.levels[level].kernels.hydrostatic_balance());
        if !want {
            return false;
        }
        let fine = &self.levels[level];
        let has_gravity = fine
            .state
            .immersed
            .as_ref()
            .is_some_and(|im| (0..im.bodies.len()).any(|b| im.bodies.get(b).has_gravity()));
        has_gravity && fine.state.fields.prim.pre_field().is_some()
    }

    /// the physical ghost fill of level `ll`. a refined level also takes the cells where a periodic
    /// axis meets a coarse-fine side, which the fill's periodic pass leaves to the side it skips:
    /// they copy from the prolonged coarse-fine ghost rows across the seam
    pub fn fill_physical_ghosts(&self, ll: usize) {
        let l = &self.levels[ll];
        l.kernels.ghost_fill(&l.state);
        if ll > 0 {
            symbi_substrate::regimes::mhd_substrate::wrap_periodic_corner_ghosts(&l.state);
        }
    }

    /// fill level `level`'s coarse-fine prim ghosts from its parent's
    /// time-interpolated prims: `(1 - alpha)*prim_old + alpha*prim_new`. pub so the decomposed
    /// driver can drive the fine subcycle (prolong -> fine ghost -> fine stages with exchange).
    pub fn prolong_cf(&self, level: usize, alpha: f64) {
        let parent = &self.levels[level - 1];
        let fine = &self.levels[level];
        let prim_old = parent
            .prim_old
            .as_ref()
            .expect("the parent of a fine level carries prim_old");
        let prim_lerp = parent
            .prim_lerp
            .as_ref()
            .expect("the parent of a fine level carries prim_lerp");
        // a slab is clipped to the fine cells the parent can actually feed: a patch that
        // runs out to the edge of a decomposed tile has ghost cells whose prolongation
        // stencil reaches past the parent's own cells, and the fine halo exchange carries
        // those from the neighbor tile's fine interior.
        let slabs: Vec<_> = cf_ghost_slabs(
            &fine.state.geom.allocated,
            &fine.state.geom.interior,
            &fine.state.boundaries,
        )
        .iter()
        .filter_map(|slab| {
            clip_to_prolongable(slab, &parent.state.geom.allocated, self.prolong_order)
        })
        .collect();
        let sweep_scratch = fine.prolong_sweep.get_or_init(|| {
            let has_pre = fine.state.fields.prim.pre_field().is_some();
            slabs
                .iter()
                .map(|s| ProlongSweepScratch::for_slab(s, self.prolong_order, has_pre))
                .collect()
        });
        let target = parent.prim_eq.as_ref().zip(fine.prim_eq.as_ref());
        let balanced = self.cf_transfer_balanced(level);
        for (slab, scratch) in slabs.iter().zip(sweep_scratch) {
            if let Some((coarse_target, fine_target)) = target {
                prolong_prims_targeted(
                    scratch,
                    prim_lerp,
                    prim_old,
                    &parent.state.fields.prim,
                    coarse_target,
                    fine_target,
                    &fine.state.fields.prim,
                    slab,
                    self.prolong_order,
                    alpha,
                );
            } else if balanced {
                // the equilibrium decomposition: encode the lerped coarse slab's
                // pressure as its departure from the mechanical equilibrium chained
                // out of the interior, prolong the departures with the unchanged
                // kernels, rebuild each fine ghost's pressure on the fine chain from
                // its nearest interior cell. baked kernel pair in the parent's lerp
                // scratch, with the slab's sweep intermediates carrying the departures.
                prolong_prims_balanced(
                    scratch,
                    prim_old,
                    &parent.state.fields.prim,
                    prim_lerp,
                    &fine.state.fields.prim,
                    slab,
                    fine.patch_interior.as_ref().unwrap_or(&fine.state.geom.interior),
                    self.prolong_order,
                    alpha,
                    &shifted_origin(&parent.state.geom),
                    &parent.state.geom.dx,
                    &shifted_origin(&fine.state.geom),
                    &fine.state.geom.dx,
                    &parent.state.immersed.as_ref().unwrap().bodies,
                    &fine.state.immersed.as_ref().unwrap().bodies,
                );
            } else {
                prolong_prims_swept(
                    scratch,
                    prim_lerp,
                    prim_old,
                    &parent.state.fields.prim,
                    &fine.state.fields.prim,
                    slab,
                    self.prolong_order,
                    alpha,
                );
            }
            // the dye rides the same time-interpolated prolongation as the rest of the primitive
            // state. it sits outside the swept pass because that pass carries a positional
            // component count (rho, DOF velocities, optional pressure) sized by its scratch.
            if let (Some(chi_old), Some(pchi), Some(fchi)) = (
                prim_old.chi_field(),
                parent.state.fields.prim.chi_field(),
                fine.state.fields.prim.chi_field(),
            ) {
                prolong_field(chi_old, pchi, fchi, slab, self.prolong_order, alpha);
            }
            // mhd: the cell-centered B ghosts feed the fine reconstruction +
            // the boundary-edge UCT emf. the fine CT evolves its own owned
            // bface, and CT preserves divB for whatever emf values it is
            // handed, so those faces stand on their own.
            if let (Some(fmhd), Some(bcell_old)) =
                (fine.state.fields.mhd.as_ref(), parent.bcell_old.as_ref())
            {
                let pmhd = parent.state.fields.mhd.as_ref().unwrap();
                for aa in 0..DOF {
                    prof("refine_prolong_face", || {
                        prolong_field(
                            &bcell_old[aa],
                            &pmhd.bcell[aa],
                            &fmhd.bcell[aa],
                            slab,
                            self.prolong_order,
                            alpha,
                        )
                    });
                }
            }
        }
        // the bface transverse halo at coarse-fine sides: the scalar ghost
        // fill skips CF faces, so the transversely-extended flux sweep would
        // read stale normal-B there — prolong it from the coarse face lattice
        // (divB and conservation are indifferent to these values; this is the
        // fine boundary-edge EMF quality).
        if let (Some(fmhd), Some(bface_old)) =
            (fine.state.fields.mhd.as_ref(), parent.bface_old.as_ref())
        {
            let pmhd = parent.state.fields.mhd.as_ref().unwrap();
            // the sides on a cut the patch continues across take their face halo from the
            // fine level's own exchange: the prolongation's cross slopes would read parent
            // faces beyond the parent's own cut, which its face lattice never holds
            let prolonged = Self::prolonged_face_boundaries(fine);
            // the parent faces come from the gathered stencil when the fine level reads one:
            // it holds the faces past the tile's lattice at both time levels. during `prime` the
            // stencil is ungathered and the halo is left to the first substep's prolongation
            let (old, new): (&[Field<f64, NDIM, Mem>], &[Field<f64, NDIM, Mem>]) = match fine.face_stencil.as_ref() {
                Some(stencil) if stencil.filled.get() => (&stencil.old[..], &stencil.new[..]),
                Some(_) if self.priming => return,
                Some(_) => panic!("the parent face stencil is read before the root stages gathered it"),
                None => (&bface_old[..], &pmhd.bface.b[..]),
            };
            for dd in 0..NDIM {
                for slab in bface_cf_halo_slabs(&fine.state.geom.interior, &prolonged, dd) {
                    prof("refine_prolong_face", || {
                        prolong_face_field(dd, old, new, &fmhd.bface[dd], &slab, alpha)
                    });
                }
            }
        }
    }
}

// =============================================================================
// well-balancing against a stationary target
// =============================================================================

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
    Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF> + Copy,
    M: Metric<f64, NDIM> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
{
    /// declare the run's stationary target state `qt`, making it an exact fixed point of the
    /// refined scheme.
    ///
    /// a steady state solves the continuum equations, so on any grid the scheme leaves a residual
    /// `R = div_h F_h(qt) - s_h(qt)` at truncation order, and an atmosphere seeded on the exact
    /// hydrostatic profile starts moving. worse, `R` is grid-dependent, so the coarse-fine flux
    /// register differences two unequal reconstructions of one exact solution and applies the
    /// difference to the coarse cells at the interface as a force. this captures both halves of
    /// the cure: `F(qt)` per level, which the registers subtract from both sides of their
    /// difference, and `R` per level, which every stage adds back.
    ///
    /// the target reaches the flux phase through the preparation an evolving state gets —
    /// conserved-to-primitive recovery, coarse-fine prolongation, physical ghost fill — and `R` is
    /// read off one explicit stage of the shared pipeline, so both quantities are what the scheme
    /// genuinely produces when handed the target, ghost band and every source included.
    ///
    /// the target must be time-independent: both quantities are evaluated once here and reused
    /// every stage of every step. each level's conserved and primitive state is left exactly as it
    /// was found, so this may be called at any point after the initial condition is seeded.
    pub fn with_equilibrium(
        mut self,
        target: impl Fn([f64; NDIM]) -> <R as Regime<f64, DOF>>::Prim,
    ) -> symbi_xpu::Result<Self> {
        for (ll, level) in self.levels.iter().enumerate() {
            assert!(
                level.state.fields.mhd.is_none(),
                "level {ll}: a cell-centered primitive cannot seed the staggered face field, so a \
                 magnetized target's interface flux is undefined"
            );
            assert!(
                level.state.fields.cons.chi_field().is_none(),
                "level {ll}: the passive scalar carries no declared target concentration, so its \
                 interface flux has nothing to difference against"
            );
        }

        let saved = self
            .levels
            .iter()
            .map(|level| save_gas_state(&level.state))
            .collect::<symbi_xpu::Result<Vec<_>>>()?;

        // the copy above runs on the device queue while the seeding below writes the same fields
        // from the host. field storage is unified memory, so both reach the same bytes, but a host
        // write is unordered against an in-flight kernel, so this barrier is what keeps the copy
        // from reading cells the target had already overwritten and "restoring" the run to the
        // target it was asked to measure.
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        for level in &self.levels {
            level.state.seed_cells(&target);
        }
        // the target has to be hierarchy-consistent: a coarse target cell equals the
        // volume-weighted average of the fine target cells covering it, or the run's own
        // restriction moves a level that was sitting on its target off it, twice per parent step.
        // an analytic profile evaluated independently per level breaks that — the two differ at
        // the profile's curvature — so the coarse target is defined as the restriction of the fine
        // one wherever a finer level exists.
        for ll in (0..self.levels.len().saturating_sub(1)).rev() {
            let (lo, hi) = self.levels.split_at(ll + 1);
            let coarse = &lo[ll];
            let fine = &hi[0];
            restrict_cons(
                &fine.state.fields.cons,
                &coarse.state.fields.cons,
                coarse.coverage.as_ref().unwrap(),
            );
        }
        self.init_levels();

        // the target's conserved state, held aside as the anchor the probe stage is differenced
        // against.
        let anchor = self
            .levels
            .iter()
            .map(|level| save_gas_state(&level.state))
            .collect::<symbi_xpu::Result<Vec<_>>>()?;

        let mut flux_eq = Vec::with_capacity(self.levels.len());
        let mut residual_eq = Vec::with_capacity(self.levels.len());
        for (ll, (level, anchored)) in self.levels.iter().zip(&anchor).enumerate() {
            level.kernels.wave_speeds(&level.state);
            for dd in 0..NDIM {
                level.kernels.flux(&level.state, dd);
            }
            flux_eq.push(snapshot_flux(&level.state)?);

            // one forward-euler stage of the shared pipeline, evaluated entirely at the target:
            // every flux and every source sees the target and no other state, so the advanced
            // state is `qt - dt R` and the probe step cancels out of the quotient. the level's own
            // cfl step is the scale that keeps that true — a far larger step drives the probe
            // state unphysical and the response stops being linear, a far smaller one loses the
            // difference `qt - advanced` to cancellation.
            let probe_dt = level.kernels.cfl(&level.state);
            // the probe's step per level: the timestep a decomposed run's probe must reproduce
            eprintln!(
                "SIMBI equilibrium probe: level={ll} dt_bits={:016x}",
                probe_dt.to_bits()
            );
            assert!(
                probe_dt.is_finite() && probe_dt > 0.0,
                "the stationary target has no finite cfl step ({probe_dt:.3e}), so it cannot be \
                 advanced to read off its imbalance"
            );
            // the probe is a diagnostic stage: the kernel's freeze count stands where the probe
            // found it, as on every tiled march
            let streak = level.kernels.freeze_streak();
            let outcome = fold_stage(
                &level.state,
                &level.kernels,
                StageArgs {
                    dt: probe_dt,
                    a0: 0.0,
                    ac: 1.0,
                    stage: 0,
                    n_stages: 1,
                    injection_weight: 1.0,
                    allow_elision: false,
                    exit_fraction: 1.0,
                },
                &mut |_hp: HookPoint| {},
            );
            level.kernels.set_freeze_streak(streak);
            // the probe stage reads the imbalance off; the stored state stands at the clock
            level.state.reset_state_clock();
            assert!(
                outcome == symbi_sim::stage::StageOutcome::Accepted,
                "the stationary target was rejected by the admissibility redo, so it is not a \
                 state the scheme can carry"
            );
            // an inadmissible probe state means the update was clipped somewhere, and a clipped
            // update departs from `qt - dt R`: what would be read off is a limiter's response in
            // place of the scheme's imbalance.
            symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
            let err = scan_c2p_errors(&level.state);
            assert!(
                err.is_ok(),
                "advancing the stationary target by one cfl step left an unrecoverable state \
                 ({err}), so its imbalance cannot be read off a linear response"
            );
            residual_eq.push(imbalance_from_stage(
                anchored.cons(),
                &level.state.fields.cons,
                &level.state.geom.allocated,
                probe_dt,
            )?);
        }

        for (level, state) in self.levels.iter().zip(&saved) {
            restore_gas_state(&level.state, state);
        }
        for (((level, flux), residual), anchored) in self
            .levels
            .iter_mut()
            .zip(flux_eq)
            .zip(residual_eq)
            .zip(anchor)
        {
            level.flux_eq = Some(flux);
            level.residual_eq = Some(residual);
            let (cons, prim) = anchored.into_parts();
            level.cons_eq = Some(cons);
            level.prim_eq = Some(prim);
        }
        self.report_target_stationarity();
        Ok(self)
    }

    /// a decomposed run's share of [`Self::with_equilibrium`] on this tile: refuse a target the
    /// scheme cannot difference, seed every level with it, and restrict the fine target into the
    /// covered coarse cells, finest pair first
    pub fn seed_declared_target(
        &self,
        target: &dyn Fn([f64; NDIM]) -> <R as Regime<f64, DOF>>::Prim,
    ) -> Result<(), String> {
        for (ll, level) in self.levels.iter().enumerate() {
            if level.state.fields.mhd.is_some() || level.state.fields.cons.chi_field().is_some() {
                return Err(format!(
                    "level {ll}: a magnetized or dyed target has no interface flux to difference against"
                ));
            }
        }
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        for level in &self.levels {
            level.state.seed_cells(target);
        }
        for ll in (0..self.levels.len().saturating_sub(1)).rev() {
            let (lo, hi) = self.levels.split_at(ll + 1);
            restrict_cons(&hi[0].state.fields.cons, &lo[ll].state.fields.cons, lo[ll].coverage.as_ref().unwrap());
        }
        Ok(())
    }

    /// the declared target must actually be a steady state of the equations being solved.
    ///
    /// the method takes the declaration as given: the imbalance is measured and subtracted whatever
    /// it is, so a state merely asserted to be stationary gets held motionless and the run proceeds
    /// silently on the wrong problem. that silence is this feature's sharpest edge.
    ///
    /// the discriminator is how the imbalance behaves under refinement. for a genuine steady state
    /// `R` is pure truncation error, `C(x) dx^p`, so halving the cell width cuts its norm over a
    /// fixed region by `2^p`. for a state that departs from the equations, `R` converges to the
    /// continuum residual, a property of the state alone that holds flat under refinement. every
    /// level pair measures this for free, over the region they both cover.
    ///
    /// the test is source-agnostic — gravity, rotation, the curvilinear geometric terms and any
    /// user source all enter `R` through the stage that produced it.
    /// report how the declared target's discrete imbalance behaves under refinement.
    ///
    /// this is advisory: it reports and lets the run continue. the deviation method carries it as a
    /// diagnostic, because declaring a non-stationary state is a silent error — the scheme holds
    /// whatever it is given, so the run proceeds quietly on the wrong problem.
    ///
    /// it is advisory because soundness and completeness pull apart here. a genuine steady state's
    /// imbalance is truncation error, which converges where the grid resolves the target; a
    /// non-stationary state leaves the continuum residual, which holds flat everywhere. for a
    /// strongly stratified target those two populations overlap — the imbalance is large where the
    /// profile is steep, and steep is where convergence stays undemonstrable either way. a check
    /// that refused on that basis would reject correct equilibria for being sharp, which is the
    /// more damaging error: it blocks valid science, where a missed warning leaves a mistake the
    /// user can still find by other means.
    ///
    /// what it reports is therefore evidence for the reader to weigh. a median near 1 on resolved
    /// cells is strong evidence the target departs from these equations; anything else is
    /// inconclusive.
    fn report_target_stationarity(&self) {
        for ll in 0..self.levels.len().saturating_sub(1) {
            let Some(measured) = self.target_imbalance_convergence(ll) else {
                continue;
            };
            let table: Vec<String> = (0..measured.ratio.len())
                .map(|cc| {
                    format!(
                        "[{cc}] ratio {:.3} scale {:.3e} cells {}",
                        measured.ratio[cc], measured.scale[cc], measured.sampled[cc]
                    )
                })
                .collect();
            // no component with broadly distributed signal on resolved cells: nothing was
            // measured. say so, because silence would otherwise read as "checked and clean".
            if (0..measured.ratio.len()).all(|cc| measured.sampled[cc] < MIN_CONVERGENCE_SAMPLE) {
                eprintln!(
                    "note: the declared stationary target between levels {ll} and {} was not \
                     checked. its imbalance is concentrated where the grid does not resolve it \
                     ({} of {} overlapping cells resolved), so no component carries signal on \
                     cells that could show convergence either way. this is expected for a sharply \
                     stratified target and is NOT evidence of a problem; it is also not evidence \
                     of correctness. every component: {}",
                    ll + 1,
                    measured.resolved,
                    measured.considered,
                    table.join("  ")
                );
                continue;
            }
            let suspect = (0..measured.ratio.len()).any(|cc| {
                measured.sampled[cc] >= MIN_CONVERGENCE_SAMPLE && measured.ratio[cc] <= 1.5
            });
            if suspect {
                eprintln!(
                    "warning: the declared stationary target may not be a steady state of these \
                     equations. between levels {ll} and {}, a conserved component's discrete \
                     imbalance did not shrink when the cell width halved, on cells where the grid \
                     resolves the target. truncation error falls by at least 2; what does not fall \
                     is the continuum residual. if that is so, well-balancing this state would \
                     hold the run motionless in something that is not an equilibrium. \
                     {} of {} overlapping cells resolved; every component: {}",
                    ll + 1,
                    measured.resolved,
                    measured.considered,
                    table.join("  ")
                );
            }
        }
    }

    /// how the target's discrete imbalance behaves under refinement, measured per cell and reduced
    /// with a median — the raw evidence behind the steady-state check, threshold-free.
    ///
    /// each coarse cell in the overlap is compared against the mean of its own children, so the
    /// statistic is local. that is what makes it usable: a ratio of summed norms is dominated by
    /// wherever the imbalance happens to be largest, and for a target with an unresolved feature —
    /// the `1/r` cusp of a point-mass atmosphere at the center of a refinement box, say — that is a
    /// handful of cells whose error stays flat because both grids under-resolve them. those cells
    /// are real, and they are silent on whether the declared state solves the equations. a median
    /// over cells is robust to them.
    ///
    /// `None` when the two levels share too little interior to compare. the region is the coverage
    /// trimmed by two coarse cells on every side: the fine level's outermost cells reconstruct
    /// through prolonged coarse-fine ghosts, whose error converges at the prolongation's order
    /// rather than the scheme's, and a mixture of the two measures neither.
    pub fn target_imbalance_convergence(&self, pair: usize) -> Option<ImbalanceConvergence> {
        const TRIM: isize = 2;
        // the imbalance is written by device kernels and read here on the host; the read goes
        // through unified memory, which leaves the barrier below to order it against the queue.
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        let coarse = self.levels.get(pair)?;
        let fine = self.levels.get(pair + 1)?;
        let coverage = coarse.coverage.as_ref()?;

        let inner = Domain::new(std::array::from_fn(|aa| {
            let s = &coverage.spaces[aa];
            Space {
                name: s.name,
                lo: s.lo + TRIM,
                hi: s.hi - TRIM,
            }
        }));
        if (0..NDIM).any(|aa| inner.spaces[aa].hi <= inner.spaces[aa].lo) {
            return None;
        }

        // the cells the question can be asked of. truncation error is a statement about a
        // solution the grid resolves; where the target changes by a large factor across one cell
        // the reconstruction is clipping rather than approximating, and its error carries no
        // order. such cells are real and are corrected like any other — they are silent on whether
        // the declared state solves the equations, so they stay out of the statistic instead of
        // setting it.
        let target = coarse.cons_eq.as_ref()?;
        let den = target.den.view();
        // a cell testifies about convergence when the grid resolves the target there: where the
        // density turns over inside one width the reconstruction is clipping rather than
        // approximating, and its error carries no order either way.
        // inside a sink the drain removes mass and energy, so the target departs from a steady
        // state of the applied equations there and those cells report the drain rather than
        // truncation, setting the statistic from a region the declaration left out.
        let mask = self.resolved_equilibrium_mask();
        let geom = &coarse.state.geom;
        let offered: Vec<[isize; NDIM]> = inner
            .iter()
            .filter(|c| {
                if mask <= 0.0 {
                    return true;
                }
                let x = geom.cell_coord(*c);
                let r2: f64 = (0..NDIM).map(|ax| x[ax] * x[ax]).sum();
                r2 >= mask * mask
            })
            .collect();
        let resolved: Vec<[isize; NDIM]> = offered
            .iter()
            .copied()
            .filter(|c| {
                let here = den.at(*c).abs().max(f64::MIN_POSITIVE);
                (0..NDIM).all(|ax| {
                    [-1isize, 1].iter().all(|step| {
                        let mut nb = *c;
                        nb[ax] += step;
                        (den.at(nb) - den.at(*c)).abs() / here < RESOLVED_VARIATION
                    })
                })
            })
            .collect();

        let coarse_comps = residual_components(coarse.residual_eq.as_ref()?);
        let fine_comps = residual_components(fine.residual_eq.as_ref()?);

        // the scale is the component's peak over the whole overlap, the resolved subset included.
        // it sets what counts as real signal, and a floor derived from the quiet cells alone would
        // admit their own roundoff as evidence.
        let scale: Vec<f64> = coarse_comps
            .iter()
            .map(|field| {
                let view = field.view();
                inner
                    .iter()
                    .map(|c| view.at(c).abs())
                    .fold(0.0_f64, f64::max)
            })
            .collect();

        let mut ratio = Vec::with_capacity(coarse_comps.len());
        let mut sampled = Vec::with_capacity(coarse_comps.len());
        for (cc, (cf, ff)) in coarse_comps.iter().zip(&fine_comps).enumerate() {
            let (cv, fv) = (cf.view(), ff.view());
            let mut ratios = Vec::new();
            for c in &resolved {
                let numerator = cv.at(*c).abs();
                // a cell whose own imbalance sits at roundoff reports noise over noise, so the
                // sample is restricted to cells that are resolved and carrying signal. a cell
                // whose residual is a millionth of this component's peak is reporting roundoff,
                // and a median of such cells reads 1 whatever the target is — the same reading a
                // genuine non-convergence gives.
                if numerator < SIGNAL_FLOOR * scale[cc].max(f64::MIN_POSITIVE) {
                    continue;
                }
                let mut child_sum = 0.0;
                let mut children = 0usize;
                for_each_child::<NDIM>(*c, RATIO, |fc| {
                    child_sum += fv.at(fc).abs();
                    children += 1;
                });
                let child_mean = child_sum / children as f64;
                if child_mean > 0.0 {
                    ratios.push(numerator / child_mean);
                }
            }
            sampled.push(ratios.len());
            ratio.push(median(&mut ratios));
        }
        Some(ImbalanceConvergence {
            ratio,
            scale,
            sampled,
            resolved: resolved.len(),
            considered: offered.len(),
        })
    }

    /// declare the stationary target as an expression of position rather than as a closure.
    ///
    /// this is the form a configured run supplies, and the form that survives a restart: the
    /// target is re-derived from the expression at whatever resolution the restarted hierarchy
    /// has, so a restart that adds a refinement level gets a target on the newly created cells.
    /// sampled field data covers the cells it was written from alone.
    ///
    /// the expression's outputs are the primitive components in order — density, one velocity
    /// component per momentum degree of freedom, then pressure when the regime carries energy —
    /// and are evaluated at each cell center at time zero, because the target is stationary.
    pub fn with_equilibrium_expression(
        self,
        config: &symbi_source_compile::EquilibriumConfig,
    ) -> symbi_xpu::Result<Self>
    where
        <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
    {
        let has_energy = self.levels[0].state.fields.cons.has_energy();
        let expected = 1 + DOF + usize::from(has_energy);
        assert_eq!(
            config.outputs.len(),
            expected,
            "the declared stationary target supplies {} primitive components; this run needs \
             {expected} — density, {DOF} velocity component(s){}",
            config.outputs.len(),
            if has_energy {
                ", and pressure"
            } else {
                " (no pressure: the regime carries no energy)"
            }
        );
        assert_eq!(
            config.dim, NDIM,
            "the declared stationary target was built for a {}-dimensional grid, but this run is \
             {NDIM}-dimensional",
            config.dim
        );
        let expression = config
            .to_expression()
            .unwrap_or_else(|err| panic!("the declared stationary target does not load: {err:?}"));

        self.with_equilibrium(|x| {
            let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
            // a stationary target is evaluated once, at t = 0; it is the same state at every later
            // time by definition.
            let slots = expression.eval(at(0), at(1), at(2), 0.0);
            <R as Regime<f64, DOF>>::Prim::from_slots(&slots)
        })
    }

    /// seed every level's conserved state from the declared target.
    ///
    /// the target a refined hierarchy holds exactly has covered cells carrying the restriction of
    /// the finer level's target, because that is what the run's own restriction produces and
    /// re-produces every parent step. a run seeded from the pointwise profile instead starts a
    /// truncation-order distance off the state the well-balancing preserves, and that distance
    /// evolves like any other perturbation. build a perturbed initial condition on top of this
    /// seeding pass.
    pub fn seed_equilibrium(&mut self) {
        for (ll, level) in self.levels.iter().enumerate() {
            let target = level.cons_eq.as_ref().unwrap_or_else(|| {
                panic!("level {ll} has no declared stationary target to seed from")
            });
            equilibrium_overwrite(
                target,
                &level.state.fields.cons,
                &level.state.geom.allocated,
            );
        }
        self.init_levels();
    }

    /// rewrite every level's interior in primitive space: `f` receives each cell's physical
    /// center and its recovered primitive and returns the primitive to store. every level
    /// evaluates that one closure at its own cell centers, so a perturbation carries each
    /// level's full resolvable content in place of the coarse grid's prolongation — the entry
    /// point for laying a multi-scale velocity field over a prolonged smooth base state.
    ///
    /// may be called repeatedly (a measured correction pass after a seeding pass, say);
    /// call `sync_perturbed` once after the last pass.
    pub fn perturb_cells(
        &self,
        f: impl Fn([f64; NDIM], <R as Regime<f64, DOF>>::Prim) -> <R as Regime<f64, DOF>>::Prim,
    ) where
        <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
    {
        for (ll, level) in self.levels.iter().enumerate() {
            assert!(
                level.state.fields.mhd.is_none(),
                "level {ll}: a cell-centered primitive rewrite cannot update the staggered \
                 face field, so a magnetized state's div(B) = 0 would not survive it"
            );
        }
        // field storage is unified memory; this barrier is what orders a host write against an
        // in-flight kernel.
        symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
        for level in &self.levels {
            level.state.perturb_cells(&f);
        }
    }

    /// restore hierarchy consistency after `perturb_cells`: covered coarse cells are
    /// re-defined as the restriction of the fine state — what the run's own sync produces
    /// every parent step — and the primitive/ghost state is re-initialized on every level.
    pub fn sync_perturbed(&mut self) {
        for ll in (0..self.levels.len().saturating_sub(1)).rev() {
            let (lo, hi) = self.levels.split_at(ll + 1);
            let coarse = &lo[ll];
            let fine = &hi[0];
            restrict_cons(
                &fine.state.fields.cons,
                &coarse.state.fields.cons,
                coarse.coverage.as_ref().unwrap(),
            );
        }
        self.init_levels();
    }
}

// =============================================================================
// helpers
// =============================================================================

/// how a declared stationary target's discrete imbalance behaves under one halving of the cell
/// width, per conserved component.
pub struct ImbalanceConvergence {
    /// median over resolved coarse cells of `|R_coarse(cell)| / mean |R_fine(its children)|`.
    /// truncation error gives `2^p` (4 for a second-order scheme); a continuum residual gives 1.
    pub ratio: Vec<f64>,
    /// the largest single-cell `|R|` over the resolved cells — the scale below which a
    /// component's ratios are quotients of roundoff and say nothing.
    pub scale: Vec<f64>,
    /// how many resolved cells carried enough signal to contribute a ratio, per component.
    pub sampled: Vec<usize>,
    /// how many cells the target is resolved on, out of `considered`.
    pub resolved: usize,
    /// how many cells the overlap offered before the resolution filter.
    pub considered: usize,
}

/// the fewest cells a median may be taken over before it is a statistic rather than an accident.
const MIN_CONVERGENCE_SAMPLE: usize = 8;

/// the largest fractional change in the target's density across one cell for that cell to count as
/// resolved. beyond it a limited reconstruction is clipping rather than approximating and its
/// error carries no order, leaving the cell silent about convergence either way. a stratified
/// atmosphere spread over its grid sits far below this; the cells it excludes are the ones where
/// the profile turns over inside a single width — a `1/r` cusp at a box center, typically.
const RESOLVED_VARIATION: f64 = 0.25;

/// the smallest share of a component's peak imbalance a cell must carry to vote on convergence.
/// below it the cell reports roundoff, and a median of such cells reads 1 whatever the target is —
/// which is exactly the value a genuinely non-stationary target gives, so admitting them makes the
/// two indistinguishable.
const SIGNAL_FLOOR: f64 = 1.0e-3;

/// the median of `values`, or 0 when empty. sorts in place; the caller owns the buffer.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

/// visit the `ratio^D` fine cells covering one coarse cell. levels share absolute index space, so
/// the children of coarse cell `c` start at `ratio * c`.
fn for_each_child<const D: usize>(c: [isize; D], ratio: usize, mut f: impl FnMut([isize; D])) {
    let base: [isize; D] = std::array::from_fn(|ax| c[ax] * ratio as isize);
    let total = ratio.pow(D as u32);
    for n in 0..total {
        let mut child = base;
        let mut rem = n;
        for slot in child.iter_mut() {
            *slot += (rem % ratio) as isize;
            rem /= ratio;
        }
        f(child);
    }
}

/// the per-stage effective flux weights of an ssp scheme: stage i's operator
/// enters the step total with `ac_i * prod_{k>i} ac_k` (each later convex
/// combine rescales the accumulated state by its ac). euler -> [1]; rk2 ->
/// [1/2, 1/2]; rk3 -> [1/6, 1/6, 2/3]. the weights sum to 1; the register
/// accumulates `weight * dt * F * A` so refluxing matches the actual flux the
/// scheme pushed through each face.
fn flux_weights(stages: &[(f64, f64)]) -> Vec<f64> {
    (0..stages.len())
        .map(|ii| {
            let mut w = stages[ii].1;
            for kk in ii + 1..stages.len() {
                w *= stages[kk].1;
            }
            w
        })
        .collect()
}

/// is this level on a uniform cartesian grid — the flux register's kernel
/// path (cpu + gpu, constant area/volume scales)? curvilinear levels keep
/// the per-coordinate host loops.
fn uniform_cartesian<R, const NDIM: usize, const DOF: usize, M, E, S, Mem>(
    state: &SimStateGeneric<R, NDIM, DOF, M, E, S, Mem>,
) -> bool
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM>,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    state.geom.coords == symbi_geometry::Geometry::Cartesian && state.geom.maps.is_none()
}

/// snapshot a level's prims (+ cell/face B for mhd) into the *_old buffers
/// via the copy kernel — parallel on the host, device-side on a gpu backend
/// (no host touch on unified memory). a serial host memcpy here was a
/// measured per-root-step stall at production sizes (~0.7 gb per step at
/// n = 256); the dispatch setup only matters on demo grids where the copy
/// is microseconds anyway.
fn save_prim_old<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    lvl: &LevelData<R, NDIM, DOF, M, E, S, Mem, K>,
) where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let po = lvl
        .prim_old
        .as_ref()
        .expect("save_prim_old: prim_old allocated");
    let prim = &lvl.state.fields.prim;

    copy_field(&prim.rho, &po.rho);
    for dd in 0..DOF {
        copy_field(&prim.vel[dd], &po.vel[dd]);
    }
    if let (Some(p), Some(pp)) = (prim.pre_field(), po.pre_field()) {
        copy_field(p, pp);
    }
    if let (Some(c), Some(pc)) = (prim.chi_field(), po.chi_field()) {
        copy_field(c, pc);
    }
    if let (Some(mhd), Some(bo)) = (lvl.state.fields.mhd.as_ref(), lvl.bcell_old.as_ref()) {
        for dd in 0..DOF {
            copy_field(&mhd.bcell[dd], &bo[dd]);
        }
    }
    if let (Some(mhd), Some(bfo)) = (lvl.state.fields.mhd.as_ref(), lvl.bface_old.as_ref()) {
        for dd in 0..NDIM {
            copy_field(&mhd.bface[dd], &bfo[dd]);
        }
    }
}

/// the coarser-level proxy of each body: the same mass, softening, and motion, with accretion
/// and every magnetic coupling removed, so the finest level alone drains, dissipates, or slips
/// the field. rigid surfaces remain active because a body may cross a coarse-fine boundary and
/// must act on uncovered coarse cells.
fn gravity_only<const D: usize>(
    bodies: &symbi_ib::BodyCollection<f64, D>,
) -> symbi_ib::BodyCollection<f64, D> {
    let mut coll = bodies.clone();
    coll.visit_all_mut(|body| {
        if let symbi_ib::BodyKind::BlackHole { softening, .. } = body.kind {
            body.kind = symbi_ib::BodyKind::Gravitational { softening };
        }
        body.spec.magnetic = symbi_ib::MagneticSpec::None;
    });
    coll
}

/// the outer radius of a sink's operator support on a grid of spacing `dx`: the accretion sphere
/// plus the width over which its drain mask `chi(phi, dx)` stays nonzero in f64, or, for a
/// magnetic-slip body, the shell `4 chi (1 - chi)` of width w centered `placement` widths off the
/// surface, whichever reaches farther, plus the three-cell reach of the current gather, the edge
/// scatter, and the curl. beyond this radius every kernel of the drain and the slip writes an
/// exact zero.
fn sink_support_radius<const D: usize>(body: &symbi_ib::Body<f64, D>, dx: f64) -> f64 {
    let racc: f64 = body.accretion_radius().unwrap_or(0.0);
    let widths = symbi_ib::sdf::mask_support_widths();
    let drain = racc + widths * dx;
    let slip = match body.spec.magnetic {
        symbi_ib::MagneticSpec::Slip {
            shell_width,
            placement,
            ..
        } => racc + placement * shell_width + widths * shell_width,
        _ => racc,
    };
    drain.max(slip) + 3.0 * dx
}

/// restore the hydrodynamic conserved state over an inactive composite-grid region.
fn restore_cons_region<const D: usize, const DOF: usize, Mem: MemorySpace>(
    src: &ConsFieldsGeneric<D, DOF, Mem>,
    dst: &ConsFieldsGeneric<D, DOF, Mem>,
    region: &Domain<D>,
) {
    copy_field_region(&src.den, &dst.den, region);
    for dd in 0..DOF {
        copy_field_region(&src.mom[dd], &dst.mom[dd], region);
    }
    if let (Some(sn), Some(dn)) = (src.nrg_field(), dst.nrg_field()) {
        copy_field_region(sn, dn, region);
    }
    if let (Some(sc), Some(dc)) = (src.chi_field(), dst.chi_field()) {
        copy_field_region(sc, dc, region);
    }
}

/// whether an absolute index is inside a domain (per-axis half-open [lo, hi)).
fn domain_contains<const D: usize>(dom: &Domain<D>, idx: &[isize; D]) -> bool {
    (0..D).all(|ax| idx[ax] >= dom.spaces[ax].lo && idx[ax] < dom.spaces[ax].hi)
}

/// the coverage extended by one cell per axis, clamped to the interior — the
/// cells whose faces the restriction + emf reflux changed.
fn shell_around<const D: usize>(cov: &Domain<D>, interior: &Domain<D>) -> Domain<D> {
    Domain::new(std::array::from_fn(|ax| Space {
        name: cov.spaces[ax].name,
        lo: (cov.spaces[ax].lo - 1).max(interior.spaces[ax].lo),
        hi: (cov.spaces[ax].hi + 1).min(interior.spaces[ax].hi),
    }))
}

/// snap a physical refinement region to parent cells: absolute parent indices
/// from the shared global origin.
/// the cells of `region` on a lattice of origin `x_lo` and spacing `dx` whose local cell 0 is
/// whole-grid cell `offset`, in local indices
fn region_to_domain<const D: usize>(
    region: &RefinementRegion<D>,
    x_lo: &[f64; D],
    dx: &[f64; D],
    offset: &[isize; D],
) -> Domain<D> {
    Domain::new(std::array::from_fn(|ax| Space {
        name: axis_name(ax),
        lo: ((region.x_lo[ax] - x_lo[ax]) / dx[ax]).round() as isize - offset[ax],
        hi: ((region.x_hi[ax] - x_lo[ax]) / dx[ax]).round() as isize - offset[ax],
    }))
}

/// the origin a kernel that places cell `i` at `x_lo + (i + 0.5) dx` needs for a lattice whose
/// local cell 0 is whole-grid cell `index_offset`; the grid's own origin on an undivided grid
fn shifted_origin<const D: usize>(geom: &symbi_sim::state::PartitionGeometry<D>) -> [f64; D] {
    std::array::from_fn(|ax| {
        if geom.index_offset[ax] == 0 {
            geom.x_lo[ax]
        } else {
            geom.x_lo[ax] + geom.index_offset[ax] as f64 * geom.dx[ax]
        }
    })
}

/// the coverage must sit inside the parent interior, and on every coarse-fine
/// (non-touching) side leave enough parent cells that the prolongation's
/// deepest read — the parent of the outermost fine ghost minus the stencil
/// halfwidth — stays inside the parent allocated domain. a violation is a
/// refusal with the remedy: on a decomposed tile the parent's allocated domain
/// ends a halo width past the cut, so a patch face closer to a cut than the
/// stencil's reach beyond that halo is refused by name.
fn validate_coverage<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    coverage: &Domain<D>,
    parent: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    fine_ng: usize,
    order: ProlongOrder,
    real_on_cut: [[bool; 2]; D],
) -> Result<(), String>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let reach = (fine_ng.div_ceil(RATIO) + order.ghost_width()) as isize;
    for ax in 0..D {
        let (clo, chi) = (coverage.spaces[ax].lo, coverage.spaces[ax].hi);
        let int = &parent.geom.interior.spaces[ax];
        let alloc = &parent.geom.allocated.spaces[ax];
        if !(clo >= int.lo && chi <= int.hi && clo < chi) {
            return Err(format!(
                "coverage [{clo}, {chi}) outside parent interior [{}, {}) on axis {ax}",
                int.lo, int.hi
            ));
        }
        let ng = (int.lo - alloc.lo) as isize;
        let clearance = reach - ng;
        // the patch's real face on a cut: its fine ghosts are prolonged from the root halo
        // alone, so the stencil has to fit inside it
        if (real_on_cut[ax][0] || real_on_cut[ax][1]) && clearance > 0 {
            return Err(format!(
                "the refinement patch's real face on axis {ax} lies on a decomposition cut and its \
                 prolongation stencil reaches {clearance} beyond the root halo; give the root {reach} \
                 ghost cells or move the face off the cut"
            ));
        }
        if clo != int.lo && clo - reach < alloc.lo {
            return Err(format!(
                "the refinement patch's low face on axis {ax} sits {} parent cell(s) from the \
                 parent's boundary, and its prolongation stencil reaches {clearance} beyond the \
                 halo; move the patch face at least {clearance} parent cells from the boundary or \
                 the decomposition cut, or onto it",
                clo - int.lo
            ));
        }
        if chi != int.hi && chi + reach > alloc.hi {
            return Err(format!(
                "the refinement patch's high face on axis {ax} sits {} parent cell(s) from the \
                 parent's boundary, and its prolongation stencil reaches {clearance} beyond the \
                 halo; move the patch face at least {clearance} parent cells from the boundary or \
                 the decomposition cut, or onto it",
                int.hi - chi
            ));
        }
    }
    Ok(())
}

// =============================================================================
// decomposed hierarchy driver (refinement x decomposition)
// =============================================================================

/// the uniform cs^2 a clipped closure seed continues: a fine ghost band the parent's allocation
/// cannot supply is continued from the fine interior, which equals the prolongation only when the
/// parent's cs^2 is one finite value everywhere it is allocated. a spatially varying closure needs
/// the parent's or the neighbor's own data there; it stops here by name.
fn assert_uniform_closure<const NDIM: usize, Mem: MemorySpace>(
    cs2: &Field<f64, NDIM, Mem>,
    allocated: &Domain<NDIM>,
    level: usize,
) {
    assert!(
        Mem::IS_HOST_ACCESSIBLE,
        "level {level}: the clipped closure seed reads the parent's cs^2 on the host"
    );
    symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
    let view = cs2.view();
    let first = *view.at(std::array::from_fn(|ax| allocated.spaces[ax].lo));
    let varying = allocated.iter().find(|c| {
        let v = *view.at(*c);
        !v.is_finite() || v.to_bits() != first.to_bits()
    });
    if let Some(cell) = varying {
        panic!(
            "level {level}: a refined patch runs to the edge of its parent's allocation, where the \
             closure seed continues cs^2 from the fine interior, but the parent's cs^2 is not one \
             finite value ({} at {cell:?} against {first}); a spatially varying closure needs the \
             parent's or the neighbor's data at that edge",
            *view.at(cell)
        );
    }
}

/// the halo topology of `level` over the tiles in `order`: the root closes the axes its tile grid
/// wraps; a refined level lies inside the domain, clear of every seam.
fn level_topology<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    order: &[usize],
) -> Topology<NDIM>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    match (level, order.first()) {
        (0, Some(&i)) => Topology::wrapping(tiles[i].root_wraps),
        (1, Some(&i)) => Topology::wrapping(tiles[i].fine_wraps),
        _ => Topology::open(),
    }
}

/// the conserved ghost cells of a level saved for the seed of the level above it
pub struct SeedGhosts(Vec<Vec<f64>>);

/// the conserved fields of a store, the dye included
fn conserved_fields<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    store: &FieldStore<NDIM, DOF, Mem, f64>,
) -> Vec<&Field<f64, NDIM, Mem>> {
    let cons = &store.fields.cons;
    let mut fields: Vec<&Field<f64, NDIM, Mem>> = vec![&cons.den];
    fields.extend(cons.mom.iter());
    fields.extend(cons.nrg_field());
    fields.extend(cons.chi_field());
    fields
}

/// the values of every conserved ghost cell of a store, field by field in allocation order
fn conserved_ghosts<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(store: &FieldStore<NDIM, DOF, Mem, f64>) -> Vec<Vec<f64>> {
    let interior = &store.geom.interior;
    conserved_fields(store)
        .into_iter()
        .map(|f| {
            let view = f.view();
            store.geom.allocated.iter().filter(|c| !interior.contains(*c)).map(|c| *view.at(c)).collect()
        })
        .collect()
}

/// write back the conserved ghost cells `conserved_ghosts` read
fn restore_conserved_ghosts<const NDIM: usize, const DOF: usize, Mem: MemorySpace>(
    store: &FieldStore<NDIM, DOF, Mem, f64>,
    saved: &[Vec<f64>],
) {
    let interior = &store.geom.interior;
    for (f, values) in conserved_fields(store).into_iter().zip(saved) {
        let view = f.view_mut();
        for (c, v) in store.geom.allocated.iter().filter(|c| !interior.contains(*c)).zip(values) {
            view.set(c, *v);
        }
    }
}

/// the sub-grid of tiles that carry one refined level. with one refined box per level the
/// refined tiles form a contiguous rectangle; `order[k]` is the tile index at fine-flatten position
/// `k`, and `devices[k]`/`counts` give that sub-grid's device map + shape for `exchange_grid`. when
/// the patch is tile-local the sub-grid is 1x1, so the fine exchange is a no-op; when
/// the patch spans cuts the sub-grid has internal cuts and the fine halos are exchanged.
pub struct FineSubgrid<const NDIM: usize> {
    pub counts: [usize; NDIM],
    pub order: Vec<usize>,
    pub devices: Vec<i32>,
}

/// seed every tile's refined levels from their parents, decomposition-aware: level by level, the
/// parent level's conserved cut halos are exchanged first, then each tile prolongs.
///
/// a refined patch spanning a cut needs this exchange on top of the per-tile
/// [`Hierarchy::seed_fine_from_coarse`]: the seed prolongs conserved components, its stencil at
/// the coverage edge reads the root's cut ghosts, and this exchange is the sole filler of
/// conserved ghosts -- the evolve loop's exchange moves primitives, on the (correct there)
/// invariant that the flux stages read primitive ghosts alone. seeded per tile, the cut-side fine
/// interior would be prolonged from each tile's standalone boundary fill in place of its
/// neighbor's data, and the decomposed hierarchy would differ from the monolithic one at step
/// zero -- measured 7.2e-3 on a smooth bump centered on the cut, decaying as the flow smooths it,
/// and growing with the prolongation order (a higher-degree interpolant weights the ghost cells
/// more heavily). tile-local patches stay clear of every cut and behave identically either way.
pub fn seed_decomposed_fine_from_coarse<
    R,
    const NDIM: usize,
    const DOF: usize,
    M,
    E,
    S,
    Mem,
    K,
    T,
>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
) -> symbi_xpu::Result<usize>
where
    R: Regime<f64, NDIM, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF> + Copy,
    M: Metric<f64, NDIM> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
    <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
{
    let depth = tiles.iter().map(|t| t.levels.len()).max().unwrap_or(1);
    let mut reseeded = 0usize;
    for level in 1..depth {
        // the parent level's cut halos, over the tiles carrying it: the conserved fields the seed
        // prolongs, and the primitives, the cell field and the transverse faces the magnetized
        // seed prolongs across the cut
        let parents: Vec<usize> = (0..tiles.len()).filter(|&i| tiles[i].levels.len() >= level).collect();
        let Some(grid) = fine_subgrid_of_level(tiles, counts, devices, level - 1) else { break };
        debug_assert_eq!(grid.order.len(), parents.len());
        // the parent's ghosts filled before its cut halos cross, so the exchange carries the cells
        // where a face's ghosts meet a cut
        let saved: Vec<(usize, SeedGhosts)> = grid
            .order
            .iter()
            .enumerate()
            .map(|(k, &i)| (i, symbi_xpu::with_device(grid.devices[k], || tiles[i].fill_seed_ghosts(level - 1))))
            .collect();
        drain_devices::<Mem>(&grid.devices);
        {
            let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> =
                grid.order.iter().map(|&i| &*tiles[i].levels[level - 1].state).collect();
            let schedule = schedule_of_tiles(&states, grid.counts, &level_topology(tiles, level - 1, &grid.order));
            symbi_sim::decomp::exchange_grid_cons(&states, &schedule, &grid.devices, transport);
            symbi_sim::decomp::exchange_grid(&states, &schedule, &grid.devices, transport);
        }
        for (k, &i) in grid.order.iter().enumerate() {
            if tiles[i].levels.len() > level {
                reseeded += symbi_xpu::with_device(grid.devices[k], || tiles[i].seed_level_from_filled_parent(level))?;
            }
        }
        for (k, (i, ghosts)) in saved.into_iter().enumerate() {
            symbi_xpu::with_device(grid.devices[k], || tiles[i].restore_seed_ghosts(level - 1, ghosts));
        }
    }
    Ok(reseeded)
}

/// declare a stationary target on every tile's hierarchy, as [`Hierarchy::with_equilibrium`]
/// declares it on the single hierarchy: every level of every tile is seeded with the target, the
/// fine target restricted into the covered coarse cells, and the levels primed coarsest first with
/// each level's cut halos exchanged before its children prolong from them. each level's anchor
/// holds the primed target with its cut halos; its interface flux and its imbalance are read off
/// one forward-euler stage at the level's cfl step, the minimum over the tiles carrying it, with
/// the troubled cells marked across the cuts. every tile's state is restored afterwards.
pub fn with_equilibrium_decomposed<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    target: impl Fn([f64; NDIM]) -> <R as Regime<f64, DOF>>::Prim,
) -> symbi_xpu::Result<()>
where
    R: Regime<f64, NDIM, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF> + Copy,
    M: Metric<f64, NDIM> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
{
    let fail = |detail: String| symbi_xpu::XpuError { operation: "stationary target", code: -1, detail };
    let depth = tiles.iter().map(|t| t.levels.len()).max().unwrap_or(1);
    let saved = tiles
        .iter()
        .map(|t| t.levels.iter().map(|l| save_gas_state(&l.state)).collect::<symbi_xpu::Result<Vec<_>>>())
        .collect::<symbi_xpu::Result<Vec<_>>>()?;
    drain_devices::<Mem>(devices);
    for (k, tile) in tiles.iter().enumerate() {
        tile.seed_declared_target(&target).map_err(|e| fail(format!("tile {k} {e}")))?;
    }
    // the prime, coarsest first: each level primed, then its own cut halos exchanged before its
    // children prolong from them
    for ll in 0..depth {
        let Some(grid) = fine_subgrid_of_level(tiles, counts, devices, ll) else { break };
        for (k, &i) in grid.order.iter().enumerate() {
            symbi_xpu::with_device(grid.devices[k], || tiles[i].prime_target_level(ll));
        }
        drain_devices::<Mem>(&grid.devices);
        {
            let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> =
                grid.order.iter().map(|&i| &*tiles[i].levels[ll].state).collect();
            let schedule = schedule_of_tiles(&states, grid.counts, &level_topology(tiles, ll, &grid.order));
            symbi_sim::decomp::exchange_grid_cons(&states, &schedule, &grid.devices, transport);
            exchange_grid(&states, &schedule, &grid.devices, transport);
        }
        for (k, &i) in grid.order.iter().enumerate() {
            symbi_xpu::with_device(grid.devices[k], || {
                tiles[i].fill_physical_ghosts(ll);
            });
            let err = scan_c2p_errors(&tiles[i].levels[ll].state);
            if err.is_err() {
                return Err(fail(format!("tile {i} level {ll}: the target's recovery failed ({err})")));
            }
        }
    }
    let anchors = tiles
        .iter()
        .map(|t| t.levels.iter().map(|l| save_gas_state(&l.state)).collect::<symbi_xpu::Result<Vec<_>>>())
        .collect::<symbi_xpu::Result<Vec<_>>>()?;
    let mut flux_eq: Vec<Vec<Option<EquilibriumFlux<NDIM, DOF, Mem>>>> =
        tiles.iter().map(|t| t.levels.iter().map(|_| None).collect()).collect();
    let mut residual_eq: Vec<Vec<Option<ConsFieldsGeneric<NDIM, DOF, Mem>>>> =
        tiles.iter().map(|t| t.levels.iter().map(|_| None).collect()).collect();
    for ll in 0..depth {
        let Some(grid) = fine_subgrid_of_level(tiles, counts, devices, ll) else { break };
        let mut probe_dt = f64::INFINITY;
        for (k, &i) in grid.order.iter().enumerate() {
            let level = &tiles[i].levels[ll];
            symbi_xpu::with_device(grid.devices[k], || {
                level.kernels.wave_speeds(&level.state);
                for dd in 0..NDIM {
                    level.kernels.flux(&level.state, dd);
                }
            });
            flux_eq[i][ll] = Some(snapshot_flux(&level.state)?);
            if tiles[i].level_evolves_cells(ll) {
                probe_dt = probe_dt.min(symbi_xpu::with_device(grid.devices[k], || level.kernels.cfl(&level.state)));
            }
        }
        // the probe's step per level, the timestep the single hierarchy's probe takes
        eprintln!("SIMBI equilibrium probe: level={ll} dt_bits={:016x}", probe_dt.to_bits());
        if !(probe_dt.is_finite() && probe_dt > 0.0) {
            return Err(fail(format!(
                "the stationary target has no finite cfl step ({probe_dt:.3e}) on level {ll}, so it cannot be advanced to read off its imbalance"
            )));
        }
        {
            let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> =
                grid.order.iter().map(|&i| &*tiles[i].levels[ll].state).collect();
            let kernels: Vec<&K> = grid.order.iter().map(|&i| &tiles[i].levels[ll].kernels).collect();
            let schedule = schedule_of_tiles(&states, grid.counts, &level_topology(tiles, ll, &grid.order));
            symbi_sim::decomp::probe_stage(&states, &kernels, &schedule, &grid.devices, transport, probe_dt)
                .map_err(|e| fail(format!("level {ll}: the stationary target is not a state the scheme can carry: {e}")))?;
        }
        for &i in &grid.order {
            let level = &tiles[i].levels[ll];
            let err = scan_c2p_errors(&level.state);
            if err.is_err() {
                return Err(fail(format!(
                    "advancing the stationary target by one cfl step left an unrecoverable state on tile {i} level {ll} ({err})"
                )));
            }
            residual_eq[i][ll] =
                Some(imbalance_from_stage(anchors[i][ll].cons(), &level.state.fields.cons, &level.state.geom.allocated, probe_dt)?);
        }
    }
    for ((tile, saved), ((anchors, fluxes), residuals)) in
        tiles.iter_mut().zip(&saved).zip(anchors.into_iter().zip(flux_eq).zip(residual_eq))
    {
        for (level, state) in tile.levels.iter().zip(saved) {
            restore_gas_state(&level.state, state);
        }
        tile.install_declared_target(anchors, fluxes, residuals);
    }
    Ok(())
}

/// the sub-grid of the tiles carrying the first refined level; None if no tile is refined.
/// asserts the refined tiles fill a rectangle (the single-box invariant). pub so the python
/// decomposed-refinement loop can gather a level with the same tile order + counts.
pub fn fine_subgrid<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
) -> Option<FineSubgrid<NDIM>>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    fine_subgrid_of_level(tiles, counts, devices, 1)
}

/// the sub-grid of the tiles carrying refined level `level`; None if no tile carries it
pub fn fine_subgrid_of_level<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    level: usize,
) -> Option<FineSubgrid<NDIM>>
where
    R: Regime<f64, NDIM>,
    M: Metric<f64, NDIM> + Copy,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy>,
    S: ExecutionSpace,
    Mem: MemorySpace,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let refined: Vec<usize> = (0..tiles.len())
        .filter(|&i| tiles[i].levels.len() > level)
        .collect();
    if refined.is_empty() {
        return None;
    }
    // the tile-coord bounding box of the refined tiles -> the fine sub-grid shape.
    let mut lo = [usize::MAX; NDIM];
    let mut hi = [0usize; NDIM];
    for &i in &refined {
        let tc = unflatten(i, counts);
        for ax in 0..NDIM {
            lo[ax] = lo[ax].min(tc[ax]);
            hi[ax] = hi[ax].max(tc[ax]);
        }
    }
    let sub_counts: [usize; NDIM] = std::array::from_fn(|ax| hi[ax] - lo[ax] + 1);
    let nsub: usize = sub_counts.iter().product();
    // place each refined tile at its position in the sub-grid's flatten order.
    let mut order = vec![usize::MAX; nsub];
    for &i in &refined {
        let tc = unflatten(i, counts);
        let rc: [usize; NDIM] = std::array::from_fn(|ax| tc[ax] - lo[ax]);
        order[flatten(rc, sub_counts)] = i;
    }
    assert!(
        order.iter().all(|&i| i != usize::MAX),
        "refinement x decomposition: refined tiles do not form a rectangle (a single refined box per level)"
    );
    let sub_devices: Vec<i32> = order.iter().map(|&i| devices[i]).collect();
    Some(FineSubgrid {
        counts: sub_counts,
        order,
        devices: sub_devices,
    })
}

pub fn seed_decomposed_hierarchy_tracers<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    global: &Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    count: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    seed_tiles_from_hierarchy(global, tiles, count, true);
}

/// `seed_decomposed_hierarchy_tracers` on the tiles one worker holds: every tile's levels take the
/// population's metadata and the records whose cells the held tiles own, the rest held elsewhere
pub fn seed_held_hierarchy_tracers<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    global: &Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    count: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    seed_tiles_from_hierarchy(global, tiles, count, false);
}

fn seed_tiles_from_hierarchy<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    global: &Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    count: usize,
    every_owner_held: bool,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let seeded = global.seed_mass_tracers(count);
    for tile in tiles.iter_mut() {
        for level in &mut tile.levels {
            level.state.tracers = Some(symbi_sim::tracers::TracerSet {
                weight: seeded.weight,
                run_seed: seeded.run_seed,
                next_id: seeded.next_id,
                ..Default::default()
            });
        }
    }
    for ii in 0..seeded.len() {
        let owner = seeded.owner[ii];
        let Some(destination) = tiles.iter().position(|tile| tile.tracer_owner_is_active(owner)) else {
            assert!(!every_owner_held, "composite tracer seed has no decomposed owner");
            continue;
        };
        let (level, coord) = tiles[destination]
            .tracer_cell(owner)
            .expect("decomposed tracer owner must resolve");
        let x = tiles[destination].levels[level].state.geom.centroid(coord);
        let tracers = tiles[destination].levels[level]
            .state
            .tracers
            .as_mut()
            .unwrap();
        tracers.x.push(x);
        tracers.id.push(seeded.id[ii]);
        tracers.cohort.push(seeded.cohort[ii]);
        tracers.flags.push(seeded.flags[ii]);
        tracers.owner.push(owner);
        tracers.step_owner.push(seeded.step_owner[ii]);
        tracers.step_flags.push(seeded.step_flags[ii]);
    }
}

pub fn gather_decomposed_hierarchy_tracers<
    R,
    const NDIM: usize,
    const DOF: usize,
    M,
    E,
    S,
    Mem,
    K,
>(
    global: &mut Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    for level in 0..global.levels.len() {
        let mut records = Vec::new();
        let mut metadata = None;
        for tile in tiles {
            let Some(tracers) = tile
                .levels
                .get(level)
                .and_then(|data| data.state.tracers.as_ref())
            else {
                continue;
            };
            metadata.get_or_insert((
                tracers.weight,
                tracers.run_seed,
                tracers.next_id,
                tracers.injection_remainder,
            ));
            records.extend((0..tracers.len()).map(|ii| {
                (
                    tracers.id[ii],
                    tracers.x[ii],
                    tracers.cohort[ii],
                    tracers.flags[ii],
                    tracers.owner[ii],
                    tracers.step_owner[ii],
                    tracers.step_flags[ii],
                )
            }));
        }
        records.sort_unstable_by_key(|record| record.0);
        let (weight, run_seed, next_id, injection_remainder) = metadata.unwrap_or_default();
        let mut gathered = symbi_sim::tracers::TracerSet {
            weight,
            run_seed,
            next_id,
            injection_remainder,
            ..Default::default()
        };
        for (id, x, cohort, flags, owner, step_owner, step_flags) in records {
            gathered.id.push(id);
            gathered.x.push(x);
            gathered.cohort.push(cohort);
            gathered.flags.push(flags);
            gathered.owner.push(owner);
            gathered.step_owner.push(step_owner);
            gathered.step_flags.push(step_flags);
        }
        global.levels[level].state.tracers = Some(gathered);
    }
}

/// exchange one level's halos across an ordered set of tiles (a sub-grid of shape `counts`), then
/// per-tile `ghost_fill` at that level. `order[k]` is the tile index at flatten position `k`;
/// `devices[k]` is order-aligned. cut faces are `CoarseFine` (so `ghost_fill` leaves them for the
/// exchange); the sub-grid's outer faces are the patch's coarse-fine boundary (prolong-filled) or a
/// physical boundary, neither of which `exchange_grid` touches (it moves only internal cuts).
/// one stage of `level` across the tiles of a decomposed level: every tile marks its troubled
/// cells, the marks cross the level's cuts into the neighbors' ghost layer next to the
/// interior, and every tile then corrects with the knowledge of trouble against its cuts, so
/// the two tiles sharing a face take the same corrected flux there. returns true on a stage
/// retry request from any tile.
#[allow(clippy::too_many_arguments)]
fn decomposed_level_stage<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    order: &[usize],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    ii: usize,
    dt: f64,
    alpha0: f64,
) -> bool
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    let mut level_stage =
        TiledLevelStage { tiles, level, order, counts, devices, transport, ii, dt, alpha0, receipt_starts: vec![0; order.len()] };
    let Ok::<_, std::convert::Infallible>(agreed) = symbi_sim::collective_stage::collective_stage(&mut level_stage);
    agreed.retry
}

/// one stage of one level across the in-process tiles holding it: each tile folds on its device
/// through the hierarchy's own stage halves, the troubled flags cross the level's cuts by the
/// in-process exchange, and every agreement is the tiles' own
struct TiledLevelStage<'a, R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    tiles: &'a mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    order: &'a [usize],
    counts: [usize; NDIM],
    devices: &'a [i32],
    transport: &'a T,
    ii: usize,
    dt: f64,
    alpha0: f64,
    /// each tile's tracer interface ledger length as its stage opened
    receipt_starts: Vec<usize>,
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T> StageCollective for TiledLevelStage<'_, R, NDIM, DOF, M, E, S, Mem, K, T>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    type Error = std::convert::Infallible;

    fn tiles(&self) -> usize {
        self.order.len()
    }

    fn mark(&mut self, k: usize) -> Result<u64, Self::Error> {
        let (tile, level, ii, dt, alpha0) = (&mut self.tiles[self.order[k]], self.level, self.ii, self.dt, self.alpha0);
        let mark = symbi_xpu::with_device(self.devices[k], || tile.level_stage_through_mark(level, ii, dt, alpha0));
        self.receipt_starts[k] = mark.receipt_start;
        Ok(mark.troubled)
    }

    fn any(&mut self, local: bool) -> Result<bool, Self::Error> {
        Ok(local)
    }

    fn exchange_trouble(&mut self) -> Result<Vec<bool>, Self::Error> {
        let (tiles, level, order, devices) = (&*self.tiles, self.level, self.order, self.devices);
        drain_devices::<Mem>(devices);
        let received = {
            let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> = order.iter().map(|&i| &*tiles[i].levels[level].state).collect();
            let schedule = schedule_of_tiles(&states, self.counts, &level_topology(tiles, level, order));
            symbi_sim::decomp::exchange_trouble(&states, &schedule, devices, self.transport)
        };
        drain_devices::<Mem>(devices);
        Ok(order
            .iter()
            .enumerate()
            .map(|(k, &i)| {
                let l = &tiles[i].levels[level];
                symbi_xpu::with_device(devices[k], || received[k].iter().any(|region| l.kernels.fofc_flags_in(&l.state, region) > 0))
            })
            .collect())
    }

    fn correct(&mut self, k: usize, cut_trouble: bool) -> Result<bool, Self::Error> {
        let (tile, level, ii, dt, alpha0, start) =
            (&mut self.tiles[self.order[k]], self.level, self.ii, self.dt, self.alpha0, self.receipt_starts[k]);
        Ok(symbi_xpu::with_device(self.devices[k], || tile.level_stage_from_correction(level, ii, dt, alpha0, cut_trouble, start)))
    }

    fn freeze_streak(&self, k: usize) -> u32 {
        self.tiles[self.order[k]].levels[self.level].kernels.freeze_streak()
    }

    fn set_freeze_streak(&mut self, k: usize, streak: u32) {
        self.tiles[self.order[k]].levels[self.level].kernels.set_freeze_streak(streak);
    }
}

/// one stage of one level of a single hierarchy: the one tile marks and corrects, with no cut to
/// carry its flags and every agreement its own
struct OneTileLevelStage<'h, R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    hier: &'h mut Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    level: usize,
    ii: usize,
    dt: f64,
    alpha0: f64,
    receipt_start: usize,
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K> StageCollective for OneTileLevelStage<'_, R, NDIM, DOF, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    type Error = std::convert::Infallible;

    fn tiles(&self) -> usize {
        1
    }

    fn mark(&mut self, _k: usize) -> Result<u64, Self::Error> {
        let mark = self.hier.level_stage_through_mark(self.level, self.ii, self.dt, self.alpha0);
        self.receipt_start = mark.receipt_start;
        Ok(mark.troubled)
    }

    fn any(&mut self, local: bool) -> Result<bool, Self::Error> {
        Ok(local)
    }

    fn exchange_trouble(&mut self) -> Result<Vec<bool>, Self::Error> {
        Ok(vec![false])
    }

    fn correct(&mut self, _k: usize, cut_trouble: bool) -> Result<bool, Self::Error> {
        Ok(self.hier.level_stage_from_correction(self.level, self.ii, self.dt, self.alpha0, cut_trouble, self.receipt_start))
    }

    fn freeze_streak(&self, _k: usize) -> u32 {
        self.hier.levels[self.level].kernels.freeze_streak()
    }

    fn set_freeze_streak(&mut self, _k: usize, streak: u32) {
        self.hier.levels[self.level].kernels.set_freeze_streak(streak);
    }
}

/// what the first half of a level stage leaves for the second: the troubled interior cell
/// count and the tracer interface ledger's length as the stage opened
#[derive(Clone, Copy, Debug)]
pub struct StageMark {
    pub troubled: u64,
    pub receipt_start: usize,
}

fn exchange_level_halos<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    order: &[usize],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    drain_devices::<Mem>(devices);
    {
        let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> = order
            .iter()
            .map(|&i| &*tiles[i].levels[level].state)
            .collect();
        let schedule = schedule_of_tiles(&states, counts, &level_topology(tiles, level, order));
        exchange_grid(&states, &schedule, devices, transport);
    }
    for (k, &i) in order.iter().enumerate() {
        symbi_xpu::with_device(devices[k], || {
            tiles[i].fill_physical_ghosts(level)
        });
    }
}

/// the register units of one fine stage across the root cuts: each root tile's contributing face
/// (a patch face on a cut) lands in the inbox of the same patch face on the root tile across the
/// cut, which holds the coarse cells its reflux corrects; every contributing face is then cleared
/// for the next stage and every landed unit folded into its receiving face, in the order the
/// single hierarchy's register accumulates. root tile `i` sits at flat position `i` of `counts`
fn exchange_flux_contributions<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    if !tiles.iter().any(Hierarchy::transports_registers) {
        return;
    }
    drain_devices::<Mem>(devices);
    for (i, src) in tiles.iter().enumerate() {
        let Some(reg) = src.flux_registers.first() else { continue };
        for (idx, faces, src_region) in reg.contributing() {
            let (ax, side) = (idx / 2, idx % 2);
            let mut tc = unflatten(i, counts);
            tc[ax] = match side {
                0 => tc[ax].checked_sub(1),
                _ => Some(tc[ax] + 1).filter(|&c| c < counts[ax]),
            }
            .unwrap_or_else(|| {
                panic!(
                    "root tile {i}'s patch face {idx} lies on the periodic seam of axis {ax}: register units cross \
                     the root cuts inside the domain alone"
                )
            });
            let j = flatten(tc, counts);
            let (_, units, dst_region) = tiles[j]
                .flux_registers
                .first()
                .and_then(|r| r.inboxes().into_iter().find(|(k, _, _)| *k == idx))
                .unwrap_or_else(|| panic!("root tile {j} across the cut holds no receiving face {idx} for tile {i}'s patch face"));
            for (face, unit) in faces.iter().zip(&units) {
                transport.copy_region(face, src_region, unit, dst_region, devices[i], devices[j]);
            }
        }
    }
    drain_devices::<Mem>(devices);
    for (i, tile) in tiles.iter().enumerate() {
        if let Some(reg) = tile.flux_registers.first() {
            symbi_xpu::with_device(devices[i], || {
                reg.zero_contributions();
                reg.fold_inboxes();
            });
        }
    }
}

fn migrate_level_tracers<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let mut migrating = Vec::new();
    for source in 0..tiles.len() {
        if level >= tiles[source].levels.len() {
            continue;
        }
        let destinations: std::collections::BTreeMap<u64, usize> = tiles[source].levels[level]
            .state
            .tracers
            .as_ref()
            .into_iter()
            .flat_map(|tracers| {
                tracers
                    .id
                    .iter()
                    .copied()
                    .zip(tracers.owner.iter().copied())
            })
            .filter_map(|(id, owner)| {
                tiles
                    .iter()
                    .enumerate()
                    .find_map(|(index, tile)| {
                        (index != source
                            && tile
                                .tracer_cell(owner)
                                .is_some_and(|(owner_level, _)| owner_level == level))
                        .then_some(index)
                    })
                    .map(|destination| (id, destination))
            })
            .collect();
        let Some(tracers) = tiles[source].levels[level].state.tracers.as_mut() else {
            continue;
        };
        let mut ii = 0usize;
        while ii < tracers.len() {
            let Some(&destination) = destinations.get(&tracers.id[ii]) else {
                ii += 1;
                continue;
            };
            tracers.x.swap_remove(ii);
            migrating.push((
                destination,
                tracers.id.swap_remove(ii),
                tracers.cohort.swap_remove(ii),
                tracers.flags.swap_remove(ii),
                tracers.owner.swap_remove(ii),
                tracers.step_owner.swap_remove(ii),
                tracers.step_flags.swap_remove(ii),
            ));
        }
    }
    for (destination, id, cohort, flags, owner, step_owner, step_flags) in migrating {
        let (_, coord) = tiles[destination]
            .tracer_cell(owner)
            .expect("tracer destination tile owns the addressed cell");
        let x = tiles[destination].levels[level].state.geom.centroid(coord);
        let tracers = tiles[destination].levels[level]
            .state
            .tracers
            .as_mut()
            .expect("every decomposed hierarchy level carries tracers");
        tracers.x.push(x);
        tracers.id.push(id);
        tracers.cohort.push(cohort);
        tracers.flags.push(flags);
        tracers.owner.push(owner);
        tracers.step_owner.push(step_owner);
        tracers.step_flags.push(step_flags);
    }
}

/// the ancestry blend of level `level`'s stage `ii` over every tile's records of the level, as the
/// single hierarchy draws it over the level's one store: the stage's convex weight and the level's
/// iteration key the draw, and a record not chosen returns to its step-entry owner
fn blend_level_tracers<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    ii: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let Some(first) = tiles.iter().filter_map(|t| t.levels.get(level)).find(|l| l.state.tracers.is_some()) else {
        return;
    };
    let (a0, ac) = first.state.timestepping.stages()[ii];
    if a0 == 0.0 && ac == 1.0 {
        return;
    }
    let key = symbi_sim::mass_transport::SamplingKey {
        run_seed: first.state.tracers.as_ref().unwrap().run_seed,
        epoch: first.state.iteration.wrapping_mul(4).wrapping_add(ii as u64),
    };
    let ids: Vec<u64> = tiles
        .iter()
        .filter_map(|t| t.levels.get(level).and_then(|l| l.state.tracers.as_ref()))
        .flat_map(|tracers| tracers.id.iter().copied())
        .collect();
    let selections: std::collections::BTreeMap<u64, bool> = symbi_sim::mass_transport::sample_convex_blend(&ids, ac, key)
        .unwrap_or_else(|detail| panic!("tracer ancestry blend: {detail}"))
        .into_iter()
        .collect();
    for tile in tiles.iter_mut() {
        let Some(tracers) = tile.levels.get_mut(level).and_then(|l| l.state.tracers.as_mut()) else {
            continue;
        };
        for kk in 0..tracers.len() {
            if !selections[&tracers.id[kk]] {
                tracers.owner[kk] = tracers.step_owner[kk];
                tracers.flags[kk] = tracers.step_flags[kk];
            }
        }
    }
}

/// the stage's tracer exchange on level `level` of the tiles: the leavers cross to the tiles
/// holding their cells, the ancestry blend ranks the level's ids across every tile, and the records
/// it returns to their step-entry cells cross back
fn exchange_level_tracers<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    ii: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    migrate_level_tracers(tiles, level);
    blend_level_tracers(tiles, level, ii);
    migrate_level_tracers(tiles, level);
    // every record owned by a cell of the level sits at that cell's centre, as the store's own
    // transport leaves it after its blend, and the stage's interface event follows, as it follows
    // the transport on the single hierarchy
    for tile in tiles.iter_mut() {
        let Some(tracers) = tile.levels.get(level).and_then(|l| l.state.tracers.as_ref()) else {
            continue;
        };
        let cells: Vec<Option<[isize; NDIM]>> = tracers
            .owner
            .iter()
            .map(|&owner| tile.tracer_cell(owner).and_then(|(lev, coord)| (lev == level).then_some(coord)))
            .collect();
        let state = &tile.levels[level].state;
        let geometry = state.geom.block_geometry(state.physics.metric);
        let centers: Vec<Option<[f64; NDIM]>> = cells
            .iter()
            .map(|cell| cell.map(|coord| {
                let center = geometry.centroid(coord);
                std::array::from_fn(|dd| center[dd])
            }))
            .collect();
        let tracers = tile.levels[level].state.tracers.as_mut().unwrap();
        for (kk, center) in centers.into_iter().enumerate() {
            if let Some(center) = center {
                tracers.x[kk] = center;
            }
        }
    }
    for tile in tiles.iter_mut() {
        tile.apply_pending_interface(level);
    }
}

/// the boundary and source injection of level `level` over every tile, spawned once from the merged
/// ledger as the single hierarchy spawns its own: ids from the run's next id, the injection
/// remainder carried, the draw keyed by the level's iteration, and each record placed at its
/// whole-grid cell's center through the tile holding the cell (every tile places a whole-grid cell
/// bit for bit) and stored there. the run's next id and the remainder ride every level of every
/// tile, as they ride every level of the single hierarchy
fn spawn_decomposed_injections<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    order: &[usize],
    level: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let Some(&holder) = order.iter().find(|&&i| tiles[i].levels.get(level).is_some_and(|l| l.state.tracers.is_some())) else {
        return;
    };
    let next_id = tiles
        .iter()
        .flat_map(|t| &t.levels)
        .filter_map(|l| l.state.tracers.as_ref())
        .map(|tracers| tracers.next_id)
        .max()
        .unwrap_or(0);
    let remainder = tiles
        .iter()
        .flat_map(|t| &t.levels)
        .filter_map(|l| l.state.tracers.as_ref())
        .map(|tracers| tracers.injection_remainder)
        .fold(0.0_f64, f64::max);
    let mut ledger = std::collections::BTreeMap::new();
    for &i in order {
        for (destination, mass) in std::mem::take(&mut tiles[i].injection_ledger) {
            *ledger.entry(destination).or_insert(0.0) += mass;
        }
    }
    let template = tiles[holder].levels[level].state.tracers.as_ref().unwrap();
    let mut spawned = symbi_sim::tracers::TracerSet {
        weight: template.weight,
        run_seed: template.run_seed,
        next_id,
        injection_remainder: remainder,
        ..Default::default()
    };
    let key = symbi_sim::mass_transport::SamplingKey {
        run_seed: spawned.run_seed,
        epoch: tiles[holder].levels[level].state.iteration | (1 << 62),
    };
    let owner_of = |owner: symbi_sim::mass_transport::ContainerId| -> Option<(usize, [isize; NDIM])> {
        order.iter().find_map(|&i| {
            let (lev, coord) = tiles[i].tracer_cell(owner)?;
            (lev == level).then_some((i, coord))
        })
    };
    symbi_sim::tracers::spawn_injected_tracers(
        &mut spawned,
        ledger.into_iter().map(|(destination, mass)| symbi_sim::mass_transport::MassTransfer { destination, mass }),
        |owner| {
            let (i, coord) = owner_of(owner).expect("a tile of the level holds every injection destination");
            let state = &tiles[i].levels[level].state;
            let center = state.geom.block_geometry(state.physics.metric).centroid(coord);
            std::array::from_fn(|dd| center[dd])
        },
        key,
    )
    .unwrap_or_else(|detail| panic!("decomposed tracer injection: {detail}"));
    let holders: Vec<usize> = (0..spawned.len())
        .map(|kk| owner_of(spawned.owner[kk]).expect("a tile of the level holds every spawned record").0)
        .collect();
    for (kk, &i) in holders.iter().enumerate() {
        let tracers = tiles[i].levels[level].state.tracers.as_mut().expect("a tile of a traced level carries tracers");
        tracers.x.push(spawned.x[kk]);
        tracers.id.push(spawned.id[kk]);
        tracers.cohort.push(spawned.cohort[kk]);
        tracers.flags.push(spawned.flags[kk]);
        tracers.owner.push(spawned.owner[kk]);
        tracers.step_owner.push(spawned.step_owner[kk]);
        tracers.step_flags.push(spawned.step_flags[kk]);
    }
    for tile in tiles.iter_mut() {
        for level in &mut tile.levels {
            if let Some(tracers) = level.state.tracers.as_mut() {
                tracers.next_id = spawned.next_id;
                tracers.injection_remainder = spawned.injection_remainder;
            }
        }
    }
}

/// the whole domain's extent and the boundary each axis presents to a continuous tracer: the root
/// tiles' bounds joined, periodic on an axis the root closes into a ring, elsewhere the condition
/// the outermost root tile carries
fn whole_continuous_domain<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
) -> ([(f64, f64); NDIM], Boundaries<NDIM>)
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let bounds_of = |t: &Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>| symbi_sim::tracers::partition_physical_bounds(&t.levels[0].state.geom);
    let bounds: [(f64, f64); NDIM] = std::array::from_fn(|dd| {
        tiles.iter().map(|t| bounds_of(t)[dd]).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (l, h)| (lo.min(l), hi.max(h)))
    });
    let boundaries = Boundaries::per_axis(std::array::from_fn(|dd| {
        if tiles[0].root_wraps[dd] {
            return [BoundaryType::Periodic; 2];
        }
        let lo = tiles.iter().min_by(|a, b| bounds_of(a)[dd].0.total_cmp(&bounds_of(b)[dd].0)).unwrap().levels[0].state.boundaries.lo(dd);
        let hi = tiles.iter().max_by(|a, b| bounds_of(a)[dd].1.total_cmp(&bounds_of(b)[dd].1)).unwrap().levels[0].state.boundaries.hi(dd);
        [lo, hi]
    }));
    (bounds, boundaries)
}

/// level `level`'s materialized ito coefficients across its cuts, after each tile filled its
/// physical and coarse-fine ghosts
fn exchange_level_ito<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    order: &[usize],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    drain_devices::<Mem>(devices);
    let states: Vec<&FieldStore<NDIM, DOF, Mem, f64>> = order.iter().map(|&i| &*tiles[i].levels[level].state).collect();
    let schedule = schedule_of_tiles(&states, counts, &level_topology(tiles, level, order));
    symbi_sim::decomp::exchange_ito_coefficients(&states, &schedule, devices, transport);
    drain_devices::<Mem>(devices);
}

/// the tile holding level `level`'s cell at `position`, with the cell's local index there
fn level_cell_holder<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
    position: [f64; NDIM],
) -> Option<(usize, [isize; NDIM])>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    tiles.iter().enumerate().find_map(|(i, t)| {
        let geom = &t.levels.get(level)?.state.geom;
        let bounds = symbi_sim::tracers::partition_physical_bounds(geom);
        if (0..NDIM).any(|dd| position[dd] < bounds[dd].0 || position[dd] >= bounds[dd].1) {
            return None;
        }
        let coord: [isize; NDIM] = std::array::from_fn(|dd| match geom.maps {
            Some(maps) => maps[dd].index_at(position[dd]) - geom.index_offset[dd],
            None => ((position[dd] - geom.x_lo[dd]) / geom.dx[dd]).floor() as isize - geom.index_offset[dd],
        });
        geom.interior.contains(coord).then_some((i, coord))
    })
}

/// level `level`'s continuous tracers moved to the tile whose level cell holds their position, so
/// the next substep interpolates them inside that tile's coefficients; a tracer outside every
/// tile's level stays where it is, as it stays on its level of the single hierarchy
fn migrate_level_continuous<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    level: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let mut moved = Vec::new();
    for source in 0..tiles.len() {
        let Some(len) = tiles[source].levels.get(level).and_then(|l| l.state.continuous_tracers.as_ref()).map(|t| t.len) else {
            continue;
        };
        let mut targets = Vec::with_capacity(len);
        {
            let tracers = tiles[source].levels[level].state.continuous_tracers.as_ref().unwrap();
            for kk in 0..len {
                let inactive = unsafe {
                    *tracers.escaped.as_ptr::<u8>().add(kk) != 0 || *tracers.crossed_sink.as_ptr::<u8>().add(kk) != 0
                };
                let position: [f64; NDIM] = unsafe { std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(kk)) };
                targets.push(if inactive { None } else { level_cell_holder(tiles, level, position).map(|(i, _)| i) });
            }
        }
        let tracers = tiles[source].levels[level].state.continuous_tracers.as_mut().unwrap();
        for kk in (0..len).rev() {
            if let Some(target) = targets[kk] {
                if target != source {
                    let record = tracers.swap_remove_host(kk).unwrap_or_else(|detail| panic!("continuous tracer migration: {detail}"));
                    moved.push((target, record));
                }
            }
        }
    }
    for (target, record) in moved {
        tiles[target].levels[level]
            .state
            .continuous_tracers
            .as_mut()
            .expect("a tile of a traced level carries continuous tracers")
            .push_host(record)
            .unwrap_or_else(|detail| panic!("continuous tracer migration: {detail}"));
    }
}

/// level `level`'s held injection over every tile, spawned once from the merged ledger as the
/// single hierarchy spawns its own after the level's tracers advanced: ids from the run's next id,
/// the remainder carried, the draw keyed by the level's iteration, each tracer placed in its
/// whole-grid cell on the moved mesh through the tile holding the cell and stored there. the next
/// id and the remainder ride every level of every tile
fn spawn_decomposed_continuous<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    order: &[usize],
    level: usize,
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let Some(&holder) = order.iter().find(|&&i| tiles[i].levels.get(level).is_some_and(|l| l.state.continuous_tracers.is_some())) else {
        return;
    };
    let mut ledger = std::collections::BTreeMap::new();
    for &i in order {
        for (destination, mass) in tiles[i].continuous_injection.remove(&level).unwrap_or_default() {
            *ledger.entry(destination).or_insert(0.0) += mass;
        }
    }
    let (order_kind, weight, run_seed, next_id, remainder) = {
        let t = tiles[holder].levels[level].state.continuous_tracers.as_ref().unwrap();
        (t.order, t.weight, t.run_seed, t.next_id, t.injection_remainder)
    };
    let mut spawned = symbi_sim::tracers::ContinuousTracerSet::<NDIM, Mem>::allocate(0, order_kind)
        .unwrap_or_else(|detail| panic!("continuous tracer injection: {detail}"));
    spawned.weight = weight;
    spawned.run_seed = run_seed;
    spawned.next_id = next_id;
    spawned.injection_remainder = remainder;
    let key = symbi_sim::mass_transport::SamplingKey {
        run_seed,
        epoch: tiles[holder].levels[level].state.iteration | (1 << 62),
    };
    let owner_of = |owner: symbi_sim::mass_transport::ContainerId| -> Option<(usize, [isize; NDIM])> {
        order.iter().find_map(|&i| {
            let (lev, coord) = tiles[i].tracer_cell(owner)?;
            (lev == level).then_some((i, coord))
        })
    };
    symbi_sim::tracers::spawn_continuous_injected_tracers(
        &mut spawned,
        ledger.into_iter().map(|(destination, mass)| symbi_sim::mass_transport::MassTransfer { destination, mass }),
        |owner| {
            let (i, coord) = owner_of(owner).expect("a tile of the level holds every injection destination");
            let state = &tiles[i].levels[level].state;
            let geometry = state.geom.block_geometry(state.physics.metric);
            let (_, scale_end, _, offset_end) = symbi_sim::tracers::continuous_tracer_mesh_step(&state.store, state.dt);
            let lo: [f64; NDIM] = state.geom.stagger_coord(coord, [symbi_sim::state::Loc::Face; NDIM]);
            let width: [f64; NDIM] = std::array::from_fn(|dd| geometry.cell_width(coord, dd));
            (std::array::from_fn(|dd| scale_end * lo[dd] + offset_end[dd]), width.map(|value| scale_end * value))
        },
        key,
    )
    .unwrap_or_else(|detail| panic!("decomposed continuous tracer injection: {detail}"));
    let records: Vec<_> = (0..spawned.len)
        .map(|kk| {
            let record = spawned.record_host(kk).unwrap_or_else(|detail| panic!("continuous tracer injection: {detail}"));
            let (i, _) = owner_of(record.owner).expect("a tile of the level holds every spawned record");
            (i, record)
        })
        .collect();
    for (i, record) in records {
        tiles[i].levels[level]
            .state
            .continuous_tracers
            .as_mut()
            .expect("a tile of a traced level carries continuous tracers")
            .push_host(record)
            .unwrap_or_else(|detail| panic!("continuous tracer injection: {detail}"));
    }
    for tile in tiles.iter_mut() {
        for level in &mut tile.levels {
            if let Some(tracers) = level.state.continuous_tracers.as_mut() {
                tracers.next_id = spawned.next_id;
                tracers.injection_remainder = spawned.injection_remainder;
            }
        }
    }
}

/// the continuous tracers relocated onto the finest level whose cell holds their position, on the
/// tile holding that cell, once per accepted root step, as the single hierarchy relocates them over
/// its levels; a tracer outside every level stays where it is
fn migrate_continuous_to_finest_decomposed<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
) where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let depth = tiles.iter().map(|t| t.levels.len()).max().unwrap_or(1);
    let locate = |tiles: &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>], position: [f64; NDIM]| {
        (0..depth).rev().find_map(|level| {
            let (i, coord) = level_cell_holder(tiles, level, position)?;
            let layout = tiles[i].tracer_layout(level);
            let interior = &tiles[i].levels[level].state.geom.interior;
            let cell: [usize; NDIM] = std::array::from_fn(|dd| layout.tile_offset[dd] + (coord[dd] - interior.spaces[dd].lo) as usize);
            let owner = symbi_sim::tracers::cell_container_id(symbi_sim::tracers::cell_address(cell, layout.global_cells), layout.level);
            Some((i, level, owner))
        })
    };
    let mut moved = Vec::new();
    for source in 0..tiles.len() {
        for source_level in 0..tiles[source].levels.len() {
            let Some(len) = tiles[source].levels[source_level].state.continuous_tracers.as_ref().map(|t| t.len) else {
                continue;
            };
            let mut plans = Vec::with_capacity(len);
            {
                let tracers = tiles[source].levels[source_level].state.continuous_tracers.as_ref().unwrap();
                for kk in 0..len {
                    let inactive = unsafe {
                        *tracers.escaped.as_ptr::<u8>().add(kk) != 0 || *tracers.crossed_sink.as_ptr::<u8>().add(kk) != 0
                    };
                    let position: [f64; NDIM] = unsafe { std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(kk)) };
                    plans.push(if inactive { None } else { locate(tiles, position) });
                }
            }
            let tracers = tiles[source].levels[source_level].state.continuous_tracers.as_mut().unwrap();
            for kk in (0..len).rev() {
                let Some((target, target_level, owner)) = plans[kk] else { continue };
                if (target, target_level) == (source, source_level) {
                    unsafe { *tracers.owner.as_mut_ptr::<symbi_sim::mass_transport::ContainerId>().add(kk) = owner };
                } else {
                    let mut record = tracers.swap_remove_host(kk).unwrap_or_else(|detail| panic!("continuous tracer refinement transfer: {detail}"));
                    record.owner = owner;
                    moved.push((target, target_level, record));
                }
            }
        }
    }
    for (target, target_level, record) in moved {
        tiles[target].levels[target_level]
            .state
            .continuous_tracers
            .as_mut()
            .expect("every level of a traced tile carries continuous tracers")
            .push_host(record)
            .unwrap_or_else(|detail| panic!("continuous tracer refinement transfer: {detail}"));
    }
}

/// the root timestep every tile of a lockstep march takes: each tile's raw estimate over its
/// levels in root units, agreed through the shared timestep agreement under the root's user cap
/// and the time left to `t_final`. each tile keeps its own raw estimate for the next crash report;
/// a fatal estimate is recorded on the tile that holds it, and the march halts on its last state.
fn agree_tiles_dt<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    devices: Option<&[i32]>,
    t_final: f64,
) -> Option<f64>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    let raws: Vec<f64> = (0..tiles.len())
        .map(|i| match devices {
            Some(devices) => symbi_xpu::with_device(devices[i], || prof("cfl", || tiles[i].raw_root_cfl())),
            None => prof("cfl", || tiles[i].raw_root_cfl()),
        })
        .collect();
    let root = &tiles[0].levels[0].state;
    match agree_timestep(raws.iter().copied(), root.max_dt, t_final - root.time, Ok::<f64, std::convert::Infallible>) {
        Ok(agreed) => {
            for (tile, raw) in tiles.iter_mut().zip(&raws) {
                tile.prev_dt_cfl = *raw;
            }
            Some(agreed.dt)
        }
        Err(TimestepFailure::Invalid(bad)) => {
            tiles[bad.index].record_cfl_crash(bad.value);
            None
        }
        Err(TimestepFailure::Collective(never)) => match never {},
    }
}

/// the one hierarchy march: N tiles advance in lockstep behind an agreed timestep, with
/// the per-tile watchdog, the step-panic catch, the crash report, and the observer
/// cadence shared by every shape. a single tile with no decomposition context steps
/// through the uni-grid transaction (`step_root_with_dt`); a decomposed set steps
/// through the collective attempt with halo exchanges. the observer fires every
/// `interval` root iterations, once when a crash is recorded (the driver snapshots
/// the `.crashed` state), and once after the final step, devices drained first.
pub fn evolve_tiles<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T, F>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    decomp: Option<(&[i32], [usize; NDIM], &T)>,
    t_final: f64,
    interval: u64,
    mut callback: F,
) -> symbi_sim::run_diagnostics::RunDiagnostics
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
    F: FnMut(&[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>]) -> ControlFlow<()>,
{
    let n = tiles.len();
    // the projection ledger's run scope: this shared driver is the one path
    // every backend run takes. a single-grid run opens the scope so the
    // projection's receipts book over the accepted step; a decomposed or
    // refined run opens nothing, since the step transaction the ledger needs is
    // modeled for one grid only, and simply records no evidence there. the
    // handle closes at return, leaving the totals queryable.
    let ledger_scope = (n == 1 && decomp.is_none() && tiles[0].levels.len() == 1)
        .then(symbi_sim::projection_ledger::open_scope);
    // the guard ledger books on every run shape — single grid, decomposed,
    // refined — so its scope opens unconditionally, and every tile's and level's
    // substages book into this thread's ledger.
    let guard_scope = symbi_sim::guard_ledger::open_scope();
    // the run-owned diagnostics: the accepted totals read from this run's own
    // scopes. the projection scope opens only on the single grid; the guard scope
    // opens always. reading through the scopes binds the evidence to this run, so
    // no other run's totals can leak in.
    let run_diagnostics = || symbi_sim::run_diagnostics::RunDiagnostics {
        projection: ledger_scope
            .as_ref()
            .map(symbi_sim::projection_ledger::LedgerScope::accepted)
            .unwrap_or_default(),
        guards: guard_scope.accepted(),
        // a hierarchy records its halt on the tile itself (`crash`), caught step panics included
        halt: None,
    };
    let drain = |decomp: Option<(&[i32], [usize; NDIM], &T)>| match decomp {
        Some((devices, _, _)) => drain_devices::<Mem>(devices),
        None => symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>(),
    };
    let root_order: Vec<usize> = (0..n).collect();
    let nstages = tiles[0].levels[0].state.timestepping.stages().len();
    // the sub-grid of every refined level; each tile learns the run's depth, since the finest
    // level's physics belongs to the run's finest level whichever levels the tile holds
    let fine: Vec<FineSubgrid<NDIM>> = match decomp {
        Some((devices, counts, _)) => {
            let depth = tiles.iter().map(|h| h.levels.len()).max().unwrap_or(1);
            for h in tiles.iter_mut() {
                h.set_run_levels(depth);
            }
            (1..depth).map_while(|level| fine_subgrid_of_level(tiles, counts, devices, level)).collect()
        }
        None => Vec::new(),
    };
    // cut halos current before the first flux.
    if let Some((devices, counts, transport)) = decomp {
        exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);
    }

    let mut last_cb = tiles[0].levels[0].state.iteration;
    while tiles[0].levels[0].state.time < t_final {
        // the agreed timestep: every tile runs the watchdog-screened selection, and the
        // minimum drives the lockstep step. a fatal estimate records the crash on its
        // tile and the march halts on the last computed state below.
        for tile in tiles.iter() {
            tile.assert_cf_order_pairing();
        }
        let agreed = agree_tiles_dt(tiles, decomp.map(|(devices, _, _)| devices), t_final);
        let dt_crashed = agreed.is_none();
        let gdt = agreed.unwrap_or(f64::INFINITY);
        if !dt_crashed {
            // a panic inside the step (the FOFC freeze-streak halt, a poisoned-cell
            // assertion) carries the same diagnostic value as a watchdog crash and
            // deserves the same exit: catch it, convert it into a crash report carrying
            // the panic message, and let the observer snapshot the `.crashed`
            // checkpoint. the default panic hook has already printed the message and
            // backtrace to stderr at the panic site, so the report reaches the log and
            // the state reaches disk. the caught state is the mid-step one the panic
            // fired on, which is exactly the state worth inspecting.
            let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match decomp {
                    Some((devices, counts, transport)) => {
                        let t = tiles[0].levels[0].state.time;
                        if let Some(adt) =
                            decomposed_root_attempt(tiles, counts, devices, transport, &fine, &root_order, nstages, gdt, t)
                        {
                            decomposed_root_post(tiles, counts, devices, transport, &root_order, adt, t);
                        }
                    }
                    None => tiles[0].step_root_with_dt(gdt),
                }
            }));
            if let Err(payload) = step {
                let msg = payload
                    .downcast_ref::<&'static str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with a non-string payload".to_string());
                let (iter, time) = {
                    let r = &tiles[0].levels[0].state;
                    (r.iteration, r.time)
                };
                tiles[0].crash = Some(CrashReport {
                    iter,
                    time,
                    dt_cfl: f64::NAN,
                    dt_prev: tiles[0].prev_dt_cfl,
                    panic: Some(msg),
                    halt: None,
                });
            }
        }
        // a crashed step records the crash and leaves the clock where it was: let the
        // observer snapshot the `.crashed` checkpoint + report it, then stop the march.
        // marching on would spin on the frozen time and clamp dt to t_final on garbage.
        if tiles.iter().any(|h| h.crash.is_some()) {
            // a crashed step never reached its accepted-step boundary, so its
            // projection receipts describe a state that did not survive: discard
            // the pending bucket before the scope closes.
            if ledger_scope.is_some() {
                symbi_sim::projection_ledger::step_discard();
            }
            // the crashed step never reached its accepted boundary, so its guard
            // acts describe a state that did not survive: discard the pending
            // bucket before the scope closes.
            symbi_sim::guard_ledger::step_discard();
            drain(decomp);
            let _ = callback(tiles);
            return run_diagnostics();
        }
        let it = tiles[0].levels[0].state.iteration;
        if it.saturating_sub(last_cb) >= interval {
            last_cb = it;
            drain(decomp);
            // a `Break` from the observer (e.g., a caught signal) stops the march early;
            // the observer has already snapshotted state for restart. drain the queue so
            // the host read in the caller is coherent, then return.
            if callback(tiles).is_break() {
                drain(decomp);
                return run_diagnostics();
            }
        }
    }
    // the caller reads fields next: drain the device queue (the host-read barrier;
    // no-op on a host backend).
    drain(decomp);
    let _ = callback(tiles);
    run_diagnostics()
}

/// the subcycle of level `parent + 1` across its tiles within one step `dt` of `parent`: ratio
/// substeps, each prolonging the coarse-fine ghosts from the parent at the substep's time
/// fraction, then the stages with the troubled marks and the halos crossing the cuts, the tail
/// ahead of the finer subcycle, the subcycle of the next level with its restriction and reflux
/// into this one, and the tail after it. `fine[k]` is the sub-grid of the tiles carrying level
/// `k + 1`; `root` is the root tiles' grid and devices, across whose cuts each level-1 stage's
/// register units travel. returns whether some tile rejected the step.
#[allow(clippy::too_many_arguments)]
fn decomposed_subcycle<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    parent: usize,
    fine: &[FineSubgrid<NDIM>],
    root: ([usize; NDIM], &[i32]),
    transport: &T,
    nstages: usize,
    dt: f64,
) -> bool
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    let level = parent + 1;
    let fg = &fine[parent];
    let fine_dt = dt / RATIO as f64;
    for sub in 0..RATIO {
        let alpha = sub as f64 / RATIO as f64;
        for (k, &i) in fg.order.iter().enumerate() {
            symbi_xpu::with_device(fg.devices[k], || {
                tiles[i].prolong_cf(level, alpha);
                tiles[i].fill_physical_ghosts(level);
            });
        }
        // the cut halos filled before the flux reads them
        exchange_level_halos(tiles, level, &fg.order, fg.counts, &fg.devices, transport);
        for &i in &fg.order {
            phase_probe::set_tile(i);
            phase_probe::sample("prolong", level, &tiles[i].levels[level].state);
        }
        for (k, &i) in fg.order.iter().enumerate() {
            symbi_xpu::with_device(fg.devices[k], || tiles[i].level_step_begin(level, fine_dt));
            phase_probe::set_tile(i);
            phase_probe::sample("begin", level, &tiles[i].levels[level].state);
        }
        for jj in 0..nstages {
            if decomposed_level_stage(tiles, level, &fg.order, fg.counts, &fg.devices, transport, jj, fine_dt, alpha) {
                return true;
            }
            exchange_level_tracers(tiles, level, jj);
            exchange_level_halos(tiles, level, &fg.order, fg.counts, &fg.devices, transport);
            if parent == 0 {
                exchange_flux_contributions(tiles, root.0, root.1, transport);
            }
            for &i in &fg.order {
                phase_probe::set_tile(i);
                phase_probe::sample(phase_probe::STAGE[jj], level, &tiles[i].levels[level].state);
            }
        }
        spawn_decomposed_injections(tiles, &fg.order, level);
        for (k, &i) in fg.order.iter().enumerate() {
            phase_probe::set_tile(i);
            symbi_xpu::with_device(fg.devices[k], || tiles[i].level_tail_pre(level, fine_dt));
        }
        // the level's accepted ito coefficients across its cuts, before a finer level prolongs them
        // and the level's tracers interpolate them
        let continuous = fg.order.iter().any(|&i| tiles[i].levels[level].state.continuous_tracers.is_some());
        if continuous {
            exchange_level_ito(tiles, level, &fg.order, fg.counts, &fg.devices, transport);
        }
        if let Some(finer) = fine.get(level) {
            if decomposed_subcycle(tiles, level, fine, root, transport, nstages, fine_dt) {
                return true;
            }
            for (k, &i) in finer.order.iter().enumerate() {
                phase_probe::set_tile(i);
                symbi_xpu::with_device(finer.devices[k], || tiles[i].level_restrict_reflux(level, alpha));
            }
            // a tile holding this level and none finer recovers it as the restriction recovers the
            // single hierarchy's
            for (k, &i) in fg.order.iter().enumerate() {
                if tiles[i].levels.len() == level + 1 {
                    symbi_xpu::with_device(fg.devices[k], || tiles[i].recover_level(level));
                }
            }
            exchange_level_halos(tiles, level, &fg.order, fg.counts, &fg.devices, transport);
            for &i in &finer.order {
                phase_probe::set_tile(i);
                phase_probe::sample("restrict", level, &tiles[i].levels[level].state);
            }
        }
        if continuous {
            // the tail split across the tiles: every tile advances its tracers, the tracers cross to
            // the tiles holding their cells, the held injection spawns once, then each tile finishes
            for (k, &i) in fg.order.iter().enumerate() {
                symbi_xpu::with_device(fg.devices[k], || tiles[i].advance_level_continuous_tracers(level, fine_dt));
            }
            migrate_level_continuous(tiles, level);
            spawn_decomposed_continuous(tiles, &fg.order, level);
            for (k, &i) in fg.order.iter().enumerate() {
                phase_probe::set_tile(i);
                symbi_xpu::with_device(fg.devices[k], || tiles[i].finish_level_tail_post(level, fine_dt));
            }
        } else {
            for (k, &i) in fg.order.iter().enumerate() {
                phase_probe::set_tile(i);
                symbi_xpu::with_device(fg.devices[k], || tiles[i].level_tail_post(level, fine_dt));
            }
        }
    }
    false
}

/// the attempts of one hierarchy's root step: every level's retry snapshot when the step can be
/// rejected, the finer levels' attempt entry, and the host side-cars (tracers, bodies, motion,
/// clocks, censuses) captured at the first attempt
struct SingleRootAttempt<'h, R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    hier: &'h mut Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>,
    restorable: bool,
    sidecars: Option<HierarchyRetrySidecars<NDIM>>,
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K> RootAttempt for SingleRootAttempt<'_, R, NDIM, DOF, M, E, S, Mem, K>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
{
    type Error = std::convert::Infallible;

    fn rejectable(&mut self) -> bool {
        self.hier.root_step_rejectable()
    }

    fn capture(&mut self, _dt: f64, restorable: bool) -> Result<(), Self::Error> {
        let hier = &mut *self.hier;
        if self.sidecars.is_none() {
            self.sidecars = Some(hier.snapshot_retry_sidecars());
        }
        self.restorable = restorable;
        if restorable {
            for level in &hier.levels {
                level.kernels.snapshot_retry(&level.state);
            }
        }
        hier.snapshot_attempt_entry();
        Ok(())
    }

    fn run(&mut self, dt: f64) -> Result<bool, Self::Error> {
        // a slip-enabled draining body runs the palindromic D-M-H-M-D coupled step in place of the
        // plain RK march; the capture above (taken before the first D) is the boundary its retry
        // restores to. a body with no magnetic slip keeps the ordinary advance bit-for-bit.
        let failed = if self.hier.finest_has_magnetic_slip() {
            self.hier.advance_slip_coupled_step(0, dt, 0.0)
        } else {
            self.hier.advance_level(0, dt, 0.0)
        };
        Ok(!failed)
    }

    fn evidence(&mut self, rejections: u32) -> Vec<String> {
        // the rejection counts against this root attempt, a fine substep's included
        let mut evidence = Vec::new();
        for (ll, level) in self.hier.levels.iter().enumerate() {
            evidence.extend(symbi_sim::retry_guard::take_evidence(&level.state, &format!("level {ll}"), rejections));
        }
        evidence
    }

    fn restore(&mut self) -> Result<(), Self::Error> {
        assert!(self.restorable, "a root step restored from an incomplete capture");
        let hier = &mut *self.hier;
        for level in &hier.levels {
            level.kernels.restore_step(&level.state);
        }
        hier.restore_attempt_entry();
        hier.restore_retry_sidecars(self.sidecars.as_ref().expect("the first attempt captured the side-cars"));
        Ok(())
    }

    fn severe_streak(&mut self) -> u32 {
        self.hier.levels[0].state.severe_streak
    }

    fn set_severe_streak(&mut self, streak: u32) {
        self.hier.levels[0].state.severe_streak = streak;
    }
}

/// the attempts of one collective root step of the decomposed march: every tile snapshots its
/// retry state, the root stages advance in lockstep with a halo exchange between each, the fine
/// subcycle runs with its own exchanges, and a rejection anywhere rolls every tile back to the
/// same entry state, its halos exchanged again, before the transaction replays it
struct TiledRootAttempt<'a, R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    tiles: &'a mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &'a [i32],
    transport: &'a T,
    fine: &'a [FineSubgrid<NDIM>],
    root_order: &'a [usize],
    nstages: usize,
    restorable: bool,
    sidecars: Option<Vec<HierarchyRetrySidecars<NDIM>>>,
}

impl<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T> RootAttempt for TiledRootAttempt<'_, R, NDIM, DOF, M, E, S, Mem, K, T>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    type Error = std::convert::Infallible;

    fn rejectable(&mut self) -> bool {
        self.tiles.iter().any(Hierarchy::root_step_rejectable)
    }

    fn capture(&mut self, _dt: f64, restorable: bool) -> Result<(), Self::Error> {
        if self.sidecars.is_none() {
            self.sidecars = Some(self.tiles.iter().map(Hierarchy::snapshot_retry_sidecars).collect());
        }
        self.restorable = restorable;
        for (i, tile) in self.tiles.iter().enumerate() {
            symbi_xpu::with_device(self.devices[i], || {
                if restorable {
                    for level in &tile.levels {
                        level.kernels.snapshot_retry(&level.state);
                    }
                }
                tile.snapshot_attempt_entry();
            });
        }
        Ok(())
    }

    fn run(&mut self, dt: f64) -> Result<bool, Self::Error> {
        let (tiles, counts, devices, transport, fine, root_order, nstages) =
            (&mut *self.tiles, self.counts, self.devices, self.transport, self.fine, self.root_order, self.nstages);
        let n = tiles.len();
        for i in 0..n {
            symbi_xpu::with_device(devices[i], || tiles[i].level_step_begin(0, dt));
            phase_probe::set_tile(i);
            phase_probe::sample("begin", 0, &tiles[i].levels[0].state);
        }
        let mut retry = false;
        // the root SSP stages, each with the troubled marks crossing the cuts before the
        // correction and a root halo exchange after.
        for ii in 0..nstages {
            retry |= decomposed_level_stage(tiles, 0, root_order, counts, devices, transport, ii, dt, 0.0);
            if retry {
                break;
            }
            exchange_level_tracers(tiles, 0, ii);
            exchange_level_halos(tiles, 0, root_order, counts, devices, transport);
            for &i in root_order {
                phase_probe::set_tile(i);
                phase_probe::sample(phase_probe::STAGE[ii], 0, &tiles[i].levels[0].state);
            }
        }
        // a rejected stage skips the root tail; the transaction restores the entry
        if !retry {
            spawn_decomposed_injections(tiles, root_order, 0);
            // root emf bookkeeping (mhd; no-op for hydro + unrefined tiles).
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || tiles[i].level_tail_emf(0, dt));
            }
            // the root excise, in the same position the uni-grid tail puts it (after the emf
            // bookkeeping). the rest of the tail — viscous, horizon ledger, penalize — belongs to the
            // finest level and is driven with the fine subcycle; the excise runs per level, because
            // the excised region is owned by whichever level contains it and a refined root still
            // owns its core.
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || tiles[i].level_tail_excise(0));
            }
            // rigid walls also act on uncovered root cells when their shape crosses the refined
            // patch boundary. covered root writes are replaced by restriction after the fine
            // subcycle; accretion remains absent from root proxies.
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || {
                    let level = &tiles[i].levels[0];
                    let has_rigid = level
                        .state
                        .immersed
                        .as_ref()
                        .is_some_and(|immersed| immersed.bodies.rigid_count() > 0);
                    if has_rigid {
                        prof("penalize", || level.kernels.penalize(&level.state, dt));
                    }
                });
            }
            // the root's continuous tracers, as the root's tail holds them on the single
            // hierarchy: the injection held until they advance, and the root's accepted ito
            // coefficients across the cuts before the fine level prolongs them
            let continuous = tiles.iter().any(|t| t.levels[0].state.continuous_tracers.is_some());
            if continuous {
                for i in 0..n {
                    tiles[i].hold_continuous_injection(0);
                    symbi_xpu::with_device(devices[i], || tiles[i].prepare_level_ito_coefficients(0));
                }
                exchange_level_ito(tiles, 0, root_order, counts, devices, transport);
            }
            // the refined levels, subcycled across their tiles, then each refined tile's root
            // takes the restriction and reflux of level 1, and the root halos carry the
            // restricted covered cells: the next step's prolongation snapshot and the
            // cut-adjacent stencils read them.
            if let Some(fg) = fine.first() {
                retry |= decomposed_subcycle(tiles, 0, fine, (counts, devices), transport, nstages, dt);
                if !retry {
                    for (k, &i) in fg.order.iter().enumerate() {
                        phase_probe::set_tile(i);
                        symbi_xpu::with_device(fg.devices[k], || {
                            tiles[i].level_restrict_reflux(0, 0.0)
                        });
                    }
                    // a tile beside a neighbor's patch face on a cut applies the units it received
                    // and recovers its cells beside the patch; every other tile holding no fine
                    // level recovers its root as the restriction recovers the single hierarchy's
                    for i in 0..n {
                        if tiles[i].levels.len() == 1 {
                            symbi_xpu::with_device(devices[i], || {
                                if tiles[i].has_flux_register(0) {
                                    tiles[i].reflux_received();
                                    tiles[i].recover_beside_patch();
                                } else {
                                    tiles[i].recover_level(0);
                                }
                            });
                        }
                    }
                    exchange_level_halos(tiles, 0, root_order, counts, devices, transport);
                    for &i in root_order {
                        phase_probe::set_tile(i);
                        phase_probe::sample("restrict", 0, &tiles[i].levels[0].state);
                    }
                }
            }
            // the root's continuous tracers advance over the root step after the fine subcycle and
            // its restriction, cross to the tiles holding their cells, and the held injection
            // spawns once
            if !retry && continuous {
                for i in 0..n {
                    symbi_xpu::with_device(devices[i], || tiles[i].advance_level_continuous_tracers(0, dt));
                }
                migrate_level_continuous(tiles, 0);
                spawn_decomposed_continuous(tiles, root_order, 0);
            }
            // the root clock; the root post-step carries no mesh motion.
            if !retry {
                for i in 0..n {
                    symbi_xpu::with_device(devices[i], || {
                        let s = &mut tiles[i].levels[0].state;
                        (s.time, s.iteration) = advance_clock(s.time, s.iteration, dt);
                    });
                }
            }
        }
        Ok(!retry)
    }

    fn evidence(&mut self, rejections: u32) -> Vec<String> {
        // the rejection counts once against this root attempt, whichever tile, level or substep
        // raised it
        let mut evidence = Vec::new();
        for (i, tile) in self.tiles.iter().enumerate() {
            for (ll, level) in tile.levels.iter().enumerate() {
                evidence.extend(symbi_sim::retry_guard::take_evidence(&level.state, &format!("tile {i}, level {ll}"), rejections));
            }
        }
        evidence
    }

    fn restore(&mut self) -> Result<(), Self::Error> {
        assert!(self.restorable, "a root step restored from an incomplete capture");
        // every tile returns to the same root-step entry, host-side tracer and body state
        // included, before any halo is exchanged
        let sidecars = self.sidecars.as_ref().expect("the first attempt captured the side-cars");
        for (i, tile) in self.tiles.iter_mut().enumerate() {
            symbi_xpu::with_device(self.devices[i], || {
                for level in &tile.levels {
                    level.kernels.restore_step(&level.state);
                }
                tile.restore_attempt_entry();
            });
            tile.restore_retry_sidecars(&sidecars[i]);
        }
        drain_devices::<Mem>(self.devices);
        exchange_level_halos(self.tiles, 0, self.root_order, self.counts, self.devices, self.transport);
        for (k, fg) in self.fine.iter().enumerate() {
            exchange_level_halos(self.tiles, k + 1, &fg.order, fg.counts, &fg.devices, self.transport);
        }
        Ok(())
    }

    fn severe_streak(&mut self) -> u32 {
        self.tiles[0].levels[0].state.severe_streak
    }

    // every tile's root carries the streak its checkpoint records
    fn set_severe_streak(&mut self, streak: u32) {
        for tile in self.tiles.iter_mut() {
            tile.levels[0].state.severe_streak = streak;
        }
    }
}

/// one collective root step of the decomposed march from its agreed candidate timestep, through
/// the shared step transaction: the accepted timestep the tiles advanced by, or none when the
/// step halted on its entry state, the halt recorded on the first tile
#[allow(clippy::too_many_arguments)]
fn decomposed_root_attempt<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    fine: &[FineSubgrid<NDIM>],
    root_order: &[usize],
    nstages: usize,
    candidate: f64,
    t: f64,
) -> Option<f64>
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    let guard = tiles[0].levels[0].state.retry_guard;
    let mut attempt =
        TiledRootAttempt { tiles: &mut *tiles, counts, devices, transport, fine, root_order, nstages, restorable: false, sidecars: None };
    let dt = match take_root_step(&mut attempt, &guard, candidate, t) {
        Ok::<_, std::convert::Infallible>(RootStep::Accepted { dt }) => dt,
        Ok(RootStep::Halted(halt)) => {
            tiles[0].record_step_halt(&halt);
            return None;
        }
    };
    // the continuous tracers onto the finest level holding them, once per accepted root step
    if tiles.iter().any(|t| t.levels.iter().any(|l| l.state.continuous_tracers.is_some())) {
        migrate_continuous_to_finest_decomposed(tiles);
    }
    Some(dt)
}

/// the accepted-step tail of the decomposed march: the per-step immersed-body bookkeeping
/// (each tile's finest level reduces its local feedback partials — the tile finest interiors
/// partition the sink region — the deltas sum across tiles into the true global reaction,
/// and the identical global delta + prescribed advance applies to every tile's bodies, so
/// all tiles stay in lockstep), then the root halo refresh for the next step's stage 0.
fn decomposed_root_post<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    root_order: &[usize],
    gdt: f64,
    t: f64,
)
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
{
    let n = tiles.len();
        // per-step immersed-body bookkeeping, the hierarchy form of the flat decomposed
        // body step: each tile's finest level reduces its local feedback partials (the
        // tile finest interiors partition the sink region — coarser levels carry a
        // gravity-only proxy, so the drain stays on the finest), the deltas are summed
        // across tiles into the true global reaction, and the identical global delta +
        // prescribed advance is applied to every tile's bodies. identical input ->
        // identical body state, so all tiles stay in lockstep.
        if tiles.iter().any(|h| h.levels[0].state.has_bodies()) {
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || tiles[i].finest_body_feedback(gdt));
            }
            drain_devices::<Mem>(devices);
            let mut global: Vec<symbi_ib::BodyDelta<f64, NDIM>> = Vec::new();
            for i in 0..n {
                for d in tiles[i].take_body_deltas() {
                    match global.iter_mut().find(|g| g.idx == d.idx) {
                        Some(g) => *g += d,
                        None => global.push(d),
                    }
                }
            }
            // the run's finest clock, advanced through its own substeps: the clock the single
            // hierarchy books its body history at, identical on every tile carrying that level
            let depth = tiles.iter().map(|h| h.levels.len()).max().unwrap_or(1);
            let t_now = tiles
                .iter()
                .find(|h| h.levels.len() == depth)
                .map_or(t + gdt, |h| h.levels[depth - 1].state.time);
            let any_refined = tiles.iter().any(|h| h.levels.len() > 1);
            for i in 0..n {
                symbi_xpu::with_device(devices[i], || {
                    tiles[i].apply_global_body_deltas(&global, gdt, t_now)
                });
                assert!(
                    !(any_refined
                        && tiles[i].levels.len() == 1
                        && tiles[i].accretion_overlaps_root()),
                    "refinement x decomposition: a sink sphere overlaps UNREFINED tile {i} — the \
                     refined patch must cover the sink on every tile it touches"
                );
            }
        }
        // the restrict/c2p recomputed root prim over the allocated domain (incl. the cut halo from
        // stale halo cons, and the cut-adjacent reflux ghosts); refresh for the next step's stage 0.
        exchange_level_halos(tiles, 0, &root_order, counts, devices, transport);
}

/// the decomposed hierarchy driver (refinement x decomposition): lockstep-advance N
/// per-tile hierarchies, decomposing every level. the root stages run with a
/// root halo exchange between them (rk2 corrector reads each neighbor's stage-1 update); the fine
/// subcycle is driven here so its stages can exchange the fine halos between fine tiles when a
/// patch spans a tile cut. for a tile-local patch the fine sub-grid is 1x1, so the fine exchange
/// is a no-op and this reduces to the tile-local case exactly. a patch face inside a tile refluxes
/// on that tile (the coarse cell and the fine cells at its face are co-located; any register write
/// to a cut-adjacent ghost is overwritten by the next root exchange); a gas patch face on a root
/// cut sends each fine stage's register unit to the tile across the cut, which refluxes the coarse
/// cells it holds after the subcycle. global dt = min over tiles of
/// `root_cfl_dt()`. `decomposed == monolithic` holds for both tile-local and cut-spanning patches.
///
/// every refined level subcycles across the tiles carrying it, with its own halo exchange across
/// the cuts, and restricts into its parent before the parent's tail completes; a tile holding
/// fewer levels than the run leaves the finest-level physics to the run's finest level.
/// hydro only in the root post-step (the root clock is advanced directly; mesh motion and immersed
/// bodies belong to the single-grid path). `on_checkpoint(iteration, time, &tiles)` fires every
/// `interval` root steps and once at the end (devices drained first).
#[allow(clippy::too_many_arguments)]
pub fn evolve_hierarchy_decomposed<R, const NDIM: usize, const DOF: usize, M, E, S, Mem, K, T, F>(
    tiles: &mut [Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>],
    counts: [usize; NDIM],
    devices: &[i32],
    transport: &T,
    ts: Timestepping,
    start_time: f64,
    t_final: f64,
    interval: u64,
    mut on_checkpoint: F,
) -> symbi_sim::run_diagnostics::RunDiagnostics
where
    R: Regime<f64, NDIM> + Copy,
    M: Metric<f64, NDIM> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, NDIM>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<NDIM, DOF, Mem, f64>,
    T: HaloTransport,
    F: FnMut(u64, f64, &[Hierarchy<R, NDIM, DOF, M, E, S, Mem, K>]) -> ControlFlow<()>,
{
    symbi_sim::retry_guard::assert_one_retry_exclusion(tiles.iter().flat_map(|h| h.levels.iter().map(|l| &l.state.retry_exclusion)));
    symbi_sim::decomp::assert_cut_axes_declare_cuts(tiles.iter().map(|h| &h.levels[0].state.boundaries), counts);
    let n = tiles.len();
    for tile in tiles.iter_mut() {
        tile.blend_across_tiles = true;
    }
    if tiles.iter().any(|t| t.levels.iter().any(|l| l.state.continuous_tracers.is_some())) {
        let domain = whole_continuous_domain(tiles);
        for tile in tiles.iter_mut() {
            tile.continuous_domain = Some(domain);
        }
    }
    // the finest-level tail owns the IBM surface physics, and the refined-
    // decomposed step drives the fine tail alone, so a refined run needs at
    // least one tile carrying a fine patch for its bodies to be penalized at
    // all — a silent physics loss otherwise, refused here.
    assert!(
        tiles.iter().any(|h| h.levels.len() > 1)
            || tiles.iter().all(|h| !h.levels[0].state.has_bodies()),
        "refined-decomposed: no tile built a fine level while immersed bodies are active; \
         the finest-level penalize would silently never run",
    );
    debug_assert_eq!(n, devices.len(), "tiles/devices length mismatch");
    debug_assert_eq!(
        ts.stages().len(),
        tiles[0].levels[0].state.timestepping.stages().len(),
        "the driver's declared timestepping matches the tiles' own"
    );
    // the tiles' own root clock is the march's time authority; the caller's start_time
    // is the same value by construction (the restart path restores both from the
    // checkpoint), asserted here so a drifting caller fails loudly.
    debug_assert!(
        (tiles[0].levels[0].state.time - start_time).abs()
            <= start_time.abs().max(1.0) * 1e-12,
        "start_time disagrees with the tiles' root clock"
    );
    // the decomposed march is the multi-tile case of the shared driver, which owns the
    // watchdog screening, the step-panic catch, the crash report, and the observer
    // cadence. the callback keeps its historical clock: iterations counted from this
    // call's entry, alongside the root time the tiles carry.
    let iter0 = tiles[0].levels[0].state.iteration;
    // a decomposed run books guard evidence across its tiles but opens no
    // projection scope, so its diagnostics carry the empty projection.
    evolve_tiles(
        tiles,
        Some((devices, counts, transport)),
        t_final,
        interval,
        |march_tiles| {
            let s = &march_tiles[0].levels[0].state;
            on_checkpoint(s.iteration - iter0, s.time, march_tiles)
        },
    )
}

fn wb_ghost_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SYMBI_WB_GHOST")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

fn wb_band_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SYMBI_WB_BAND")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}

#[cfg(test)]
mod cfl_watchdog_tests {
    use super::fatal_cfl_dt;

    #[test]
    fn recovered_transient_wave_speed_is_not_a_crash() {
        // A pressure-supported density hole can briefly make c_s, and hence the CFL rate,
        // enormous. Once it refills, an arbitrarily larger positive dt is safe.
        assert!(!fatal_cfl_dt(1.653_851e-10));
        assert!(!fatal_cfl_dt(4.127_354e-5));
        assert!(!fatal_cfl_dt(f64::INFINITY));
        assert!(fatal_cfl_dt(f64::NAN));
        assert!(fatal_cfl_dt(0.0));
        assert!(fatal_cfl_dt(-1.0));
    }
}
