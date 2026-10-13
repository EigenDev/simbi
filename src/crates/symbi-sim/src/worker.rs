// =============================================================================
// worker.rs
//
// the evolution loop of one worker over the tiles it holds. the step is the
// decomposed step of `evolve_scheduled` with its four cross-tile operations
// carried by the fabric: the timestep is the fabric minimum of every worker's
// owned-tile candidate; a stage's troubled cells are marked on every owned
// tile, the fabric any over every worker's count decides whether the flags
// cross the cuts, and a tile whose cut ghosts received a set flag corrects
// with a clean interior, so a face two tiles share takes one flux; a stage's
// rejection is the fabric any over every worker's outcome and is decided
// before that stage's halos move; and every exchange runs the compiled plan
// across workers under grants. each worker issues one collective per
// decision whatever number of tiles it holds. the tile operations themselves
// are the same kernel calls in the same order, so the owned tiles of a worker
// match the single-process reference bitwise.
//
// the loop evolves single-level hydro and ideal MHD with their dye, immersed
// bodies and mesh motion; a store carrying a horizon body, tracers, or
// excision is refused at entry. a local fatal error tells every peer through
// the abort relay before the loop returns it.
//
// usage:
//  let report = evolve_worker(&tiles, &mut stores, &kernels, &devices, &exchange,
//      &LocalCopy, &mut fabric, &cfg, |iter, t, owned, fabric| Ok(ControlFlow::Continue(())))?;
// =============================================================================

use crate::atlas::{AuditAt, ExchangePoint, TileId};
use crate::decomp::{HaloTransport, plan_fields};
use crate::driver::{
    advance_clock, advance_state_clock, downstream_injection_weight, needs_step_snapshot,
    AgreedTimestep, InvalidCandidate, TimestepFailure, agree_timestep, stage_schedule,
};
use crate::plan_exchange::{AuditCost, AuditError, PlanExchange, StagingFault};
use crate::stage::{
    StageArgs, StageOutcome, fold_stage_curl_through_mark, fold_stage_from_correction, fold_stage_through_emf,
    fold_stage_through_mark, raise_troubled_cell,
};
use crate::state::{FieldStore, Timestepping};
use crate::substrate_seam::{KernelSet, RegimeKind, RetryExclusion};
use std::ops::ControlFlow;
use std::time::{Duration, Instant};
use symbi_fabric::{Fabric, FabricError, Link, OpKind, WorkerId};
use symbi_grid::Field;
use symbi_xpu::{MemorySpace, with_device};

/// outcomes a test harness forces at a named step, outside the numerical path.
#[derive(Debug, Clone, Default)]
pub struct Injection {
    /// report a rejection from this worker at each (step, stage) on that step's first attempt
    pub reject_at: Vec<(u64, usize)>,
    /// take at most this timestep (as f64 bits) at this step: how a clean run follows the
    /// accepted timestep sequence of a run that rejected and halved
    pub cap_dt_at: Option<(u64, u64)>,
    /// report a NaN timestep candidate from this worker's first tile at this step, leaving
    /// its other tiles' candidates valid
    pub invalid_cfl_at: Option<u64>,
    /// raise the troubled-cell flag on a global cell at (step, stage) on the first attempt,
    /// on whichever worker holds it; the first `D` entries index the cell. the cell's state
    /// stays physical, so the correction it triggers is the conservative first-order redo.
    pub trouble_at: Option<(u64, usize, [isize; 3])>,
    /// hold this worker between its retry-guard decision and the guard's collective: the
    /// worker holding the rejecting evidence arriving last there
    pub hold_before_guard: Option<Duration>,
    /// end this worker's process at its retry-guard decision without a word to its peers: a
    /// peer lost while the guard's collective waits for it
    pub vanish_before_guard: bool,
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// the regime the kernels implement, from `RegimeKind::of::<f64, D, R>()`
    pub regime: RegimeKind,
    pub timestepping: Timestepping,
    pub start_time: f64,
    pub t_final: f64,
    /// the root iteration the march stops at, a resumed run's checkpointed iterations included;
    /// zero is unbounded
    pub max_steps: u64,
    pub deadline: Duration,
    pub injection: Injection,
    /// audit the shared interface faces after every exchange point, where the default audits
    /// them after the prime exchange alone
    pub audit_every_point: bool,
}

#[derive(Debug, Clone, Default)]
pub struct WorkerReport {
    pub steps: u64,
    pub rejections: u64,
    pub time: f64,
    /// the accepted timestep of every step, in order
    pub dt_sequence: Vec<f64>,
    /// wall time by activity; the buckets are disjoint and sum to the march's wall time
    /// less the step callback
    pub timing: WorkerTiming,
    /// stages in which some worker was troubled and the flags crossed the cuts
    pub troubled_exchanges: u64,
    /// the tracers that crossed to another tile over the session, every worker counting all
    pub tracer_migrations: crate::worker_tracers::Migrations,
    /// corrections this worker ran on a tile with a clean interior, because a neighbor's
    /// troubled cell reached that tile's cut ghosts
    pub neighbor_corrections: u64,
    /// the troubled and frozen cells of this worker's accepted steps
    pub troubled_cells: u64,
    pub frozen_cells: u64,
    /// interface audits run, and what they cost this worker in total
    pub audits: u64,
    pub audit_cost: AuditCost,
    /// the fine level's troubled exchanges on a refined run
    pub fine_troubled_exchanges: u64,
    /// `neighbor_corrections` on the fine level
    pub fine_neighbor_corrections: u64,
    /// the most grants held at once from each peer over the session
    pub grant_high_water: Vec<usize>,
    /// the cause when every worker agreed to halt the march at a step entry it cannot advance (an
    /// invalid timestep candidate on some worker, the retry guard, a retry timestep the clock
    /// cannot represent): the stores hold that step entry, the last accepted state
    pub halt: Option<String>,
}

/// where a worker's wall time went. `collectives` and `exchange` include the time spent
/// waiting for slower peers, so a load imbalance shows up there.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerTiming {
    /// kernels on owned tiles: recovery, ghost fill, timestep candidates, stages, restores
    pub compute: Duration,
    /// the timestep minimum and the rejection votes
    pub collectives: Duration,
    /// halo exchange: pack, grants, transfer, scatter
    pub exchange: Duration,
}

#[derive(Debug)]
pub enum WorkerError {
    Fabric(FabricError),
    Numerics(String),
    Refused(String),
    Checkpoint(String),
    /// two tiles hold different values for a shared interface face
    Interface(crate::plan_exchange::InterfaceMismatch),
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerError::Fabric(e) => write!(f, "fabric: {e}"),
            WorkerError::Numerics(d) => write!(f, "numerics: {d}"),
            WorkerError::Refused(d) => write!(f, "refused: {d}"),
            WorkerError::Checkpoint(d) => write!(f, "checkpoint: {d}"),
            WorkerError::Interface(m) => write!(f, "interface: {m}"),
        }
    }
}

impl std::error::Error for WorkerError {}

impl From<FabricError> for WorkerError {
    fn from(e: FabricError) -> Self {
        WorkerError::Fabric(e)
    }
}

/// whether a store carries immersed bodies but no horizon body: a step of such a run has no body
/// restore on the launched march
fn bodies_without_horizon<const D: usize, const DOF: usize, M: MemorySpace>(store: &FieldStore<D, DOF, M, f64>) -> bool {
    store.immersed.is_some() && crate::driver::horizon_request(store).is_none()
}

/// the whole run's horizon partial from every worker's: each worker supplies its own partial's
/// words at its own window, every worker receives every partial, and the merge is exact, so every
/// worker holds the integer the whole grid's sum holds whatever the partition
pub fn combine_horizon_partials<L: Link>(
    fabric: &mut Fabric<L>,
    workers: usize,
    local: &crate::driver::HorizonPartial,
    deadline: Duration,
) -> Result<crate::driver::HorizonPartial, WorkerError> {
    use crate::driver::HorizonPartial;
    let words = HorizonPartial::WORDS;
    let base = fabric.me().0 as u64 * words as u64;
    let entries: Vec<(u64, u64)> = local.to_words().into_iter().enumerate().map(|(i, w)| (base + i as u64, w)).collect();
    let assembled = fabric.assemble_windowed((workers * words) as u64, &entries, deadline).map_err(|e| fail(fabric, e.into()))?;
    let mut total = HorizonPartial::default();
    for w in 0..workers {
        let block = &assembled[w * words..(w + 1) * words];
        let partial = HorizonPartial::from_words(block).map_err(|e| {
            fail(fabric, WorkerError::Numerics(format!("worker {w}'s horizon partial is missing or malformed: {e}")))
        })?;
        total.merge(&partial);
    }
    Ok(total)
}

/// the launched worker's scope: 1D to 3D grids in host memory, a curved spacetime only where
/// `curved` admits it (the relativistic gas and magnetized gas), a horizon body where the kernel
/// set books its diagnostic shell, and tracers on the cartesian chart of a flat spacetime alone. the charts and settings a launch admits are decided
/// at the launch's front door.
pub fn refuse<const D: usize, const DOF: usize, M: MemorySpace, K: KernelSet<D, DOF, M, f64>>(
    stores: &[&FieldStore<D, DOF, M, f64>],
    _kernels: &[&K],
    curved: bool,
) -> Result<(), WorkerError> {
    let refused = |reason: &str| {
        WorkerError::Refused(format!("{reason} is outside the distributed first release"))
    };
    if !matches!(D, 1..=3) {
        return Err(refused(&format!("a {D}-dimensional grid")));
    }
    if M::IS_DEVICE_ACCESSIBLE {
        return Err(refused("device memory"));
    }
    for store in stores {
        let flat = store.geom.spacetime == symbi_geometry::Spacetime::Minkowski;
        let reason = if !flat && !curved {
            Some(format!("the {:?} spacetime", store.geom.spacetime))
        } else if let Some((_, radius)) = crate::driver::horizon_request(store)
            && _kernels.iter().any(|k| k.horizon_accretion(store, radius).is_none())
        {
            Some("a horizon body on a kernel set without the diagnostic-shell kernels".to_string())
        } else if (store.tracers.is_some() || store.continuous_tracers.is_some()) && !flat {
            Some(format!("a tracer population on the {:?} spacetime", store.geom.spacetime))
        } else if store.continuous_tracers.is_some() && store.geom.coords != symbi_geometry::Geometry::Cartesian {
            Some("an Ito tracer population on a curvilinear chart".to_string())
        } else if store.tracers.is_some() && store.geom.coords != symbi_geometry::Geometry::Cartesian {
            Some("a tracer population on a curvilinear chart".to_string())
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(refused(&reason));
        }
    }
    Ok(())
}

/// `fields[tile][field]` for every held tile, in schema order; unheld tiles are empty.
pub fn field_lists<'a, const D: usize, const DOF: usize, M: MemorySpace>(
    tiles: &[TileId],
    stores: &[&'a FieldStore<D, DOF, M, f64>],
    n_tiles: usize,
) -> Vec<Vec<&'a Field<f64, D, M>>> {
    let mut lists: Vec<Vec<&'a Field<f64, D, M>>> = (0..n_tiles).map(|_| Vec::new()).collect();
    for (tile, store) in tiles.iter().zip(stores) {
        lists[tile.0 as usize] = plan_fields(store);
    }
    lists
}

/// one exchange point: every axis phase in plan order, each waited to completion.
#[allow(clippy::too_many_arguments)]
pub fn exchange_point<const D: usize, const DOF: usize, M, T, L>(
    exchange: &PlanExchange<'_, D>,
    tiles: &[TileId],
    stores: &[&FieldStore<D, DOF, M, f64>],
    n_tiles: usize,
    devices: &[i32],
    transport: &T,
    fabric: &mut Fabric<L>,
    point: ExchangePoint,
    step: u64,
    attempt: u16,
    deadline: Duration,
    spent: &mut Duration,
) -> Result<(), FabricError>
where
    M: MemorySpace,
    T: HaloTransport,
    L: Link,
{
    let fields = field_lists(tiles, stores, n_tiles);
    exchange_point_fields(exchange, &fields, devices, transport, fabric, point, step, attempt, deadline, spent)
}

/// one forward-euler stage of length `dt` over the held tiles, before the prime of a run that
/// declares a stationary target: the seeded target's halos cross the cuts at the
/// equilibrium-prime point, every tile marks its troubled cells, the workers agree whether any
/// did and carry the flags across the cuts, and every tile corrects; the workers agree whether
/// any tile's redo rejected the stage, an error on every worker. the stage clocks stand at their
/// entry afterwards. the caller saves and restores the state.
#[allow(clippy::too_many_arguments)]
pub fn probe_equilibrium<const D: usize, const DOF: usize, M, K, T, L>(
    exchange: &PlanExchange<'_, D>,
    tiles: &[TileId],
    stores: &[&FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
    devices: &[i32],
    transport: &T,
    fabric: &mut Fabric<L>,
    deadline: Duration,
    dt_of: &mut dyn FnMut(&mut Fabric<L>) -> Result<f64, WorkerError>,
) -> Result<f64, WorkerError>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    let n_tiles = devices.len();
    let dev = |k: usize| devices[tiles[k].0 as usize];
    let mut spent = Duration::ZERO;
    for k in 0..stores.len() {
        with_device(dev(k), || {
            kernels[k].c2p(stores[k]);
            kernels[k].ghost_fill(stores[k]);
        });
    }
    exchange_point(exchange, tiles, stores, n_tiles, devices, transport, fabric, ExchangePoint::EquilibriumPrime, 0, 0, deadline, &mut spent)?;
    flip_polar_bands(exchange, tiles, stores, kernels, devices);
    for k in 0..stores.len() {
        with_device(dev(k), || kernels[k].ghost_fill(stores[k]));
    }
    // the caller anchors the primed target before the fluxes are evaluated
    let dt = dt_of(fabric)?;
    let args = crate::stage::StageArgs {
        dt,
        a0: 0.0,
        ac: 1.0,
        stage: 0,
        n_stages: 1,
        injection_weight: 1.0,
        allow_elision: false,
        exit_fraction: 1.0,
    };
    let mut troubled = 0u64;
    for k in 0..stores.len() {
        troubled += with_device(dev(k), || fold_stage_through_mark(stores[k], kernels[k], args, &mut |_| {}));
    }
    let any_troubled = fabric.collective(OpKind::Any, u64::from(troubled > 0), deadline)?;
    let mut cut_trouble = vec![false; stores.len()];
    if any_troubled != 0 {
        exchange_point(exchange, tiles, stores, n_tiles, devices, transport, fabric, ExchangePoint::EquilibriumTroubled, 0, 0, deadline, &mut spent)?;
        for k in 0..stores.len() {
            cut_trouble[k] = with_device(dev(k), || {
                exchange.trouble_regions(tiles[k]).iter().any(|region| kernels[k].fofc_flags_in(stores[k], region) > 0)
            });
        }
    }
    let mut rejected = false;
    for k in 0..stores.len() {
        // the probe is a diagnostic stage: each kernel's freeze count stands where the probe
        // found it, so the march's level-wide count starts equal on every tile of every worker
        let streak = kernels[k].freeze_streak();
        let outcome = with_device(dev(k), || fold_stage_from_correction(stores[k], kernels[k], args, cut_trouble[k], &mut |_| {}));
        kernels[k].set_freeze_streak(streak);
        rejected |= outcome != StageOutcome::Accepted;
        stores[k].reset_state_clock();
    }
    if fabric.collective(OpKind::Any, u64::from(rejected), deadline)? != 0 {
        return Err(WorkerError::Numerics("the stationary target's probe stage was rejected by the admissibility redo".into()));
    }
    Ok(dt)
}

/// one plan's share of a phase several plans open together: the plan, the fields of the
/// tiles it moves (in its schema order, indexed by its tile ids) and the device per tile
pub struct PlanPart<'a, const D: usize, M: MemorySpace> {
    pub plan: &'a PlanExchange<'a, D>,
    pub fields: &'a [Vec<&'a Field<f64, D, M>>],
    pub devices: &'a [i32],
    /// a fault injected into this worker's sends of the part, for the exactly-once gates
    pub fault: Option<StagingFault>,
    /// the fields the part writes, where they differ from the fields it reads
    pub dst_fields: Option<&'a [Vec<&'a Field<f64, D, M>>]>,
}

/// one exchange point over several plans: every axis phase opens once with the union of
/// the plans' sends and receives (the plans hold disjoint fabric id ranges), so two levels,
/// or the root state and its conserved seed, move in one phase of the table
#[allow(clippy::too_many_arguments)]
pub fn exchange_point_plans<const D: usize, M, T, L>(
    parts: &[PlanPart<'_, D, M>],
    transport: &T,
    fabric: &mut Fabric<L>,
    point: ExchangePoint,
    step: u64,
    attempt: u16,
    deadline: Duration,
    spent: &mut Duration,
) -> Result<(), FabricError>
where
    M: MemorySpace,
    T: HaloTransport,
    L: Link,
{
    let clock = Instant::now();
    let first = parts.first().expect("at least one plan");
    let mut run = || -> Result<(), FabricError> {
        for axis in 0..D {
            let epoch = first.plan.epoch(point, step, attempt, axis);
            let mut sends = Vec::new();
            let mut receives = Vec::new();
            for part in parts {
                let dst = part.dst_fields.unwrap_or(part.fields);
                let (s, r) = part.plan.stage_axis_into(part.fields, dst, part.devices, transport, fabric, epoch, part.fault)?;
                sends.extend(s);
                receives.extend(r);
            }
            fabric.open(symbi_fabric::PhaseSpec { epoch, sends: &sends, receives: &receives })?;
            fabric.wait(deadline)?;
            for part in parts {
                part.plan.unpack_axis_at(part.dst_fields.unwrap_or(part.fields), fabric, axis, point)?;
            }
            fabric.close()?;
        }
        Ok(())
    };
    let result = run();
    *spent += clock.elapsed();
    result
}

/// `exchange_point` over a prepared field list (`fields[tile][field]` in the plan's schema
/// order), for a plan over any field set (the conserved seed, a fine level's primitives)
#[allow(clippy::too_many_arguments)]
pub fn exchange_point_fields<const D: usize, M, T, L>(
    exchange: &PlanExchange<'_, D>,
    fields: &[Vec<&Field<f64, D, M>>],
    devices: &[i32],
    transport: &T,
    fabric: &mut Fabric<L>,
    point: ExchangePoint,
    step: u64,
    attempt: u16,
    deadline: Duration,
    spent: &mut Duration,
) -> Result<(), FabricError>
where
    M: MemorySpace,
    T: HaloTransport,
    L: Link,
{
    let clock = Instant::now();
    let mut run = || -> Result<(), FabricError> {
        for axis in 0..D {
            let epoch = exchange.epoch(point, step, attempt, axis);
            exchange.open_axis(&fields, devices, transport, fabric, epoch)?;
            fabric.wait(deadline)?;
            exchange.finish_axis_at(&fields, fabric, axis, point)?;
        }
        Ok(())
    };
    let result = run();
    *spent += clock.elapsed();
    result
}

/// the post-step context of a worker: each tile's work on its device, a count combined over
/// the tiles it holds, and a halo refresh through the post-step exchange point, the polar
/// bands signed and the physical ghosts refilled at the cut corners. a fabric failure during
/// the refresh is kept for the caller, which ends the session
struct WorkerStep<'a, 'p, const D: usize, K, T, L: Link> {
    exchange: &'a PlanExchange<'p, D>,
    tiles: &'a [TileId],
    n_tiles: usize,
    devices: &'a [i32],
    transport: &'a T,
    fabric: &'a mut Fabric<L>,
    kernels: &'a [&'a K],
    step: u64,
    attempt: u16,
    deadline: Duration,
    spent: &'a mut Duration,
    failure: Option<FabricError>,
    /// the stage whose excision this context refreshes, the stage count for the post-step
    excise_at: u8,
    /// the excision sweep count every worker agreed at the session's start
    excise_passes: usize,
}

impl<'a, 'p, const D: usize, const DOF: usize, M, K, T, L> crate::stage::StepContext<D, DOF, M>
    for WorkerStep<'a, 'p, D, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    fn on_tile(&self, k: usize, f: &mut dyn FnMut()) {
        with_device(self.devices[self.tiles[k].0 as usize], || f())
    }
    /// the excision sweep count, the only count the recipe combines: fixed by the run's radius
    /// and agreed by every worker at the session's start
    fn combine_max(&mut self, local: usize) -> usize {
        debug_assert!(local <= self.excise_passes, "a tile sweeps more often than the agreed count");
        self.excise_passes
    }
    fn refresh_halos(&mut self, stores: &[&FieldStore<D, DOF, M, f64>], at: crate::stage::RefreshAt) {
        use crate::stage::RefreshAt;
        if self.failure.is_some() {
            return;
        }
        // one sweep per excision, agreed at the session's start: each refresh has its own point
        let point = match at {
            RefreshAt::Phase => ExchangePoint::PostStep,
            RefreshAt::ExciseSweep(0) => ExchangePoint::ExciseSweep(self.excise_at),
            RefreshAt::ExciseSweep(pass) => unreachable!("excision sweep {pass} beyond the agreed single sweep"),
            RefreshAt::ExciseRebuild => ExchangePoint::ExciseRebuild(self.excise_at),
        };
        let moved = exchange_point(
            self.exchange,
            self.tiles,
            stores,
            self.n_tiles,
            self.devices,
            self.transport,
            self.fabric,
            point,
            self.step,
            self.attempt,
            self.deadline,
            self.spent,
        );
        if let Err(e) = moved {
            self.failure = Some(e);
            return;
        }
        flip_polar_bands(self.exchange, self.tiles, stores, self.kernels, self.devices);
        for (k, store) in stores.iter().enumerate() {
            with_device(self.devices[self.tiles[k].0 as usize], || self.kernels[k].ghost_fill(store));
        }
    }
}

/// the words one body's delta travels in: force, normal force, torque, then the accreted mass,
/// the previous step's mass, the accreted energy and the exported slip heat
/// the words of one body's delta: force, normal force, torque, and the four scalar exchanges
pub const fn body_delta_words(d: usize) -> usize {
    2 * d + 7
}

pub fn encode_body_delta<const D: usize>(delta: &symbi_ib::BodyDelta<f64, D>, out: &mut Vec<u64>) {
    out.extend((0..D).map(|ax| delta.force_delta[ax].to_bits()));
    out.extend((0..D).map(|ax| delta.force_normal_delta[ax].to_bits()));
    out.extend((0..3).map(|ax| delta.torque_delta[ax].to_bits()));
    for v in [delta.mass_delta, delta.prev_mass_delta, delta.energy_delta, delta.slip_heat_delta] {
        out.push(v.to_bits());
    }
}

pub fn decode_body_delta<const D: usize>(idx: usize, words: &[u64]) -> symbi_ib::BodyDelta<f64, D> {
    let at = |k: usize| f64::from_bits(words[k]);
    let mut delta = symbi_ib::BodyDelta::new(idx);
    delta.force_delta = symbi_algebra::Tensor::new(std::array::from_fn(|ax| at(ax)));
    delta.force_normal_delta = symbi_algebra::Tensor::new(std::array::from_fn(|ax| at(D + ax)));
    delta.torque_delta = symbi_algebra::Tensor::new(std::array::from_fn(|ax| at(2 * D + ax)));
    delta.mass_delta = at(2 * D + 3);
    delta.prev_mass_delta = at(2 * D + 4);
    delta.energy_delta = at(2 * D + 5);
    delta.slip_heat_delta = at(2 * D + 6);
    delta
}

/// negate the odd components of every polar ghost band this worker holds, after an antipodal
/// exchange filled it with rotated interior copies and before the physical refill reads the
/// band's edge. a plan without a seam has no band.
pub fn flip_polar_bands<const D: usize, const DOF: usize, M, K>(
    exchange: &PlanExchange<'_, D>,
    tiles: &[TileId],
    stores: &[&FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
    devices: &[i32],
) where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
{
    let plan = exchange.plan();
    let Some(seam) = plan.seam.as_ref() else {
        return;
    };
    for &(tile, side) in &plan.polar_bands {
        if let Some(k) = tiles.iter().position(|&t| t == tile) {
            with_device(devices[tile.0 as usize], || {
                kernels[k].flip_polar_band(stores[k], seam.mirror, side, plan.reach)
            });
        }
    }
}

/// audit the shared interface faces after the exchange `at`, booking the cost. a plan with
/// no face field along a cut has nothing to audit and opens no phase.
#[allow(clippy::too_many_arguments)]
pub fn audit_point<const D: usize, const DOF: usize, M, L>(
    exchange: &PlanExchange<'_, D>,
    tiles: &[TileId],
    stores: &[&FieldStore<D, DOF, M, f64>],
    n_tiles: usize,
    fabric: &mut Fabric<L>,
    at: AuditAt,
    step: u64,
    attempt: u16,
    deadline: Duration,
    report: &mut WorkerReport,
) -> Result<(), WorkerError>
where
    M: MemorySpace,
    L: Link,
{
    if !exchange.has_audits() {
        return Ok(());
    }
    let fields = field_lists(tiles, stores, n_tiles);
    let cost = exchange
        .audit(&fields, fabric, at, step, attempt, deadline)
        .map_err(|e| match e {
            AuditError::Fabric(e) => WorkerError::Fabric(e),
            AuditError::Mismatch(m) => WorkerError::Interface(m),
        })?;
    report.audits += 1;
    report.audit_cost.faces += cost.faces;
    report.audit_cost.wire_bytes += cost.wire_bytes;
    report.audit_cost.elapsed += cost.elapsed;
    Ok(())
}

/// a local fatal error: every peer is told before it is returned.
pub fn fail<L: Link>(fabric: &mut Fabric<L>, err: WorkerError) -> WorkerError {
    // the step in progress ends here: its guard acts are discarded, so the run scope closes
    // resolved and the error reaches the caller instead of the ledger's unresolved-step halt.
    if crate::guard_ledger::scope_is_open() {
        crate::guard_ledger::step_discard();
    }
    match &err {
        WorkerError::Fabric(FabricError::PeerAborted { .. } | FabricError::Disconnected { .. }) => {
        }
        other => fabric.abort(&other.to_string()),
    }
    err
}

/// one stage of a worker's tiles: each tile folds on its device under the run's stage hook, the
/// axis rings are assembled across the workers, the troubled flags cross the cuts at the stage's
/// troubled exchange point, and every agreement is one vote of this worker over the fabric
struct WorkerStage<'a, 'p, 'h, const D: usize, const DOF: usize, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    stores: &'a [&'a FieldStore<D, DOF, M, f64>],
    kernels: &'a [&'a K],
    tiles: &'a [TileId],
    devices: &'a [i32],
    exchange: &'a PlanExchange<'p, D>,
    transport: &'a T,
    fabric: &'a mut Fabric<L>,
    timing: &'a mut WorkerTiming,
    stage_hook: &'a mut crate::decomp::TileStageHook<'h, D, DOF, M>,
    args: StageArgs,
    axis_rings: bool,
    n_cells: [usize; D],
    ring_len: u64,
    deadline: Duration,
    iter: u64,
    attempt: u16,
    /// the global cell a test fault raises troubled at this stage
    injected: Option<[isize; D]>,
    /// a test fault rejects this stage
    forced: bool,
}

impl<const D: usize, const DOF: usize, M, K, T, L> WorkerStage<'_, '_, '_, D, DOF, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    fn dev(&self, k: usize) -> i32 {
        self.devices[self.tiles[k].0 as usize]
    }
}

impl<const D: usize, const DOF: usize, M, K, T, L> crate::collective_stage::StageCollective for WorkerStage<'_, '_, '_, D, DOF, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    type Error = WorkerError;

    fn tiles(&self) -> usize {
        self.stores.len()
    }

    fn mark(&mut self, k: usize) -> Result<u64, WorkerError> {
        let (store, kernel, args, dev, hook) = (self.stores[k], self.kernels[k], self.args, self.dev(k), &mut *self.stage_hook);
        Ok(with_device(dev, || fold_stage_through_mark(store, kernel, args, &mut |hp| hook(k, store, hp, args))))
    }

    fn axis_rings(&self) -> bool {
        self.axis_rings
    }

    fn mark_through_emf(&mut self, k: usize) -> Result<(), WorkerError> {
        let (store, kernel, args, dev, hook) = (self.stores[k], self.kernels[k], self.args, self.dev(k), &mut *self.stage_hook);
        with_device(dev, || fold_stage_through_emf(store, kernel, args, &mut |hp| hook(k, store, hp, args)));
        Ok(())
    }

    // every worker supplies the ring copies its tiles own to one assembled vector and reconciles
    // the rings it holds from it
    fn reconcile_rings(&mut self) -> Result<(), WorkerError> {
        let entries: Vec<(u64, u64)> =
            self.stores.iter().flat_map(|s| crate::axis_ring::ring_entries(s, self.n_cells)).collect();
        let clock = Instant::now();
        let deadline = self.deadline;
        let assembled = self
            .fabric
            .assemble_windowed(self.ring_len, &entries, deadline)
            .map_err(|e| fail(self.fabric, e.into()))?;
        self.timing.collectives += clock.elapsed();
        for s in self.stores {
            crate::axis_ring::apply_assembled(s, self.n_cells, &assembled, symbi_fabric::ABSENT);
        }
        Ok(())
    }

    fn mark_from_curl(&mut self, k: usize) -> Result<u64, WorkerError> {
        let (store, kernel, args, dev, hook) = (self.stores[k], self.kernels[k], self.args, self.dev(k), &mut *self.stage_hook);
        Ok(with_device(dev, || fold_stage_curl_through_mark(store, kernel, args, &mut |hp| hook(k, store, hp, args))))
    }

    fn injected_trouble(&mut self, k: usize) -> u64 {
        self.injected
            .and_then(|cell| self.exchange.local_cell(self.tiles[k], cell))
            .map_or(0, |local| raise_troubled_cell(self.stores[k], local))
    }

    fn any(&mut self, local: bool) -> Result<bool, WorkerError> {
        let clock = Instant::now();
        let deadline = self.deadline;
        let any = self.fabric.collective(OpKind::Any, u64::from(local), deadline).map_err(|e| fail(self.fabric, e.into()))?;
        self.timing.collectives += clock.elapsed();
        Ok(any != 0)
    }

    // a face on a cut takes the first-order flux when either cell sharing it is troubled: every
    // flag ghost on a cut is rewritten here, so none survives from an earlier stage or attempt
    fn exchange_trouble(&mut self) -> Result<Vec<bool>, WorkerError> {
        let deadline = self.deadline;
        exchange_point(
            self.exchange,
            self.tiles,
            self.stores,
            self.devices.len(),
            self.devices,
            self.transport,
            self.fabric,
            ExchangePoint::Troubled(self.args.stage as u8),
            self.iter,
            self.attempt,
            deadline,
            &mut self.timing.exchange,
        )
        .map_err(|e| fail(self.fabric, e.into()))?;
        Ok((0..self.stores.len())
            .map(|k| {
                with_device(self.dev(k), || {
                    self.exchange.trouble_regions(self.tiles[k]).iter().any(|region| self.kernels[k].fofc_flags_in(self.stores[k], region) > 0)
                })
            })
            .collect())
    }

    fn correct(&mut self, k: usize, cut_trouble: bool) -> Result<bool, WorkerError> {
        let (store, kernel, args, dev, hook) = (self.stores[k], self.kernels[k], self.args, self.dev(k), &mut *self.stage_hook);
        let outcome = with_device(dev, || fold_stage_from_correction(store, kernel, args, cut_trouble, &mut |hp| hook(k, store, hp, args)));
        Ok(outcome == StageOutcome::RetryStep)
    }

    fn freeze_streak(&self, k: usize) -> u32 {
        self.kernels[k].freeze_streak()
    }

    fn set_freeze_streak(&mut self, k: usize, streak: u32) {
        self.kernels[k].set_freeze_streak(streak);
    }

    fn forced_rejection(&mut self) -> bool {
        self.forced
    }
}

/// the stage tail of a worker's tiles: the excision through `WorkerStep` at the session-agreed
/// sweep count, the halos at the stage's exchange point (and the audit of every shared face when
/// asked), the tracers' layouts from the plan, and the tracers crossing to other workers' tiles
/// through the fabric
struct WorkerTileTail<'a, 's, 'p, const D: usize, const DOF: usize, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    stores: &'a mut [&'s mut FieldStore<D, DOF, M, f64>],
    kernels: &'a [&'a K],
    tiles: &'a [TileId],
    devices: &'a [i32],
    exchange: &'a PlanExchange<'p, D>,
    transport: &'a T,
    fabric: &'a mut Fabric<L>,
    timing: &'a mut WorkerTiming,
    report: &'a mut WorkerReport,
    ledgers: &'a mut Vec<std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>>,
    layouts: &'a [crate::tracers::TransportLayout<D>],
    discrete: bool,
    continuous: bool,
    audit: bool,
    excise_passes: usize,
    deadline: Duration,
    iter: u64,
    attempt: u16,
    stage: u8,
}

impl<const D: usize, const DOF: usize, M, K, T, L> crate::stage_tail::FlatStageTail<D, DOF, M>
    for WorkerTileTail<'_, '_, '_, D, DOF, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    type Error = WorkerError;

    // the sweep and the rebuild refreshed at their own points, before the stage's halos move
    fn excise(&mut self) -> Result<(), WorkerError> {
        if self.excise_passes == 0 {
            return Ok(());
        }
        let sh: Vec<&FieldStore<D, DOF, M, f64>> = self.stores.iter().map(|s| &**s).collect();
        let mut step = WorkerStep {
            exchange: self.exchange,
            tiles: self.tiles,
            n_tiles: self.devices.len(),
            devices: self.devices,
            transport: self.transport,
            fabric: &mut *self.fabric,
            kernels: self.kernels,
            step: self.iter,
            attempt: self.attempt,
            deadline: self.deadline,
            spent: &mut self.timing.exchange,
            failure: None,
            excise_at: self.stage,
            excise_passes: self.excise_passes,
        };
        let folded = crate::stage::fold_stage_excision(&sh, self.kernels, &mut step);
        if let Some(e) = step.failure.take() {
            return Err(fail(self.fabric, e.into()));
        }
        folded.map_err(|e| fail(self.fabric, WorkerError::Numerics(e.to_string())))
    }

    fn refresh_halos(&mut self) -> Result<(), WorkerError> {
        let sh: Vec<&FieldStore<D, DOF, M, f64>> = self.stores.iter().map(|s| &**s).collect();
        let (tiles, devices, kernels) = (self.tiles, self.devices, self.kernels);
        exchange_point(
            self.exchange,
            tiles,
            &sh,
            devices.len(),
            devices,
            self.transport,
            self.fabric,
            ExchangePoint::Stage(self.stage),
            self.iter,
            self.attempt,
            self.deadline,
            &mut self.timing.exchange,
        )
        .map_err(|e| fail(self.fabric, e.into()))?;
        flip_polar_bands(self.exchange, tiles, &sh, kernels, devices);
        for k in 0..sh.len() {
            with_device(devices[tiles[k].0 as usize], || kernels[k].ghost_fill(sh[k]));
        }
        if self.audit {
            audit_point(self.exchange, tiles, &sh, devices.len(), self.fabric, AuditAt::Stage(self.stage), self.iter, self.attempt, self.deadline, self.report)
                .map_err(|e| fail(self.fabric, e))?;
        }
        Ok(())
    }

    fn tracers(&self) -> (bool, bool) {
        (self.discrete, self.continuous)
    }

    fn tiles(&self) -> usize {
        self.stores.len()
    }

    fn tile(&mut self, k: usize) -> (&mut FieldStore<D, DOF, M, f64>, &mut std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>) {
        (&mut *self.stores[k], &mut self.ledgers[k])
    }

    fn layout(&self, k: usize) -> crate::tracers::TransportLayout<D> {
        self.layouts[k]
    }

    fn migrate(&mut self) -> Result<(), WorkerError> {
        let moves = crate::worker_tracers::migrate_discrete(self.stores, self.tiles, self.exchange.plan(), self.fabric, self.deadline)
            .map_err(|e| fail(self.fabric, e.into()))?;
        self.report.tracer_migrations.add(moves);
        Ok(())
    }

    fn blend(&mut self, ac: f64, stage: usize) {
        crate::worker_tracers::blend_discrete(self.stores, ac, stage);
    }

    fn failure(&mut self, detail: String) -> WorkerError {
        fail(self.fabric, WorkerError::Numerics(detail))
    }
}

/// the body step of a worker's tiles: the Alfven stiffness and the deltas assembled from every
/// tile of every worker, the halos at the post-body exchange point
struct WorkerBodies<'a, 's, 'p, const D: usize, const DOF: usize, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    stores: &'a mut [&'s mut FieldStore<D, DOF, M, f64>],
    kernels: &'a [&'a K],
    tiles: &'a [TileId],
    devices: &'a [i32],
    tile_devices: &'a [i32],
    exchange: &'a PlanExchange<'p, D>,
    transport: &'a T,
    fabric: &'a mut Fabric<L>,
    spent: &'a mut Duration,
    layouts: &'a [crate::tracers::TransportLayout<D>],
    deadline: Duration,
    iter: u64,
    attempt: u16,
}

impl<'s, const D: usize, const DOF: usize, M, K, T, L> crate::accepted_tail::FlatBodies<'s, D, DOF, M>
    for WorkerBodies<'_, 's, '_, D, DOF, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    type Error = WorkerError;
    type Kernels = K;

    fn held(&mut self) -> crate::accepted_tail::HeldTiles<'_, 's, D, DOF, M, K> {
        crate::accepted_tail::HeldTiles { stores: &mut *self.stores, kernels: self.kernels, devices: self.tile_devices }
    }

    // every tile's value at its tile id, so the maximum is over every tile of every worker
    fn max(&mut self, per_tile: &[f64]) -> Result<f64, WorkerError> {
        let n_tiles = self.devices.len();
        let local: Vec<(u64, u64)> = per_tile.iter().enumerate().map(|(k, v)| (self.tiles[k].0 as u64, v.to_bits())).collect();
        let assembled = self.fabric.assemble_windowed(n_tiles as u64, &local, self.deadline).map_err(|e| fail(self.fabric, e.into()))?;
        let mut max = 0.0_f64;
        for (tile, bits) in assembled.iter().enumerate() {
            if *bits == symbi_fabric::ABSENT {
                return Err(fail(self.fabric, WorkerError::Numerics(format!("no worker supplied tile {tile}'s Alfven stiffness"))));
            }
            max = max.max(f64::from_bits(*bits));
        }
        Ok(max)
    }

    fn refresh_halos(&mut self) -> Result<(), WorkerError> {
        let sh: Vec<&FieldStore<D, DOF, M, f64>> = self.stores.iter().map(|s| &**s).collect();
        exchange_point(
            self.exchange,
            self.tiles,
            &sh,
            self.devices.len(),
            self.devices,
            self.transport,
            self.fabric,
            ExchangePoint::PostBody,
            self.iter,
            self.attempt,
            self.deadline,
            self.spent,
        )
        .map_err(|e| fail(self.fabric, e.into()))?;
        flip_polar_bands(self.exchange, self.tiles, &sh, self.kernels, self.devices);
        for k in 0..sh.len() {
            with_device(self.tile_devices[k], || self.kernels[k].ghost_fill(sh[k]));
        }
        Ok(())
    }

    fn layout(&self, k: usize) -> crate::tracers::TransportLayout<D> {
        self.layouts[k]
    }

    // every tile's partials at its tile id, in flat tile order
    fn gather_deltas(&mut self) -> Result<Vec<Vec<symbi_ib::BodyDelta<f64, D>>>, WorkerError> {
        let n_tiles = self.devices.len();
        let n_bodies = self.stores.iter().find_map(|s| s.immersed.as_ref().map(|im| im.bodies.len())).unwrap_or(0);
        let words = body_delta_words(D);
        let per_tile = n_bodies * words;
        let mut entries: Vec<(u64, u64)> = Vec::with_capacity(self.stores.len() * per_tile);
        for (k, s) in self.stores.iter().enumerate() {
            let Some(im) = s.immersed.as_ref() else {
                continue;
            };
            let mut encoded = Vec::with_capacity(per_tile);
            for delta in im.diagnostics.consolidate() {
                encode_body_delta(&delta, &mut encoded);
            }
            let base = self.tiles[k].0 as u64 * per_tile as u64;
            entries.extend(encoded.into_iter().enumerate().map(|(i, bits)| (base + i as u64, bits)));
        }
        let assembled =
            self.fabric.assemble_windowed((n_tiles * per_tile) as u64, &entries, self.deadline).map_err(|e| fail(self.fabric, e.into()))?;
        let mut tile_deltas = Vec::with_capacity(n_tiles);
        for tile in 0..n_tiles {
            let block = &assembled[tile * per_tile..(tile + 1) * per_tile];
            if block.contains(&symbi_fabric::ABSENT) {
                return Err(fail(
                    self.fabric,
                    WorkerError::Numerics(format!(
                        "tile {tile}'s body feedback is missing from the assembled vector (or holds the absent bits)"
                    )),
                ));
            }
            tile_deltas.push((0..n_bodies).map(|b| decode_body_delta::<D>(b, &block[b * words..(b + 1) * words])).collect::<Vec<_>>());
        }
        Ok(tile_deltas)
    }

    fn failure(&mut self, detail: String) -> WorkerError {
        fail(self.fabric, WorkerError::Numerics(detail))
    }
}

/// the continuous-tracer step of a worker's tiles: the coefficients cross at their exchange point,
/// the leavers and the injection through the fabric
struct WorkerContinuous<'a, 's, 'p, 'c, const D: usize, const DOF: usize, M, T, L>
where
    M: MemorySpace,
    T: HaloTransport,
    L: Link,
{
    stores: &'a mut [&'s mut FieldStore<D, DOF, M, f64>],
    tiles: &'a [TileId],
    devices: &'a [i32],
    exchange: &'a PlanExchange<'p, D>,
    transport: &'a T,
    fabric: &'a mut Fabric<L>,
    timing: &'a mut WorkerTiming,
    report: &'a mut WorkerReport,
    context: &'a ContinuousTracerContext<'c, D>,
    ledgers: &'a [std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>],
    deadline: Duration,
    iter: u64,
    attempt: u16,
}

impl<'s, const D: usize, const DOF: usize, M, T, L> crate::accepted_tail::FlatContinuous<'s, D, DOF, M>
    for WorkerContinuous<'_, 's, '_, '_, D, DOF, M, T, L>
where
    M: MemorySpace,
    T: HaloTransport,
    L: Link,
{
    type Error = WorkerError;

    fn stores(&mut self) -> &mut [&'s mut FieldStore<D, DOF, M, f64>] {
        &mut *self.stores
    }

    fn domain(&self) -> ([(f64, f64); D], crate::state::Boundaries<D>) {
        (self.context.bounds, self.context.boundaries)
    }

    fn exchange_coefficients(&mut self) -> Result<(), WorkerError> {
        let n_tiles = self.devices.len();
        let mut fields: Vec<Vec<&Field<f64, D, M>>> = (0..n_tiles).map(|_| Vec::new()).collect();
        for (k, store) in self.stores.iter().enumerate() {
            let c = store.ito_coefficients.as_ref().expect("ito coefficients were materialized");
            fields[self.tiles[k].0 as usize] = c.drift.iter().chain(c.variance.iter()).chain(c.third.iter()).collect();
        }
        exchange_point_fields(
            self.context.coefficients,
            &fields,
            self.devices,
            self.transport,
            self.fabric,
            ExchangePoint::TracerCoefficients,
            self.iter,
            self.attempt,
            self.deadline,
            &mut self.timing.exchange,
        )
        .map_err(|e| fail(self.fabric, e.into()))
    }

    fn migrate(&mut self) -> Result<(), WorkerError> {
        let moves = crate::worker_tracers::migrate_continuous(self.stores, self.tiles, self.exchange.plan(), self.context.bounds, self.fabric, self.deadline)
            .map_err(|e| fail(self.fabric, WorkerError::Numerics(format!("continuous tracer migration: {e}"))))?;
        self.report.tracer_migrations.add(moves);
        Ok(())
    }

    fn spawn(&mut self, dt: f64) -> Result<(), WorkerError> {
        crate::worker_tracers::spawn_continuous_injection(self.stores, self.tiles, self.exchange.plan(), self.ledgers, dt, self.fabric, self.deadline)
            .map_err(|e| fail(self.fabric, WorkerError::Numerics(format!("continuous tracer injection: {e}"))))
    }

    fn failure(&mut self, detail: String) -> WorkerError {
        fail(self.fabric, WorkerError::Numerics(detail))
    }
}

/// the attempts of one root step on a worker: every held tile captures its retry snapshot, the
/// stages advance with the troubled marks, the freeze streak and the rejection agreed across the
/// workers, and a rejection restores every held tile and refreshes the halos at the rollback
/// exchange point
struct WorkerRootAttempt<'a, 's, 'p, 'h, const D: usize, const DOF: usize, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
{
    tiles: &'a [TileId],
    stores: &'a mut [&'s mut FieldStore<D, DOF, M, f64>],
    kernels: &'a [&'a K],
    devices: &'a [i32],
    exchange: &'a PlanExchange<'p, D>,
    transport: &'a T,
    fabric: &'a mut Fabric<L>,
    cfg: &'a WorkerConfig,
    stage_hook: &'a mut crate::decomp::TileStageHook<'h, D, DOF, M>,
    timing: &'a mut WorkerTiming,
    report: &'a mut WorkerReport,
    schedule: &'a [crate::driver::StageSpec],
    stages: &'a [(f64, f64)],
    multistage: bool,
    has_axis_rings: bool,
    has_tracers: bool,
    has_continuous: bool,
    n_cells: [usize; D],
    ring_len: u64,
    excise_passes: usize,
    tracer_layouts: &'a [crate::tracers::TransportLayout<D>],
    motion_n: Vec<symbi_geometry::MotionState<f64>>,
    a_n: Vec<f64>,
    iter: u64,
    /// the attempt's ordinal within the root step, the exchange points' and injections' label
    attempt: u16,
    restorable: bool,
    /// each tile's tracer injections of the last attempt, spawned once the step is accepted
    injection_ledgers: Vec<std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>>,
    /// each held tile's tracers at the step entry: the stages migrate records across the cuts and
    /// the workers, and every worker restoring its own tiles' entry populations returns each record
    /// to the tile that held it
    tracer_entry: Option<Vec<crate::tracers::TileTracerSnapshot<D>>>,
}

impl<const D: usize, const DOF: usize, M, K, T, L> crate::step_transaction::RootAttempt
    for WorkerRootAttempt<'_, '_, '_, '_, D, DOF, M, K, T, L>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
{
    type Error = WorkerError;

    fn rejectable(&mut self) -> bool {
        self.kernels.iter().any(|k| k.fofc_active()) || !self.cfg.injection.reject_at.is_empty()
    }

    fn capture(&mut self, dt: f64, restorable: bool) -> Result<(), WorkerError> {
        self.restorable = restorable;
        if restorable && (self.has_tracers || self.has_continuous) && self.tracer_entry.is_none() {
            let entry = self.stores.iter().map(|s| crate::tracers::TileTracerSnapshot::of(&**s)).collect::<Result<Vec<_>, _>>();
            self.tracer_entry = Some(entry.map_err(|e| fail(self.fabric, WorkerError::Numerics(format!("tracer retry snapshot: {e}"))))?);
        }
        let (tiles, devices) = (self.tiles, self.devices);
        let dev = |k: usize| devices[tiles[k].0 as usize];
        for k in 0..self.stores.len() {
            let (store, kernel) = (&*self.stores[k], self.kernels[k]);
            if restorable {
                with_device(dev(k), || kernel.snapshot_retry(store));
            }
            if self.multistage {
                with_device(dev(k), || kernel.snapshot(store));
            }
        }
        for s in self.stores.iter_mut() {
            s.dt = dt;
            if self.has_tracers {
                crate::tracers::snapshot_transport_state(&mut **s);
            }
            if self.has_continuous {
                let geometry = s.geom.block_geometry(symbi_geometry::Cartesian);
                crate::tracers::begin_ito_transport_store(&mut **s, &geometry)
                    .map_err(|e| fail(self.fabric, WorkerError::Numerics(format!("ito transport initialization: {e}"))))?;
            }
        }
        Ok(())
    }

    fn run(&mut self, dt: f64) -> Result<bool, WorkerError> {
        let (tiles, kernels, devices, exchange, transport, cfg) =
            (self.tiles, self.kernels, self.devices, self.exchange, self.transport, self.cfg);
        let (schedule, stages, tracer_layouts) = (self.schedule, self.stages, self.tracer_layouts);
        let (has_axis_rings, has_tracers, has_continuous) = (self.has_axis_rings, self.has_tracers, self.has_continuous);
        let (n_cells, ring_len, excise_passes, iter, attempt) = (self.n_cells, self.ring_len, self.excise_passes, self.iter, self.attempt);
        let a_n = &self.a_n;
        let stores = &mut *self.stores;
        let fabric = &mut *self.fabric;
        let stage_hook = &mut *self.stage_hook;
        let timing = &mut *self.timing;
        let report = &mut *self.report;
        let deadline = cfg.deadline;
        let n = stores.len();
        macro_rules! shared {
            () => {{
                let sh: Vec<&FieldStore<D, DOF, M, f64>> = stores.iter().map(|s| &**s).collect();
                sh
            }};
        }
        self.injection_ledgers = vec![std::collections::BTreeMap::new(); n];
        let injection_ledgers = &mut self.injection_ledgers;
        let mut rejected = false;
        for stage in schedule {
            for (k, s) in stores.iter_mut().enumerate() {
                let t_entry = s.time + stage.entry * dt;
                let law_value = s.motion_law.as_ref().map(|law| (law.a_at(t_entry), law.adot_at(t_entry)));
                crate::driver::set_stage_motion(&mut s.motion, law_value, dt, a_n[k], stage.entry);
            }
            let sh = shared!();
            let args = StageArgs {
                dt,
                a0: stage.a0,
                ac: stage.ac,
                stage: stage.index,
                n_stages: stages.len(),
                injection_weight: downstream_injection_weight(stages, stage.index),
                allow_elision: false,
                exit_fraction: crate::driver::stage_exit_fraction(stages, stage.index),
            };
            // the stage runs in two spans around the correction, every agreement one vote of this
            // worker for all of its tiles, so workers holding different numbers of tiles issue the
            // same collective sequence
            let injected = cfg
                .injection
                .trouble_at
                .filter(|&(at_step, at_stage, _)| attempt == 0 && (at_step, at_stage) == (iter, stage.index))
                .map(|(_, _, cell)| std::array::from_fn(|ax| cell[ax]));
            let mut level_stage = WorkerStage {
                stores: &sh,
                kernels,
                tiles,
                devices,
                exchange,
                transport,
                fabric: &mut *fabric,
                timing: &mut *timing,
                stage_hook: &mut *stage_hook,
                args,
                axis_rings: has_axis_rings,
                n_cells,
                ring_len,
                deadline,
                iter,
                attempt,
                injected,
                forced: attempt == 0 && cfg.injection.reject_at.contains(&(iter, stage.index)),
            };
            let agreed = crate::collective_stage::collective_stage(&mut level_stage)?;
            report.troubled_exchanges += u64::from(agreed.exchanged);
            report.neighbor_corrections += agreed.neighbor_corrections;
            if agreed.retry {
                rejected = true;
                break;
            }
            drop(sh);
            let mut tail = WorkerTileTail {
                stores: &mut *stores,
                kernels,
                tiles,
                devices,
                exchange,
                transport,
                fabric: &mut *fabric,
                timing: &mut *timing,
                report: &mut *report,
                ledgers: &mut *injection_ledgers,
                layouts: tracer_layouts,
                discrete: has_tracers,
                continuous: has_continuous,
                audit: cfg.audit_every_point,
                excise_passes,
                deadline,
                iter,
                attempt,
                stage: stage.index as u8,
            };
            crate::stage_tail::flat_stage_tail(&mut tail, stage)?;
        }
        if rejected {
            report.rejections += 1;
        }
        Ok(!rejected)
    }

    fn evidence(&mut self, rejections: u32) -> Vec<String> {
        // every worker books the agreed rejection with the evidence its own tiles hold, so every
        // worker reaches the guard's decision at the same attempt
        let mut evidence = Vec::new();
        for (k, store) in self.stores.iter().enumerate() {
            let site = format!("worker {}, tile {}, level 0", self.fabric.me().0, self.tiles[k].0);
            evidence.extend(crate::retry_guard::take_evidence(&**store, &site, rejections));
        }
        evidence
    }

    fn restore(&mut self) -> Result<(), WorkerError> {
        assert!(self.restorable, "a root step restored from an incomplete capture");
        for (k, s) in self.stores.iter_mut().enumerate() {
            s.motion = self.motion_n[k];
            if let Some(entry) = self.tracer_entry.as_ref() {
                if let Err(e) = entry[k].restore(&mut **s) {
                    return Err(fail(self.fabric, WorkerError::Numerics(format!("tracer retry restore: {e}"))));
                }
            }
        }
        let (tiles, devices, kernels) = (self.tiles, self.devices, self.kernels);
        let dev = |k: usize| devices[tiles[k].0 as usize];
        let sh: Vec<&FieldStore<D, DOF, M, f64>> = self.stores.iter().map(|s| &**s).collect();
        for k in 0..sh.len() {
            // a replay from anything but the step entry is a different computation dressed as a
            // retry: every worker ends the session instead. the attempt restores the tracers
            // itself; the kernel set restores the fields. the bodies of a run carrying a horizon
            // change after acceptance alone (the ledger, the feedback and the motion), so its
            // replay starts from the entry bodies; other immersed bodies have no restore here
            if !kernels[k].field_rollback_is_complete(sh[k]) || bodies_without_horizon(sh[k]) {
                return Err(fail(
                    self.fabric,
                    WorkerError::Numerics(
                        "a step was rejected on a kernel set without a complete rollback \
                         snapshot; the replay cannot start from the step entry"
                            .into(),
                    ),
                ));
            }
            with_device(dev(k), || kernels[k].restore_step(sh[k]));
            // the tile stands at its step entry again, the clock its ghost refills read
            sh[k].reset_state_clock();
        }
        let deadline = self.cfg.deadline;
        exchange_point(
            self.exchange,
            tiles,
            &sh,
            devices.len(),
            devices,
            self.transport,
            self.fabric,
            ExchangePoint::Rollback,
            self.iter,
            self.attempt,
            deadline,
            &mut self.timing.exchange,
        )
        .map_err(|e| fail(self.fabric, e.into()))?;
        flip_polar_bands(self.exchange, tiles, &sh, kernels, devices);
        for k in 0..sh.len() {
            with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
        }
        if self.cfg.audit_every_point {
            audit_point(self.exchange, tiles, &sh, devices.len(), self.fabric, AuditAt::Rollback, self.iter, self.attempt, deadline, self.report)
                .map_err(|e| fail(self.fabric, e))?;
        }
        self.attempt += 1;
        Ok(())
    }

    fn severe_streak(&mut self) -> u32 {
        self.stores[0].severe_streak
    }

    fn set_severe_streak(&mut self, streak: u32) {
        for store in self.stores.iter_mut() {
            store.severe_streak = streak;
        }
    }
}

/// end the run on a retry-guard failure: the guard's decision follows from the agreed timestep and
/// the agreed rejections, so every worker reaches it at the same point, and a collective there lets
/// each worker report its own evidence before any peer's abort ends its session
///
/// every worker enters the collective, whether or not it holds rejecting cells; it returns at the
/// deadline or on a lost peer, and the failure stands either way
/// the root timestep of a worker's step: a step every worker takes, or a halt every worker
/// takes at this step entry, with the invalid candidate when this worker holds it
pub enum AgreedStep {
    Step(AgreedTimestep),
    Halt(Option<InvalidCandidate>),
}

/// the timestep every worker of the session agrees on. each worker judges its own raw candidates
/// first, since a minimum over f64 prefers the finite operand; one vote over the workers then
/// halts the march on every worker at this step entry when any holds an invalid candidate, and
/// otherwise the valid minimum is reduced across the workers as `agree_timestep` reduces it
pub fn agree_worker_timestep<L: Link>(
    candidates: Vec<f64>,
    max_dt: f64,
    remaining: f64,
    fabric: &mut Fabric<L>,
    deadline: Duration,
    collectives: &mut Duration,
) -> Result<AgreedStep, WorkerError> {
    let bad = match candidates.iter().position(|v| !v.is_finite() || *v <= 0.0) {
        Some(index) => Some(InvalidCandidate { index, value: candidates[index] }),
        None if candidates.is_empty() => Some(InvalidCandidate { index: 0, value: f64::INFINITY }),
        None => None,
    };
    let clock = Instant::now();
    let any = fabric.collective(OpKind::Any, u64::from(bad.is_some()), deadline);
    *collectives += clock.elapsed();
    if any? != 0 {
        return Ok(AgreedStep::Halt(bad));
    }
    let agreed = agree_timestep(candidates, max_dt, remaining, |local| {
        let clock = Instant::now();
        let global = fabric.collective(OpKind::Min, local.to_bits(), deadline);
        *collectives += clock.elapsed();
        global.map(f64::from_bits)
    });
    match agreed {
        Ok(agreed) => Ok(AgreedStep::Step(agreed)),
        Err(TimestepFailure::Invalid(bad)) => Ok(AgreedStep::Halt(Some(bad))),
        Err(TimestepFailure::Collective(e)) => Err(e.into()),
    }
}

pub fn guard_failure<L: Link>(
    fabric: &mut Fabric<L>,
    failure: crate::retry_guard::RetryGuardFailure,
    injection: &Injection,
    deadline: Duration,
) -> Result<String, WorkerError> {
    if let Some(hold) = injection.hold_before_guard {
        std::thread::sleep(hold);
    }
    if injection.vanish_before_guard {
        std::process::exit(3);
    }
    // every live worker reaches the guard's decision at the same attempt; with a worker gone the
    // session ends through the abort path, carrying the guard's reason
    match fabric.collective(OpKind::Any, 1, deadline) {
        Ok(_) => Ok(failure.detail),
        Err(_) => Err(fail(fabric, WorkerError::Numerics(failure.detail))),
    }
}

/// the retry guard state every worker of the session holds: the same consecutive severely reduced
/// root steps, the same thresholds and the same retry exclusion, on every store of every worker.
/// each word is reduced to its
/// minimum over the workers as its two 32-bit halves sent as the exact integers one above them
/// (the minimum reduces over positive finite numbers); a worker whose own words differ from the
/// minimum, or whose stores disagree among themselves, votes, and one vote refuses the session
pub fn agree_retry_guard<const D: usize, const DOF: usize, M: MemorySpace, L: Link>(
    stores: &[&FieldStore<D, DOF, M, f64>],
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), WorkerError> {
    let words = |s: &FieldStore<D, DOF, M, f64>| -> [u64; 6] {
        let [kind, radius] = s.retry_exclusion.words();
        [
            u64::from(s.severe_streak),
            u64::from(s.retry_guard.max_rejections),
            s.retry_guard.severe_ratio.to_bits(),
            u64::from(s.retry_guard.sustained_steps),
            kind,
            radius,
        ]
    };
    let mine = words(stores[0]);
    let locally_agreed = stores.iter().all(|s| words(s) == mine);
    let mut agreed = [0u64; 6];
    for (w, word) in agreed.iter_mut().enumerate() {
        let mut halves = [0u64; 2];
        for (i, half) in halves.iter_mut().enumerate() {
            let own = ((mine[w] >> (32 * i)) & 0xffff_ffff) as f64 + 1.0;
            *half = (f64::from_bits(fabric.collective(OpKind::Min, own.to_bits(), deadline)?) - 1.0) as u64;
        }
        *word = halves[0] | halves[1] << 32;
    }
    let differs = !locally_agreed || mine != agreed;
    if fabric.collective(OpKind::Any, u64::from(differs), deadline)? != 0 {
        let show = |w: [u64; 6]| {
            let exclusion = RetryExclusion::from_words([w[4], w[5]])
                .map_or_else(|| format!("unrecognized ({}, {})", w[4], w[5]), |e| e.to_string());
            format!(
                "streak {} max_rejections {} severe_ratio {} sustained_steps {} retry exclusion {exclusion}",
                w[0],
                w[1],
                f64::from_bits(w[2]),
                w[3]
            )
        };
        return Err(WorkerError::Numerics(format!(
            "the retry guard or retry exclusion differs across the workers (this worker's stores agree: \
             {locally_agreed}; here {}, the minimum over the workers {}); the session does not start",
            show(mine),
            show(agreed)
        )));
    }
    Ok(())
}

/// evolve this worker's tiles from `cfg.start_time` until `cfg.t_final` or
/// `cfg.max_steps`. `tiles`, `stores`, `kernels` are parallel over the held
/// tiles; `devices` is indexed by tile id over the whole plan. `on_step` runs
/// after every accepted step with the held stores and the fabric, so a
/// checkpoint runs there; it may stop the run, and its error reaches every
/// peer through the abort relay. the session stays open on return, so the
/// caller writes its final checkpoint and then finishes the fabric.
#[allow(clippy::too_many_arguments)]
pub fn evolve_worker<const D: usize, const DOF: usize, M, K, T, L, F>(
    tiles: &[TileId],
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
    devices: &[i32],
    exchange: &PlanExchange<'_, D>,
    transport: &T,
    fabric: &mut Fabric<L>,
    cfg: &WorkerConfig,
    on_step: F,
) -> Result<WorkerReport, WorkerError>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
    F: FnMut(
        u64,
        f64,
        &[&FieldStore<D, DOF, M, f64>],
        &mut Fabric<L>,
    ) -> Result<ControlFlow<()>, WorkerError>,
{
    evolve_worker_traced(tiles, stores, kernels, devices, exchange, transport, fabric, cfg, None, &mut |_, _, _, _| {}, on_step)
}

/// what a continuous-tracer population needs beyond the state plan: the plan that carries its
/// coefficients across the cuts, and the whole domain's faces and boundary conditions the
/// tracers meet
pub struct ContinuousTracerContext<'a, const D: usize> {
    pub coefficients: &'a PlanExchange<'a, D>,
    pub bounds: [(f64, f64); D],
    pub boundaries: crate::state::Boundaries<D>,
}

/// `evolve_worker` with the context a continuous-tracer population rides
#[allow(clippy::too_many_arguments)]
pub fn evolve_worker_traced<const D: usize, const DOF: usize, M, K, T, L, F>(
    tiles: &[TileId],
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
    devices: &[i32],
    exchange: &PlanExchange<'_, D>,
    transport: &T,
    fabric: &mut Fabric<L>,
    cfg: &WorkerConfig,
    continuous: Option<&ContinuousTracerContext<'_, D>>,
    stage_hook: &mut crate::decomp::TileStageHook<'_, D, DOF, M>,
    mut on_step: F,
) -> Result<WorkerReport, WorkerError>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
    F: FnMut(
        u64,
        f64,
        &[&FieldStore<D, DOF, M, f64>],
        &mut Fabric<L>,
    ) -> Result<ControlFlow<()>, WorkerError>,
{
    let n = stores.len();
    assert_eq!(n, tiles.len(), "tiles/stores length mismatch");
    assert_eq!(n, kernels.len(), "stores/kernels length mismatch");
    let n_tiles = devices.len();
    let dev = |k: usize| devices[tiles[k].0 as usize];
    macro_rules! shared {
        () => {{
            let sh: Vec<&FieldStore<D, DOF, M, f64>> = stores.iter().map(|s| &**s).collect();
            sh
        }};
    }
    {
        let sh = shared!();
        // the relativistic gas and the relativistic magnetized gas carry a curved spacetime
        let curved = matches!(cfg.regime, crate::substrate_seam::RegimeKind::Rhd | crate::substrate_seam::RegimeKind::Rmhd);
        refuse(&sh, kernels, curved).map_err(|e| fail(fabric, e))?;
        // a rejection replays the step from its entry, which needs the whole entry state kept
        if !cfg.injection.reject_at.is_empty() {
            for (store, kernel) in sh.iter().zip(kernels) {
                if !kernel.field_rollback_is_complete(store) || bodies_without_horizon(store) {
                    return Err(fail(
                        fabric,
                        WorkerError::Refused(
                            "a forced rejection on a kernel set without a complete rollback \
                             snapshot is outside the distributed first release"
                                .into(),
                        ),
                    ));
                }
            }
        }
    }
    let deadline = cfg.deadline;
    let mut timing = WorkerTiming::default();
    let march_start = Instant::now();
    let mut in_callback = Duration::ZERO;
    let stages = cfg.timestepping.stages();
    if exchange.table().stages() as usize != stages.len() || exchange.table().substeps() != 0 {
        return Err(fail(
            fabric,
            WorkerError::Refused(format!(
                "the session's phase table holds {} root stages and {} fine substeps; this worker \
                 runs {} stages on a flat grid",
                exchange.table().stages(),
                exchange.table().substeps(),
                stages.len()
            )),
        ));
    }
    // whether any worker holds an edge on a coordinate axis, agreed once so every worker issues
    // the same collective sequence; the ring vector spans the whole grid of the plan's tiles
    let n_cells: [usize; D] = std::array::from_fn(|ax| {
        exchange
            .plan()
            .tiles
            .iter()
            .map(|t| (t.offset[ax] + t.interior[ax].size() as isize) as usize)
            .max()
            .unwrap_or(0)
    });
    let local_rings = {
        let sh = shared!();
        sh.iter().any(|s| !crate::axis_ring::axis_rings(*s).is_empty())
    };
    let has_axis_rings = fabric
        .collective(OpKind::Any, u64::from(local_rings), deadline)
        .map_err(|e| fail(fabric, e.into()))?
        != 0;
    let ring_len = {
        let sh = shared!();
        sh.first().map_or(0, |s| crate::axis_ring::ring_vector_len(n_cells, s.geom.coords))
    };
    let multistage = needs_step_snapshot(stages);
    let schedule = stage_schedule(stages);
    // the discrete tracer population: every tile of a traced run carries one, its transport laid
    // over the whole grid from the tile's place in the plan
    let has_tracers = stores.iter().any(|s| s.tracers.is_some());
    let has_continuous = stores.iter().any(|s| s.continuous_tracers.is_some());
    // whether each population's transport runs is the configuration's: every tile of a traced
    // run carries a set, empty or not. the workers agree on it once, so each issues the same
    // collectives and exchange points whatever particles it holds, and a disagreement ends the
    // session by name
    for (local, what) in [(has_tracers, "a discrete tracer population"), (has_continuous, "an Ito tracer population")] {
        let any_enabled = fabric
            .collective(OpKind::Any, u64::from(local), deadline)
            .map_err(|e| fail(fabric, e.into()))?
            != 0;
        let any_disabled = fabric
            .collective(OpKind::Any, u64::from(!local), deadline)
            .map_err(|e| fail(fabric, e.into()))?
            != 0;
        // both reductions reach every worker, so a split ends every session at this point
        if any_enabled && any_disabled {
            return Err(fail(fabric, WorkerError::Refused(format!("the workers disagree on carrying {what}"))));
        }
    }
    if has_continuous && continuous.is_none() {
        return Err(fail(
            fabric,
            WorkerError::Refused("an Ito tracer population without its coefficient plan is outside the distributed first release".into()),
        ));
    }
    let tracer_layouts: Vec<crate::tracers::TransportLayout<D>> = tiles
        .iter()
        .map(|tile| {
            let layout = exchange.plan().tiles.iter().find(|l| l.tile == *tile).expect("the plan lays out every tile");
            crate::tracers::TransportLayout {
                global_cells: n_cells,
                tile_offset: std::array::from_fn(|ax| layout.offset[ax] as usize),
                wraps: exchange.plan().wraps,
                level: 0,
            }
        })
        .collect();
    let mut injection_ledgers: Vec<std::collections::BTreeMap<crate::mass_transport::ContainerId, f64>>;

    let mut report = WorkerReport::default();

    // prime: primitives and physical ghosts, the cut halos, the physical ghosts again at the
    // cut corners, the audit of every shared interface face as the workers seeded it, then
    // the initial-condition check. a resumed tile keeps the primitives its checkpoint stored,
    // as the single grid's resume does
    {
        let sh = shared!();
        for k in 0..n {
            with_device(dev(k), || {
                if !sh[k].primitives_loaded {
                    kernels[k].c2p(sh[k]);
                }
                kernels[k].ghost_fill(sh[k]);
            });
        }
        exchange_point(
            exchange,
            tiles,
            &sh,
            n_tiles,
            devices,
            transport,
            fabric,
            ExchangePoint::Prime,
            0,
            0,
            deadline,
            &mut timing.exchange,
        )
        .map_err(|e| fail(fabric, e.into()))?;
        flip_polar_bands(exchange, tiles, &sh, kernels, devices);
        for k in 0..n {
            with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
        }
        audit_point(
            exchange, tiles, &sh, n_tiles, fabric, AuditAt::Prime, 0, 0, deadline, &mut report,
        )
        .map_err(|e| fail(fabric, e))?;
        for (k, store) in sh.iter().enumerate() {
            let err = crate::hydro_ops::scan_c2p_errors(store);
            if err.is_err() {
                return Err(fail(
                    fabric,
                    WorkerError::Numerics(format!(
                        "c2p failed on initial conditions (tile {:?}): {err}",
                        tiles[k]
                    )),
                ));
            }
        }
    }

    let guard_scope = crate::guard_ledger::open_scope();
    // every worker resumes the same retry guard: the streak a checkpoint recorded and the
    // configured thresholds
    {
        let sh = shared!();
        agree_retry_guard(&sh, fabric, cfg.deadline).map_err(|e| fail(fabric, e))?;
    }
    // the horizon excision's sweep count is fixed by the run's radius: every worker holds the
    // same one, checked from both sides, and the excision points are named for a single sweep
    let excise_passes = {
        let local = {
            let sh = shared!();
            (0..n).map(|k| kernels[k].excise_pass_count(sh[k])).max().unwrap_or(0)
        };
        let mut vote = |flag: bool| -> Result<bool, WorkerError> {
            Ok(fabric.collective(OpKind::Any, u64::from(flag), cfg.deadline).map_err(|e| fail(fabric, e.into()))? != 0)
        };
        let sweeping = vote(local > 0)?;
        let still = vote(local == 0)?;
        let repeated = vote(local > 1)?;
        if sweeping && still {
            return Err(fail(fabric, WorkerError::Refused("the workers disagree on whether the run excises".into())));
        }
        if repeated {
            return Err(fail(
                fabric,
                WorkerError::Refused("an excision of more than one sweep per step is outside the distributed first release".into()),
            ));
        }
        usize::from(sweeping)
    };
    let mut t = cfg.start_time;
    let mut iter: u64 = 0;
    // the cap bounds the root iteration itself, a resumed run's checkpointed iterations included
    while t < cfg.t_final && (cfg.max_steps == 0 || stores[0].iteration < cfg.max_steps) {
        // the timestep: the exact minimum over owned cells, then the fabric minimum over
        // workers, consumed as the coordinator's bits, then the final-time clamp.
        // every tile's candidate is validated before the fold: a minimum over f64 prefers the
        // finite operand, so one bad tile would vanish behind a good one.
        let (candidates, max_dt) = {
            let sh = shared!();
            let candidates: Vec<f64> = (0..n)
                .map(|k| {
                    let candidate = with_device(dev(k), || kernels[k].cfl(sh[k]));
                    if k == 0 && cfg.injection.invalid_cfl_at == Some(iter) { f64::NAN } else { candidate }
                })
                .collect();
            (candidates, sh[0].max_dt)
        };
        let agreed = agree_worker_timestep(candidates, max_dt, cfg.t_final - t, fabric, deadline, &mut timing.collectives)
            .map_err(|e| fail(fabric, e))?;
        let mut dt = match agreed {
            AgreedStep::Step(agreed) => agreed.dt,
            AgreedStep::Halt(Some(bad)) => {
                report.halt = Some(format!(
                    "invalid CFL candidate {:e} on tile {:?} at iter {iter} (time {t:.4e})",
                    bad.value, tiles[bad.index]
                ));
                break;
            }
            AgreedStep::Halt(None) => {
                report.halt = Some(format!("an invalid CFL candidate on another worker at iter {iter} (time {t:.4e})"));
                break;
            }
        };
        if let Some((at, bits)) = cfg.injection.cap_dt_at {
            if at == iter {
                dt = dt.min(f64::from_bits(bits));
            }
        }
        // homologous mesh motion: each stage's dispatches bind the geometry and grid-velocity
        // scalars from the tile's motion state, which holds a(t) at the stage's entry; the step
        // entry is kept for a rollback and for the step-tail advance
        let motion_n: Vec<_> = stores.iter().map(|s| s.motion).collect();
        let a_n: Vec<f64> = motion_n.iter().map(|motion| motion.a).collect();
        // the ancestry blend ranks the population's ids 0..next_id, the whole id set: the tile
        // counts and the id bound confirm it once per step
        if has_tracers {
            let sh = shared!();
            let whole = crate::worker_tracers::contiguous_ids(&sh, tiles, n_tiles, fabric, deadline)
                .map_err(|e| fail(fabric, e.into()))?;
            if !whole {
                return Err(fail(
                    fabric,
                    WorkerError::Numerics("the tracer ids are not the range 0..next_id the ancestry blend ranks".into()),
                ));
            }
        }
        // the root step from its candidate, the agreed timestep after its final-time clamp and caps
        let guard = stores[0].retry_guard;
        let mut root = WorkerRootAttempt {
            tiles,
            stores: &mut *stores,
            kernels,
            devices,
            exchange,
            transport,
            fabric: &mut *fabric,
            cfg,
            stage_hook: &mut *stage_hook,
            timing: &mut timing,
            report: &mut report,
            schedule: &schedule,
            stages,
            multistage,
            has_axis_rings,
            has_tracers,
            has_continuous,
            n_cells,
            ring_len,
            excise_passes,
            tracer_layouts: &tracer_layouts,
            motion_n,
            a_n: a_n.clone(),
            iter,
            attempt: 0,
            restorable: false,
            injection_ledgers: Vec::new(),
            tracer_entry: None,
        };
        let step = crate::step_transaction::take_root_step(&mut root, &guard, dt, t)?;
        injection_ledgers = std::mem::take(&mut root.injection_ledgers);
        // the accepted attempt's ordinal labels the step tail's exchange points
        let attempt = root.attempt;
        drop(root);
        let dt = match step {
            crate::step_transaction::RootStep::Accepted { dt } => dt,
            // the halted step left every store at its entry, on every worker alike
            crate::step_transaction::RootStep::Halted(crate::step_transaction::StepHalt::Guard(failure)) => {
                report.halt = Some(guard_failure(fabric, failure, &cfg.injection, deadline)?);
                break;
            }
            crate::step_transaction::RootStep::Halted(crate::step_transaction::StepHalt::Timestep { detail, .. }) => {
                report.halt = Some(detail);
                break;
            }
        };
        // the stage refresh moved a; the step-tail advance starts from the step-entry value, as
        // on the single grid
        for (k, s) in stores.iter_mut().enumerate() {
            s.motion.a = a_n[k];
        }
        if has_tracers {
            crate::worker_tracers::spawn_discrete_injection(stores, tiles, exchange.plan(), &injection_ledgers, fabric, deadline)
                .map_err(|e| fail(fabric, e.into()))?;
        }
        // the accepted step's continuous transport: each tile materializes its coefficients and
        // fills their physical ghosts, the cut ghosts cross, the tracers advance and meet the
        // domain's faces, the leavers cross to their cells' tiles, and the injection spawns
        if let Some(context) = continuous.filter(|_| has_continuous) {
            let mut continuous = WorkerContinuous {
                stores: &mut *stores,
                tiles,
                devices,
                exchange,
                transport,
                fabric: &mut *fabric,
                timing: &mut timing,
                report: &mut report,
                context,
                ledgers: &injection_ledgers,
                deadline,
                iter,
                attempt,
            };
            crate::accepted_tail::flat_continuous_step(&mut continuous, dt)?;
        }
        {
            // the post-step recipe on the accepted step: an operation that writes the conserved
            // interior recovers the primitives and refreshes the cut halos at the post-step
            // point, so the next step's first reader and the checkpoint see one state. a
            // recovery the accepted state cannot land ends the run on every worker
            let sh = shared!();
            let mut step = WorkerStep {
                exchange,
                tiles,
                n_tiles,
                devices,
                transport,
                fabric,
                kernels,
                step: iter,
                attempt,
                deadline,
                spent: &mut timing.exchange,
                failure: None,
                excise_at: exchange.table().stages(),
                excise_passes,
            };
            let folded = crate::stage::fold_post_step(&sh, kernels, dt, &mut step, |_| true);
            let failure = step.failure.take();
            if let Some(e) = failure {
                return Err(fail(fabric, e.into()));
            }
            if let Err(e) = folded {
                return Err(fail(fabric, WorkerError::Numerics(e.to_string())));
            }
        }
        // the root's horizon ledger, once per accepted step: every worker merges its tiles' exact
        // shell partials, the workers' partials merge into the whole grid's, and every tile books
        // the rates rounded once, as the single grid does
        if let Some((index, radius)) = stores.first().and_then(|s| crate::driver::horizon_request(&**s)) {
            let mut local = crate::driver::HorizonPartial::default();
            {
                let sh = shared!();
                for (k, store) in sh.iter().enumerate() {
                    let partial = with_device(devices[tiles[k].0 as usize], || {
                        crate::driver::horizon_partial_of(kernels[k], store, radius)
                    });
                    local.merge(&partial);
                }
            }
            let total = combine_horizon_partials(fabric, exchange.placement().workers as usize, &local, deadline)?;
            let (mdot, edot) = total.rates();
            for s in stores.iter_mut() {
                crate::driver::book_horizon_receipt(&mut **s, index, mdot, edot, dt);
            }
        }
        if stores.iter().any(|s| s.immersed.is_some()) {
            let tile_devices: Vec<i32> = tiles.iter().map(|t| devices[t.0 as usize]).collect();
            let mut bodies = WorkerBodies {
                stores: &mut *stores,
                kernels,
                tiles,
                devices,
                tile_devices: &tile_devices,
                exchange,
                transport,
                fabric: &mut *fabric,
                spent: &mut timing.exchange,
                layouts: &tracer_layouts,
                deadline,
                iter,
                attempt,
            };
            crate::accepted_tail::flat_body_step(&mut bodies, dt)?;
        }
        for s in stores.iter_mut() {
            advance_state_clock(&mut **s, dt);
        }
        if has_tracers {
            for (k, s) in stores.iter_mut().enumerate() {
                let geometry = s.geom.block_geometry(symbi_geometry::Cartesian);
                crate::tracers::refresh_derived_positions_store(&mut **s, &geometry, tracer_layouts[k]);
            }
        }
        (t, iter) = advance_clock(t, iter, dt);
        report.steps = iter;
        report.time = t;
        report.dt_sequence.push(dt);
        let sh = shared!();
        let clock = Instant::now();
        let flow = on_step(iter, t, &sh, fabric);
        in_callback += clock.elapsed();
        match flow {
            Ok(ControlFlow::Continue(())) => {}
            Ok(ControlFlow::Break(())) => break,
            Err(e) => return Err(fail(fabric, e)),
        }
    }
    // compute is the remainder: every kernel call on owned tiles, outside the fabric and the
    // step callback
    timing.compute = march_start
        .elapsed()
        .saturating_sub(timing.collectives + timing.exchange + in_callback);
    report.timing = timing;
    report.grant_high_water = (0..exchange.placement().workers)
        .map(|p| fabric.grants().high_water(WorkerId(p)))
        .collect();
    let accepted = guard_scope.accepted();
    report.troubled_cells = accepted.troubled_cells.total;
    report.frozen_cells = accepted.frozen_cells.total;
    Ok(report)
}
