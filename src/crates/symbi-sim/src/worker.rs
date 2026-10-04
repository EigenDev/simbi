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
    retry_timestep, select_timestep, stage_schedule,
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
    /// steps to take; zero is unbounded
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

/// the first-release scope: 2D and 3D newtonian and relativistic hydro and MHD,
/// adiabatic or globally isothermal, with or without a passive scalar and immersed bodies, on
/// the cartesian, spherical and cylindrical charts of a flat spacetime, in host memory, on a
/// static mesh, without horizon bodies, tracers, or excision.
pub fn refuse<const D: usize, const DOF: usize, M: MemorySpace, K: KernelSet<D, DOF, M, f64>>(
    stores: &[&FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
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
    for (store, kernel) in stores.iter().zip(kernels) {
        let reason = if store.geom.spacetime != symbi_geometry::Spacetime::Minkowski {
            Some(format!("the {:?} spacetime", store.geom.spacetime))
        } else if crate::driver::horizon_request(store).is_some() {
            Some("a horizon body".to_string())
        } else if store.continuous_tracers.is_some() && store.geom.coords != symbi_geometry::Geometry::Cartesian {
            Some("an Ito tracer population on a curvilinear chart".to_string())
        } else if store.tracers.is_some() && store.geom.coords != symbi_geometry::Geometry::Cartesian {
            Some("a tracer population on a curvilinear chart".to_string())
        } else if let Some(property) = crate::stage::post_step_refresh_needed(store, *kernel) {
            // the launched path carries no post-step exchange point, so an operation that
            // leaves the cut halos stale is refused by the recipe's own declaration
            Some(property.to_string())
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
        let outcome = with_device(dev(k), || fold_stage_from_correction(stores[k], kernels[k], args, cut_trouble[k], &mut |_| {}));
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
    fn combine_max(&mut self, local: usize) -> usize {
        local
    }
    fn refresh_halos(&mut self, stores: &[&FieldStore<D, DOF, M, f64>]) {
        if self.failure.is_some() {
            return;
        }
        let moved = exchange_point(
            self.exchange,
            self.tiles,
            stores,
            self.n_tiles,
            self.devices,
            self.transport,
            self.fabric,
            ExchangePoint::PostStep,
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

/// the per-step immersed-body bookkeeping on the held tiles, after the accepted step's post-step
/// recipe, in the order the in-process tiles keep: the wall relaxation's Alfven stiffness takes
/// the largest value of any tile; each tile's surface operations rescale its conserved state, its
/// primitives and ghost band are recovered and the cut halos carry the drained gas; each tile
/// reduces its feedback partials; every tile's partials cross the workers in one assembled
/// vector and are summed in flat tile order, the order the in-process tiles sum them in; and the
/// global delta is applied to every held tile's bodies alike
#[allow(clippy::too_many_arguments)]
fn body_step<const D: usize, const DOF: usize, M, K, T, L>(
    exchange: &PlanExchange<'_, D>,
    tiles: &[TileId],
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
    n_tiles: usize,
    devices: &[i32],
    transport: &T,
    fabric: &mut Fabric<L>,
    dt: f64,
    step: u64,
    attempt: u16,
    deadline: Duration,
    spent: &mut Duration,
    tracer_layouts: &[crate::tracers::TransportLayout<D>],
) -> Result<(), WorkerError>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
    symbi_geometry::Cartesian: symbi_geometry::Metric<f64, D>,
{
    let dev = |k: usize| devices[tiles[k].0 as usize];
    // the density each tile's sinks drain from, kept for the tracers the drain carries into the
    // sink reservoirs
    let traced = stores.iter().any(|s| s.tracers.is_some() || s.continuous_tracers.is_some());
    let mut accretion_density = Vec::new();
    {
        let sh: Vec<&FieldStore<D, DOF, M, f64>> = stores.iter().map(|s| &**s).collect();
        // the largest Alfven stiffness of any tile (zero off MHD), on every tile
        let local: Vec<(u64, u64)> = sh
            .iter()
            .enumerate()
            .map(|(k, s)| (tiles[k].0 as u64, crate::state::local_c_a2_max(*s).to_bits()))
            .collect();
        let assembled = fabric.assemble_windowed(n_tiles as u64, &local, deadline)?;
        let mut c_a2 = 0.0_f64;
        for (tile, bits) in assembled.iter().enumerate() {
            if *bits == symbi_fabric::ABSENT {
                return Err(WorkerError::Numerics(format!("no worker supplied tile {tile}'s Alfven stiffness")));
            }
            c_a2 = c_a2.max(f64::from_bits(*bits));
        }
        for s in &sh {
            if let Some(im) = s.immersed.as_ref() {
                im.set_c_a2_override(c_a2);
            }
        }
        if traced {
            accretion_density = sh.iter().map(|s| crate::tracers::snapshot_accretion_density(*s)).collect();
        }
        // the bodies are the same on every tile, so every worker takes the same branch
        let penalizes = sh.iter().any(|s| s.immersed.as_ref().is_some_and(|im| im.bodies.penalizes()));
        for k in 0..sh.len() {
            with_device(dev(k), || {
                kernels[k].penalize(sh[k], dt);
                if penalizes {
                    kernels[k].c2p(sh[k]);
                    kernels[k].ghost_fill(sh[k]);
                }
            });
        }
        if penalizes {
            exchange_point(
                exchange, tiles, &sh, n_tiles, devices, transport, fabric, ExchangePoint::PostBody, step, attempt,
                deadline, spent,
            )?;
            flip_polar_bands(exchange, tiles, &sh, kernels, devices);
            for k in 0..sh.len() {
                with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
            }
        }
        for k in 0..sh.len() {
            if sh[k].immersed.as_ref().is_some_and(|im| im.bodies.needs_feedback()) {
                with_device(dev(k), || kernels[k].body_feedback(sh[k], dt));
            }
        }
    }
    // each tile's tracers in the drained cells cross into the sink reservoirs at the step's end
    for (k, store) in stores.iter_mut().enumerate().filter(|_| traced) {
        let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
        let crossing_time = store.time + dt;
        crate::tracers::advance_accretion_transport_store(&mut **store, &geometry, tracer_layouts[k], &accretion_density[k], crossing_time)
            .and_then(|_| {
                crate::tracers::advance_continuous_accretion_transport_store(
                    &mut **store,
                    &geometry,
                    tracer_layouts[k],
                    &accretion_density[k],
                    crossing_time,
                )
            })
            .map_err(|e| WorkerError::Numerics(format!("tracer accretion transport: {e}")))?;
    }
    // every tile's partials, in flat tile order
    let n_bodies = stores.iter().find_map(|s| s.immersed.as_ref().map(|im| im.bodies.len())).unwrap_or(0);
    let words = body_delta_words(D);
    let per_tile = n_bodies * words;
    let mut entries: Vec<(u64, u64)> = Vec::with_capacity(stores.len() * per_tile);
    for (k, s) in stores.iter().enumerate() {
        let Some(im) = s.immersed.as_ref() else {
            continue;
        };
        let mut encoded = Vec::with_capacity(per_tile);
        for delta in im.diagnostics.consolidate() {
            encode_body_delta(&delta, &mut encoded);
        }
        let base = tiles[k].0 as u64 * per_tile as u64;
        entries.extend(encoded.into_iter().enumerate().map(|(i, bits)| (base + i as u64, bits)));
    }
    let assembled = fabric.assemble_windowed((n_tiles * per_tile) as u64, &entries, deadline)?;
    let mut tile_deltas = Vec::with_capacity(n_tiles);
    for tile in 0..n_tiles {
        let block = &assembled[tile * per_tile..(tile + 1) * per_tile];
        if block.contains(&symbi_fabric::ABSENT) {
            return Err(WorkerError::Numerics(format!(
                "tile {tile}'s body feedback is missing from the assembled vector (or holds the absent bits)"
            )));
        }
        tile_deltas.push((0..n_bodies).map(|b| decode_body_delta::<D>(b, &block[b * words..(b + 1) * words])).collect::<Vec<_>>());
    }
    let global = crate::decomp::merge_body_deltas(tile_deltas);
    crate::decomp::apply_global_body_step(stores, &global, dt);
    Ok(())
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

/// end the run on a retry-guard failure: the guard's decision follows from the agreed timestep and
/// the agreed rejections, so every worker reaches it at the same point, and a collective there lets
/// each worker report its own evidence before any peer's abort ends its session
///
/// every worker enters the collective, whether or not it holds rejecting cells; it returns at the
/// deadline or on a lost peer, and the failure stands either way
pub fn guard_failure<L: Link>(
    fabric: &mut Fabric<L>,
    failure: crate::retry_guard::RetryGuardFailure,
    injection: &Injection,
    deadline: Duration,
) -> WorkerError {
    if let Some(hold) = injection.hold_before_guard {
        std::thread::sleep(hold);
    }
    if injection.vanish_before_guard {
        std::process::exit(3);
    }
    let _ = fabric.collective(OpKind::Any, 1, deadline);
    fail(fabric, WorkerError::Numerics(failure.detail))
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
        refuse(&sh, kernels).map_err(|e| fail(fabric, e))?;
        // a rejection replays the step from its entry, which needs the whole entry state kept
        if !cfg.injection.reject_at.is_empty() {
            for (store, kernel) in sh.iter().zip(kernels) {
                if !kernel.rollback_snapshot_is_complete(store) {
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
    // the initial-condition check.
    {
        let sh = shared!();
        for k in 0..n {
            with_device(dev(k), || {
                kernels[k].c2p(sh[k]);
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
    let mut t = cfg.start_time;
    let mut iter: u64 = 0;
    while t < cfg.t_final && (cfg.max_steps == 0 || iter < cfg.max_steps) {
        // the timestep: the exact minimum over owned cells, then the fabric minimum over
        // workers, consumed as the coordinator's bits, then the final-time clamp.
        // every tile's candidate is validated before the fold: a minimum over f64 prefers the
        // finite operand, so one bad tile would vanish behind a good one.
        let mut local = f64::INFINITY;
        {
            let sh = shared!();
            for k in 0..n {
                let mut candidate = with_device(dev(k), || kernels[k].cfl(sh[k]));
                if k == 0 && cfg.injection.invalid_cfl_at == Some(iter) {
                    candidate = f64::NAN;
                }
                if !candidate.is_finite() || candidate <= 0.0 {
                    return Err(fail(
                        fabric,
                        WorkerError::Numerics(format!(
                            "invalid CFL candidate {candidate:e} on tile {:?} at iter {iter} (time {t:.4e})",
                            tiles[k]
                        )),
                    ));
                }
                local = local.min(candidate);
            }
        }
        let clock = Instant::now();
        let global = fabric
            .collective(OpKind::Min, local.to_bits(), deadline)
            .map_err(|e| fail(fabric, e.into()))?;
        timing.collectives += clock.elapsed();
        let mut dt = select_timestep([f64::from_bits(global)], cfg.t_final - t, iter, t)
            .map_err(|e| fail(fabric, WorkerError::Numerics(e.detail)))?;
        dt = crate::decomp::clamp_user_dt(dt, shared!()[0].max_dt);
        if let Some((at, bits)) = cfg.injection.cap_dt_at {
            if at == iter {
                dt = dt.min(f64::from_bits(bits));
            }
        }
        // the root step's candidate: the agreed timestep after its final-time clamp and caps
        let guard = stores[0].retry_guard;
        let mut ledger = crate::retry_guard::RetryLedger::new(dt);
        let mut attempt: u16 = 0;
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
        loop {
            {
                let sh = shared!();
                for k in 0..n {
                    if kernels[k].fofc_active() {
                        with_device(dev(k), || kernels[k].snapshot_retry(sh[k]));
                    }
                    if multistage {
                        with_device(dev(k), || kernels[k].snapshot(sh[k]));
                    }
                }
            }
            for s in stores.iter_mut() {
                s.dt = dt;
                if has_tracers {
                    crate::tracers::snapshot_transport_state(&mut **s);
                }
                if has_continuous {
                    let geometry = s.geom.block_geometry(symbi_geometry::Cartesian);
                    crate::tracers::begin_ito_transport_store(&mut **s, &geometry)
                        .map_err(|e| fail(fabric, WorkerError::Numerics(format!("ito transport initialization: {e}"))))?;
                }
            }
            injection_ledgers = vec![std::collections::BTreeMap::new(); n];
            let mut rejected = false;
            for stage in &schedule {
                for (k, s) in stores.iter_mut().enumerate() {
                    let t_entry = s.time + stage.entry * dt;
                    let law_value = s.motion_law.as_ref().map(|law| (law.a_at(t_entry), law.adot_at(t_entry)));
                    crate::driver::set_stage_motion(&mut s.motion, law_value, dt, a_n[k], stage.entry);
                }
                let sh = shared!();
                let mut retry = false;
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
                // the stage runs in two spans. the first ends with every owned tile's troubled
                // cells marked; the worker then makes one vote for all of them, so workers
                // holding different numbers of tiles issue the same collective sequence.
                let mut troubled = 0u64;
                if has_axis_rings {
                    // the edges on a coordinate axis are reconciled over the whole ring between
                    // the stage's complete EMF and its curl: every worker supplies the copies its
                    // tiles own to one assembled vector and reconciles the rings it holds from it
                    for k in 0..n {
                        with_device(dev(k), || fold_stage_through_emf(sh[k], kernels[k], args, &mut |hp| stage_hook(k, sh[k], hp, args)));
                    }
                    let entries: Vec<(u64, u64)> =
                        (0..n).flat_map(|k| crate::axis_ring::ring_entries(sh[k], n_cells)).collect();
                    let clock = Instant::now();
                    let assembled = fabric
                        .assemble_windowed(ring_len, &entries, deadline)
                        .map_err(|e| fail(fabric, e.into()))?;
                    timing.collectives += clock.elapsed();
                    for k in 0..n {
                        crate::axis_ring::apply_assembled(sh[k], n_cells, &assembled, symbi_fabric::ABSENT);
                        troubled += with_device(dev(k), || fold_stage_curl_through_mark(sh[k], kernels[k], args, &mut |hp| stage_hook(k, sh[k], hp, args)));
                    }
                } else {
                    for k in 0..n {
                        troubled += with_device(dev(k), || {
                            fold_stage_through_mark(sh[k], kernels[k], args, &mut |hp| stage_hook(k, sh[k], hp, args))
                        });
                    }
                }
                if attempt == 0 {
                    if let Some((at_step, at_stage, cell)) = cfg.injection.trouble_at {
                        if (at_step, at_stage) == (iter, stage.index) {
                            let cell: [isize; D] = std::array::from_fn(|ax| cell[ax]);
                            for k in 0..n {
                                if let Some(local) = exchange.local_cell(tiles[k], cell) {
                                    troubled += raise_troubled_cell(sh[k], local);
                                }
                            }
                        }
                    }
                }
                let clock = Instant::now();
                let any_troubled = fabric
                    .collective(OpKind::Any, u64::from(troubled > 0), deadline)
                    .map_err(|e| fail(fabric, e.into()))?;
                timing.collectives += clock.elapsed();
                // a face on a cut takes the first-order flux when either cell sharing it is
                // troubled, and those cells sit on different tiles: the flags cross the cuts
                // before any tile corrects, and a tile whose cut ghosts received a set flag
                // corrects with a clean interior. every flag ghost on a cut is rewritten here,
                // so none survives from an earlier stage or attempt.
                let mut cut_trouble = vec![false; n];
                let mut clean = vec![false; n];
                if any_troubled != 0 {
                    report.troubled_exchanges += 1;
                    for k in 0..n {
                        clean[k] = with_device(dev(k), || {
                            kernels[k].fofc_flags_in(sh[k], &sh[k].geom.interior) == 0
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
                        ExchangePoint::Troubled(stage.index as u8),
                        iter,
                        attempt,
                        deadline,
                        &mut timing.exchange,
                    )
                    .map_err(|e| fail(fabric, e.into()))?;
                    for k in 0..n {
                        cut_trouble[k] = with_device(dev(k), || {
                            exchange
                                .trouble_regions(tiles[k])
                                .iter()
                                .any(|region| kernels[k].fofc_flags_in(sh[k], region) > 0)
                        });
                    }
                }
                for k in 0..n {
                    let outcome = with_device(dev(k), || {
                        fold_stage_from_correction(
                            sh[k],
                            kernels[k],
                            args,
                            cut_trouble[k],
                            &mut |hp| stage_hook(k, sh[k], hp, args),
                        )
                    });
                    retry |= outcome == StageOutcome::RetryStep;
                    report.neighbor_corrections += u64::from(cut_trouble[k] && clean[k]);
                }
                if attempt == 0 && cfg.injection.reject_at.contains(&(iter, stage.index)) {
                    retry = true;
                }
                // the rejection is decided before this stage's halos move, so no halo is
                // computed from a rejected state.
                let clock = Instant::now();
                let reject = fabric
                    .collective(OpKind::Any, u64::from(retry), deadline)
                    .map_err(|e| fail(fabric, e.into()))?;
                timing.collectives += clock.elapsed();
                if reject != 0 {
                    rejected = true;
                    break;
                }
                exchange_point(
                    exchange,
                    tiles,
                    &sh,
                    n_tiles,
                    devices,
                    transport,
                    fabric,
                    ExchangePoint::Stage(stage.index as u8),
                    iter,
                    attempt,
                    deadline,
            &mut timing.exchange,
        )
                .map_err(|e| fail(fabric, e.into()))?;
                flip_polar_bands(exchange, tiles, &sh, kernels, devices);
                for k in 0..n {
                    with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
                }
                if cfg.audit_every_point {
                    let at = AuditAt::Stage(stage.index as u8);
                    audit_point(
                        exchange, tiles, &sh, n_tiles, fabric, at, iter, attempt, deadline,
                        &mut report,
                    )
                    .map_err(|e| fail(fabric, e))?;
                }
                drop(sh);
                // the stage's tracer transport on the exchanged state: each tile books its
                // boundary and source injection and moves its tracers' owners, the leavers cross
                // to their owners' tiles, the ancestry blend runs, and the reverted cross again
                if has_tracers || has_continuous {
                    for k in 0..n {
                        let store = &mut *stores[k];
                        let geometry = store.geom.block_geometry(symbi_geometry::Cartesian);
                        let layout = tracer_layouts[k];
                        if has_continuous {
                            crate::tracers::accumulate_ito_transport_stage_store(store, &geometry, stage.ac)
                                .map_err(|e| fail(fabric, WorkerError::Numerics(format!("ito transport accumulation: {e}"))))?;
                        }
                        let mut injections = crate::tracers::boundary_injection_transfers_store(&*store, &geometry, layout);
                        injections.extend(crate::tracers::source_injection_transfers_store(
                            &*store, &geometry, layout, stage.a0, stage.ac,
                        ));
                        crate::tracers::fold_injection_ledger(&mut injection_ledgers[k], injections, stage.ac);
                        if has_tracers {
                            crate::tracers::advance_stage_mass_transport_tile(store, &geometry, layout, stage.a0, stage.ac, stage.index)
                                .map_err(|e| fail(fabric, WorkerError::Numerics(format!("tracer mass transport: {e}"))))?;
                        }
                    }
                }
                if has_tracers {
                    let moves = crate::worker_tracers::migrate_discrete(stores, tiles, exchange.plan(), fabric, deadline)
                        .map_err(|e| fail(fabric, e.into()))?;
                    report.tracer_migrations.add(moves);
                    crate::worker_tracers::blend_discrete(stores, stage.ac, stage.index);
                    let moves = crate::worker_tracers::migrate_discrete(stores, tiles, exchange.plan(), fabric, deadline)
                        .map_err(|e| fail(fabric, e.into()))?;
                    report.tracer_migrations.add(moves);
                }
            }
            if !rejected {
                crate::guard_ledger::step_commit();
                break;
            }
            crate::guard_ledger::step_discard();
            // a migrated tracer lives in another tile's storage, and no per-tile restore puts it
            // back, so a traced run ends instead of replaying
            if has_tracers || has_continuous {
                return Err(fail(
                    fabric,
                    WorkerError::Numerics(
                        "a step was rejected with tracers attached: cross-tile tracer migration has no per-tile inverse".into(),
                    ),
                ));
            }
            report.rejections += 1;
            // every worker books the agreed rejection with the evidence its own tiles hold, so
            // every worker reaches the guard's decision at the same attempt
            {
                let mut evidence = Vec::new();
                for (k, store) in stores.iter().enumerate() {
                    let site = format!("worker {}, tile {}, level 0", fabric.me().0, tiles[k].0);
                    evidence.extend(crate::retry_guard::take_evidence(&**store, &site, ledger.rejections()));
                }
                if let Err(failure) = ledger.reject(&guard, dt, evidence) {
                    return Err(guard_failure(fabric, failure, &cfg.injection, deadline));
                }
            }
            for (k, s) in stores.iter_mut().enumerate() {
                s.motion = motion_n[k];
            }
            {
                let sh = shared!();
                for k in 0..n {
                    // a replay from anything but the step entry is a different computation
                    // dressed as a retry: every worker ends the session instead
                    if !kernels[k].rollback_snapshot_is_complete(sh[k]) {
                        return Err(fail(
                            fabric,
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
                exchange_point(
                    exchange,
                    tiles,
                    &sh,
                    n_tiles,
                    devices,
                    transport,
                    fabric,
                    ExchangePoint::Rollback,
                    iter,
                    attempt,
                    deadline,
            &mut timing.exchange,
        )
                .map_err(|e| fail(fabric, e.into()))?;
                flip_polar_bands(exchange, tiles, &sh, kernels, devices);
                for k in 0..n {
                    with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
                }
                if cfg.audit_every_point {
                    audit_point(
                        exchange, tiles, &sh, n_tiles, fabric, AuditAt::Rollback, iter, attempt,
                        deadline, &mut report,
                    )
                    .map_err(|e| fail(fabric, e))?;
                }
            }
            dt =
                retry_timestep(dt, t).map_err(|e| fail(fabric, WorkerError::Numerics(e.detail)))?;
            attempt += 1;
        }
        // the stage refresh moved a; the step-tail advance starts from the step-entry value, as
        // on the single grid
        for (k, s) in stores.iter_mut().enumerate() {
            s.motion.a = a_n[k];
        }
        // the sustained-reduction guard, once per accepted root step, before the post-step
        // recipe and any checkpoint publish it; the agreed timesteps give every worker the same
        // decision
        {
            let mut streak = stores[0].severe_streak;
            if let Err(failure) = crate::retry_guard::accept(&guard, &mut streak, &ledger, dt) {
                return Err(guard_failure(fabric, failure, &cfg.injection, deadline));
            }
            for store in stores.iter_mut() {
                store.severe_streak = streak;
            }
        }
        if has_tracers {
            crate::worker_tracers::spawn_discrete_injection(stores, tiles, exchange.plan(), &injection_ledgers, fabric, deadline)
                .map_err(|e| fail(fabric, e.into()))?;
        }
        // the accepted step's continuous transport: each tile materializes its coefficients and
        // fills their physical ghosts, the cut ghosts cross, the tracers advance and meet the
        // domain's faces, the leavers cross to their cells' tiles, and the injection spawns
        if let Some(context) = continuous.filter(|_| has_continuous) {
            for s in stores.iter_mut() {
                let geometry = s.geom.block_geometry(symbi_geometry::Cartesian);
                crate::tracers::materialize_ito_coefficients_store(&mut **s, &geometry)
                    .map_err(|e| fail(fabric, WorkerError::Numerics(format!("ito coefficient materialization: {e}"))))?;
                crate::tracers::fill_ito_coefficient_boundaries_host(
                    s.ito_coefficients.as_ref().expect("ito coefficients were materialized"),
                    &s.geom,
                    s.boundaries,
                )
                .map_err(|e| fail(fabric, WorkerError::Numerics(format!("ito coefficient boundaries: {e}"))))?;
            }
            {
                let sh = shared!();
                let mut fields: Vec<Vec<&Field<f64, D, M>>> = (0..n_tiles).map(|_| Vec::new()).collect();
                for (k, store) in sh.iter().enumerate() {
                    let c = store.ito_coefficients.as_ref().expect("ito coefficients were materialized");
                    fields[tiles[k].0 as usize] = c.drift.iter().chain(c.variance.iter()).chain(c.third.iter()).collect();
                }
                exchange_point_fields(
                    context.coefficients,
                    &fields,
                    devices,
                    transport,
                    fabric,
                    ExchangePoint::TracerCoefficients,
                    iter,
                    attempt,
                    deadline,
                    &mut timing.exchange,
                )
                .map_err(|e| fail(fabric, e.into()))?;
            }
            for s in stores.iter_mut() {
                let mut tracers = s.continuous_tracers.take().expect("every tile of a traced run carries tracers");
                let (scale_start, scale_end, offset_start, offset_end) = crate::tracers::continuous_tracer_mesh_step(&**s, dt);
                let advanced = crate::tracers::advance_continuous_tracers(
                    &mut tracers,
                    s.ito_coefficients.as_ref().expect("ito coefficients were materialized"),
                    &s.geom,
                    scale_start,
                    scale_end,
                    offset_start,
                    offset_end,
                    dt,
                )
                .and_then(|_| {
                    let bounds = crate::tracers::map_continuous_tracer_bounds(context.bounds, scale_end, offset_end);
                    crate::tracers::apply_continuous_boundaries_host(&mut tracers, bounds, context.boundaries)
                });
                s.continuous_tracers = Some(tracers);
                advanced.map_err(|e| fail(fabric, WorkerError::Numerics(format!("ito tracer advancement: {e}"))))?;
            }
            let moves = crate::worker_tracers::migrate_continuous(stores, tiles, exchange.plan(), context.bounds, fabric, deadline)
                .map_err(|e| fail(fabric, WorkerError::Numerics(format!("continuous tracer migration: {e}"))))?;
            report.tracer_migrations.add(moves);
            crate::worker_tracers::spawn_continuous_injection(stores, tiles, exchange.plan(), &injection_ledgers, dt, fabric, deadline)
                .map_err(|e| fail(fabric, WorkerError::Numerics(format!("continuous tracer injection: {e}"))))?;
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
        if stores.iter().any(|s| s.immersed.is_some()) {
            body_step(
                exchange, tiles, stores, kernels, n_tiles, devices, transport, fabric, dt, iter, attempt, deadline,
                &mut timing.exchange, &tracer_layouts,
            )
            .map_err(|e| fail(fabric, e))?;
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
