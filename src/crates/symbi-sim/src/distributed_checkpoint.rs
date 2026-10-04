// =============================================================================
// distributed_checkpoint.rs
//
// one checkpoint written by every worker through the coordinator. the
// coordinator opens the temporary and declares every dataset, every worker
// walks its owned tiles with the same block walker the local writer uses and
// sends each block under credit, the coordinator writes each block as it
// arrives and records it in a coverage ledger sized from the plan, and the
// file takes its name only once the ledger shows every owned box exactly once.
// two collectives bracket the write: CheckpointOpen carries the coordinator's
// open result to every worker, CheckpointClose carries the ledger's verdict
// and the publish result.
//
// usage:
//  distributed_checkpoint::<R, D, DOF, Mem, L>(&mut fabric, &plan, &placement, &identity,
//      &level, &tile_ids, &CheckpointRequest { .. })?;
// =============================================================================

use crate::atlas::{ExchangePlan, Placement, TileId, domain_of};
use crate::checkpoint::{
    CheckpointStream, GlobalGrid, LevelTiles, PhysicsIdentity, cell_datasets, chunk_region,
    dataset_paths, owned_blocks, owned_cell_region, owned_face_region, slab_start_and_count,
    touches_domain_edges,
};
use std::time::{Duration, Instant};
use symbi_fabric::{Fabric, FabricError, Link, OpKind, WorkerId};
use symbi_hydro::regime::Regime;
use symbi_io::{IoError, Metadata};
use symbi_xpu::MemorySpace;

/// the fixed part of a block payload: dataset u16, tile u32, then start and count.
fn block_header_len(d: usize) -> usize {
    2 + 4 + 8 * d
}

/// the block credit that lets one full block of `budget` cells travel.
pub fn block_credit_for(budget: usize, d: usize) -> u32 {
    (block_header_len(d) + 8 * budget) as u32
}

#[derive(Debug)]
pub enum CheckpointError {
    Fabric(FabricError),
    Io(IoError),
    /// the collective verdict was negative: a worker or the coordinator failed
    Refused(String),
    /// two tiles hold different values for a shared interface face of the state to be written
    Interface(crate::plan_exchange::InterfaceMismatch),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Fabric(e) => write!(f, "checkpoint fabric: {e}"),
            CheckpointError::Io(e) => write!(f, "checkpoint io: {e}"),
            CheckpointError::Refused(d) => write!(f, "checkpoint refused: {d}"),
            CheckpointError::Interface(m) => write!(f, "checkpoint audit: {m}"),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<FabricError> for CheckpointError {
    fn from(e: FabricError) -> Self {
        CheckpointError::Fabric(e)
    }
}

impl From<IoError> for CheckpointError {
    fn from(e: IoError) -> Self {
        CheckpointError::Io(e)
    }
}

/// what a worker writes and, for the gates, what it withholds or repeats.
pub struct CheckpointRequest<'a> {
    pub path: &'a str,
    pub extras: &'a Metadata,
    /// cells per block; the block credit must admit `block_credit_for(budget, D)` bytes
    pub budget: usize,
    pub deadline: Duration,
    /// withhold this worker's block of this index: the missing-block gate
    pub drop_block: Option<usize>,
    /// send this worker's first block twice: the overlapping-block gate
    pub duplicate_first_block: bool,
    /// send every block and then vote against the close: the negative-vote gate
    pub vote_no: bool,
}

fn encode_block<const D: usize>(
    dataset: u16,
    tile: TileId,
    start: &[usize],
    count: &[usize],
    data: &[f64],
    out: &mut Vec<u8>,
) {
    out.clear();
    out.extend_from_slice(&dataset.to_le_bytes());
    out.extend_from_slice(&tile.0.to_le_bytes());
    for s in start {
        out.extend_from_slice(&(*s as u32).to_le_bytes());
    }
    for c in count {
        out.extend_from_slice(&(*c as u32).to_le_bytes());
    }
    for v in data {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
}

struct Block<'a> {
    dataset: usize,
    tile: TileId,
    start: Vec<usize>,
    count: Vec<usize>,
    data: &'a [u8],
}

fn decode_block<const D: usize>(payload: &[u8]) -> Option<Block<'_>> {
    if payload.len() < block_header_len(D) {
        return None;
    }
    let dataset = u16::from_le_bytes(payload[0..2].try_into().ok()?) as usize;
    let tile = TileId(u32::from_le_bytes(payload[2..6].try_into().ok()?));
    let mut pos = 6;
    let mut read = || {
        let v = u32::from_le_bytes(payload[pos..pos + 4].try_into().expect("four bytes")) as usize;
        pos += 4;
        v
    };
    let start: Vec<usize> = (0..D).map(|_| read()).collect();
    let count: Vec<usize> = (0..D).map(|_| read()).collect();
    Some(Block {
        dataset,
        tile,
        start,
        count,
        data: &payload[block_header_len(D)..],
    })
}

/// one (tile, dataset) entry of the coverage ledger.
struct Entry {
    owned_start: Vec<usize>,
    owned_count: Vec<usize>,
    expected: usize,
    capacity: usize,
    boxes: Vec<(Vec<usize>, Vec<usize>)>,
    received: usize,
}

impl Entry {
    fn inside(&self, start: &[usize], count: &[usize]) -> bool {
        start
            .iter()
            .zip(count)
            .zip(self.owned_start.iter().zip(&self.owned_count))
            .all(|((s, c), (os, oc))| *s >= *os && s + c <= os + oc)
    }

    fn overlaps_recorded(&self, start: &[usize], count: &[usize]) -> bool {
        self.boxes.iter().any(|(bs, bc)| {
            start
                .iter()
                .zip(count)
                .zip(bs.iter().zip(bc))
                .all(|((s, c), (os, oc))| *s < os + oc && *os < s + c)
        })
    }

    fn complete(&self) -> bool {
        self.received == self.expected
    }
}

/// the ledger over every tile of the plan and every dataset, sized from the plan.
fn build_ledger<R, const D: usize, const DOF: usize, Mem>(
    plan: &ExchangePlan<D>,
    grid: &GlobalGrid<D>,
    authority: &crate::state::FieldStore<D, DOF, Mem, f64>,
    budget: usize,
    origin: [isize; D],
) -> Vec<Vec<Entry>>
where
    R: Regime<f64, D>,
    Mem: MemorySpace,
{
    let n_cell = cell_datasets::<R, D, DOF, Mem>(authority).len();
    let n_datasets = dataset_paths::<R, D, DOF, Mem>(0, authority).len();
    let ng = grid.ng as isize;
    plan.tiles
        .iter()
        .map(|layout| {
            let interior = domain_of(&layout.interior);
            let alloc = domain_of(&layout.allocated);
            let size: [usize; D] = std::array::from_fn(|ax| layout.interior[ax].size());
            let offset: [isize; D] = std::array::from_fn(|ax| layout.offset[ax] + origin[ax]);
            let (first, last) = touches_domain_edges(&offset, &size, grid);
            let shift: [isize; D] = std::array::from_fn(|ax| offset[ax] - layout.interior[ax].lo - grid.interior_lo[ax]);
            (0..n_datasets)
                .map(|i| {
                    let (region, file_of): (_, Box<dyn Fn(usize, isize) -> usize>) = if i < n_cell {
                        (
                            owned_cell_region(&interior, &alloc, &first, &last),
                            Box::new(move |ax, c| (c + shift[ax] + ng) as usize),
                        )
                    } else {
                        (
                            owned_face_region(&interior, i - n_cell, &last),
                            Box::new(move |ax, c| (c + shift[ax]) as usize),
                        )
                    };
                    let (owned_start, owned_count) = slab_start_and_count(&region, &*file_of);
                    let capacity = chunk_region(&region, budget).len();
                    Entry {
                        owned_start,
                        owned_count,
                        expected: region.volume(),
                        capacity,
                        boxes: Vec::with_capacity(capacity),
                        received: 0,
                    }
                })
                .collect()
        })
        .collect()
}

/// record one block in the ledger and write it: the block's tile must be owned by
/// `from`, the box must lie inside that tile's owned box for the dataset, must overlap no
/// recorded box, and must fit the ledger's capacity.
fn admit_block(
    ledger: &mut [Vec<Entry>],
    datasets: &[String],
    placement: &Placement,
    stream: &mut CheckpointStream,
    from: WorkerId,
    block: &Block<'_>,
    budget: usize,
) -> Result<(), FabricError> {
    let reject = |detail: String| FabricError::protocol(from, detail);
    let Some(per_tile) = ledger.get_mut(block.tile.0 as usize) else {
        return Err(reject(format!("block for unknown tile {:?}", block.tile)));
    };
    if placement.owner_of(block.tile) != from {
        return Err(reject(format!(
            "block for {:?}, which {from:?} does not hold",
            block.tile
        )));
    }
    let Some(entry) = per_tile.get_mut(block.dataset) else {
        return Err(reject(format!(
            "block for unknown dataset {}",
            block.dataset
        )));
    };
    let volume: usize = block.count.iter().product();
    if block.data.len() != 8 * volume || volume > budget || volume == 0 {
        return Err(reject(format!(
            "block of {} bytes for a box of {volume} cells",
            block.data.len()
        )));
    }
    if !entry.inside(&block.start, &block.count) {
        return Err(reject(format!(
            "block outside the owned box of {:?}",
            block.tile
        )));
    }
    if entry.overlaps_recorded(&block.start, &block.count) {
        return Err(reject(format!(
            "block overlaps one already received for {:?}",
            block.tile
        )));
    }
    if entry.boxes.len() >= entry.capacity {
        return Err(reject(format!(
            "more blocks than the ledger holds for {:?}",
            block.tile
        )));
    }
    let values: Vec<f64> = block
        .data
        .chunks_exact(8)
        .map(|c| f64::from_bits(u64::from_le_bytes(c.try_into().expect("eight bytes"))))
        .collect();
    stream
        .block(
            &datasets[block.dataset],
            &block.start,
            &block.count,
            &values,
        )
        .map_err(|e| FabricError::protocol(from, format!("write of a block failed: {e}")))?;
    entry.boxes.push((block.start.clone(), block.count.clone()));
    entry.received += volume;
    Ok(())
}

/// one level of a distributed checkpoint: the plan whose tiles partition it, their placement,
/// and the whole-grid index of the plan's index origin (zero on the root; a fine level's plan
/// indexes its patch from zero)
pub struct LevelPlan<'a, const D: usize> {
    pub plan: &'a ExchangePlan<D>,
    pub placement: &'a Placement,
    pub origin: [isize; D],
    /// the plan's ids of this worker's tiles, parallel to the level's tiles
    pub tile_ids: &'a [TileId],
}

/// write one checkpoint across every worker. `level` holds this worker's tiles and the
/// global grid; `tile_ids` is parallel to `level.tiles`.
#[allow(clippy::too_many_arguments)]
pub fn distributed_checkpoint<R, const D: usize, const DOF: usize, Mem, L>(
    fabric: &mut Fabric<L>,
    plan: &ExchangePlan<D>,
    placement: &Placement,
    identity: &PhysicsIdentity,
    level: &LevelTiles<'_, D, DOF, Mem>,
    tile_ids: &[TileId],
    req: &CheckpointRequest<'_>,
) -> Result<(), CheckpointError>
where
    R: Regime<f64, D>,
    Mem: MemorySpace,
    L: Link,
{
    let plans = [LevelPlan { plan, placement, origin: [0; D], tile_ids }];
    distributed_checkpoint_levels::<R, D, DOF, Mem, L>(fabric, &plans, identity, std::slice::from_ref(level), req)
}

/// write one checkpoint of several levels across every worker: `levels[i]` holds this
/// worker's tiles of level `i` (none on a worker holding no tile of a fine level, whose header
/// then names the level's clock and field set), `plans[i]` the plan that partitions it. a
/// block's dataset index counts the datasets of every level in order, so it names its level.
pub fn distributed_checkpoint_levels<R, const D: usize, const DOF: usize, Mem, L>(
    fabric: &mut Fabric<L>,
    plans: &[LevelPlan<'_, D>],
    identity: &PhysicsIdentity,
    levels: &[LevelTiles<'_, D, DOF, Mem>],
    req: &CheckpointRequest<'_>,
) -> Result<(), CheckpointError>
where
    R: Regime<f64, D>,
    Mem: MemorySpace,
    L: Link,
{
    assert_eq!(plans.len(), levels.len(), "one plan per level");
    for (p, l) in plans.iter().zip(levels) {
        assert_eq!(l.tiles.len(), p.tile_ids.len(), "tiles and ids are parallel");
    }
    let budget = req.budget;
    if block_credit_for(budget, D) > fabric.block_credit() {
        return Err(CheckpointError::Refused(format!(
            "a block of {budget} cells needs {} bytes of credit; {} agreed",
            block_credit_for(budget, D),
            fabric.block_credit()
        )));
    }
    let workers = plans[0].placement.workers as usize;
    // every level's datasets, numbered in level order: (level, index within the level, path)
    let mut datasets: Vec<(usize, usize, String)> = Vec::new();
    for (li, level) in levels.iter().enumerate() {
        for (di, path) in dataset_paths::<R, D, DOF, Mem>(li, level.template()).into_iter().enumerate() {
            datasets.push((li, di, path));
        }
    }
    let global_index = |li: usize, path: &str| -> usize {
        datasets
            .iter()
            .position(|(l, _, p)| *l == li && p == path)
            .expect("the walker names a declared dataset")
    };
    // the tracer population of every worker's tiles on every level, gathered level by level so the
    // coordinator writes it whole; every tile of a traced run carries one on each of its levels, and
    // every worker holds root tiles, so the root decides for every worker whether the gathers run
    let traced = levels[0].tiles.iter().any(|view| view.state.tracers.is_some());
    let population = if !traced {
        None
    } else {
        let mut sets = Vec::new();
        for (plan, level) in plans.iter().zip(levels) {
            let local: Vec<(usize, &crate::tracers::TracerSet<D>)> = plan
                .tile_ids
                .iter()
                .zip(&level.tiles)
                .filter_map(|(tile, view)| view.state.tracers.as_ref().map(|set| (tile.0 as usize, set)))
                .collect();
            sets.extend(crate::worker_tracers::gather_discrete(fabric, plan.plan.n_tiles(), &local, req.deadline)?);
        }
        Some(sets)
    };
    let local: Vec<(usize, &crate::tracers::ContinuousTracerSet<D, Mem>)> = plans[0]
        .tile_ids
        .iter()
        .zip(&levels[0].tiles)
        .filter_map(|(tile, view)| view.state.continuous_tracers.as_ref().map(|set| (tile.0 as usize, set)))
        .collect();
    let continuous = if local.is_empty() {
        None
    } else {
        Some(
            crate::worker_tracers::gather_continuous(fabric, plans[0].plan.n_tiles(), &local, req.deadline)
                .map_err(|e| CheckpointError::Refused(format!("continuous tracer gather: {e}")))?,
        )
    };
    if fabric.is_coordinator() {
        let opened = CheckpointStream::open::<R, D, DOF, Mem>(
            identity,
            levels,
            req.path,
            req.extras,
            budget,
            false,
            population.as_deref(),
            continuous.as_deref(),
        );
        let ok = fabric.collective(OpKind::CheckpointOpen, u64::from(opened.is_ok()), req.deadline)?;
        let mut stream = opened?;
        if ok == 0 {
            return Err(CheckpointError::Refused("a worker could not begin the checkpoint".into()));
        }
        let mut ledgers: Vec<Vec<Vec<Entry>>> = plans
            .iter()
            .zip(levels)
            .map(|(p, l)| build_ledger::<R, D, DOF, Mem>(p.plan, &l.grid, l.template(), budget, p.origin))
            .collect();
        let admit = |ledgers: &mut Vec<Vec<Vec<Entry>>>, stream: &mut CheckpointStream, from: WorkerId, block: &Block<'_>| {
            let Some((li, di, _)) = datasets.get(block.dataset) else {
                return Err(FabricError::protocol(from, format!("block for unknown dataset {}", block.dataset)));
            };
            let local = Block { dataset: *di, tile: block.tile, start: block.start.clone(), count: block.count.clone(), data: block.data };
            let level_paths: Vec<String> = datasets.iter().filter(|(l, _, _)| l == li).map(|(_, _, p)| p.clone()).collect();
            admit_block(&mut ledgers[*li], &level_paths, plans[*li].placement, stream, from, &local, budget)
        };
        let me = fabric.me();
        for (li, (p, level)) in plans.iter().zip(levels).enumerate() {
            for (tile, view) in p.tile_ids.iter().zip(&level.tiles) {
                let mut sink = |path: &str, start: &[usize], count: &[usize], data: &[f64]| {
                    let mut bytes = Vec::new();
                    encode_block::<D>(global_index(li, path) as u16, *tile, start, count, data, &mut bytes);
                    let block = decode_block::<D>(&bytes).expect("an encoded block decodes");
                    admit(&mut ledgers, &mut stream, me, &block).map_err(|e| IoError::Backend(e.to_string()))
                };
                owned_blocks::<R, D, DOF, Mem>(li, view, &level.grid, budget, &mut sink)?;
            }
        }
        let start = Instant::now();
        while fabric.contributions_to_current() < workers - 1 {
            fabric.service()?;
            if let Some((from, _seq, payload)) = fabric.staged_block() {
                let block = decode_block::<D>(payload).ok_or_else(|| FabricError::protocol(from, "malformed block payload"))?;
                admit(&mut ledgers, &mut stream, from, &block)?;
                fabric.release_block()?;
            }
            if start.elapsed() > req.deadline {
                return Err(CheckpointError::Fabric(FabricError::Deadline {
                    phase: symbi_fabric::Deadline::Checkpoint,
                    peer: None,
                    epoch: None,
                    pending: workers - 1 - fabric.contributions_to_current(),
                }));
            }
            std::thread::sleep(Duration::from_micros(50));
        }
        let complete = ledgers.iter().all(|l| l.iter().all(|per_tile| per_tile.iter().all(Entry::complete)));
        let workers_ok = fabric.current_contributions_all_nonzero();
        let published = complete && workers_ok && stream.publish().is_ok();
        let ok = fabric.collective(OpKind::CheckpointClose, u64::from(published), req.deadline)?;
        if ok == 0 {
            return Err(CheckpointError::Refused(if !complete {
                "the coverage ledger is incomplete; no file was published".into()
            } else if !workers_ok {
                "a worker reported a failed write; no file was published".into()
            } else {
                "the checkpoint could not be published".into()
            }));
        }
        Ok(())
    } else {
        let ok = fabric.collective(OpKind::CheckpointOpen, 1, req.deadline)?;
        if ok == 0 {
            return Err(CheckpointError::Refused("the coordinator could not open the checkpoint".into()));
        }
        let mut bytes = Vec::with_capacity(block_credit_for(budget, D) as usize);
        let mut index = 0usize;
        let mut sent_ok = true;
        for (li, (p, level)) in plans.iter().zip(levels).enumerate() {
            for (tile, view) in p.tile_ids.iter().zip(&level.tiles) {
                let mut sink = |path: &str, start: &[usize], count: &[usize], data: &[f64]| {
                    let this = index;
                    index += 1;
                    if req.drop_block == Some(this) {
                        return Ok(());
                    }
                    encode_block::<D>(global_index(li, path) as u16, *tile, start, count, data, &mut bytes);
                    let repeats = if this == 0 && req.duplicate_first_block { 2 } else { 1 };
                    for _ in 0..repeats {
                        if let Err(e) = fabric.send_block(&bytes, req.deadline) {
                            sent_ok = false;
                            return Err(IoError::Backend(format!("fabric: {e}")));
                        }
                    }
                    Ok(())
                };
                owned_blocks::<R, D, DOF, Mem>(li, view, &level.grid, budget, &mut sink)?;
            }
        }
        let vote = sent_ok && !req.vote_no;
        let ok = fabric.collective(OpKind::CheckpointClose, u64::from(vote), req.deadline)?;
        if ok == 0 {
            return Err(CheckpointError::Refused("the checkpoint was not published".into()));
        }
        Ok(())
    }
}


/// what an audited checkpoint cost: the interface audit apart from the checkpoint's own
/// collectives and I/O.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AuditedCheckpointCost {
    pub audit: crate::plan_exchange::AuditCost,
    /// the checkpoint from its open vote to its publication
    pub checkpoint: std::time::Duration,
}

/// the audit a checkpoint's state has, or gets, before it is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditBasis {
    /// the state as the prime exchange left it. the prime audit certified every shared face
    /// of that state, and the invariant the caller asserts by choosing this is that no
    /// stage, exchange, rollback or restore has touched a face field since: the caller
    /// writes this checkpoint before its march begins. no second audit runs.
    Primed,
    /// the state that completed step `k`. an audit runs first, at an epoch that sorts after
    /// every exchange of step `k` and before the next step's.
    AfterStep(u64),
    /// the state that completed step `k`, which an `AfterStep(k)` audit already certified for an
    /// earlier checkpoint of the same step: nothing has touched a face field since, and the audit
    /// epoch of step `k` is spent. no second audit runs.
    Certified(u64),
}

impl AuditBasis {
    /// `Primed` when the plan audits nothing, whatever the basis: there is no face to compare.
    fn filter_audited(self, has_audits: bool) -> Self {
        if has_audits { self } else { AuditBasis::Primed }
    }
}

/// audit every shared interface face of the state to be written, then write the checkpoint.
/// the audit runs before the checkpoint transaction opens, so a mismatch ends the session
/// with no temporary created and no file published.
#[allow(clippy::too_many_arguments)]
pub fn audited_checkpoint<R, const D: usize, const DOF: usize, Mem, L>(
    fabric: &mut Fabric<L>,
    exchange: &crate::plan_exchange::PlanExchange<'_, D>,
    identity: &PhysicsIdentity,
    level: &LevelTiles<'_, D, DOF, Mem>,
    tile_ids: &[TileId],
    req: &CheckpointRequest<'_>,
    basis: AuditBasis,
) -> Result<AuditedCheckpointCost, CheckpointError>
where
    R: Regime<f64, D>,
    Mem: MemorySpace,
    L: Link,
{
    let mut cost = AuditedCheckpointCost::default();
    if let AuditBasis::AfterStep(step) = basis.filter_audited(exchange.has_audits()) {
        let stores: Vec<&crate::state::FieldStore<D, DOF, Mem, f64>> =
            level.tiles.iter().map(|t| t.state).collect();
        let fields = crate::worker::field_lists(tile_ids, &stores, exchange.plan().n_tiles());
        cost.audit = exchange
            .audit(
                &fields,
                fabric,
                crate::atlas::AuditAt::Checkpoint,
                step,
                u16::MAX,
                req.deadline,
            )
            .map_err(|e| match e {
                crate::plan_exchange::AuditError::Fabric(e) => CheckpointError::Fabric(e),
                crate::plan_exchange::AuditError::Mismatch(m) => CheckpointError::Interface(m),
            })?;
    }
    let clock = std::time::Instant::now();
    distributed_checkpoint::<R, D, DOF, Mem, L>(
        fabric,
        exchange.plan(),
        exchange.placement(),
        identity,
        level,
        tile_ids,
        req,
    )?;
    cost.checkpoint = clock.elapsed();
    Ok(cost)
}
