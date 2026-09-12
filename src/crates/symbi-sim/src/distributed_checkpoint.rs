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
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Fabric(e) => write!(f, "checkpoint fabric: {e}"),
            CheckpointError::Io(e) => write!(f, "checkpoint io: {e}"),
            CheckpointError::Refused(d) => write!(f, "checkpoint refused: {d}"),
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
            let (first, last) = touches_domain_edges(&layout.offset, &size, grid);
            let shift: [isize; D] = std::array::from_fn(|ax| {
                layout.offset[ax] - layout.interior[ax].lo - grid.interior_lo[ax]
            });
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
    assert_eq!(
        level.tiles.len(),
        tile_ids.len(),
        "tiles and ids are parallel"
    );
    let budget = req.budget;
    if block_credit_for(budget, D) > fabric.block_credit() {
        return Err(CheckpointError::Refused(format!(
            "a block of {budget} cells needs {} bytes of credit; {} agreed",
            block_credit_for(budget, D),
            fabric.block_credit()
        )));
    }
    let workers = placement.workers as usize;
    let authority = level.tiles[0].state;
    let datasets = dataset_paths::<R, D, DOF, Mem>(0, authority);
    if fabric.is_coordinator() {
        let opened = CheckpointStream::open::<R, D, DOF, Mem>(
            identity,
            std::slice::from_ref(level),
            req.path,
            req.extras,
            budget,
            false,
        );
        let ok = fabric.collective(
            OpKind::CheckpointOpen,
            u64::from(opened.is_ok()),
            req.deadline,
        )?;
        let mut stream = opened?;
        if ok == 0 {
            return Err(CheckpointError::Refused(
                "a worker could not begin the checkpoint".into(),
            ));
        }
        let mut ledger = build_ledger::<R, D, DOF, Mem>(plan, &level.grid, authority, budget);
        // the coordinator's own blocks go through the same admission as a remote block
        let me = fabric.me();
        for (tile, view) in tile_ids.iter().zip(&level.tiles) {
            let mut sink = |path: &str, start: &[usize], count: &[usize], data: &[f64]| {
                let dataset = datasets
                    .iter()
                    .position(|d| d == path)
                    .expect("the walker names a declared dataset");
                let mut bytes = Vec::new();
                encode_block::<D>(dataset as u16, *tile, start, count, data, &mut bytes);
                let block = decode_block::<D>(&bytes).expect("an encoded block decodes");
                admit_block(
                    &mut ledger,
                    &datasets,
                    placement,
                    &mut stream,
                    me,
                    &block,
                    budget,
                )
                .map_err(|e| IoError::Backend(e.to_string()))
            };
            owned_blocks::<R, D, DOF, Mem>(0, view, &level.grid, budget, &mut sink)?;
        }
        // remote blocks until every worker has contributed its close vote; a worker's vote
        // follows its last block on the same ordered stream
        let start = Instant::now();
        while fabric.contributions_to_current() < workers - 1 {
            fabric.service()?;
            if let Some((from, _seq, payload)) = fabric.staged_block() {
                let block = decode_block::<D>(payload)
                    .ok_or_else(|| FabricError::protocol(from, "malformed block payload"))?;
                admit_block(
                    &mut ledger,
                    &datasets,
                    placement,
                    &mut stream,
                    from,
                    &block,
                    budget,
                )?;
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
        // the file takes its name only when every owned box arrived exactly once and every
        // worker's close vote is affirmative; the coordinator's own vote follows the rename
        let complete = ledger
            .iter()
            .all(|per_tile| per_tile.iter().all(Entry::complete));
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
            return Err(CheckpointError::Refused(
                "the coordinator could not open the checkpoint".into(),
            ));
        }
        let mut bytes = Vec::with_capacity(block_credit_for(budget, D) as usize);
        let mut index = 0usize;
        let mut sent_ok = true;
        for (tile, view) in tile_ids.iter().zip(&level.tiles) {
            let mut sink = |path: &str, start: &[usize], count: &[usize], data: &[f64]| {
                let dataset = datasets
                    .iter()
                    .position(|d| d == path)
                    .expect("the walker names a declared dataset");
                let this = index;
                index += 1;
                if req.drop_block == Some(this) {
                    return Ok(());
                }
                encode_block::<D>(dataset as u16, *tile, start, count, data, &mut bytes);
                let repeats = if this == 0 && req.duplicate_first_block {
                    2
                } else {
                    1
                };
                for _ in 0..repeats {
                    if let Err(e) = fabric.send_block(&bytes, req.deadline) {
                        sent_ok = false;
                        return Err(IoError::Backend(format!("fabric: {e}")));
                    }
                }
                Ok(())
            };
            owned_blocks::<R, D, DOF, Mem>(0, view, &level.grid, budget, &mut sink)?;
        }
        let vote = sent_ok && !req.vote_no;
        let ok = fabric.collective(OpKind::CheckpointClose, u64::from(vote), req.deadline)?;
        if ok == 0 {
            return Err(CheckpointError::Refused(
                "the checkpoint was not published".into(),
            ));
        }
        Ok(())
    }
}
