// =============================================================================
// worker_tracers.rs
//
// the discrete tracer population on launched workers. each worker holds the
// tracers its tiles own; the population crosses workers through assembled
// collectives, every worker issuing the same sequence:
// - assemble_blocks: every tile's variable-length block of fixed-width records,
//   in flat tile order (a first pass agrees the counts, a second the words)
// - migrate_discrete: tracers whose owner cell lies on another tile move there,
//   in the in-process driver's order (source tile in flat order, removal order)
// - blend_discrete: the ssp ancestry selection over the population's ids
// - contiguous_ids: the per-step check that the ids are 0..next_id, the id set
//   the blend ranks
// - spawn_discrete_injection: every tile's injection ledger merged in flat tile
//   order and spawned alike on every worker; each keeps its tiles' tracers
// - gather_discrete: every tile's population, for the checkpoint
// - migrate_continuous, spawn_continuous_injection, gather_continuous: the same
//   for the ito population, owned by the cell its position lies in
// - whole_bounds: the whole domain's faces from any tile's geometry
//
// usage:
//   migrate_discrete(stores, tiles, plan, fabric, deadline)?;
//   blend_discrete(stores, stage.ac, stage.index);
// =============================================================================

use std::collections::BTreeMap;
use std::time::Duration;

use symbi_fabric::{ABSENT, Fabric, FabricError, Link};
use symbi_xpu::MemorySpace;

use crate::atlas::{ExchangePlan, TileId};
use crate::mass_transport::ContainerId;
use crate::state::FieldStore;
use crate::tracers::{ContinuousTracerRecord, ContinuousTracerSet, TracerFlags, TracerSet};

/// the tracers that crossed to another tile, over the whole session: every move, and the moves
/// whose source and destination tiles differ along two or more partition axes (through a corner)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Migrations {
    pub moved: u64,
    pub corner: u64,
}

impl Migrations {
    fn count<const D: usize>(&mut self, plan: &ExchangePlan<D>, source: usize, target: usize) {
        let offset = |tile: usize| plan.tiles.iter().find(|t| t.tile.0 as usize == tile).map(|t| t.offset);
        let (Some(from), Some(to)) = (offset(source), offset(target)) else {
            return;
        };
        self.moved += 1;
        if (0..D).filter(|&ax| from[ax] != to[ax]).count() >= 2 {
            self.corner += 1;
        }
    }

    pub fn add(&mut self, other: Migrations) {
        self.moved += other.moved;
        self.corner += other.corner;
    }
}

/// the words of one tracer: position, id, cohort, owner, step-entry owner, flags, step flags
pub const fn record_words(d: usize) -> usize {
    d + 8
}

fn pack(flags: TracerFlags) -> [u64; 2] {
    [u64::from(flags.escaped) | (u64::from(flags.crossed_sink) << 1), flags.crossing_time.to_bits()]
}

fn unpack(words: [u64; 2]) -> TracerFlags {
    TracerFlags {
        escaped: words[0] & 1 != 0,
        crossed_sink: words[0] & 2 != 0,
        crossing_time: f64::from_bits(words[1]),
    }
}

/// append record `ii` of `set` to `out` in `record_words(D)` words
pub fn encode<const D: usize>(set: &TracerSet<D>, ii: usize, out: &mut Vec<u64>) {
    out.extend(set.x[ii].iter().map(|v| v.to_bits()));
    out.push(set.id[ii]);
    out.push(u64::from(set.cohort[ii]));
    out.push(set.owner[ii].0);
    out.push(set.step_owner[ii].0);
    out.extend(pack(set.flags[ii]));
    out.extend(pack(set.step_flags[ii]));
}

/// push the record `words` encodes onto `set`
pub fn push_decoded<const D: usize>(set: &mut TracerSet<D>, words: &[u64]) {
    set.x.push(std::array::from_fn(|dd| f64::from_bits(words[dd])));
    set.id.push(words[D]);
    set.cohort.push(words[D + 1] as u16);
    set.owner.push(ContainerId(words[D + 2]));
    set.step_owner.push(ContainerId(words[D + 3]));
    set.flags.push(unpack([words[D + 4], words[D + 5]]));
    set.step_flags.push(unpack([words[D + 6], words[D + 7]]));
}

/// one fixed-size header per tile, assembled on every worker in flat tile order. every
/// assembly here moves in the windows the session's agreed frame bound carries
pub fn assemble_headers<L: Link, const K: usize>(
    fabric: &mut Fabric<L>,
    n_tiles: usize,
    local: &[(usize, [u64; K])],
    deadline: Duration,
) -> Result<Vec<[u64; K]>, FabricError> {
    let entries: Vec<(u64, u64)> = local
        .iter()
        .flat_map(|(tile, words)| words.iter().enumerate().map(move |(w, &v)| ((tile * K + w) as u64, v)))
        .collect();
    let dense = fabric.assemble_windowed((n_tiles * K) as u64, &entries, deadline)?;
    Ok((0..n_tiles).map(|tile| std::array::from_fn(|w| dense[tile * K + w])).collect())
}

/// every tile's block of `width`-word records, assembled on every worker in flat tile order: a
/// first pass agrees each tile's record count, a second carries the words into the slots the
/// counts lay out
pub fn assemble_blocks<L: Link>(
    fabric: &mut Fabric<L>,
    n_tiles: usize,
    local: &[(usize, Vec<u64>)],
    width: usize,
    deadline: Duration,
) -> Result<Vec<Vec<u64>>, FabricError> {
    let me = fabric.me();
    let counts: Vec<(usize, [u64; 1])> = local.iter().map(|(tile, words)| (*tile, [(words.len() / width) as u64])).collect();
    let counts = assemble_headers(fabric, n_tiles, &counts, deadline)?;
    if let Some(tile) = counts.iter().position(|c| c[0] == ABSENT) {
        return Err(FabricError::protocol(me, format!("tile {tile}'s record count is missing from the assembled vector")));
    }
    let mut starts = Vec::with_capacity(n_tiles + 1);
    starts.push(0usize);
    for count in &counts {
        starts.push(starts.last().unwrap() + count[0] as usize * width);
    }
    let total = starts[n_tiles];
    if total == 0 {
        return Ok(vec![Vec::new(); n_tiles]);
    }
    let entries: Vec<(u64, u64)> = local
        .iter()
        .flat_map(|(tile, words)| {
            let start = starts[*tile];
            words.iter().enumerate().map(move |(w, &v)| ((start + w) as u64, v))
        })
        .collect();
    let dense = fabric.assemble_windowed(total as u64, &entries, deadline)?;
    Ok((0..n_tiles).map(|tile| dense[starts[tile]..starts[tile + 1]].to_vec()).collect())
}

/// the whole grid's cell counts, from the plan's tiles
pub fn plan_cells<const D: usize>(plan: &ExchangePlan<D>) -> [usize; D] {
    std::array::from_fn(|ax| {
        plan.tiles
            .iter()
            .map(|t| (t.offset[ax] + t.interior[ax].size() as isize) as usize)
            .max()
            .unwrap_or(0)
    })
}

/// the flat tile holding the root cell a tracer owner names (axis-0-fastest over the whole
/// grid), by the plan's tile boxes
pub fn owner_tile<const D: usize>(plan: &ExchangePlan<D>, cells: [usize; D], owner: ContainerId) -> usize {
    let mut linear = owner.0 as usize;
    let cell: [isize; D] = std::array::from_fn(|ax| {
        let index = linear % cells[ax];
        linear /= cells[ax];
        index as isize
    });
    plan.tiles
        .iter()
        .find(|t| (0..D).all(|ax| cell[ax] >= t.offset[ax] && cell[ax] < t.offset[ax] + t.interior[ax].size() as isize))
        .map(|t| t.tile.0 as usize)
        .expect("a root cell lies on one tile of the plan")
}

/// move every tracer whose owner cell lies on another tile to that tile. leavers come out of
/// their tile by swap removal in index order and reach their targets in (source tile, removal)
/// order, the in-process driver's order; an escaped tracer stays where it is.
pub fn migrate_discrete<const D: usize, const DOF: usize, M: MemorySpace, L: Link>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    tiles: &[TileId],
    plan: &ExchangePlan<D>,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<Migrations, FabricError> {
    let cells = plan_cells(plan);
    let width = 1 + record_words(D);
    let mut local = Vec::with_capacity(stores.len());
    for (k, store) in stores.iter_mut().enumerate() {
        let source = tiles[k].0 as usize;
        let mut words = Vec::new();
        if let Some(tracers) = store.tracers.as_mut() {
            let mut ii = 0;
            while ii < tracers.len() {
                // a tracer owned by no cell (escaped, or in a sink reservoir) is frozen and stays
                let frozen = tracers.flags[ii].escaped || crate::tracers::cell_container_address(tracers.owner[ii]).is_none();
                let target = if frozen { source } else { owner_tile(plan, cells, tracers.owner[ii]) };
                if target == source {
                    ii += 1;
                    continue;
                }
                let last = tracers.len() - 1;
                tracers.x.swap(ii, last);
                tracers.id.swap(ii, last);
                tracers.cohort.swap(ii, last);
                tracers.flags.swap(ii, last);
                tracers.owner.swap(ii, last);
                tracers.step_owner.swap(ii, last);
                tracers.step_flags.swap(ii, last);
                words.push(target as u64);
                encode(tracers, last, &mut words);
                tracers.x.pop();
                tracers.id.pop();
                tracers.cohort.pop();
                tracers.flags.pop();
                tracers.owner.pop();
                tracers.step_owner.pop();
                tracers.step_flags.pop();
            }
        }
        local.push((source, words));
    }
    let blocks = assemble_blocks(fabric, plan.tiles.len(), &local, width, deadline)?;
    let mut moves = Migrations::default();
    for (source, block) in blocks.iter().enumerate() {
        for record in block.chunks_exact(width) {
            let target = record[0] as usize;
            moves.count(plan, source, target);
            if let Some(k) = tiles.iter().position(|t| t.0 as usize == target) {
                let tracers = stores[k].tracers.as_mut().expect("every tile of a traced run carries tracers");
                push_decoded(tracers, &record[1..]);
            }
        }
    }
    Ok(moves)
}

/// the ssp ancestry selection of stage `stage`: a tracer outside the systematic sample of the
/// candidate weight returns to its step-entry owner. the sample ranks the population's ids
/// 0..next_id, the whole id set `contiguous_ids` checks each step.
pub fn blend_discrete<const D: usize, const DOF: usize, M: MemorySpace>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    candidate_weight: f64,
    stage: usize,
) {
    if candidate_weight == 1.0 {
        return;
    }
    let first = stores.iter().find_map(|s| s.tracers.as_ref()).expect("a traced run carries tracers");
    let ids: Vec<u64> = (0..first.next_id).collect();
    let key = crate::mass_transport::SamplingKey {
        run_seed: first.run_seed,
        epoch: stores[0].iteration.wrapping_mul(4).wrapping_add(stage as u64),
    };
    let selections = crate::mass_transport::sample_convex_blend(&ids, candidate_weight, key)
        .expect("valid ssp ancestry weight");
    for store in stores.iter_mut() {
        let Some(tracers) = store.tracers.as_mut() else {
            continue;
        };
        for ii in 0..tracers.len() {
            if !selections[tracers.id[ii] as usize].1 {
                tracers.owner[ii] = tracers.step_owner[ii];
                tracers.flags[ii] = tracers.step_flags[ii];
            }
        }
    }
}

/// whether the population's ids are exactly 0..next_id: the tile counts sum to next_id and every
/// id lies below it (ids are unique by construction)
pub fn contiguous_ids<const D: usize, const DOF: usize, M: MemorySpace, L: Link>(
    stores: &[&FieldStore<D, DOF, M, f64>],
    tiles: &[TileId],
    n_tiles: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<bool, FabricError> {
    let mut below = true;
    let mut local = Vec::with_capacity(stores.len());
    let mut next_id = 0;
    for (k, store) in stores.iter().enumerate() {
        let tracers = store.tracers.as_ref().expect("every tile of a traced run carries tracers");
        next_id = tracers.next_id;
        below &= tracers.id.iter().all(|&id| id < tracers.next_id);
        local.push((tiles[k].0 as usize, [tracers.len() as u64]));
    }
    let counts = assemble_headers(fabric, n_tiles, &local, deadline)?;
    let total: u64 = counts.iter().map(|c| c[0]).sum();
    Ok(below && total == next_id)
}

/// spawn the step's boundary and source injection: every tile's ledger merged in flat tile order
/// and spawned alike on every worker (ids, remainder and positions follow the whole grid), each
/// worker keeping the tracers its tiles own. the remainder is booked on the grid's first tile.
pub fn spawn_discrete_injection<const D: usize, const DOF: usize, M: MemorySpace, L: Link>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    tiles: &[TileId],
    plan: &ExchangePlan<D>,
    ledgers: &[BTreeMap<ContainerId, f64>],
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError> {
    let n_tiles = plan.tiles.len();
    let mut headers = Vec::with_capacity(stores.len());
    let mut blocks = Vec::with_capacity(stores.len());
    for (k, store) in stores.iter().enumerate() {
        let tracers = store.tracers.as_ref().expect("every tile of a traced run carries tracers");
        headers.push((tiles[k].0 as usize, [tracers.next_id, tracers.injection_remainder.to_bits()]));
        blocks.push((tiles[k].0 as usize, ledgers[k].iter().flat_map(|(dest, mass)| [dest.0, mass.to_bits()]).collect()));
    }
    let headers = assemble_headers(fabric, n_tiles, &headers, deadline)?;
    let blocks = assemble_blocks(fabric, n_tiles, &blocks, 2, deadline)?;
    let mut injections = BTreeMap::new();
    for block in &blocks {
        for pair in block.chunks_exact(2) {
            *injections.entry(ContainerId(pair[0])).or_insert(0.0) += f64::from_bits(pair[1]);
        }
    }
    if injections.is_empty() {
        return Ok(());
    }
    let next_id = headers.iter().map(|h| h[0]).max().unwrap_or(0);
    let remainder: f64 = headers.iter().map(|h| f64::from_bits(h[1])).sum();
    let cells = plan_cells(plan);
    let spawned = crate::decomp::spawn_injection_set(&*stores[0], cells, injections, next_id, remainder);
    for (k, store) in stores.iter_mut().enumerate() {
        let tracers = store.tracers.as_mut().expect("every tile of a traced run carries tracers");
        tracers.next_id = spawned.next_id;
        tracers.injection_remainder = if tiles[k].0 == 0 { spawned.injection_remainder } else { 0.0 };
    }
    let mut words = Vec::with_capacity(record_words(D));
    for ii in 0..spawned.len() {
        let target = owner_tile(plan, cells, spawned.owner[ii]);
        if let Some(k) = tiles.iter().position(|t| t.0 as usize == target) {
            words.clear();
            encode(&spawned, ii, &mut words);
            push_decoded(stores[k].tracers.as_mut().expect("every tile of a traced run carries tracers"), &words);
        }
    }
    Ok(())
}

/// every tile's tracer population, in flat tile order, on every worker
pub fn gather_discrete<const D: usize, L: Link>(
    fabric: &mut Fabric<L>,
    n_tiles: usize,
    local: &[(usize, &TracerSet<D>)],
    deadline: Duration,
) -> Result<Vec<TracerSet<D>>, FabricError> {
    let me = fabric.me();
    let headers: Vec<(usize, [u64; 4])> = local
        .iter()
        .map(|(tile, set)| (*tile, [set.weight.to_bits(), set.run_seed, set.next_id, set.injection_remainder.to_bits()]))
        .collect();
    let blocks: Vec<(usize, Vec<u64>)> = local
        .iter()
        .map(|(tile, set)| {
            let mut words = Vec::with_capacity(set.len() * record_words(D));
            for ii in 0..set.len() {
                encode(set, ii, &mut words);
            }
            (*tile, words)
        })
        .collect();
    let headers = assemble_headers(fabric, n_tiles, &headers, deadline)?;
    let blocks = assemble_blocks(fabric, n_tiles, &blocks, record_words(D), deadline)?;
    let mut sets = Vec::with_capacity(n_tiles);
    for (tile, (header, block)) in headers.iter().zip(&blocks).enumerate() {
        if header.iter().all(|&w| w == ABSENT) {
            return Err(FabricError::protocol(me, format!("tile {tile}'s tracer header is missing from the assembled vector")));
        }
        let mut set = TracerSet::<D> {
            weight: f64::from_bits(header[0]),
            run_seed: header[1],
            next_id: header[2],
            injection_remainder: f64::from_bits(header[3]),
            ..Default::default()
        };
        for record in block.chunks_exact(record_words(D)) {
            push_decoded(&mut set, record);
        }
        sets.push(set);
    }
    Ok(sets)
}

// ---- the continuous (ito) population ----------------------------------------------------

/// the words of one continuous tracer: position, step position, id, cohort, owner, flags,
/// crossing time and noise counter
pub const fn continuous_words(d: usize) -> usize {
    2 * d + 7
}

/// append the continuous record to `out` in `continuous_words(D)` words
pub fn encode_continuous<const D: usize>(record: &ContinuousTracerRecord<D>, out: &mut Vec<u64>) {
    out.extend(record.x.iter().map(|v| v.to_bits()));
    out.extend(record.step_x.iter().map(|v| v.to_bits()));
    out.push(record.id);
    out.push(u64::from(record.cohort));
    out.push(record.owner.0);
    out.push(u64::from(record.escaped));
    out.push(u64::from(record.crossed_sink));
    out.push(record.crossing_time.to_bits());
    out.push(record.random_counter);
}

/// the continuous record `words` encodes
pub fn decode_continuous<const D: usize>(words: &[u64]) -> ContinuousTracerRecord<D> {
    ContinuousTracerRecord {
        x: std::array::from_fn(|dd| f64::from_bits(words[dd])),
        step_x: std::array::from_fn(|dd| f64::from_bits(words[D + dd])),
        id: words[2 * D],
        cohort: words[2 * D + 1] as u16,
        owner: ContainerId(words[2 * D + 2]),
        escaped: words[2 * D + 3] as u8,
        crossed_sink: words[2 * D + 4] as u8,
        crossing_time: f64::from_bits(words[2 * D + 5]),
        random_counter: words[2 * D + 6],
    }
}

/// a position folded into the whole domain on the axes that close into a ring
fn fold_into_domain<const D: usize>(mut position: [f64; D], wraps: [bool; D], bounds: [(f64, f64); D]) -> [f64; D] {
    for dd in 0..D {
        // a position inside the domain is the one the trajectory reached; re-folding it would
        // round (p - lo) + lo away from p
        if !wraps[dd] || (position[dd] >= bounds[dd].0 && position[dd] < bounds[dd].1) {
            continue;
        }
        let span = bounds[dd].1 - bounds[dd].0;
        if span > 0.0 {
            let offset = (position[dd] - bounds[dd].0).rem_euclid(span);
            position[dd] = bounds[dd].0 + offset;
        }
    }
    position
}

/// move every continuous tracer whose position lies in another tile's cells to that tile, its
/// owner the whole-grid cell it sits in. leavers come out by swap removal in index order and
/// reach their targets in (source tile, removal) order, the in-process driver's order; an
/// escaped, absorbed or out-of-domain tracer stays where it is. `bounds` is the whole domain.
pub fn migrate_continuous<const D: usize, const DOF: usize, M: MemorySpace, L: Link>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    tiles: &[TileId],
    plan: &ExchangePlan<D>,
    bounds: [(f64, f64); D],
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<Migrations, String> {
    let cells = plan_cells(plan);
    let width = 1 + continuous_words(D);
    let mut local = Vec::with_capacity(stores.len());
    for (k, store) in stores.iter_mut().enumerate() {
        let source = tiles[k].0 as usize;
        let mut words = Vec::new();
        if let Some(tracers) = store.continuous_tracers.as_mut() {
            let mut ii = 0;
            while ii < tracers.len {
                let record = tracers.record_host(ii)?;
                if record.escaped != 0 || record.crossed_sink != 0 {
                    ii += 1;
                    continue;
                }
                let position = fold_into_domain(record.x, plan.wraps, bounds);
                if (0..D).any(|dd| position[dd] < bounds[dd].0 || position[dd] >= bounds[dd].1) {
                    ii += 1;
                    continue;
                }
                if position != record.x {
                    tracers.set_position_host(ii, position)?;
                }
                // the whole-grid cell the position lies in, on a uniform or a mapped axis
                let cell: [usize; D] = std::array::from_fn(|dd| (store.geom.cell_at(dd, position[dd]) + store.geom.index_offset[dd]) as usize);
                let owner = crate::tracers::cell_container_id(crate::tracers::cell_address(cell, cells), 0);
                let target = owner_tile(plan, cells, owner);
                if target == source {
                    tracers.set_owner_host(ii, owner)?;
                    ii += 1;
                } else {
                    let mut record = tracers.swap_remove_host(ii)?;
                    record.owner = owner;
                    words.push(target as u64);
                    encode_continuous(&record, &mut words);
                }
            }
        }
        local.push((source, words));
    }
    let blocks = assemble_blocks(fabric, plan.tiles.len(), &local, width, deadline).map_err(|e| e.to_string())?;
    let mut moves = Migrations::default();
    for (source, block) in blocks.iter().enumerate() {
        for record in block.chunks_exact(width) {
            let target = record[0] as usize;
            moves.count(plan, source, target);
            if let Some(k) = tiles.iter().position(|t| t.0 as usize == target) {
                let tracers = stores[k].continuous_tracers.as_mut().expect("every tile of a traced run carries tracers");
                tracers.push_host(decode_continuous(&record[1..]))?;
            }
        }
    }
    Ok(moves)
}

/// spawn the step's injection as continuous tracers: every tile's ledger merged in flat tile
/// order and spawned alike on every worker on the moved mesh, each worker keeping the tracers its
/// tiles own, pushed in the in-process order (last spawned first). the remainder is one global
/// value every tile carries.
pub fn spawn_continuous_injection<const D: usize, const DOF: usize, M: MemorySpace, L: Link>(
    stores: &mut [&mut FieldStore<D, DOF, M, f64>],
    tiles: &[TileId],
    plan: &ExchangePlan<D>,
    ledgers: &[BTreeMap<ContainerId, f64>],
    dt: f64,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), String> {
    let n_tiles = plan.tiles.len();
    let mut headers = Vec::with_capacity(stores.len());
    let mut blocks = Vec::with_capacity(stores.len());
    for (k, store) in stores.iter().enumerate() {
        let tracers = store.continuous_tracers.as_ref().expect("every tile of a traced run carries tracers");
        headers.push((tiles[k].0 as usize, [tracers.next_id]));
        blocks.push((tiles[k].0 as usize, ledgers[k].iter().flat_map(|(dest, mass)| [dest.0, mass.to_bits()]).collect()));
    }
    let headers = assemble_headers(fabric, n_tiles, &headers, deadline).map_err(|e| e.to_string())?;
    let blocks = assemble_blocks(fabric, n_tiles, &blocks, 2, deadline).map_err(|e| e.to_string())?;
    let mut injections = BTreeMap::new();
    for block in &blocks {
        for pair in block.chunks_exact(2) {
            *injections.entry(ContainerId(pair[0])).or_insert(0.0) += f64::from_bits(pair[1]);
        }
    }
    if injections.is_empty() {
        return Ok(());
    }
    let next_id = headers.iter().map(|h| h[0]).max().unwrap_or(0);
    let cells = plan_cells(plan);
    let mut spawned = crate::decomp::spawn_continuous_injection_set(&*stores[0], cells, injections, next_id, dt)?;
    for store in stores.iter_mut() {
        let tracers = store.continuous_tracers.as_mut().expect("every tile of a traced run carries tracers");
        tracers.next_id = spawned.next_id;
        tracers.injection_remainder = spawned.injection_remainder;
    }
    while spawned.len > 0 {
        let record = spawned.swap_remove_host(spawned.len - 1)?;
        let target = owner_tile(plan, cells, record.owner);
        if let Some(k) = tiles.iter().position(|t| t.0 as usize == target) {
            stores[k].continuous_tracers.as_mut().expect("every tile of a traced run carries tracers").push_host(record)?;
        }
    }
    Ok(())
}

/// every tile's continuous population, in flat tile order, on every worker
pub fn gather_continuous<const D: usize, M: MemorySpace, L: Link>(
    fabric: &mut Fabric<L>,
    n_tiles: usize,
    local: &[(usize, &ContinuousTracerSet<D, M>)],
    deadline: Duration,
) -> Result<Vec<ContinuousTracerSet<D, M>>, String> {
    let me = fabric.me();
    let headers: Vec<(usize, [u64; 5])> = local
        .iter()
        .map(|(tile, set)| {
            let order = match set.order {
                crate::mass_transport::ItoOrder::Two => 2,
                crate::mass_transport::ItoOrder::Three => 3,
            };
            (*tile, [order, set.weight.to_bits(), set.run_seed, set.next_id, set.injection_remainder.to_bits()])
        })
        .collect();
    let mut blocks = Vec::with_capacity(local.len());
    for (tile, set) in local {
        let mut words = Vec::with_capacity(set.len * continuous_words(D));
        for ii in 0..set.len {
            encode_continuous(&set.record_host(ii)?, &mut words);
        }
        blocks.push((*tile, words));
    }
    let headers = assemble_headers(fabric, n_tiles, &headers, deadline).map_err(|e| e.to_string())?;
    let blocks = assemble_blocks(fabric, n_tiles, &blocks, continuous_words(D), deadline).map_err(|e| e.to_string())?;
    let mut sets = Vec::with_capacity(n_tiles);
    for (tile, (header, block)) in headers.iter().zip(&blocks).enumerate() {
        let order = match header[0] {
            2 => crate::mass_transport::ItoOrder::Two,
            3 => crate::mass_transport::ItoOrder::Three,
            _ => return Err(format!("worker {}: tile {tile}'s continuous tracer header is missing from the assembled vector", me.0)),
        };
        let mut set = ContinuousTracerSet::<D, M>::allocate(0, order)?;
        set.weight = f64::from_bits(header[1]);
        set.run_seed = header[2];
        set.next_id = header[3];
        set.injection_remainder = f64::from_bits(header[4]);
        for record in block.chunks_exact(continuous_words(D)) {
            set.push_host(decode_continuous(record))?;
        }
        sets.push(set);
    }
    Ok(sets)
}

/// the whole domain's lower and upper faces per axis from one tile's geometry: the whole grid's
/// cells 0 and `cells` in the tile's index frame, the faces the grid's first and last tiles hold
pub fn whole_bounds<const D: usize>(geom: &crate::state::PartitionGeometry<D>, cells: [usize; D]) -> [(f64, f64); D] {
    std::array::from_fn(|dd| {
        let face = |global: isize| {
            let mut idx = [0isize; D];
            idx[dd] = global - geom.index_offset[dd];
            geom.stagger_coord(idx, [crate::state::Loc::Face; D])[dd]
        };
        (face(0), face(cells[dd] as isize))
    })
}
