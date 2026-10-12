// =============================================================================
// launched_tracers.rs
//
// the discrete tracer population of a refined run on launched workers, level by
// level, as the single hierarchy carries it: every stage's records cross to the
// tiles holding their owner cells, the ancestry blend ranks the level's ids over
// every worker, the returned records cross back, every record sits at its owner
// cell's centre, and the stage's coarse-fine interface event follows; each
// level's injection spawns once from the merged ledger with the run's next id
// and remainder, every worker keeping the records its tiles own. a level atlas
// of every tile's level box in level-global cells, agreed once, maps any owner
// to its tile on every worker.
//
// usage:
//  let atlas = LevelAtlas::agree(tiles, held, depth, fabric, deadline)?;
//  exchange_level_tracers(tiles, held, &atlas, level, stage, fabric, deadline)?;
//  spawn_level_injection(tiles, held, &atlas, level, fabric, deadline)?;
// =============================================================================
use std::collections::BTreeMap;
use std::time::Duration;

use symbi_fabric::{ABSENT, Fabric, FabricError, Link};
use symbi_hydro::regime::Regime;
use symbi_sim::atlas::TileId;
use symbi_sim::mass_transport::{ContainerId, SamplingKey};

use symbi_sim::substrate_seam::KernelSet;
use symbi_sim::worker_tracers::{assemble_blocks, assemble_headers, encode, push_decoded, record_words};
use symbi_xpu::{ExecutionSpace, MemorySpace};
use symbi_geometry::Metric;

use super::hierarchy::{Hierarchy, RATIO};
use super::launched::Held;

/// every tile's box of each level in that level's whole-grid cells, by the level's tile id, and the
/// root lattice every level refines
pub struct LevelAtlas<const D: usize> {
    boxes: Vec<Vec<([usize; D], [usize; D])>>,
    root_cells: [usize; D],
    x_lo: [f64; D],
    dx: [f64; D],
}

/// the id on level `level`'s plan of held root tile `k`'s tile of the level
pub(crate) fn level_id(held: &Held<'_>, level: usize, k: usize) -> Option<TileId> {
    if level == 0 { Some(held.root_ids[k]) } else { held.level_ids(level)[k] }
}

fn level_count(held: &Held<'_>, level: usize) -> usize {
    if level == 0 { held.n_root } else { held.level_tiles(level) }
}

impl<const D: usize> LevelAtlas<D> {
    /// every worker's held tile boxes, assembled once per level
    pub fn agree<R, const DOF: usize, M, E, S, Mem, K, L>(
        tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
        held: &Held<'_>,
        depth: usize,
        fabric: &mut Fabric<L>,
        deadline: Duration,
    ) -> Result<Self, FabricError>
    where
        R: Regime<f64, D> + Copy,
        M: Metric<f64, D> + Copy + Send + Sync,
        E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
        S: ExecutionSpace,
        Mem: MemorySpace + Sync,
        K: KernelSet<D, DOF, Mem, f64>,
        L: Link,
    {
        let root_cells = tiles[0].tracer_root_cells();
        let mut boxes = Vec::with_capacity(depth);
        for level in 0..depth {
            let mut local: Vec<(usize, Vec<u64>)> = Vec::new();
            for (k, tile) in tiles.iter().enumerate() {
                let Some(id) = level_id(held, level, k) else { continue };
                let layout = tile.tracer_layout(level);
                let interior = &tile.levels[level].state.geom.interior;
                let lo: [usize; D] = layout.tile_offset;
                let hi: [usize; D] = std::array::from_fn(|ax| lo[ax] + interior.spaces[ax].size());
                local.push((id.0 as usize, lo.iter().chain(hi.iter()).map(|&v| v as u64).collect()));
            }
            let blocks = assemble_blocks(fabric, level_count(held, level), &local, 2 * D, deadline)?;
            boxes.push(
                blocks
                    .iter()
                    .map(|b| (std::array::from_fn(|ax| b[ax] as usize), std::array::from_fn(|ax| b[D + ax] as usize)))
                    .collect(),
            );
        }
        let geom = &tiles[0].levels[0].state.geom;
        Ok(Self { boxes, root_cells, x_lo: geom.x_lo, dx: geom.dx })
    }

    /// the whole-grid cell of level `level` holding `position`, inside the level's lattice
    fn level_cell(&self, level: usize, position: [f64; D]) -> Option<[usize; D]> {
        let scale = RATIO.pow(level as u32);
        let mut cell = [0usize; D];
        for ax in 0..D {
            let index = ((position[ax] - self.x_lo[ax]) / (self.dx[ax] / scale as f64)).floor();
            if !(index >= 0.0 && (index as usize) < self.root_cells[ax] * scale) {
                return None;
            }
            cell[ax] = index as usize;
        }
        Some(cell)
    }

    /// the level tile id whose box holds level `level`'s cell at `position`, with the cell
    fn holder_at(&self, level: usize, position: [f64; D]) -> Option<(usize, [usize; D])> {
        let cell = self.level_cell(level, position)?;
        self.boxes
            .get(level)?
            .iter()
            .position(|(lo, hi)| (0..D).all(|ax| cell[ax] >= lo[ax] && cell[ax] < hi[ax]))
            .map(|id| (id, cell))
    }

    /// the level and the level tile id whose box holds the cell `owner` names
    pub fn holder(&self, owner: ContainerId) -> Option<(usize, usize)> {
        let (level, mut linear) = symbi_sim::tracers::cell_container_address(owner)?;
        let level = level as usize;
        let boxes = self.boxes.get(level)?;
        let scale = RATIO.pow(level as u32);
        let cell: [usize; D] = std::array::from_fn(|ax| {
            let cells = self.root_cells[ax] * scale;
            let index = linear % cells;
            linear /= cells;
            index
        });
        boxes
            .iter()
            .position(|(lo, hi)| (0..D).all(|ax| cell[ax] >= lo[ax] && cell[ax] < hi[ax]))
            .map(|id| (level, id))
    }
}

/// level `level`'s records whose owner cell lies on another tile of the level cross to it, over
/// every worker; a record owned on no tile of the level stays
fn migrate_level<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    atlas: &LevelAtlas<D>,
    level: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    let width = 1 + record_words(D);
    let mut local = Vec::new();
    for k in 0..tiles.len() {
        let Some(id) = level_id(held, level, k) else { continue };
        let source = id.0 as usize;
        let mut words = Vec::new();
        if let Some(tracers) = tiles[k].levels[level].state.tracers.as_mut() {
            let mut ii = 0;
            while ii < tracers.len() {
                let target = match atlas.holder(tracers.owner[ii]) {
                    Some((owner_level, target)) if owner_level == level && target != source => target,
                    _ => {
                        ii += 1;
                        continue;
                    }
                };
                words.push(target as u64);
                encode(tracers, ii, &mut words);
                tracers.x.swap_remove(ii);
                tracers.id.swap_remove(ii);
                tracers.cohort.swap_remove(ii);
                tracers.flags.swap_remove(ii);
                tracers.owner.swap_remove(ii);
                tracers.step_owner.swap_remove(ii);
                tracers.step_flags.swap_remove(ii);
            }
        }
        local.push((source, words));
    }
    let blocks = assemble_blocks(fabric, level_count(held, level), &local, width, deadline)?;
    for block in &blocks {
        for record in block.chunks_exact(width) {
            let target = record[0] as usize;
            if let Some(k) = (0..tiles.len()).find(|&k| level_id(held, level, k).is_some_and(|id| id.0 as usize == target)) {
                let tracers = tiles[k].levels[level].state.tracers.as_mut().expect("every level of a traced tile carries tracers");
                push_decoded(tracers, &record[1..]);
            }
        }
    }
    Ok(())
}

/// the ancestry blend of level `level`'s stage `ii` over the level's ids on every worker, keyed as
/// the single hierarchy keys it over its level store
fn blend_level<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    level: usize,
    ii: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    let (a0, ac) = tiles[0].levels[0].state.timestepping.stages()[ii];
    if a0 == 0.0 && ac == 1.0 {
        return Ok(());
    }
    let mut local = Vec::new();
    for (k, tile) in tiles.iter().enumerate() {
        let Some(id) = level_id(held, level, k) else { continue };
        let ids = tile.levels[level].state.tracers.as_ref().map(|t| t.id.clone()).unwrap_or_default();
        local.push((id.0 as usize, ids));
    }
    let blocks = assemble_blocks(fabric, level_count(held, level), &local, 1, deadline)?;
    let Some(first) = (0..tiles.len()).find(|&k| level_id(held, level, k).is_some()) else {
        return Ok(());
    };
    let state = &tiles[first].levels[level].state;
    let key = SamplingKey {
        run_seed: state.tracers.as_ref().expect("a traced level carries tracers").run_seed,
        epoch: state.iteration.wrapping_mul(4).wrapping_add(ii as u64),
    };
    let ids: Vec<u64> = blocks.into_iter().flatten().collect();
    let selections: BTreeMap<u64, bool> = symbi_sim::mass_transport::sample_convex_blend(&ids, ac, key)
        .unwrap_or_else(|detail| panic!("tracer ancestry blend: {detail}"))
        .into_iter()
        .collect();
    for (k, tile) in tiles.iter_mut().enumerate() {
        if level_id(held, level, k).is_none() {
            continue;
        }
        let Some(tracers) = tile.levels[level].state.tracers.as_mut() else { continue };
        for kk in 0..tracers.len() {
            if !selections[&tracers.id[kk]] {
                tracers.owner[kk] = tracers.step_owner[kk];
                tracers.flags[kk] = tracers.step_flags[kk];
            }
        }
    }
    Ok(())
}

/// every record a held tile's level cell owns placed at that cell's centre
fn place_level<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    level: usize,
) where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    for (k, tile) in tiles.iter_mut().enumerate() {
        if level_id(held, level, k).is_none() {
            continue;
        }
        let Some(tracers) = tile.levels[level].state.tracers.as_ref() else { continue };
        let centers: Vec<Option<[f64; D]>> = {
            let state = &tile.levels[level].state;
            let geometry = state.geom.block_geometry(state.physics.metric);
            tracers
                .owner
                .iter()
                .map(|&owner| {
                    tile.tracer_cell(owner).and_then(|(lev, coord)| {
                        (lev == level).then(|| {
                            let center = geometry.centroid(coord);
                            std::array::from_fn(|dd| center[dd])
                        })
                    })
                })
                .collect()
        };
        let tracers = tile.levels[level].state.tracers.as_mut().unwrap();
        for (kk, center) in centers.into_iter().enumerate() {
            if let Some(center) = center {
                tracers.x[kk] = center;
            }
        }
    }
}

/// level `level`'s stage-`ii` tracer exchange over every worker: the leavers cross, the blend ranks
/// the level's ids, the returned records cross back, every record sits at its owner cell's centre,
/// and the stage's interface event follows
#[allow(clippy::too_many_arguments)]
pub fn exchange_level_tracers<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    atlas: &LevelAtlas<D>,
    level: usize,
    ii: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    migrate_level(tiles, held, atlas, level, fabric, deadline)?;
    blend_level(tiles, held, level, ii, fabric, deadline)?;
    migrate_level(tiles, held, atlas, level, fabric, deadline)?;
    place_level(tiles, held, level);
    for (k, tile) in tiles.iter_mut().enumerate() {
        if level_id(held, level, k).is_some() {
            tile.apply_pending_interface(level);
        }
    }
    Ok(())
}

/// level `level`'s injection over every worker, spawned once from the merged ledger with the run's
/// next id and remainder and keyed by the level's iteration, as the single hierarchy spawns its
/// own; every worker spawns the same records and keeps those its tiles own, each at its cell's
/// centre. the next id and the remainder ride every held level
pub fn spawn_level_injection<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    atlas: &LevelAtlas<D>,
    level: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    let me = fabric.me();
    let n_level = level_count(held, level);
    let mut ledgers = Vec::new();
    let mut clocks = Vec::new();
    for k in 0..tiles.len() {
        let Some(id) = level_id(held, level, k) else { continue };
        let ledger = tiles[k].take_injection_ledger();
        ledgers.push((id.0 as usize, ledger.iter().flat_map(|(dest, mass)| [dest.0, mass.to_bits()]).collect::<Vec<u64>>()));
        clocks.push((id.0 as usize, [tiles[k].levels[level].state.iteration]));
    }
    let blocks = assemble_blocks(fabric, n_level, &ledgers, 2, deadline)?;
    let clocks = assemble_headers(fabric, n_level, &clocks, deadline)?;
    let mut injections = BTreeMap::new();
    for block in &blocks {
        for pair in block.chunks_exact(2) {
            *injections.entry(ContainerId(pair[0])).or_insert(0.0) += f64::from_bits(pair[1]);
        }
    }
    if injections.is_empty() {
        return Ok(());
    }
    let iteration = clocks
        .iter()
        .map(|c| c[0])
        .find(|&w| w != ABSENT)
        .ok_or_else(|| FabricError::protocol(me, format!("no tile of level {level} published its clock")))?;
    let template = tiles[0].levels[0].state.tracers.as_ref().expect("a traced tile carries tracers");
    let mut spawned = symbi_sim::tracers::TracerSet::<D> {
        weight: template.weight,
        run_seed: template.run_seed,
        next_id: template.next_id,
        injection_remainder: template.injection_remainder,
        ..Default::default()
    };
    let key = SamplingKey { run_seed: spawned.run_seed, epoch: iteration | (1 << 62) };
    let place = |owner: ContainerId| -> [f64; D] {
        for (k, tile) in tiles.iter().enumerate() {
            if level_id(held, level, k).is_none() {
                continue;
            }
            if let Some((lev, coord)) = tile.tracer_cell(owner) {
                if lev == level {
                    let state = &tile.levels[level].state;
                    let center = state.geom.block_geometry(state.physics.metric).centroid(coord);
                    return std::array::from_fn(|dd| center[dd]);
                }
            }
        }
        // a cell another worker holds: that worker places the record
        [0.0; D]
    };
    symbi_sim::tracers::spawn_injected_tracers(
        &mut spawned,
        injections.into_iter().map(|(destination, mass)| symbi_sim::mass_transport::MassTransfer { destination, mass }),
        place,
        key,
    )
    .unwrap_or_else(|detail| panic!("launched refined tracer injection: {detail}"));
    let mut words = Vec::with_capacity(record_words(D));
    for kk in 0..spawned.len() {
        let Some((owner_level, target)) = atlas.holder(spawned.owner[kk]) else { continue };
        debug_assert_eq!(owner_level, level);
        if let Some(k) = (0..tiles.len()).find(|&k| level_id(held, level, k).is_some_and(|id| id.0 as usize == target)) {
            words.clear();
            encode(&spawned, kk, &mut words);
            push_decoded(tiles[k].levels[level].state.tracers.as_mut().expect("a traced level carries tracers"), &words);
        }
    }
    for tile in tiles.iter_mut() {
        for data in &mut tile.levels {
            if let Some(tracers) = data.state.tracers.as_mut() {
                tracers.next_id = spawned.next_id;
                tracers.injection_remainder = spawned.injection_remainder;
            }
        }
    }
    Ok(())
}

// ---- the continuous (ito) population ----------------------------------------------------

/// level `level`'s accepted ito coefficients across its cuts at the level's coefficient point
#[allow(clippy::too_many_arguments)]
pub fn exchange_level_coefficients<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plan: &symbi_sim::plan_exchange::PlanExchange<'_, D>,
    level: usize,
    point: symbi_sim::atlas::ExchangePoint,
    step: u64,
    attempt: u16,
    transport: &T,
    fabric: &mut Fabric<L>,
    deadline: Duration,
    spent: &mut Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: symbi_sim::decomp::HaloTransport,
    L: Link,
{
    let mut fields: Vec<Vec<&symbi_grid::Field<f64, D, Mem>>> = (0..plan.plan().n_tiles()).map(|_| Vec::new()).collect();
    for (k, tile) in tiles.iter().enumerate() {
        let Some(id) = level_id(held, level, k) else { continue };
        let c = tile.levels[level].state.ito_coefficients.as_ref().expect("ito coefficients were materialized");
        fields[id.0 as usize] = c.drift.iter().chain(c.variance.iter()).chain(c.third.iter()).collect();
    }
    let devices = if level == 0 { held.root_devices } else { held.level_devices(level) };
    let part = symbi_sim::worker::PlanPart { plan, fields: &fields, devices, fault: None, dst_fields: None };
    symbi_sim::worker::exchange_point_plans(&[part], transport, fabric, point, step, attempt, deadline, spent)
}

/// level `level`'s continuous tracers cross to the tile whose level cell holds their position, over
/// every worker; a tracer outside the level's tiles stays, as it stays on its level of the single
/// hierarchy
pub fn migrate_level_continuous<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    atlas: &LevelAtlas<D>,
    level: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    use symbi_sim::worker_tracers::{continuous_words, decode_continuous, encode_continuous};
    let width = 1 + continuous_words(D);
    let mut local = Vec::new();
    for k in 0..tiles.len() {
        let Some(id) = level_id(held, level, k) else { continue };
        let source = id.0 as usize;
        let mut words = Vec::new();
        if let Some(tracers) = tiles[k].levels[level].state.continuous_tracers.as_mut() {
            for kk in (0..tracers.len).rev() {
                let (inactive, position) = unsafe {
                    (
                        *tracers.escaped.as_ptr::<u8>().add(kk) != 0 || *tracers.crossed_sink.as_ptr::<u8>().add(kk) != 0,
                        std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(kk)),
                    )
                };
                if inactive {
                    continue;
                }
                let Some((target, _)) = atlas.holder_at(level, position) else { continue };
                if target == source {
                    continue;
                }
                let record = tracers.swap_remove_host(kk).unwrap_or_else(|detail| panic!("continuous tracer migration: {detail}"));
                words.push(target as u64);
                encode_continuous(&record, &mut words);
            }
        }
        local.push((source, words));
    }
    let blocks = assemble_blocks(fabric, level_count(held, level), &local, width, deadline)?;
    for block in &blocks {
        for record in block.chunks_exact(width) {
            let target = record[0] as usize;
            if let Some(k) = (0..tiles.len()).find(|&k| level_id(held, level, k).is_some_and(|id| id.0 as usize == target)) {
                tiles[k].levels[level]
                    .state
                    .continuous_tracers
                    .as_mut()
                    .expect("every level of a traced tile carries continuous tracers")
                    .push_host(decode_continuous::<D>(&record[1..]))
                    .unwrap_or_else(|detail| panic!("continuous tracer migration: {detail}"));
            }
        }
    }
    Ok(())
}

/// level `level`'s held continuous injection over every worker, spawned once from the gathered
/// ledgers after the level's tracers advanced, as the single hierarchy spawns its own: ids from the
/// run's next id, the remainder carried, the draw keyed by the level's iteration, each tracer placed
/// in its cell on the moved mesh; every worker keeps the tracers its tiles own. the next id and the
/// remainder ride every held level
pub fn spawn_level_continuous<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    atlas: &LevelAtlas<D>,
    level: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    let me = fabric.me();
    let n_level = level_count(held, level);
    let mut ledgers = Vec::new();
    let mut clocks = Vec::new();
    for k in 0..tiles.len() {
        let Some(id) = level_id(held, level, k) else { continue };
        let ledger = tiles[k].take_continuous_injection(level);
        ledgers.push((id.0 as usize, ledger.iter().flat_map(|(dest, mass)| [dest.0, mass.to_bits()]).collect::<Vec<u64>>()));
        clocks.push((id.0 as usize, [tiles[k].levels[level].state.iteration]));
    }
    let blocks = assemble_blocks(fabric, n_level, &ledgers, 2, deadline)?;
    let clocks = assemble_headers(fabric, n_level, &clocks, deadline)?;
    let mut injections = BTreeMap::new();
    for block in &blocks {
        for pair in block.chunks_exact(2) {
            *injections.entry(ContainerId(pair[0])).or_insert(0.0) += f64::from_bits(pair[1]);
        }
    }
    if injections.is_empty() {
        return Ok(());
    }
    let iteration = clocks
        .iter()
        .map(|c| c[0])
        .find(|&w| w != ABSENT)
        .ok_or_else(|| FabricError::protocol(me, format!("no tile of level {level} published its clock")))?;
    let (order, weight, run_seed, next_id, remainder) = {
        let t = tiles[0].levels[0].state.continuous_tracers.as_ref().expect("a traced tile carries continuous tracers");
        (t.order, t.weight, t.run_seed, t.next_id, t.injection_remainder)
    };
    let mut spawned = symbi_sim::tracers::ContinuousTracerSet::<D, Mem>::allocate(0, order)
        .unwrap_or_else(|detail| panic!("continuous tracer injection: {detail}"));
    spawned.weight = weight;
    spawned.run_seed = run_seed;
    spawned.next_id = next_id;
    spawned.injection_remainder = remainder;
    let key = SamplingKey { run_seed, epoch: iteration | (1 << 62) };
    let place = |owner: ContainerId| -> ([f64; D], [f64; D]) {
        for (k, tile) in tiles.iter().enumerate() {
            if level_id(held, level, k).is_none() {
                continue;
            }
            if let Some((lev, coord)) = tile.tracer_cell(owner) {
                if lev == level {
                    let state = &tile.levels[level].state;
                    let geometry = state.geom.block_geometry(state.physics.metric);
                    let (_, scale_end, _, offset_end) = symbi_sim::tracers::continuous_tracer_mesh_step(&state.store, state.dt);
                    let lo: [f64; D] = state.geom.stagger_coord(coord, [symbi_sim::state::Loc::Face; D]);
                    let width: [f64; D] = std::array::from_fn(|dd| geometry.cell_width(coord, dd));
                    return (std::array::from_fn(|dd| scale_end * lo[dd] + offset_end[dd]), width.map(|value| scale_end * value));
                }
            }
        }
        // a cell another worker holds: that worker places the tracer
        ([0.0; D], [1.0; D])
    };
    symbi_sim::tracers::spawn_continuous_injected_tracers(
        &mut spawned,
        injections.into_iter().map(|(destination, mass)| symbi_sim::mass_transport::MassTransfer { destination, mass }),
        place,
        key,
    )
    .unwrap_or_else(|detail| panic!("launched refined continuous tracer injection: {detail}"));
    for kk in 0..spawned.len {
        let record = spawned.record_host(kk).unwrap_or_else(|detail| panic!("continuous tracer injection: {detail}"));
        let Some((_, target)) = atlas.holder(record.owner) else { continue };
        if let Some(k) = (0..tiles.len()).find(|&k| level_id(held, level, k).is_some_and(|id| id.0 as usize == target)) {
            tiles[k].levels[level]
                .state
                .continuous_tracers
                .as_mut()
                .expect("a traced level carries continuous tracers")
                .push_host(record)
                .unwrap_or_else(|detail| panic!("continuous tracer injection: {detail}"));
        }
    }
    for tile in tiles.iter_mut() {
        for data in &mut tile.levels {
            if let Some(tracers) = data.state.continuous_tracers.as_mut() {
                tracers.next_id = spawned.next_id;
                tracers.injection_remainder = spawned.injection_remainder;
            }
        }
    }
    Ok(())
}

/// the continuous tracers relocated onto the finest level whose cell holds their position, on the
/// tile holding that cell, over every worker, once per accepted root step, as the single hierarchy
/// relocates them over its levels; a tracer outside every level stays where it is
pub fn relocate_continuous_to_finest<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    atlas: &LevelAtlas<D>,
    depth: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    use symbi_sim::worker_tracers::{continuous_words, decode_continuous, encode_continuous};
    let locate = |position: [f64; D]| {
        (0..depth).rev().find_map(|level| {
            let (id, cell) = atlas.holder_at(level, position)?;
            let scale = RATIO.pow(level as u32);
            let cells: [usize; D] = std::array::from_fn(|ax| atlas.root_cells[ax] * scale);
            let owner = symbi_sim::tracers::cell_container_id(symbi_sim::tracers::cell_address(cell, cells), level as u8);
            Some((level, id, owner))
        })
    };
    let held_at = |tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], level: usize, id: usize| {
        (0..tiles.len()).find(|&k| level < tiles[k].levels.len() && level_id(held, level, k).is_some_and(|t| t.0 as usize == id))
    };
    let width = 2 + continuous_words(D);
    let mut local = Vec::new();
    let mut kept = Vec::new();
    for k in 0..tiles.len() {
        let mut words = Vec::new();
        for source_level in 0..tiles[k].levels.len() {
            let Some(len) = tiles[k].levels[source_level].state.continuous_tracers.as_ref().map(|t| t.len) else { continue };
            for kk in (0..len).rev() {
                let tracers = tiles[k].levels[source_level].state.continuous_tracers.as_mut().unwrap();
                let (inactive, position) = unsafe {
                    (
                        *tracers.escaped.as_ptr::<u8>().add(kk) != 0 || *tracers.crossed_sink.as_ptr::<u8>().add(kk) != 0,
                        std::array::from_fn(|dd| *tracers.x[dd].as_ptr::<f64>().add(kk)),
                    )
                };
                if inactive {
                    continue;
                }
                let Some((level, id, owner)) = locate(position) else { continue };
                let here = level_id(held, level, k).is_some_and(|t| t.0 as usize == id);
                if here && level == source_level {
                    unsafe { *tracers.owner.as_mut_ptr::<ContainerId>().add(kk) = owner };
                    continue;
                }
                let mut record = tracers.swap_remove_host(kk).unwrap_or_else(|detail| panic!("continuous tracer refinement transfer: {detail}"));
                record.owner = owner;
                if let Some(target) = held_at(tiles, level, id) {
                    kept.push((target, level, record));
                } else {
                    words.push(level as u64);
                    words.push(id as u64);
                    encode_continuous(&record, &mut words);
                }
            }
        }
        local.push((held.root_ids[k].0 as usize, words));
    }
    for (target, level, record) in kept {
        tiles[target].levels[level]
            .state
            .continuous_tracers
            .as_mut()
            .expect("every level of a traced tile carries continuous tracers")
            .push_host(record)
            .unwrap_or_else(|detail| panic!("continuous tracer refinement transfer: {detail}"));
    }
    let blocks = assemble_blocks(fabric, held.n_root, &local, width, deadline)?;
    for block in &blocks {
        for words in block.chunks_exact(width) {
            let (level, id) = (words[0] as usize, words[1] as usize);
            if let Some(k) = held_at(tiles, level, id) {
                tiles[k].levels[level]
                    .state
                    .continuous_tracers
                    .as_mut()
                    .expect("every level of a traced tile carries continuous tracers")
                    .push_host(decode_continuous::<D>(&words[2..]))
                    .unwrap_or_else(|detail| panic!("continuous tracer refinement transfer: {detail}"));
            }
        }
    }
    Ok(())
}
