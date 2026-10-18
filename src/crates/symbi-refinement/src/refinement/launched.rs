// =============================================================================
// launched.rs
//
// the refined hydro transaction on the launched path: one root level and one
// fine level of ratio 2 over a patch that may span tile cuts, every worker
// holding its root tiles and the fine tiles under them (or no fine tile at
// all), advanced in lockstep through the fabric. the root step is one
// transaction: the root-attempt snapshot of every level, the root stages with
// their troubled and state exchanges, the root tail, the fine subcycle (each
// substep prolonged from the time-interpolated root, its stages exchanged on
// the fine plan, its tail refreshed at the post-tail point), the restriction
// with its reflux and the root's post-restriction refresh, then the clock. a
// rejection at any stage of any level restores every level's snapshot on
// every worker, refreshes both levels' halos at the rollback point and replays
// the step at a halved timestep. the fine tail's viscous pass and bodies run on
// the globally finest level alone and the root books the horizon ledger, as the
// single hierarchy runs them. three plans share the fabric: the root state, the root conserved
// seed (moved at the prime alone, from which the fine level is prolonged) and
// the fine state.
//
// usage:
//  let report = evolve_refined_worker(&mut tiles, &held, &plans, transport, &mut fabric, &cfg, on_step)?;
// =============================================================================
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

use symbi_fabric::{CHECKPOINT_ATTEMPT, Fabric, Link, OpKind, WorkerId};
use symbi_sim::checkpoint::{
    GlobalGrid, LevelHeader, LevelTiles, PhysicsIdentity, TileView, checkpoint_level_count, load_partitioned_level,
};
use symbi_sim::distributed_checkpoint::{CheckpointRequest, LevelPlan, distributed_checkpoint_levels};
use symbi_grid::Field;
use symbi_hydro::regime::Regime;
use symbi_sim::atlas::{AuditAt, ExchangePoint, FinePoint, TileId};
use symbi_sim::decomp::{HaloTransport, plan_cons_fields, plan_fields};
use symbi_sim::driver::advance_clock;
use symbi_sim::step_transaction::{AttemptOp, AttemptReply, RootStep, ScopedAttempt, StepHalt, take_root_step};
use symbi_sim::hydro_ops::scan_c2p_errors;
use symbi_sim::plan_exchange::{AuditError, PlanExchange, PlanRole, StagingFault, audit_plans};
use symbi_sim::state::FieldStore;
use symbi_sim::substrate_seam::KernelSet;
use symbi_sim::worker::{
    PlanPart, WorkerConfig, WorkerError, WorkerReport, WorkerTiming, exchange_point_plans, fail,
    refuse,
};
use symbi_xpu::{ExecutionSpace, MemorySpace, with_device};
use symbi_geometry::Metric;

use super::hierarchy::{Hierarchy, PrimStencil, RATIO, SeedGhosts};
use super::launched_tracers::{
    LevelAtlas, exchange_level_coefficients, exchange_level_tracers, migrate_level_continuous, relocate_continuous_to_finest,
    spawn_level_continuous, spawn_level_injection,
};

/// the three plans of a refined session and the fabric they share
pub struct RefinedPlans<'p, const D: usize> {
    /// the root state, bands and audits at the root points
    pub root: PlanExchange<'p, D>,
    /// the root conserved halos at the prime, the fine level's seed
    pub seed: PlanExchange<'p, D>,
    /// the fine state, bands and audits at the fine points and the rollback
    pub fine: PlanExchange<'p, D>,
    /// the flux-register contributions of the patch faces lying on cuts, at every fine
    /// stage point; absent when no patch face lies on a cut
    pub flux: Option<PlanExchange<'p, D>>,
    /// the EMF-register contributions of those faces, at every fine post-tail point; absent
    /// on a hydrodynamic run or when no patch face lies on a cut
    pub emf: Option<PlanExchange<'p, D>>,
    /// the neighbor's copies of the staggered faces on those cuts, at the last substep's
    /// post-tail point after the restriction; absent with `emf`
    pub faces: Option<PlanExchange<'p, D>>,
    /// the copies across a tile's own cut of the faces its received units corrected, at the
    /// corrected-faces point; absent with `emf` or when no such face lies on a cut
    pub corrected: Option<PlanExchange<'p, D>>,
    /// the refined tiles' parent face stencils at the attempt's entry and after the root
    /// stages, gathered at the last root stage point: two instances of one face-stencil plan,
    /// present on every magnetized refined run
    pub stencil: Option<(PlanExchange<'p, D>, PlanExchange<'p, D>)>,
    /// the refined tiles' parent prim stencils at the attempt's entry and after the root stages,
    /// gathered at the last root stage point and at the primed point: two instances of one
    /// cell-stencil plan, present on a gas run whose patch face on a cut prolongs past the root halo
    pub prim_stencil: Option<(PlanExchange<'p, D>, PlanExchange<'p, D>)>,
    /// the plans of every nested level, the level `l` at `l - 2`: absent on a run of one
    /// refined level
    pub nested: Vec<NestedPlans<'p, D>>,
    /// the continuous tracers' ito coefficients of every level, the level `l` at `l`, at the
    /// level's coefficient points: empty on a run without them
    pub coefficients: Vec<PlanExchange<'p, D>>,
    /// the band plans of every level, the level `l` at `l`, at the band points of the seams in
    /// `band_seams`: empty on a run whose balanced bands each lie on one tile
    pub bands: Vec<PlanExchange<'p, D>>,
    /// the seams whose balanced band a cut runs inside, with the cuts' distances from the seam, in
    /// the order the restriction rewrites them
    pub band_seams: Vec<symbi_sim::phase_table::BandHops>,
    /// the faces each cut of the finest level shares, from their owner, at the slip points: present
    /// on a run carrying a magnetic-slip body
    pub slip_shared: Option<PlanExchange<'p, D>>,
    /// one cell field of the finest level at the slip points, present with `slip_shared`
    pub slip_cells: Option<PlanExchange<'p, D>>,
}

/// the plans of one nested refined level (two or deeper)
pub struct NestedPlans<'p, const D: usize> {
    /// the level's state at its own points, after the priming, after the restriction of the
    /// level below it, and at the rollback
    pub state: PlanExchange<'p, D>,
    /// the parent level's conserved halos at the level's seed
    pub seed: PlanExchange<'p, D>,
    /// the level's parent face stencils at the parent substep's start and after its stages,
    /// gathered at the parent's last stage point: two instances of one plan on the parent's
    /// partition, present on a magnetized run
    pub stencil: Option<(PlanExchange<'p, D>, PlanExchange<'p, D>)>,
    /// the flux-register contributions of the level's patch faces lying on cuts of its parent, at
    /// every stage point of the level, on the parent's partition: absent when no such face lies on
    /// a cut
    pub flux: Option<PlanExchange<'p, D>>,
    /// the level's parent prim stencils at the parent substep's start and after its stages,
    /// gathered at the parent's last stage point and at the primed point: two instances of one
    /// cell-stencil plan on the parent's partition, present on a gas run whose level face on a
    /// parent cut prolongs past the parent's halo
    pub prim_stencil: Option<(PlanExchange<'p, D>, PlanExchange<'p, D>)>,
}

/// the tiles of one nested level this worker holds, parallel over the held root tiles
pub struct NestedHeld {
    /// the level's tile under each held root tile, when the level reaches it
    pub ids: Vec<Option<TileId>>,
    pub n: usize,
    /// the device of every tile of the level's plan, by tile id
    pub devices: Vec<i32>,
}

/// the restriction of level `level + 1` into level `level` on the held tiles holding both, with
/// the seam bands rewritten seam by seam in the single hierarchy's order: the tiles holding a
/// band's uncovered cell rewrite their part, and at each seam whose band a cut runs inside every
/// worker opens the seam's band point, where the restricted density and the band equilibria beside
/// the cuts cross them, before the tiles past the cut resume their chains there. a worker holding
/// no band cell opens the points with nothing to send. the equilibria a far tile reads are the ones
/// the near tiles wrote in this attempt, delivered at this attempt's epoch
#[allow(clippy::too_many_arguments)]
fn restrict_with_bands<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    level: usize,
    path: u32,
    alpha: f64,
    step: u64,
    attempt: u16,
    transport: &T,
    fabric: &mut Fabric<L>,
    deadline: Duration,
    spent: &mut Duration,
) -> Result<(), symbi_fabric::FabricError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
{
    use crate::refinement::launched_tracers::level_id;
    use crate::refinement::transfer::{BandSide, band_distance};
    let devices_of = |lv: usize| if lv == 0 { held.root_devices } else { held.level_devices(lv) };
    let dev = |k: usize| devices_of(level)[level_id(held, level, k).expect("a held tile of the level").0 as usize];
    let holders: Vec<usize> = (0..tiles.len()).filter(|&k| tiles[k].levels.len() > level + 1).collect();
    for &k in &holders {
        with_device(dev(k), || tiles[k].level_restrict_cells(level));
    }
    for axis in 0..D {
        for high in [false, true] {
            let seams: Vec<(usize, crate::refinement::transfer::BandSeam<D>)> = holders
                .iter()
                .filter(|&&k| tiles[k].rewrites_bands(level))
                .filter_map(|&k| tiles[k].band_seam(level, axis, high).map(|seam| (k, seam)))
                .collect();
            for (k, seam) in seams.iter().filter(|(_, seam)| seam.side == BandSide::Near) {
                with_device(dev(*k), || tiles[*k].rewrite_band_seam(level, seam));
            }
            let hops = plans.band_seams.iter().find(|b| (b.level, b.axis, b.high) == (level as u8, axis as u8, high));
            // every far band part this worker holds resumes at one of the plan's hops
            for (_, seam) in seams.iter().filter(|(_, seam)| seam.side != BandSide::Near) {
                assert!(
                    hops.is_some_and(|h| h.distances.contains(&band_distance(seam))),
                    "a far part of the level-{level} band on axis {axis} (high {high}) starts {} cells from the \
                     seam, a cut the launch's band plan holds no hop for",
                    band_distance(seam)
                );
            }
            let Some(hops) = hops else {
                continue;
            };
            // the far tiles resume the chains hop by hop: across the cut nearest the seam first,
            // each from the equilibrium the tile toward the seam reached beside it
            for (hop, &distance) in hops.distances.iter().enumerate() {
                let mut fields: [Vec<Vec<&symbi_grid::Field<f64, D, Mem>>>; 2] = [level, level + 1]
                    .map(|lv| (0..plans.bands[lv].plan().n_tiles()).map(|_| Vec::new()).collect());
                for (side, lv) in [level, level + 1].into_iter().enumerate() {
                    for (k, tile) in tiles.iter().enumerate() {
                        let Some(id) = level_id(held, lv, k) else { continue };
                        if tile.levels.len() <= lv {
                            continue;
                        }
                        let l = &tile.levels[lv];
                        let anchor = l.band_anchor.get_or_init(|| {
                            symbi_grid::Field::zeros(&l.state.geom.allocated).expect("band anchor scratch")
                        });
                        fields[side][id.0 as usize] = symbi_sim::decomp::plan_band_fields(&l.state, anchor);
                    }
                }
                let parts = [
                    PlanPart { plan: &plans.bands[level], fields: &fields[0], devices: devices_of(level), fault: None, dst_fields: None },
                    PlanPart { plan: &plans.bands[level + 1], fields: &fields[1], devices: devices_of(level + 1), fault: None, dst_fields: None },
                ];
                let point = ExchangePoint::BandSeam { level: level as u8, path, axis: axis as u8, high, hop: hop as u8 };
                exchange_point_plans(&parts, transport, fabric, point, step, attempt, deadline, spent)?;
                for (k, seam) in seams.iter().filter(|(_, seam)| seam.side != BandSide::Near && band_distance(seam) == distance) {
                    with_device(dev(*k), || tiles[*k].rewrite_band_seam(level, seam));
                }
            }
        }
    }
    for &k in &holders {
        with_device(dev(k), || tiles[k].level_restrict_finish(level, alpha));
    }
    Ok(())
}

/// the parent prim stencil fields of refined level `level >= 2`, per tile of its parent level: the
/// cells each parent tile gives at the parent step's entry, its current primitives, and the
/// stencil's old and new time levels
#[allow(clippy::type_complexity)]
fn nested_prim_stencil_lists<'t, R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &'t [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    level: usize,
) -> [Vec<Vec<&'t Field<f64, D, Mem>>>; 4]
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    let parent = level - 1;
    let n = held.level_tiles(parent);
    let mut lists: [Vec<Vec<&Field<f64, D, Mem>>>; 4] = std::array::from_fn(|_| (0..n).map(|_| Vec::new()).collect());
    for (k, h) in tiles.iter().enumerate() {
        let Some(id) = held.level_ids(parent)[k] else { continue };
        let t = id.0 as usize;
        if let Some(entry) = h.prim_entry_at(parent) {
            lists[0][t] = PrimStencil::components(entry);
        }
        lists[1][t] = PrimStencil::components(&h.levels[parent].state.fields.prim);
        if let Some(stencil) = h.prim_stencil_at(level) {
            lists[2][t] = PrimStencil::components(&stencil.old);
            lists[3][t] = PrimStencil::components(&stencil.new);
        }
    }
    lists
}

/// the register slabs of level `parent`'s flux registers (its interface with the finer level), per
/// tile of level `parent >= 1`
fn level_flux_lists<'t, R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &'t [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    parent: usize,
) -> Vec<Vec<&'t Field<f64, D, Mem>>>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.level_tiles(parent)).map(|_| Vec::new()).collect();
    for (k, h) in tiles.iter().enumerate() {
        if let (Some(id), true) = (held.level_ids(parent)[k], h.has_flux_register(parent)) {
            lists[id.0 as usize] = h.flux_registers[parent].plan_fields(&h.levels[parent].state.fields.cons.den);
        }
    }
    lists
}

/// after a stage's units crossed the cuts of level `parent >= 1`: every contributing face of the
/// level's registers cleared for the next stage and every landed unit folded into its receiving
/// face, in the order the register's owner accumulates
fn fold_level_contributions<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    parent: usize,
) where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    for (k, h) in tiles.iter().enumerate() {
        if let (Some(id), true) = (held.level_ids(parent)[k], h.has_flux_register(parent)) {
            with_device(held.level_devices(parent)[id.0 as usize], || {
                h.flux_registers[parent].zero_contributions();
                h.flux_registers[parent].fold_inboxes();
            });
        }
    }
}

/// the tiles holding level `level >= 1` and none finer beside a finer patch face on a cut of the
/// level apply the units they received and recover the level, once, before the level's restriction
/// and bands, as the single hierarchy refluxes and recovers every uncovered cell before its bands
fn reflux_received_nested<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
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
    for (k, h) in tiles.iter().enumerate() {
        if let (Some(id), true) = (held.level_ids(level)[k], h.levels.len() == level + 1 && h.has_flux_register(level)) {
            with_device(held.level_devices(level)[id.0 as usize], || h.reflux_received_at(level));
        }
    }
}

/// the tiles this worker holds: parallel over the held root tiles
pub struct Held<'a> {
    pub root_ids: &'a [TileId],
    /// the fine tile under each held root tile, when the patch reaches it
    pub fine_ids: &'a [Option<TileId>],
    pub n_root: usize,
    pub n_fine: usize,
    /// the device of every root tile of the plan, by root tile id
    pub root_devices: &'a [i32],
    /// the device of every fine tile of the plan, by fine tile id
    pub fine_devices: &'a [i32],
    /// the tiles of every nested level, the level `l` at `l - 2`
    pub nested: &'a [NestedHeld],
}

impl Held<'_> {
    /// the tile ids of refined level `level >= 1`, parallel over the held root tiles
    pub(crate) fn level_ids(&self, level: usize) -> &[Option<TileId>] {
        if level == 1 { self.fine_ids } else { &self.nested[level - 2].ids }
    }

    pub(crate) fn level_tiles(&self, level: usize) -> usize {
        if level == 1 { self.n_fine } else { self.nested[level - 2].n }
    }

    pub(crate) fn level_devices(&self, level: usize) -> &[i32] {
        if level == 1 { self.fine_devices } else { &self.nested[level - 2].devices }
    }
}

#[derive(Debug, Clone)]
pub struct RefinedConfig {
    pub base: WorkerConfig,
    /// report a NaN timestep candidate from this level of the worker's first hierarchy at
    /// this step, leaving every other candidate valid: (step, level)
    pub invalid_level_cfl_at: Option<(u64, usize)>,
    /// raise the troubled-cell flag on a fine cell at (step, substep, stage) on the first
    /// attempt, on whichever worker holds it; the cell is indexed on the fine plan's grid.
    /// the cell's state stays physical, so the correction it triggers is the conservative
    /// first-order redo
    pub fine_trouble_at: Option<(u64, u8, usize, [isize; 3])>,
    /// a fault in this worker's flux contributions at (step, substep, stage) on the first
    /// attempt, for the exactly-once gates
    pub contribution_fault: Option<(u64, u8, usize, StagingFault)>,
    /// a fault in this worker's flux contributions of nested level `level` at (step, level, path,
    /// stage) on the first attempt, the path numbering the level's substeps across the root step
    pub nested_contribution_fault: Option<(u64, u8, u32, usize, StagingFault)>,
    /// move one copy of a fine shared face on a fine cut by one unit in the last place after
    /// this many accepted steps (after the initialization at zero, before the prime audit), on
    /// the first held tile whose fine level has a cut on its low side of axis 0: the next audit
    /// of the fine level must name it
    pub perturb_fine_face_after: Option<u64>,
    /// move the clock of one held fine tile by one unit in the last place (time, dt) or one
    /// step (iteration) after this many accepted steps: (step, which, local fine tile ordinal)
    pub clock_fault: Option<(u64, ClockField, usize)>,
    /// the tiles hold a state loaded from a checkpoint (`load_refined_tiles`): the prime
    /// exchanges and fills it and keeps the loaded fine level, which a fresh start seeds by
    /// prolongation from the root
    pub restart: bool,
}

/// the member of a fine level's clock a fault moves
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockField {
    Time,
    Dt,
    Iteration,
}

/// the fine faces a fault moves: the first held fine tile with a cut on its low side of axis 0,
/// its opening face there on the second interior row
fn perturb_fine_face<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>])
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    let Some(fine) = tiles.iter().filter(|h| h.levels.len() > 1).map(|h| &h.levels[1]).find(|l| l.cut_sides[0][0]) else {
        return;
    };
    let Some(mhd) = fine.state.fields.mhd.as_ref() else { return };
    let at: [isize; D] = std::array::from_fn(|ax| fine.state.geom.interior.spaces[ax].lo + isize::from(ax == 1));
    let field = &mhd.bface.b[0];
    let old = *field.view().at(at);
    *field.view_mut().at_mut(at) = f64::from_bits(old.to_bits() + 1);
}

/// evolve this worker's hierarchies from the configured start until the final time or
/// the step bound. on a fresh start `on_seeded` edits each hierarchy once every level is seeded
/// and before the priming. `on_step` runs once on the primed state with zero accepted steps, then after
/// every accepted step with the count of accepted steps, the time, the tiles and the fabric; a
/// checkpoint written there (`refined_checkpoint`) holds that state.
#[allow(clippy::too_many_arguments)]
pub fn evolve_refined_worker<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L, F, G>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    transport: &T,
    fabric: &mut Fabric<L>,
    cfg: &RefinedConfig,
    mut seed_target: Option<&mut dyn FnMut(&Hierarchy<R, D, DOF, M, E, S, Mem, K>) -> Result<(), String>>,
    mut on_seeded: G,
    mut on_step: F,
) -> Result<WorkerReport, WorkerError>
where
    R: Regime<f64, D, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF> + Copy,
    M: Metric<f64, D> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
    F: FnMut(u64, f64, &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], &mut Fabric<L>) -> Result<ControlFlow<()>, WorkerError>,
    G: FnMut(&mut Hierarchy<R, D, DOF, M, E, S, Mem, K>) -> Result<(), String>,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
    <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
{
    let n = tiles.len();
    assert_eq!(n, held.root_ids.len(), "tiles/root ids length mismatch");
    assert_eq!(n, held.fine_ids.len(), "tiles/fine ids length mismatch");
    assert_eq!(plans.root.role(), PlanRole::RootState);
    assert_eq!(plans.seed.role(), PlanRole::RootSeed);
    assert_eq!(plans.fine.role(), PlanRole::Fine);
    if let Some(flux) = plans.flux.as_ref() {
        assert_eq!(flux.role(), PlanRole::FluxContribution);
    }
    if let Some(emf) = plans.emf.as_ref() {
        assert_eq!(emf.role(), PlanRole::EmfContribution);
    }
    if let Some(faces) = plans.faces.as_ref() {
        assert_eq!(faces.role(), PlanRole::InterfaceFace);
    }
    if let Some(corrected) = plans.corrected.as_ref() {
        assert_eq!(corrected.role(), PlanRole::CorrectedFace);
    }
    assert_eq!(plans.emf.is_some(), plans.faces.is_some(), "the EMF contributions and the interface faces travel together");
    if let Some((old, new)) = plans.stencil.as_ref() {
        assert_eq!((old.role(), new.role()), (PlanRole::FaceStencil, PlanRole::FaceStencil));
    }
    for (i, nested) in plans.nested.iter().enumerate() {
        let level = (i + 2) as u8;
        assert_eq!((nested.state.role(), nested.seed.role()), (PlanRole::Nested { level }, PlanRole::NestedSeed { level }));
        if let Some(flux) = nested.flux.as_ref() {
            assert_eq!(flux.role(), PlanRole::NestedFluxContribution { level });
        }
        if let Some((old, new)) = nested.prim_stencil.as_ref() {
            assert_eq!((old.role(), new.role()), (PlanRole::NestedPrimStencil { level }, PlanRole::NestedPrimStencil { level }));
        }
    }
    assert_eq!(held.nested.len(), plans.nested.len(), "one held level per nested plan");
    // the levels of the session: the root, the fine level, and every nested one
    let depth = 2 + plans.nested.len();
    // a nested patch face on a cut of its parent level carries its flux-register units on the level's
    // contribution plan; its edge-field units ride no launched plan
    let unplanned = tiles.iter().any(|h| {
        h.flux_registers.iter().enumerate().skip(1).any(|(parent, r)| {
            (r.has_contributing() || r.has_receiving()) && plans.nested.get(parent - 1).is_none_or(|n| n.flux.is_none())
        }) || h.emf_registers.iter().skip(1).flatten().any(|r| r.has_contributing() || r.has_receiving())
    });
    if unplanned {
        return Err(fail(
            fabric,
            WorkerError::Refused("a nested patch face on a cut of its parent level carries register units no launched plan moves".into()),
        ));
    }
    // a nested level reading a parent prim stencil gathers it on its level's plan; the primed point
    // gathers every stencil in one phase, so the session carries them on one level alone: a finer
    // stencil reads its parent's coarse-fine ghosts, which a coarser stencil re-prolongs after it
    let unplanned_stencil = tiles
        .iter()
        .any(|h| (2..h.levels.len()).any(|level| h.prim_stencil_at(level).is_some() && plans.nested[level - 2].prim_stencil.is_none()));
    if unplanned_stencil {
        return Err(fail(
            fabric,
            WorkerError::Refused(
                "a nested tile reads a parent prim stencil the session carries no plan for: its patch face on a cut of \
                 its parent level prolongs past that level's halo"
                    .into(),
            ),
        ));
    }
    let stencil_levels = usize::from(plans.prim_stencil.is_some()) + plans.nested.iter().filter(|n| n.prim_stencil.is_some()).count();
    if stencil_levels > 1 {
        return Err(fail(
            fabric,
            WorkerError::Refused(format!(
                "parent prim stencils on {stencil_levels} levels of one launched session: the primed point gathers \
                 them in one phase, before a finer stencil's parent level holds the coarse-fine ghosts a coarser \
                 stencil prolongs; keep the patch faces of all but one level off the cuts"
            )),
        ));
    }
    // a gas run refluxes the tiles beside a neighbor's patch before the root's bands; a
    // magnetized run applies its units around the corrected-faces exchange after the subcycle
    let gas_reflux_first = plans.emf.is_none() && plans.corrected.is_none() && tiles.iter().all(|h| h.levels[0].state.fields.mhd.is_none());
    let band_start = plans.band_seams.iter().any(|b| b.level == 0 && b.start_across_cut);
    if band_start && !gas_reflux_first {
        return Err(fail(
            fabric,
            WorkerError::Refused("a balanced band starting across a cut on a magnetized refined launch".into()),
        ));
    }

    let base = &cfg.base;
    let deadline = base.deadline;
    let stages = base.timestepping.stages();
    let nstages = stages.len();
    let table = plans.root.table();
    if table.stages() as usize != nstages
        || table.substeps() as usize != RATIO
        || table.fine_stages() as usize != nstages
        || table.depth() as usize != depth
    {
        return Err(fail(
            fabric,
            WorkerError::Refused(format!(
                "the session's phase table holds {} root stages, {} substeps of {} stages over {} levels; \
                 this worker runs {nstages} stages with {RATIO} substeps over {depth} levels",
                table.stages(),
                table.substeps(),
                table.fine_stages(),
                table.depth()
            )),
        ));
    }
    for h in tiles.iter() {
        if h.levels.len() > depth {
            return Err(fail(
                fabric,
                WorkerError::Refused(format!(
                    "a hierarchy of {} levels in a session of {depth} levels is outside the launched refinement",
                    h.levels.len()
                )),
            ));
        }
    }
    // the finest-level physics belongs to the session's finest level, whichever levels a tile holds
    for h in tiles.iter_mut() {
        h.set_run_levels(depth);
    }
    let root_dev = |k: usize| held.root_devices[held.root_ids[k].0 as usize];
    let fine_dev = |k: usize| held.fine_devices[held.fine_ids[k].expect("a refined tile").0 as usize];
    let refined: Vec<usize> = (0..n).filter(|&k| tiles[k].levels.len() > 1).collect();
    for (k, h) in tiles.iter().enumerate() {
        assert_eq!(h.levels.len() > 1, held.fine_ids[k].is_some(), "a fine tile id per refined tile");
    }
    // the regime and operator scope, on every level this worker holds
    {
        let root_stores: Vec<&FieldStore<D, DOF, Mem, f64>> = tiles.iter().map(|h| &*h.levels[0].state).collect();
        let root_kernels: Vec<&K> = tiles.iter().map(|h| &h.levels[0].kernels).collect();
        // the relativistic gas and the relativistic magnetized gas carry a curved spacetime on the
        // refined workers
        let curved = matches!(<R as Regime<f64, D>>::SPEC.name, "rhd" | "rmhd");
        refuse(&root_stores, &root_kernels, curved).map_err(|e| fail(fabric, e))?;
        let fine_stores: Vec<&FieldStore<D, DOF, Mem, f64>> = refined.iter().map(|&k| &*tiles[k].levels[1].state).collect();
        let fine_kernels: Vec<&K> = refined.iter().map(|&k| &tiles[k].levels[1].kernels).collect();
        refuse(&fine_stores, &fine_kernels, curved).map_err(|e| fail(fabric, e))?;
        if plans.root.plan().seam.is_some() {
            return Err(fail(fabric, WorkerError::Refused("a polar seam with refinement is outside the launched refinement".into())));
        }
        refuse_transported_registers(tiles).map_err(|e| fail(fabric, e))?;
        let magnetized = tiles.iter().any(|h| h.levels[0].state.fields.mhd.is_some());
        if magnetized && plans.nested.iter().any(|n| n.stencil.is_none()) {
            return Err(fail(
                fabric,
                WorkerError::Refused(
                    "a magnetized nested level without its parent face stencil is outside the launched refinement: \
                     its face prolongation reads parent faces past a tile's face lattice"
                        .into(),
                ),
            ));
        }
        if magnetized
            && (plans.stencil.is_none()
                || refined.iter().any(|&k| tiles[k].face_stencil().is_none())
                || tiles.iter().any(|h| h.face_entry().is_none()))
        {
            return Err(fail(
                fabric,
                WorkerError::Refused(
                    "a magnetized refined run without the parent face stencil is outside the launched refinement: \
                     the fine face prolongation reads parent faces past a tile's face lattice"
                        .into(),
                ),
            ));
        }
        // a patch face on a cut prolonging past the root halo reads a prim stencil the session gathers
        if tiles.iter().any(|h| h.prim_stencil().is_some()) && plans.prim_stencil.is_none() {
            return Err(fail(
                fabric,
                WorkerError::Refused(
                    "a refined tile reads a parent prim stencil the session carries no plan for: its patch face on a \
                     cut prolongs past the root halo"
                        .into(),
                ),
            ));
        }
        for h in tiles.iter() {
            h.prim_stencil_admission().map_err(|e| fail(fabric, WorkerError::Refused(e)))?;
        }
        let forced = !base.injection.reject_at.is_empty() || tiles.iter().any(|h| h.forced_fine_rejection_pending());
        if forced {
            for h in tiles.iter() {
                for level in &h.levels {
                    if !level.kernels.field_rollback_is_complete(&level.state) {
                        return Err(fail(
                            fabric,
                            WorkerError::Refused(
                                "a forced rejection on a kernel set whose snapshot leaves out a grid field is \
                                 outside the distributed first release"
                                    .into(),
                            ),
                        ));
                    }
                }
            }
        }
    }
    let mut timing = WorkerTiming::default();
    let march_start = Instant::now();
    let mut in_callback = Duration::ZERO;
    let mut report = WorkerReport::default();

    // the field lists of each plan: root primitives by root tile id, root conserved by root
    // tile id, fine primitives by fine tile id; rebuilt per point since the borrows are short
    macro_rules! root_fields {
        () => {{
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                lists[held.root_ids[k].0 as usize] = plan_fields(&h.levels[0].state);
            }
            lists
        }};
    }
    macro_rules! seed_fields {
        () => {{
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                lists[held.root_ids[k].0 as usize] = plan_cons_fields(&h.levels[0].state);
            }
            lists
        }};
    }
    macro_rules! fine_fields {
        () => {{
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_fine).map(|_| Vec::new()).collect();
            for &k in &refined {
                lists[held.fine_ids[k].unwrap().0 as usize] = plan_fields(&tiles[k].levels[1].state);
            }
            lists
        }};
    }
    // the state or the conserved fields of refined level `level`, by that level's tile id
    macro_rules! level_fields {
        ($level:expr, $of:ident) => {{
            let level: usize = $level;
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.level_tiles(level)).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                if let Some(id) = held.level_ids(level)[k] {
                    lists[id.0 as usize] = $of(&h.levels[level].state);
                }
            }
            lists
        }};
    }
    // the stencil of nested level `$level` from its parent level's faces: the old time level from
    // the parent's faces at its substep's start, the new one from its faces after the stages; the
    // sources and destinations indexed by the parent level's tile id
    macro_rules! nested_stencil_lists {
        ($level:expr) => {{
            let level: usize = $level;
            let parent = level - 1;
            let n = held.level_tiles(parent);
            let mut old_src: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
            let mut new_src: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
            let mut old_dst: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
            let mut new_dst: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                if let Some(id) = held.level_ids(parent)[k] {
                    let t = id.0 as usize;
                    if let Some(entry) = h.level_face_entry(parent) {
                        old_src[t] = entry.iter().collect();
                    }
                    if let Some(mhd) = h.levels[parent].state.fields.mhd.as_ref() {
                        new_src[t] = mhd.bface.b.iter().collect();
                    }
                    if let Some(stencil) = h.face_stencil_of(level) {
                        old_dst[t] = stencil.old.iter().collect();
                        new_dst[t] = stencil.new.iter().collect();
                    }
                }
            }
            (old_src, new_src, old_dst, new_dst)
        }};
    }
    macro_rules! root_point {
        ($point:expr, $step:expr, $attempt:expr) => {{
            let fields = root_fields!();
            let part = PlanPart { plan: &plans.root, fields: &fields, devices: held.root_devices, fault: None, dst_fields: None };
            exchange_point_plans(&[part], transport, fabric, $point, $step, $attempt, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
            for (k, h) in tiles.iter().enumerate() {
                with_device(root_dev(k), || h.levels[0].kernels.ghost_fill(&h.levels[0].state));
            }
        }};
    }
    macro_rules! fine_point {
        ($point:expr, $step:expr, $attempt:expr) => {{
            let fields = fine_fields!();
            let part = PlanPart { plan: &plans.fine, fields: &fields, devices: held.fine_devices, fault: None, dst_fields: None };
            exchange_point_plans(&[part], transport, fabric, $point, $step, $attempt, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
            for &k in &refined {
                let h = &tiles[k];
                with_device(fine_dev(k), || h.fill_physical_ghosts(1));
            }
        }};
    }
    // the magnetic-slip palindrome on the run's finest level: that level's state plan, the tiles
    // of it this worker holds, and the level's halos after each split operator
    let slip_level = depth - 1;
    macro_rules! slip_plan {
        () => {
            if slip_level == 1 { &plans.fine } else { &plans.nested[slip_level - 2].state }
        };
    }
    let slip_holders: Vec<usize> = (0..n).filter(|&k| tiles[k].levels.len() > slip_level).collect();
    // the finest level's primitives recovered, then its halos across the cuts at `$point` and its
    // physical fill, as the single hierarchy rebuilds after a split operator
    macro_rules! slip_rebuild {
        ($point:expr, $step:expr, $attempt:expr) => {{
            for &k in &slip_holders {
                let level = &tiles[k].levels[slip_level];
                with_device(root_dev(k), || level.kernels.c2p(&level.state));
            }
            let fields = level_fields!(slip_level, plan_fields);
            let part = PlanPart { plan: slip_plan!(), fields: &fields, devices: held.level_devices(slip_level), fault: None, dst_fields: None };
            exchange_point_plans(&[part], transport, fabric, $point, $step, $attempt, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
            for &k in &slip_holders {
                let h = &tiles[k];
                with_device(root_dev(k), || h.fill_physical_ghosts(slip_level));
            }
        }};
    }
    // the material drain of half `$half` on the finest level at the whole level's Alfven stiffness
    // (a maximum over every worker's tiles of it), then the rebuild
    macro_rules! slip_drain {
        ($dt:expr, $half:expr, $step:expr, $attempt:expr) => {{
            let ids = held.level_ids(slip_level);
            let local: Vec<(u64, u64)> = slip_holders
                .iter()
                .map(|&k| (ids[k].expect("a holder holds the level").0 as u64, symbi_sim::state::local_c_a2_max(&tiles[k].levels[slip_level].state).to_bits()))
                .collect();
            let assembled = fabric.assemble_windowed(held.level_tiles(slip_level) as u64, &local, deadline).map_err(|e| fail(fabric, e.into()))?;
            let mut c_a2 = 0.0_f64;
            for (tile, bits) in assembled.iter().enumerate() {
                if *bits == symbi_fabric::ABSENT {
                    return Err(fail(fabric, WorkerError::Numerics(format!("no worker supplied level {slip_level} tile {tile}'s Alfven stiffness"))));
                }
                c_a2 = c_a2.max(f64::from_bits(*bits));
            }
            for &k in &slip_holders {
                let level = &tiles[k].levels[slip_level];
                if let Some(im) = level.state.immersed.as_ref() {
                    im.set_c_a2_override(c_a2);
                }
                with_device(root_dev(k), || level.kernels.penalize(&level.state, $dt));
            }
            slip_rebuild!(ExchangePoint::SlipDrain { half: $half }, $step, $attempt);
        }};
    }
    // the slip midpoint of half `$half` over every worker's tiles of the finest level in lockstep,
    // a worker holding none of them joining every exchange and sum; its heat booked once, by the
    // owner of the level's first tile. evaluates to whether the solve converged
    macro_rules! slip_mid {
        ($dt:expr, $half:expr, $step:expr, $attempt:expr) => {{
            let level_ids = held.level_ids(slip_level);
            let ids: Vec<TileId> = slip_holders.iter().map(|&k| level_ids[k].expect("a holder holds the level")).collect();
            let books = slip_plan!().placement().owner_of(TileId(0)) == fabric.me();
            let converged = {
                let stores: Vec<&FieldStore<D, DOF, Mem, f64>> = slip_holders.iter().map(|&k| &*tiles[k].levels[slip_level].state).collect();
                let (reference, kernels) = match slip_holders.first() {
                    Some(&k) => (&*tiles[k].levels[slip_level].state, &tiles[k].levels[slip_level].kernels),
                    None => (&*tiles[0].levels[0].state, &tiles[0].levels[0].kernels),
                };
                let cuts = symbi_sim::worker::WorkerSlipCuts::new(
                    slip_plan!(),
                    plans.slip_shared.as_ref().expect("a slip run carries its shared-face plan"),
                    plans.slip_cells.as_ref().expect("a slip run carries its cell plan"),
                    &ids,
                    held.level_devices(slip_level),
                    transport,
                    &mut *fabric,
                    $half,
                    $step,
                    $attempt,
                    deadline,
                );
                let converged = kernels.magnetic_slip_step_tiles(&stores, reference, $dt, &cuts, books);
                timing.exchange += cuts.finish()?;
                converged
            };
            if converged {
                slip_rebuild!(ExchangePoint::SlipRebuild { half: $half }, $step, $attempt);
            }
            converged
        }};
    }
    // every tile's covered coarser cells take the finest state, then every coarser level's halos
    // cross the cuts and its physical ghosts fill
    macro_rules! slip_sync {
        ($half:expr, $step:expr, $attempt:expr) => {{
            for (k, h) in tiles.iter().enumerate() {
                with_device(root_dev(k), || h.sync_all_fine_to_coarse());
            }
            let root = root_fields!();
            let coarser: Vec<Vec<Vec<&Field<f64, D, Mem>>>> = (1..slip_level).map(|level| level_fields!(level, plan_fields)).collect();
            let mut parts = vec![PlanPart { plan: &plans.root, fields: &root, devices: held.root_devices, fault: None, dst_fields: None }];
            for (i, level) in (1..slip_level).enumerate() {
                let plan = if level == 1 { &plans.fine } else { &plans.nested[level - 2].state };
                parts.push(PlanPart { plan, fields: &coarser[i], devices: held.level_devices(level), fault: None, dst_fields: None });
            }
            exchange_point_plans(&parts, transport, fabric, ExchangePoint::SlipSync { half: $half }, $step, $attempt, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
            for (k, h) in tiles.iter().enumerate() {
                with_device(root_dev(k), || {
                    h.levels[0].kernels.ghost_fill(&h.levels[0].state);
                    for level in 1..slip_level.min(h.levels.len()) {
                        h.fill_physical_ghosts(level);
                    }
                });
            }
        }};
    }
    // the register slabs of the contributions plan, per root tile: a tile without a
    // register lists nothing, since the plan references none of its slots
    macro_rules! flux_fields {
        () => {{
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                if h.has_flux_register(0) {
                    lists[held.root_ids[k].0 as usize] = h.flux_registers[0].plan_fields(&h.levels[0].state.fields.cons.den);
                }
            }
            lists
        }};
    }
    // the EMF slabs of the contributions plan and the staggered faces of the replacement
    // plan, per root tile
    macro_rules! emf_fields {
        () => {{
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                if let Some(Some(reg)) = h.emf_registers.first() {
                    lists[held.root_ids[k].0 as usize] = reg.plan_fields(&h.levels[0].state.fields.cons.den);
                }
            }
            lists
        }};
    }
    macro_rules! face_fields {
        () => {{
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                if let Some(mhd) = h.levels[0].state.fields.mhd.as_ref() {
                    lists[held.root_ids[k].0 as usize] = mhd.bface.b.iter().collect();
                }
            }
            lists
        }};
    }
    // the root faces at the attempt's entry, per root tile, the old-time stencil source
    macro_rules! entry_fields {
        () => {{
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                if let Some(entry) = h.face_entry() {
                    lists[held.root_ids[k].0 as usize] = entry.iter().collect();
                }
            }
            lists
        }};
    }
    // the prim stencil's sources and destinations per root tile: each tile's donated cells at the
    // attempt's entry, its current primitives, and the stencil's two time levels
    macro_rules! prim_stencil_fields {
        () => {{
            let empty = || -> Vec<Vec<&Field<f64, D, Mem>>> { (0..held.n_root).map(|_| Vec::new()).collect() };
            let (mut entry, mut now, mut old, mut new) = (empty(), empty(), empty(), empty());
            for (k, h) in tiles.iter().enumerate() {
                let id = held.root_ids[k].0 as usize;
                now[id] = PrimStencil::components(&h.levels[0].state.fields.prim);
                if let Some(e) = h.prim_entry() {
                    entry[id] = PrimStencil::components(e);
                }
                if let Some(st) = h.prim_stencil() {
                    old[id] = PrimStencil::components(&st.old);
                    new[id] = PrimStencil::components(&st.new);
                }
            }
            (entry, now, old, new)
        }};
    }
    // the refined tiles' stencil fields at both time levels, per root tile
    macro_rules! stencil_fields {
        () => {{
            let mut old: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            let mut new: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
            for (k, h) in tiles.iter().enumerate() {
                if let Some(stencil) = h.face_stencil() {
                    old[held.root_ids[k].0 as usize] = stencil.old.iter().collect();
                    new[held.root_ids[k].0 as usize] = stencil.new.iter().collect();
                }
            }
            (old, new)
        }};
    }
    // the fine post-tail point: the fine state and, with it, the EMF contributions of the
    // substep and, at the last substep, the corrected staggered faces on the cuts; every
    // contributing slab is then cleared and every landed unit folded in substep order
    macro_rules! fine_tail_point {
        ($sub:expr, $step:expr, $attempt:expr) => {{
            let point = ExchangePoint::Fine { substep: $sub as u8, point: FinePoint::PostTail };
            let fields = fine_fields!();
            let emf_fields = emf_fields!();
            let face_fields = face_fields!();
            let mut parts = vec![PlanPart { plan: &plans.fine, fields: &fields, devices: held.fine_devices, fault: None, dst_fields: None }];
            if let Some(emf) = plans.emf.as_ref() {
                parts.push(PlanPart { plan: emf, fields: &emf_fields, devices: held.root_devices, fault: None, dst_fields: None });
            }
            if let Some(faces) = plans.faces.as_ref() {
                parts.push(PlanPart { plan: faces, fields: &face_fields, devices: held.root_devices, fault: None, dst_fields: None });
            }
            exchange_point_plans(&parts, transport, fabric, point, $step, $attempt, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
            for &k in &refined {
                let h = &tiles[k];
                with_device(fine_dev(k), || h.fill_physical_ghosts(1));
            }
            if plans.emf.is_some() {
                for (k, h) in tiles.iter().enumerate() {
                    if let Some(Some(reg)) = h.emf_registers.first() {
                        with_device(root_dev(k), || {
                            reg.zero_contributions();
                            reg.fold_inboxes();
                        });
                    }
                }
            }
        }};
    }
    // the fine stage point: the fine state and, with it, the flux contributions of the
    // stage; every contributing face is then cleared for the next stage and every landed
    // unit folded into its receiving face, in the order the register's owner accumulates
    macro_rules! fine_stage_point {
        ($sub:expr, $stage:expr, $step:expr, $attempt:expr) => {{
            let point = ExchangePoint::Fine { substep: $sub as u8, point: FinePoint::Stage($stage as u8) };
            let fields = fine_fields!();
            let flux_fields = flux_fields!();
            // the last fine stage also gathers the level-2 parent face stencils of a nested run
            let nested_stencil = plans.nested.first().and_then(|n| n.stencil.as_ref()).filter(|_| $stage + 1 == nstages);
            let (old_src, new_src, old_dst, new_dst) = if nested_stencil.is_some() {
                nested_stencil_lists!(2)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
            // and the level-2 parent prim stencils: level 1's cells at the substep's start and after
            // its stages
            let nested_prim = plans.nested.first().and_then(|n| n.prim_stencil.as_ref()).filter(|_| $stage + 1 == nstages);
            let [prim_entry, prim_now, prim_old, prim_new] = if nested_prim.is_some() {
                nested_prim_stencil_lists(tiles, held, 2)
            } else {
                std::array::from_fn(|_| Vec::new())
            };
            let mut parts = vec![PlanPart { plan: &plans.fine, fields: &fields, devices: held.fine_devices, fault: None, dst_fields: None }];
            if let Some((old, new)) = nested_stencil {
                parts.push(PlanPart { plan: old, fields: &old_src, devices: held.fine_devices, fault: None, dst_fields: Some(&old_dst) });
                parts.push(PlanPart { plan: new, fields: &new_src, devices: held.fine_devices, fault: None, dst_fields: Some(&new_dst) });
            }
            if let Some((old, new)) = nested_prim {
                parts.push(PlanPart { plan: old, fields: &prim_entry, devices: held.fine_devices, fault: None, dst_fields: Some(&prim_old) });
                parts.push(PlanPart { plan: new, fields: &prim_now, devices: held.fine_devices, fault: None, dst_fields: Some(&prim_new) });
            }
            if let Some(flux) = plans.flux.as_ref() {
                let fault = cfg.contribution_fault.and_then(|(at_step, at_sub, at_stage, fault)| {
                    ($attempt == 0 && (at_step, at_sub, at_stage) == ($step, $sub as u8, $stage)).then_some(fault)
                });
                parts.push(PlanPart { plan: flux, fields: &flux_fields, devices: held.root_devices, fault, dst_fields: None });
            }
            exchange_point_plans(&parts, transport, fabric, point, $step, $attempt, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
            for &k in &refined {
                let h = &tiles[k];
                with_device(fine_dev(k), || h.fill_physical_ghosts(1));
            }
            if nested_stencil.is_some() {
                for h in tiles.iter() {
                    h.mark_face_stencil_filled_of(2);
                }
            }
            if nested_prim.is_some() {
                for h in tiles.iter() {
                    h.mark_prim_stencil_filled_of(2);
                }
            }
            if plans.flux.is_some() {
                for (k, h) in tiles.iter().enumerate() {
                    if h.has_flux_register(0) {
                        with_device(root_dev(k), || {
                            h.flux_registers[0].zero_contributions();
                            h.flux_registers[0].fold_inboxes();
                        });
                    }
                }
            }
        }};
    }
    // the shared faces of both levels in one audit phase
    macro_rules! audit_levels {
        ($point:expr, $step:expr, $attempt:expr) => {{
            if plans.root.has_audits() || plans.fine.has_audits() || plans.nested.iter().any(|p| p.state.has_audits()) {
                let root_lists = root_fields!();
                let fine_lists = fine_fields!();
                let nested_lists: Vec<_> = (2..depth).map(|level| level_fields!(level, plan_fields)).collect();
                let mut audits = vec![(&plans.root, &root_lists[..]), (&plans.fine, &fine_lists[..])];
                for (i, nested) in plans.nested.iter().enumerate() {
                    audits.push((&nested.state, &nested_lists[i][..]));
                }
                let cost = audit_plans(&audits, fabric, $point, $step, $attempt, deadline)
                    .map_err(|e| match e {
                        AuditError::Fabric(e) => fail(fabric, WorkerError::Fabric(e)),
                        AuditError::Mismatch(m) => fail(fabric, WorkerError::Interface(m)),
                    })?;
                report.audits += 1;
                report.audit_cost.faces += cost.faces;
                report.audit_cost.wire_bytes += cost.wire_bytes;
                report.audit_cost.elapsed += cost.elapsed;
            }
        }};
    }
    macro_rules! audit {
        ($plan:expr, $fields:expr, $point:expr, $step:expr, $attempt:expr) => {{
            if $plan.has_audits() {
                let fields = $fields;
                let cost = $plan
                    .audit_at(&fields, fabric, $point, $step, $attempt, deadline)
                    .map_err(|e| match e {
                        AuditError::Fabric(e) => fail(fabric, WorkerError::Fabric(e)),
                        AuditError::Mismatch(m) => fail(fabric, WorkerError::Interface(m)),
                    })?;
                report.audits += 1;
                report.audit_cost.faces += cost.faces;
                report.audit_cost.wire_bytes += cost.wire_bytes;
                report.audit_cost.elapsed += cost.elapsed;
            }
        }};
    }

    // prime: the root primitives and physical ghosts, then the root state and the root
    // conserved halos across the cuts in one phase, the physical ghosts again, the prime
    // audit, then the fine level seeded by prolongation from the exchanged root. a resumed root
    // keeps the primitives its checkpoint stored, as the single hierarchy's resume does
    {
        for (k, h) in tiles.iter().enumerate() {
            with_device(root_dev(k), || {
                if !h.levels[0].state.primitives_loaded {
                    h.levels[0].kernels.c2p(&h.levels[0].state);
                }
                h.levels[0].kernels.ghost_fill(&h.levels[0].state);
            });
        }
        // every gathered stencil lands inside the stencil fields the builder gave each held tile:
        // the plans derive the boxes from the partition, the builder from the level's own sides
        {
            let mut checks: Vec<Result<(), String>> = Vec::new();
            if let Some((old, new)) = plans.stencil.as_ref() {
                let (stencil_old, stencil_new) = stencil_fields!();
                checks.push(old.check_destinations(&stencil_old));
                checks.push(new.check_destinations(&stencil_new));
            }
            if let Some((old, new)) = plans.prim_stencil.as_ref() {
                let (_, _, stencil_old, stencil_new) = prim_stencil_fields!();
                checks.push(old.check_destinations(&stencil_old));
                checks.push(new.check_destinations(&stencil_new));
            }
            for (i, nested) in plans.nested.iter().enumerate() {
                if let Some((old, new)) = nested.stencil.as_ref() {
                    let (_, _, old_dst, new_dst) = nested_stencil_lists!(i + 2);
                    checks.push(old.check_destinations(&old_dst));
                    checks.push(new.check_destinations(&new_dst));
                }
                if let Some((old, new)) = nested.prim_stencil.as_ref() {
                    let [_, _, old_dst, new_dst] = nested_prim_stencil_lists(tiles, held, i + 2);
                    checks.push(old.check_destinations(&old_dst));
                    checks.push(new.check_destinations(&new_dst));
                }
            }
            if let Some(Err(e)) = checks.into_iter().find(Result::is_err) {
                return Err(fail(fabric, WorkerError::Refused(format!("a gathered stencil outgrows its held field: {e}"))));
            }
        }
        // a fresh start fills the root's ghosts for the fine seed before its cut halos cross, so
        // the exchange carries the cells where a face's ghosts meet a cut
        let root_ghosts: Vec<SeedGhosts> = if cfg.restart {
            Vec::new()
        } else {
            tiles.iter().enumerate().map(|(k, h)| with_device(root_dev(k), || h.fill_seed_ghosts(0))).collect()
        };
        {
            let prims = root_fields!();
            let cons = seed_fields!();
            let parts = [
                PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None, dst_fields: None },
                PlanPart { plan: &plans.seed, fields: &cons, devices: held.root_devices, fault: None, dst_fields: None },
            ];
            exchange_point_plans(&parts, transport, fabric, ExchangePoint::Prime, 0, 0, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
        }
        for (k, h) in tiles.iter().enumerate() {
            with_device(root_dev(k), || h.levels[0].kernels.ghost_fill(&h.levels[0].state));
        }
        for (k, h) in tiles.iter().enumerate() {
            let err = scan_c2p_errors(&h.levels[0].state);
            if err.is_err() {
                return Err(fail(
                    fabric,
                    WorkerError::Numerics(format!("c2p failed on initial conditions (tile {:?}): {err}", held.root_ids[k])),
                ));
            }
        }
        if cfg.restart {
            // the loaded fine level stands: its interior, its cut halos from the neighbors'
            // interiors and its coarse-fine ghosts come from the file; the physical ghosts fill
            for &k in &refined {
                with_device(fine_dev(k), || tiles[k].fill_physical_ghosts(1));
            }
        } else {
            for &k in &refined {
                with_device(fine_dev(k), || tiles[k].seed_level_from_filled_parent(1))
                    .map_err(|e| fail(fabric, WorkerError::Numerics(format!("fine seed: {e:?}"))))?;
            }
            for (k, (h, ghosts)) in tiles.iter().zip(root_ghosts).enumerate() {
                with_device(root_dev(k), || h.restore_seed_ghosts(0, ghosts));
            }
            // each nested level seeds from its parent once the parent's state (the primitives,
            // the cell field and the staggered faces) and its conserved halos crossed the parent's
            // cuts, as the root's cross before the fine level seeds: the prolongation of a
            // patch's faces beside a cut reads the parent's faces and cell field past it
            for level in 2..depth {
                // the parent's ghosts filled for the seed before its cut halos cross
                let parent_ghosts: Vec<(usize, SeedGhosts)> = tiles
                    .iter()
                    .enumerate()
                    .filter_map(|(k, h)| {
                        held.level_ids(level - 1)[k].map(|id| (k, with_device(held.level_devices(level - 1)[id.0 as usize], || h.fill_seed_ghosts(level - 1))))
                    })
                    .collect();
                let parent_state = level_fields!(level - 1, plan_fields);
                let parent = level_fields!(level - 1, plan_cons_fields);
                let state_plan = if level == 2 { &plans.fine } else { &plans.nested[level - 3].state };
                let parts = [
                    PlanPart { plan: state_plan, fields: &parent_state, devices: held.level_devices(level - 1), fault: None, dst_fields: None },
                    PlanPart { plan: &plans.nested[level - 2].seed, fields: &parent, devices: held.level_devices(level - 1), fault: None, dst_fields: None },
                ];
                exchange_point_plans(&parts, transport, fabric, ExchangePoint::Seed { level: level as u8 }, 0, 0, deadline, &mut timing.exchange)
                    .map_err(|e| fail(fabric, e.into()))?;
                for (k, h) in tiles.iter().enumerate() {
                    if let Some(id) = held.level_ids(level - 1)[k] {
                        with_device(held.level_devices(level - 1)[id.0 as usize], || h.fill_physical_ghosts(level - 1));
                    }
                }
                for (k, h) in tiles.iter().enumerate() {
                    if let Some(id) = held.level_ids(level)[k] {
                        with_device(held.level_devices(level)[id.0 as usize], || h.seed_level_from_filled_parent(level))
                            .map_err(|e| fail(fabric, WorkerError::Numerics(format!("level {level} seed: {e:?}"))))?;
                    }
                }
                for (k, ghosts) in parent_ghosts {
                    let id = held.level_ids(level - 1)[k].expect("a tile of the parent level");
                    with_device(held.level_devices(level - 1)[id.0 as usize], || tiles[k].restore_seed_ghosts(level - 1, ghosts));
                }
            }
        }
        // a declared stationary target, read off once every level is seeded or loaded
        if let Some(seed) = seed_target.as_mut() {
            probe_refined_target(tiles, held, plans, depth, transport, fabric, deadline, &mut timing, &mut **seed)?;
        }
        if !cfg.restart {
            // a fresh start lays its initial-state edits (a declared perturbation) over every
            // seeded level before the priming fills and certifies them
            for h in tiles.iter_mut() {
                on_seeded(h).map_err(|e| fail(fabric, WorkerError::Refused(e)))?;
                h.prime();
            }
        }
        // the primed root reaches the neighbors' halos: the priming restricted the seeded fine
        // level into each tile's covered cells and faces after the prime exchange, and the first
        // stage reads those cells across the cuts (an edge EMF differences the faces there). a
        // resumed run exchanges its loaded root here as well, so every worker walks one table
        {
            // the primed root across its cuts, and into every refined tile's level-1 prim stencil at
            // both time levels
            {
                let prims = root_fields!();
                let (_, now, stencil_old, stencil_new) = prim_stencil_fields!();
                let mut parts = vec![PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None, dst_fields: None }];
                if let Some((old, new)) = plans.prim_stencil.as_ref() {
                    parts.push(PlanPart { plan: old, fields: &now, devices: held.root_devices, fault: None, dst_fields: Some(&stencil_old) });
                    parts.push(PlanPart { plan: new, fields: &now, devices: held.root_devices, fault: None, dst_fields: Some(&stencil_new) });
                }
                exchange_point_plans(&parts, transport, fabric, ExchangePoint::Primed, 0, 0, deadline, &mut timing.exchange)
                    .map_err(|e| fail(fabric, e.into()))?;
                for (k, h) in tiles.iter().enumerate() {
                    with_device(root_dev(k), || h.levels[0].kernels.ghost_fill(&h.levels[0].state));
                }
            }
            // level by level, coarsest first, on every worker: a fresh gas level primes again from its
            // parent's crossed cut halos (and its gathered prim stencil), as the single hierarchy's
            // prime prolongs from its parent's interior, a resumed level keeping the coarse-fine ghosts
            // it loaded; then the level's cut halos cross, and the next level's prim stencils gather
            // from its primed cells
            for level in 1..depth {
                for (k, h) in tiles.iter_mut().enumerate() {
                    if h.levels.len() > level {
                        let gas = h.levels[level].state.fields.mhd.is_none();
                        with_device(root_dev(k), || h.reprime_level(level, gas && !cfg.restart));
                    }
                }
                let state = level_fields!(level, plan_fields);
                let state_plan = if level == 1 { &plans.fine } else { &plans.nested[level - 2].state };
                let devices = held.level_devices(level);
                let child = plans.nested.get(level - 1).and_then(|n| n.prim_stencil.as_ref());
                let [_, now, old_dst, new_dst] =
                    if child.is_some() { nested_prim_stencil_lists(tiles, held, level + 1) } else { std::array::from_fn(|_| Vec::new()) };
                let mut parts = vec![PlanPart { plan: state_plan, fields: &state, devices, fault: None, dst_fields: None }];
                if let Some((old, new)) = child {
                    parts.push(PlanPart { plan: old, fields: &now, devices, fault: None, dst_fields: Some(&old_dst) });
                    parts.push(PlanPart { plan: new, fields: &now, devices, fault: None, dst_fields: Some(&new_dst) });
                }
                exchange_point_plans(&parts, transport, fabric, ExchangePoint::PrimedLevel { level: level as u8 }, 0, 0, deadline, &mut timing.exchange)
                    .map_err(|e| fail(fabric, e.into()))?;
                for (k, h) in tiles.iter().enumerate() {
                    if h.levels.len() > level {
                        with_device(root_dev(k), || h.levels[level].kernels.ghost_fill(&h.levels[level].state));
                    }
                }
            }
        }
        if cfg.perturb_fine_face_after == Some(0) {
            perturb_fine_face(tiles);
        }
        // the prime audit certifies the initialized state of both levels: the root after its
        // exchange, the fine level seeded or loaded and primed. the initial checkpoint writes this
        // state
        audit_levels!(ExchangePoint::Audit(AuditAt::Prime), 0, 0);
    }

    // a discrete population: every level stage leaves its ancestry blend to the exchange below, and
    // the level atlas maps any owner to its tile on every worker
    let atlas = if tiles.iter().any(|h| h.levels[0].state.tracers.is_some() || h.levels[0].state.continuous_tracers.is_some()) {
        for h in tiles.iter_mut() {
            h.set_blend_across_tiles();
        }
        Some(LevelAtlas::agree(tiles, held, depth, fabric, deadline).map_err(|e| fail(fabric, e.into()))?)
    } else {
        None
    };
    // a continuous population: the levels' coefficients cross their cuts at their own points
    let continuous = !plans.coefficients.is_empty();
    // the discrete population's stage exchanges and spawns, on a run that carries one
    let discrete_atlas = atlas.as_ref().filter(|_| tiles.iter().any(|h| h.levels[0].state.tracers.is_some()));

    let guard_scope = symbi_sim::guard_ledger::open_scope();
    // every worker resumes the same retry guard: the streak a checkpoint recorded and the
    // configured thresholds, on every held tile's root
    {
        let roots: Vec<_> = tiles.iter().map(|h| &*h.levels[0].state).collect();
        symbi_sim::worker::agree_retry_guard(&roots, fabric, deadline).map_err(|e| fail(fabric, e))?;
    }
    let mut t = base.start_time;
    let mut iter: u64 = 0;
    // the primed state, before the first step: the callback sees zero accepted steps, where an
    // initial checkpoint holds the seeded fine level
    match on_step(0, t, &*tiles, fabric) {
        Ok(ControlFlow::Continue(())) => {}
        Ok(ControlFlow::Break(())) => return Ok(report),
        Err(e) => return Err(fail(fabric, e)),
    }
    // the cap bounds the root iteration itself, a resumed run's checkpointed iterations included
    while t < base.t_final && (base.max_steps == 0 || tiles[0].levels[0].state.iteration < base.max_steps) {
        // the global timestep: the minimum root cfl over the held hierarchies, then over the
        // workers, consumed as the coordinator's bits
        // every level's raw candidate is judged before the user clamp, which would replace a
        // NaN with the cap
        // every held tile's levels in root units, tile by tile, each candidate named by its tile
        // and level
        let mut origin: Vec<(usize, usize)> = Vec::new();
        let mut candidates: Vec<f64> = Vec::new();
        for (k, h) in tiles.iter().enumerate() {
            for (ll, candidate) in with_device(root_dev(k), || h.level_cfl_candidates()) {
                let poisoned = k == 0 && (base.injection.invalid_cfl_at == Some(iter) && ll == 0 || cfg.invalid_level_cfl_at == Some((iter, ll)));
                origin.push((k, ll));
                candidates.push(if poisoned { f64::NAN } else { candidate });
            }
        }
        let max_dt = tiles[0].levels[0].state.max_dt;
        let agreed = symbi_sim::worker::agree_worker_timestep(candidates, max_dt, base.t_final - t, fabric, deadline, &mut timing.collectives)
            .map_err(|e| fail(fabric, e))?;
        let mut dt = match agreed {
            symbi_sim::worker::AgreedStep::Step(agreed) => agreed.dt,
            symbi_sim::worker::AgreedStep::Halt(Some(bad)) => {
                let (k, ll) = origin[bad.index];
                report.halt = Some(format!(
                    "invalid CFL candidate {:e} on level {ll} of tile {:?} at iter {iter} (time {t:.4e})",
                    bad.value, held.root_ids[k]
                ));
                break;
            }
            symbi_sim::worker::AgreedStep::Halt(None) => {
                report.halt = Some(format!("an invalid CFL candidate on another worker at iter {iter} (time {t:.4e})"));
                break;
            }
        };
        if let Some((at, bits)) = base.injection.cap_dt_at {
            if at == iter {
                dt = dt.min(f64::from_bits(bits));
            }
        }
        // the root step from its candidate, the agreed timestep after its final-time clamp and
        // caps, through the shared step transaction; the attempt's mechanics answer it from this
        // scope, beside the plans and exchange points they drive
        let guard = tiles[0].levels[0].state.retry_guard;
        let mut attempt: u16 = 0;
        let mut sidecars: Vec<crate::refinement::hierarchy::HierarchyRetrySidecars<D>> = Vec::new();
        let mut restorable = false;
        let mut scoped = ScopedAttempt(|op: AttemptOp| -> Result<AttemptReply, WorkerError> {
            match op {
                AttemptOp::Rejectable => Ok(AttemptReply::Flag(
                    tiles.iter().any(Hierarchy::root_step_rejectable) || !base.injection.reject_at.is_empty() || plans.slip_shared.is_some(),
                )),
                AttemptOp::Capture { restorable: complete, .. } => {
                    // the root-attempt snapshot of every level and its sidecars
                    restorable = complete;
                    sidecars = tiles.iter().map(|h| h.snapshot_retry_sidecars()).collect();
                    for (k, h) in tiles.iter_mut().enumerate() {
                        if complete {
                            for level in &h.levels {
                                with_device(root_dev(k), || level.kernels.snapshot_retry(&level.state));
                            }
                        }
                        with_device(root_dev(k), || {
                            h.snapshot_attempt_entry();
                            h.snapshot_face_entry();
                        });
                    }
                    Ok(AttemptReply::Done)
                }
                AttemptOp::Run { dt } => {
                    // a slip run opens its palindrome on the finest level before the root step,
                    // whose drains it then withholds
                    let slipping = plans.slip_shared.is_some();
                    let half_dt = 0.5 * dt;
                    if slipping {
                        slip_drain!(half_dt, 0u8, iter, attempt);
                        if !slip_mid!(half_dt, 0u8, iter, attempt) {
                            report.rejections += 1;
                            return Ok(AttemptReply::Flag(false));
                        }
                        slip_sync!(0u8, iter, attempt);
                        for h in tiles.iter_mut() {
                            h.set_drain_suppressed(true);
                        }
                    }
                    for (k, h) in tiles.iter_mut().enumerate() {
                        with_device(root_dev(k), || h.level_step_begin(0, dt));
                    }
                    let mut rejected = false;
                    // the root stages
                    for ii in 0..nstages {
                        let injected = base
                            .injection
                            .trouble_at
                            .filter(|&(at_step, at_stage, _)| attempt == 0 && (at_step, at_stage) == (iter, ii))
                            .map(|(_, _, cell)| std::array::from_fn(|ax| cell[ax]));
                        let all: Vec<usize> = (0..n).collect();
                        let mut level_stage = LaunchedLevelStage {
                            tiles: &mut *tiles,
                            mine: &all,
                            ids: held.root_ids,
                            devices: held.root_devices,
                            plan: &plans.root,
                            level: 0,
                            ii,
                            dt,
                            alpha: 0.0,
                            point: ExchangePoint::Troubled(ii as u8),
                            step: iter,
                            attempt,
                            transport,
                            fabric: &mut *fabric,
                            deadline,
                            timing: &mut timing,
                            injected,
                            forced: attempt == 0 && base.injection.reject_at.contains(&(iter, ii)),
                            receipt_starts: vec![0; n],
                        };
                        let agreed = symbi_sim::collective_stage::collective_stage(&mut level_stage)?;
                        report.troubled_exchanges += u64::from(agreed.exchanged);
                        report.neighbor_corrections += agreed.neighbor_corrections;
                        if agreed.retry {
                            rejected = true;
                            break;
                        }
                        if let Some(atlas) = discrete_atlas {
                            exchange_level_tracers(tiles, held, atlas, 0, ii, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
                        }
                        if ii + 1 == nstages && plans.stencil.is_some() {
                            // the last root stage point also gathers every refined tile's parent face
                            // stencil at both time levels, the faces final for the fine subcycle
                            let (old, new) = plans.stencil.as_ref().unwrap();
                            let prims = root_fields!();
                            let entry = entry_fields!();
                            let faces = face_fields!();
                            let (stencil_old, stencil_new) = stencil_fields!();
                            let parts = [
                                PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None, dst_fields: None },
                                PlanPart { plan: old, fields: &entry, devices: held.root_devices, fault: None, dst_fields: Some(&stencil_old) },
                                PlanPart { plan: new, fields: &faces, devices: held.root_devices, fault: None, dst_fields: Some(&stencil_new) },
                            ];
                            exchange_point_plans(&parts, transport, fabric, ExchangePoint::Stage(ii as u8), iter, attempt, deadline, &mut timing.exchange)
                                .map_err(|e| fail(fabric, e.into()))?;
                            for (k, h) in tiles.iter().enumerate() {
                                with_device(root_dev(k), || {
                                    h.levels[0].kernels.ghost_fill(&h.levels[0].state);
                                    h.mark_face_stencil_filled();
                                });
                            }
                        } else if ii + 1 == nstages && plans.prim_stencil.is_some() {
                            // the last root stage point also gathers every refined tile's parent prim
                            // stencil at both time levels: the donated cells at the attempt's entry and
                            // the owners' primitives after the stages
                            let (old, new) = plans.prim_stencil.as_ref().unwrap();
                            let prims = root_fields!();
                            let (entry, now, stencil_old, stencil_new) = prim_stencil_fields!();
                            let parts = [
                                PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None, dst_fields: None },
                                PlanPart { plan: old, fields: &entry, devices: held.root_devices, fault: None, dst_fields: Some(&stencil_old) },
                                PlanPart { plan: new, fields: &now, devices: held.root_devices, fault: None, dst_fields: Some(&stencil_new) },
                            ];
                            exchange_point_plans(&parts, transport, fabric, ExchangePoint::Stage(ii as u8), iter, attempt, deadline, &mut timing.exchange)
                                .map_err(|e| fail(fabric, e.into()))?;
                            for (k, h) in tiles.iter().enumerate() {
                                with_device(root_dev(k), || {
                                    h.levels[0].kernels.ghost_fill(&h.levels[0].state);
                                    h.mark_prim_stencil_filled();
                                });
                            }
                        } else {
                            root_point!(ExchangePoint::Stage(ii as u8), iter, attempt);
                        }
                        if base.audit_every_point {
                            audit!(plans.root, root_fields!(), ExchangePoint::Audit(AuditAt::Stage(ii as u8)), iter, attempt);
                        }
                    }
                    if !rejected {
                        if let Some(atlas) = discrete_atlas {
                            spawn_level_injection(tiles, held, atlas, 0, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
                        }
                        for (k, h) in tiles.iter().enumerate() {
                            with_device(root_dev(k), || {
                                h.level_tail_emf(0, dt);
                                h.level_tail_excise(0);
                            });
                        }
                        // the root's horizon ledger: every held tile's exact shell partial, merged
                        // across the workers into the whole root's and booked on every level of every
                        // held tile, once per root step inside the attempt, whose restore unbooks it
                        if let Some((_, radius)) = symbi_sim::driver::horizon_request(&tiles[0].levels[0].state) {
                            let mut local = symbi_sim::driver::HorizonPartial::default();
                            for (k, h) in tiles.iter().enumerate() {
                                let root = &h.levels[0];
                                let partial = with_device(root_dev(k), || {
                                    symbi_sim::driver::horizon_partial_of(&root.kernels, &root.state, radius)
                                });
                                local.merge(&partial);
                            }
                            let workers = plans.root.placement().workers as usize;
                            let total = symbi_sim::worker::combine_horizon_partials(fabric, workers, &local, deadline)?;
                            for h in tiles.iter_mut() {
                                h.book_horizon_on_every_level(&total, dt);
                            }
                        }
                        for (k, h) in tiles.iter().enumerate() {
                            with_device(root_dev(k), || {
                                // rigid walls also act on the uncovered root cells; the restriction
                                // replaces the covered ones after the fine subcycle, and the sinks stay
                                // off the root's gravity-only proxies
                                let level = &h.levels[0];
                                if level.state.immersed.as_ref().is_some_and(|im| im.bodies.rigid_count() > 0) {
                                    level.kernels.penalize(&level.state, dt);
                                }
                            });
                        }
                        // the root's continuous tracers: the injection held until they advance, the root's
                        // accepted ito coefficients across the cuts before the fine level prolongs them
                        if continuous {
                            for (k, h) in tiles.iter_mut().enumerate() {
                                h.hold_continuous_injection(0);
                                with_device(root_dev(k), || h.prepare_level_ito_coefficients(0));
                            }
                            exchange_level_coefficients(
                                tiles, held, &plans.coefficients[0], 0, ExchangePoint::RootCoefficients, iter, attempt, transport, fabric, deadline,
                                &mut timing.exchange,
                            )
                            .map_err(|e| fail(fabric, e.into()))?;
                        }
                        // the fine subcycle: every worker walks its phases, a worker without fine
                        // tiles with no local work
                        let fine_dt = dt / RATIO as f64;
                        'subcycle: for sub in 0..RATIO {
                            let alpha = sub as f64 / RATIO as f64;
                            for &k in &refined {
                                with_device(fine_dev(k), || {
                                    tiles[k].prolong_cf(1, alpha);
                                    tiles[k].fill_physical_ghosts(1);
                                });
                            }
                            fine_point!(ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Prolong }, iter, attempt);
                            for &k in &refined {
                                with_device(fine_dev(k), || {
                                    tiles[k].level_step_begin(1, fine_dt);
                                    // the fine faces at the substep's start, the old time level of the
                                    // nested level's stencil
                                    tiles[k].snapshot_level_faces(1);
                                });
                            }
                            for jj in 0..nstages {
                                let fine_ids: Vec<TileId> = refined.iter().map(|&k| held.fine_ids[k].unwrap()).collect();
                                let injected = cfg
                                    .fine_trouble_at
                                    .filter(|&(at_step, at_sub, at_stage, _)| attempt == 0 && (at_step, at_sub, at_stage) == (iter, sub as u8, jj))
                                    .map(|(_, _, _, cell)| std::array::from_fn(|ax| cell[ax]));
                                let mut level_stage = LaunchedLevelStage {
                                    tiles: &mut *tiles,
                                    mine: &refined,
                                    ids: &fine_ids,
                                    devices: held.fine_devices,
                                    plan: &plans.fine,
                                    level: 1,
                                    ii: jj,
                                    dt: fine_dt,
                                    alpha,
                                    point: ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Troubled(jj as u8) },
                                    step: iter,
                                    attempt,
                                    transport,
                                    fabric: &mut *fabric,
                                    deadline,
                                    timing: &mut timing,
                                    injected,
                                    forced: false,
                                    receipt_starts: vec![0; refined.len()],
                                };
                                let agreed = symbi_sim::collective_stage::collective_stage(&mut level_stage)?;
                                report.fine_troubled_exchanges += u64::from(agreed.exchanged);
                                report.fine_neighbor_corrections += agreed.neighbor_corrections;
                                if agreed.retry {
                                    rejected = true;
                                    break 'subcycle;
                                }
                                if let Some(atlas) = discrete_atlas {
                                    exchange_level_tracers(tiles, held, atlas, 1, jj, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
                                }
                                fine_stage_point!(sub, jj, iter, attempt);
                                if base.audit_every_point {
                                    audit!(plans.fine, fine_fields!(), ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Audit(jj as u8) }, iter, attempt);
                                }
                            }
                            if let Some(atlas) = discrete_atlas {
                                spawn_level_injection(tiles, held, atlas, 1, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
                            }
                            // the fine tail on the globally finest level, then its recovery and halo
                            // refresh at the post-tail point before any later reader
                            let mut retry = false;
                            if depth > 2 {
                                // the fine tail around the nested subcycle: the nested levels advance across
                                // their own tiles, restrict into the fine level, and its halos carry the
                                // restriction before the tail completes
                                for &k in &refined {
                                    with_device(fine_dev(k), || tiles[k].level_tail_pre(1, fine_dt));
                                }
                                if continuous {
                                    exchange_level_coefficients(
                                        tiles, held, &plans.coefficients[1], 1, ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Coefficients },
                                        iter, attempt, transport, fabric, deadline, &mut timing.exchange,
                                    )
                                    .map_err(|e| fail(fabric, e.into()))?;
                                }
                                let nested = NestedStep { held, plans, transport, deadline, nstages, depth, audit_every: base.audit_every_point, step: iter, attempt, atlas: atlas.as_ref(), discrete: discrete_atlas.is_some(), nested_contribution_fault: cfg.nested_contribution_fault };
                                if nested_subcycle(tiles, &nested, fabric, &mut timing, &mut report, 2, sub as u32, fine_dt)? {
                                    rejected = true;
                                    break 'subcycle;
                                }
                                reflux_received_nested(tiles, held, 1);
                                restrict_with_bands(tiles, held, plans, 1, sub as u32, alpha, iter, attempt, transport, fabric, deadline, &mut timing.exchange)
                                    .map_err(|e| fail(fabric, e.into()))?;
                                {
                                    let fine = fine_fields!();
                                    let part = PlanPart { plan: &plans.fine, fields: &fine, devices: held.fine_devices, fault: None, dst_fields: None };
                                    exchange_point_plans(&[part], transport, fabric, ExchangePoint::NestedRestrict { level: 1, path: sub as u32 }, iter, attempt, deadline, &mut timing.exchange)
                                        .map_err(|e| fail(fabric, e.into()))?;
                                }
                                for &k in &refined {
                                    with_device(fine_dev(k), || tiles[k].fill_physical_ghosts(1));
                                }
                                if continuous {
                                    continuous_tail(tiles, held, atlas.as_ref().unwrap(), 1, fine_dt, fabric, deadline)?;
                                } else {
                                    for &k in &refined {
                                        with_device(fine_dev(k), || tiles[k].level_tail_post(1, fine_dt));
                                    }
                                }
                            } else if continuous {
                                // the finest fine level's tail split around its coefficients and its tracers'
                                // crossing, as the single hierarchy runs it
                                for &k in &refined {
                                    with_device(fine_dev(k), || tiles[k].level_tail_pre(1, fine_dt));
                                }
                                exchange_level_coefficients(
                                    tiles, held, &plans.coefficients[1], 1, ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Coefficients },
                                    iter, attempt, transport, fabric, deadline, &mut timing.exchange,
                                )
                                .map_err(|e| fail(fabric, e.into()))?;
                                continuous_tail(tiles, held, atlas.as_ref().unwrap(), 1, fine_dt, fabric, deadline)?;
                            } else {
                                for &k in &refined {
                                    retry |= with_device(fine_dev(k), || tiles[k].level_step_tail(1, fine_dt, alpha));
                                }
                            }
                            if vote_retry(fabric, retry, deadline, &mut timing)? {
                                rejected = true;
                                break 'subcycle;
                            }
                            let workers = plans.root.placement().workers as usize;
                            launched_level_census(tiles, 1, fine_dev, workers, fabric, deadline)?;
                            // the last substep: the refined tiles restrict and reflux before the
                            // post-tail exchange, so the staggered faces it carries to the neighbors
                            // across the cuts are the corrected ones
                            if sub + 1 == RATIO {
                                // a gas run: the tiles beside a neighbor's patch apply their received
                                // units and recover their cells beside the patch now, as the single
                                // hierarchy refluxes every uncovered cell before its bands, and a band
                                // whose patch face lies on a cut takes its start cell from the root
                                // halos at the band start point
                                if gas_reflux_first {
                                    for k in (0..n).filter(|&k| tiles[k].levels.len() == 1 && tiles[k].has_flux_register(0)) {
                                        with_device(root_dev(k), || {
                                            tiles[k].reflux_received();
                                            tiles[k].recover_beside_patch();
                                        });
                                    }
                                    if band_start {
                                        root_point!(ExchangePoint::BandStart { level: 0, path: 0 }, iter, attempt);
                                    }
                                }
                                restrict_with_bands(tiles, held, plans, 0, 0, 0.0, iter, attempt, transport, fabric, deadline, &mut timing.exchange)
                                    .map_err(|e| fail(fabric, e.into()))?;
                            }
                            fine_tail_point!(sub, iter, attempt);
                            if base.audit_every_point {
                                audit!(plans.fine, fine_fields!(), ExchangePoint::Fine { substep: sub as u8, point: FinePoint::AuditTail }, iter, attempt);
                            }
                        }
                    }
                    if !rejected {
                        // the tiles beside a neighbor's patch apply their received units, the faces
                        // that application wrote on their own cuts reach the tiles across them, and
                        // every such tile then recovers its cells beside the patch
                        let beside: Vec<usize> = if gas_reflux_first {
                            Vec::new()
                        } else {
                            (0..n).filter(|&k| tiles[k].levels.len() == 1 && tiles[k].has_flux_register(0)).collect()
                        };
                        for &k in &beside {
                            with_device(root_dev(k), || tiles[k].reflux_received());
                        }
                        if let Some(corrected) = plans.corrected.as_ref() {
                            let fields = face_fields!();
                            let part = PlanPart { plan: corrected, fields: &fields, devices: held.root_devices, fault: None, dst_fields: None };
                            exchange_point_plans(&[part], transport, fabric, ExchangePoint::CorrectedFaces, iter, attempt, deadline, &mut timing.exchange)
                                .map_err(|e| fail(fabric, e.into()))?;
                        }
                        for &k in &beside {
                            with_device(root_dev(k), || tiles[k].recover_beside_patch());
                        }
                        // every other tile holding no fine level recovers its root as the
                        // restriction recovers the single hierarchy's
                        for k in (0..n).filter(|&k| tiles[k].levels.len() == 1 && !tiles[k].has_flux_register(0)) {
                            with_device(root_dev(k), || tiles[k].recover_level(0));
                        }
                        root_point!(ExchangePoint::Restrict, iter, attempt);
                        if base.audit_every_point {
                            audit!(plans.root, root_fields!(), ExchangePoint::Audit(AuditAt::Restrict), iter, attempt);
                        }
                        // the root's continuous tracers advance over the root step after the fine subcycle
                        // and its restriction, cross to the tiles holding their cells, and the held
                        // injection spawns once
                        if continuous {
                            let atlas = atlas.as_ref().unwrap();
                            for (k, h) in tiles.iter_mut().enumerate() {
                                with_device(root_dev(k), || h.advance_level_continuous_tracers(0, dt));
                            }
                            migrate_level_continuous(tiles, held, atlas, 0, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
                            spawn_level_continuous(tiles, held, atlas, 0, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
                        }
                        for (k, h) in tiles.iter_mut().enumerate() {
                            with_device(root_dev(k), || {
                                let s = &mut h.levels[0].state;
                                (s.time, s.iteration) = advance_clock(s.time, s.iteration, dt);
                            });
                        }
                    }
                    // the closing half: the slip midpoint and the drain on the finest level, then the
                    // coarser levels take the finest state
                    if slipping {
                        for h in tiles.iter_mut() {
                            h.set_drain_suppressed(false);
                        }
                        if !rejected {
                            if slip_mid!(half_dt, 1u8, iter, attempt) {
                                slip_drain!(half_dt, 1u8, iter, attempt);
                                slip_sync!(1u8, iter, attempt);
                            } else {
                                rejected = true;
                            }
                        }
                    }
                    if rejected {
                        report.rejections += 1;
                    }
                    Ok(AttemptReply::Flag(!rejected))
                }
                AttemptOp::Evidence { rejections } => {
                    // every worker books the agreed rejection with the evidence its own tiles
                    // hold, a fine substep's included, so every worker reaches the guard's
                    // decision at the same attempt
                    let mut evidence = Vec::new();
                    for (k, h) in tiles.iter().enumerate() {
                        for (ll, level) in h.levels.iter().enumerate() {
                            let tile = if ll == 0 { held.root_ids[k].0 } else { held.fine_ids[k].map_or(u32::MAX, |t| t.0) };
                            let site = format!("worker {}, tile {tile}, level {ll}", fabric.me().0);
                            evidence.extend(symbi_sim::retry_guard::take_evidence(&level.state, &site, rejections));
                        }
                    }
                    Ok(AttemptReply::Evidence(evidence))
                }
                AttemptOp::Restore => {
                    assert!(restorable, "a root step restored from an incomplete capture");
                    // the collective rollback: every level of every tile returns to the
                    // root-attempt snapshot, and every level's halos are refreshed at the
                    // rollback point
                    for (k, h) in tiles.iter_mut().enumerate() {
                        for level in &h.levels {
                            if !level.kernels.field_rollback_is_complete(&level.state) {
                                return Err(fail(
                                    fabric,
                                    WorkerError::Numerics(
                                        "a step was rejected on a kernel set without a complete rollback snapshot; the \
                                         replay cannot start from the step entry"
                                            .into(),
                                    ),
                                ));
                            }
                            with_device(root_dev(k), || level.kernels.restore_step(&level.state));
                        }
                        with_device(root_dev(k), || {
                            h.restore_attempt_entry();
                            h.clear_pending_contributions();
                        });
                        h.restore_retry_sidecars(&sidecars[k]);
                    }
                    {
                        let prims = root_fields!();
                        let fine = fine_fields!();
                        let nested: Vec<_> = (2..depth).map(|level| level_fields!(level, plan_fields)).collect();
                        let mut parts = vec![
                            PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None, dst_fields: None },
                            PlanPart { plan: &plans.fine, fields: &fine, devices: held.fine_devices, fault: None, dst_fields: None },
                        ];
                        for (i, level) in plans.nested.iter().enumerate() {
                            parts.push(PlanPart { plan: &level.state, fields: &nested[i], devices: held.level_devices(i + 2), fault: None, dst_fields: None });
                        }
                        exchange_point_plans(&parts, transport, fabric, ExchangePoint::Rollback, iter, attempt, deadline, &mut timing.exchange)
                            .map_err(|e| fail(fabric, e.into()))?;
                    }
                    for (k, h) in tiles.iter().enumerate() {
                        for level in &h.levels {
                            with_device(root_dev(k), || level.kernels.ghost_fill(&level.state));
                        }
                    }
                    if base.audit_every_point {
                        audit!(plans.root, root_fields!(), ExchangePoint::Audit(AuditAt::Rollback), iter, attempt);
                    }
                    attempt += 1;
                    Ok(AttemptReply::Done)
                }
                AttemptOp::SevereStreak => Ok(AttemptReply::Streak(tiles[0].levels[0].state.severe_streak)),
                // every tile's root carries the streak its checkpoint records
                AttemptOp::SetSevereStreak(streak) => {
                    for h in tiles.iter_mut() {
                        h.levels[0].state.severe_streak = streak;
                    }
                    Ok(AttemptReply::Done)
                }
            }
        });
        let step = take_root_step(&mut scoped, &guard, dt, t)?;
        drop(scoped);
        let dt = match step {
            RootStep::Accepted { dt } => dt,
            // the halted step left every level of every tile at its entry, on every worker alike
            RootStep::Halted(StepHalt::Guard(failure)) => {
                report.halt = Some(symbi_sim::worker::guard_failure(fabric, failure, &base.injection, deadline)?);
                break;
            }
            RootStep::Halted(StepHalt::Timestep { detail, .. }) => {
                report.halt = Some(detail);
                break;
            }
        };
        // the continuous tracers onto the finest level holding them, once per accepted root step
        if continuous {
            relocate_continuous_to_finest(tiles, held, atlas.as_ref().unwrap(), depth, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
        }
        // the root's own per-level census row at its new clock, ahead of the body step as the
        // single hierarchy records it in the root's tail
        launched_level_census(tiles, 0, root_dev, plans.root.placement().workers as usize, fabric, deadline)?;
        if tiles.iter().any(|h| h.levels[0].state.has_bodies()) {
            refined_body_step(tiles, held, plans, fabric, dt, deadline).map_err(|e| fail(fabric, e))?;
        }
        // the root-step censuses at the root's new clock, after the body step as on the single
        // hierarchy, so a moving body masks the cells it covers at that clock: every worker merges
        // its tiles' exact partials of every level, the workers' partials merge into the whole
        // hierarchy's sample, and every tile's root records it
        if !tiles[0].levels[0].state.censuses.is_empty() {
            let local = crate::refinement::hierarchy::tiles_root_step_census(tiles, root_dev);
            let workers = plans.root.placement().workers as usize;
            let total = symbi_sim::worker::combine_census_partials(fabric, workers, &tiles[0].levels[0].state.censuses, &local, deadline)?;
            let now = tiles[0].levels[0].state.time;
            for h in tiles.iter_mut() {
                h.levels[0].state.store.record_census(now, 0, &total);
            }
        }
        (t, iter) = advance_clock(t, iter, dt);
        report.steps = iter;
        report.time = t;
        report.dt_sequence.push(dt);
        if cfg.perturb_fine_face_after == Some(iter) {
            perturb_fine_face(tiles);
        }
        if let Some((at, which, ordinal)) = cfg.clock_fault {
            if at == iter {
                if let Some(h) = tiles.iter_mut().filter(|h| h.levels.len() > 1).nth(ordinal) {
                    let st = &mut h.levels[1].state;
                    match which {
                        ClockField::Time => st.time = f64::from_bits(st.time.to_bits() + 1),
                        ClockField::Dt => st.dt = f64::from_bits(st.dt.to_bits() + 1),
                        ClockField::Iteration => st.iteration += 1,
                    }
                }
            }
        }
        let clock = Instant::now();
        let flow = on_step(iter, t, &*tiles, fabric);
        in_callback += clock.elapsed();
        match flow {
            Ok(ControlFlow::Continue(())) => {}
            Ok(ControlFlow::Break(())) => break,
            Err(e) => return Err(fail(fabric, e)),
        }
    }
    timing.compute = march_start
        .elapsed()
        .saturating_sub(timing.collectives + timing.exchange + in_callback);
    report.timing = timing;
    report.grant_high_water = (0..plans.root.placement().workers)
        .map(|p| fabric.grants().high_water(WorkerId(p)))
        .collect();
    let accepted = guard_scope.accepted();
    report.troubled_cells = accepted.troubled_cells.total;
    report.frozen_cells = accepted.frozen_cells.total;
    Ok(report)
}

/// what a nested subcycle reads of its session: the held tiles, the plans, the transport and
/// the scheme, for one attempt of one root step
struct NestedStep<'a, 'p, const D: usize, T> {
    held: &'a Held<'a>,
    plans: &'a RefinedPlans<'p, D>,
    transport: &'a T,
    deadline: Duration,
    nstages: usize,
    depth: usize,
    audit_every: bool,
    step: u64,
    attempt: u16,
    /// the tracer population's level atlas, when the run carries one
    atlas: Option<&'a LevelAtlas<D>>,
    /// the run carries a discrete population, whose stages exchange and spawn here
    discrete: bool,
    /// a fault in this worker's nested flux contributions, for the exactly-once gates
    nested_contribution_fault: Option<(u64, u8, u32, usize, StagingFault)>,
}

/// the subcycle of nested level `level >= 2` across its tiles within one step `dt` of its parent's
/// substep `parent_path`: each of the ratio substeps prolongs the coarse-fine ghosts at the
/// substep's time fraction, runs the stages with the troubled marks crossing the cuts, the tail
/// ahead of the level below, that level's subcycle with its restriction into this one and this
/// level's exchange after it, and the tail after it, then the post-tail exchange. the units of the
/// level's patch faces on its parent's cuts cross at each stage point to the parent tiles holding
/// the cells they correct. returns whether some worker rejected the step
#[allow(clippy::too_many_arguments)]
fn nested_subcycle<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    cx: &NestedStep<'_, '_, D, T>,
    fabric: &mut Fabric<L>,
    timing: &mut WorkerTiming,
    report: &mut WorkerReport,
    level: usize,
    parent_path: u32,
    dt: f64,
) -> Result<bool, WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
{
    let (step, attempt, deadline) = (cx.step, cx.attempt, cx.deadline);
    let plan = &cx.plans.nested[level - 2].state;
    let devices = cx.held.level_devices(level);
    let ids_by_root = cx.held.level_ids(level);
    let mine: Vec<usize> = (0..tiles.len()).filter(|&k| ids_by_root[k].is_some()).collect();
    let ids: Vec<TileId> = mine.iter().map(|&k| ids_by_root[k].expect("a tile of the level")).collect();
    let dev = |r: usize| devices[ids[r].0 as usize];
    let level_dt = dt / RATIO as f64;
    // the level's state at one of its points, then its physical ghosts
    let exchange = |tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], fabric: &mut Fabric<L>, timing: &mut WorkerTiming, point: ExchangePoint| {
        let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..plan.plan().n_tiles()).map(|_| Vec::new()).collect();
        for (r, &k) in mine.iter().enumerate() {
            lists[ids[r].0 as usize] = plan_fields(&tiles[k].levels[level].state);
        }
        let part = PlanPart { plan, fields: &lists, devices, fault: None, dst_fields: None };
        exchange_point_plans(&[part], cx.transport, fabric, point, step, attempt, deadline, &mut timing.exchange)
            .map_err(|e| fail(fabric, e.into()))?;
        for (r, &k) in mine.iter().enumerate() {
            with_device(dev(r), || tiles[k].fill_physical_ghosts(level));
        }
        Ok::<(), WorkerError>(())
    };
    let audit = |tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], fabric: &mut Fabric<L>, report: &mut WorkerReport, point: ExchangePoint| {
        if !(cx.audit_every && plan.has_audits()) {
            return Ok::<(), WorkerError>(());
        }
        let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..plan.plan().n_tiles()).map(|_| Vec::new()).collect();
        for (r, &k) in mine.iter().enumerate() {
            lists[ids[r].0 as usize] = plan_fields(&tiles[k].levels[level].state);
        }
        let cost = plan.audit_at(&lists, fabric, point, step, attempt, deadline).map_err(|e| match e {
            AuditError::Fabric(e) => fail(fabric, WorkerError::Fabric(e)),
            AuditError::Mismatch(m) => fail(fabric, WorkerError::Interface(m)),
        })?;
        report.audits += 1;
        report.audit_cost.faces += cost.faces;
        report.audit_cost.wire_bytes += cost.wire_bytes;
        report.audit_cost.elapsed += cost.elapsed;
        Ok(())
    };
    for sub in 0..RATIO {
        let path = parent_path * RATIO as u32 + sub as u32;
        let alpha = sub as f64 / RATIO as f64;
        let at = |point| ExchangePoint::Nested { level: level as u8, path, point };
        for (r, &k) in mine.iter().enumerate() {
            with_device(dev(r), || {
                tiles[k].prolong_cf(level, alpha);
                tiles[k].fill_physical_ghosts(level);
            });
        }
        exchange(tiles, fabric, timing, at(FinePoint::Prolong))?;
        for (r, &k) in mine.iter().enumerate() {
            with_device(dev(r), || {
                tiles[k].level_step_begin(level, level_dt);
                tiles[k].snapshot_level_faces(level);
            });
        }
        for jj in 0..cx.nstages {
            let mut level_stage = LaunchedLevelStage {
                tiles: &mut *tiles,
                mine: &mine,
                ids: &ids,
                devices,
                plan,
                level,
                ii: jj,
                dt: level_dt,
                alpha,
                point: at(FinePoint::Troubled(jj as u8)),
                step,
                attempt,
                transport: cx.transport,
                fabric: &mut *fabric,
                deadline,
                timing: &mut *timing,
                injected: None,
                forced: false,
                receipt_starts: vec![0; mine.len()],
            };
            let agreed = symbi_sim::collective_stage::collective_stage(&mut level_stage)?;
            report.fine_troubled_exchanges += u64::from(agreed.exchanged);
            report.fine_neighbor_corrections += agreed.neighbor_corrections;
            if agreed.retry {
                return Ok(true);
            }
            if let Some(atlas) = cx.atlas.filter(|_| cx.discrete) {
                exchange_level_tracers(tiles, cx.held, atlas, level, jj, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
            }
            let last = jj + 1 == cx.nstages;
            let child_stencil = cx.plans.nested.get(level - 1).and_then(|n| n.stencil.as_ref()).filter(|_| last);
            let child_prim = cx.plans.nested.get(level - 1).and_then(|n| n.prim_stencil.as_ref()).filter(|_| last);
            let own_flux = cx.plans.nested[level - 2].flux.as_ref();
            if child_stencil.is_none() && child_prim.is_none() && own_flux.is_none() {
                exchange(tiles, fabric, timing, at(FinePoint::Stage(jj as u8)))?;
            } else {
                // the stage point also carries, on the last stage, the next level's parent face and
                // prim stencils from this level's state at the substep's start and after its stages,
                // and at every stage the units of this level's patch faces on its parent's cuts
                let n = plan.plan().n_tiles();
                let mut state: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
                let mut old_src: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
                let mut new_src: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
                let mut old_dst: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
                let mut new_dst: Vec<Vec<&Field<f64, D, Mem>>> = (0..n).map(|_| Vec::new()).collect();
                for (r, &k) in mine.iter().enumerate() {
                    let t = ids[r].0 as usize;
                    let h = &tiles[k];
                    state[t] = plan_fields(&h.levels[level].state);
                    if child_stencil.is_some() {
                        if let Some(entry) = h.level_face_entry(level) {
                            old_src[t] = entry.iter().collect();
                        }
                        if let Some(mhd) = h.levels[level].state.fields.mhd.as_ref() {
                            new_src[t] = mhd.bface.b.iter().collect();
                        }
                        if let Some(stencil) = h.face_stencil_of(level + 1) {
                            old_dst[t] = stencil.old.iter().collect();
                            new_dst[t] = stencil.new.iter().collect();
                        }
                    }
                }
                let [prim_entry, prim_now, prim_old, prim_new] = if child_prim.is_some() {
                    nested_prim_stencil_lists(tiles, cx.held, level + 1)
                } else {
                    std::array::from_fn(|_| Vec::new())
                };
                let flux_fields = if own_flux.is_some() { level_flux_lists(tiles, cx.held, level - 1) } else { Vec::new() };
                let mut parts = vec![PlanPart { plan, fields: &state, devices, fault: None, dst_fields: None }];
                if let Some((old, new)) = child_stencil {
                    parts.push(PlanPart { plan: old, fields: &old_src, devices, fault: None, dst_fields: Some(&old_dst) });
                    parts.push(PlanPart { plan: new, fields: &new_src, devices, fault: None, dst_fields: Some(&new_dst) });
                }
                if let Some((old, new)) = child_prim {
                    parts.push(PlanPart { plan: old, fields: &prim_entry, devices, fault: None, dst_fields: Some(&prim_old) });
                    parts.push(PlanPart { plan: new, fields: &prim_now, devices, fault: None, dst_fields: Some(&prim_new) });
                }
                if let Some(flux) = own_flux {
                    let fault = cx.nested_contribution_fault.and_then(|(at_step, at_level, at_path, at_stage, fault)| {
                        (attempt == 0 && (at_step, at_level as usize, at_path, at_stage) == (step, level, path, jj)).then_some(fault)
                    });
                    parts.push(PlanPart { plan: flux, fields: &flux_fields, devices: cx.held.level_devices(level - 1), fault, dst_fields: None });
                }
                exchange_point_plans(&parts, cx.transport, fabric, at(FinePoint::Stage(jj as u8)), step, attempt, deadline, &mut timing.exchange)
                    .map_err(|e| fail(fabric, e.into()))?;
                for (r, &k) in mine.iter().enumerate() {
                    let h = &tiles[k];
                    with_device(dev(r), || {
                        h.fill_physical_ghosts(level);
                        if child_stencil.is_some() {
                            h.mark_face_stencil_filled_of(level + 1);
                        }
                        if child_prim.is_some() {
                            h.mark_prim_stencil_filled_of(level + 1);
                        }
                    });
                }
                if own_flux.is_some() {
                    fold_level_contributions(tiles, cx.held, level - 1);
                }
            }
            audit(tiles, fabric, report, at(FinePoint::Audit(jj as u8)))?;
        }
        if let Some(atlas) = cx.atlas.filter(|_| cx.discrete) {
            spawn_level_injection(tiles, cx.held, atlas, level, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
        }
        for (r, &k) in mine.iter().enumerate() {
            with_device(dev(r), || tiles[k].level_tail_pre(level, level_dt));
        }
        if let Some(plan) = cx.plans.coefficients.get(level) {
            exchange_level_coefficients(
                tiles, cx.held, plan, level, at(FinePoint::Coefficients), step, attempt, cx.transport, fabric, deadline, &mut timing.exchange,
            )
            .map_err(|e| fail(fabric, e.into()))?;
        }
        if level + 1 < cx.depth {
            if nested_subcycle(tiles, cx, fabric, timing, report, level + 1, path, level_dt)? {
                return Ok(true);
            }
            reflux_received_nested(tiles, cx.held, level);
            restrict_with_bands(tiles, cx.held, cx.plans, level, path, alpha, step, attempt, cx.transport, fabric, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
            // every other tile holding this level and none finer recovers it as the restriction
            // recovers the single hierarchy's
            for (r, &k) in mine.iter().enumerate() {
                if tiles[k].levels.len() <= level + 1 && !tiles[k].has_flux_register(level) {
                    with_device(dev(r), || tiles[k].recover_level(level));
                }
            }
            exchange(tiles, fabric, timing, ExchangePoint::NestedRestrict { level: level as u8, path })?;
        }
        if !cx.plans.coefficients.is_empty() {
            continuous_tail(tiles, cx.held, cx.atlas.expect("a continuous population carries its level atlas"), level, level_dt, fabric, deadline)?;
        } else {
            for (r, &k) in mine.iter().enumerate() {
                with_device(dev(r), || tiles[k].level_tail_post(level, level_dt));
            }
        }
        let held = cx.held;
        let level_device = |k: usize| held.level_devices(level)[held.level_ids(level)[k].expect("a holder of the level").0 as usize];
        launched_level_census(tiles, level, level_device, cx.plans.root.placement().workers as usize, fabric, deadline)?;
        exchange(tiles, fabric, timing, at(FinePoint::PostTail))?;
        audit(tiles, fabric, report, at(FinePoint::AuditTail))?;
    }
    Ok(false)
}

/// the collective retry vote of one stage: true when any worker rejects
/// a declared stationary target on the refined workers, as [`Hierarchy::with_equilibrium`]
/// declares it on the single hierarchy: every held level seeded with the target through `seed`,
/// the levels primed coarsest first with each level's cut halos crossing at its own point before a
/// child prolongs from them; each level's anchor, interface flux and imbalance read off one
/// forward-euler stage at the level's cfl step, the minimum over every worker's tiles of the level,
/// with the troubled flags across the cuts; every tile restored afterwards. a worker holding no tile
/// of a level takes every point and vote of it
#[allow(clippy::too_many_arguments)]
fn probe_refined_target<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    depth: usize,
    transport: &T,
    fabric: &mut Fabric<L>,
    deadline: Duration,
    timing: &mut WorkerTiming,
    seed: &mut dyn FnMut(&Hierarchy<R, D, DOF, M, E, S, Mem, K>) -> Result<(), String>,
) -> Result<(), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
{
    use super::equilibrium::{imbalance_from_stage, restore_gas_state, save_gas_state, snapshot_flux};
    use symbi_sim::stage::{StageArgs, StageOutcome, fold_stage_from_correction, fold_stage_through_mark};
    let numerics = |detail: String| WorkerError::Numerics(detail);
    // the tiles holding level `ll` on this worker, their ids on the level's plan, the plan and the
    // devices by id
    let members = |ll: usize, tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>]| -> (Vec<usize>, Vec<TileId>) {
        let mut ks = Vec::new();
        let mut ids = Vec::new();
        for k in 0..tiles.len() {
            let id = if ll == 0 { Some(held.root_ids[k]) } else { held.level_ids(ll)[k] };
            if let Some(id) = id {
                ks.push(k);
                ids.push(id);
            }
        }
        (ks, ids)
    };
    let plan_of = |ll: usize| -> &PlanExchange<'_, D> {
        match ll {
            0 => &plans.root,
            1 => &plans.fine,
            _ => &plans.nested[ll - 2].state,
        }
    };
    let devices_of = |ll: usize| -> &[i32] { if ll == 0 { held.root_devices } else { held.level_devices(ll) } };
    let saved = tiles
        .iter()
        .map(|t| t.levels.iter().map(|l| save_gas_state(&l.state)).collect::<symbi_xpu::Result<Vec<_>>>())
        .collect::<symbi_xpu::Result<Vec<_>>>()
        .map_err(|e| fail(fabric, numerics(format!("stationary target save: {e:?}"))))?;
    for tile in tiles.iter() {
        seed(tile).map_err(|e| fail(fabric, WorkerError::Refused(e)))?;
    }
    for ll in 0..depth {
        let (ks, ids) = members(ll, tiles);
        let (plan, devices) = (plan_of(ll), devices_of(ll));
        for (r, &k) in ks.iter().enumerate() {
            with_device(devices[ids[r].0 as usize], || tiles[k].prime_target_level(ll));
        }
        {
            let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..plan.plan().n_tiles()).map(|_| Vec::new()).collect();
            for (r, &k) in ks.iter().enumerate() {
                lists[ids[r].0 as usize] = plan_fields(&tiles[k].levels[ll].state);
            }
            let part = PlanPart { plan, fields: &lists, devices, fault: None, dst_fields: None };
            let point = ExchangePoint::EquilibriumLevel { level: ll as u8, troubled: false };
            exchange_point_plans(&[part], transport, fabric, point, 0, 0, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
        }
        for (r, &k) in ks.iter().enumerate() {
            let level = &tiles[k].levels[ll];
            with_device(devices[ids[r].0 as usize], || level.kernels.ghost_fill(&level.state));
            let err = scan_c2p_errors(&level.state);
            if err.is_err() {
                return Err(fail(fabric, numerics(format!("level {ll}: the stationary target's recovery failed ({err})"))));
            }
        }
    }
    let anchors = tiles
        .iter()
        .map(|t| t.levels.iter().map(|l| save_gas_state(&l.state)).collect::<symbi_xpu::Result<Vec<_>>>())
        .collect::<symbi_xpu::Result<Vec<_>>>()
        .map_err(|e| fail(fabric, numerics(format!("stationary target anchor: {e:?}"))))?;
    let mut fluxes: Vec<Vec<_>> = tiles.iter().map(|t| t.levels.iter().map(|_| None).collect()).collect();
    let mut residuals: Vec<Vec<_>> = tiles.iter().map(|t| t.levels.iter().map(|_| None).collect()).collect();
    for ll in 0..depth {
        let (ks, ids) = members(ll, tiles);
        let (plan, devices) = (plan_of(ll), devices_of(ll));
        // a worker without a tile of the level offers the largest finite step, which every other
        // offer undercuts
        let mut local = f64::MAX;
        for (r, &k) in ks.iter().enumerate() {
            let level = &tiles[k].levels[ll];
            let dev = devices[ids[r].0 as usize];
            with_device(dev, || {
                level.kernels.wave_speeds(&level.state);
                for dd in 0..D {
                    level.kernels.flux(&level.state, dd);
                }
            });
            fluxes[k][ll] = Some(snapshot_flux(&level.state).map_err(|e| fail(fabric, numerics(format!("stationary target flux: {e:?}"))))?);
            // a tile whose level lies wholly under a finer level evolves no cell of it and offers
            // no step
            if !tiles[k].level_evolves_cells(ll) {
                continue;
            }
            let candidate = with_device(dev, || level.kernels.cfl(&level.state));
            if !(candidate.is_finite() && candidate > 0.0) {
                return Err(fail(fabric, numerics(format!(
                    "the stationary target has no finite cfl step ({candidate:.3e}) on level {ll}, so it cannot be advanced to read off its imbalance"
                ))));
            }
            local = local.min(candidate);
        }
        let dt = f64::from_bits(fabric.collective(OpKind::Min, local.to_bits(), deadline).map_err(|e| fail(fabric, e.into()))?);
        // the probe's step per level, the timestep the single hierarchy's probe takes
        eprintln!("SIMBI equilibrium probe: level={ll} dt_bits={:016x}", dt.to_bits());
        let args = StageArgs {
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
        for (r, &k) in ks.iter().enumerate() {
            let level = &tiles[k].levels[ll];
            troubled += with_device(devices[ids[r].0 as usize], || fold_stage_through_mark(&level.state, &level.kernels, args, &mut |_| {}));
        }
        let mut exchanges = 0u64;
        let point = ExchangePoint::EquilibriumLevel { level: ll as u8, troubled: true };
        let cut_trouble =
            fine_level_trouble(tiles, ll, &ks, plan, &ids, devices, troubled, point, 0, 0, transport, fabric, deadline, timing, &mut exchanges)?;
        let mut rejected = false;
        for (r, &k) in ks.iter().enumerate() {
            let level = &tiles[k].levels[ll];
            // the probe is a diagnostic stage: each kernel's freeze count stands where the probe
            // found it, so the march's level-wide count starts equal on every tile of every worker
            let streak = level.kernels.freeze_streak();
            let outcome = with_device(devices[ids[r].0 as usize], || {
                fold_stage_from_correction(&level.state, &level.kernels, args, cut_trouble[r], &mut |_| {})
            });
            level.kernels.set_freeze_streak(streak);
            rejected |= outcome != StageOutcome::Accepted;
            level.state.reset_state_clock();
        }
        if vote_retry(fabric, rejected, deadline, timing)? {
            return Err(fail(fabric, numerics(format!("level {ll}: the stationary target's probe stage was rejected by the admissibility redo"))));
        }
        for &k in &ks {
            let level = &tiles[k].levels[ll];
            let err = scan_c2p_errors(&level.state);
            if err.is_err() {
                return Err(fail(fabric, numerics(format!(
                    "advancing the stationary target by one cfl step left an unrecoverable state on level {ll} ({err})"
                ))));
            }
            residuals[k][ll] = Some(
                imbalance_from_stage(anchors[k][ll].cons(), &level.state.fields.cons, &level.state.geom.allocated, dt)
                    .map_err(|e| fail(fabric, numerics(format!("stationary target residual: {e:?}"))))?,
            );
        }
    }
    for ((tile, saved), ((anchors, fluxes), residuals)) in
        tiles.iter_mut().zip(&saved).zip(anchors.into_iter().zip(fluxes).zip(residuals))
    {
        for (level, state) in tile.levels.iter().zip(saved) {
            restore_gas_state(&level.state, state);
        }
        tile.install_declared_target(anchors, fluxes, residuals);
    }
    Ok(())
}

/// the tail of refined level `level` after its coefficients crossed the cuts, on a run carrying
/// continuous tracers: every held tile advances its tracers, the tracers cross to the tiles holding
/// their cells, the held injection spawns once, and every held tile finishes its tail
#[allow(clippy::too_many_arguments)]
fn continuous_tail<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    atlas: &LevelAtlas<D>,
    level: usize,
    dt: f64,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    let mine: Vec<usize> = (0..tiles.len()).filter(|&k| held.level_ids(level)[k].is_some()).collect();
    let device = |k: usize| held.level_devices(level)[held.level_ids(level)[k].unwrap().0 as usize];
    for &k in &mine {
        with_device(device(k), || tiles[k].advance_level_continuous_tracers(level, dt));
    }
    migrate_level_continuous(tiles, held, atlas, level, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
    spawn_level_continuous(tiles, held, atlas, level, fabric, deadline).map_err(|e| fail(fabric, e.into()))?;
    for &k in &mine {
        with_device(device(k), || tiles[k].finish_level_tail_post(level, dt));
    }
    Ok(())
}

/// the per-level census row of `level` once every worker's tiles have finished its substep: the
/// level's clock agreed across the workers (the root's is every worker's own), each worker's
/// holders merged exactly, the workers' partials merged into the whole level's sample, and the
/// row recorded on every tile's root. a worker holding no tile of the level joins with empty
/// partials, so every worker keeps the same schedule
#[allow(clippy::too_many_arguments)]
fn launched_level_census<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    level: usize,
    device: impl Fn(usize) -> i32,
    workers: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    if !tiles[0].has_per_level_census() {
        return Ok(());
    }
    let now = if level == 0 { tiles[0].levels[0].state.time } else { agree_fine_clock(tiles, level, fabric, deadline)?.0 };
    let local = crate::refinement::hierarchy::tiles_level_census(tiles, level, now, device);
    let total = symbi_sim::worker::combine_census_partials(fabric, workers, &tiles[0].levels[0].state.censuses, &local, deadline)?;
    crate::refinement::hierarchy::record_tiles_level_census(tiles, level, now, &total);
    Ok(())
}

fn vote_retry<L: Link>(
    fabric: &mut Fabric<L>,
    retry: bool,
    deadline: Duration,
    timing: &mut WorkerTiming,
) -> Result<bool, WorkerError> {
    let clock = Instant::now();
    let reject = fabric
        .collective(OpKind::Any, u64::from(retry), deadline)
        .map_err(|e| fail(fabric, e.into()))?;
    timing.collectives += clock.elapsed();
    Ok(reject != 0)
}

/// the troubled flags of one level across its cuts at `point`, from the tiles `mine` this worker
/// holds at the level (`ids` their tile ids, `devices` the level's devices by tile id); per held
/// tile, whether a cut ghost received a flag (a clean tile then corrects at the cut)
#[allow(clippy::too_many_arguments)]
fn exchange_level_flags<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    level: usize,
    mine: &[usize],
    plan: &PlanExchange<'_, D>,
    ids: &[TileId],
    devices: &[i32],
    point: ExchangePoint,
    step: u64,
    attempt: u16,
    transport: &T,
    fabric: &mut Fabric<L>,
    deadline: Duration,
    timing: &mut WorkerTiming,
) -> Result<Vec<bool>, WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
{
    let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..plan.plan().n_tiles()).map(|_| Vec::new()).collect();
    for (r, &k) in mine.iter().enumerate() {
        lists[ids[r].0 as usize] = plan_fields(&tiles[k].levels[level].state);
    }
    let part = PlanPart { plan, fields: &lists, devices, fault: None, dst_fields: None };
    exchange_point_plans(&[part], transport, fabric, point, step, attempt, deadline, &mut timing.exchange)
        .map_err(|e| fail(fabric, e.into()))?;
    Ok(mine
        .iter()
        .enumerate()
        .map(|(r, &k)| {
            let l = &tiles[k].levels[level];
            with_device(devices[ids[r].0 as usize], || {
                plan.trouble_regions(ids[r]).iter().any(|region| l.kernels.fofc_flags_in(&l.state, region) > 0)
            })
        })
        .collect())
}

/// the troubled marks of one level across the cuts for a single probe stage: one vote for every
/// tile this worker holds, then the flag exchange when any worker is troubled
#[allow(clippy::too_many_arguments)]
fn fine_level_trouble<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    level: usize,
    refined: &[usize],
    plan: &PlanExchange<'_, D>,
    fine_ids: &[TileId],
    devices: &[i32],
    troubled: u64,
    point: ExchangePoint,
    step: u64,
    attempt: u16,
    transport: &T,
    fabric: &mut Fabric<L>,
    deadline: Duration,
    timing: &mut WorkerTiming,
    exchanges: &mut u64,
) -> Result<Vec<bool>, WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
{
    let clock = Instant::now();
    let any = fabric
        .collective(OpKind::Any, u64::from(troubled > 0), deadline)
        .map_err(|e| fail(fabric, e.into()))?;
    timing.collectives += clock.elapsed();
    if any == 0 {
        return Ok(vec![false; refined.len()]);
    }
    *exchanges += 1;
    exchange_level_flags(tiles, level, refined, plan, fine_ids, devices, point, step, attempt, transport, fabric, deadline, timing)
}

/// one stage of one refined level on a worker: the tiles `mine` it holds at the level fold through
/// the hierarchy's own stage halves on the level's devices, a test fault may write a global cell
/// troubled, the flags cross the level's cuts at the stage's troubled exchange point, and every
/// agreement is one vote of this worker over the fabric. a worker holding no tile at the level
/// votes with neutral inputs
struct LaunchedLevelStage<'a, 'p, R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
{
    tiles: &'a mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    mine: &'a [usize],
    ids: &'a [TileId],
    devices: &'a [i32],
    plan: &'a PlanExchange<'p, D>,
    level: usize,
    ii: usize,
    dt: f64,
    alpha: f64,
    point: ExchangePoint,
    step: u64,
    attempt: u16,
    transport: &'a T,
    fabric: &'a mut Fabric<L>,
    deadline: Duration,
    timing: &'a mut WorkerTiming,
    /// the global cell a test fault writes troubled at this stage
    injected: Option<[isize; D]>,
    /// a test fault rejects this stage
    forced: bool,
    /// each held tile's tracer interface ledger length as its stage opened
    receipt_starts: Vec<usize>,
}

impl<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L> symbi_sim::collective_stage::StageCollective
    for LaunchedLevelStage<'_, '_, R, D, DOF, M, E, S, Mem, K, T, L>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
{
    type Error = WorkerError;

    fn tiles(&self) -> usize {
        self.mine.len()
    }

    fn mark(&mut self, r: usize) -> Result<u64, WorkerError> {
        let (tile, level, ii, dt, alpha) = (&mut self.tiles[self.mine[r]], self.level, self.ii, self.dt, self.alpha);
        let mark = with_device(self.devices[self.ids[r].0 as usize], || tile.level_stage_through_mark(level, ii, dt, alpha));
        self.receipt_starts[r] = mark.receipt_start;
        Ok(mark.troubled)
    }

    fn injected_trouble(&mut self, r: usize) -> u64 {
        let Some(local) = self.injected.and_then(|cell| self.plan.local_cell(self.ids[r], cell)) else { return 0 };
        *self.tiles[self.mine[r]].levels[self.level].state.workspace.fofc_flag.view_mut().at_mut(local) = 1.0;
        1
    }

    fn any(&mut self, local: bool) -> Result<bool, WorkerError> {
        let clock = Instant::now();
        let any = self
            .fabric
            .collective(OpKind::Any, u64::from(local), self.deadline)
            .map_err(|e| fail(self.fabric, e.into()))?;
        self.timing.collectives += clock.elapsed();
        Ok(any != 0)
    }

    fn exchange_trouble(&mut self) -> Result<Vec<bool>, WorkerError> {
        exchange_level_flags(
            self.tiles,
            self.level,
            self.mine,
            self.plan,
            self.ids,
            self.devices,
            self.point,
            self.step,
            self.attempt,
            self.transport,
            self.fabric,
            self.deadline,
            self.timing,
        )
    }

    fn correct(&mut self, r: usize, cut_trouble: bool) -> Result<bool, WorkerError> {
        let (tile, level, ii, dt, alpha, start) = (&mut self.tiles[self.mine[r]], self.level, self.ii, self.dt, self.alpha, self.receipt_starts[r]);
        Ok(with_device(self.devices[self.ids[r].0 as usize], || tile.level_stage_from_correction(level, ii, dt, alpha, cut_trouble, start)))
    }

    fn freeze_streak(&self, r: usize) -> u32 {
        self.tiles[self.mine[r]].levels[self.level].kernels.freeze_streak()
    }

    fn set_freeze_streak(&mut self, r: usize, streak: u32) {
        self.tiles[self.mine[r]].levels[self.level].kernels.set_freeze_streak(streak);
    }

    fn forced_rejection(&mut self) -> bool {
        self.forced
    }
}

/// a register unit crossing a cut is admitted on a uniform cartesian root alone: the
/// constant-area register kernels are the gated arithmetic; the metric-weighted register
/// path of a curvilinear chart awaits its own arithmetic and process gates
pub fn refuse_transported_registers<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
) -> Result<(), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    if tiles.iter().any(|h| h.transports_registers()) && !tiles.iter().all(|h| h.root_is_uniform_cartesian()) {
        return Err(WorkerError::Refused(
            "a flux-register contribution across a cut on a curvilinear chart is outside the launched \
             refinement until its metric-weighted arithmetic is gated"
                .into(),
        ));
    }
    Ok(())
}

/// the global grids of a refined session's two levels as the checkpoint records them: the
/// root's, and the fine level's over the patch at twice the resolution, its first interior
/// cell at twice the patch's lower corner, its halo one deeper than the root's
#[derive(Clone, Debug)]
pub struct RefinedGrids<const D: usize> {
    pub root: GlobalGrid<D>,
    pub fine: GlobalGrid<D>,
    /// the grid of every nested level, the level `l` at `l - 2`
    pub nested: Vec<GlobalGrid<D>>,
}

impl<const D: usize> RefinedGrids<D> {
    /// the grids of a root of `cells` with its first cell at `x_lo`, spacing `dx` and halo `ng`,
    /// refining `patch` (root cells per axis) at ratio 2
    pub fn of(cells: [usize; D], x_lo: [f64; D], dx: [f64; D], ng: usize, patch: [(usize, usize); D]) -> Self {
        Self {
            root: GlobalGrid { cells, interior_lo: [0; D], ng, x_lo, dx, maps: None },
            fine: GlobalGrid {
                cells: std::array::from_fn(|a| (patch[a].1 - patch[a].0) * RATIO),
                interior_lo: std::array::from_fn(|a| (patch[a].0 * RATIO) as isize),
                ng: ng + 1,
                x_lo,
                dx: std::array::from_fn(|a| dx[a] / RATIO as f64),
                maps: None,
            },
            nested: Vec::new(),
        }
    }

    /// these grids with nested level `fine.len() + 2` over `cells` of that level, its first cell
    /// at whole-level index `interior_lo`
    pub fn with_nested(mut self, cells: [usize; D], interior_lo: [isize; D]) -> Self {
        let parent = self.nested.last().unwrap_or(&self.fine);
        let level = GlobalGrid {
            cells,
            interior_lo,
            ng: parent.ng,
            x_lo: parent.x_lo,
            dx: std::array::from_fn(|a| parent.dx[a] / RATIO as f64),
            maps: None,
        };
        self.nested.push(level);
        self
    }

    /// the grid of refined level `level >= 1`
    pub fn level(&self, level: usize) -> &GlobalGrid<D> {
        if level == 1 { &self.fine } else { &self.nested[level - 2] }
    }
}

/// the per-root-step body bookkeeping of a refined session, in the in-process refined tiles'
/// order: each tile's finest level reduces its feedback partials, every tile's deltas cross the
/// workers (a tile's list as long as the bodies its finest level touched, keyed by body) and are
/// summed in flat root tile order, and the global delta is applied to every held tile's bodies at
/// the finest level's clock, the one the single hierarchy books its history at
fn refined_body_step<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    fabric: &mut Fabric<L>,
    dt: f64,
    deadline: Duration,
) -> Result<(), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    use symbi_sim::worker::{body_delta_words, decode_body_delta, encode_body_delta};
    for h in tiles.iter_mut() {
        h.finest_body_feedback(dt);
    }
    let width = 1 + body_delta_words(D);
    let local: Vec<(usize, Vec<u64>)> = tiles
        .iter_mut()
        .enumerate()
        .map(|(k, h)| {
            let mut words = Vec::new();
            for delta in h.take_body_deltas() {
                words.push(delta.idx as u64);
                encode_body_delta(&delta, &mut words);
            }
            (held.root_ids[k].0 as usize, words)
        })
        .collect();
    let blocks = symbi_sim::worker_tracers::assemble_blocks(fabric, held.n_root, &local, width, deadline)?;
    let mut per_tile = Vec::with_capacity(blocks.len());
    for block in &blocks {
        let deltas = block
            .chunks_exact(width)
            .map(|record| decode_body_delta::<D>(record[0] as usize, &record[1..]))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| WorkerError::Numerics(format!("a tile's body feedback is malformed: {e}")))?;
        per_tile.push(deltas);
    }
    let global = symbi_sim::decomp::merge_body_deltas(per_tile);
    let finest = 1 + plans.nested.len();
    let (t_now, _, _) = agree_fine_clock(tiles, finest, fabric, deadline)?;
    for (k, h) in tiles.iter_mut().enumerate() {
        h.try_apply_global_body_deltas(&global, dt, t_now).map_err(WorkerError::Numerics)?;
        if h.levels.len() == 1 && h.accretion_overlaps_root() {
            return Err(WorkerError::Numerics(format!(
                "a sink sphere overlaps the unrefined root tile {}: the refined patch must cover the sink on every tile it touches",
                held.root_ids[k].0
            )));
        }
    }
    Ok(())
}

/// record on every level of every held tile its whole level's interior in the tile's own index
/// frame, from the level grids and the plan offsets the checkpoint places the tile's blocks by:
/// a bound that reads the level's extent (the alpha viscosity's slowest orbit) takes the whole
/// level's, as the single hierarchy does, whichever tile holds a part of it
pub fn record_level_spans<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    grids: &RefinedGrids<D>,
) where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    let depth = 2 + plans.nested.len();
    for (k, h) in tiles.iter_mut().enumerate() {
        let offset = plans.root.plan().tiles[held.root_ids[k].0 as usize].offset;
        let whole = grids.root.interior_span(&h.levels[0].state.geom, offset);
        h.levels[0].state.geom.whole_span = Some(whole);
        for level in 1..depth.min(h.levels.len()) {
            let Some(id) = held.level_ids(level)[k] else {
                continue;
            };
            let plan = if level == 1 { &plans.fine } else { &plans.nested[level - 2].state };
            let grid = grids.level(level);
            let layout = &plan.plan().tiles[id.0 as usize];
            let offset: [isize; D] = std::array::from_fn(|a| grid.interior_lo[a] + layout.offset[a]);
            let whole = grid.interior_span(&h.levels[level].state.geom, offset);
            h.levels[level].state.geom.whole_span = Some(whole);
        }
    }
}

/// write one checkpoint of every level across every worker after `accepted` accepted steps:
/// the shared faces are audited at the checkpoint epoch unless `audit` is off (a second
/// checkpoint of a step an earlier one certified), each refined level's clock is
/// agreed from the workers holding fine tiles, and every worker streams its owned blocks of
/// both levels to the coordinator, which publishes the file when every block arrived once
#[allow(clippy::too_many_arguments)]
pub fn refined_checkpoint<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    grids: &RefinedGrids<D>,
    identity: &PhysicsIdentity,
    req: &CheckpointRequest<'_>,
    fabric: &mut Fabric<L>,
    accepted: u64,
    audit: bool,
) -> Result<(), WorkerError>
where
    R: Regime<f64, D, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF> + Copy,
    M: Metric<f64, D> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
    <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
{
    // every held tile level's conserved halo published as the conserved state of its halo
    // primitives, restored once the file is written
    let halos: Vec<Vec<_>> = tiles.iter().map(|h| h.levels.iter().map(|l| l.state.prepare_checkpoint_halo()).collect()).collect();
    let written = refined_checkpoint_levels(tiles, held, plans, grids, identity, req, fabric, accepted, audit);
    for (h, tile_halos) in tiles.iter().zip(&halos) {
        for (l, halo) in h.levels.iter().zip(tile_halos) {
            l.state.restore_checkpoint_halo(halo);
        }
    }
    written
}

/// the levels of `refined_checkpoint` written as they stand
#[allow(clippy::too_many_arguments)]
fn refined_checkpoint_levels<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    grids: &RefinedGrids<D>,
    identity: &PhysicsIdentity,
    req: &CheckpointRequest<'_>,
    fabric: &mut Fabric<L>,
    accepted: u64,
    audit: bool,
) -> Result<(), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    let checkpoint_err = |e: String| WorkerError::Checkpoint(e);
    // the shared faces of both levels at the checkpoint epoch of the last accepted step; the
    // primed state was certified by the prime audit after initialization
    let depth = 2 + plans.nested.len();
    let audited = plans.root.has_audits() || plans.fine.has_audits() || plans.nested.iter().any(|p| p.state.has_audits());
    if audited && accepted > 0 && audit {
        let mut root_lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
        for (k, h) in tiles.iter().enumerate() {
            root_lists[held.root_ids[k].0 as usize] = plan_fields(&h.levels[0].state);
        }
        // every refined level's lists, by that level's tile id
        let level_lists: Vec<Vec<Vec<&Field<f64, D, Mem>>>> = (1..depth)
            .map(|level| {
                let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.level_tiles(level)).map(|_| Vec::new()).collect();
                for (k, h) in tiles.iter().enumerate() {
                    if let Some(id) = held.level_ids(level)[k] {
                        lists[id.0 as usize] = plan_fields(&h.levels[level].state);
                    }
                }
                lists
            })
            .collect();
        let mut audits: Vec<(&PlanExchange<'_, D>, &[Vec<&Field<f64, D, Mem>>])> = vec![(&plans.root, &root_lists[..]), (&plans.fine, &level_lists[0][..])];
        for (i, nested) in plans.nested.iter().enumerate() {
            audits.push((&nested.state, &level_lists[i + 1][..]));
        }
        audit_plans(
            &audits,
            fabric,
            ExchangePoint::Audit(AuditAt::Checkpoint),
            accepted - 1,
            CHECKPOINT_ATTEMPT,
            req.deadline,
        )
        .map_err(|e| match e {
            AuditError::Fabric(e) => WorkerError::Fabric(e),
            AuditError::Mismatch(m) => WorkerError::Interface(m),
        })?;
    }
    let root_tiles: Vec<TileView<'_, D, DOF, Mem>> = tiles
        .iter()
        .zip(held.root_ids)
        .map(|(h, id)| TileView { state: &h.levels[0].state, offset: plans.root.plan().tiles[id.0 as usize].offset })
        .collect();
    let mut levels = vec![LevelTiles { tiles: root_tiles, grid: grids.root.clone(), header: None }];
    let mut ids_by_level: Vec<Vec<TileId>> = Vec::with_capacity(depth - 1);
    for level in 1..depth {
        // every refined level's clock, agreed from the workers holding its tiles
        let (time, dt, iteration) = agree_fine_clock(tiles, level, fabric, req.deadline)?;
        let plan = if level == 1 { &plans.fine } else { &plans.nested[level - 2].state };
        let grid = grids.level(level);
        let mut ids = Vec::new();
        let level_tiles: Vec<TileView<'_, D, DOF, Mem>> = tiles
            .iter()
            .zip(held.level_ids(level))
            .filter_map(|(h, id)| id.map(|id| (h, id)))
            .map(|(h, id)| {
                ids.push(id);
                let layout = &plan.plan().tiles[id.0 as usize];
                TileView { state: &h.levels[level].state, offset: std::array::from_fn(|a| grid.interior_lo[a] + layout.offset[a]) }
            })
            .collect();
        ids_by_level.push(ids);
        // the level's template is this worker's own tile of the level when it holds one: its field
        // set, and the finest level's body state and history the file records; a worker holding
        // no tile of the level names its first tile's deepest level, which carries the full bodies
        // and books the global body delta every tile applies (coarser levels hold gravity-only
        // proxies)
        let template = tiles
            .iter()
            .zip(held.level_ids(level))
            .find_map(|(h, id)| id.map(|_| &h.levels[level].state))
            .unwrap_or(&tiles[0].levels[tiles[0].levels.len() - 1].state);
        levels.push(LevelTiles {
            tiles: level_tiles,
            grid: grid.clone(),
            header: Some(LevelHeader { template, time, dt, iteration }),
        });
    }
    let mut level_plans =
        vec![LevelPlan { plan: plans.root.plan(), placement: plans.root.placement(), origin: [0; D], tile_ids: held.root_ids }];
    for level in 1..depth {
        let plan = if level == 1 { &plans.fine } else { &plans.nested[level - 2].state };
        level_plans.push(LevelPlan {
            plan: plan.plan(),
            placement: plan.placement(),
            origin: grids.level(level).interior_lo,
            tile_ids: &ids_by_level[level - 1],
        });
    }
    distributed_checkpoint_levels::<R, D, DOF, Mem, L>(fabric, &level_plans, identity, &levels, req)
        .map_err(|e| checkpoint_err(format!("{e}")))
}

/// load both levels of every held tile from a checkpoint of a refined session written on any
/// partition: each tile reads its interior and halo boxes of each level from the whole-grid
/// datasets, the cut halos from the neighbors' interiors, the clocks from the level groups
pub fn load_refined_tiles<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    grids: &RefinedGrids<D>,
    path: &str,
) -> Result<(), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    let err = |e: symbi_sim::checkpoint::IoError| WorkerError::Checkpoint(format!("{path}: {e}"));
    let depth = 2 + plans.nested.len();
    let stored = checkpoint_level_count(path).map_err(err)?;
    // a checkpoint deeper than the session would lose levels; a shallower one seeds the levels it
    // lacks from their parents after the load, as the single hierarchy's deeper restart does
    if stored > depth {
        return Err(WorkerError::Refused(format!(
            "{path} carries {stored} level(s); a refined session of {depth} levels may add levels to a checkpoint, never drop them"
        )));
    }
    for (k, h) in tiles.iter_mut().enumerate() {
        let root_offset = plans.root.plan().tiles[held.root_ids[k].0 as usize].offset;
        load_partitioned_level(&mut h.levels[0].state, path, 0, root_offset, &grids.root).map_err(err)?;
        for level in 1..stored {
            if let Some(id) = held.level_ids(level)[k] {
                let plan = if level == 1 { &plans.fine } else { &plans.nested[level - 2].state };
                let grid = grids.level(level);
                let layout = &plan.plan().tiles[id.0 as usize];
                let offset: [isize; D] = std::array::from_fn(|a| grid.interior_lo[a] + layout.offset[a]);
                load_partitioned_level(&mut h.levels[level].state, path, level, offset, grid).map_err(err)?;
            }
        }
        h.inject_levels_from(stored).map_err(WorkerError::Refused)?;
    }
    Ok(())
}

/// the parent face stencil boxes of the fine level of root tile `tile`, in that tile's root
/// index space, derived from the partition and the patch alone as the hierarchy's builder
/// derives the fine level: its interior over the patch clipped to the tile at twice the
/// resolution; each side a coarse-fine boundary except where the clipped patch meets the
/// tile's physical boundary; a side on a cut the patch continues across takes its face halo
/// from the fine exchange. `None` for a tile the patch misses. every worker derives every
/// tile's boxes this way to compile the stencil plan; `Hierarchy::face_stencil_boxes` of a
/// held tile equals it
pub fn face_stencil_boxes_of_tile<const D: usize>(
    partition: &symbi_sim::decomp::Partition<D>,
    patch: [(usize, usize); D],
    tile: TileId,
) -> Option<[Option<symbi_algebra::Domain<D>>; D]> {
    face_stencil_boxes_of_level_tile(partition, patch, tile, true, [false; D])
}

/// `face_stencil_boxes_of_tile` on the partition of any level: `edges_physical` says whether the
/// partition's outer edges are the domain's walls (the root) or its patch's coarse-fine boundary (a
/// refined level parenting a nested one), and `spans` names the periodic axes the patch spans
/// whole, along which its level wraps (within a tile, or across the tiles at the seam) and takes
/// its face halo from its own wrap, as the builder's level does
pub fn face_stencil_boxes_of_level_tile<const D: usize>(
    partition: &symbi_sim::decomp::Partition<D>,
    patch: [(usize, usize); D],
    tile: TileId,
    edges_physical: bool,
    spans: [bool; D],
) -> Option<[Option<symbi_algebra::Domain<D>>; D]> {
    use symbi_algebra::{Domain, Space};
    use symbi_sim::state::{Boundaries, BoundaryType, axis_name};
    let counts = partition.counts();
    let tc = symbi_sim::decomp::unflatten(tile.0 as usize, counts);
    let ext = partition.tile_extents(tc);
    let lo: [usize; D] = std::array::from_fn(|a| patch[a].0.max(ext[a].0));
    let hi: [usize; D] = std::array::from_fn(|a| patch[a].1.min(ext[a].0 + ext[a].1));
    if (0..D).any(|a| hi[a] <= lo[a]) {
        return None;
    }
    let r = RATIO as isize;
    let interior = Domain::new(std::array::from_fn(|a| Space {
        name: axis_name(a),
        lo: (lo[a] - ext[a].0) as isize * r,
        hi: (hi[a] - ext[a].0) as isize * r,
    }));
    let boundaries = Boundaries(std::array::from_fn(|a| {
        let side = |at_tile_edge: bool, tile_is_domain_edge: bool, continues: bool| {
            if !at_tile_edge || continues {
                // a coarse-fine side, or a cut the patch continues across: the face halo comes
                // from the root lattice by prolongation, or from the fine exchange
                if at_tile_edge && continues { BoundaryType::Outflow } else { BoundaryType::CoarseFine }
            } else if tile_is_domain_edge && edges_physical {
                BoundaryType::Outflow
            } else {
                // the patch's real face on a cut: a coarse-fine side
                BoundaryType::CoarseFine
            }
        };
        [
            side(lo[a] == ext[a].0, tc[a] == 0, lo[a] > patch[a].0 || spans[a]),
            side(hi[a] == ext[a].0 + ext[a].1, tc[a] + 1 == counts[a], hi[a] < patch[a].1 || spans[a]),
        ]
    }));
    Some(super::transfer::face_stencil_boxes(&interior, &boundaries))
}

/// the parent prim stencil of the fine level of every tile of a partition, in whole-partition cell
/// indices, derived from the partition and the patch alone as the hierarchy's builder derives it:
/// a tile holding a real patch face on a cut, under a parent halo of `ng` cells shallower than the
/// prolongation reach of the fine level's `fine_ng` ghosts, holds the patch coverage grown by that
/// reach, past its allocation on those sides and clipped to it elsewhere; every other tile holds
/// none. every refined level carries one ghost cell more than the root, so a first level's
/// `fine_ng` is the root's `ng + 1` and a nested level's equals its parent's `ng`. `wraps` names the root axes
/// closed into a ring, where a cut domain face is a seam and a patch spanning the axis wraps across it.
/// `Hierarchy::prim_stencil` of a held tile equals its entry, in the tile's own index space
pub fn prim_stencil_boxes<const D: usize>(
    partition: &symbi_sim::decomp::Partition<D>,
    patch: [(usize, usize); D],
    ng: usize,
    fine_ng: usize,
    order: super::transfer::ProlongOrder,
    wraps: [bool; D],
) -> Vec<Option<[symbi_sim::atlas::Span; D]>> {
    use symbi_sim::atlas::Span;
    let counts = partition.counts();
    let reach = (fine_ng.div_ceil(RATIO) + order.ghost_width()) as isize;
    let cells: [usize; D] = std::array::from_fn(|a| {
        (0..counts[a]).map(|ii| { let mut tc = [0; D]; tc[a] = ii; partition.tile_extents(tc)[a].1 }).sum()
    });
    (0..partition.n_tiles())
        .map(|t| {
            let tc = symbi_sim::decomp::unflatten(t, counts);
            let ext = partition.tile_extents(tc);
            let lo: [isize; D] = std::array::from_fn(|a| patch[a].0.max(ext[a].0) as isize);
            let hi: [isize; D] = std::array::from_fn(|a| patch[a].1.min(ext[a].0 + ext[a].1) as isize);
            if (0..D).any(|a| hi[a] <= lo[a]) {
                return None;
            }
            let (tlo, thi): ([isize; D], [isize; D]) =
                (std::array::from_fn(|a| ext[a].0 as isize), std::array::from_fn(|a| (ext[a].0 + ext[a].1) as isize));
            let donated: [[bool; 2]; D] = std::array::from_fn(|a| {
                let seams = wraps[a] && counts[a] > 1;
                let spans = wraps[a] && patch[a] == (0, cells[a]);
                let short = reach > ng as isize && !spans;
                [
                    short && lo[a] == tlo[a] && patch[a].0 as isize == tlo[a] && (tc[a] > 0 || seams),
                    short && hi[a] == thi[a] && patch[a].1 as isize == thi[a] && (tc[a] + 1 < counts[a] || seams),
                ]
            });
            if donated.iter().flatten().all(|&d| !d) {
                return None;
            }
            let g = ng as isize;
            Some(std::array::from_fn(|a| Span {
                lo: if donated[a][0] { lo[a] - reach } else { (lo[a] - reach).max(tlo[a] - g) },
                hi: if donated[a][1] { hi[a] + reach } else { (hi[a] + reach).min(thi[a] + g) },
            }))
        })
        .collect()
}

/// the receiving registers of root tile `tile` beside the patch, which it does not hold: the
/// patch box in the tile's root index space (clipped across the faces it does not sit beside)
/// and the patch faces on cuts it sits beside. `None` for a tile whose interior misses the
/// one-cell shell around the patch; a tile meeting the shell with no such face still carries
/// an empty register, whose recovery reconciles the corrected faces it receives
pub fn receiving_faces_of_tile<const D: usize>(
    partition: &symbi_sim::decomp::Partition<D>,
    patch: [(usize, usize); D],
    tile: TileId,
) -> Option<(symbi_algebra::Domain<D>, [[bool; 2]; D])> {
    use symbi_algebra::{Domain, Space};
    let ext = partition.tile_extents(symbi_sim::decomp::unflatten(tile.0 as usize, partition.counts()));
    let mut faces = [[false; 2]; D];
    for ax in 0..D {
        for side in 0..2 {
            let face_pos = if side == 0 { patch[ax].0 } else { patch[ax].1 };
            let beside = if side == 1 { ext[ax].0 == face_pos } else { ext[ax].0 + ext[ax].1 == face_pos };
            let across = (0..D).all(|aa| aa == ax || (ext[aa].0 < patch[aa].1 && ext[aa].0 + ext[aa].1 > patch[aa].0));
            faces[ax][side] = beside && across;
        }
    }
    let meets_shell = (0..D).all(|a| ext[a].0 < patch[a].1 + 1 && ext[a].0 + ext[a].1 > patch[a].0.saturating_sub(1));
    if !faces.iter().flatten().any(|&f| f) && !meets_shell {
        return None;
    }
    let patch_local = Domain::new(std::array::from_fn(|a| {
        let origin = ext[a].0 as isize;
        let (lo, hi) = if faces[a].iter().any(|&f| f) {
            (patch[a].0 as isize - origin, patch[a].1 as isize - origin)
        } else {
            (patch[a].0.max(ext[a].0) as isize - origin, patch[a].1.min(ext[a].0 + ext[a].1) as isize - origin)
        };
        Space { name: symbi_sim::state::axis_name(a), lo, hi }
    }));
    Some((patch_local, faces))
}

/// the tile grids of the refined levels: entry `k` partitions level `k + 1`'s patch by the cuts
/// of the root falling inside it. `patches[0]` is the level-1 patch in root cells, `patches[k]`
/// the level-`k + 1` patch in level-`k` cells from the level-`k` patch's origin; every level
/// refines its parent by a ratio of two
pub fn nested_partitions<const D: usize>(
    root: &symbi_sim::decomp::Partition<D>,
    patches: &[[(usize, usize); D]],
) -> Result<Vec<symbi_sim::decomp::FinePartition<D>>, String> {
    let mut out: Vec<symbi_sim::decomp::FinePartition<D>> = Vec::with_capacity(patches.len());
    for patch in patches {
        let parent = out.last().map_or(root, |f| &f.partition);
        out.push(symbi_sim::decomp::FinePartition::of(parent, *patch, 2)?);
    }
    Ok(out)
}

/// root tile `root_tile`'s tile on refined level `level` (>= 1) and that tile's index frame: its
/// id in `fine[level - 1]`'s partition and the level index (whole-grid level index less the root
/// tile's origin times `2^level`) of its first interior cell. `None` for a root tile holding no
/// part of the level. `fine` is `nested_partitions(root, patches)`
pub fn nested_tile_of<const D: usize>(
    root: &symbi_sim::decomp::Partition<D>,
    fine: &[symbi_sim::decomp::FinePartition<D>],
    patches: &[[(usize, usize); D]],
    root_tile: TileId,
    level: usize,
) -> Option<(TileId, [isize; D])> {
    use symbi_sim::decomp::{flatten, unflatten};
    let root_tc = unflatten(root_tile.0 as usize, root.counts());
    let mut tc = root_tc;
    for f in &fine[..level] {
        tc = f.fine_tile_of(tc)?;
    }
    let part = &fine[level - 1].partition;
    // the level-`level` patch's whole-grid origin: each level doubles its parent's index and adds
    // its patch's offset in the parent's cells
    let origin: [isize; D] = std::array::from_fn(|a| patches[..level].iter().fold(0isize, |o, p| 2 * (o + p[a].0 as isize)));
    let ext = part.tile_extents(tc);
    let root_ext = root.tile_extents(root_tc);
    let frame = std::array::from_fn(|a| origin[a] + ext[a].0 as isize - ((root_ext[a].0 as isize) << level));
    Some((TileId(flatten(tc, part.counts()) as u32), frame))
}

/// the receiving register of root tile `root_tile` on refined level `level` (>= 1), which it
/// holds while holding none of level `level + 1`: the level-`level + 1` patch box in the tile's
/// own level-`level` index frame (`nested_tile_of`) and the patch faces on cuts of level `level`
/// it sits beside. `None` for a tile holding no part of level `level` or beside no such face.
/// `fine` is `nested_partitions(root, patches)`
pub fn nested_receiving_faces_of_tile<const D: usize>(
    root: &symbi_sim::decomp::Partition<D>,
    fine: &[symbi_sim::decomp::FinePartition<D>],
    patches: &[[(usize, usize); D]],
    root_tile: TileId,
    level: usize,
) -> Option<(symbi_algebra::Domain<D>, [[bool; 2]; D])> {
    assert!(level >= 1 && level < patches.len(), "a nested receiving register lies on a refined level with a finer patch");
    let (tile, frame) = nested_tile_of(root, fine, patches, root_tile, level)?;
    let part = &fine[level - 1].partition;
    let (patch_local, faces) = receiving_faces_of_tile(part, patches[level], tile)?;
    if !faces.iter().flatten().any(|&f| f) {
        return None;
    }
    // from the tile's partition-local cells to its own level frame
    let shifted = symbi_algebra::Domain::new(std::array::from_fn(|a| symbi_algebra::Space {
        name: patch_local.spaces[a].name,
        lo: patch_local.spaces[a].lo + frame[a],
        hi: patch_local.spaces[a].hi + frame[a],
    }));
    Some((shifted, faces))
}

/// the fine level's clock, agreed by every worker and verified: each worker's fine tiles must
/// share one clock; the words are reduced over the workers holding fine tiles, each 64-bit word
/// as its two 32-bit halves sent as the exact integers one above them, since the minimum
/// reduces over positive finite numbers (a worker holding no fine tile contributes the largest
/// finite number); every holder compares the agreed words with its own, and one vote refuses
/// the checkpoint on any disagreement, or when no worker holds a fine tile
fn agree_fine_clock<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    level: usize,
    fabric: &mut Fabric<L>,
    deadline: Duration,
) -> Result<(f64, f64, u64), WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    L: Link,
{
    let clocks: Vec<[u64; 3]> = tiles
        .iter()
        .filter(|h| h.levels.len() > level)
        .map(|h| {
            let s = &h.levels[level].state;
            [s.time.to_bits(), s.dt.to_bits(), s.iteration]
        })
        .collect();
    let mine = clocks.first().copied();
    let locally_agreed = clocks.iter().all(|c| Some(*c) == mine);
    let holders = fabric.collective(OpKind::Any, u64::from(mine.is_some()), deadline)?;
    if holders == 0 {
        return Err(WorkerError::Checkpoint(format!("no worker holds a tile of level {level}; the checkpoint has no clock for it")));
    }
    let mut agreed = [0u64; 3];
    for (w, word) in agreed.iter_mut().enumerate() {
        let mut halves = [0u64; 2];
        for (i, half) in halves.iter_mut().enumerate() {
            let own = mine.map(|c| ((c[w] >> (32 * i)) & 0xffff_ffff) as f64 + 1.0).unwrap_or(f64::MAX);
            let got = f64::from_bits(fabric.collective(OpKind::Min, own.to_bits(), deadline)?);
            *half = (got - 1.0) as u64;
        }
        *word = halves[0] | halves[1] << 32;
    }
    let differs = !locally_agreed || mine.is_some_and(|c| c != agreed);
    if fabric.collective(OpKind::Any, u64::from(differs), deadline)? != 0 {
        return Err(WorkerError::Checkpoint(format!(
            "the clock of level {level} differs across its tiles (this worker's tiles agree: {locally_agreed}; \
             time, dt, iteration here {:?}, agreed {:?}); no checkpoint is written",
            mine.map(|c| (f64::from_bits(c[0]), f64::from_bits(c[1]), c[2])),
            (f64::from_bits(agreed[0]), f64::from_bits(agreed[1]), agreed[2]),
        )));
    }
    Ok((f64::from_bits(agreed[0]), f64::from_bits(agreed[1]), agreed[2]))
}
