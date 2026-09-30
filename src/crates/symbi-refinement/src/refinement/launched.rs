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
// the step at a halved timestep. the fine tail's viscous pass, horizon ledger
// and bodies run on the globally finest level alone, as the single hierarchy
// runs them. three plans share the fabric: the root state, the root conserved
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
use symbi_sim::driver::{advance_clock, retry_timestep, select_timestep};
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

use super::hierarchy::{Hierarchy, RATIO, StageMark};

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
/// the step bound. `on_step` runs once on the primed state with zero accepted steps, then after
/// every accepted step with the count of accepted steps, the time, the tiles and the fabric; a
/// checkpoint written there (`refined_checkpoint`) holds that state.
#[allow(clippy::too_many_arguments)]
pub fn evolve_refined_worker<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L, F>(
    tiles: &mut [Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    held: &Held<'_>,
    plans: &RefinedPlans<'_, D>,
    transport: &T,
    fabric: &mut Fabric<L>,
    cfg: &RefinedConfig,
    mut on_step: F,
) -> Result<WorkerReport, WorkerError>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    T: HaloTransport,
    L: Link,
    F: FnMut(u64, f64, &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], &mut Fabric<L>) -> Result<ControlFlow<()>, WorkerError>,
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
    let base = &cfg.base;
    let deadline = base.deadline;
    let stages = base.timestepping.stages();
    let nstages = stages.len();
    let table = plans.root.table();
    if table.stages() as usize != nstages || table.substeps() as usize != RATIO || table.fine_stages() as usize != nstages
    {
        return Err(fail(
            fabric,
            WorkerError::Refused(format!(
                "the session's phase table holds {} root stages, {} substeps of {} stages; this worker \
                 runs {nstages} stages with {RATIO} substeps",
                table.stages(),
                table.substeps(),
                table.fine_stages()
            )),
        ));
    }
    for h in tiles.iter() {
        if h.levels.len() > 2 {
            return Err(fail(
                fabric,
                WorkerError::Refused("a hierarchy deeper than one fine level is outside the launched refinement".into()),
            ));
        }
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
        refuse(&root_stores, &root_kernels).map_err(|e| fail(fabric, e))?;
        let fine_stores: Vec<&FieldStore<D, DOF, Mem, f64>> = refined.iter().map(|&k| &*tiles[k].levels[1].state).collect();
        let fine_kernels: Vec<&K> = refined.iter().map(|&k| &tiles[k].levels[1].kernels).collect();
        refuse(&fine_stores, &fine_kernels).map_err(|e| fail(fabric, e))?;
        if plans.root.plan().seam.is_some() {
            return Err(fail(fabric, WorkerError::Refused("a polar seam with refinement is outside the launched refinement".into())));
        }
        refuse_transported_registers(tiles).map_err(|e| fail(fabric, e))?;
        let magnetized = tiles.iter().any(|h| h.levels[0].state.fields.mhd.is_some());
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
        let forced = !base.injection.reject_at.is_empty() || tiles.iter().any(|h| h.forced_fine_rejection_pending());
        if forced {
            for h in tiles.iter() {
                for level in &h.levels {
                    if !level.kernels.rollback_snapshot_is_complete(&level.state) {
                        return Err(fail(
                            fabric,
                            WorkerError::Refused(
                                "a forced rejection on a kernel set without a complete rollback snapshot is \
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
                with_device(fine_dev(k), || h.levels[1].kernels.ghost_fill(&h.levels[1].state));
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
                with_device(fine_dev(k), || h.levels[1].kernels.ghost_fill(&h.levels[1].state));
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
            let mut parts = vec![PlanPart { plan: &plans.fine, fields: &fields, devices: held.fine_devices, fault: None, dst_fields: None }];
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
                with_device(fine_dev(k), || h.levels[1].kernels.ghost_fill(&h.levels[1].state));
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
            if plans.root.has_audits() || plans.fine.has_audits() {
                let root_lists = root_fields!();
                let fine_lists = fine_fields!();
                let cost = audit_plans(&[(&plans.root, &root_lists[..]), (&plans.fine, &fine_lists[..])], fabric, $point, $step, $attempt, deadline)
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
    // audit, then the fine level seeded by prolongation from the exchanged root
    {
        for (k, h) in tiles.iter().enumerate() {
            with_device(root_dev(k), || {
                h.levels[0].kernels.c2p(&h.levels[0].state);
                h.levels[0].kernels.ghost_fill(&h.levels[0].state);
            });
        }
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
                with_device(fine_dev(k), || tiles[k].levels[1].kernels.ghost_fill(&tiles[k].levels[1].state));
            }
        } else {
            for &k in &refined {
                with_device(fine_dev(k), || tiles[k].seed_fine_from_coarse())
                    .map_err(|e| fail(fabric, WorkerError::Numerics(format!("fine seed: {e:?}"))))?;
            }
            for h in tiles.iter_mut() {
                h.prime();
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
    while t < base.t_final && (base.max_steps == 0 || iter < base.max_steps) {
        // the global timestep: the minimum root cfl over the held hierarchies, then over the
        // workers, consumed as the coordinator's bits
        // every level's raw candidate is judged before the user clamp, which would replace a
        // NaN with the cap
        let mut local = f64::INFINITY;
        for (k, h) in tiles.iter().enumerate() {
            let mut raw = f64::INFINITY;
            for (ll, mut candidate) in with_device(root_dev(k), || h.level_cfl_candidates()).into_iter().enumerate() {
                if k == 0 && (base.injection.invalid_cfl_at == Some(iter) && ll == 0 || cfg.invalid_level_cfl_at == Some((iter, ll))) {
                    candidate = f64::NAN;
                }
                if !candidate.is_finite() || candidate <= 0.0 {
                    return Err(fail(
                        fabric,
                        WorkerError::Numerics(format!(
                            "invalid CFL candidate {candidate:e} on level {ll} of tile {:?} at iter {iter} (time {t:.4e})",
                            held.root_ids[k]
                        )),
                    ));
                }
                raw = raw.min(candidate);
            }
            local = local.min(h.clamp_dt(raw));
        }
        let clock = Instant::now();
        let global = fabric
            .collective(OpKind::Min, local.to_bits(), deadline)
            .map_err(|e| fail(fabric, e.into()))?;
        timing.collectives += clock.elapsed();
        let mut dt = select_timestep([f64::from_bits(global)], base.t_final - t, iter, t)
            .map_err(|e| fail(fabric, WorkerError::Numerics(e.detail)))?;
        if let Some((at, bits)) = base.injection.cap_dt_at {
            if at == iter {
                dt = dt.min(f64::from_bits(bits));
            }
        }
        // the root step's candidate: the agreed timestep after its final-time clamp and caps
        let guard = tiles[0].levels[0].state.retry_guard;
        let mut ledger = symbi_sim::retry_guard::RetryLedger::new(dt);
        let mut attempt: u16 = 0;
        'attempt: loop {
            // the root-attempt snapshot of every level and its sidecars
            let sidecars: Vec<_> = tiles.iter().map(|h| h.snapshot_retry_sidecars()).collect();
            for (k, h) in tiles.iter_mut().enumerate() {
                for level in &h.levels {
                    if level.kernels.fofc_active() {
                        with_device(root_dev(k), || level.kernels.snapshot_retry(&level.state));
                    }
                }
                with_device(root_dev(k), || {
                    h.snapshot_attempt_entry();
                    h.snapshot_face_entry();
                    h.level_step_begin(0, dt);
                });
            }
            let mut rejected = false;
            // the root stages
            for ii in 0..nstages {
                let marks: Vec<StageMark> = (0..n)
                    .map(|k| with_device(root_dev(k), || tiles[k].level_stage_through_mark(0, ii, dt, 0.0)))
                    .collect();
                let mut troubled: u64 = marks.iter().map(|m| m.troubled).sum();
                if attempt == 0 {
                    if let Some((at_step, at_stage, cell)) = base.injection.trouble_at {
                        if (at_step, at_stage) == (iter, ii) {
                            let cell: [isize; D] = std::array::from_fn(|ax| cell[ax]);
                            for k in 0..n {
                                if let Some(local) = plans.root.local_cell(held.root_ids[k], cell) {
                                    *tiles[k].levels[0].state.workspace.fofc_flag.view_mut().at_mut(local) = 1.0;
                                    troubled += 1;
                                }
                            }
                        }
                    }
                }
                let cut_trouble = level_trouble(
                    tiles, &plans.root, held.root_ids, held.root_devices, 0, troubled,
                    ExchangePoint::Troubled(ii as u8), iter, attempt, transport, fabric, deadline, &mut timing,
                    &mut report.troubled_exchanges,
                )?;
                let mut retry = false;
                for k in 0..n {
                    report.neighbor_corrections += u64::from(cut_trouble[k] && marks[k].troubled == 0);
                    retry |= with_device(root_dev(k), || {
                        tiles[k].level_stage_from_correction(0, ii, dt, 0.0, cut_trouble[k], marks[k].receipt_start)
                    });
                }
                if attempt == 0 && base.injection.reject_at.contains(&(iter, ii)) {
                    retry = true;
                }
                if vote_retry(fabric, retry, deadline, &mut timing)? {
                    rejected = true;
                    break;
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
                } else {
                    root_point!(ExchangePoint::Stage(ii as u8), iter, attempt);
                }
                if base.audit_every_point {
                    audit!(plans.root, root_fields!(), ExchangePoint::Audit(AuditAt::Stage(ii as u8)), iter, attempt);
                }
            }
            if !rejected {
                for (k, h) in tiles.iter().enumerate() {
                    with_device(root_dev(k), || {
                        h.level_tail_emf(0, dt);
                        h.level_tail_excise(0);
                    });
                }
                // the fine subcycle: every worker walks its phases, a worker without fine
                // tiles with no local work
                let fine_dt = dt / RATIO as f64;
                'subcycle: for sub in 0..RATIO {
                    let alpha = sub as f64 / RATIO as f64;
                    for &k in &refined {
                        with_device(fine_dev(k), || {
                            tiles[k].prolong_cf(1, alpha);
                            tiles[k].levels[1].kernels.ghost_fill(&tiles[k].levels[1].state);
                        });
                    }
                    fine_point!(ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Prolong }, iter, attempt);
                    for &k in &refined {
                        with_device(fine_dev(k), || tiles[k].level_step_begin(1, fine_dt));
                    }
                    for jj in 0..nstages {
                        let marks: Vec<StageMark> = refined
                            .iter()
                            .map(|&k| with_device(fine_dev(k), || tiles[k].level_stage_through_mark(1, jj, fine_dt, alpha)))
                            .collect();
                        let mut troubled: u64 = marks.iter().map(|m| m.troubled).sum();
                        let fine_ids: Vec<TileId> = refined.iter().map(|&k| held.fine_ids[k].unwrap()).collect();
                        if attempt == 0 {
                            if let Some((at_step, at_sub, at_stage, cell)) = cfg.fine_trouble_at {
                                if (at_step, at_sub, at_stage) == (iter, sub as u8, jj) {
                                    let cell: [isize; D] = std::array::from_fn(|ax| cell[ax]);
                                    for (r, &k) in refined.iter().enumerate() {
                                        if let Some(local) = plans.fine.local_cell(fine_ids[r], cell) {
                                            *tiles[k].levels[1].state.workspace.fofc_flag.view_mut().at_mut(local) = 1.0;
                                            troubled += 1;
                                        }
                                    }
                                }
                            }
                        }
                        let cut_trouble = fine_level_trouble(
                            tiles, &refined, &plans.fine, &fine_ids, held.fine_devices, troubled,
                            ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Troubled(jj as u8) },
                            iter, attempt, transport, fabric, deadline, &mut timing, &mut report.fine_troubled_exchanges,
                        )?;
                        let mut retry = false;
                        for (r, &k) in refined.iter().enumerate() {
                            report.fine_neighbor_corrections += u64::from(cut_trouble[r] && marks[r].troubled == 0);
                            retry |= with_device(fine_dev(k), || {
                                tiles[k].level_stage_from_correction(1, jj, fine_dt, alpha, cut_trouble[r], marks[r].receipt_start)
                            });
                        }
                        if vote_retry(fabric, retry, deadline, &mut timing)? {
                            rejected = true;
                            break 'subcycle;
                        }
                        fine_stage_point!(sub, jj, iter, attempt);
                        if base.audit_every_point {
                            audit!(plans.fine, fine_fields!(), ExchangePoint::Fine { substep: sub as u8, point: FinePoint::Audit(jj as u8) }, iter, attempt);
                        }
                    }
                    // the fine tail on the globally finest level, then its recovery and halo
                    // refresh at the post-tail point before any later reader
                    let mut retry = false;
                    for &k in &refined {
                        retry |= with_device(fine_dev(k), || tiles[k].level_step_tail(1, fine_dt, alpha));
                    }
                    if vote_retry(fabric, retry, deadline, &mut timing)? {
                        rejected = true;
                        break 'subcycle;
                    }
                    // the last substep: the refined tiles restrict and reflux before the
                    // post-tail exchange, so the staggered faces it carries to the neighbors
                    // across the cuts are the corrected ones
                    if sub + 1 == RATIO {
                        for &k in &refined {
                            with_device(root_dev(k), || tiles[k].level_restrict_reflux(0, 0.0));
                        }
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
                let beside: Vec<usize> = (0..n).filter(|&k| tiles[k].levels.len() == 1 && tiles[k].has_flux_register(0)).collect();
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
                root_point!(ExchangePoint::Restrict, iter, attempt);
                if base.audit_every_point {
                    audit!(plans.root, root_fields!(), ExchangePoint::Audit(AuditAt::Restrict), iter, attempt);
                }
                for (k, h) in tiles.iter_mut().enumerate() {
                    with_device(root_dev(k), || {
                        let s = &mut h.levels[0].state;
                        (s.time, s.iteration) = advance_clock(s.time, s.iteration, dt);
                    });
                }
                symbi_sim::guard_ledger::step_commit();
                break 'attempt;
            }
            // the collective rollback: every level of every tile returns to the root-attempt
            // snapshot, both levels' halos are refreshed at the rollback point, the step
            // replays at a halved timestep
            symbi_sim::guard_ledger::step_discard();
            report.rejections += 1;
            // every worker books the agreed rejection with the evidence its own tiles hold, a fine
            // substep's included, so every worker reaches the guard's decision at the same attempt
            {
                let mut evidence = Vec::new();
                for (k, h) in tiles.iter().enumerate() {
                    for (ll, level) in h.levels.iter().enumerate() {
                        let tile = if ll == 0 { held.root_ids[k].0 } else { held.fine_ids[k].map_or(u32::MAX, |t| t.0) };
                        let site = format!("worker {}, tile {tile}, level {ll}", fabric.me().0);
                        evidence.extend(symbi_sim::retry_guard::take_evidence(&level.state, &site, ledger.rejections()));
                    }
                }
                if let Err(failure) = ledger.reject(&guard, dt, evidence) {
                    return Err(symbi_sim::worker::guard_failure(fabric, failure, &base.injection, deadline));
                }
            }
            for (k, h) in tiles.iter_mut().enumerate() {
                for level in &h.levels {
                    if !level.kernels.rollback_snapshot_is_complete(&level.state) {
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
                let parts = [
                    PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None, dst_fields: None },
                    PlanPart { plan: &plans.fine, fields: &fine, devices: held.fine_devices, fault: None, dst_fields: None },
                ];
                exchange_point_plans(&parts, transport, fabric, ExchangePoint::Rollback, iter, attempt, deadline, &mut timing.exchange)
                    .map_err(|e| fail(fabric, e.into()))?;
            }
            for (k, h) in tiles.iter().enumerate() {
                with_device(root_dev(k), || h.levels[0].kernels.ghost_fill(&h.levels[0].state));
                if h.levels.len() > 1 {
                    with_device(root_dev(k), || h.levels[1].kernels.ghost_fill(&h.levels[1].state));
                }
            }
            if base.audit_every_point {
                audit!(plans.root, root_fields!(), ExchangePoint::Audit(AuditAt::Rollback), iter, attempt);
            }
            dt = retry_timestep(dt, t).map_err(|e| fail(fabric, WorkerError::Numerics(e.detail)))?;
            attempt += 1;
        }
        // the sustained-reduction guard, once per accepted root step, before any checkpoint
        // publishes it; the agreed timesteps give every worker the same decision
        {
            let mut streak = tiles[0].levels[0].state.severe_streak;
            if let Err(failure) = symbi_sim::retry_guard::accept(&guard, &mut streak, &ledger, dt) {
                return Err(symbi_sim::worker::guard_failure(fabric, failure, &base.injection, deadline));
            }
            for h in tiles.iter_mut() {
                h.levels[0].state.severe_streak = streak;
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

/// the collective retry vote of one stage: true when any worker rejects
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

/// the troubled marks of the root level across the cuts: one vote for every tile this
/// worker holds, the flag exchange when any worker is troubled, and per tile whether a cut
/// ghost received a flag (a clean tile then corrects at the cut)
#[allow(clippy::too_many_arguments)]
fn level_trouble<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
    plan: &PlanExchange<'_, D>,
    ids: &[TileId],
    devices: &[i32],
    level: usize,
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
    let mut cut_trouble = vec![false; tiles.len()];
    if any == 0 {
        return Ok(cut_trouble);
    }
    *exchanges += 1;
    let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..plan.plan().n_tiles()).map(|_| Vec::new()).collect();
    for (k, h) in tiles.iter().enumerate() {
        lists[ids[k].0 as usize] = plan_fields(&h.levels[level].state);
    }
    let part = PlanPart { plan, fields: &lists, devices, fault: None, dst_fields: None };
    exchange_point_plans(&[part], transport, fabric, point, step, attempt, deadline, &mut timing.exchange)
        .map_err(|e| fail(fabric, e.into()))?;
    for (k, h) in tiles.iter().enumerate() {
        let l = &h.levels[level];
        cut_trouble[k] = with_device(devices[ids[k].0 as usize], || {
            plan.trouble_regions(ids[k]).iter().any(|region| l.kernels.fofc_flags_in(&l.state, region) > 0)
        });
    }
    Ok(cut_trouble)
}

/// `level_trouble` for the fine level, over the refined tiles alone
#[allow(clippy::too_many_arguments)]
fn fine_level_trouble<R, const D: usize, const DOF: usize, M, E, S, Mem, K, T, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
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
    let mut cut_trouble = vec![false; refined.len()];
    if any == 0 {
        return Ok(cut_trouble);
    }
    *exchanges += 1;
    let mut lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..plan.plan().n_tiles()).map(|_| Vec::new()).collect();
    for (r, &k) in refined.iter().enumerate() {
        lists[fine_ids[r].0 as usize] = plan_fields(&tiles[k].levels[1].state);
    }
    let part = PlanPart { plan, fields: &lists, devices, fault: None, dst_fields: None };
    exchange_point_plans(&[part], transport, fabric, point, step, attempt, deadline, &mut timing.exchange)
        .map_err(|e| fail(fabric, e.into()))?;
    for (r, &k) in refined.iter().enumerate() {
        let l = &tiles[k].levels[1];
        cut_trouble[r] = with_device(devices[fine_ids[r].0 as usize], || {
            plan.trouble_regions(fine_ids[r]).iter().any(|region| l.kernels.fofc_flags_in(&l.state, region) > 0)
        });
    }
    Ok(cut_trouble)
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
        }
    }
}

/// write one checkpoint of both levels across every worker after `accepted` accepted steps:
/// the root's shared faces are audited at the checkpoint epoch, the fine level's clock is
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
    if (plans.root.has_audits() || plans.fine.has_audits()) && accepted > 0 {
        let mut root_lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_root).map(|_| Vec::new()).collect();
        let mut fine_lists: Vec<Vec<&Field<f64, D, Mem>>> = (0..held.n_fine).map(|_| Vec::new()).collect();
        for (k, h) in tiles.iter().enumerate() {
            root_lists[held.root_ids[k].0 as usize] = plan_fields(&h.levels[0].state);
            if let Some(id) = held.fine_ids[k] {
                fine_lists[id.0 as usize] = plan_fields(&h.levels[1].state);
            }
        }
        audit_plans(
            &[(&plans.root, &root_lists[..]), (&plans.fine, &fine_lists[..])],
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
    let (fine_time, fine_dt, fine_iteration) = agree_fine_clock(tiles, fabric, req.deadline)?;
    let root_tiles: Vec<TileView<'_, D, DOF, Mem>> = tiles
        .iter()
        .zip(held.root_ids)
        .map(|(h, id)| TileView { state: &h.levels[0].state, offset: plans.root.plan().tiles[id.0 as usize].offset })
        .collect();
    let mut fine_ids = Vec::new();
    let fine_tiles: Vec<TileView<'_, D, DOF, Mem>> = tiles
        .iter()
        .zip(held.fine_ids)
        .filter_map(|(h, id)| id.map(|id| (h, id)))
        .map(|(h, id)| {
            fine_ids.push(id);
            let layout = &plans.fine.plan().tiles[id.0 as usize];
            TileView {
                state: &h.levels[1].state,
                offset: std::array::from_fn(|a| grids.fine.interior_lo[a] + layout.offset[a]),
            }
        })
        .collect();
    let levels = [
        LevelTiles { tiles: root_tiles, grid: grids.root.clone(), header: None },
        LevelTiles {
            tiles: fine_tiles,
            grid: grids.fine.clone(),
            header: Some(LevelHeader {
                template: &tiles[0].levels[0].state,
                time: fine_time,
                dt: fine_dt,
                iteration: fine_iteration,
            }),
        },
    ];
    let level_plans = [
        LevelPlan { plan: plans.root.plan(), placement: plans.root.placement(), origin: [0; D], tile_ids: held.root_ids },
        LevelPlan { plan: plans.fine.plan(), placement: plans.fine.placement(), origin: grids.fine.interior_lo, tile_ids: &fine_ids },
    ];
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
    let stored = checkpoint_level_count(path).map_err(err)?;
    if stored != 2 {
        return Err(WorkerError::Refused(format!(
            "{path} carries {stored} level(s); a refined session restarts from a checkpoint of its two levels"
        )));
    }
    for (k, h) in tiles.iter_mut().enumerate() {
        let root_offset = plans.root.plan().tiles[held.root_ids[k].0 as usize].offset;
        load_partitioned_level(&mut h.levels[0].state, path, 0, root_offset, &grids.root).map_err(err)?;
        if let Some(id) = held.fine_ids[k] {
            let layout = &plans.fine.plan().tiles[id.0 as usize];
            let offset: [isize; D] = std::array::from_fn(|a| grids.fine.interior_lo[a] + layout.offset[a]);
            load_partitioned_level(&mut h.levels[1].state, path, 1, offset, &grids.fine).map_err(err)?;
        }
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
            } else if tile_is_domain_edge {
                BoundaryType::Outflow
            } else {
                // the patch's real face on a cut: a coarse-fine side
                BoundaryType::CoarseFine
            }
        };
        [
            side(lo[a] == ext[a].0, tc[a] == 0, lo[a] > patch[a].0),
            side(hi[a] == ext[a].0 + ext[a].1, tc[a] + 1 == counts[a], hi[a] < patch[a].1),
        ]
    }));
    Some(super::transfer::face_stencil_boxes(&interior, &boundaries))
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

/// the fine level's clock, agreed by every worker and verified: each worker's fine tiles must
/// share one clock; the words are reduced over the workers holding fine tiles, each 64-bit word
/// as its two 32-bit halves sent as the exact integers one above them, since the minimum
/// reduces over positive finite numbers (a worker holding no fine tile contributes the largest
/// finite number); every holder compares the agreed words with its own, and one vote refuses
/// the checkpoint on any disagreement, or when no worker holds a fine tile
fn agree_fine_clock<R, const D: usize, const DOF: usize, M, E, S, Mem, K, L>(
    tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>],
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
        .filter(|h| h.levels.len() > 1)
        .map(|h| {
            let s = &h.levels[1].state;
            [s.time.to_bits(), s.dt.to_bits(), s.iteration]
        })
        .collect();
    let mine = clocks.first().copied();
    let locally_agreed = clocks.iter().all(|c| Some(*c) == mine);
    let holders = fabric.collective(OpKind::Any, u64::from(mine.is_some()), deadline)?;
    if holders == 0 {
        return Err(WorkerError::Checkpoint("no worker holds a tile of the fine level; the checkpoint has no fine clock".into()));
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
            "the fine level's clock differs across its tiles (this worker's tiles agree: {locally_agreed}; \
             time, dt, iteration here {:?}, agreed {:?}); no checkpoint is written",
            mine.map(|c| (f64::from_bits(c[0]), f64::from_bits(c[1]), c[2])),
            (f64::from_bits(agreed[0]), f64::from_bits(agreed[1]), agreed[2]),
        )));
    }
    Ok((f64::from_bits(agreed[0]), f64::from_bits(agreed[1]), agreed[2]))
}
