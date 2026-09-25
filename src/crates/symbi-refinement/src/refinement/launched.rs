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

use symbi_fabric::{Fabric, Link, OpKind, WorkerId};
use symbi_grid::Field;
use symbi_hydro::regime::Regime;
use symbi_sim::atlas::{AuditAt, ExchangePoint, FinePoint, TileId};
use symbi_sim::decomp::{HaloTransport, plan_cons_fields, plan_fields};
use symbi_sim::driver::{advance_clock, retry_timestep, select_timestep};
use symbi_sim::hydro_ops::scan_c2p_errors;
use symbi_sim::plan_exchange::{AuditError, PlanExchange, PlanRole, StagingFault};
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
}

/// evolve this worker's hierarchies from the configured start until the final time or
/// the step bound. `on_step` runs after every accepted step with the fabric.
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
    F: FnMut(u64, f64, &mut Fabric<L>) -> Result<ControlFlow<()>, WorkerError>,
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
        refuse(base.regime, &root_stores, &root_kernels).map_err(|e| fail(fabric, e))?;
        let fine_stores: Vec<&FieldStore<D, DOF, Mem, f64>> = refined.iter().map(|&k| &*tiles[k].levels[1].state).collect();
        let fine_kernels: Vec<&K> = refined.iter().map(|&k| &tiles[k].levels[1].kernels).collect();
        refuse(base.regime, &fine_stores, &fine_kernels).map_err(|e| fail(fabric, e))?;
        if plans.root.plan().seam.is_some() {
            return Err(fail(fabric, WorkerError::Refused("a polar seam with refinement is outside the launched refinement".into())));
        }
        if tiles.iter().any(|h| h.levels[0].state.fields.mhd.is_some()) {
            return Err(fail(
                fabric,
                WorkerError::Refused("a magnetized refined run is outside the launched refinement until its registers move".into()),
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
            let part = PlanPart { plan: &plans.root, fields: &fields, devices: held.root_devices, fault: None };
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
            let part = PlanPart { plan: &plans.fine, fields: &fields, devices: held.fine_devices, fault: None };
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
    // the fine stage point: the fine state and, with it, the flux contributions of the
    // stage; every contributing face is then cleared for the next stage and every landed
    // unit folded into its receiving face, in the order the register's owner accumulates
    macro_rules! fine_stage_point {
        ($sub:expr, $stage:expr, $step:expr, $attempt:expr) => {{
            let point = ExchangePoint::Fine { substep: $sub as u8, point: FinePoint::Stage($stage as u8) };
            let fields = fine_fields!();
            let flux_fields = flux_fields!();
            let mut parts = vec![PlanPart { plan: &plans.fine, fields: &fields, devices: held.fine_devices, fault: None }];
            if let Some(flux) = plans.flux.as_ref() {
                let fault = cfg.contribution_fault.and_then(|(at_step, at_sub, at_stage, fault)| {
                    ($attempt == 0 && (at_step, at_sub, at_stage) == ($step, $sub as u8, $stage)).then_some(fault)
                });
                parts.push(PlanPart { plan: flux, fields: &flux_fields, devices: held.root_devices, fault });
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
                PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None },
                PlanPart { plan: &plans.seed, fields: &cons, devices: held.root_devices, fault: None },
            ];
            exchange_point_plans(&parts, transport, fabric, ExchangePoint::Prime, 0, 0, deadline, &mut timing.exchange)
                .map_err(|e| fail(fabric, e.into()))?;
        }
        for (k, h) in tiles.iter().enumerate() {
            with_device(root_dev(k), || h.levels[0].kernels.ghost_fill(&h.levels[0].state));
        }
        audit!(plans.root, root_fields!(), ExchangePoint::Audit(AuditAt::Prime), 0, 0);
        for (k, h) in tiles.iter().enumerate() {
            let err = scan_c2p_errors(&h.levels[0].state);
            if err.is_err() {
                return Err(fail(
                    fabric,
                    WorkerError::Numerics(format!("c2p failed on initial conditions (tile {:?}): {err}", held.root_ids[k])),
                ));
            }
        }
        for &k in &refined {
            with_device(fine_dev(k), || tiles[k].seed_fine_from_coarse())
                .map_err(|e| fail(fabric, WorkerError::Numerics(format!("fine seed: {e:?}"))))?;
        }
        for h in tiles.iter_mut() {
            h.prime();
        }
    }

    let guard_scope = symbi_sim::guard_ledger::open_scope();
    let mut t = base.start_time;
    let mut iter: u64 = 0;
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
                root_point!(ExchangePoint::Stage(ii as u8), iter, attempt);
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
                    fine_point!(ExchangePoint::Fine { substep: sub as u8, point: FinePoint::PostTail }, iter, attempt);
                    if base.audit_every_point {
                        audit!(plans.fine, fine_fields!(), ExchangePoint::Fine { substep: sub as u8, point: FinePoint::AuditTail }, iter, attempt);
                    }
                }
            }
            if !rejected {
                // restriction on the refined tiles, the reflux on every tile whose register
                // holds a face: a receiving register corrects the cell beside a neighbor's patch
                for k in 0..n {
                    if tiles[k].levels.len() > 1 || tiles[k].has_flux_register(0) {
                        with_device(root_dev(k), || tiles[k].level_restrict_reflux(0, 0.0));
                    }
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
                with_device(root_dev(k), || h.restore_attempt_entry());
                h.restore_retry_sidecars(&sidecars[k]);
            }
            {
                let prims = root_fields!();
                let fine = fine_fields!();
                let parts = [
                    PlanPart { plan: &plans.root, fields: &prims, devices: held.root_devices, fault: None },
                    PlanPart { plan: &plans.fine, fields: &fine, devices: held.fine_devices, fault: None },
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
        (t, iter) = advance_clock(t, iter, dt);
        report.steps = iter;
        report.time = t;
        report.dt_sequence.push(dt);
        let clock = Instant::now();
        let flow = on_step(iter, t, fabric);
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
    let part = PlanPart { plan, fields: &lists, devices, fault: None };
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
    let part = PlanPart { plan, fields: &lists, devices, fault: None };
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
