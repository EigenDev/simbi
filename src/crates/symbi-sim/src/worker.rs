// =============================================================================
// worker.rs
//
// the evolution loop of one worker over the tiles it holds. the step is the
// decomposed step of `evolve_scheduled` with its three cross-tile operations
// carried by the fabric: the timestep is the fabric minimum of every worker's
// owned-tile candidate, a stage's rejection is the fabric any over every
// worker's outcome and is decided before that stage's halos move, and every
// halo exchange runs the compiled plan across workers under grants. the tile
// operations themselves are the same kernel calls in the same order, so the
// owned tiles of a worker match the single-process reference bitwise.
//
// the first release evolves single-level cartesian hydro: a store carrying
// bodies, tracers, mesh motion, magnetic fields, or excision is refused at
// entry. a local fatal error tells every peer through the abort relay before
// the loop returns it.
//
// usage:
//  let report = evolve_worker(&tiles, &mut stores, &kernels, &devices, &exchange,
//      &LocalCopy, &mut fabric, &cfg, |iter, t, owned, fabric| Ok(ControlFlow::Continue(())))?;
// =============================================================================

use crate::atlas::{ExchangePoint, TileId};
use crate::decomp::{HaloTransport, plan_fields};
use crate::driver::{
    advance_clock, advance_state_clock, downstream_injection_weight, needs_step_snapshot,
    retry_timestep, select_timestep, stage_schedule,
};
use crate::plan_exchange::PlanExchange;
use crate::stage::{StageArgs, StageOutcome, fold_stage};
use crate::state::{FieldStore, Timestepping};
use crate::substrate_seam::{KernelSet, RegimeKind};
use std::ops::ControlFlow;
use std::time::Duration;
use symbi_fabric::{Fabric, FabricError, Link, OpKind};
use symbi_grid::Field;
use symbi_xpu::{MemorySpace, with_device};

/// outcomes a test harness forces at a named step, outside the numerical path.
#[derive(Debug, Clone, Copy, Default)]
pub struct Injection {
    /// report a rejection from this worker at (step, stage) on the first attempt
    pub reject_at: Option<(u64, usize)>,
    /// report a NaN timestep candidate from this worker's first tile at this step, leaving
    /// its other tiles' candidates valid
    pub invalid_cfl_at: Option<u64>,
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
}

#[derive(Debug, Clone, Default)]
pub struct WorkerReport {
    pub steps: u64,
    pub rejections: u64,
    pub time: f64,
    /// the accepted timestep of every step, in order
    pub dt_sequence: Vec<f64>,
}

#[derive(Debug)]
pub enum WorkerError {
    Fabric(FabricError),
    Numerics(String),
    Refused(String),
    Checkpoint(String),
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerError::Fabric(e) => write!(f, "fabric: {e}"),
            WorkerError::Numerics(d) => write!(f, "numerics: {d}"),
            WorkerError::Refused(d) => write!(f, "refused: {d}"),
            WorkerError::Checkpoint(d) => write!(f, "checkpoint: {d}"),
        }
    }
}

impl std::error::Error for WorkerError {}

impl From<FabricError> for WorkerError {
    fn from(e: FabricError) -> Self {
        WorkerError::Fabric(e)
    }
}

/// the first-release scope: 2D cartesian adiabatic newtonian hydro in host memory, on a
/// static mesh, inviscid, without dye, bodies, tracers, magnetic fields, or excision.
fn refuse<const D: usize, const DOF: usize, M: MemorySpace, K: KernelSet<D, DOF, M, f64>>(
    regime: RegimeKind,
    stores: &[&FieldStore<D, DOF, M, f64>],
    kernels: &[&K],
) -> Result<(), WorkerError> {
    let refused = |reason: &str| {
        WorkerError::Refused(format!("{reason} is outside the distributed first release"))
    };
    if regime != RegimeKind::Newtonian {
        return Err(refused(&format!("the {regime:?} regime")));
    }
    if D != 2 {
        return Err(refused(&format!("a {D}-dimensional grid")));
    }
    if M::IS_DEVICE_ACCESSIBLE {
        return Err(refused("device memory"));
    }
    for (store, kernel) in stores.iter().zip(kernels) {
        let reason = if store.geom.coords != symbi_geometry::Geometry::Cartesian {
            Some(format!("the {:?} chart", store.geom.coords))
        } else if store.geom.spacetime != symbi_geometry::Spacetime::Minkowski {
            Some(format!("the {:?} spacetime", store.geom.spacetime))
        } else if store.fields.prim.pre.is_none() {
            Some("an isothermal closure".to_string())
        } else if store.has_passive_scalar()
            || store.fields.prim.chi.is_some()
            || store.fields.cons.chi.is_some()
        {
            Some("a passive scalar".to_string())
        } else if kernel.is_viscous() {
            Some("viscous transport".to_string())
        } else if store.immersed.is_some() {
            Some("an immersed body".to_string())
        } else if store.tracers.is_some() || store.continuous_tracers.is_some() {
            Some("a tracer population".to_string())
        } else if store.motion_law.is_some() {
            Some("mesh motion".to_string())
        } else if store.fields.mhd.is_some() {
            Some("a magnetic field".to_string())
        } else if kernel.excise_pass_count(store) > 0 {
            Some("horizon excision".to_string())
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
fn field_lists<'a, const D: usize, const DOF: usize, M: MemorySpace>(
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
fn exchange_point<const D: usize, const DOF: usize, M, T, L>(
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
) -> Result<(), FabricError>
where
    M: MemorySpace,
    T: HaloTransport,
    L: Link,
{
    let fields = field_lists(tiles, stores, n_tiles);
    for axis in 0..D {
        let epoch = PlanExchange::<D>::epoch(point, step, attempt, axis);
        exchange.open_axis(&fields, devices, transport, fabric, epoch)?;
        fabric.wait(deadline)?;
        exchange.finish_axis(&fields, fabric, axis)?;
    }
    Ok(())
}

/// a local fatal error: every peer is told before it is returned.
fn fail<L: Link>(fabric: &mut Fabric<L>, err: WorkerError) -> WorkerError {
    match &err {
        WorkerError::Fabric(FabricError::PeerAborted { .. } | FabricError::Disconnected { .. }) => {
        }
        other => fabric.abort(&other.to_string()),
    }
    err
}

/// evolve this worker's tiles from `cfg.start_time` until `cfg.t_final` or
/// `cfg.max_steps`. `tiles`, `stores`, `kernels` are parallel over the held
/// tiles; `devices` is indexed by tile id over the whole plan. `on_step` runs
/// after every accepted step with the held stores and the fabric, so a
/// checkpoint runs there; it may stop the run, and its error reaches every
/// peer through the abort relay.
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
    mut on_step: F,
) -> Result<WorkerReport, WorkerError>
where
    M: MemorySpace,
    K: KernelSet<D, DOF, M, f64>,
    T: HaloTransport,
    L: Link,
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
        refuse(cfg.regime, &sh, kernels).map_err(|e| fail(fabric, e))?;
    }
    let deadline = cfg.deadline;
    let stages = cfg.timestepping.stages();
    let multistage = needs_step_snapshot(stages);
    let schedule = stage_schedule(stages);

    // prime: primitives and physical ghosts, the cut halos, the physical ghosts again at the
    // cut corners, then the initial-condition check.
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
        )
        .map_err(|e| fail(fabric, e.into()))?;
        for k in 0..n {
            with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
        }
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

    let _guard_scope = crate::guard_ledger::open_scope();
    let mut report = WorkerReport::default();
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
        let global = fabric
            .collective(OpKind::Min, local.to_bits(), deadline)
            .map_err(|e| fail(fabric, e.into()))?;
        let mut dt = select_timestep([f64::from_bits(global)], cfg.t_final - t, iter, t)
            .map_err(|e| fail(fabric, WorkerError::Numerics(e.detail)))?;
        let mut attempt: u16 = 0;
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
            }
            let mut rejected = false;
            for stage in &schedule {
                let sh = shared!();
                let mut retry = false;
                for k in 0..n {
                    let outcome = with_device(dev(k), || {
                        fold_stage(
                            sh[k],
                            kernels[k],
                            StageArgs {
                                dt,
                                a0: stage.a0,
                                ac: stage.ac,
                                stage: stage.index,
                                n_stages: stages.len(),
                                injection_weight: downstream_injection_weight(stages, stage.index),
                                allow_elision: false,
                            },
                            &mut |_| {},
                        )
                    });
                    retry |= outcome == StageOutcome::RetryStep;
                }
                if attempt == 0 && cfg.injection.reject_at == Some((iter, stage.index)) {
                    retry = true;
                }
                // the rejection is decided before this stage's halos move, so no halo is
                // computed from a rejected state.
                let reject = fabric
                    .collective(OpKind::Any, u64::from(retry), deadline)
                    .map_err(|e| fail(fabric, e.into()))?;
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
                )
                .map_err(|e| fail(fabric, e.into()))?;
                for k in 0..n {
                    with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
                }
            }
            if !rejected {
                crate::guard_ledger::step_commit();
                break;
            }
            crate::guard_ledger::step_discard();
            report.rejections += 1;
            {
                let sh = shared!();
                for k in 0..n {
                    if !kernels[k].fofc_active() {
                        return Err(fail(
                            fabric,
                            WorkerError::Numerics(
                                "a step was rejected on a kernel set without a retry snapshot"
                                    .into(),
                            ),
                        ));
                    }
                    with_device(dev(k), || kernels[k].restore_step(sh[k]));
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
                )
                .map_err(|e| fail(fabric, e.into()))?;
                for k in 0..n {
                    with_device(dev(k), || kernels[k].ghost_fill(sh[k]));
                }
            }
            dt =
                retry_timestep(dt, t).map_err(|e| fail(fabric, WorkerError::Numerics(e.detail)))?;
            attempt += 1;
        }
        {
            let sh = shared!();
            for k in 0..n {
                with_device(dev(k), || kernels[k].viscous(sh[k], dt));
            }
        }
        for s in stores.iter_mut() {
            advance_state_clock(&mut **s, dt);
        }
        (t, iter) = advance_clock(t, iter, dt);
        report.steps = iter;
        report.time = t;
        report.dt_sequence.push(dt);
        let sh = shared!();
        match on_step(iter, t, &sh, fabric) {
            Ok(ControlFlow::Continue(())) => {}
            Ok(ControlFlow::Break(())) => break,
            Err(e) => return Err(fail(fabric, e)),
        }
    }
    fabric
        .finish(deadline)
        .map_err(|e| fail(fabric, e.into()))?;
    Ok(report)
}
