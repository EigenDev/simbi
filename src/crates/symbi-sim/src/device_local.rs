// =============================================================================
// device_local.rs
//
// the explicit transfers between a host store and a store in another memory space, and the
// scope a store in device-local memory accepts. a device-local store is built as a mirror of a
// host store: every field allocated in the target space over its source's domain and filled by
// one synchronized upload; geometry, boundaries and clocks are host data and copy as values.
// the state returns to a host store by one synchronized download per field (a checkpoint) and
// goes back the same way (a restart).
//
// the device-local scope: one grid of flat cartesian hydrodynamics with an energy equation,
// outflow / periodic / reflecting faces, no refinement, magnetic field, passive scalar, source
// field, immersed bodies, tracers, censuses or mesh motion. a store outside it is refused with
// the reason.
//
// usage:
//   let mut device = mirror_sim::<_, _, _, _, _, _, _, HipDeviceLocal>(&host, 0)?;
//   evolve(&mut device, &kernels, t_end)?;
//   download_state(&device, &mut host)?;   // then write_checkpoint(&host, ..)
//   upload_state(&host, &device)?;         // after load_checkpoint(&mut host, ..)
// =============================================================================

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use symbi_geometry::Metric;
use symbi_grid::Field;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, Executor, MemorySpace, MemoryStorage};

use crate::state::{
    BoundaryType, ConsFieldsGeneric, Context, FieldStore, PartitionFieldsGeneric, PartitionGeometry, PrimFieldsGeneric,
    RkWorkspaceGeneric, SimStateGeneric,
};
use crate::substrate_seam::RetryExclusion;

/// why a store lies outside the device-local scope, or a transfer failed
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceLocalError(pub String);

impl std::fmt::Display for DeviceLocalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<symbi_xpu::XpuError> for DeviceLocalError {
    fn from(e: symbi_xpu::XpuError) -> Self {
        DeviceLocalError(format!("transfer failed: {e}"))
    }
}

type Result<T> = std::result::Result<T, DeviceLocalError>;

fn refuse(reason: &str) -> Result<()> {
    Err(DeviceLocalError(format!("device-local storage refuses {reason}")))
}

/// `Ok` when `store` lies inside the device-local scope, else the first feature that takes it out
pub fn check_scope<const D: usize, const DOF: usize, Mem: MemoryStorage>(store: &FieldStore<D, DOF, Mem, f64>) -> Result<()> {
    let geom = &store.geom;
    if geom.coords != symbi_geometry::Geometry::Cartesian {
        return refuse("a curvilinear chart");
    }
    if geom.spacetime != symbi_geometry::Spacetime::Minkowski {
        return refuse("a curved spacetime");
    }
    if geom.maps.is_some() {
        return refuse("a graded spacing map");
    }
    if geom.index_offset != [0; D] || geom.whole_span.is_some() {
        return refuse("a tile of a decomposed grid");
    }
    if DOF != D {
        return refuse("an out-of-plane velocity component");
    }
    let fields = &store.fields;
    if fields.mhd.is_some() {
        return refuse("a magnetic field");
    }
    if fields.cons.nrg.is_none() || fields.prim.pre.is_none() || fields.cs2.is_some() {
        return refuse("an isothermal closure");
    }
    if fields.cons.chi.is_some() || fields.prim.chi.is_some() {
        return refuse("a passive scalar");
    }
    if fields.source.is_some() {
        return refuse("a source field");
    }
    let ownership = &store.composite_ownership;
    if ownership.coverage.is_some() || ownership.inactive.is_some() || ownership.decision_excluded.is_some() {
        return refuse("a refined level");
    }
    if store.retry_exclusion != RetryExclusion::None {
        return refuse("a retry exclusion region");
    }
    if store.immersed.is_some() {
        return refuse("immersed bodies");
    }
    if store.tracers.is_some() || store.continuous_tracers.is_some() || store.ito_coefficients.is_some() {
        return refuse("tracers");
    }
    if !store.censuses.is_empty() {
        return refuse("a registered census");
    }
    if store.motion.a != 1.0 || store.motion.a_dot != 0.0 || store.motion_law.is_some() {
        return refuse("mesh motion");
    }
    for face in store.boundaries.0.iter().flatten() {
        if !matches!(face, BoundaryType::Outflow | BoundaryType::Periodic | BoundaryType::Reflect) {
            return refuse(&format!("a {face:?} boundary"));
        }
    }
    Ok(())
}

/// `src` copied into a fresh field of space `Dst` over the same domain
fn mirror_field<Src: MemoryStorage, Dst: MemoryStorage, const D: usize>(src: &Field<f64, D, Src>) -> Result<Field<f64, D, Dst>> {
    let dst = Field::<f64, D, Dst>::zeros(src.domain())?;
    dst.upload(&src.download()?)?;
    Ok(dst)
}

fn mirror_opt<Src: MemoryStorage, Dst: MemoryStorage, const D: usize>(
    src: &Option<Field<f64, D, Src>>,
) -> Result<Option<Field<f64, D, Dst>>> {
    src.as_ref().map(mirror_field).transpose()
}

fn mirror_array<Src: MemoryStorage, Dst: MemoryStorage, const D: usize, const N: usize>(
    src: &[Field<f64, D, Src>; N],
) -> Result<[Field<f64, D, Dst>; N]> {
    let fields = src.iter().map(mirror_field).collect::<Result<Vec<_>>>()?;
    Ok(fields.try_into().unwrap_or_else(|_| unreachable!()))
}

fn mirror_cons<Src: MemoryStorage, Dst: MemoryStorage, const D: usize, const DOF: usize>(
    src: &ConsFieldsGeneric<D, DOF, Src, f64>,
) -> Result<ConsFieldsGeneric<D, DOF, Dst, f64>> {
    Ok(ConsFieldsGeneric {
        den: mirror_field(&src.den)?,
        mom: mirror_array(&src.mom)?,
        nrg: mirror_opt(&src.nrg)?,
        chi: mirror_opt(&src.chi)?,
    })
}

fn mirror_prim<Src: MemoryStorage, Dst: MemoryStorage, const D: usize, const DOF: usize>(
    src: &PrimFieldsGeneric<D, DOF, Src, f64>,
) -> Result<PrimFieldsGeneric<D, DOF, Dst, f64>> {
    Ok(PrimFieldsGeneric {
        rho: mirror_field(&src.rho)?,
        vel: mirror_array(&src.vel)?,
        pre: mirror_opt(&src.pre)?,
        chi: mirror_opt(&src.chi)?,
    })
}

fn mirror_cons_array<Src: MemoryStorage, Dst: MemoryStorage, const D: usize, const DOF: usize>(
    src: &[ConsFieldsGeneric<D, DOF, Src, f64>; D],
) -> Result<[ConsFieldsGeneric<D, DOF, Dst, f64>; D]> {
    let groups = src.iter().map(mirror_cons).collect::<Result<Vec<_>>>()?;
    Ok(groups.try_into().unwrap_or_else(|_| unreachable!()))
}

fn copy_flag(src: &AtomicBool) -> AtomicBool {
    AtomicBool::new(src.load(Ordering::Relaxed))
}

/// `src` mirrored into space `Dst`: every field allocated in `Dst` and filled by a synchronized
/// upload, every host datum copied. refuses a store outside the device-local scope
pub fn mirror_store<Src: MemorySpace, Dst: MemoryStorage, const D: usize, const DOF: usize>(
    src: &FieldStore<D, DOF, Src, f64>,
) -> Result<FieldStore<D, DOF, Dst, f64>> {
    check_scope(src)?;
    let (f, w) = (&src.fields, &src.workspace);
    let fields = PartitionFieldsGeneric {
        cons: mirror_cons(&f.cons)?,
        prim: mirror_prim(&f.prim)?,
        flux: mirror_cons_array(&f.flux)?,
        c2p_error: mirror_field(&f.c2p_error)?,
        mhd: None,
        source: None,
        cs2: None,
    };
    let workspace = RkWorkspaceGeneric {
        u_n: mirror_cons(&w.u_n)?,
        prim_n: mirror_prim(&w.prim_n)?,
        u_stage: mirror_cons(&w.u_stage)?,
        stage_input_is_un: copy_flag(&w.stage_input_is_un),
        elide_stage_snapshot: copy_flag(&w.elide_stage_snapshot),
        fofc_cut_trouble: copy_flag(&w.fofc_cut_trouble),
        flux_reads_stage_faces: copy_flag(&w.flux_reads_stage_faces),
        fofc_marked: copy_flag(&w.fofc_marked),
        flux_ho: mirror_cons_array(&w.flux_ho)?,
        fofc_flag: mirror_field(&w.fofc_flag)?,
        freeze_applied: mirror_field(&w.freeze_applied)?,
        fofc_evidence: mirror_field(&w.fofc_evidence)?,
        fofc_inadmissible: mirror_field(&w.fofc_inadmissible)?,
        rejection_evidence: Mutex::new(Vec::new()),
        unrecoverable_once: Mutex::new(w.unrecoverable_once.lock().expect("the unrecoverable-cell slots").clone()),
        troubled_once: Mutex::new(*w.troubled_once.lock().expect("the troubled-cell slot")),
        inadmissible_once: Mutex::new(*w.inadmissible_once.lock().expect("the inadmissible-candidate slot")),
        body_scratch: std::sync::OnceLock::new(),
        #[cfg(debug_assertions)]
        stage_writes: Mutex::new(None),
        census_scratch: std::sync::OnceLock::new(),
    };
    let g = &src.geom;
    let geom = PartitionGeometry {
        dx: g.dx,
        x_lo: g.x_lo,
        allocated: g.allocated.clone(),
        interior: g.interior.clone(),
        ng: g.ng,
        coords: g.coords,
        spacetime: g.spacetime,
        spacetime_scalars: g.spacetime_scalars.clone(),
        axes: g.axes,
        maps: None,
        index_offset: g.index_offset,
        whole_span: None,
    };
    Ok(FieldStore {
        fields,
        workspace,
        geom,
        boundaries: src.boundaries,
        composite_ownership: src.composite_ownership.clone(),
        retry_exclusion: src.retry_exclusion,
        retry_guard: src.retry_guard,
        severe_streak: src.severe_streak,
        retry_guard_recorded: src.retry_guard_recorded,
        primitives_loaded: src.primitives_loaded,
        tracers: None,
        continuous_tracers: None,
        ito_coefficients: None,
        ito_transport: None,
        primitives_recovered: copy_flag(&src.primitives_recovered),
        stage_input_captured: copy_flag(&src.stage_input_captured),
        time: src.time,
        dt: src.dt,
        state_clock: AtomicU64::new(src.state_clock.load(Ordering::Relaxed)),
        max_dt: src.max_dt,
        iteration: src.iteration,
        cfl: src.cfl,
        timestepping: src.timestepping,
        motion: src.motion,
        motion_at_step_entry: src.motion_at_step_entry,
        motion_law: None,
        immersed: None,
        censuses: Vec::new(),
    })
}

/// a whole simulation state mirrored into space `Dst` (see [`mirror_store`]), its executor on
/// device `device_id`
pub fn mirror_sim<R, const D: usize, const DOF: usize, M, E, S, Src, Dst>(
    src: &SimStateGeneric<R, D, DOF, M, E, S, Src>,
    device_id: i64,
) -> Result<SimStateGeneric<R, D, DOF, M, E, S, Dst>>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy,
    E: Eos<f64> + Copy,
    S: ExecutionSpace,
    Src: MemorySpace,
    Dst: MemoryStorage,
{
    Ok(SimStateGeneric {
        store: mirror_store(&src.store)?,
        physics: src.physics,
        ctx: Context { exec: Executor::<S>::new(device_id)? },
    })
}

/// every field pair of the evolved state, conserved, primitive and recovery status: the fields a
/// checkpoint writes and a restart reads
fn state_pairs<'a, A: MemoryStorage, B: MemoryStorage, const D: usize, const DOF: usize>(
    a: &'a FieldStore<D, DOF, A, f64>,
    b: &'a FieldStore<D, DOF, B, f64>,
) -> Vec<(&'a Field<f64, D, A>, &'a Field<f64, D, B>)> {
    let (ca, cb) = (&a.fields.cons, &b.fields.cons);
    let (pa, pb) = (&a.fields.prim, &b.fields.prim);
    let mut pairs = vec![(&ca.den, &cb.den), (&pa.rho, &pb.rho), (&a.fields.c2p_error, &b.fields.c2p_error)];
    pairs.extend(ca.mom.iter().zip(cb.mom.iter()));
    pairs.extend(pa.vel.iter().zip(pb.vel.iter()));
    pairs.extend(ca.nrg.iter().zip(cb.nrg.iter()));
    pairs.extend(pa.pre.iter().zip(pb.pre.iter()));
    pairs
}

fn copy_clocks<A: MemoryStorage, B: MemoryStorage, const D: usize, const DOF: usize>(
    from: &FieldStore<D, DOF, A, f64>,
    to: &mut FieldStore<D, DOF, B, f64>,
) {
    to.time = from.time;
    to.dt = from.dt;
    to.iteration = from.iteration;
    to.severe_streak = from.severe_streak;
    to.primitives_loaded = from.primitives_loaded;
    to.primitives_recovered.store(from.primitives_recovered.load(Ordering::Relaxed), Ordering::Relaxed);
    to.state_clock.store(from.state_clock.load(Ordering::Relaxed), Ordering::Relaxed);
}

/// the evolved state of `device` and its clocks copied into `host`, one synchronized download per
/// field, for a checkpoint written from `host`
pub fn download_state<Dev: MemoryStorage, Host: MemorySpace, const D: usize, const DOF: usize>(
    device: &FieldStore<D, DOF, Dev, f64>,
    host: &mut FieldStore<D, DOF, Host, f64>,
) -> Result<()> {
    for (from, to) in state_pairs(device, host) {
        to.upload(&from.download()?)?;
    }
    copy_clocks(device, host);
    Ok(())
}

/// the evolved state of `host` and its clocks copied into `device`, one synchronized upload per
/// field, after a restart loaded `host` from a checkpoint
pub fn upload_state<Host: MemorySpace, Dev: MemoryStorage, const D: usize, const DOF: usize>(
    host: &FieldStore<D, DOF, Host, f64>,
    device: &mut FieldStore<D, DOF, Dev, f64>,
) -> Result<()> {
    check_scope(host)?;
    for (from, to) in state_pairs(host, device) {
        to.upload(&from.download()?)?;
    }
    copy_clocks(host, device);
    Ok(())
}

/// the bytes one call of [`download_state`] or [`upload_state`] moves for `store`
pub fn state_bytes<Mem: MemoryStorage, const D: usize, const DOF: usize>(store: &FieldStore<D, DOF, Mem, f64>) -> usize {
    state_pairs(store, store).iter().map(|(f, _)| f.domain().volume() * std::mem::size_of::<f64>()).sum()
}

fn cons_volume<Mem: MemoryStorage, const D: usize, const DOF: usize>(c: &ConsFieldsGeneric<D, DOF, Mem, f64>) -> usize {
    let mut fields = vec![&c.den];
    fields.extend(c.mom.iter());
    fields.extend(c.nrg.iter());
    fields.extend(c.chi.iter());
    fields.iter().map(|f| f.domain().volume()).sum()
}

fn prim_volume<Mem: MemoryStorage, const D: usize, const DOF: usize>(p: &PrimFieldsGeneric<D, DOF, Mem, f64>) -> usize {
    let mut fields = vec![&p.rho];
    fields.extend(p.vel.iter());
    fields.extend(p.pre.iter());
    fields.extend(p.chi.iter());
    fields.iter().map(|f| f.domain().volume()).sum()
}

/// the bytes [`mirror_store`] uploads for `store`: every field of the evolved state and the
/// step workspace
pub fn resident_bytes<Mem: MemoryStorage, const D: usize, const DOF: usize>(store: &FieldStore<D, DOF, Mem, f64>) -> usize {
    let (f, w) = (&store.fields, &store.workspace);
    let cells = cons_volume(&f.cons)
        + prim_volume(&f.prim)
        + f.flux.iter().map(cons_volume).sum::<usize>()
        + f.c2p_error.domain().volume()
        + cons_volume(&w.u_n)
        + prim_volume(&w.prim_n)
        + cons_volume(&w.u_stage)
        + w.flux_ho.iter().map(cons_volume).sum::<usize>()
        + [&w.fofc_flag, &w.freeze_applied, &w.fofc_evidence, &w.fofc_inadmissible]
            .iter()
            .map(|f| f.domain().volume())
            .sum::<usize>();
    cells * std::mem::size_of::<f64>()
}
