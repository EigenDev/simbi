// =============================================================================
// fofc.rs
//
// the first-order flux correction (FOFC) orchestration, shared by every substrate that
// opts in (the MHD regimes + the hydro regimes). after a high-order stage the c2p of some
// zones goes unphysical (or non-finite); FOFC redoes those zones with a first-order (PCM +
// robust HLLE/light-cone) update from the physical stage-input state and, per zone, keeps
// whichever tier is physical: high-order, else first-order, else the frozen stage input.
//
// GRMHD replaces the last two tiers with a CONSERVATIVE recovery (`SourceReplay`): the spliced
// first-order flux and edge EMF are kept single-valued, and the only thing clipped is the
// pointwise geometric (metric) source, scaled to the largest fraction of the source ray that
// stays inside the Wu & Tang admissible set. that leaves nothing to freeze — a state the
// source-free low-order operator cannot make admissible is a statement about the TIMESTEP, and
// the orchestrator reports it up so the driver rejects the step and replays at a smaller dt.
//
// usage:
//  fofc_orchestrate(sim, prefix, has_additive, first_order_flux, c2p, godunov, source_apply);
//  where the closures dispatch the substrate's own first-order flux / c2p / godunov / source.
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_grid::Field;
use symbi_ir::algebra::Scalar;
use symbi_sim::state::{ConsFieldsGeneric, FieldStore, PrimFieldsGeneric};
use symbi_xpu::MemorySpace;

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use crate::kernels::support::FaceDomain;
use crate::regimes::substrate_gpu::field_reduce;
use crate::regimes::substrate_kernels::{
    coord_suffix, dispatch_fields_each, dispatch_named, kernel_field_binds, resolve_body_scalars,
};
use symbi_ir::emit::ReductionOp;

// FOFC observability counters — the running totals of DELIBERATE fallback events over a run, so a
// run can surface how often/where the first-order flux correction and the last-resort freeze fired
// (a limiter is fine as long as it is visible). `FALLBACK_CELLS` counts flagged-cell x
// substage events (a cell whose high-order c2p was unphysical and took the first-order redo);
// `FREEZE_CELLS` counts the frozen-cell x substage events (neither order recovered it -> held at the
// stage input). read + reset by the driver at its benchmark cadence to show a per-window delta.
static FOFC_FALLBACK_CELLS: AtomicU64 = AtomicU64::new(0);
static FOFC_FREEZE_CELLS: AtomicU64 = AtomicU64::new(0);

// the horizon-region subtotals: events whose cell center lies inside the configured
// horizon radius. on a black-hole run everything inside r_+ is causally disconnected
// from the exterior, so fallbacks there (e.g. the ring at the metric-guard radius,
// where the clamped metric's source is discontinuous) are expected and harmless —
// the acceptance criterion for a production run is EXTERIOR events == 0, and that
// signal must not be drowned by the interior's steady, legitimate firing.
static FOFC_FALLBACK_CELLS_HORIZON: AtomicU64 = AtomicU64::new(0);
static FOFC_FREEZE_CELLS_HORIZON: AtomicU64 = AtomicU64::new(0);
// the horizon radius (f64 bits) for the region split; 0.0 = no split (flat runs,
// and charts without a euclidean cell-center radius).
static FOFC_HORIZON_RADIUS_BITS: AtomicU64 = AtomicU64::new(0);
static FOFC_DIAGNOSTIC_POSTED: AtomicBool = AtomicBool::new(false);

/// the cumulative (fallback-cell, freeze-cell) FOFC event totals since the last `fofc_reset_stats`.
/// each is a sum over substages of the per-substage flagged/frozen interior-cell count.
pub fn fofc_stats() -> (u64, u64) {
    (
        FOFC_FALLBACK_CELLS.load(Ordering::Relaxed),
        FOFC_FREEZE_CELLS.load(Ordering::Relaxed),
    )
}

/// the (fallback, freeze) subtotals whose cells lie INSIDE the configured horizon radius —
/// the causally disconnected region of a black-hole run. exterior counts are the totals
/// minus these. both zero when no horizon radius is configured.
pub fn fofc_horizon_stats() -> (u64, u64) {
    (
        FOFC_FALLBACK_CELLS_HORIZON.load(Ordering::Relaxed),
        FOFC_FREEZE_CELLS_HORIZON.load(Ordering::Relaxed),
    )
}

/// configure the horizon radius for the FOFC region split: events at euclidean cell-center
/// radius |x| < r_h book into the horizon subtotals. CARTESIAN charts only (the euclidean
/// center radius is meaningless on an angular grid); 0 disables the split. host-memory runs
/// only — a device run keeps global counts (the masked count would force a device round-trip).
pub fn fofc_set_horizon_radius(r_h: f64) {
    FOFC_HORIZON_RADIUS_BITS.store(r_h.to_bits(), Ordering::Relaxed);
}

/// zero the fofc event counters (call at run start / after reading a window delta).
pub fn fofc_reset_stats() {
    FOFC_FALLBACK_CELLS.store(0, Ordering::Relaxed);
    FOFC_FREEZE_CELLS.store(0, Ordering::Relaxed);
    FOFC_FALLBACK_CELLS_HORIZON.store(0, Ordering::Relaxed);
    FOFC_FREEZE_CELLS_HORIZON.store(0, Ordering::Relaxed);
    FOFC_DIAGNOSTIC_POSTED.store(false, Ordering::Relaxed);
}

fn post_first_fallback<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    flag: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    if std::env::var_os("SIMBI_FOFC_DIAGNOSTICS").is_none()
        || Mem::IS_DEVICE_ACCESSIBLE
        || FOFC_DIAGNOSTIC_POSTED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    let fv = flag.view();
    let rho = sim.fields.prim.rho.view();
    let pre = sim.fields.prim.pre_field().map(Field::view);
    for c in sim.geom.interior.iter() {
        if (*fv.at(c)).to_f64() <= 0.5 {
            continue;
        }
        let x: [f64; D] =
            std::array::from_fn(|aa| sim.geom.x_lo[aa] + (c[aa] as f64 + 0.5) * sim.geom.dx[aa]);
        let radius = x.iter().map(|xx| xx * xx).sum::<f64>().sqrt();
        let pressure = pre.as_ref().map(|view| (*view.at(c)).to_f64());
        eprintln!(
            "fofc diagnostic: first fallback coord={c:?} x={x:?} r={radius:.9e} \
             rho={:.9e} pre={pressure:?}",
            (*rho.at(c)).to_f64(),
        );
        return;
    }
}

/// count the flagged cells (flag > 0.5) whose cell center |x| lies inside `r_h`, on a
/// uniform cartesian grid with origin `x_lo` and spacing `dx`. the boundary test is a
/// diagnostic split with no physics dependence — half-cell classification error at the
/// horizon is irrelevant (the interior fire sits deep inside, the exterior gate cares
/// about cells well outside).
pub fn horizon_flagged_count<const D: usize, Mem, Sc>(
    flag: &Field<Sc, D, Mem>,
    interior: &symbi_algebra::Domain<D>,
    x_lo: &[f64; D],
    dx: &[f64; D],
    r_h: f64,
) -> u64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let view = flag.view();
    let mut count = 0u64;
    for c in interior.iter() {
        let mut r2 = 0.0f64;
        for a in 0..D {
            let x = x_lo[a] + (c[a] as f64 + 0.5) * dx[a];
            r2 += x * x;
        }
        if r2 < r_h * r_h && (*view.at(c)).to_f64() > 0.5 {
            count += 1;
        }
    }
    count
}

/// book a per-substage flagged count into a horizon subtotal, when the split is active:
/// a configured radius, a cartesian chart, and host-resident memory. returns the number of
/// events booked (the count INSIDE the horizon), or 0 when no split applies — so the caller can
/// derive the exterior count as `total - returned` and treat a run with no configured horizon as
/// all-exterior (unchanged behavior).
fn book_horizon_events<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    flag: &Field<Sc, D, Mem>,
    counter: &AtomicU64,
) -> u64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let r_h = f64::from_bits(FOFC_HORIZON_RADIUS_BITS.load(Ordering::Relaxed));
    if r_h <= 0.0
        || Mem::IS_DEVICE_ACCESSIBLE
        || sim.geom.coords != symbi_geometry::Geometry::Cartesian
    {
        return 0;
    }
    let n = horizon_flagged_count(flag, &sim.geom.interior, &sim.geom.x_lo, &sim.geom.dx, r_h);
    counter.fetch_add(n, Ordering::Relaxed);
    n
}

/// consecutive substages the FOFC freeze tier may fire before the run halts. the freeze is the
/// last-resort MOOD parachute (neither high- nor first-order recovered a zone); it deploys rarely
/// and in ISOLATION for a genuine hard cell (the magnetized-torus inner-cliff pathology freezes on
/// ~0.5% of stages, never consecutively), but a poisoned source / initial datum that FOFC cannot fix
/// freezes EVERY stage. so a long consecutive streak is the honest "genuinely broken" fail-loud that
/// survives FOFC recovery, without false-halting the rare correct parachute. generous margin.
const FOFC_FREEZE_HALT_STREAK: u32 = 16;

fn advance_freeze_streak(freeze_streak: &AtomicU32, exterior_froze: u64) -> u32 {
    if exterior_froze > 0 {
        freeze_streak.fetch_add(1, Ordering::Relaxed) + 1
    } else {
        freeze_streak.store(0, Ordering::Relaxed);
        0
    }
}

/// resolve a FOFC copy/select slot name to its field: `den`/`mom_k`/`nrg` (conserved),
/// `rho`/`vel_k`/`pre` (primitive). regime-generic over the degrees of freedom `DOF`.
pub(crate) fn fofc_comp<'a, const D: usize, const DOF: usize, Mem, Sc>(
    cons: &'a ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim: &'a PrimFieldsGeneric<D, DOF, Mem, Sc>,
    name: &str,
) -> &'a Field<Sc, D, Mem>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    match name {
        "den" => &cons.den,
        "nrg" => cons.nrg_field().expect("fofc: energy field"),
        "rho" => &prim.rho,
        "pre" => prim.pre_field().expect("fofc: pressure field"),
        s if s.starts_with("mom_") => &cons.mom[s[4..].parse::<usize>().unwrap()],
        s if s.starts_with("vel_") => &prim.vel[s[4..].parse::<usize>().unwrap()],
        o => panic!("fofc_comp: unknown component '{o}'"),
    }
}

/// dispatch the componentwise copy kernel `{prefix}_fofc_{tag}_{D}d` (src `s_*` -> dst `d_*`),
/// over the whole allocated domain (ghosts too, so the first-order reconstruction has valid
/// ghost inputs).
pub(crate) fn fofc_copy<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dof_sfx: &str,
    tag: &str,
    src: (
        &ConsFieldsGeneric<D, DOF, Mem, Sc>,
        &PrimFieldsGeneric<D, DOF, Mem, Sc>,
    ),
    dst: (
        &ConsFieldsGeneric<D, DOF, Mem, Sc>,
        &PrimFieldsGeneric<D, DOF, Mem, Sc>,
    ),
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_{tag}{dof_sfx}_{D}d");
    let slot = |s: &str| -> &Field<Sc, D, Mem> {
        let comp = &s[2..]; // strip "s_" / "d_"
        if s.starts_with("s_") {
            fofc_comp(src.0, src.1, comp)
        } else {
            fofc_comp(dst.0, dst.1, comp)
        }
    };
    let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    for (bind, is_out) in kernel_field_binds(&name).iter() {
        let fld = slot(&bind.name());
        if *is_out {
            outputs.push(fld);
        } else {
            inputs.push(fld);
        }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.allocated, &inputs, &outputs, &[], &[]);
}

/// dispatch the FREEZE-tier select kernel `{prefix}_fofc_select_{D}d` over the interior: keep the
/// live spliced first-order conserved (`x_*` = the in-place cons/prim) where physical, else FREEZE to
/// the stage input (`us_*` = u_stage). the face-based splice already made every kept cell
/// conservative; this handles only the rare cell no flux can update admissibly (the documented
/// single-cell conservation waiver). only the conserved is chosen; the primitive is re-derived by the
/// c2p after the select.
pub fn fofc_select<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dof_sfx: &str,
    u_stage: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    cons: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim: &PrimFieldsGeneric<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_select{dof_sfx}_{D}d");
    let slot = |s: &str| -> &Field<Sc, D, Mem> {
        if let Some(c) = s.strip_prefix("us_") {
            // the stage-input freeze tier: conserved only (den/mom/nrg); the prim arg is unread.
            fofc_comp(u_stage, prim, c)
        } else if let Some(c) = s.strip_prefix("x_") {
            // the in-place cons: read (spliced first-order) + write (select result), one binding. the
            // prim (x_rho/x_pre) is read-only for the physicality test.
            fofc_comp(cons, prim, c)
        } else {
            panic!("fofc_select: unknown slot '{s}'")
        }
    };
    let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    for (bind, is_out) in kernel_field_binds(&name).iter() {
        let fld = slot(&bind.name());
        if *is_out {
            outputs.push(fld);
        } else {
            inputs.push(fld);
        }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.interior, &inputs, &outputs, &[], &[]);
}

/// the FREEZE-tier select with the immersed-body source composed INLINE: the `has_bodies` twin of
/// `fofc_select`. dispatches `{prefix}_fofc_select_with_body{coords}_{D}d`, whose freeze parachute is
/// the stage input EVOLVED by the body source (gravity + accretion) over `dt`, guarded to a physical
/// state — so a frozen cell near a body keeps its body-evolved gravity (the raw pre-body `u_stage` would lose it). field
/// binding is identical to `fofc_select` (us_* stage input, x_* live cons/prim); the body + grid
/// scalars are resolved by the kernel manifest exactly as for the standalone body source.
pub fn fofc_select_with_body<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dt: f64,
    gamma: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!(
        "{prefix}_fofc_select_with_body{}_{D}d",
        coord_suffix(sim.geom.coords)
    );
    let scalars = resolve_body_scalars(sim, dt, gamma, &name);
    let u_stage = sim.stage_input();
    let cons = &sim.fields.cons;
    let prim = &sim.fields.prim;
    let slot = |s: &str| -> &Field<Sc, D, Mem> {
        if let Some(c) = s.strip_prefix("us_") {
            fofc_comp(u_stage, prim, c)
        } else if let Some(c) = s.strip_prefix("x_") {
            fofc_comp(cons, prim, c)
        } else {
            panic!("fofc_select_with_body: unknown slot '{s}'")
        }
    };
    let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    for (bind, is_out) in kernel_field_binds(&name).iter() {
        let fld = slot(&bind.name());
        if *is_out {
            outputs.push(fld);
        } else {
            inputs.push(fld);
        }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.interior, &inputs, &outputs, &[], &scalars);
}

/// write the per-cell inadmissibility mask (1 where the current primitive state is unphysical)
/// over the interior into `bad`.
pub(crate) fn fofc_probe<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dof_sfx: &str,
    pre: &Field<Sc, D, Mem>,
    bad: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_probe{dof_sfx}_{D}d");
    dispatch_named(sim, pre, Some(bad), 0, &name, &sim.geom.interior, &[], &[]);
}

/// the number of set cells in a 0/1 interior mask.
pub(crate) fn fofc_flag_count<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    flag: &Field<Sc, D, Mem>,
) -> u64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    field_reduce(flag, &sim.geom.interior, ReductionOp::Add) as u64
}

/// dispatch the FACE-BASED FLUX SPLICE kernel `{prefix}_fofc_splice_{D}d_{dir}` over the axis-`dir`
/// interior face domain: on each face the live first-order flux (`fo_*` = `fields.flux[dir]`, spliced
/// in place) is kept where either adjacent cell is flagged, else replaced with the saved high-order
/// flux (`ho_*` = `flux_ho[dir]`). the flag is the 0/1 cell field. after all axes are spliced every
/// face carries ONE flux and the following godunov telescopes conservatively.
pub(crate) fn fofc_splice<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dof_sfx: &str,
    dir: usize,
    flag: &Field<Sc, D, Mem>,
    flux_ho: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_splice{dof_sfx}_{D}d_{dir}");
    let flux = &sim.fields.flux[dir];
    let prim = &sim.fields.prim;
    let slot = |s: &str| -> &Field<Sc, D, Mem> {
        if s == "flag" {
            flag
        } else if let Some(c) = s.strip_prefix("fo_") {
            // the live flux buffer: read (first-order) + write (spliced), one in-place binding.
            fofc_comp(flux, prim, c)
        } else if let Some(c) = s.strip_prefix("ho_") {
            fofc_comp(flux_ho, prim, c)
        } else {
            panic!("fofc_splice: unknown slot '{s}'")
        }
    };
    let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    for (bind, is_out) in kernel_field_binds(&name).iter() {
        let fld = slot(&bind.name());
        if *is_out {
            outputs.push(fld);
        } else {
            inputs.push(fld);
        }
    }
    dispatch_fields_each::<Sc, Mem, D>(
        &name,
        &sim.geom.interior.face_domain(dir),
        &inputs,
        &outputs,
        &[],
        &[],
    );
}

/// FOFC FREEZE DIAGNOSTIC: count the zones where the SPLICED first-order result is still unphysical —
/// the zones the freeze tier holds at the stage input (the conservation waiver). writes the per-zone
/// freeze flag to the scratch (`{prefix}_fofc_freeze{dof_sfx}_{D}d`) and max-reduces it over the
/// interior: > 0 iff some zone froze this substage. dispatched after the spliced-godunov c2p, so `x_*`
/// is the live conserved the select tests.
pub(crate) fn fofc_freeze_count<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dof_sfx: &str,
    scratch: &Field<Sc, D, Mem>,
    cons: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim: &PrimFieldsGeneric<D, DOF, Mem, Sc>,
) -> f64
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_freeze{dof_sfx}_{D}d");
    let slot = |s: &str| -> &Field<Sc, D, Mem> {
        if s == "freeze" {
            scratch
        } else if let Some(c) = s.strip_prefix("x_") {
            fofc_comp(cons, prim, c)
        } else {
            panic!("fofc_freeze_count: unknown slot '{s}'")
        }
    };
    let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    for (bind, is_out) in kernel_field_binds(&name).iter() {
        let fld = slot(&bind.name());
        if *is_out {
            outputs.push(fld);
        } else {
            inputs.push(fld);
        }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.interior, &inputs, &outputs, &[], &[]);
    // SUM: the freeze flag is 0/1, so this is the COUNT of frozen zones (> 0 iff any froze — the
    // caller's streak test and the observability tally both read it).
    field_reduce(scratch, &sim.geom.interior, ReductionOp::Add)
}

/// what the GRMHD conservative source replay did with the spliced first-order state. every other
/// regime reports `NotApplicable` and takes the shared redo.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceReplay {
    /// the regime has no source replay; the shared godunov / source / body redo completes the
    /// substage.
    NotApplicable,
    /// the replay completed the substage itself (godunov, CT re-sync, additive sources) with the
    /// geometric source scaled to the largest admissible fraction.
    Completed,
}

/// the constrained-transport hooks of the FOFC redo, carried as one value: four
/// same-signature closures traveled as positional arguments, so transposing two of
/// them typechecked and linked while corrupting `bcell` only on FOFC cells. hydro
/// regimes pass `CtHooks::none()`; MHD names each field at the call site. a flagged
/// cell needs FIRST-ORDER (diffused) B to recover, so the redo re-runs the CT with
/// the edge EMF SPLICED (HO off the fallback region, FO on it):
///  - `save`         : bflux -> bflux_ho + efield -> efield_ho (the HO induction flux +
///    edge EMF), before the FO flux redo overwrites them.
///  - `restore`      : bcell <- bcell_stage (the stage-input cell B, so the recomputed
///    EMF + the cell-B predictor read the correct base).
///  - `flux_and_curl`: AFTER the FO flux + gas splice, BEFORE the godunov — splice
///    bflux, recompute the FO edge EMF (Contact/HLL), splice the edge EMF (HO on
///    non-flagged edges, FO on flagged), restore bface <- bface_n and re-curl. gives
///    flagged cells FO B, leaves non-flagged B unchanged.
///  - `resync`       : bcell_from_bface (the patch-applying stages only) — re-interp
///    `bcell` from the re-curled face field + the small (FO-vs-FO) energy patch.
pub(crate) struct CtHooks<S1, S2, S3, S4>
where
    S1: Fn(),
    S2: Fn(),
    S3: Fn(),
    S4: Fn(),
{
    pub save: S1,
    pub restore: S2,
    pub flux_and_curl: S3,
    pub resync: S4,
}

fn ct_noop() {}

impl CtHooks<fn(), fn(), fn(), fn()> {
    /// the hydro case: no induction flux, no cell B, nothing to re-curl or re-sync.
    pub fn none() -> Self {
        CtHooks {
            save: ct_noop,
            restore: ct_noop,
            flux_and_curl: ct_noop,
            resync: ct_noop,
        }
    }
}

/// the FACE-BASED FOFC flow. (1) flag every zone whose high-order c2p is unphysical (the probe write
/// over the interior), boundary-fill the flag; early-out if none. (2) save the high-order fluxes.
/// (3) restore cons <- u_stage so the first-order flux reconstructs from the PHYSICAL stage input;
/// c2p. (4) re-flux at first order (the caller's `first_order_flux`). (5) SPLICE per face: keep the
/// first-order flux where either adjacent cell is flagged, else the saved high-order flux — one flux
/// per face. (6) the SINGLE conservative godunov from the spliced fluxes (+ additive source); c2p.
/// (7) the freeze tier holds the stage input on any cell the spliced update still left unphysical
/// (the documented single-cell conservation waiver); c2p to re-derive the primitive. because every
/// face carries ONE flux, the update telescopes exactly across every fallback boundary — the mass /
/// momentum / energy created at flag boundaries by a per-cell state replacement is gone. the caller
/// supplies the substrate's own first-order flux (per direction) / c2p / godunov / source_apply.
///
/// step (6) is where GRMHD diverges: `source_replay` completes the substage itself with the
/// geometric source limited per cell, and steps (7)'s freeze tier then only ever counts. returns
/// TRUE when the substage cannot be completed at this timestep at all and the caller must reject
/// and replay the whole step.
pub(crate) fn fofc_orchestrate<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    // the DOF-lift tag distinguishing the spherical-swirl (DOF > D) fofc kernels from the DOF == D
    // ones (both share the `_{D}d` grid tag); "" for DOF == D and for the always-3-vector MHD.
    dof_sfx: &str,
    has_additive: bool,
    scratch: &Field<Sc, D, Mem>,
    pre_bind: &Field<Sc, D, Mem>,
    freeze_streak: &AtomicU32,
    first_order_flux: impl Fn(usize),
    c2p: impl Fn(),
    godunov: impl Fn(),
    source_apply: impl Fn(),
    // the immersed-body source (gravity + accretion), re-applied in the redo exactly as the
    // high-order substage does it (hierarchy.rs: godunov -> additive -> body). the redo restores
    // `u_stage` (the stage input, BEFORE any source) and re-runs the godunov, so without this a
    // FO/freeze-selected cell near a body would lose its body source for the substage — precisely
    // the cells where fallback is most likely. self-gating (no-op when the sim has no bodies); a
    // no-op where the godunov already fuses the body (iso cartesian), so it never double-applies.
    body_apply: impl Fn(),
    // the ADMISSIBLE-BOUNDARY PROJECTION (GR-hydro): blend every spliced cell toward the admissible
    // stage-input anchor onto partial-G BEFORE the c2p, so the recovery succeeds on every cell and the
    // freeze tier below fires only on a genuinely inadmissible anchor. a no-op for regimes without a
    // baked projection (flat, MHD, iso), which keep the freeze parachute.
    project: impl Fn(),
    // the FREEZE-tier body parachute: `Some((dt_eff, gamma))` when the sim carries immersed bodies AND
    // the regime has a `_with_body` freeze-select kernel (adiabatic only). a frozen cell holds the
    // stage input `u_stage` (pre-body), so `body_apply` above — which targets the LIVE cons — never
    // reaches it; the with-body select instead evolves the parachute by the body source inline. `None`
    // falls back to the plain `fofc_select` (regimes without the kernel keep the pre-body freeze).
    body_freeze: Option<(f64, f64)>,
    // the MHD constrained-transport hooks (see `CtHooks`); `CtHooks::none()` for hydro.
    ct: CtHooks<impl Fn(), impl Fn(), impl Fn(), impl Fn()>,
    // the GRMHD conservative source replay, called once the single-valued gas flux and edge EMF are
    // spliced: it limits the (non-conservative) geometric source to the largest admissible fraction
    // of the source ray instead of blending whole cell states, so the flux and CT operators stay
    // untouched. see `SourceReplay`. every other regime reports `NotApplicable`.
    source_replay: impl Fn() -> SourceReplay,
    // whether an exterior cell the projection could not recover should REJECT the step (replay it
    // at a smaller dt) rather than freeze. true only where a projection exists to have tried and
    // failed first; a regime with no projection has no tier below the freeze and must keep it.
    retry_on_freeze: bool,
    // whether the caller must reject the whole step and replay it at a smaller timestep.
) -> bool
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let ws = &sim.workspace;
    let flag = &ws.fofc_flag;
    // HOST GATE + FLAG: write the per-cell fallback flag (1 where the high-order c2p is unphysical)
    // over the interior. FOFC only corrects a flagged cell (a physical one keeps its high-order flux
    // on every non-fallback face), so with none the whole pass is a no-op — skip it. a clean substage
    // costs one pointwise probe + a reduction, skipping the flux sweep + two extra c2p passes.
    let probe = format!("{prefix}_fofc_probe{dof_sfx}_{D}d");
    dispatch_named(
        sim,
        pre_bind,
        Some(flag),
        0,
        &probe,
        &sim.geom.interior,
        &[],
        &[],
    );
    // SUM (not max): the flag is 0/1, so the reduction is the COUNT of flagged cells — it doubles as
    // the fire/skip decision (count == 0 iff every high-order c2p was physical) AND the observability
    // tally, in a single pass.
    let fallback_cells = field_reduce(flag, &sim.geom.interior, ReductionOp::Add);
    if fallback_cells < 0.5 {
        advance_freeze_streak(freeze_streak, 0);
        return false;
    }
    FOFC_FALLBACK_CELLS.fetch_add(fallback_cells as u64, Ordering::Relaxed);
    post_first_fallback(sim, flag);
    // the horizon-region subtotal (inert without a configured split radius).
    book_horizon_events(sim, flag, &FOFC_FALLBACK_CELLS_HORIZON);
    // boundary-consistent ghosts for the flag: a face straddling the periodic wrap (or any boundary)
    // must take ONE first-order decision from both of its cells, else the splice re-creates the very
    // non-conservation it exists to remove.
    crate::regimes::mhd_substrate::flag_ghost_fill(
        sim,
        flag,
        crate::kernels::support::to_bc_array::<D>(&sim.boundaries),
    );

    let (cons, prim) = (&sim.fields.cons, &sim.fields.prim);
    // save the HIGH-ORDER fluxes before the first-order redo overwrites the live buffers. the
    // componentwise conserved copy is exactly `fofc_restore` (d_x = s_x), reused on the per-direction
    // flux ConsFields.
    for dir in 0..D {
        fofc_copy(
            sim,
            prefix,
            dof_sfx,
            "restore",
            (&sim.fields.flux[dir], prim),
            (&ws.flux_ho[dir], prim),
        );
    }
    (ct.save)(); // MHD: bflux -> bflux_ho + efield -> efield_ho (the HO induction flux + edge EMF)
    // the stage input via THE accessor: at the first stage of a multi-stage scheme
    // the driver elides the cons -> u_stage copy (u_n IS the stage input), so a
    // direct ws.u_stage read restores from a stale (first step: zeroed) buffer —
    // the redo then rebuilds the flow from garbage and the freeze tier both
    // fires where it should not and parachutes into zeros where it fires.
    fofc_copy(
        sim,
        prefix,
        dof_sfx,
        "restore",
        (sim.stage_input(), prim),
        (cons, prim),
    );
    (ct.restore)(); // MHD: bcell <- bcell_stage (stage-input cell B, the correct EMF/predictor base)
    c2p();
    for dir in 0..D {
        first_order_flux(dir);
    }
    for dir in 0..D {
        fofc_splice(sim, prefix, dof_sfx, dir, flag, &ws.flux_ho[dir]);
    }
    (ct.flux_and_curl)(); // MHD: splice bflux, recompute + splice the edge EMF, re-curl bface_n -> FO B on flagged
    match source_replay() {
        SourceReplay::Completed => {}
        SourceReplay::NotApplicable => {
            godunov();
            (ct.resync)(); // MHD: bcell_from_bface (patch stages) — re-interp bcell + the small FO-vs-FO patch
            if has_additive {
                source_apply();
            }
            body_apply(); // re-apply the immersed-body source on the redo (mirrors the HO godunov->additive->body order)
        }
    }
    // ADMISSIBLE-BOUNDARY PROJECTION — the tier BELOW every conservative correction above, and it
    // must sit below ALL of them: the flux splice and the source limiter both act on the OPERATOR
    // (a face flux, a source magnitude) and neither can move a cell that is already outside G. this
    // maps such a cell onto partial-G along the segment to the admissible stage-input anchor, so the
    // freeze tier afterwards counts only a genuinely inadmissible ANCHOR. exact passthrough on an
    // admissible cell, hence a no-op wherever the tiers above already succeeded. a no-op entirely
    // for regimes with no baked projection (flat MHD, iso), which keep the freeze parachute.
    project();
    c2p();
    // PERSISTENT-FREEZE FAIL-LOUD: the freeze tier holds the stage input where even full first-order
    // fluxes leave a cell unphysical. it is the rare correct parachute for a genuinely hard cell
    // (isolated in time), so a single firing is not a failure — but a poisoned source / initial datum
    // that FOFC cannot fix freezes EVERY stage. track the consecutive streak and halt loudly once it
    // persists.
    let froze = fofc_freeze_count(sim, prefix, dof_sfx, scratch, cons, prim);
    // the freeze-streak HALT gates on the EXTERIOR (r > r_+) freeze count only. a cell inside the
    // event horizon is causally disconnected from the exterior — no numerical signal crosses r_+
    // outward faster than the light cone the CFL bounds — so a persistent freeze there is expected
    // and harmless: the near-vacuum supersonic infall between the excision surface and the horizon
    // is the stiffest gas in the domain, and the acceptance criterion for a black-hole run is
    // EXTERIOR events == 0 (the small residual interior->exterior leak is separately bounded by the
    // excision-leakage gate). the panic exists to catch a poison spreading across the PHYSICAL
    // domain; charging it for causally-disconnected interior fiction halts a healthy run. a run with
    // no configured horizon (flat, angular chart, device) books zero interior events, so exterior ==
    // total and the halt is bit-unchanged.
    let interior_froze = if froze > 0.5 {
        FOFC_FREEZE_CELLS.fetch_add(froze as u64, Ordering::Relaxed);
        // the freeze flag lives in `scratch` after fofc_freeze_count.
        book_horizon_events(sim, scratch, &FOFC_FREEZE_CELLS_HORIZON)
    } else {
        0
    };
    let exterior_froze = (froze as u64).saturating_sub(interior_froze);
    let streak = advance_freeze_streak(freeze_streak, exterior_froze);
    // TIER 3, the last resort below the projection: an EXTERIOR cell that even the projection could
    // not land in G has an inadmissible ANCHOR — its stage input is already unrecoverable — so the
    // substage cannot be completed at this timestep. replaying the whole step at a smaller dt is
    // strictly better than the freeze below, which holds the cell at that same anchor and waives
    // its conservation. bounded by the SAME streak the halt uses, so a state no timestep can rescue
    // still fails loudly instead of halving forever.
    if exterior_froze > 0 && streak < FOFC_FREEZE_HALT_STREAK && retry_on_freeze {
        return true;
    }
    if streak >= FOFC_FREEZE_HALT_STREAK {
        let c2p_errors = symbi_sim::hydro_ops::scan_c2p_errors(sim);
        let first_error = symbi_sim::hydro_ops::first_c2p_error(sim)
            .map(|(coord, code)| format!("{coord:?}:{code}"))
            .unwrap_or_else(|| "unavailable".to_string());
        let first_state = symbi_sim::hydro_ops::first_c2p_failure_state(sim)
            .unwrap_or_else(|| "unavailable".to_string());
        panic!(
            "FOFC last-resort freeze fired on {streak} consecutive substages in the EXTERIOR (r > r_+, \
             regime={prefix}{dof_sfx}): a physical zone is persistently unrecoverable by the first-order \
             redo — a genuine breakdown, not the rare isolated parachute. projected c2p errors: \
             {c2p_errors}; first host error: {first_error}; first state: {first_state}. \
             check the source / initial \
             data / boundary for a poison. (freezes inside the horizon are expected and do not halt.)"
        );
    }
    match body_freeze {
        Some((dt_eff, gamma)) => fofc_select_with_body(sim, prefix, dt_eff, gamma),
        None => fofc_select(sim, prefix, dof_sfx, sim.stage_input(), cons, prim),
    }
    c2p();
    false
}

#[cfg(test)]
mod horizon_split_tests {
    use super::{FOFC_FREEZE_HALT_STREAK, advance_freeze_streak, horizon_flagged_count};
    use std::sync::atomic::{AtomicU32, Ordering};
    use symbi_algebra::{Domain, Space};
    use symbi_grid::Field;
    use symbi_xpu::HostMemory;

    #[test]
    fn clean_substage_breaks_the_persistent_freeze_streak() {
        let streak = AtomicU32::new(0);
        for _ in 0..FOFC_FREEZE_HALT_STREAK {
            assert_eq!(advance_freeze_streak(&streak, 1), 1);
            assert_eq!(advance_freeze_streak(&streak, 0), 0);
        }
        assert_eq!(streak.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn masked_count_splits_by_cell_center_radius() {
        // a 16x16 grid on (-2, 2)^2: flag one cell deep inside r_h = 1 and one far
        // outside; only the inside one books into the horizon subtotal.
        let n = 16isize;
        let dom = Domain::new([
            Space {
                name: "i",
                lo: 0,
                hi: n,
            },
            Space {
                name: "j",
                lo: 0,
                hi: n,
            },
        ]);
        let flag = Field::<f64, 2, HostMemory>::zeros(&dom).unwrap();
        let (x_lo, dx) = ([-2.0, -2.0], [0.25, 0.25]);
        // cell [8, 8] center = (0.125, 0.125), r ~ 0.18 < 1: inside.
        flag.set([8, 8], 1.0);
        // cell [15, 15] center = (1.875, 1.875), r ~ 2.65 > 1: outside.
        flag.set([15, 15], 1.0);
        assert_eq!(horizon_flagged_count(&flag, &dom, &x_lo, &dx, 1.0), 1);
        // no split radius -> nothing books.
        assert_eq!(horizon_flagged_count(&flag, &dom, &x_lo, &dx, 0.0), 0);
        // a radius covering the grid books both.
        assert_eq!(horizon_flagged_count(&flag, &dom, &x_lo, &dx, 10.0), 2);
    }
}
