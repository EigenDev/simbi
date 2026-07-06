// =============================================================================
// fofc.rs
//
// the first-order flux correction (FOFC) orchestration, shared by every substrate that
// opts in (the MHD regimes + the hydro regimes). after a high-order stage the c2p of some
// zones goes unphysical (or non-finite); FOFC redoes those zones with a first-order (PCM +
// robust HLLE/light-cone) update from the physical stage-input state and, per zone, keeps
// whichever tier is physical: high-order, else first-order, else the frozen stage input.
//
// usage:
//  fofc_orchestrate(sim, prefix, has_additive, first_order_flux, c2p, godunov, source_apply);
//  where the closures dispatch the substrate's own first-order flux / c2p / godunov / source.
// =============================================================================

use symbi_grid::Field;
use symbi_ir::algebra::Scalar;
use symbi_algebra::OrderedNumeric;
use symbi_xpu::MemorySpace;
use symbi_sim::state::{ConsFieldsGeneric, FieldStore, PrimFieldsGeneric};

use std::sync::atomic::{AtomicU32, Ordering};

use crate::regimes::substrate_kernels::{dispatch_fields_each, dispatch_named, kernel_field_binds};
use crate::regimes::substrate_gpu::field_max_reduce;
use crate::kernels::support::FaceDomain;

/// consecutive substages the FOFC freeze tier may fire before the run halts. the freeze is the
/// last-resort MOOD parachute (neither high- nor first-order recovered a zone); it deploys rarely
/// and in ISOLATION for a genuine hard cell (the magnetized-torus inner-cliff pathology freezes on
/// ~0.5% of stages, never consecutively), but a poisoned source / initial datum that FOFC cannot fix
/// freezes EVERY stage. so a long consecutive streak is the honest "genuinely broken" fail-loud that
/// survives FOFC recovery, without false-halting the rare correct parachute. generous margin.
const FOFC_FREEZE_HALT_STREAK: u32 = 16;

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
    src: (&ConsFieldsGeneric<D, DOF, Mem, Sc>, &PrimFieldsGeneric<D, DOF, Mem, Sc>),
    dst: (&ConsFieldsGeneric<D, DOF, Mem, Sc>, &PrimFieldsGeneric<D, DOF, Mem, Sc>),
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
        if *is_out { outputs.push(fld); } else { inputs.push(fld); }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.allocated, &inputs, &outputs, &[], &[]);
}

/// dispatch the FREEZE-tier select kernel `{prefix}_fofc_select_{D}d` over the interior: keep the
/// live spliced first-order conserved (`x_*` = the in-place cons/prim) where physical, else FREEZE to
/// the stage input (`us_*` = u_stage). the face-based splice already made every kept cell
/// conservative; this handles only the rare cell no flux can update admissibly (the documented
/// single-cell conservation waiver). only the conserved is chosen; the primitive is re-derived by the
/// c2p after the select.
pub(crate) fn fofc_select<const D: usize, const DOF: usize, Mem, Sc>(
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
        if *is_out { outputs.push(fld); } else { inputs.push(fld); }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.interior, &inputs, &outputs, &[], &[]);
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
        if *is_out { outputs.push(fld); } else { inputs.push(fld); }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.interior.face_domain(dir), &inputs, &outputs, &[], &[]);
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
        if *is_out { outputs.push(fld); } else { inputs.push(fld); }
    }
    dispatch_fields_each::<Sc, Mem, D>(&name, &sim.geom.interior, &inputs, &outputs, &[], &[]);
    field_max_reduce(scratch, &sim.geom.interior)
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
    // MHD-only: re-run the face->cell B interpolation + magnetic-energy patch after the gas-only
    // godunov redo, so the FO-tier cons.nrg regains the patch that gas-only overwrote (fixes the C2
    // energy inconsistency). gated by the caller to the RK stages that apply the patch (single /
    // corrector). a NO-OP closure for every hydro regime (no cell B).
    ct_resync: impl Fn(),
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let ws = &sim.workspace;
    let flag = &ws.fofc_flag;
    // HOST GATE + FLAG: write the per-cell fallback flag (1 where the high-order c2p is unphysical)
    // over the interior. FOFC only corrects a flagged cell (a physical one keeps its high-order flux
    // on every non-fallback face), so with none the whole pass is a no-op — skip it. a clean substage
    // costs one pointwise probe + a reduction, not the flux sweep + two extra c2p passes.
    let probe = format!("{prefix}_fofc_probe{dof_sfx}_{D}d");
    dispatch_named(sim, pre_bind, Some(flag), 0, &probe, &sim.geom.interior, &[], &[]);
    if field_max_reduce(flag, &sim.geom.interior) <= 0.5 {
        return;
    }
    // boundary-consistent ghosts for the flag: a face straddling the periodic wrap (or any boundary)
    // must take ONE first-order decision from both of its cells, else the splice re-creates the very
    // non-conservation it exists to remove.
    crate::regimes::mhd_substrate::flag_ghost_fill(sim, flag);

    let (cons, prim) = (&sim.fields.cons, &sim.fields.prim);
    // save the HIGH-ORDER fluxes before the first-order redo overwrites the live buffers. the
    // componentwise conserved copy is exactly `fofc_restore` (d_x = s_x), reused on the per-direction
    // flux ConsFields.
    for dir in 0..D {
        fofc_copy(sim, prefix, dof_sfx, "restore", (&sim.fields.flux[dir], prim), (&ws.flux_ho[dir], prim));
    }
    fofc_copy(sim, prefix, dof_sfx, "restore", (&ws.u_stage, prim), (cons, prim));
    c2p();
    for dir in 0..D {
        first_order_flux(dir);
    }
    for dir in 0..D {
        fofc_splice(sim, prefix, dof_sfx, dir, flag, &ws.flux_ho[dir]);
    }
    godunov();
    if has_additive {
        source_apply();
    }
    // re-sync the cell B + magnetic-energy patch from the (unchanged, CT-consistent) face field: the
    // gas-only godunov above overwrote cons.nrg without the patch, so an MHD redo would otherwise
    // lose it (C2). idempotent, so it also corrects the doubly-advanced cell B. hydro: no-op.
    ct_resync();
    c2p();
    // PERSISTENT-FREEZE FAIL-LOUD: the freeze tier holds the stage input where even full first-order
    // fluxes leave a cell unphysical. it is the rare correct parachute for a genuinely hard cell
    // (isolated in time), so a single firing is not a failure — but a poisoned source / initial datum
    // that FOFC cannot fix freezes EVERY stage. track the consecutive streak and halt loudly once it
    // persists.
    let froze = fofc_freeze_count(sim, prefix, dof_sfx, scratch, cons, prim);
    if froze > 0.5 {
        let streak = freeze_streak.fetch_add(1, Ordering::Relaxed) + 1;
        assert!(
            streak < FOFC_FREEZE_HALT_STREAK,
            "FOFC last-resort freeze fired on {streak} consecutive substages (regime={prefix}{dof_sfx}): \
             a zone is persistently unrecoverable by the first-order redo — a genuine breakdown, not \
             the rare isolated parachute. check the source / initial data / boundary for a poison."
        );
    } else {
        freeze_streak.store(0, Ordering::Relaxed);
    }
    fofc_select(sim, prefix, dof_sfx, &ws.u_stage, cons, prim);
    c2p();
}
