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

use crate::regimes::substrate_kernels::{dispatch_fields_each, dispatch_named, kernel_field_binds};
use crate::regimes::substrate_gpu::field_max_reduce;

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

/// dispatch the three-tier select kernel `{prefix}_fofc_select_{D}d` over the interior: choose the
/// conserved per zone as high-order (`ho_*` = u_fofc/prim_fofc) if physical, else first-order
/// (`x_*` = the in-place live cons/prim) if physical, else FREEZE to the stage input (`us_*` =
/// u_stage). only the conserved is chosen; the primitive is re-derived by the c2p after the select.
pub(crate) fn fofc_select<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dof_sfx: &str,
    u_fofc: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim_fofc: &PrimFieldsGeneric<D, DOF, Mem, Sc>,
    u_stage: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    cons: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim: &PrimFieldsGeneric<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_select{dof_sfx}_{D}d");
    let slot = |s: &str| -> &Field<Sc, D, Mem> {
        if let Some(c) = s.strip_prefix("ho_") {
            fofc_comp(u_fofc, prim_fofc, c)
        } else if let Some(c) = s.strip_prefix("us_") {
            // the stage-input freeze tier: conserved only (den/mom/nrg); the prim arg is unread.
            fofc_comp(u_stage, prim, c)
        } else if let Some(c) = s.strip_prefix("x_") {
            // the in-place cons: read (first-order) + write (select result), one binding. the
            // prim (x_rho/x_pre) is read-only for the first-order physicality test.
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

/// FOFC FREEZE DIAGNOSTIC (task 1 instrumentation): count the zones where neither the high-order
/// nor the first-order tier is physical — the zones the select freezes to the stage input. writes
/// the per-zone freeze flag to the scratch (reusing `{prefix}_fofc_freeze{dof_sfx}_{D}d`) and
/// max-reduces it over the interior: > 0 iff some zone froze this substage. dispatched between the
/// first-order c2p and the select, so `ho_*` is the high-order snapshot and `x_*` is the first-order
/// live state — exactly the pair the select's tiers test.
pub(crate) fn fofc_freeze_count<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    dof_sfx: &str,
    scratch: &Field<Sc, D, Mem>,
    u_fofc: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim_fofc: &PrimFieldsGeneric<D, DOF, Mem, Sc>,
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
        } else if let Some(c) = s.strip_prefix("ho_") {
            fofc_comp(u_fofc, prim_fofc, c)
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


/// the FOFC flow: (1) snapshot the high-order cons+prim -> u_fofc/prim_fofc; (2) restore
/// cons <- u_stage so the redo reconstructs from the PHYSICAL stage-input state; (3) c2p; (4)
/// re-flux at first order (the caller's `first_order_flux`); (5) re-godunov; (6) additive source
/// when present; (7) re-c2p; (8) three-tier select; (9) c2p to re-derive the primitive from the
/// selected conserved (the frozen zones carry admissible u_stage, so c2p converges). the caller
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
    first_order_flux: impl Fn(usize),
    c2p: impl Fn(),
    godunov: impl Fn(),
    source_apply: impl Fn(),
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    // HOST GATE: probe the high-order c2p for any unphysical zone. FOFC only ever modifies an
    // unphysical zone (a physical one keeps its high-order value), so with none the whole pass is a
    // no-op — skip it. a clean substage costs one pointwise probe + a reduction, not the flux
    // sweep + two extra c2p passes.
    let probe = format!("{prefix}_fofc_probe{dof_sfx}_{D}d");
    dispatch_named(sim, pre_bind, Some(scratch), 0, &probe, &sim.geom.interior, &[], &[]);
    if field_max_reduce(scratch, &sim.geom.interior) <= 0.5 {
        return;
    }
    let (cons, prim) = (&sim.fields.cons, &sim.fields.prim);
    let ws = &sim.workspace;
    fofc_copy(sim, prefix, dof_sfx, "snap", (cons, prim), (&ws.u_fofc, &ws.prim_fofc));
    fofc_copy(sim, prefix, dof_sfx, "restore", (&ws.u_stage, prim), (cons, prim));
    c2p();
    for dir in 0..D {
        first_order_flux(dir);
    }
    godunov();
    if has_additive {
        source_apply();
    }
    c2p();
    // task 1 instrumentation: count the zones about to hit the freeze tier (neither HO nor FO
    // physical). a fully-PCP pipeline drives this to zero; a nonzero count flags a leak.
    let froze = fofc_freeze_count(sim, prefix, dof_sfx, scratch, &ws.u_fofc, &ws.prim_fofc, cons, prim);
    if froze > 0.5 {
        eprintln!("[fofc-diag] freeze tier fired (regime={prefix}{dof_sfx})");
    }
    fofc_select(sim, prefix, dof_sfx, &ws.u_fofc, &ws.prim_fofc, &ws.u_stage, cons, prim);
    c2p();
}
