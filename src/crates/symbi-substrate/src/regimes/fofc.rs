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

use crate::regimes::substrate_kernels::{dispatch_fields_each, kernel_field_binds};

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
    tag: &str,
    src: (&ConsFieldsGeneric<D, DOF, Mem, Sc>, &PrimFieldsGeneric<D, DOF, Mem, Sc>),
    dst: (&ConsFieldsGeneric<D, DOF, Mem, Sc>, &PrimFieldsGeneric<D, DOF, Mem, Sc>),
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_{tag}_{D}d");
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
    u_fofc: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim_fofc: &PrimFieldsGeneric<D, DOF, Mem, Sc>,
    u_stage: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    cons: &ConsFieldsGeneric<D, DOF, Mem, Sc>,
    prim: &PrimFieldsGeneric<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("{prefix}_fofc_select_{D}d");
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

/// the FOFC flow: (1) snapshot the high-order cons+prim -> u_fofc/prim_fofc; (2) restore
/// cons <- u_stage so the redo reconstructs from the PHYSICAL stage-input state; (3) c2p; (4)
/// re-flux at first order (the caller's `first_order_flux`); (5) re-godunov; (6) additive source
/// when present; (7) re-c2p; (8) three-tier select; (9) c2p to re-derive the primitive from the
/// selected conserved (the frozen zones carry admissible u_stage, so c2p converges). the caller
/// supplies the substrate's own first-order flux (per direction) / c2p / godunov / source_apply.
pub(crate) fn fofc_orchestrate<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    has_additive: bool,
    first_order_flux: impl Fn(usize),
    c2p: impl Fn(),
    godunov: impl Fn(),
    source_apply: impl Fn(),
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let (cons, prim) = (&sim.fields.cons, &sim.fields.prim);
    let ws = &sim.workspace;
    fofc_copy(sim, prefix, "snap", (cons, prim), (&ws.u_fofc, &ws.prim_fofc));
    fofc_copy(sim, prefix, "restore", (&ws.u_stage, prim), (cons, prim));
    c2p();
    for dir in 0..D {
        first_order_flux(dir);
    }
    godunov();
    if has_additive {
        source_apply();
    }
    c2p();
    fofc_select(sim, prefix, &ws.u_fofc, &ws.prim_fofc, &ws.u_stage, cons, prim);
    c2p();
}
