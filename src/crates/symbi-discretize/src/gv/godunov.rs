// =============================================================================
// godunov.rs
//
// the conserved-update godunov family: snapshot, ssp stage, fused sources, and the unified dag-application operator.
// =============================================================================

use super::*;
use symbi_geometry::{KerrKS, KerrKSCartesian, KerrKSCylindrical, Schwarzschild, SchwarzschildKS, SchwarzschildKSCartesian, SchwarzschildKSCylindrical};
use symbi_geometry::grhd_source::{grhd_covariant_source, grmhd_covariant_source};
use symbi_algebra::Tensor;
use symbi_ir::dual::Dual;


/// snapshot `u_n = cons` — a pure pointwise copy (the RK2 stage-0 hold), geometry-INDEPENDENT
/// (works for every coord system). copies the energy too when `has_energy`. write root == the
/// read field node (a direct buffer copy).
pub fn snapshot_gv(ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let nrg = has_energy.then(|| Gv::field("cons_nrg", FieldRef::cons_nrg()));
    let mut writes = vec![("u_n_den".to_string(), FieldRef::un_den().into(), den.node())];
    for (k, m) in mom.iter().enumerate() {
        writes.push((format!("u_n_mom_{k}"), FieldRef::un_mom(k as u8).into(), m.node()));
    }
    if let Some(n) = nrg {
        writes.push(("u_n_nrg".to_string(), FieldRef::un_nrg().into(), n.node()));
    }
    (end_trace(), writes)
}


/// a componentwise conserved-field copy `dst = src` over the gas conserved (den, mom[k], nrg?).
/// used to (a) restore `cons <- u_stage` so the first-order redo reconstructs from the physical
/// stage-input state, and (b) save the high-order per-direction fluxes before the redo overwrites the
/// live flux buffers (both are ConsFields). explicit-field dispatch: slots `s_*` (source) -> `d_*`
/// (dest).
pub fn fofc_copy_gv(ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    let mut cp = |name: &str| {
        let v = Gv::field(&format!("s_{name}"), &format!("s_{name}"));
        writes.push((format!("d_{name}"), format!("d_{name}").into(), v.node()));
    };
    cp("den");
    for k in 0..ncomp {
        cp(&format!("mom_{k}"));
    }
    if has_energy {
        cp("nrg");
    }
    (end_trace(), writes)
}


/// the FIRST-ORDER FLUX-CORRECTION select: `out = physical(ho_prim) ? ho : fo`, componentwise over
/// the gas conserved (den, mom[k], nrg?) and primitive (rho, vel[k], pre?). the high-order state
/// `ho_*` is the snapshot taken before the substage was redone at first order; `fo_*` is the redone
/// (PCM + HLLE) result, aliased to the live cons/prim `out_*` (in-place read+write). the failure
/// test is metric-free and needs only rho/pre: both relativistic recoveries drive an out-of-cone
/// state to a FINITE flagged result — density from the ceiling-clamped Lorentz factor, pressure to
/// the non-positive `C2P_CONE_FAIL_PRESSURE` sentinel (see c2p_result) — so an unphysical recovery
/// always shows up as rho <= 0 or pre <= 0 (both fail `> 0`), never needing a velocity test. so a
/// cell whose HIGH-ORDER c2p is physical keeps its sharp state; only the failed cells take the
/// diffusive first-order result. carrier-generic, regime-generic (has_energy toggles the pressure law).
/// the FOFC HOST GATE probe: write 1 to the scratch where the high-order c2p is unphysical (density
/// or, for an energy regime, pressure non-finite or non-positive), else 0. a max-reduce over the
/// interior is > 0 exactly when some zone needs correcting; a clean substage reduces to 0 and skips
/// the whole FOFC pass (which would keep the high-order everywhere anyway — bit-identical to skip).
pub fn fofc_probe_gv(ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let finite_pos = |v: Gv| (v - v).cmp_eq(Gv::ZERO) & v.cmp_gt(Gv::ZERO);
    let finite = |v: Gv| (v - v).cmp_eq(Gv::ZERO);
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let mut physical = if has_energy {
        let pre = Gv::field("prim_pre", FieldRef::PrimPre);
        finite_pos(rho) & finite_pos(pre)
    } else {
        finite_pos(rho)
    };
    // the full state vector: each velocity component must be FINITE (its SIGN is physical, so no
    // positivity test). catches a non-finite momentum the density/pressure test misses — notably for
    // iso, whose only other guard is the density, so a NaN momentum would otherwise ride through the
    // FOFC gate until the next flux divergence poisons the density one step later.
    for k in 0..ncomp {
        physical = physical & finite(Gv::field(&format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8)));
    }
    let flag = Gv::select(physical, Gv::ZERO, Gv::ONE);
    (end_trace(), vec![("flag".to_string(), FieldRef::Scratch.into(), flag.node())])
}


/// GHOST-BAND FAIL-LOUD probe: write 1 where the density is non-finite (NaN or +-inf via
/// `(rho - rho) != 0`), else 0. run over the ALLOCATED domain (interior + ghosts): first-order flux
/// correction keeps the INTERIOR finite, but it never touches the ghost band, so a poisoned boundary
/// (a driven-inflow expression producing NaN, a broken BC) leaves a non-finite ghost that FOFC cannot
/// mask. a max-reduce > 0 forces the CFL rate to +inf (dt -> 0, the driver halts) — the fail-loud that
/// survives FOFC recovery. density-only, so regime- and energy-independent (one kernel per dimension);
/// a poison in any primitive reaches the density within one c2p / flux divergence.
pub fn state_finite_probe_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let flag = Gv::select((rho - rho).cmp_eq(Gv::ZERO), Gv::ZERO, Gv::ONE);
    (end_trace(), vec![("flag".to_string(), FieldRef::Scratch.into(), flag.node())])
}


/// FOFC FREEZE diagnostic: write 1 to `freeze` where the SPLICED first-order result (`x_*`, the live
/// cons after the face-based redo) is still unphysical — the zones the freeze tier holds at the
/// stage input. reduced over the interior to count freezes per substage; a fully
/// physical-constraint-preserving low-order scheme recovers every flagged cell (full first-order
/// fluxes on all its faces), driving this to zero, so a nonzero count localizes where a PCP
/// assumption leaks and the run trades a cell's conservation for finiteness.
pub fn fofc_freeze_probe_gv(ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let finite_pos = |v: Gv| (v - v).cmp_eq(Gv::ZERO) & v.cmp_gt(Gv::ZERO);
    let finite = |v: Gv| (v - v).cmp_eq(Gv::ZERO);
    let x_rho = Gv::field("x_rho", "x_rho");
    let mut physical = if has_energy {
        let x_pre = Gv::field("x_pre", "x_pre");
        finite_pos(x_rho) & finite_pos(x_pre)
    } else {
        finite_pos(x_rho)
    };
    // the freeze count mirrors the select's physicality: each spliced velocity must be FINITE too
    // (sign is physical), so a non-finite momentum the density/pressure test misses is still counted
    // (and, in the select, frozen to the stage input).
    for k in 0..ncomp {
        let p = format!("x_vel_{k}");
        physical = physical & finite(Gv::field(&p, &p));
    }
    let frozen = Gv::select(physical, Gv::ZERO, Gv::ONE);
    (end_trace(), vec![("freeze".to_string(), "freeze".into(), frozen.node())])
}


/// the FOFC FREEZE tier select (the face-based redo's only per-cell state replacement): keep the
/// live spliced first-order conserved (`x_*`) where it is physical, else FREEZE to the stage-input
/// state `u_stage` (`us_*`) — the pre-godunov conserved, admissible from stage entry, so the final
/// c2p converges on it. the face-based splice already made every kept cell conservative (one flux per
/// face); this handles only the rare cell that no flux can update admissibly, holding its stage input
/// so no NaN propagates. that single-cell hold is the documented conservation waiver — it
/// discards the cell's flux exchange, bounded by the persistent-freeze fail-loud. only the conserved
/// is chosen; the primitive is re-derived by the c2p that follows.
pub fn fofc_select_gv(ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // finite AND positive: (v - v) is 0 for a finite value and NaN for NaN OR +-inf (inf - inf =
    // NaN), so cmp_eq(0) rejects every non-finite value; the > 0 rejects a vacuum/negative one. a
    // "physical" cell is one whose density (and pressure, when modelled) passes both.
    let finite_pos = |v: Gv| (v - v).cmp_eq(Gv::ZERO) & v.cmp_gt(Gv::ZERO);
    let finite = |v: Gv| (v - v).cmp_eq(Gv::ZERO);
    let x_rho = Gv::field("x_rho", "x_rho");
    let mut physical = if has_energy {
        let x_pre = Gv::field("x_pre", "x_pre");
        finite_pos(x_rho) & finite_pos(x_pre)
    } else {
        finite_pos(x_rho)
    };
    // the full state vector: each spliced velocity must be FINITE (its sign is physical) — else a
    // non-finite momentum with a finite density/pressure would be kept and propagated instead of
    // frozen to the admissible stage input.
    for k in 0..ncomp {
        let p = format!("x_vel_{k}");
        physical = physical & finite(Gv::field(&p, &p));
    }
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    // the live cons (`x_*`) is read+write IN PLACE: it holds the spliced first-order result and is
    // overwritten with the chosen tier. one slot per component (read path == write path) so the IR
    // dedups it to a single in-place binding (the CT-`b` pattern) — no input/output aliasing.
    let mut sel_inplace = |comp: &str, us: Gv| {
        let path = format!("x_{comp}");
        let x = Gv::field(&path, &path);
        let chosen = Gv::select(physical, x, us);
        writes.push((path.clone(), path.into(), chosen.node()));
    };
    sel_inplace("den", Gv::field("us_den", "us_den"));
    for k in 0..ncomp {
        sel_inplace(&format!("mom_{k}"), Gv::field(&format!("us_mom_{k}"), &format!("us_mom_{k}")));
    }
    if has_energy {
        sel_inplace("nrg", Gv::field("us_nrg", "us_nrg"));
    }
    (end_trace(), writes)
}

/// the FREEZE-tier select WITH the immersed-body source composed INLINE — the LAZY, buffer-free
/// answer to "a frozen cell must not lose its body gravity/accretion". identical to `fofc_select_gv`
/// except the freeze parachute is the stage input EVOLVED by the body source in registers,
/// `u_stage + dt*body(u_stage)` (via `body_evolved_gv` / `body_evolved_iso_gv`). no buffer is materialized: the body delta AND the c2p pressure used to GUARD it are
/// closed forms of `us_*`. the guard preserves the freeze tier's physical-parachute invariant — a body
/// kick that would drive the parachute unphysical (a strong pull on a low-internal-energy cell) falls
/// back to the bare stage input. `has_energy` selects the adiabatic (evolves nrg, eos param = gamma,
/// pressure guard) vs the isothermal (no nrg, eos param = cs, `p = cs^2 * rho > 0` so only the density
/// guard) form. body-free regimes keep `fofc_select_gv`.
pub fn fofc_select_with_body_gv(
    ncomp: usize,
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    has_energy: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let finite_pos = |v: Gv| (v - v).cmp_eq(Gv::ZERO) & v.cmp_gt(Gv::ZERO);
    let finite = |v: Gv| (v - v).cmp_eq(Gv::ZERO);
    // the spliced first-order result's physicality, IDENTICAL to `fofc_select_gv`: density always,
    // pressure only when the energy is modelled (iso keeps p in a separate cs^2 buffer, so its
    // select gates on the density alone), plus finiteness of every velocity component.
    let x_rho = Gv::field("x_rho", "x_rho");
    let mut physical_fo = if has_energy {
        finite_pos(x_rho) & finite_pos(Gv::field("x_pre", "x_pre"))
    } else {
        finite_pos(x_rho)
    };
    for k in 0..ncomp {
        let p = format!("x_vel_{k}");
        physical_fo = physical_fo & finite(Gv::field(&p, &p));
    }
    // the stage input evolved by the body source, INLINE (no buffer, no separate pass).
    let dt = Gv::scalar("dt");
    let us_den = Gv::field("us_den", "us_den");
    let us_mom: Vec<Gv> = (0..ncomp)
        .map(|k| Gv::field(&format!("us_mom_{k}"), &format!("us_mom_{k}")))
        .collect();
    // the body-evolved conserved parachute + its physicality, energy-aware. `us_nrg` is bound only in
    // the adiabatic form so the isothermal kernel manifest carries no energy field.
    let (b_den, b_mom, b_nrg, usb_ok) = if has_energy {
        let gamma = Gv::scalar("gamma");
        let us_nrg = Gv::field("us_nrg", "us_nrg");
        let (b_den, b_mom, b_nrg) = crate::gv_immersed::body_evolved_gv(
            us_den, &us_mom, us_nrg, dt, gamma, n_bodies, coords, ndim, ncomp, axes,
        );
        // GUARD: the parachute must itself be physical. rho = den (Newtonian, W = 1); the adiabatic
        // pressure p = (gamma-1)(nrg - 0.5|mom|^2/den) is a closed form of the evolved cons.
        let mut ke = Gv::ZERO;
        for m in &b_mom {
            ke = ke + *m * *m;
        }
        let b_pre = (gamma - Gv::ONE) * (b_nrg - Gv::from_f64(0.5) * ke / b_den);
        let usb_ok = finite_pos(b_den) & finite_pos(b_pre);
        (b_den, b_mom, Some((us_nrg, b_nrg)), usb_ok)
    } else {
        // isothermal EOS: p = cs^2 * rho, so the stage-input pressure is a closed form of us_den; the
        // pressure stays positive wherever the density does, hence only the density guard.
        let cs = Gv::scalar("cs");
        let us_pre = cs * cs * us_den;
        let (b_den, b_mom) = crate::gv_immersed::body_evolved_iso_gv(
            us_den, &us_mom, us_pre, dt, n_bodies, coords, ndim, ncomp, axes,
        );
        let usb_ok = finite_pos(b_den);
        (b_den, b_mom, None, usb_ok)
    };
    let parachute = |ub: Gv, us: Gv| Gv::select(usb_ok, ub, us);
    // main select IN PLACE: `x_*` (the spliced first-order cons) is kept where physical, else FROZEN
    // to the guarded body-evolved stage input.
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    let mut sel = |comp: &str, par: Gv| {
        let path = format!("x_{comp}");
        let x = Gv::field(&path, &path);
        let chosen = Gv::select(physical_fo, x, par);
        writes.push((path.clone(), path.into(), chosen.node()));
    };
    sel("den", parachute(b_den, us_den));
    for k in 0..ncomp {
        sel(&format!("mom_{k}"), parachute(b_mom[k], us_mom[k]));
    }
    if let Some((us_nrg, b_nrg)) = b_nrg {
        sel("nrg", parachute(b_nrg, us_nrg));
    }
    (end_trace(), writes)
}


/// the FOFC FACE-BASED FLUX SPLICE for axis `dir`: choose, per interior face, the FIRST-ORDER flux
/// (`fo_*`, the redone HLLE/rusanov flux held in the live `fields.flux[dir]`) where either adjacent
/// cell is flagged for fallback, else the HIGH-ORDER flux (`ho_*`, saved in `flux_ho[dir]` before the
/// redo overwrote the live buffer). the finite-volume convention stores the axis-`dir` flux at cell
/// `c` on the LOW face of `c` (between `c - e_dir` and `c`), so the face is first-order iff
/// `flag[c] > 0 OR flag[c - e_dir] > 0`. the `fo_*` slot is read+write IN PLACE (the live flux
/// buffer), so after the splice every face carries ONE flux value and the following godunov
/// telescopes conservatively across every fallback boundary. componentwise over the conserved flux
/// (den, mom[k], nrg?); the flag is a plain 0/1 cell field with boundary-consistent ghosts.
pub fn fofc_splice_gv(ndim: usize, dir: usize, ncomp: usize, has_energy: bool) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let nd = ndim as u8;
    let d = dir as u8;
    let flag_c = Gv::field("flag", "flag");
    let flag_lo = Gv::field_shifted("flag", "flag", nd, d, -1);
    let face_fo = flag_c.cmp_gt(Gv::ZERO) | flag_lo.cmp_gt(Gv::ZERO);
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    let mut splice = |comp: &str| {
        let fo_name = format!("fo_{comp}");
        let fo = Gv::field(&fo_name, &fo_name);
        let ho = Gv::field(&format!("ho_{comp}"), &format!("ho_{comp}"));
        let chosen = Gv::select(face_fo, fo, ho);
        writes.push((fo_name.clone(), fo_name.into(), chosen.node()));
    };
    splice("den");
    for k in 0..ncomp {
        splice(&format!("mom_{k}"));
    }
    if has_energy {
        splice("nrg");
    }
    (end_trace(), writes)
}


/// the FOFC FACE-BASED INDUCTION-FLUX SPLICE for axis `dir`: the magnetic mirror of
/// `fofc_splice_gv`. per B-component `c` in `0..ncomp`, choose the FIRST-ORDER induction flux
/// (`fo_bflux_{c}`, the redone flux in the live `bflux[dir][c]`) on faces adjacent to a flagged cell,
/// else the HIGH-ORDER flux (`ho_bflux_{c}`, saved in `bflux_ho[dir][c]`). the axis-`dir` induction
/// flux shares the gas flux's face indexing (stored at cell `c` on the low face of `c`), so the face
/// is first-order iff `flag[c] > 0 OR flag[c - e_dir] > 0` — the identical mask to the gas splice.
/// `fo_bflux_{c}` is read+write IN PLACE; the spliced induction flux feeds the cell-B predictor (HO
/// off the fallback region, FO on it) and the Contact FO edge EMF.
pub fn fofc_bflux_splice_gv(ndim: usize, dir: usize, ncomp: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let nd = ndim as u8;
    let d = dir as u8;
    let flag_c = Gv::field("flag", "flag");
    let flag_lo = Gv::field_shifted("flag", "flag", nd, d, -1);
    let face_fo = flag_c.cmp_gt(Gv::ZERO) | flag_lo.cmp_gt(Gv::ZERO);
    let mut writes: Vec<(String, FieldBind, NodeId)> = Vec::new();
    for c in 0..ncomp {
        let fo_name = format!("fo_bflux_{c}");
        let fo = Gv::field(&fo_name, &fo_name);
        let ho = Gv::field(&format!("ho_bflux_{c}"), &format!("ho_bflux_{c}"));
        let chosen = Gv::select(face_fo, fo, ho);
        writes.push((fo_name.clone(), fo_name.into(), chosen.node()));
    }
    (end_trace(), writes)
}


/// the single mass-law godunov step to a SEPARATE output buffer:
/// `rho_new = rho - dt*div(mass_flux)`. cartesian-uniform OR curvilinear (area-weighted).
/// write -> `cons.den_new`.
pub fn godunov_mass_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let geo = (!is_cartesian_uniform(coords, spacing))
        .then(|| cell_geometry_gv(coords, spacing, axes, ndim as usize));
    let rho = Gv::field("rho", FieldRef::cons_den());
    let rho_new = rho - dt * gv_divergence("mass_flux", ndim, &geo);
    let writes = vec![("rho_new".to_string(), "cons.den_new".into(), rho_new.node())];
    (end_trace(), writes)
}


/// the in-place SSP Shu-Osher stage update `cons = a0*u_n + ac*fe(cons)`, where the
/// forward-Euler operator is `fe(u) = u - dt*div(F) (+ dt*S_geom)`. ONE builder for every
/// explicit SSP scheme: the per-stage convex coefficients `(a0, ac)` arrive as RUNTIME
/// scalars, so a SINGLE compiled kernel serves forward-Euler `[(0,1)]`, SSP-RK2
/// `[(0,1),(1/2,1/2)]`, and SSP-RK3 `[(0,1),(3/4,1/4),(1/3,2/3)]` — the integrator is data,
/// not codegen. forward-Euler is the `(a0,ac)=(0,1)` instantiation (the `a0*u_n` term reads
/// the snapshot held by `snapshot_gv` and multiplies it by 0).
///
/// mass + one scalar law per momentum component (+ energy when `has_energy`). cartesian =
/// unweighted divergence, no source; curvilinear = area-weighted divergence + the geometric
/// momentum `source` carried inside the forward-Euler stage. write path == input path (in
/// place). EOS- AND geom-generic.
///
/// this is the no-overlay case of [`godunov_stage_gv_with_fused_sources`] — the full stage
/// body lives there, and the empty source slice traces exactly the plain SSP stage (the splice
/// helper short-circuits on no overlays, so there are no dead vocabulary nodes). kept as a named
/// entry point for the common no-source case.
pub fn godunov_stage_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    godunov_stage_gv_with_fused_sources(coords, spacetime, spacing, axes, ndim, ncomp, has_energy, source, &[], false)
}


/// per-field NodeId contributions from a list of spec sources, bucketed by
/// `target_field`. consumed by `godunov_stage_gv_with_fused_sources` — the spec
/// vocabulary is spliced once, then each conserved law adds its bucket inside the
/// forward-Euler stage.
///
/// **structural shape contract**: spliced outputs MUST have the expected per-target
/// arity (1 for den/nrg, D for mom); spec authors that violate this get a panic that prevents
/// a silent wrong-component write.
struct FusedContribs {
    /// each entry is a `S_den` NodeId to add to `rho_new`.
    den: Vec<NodeId>,
    /// `mom[k]` is the list of `S_mom_k` NodeIds for momentum component k.
    mom: Vec<Vec<NodeId>>,
    /// each entry is a `S_nrg` NodeId to add to `nrg_new`.
    nrg: Vec<NodeId>,
    /// `mag[k]` is the per-component cell-B prescription, ONLY for a driven-boundary
    /// (`WriteMode::Assign`) MHD `bcell` slot. unused (empty) for hydro and for the
    /// accumulate (godunov source) path — the conservation-law lifts never target B.
    mag: Vec<Vec<NodeId>>,
}


/// fused-source splice helper. requires an ACTIVE Gv trace
/// (the caller holds `begin_trace` / `end_trace`). builds the shared primitive
/// vocabulary (`rho`, `vel_k`, lazy `x_k` ↔ centroid), then splices every
/// spec into the trace and buckets the outputs by `target_field`. with no overlays it
/// returns empty buckets WITHOUT touching the trace — so the no-source `godunov_stage_gv`
/// wrapper traces exactly the plain SSP stage, no dead `mom/rho` vocabulary nodes.
fn splice_fused_sources_to_contribs(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    geo: &Option<CellGeometryGv>,
    // the STATE vocabulary the DAG reads `rho`/`vel_k` from (`StateEnv`). `Some((rho,
    // mom))` binds them (sources read the stage/conserved state); `None` binds NOTHING from state — a
    // pure coordinate prescription (a driven boundary, whose DAG outputs the state it does not read).
    // `x_k` (centroid) + scalar params are bound regardless.
    state: Option<(Gv, &[Gv])>,
    // (target_field, built) pairs — the BuiltSource VALUES, so this serves both the AOT path
    // (SourceSpec.build_source(ndim)) and the RUNTIME path (build_user_source's loaded values).
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
) -> FusedContribs {
    use std::collections::HashMap;

    if sources.is_empty() {
        return FusedContribs { den: Vec::new(), mom: vec![Vec::new(); ncomp], nrg: Vec::new(), mag: vec![Vec::new(); ncomp] };
    }

    // ----- shared primitive vocabulary, declared ONCE; CSE collapses the
    // `mom_k / rho` divisions across every overlay that uses them. bound only when the DAG reads
    // state (`StateEnv::Stage`); a coordinate prescription (`StateEnv::Coord`) skips it.
    let mut shared_params: HashMap<String, NodeId> = HashMap::new();
    if let Some((rho, mom)) = state {
        shared_params.insert("rho".to_string(), rho.node());
        for k in 0..ncomp {
            let v_k = mom[k] / rho;
            shared_params.insert(format!("vel_{k}"), v_k.node());
        }
        // pressure-reading sources (e.g., radiative cooling Lambda(rho, T), T = pre/rho): bind `pre`
        // to the c2p-computed `prim.pre` field. at source-apply / fused-godunov time prim is the SSP
        // stage input (not yet recomputed), so this is consistent with rho/vel above. energy-bearing
        // regimes only — iso has no pressure field. bound ONLY when a source actually references
        // `pre` (mirrors `needs_position`): an unconditional bind adds a manifest `prim.pre` read that
        // DUPLICATES the adiabatic godunov's own flux-reconstruction read -> input/output aliasing.
        let needs_pre = sources.iter().any(|(_, b)| b.params.iter().any(|p| p == "pre"));
        if has_energy && needs_pre {
            shared_params.insert("pre".to_string(), Gv::field("pre", FieldRef::PrimPre).node());
        }
    }
    // LAZY centroid binding. `x_k` ↔ cell centroid for specs
    // that declare position params (gravity, immersed bodies). walk the
    // spec params FIRST to detect which axes are needed, then call
    // `cell_geometry_gv` (which declares `x_lo_k` / `dx_k` scalars in the
    // trace) ONLY if at least one axis is referenced. specs without
    // position dependence keep the prior scalar manifest unchanged.
    let needs_position = sources.iter().any(|(_, built)| {
        built.params.iter().any(|p| (0..(ndim as usize))
            .any(|k| *p == format!("x_{k}")))
    });
    if needs_position {
        let centroid_geo = geo.clone().unwrap_or_else(
            || cell_geometry_gv(coords, spacing, axes, ndim as usize),
        );
        for k in 0..(ndim as usize) {
            shared_params.insert(format!("x_{k}"), centroid_geo.centroid[k].node());
        }
    }

    // scalar-leaf cache so the SAME spec param across multiple overlays
    // (e.g., `g_ext_0` in the mom + nrg specs of uniform_acceleration)
    // resolves to ONE Gv leaf — runtime fills one scalar, CSE collapses.
    let mut scalar_leaves: HashMap<String, NodeId> = HashMap::new();
    let mut out = FusedContribs {
        den: Vec::new(),
        mom: vec![Vec::new(); ncomp],
        nrg: Vec::new(),
        mag: vec![Vec::new(); ncomp],
    };
    for (target_field, built) in sources {
        let mut name_to_node = shared_params.clone();
        for pname in &built.params {
            if name_to_node.contains_key(pname) { continue; }
            let nid = *scalar_leaves.entry(pname.clone())
                .or_insert_with(|| Gv::scalar(pname).node());
            name_to_node.insert(pname.clone(), nid);
        }
        let spliced = with_trace(|t| {
            symbi_hydro::source_spec::splice_built_source_into(
                built, t.graph(), &name_to_node,
            )
        });
        match *target_field {
            "den" => {
                assert_eq!(spliced.len(), 1,
                    "splice_fused_sources: den overlay must emit 1 scalar, got {}", spliced.len());
                out.den.push(spliced[0]);
            }
            "mom" => {
                assert_eq!(spliced.len(), ncomp,
                    "splice_fused_sources: mom overlay must emit {ncomp} components, got {}",
                    spliced.len());
                for k in 0..ncomp { out.mom[k].push(spliced[k]); }
            }
            "nrg" => {
                assert!(has_energy,
                    "splice_fused_sources: nrg overlay requires has_energy=true");
                assert_eq!(spliced.len(), 1,
                    "splice_fused_sources: nrg overlay must emit 1 scalar, got {}", spliced.len());
                out.nrg.push(spliced[0]);
            }
            // cell-B prescription (MHD driven boundary): the ncomp-component bcell vector.
            // only valid in the Assign (prescription) mode — the conservation-law source lifts
            // never target B, so the accumulate path asserts mag stays empty.
            "bcell" => {
                assert_eq!(spliced.len(), ncomp,
                    "splice_fused_sources: bcell overlay must emit {ncomp} components, got {}",
                    spliced.len());
                for k in 0..ncomp { out.mag[k].push(spliced[k]); }
            }
            other => panic!("splice_fused_sources: unsupported target_field {other:?}"),
        }
    }
    out
}


/// the SSP Shu-Osher stage update WITH a fused list of spec sources — the
/// `godunov_stage_gv` body (runtime `(a0, ac)` convex coefficients, `cons = a0*u_n + ac*fe`)
/// with the spec contributions spliced into the forward-Euler operator:
/// `fe(u, div, src) = u - dt*div + dt*(geo_src + \sum spec_src)`. one launch folds flux
/// divergence + geometric source + every user overlay + the integrator combine. the dispatch
/// `{prefix}_godunov_stage_with_{slug}_{D}d` resolves here.
///
/// the spec contributions live inside `fe`, so the stage's `ac` weight multiplies them — the
/// same convex coefficient that weights the flux divergence — which is exactly the SSP
/// source treatment (`ac*dt*S` per stage). pass an empty slice for the no-overlay variant.
///
/// this is the COMPILE-TIME entry: it materializes each `SourceSpec`'s `BuiltSource`
/// (`build_source(ndim)`) then delegates to [`godunov_stage_gv_with_fused_built`], the core
/// over `BuiltSource` VALUES that the AOT bake and the RUNTIME user-source path share. the
/// godunov+source trace lives ONCE, in the core.
pub fn godunov_stage_gv_with_fused_sources(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    user_sources: &[&symbi_hydro::source_spec::SourceSpec],
    // when this stage is FUSED with the cell-B predictor, the magnetic geo source reads cell-B
    // via the predictor's `bc_k` key so try_fuse merges the two reads (no input/output alias).
    // the plain (unfused) stage passes false -> reads `prim.mag[k]`.
    mag_from_bcell: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    let builts: Vec<(&str, symbi_hydro::source_spec::BuiltSource)> = user_sources.iter()
        .map(|s| (s.target_field, (s.build_source)(ndim as usize)))
        .collect();
    let src_refs: Vec<(&str, &symbi_hydro::source_spec::BuiltSource)> =
        builts.iter().map(|(t, b)| (*t, b)).collect();
    // spec-overlay fusion (AOT `_with_{slug}` bakes) never carries an immersed body; that is the
    // runtime-source path's job (build_fused_cpu_kernel threads the real count).
    godunov_stage_gv_with_fused_built(
        coords, spacetime, spacing, axes, ndim, ncomp, has_energy, source, &src_refs, mag_from_bcell, 0,
    )
}


/// the SSP stage core over PRE-BUILT sources — `BuiltSource` VALUES paired with their target
/// field, the shape `splice_fused_sources_to_contribs` consumes. the SourceSpec entry
/// [`godunov_stage_gv_with_fused_sources`] feeds AOT specs (`build_source(ndim)`); the runtime
/// user-source CPU fusion feeds `RuntimeSource`'s loaded `BuiltSource`s directly. ONE trace, both
/// paths — no duplicated godunov+source lowering. `sources` is `(target_field, built)` pairs.
#[allow(clippy::too_many_arguments)]
pub fn godunov_stage_gv_with_fused_built(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
    mag_from_bcell: bool,
    // immersed-body count. > 0 wraps the post-combine cons with `body_evolved_gv` (gravity +
    // accretion drain) at weight `ac*dt`, so the single fused sweep IS `plain godunov + source_apply
    // + body_source`, in that order, bit-for-bit. 0 leaves the update body-free. adiabatic only.
    n_bodies: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let a0 = Gv::scalar("a0");
    let ac = Gv::scalar("ac");
    // the SSP source weight. computed as `ac*dt` so it is BIT-IDENTICAL to the standalone
    // `source_apply_gv` pass's `dt` scalar (the driver fills that with `ac*sim.dt` — the same IEEE
    // f64 product). this is what makes `fused == plain godunov + source_apply` bit-exact: the
    // user source is added as a SEPARATE post-combine term with this weight, never
    // folded into the `ac*fe` multiply (which would distribute the rounding differently).
    let ac_dt = ac * dt;
    // flat spacetime: the physical (orthonormal) finite-volume geometry. curved (GR): the
    // COVARIANT geometry — coordinate-form angular face weights (the alpha sqrt(gamma) measure),
    // matching the covariant momentum S_i and the contravariant fluxes v^i; the orthonormal
    // angular weights would leave every theta-direction force on S_theta short by a factor r.
    // radial faces and the volume coincide, so 1D radial GR is bit-identical.
    // a curved spacetime ALWAYS needs the geometry (the metric position + the alpha sqrt(gamma)
    // densitization measure), even on a cartesian-uniform grid where flat hydro skips it.
    let geo = (!is_cartesian_uniform(coords, spacing) || spacetime != Spacetime::Minkowski)
        .then(|| match spacetime {
        Spacetime::Minkowski => cell_geometry_gv(coords, spacing, axes, ndim as usize),
        // spinning kerr: the densitized measure is Sigma sin(theta) — the spin rides the
        // `kerr_spin` kernel scalar into the face/volume moments.
        Spacetime::Kerr => cell_geometry_covariant_gv(
            coords, spacing, axes, ndim as usize, Some(Gv::scalar("kerr_spin")),
        ),
        _ => cell_geometry_covariant_gv(coords, spacing, axes, ndim as usize, None),
    });
    let rho = Gv::field("rho", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    // on a curved background the flat velocity-quadratic inertial is the WRONG contraction for the
    // covariant momentum S_i (it treats the components as flat); the covariant stress-energy
    // contraction below carries those blocks instead, so the hydro geometric source keeps ONLY its
    // discrete well-balanced pressure form `p (A_hi - A_lo) / V` — which cancels the pressure flux
    // divergence bit-exactly at a uniform-p hydrostatic state, unlike the analytic pressure block
    // `p d_j ln(alpha sqrt(gamma))` of the contraction.
    // the ideal-MHD stress moves to the covariant contraction on the GR path too — the flat
    // Rmhd curvilinear source would double-count the inertia/tension with the WRONG (flat)
    // contraction for covariant S_i; only the GAS-pressure discrete block stays.
    let source_discrete = match (spacetime, source) {
        (Spacetime::Minkowski, s) => s,
        (_, GeoSource::Hydro { .. }) | (_, GeoSource::Rmhd) => GeoSource::Hydro { inertial: false },
        (_, s) => s,
    };
    let src = geo
        .as_ref()
        .map(|g| gv_geometric_source(coords, axes, ndim as usize, ncomp, g, source_discrete, &mom, mag_from_bcell));

    let contribs = splice_fused_sources_to_contribs(
        coords, spacing, axes, ndim, ncomp, has_energy, &geo, Some((rho, &mom)), sources,
    );

    // the plain forward-Euler stage carries ONLY the flux divergence + the (well-balanced)
    // geometric source — NOT the user sources. `cons_new = a0*u_n + ac*fe`, identical to
    // `godunov_stage_gv`. the homologous-mesh dilution `-mesh_hdil * u` (with
    // `mesh_hdil = ndim * a_dot / a`, the comoving volume-growth rate) rides every
    // conserved law; the static binding mesh_hdil = 0 subtracts an exact zero.
    let h_dil = Gv::scalar("mesh_hdil");
    // GR densitization (Valencia 3+1, static diagonal background): the spatial RHS — the flux
    // divergence + the geometric momentum source — is weighted by the lapse `alpha(x)`. NOT the
    // `u` snapshot or the mesh-dilution term (those are the time / comoving parts). flat
    // spacetime -> `None` -> untouched, bit-identical (see `gv_lapse_weight`).
    // the coordinate-indexed cell centroid (r at slot 0) for the lapse alpha(x); only the
    // curvilinear path carries one (cartesian-uniform geo = None is always Minkowski -> unused).
    let coord_centroid: Vec<Gv> = match &geo {
        Some(g) => {
            let mut c = vec![Gv::ZERO; 3];
            for d in 0..(ndim as usize) {
                c[axes[d]] = g.centroid[d];
            }
            c
        }
        None => Vec::new(),
    };
    assert!(
        spacetime == Spacetime::Minkowski
            || matches!(source, GeoSource::Rmhd | GeoSource::Hydro { .. }),
        "the GR godunov source carries the perfect-fluid or ideal-MHD stress only"
    );
    let lapse = gv_lapse_weight(coords, spacetime, &coord_centroid);
    // the GR geodesic sources from the FULL covariant contraction `grhd_covariant_source`: the
    // per-coordinate momentum source S_j = (1/2) T^{mu nu} d_j g_{mu nu} and the energy source
    // S_tau, one forward-autodiff pass per axis at the metric's full spherical D = 3 (the metric
    // supplies only its ADM line element — no hand-derived christoffels). the MOMENTUM call takes
    // p = 0: the E-part only (gravity + covariant centrifugal), because the pressure block
    // `p d_j ln(alpha sqrt(gamma))` rides the DISCRETE well-balanced form in gv_geometric_source
    // above. the ENERGY call takes the full p — S_tau needs no discrete balance (it vanishes
    // identically at a zero-shift hydrostatic state). the polar angle is the cell centroid when
    // gridded, else pi/2 (exact: with no polar grid every theta-dependence cancels). flat -> None.
    // GRMHD-ready: the EM stress just changes T^{mu nu}.
    let geodesic: Option<(Tensor<Gv, 3>, Gv)> = match spacetime {
        Spacetime::Minkowski => None,
        _ => {
            let mass = Dual::constant(Gv::scalar("schwarzschild_mass")); // constant w.r.t. position
            // coordinate-indexed metric position: each gridded coordinate at its centroid, each
            // ungridded coordinate at its chart symmetry default (spherical polar -> pi/2, else 0).
            let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                if axes.contains(&c) { coord_centroid[c] } else { gv_ungridded_slot(coords, c) }
            }));
            let e = rho + Gv::field("nrg", FieldRef::cons_nrg()) + Gv::field("pre", FieldRef::PrimPre);
            let p = Gv::field("pre", FieldRef::PrimPre);
            // the CONTRAVARIANT velocity in coordinate slots (the metric-aware c2p output);
            // spherical GR momentum slots are coordinate-ordered, so slot k == coordinate k.
            // coordinates without a momentum slot carry zero.
            let v = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                if c < ncomp {
                    Gv::field(&format!("prim_v{c}"), FieldRef::PrimVel(c as u8))
                } else {
                    Gv::ZERO
                }
            }));
            if matches!(source, GeoSource::Rmhd) {
                // GRMHD: the ideal-MHD stress in the same contraction. the source takes the
                // METRIC-FREE rest enthalpy density rho_h = rho + Gamma/(Gamma-1) p (it builds
                // W and b^mu from the harvested gamma internally); B reads the cell field under
                // the same key convention as the discrete magnetic geo source. the momentum call
                // takes p = 0 (the gas-pressure block rides the discrete well-balanced form) but
                // keeps the FULL magnetic stress — the b^2/2 isotropic block is analytic; the
                // one-step-residual instrument adjudicates its balance. spinning kerr is not
                // covered here (the dragging-consistent reconstruction does not yet extend to B).
                let gamma_eos = Gv::scalar("gamma");
                let prim_rho = Gv::field("prim_rho", FieldRef::PrimRho);
                let rho_h = prim_rho + gamma_eos / (gamma_eos - Gv::ONE) * p;
                let b = Tensor::<Gv, 3>::new(std::array::from_fn(|k| {
                    if mag_from_bcell {
                        Gv::field(&format!("bc_{k}"), FieldRef::BCell(k as u8))
                    } else {
                        Gv::field(&format!("prim_b{k}"), &format!("prim.mag[{k}]"))
                    }
                }));
                let src_at = |pp: Gv| match spacetime {
                    Spacetime::Schwarzschild => {
                        grmhd_covariant_source(&Schwarzschild { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::KerrSchild if coords == Coords::Cartesian => {
                        grmhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::KerrSchild if coords == Coords::Cylindrical => {
                        grmhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::KerrSchild => {
                        grmhd_covariant_source(&SchwarzschildKS { mass }, x, rho_h, v, pp, b)
                    }
                    Spacetime::Kerr if coords == Coords::Cartesian => {
                        // cartesian spinning kerr: the rank-1 kerr-schild metric at the FULL
                        // cartesian position; derivatives ride the same autodiff Dual pass.
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grmhd_covariant_source(&KerrKSCartesian { mass, spin }, x, rho_h, v, pp, b)
                    }
                    Spacetime::Kerr if coords == Coords::Cylindrical => {
                        // cylindrical spinning kerr: the rank-1 update on the diag(1, R^2, 1)
                        // base at the FULL (R, phi, z) position; same autodiff Dual pass.
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grmhd_covariant_source(&KerrKSCylindrical { mass, spin }, x, rho_h, v, pp, b)
                    }
                    Spacetime::Kerr => {
                        // the generic covariant stress contraction S_j = (1/2) T^{mu nu} d_j g_{mu nu}
                        // with the EM stress; the non-diagonal kerr metric enters only through the
                        // autodiff Dual pass, no per-block closed form.
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grmhd_covariant_source(&KerrKS { mass, spin }, x, rho_h, v, pp, b)
                    }
                    Spacetime::Minkowski => unreachable!("flat handled above"),
                };
                let (s_mom, _) = src_at(Gv::ZERO);
                let (_, s_tau) = src_at(p);
                Some((s_mom, s_tau))
            } else {
                let src_at = |pp: Gv| match spacetime {
                    Spacetime::Schwarzschild => grhd_covariant_source(&Schwarzschild { mass }, x, e, v, pp),
                    Spacetime::KerrSchild if coords == Coords::Cartesian => {
                        grhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, e, v, pp)
                    }
                    Spacetime::KerrSchild if coords == Coords::Cylindrical => {
                        grhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, e, v, pp)
                    }
                    Spacetime::KerrSchild => grhd_covariant_source(&SchwarzschildKS { mass }, x, e, v, pp),
                    Spacetime::Kerr if coords == Coords::Cartesian => {
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grhd_covariant_source(&KerrKSCartesian { mass, spin }, x, e, v, pp)
                    }
                    Spacetime::Kerr if coords == Coords::Cylindrical => {
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grhd_covariant_source(&KerrKSCylindrical { mass, spin }, x, e, v, pp)
                    }
                    Spacetime::Kerr => {
                        let spin = Dual::constant(Gv::scalar("kerr_spin"));
                        grhd_covariant_source(&KerrKS { mass, spin }, x, e, v, pp)
                    }
                    Spacetime::Minkowski => unreachable!("flat handled above"),
                };
                let (s_mom, _) = src_at(Gv::ZERO);
                let (_, s_tau) = src_at(p);
                Some((s_mom, s_tau))
            }
        }
    };
    let mom_gravity: Option<Tensor<Gv, 3>> = geodesic.map(|(s_mom, _)| s_mom);
    // the GR geodesic ENERGY source S_tau — the second output of the contraction (gravity's rate
    // of work on the infalling gas). zero on a flat background.
    let nrg_gravity: Option<Gv> = geodesic.map(|(_, s_tau)| s_tau);
    let fe = |u: Gv, div: Gv, geo_src: Option<Gv>| {
        let div = match lapse { Some(a) => a * div, None => div };
        let mut r = u - dt * div - dt * (h_dil * u);
        if let Some(s) = geo_src {
            let s = match lapse { Some(a) => a * s, None => s };
            r = r + dt * s;
        }
        r
    };
    let combine = |un: Gv, fe: Gv| a0 * un + ac * fe;
    // the USER sources ride as a SEPARATE additive term after the combine: `+ \sum ac*dt*contrib`,
    // accumulated exactly as `source_apply_gv` accumulates it (start from the combine result,
    // `+= ac_dt*contrib` per spec). so the fused kernel IS `plain godunov + the additive pass`,
    // bit-for-bit, fused into one launch (proven by the fused-equivalence test).
    let with_sources = |base: Gv, srcs: &[NodeId]| {
        let mut r = base;
        for c in srcs {
            r = r + ac_dt * Gv::of(*c);
        }
        r
    };

    let u_n_rho = Gv::field("u_n_rho", FieldRef::un_den());
    let rho_g = with_sources(
        combine(u_n_rho, fe(rho, gv_divergence("mass_flux", ndim, &geo), None)),
        &contribs.den,
    );
    let mut mom_g: Vec<Gv> = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let u_n_mom = Gv::field(&format!("u_n_mom_{k}"), FieldRef::un_mom(k as u8));
        let div = gv_divergence(&format!("mom_flux_{k}"), ndim, &geo);
        let geo_src = src.as_ref().map(|s| s[k]);
        // every momentum slot carries its covariant geodesic block (gravity + covariant
        // centrifugal, coordinate k of the contraction) on top of the discrete pressure form in
        // geo_src; a suppressed axisymmetric slot's block is identically zero (the metric never
        // reads phi, so its autodiff tangent vanishes — angular-momentum conservation).
        let mom_src = match mom_gravity {
            Some(g) => Some(geo_src.map_or(g[k], |s| s + g[k])),
            None => geo_src,
        };
        // Valencia covariant storage: the conserved momentum is the COVARIANT S_i = rho h W^2
        // gamma_ij v^j (the metric-aware c2p + flux), and the geodesic source is written for that
        // covariant S_i, so d_t S_i = -alpha div(F) + alpha S — a SINGLE, uniform lapse on every
        // conserved law, supplied by the `fe` weight. no orthonormal alpha^2 asymmetry: the flux
        // kernel already carries the contravariant v^n (no V_rhat), and the metric coefficient
        // gamma_ij rides inside S_i, above the densitization.
        mom_g.push(with_sources(
            combine(u_n_mom, fe(mom[k], div, mom_src)),
            &contribs.mom[k],
        ));
    }
    let nrg_g = has_energy.then(|| {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        let u_n_nrg = Gv::field("u_n_nrg", FieldRef::un_nrg());
        with_sources(
            combine(u_n_nrg, fe(nrg, gv_divergence("nrg_flux", ndim, &geo), nrg_gravity)),
            &contribs.nrg,
        )
    });

    // immersed-body wrap. the body is a POST-combine operator `(cons_g + ac_dt*S_grav)*f` with
    // `f = exp(-drain*ac_dt)`, reading the godunov+source-combined state. that is exactly the
    // two-pass execution order (godunov -> source_apply -> body_source, every stage at weight
    // ac*dt), so the fused sweep stays bit-identical to plain godunov followed by the standalone
    // `body_source` pass: storing cons_g to an f64 buffer and reading it back is exact, so the
    // register-resident cons_g the body reads here equals the memory value the two-pass body reads.
    // gravity is additive but the accretion drain is multiplicative, so neither rides the additive
    // `contribs` accumulation — the body wraps the final nodes. adiabatic (energy) only; the iso
    // body (`body_source_iso_gv`, cs from prim.pre) is a follow-on.
    let (rho_final, mom_final, nrg_final) = if n_bodies > 0 {
        let nrg_in = nrg_g.expect("fused body source requires has_energy (iso body fusion pending)");
        let gamma = Gv::scalar("gamma");
        let (den_b, mom_b, nrg_b) = crate::gv_immersed::body_evolved_gv(
            rho_g, &mom_g, nrg_in, ac_dt, gamma, n_bodies, coords, ndim as usize, ncomp, axes,
        );
        (den_b, mom_b, Some(nrg_b))
    } else {
        (rho_g, mom_g, nrg_g)
    };

    let mut writes = vec![("rho".to_string(), FieldRef::cons_den().into(), rho_final.node())];
    for (k, m) in mom_final.iter().enumerate() {
        writes.push((format!("mom_{k}"), FieldRef::cons_mom(k as u8).into(), m.node()));
    }
    if let Some(nrg_new) = nrg_final {
        writes.push(("nrg".to_string(), FieldRef::cons_nrg().into(), nrg_new.node()));
    }
    (end_trace(), writes)
}


/// the standalone ADDITIVE source pass: `cons += dt * \sum S(prim, x; params)`, in place, per
/// conserved slot, for a list of spec sources. the GENERAL source executor — it runs ANY composed
/// source as a SEPARATE per-stage kernel (the `body_source_gv` mechanism, generalized to
/// `SourceSpec`s), as opposed to FUSING the source into the godunov stage.
///
/// it splices the SAME `splice_fused_sources_to_contribs` the fused godunov uses, so a plain
/// `godunov_stage_gv` (flux + geometric source, no user sources) followed by this pass is the
/// proven-equivalent DECOMPOSITION of `godunov_stage_gv_with_fused_sources`. the driver passes
/// `dt = ac*dt` (the SSP Shu-Osher stage weight — identical to how `body_source` is invoked), so
/// `S` lands with the same `ac*dt` weight the fused stage applies inside its `ac*fe` combine.
///
// =============================================================================
// THE UNIFIED DAG-APPLICATION OPERATOR.
//
// `apply_dag_core_gv` is the ONE kernel builder behind BOTH the interior source pass and
// the driven-boundary pass. it factors out the decisions a source/boundary
// makes: WHERE the DAG reads state (`StateEnv`), and HOW its result lands in
// the target field (`WriteMode`). the iteration domain + target-field binding are the dispatch's job
// (the same `dispatch_runtime_ir` + `resolve_path` serve cons.* and prim.*), so this builder is the
// whole difference between a source and a boundary prescription. doc 32's user `combine` projects
// onto `WriteMode`: add/relax -> Accumulate (differ only in the constructed expression), overwrite ->
// Assign.
// =============================================================================

/// the state vocabulary the DAG reads `rho`/`vel_k` from. `Stage` binds them from the SSP stage
/// snapshot `u_stage` (an interior source evaluates at its stage input — the stage-input invariant); `Coord`
/// binds NOTHING from state (a pure coordinate prescription — a driven boundary, whose DAG OUTPUTS
/// the state). `x_k` (centroid) + scalar params bind regardless of this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StateEnv {
    Stage,
    Coord,
}


/// how the DAG result lands in the target field. `Accumulate` is the RHS form `target = read(target)
/// + dt * \sum contrib` (in place; the `dt` scalar is the SSP stage weight) — sources. `Assign` is the
/// prescription `target = expr` (write-only, no base, no weight) — driven boundaries. doc 32's
/// `combine`: add + relax both map to `Accumulate`, overwrite to `Assign`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteMode {
    Accumulate,
    Assign,
}


/// the unified core: trace a kernel that evaluates each `(slot, BuiltSource)` DAG per cell and writes
/// it to the slot's field under `mode`. `slot` names the STRUCTURAL conserved slot (`"den"` mass /
/// `"mom"` momentum-vector / `"nrg"` energy); `mode` + the slot pick the runtime path (Accumulate ->
/// `cons.{den,mom_k,nrg}`; Assign -> `prim.{rho,vel_k,pre}`). shared: trace, geometry, the
/// `splice_fused_sources_to_contribs` primitive (leaf binding + per-DAG lowering).
fn apply_dag_core_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    state: StateEnv,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
    mode: WriteMode,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let geo = (!is_cartesian_uniform(coords, spacing))
        .then(|| cell_geometry_gv(coords, spacing, axes, ndim as usize));

    // bind the state vocabulary the DAG reads from. `Stage` reads the stage-input snapshot `u_stage`
    // (NOT post-godunov `cons`): the fused stage evaluates at its stage input, so this standalone
    // pass must too, for `plain + this == fused` bit-for-bit. `Coord` reads no state.
    let state_vocab: Option<(Gv, Vec<Gv>)> = match state {
        StateEnv::Stage => {
            let rho = Gv::field("rho", FieldRef::ustage_den());
            let mom = (0..ncomp)
                .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::ustage_mom(k as u8)))
                .collect();
            Some((rho, mom))
        }
        StateEnv::Coord => None,
    };
    let state_ref = state_vocab.as_ref().map(|(r, m)| (*r, m.as_slice()));

    let contribs = splice_fused_sources_to_contribs(
        coords, spacing, axes, ndim, ncomp, has_energy, &geo, state_ref, sources,
    );

    let writes = match mode {
        WriteMode::Accumulate => {
            // RHS in place: `cons_slot = cons_slot + \sum dt*contrib`, accumulated exactly as the fused
            // stage's `with_sources` — so fused and (plain godunov + this pass) agree bit-for-bit.
            let dt = Gv::scalar("dt"); // the driver fills this with ac*dt (the SSP stage weight)
            let cons_den = Gv::field("cons_den", FieldRef::cons_den());
            let mut rho_new = cons_den;
            for c in &contribs.den {
                rho_new = rho_new + dt * Gv::of(*c);
            }
            let mut writes = vec![("rho".to_string(), FieldRef::cons_den().into(), rho_new.node())];
            for k in 0..ncomp {
                let cons_mom = Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8));
                let mut mom_new = cons_mom;
                for c in &contribs.mom[k] {
                    mom_new = mom_new + dt * Gv::of(*c);
                }
                writes.push((format!("mom_{k}"), FieldRef::cons_mom(k as u8).into(), mom_new.node()));
            }
            if has_energy {
                let cons_nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
                let mut nrg_new = cons_nrg;
                for c in &contribs.nrg {
                    nrg_new = nrg_new + dt * Gv::of(*c);
                }
                writes.push(("nrg".to_string(), FieldRef::cons_nrg().into(), nrg_new.node()));
            }
            // the godunov-source (accumulate) path never targets B — the safe conservation-law
            // lifts touch only den/mom/nrg, and `raw` is gated to those slots. a bcell contrib
            // here means a mis-routed source; fail loud so it is never silently dropped.
            debug_assert!(
                contribs.mag.iter().all(|m| m.is_empty()),
                "accumulate (godunov source) path does not support a `bcell` target",
            );
            writes
        }
        WriteMode::Assign => {
            // prescription: `prim_slot = expr` (write-only, no base, no weight). a prescription is a
            // COMPLETE state — exactly ONE DAG per slot (not summed overlays).
            assert_eq!(contribs.den.len(), 1, "Assign: prim.rho needs exactly one source DAG");
            let mut writes = vec![("rho".to_string(), FieldRef::PrimRho.into(), contribs.den[0])];
            for k in 0..ncomp {
                assert_eq!(contribs.mom[k].len(), 1,
                    "Assign: prim.vel_{k} needs exactly one source DAG");
                writes.push((format!("vel_{k}"), FieldRef::PrimVel(k as u8).into(), contribs.mom[k][0]));
            }
            if has_energy {
                assert_eq!(contribs.nrg.len(), 1, "Assign: prim.pre needs exactly one source DAG");
                writes.push(("pre".to_string(), FieldRef::PrimPre.into(), contribs.nrg[0]));
            }
            // MHD driven boundary: prescribe the cell-B vector (prim.mag). out-of-plane B_phi
            // (cell-centered, flux-evolved) is the SAFE toroidal case; in-plane components are
            // the user's responsibility to keep div-compatible (=0 for a purely toroidal field).
            // absent for a hydro prescription (no bcell slot -> empty mag buckets).
            if contribs.mag.iter().any(|m| !m.is_empty()) {
                for k in 0..ncomp {
                    assert_eq!(contribs.mag[k].len(), 1,
                        "Assign: prim.mag_{k} needs exactly one source DAG");
                    writes.push((format!("mag_{k}"), FieldRef::PrimMag(k as u8).into(), contribs.mag[k][0]));
                }
            }
            writes
        }
    };
    (end_trace(), writes)
}


/// AOT entry: the in-place source-apply kernel from declarative `SourceSpec`s (each `build_source`d
/// at dimension `ndim`). build.rs bakes this per (regime, ndim). the `(Stage, Accumulate)` instance
/// of [`apply_dag_core_gv`].
pub fn source_apply_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    user_sources: &[&symbi_hydro::source_spec::SourceSpec],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    let builts: Vec<(&str, symbi_hydro::source_spec::BuiltSource)> = user_sources.iter()
        .map(|s| (s.target_field, (s.build_source)(ndim as usize)))
        .collect();
    let src_refs: Vec<(&str, &symbi_hydro::source_spec::BuiltSource)> =
        builts.iter().map(|(t, b)| (*t, b)).collect();
    apply_dag_core_gv(coords, spacing, axes, ndim, ncomp, has_energy, StateEnv::Stage, &src_refs, WriteMode::Accumulate)
}


/// RUNTIME entry (Path B): the SAME in-place source-apply kernel, but from already-lowered
/// `(target_field, BuiltSource)` values — e.g., `expr_bridge::build_user_source`'s output from a
/// SourceConfig loaded at sim startup. the `(Stage, Accumulate)` instance of [`apply_dag_core_gv`].
pub fn source_apply_from_built_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    apply_dag_core_gv(coords, spacing, axes, ndim, ncomp, has_energy, StateEnv::Stage, sources, WriteMode::Accumulate)
}


/// DRIVEN-BOUNDARY entry: prescribe the primitive state from coordinate DAGs — the
/// `(Coord, Assign)` instance of [`apply_dag_core_gv`]. `sources` are `(slot, BuiltSource)` with slot
/// `"den"`/`"mom"`/`"nrg"` mapping to `prim.rho`/`prim.vel_k`/`prim.pre`; each DAG reads only
/// `x_k`/`t`/`p_i` and OUTPUTS the prescribed value. dispatched over a face's ghost band.
pub fn boundary_fill_from_built_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    apply_dag_core_gv(coords, spacing, axes, ndim, ncomp, has_energy, StateEnv::Coord, sources, WriteMode::Assign)
}


/// the cell-B induction-flux divergence for component `c` (mirror of `rmhd::bcell_flux_div`):
/// cartesian `sum_d (bf_d_c[+e_d] - bf_d_c)/dx_d`; curvilinear the area-weighted `inv_V sum_d
/// (A_hi_d bf_d_c[+e_d] - A_lo_d bf_d_c)` from `geo` — the SAME divergence the gas godunov uses.
fn bcell_flux_div_gv(c: usize, ndim: usize, geo: &Option<CellGeometryGv>, dx: &[Gv]) -> Gv {
    let off = |d: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[d] = 1;
        o
    };
    let zero = vec![0i32; ndim];
    let mut div: Option<Gv> = None;
    for d in 0..ndim {
        let key = format!("bf_{d}_{c}");
        let here = gv_field_at(&key, &key, ndim, &zero);
        let plus = gv_field_at(&key, &key, ndim, &off(d));
        let term = match geo {
            None => (plus - here) / dx[d],
            Some(g) => g.area_hi[d] * plus - g.area_lo[d] * here,
        };
        div = Some(match div {
            None => term,
            Some(a) => a + term,
        });
    }
    let div = div.unwrap();
    match geo {
        Some(g) => g.inv_volume * div,
        None => div,
    }
}


/// the PLAIN (metric-free) cell-B induction-flux divergence `sum_d (bf_d_c[+e_d] - bf_d_c)/width_d`
/// with the per-axis COORDINATE width read in-kernel (gv_axis_face_at). used for the OUT-OF-PLANE
/// B component whose curl carries no Lame factor — see `metric_free_oop_component`.
fn bcell_flux_div_plain_gv(c: usize, ndim: usize, spacing: &[Spacing]) -> Gv {
    let off = |d: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[d] = 1;
        o
    };
    let zero = vec![0i32; ndim];
    let mut div: Option<Gv> = None;
    for d in 0..ndim {
        let key = format!("bf_{d}_{c}");
        let here = gv_field_at(&key, &key, ndim, &zero);
        let plus = gv_field_at(&key, &key, ndim, &off(d));
        let width = gv_axis_face_at(d, spacing[d], 1) - gv_axis_face_at(d, spacing[d], 0);
        let term = (plus - here) / width;
        div = Some(match div {
            None => term,
            Some(a) => a + term,
        });
    }
    div.unwrap()
}


/// the divergence operator for the OUT-OF-PLANE B component (the one not in `axes`) on a FLAT
/// background, where the stored component is PHYSICAL (orthonormal) and its induction curl is
/// `d_t B_c = -(1/(h1 h2))[d_1(h2 F^1) + d_2(h1 F^2)]` over the in-plane lame factors — which
/// only sometimes coincides with the gas area-weighted divergence.
enum OopDiv {
    /// in-plane lame factors are both 1 (cyl r-z: (curl E)_phi = d_z E_r - d_r E_z), so the
    /// operator is the plain unweighted divergence; the gas h_phi = r cell volume would inject
    /// a spurious F_r/r source.
    Plain,
    /// a non-unit in-plane lame factor rides the curl as a face weight (sph r-theta: h_theta =
    /// r, so `d_t B_phi = -(1/r)[d_r(r F^r) + d_theta F^theta]`); the gas r^2 sin(theta)
    /// measure would inject spurious `-F^r/r - cot(theta) F^theta/r` sources.
    Curl(CellGeometryGv),
}


/// the out-of-plane B component and its curl divergence for a FLAT (physical-component) plane:
/// - cyl r-z (axes [0,2]) -> B_phi (comp 1), metric-free (Plain).
/// - sph r-theta (axes [0,1]) -> B_phi (comp 2), the (r, 1)-weighted curl on the r dr dtheta
///   measure (Curl).
/// - cyl r-phi (axes [0,1]) -> B_z: the z-curl `(1/R)[d_R(R F^R) + d_phi F^phi]` IS the gas
///   R-measure divergence -> None (the gas path is already the curl).
/// - cartesian / fully-gridded (3D): None (plain == gas, or no out-of-plane component).
/// a CURVED spacetime never takes these shortcuts: B is stored CONTRAVARIANT and every
/// component obeys the densitized conservation law `d_t(sqrt(gamma) B^i) + d_j(alpha
/// sqrt(gamma) G^j) = 0` — the covariant area-weighted divergence with the lapse weight.
fn flat_oop_divergence(coords: Coords, spacing: &[Spacing], axes: &[usize], ncomp: usize) -> Option<(usize, OopDiv)> {
    match (coords, axes) {
        (Coords::Cylindrical, [0, 2]) if ncomp > 1 => Some((1, OopDiv::Plain)),
        (Coords::Spherical, [0, 1]) if ncomp > 2 => {
            Some((2, OopDiv::Curl(oop_curl_geometry_sph_rtheta_gv(spacing))))
        }
        _ => None,
    }
}


/// the per-component induction-flux divergences for the cell-B predictor, GR-lapse-weighted.
/// flat: the gas area-weighted divergence, except the out-of-plane component's curl operator
/// (`flat_oop_divergence`). curved: the covariant `alpha sqrt(gamma)` measure for EVERY
/// component, times the lapse `alpha(centroid)` — the same densitization contract as the gas
/// godunov (the face kernel writes `G = F - (beta^n/alpha) U`, deferring one alpha to the
/// divergence; see `gv_lapse_weight`). flat spacetime elides the weight (bit-identical).
fn bcell_flux_divs_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    comps: &[usize],
    ncomp: usize,
    axes: &[usize],
) -> Vec<Gv> {
    let (geo, dx) = bcell_godunov_geom(coords, spacetime, spacing, ndim, axes);
    let oop = match spacetime {
        Spacetime::Minkowski => flat_oop_divergence(coords, spacing, axes, ncomp),
        _ => None,
    };
    // coordinate-indexed cell centroid for the lapse alpha(x) (matching the gas godunov's
    // convention: gridded axes at their centroids, ungridded slots zero). the curved path
    // always carries a geometry (bcell_godunov_geom), so the centroid exists whenever the
    // lapse is non-unit.
    let coord_centroid: Vec<Gv> = match &geo {
        Some(g) => {
            let mut c = vec![Gv::ZERO; 3];
            for d in 0..ndim {
                c[axes[d]] = g.centroid[d];
            }
            c
        }
        None => Vec::new(),
    };
    let lapse = gv_lapse_weight(coords, spacetime, &coord_centroid);
    // one divergence per REQUESTED component (the predictor evaluates only the out-of-plane set),
    // returned in `comps` order.
    comps
        .iter()
        .map(|&c| {
            let div = match &oop {
                Some((co, OopDiv::Plain)) if c == *co => bcell_flux_div_plain_gv(c, ndim, spacing),
                Some((co, OopDiv::Curl(g))) if c == *co => {
                    bcell_flux_div_gv(c, ndim, &Some(g.clone()), &dx)
                }
                _ => bcell_flux_div_gv(c, ndim, &geo, &dx),
            };
            match lapse {
                Some(a) => a * div,
                None => div,
            }
        })
        .collect()
}

/// the OUT-OF-PLANE (non-CT) magnetic components for a chart: the B-vector slots whose coordinate
/// is NOT one of the grid axes. those live on staggered faces and are re-derived cell-centered by
/// `bcell_from_bface = interp(bface)`, so the predictor must leave them alone; the complement — the
/// out-of-plane components — have no face to curl and ARE evolved here as cell-centered conserved
/// variables. cartesian `[0..ndim)` grid -> `[ndim..ncomp)`; cyl r-z (axes [0,2])
/// -> {phi=1}; sph r-theta (axes [0,1]) -> {phi=2}; a fully-gridded 3D chart -> empty.
fn oop_components(ncomp: usize, axes: &[usize]) -> Vec<usize> {
    (0..ncomp).filter(|c| !axes.contains(c)).collect()
}


/// the RMHD cell-B FLUX PREDICTOR (Euler): `bcell[c] -= dt*div(bflux_c)`, in-place, for the
/// OUT-OF-PLANE components ONLY (`oop_components`). those are the genuinely cell-centered magnetic
/// slots — no staggered face, so not CT-evolved (reduced-dimension MHD). the
/// in-plane components are re-derived by `bcell_from_bface = interp(bface)` and must NOT be
/// flux-evolved here: their transient predictor value poisons the FOFC/c2p recoverability probe once
/// the magnetic-energy patch is gone. a fully-gridded chart (3D)
/// has no out-of-plane component and yields an EMPTY kernel — its dispatch is elided at `ndim==ncomp`.
pub fn rmhd_bcell_godunov_euler_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let oop = oop_components(ncomp, axes);
    let bc: Vec<Gv> = oop.iter().map(|&c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    // pin the ndim*|oop| induction-flux inputs in d-outer/c-inner order (the positional dispatch
    // order [bf_0_c, bf_1_c, ..]) before bcell_flux_div_gv reads them (it loops d).
    for d in 0..ndim {
        for &c in &oop {
            gv_register_field(&format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
        }
    }
    let dt = Gv::scalar("dt");
    let divs = bcell_flux_divs_gv(coords, spacetime, spacing, ndim, &oop, ncomp, axes);
    let writes = (0..oop.len())
        .map(|i| {
            let c = oop[i];
            let bnew = bc[i] - dt * divs[i];
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bnew.node())
        })
        .collect();
    (end_trace(), writes)
}


/// the RMHD cell-B FLUX PREDICTOR (RK2 stage 2): `bcell[c] = 0.5*(bcell_n[c] + (bcell[c] -
/// dt*div(bflux_c)))`, in-place, for the OUT-OF-PLANE components ONLY (`oop_components`; see the
/// Euler predictor). a fully-gridded chart (3D) yields an EMPTY kernel (dispatch elided).
pub fn rmhd_bcell_godunov_rk2_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let oop = oop_components(ncomp, axes);
    let bcn: Vec<Gv> = oop.iter().map(|&c| Gv::field(&format!("bcn_{c}"), FieldRef::BCellN(c as u8))).collect();
    let bc: Vec<Gv> = oop.iter().map(|&c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    for d in 0..ndim {
        for &c in &oop {
            gv_register_field(&format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
        }
    }
    let dt = Gv::scalar("dt");
    let half = Gv::from_f64(0.5);
    let divs = bcell_flux_divs_gv(coords, spacetime, spacing, ndim, &oop, ncomp, axes);
    let writes = (0..oop.len())
        .map(|i| {
            let c = oop[i];
            let bc_star = bc[i] - dt * divs[i];
            let bnew = half * (bcn[i] + bc_star);
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bnew.node())
        })
        .collect();
    (end_trace(), writes)
}


/// the cell-B godunov geometry: curvilinear or curved -> the gv cell geometry (area-weighted
/// div); flat cartesian -> the uniform `dx_d` scalars. a curved CARTESIAN chart (kerr-schild)
/// still carries the (flat-equal) cartesian geometry so the lapse weight has a centroid to
/// evaluate at — its covariant measure `alpha sqrt(gamma) = 1` equals the coordinate volume.
fn bcell_godunov_geom(coords: Coords, spacetime: Spacetime, spacing: &[Spacing], ndim: usize, axes: &[usize]) -> (Option<CellGeometryGv>, Vec<Gv>) {
    if coords == Coords::Cartesian && spacetime == Spacetime::Minkowski {
        (None, (0..ndim).map(|d| Gv::scalar(&format!("dx_{d}"))).collect())
    } else {
        // axes maps grid axis -> coordinate (identity for sph/3d-cyl; [0,2] for cyl r-z) so the
        // area-weighted divergence uses the right radial axis for the cylindrical metric. a
        // curved spacetime takes the COVARIANT (alpha sqrt(gamma)) measure — the mag rows are
        // densitized conserved laws of the same form as the gas (d_t(sqrt(g) B) + coordinate
        // divergence), exactly like the gas godunov's geometry selection.
        let g = match spacetime {
            Spacetime::Minkowski => cell_geometry_gv(coords, spacing, axes, ndim),
            Spacetime::Kerr => cell_geometry_covariant_gv(
                coords, spacing, axes, ndim, Some(Gv::scalar("kerr_spin")),
            ),
            _ => cell_geometry_covariant_gv(coords, spacing, axes, ndim, None),
        };
        (Some(g), Vec::new())
    }
}

