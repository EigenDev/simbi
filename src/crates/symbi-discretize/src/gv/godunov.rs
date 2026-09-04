// =============================================================================
// godunov.rs
//
// the conserved-update godunov family: snapshot, ssp stage, fused sources, and the unified dag-application operator.
// =============================================================================

use super::*;
use symbi_algebra::Tensor;
use symbi_carrier::Dual;
use symbi_geometry::grhd_source::{grhd_covariant_source, grmhd_covariant_source};
use symbi_geometry::{
    KerrKS, KerrKSCartesian, KerrKSCylindrical, SchwarzschildKS, SchwarzschildKSCartesian,
    SchwarzschildKSCylindrical,
};
use symbi_ir::{CtCellCt, CtFaceCt, PhysComp};
use symbi_ir::{KernelProgram, KernelWrite, KernelWrites, trace_kernel};
use symbi_source_compile::{AdmittedSources, BoundaryPrescription};

/// snapshot `u_n = cons` — a pure pointwise copy (the RK2 stage-0 hold), geometry-independent
/// (works for every coord system). copies the energy too when `has_energy`. write root == the
/// read field node (a direct buffer copy).
pub fn snapshot_gv(ncomp: usize, has_energy: bool) -> KernelProgram {
    trace_kernel(|cx| {
        let den = cx.field("cons_den", FieldRef::cons_den());
        let mom: Vec<Gv> = (0..ncomp)
            .map(|k| cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
            .collect();
        let nrg = has_energy.then(|| cx.field("cons_nrg", FieldRef::cons_nrg()));
        let mut writes = vec![KernelWrite::new("u_n_den", FieldRef::un_den(), den.node())];
        for (k, m) in mom.iter().enumerate() {
            writes.push(KernelWrite::new(
                format!("u_n_mom_{k}"),
                FieldRef::un_mom(k as u8),
                m.node(),
            ));
        }
        if let Some(n) = nrg {
            writes.push(KernelWrite::new("u_n_nrg", FieldRef::un_nrg(), n.node()));
        }
        writes
    })
}

/// a componentwise conserved-field copy `dst = src` over the gas conserved (den, mom[k], nrg?).
/// used to (a) restore `cons <- u_stage` so the first-order redo reconstructs from the physical
/// stage-input state, and (b) save the high-order per-direction fluxes before the redo overwrites the
/// live flux buffers (both are ConsFields). explicit-field dispatch: slots `s_*` (source) -> `d_*`
/// (dest).
pub fn fofc_copy_gv(ncomp: usize, has_energy: bool) -> KernelProgram {
    trace_kernel(|cx| {
        let mut writes = KernelWrites::new();
        let mut cp = |name: &str| {
            let v = cx.field(&format!("s_{name}"), &format!("s_{name}"));
            writes.push(KernelWrite::new(
                format!("d_{name}"),
                format!("d_{name}"),
                v.node(),
            ));
        };
        cp("den");
        for k in 0..ncomp {
            cp(&format!("mom_{k}"));
        }
        if has_energy {
            cp("nrg");
        }
        writes
    })
}

/// the TroubledCell decode: materialize the fallback flag from the
/// authoritative C2pStatus channel the recovery kernel wrote — zero accepted,
/// nonzero rejected — as the exact 0/1 mask the splice kernels consume. the
/// recovery classified this very state and nothing mutates the primitives
/// between the recovery and the fallback pass, so the decode carries the same
/// fact; classification lives with the recovery, this only re-encodes it.
pub fn fofc_flag_from_status_gv() -> KernelProgram {
    trace_kernel(|cx| {
        let status = cx.field("status", "status");
        let flag = Gv::select(status.cmp_eq(Gv::ZERO), Gv::ZERO, Gv::ONE);
        vec![KernelWrite::new("flag", "flag", flag.node())]
    })
}

/// keep only the flagged cells whose admissibility carries information about the timestep: drop
/// everything on the causally disconnected side of the mask surface. the outer horizon
/// `r_+ = M + sqrt(M^2 - a^2)` is one-way on a horizon-penetrating chart, so the exterior's
/// stability is set by exterior cells alone; and an excised cell is frozen at the vacuum floor,
/// numerical padding that can sit arbitrarily close to the admissible boundary forever. the
/// threshold is therefore the larger of `r_+` and the excision surface, tested on the kerr-schild
/// radius level set — the same surface and the same test the source-admissibility CFL masks on, so
/// the two agree cell for cell.
/// cartesian kerr-schild charts only; every other chart passes each flag through unchanged.
pub fn fofc_exterior_flag_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
) -> KernelProgram {
    trace_kernel(|cx| {
        let ndim = axes.len();
        let centroid = cell_geometry_gv(cx, coords, spacing, axes, ndim).centroid;
        let mut position = [Gv::ZERO; 3];
        for (grid_axis, &coordinate) in axes.iter().enumerate() {
            position[coordinate] = centroid[grid_axis];
        }
        let bad = cx.field("bad", "bad");
        let exterior = if coords == Coords::Cartesian
            && matches!(spacetime, Spacetime::SchwarzschildKS | Spacetime::KerrKS)
        {
            let spin = if spacetime == Spacetime::KerrKS {
                cx.scalar("kerr_spin")
            } else {
                Gv::ZERO
            };
            let mass = cx.scalar("schwarzschild_mass");
            let r_plus = mass + (mass * mass - spin * spin).max(Gv::ZERO).sqrt();
            let r_mask = r_plus.max(cx.scalar("excision_radius"));
            let excised = symbi_ib::excise::ks_excised(&position, spin, r_mask);
            Gv::select(excised, Gv::ZERO, bad)
        } else {
            bad
        };
        vec![KernelWrite::new("exterior", "exterior", exterior.node())]
    })
}

/// ghost-band fail-loud probe: write 1 where the density is non-finite (NaN or +-inf via
/// `(rho - rho) != 0`), else 0. run over the allocated domain (interior + ghosts): first-order flux
/// correction keeps the interior finite and acts on the interior alone, so a poisoned boundary
/// (a driven-inflow expression producing NaN, a broken bc) leaves a non-finite ghost beyond FOFC's
/// reach. a max-reduce > 0 forces the CFL rate to +inf (dt -> 0, the driver halts) — the fail-loud that
/// survives FOFC recovery. density-only, so regime- and energy-independent (one kernel per dimension);
/// a poison in any primitive reaches the density within one c2p / flux divergence.
pub fn state_finite_probe_gv() -> KernelProgram {
    trace_kernel(|cx| {
        let rho = cx.field("prim_rho", FieldRef::PrimRho);
        let flag = Gv::select((rho - rho).cmp_eq(Gv::ZERO), Gv::ZERO, Gv::ONE);
        vec![KernelWrite::new("flag", FieldRef::Scratch, flag.node())]
    })
}

/// the FOFC freeze tier select (the face-based redo's only per-cell state replacement): keep the
/// live spliced first-order conserved (`x_*`) where it is physical, else freeze to the stage-input
/// state `u_stage` (`us_*`) — the pre-godunov conserved, admissible from stage entry, so the final
/// c2p converges on it. the face-based splice already made every kept cell conservative (one flux per
/// face); this handles only the rare cell whose every flux update lands outside the admissible set,
/// holding its stage input so the state stays finite. that single-cell hold is the documented
/// conservation waiver — it discards the cell's flux exchange, bounded by the persistent-freeze
/// fail-loud. only the conserved is chosen; the primitive is re-derived by the c2p that follows.
pub fn fofc_select_gv(ncomp: usize, has_energy: bool) -> KernelProgram {
    trace_kernel(|cx| {
        // finite and positive: (v - v) is 0 for a finite value and NaN for NaN or +-inf (inf - inf =
        // NaN), so cmp_eq(0) rejects every non-finite value; the > 0 rejects a vacuum/negative one. a
        // "physical" cell is one whose density (and pressure, when modeled) passes both.
        // finiteness probes as named-brand fns: a local closure cannot name the trace
        // brand, and annotating the elided lifetime mints regions invariance rejects.
        fn finite<'t>(v: Gv<'t>) -> symbi_ir::GvMask<'t> {
            (v - v).cmp_eq(Gv::ZERO)
        }
        fn finite_pos<'t>(v: Gv<'t>) -> symbi_ir::GvMask<'t> {
            finite(v) & v.cmp_gt(Gv::ZERO)
        }
        let x_rho = cx.field("x_rho", "x_rho");
        let mut physical = if has_energy {
            let x_pre = cx.field("x_pre", "x_pre");
            finite_pos(x_rho) & finite_pos(x_pre)
        } else {
            finite_pos(x_rho)
        };
        // the full state vector: each spliced velocity is tested for finiteness (its sign is physical),
        // so a non-finite momentum riding an otherwise finite density/pressure freezes to the
        // admissible stage input.
        for k in 0..ncomp {
            let p = format!("x_vel_{k}");
            physical = physical & finite(cx.field(&p, &p));
        }
        let mut writes = KernelWrites::new();
        // the live cons (`x_*`) is read+write in place: it holds the spliced first-order result and is
        // overwritten with the chosen tier. one slot per component (read path == write path) so the IR
        // dedups it to a single in-place binding (the CT-`b` pattern), keeping input and output on a
        // single binding.
        let mut sel_inplace = |comp: &str, us| {
            let path = format!("x_{comp}");
            let x = cx.field(&path, &path);
            let chosen = Gv::select(physical, x, us);
            writes.push(KernelWrite::new(path.clone(), path, chosen.node()));
        };
        sel_inplace("den", cx.field("us_den", "us_den"));
        for k in 0..ncomp {
            sel_inplace(
                &format!("mom_{k}"),
                cx.field(&format!("us_mom_{k}"), &format!("us_mom_{k}")),
            );
        }
        if has_energy {
            sel_inplace("nrg", cx.field("us_nrg", "us_nrg"));
        }
        // the FreezeApplied channel: this select is the component performing
        // the freeze, so it reports the act — 1 where the candidate was
        // rejected and the stage-input parachute deployed, else 0.
        writes.push(KernelWrite::new(
            "freeze",
            "freeze",
            Gv::select(physical, Gv::ZERO, Gv::ONE).node(),
        ));
        writes
    })
}

/// the freeze-tier select with the immersed-body source composed inline — the lazy, buffer-free
/// form that carries a frozen cell's body gravity/accretion with it. identical to `fofc_select_gv`
/// except the freeze parachute is the stage input evolved by the body source in registers,
/// `u_stage + dt*body(u_stage)` (via `body_evolved_gv` / `body_evolved_iso_gv`). the body delta and
/// the c2p pressure that guards it are closed forms of `us_*`, so both stay in registers.
/// the guard preserves the freeze tier's physical-parachute invariant — a body
/// kick that would drive the parachute unphysical (a strong pull on a low-internal-energy cell) falls
/// back to the bare stage input. `has_energy` selects the adiabatic (evolves nrg, eos param = gamma,
/// pressure guard) vs the isothermal (density + momentum, eos param = cs, `p = cs^2 * rho > 0` so the
/// density guard alone) form. body-free regimes keep `fofc_select_gv`.
pub fn fofc_select_with_body_gv(
    ncomp: usize,
    n_bodies: usize,
    coords: Coords,
    ndim: usize,
    axes: &[usize],
    has_energy: bool,
) -> KernelProgram {
    trace_kernel(|cx| {
        // finiteness probes as named-brand fns: a local closure cannot name the trace
        // brand, and annotating the elided lifetime mints regions invariance rejects.
        fn finite<'t>(v: Gv<'t>) -> symbi_ir::GvMask<'t> {
            (v - v).cmp_eq(Gv::ZERO)
        }
        fn finite_pos<'t>(v: Gv<'t>) -> symbi_ir::GvMask<'t> {
            finite(v) & v.cmp_gt(Gv::ZERO)
        }
        // the spliced first-order result's physicality, identical to `fofc_select_gv`: density always,
        // pressure only when the energy is modeled (iso keeps p in a separate cs^2 buffer, so its
        // select gates on the density alone), plus finiteness of every velocity component.
        let x_rho = cx.field("x_rho", "x_rho");
        let mut physical_fo = if has_energy {
            finite_pos(x_rho) & finite_pos(cx.field("x_pre", "x_pre"))
        } else {
            finite_pos(x_rho)
        };
        for k in 0..ncomp {
            let p = format!("x_vel_{k}");
            physical_fo = physical_fo & finite(cx.field(&p, &p));
        }
        // the stage input evolved by the body source, inline in registers within this kernel.
        let dt = cx.scalar("dt");
        let us_den = cx.field("us_den", "us_den");
        let us_mom: Vec<Gv> = (0..ncomp)
            .map(|k| cx.field(&format!("us_mom_{k}"), &format!("us_mom_{k}")))
            .collect();
        // the body-evolved conserved parachute + its physicality, energy-aware. `us_nrg` is bound only in
        // the adiabatic form so the isothermal kernel manifest carries the density and momentum alone.
        let (b_den, b_mom, b_nrg, usb_ok) = if has_energy {
            let gamma = cx.scalar("gamma");
            let us_nrg = cx.field("us_nrg", "us_nrg");
            let (b_den, b_mom, b_nrg, _drain) = crate::gv_immersed::body_evolved_gv(
                cx, us_den, &us_mom, us_nrg, dt, gamma, n_bodies, coords, ndim, ncomp, axes,
            );
            // guard: the parachute must itself be physical. rho = den (newtonian, W = 1); the adiabatic
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
            let cs = cx.scalar("cs");
            let us_pre = cs * cs * us_den;
            let (b_den, b_mom) = crate::gv_immersed::body_evolved_iso_gv(
                cx, us_den, &us_mom, us_pre, dt, n_bodies, coords, ndim, ncomp, axes,
            );
            let usb_ok = finite_pos(b_den);
            (b_den, b_mom, None, usb_ok)
        };
        let parachute = |ub, us| Gv::select(usb_ok, ub, us);
        // main select in place: `x_*` (the spliced first-order cons) is kept where physical, else frozen
        // to the guarded body-evolved stage input.
        let mut writes = KernelWrites::new();
        let mut sel = |comp: &str, par| {
            let path = format!("x_{comp}");
            let x = cx.field(&path, &path);
            let chosen = Gv::select(physical_fo, x, par);
            writes.push(KernelWrite::new(path.clone(), path, chosen.node()));
        };
        sel("den", parachute(b_den, us_den));
        for k in 0..ncomp {
            sel(&format!("mom_{k}"), parachute(b_mom[k], us_mom[k]));
        }
        if let Some((us_nrg, b_nrg)) = b_nrg {
            sel("nrg", parachute(b_nrg, us_nrg));
        }
        // the FreezeApplied channel: 1 where the candidate was rejected and a
        // parachute deployed — the body-evolved one or, when its guard fails,
        // the bare stage input; both waive the cell's conservation, so both
        // are the freeze act this select reports.
        writes.push(KernelWrite::new(
            "freeze",
            "freeze",
            Gv::select(physical_fo, Gv::ZERO, Gv::ONE).node(),
        ));
        writes
    })
}

/// the FOFC face-based flux splice for axis `dir`: choose, per interior face, the first-order flux
/// (`fo_*`, the redone HLLE/rusanov flux held in the live `fields.flux[dir]`) where either adjacent
/// cell is flagged for fallback, else the high-order flux (`ho_*`, saved in `flux_ho[dir]` before the
/// redo overwrote the live buffer). the finite-volume convention stores the axis-`dir` flux at cell
/// `c` on the low face of `c` (between `c - e_dir` and `c`), so the face is first-order iff
/// `flag[c] > 0 OR flag[c - e_dir] > 0`. the `fo_*` slot is read+write in place (the live flux
/// buffer), so after the splice every face carries a single flux value and the following godunov
/// telescopes conservatively across every fallback boundary. componentwise over the conserved flux
/// (den, mom[k], nrg?); the flag is a plain 0/1 cell field with boundary-consistent ghosts.
pub fn fofc_splice_gv(ndim: usize, dir: usize, ncomp: usize, has_energy: bool) -> KernelProgram {
    trace_kernel(|cx| {
        let nd = ndim as u8;
        let d = dir as u8;
        let flag_c = cx.field("flag", CtCellCt::FofcFlag);
        let flag_lo = cx.field_shifted("flag", CtCellCt::FofcFlag, nd, d, -1);
        let face_fo = flag_c.cmp_gt(Gv::ZERO) | flag_lo.cmp_gt(Gv::ZERO);
        let mut writes = KernelWrites::new();
        let mut splice = |comp: &str| {
            let fo_name = format!("fo_{comp}");
            let fo = cx.field(&fo_name, &fo_name);
            let ho = cx.field(&format!("ho_{comp}"), &format!("ho_{comp}"));
            let chosen = Gv::select(face_fo, fo, ho);
            writes.push(KernelWrite::new(fo_name.clone(), fo_name, chosen.node()));
        };
        splice("den");
        for k in 0..ncomp {
            splice(&format!("mom_{k}"));
        }
        if has_energy {
            splice("nrg");
        }
        writes
    })
}

/// the FOFC face-based induction-flux splice for axis `dir`: the magnetic mirror of
/// `fofc_splice_gv`. per B-component `c` in `0..ncomp`, choose the first-order induction flux
/// (`fo_bflux_{c}`, the redone flux in the live `bflux[dir][c]`) on faces adjacent to a flagged cell,
/// else the high-order flux (`ho_bflux_{c}`, saved in `bflux_ho[dir][c]`). the axis-`dir` induction
/// flux shares the gas flux's face indexing (stored at cell `c` on the low face of `c`), so the face
/// is first-order iff `flag[c] > 0 OR flag[c - e_dir] > 0` — the identical mask to the gas splice.
/// `fo_bflux_{c}` is read+write in place; the spliced induction flux feeds the cell-B predictor (HO
/// off the fallback region, FO on it) and the Contact FO edge EMF.
pub fn fofc_bflux_splice_gv(ndim: usize, dir: usize, ncomp: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let nd = ndim as u8;
        let d = dir as u8;
        let flag_c = cx.field("flag", CtCellCt::FofcFlag);
        let flag_lo = cx.field_shifted("flag", CtCellCt::FofcFlag, nd, d, -1);
        let face_fo = flag_c.cmp_gt(Gv::ZERO) | flag_lo.cmp_gt(Gv::ZERO);
        let mut writes = KernelWrites::new();
        for c in 0..ncomp {
            let fo_name = format!("fo_bflux_{c}");
            let fo = cx.field(&fo_name, CtFaceCt::BFluxFirstOrder(PhysComp::new(c)));
            let ho = cx.field(
                &format!("ho_bflux_{c}"),
                CtFaceCt::BFluxHighOrder(PhysComp::new(c)),
            );
            let chosen = Gv::select(face_fo, fo, ho);
            writes.push(KernelWrite::new(
                fo_name,
                CtFaceCt::BFluxFirstOrder(PhysComp::new(c)),
                chosen.node(),
            ));
        }
        writes
    })
}

/// the single mass-law godunov step to a separate output buffer:
/// `rho_new = rho - dt*div(mass_flux)`. cartesian-uniform or curvilinear (area-weighted).
/// write -> `cons.den_new`.
pub fn godunov_mass_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
) -> KernelProgram {
    trace_kernel(|cx| {
        let dt = cx.scalar("dt");
        let geo = (!is_cartesian_uniform(coords, spacing))
            .then(|| cell_geometry_gv(cx, coords, spacing, axes, ndim as usize));
        let rho = cx.field("rho", FieldRef::cons_den());
        let rho_new = rho - dt * gv_divergence(cx, "mass_flux", ndim, &geo, spacing);
        let writes = vec![KernelWrite::new("rho_new", "cons.den_new", rho_new.node())];
        writes
    })
}

/// the in-place SSP shu-osher stage update `cons = a0*u_n + ac*fe(cons)`, where the
/// forward-euler operator is `fe(u) = u - dt*div(F) (+ dt*S_geom)`. one builder for every
/// explicit SSP scheme: the per-stage convex coefficients `(a0, ac)` arrive as runtime
/// scalars, so a single compiled kernel serves forward-euler `[(0,1)]`, SSP-RK2
/// `[(0,1),(1/2,1/2)]`, and SSP-RK3 `[(0,1),(3/4,1/4),(1/3,2/3)]` — the integrator arrives as
/// data at runtime. forward-euler is the `(a0,ac)=(0,1)` instantiation (the `a0*u_n` term reads
/// the snapshot held by `snapshot_gv` and multiplies it by 0).
///
/// mass + one scalar law per momentum component (+ energy when `has_energy`). cartesian =
/// unweighted divergence alone; curvilinear = area-weighted divergence + the geometric
/// momentum `source` carried inside the forward-euler stage. write path == input path (in
/// place). EOS- and geom-generic.
///
/// this is the no-overlay case of [`godunov_stage_gv_with_fused_sources`] — the full stage
/// body lives there, and the empty source slice traces exactly the plain SSP stage (the splice
/// helper short-circuits on an empty overlay list, so the trace keeps only live vocabulary
/// nodes). kept as a named entry point for the common no-source case.
pub fn godunov_stage_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
) -> KernelProgram {
    godunov_stage_gv_with_fused_sources(
        coords,
        spacetime,
        spacing,
        axes,
        ndim,
        ncomp,
        has_energy,
        source,
        &AdmittedSources::none(),
        false,
    )
}

/// per-field NodeId contributions from a list of spec sources, bucketed by
/// `target_field`. consumed by `godunov_stage_gv_with_fused_sources` — the spec
/// vocabulary is spliced once, then each conserved law adds its bucket inside the
/// forward-euler stage.
///
/// **structural shape contract**: spliced outputs carry the expected per-target
/// arity (1 for den/nrg, D for mom); a violating spec panics, which prevents
/// a silent wrong-component write.
struct FusedContribs {
    /// each entry is a `S_den` NodeId to add to `rho_new`.
    den: Vec<NodeId>,
    /// `mom[k]` is the list of `S_mom_k` NodeIds for momentum component k.
    mom: Vec<Vec<NodeId>>,
    /// each entry is a `S_nrg` NodeId to add to `nrg_new`.
    nrg: Vec<NodeId>,
    /// `mag[k]` is the per-component cell-B prescription, only for a driven-boundary
    /// (`WriteMode::Assign`) MHD `bcell` slot. empty for hydro and for the
    /// accumulate (godunov source) path — the conservation-law lifts target den/mom/nrg alone.
    mag: Vec<Vec<NodeId>>,
    /// the dye concentration prescription, only for a driven-boundary (`WriteMode::Assign`)
    /// `chi` slot: injected fluid carries a concentration supplied by the boundary itself. empty
    /// for the accumulate path, where the dye moves by the mass flux alone.
    chi: Vec<NodeId>,
}

/// fused-source splice helper. runs inside an open Gv trace
/// (the caller holds the open trace via its `TraceCx`). builds the shared primitive
/// vocabulary (`rho`, `vel_k`, lazy `x_k` <-> centroid), then splices every
/// spec into the trace and buckets the outputs by `target_field`. on an empty overlay list it
/// returns empty buckets and leaves the trace untouched — so the no-source `godunov_stage_gv`
/// wrapper traces exactly the plain SSP stage, keeping only live `mom/rho` vocabulary nodes.
fn splice_fused_sources_to_contribs<'t>(
    cx: TraceCx<'t>,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    geo: &Option<CellGeometryGv<'t>>,
    // the state vocabulary the DAG reads `rho`/`vel_k` from (`StateEnv`). `Some((rho,
    // mom))` binds them (sources read the stage/conserved state); `None` is a pure coordinate
    // prescription — a driven boundary, whose DAG outputs the state and reads coordinates alone.
    // `x_k` (centroid) + scalar params are bound regardless.
    state: Option<(Gv<'t>, &[Gv<'t>])>,
    // (target_field, built) pairs — the programs an admission witness (`AdmittedSources`, or
    // `BoundaryPrescription` for a coordinate prescription) carries, so this serves the AOT
    // bake and the runtime paths from one shape.
    sources: &[(String, symbi_source_compile::source_spec::SourceProgram)],
) -> FusedContribs {
    use std::collections::HashMap;

    if sources.is_empty() {
        return FusedContribs {
            den: Vec::new(),
            mom: vec![Vec::new(); ncomp],
            nrg: Vec::new(),
            mag: vec![Vec::new(); ncomp],
            chi: Vec::new(),
        };
    }

    // ----- shared primitive vocabulary, declared once; CSE collapses the
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
        // to the c2p-computed `prim.pre` field. at source-apply / fused-godunov time prim still holds
        // the SSP stage input, so this is consistent with rho/vel above. energy-bearing
        // regimes only — the `prim.pre` field exists there. bound when a source actually references
        // `pre` (mirrors `needs_position`): an unconditional bind adds a manifest `prim.pre` read that
        // duplicates the adiabatic godunov's own flux-reconstruction read -> input/output aliasing.
        let needs_pre = sources
            .iter()
            .any(|(_, b)| b.params().iter().any(|p| p == "pre"));
        if has_energy && needs_pre {
            shared_params.insert("pre".to_string(), cx.field("pre", FieldRef::PrimPre).node());
        }
    }
    // lazy centroid binding. `x_k` <-> cell centroid for specs
    // that declare position params (gravity, immersed bodies). walk the
    // spec params first to detect which axes are needed, then call
    // `cell_geometry_gv` (which declares `x_lo_k` / `dx_k` scalars in the
    // trace) only if at least one axis is referenced. position-free specs
    // keep the prior scalar manifest unchanged.
    let needs_position = sources.iter().any(|(_, built)| {
        built
            .params()
            .iter()
            .any(|p| (0..(ndim as usize)).any(|k| *p == format!("x_{k}")))
    });
    if needs_position {
        let centroid_geo = geo
            .clone()
            .unwrap_or_else(|| cell_geometry_gv(cx, coords, spacing, axes, ndim as usize));
        for k in 0..(ndim as usize) {
            shared_params.insert(format!("x_{k}"), centroid_geo.centroid[k].node());
        }
    }

    // scalar-leaf cache so the same spec param across multiple overlays
    // (e.g., `g_ext_0` in the mom + nrg specs of uniform_acceleration)
    // resolves to a single Gv leaf — runtime fills one scalar, CSE collapses.
    let mut scalar_leaves: HashMap<String, NodeId> = HashMap::new();
    let mut out = FusedContribs {
        den: Vec::new(),
        mom: vec![Vec::new(); ncomp],
        nrg: Vec::new(),
        mag: vec![Vec::new(); ncomp],
        chi: Vec::new(),
    };
    for (target_field, built) in sources {
        let mut name_to_node = shared_params.clone();
        for pname in built.params() {
            if name_to_node.contains_key(pname) {
                continue;
            }
            let nid = *scalar_leaves
                .entry(pname.clone())
                .or_insert_with(|| cx.scalar(pname).node());
            name_to_node.insert(pname.clone(), nid);
        }
        let spliced = cx.with_trace(|t| built.splice_into(t.graph(), &name_to_node));
        match target_field.as_str() {
            "den" => {
                assert_eq!(
                    spliced.len(),
                    1,
                    "splice_fused_sources: den overlay must emit 1 scalar, got {}",
                    spliced.len()
                );
                out.den.push(spliced[0]);
            }
            "mom" => {
                assert_eq!(
                    spliced.len(),
                    ncomp,
                    "splice_fused_sources: mom overlay must emit {ncomp} components, got {}",
                    spliced.len()
                );
                for k in 0..ncomp {
                    out.mom[k].push(spliced[k]);
                }
            }
            "nrg" => {
                assert!(
                    has_energy,
                    "splice_fused_sources: nrg overlay requires has_energy=true"
                );
                assert_eq!(
                    spliced.len(),
                    1,
                    "splice_fused_sources: nrg overlay must emit 1 scalar, got {}",
                    spliced.len()
                );
                out.nrg.push(spliced[0]);
            }
            // cell-B prescription (MHD driven boundary): the ncomp-component bcell vector.
            // only valid in the Assign (prescription) mode — the conservation-law source lifts
            // target den/mom/nrg alone, so the accumulate path asserts mag stays empty.
            "bcell" => {
                assert_eq!(
                    spliced.len(),
                    ncomp,
                    "splice_fused_sources: bcell overlay must emit {ncomp} components, got {}",
                    spliced.len()
                );
                for k in 0..ncomp {
                    out.mag[k].push(spliced[k]);
                }
            }
            // dye prescription (driven boundary): one scalar concentration for the injected fluid.
            // Assign mode only — the dye rides the mass flux, which is its whole transport.
            "chi" => {
                assert_eq!(
                    spliced.len(),
                    1,
                    "splice_fused_sources: chi overlay must emit 1 scalar, got {}",
                    spliced.len()
                );
                out.chi.push(spliced[0]);
            }
            other => panic!("splice_fused_sources: unsupported target_field {other:?}"),
        }
    }
    out
}

/// the SSP shu-osher stage update with a fused list of spec sources — the
/// `godunov_stage_gv` body (runtime `(a0, ac)` convex coefficients, `cons = a0*u_n + ac*fe`)
/// with the spec contributions spliced into the forward-euler operator:
/// `fe(u, div, src) = u - dt*div + dt*(geo_src + \sum spec_src)`. one launch folds flux
/// divergence + geometric source + every user overlay + the integrator combine. the dispatch
/// `{prefix}_godunov_stage_with_{slug}_{D}d` resolves here.
///
/// the spec contributions live inside `fe`, so the stage's `ac` weight multiplies them — the
/// same convex coefficient that weights the flux divergence — which is exactly the SSP
/// source treatment (`ac*dt*S` per stage). pass an empty slice for the no-overlay variant.
///
/// the body-free entry: the bake-time producer for the AOT `_with_{slug}` variants, whose
/// sources are admitted specs (`AdmittedSources::admit_specs`), delegating to
/// [`godunov_stage_gv_with_fused_bodies`] with no immersed body. the immersed body belongs to
/// the runtime-source path, which threads the real count through the core.
pub fn godunov_stage_gv_with_fused_sources(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    sources: &AdmittedSources,
    // when this stage is fused with the cell-B predictor, the magnetic geo source reads cell-B
    // via the predictor's `bc_k` key so try_fuse merges the two reads onto one binding.
    // the plain (unfused) stage passes false -> reads `prim.mag[k]`.
    mag_from_bcell: bool,
) -> KernelProgram {
    godunov_stage_gv_with_fused_bodies(
        coords,
        spacetime,
        spacing,
        axes,
        ndim,
        ncomp,
        has_energy,
        source,
        sources,
        mag_from_bcell,
        0,
    )
}

/// the SSP stage core over admitted sources — the `SourceProgram` values the contribution
/// door admitted, paired with their target field. the AOT bake feeds admitted specs; the
/// runtime user-source CPU fusion feeds `RuntimeSource`'s admitted programs. one trace, both
/// paths — the godunov+source lowering lives in one place.
#[allow(clippy::too_many_arguments)]
pub fn godunov_stage_gv_with_fused_bodies(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    sources: &AdmittedSources,
    mag_from_bcell: bool,
    // immersed-body count. > 0 wraps the post-combine cons with `body_evolved_gv` (gravity +
    // accretion drain) at weight `ac*dt`, so the single fused sweep equals `plain godunov +
    // source_apply + body_source`, in that order, bit-for-bit. 0 leaves the update body-free.
    n_bodies: usize,
) -> KernelProgram {
    godunov_stage_gv_with_fused_bodies_and_geo_weight(
        coords,
        spacetime,
        spacing,
        axes,
        ndim,
        ncomp,
        has_energy,
        source,
        sources,
        mag_from_bcell,
        n_bodies,
        false,
    )
}

/// the ssp stage core with an optional per-cell geometric-source multiplier.
///
/// `weighted_geo_source` is reserved for the grmhd fofc replay: the flux divergence,
/// mesh dilution, user sources, and immersed-body operators remain unchanged while the
/// local metric source is multiplied by the typed scratch field. the ordinary stage
/// delegates here with `false`, so its graph and abi are unchanged.
#[allow(clippy::too_many_arguments)]
pub fn godunov_stage_gv_with_fused_bodies_and_geo_weight(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    sources: &AdmittedSources,
    mag_from_bcell: bool,
    n_bodies: usize,
    weighted_geo_source: bool,
) -> KernelProgram {
    trace_kernel(|cx| {
        let dt = cx.scalar("dt");
        let a0 = cx.scalar("a0");
        let ac = cx.scalar("ac");
        // the SSP source weight. computed as `ac*dt` so it is bit-identical to the standalone
        // `source_apply_gv` pass's `dt` scalar (the driver fills that with `ac*sim.dt` — the same IEEE
        // f64 product). this is what makes `fused == plain godunov + source_apply` bit-exact: the
        // user source rides as a separate post-combine term at this weight; folding it into the
        // `ac*fe` multiply would distribute the rounding differently.
        let ac_dt = ac * dt;
        // flat spacetime: the physical (orthonormal) finite-volume geometry. curved (GR): the
        // covariant geometry — coordinate-form angular face weights (the alpha sqrt(gamma) measure),
        // matching the covariant momentum S_i and the contravariant fluxes v^i; the orthonormal
        // angular weights would leave every theta-direction force on S_theta short by a factor r.
        // radial faces and the volume coincide, so 1D radial GR is bit-identical.
        // a curved spacetime always needs the geometry (the metric position + the alpha sqrt(gamma)
        // densitization measure), even on a cartesian-uniform grid where flat hydro skips it.
        let geo = (!is_cartesian_uniform(coords, spacing) || spacetime != Spacetime::Minkowski)
            .then(|| {
                match spacetime {
                    Spacetime::Minkowski => {
                        cell_geometry_gv(cx, coords, spacing, axes, ndim as usize)
                    }
                    // spinning kerr: the densitized measure is Sigma sin(theta) — the spin rides the
                    // `kerr_spin` kernel scalar into the face/volume moments.
                    Spacetime::KerrKS => cell_geometry_covariant_gv(
                        cx,
                        coords,
                        spacing,
                        axes,
                        ndim as usize,
                        Some(cx.scalar("kerr_spin")),
                    ),
                    _ => cell_geometry_covariant_gv(cx, coords, spacing, axes, ndim as usize, None),
                }
            });
        // GR hydro evolves the fully densitized state (gammie et al. 2003; stone et al. 2024 eq. 20):
        // U = sqrt(-g)[rho u^t, T^t_i, -(T^t_t + rho u^t)] with the flux carrying the same measure, so
        // the divergence is plain coordinate differencing with the lapse absorbed into the measure on
        // both sides, the geometry arrives through the pointwise connection source
        // (1/2) sqrt(-g) (d_i g_ab) T^ab, and the energy source is identically zero because the metric
        // is stationary. GR MHD keeps the area-weighted valencia path (its induction and CT seam carry
        // the area-weighted form).
        let densitized =
            spacetime != Spacetime::Minkowski && matches!(source, GeoSource::Hydro { .. });
        assert!(
            !densitized || (sources.is_empty() && n_bodies == 0),
            "a densitized GR state cannot take undensitized user-source or immersed-body rates"
        );
        let rho = cx.field("rho", FieldRef::cons_den());
        let mom: Vec<Gv> = (0..ncomp)
            .map(|k| cx.field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
            .collect();
        // on a curved background the covariant momentum S_i takes its inertial blocks from the
        // covariant stress-energy contraction below (the flat velocity-quadratic inertial treats the
        // components as flat), so the hydro geometric source keeps only its
        // discrete well-balanced pressure form `p (A_hi - A_lo) / V` — which cancels the pressure flux
        // divergence bit-exactly at a uniform-p hydrostatic state, while the analytic pressure block
        // `p d_j ln(alpha sqrt(gamma))` of the contraction cancels only to truncation order.
        // the ideal-MHD stress moves to the covariant contraction on the GR path too — the flat
        // Rmhd curvilinear source would double-count the inertia/tension with the flat contraction
        // for covariant S_i; only the gas-pressure discrete block stays.
        let source_discrete = match (spacetime, source) {
            (Spacetime::Minkowski, s) => s,
            (_, GeoSource::Hydro { .. }) | (_, GeoSource::Rmhd) => {
                GeoSource::Hydro { inertial: false }
            }
            (_, s) => s,
        };
        // under densitization the whole geometric momentum contribution — pressure block included —
        // arrives through the connection source below, so the discrete area-weighted form is retired.
        let src = (!densitized).then(|| geo.as_ref()).flatten().map(|g| {
            gv_geometric_source(
                cx,
                coords,
                axes,
                ndim as usize,
                ncomp,
                g,
                source_discrete,
                &mom,
                mag_from_bcell,
            )
        });

        let contribs = splice_fused_sources_to_contribs(
            cx,
            coords,
            spacing,
            axes,
            ndim,
            ncomp,
            has_energy,
            &geo,
            Some((rho, &mom)),
            sources.pairs(),
        );

        // the plain forward-euler stage carries the flux divergence + the (well-balanced)
        // geometric source; the user sources ride outside it, added after the combine.
        // `cons_new = a0*u_n + ac*fe`, identical to `godunov_stage_gv`. the homologous-mesh
        // dilution `-mesh_hdil * u` (with `mesh_hdil = ndim * a_dot / a`, the comoving
        // volume-growth rate) rides every conserved law; the static binding mesh_hdil = 0
        // subtracts an exact zero.
        let h_dil = cx.scalar("mesh_hdil");
        // GR densitization (valencia 3+1, static diagonal background): the spatial RHS — the flux
        // divergence + the geometric momentum source — is weighted by the lapse `alpha(x)`. the `u`
        // snapshot and the mesh-dilution term stay unweighted: they are the time / comoving parts.
        // flat spacetime -> `None` -> untouched, bit-identical (see `gv_lapse_weight`).
        // the coordinate-indexed cell centroid (r at slot 0) for the lapse alpha(x); only the
        // curvilinear path carries one (cartesian-uniform geo = None is always minkowski -> unused).
        let coord_centroid: Vec<Gv> = match &geo {
            Some(g) => {
                // the densitized law's cell average is over the plain coordinate volume, so its metric
                // sampling point is the arithmetic midpoint; the area-weighted law's average carries
                // the chart's volume element and reads the volume-weighted centroid.
                let mid = densitized.then(|| gv_cell_midpoints(cx, spacing, ndim as usize));
                let mut c = vec![Gv::ZERO; 3];
                for d in 0..(ndim as usize) {
                    c[axes[d]] = match &mid {
                        Some(m) => m[d],
                        None => g.centroid[d],
                    };
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
        // the cell lapse. on the valencia path it weights the spatial RHS; under densitization it is
        // one factor of the measure sqrt(-g) = alpha sqrt(det gamma), and the RHS weight is identically
        // 1 (sqrt(-g) sits on both the state and the flux, so the whole lapse is already placed).
        let cell_lapse = gv_lapse_weight(cx, coords, spacetime, &coord_centroid);
        let lapse = if densitized { None } else { cell_lapse };
        // coordinate-indexed metric position: each gridded coordinate at its centroid, each ungridded
        // coordinate at its chart symmetry default (spherical polar -> pi/2, else 0). a flat spacetime
        // leaves this `None`: its metric is constant, and a cartesian-uniform grid's centroid nodes are
        // absent from the trace.
        let x_cell: Option<Tensor<Gv, 3>> = (spacetime != Spacetime::Minkowski).then(|| {
            Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                if axes.contains(&c) {
                    coord_centroid[c]
                } else {
                    gv_ungridded_slot(coords, c)
                }
            }))
        });
        // the cell measure, evaluated at the metric's full spatial dimension so a reduced grid keeps
        // the suppressed directions' volume element. the state carries sqrt(det gamma), the flux and
        // the connection source carry sqrt(-g) = alpha sqrt(det gamma).
        let sqrt_gamma_cell = densitized.then(|| {
            gv_metric_volume_factor_at(
                cx,
                spacetime,
                coords,
                x_cell.expect("a curved spacetime has a metric position"),
            )
        });
        let sqrt_neg_g =
            sqrt_gamma_cell.map(|g| cell_lapse.expect("a curved spacetime has a lapse") * g);
        // the GR geodesic sources from the full covariant contraction `grhd_covariant_source`: the
        // per-coordinate momentum source S_j = (1/2) T^{mu nu} d_j g_{mu nu} and the energy source
        // S_tau, one forward-autodiff pass per axis at the metric's full spherical D = 3 (the metric
        // supplies its ADM line element and autodiff supplies every derivative).
        //
        // on the valencia path the momentum call takes p = 0 (the E-part only: gravity + covariant
        // centrifugal), because the pressure block `p d_j ln(alpha sqrt(gamma))` rides the discrete
        // well-balanced form in gv_geometric_source above, and the energy call takes the full p.
        // under densitization the connection source is the whole geometric contribution, so one pass
        // at the full pressure serves the momentum and the energy source is structurally zero.
        // GRMHD-ready: the EM stress just changes T^{mu nu}.
        let geodesic: Option<(Tensor<Gv, 3>, Gv)> = match spacetime {
            Spacetime::Minkowski => None,
            _ => {
                let mass = Dual::constant(cx.scalar("schwarzschild_mass")); // constant w.r.t. position
                let x = x_cell.expect("a curved spacetime has a metric position");
                // the effective inertia e = rho h W^2 for the covariant stress. reconstructed metric-free
                // as h D^2 / rho_prim (W = D/rho_prim) — independent of the energy variable, so it holds
                // whether the nrg slot stores the valencia tau (RMHD) or the killing energy (RHD, whose
                // stored value differs from D + tau + p). the mass slot is densitized on the GR
                // hydro path, so the baryon density D is recovered as cons_den / sqrt(det gamma).
                let p = cx.field("pre", FieldRef::PrimPre);
                let e = {
                    let prim_rho = cx.field("prim_rho", FieldRef::PrimRho);
                    let gamma_eos = cx.scalar("gamma");
                    let h_enth = Gv::ONE + gamma_eos / (gamma_eos - Gv::ONE) * p / prim_rho;
                    let d_baryon = match sqrt_gamma_cell {
                        Some(g) => rho / g,
                        None => rho,
                    };
                    h_enth * d_baryon * d_baryon / prim_rho
                };
                // the contravariant velocity in coordinate slots (the metric-aware c2p output);
                // spherical GR momentum slots are coordinate-ordered, so slot k == coordinate k.
                // a coordinate beyond the momentum slots carries zero.
                let v = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
                    if c < ncomp {
                        cx.field(&format!("prim_v{c}"), FieldRef::PrimVel(c as u8))
                    } else {
                        Gv::ZERO
                    }
                }));
                if matches!(source, GeoSource::Rmhd) {
                    // GRMHD: the ideal-MHD stress in the same contraction. the source takes the
                    // metric-free rest enthalpy density rho_h = rho + Gamma/(Gamma-1) p (it builds
                    // W and b^mu from the harvested gamma internally); B reads the cell field under
                    // the same key convention as the discrete magnetic geo source. the momentum call
                    // takes p = 0 (the gas-pressure block rides the discrete well-balanced form) but
                    // keeps the full magnetic stress — the b^2/2 isotropic block is analytic; the
                    // one-step-residual instrument adjudicates its balance. this path applies on a
                    // static background: the dragging-consistent reconstruction extends to the gas alone.
                    let gamma_eos = cx.scalar("gamma");
                    let prim_rho = cx.field("prim_rho", FieldRef::PrimRho);
                    let rho_h = prim_rho + gamma_eos / (gamma_eos - Gv::ONE) * p;
                    let b = Tensor::<Gv, 3>::new(std::array::from_fn(|k| {
                        if mag_from_bcell {
                            cx.field(&format!("bc_{k}"), FieldRef::BCell(k as u8))
                        } else {
                            cx.field(&format!("prim_b{k}"), &format!("prim.mag[{k}]"))
                        }
                    }));
                    let src_at = |pp| match spacetime {
                        Spacetime::SchwarzschildKS if coords == Coords::Cartesian => {
                            grmhd_covariant_source(
                                &SchwarzschildKSCartesian { mass },
                                x,
                                rho_h,
                                v,
                                pp,
                                b,
                            )
                        }
                        Spacetime::SchwarzschildKS if coords == Coords::Cylindrical => {
                            grmhd_covariant_source(
                                &SchwarzschildKSCylindrical { mass },
                                x,
                                rho_h,
                                v,
                                pp,
                                b,
                            )
                        }
                        Spacetime::SchwarzschildKS => {
                            grmhd_covariant_source(&SchwarzschildKS { mass }, x, rho_h, v, pp, b)
                        }
                        Spacetime::KerrKS if coords == Coords::Cartesian => {
                            // cartesian spinning kerr: the rank-1 kerr-schild metric at the full
                            // cartesian position; derivatives ride the same autodiff Dual pass.
                            let spin = Dual::constant(cx.scalar("kerr_spin"));
                            grmhd_covariant_source(
                                &KerrKSCartesian { mass, spin },
                                x,
                                rho_h,
                                v,
                                pp,
                                b,
                            )
                        }
                        Spacetime::KerrKS if coords == Coords::Cylindrical => {
                            // cylindrical spinning kerr: the rank-1 update on the diag(1, R^2, 1)
                            // base at the full (R, phi, z) position; same autodiff Dual pass.
                            let spin = Dual::constant(cx.scalar("kerr_spin"));
                            grmhd_covariant_source(
                                &KerrKSCylindrical { mass, spin },
                                x,
                                rho_h,
                                v,
                                pp,
                                b,
                            )
                        }
                        Spacetime::KerrKS => {
                            // the generic covariant stress contraction S_j = (1/2) T^{mu nu} d_j g_{mu nu}
                            // with the EM stress; the non-diagonal kerr metric enters through the
                            // autodiff Dual pass alone, which keeps the contraction generic.
                            let spin = Dual::constant(cx.scalar("kerr_spin"));
                            grmhd_covariant_source(&KerrKS { mass, spin }, x, rho_h, v, pp, b)
                        }
                        Spacetime::Minkowski => unreachable!("flat handled above"),
                    };
                    let (s_mom, _) = src_at(Gv::ZERO);
                    let (_, s_tau) = src_at(p);
                    Some((s_mom, s_tau))
                } else {
                    let src_at = |pp| match spacetime {
                        Spacetime::SchwarzschildKS if coords == Coords::Cartesian => {
                            grhd_covariant_source(&SchwarzschildKSCartesian { mass }, x, e, v, pp)
                        }
                        Spacetime::SchwarzschildKS if coords == Coords::Cylindrical => {
                            grhd_covariant_source(&SchwarzschildKSCylindrical { mass }, x, e, v, pp)
                        }
                        Spacetime::SchwarzschildKS => {
                            grhd_covariant_source(&SchwarzschildKS { mass }, x, e, v, pp)
                        }
                        Spacetime::KerrKS if coords == Coords::Cartesian => {
                            let spin = Dual::constant(cx.scalar("kerr_spin"));
                            grhd_covariant_source(&KerrKSCartesian { mass, spin }, x, e, v, pp)
                        }
                        Spacetime::KerrKS if coords == Coords::Cylindrical => {
                            let spin = Dual::constant(cx.scalar("kerr_spin"));
                            grhd_covariant_source(&KerrKSCylindrical { mass, spin }, x, e, v, pp)
                        }
                        Spacetime::KerrKS => {
                            let spin = Dual::constant(cx.scalar("kerr_spin"));
                            grhd_covariant_source(&KerrKS { mass, spin }, x, e, v, pp)
                        }
                        Spacetime::Minkowski => unreachable!("flat handled above"),
                    };
                    if densitized {
                        // the momentum call takes p = 0: the pressure block rides the discrete
                        // well-balanced form below, exactly as it does on the flat curvilinear path.
                        // the free-index-down energy source vanishes on a stationary metric.
                        let (s_mom, _) = src_at(Gv::ZERO);
                        Some((s_mom, Gv::ZERO))
                    } else {
                        let (s_mom, _) = src_at(Gv::ZERO);
                        let (_, s_tau) = src_at(p);
                        Some((s_mom, s_tau))
                    }
                }
            }
        };
        // the connection source is the right side of d_t U + d_j F^j = sqrt(-g)(1/2)(d_i g_ab) T^ab, so
        // it takes the same measure the state and the flux carry.
        let geodesic = match (geodesic, sqrt_neg_g) {
            (Some((s_mom, s_tau)), Some(m)) => Some((s_mom.scale(m), s_tau)),
            (g, _) => g,
        };
        // the well-balanced pressure block of the densitized momentum source.
        //
        // the momentum flux carries sqrt(-g) p delta^j_i, so its divergence contributes
        // p d_i sqrt(-g) + sqrt(-g) d_i p, while the connection source's pressure part is
        // (1/2) sqrt(-g) p g^{ab} d_i g_ab = p d_i sqrt(-g). the two p d_i sqrt(-g) terms cancel
        // analytically, leaving sqrt(-g) d_i p — identically zero at uniform pressure. discrete
        // cancellation requires both sides to use one operator: differencing sqrt(-g) p across the
        // faces on the flux side while taking a pointwise derivative at the center on the source side
        // leaves an O(dx^2) mismatch, the same order as the truncation error, landing on exactly the
        // momentum components.
        //
        // so the pressure block is discretized with the same operator the divergence uses — a face
        // difference of the measure over the coordinate width — and a uniform-p state then cancels the
        // pressure flux divergence bit-exactly. only a gridded coordinate has a pressure gradient.
        let mom_pressure: Option<Vec<Gv>> = densitized.then(|| {
            let p = cx.field("pre", FieldRef::PrimPre);
            // the measure at a face of this cell: the swept coordinate at the face, every other slot
            // where the flux kernel puts it — gridded slots at the cell midpoint, ungridded slots at
            // the chart's symmetry default. taking the raw centroid vector instead would leave an
            // ungridded spherical polar slot at theta = 0, where sin(theta) zeroes the measure and the
            // whole block with it.
            let base = x_cell.expect("a curved spacetime has a metric position");
            let measure_at = |slot: usize, offset: i64| {
                let d = axes
                    .iter()
                    .position(|&a| a == slot)
                    .expect("a gridded coordinate");
                let face = gv_axis_face_at(cx, d, spacing[d], offset);
                let xf: Vec<Gv> = (0..3)
                    .map(|c| if c == slot { face } else { base[c] })
                    .collect();
                let lapse_f = gv_lapse_weight(cx, coords, spacetime, &xf)
                    .expect("a curved spacetime has a lapse");
                let xt = Tensor::<Gv, 3>::new(std::array::from_fn(|c| xf[c]));
                lapse_f * gv_metric_volume_factor_at(cx, spacetime, coords, xt)
            };
            (0..ncomp)
                .map(|coord| match axes.iter().position(|&c| c == coord) {
                    Some(d) => {
                        let width = gv_axis_face_at(cx, d, spacing[d], 1)
                            - gv_axis_face_at(cx, d, spacing[d], 0);
                        p * (measure_at(coord, 1) - measure_at(coord, 0)) / width
                    }
                    None => Gv::ZERO,
                })
                .collect()
        });
        let mom_gravity: Option<Tensor<Gv, 3>> = geodesic.map(|(s_mom, _)| s_mom);
        // the GR geodesic energy source S_tau — the second output of the contraction (gravity's rate
        // of work on the infalling gas). zero on a flat background.
        let nrg_gravity: Option<Gv> = geodesic.map(|(_, s_tau)| s_tau);
        let geo_weight =
            weighted_geo_source.then(|| cx.field("geo_source_weight", FieldRef::Scratch));
        let fe = |u, div, geo_src: Option<_>| {
            let div = match lapse {
                Some(a) => a * div,
                None => div,
            };
            let mut r = u - dt * div - dt * (h_dil * u);
            if let Some(s) = geo_src {
                let s = geo_weight.map_or(s, |w| w * s);
                let s = match lapse {
                    Some(a) => a * s,
                    None => s,
                };
                r = r + dt * s;
            }
            r
        };
        // the densitized law differences the face fluxes in plain coordinates — sqrt(-g) already
        // carries the geometry, so every chart differences with unit weights.
        let divergence = |base: &str| {
            if densitized {
                gv_divergence_coord(cx, base, ndim, spacing)
            } else {
                gv_divergence(cx, base, ndim, &geo, spacing)
            }
        };
        let combine = |un, fe| a0 * un + ac * fe;
        // the user sources ride as a separate additive term after the combine: `+ \sum ac*dt*contrib`,
        // accumulated exactly as `source_apply_gv` accumulates it (start from the combine result,
        // `+= ac_dt*contrib` per spec). so the fused kernel equals `plain godunov + the additive pass`,
        // bit-for-bit, fused into one launch (proven by the fused-equivalence test).
        let with_sources = |base, srcs: &[NodeId]| {
            let mut r = base;
            for c in srcs {
                r = r + ac_dt * cx.gv(*c);
            }
            r
        };

        let u_n_rho = cx.field("u_n_rho", FieldRef::un_den());
        let rho_g = with_sources(
            combine(u_n_rho, fe(rho, divergence("mass_flux"), None)),
            &contribs.den,
        );
        let mut mom_g: Vec<Gv> = Vec::with_capacity(ncomp);
        for k in 0..ncomp {
            let u_n_mom = cx.field(&format!("u_n_mom_{k}"), FieldRef::un_mom(k as u8));
            let div = divergence(&format!("mom_flux_{k}"));
            let geo_src = src.as_ref().map(|s| s[k]);
            // every momentum slot carries its covariant geodesic block (gravity + covariant
            // centrifugal, coordinate k of the contraction) on top of the discrete pressure form in
            // geo_src; a suppressed axisymmetric slot's block is identically zero (the metric is
            // independent of phi, so its autodiff tangent vanishes — angular-momentum conservation).
            let mom_src = match mom_gravity {
                Some(g) => Some(geo_src.map_or(g[k], |s| s + g[k])),
                None => geo_src,
            };
            // the densitized path's pressure block, in the divergence's own face-difference form.
            let mom_src = match &mom_pressure {
                Some(pb) => Some(mom_src.map_or(pb[k], |s| s + pb[k])),
                None => mom_src,
            };
            // valencia covariant storage: the conserved momentum is the covariant S_i = rho h W^2
            // gamma_ij v^j (the metric-aware c2p + flux), and the geodesic source is written for that
            // covariant S_i, so d_t S_i = -alpha div(F) + alpha S — a single, uniform lapse on every
            // conserved law, supplied by the `fe` weight. the lapse enters at one power, matching the
            // contravariant v^n the flux kernel carries (the orthonormal V_rhat would bring an alpha^2
            // asymmetry), and the metric coefficient gamma_ij rides inside S_i, above the densitization.
            mom_g.push(with_sources(
                combine(u_n_mom, fe(mom[k], div, mom_src)),
                &contribs.mom[k],
            ));
        }
        // relativistic hydro and MHD on a curved background evolve the covariant (killing) energy ehat,
        // whose flux is the free-index-down `-sqrt(-g)(T^t_t + rho u^t)` current and whose source vanishes
        // on a stationary metric (HARM/AthenaK; docs/covariant_energy.md). RHD builds f_ehat in the regime;
        // RMHD builds it as a linear re-split of the valencia numerical fluxes in the flux kernel (keeping
        // the valencia-native HLLD / CT machinery). both feed the same pure-conservation energy law here.
        let is_covariant_energy = spacetime != Spacetime::Minkowski
            && matches!(source, GeoSource::Hydro { .. } | GeoSource::Rmhd);
        let nrg_g = has_energy.then(|| {
            let nrg = cx.field("nrg", FieldRef::cons_nrg());
            let u_n_nrg = cx.field("u_n_nrg", FieldRef::un_nrg());
            let div = divergence("nrg_flux");
            let stage = if is_covariant_energy {
                // conservation of the killing energy, d_t(sqrt(gm) ehat) + d_n(sqrt(gm) f_ehat) = 0,
                // reduced to the code's flat coordinate measure (sqrt(gm) = sqrt(gm_flat)/alpha, static):
                // d_t ehat + alpha_cell * div_flat(f_ehat/alpha_face) = 0. the flux kernel stores
                // f_ehat/alpha at the face; the cell lapse rides here through `fe` — the same uniform
                // lapse every other conserved law takes. the geodesic energy source is identically zero
                // for the killing energy on a stationary metric (the normal-observer tau carries one),
                // so this law is pure conservation.
                fe(nrg, div, None)
            } else {
                fe(nrg, div, nrg_gravity)
            };
            with_sources(combine(u_n_nrg, stage), &contribs.nrg)
        });

        // immersed-body wrap: `(cons_g + ac_dt*S_grav(cons))*f` with `f = exp(-drain*ac_dt)`. the body
        // contribution is evaluated at the stage input — `rho`/`mom`/`nrg`, which is what this kernel
        // reads `cons` as — and applied to the flux-combined `cons_g`. an explicit scheme evaluates
        // the flux divergence and every source at one state and sums them into one convex update;
        // evaluating the body at the combined state instead would compose two operators sequentially,
        // which is first order in dt at any Runge-Kutta order and leaks internal energy one-signed
        // every stage.
        //
        // that is exactly the two-pass execution order (godunov -> source_apply -> body_source, every
        // stage at weight ac*dt), so the fused sweep stays bit-identical to plain godunov followed by
        // the standalone `body_source` pass: storing cons_g to an f64 buffer and reading it back is
        // exact, so the register-resident cons_g the body reads here equals the memory value the
        // two-pass body reads. gravity is additive and the accretion drain multiplicative, so the body
        // wraps the final nodes, outside the additive `contribs` accumulation.
        // adiabatic (energy) only; the iso body (`body_source_iso_gv`, cs from prim.pre) is a follow-on.
        let (rho_final, mom_final, nrg_final) = if n_bodies > 0 {
            if let Some(nrg_in) = nrg_g {
                let gamma = cx.scalar("gamma");
                // the stage input, re-bound: these hash-cons to the same nodes the flux read.
                let us_nrg = cx.field("nrg", FieldRef::cons_nrg());
                let (den_b, mom_b, nrg_b, _drain) = crate::gv_immersed::body_applied_gv(
                    cx,
                    rho_g,
                    &mom_g,
                    nrg_in,
                    rho,
                    &mom,
                    us_nrg,
                    ac_dt,
                    gamma,
                    n_bodies,
                    coords,
                    ndim as usize,
                    ncomp,
                    axes,
                );
                (den_b, mom_b, Some(nrg_b))
            } else {
                let pre = cx.field("prim_pre", FieldRef::PrimPre);
                let (den_b, mom_b) = crate::gv_immersed::body_applied_iso_gv(
                    cx,
                    rho_g,
                    &mom_g,
                    rho,
                    &mom,
                    pre,
                    ac_dt,
                    n_bodies,
                    coords,
                    ndim as usize,
                    ncomp,
                    axes,
                );
                (den_b, mom_b, None)
            }
        } else {
            (rho_g, mom_g, nrg_g)
        };

        let mut writes = vec![KernelWrite::new(
            "rho",
            FieldRef::cons_den(),
            rho_final.node(),
        )];
        for (k, m) in mom_final.iter().enumerate() {
            writes.push(KernelWrite::new(
                format!("mom_{k}"),
                FieldRef::cons_mom(k as u8),
                m.node(),
            ));
        }
        if let Some(nrg_new) = nrg_final {
            writes.push(KernelWrite::new(
                "nrg",
                FieldRef::cons_nrg(),
                nrg_new.node(),
            ));
        }
        writes
    })
}

/// the standalone additive source pass: `cons += dt * \sum S(prim, x; params)`, in place, per
/// conserved slot, for a list of spec sources. the general source executor — it runs any composed
/// source as a separate per-stage kernel (the `body_source_gv` mechanism, generalized to
/// `SourceSpec`s); the fused path folds the same source into the godunov stage.
///
/// it splices the same `splice_fused_sources_to_contribs` the fused godunov uses, so a plain
/// `godunov_stage_gv` (flux + geometric source, no user sources) followed by this pass is the
/// proven-equivalent decomposition of `godunov_stage_gv_with_fused_sources`. the driver passes
/// `dt = ac*dt` (the SSP shu-osher stage weight — identical to how `body_source` is invoked), so
/// `S` lands with the same `ac*dt` weight the fused stage applies inside its `ac*fe` combine.
///
// =============================================================================
// the unified DAG-application operator.
//
// `apply_dag_core_gv` is the single kernel builder behind both the interior source pass and
// the driven-boundary pass. it factors out the decisions a source/boundary
// makes: where the DAG reads state (`StateEnv`), and how its result lands in
// the target field (`WriteMode`). the iteration domain + target-field binding are the dispatch's job
// (the same `dispatch_runtime_ir` + `resolve_path` serve cons.* and prim.*), so this builder is the
// whole difference between a source and a boundary prescription. doc 32's user `combine` projects
// onto `WriteMode`: add/relax -> Accumulate (differ only in the constructed expression), overwrite ->
// Assign.
// =============================================================================

/// the state vocabulary the DAG reads `rho`/`vel_k` from. `Stage` binds them from the SSP stage
/// snapshot `u_stage` (an interior source evaluates at its stage input — the stage-input invariant); `Coord`
/// is a pure coordinate prescription — a driven boundary, whose DAG outputs the state and reads
/// coordinates alone. `x_k` (centroid) + scalar params bind regardless of this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StateEnv {
    Stage,
    Coord,
}

/// how the DAG result lands in the target field. `Accumulate` is the RHS form `target = read(target)
/// + dt * \sum contrib` (in place; the `dt` scalar is the SSP stage weight) — sources. `Assign` is the
/// prescription `target = expr` (write-only; the expression is the whole value) — driven
/// boundaries. doc 32's `combine`: add + relax both map to `Accumulate`, overwrite to `Assign`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WriteMode {
    Accumulate,
    Assign,
}

/// the unified core: trace a kernel that evaluates each `(slot, SourceProgram)` DAG per cell and writes
/// it to the slot's field under `mode`. `slot` names the structural conserved slot (`"den"` mass /
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
    sources: &[(String, symbi_source_compile::source_spec::SourceProgram)],
    mode: WriteMode,
) -> KernelProgram {
    trace_kernel(|cx| {
        let geo = (!is_cartesian_uniform(coords, spacing))
            .then(|| cell_geometry_gv(cx, coords, spacing, axes, ndim as usize));

        // bind the state vocabulary the DAG reads from. `Stage` reads the stage-input snapshot `u_stage`
        // in place of the post-godunov `cons`: the fused stage evaluates at its stage input, so this
        // standalone pass matches it, giving `plain + this == fused` bit-for-bit. `Coord` reads
        // coordinates alone.
        let state_vocab: Option<(Gv, Vec<Gv>)> = match state {
            StateEnv::Stage => {
                let rho = cx.field("rho", FieldRef::ustage_den());
                let mom = (0..ncomp)
                    .map(|k| cx.field(&format!("mom_{k}"), FieldRef::ustage_mom(k as u8)))
                    .collect();
                Some((rho, mom))
            }
            StateEnv::Coord => None,
        };
        let state_ref = state_vocab.as_ref().map(|(r, m)| (*r, m.as_slice()));

        let contribs = splice_fused_sources_to_contribs(
            cx, coords, spacing, axes, ndim, ncomp, has_energy, &geo, state_ref, sources,
        );

        let writes = match mode {
            WriteMode::Accumulate => {
                // RHS in place: `cons_slot = cons_slot + \sum dt*contrib`, accumulated exactly as the fused
                // stage's `with_sources` — so fused and (plain godunov + this pass) agree bit-for-bit.
                let dt = cx.scalar("dt"); // the driver fills this with ac*dt (the SSP stage weight)
                let cons_den = cx.field("cons_den", FieldRef::cons_den());
                let mut rho_new = cons_den;
                for c in &contribs.den {
                    rho_new = rho_new + dt * cx.gv(*c);
                }
                let mut writes = vec![KernelWrite::new(
                    "rho",
                    FieldRef::cons_den(),
                    rho_new.node(),
                )];
                for k in 0..ncomp {
                    let cons_mom = cx.field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8));
                    let mut mom_new = cons_mom;
                    for c in &contribs.mom[k] {
                        mom_new = mom_new + dt * cx.gv(*c);
                    }
                    writes.push(KernelWrite::new(
                        format!("mom_{k}"),
                        FieldRef::cons_mom(k as u8),
                        mom_new.node(),
                    ));
                }
                if has_energy {
                    let cons_nrg = cx.field("cons_nrg", FieldRef::cons_nrg());
                    let mut nrg_new = cons_nrg;
                    for c in &contribs.nrg {
                        nrg_new = nrg_new + dt * cx.gv(*c);
                    }
                    writes.push(KernelWrite::new(
                        "nrg",
                        FieldRef::cons_nrg(),
                        nrg_new.node(),
                    ));
                }
                // the godunov-source (accumulate) path targets den/mom/nrg alone — the safe
                // conservation-law lifts touch those slots, and `raw` is gated to them. a bcell contrib
                // here means a mis-routed source; fail loud so it surfaces at once.
                debug_assert!(
                    contribs.mag.iter().all(|m| m.is_empty()),
                    "accumulate (godunov source) path does not support a `bcell` target",
                );
                // the dye is advected by the mass flux alone, so a chi contrib here is a mis-routed
                // prescription. fail loud so it surfaces at once.
                debug_assert!(
                    contribs.chi.is_empty(),
                    "accumulate (godunov source) path does not support a `chi` target",
                );
                writes
            }
            WriteMode::Assign => {
                // prescription: `prim_slot = expr` (write-only; the expression is the whole value). a
                // prescription is a complete state — a single DAG per slot, taken whole.
                assert_eq!(
                    contribs.den.len(),
                    1,
                    "Assign: prim.rho needs exactly one source DAG"
                );
                let mut writes = vec![KernelWrite::new("rho", FieldRef::PrimRho, contribs.den[0])];
                for k in 0..ncomp {
                    assert_eq!(
                        contribs.mom[k].len(),
                        1,
                        "Assign: prim.vel_{k} needs exactly one source DAG"
                    );
                    writes.push(KernelWrite::new(
                        format!("vel_{k}"),
                        FieldRef::PrimVel(k as u8),
                        contribs.mom[k][0],
                    ));
                }
                if has_energy {
                    assert_eq!(
                        contribs.nrg.len(),
                        1,
                        "Assign: prim.pre needs exactly one source DAG"
                    );
                    writes.push(KernelWrite::new("pre", FieldRef::PrimPre, contribs.nrg[0]));
                }
                // MHD driven boundary: prescribe the cell-B vector (prim.mag). out-of-plane B_phi
                // (cell-centered, flux-evolved) is the safe toroidal case; in-plane components are
                // the user's responsibility to keep div-compatible (=0 for a purely toroidal field).
                // absent for a hydro prescription (no bcell slot -> empty mag buckets).
                if contribs.mag.iter().any(|m| !m.is_empty()) {
                    for k in 0..ncomp {
                        assert_eq!(
                            contribs.mag[k].len(),
                            1,
                            "Assign: prim.mag_{k} needs exactly one source DAG"
                        );
                        writes.push(KernelWrite::new(
                            format!("mag_{k}"),
                            FieldRef::PrimMag(k as u8),
                            contribs.mag[k][0],
                        ));
                    }
                }
                // the dye of injected fluid. absent for an undyed prescription (no chi slot -> empty
                // bucket), in which case the face's dye ghosts stay whatever the scalar pullback left.
                if !contribs.chi.is_empty() {
                    assert_eq!(
                        contribs.chi.len(),
                        1,
                        "Assign: prim.chi needs exactly one source DAG"
                    );
                    writes.push(KernelWrite::new("chi", FieldRef::PrimChi, contribs.chi[0]));
                }
                writes
            }
        };
        writes
    })
}

/// the in-place source-apply kernel `cons += dt*S` over admitted sources — the AOT bake feeds
/// admitted specs per (regime, ndim), the runtime feeds `RuntimeSource`'s admitted programs
/// loaded from a `SourceConfig` at sim startup. the `(Stage, Accumulate)` instance of
/// [`apply_dag_core_gv`].
pub fn source_apply_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    sources: &AdmittedSources,
) -> KernelProgram {
    apply_dag_core_gv(
        coords,
        spacing,
        axes,
        ndim,
        ncomp,
        has_energy,
        StateEnv::Stage,
        sources.pairs(),
        WriteMode::Accumulate,
    )
}

/// driven-boundary entry: prescribe the primitive state from coordinate DAGs — the
/// `(Coord, Assign)` instance of [`apply_dag_core_gv`]. the prescription pairs a slot
/// `"den"`/`"mom"`/`"nrg"` (mapping to `prim.rho`/`prim.vel_k`/`prim.pre`) with the DAG that
/// reads only `x_k`/`t`/`p_i` and outputs the prescribed value. dispatched over a face's
/// ghost band.
pub fn boundary_fill_from_prescription_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    prescription: &BoundaryPrescription,
) -> KernelProgram {
    apply_dag_core_gv(
        coords,
        spacing,
        axes,
        ndim,
        ncomp,
        has_energy,
        StateEnv::Coord,
        prescription.pairs(),
        WriteMode::Assign,
    )
}

/// the cell-B induction-flux divergence for component `c` (mirror of `rmhd::bcell_flux_div`):
/// cartesian `sum_d (bf_d_c[+e_d] - bf_d_c)/dx_d`; curvilinear the area-weighted `inv_V sum_d
/// (A_hi_d bf_d_c[+e_d] - A_lo_d bf_d_c)` from `geo` — the same divergence the gas godunov uses.
fn bcell_flux_div_gv<'t>(
    cx: TraceCx<'t>,
    c: usize,
    ndim: usize,
    geo: &Option<CellGeometryGv<'t>>,
    dx: &[Gv<'t>],
) -> Gv<'t> {
    let off = |d: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[d] = 1;
        o
    };
    let zero = vec![0i32; ndim];
    let mut div: Option<Gv> = None;
    for d in 0..ndim {
        let key = format!("bf_{d}_{c}");
        let here = gv_field_at(cx, &key, &key, ndim, &zero);
        let plus = gv_field_at(cx, &key, &key, ndim, &off(d));
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

/// the plain (metric-free) cell-B induction-flux divergence `sum_d (bf_d_c[+e_d] - bf_d_c)/width_d`
/// with the per-axis coordinate width read in-kernel (gv_axis_face_at). used for the out-of-plane
/// B component whose curl carries unit lame factors — see `metric_free_oop_component`.
fn bcell_flux_div_plain_gv<'t>(
    cx: TraceCx<'t>,
    c: usize,
    ndim: usize,
    spacing: &[Spacing],
) -> Gv<'t> {
    let off = |d: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[d] = 1;
        o
    };
    let zero = vec![0i32; ndim];
    let mut div: Option<Gv> = None;
    for d in 0..ndim {
        let key = format!("bf_{d}_{c}");
        let here = gv_field_at(cx, &key, &key, ndim, &zero);
        let plus = gv_field_at(cx, &key, &key, ndim, &off(d));
        let width = gv_axis_face_at(cx, d, spacing[d], 1) - gv_axis_face_at(cx, d, spacing[d], 0);
        let term = (plus - here) / width;
        div = Some(match div {
            None => term,
            Some(a) => a + term,
        });
    }
    div.unwrap()
}

/// the divergence operator for the out-of-plane B component (the one whose coordinate lies outside
/// `axes`) on a flat background, where the stored component is physical (orthonormal) and its
/// induction curl is `d_t B_c = -(1/(h1 h2))[d_1(h2 F^1) + d_2(h1 F^2)]` over the in-plane lame
/// factors — which only sometimes coincides with the gas area-weighted divergence.
enum OopDiv<'t> {
    /// in-plane lame factors are both 1 (cyl r-z: (curl E)_phi = d_z E_r - d_r E_z), so the
    /// operator is the plain unweighted divergence; the gas h_phi = r cell volume would inject
    /// a spurious F_r/r source.
    Plain,
    /// a non-unit in-plane lame factor rides the curl as a face weight (sph r-theta: h_theta =
    /// r, so `d_t B_phi = -(1/r)[d_r(r F^r) + d_theta F^theta]`); the gas r^2 sin(theta)
    /// measure would inject spurious `-F^r/r - cot(theta) F^theta/r` sources.
    Curl(CellGeometryGv<'t>),
}

/// the out-of-plane B component and its curl divergence for a flat (physical-component) plane:
/// - cyl r-z (axes [0,2]) -> B_phi (comp 1), metric-free (Plain).
/// - sph r-theta (axes [0,1]) -> B_phi (comp 2), the (r, 1)-weighted curl on the r dr dtheta
///   measure (Curl).
/// - cyl r-phi (axes [0,1]) -> B_z: the z-curl `(1/R)[d_R(R F^R) + d_phi F^phi]` is the gas
///   R-measure divergence -> None (the gas path is already the curl).
/// - cartesian / fully-gridded (3D): None (plain == gas, or the out-of-plane set is empty).
/// on a curved spacetime every component takes the covariant law: B is stored
/// contravariant and obeys the densitized conservation `d_t(sqrt(gamma) B^i) + d_j(alpha
/// sqrt(gamma) G^j) = 0` — the covariant area-weighted divergence with the lapse weight.
fn flat_oop_divergence<'t>(
    cx: TraceCx<'t>,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ncomp: usize,
) -> Option<(usize, OopDiv<'t>)> {
    match (coords, axes) {
        (Coords::Cylindrical, [0, 2]) if ncomp > 1 => Some((1, OopDiv::Plain)),
        (Coords::Spherical, [0, 1]) if ncomp > 2 => Some((
            2,
            OopDiv::Curl(oop_curl_geometry_sph_rtheta_gv(cx, spacing)),
        )),
        _ => None,
    }
}

/// the per-component induction-flux divergences for the cell-B predictor, GR-lapse-weighted.
/// flat: the gas area-weighted divergence, except the out-of-plane component's curl operator
/// (`flat_oop_divergence`). curved: the covariant `alpha sqrt(gamma)` measure for every
/// component, times the lapse `alpha(centroid)` — the same densitization contract as the gas
/// godunov (the face kernel writes `G = F - (beta^n/alpha) U`, deferring one alpha to the
/// divergence; see `gv_lapse_weight`). flat spacetime elides the weight (bit-identical).
fn bcell_flux_divs_gv<'t>(
    cx: TraceCx<'t>,
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    comps: &[usize],
    ncomp: usize,
    axes: &[usize],
) -> Vec<Gv<'t>> {
    let (geo, dx) = bcell_godunov_geom(cx, coords, spacetime, spacing, ndim, axes);
    let oop = match spacetime {
        Spacetime::Minkowski => flat_oop_divergence(cx, coords, spacing, axes, ncomp),
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
    let lapse = gv_lapse_weight(cx, coords, spacetime, &coord_centroid);
    // one divergence per requested component (the predictor evaluates only the out-of-plane set),
    // returned in `comps` order.
    comps
        .iter()
        .map(|&c| {
            let div = match &oop {
                Some((co, OopDiv::Plain)) if c == *co => {
                    bcell_flux_div_plain_gv(cx, c, ndim, spacing)
                }
                Some((co, OopDiv::Curl(g))) if c == *co => {
                    bcell_flux_div_gv(cx, c, ndim, &Some(g.clone()), &dx)
                }
                _ => bcell_flux_div_gv(cx, c, ndim, &geo, &dx),
            };
            match lapse {
                Some(a) => a * div,
                None => div,
            }
        })
        .collect()
}

/// the out-of-plane (non-CT) magnetic components for a chart: the B-vector slots whose coordinate
/// lies outside the grid axes. the in-plane slots live on staggered faces and are re-derived
/// cell-centered by `bcell_from_bface = interp(bface)`, so the predictor leaves them alone; the
/// complement — the out-of-plane components — live purely at cell centers and are evolved here as
/// conserved variables. cartesian `[0..ndim)` grid -> `[ndim..ncomp)`; cyl r-z (axes [0,2])
/// -> {phi=1}; sph r-theta (axes [0,1]) -> {phi=2}; a fully-gridded 3D chart -> empty.
fn oop_components(ncomp: usize, axes: &[usize]) -> Vec<usize> {
    (0..ncomp).filter(|c| !axes.contains(c)).collect()
}

/// the RMHD cell-B flux predictor (euler): `bcell[c] -= dt*div(bflux_c)`, in-place, for the
/// out-of-plane components (`oop_components`). those are the genuinely cell-centered magnetic
/// slots — stored at cell centers, hence flux-evolved (reduced-dimension MHD) while CT governs the
/// staggered faces. the in-plane components are re-derived by `bcell_from_bface = interp(bface)`
/// and keep that value through this pass: a transient predictor value on them poisons the FOFC/c2p
/// recoverability probe once the magnetic-energy patch is gone. on a fully-gridded chart (3D) the
/// out-of-plane set is empty and the kernel comes out empty — its dispatch is elided at
/// `ndim==ncomp`.
pub fn rmhd_bcell_godunov_euler_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> KernelProgram {
    trace_kernel(|cx| {
        let oop = oop_components(ncomp, axes);
        let bc: Vec<Gv> = oop
            .iter()
            .map(|&c| cx.field(&format!("bc_{c}"), FieldRef::BCell(c as u8)))
            .collect();
        // pin the ndim*|oop| induction-flux inputs in d-outer/c-inner order (the positional dispatch
        // order [bf_0_c, bf_1_c, ..]) before bcell_flux_div_gv reads them (it loops d).
        for d in 0..ndim {
            for &c in &oop {
                gv_register_field(cx, &format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
            }
        }
        let dt = cx.scalar("dt");
        let divs = bcell_flux_divs_gv(cx, coords, spacetime, spacing, ndim, &oop, ncomp, axes);
        let writes = (0..oop.len())
            .map(|i| {
                let c = oop[i];
                let bnew = bc[i] - dt * divs[i];
                KernelWrite::new(format!("bc_{c}_new"), format!("bc_{c}"), bnew.node())
            })
            .collect();
        writes
    })
}

/// the RMHD cell-B flux predictor (RK2 stage 2): `bcell[c] = 0.5*(bcell_n[c] + (bcell[c] -
/// dt*div(bflux_c)))`, in-place, for the out-of-plane components (`oop_components`; see the
/// euler predictor). a fully-gridded chart (3D) yields an empty kernel (dispatch elided).
pub fn rmhd_bcell_godunov_rk2_gv(
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> KernelProgram {
    trace_kernel(|cx| {
        let oop = oop_components(ncomp, axes);
        let bcn: Vec<Gv> = oop
            .iter()
            .map(|&c| cx.field(&format!("bcn_{c}"), FieldRef::BCellN(c as u8)))
            .collect();
        let bc: Vec<Gv> = oop
            .iter()
            .map(|&c| cx.field(&format!("bc_{c}"), FieldRef::BCell(c as u8)))
            .collect();
        for d in 0..ndim {
            for &c in &oop {
                gv_register_field(cx, &format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
            }
        }
        let dt = cx.scalar("dt");
        let half = Gv::from_f64(0.5);
        let divs = bcell_flux_divs_gv(cx, coords, spacetime, spacing, ndim, &oop, ncomp, axes);
        let writes = (0..oop.len())
            .map(|i| {
                let c = oop[i];
                let bc_star = bc[i] - dt * divs[i];
                let bnew = half * (bcn[i] + bc_star);
                KernelWrite::new(format!("bc_{c}_new"), format!("bc_{c}"), bnew.node())
            })
            .collect();
        writes
    })
}

/// the cell-B godunov geometry: curvilinear or curved -> the gv cell geometry (area-weighted
/// div); flat cartesian -> the uniform `dx_d` scalars. a curved cartesian chart (kerr-schild)
/// still carries the (flat-equal) cartesian geometry so the lapse weight has a centroid to
/// evaluate at — its covariant measure `alpha sqrt(gamma) = 1` equals the coordinate volume.
fn bcell_godunov_geom<'t>(
    cx: TraceCx<'t>,
    coords: Coords,
    spacetime: Spacetime,
    spacing: &[Spacing],
    ndim: usize,
    axes: &[usize],
) -> (Option<CellGeometryGv<'t>>, Vec<Gv<'t>>) {
    if coords == Coords::Cartesian && spacetime == Spacetime::Minkowski {
        (
            None,
            // per-cell widths: an unmapped axis reduces to its `dx_d` scalar, a graded one
            // differences its own faces.
            (0..ndim)
                .map(|d| gv_axis_width(cx, d, spacing[d]))
                .collect(),
        )
    } else {
        // axes maps grid axis -> coordinate (identity for sph/3d-cyl; [0,2] for cyl r-z) so the
        // area-weighted divergence uses the right radial axis for the cylindrical metric. a
        // curved spacetime takes the covariant (alpha sqrt(gamma)) measure — the mag rows are
        // densitized conserved laws of the same form as the gas (d_t(sqrt(g) B) + coordinate
        // divergence), exactly like the gas godunov's geometry selection.
        let g = match spacetime {
            Spacetime::Minkowski => cell_geometry_gv(cx, coords, spacing, axes, ndim),
            Spacetime::KerrKS => cell_geometry_covariant_gv(
                cx,
                coords,
                spacing,
                axes,
                ndim,
                Some(cx.scalar("kerr_spin")),
            ),
            _ => cell_geometry_covariant_gv(cx, coords, spacing, axes, ndim, None),
        };
        (Some(g), Vec::new())
    }
}

// =============================================================================
// passive-scalar (dye) transport: D_chi = rho*chi rides the already-materialized
// mass flux, donor-cell upwinded on its sign per face:
//   F_chi(face) = mass_flux(face) * (chi_donor)
// so the dye moves exactly with the mass it is painted on: a uniform chi = c
// gives F_chi = c * F_rho at every face and the update telescopes to c times the
// mass update (uniform-preservation is bit-exact). the upwind chi comes from the
// primitive chi (the stage-input concentration, consistent with the primitives
// the flux pass reconstructed from); the kernels read prim.chi at neighbor
// offsets and write cons.chi at offset 0 only, so the in-place update is
// race-free under tiled execution. cartesian flat charts only (dx_d scalars).
// =============================================================================

// the donor-cell chi flux divergence: sum_d (F_hi - F_lo)/dx_d, reading the stored interface dye
// flux `flux[d].chi` written by `chi_flux_gv`. same convention as the gas: the flux field at a cell
// index holds the flux through that cell's low face on axis d.
fn chi_flux_div_gv<'t>(cx: TraceCx<'t>, ndim: usize, spacing: &[Spacing]) -> Gv<'t> {
    let zero_off = vec![0i32; ndim];
    let mut div: Option<Gv> = None;
    for d in 0..ndim {
        let mut plus = zero_off.clone();
        plus[d] = 1;
        let key = format!("chi_flux_{d}");
        let path = FieldRef::ChiFlux(d as u8).name();
        let f_lo = gv_field_at(cx, &key, &path, ndim, &zero_off);
        let f_hi = gv_field_at(cx, &key, &path, ndim, &plus);
        let term = (f_hi - f_lo) / gv_axis_width(cx, d, spacing[d]);
        div = Some(match div {
            None => term,
            Some(a) => a + term,
        });
    }
    div.unwrap()
}

/// the interface dye flux on one axis: `flux[d].chi = mass_flux_d * upwind(prim.chi)`, the
/// concentration taken from whichever side the mass is flowing out of. donor-cell upwinding on the
/// sign of the mass flux at that same face, so the dye flux telescopes with the mass flux and a
/// uniform concentration is carried exactly.
///
/// materialized as a stored field so a coarse-fine reflux can correct the conserved dye from the
/// difference between the fine-time-summed and coarse fluxes at the interface. `F_chi` is nonlinear
/// in the state, so that correction has to read the stored dye flux itself; the mass-flux
/// correction alone underdetermines it.
pub fn chi_flux_gv(ndim: usize, dir: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let zero_off = vec![0i32; ndim];
        let mut minus = zero_off.clone();
        minus[dir] = -1;
        let zero = Gv::from_f64(0.0);
        let mf_key = format!("mass_flux_{dir}");
        let mf_path = FieldRef::MassFlux(dir as u8).name();
        let chi_path = FieldRef::PrimChi.name();
        let f = gv_field_at(cx, &mf_key, &mf_path, ndim, &zero_off);
        let chi_m = gv_field_at(cx, "prim_chi", &chi_path, ndim, &minus);
        let chi_0 = gv_field_at(cx, "prim_chi", &chi_path, ndim, &zero_off);
        let upwind = Gv::select(f.cmp_ge(zero), chi_m, chi_0);
        let writes = vec![KernelWrite::new(
            format!("chi_flux_{dir}_new"),
            FieldRef::ChiFlux(dir as u8),
            (f * upwind).node(),
        )];
        writes
    })
}

/// the dye godunov, in the same SSP shu-osher form as the gas stage:
/// `cons.chi = a0*u_n.chi + ac*(cons.chi - dt*div(F_chi))`, in place, with the
/// per-stage convex coefficients as runtime scalars (forward-euler = (0, 1)) —
/// one kernel serves every explicit SSP scheme.
pub fn chi_godunov_gv(ndim: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let chin = cx.field("un_chi", FieldRef::un_chi());
        let dchi = cx.field("cons_chi", FieldRef::cons_chi());
        let dt = cx.scalar("dt");
        let a0 = cx.scalar("a0");
        let ac = cx.scalar("ac");
        // the homologous-mesh dilution `-mesh_hdil * D_chi` rides every conserved law, and the dye is
        // one: expansion dilutes `D_chi = rho chi` exactly as it dilutes `rho`, leaving the
        // concentration invariant. the static binding `mesh_hdil = 0` subtracts an exact zero.
        let h_dil = cx.scalar("mesh_hdil");
        // face positions are selected at runtime by `map_kind_d`, so the widths come out per-cell on
        // any mesh, whatever the bake-time spacing tag says.
        let spacing = vec![Spacing::Uniform; ndim];
        let new = a0 * chin
            + ac * (dchi - dt * chi_flux_div_gv(cx, ndim, &spacing) - dt * (h_dil * dchi));
        let writes = vec![KernelWrite::new(
            "chi_new",
            FieldRef::cons_chi(),
            new.node(),
        )];
        writes
    })
}

/// the dye concentration recovery: `prim.chi = cons.chi / cons.den` — chi's whole
/// cons2prim, run after the stage's density is final so the concentration is
/// consistent with the same-instant mass.
pub fn chi_c2p_gv() -> KernelProgram {
    trace_kernel(|cx| {
        let dchi = cx.field("cons_chi", FieldRef::cons_chi());
        let den = cx.field("cons_den", FieldRef::cons_den());
        let writes = vec![KernelWrite::new(
            "prim_chi_new",
            FieldRef::PrimChi,
            (dchi / den).node(),
        )];
        writes
    })
}

/// the dye step snapshot: `u_n.chi <- cons.chi`, the rk2 combine's step-start state.
pub fn chi_snapshot_gv() -> KernelProgram {
    trace_kernel(|cx| {
        let dchi = cx.field("cons_chi", FieldRef::cons_chi());
        let writes = vec![KernelWrite::new(
            "un_chi_new",
            FieldRef::un_chi(),
            dchi.node(),
        )];
        writes
    })
}

#[cfg(test)]
mod pcp_source_weight_tests {
    use super::*;

    fn rmhd_stage(weighted: bool) -> symbi_ir::GvKernel {
        godunov_stage_gv_with_fused_bodies_and_geo_weight(
            Coords::Cartesian,
            Spacetime::SchwarzschildKS,
            &[Spacing::Uniform; 3],
            &[0, 1, 2],
            3,
            3,
            true,
            GeoSource::Rmhd,
            &AdmittedSources::none(),
            false,
            0,
            weighted,
        )
        .into_kernel()
    }

    #[test]
    fn ordinary_stage_does_not_carry_source_weight_field() {
        let kernel = rmhd_stage(false);
        assert!(
            kernel
                .field_inputs()
                .iter()
                .all(|(_, bind)| *bind != FieldRef::Scratch.into())
        );
    }

    #[test]
    fn pcp_stage_carries_typed_source_weight_field() {
        let kernel = rmhd_stage(true);
        let weights = kernel
            .field_inputs()
            .iter()
            .filter(|(_, bind)| *bind == FieldRef::Scratch.into())
            .count();
        assert_eq!(weights, 1);
    }
}
