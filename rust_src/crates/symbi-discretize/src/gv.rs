// =============================================================================
// gv.rs
//
// the Gv KERNEL BUILDERS: each `*_gv` fn instantiates a carrier-generic
// symbi-hydro physics function (written over `S: Scalar`) at `S = Gv` and traces
// it into a stencil DAG — the dispatchable kernel (graph + ABI manifest). the
// `Gv` carrier + the trace itself live in `symbi-core`; this module is the
// discretization layer that drives it: it picks coords/spacing/reconstruction
// (the numerical choices) and builds c2p / flux / godunov / wave-speed / CT /
// ghost-fill / geometry kernels (design/gv_algebra_unification.md §3). `S = f64`
// gives the host body; `S = Gv` gives the kernel graph — one physics source.
//
// raw index/stencil IR (integer coord arithmetic, lattice-map boundary source,
// multi-axis load_at) is built directly against `symbi_core::with_trace` — the
// f64 `Gv` carrier deliberately does not route integer addressing through itself.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;
use symbi_ir::algebra::Scalar;
use symbi_ir::{FieldBind, FieldRef};
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::mhd_state::{MhdCons, MhdPrim, IsoMhdCons, IsoMhdPrim};
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::newtonian_mhd::{nmhd_recover, NewtonianMhd};
use symbi_hydro::isothermal_mhd::{imhd_recover, IsothermalMhd};
use symbi_hydro::regime::Regime;
use symbi_hydro::riemann::{hlle, hlle_with_speeds, hllc, hllc_srhd, hllc_rmhd, hllc_newtonian, hlld_rmhd, hlld_rmhd_states, HlldStates, hlld_newtonian, hlld_isothermal};
use symbi_hydro::ShockwaveLimiter;
use symbi_hydro::rmhd::{rmhd_magnetosonic_cfl_speeds, rmhd_recover, rmhd_source_quantities, Rmhd};
use symbi_hydro::srhd::{srhd_recover, Srhd};
use symbi_hydro::state::{Cons, Prim, ConsG, PrimG};
use symbi_hydro::energy::Zero;
use symbi_ir::Symbol;
use symbi_ir::graph::{ConstValue, ElementWiseOp, NodeId};

// the carrier + trace live alongside Op + Graph in symbi-ir (consolidated 2026-05-30;
// symbi-core was folded in). the builders below instantiate carrier-generic
// symbi-hydro physics at S = Gv and trace it into the IR.
use symbi_ir::{begin_trace, end_trace, with_trace, Gv, GvKernel, MeshScalar, TileSpec};

use super::coords::{Coords, Spacing};
/// trace the REAL adiabatic (ideal-gas) c2p — symbi-hydro's `Cons::to_primitive` at
/// `S = Gv` — into a dispatchable kernel. the carrier-generic physics IS the kernel
/// builder; this is what replaces the hand-written `adiabatic_c2p` Expr builder. returns
/// the `GvKernel` (graph + ABI manifest) and the `(write_key, runtime_path, root)` writes.
/// note: the `Regime::to_primitive` WRAPPER's native error-code branches are host-only
/// diagnostics — the kernel traces only the branch-free math `Cons::to_primitive`.
pub fn adiabatic_c2p_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding: the conserved fields + the eos scalar, as Gv leaves.
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..D)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let gamma = Gv::scalar("gamma");

    // the SINGLE-SOURCE physics, instantiated at the tracing carrier.
    let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
    let cons = Cons::<Gv, D> { den, mom: Tensor::new(mom_arr), nrg };
    let prim: Prim<Gv, D> = cons.to_primitive(&IdealGas { gamma });

    // decompose the recovered primitive into field writes.
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..D {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}

/// trace the REAL isothermal c2p — symbi-hydro's `IsoNewtonian::to_primitive` (the pure
/// `rho = den`, `vel = mom / rho` kinematics) plus the `Isothermal::pressure` closure
/// `p = cs^2 * rho` — at `S = Gv`. replaces the hand-written `iso_c2p` Expr builder.
///
/// `IsoModel`'s `prim.pre` is a ZST: the HOST runtime elides pressure storage and
/// recomputes `cs^2 * rho` in the flux. the SUBSTRATE stores a real `prim.pre` field
/// (the iso face flux PLM-reconstructs it), so the materialized closure is traced
/// explicitly here — the value (`cs^2 * rho`) is the single source either way.
///
/// iso c2p is geometry-independent and ncomp == ndim (the cyl r-z swirl has no iso c2p),
/// so the `<D>` instance is a complete drop-in: no geom branch, no retained builder.
pub fn iso_c2p_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding: the conserved fields + the prescribed per-cell sound-speed-squared
    // field `cs2` (the local temperature; global isothermal is a uniform cs2). NO scalar —
    // cs2 is a FIELD so the run can be LOCALLY isothermal (cs varies per cell).
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..D)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let cs2 = Gv::field("cs2", "cs2");

    // the SINGLE-SOURCE physics: symbi-hydro's LOCALLY-isothermal recovery (state.rs
    // `locally_isothermal_recover`) — `Cons::to_primitive` with the Isothermal eos reads
    // cs^2 from the nrg slot: rho = den, vel = mom/rho, p = recover_pressure = cs2 * rho.
    // the cs2 is the separate prescribed field, fed through the compute struct's nrg slot.
    let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
    let cons = Cons::<Gv, D> { den, mom: Tensor::new(mom_arr), nrg: cs2 };
    let prim = cons.to_primitive(&Isothermal { cs: Gv::ONE }); // cs unused: recover reads nrg

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..D {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}

/// trace the REAL SRHD c2p — symbi-hydro's branch-free `srhd_recover` (the iterative
/// relativistic cons->prim: a carrier-generic Newton on the pressure root, then the
/// algebraic velocity/Lorentz/density recovery) at `S = Gv`. the Newton lowers to one
/// `Op::IterateInline` (body traced once); `max_iters` bakes the fixed loop count. this
/// is the FIRST iterative gv kernel — replaces the hand-written `srhd_c2p` Expr builder.
///
/// numerically equivalent within ULP, NOT bit-identical (the builder hand-cancels rho in
/// `c2`/`h`; the EOS-generic form keeps `eos.pressure`/`sound_speed_sq`/explicit `h`).
/// the host wrapper's input guard + post-hoc diagnostics are host-only — the kernel
/// computes the raw recovery, exactly as the substrate already does.
pub fn srhd_c2p_gv<const D: usize>(max_iters: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding: the conserved fields + the eos scalar, as Gv leaves.
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..D)
        .map(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let gamma = Gv::scalar("gamma");

    // the SINGLE-SOURCE physics, instantiated at the tracing carrier.
    let mom_arr: [Gv; D] = mom.try_into().expect("D momentum components");
    let cons = Cons::<Gv, D> { den, mom: Tensor::new(mom_arr), nrg };
    let prim = srhd_recover(&IdealGas { gamma }, &cons, max_iters);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..D {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}

/// trace the REAL RMHD c2p — symbi-hydro's branch-free `rmhd_recover` (the KKC
/// false-position: a 6-state bracketed iterate over `kkc_fmu44` + `find_mu_plus`,
/// Illinois half-damp, sticky `done`) at `S = Gv`. the LAST + hardest c2p: the
/// bracketed solve lowers to a multi-accumulator `Op::IterateInline` via the new
/// `Scalar::iterate_vec`. replaces the hand-written `rmhd_c2p` Expr builder.
///
/// RMHD vectors are ALWAYS 3-component (the physics is 3D; grid symmetry handles the
/// 1D/2D cases), so this always traces `rmhd_recover::<Gv, 3>` — `ndim` only selects
/// the emit grid loop. reads the 8-field conserved (den, mom_{0,1,2}, nrg, mag_{0,1,2})
/// + gamma; writes (rho, vel_{0,1,2}, pre). B passes through (CT-evolved, not recovered).
pub fn rmhd_c2p_gv(max_iters: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // input binding, in the substrate's field-read order: den, mom, nrg (tau), mag, gamma.
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
    let gamma = Gv::scalar("gamma");

    // the SINGLE-SOURCE physics at the tracing carrier (3-component RMHD state).
    let cons = MhdCons::<Gv, 3> {
        hydro: Cons { den, mom: Tensor::new(mom), nrg },
        mag: Tensor::new(mag),
    };
    let prim = rmhd_recover(&IdealGas { gamma }, &cons, max_iters);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..3 {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}

// =============================================================================
// newtonian MHD — the non-relativistic ideal-MHD regime. ALGEBRAIC c2p (no
// iteration -> no current-sheet failure mode), closed-form fast-magnetosonic
// wave speeds (cheap enough to compute inline in the flux from the reconstructed
// face states; NO per-cell materialization needed, unlike the RMHD quartic). all
// three builders trace the SAME `NewtonianMhd` carrier-generic physics validated
// at f64 in symbi-hydro. B passes through c2p unchanged (CT-evolved, not recovered).
// =============================================================================

/// trace the newtonian-MHD c2p — the carrier-safe algebraic `nmhd_recover` at
/// `S = Gv`. binds cons (den, mom, nrg, mag) + gamma; writes the recovered hydro
/// (rho, vel, pre). the host-side `to_primitive` error codes are NOT traced (the
/// math is branch-free; comparisons stay on the host). reads `cons_mag_k` because
/// recovering the gas pressure requires stripping 1/2|B|^2 from the total energy.
pub fn nmhd_c2p_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));
    let gamma = Gv::scalar("gamma");

    let cons = MhdCons::<Gv, 3> {
        hydro: Cons { den, mom: Tensor::new(mom), nrg },
        mag: Tensor::new(mag),
    };
    let prim = nmhd_recover(&IdealGas { gamma }, &cons);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..3 {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), prim.pre.node()));

    (end_trace(), writes)
}

/// trace the newtonian-MHD face flux — PLM-reconstruct the 8-component MHD
/// primitive (rho, v_{0,1,2}, pre, B_{0,1,2}) to the face, then the canonical
/// `riemann::hlle(&NewtonianMhd, ...)`. unlike `rmhd_flux_gv`, the Davis fan
/// speeds are computed INLINE by `hlle` from the reconstructed L/R states (the
/// closed-form magnetosonic is cheap) rather than read from a materialized
/// per-cell field — one fewer kernel. `ndim` is the reconstruction grid; `dir`
/// the sweep axis (RMHD/NMHD are fixed 3D in the velocity/field components).
// shared NMHD face-flux reconstruction: bind gamma + theta, PLM-reconstruct the
// 8-component MHD primitive (rho, v_{0..2}, pre, B_{0..2}) to the face. assumes
// begin_trace() is active. returns the eos + L/R primitives + the sweep normal —
// the solver (HLLE / HLLC / HLLD) is the only thing that differs.
// reconstruct the L/R MHD primitives at the `dir`-grid face. the PLM stencil shifts along
// GRID axis `dir`; the NORMAL is physical component `coord_n` (= axes[dir]; == dir for
// cartesian/identity, [0,2][dir] for cyl r-z) — nhat and the staggered normal-B override
// both index `coord_n`, while the face field is read along grid `dir`.
fn nmhd_reconstruct(ndim: u8, dir: u8, coord_n: usize) -> (IdealGas<Gv>, MhdPrim<Gv, 3>, MhdPrim<Gv, 3>, Tensor<Gv, 3>) {
    let gamma = Gv::scalar("gamma");
    let theta = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), ndim, dir, theta);
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim, dir, theta);
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_b{k}"), &format!("prim.mag[{k}]"), ndim, dir, theta);
        bl.push(l);
        br.push(r);
    }
    // the NORMAL field is the staggered, divergence-free FACE field — read it DIRECTLY (one
    // value at the face, no reconstruction) and override the cell-reconstructed B normal
    // component. Gardiner-Stone (2005) CT-Godunov coupling: reconstructed bcell gives
    // bn_l != bn_r, breaking the Riemann solver's constant-Bn assumption (OT noise/blow-up).
    // the face field is read along GRID axis `dir`; the overridden component is the physical
    // normal `coord_n` (they coincide for cartesian; differ for the cyl r-z swirl/axisym).
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let mk = |rho: Gv, v: &[Gv], p: Gv, b: &[Gv]| MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new([v[0], v[1], v[2]]), pre: p },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    let left = mk(rho_l, &vl, pre_l, &bl);
    let right = mk(rho_r, &vr, pre_r, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);
    (IdealGas { gamma }, left, right, nhat)
}

// the 8 conserved face-flux writes (D, S_{0..2}, nrg, B_{0..2}).
fn nmhd_flux_writes(flux: &MhdCons<Gv, 3>) -> Vec<(String, FieldBind, NodeId)> {
    let mut writes = vec![("flux_den".to_string(), FieldRef::flux_den().into(), flux.den.node())];
    for k in 0..3 {
        writes.push((format!("flux_mom_{k}"), FieldRef::flux_mom(k as u8).into(), flux.mom[k].node()));
    }
    writes.push(("flux_nrg".to_string(), FieldRef::flux_nrg().into(), flux.nrg.node()));
    for k in 0..3 {
        writes.push((format!("flux_mag_{k}"), format!("flux.mag_{k}").into(), flux.mag[k].node()));
    }
    writes
}

pub fn nmhd_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hlle(&NewtonianMhd, &eos, &left, &right, &nhat, Gv::ZERO);
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// NMHD HLLC face flux — `hllc_newtonian` (Li 2005, contact-resolving, transverse-B
/// continuous) on the reconstructed L/R states. inline wave speeds (no ws_l/ws_r).
pub fn nmhd_hllc_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hllc_newtonian(&eos, &left, &right, &nhat, Gv::ZERO, ShockwaveLimiter::Standard);
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// NMHD HLLD face flux — `hlld_newtonian` (Miyoshi-Kusano 2005, full 5-wave). the
/// robust solver: the algebraic c2p + this closed-form HLLD make Orszag-Tang stable.
pub fn nmhd_hlld_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hlld_newtonian(&eos, &left, &right, &nhat, Gv::ZERO);
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// trace the newtonian-MHD CFL wave-speed map — `NewtonianMhd::wave_speeds` (the
/// EXACT closed-form fast magnetosonic, not a bound; it is already cheap) folded
/// with the geometry inverse-width into `lambda = max_d (max(|sl|,|sr|) inv_w_d)`.
/// the SAME speed the flux's HLLE consumes (one physics, two consumers).
pub fn nmhd_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        let (sl, sr) = NewtonianMhd.wave_speeds(&eos, &prim, &nhat);
        lambda = lambda.max(sl.abs().max(sr.abs()) * inv_w[d]);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}

// =============================================================================
// ISOTHERMAL MHD gv builders — the same shapes as the NMHD ones, but over the
// energy-model-generic state at E = IsoModel: the conserved vector is {den, mom,
// mag} (NO nrg), c2p is trivial (rho = den, v = mom/den, no pressure), and the
// closure is p = cs^2 rho (Isothermal EOS, scalar `cs` replaces `gamma`). the
// flux is `IsothermalMhd::to_flux` -> HLLE / the 3-state `hlld_isothermal`.
// =============================================================================

/// trace the isothermal-MHD c2p — trivial inversion (rho = den, v = mom/den); no
/// energy/pressure output. the single source the substrate c2p kernel renders.
pub fn imhd_c2p_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let den = Gv::field("cons_den", FieldRef::cons_den());
    let mom: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mom_{k}"), FieldRef::cons_mom(k as u8)));
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("cons_mag_{k}"), &format!("cons.mag_{k}")));

    let cons = IsoMhdCons::<Gv, 3> {
        hydro: ConsG { den, mom: Tensor::new(mom), nrg: Zero::default() },
        mag: Tensor::new(mag),
    };
    // imhd_recover ignores the EOS (pure kinematics); Gv::ZERO -> no `cs` param.
    let prim = imhd_recover(&Isothermal { cs: Gv::ZERO }, &cons);

    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), prim.rho.node())];
    for k in 0..3 {
        writes.push((format!("prim_vel_{k}"), FieldRef::PrimVel(k as u8).into(), prim.vel[k].node()));
    }
    // no prim.pre — the isothermal closure has no independent pressure.
    (end_trace(), writes)
}

// shared isothermal face-flux reconstruction: bind cs + theta, PLM-reconstruct the
// 7-component iso-MHD primitive (rho, v_{0..2}, B_{0..2}) to the face. NO pre. the
// NORMAL field comes from the staggered face field (bface coupling, see
// nmhd_reconstruct). returns the Isothermal eos + L/R primitives + the sweep normal.
fn imhd_reconstruct(ndim: u8, dir: u8, coord_n: usize) -> (Isothermal<Gv>, IsoMhdPrim<Gv, 3>, IsoMhdPrim<Gv, 3>, Tensor<Gv, 3>) {
    let cs = Gv::scalar("cs");
    let theta = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), ndim, dir, theta);
        vl.push(l);
        vr.push(r);
    }
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_b{k}"), &format!("prim.mag[{k}]"), ndim, dir, theta);
        bl.push(l);
        br.push(r);
    }
    // staggered div-free normal FACE field (Gardiner-Stone CT coupling): read along grid `dir`,
    // override the physical normal component `coord_n` (= axes[dir]). see nmhd_reconstruct.
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let mk = |rho: Gv, v: &[Gv], b: &[Gv]| IsoMhdPrim::<Gv, 3> {
        hydro: PrimG { rho, vel: Tensor::new([v[0], v[1], v[2]]), pre: Zero::default() },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    let left = mk(rho_l, &vl, &bl);
    let right = mk(rho_r, &vr, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);
    (Isothermal { cs }, left, right, nhat)
}

// the 7 conserved face-flux writes (D, S_{0..2}, B_{0..2}) — NO nrg.
fn imhd_flux_writes(flux: &IsoMhdCons<Gv, 3>) -> Vec<(String, FieldBind, NodeId)> {
    let mut writes = vec![("flux_den".to_string(), FieldRef::flux_den().into(), flux.den.node())];
    for k in 0..3 {
        writes.push((format!("flux_mom_{k}"), FieldRef::flux_mom(k as u8).into(), flux.mom[k].node()));
    }
    for k in 0..3 {
        writes.push((format!("flux_mag_{k}"), format!("flux.mag_{k}").into(), flux.mag[k].node()));
    }
    writes
}

/// isothermal-MHD HLLE face flux.
pub fn imhd_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = imhd_reconstruct(ndim, dir, coord_n);
    let flux = hlle(&IsothermalMhd, &eos, &left, &right, &nhat, Gv::ZERO);
    let writes = imhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// isothermal-MHD HLLD face flux — `hlld_isothermal` (Mignone 2007, 3-state).
pub fn imhd_hlld_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = imhd_reconstruct(ndim, dir, coord_n);
    let flux = hlld_isothermal(&eos, &left, &right, &nhat, Gv::ZERO);
    let writes = imhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// isothermal-MHD CFL wave-speed map — `IsothermalMhd::wave_speeds` (fast
/// magnetosonic at a^2 = cs^2) folded with the geometry inverse-width.
pub fn imhd_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let cs = Gv::scalar("cs");
    let eos = Isothermal { cs };
    let prim = IsoMhdPrim::<Gv, 3> {
        hydro: PrimG { rho, vel: Tensor::new(vel), pre: Zero::default() },
        mag: Tensor::new(mag),
    };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        let (sl, sr) = IsothermalMhd.wave_speeds(&eos, &prim, &nhat);
        lambda = lambda.max(sl.abs().max(sr.abs()) * inv_w[d]);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}

// =============================================================================
// face flux — PLM reconstruction (Gv stencil) composed with the carrier-generic
// `riemann::hlle` (symbi-hydro). the reconstruction is codegen-only (the host uses
// the compiled kernel, not a DomainForEach); the HLLE physics is the SINGLE source.
// =============================================================================

/// the moving-mesh grid velocity at the face this thread owns:
/// `vface = mesh_adot_{dir} * x_face + mesh_vtrans_{dir}` with the face
/// coordinate `x_lo + i*dx` along sweep axis `dir` (the thread coordinate on a
/// face domain IS the face index). the dispatch decides the semantics per
/// instance: homologous binds `mesh_adot_{dir} = a_dot/a` with PHYSICAL geometry
/// scalars (so vface = H * r, and zero on non-expanding curvilinear axes);
/// uniform translation binds `mesh_vtrans_{dir} = a_dot` on axis 0. the static
/// binding (both zero) traces arithmetic that is bit-identical to the
/// static flux. the formula assumes uniform spacing — asserted at the
/// evolve entry. the per-axis names are the SAME convention the wave-speed map
/// uses, minted through `MeshScalar` so the trace and the dispatch cannot drift.
fn mesh_face_velocity_gv(dir: u8) -> Gv {
    let mesh_adot = Gv::scalar(&MeshScalar::Adot(dir).name());
    let x_face = Gv::scalar(&format!("x_lo_{dir}"))
        + Gv::coord(dir) * Gv::scalar(&format!("dx_{dir}"));
    mesh_adot * x_face + Gv::scalar(&MeshScalar::Vtrans(dir).name())
}

// shared euler (ideal-gas Newtonian/relativistic) face reconstruction: bind the
// scalar tail (gamma, theta), theta-MC PLM-reconstruct the (rho, vel_{0..D}, pre)
// primitive to the `dir`-grid face, and return the IdealGas eos + L/R primitives +
// the sweep normal + the moving-face velocity. the solver (HLLE / HLLC) is the only
// thing that differs. `ndim` is the reconstruction grid (stencil shifts along grid
// axis `dir`); `coord_n` is the sweep COORDINATE (normal velocity is vel[coord_n]).
fn euler_reconstruct<const D: usize>(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (IdealGas<Gv>, Prim<Gv, D>, Prim<Gv, D>, Tensor<Gv, D>, Gv) {
    // scalars first (manifest order [gamma, theta]); the free-theta theta-MC limiter is
    // regime-generic — theta == 1 reduces it EXACTLY to plain minmod (the hydro default).
    let gamma = Gv::scalar("gamma");
    let theta = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(D);
    let mut vr = Vec::with_capacity(D);
    for k in 0..D {
        let (l, r) = plm_theta_gv(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), ndim, dir, theta);
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim, dir, theta);

    let eos = IdealGas { gamma };
    let vl_arr: [Gv; D] = vl.try_into().expect("D velocity components");
    let vr_arr: [Gv; D] = vr.try_into().expect("D velocity components");
    let left = Prim::<Gv, D> { rho: rho_l, vel: Tensor::new(vl_arr), pre: pre_l };
    let right = Prim::<Gv, D> { rho: rho_r, vel: Tensor::new(vr_arr), pre: pre_r };
    let nhat = Tensor::<Gv, D>::unit(coord_n);
    let vface = mesh_face_velocity_gv(dir);
    (eos, left, right, nhat, vface)
}

// the D+2 conserved face-flux writes (D, S_{0..D}, nrg) for an euler-shaped Cons.
fn euler_flux_writes<const D: usize>(flux: &Cons<Gv, D>) -> Vec<(String, FieldBind, NodeId)> {
    let mut writes = vec![("flux_den".to_string(), FieldRef::flux_den().into(), flux.den.node())];
    for k in 0..D {
        writes.push((format!("flux_mom_{k}"), FieldRef::flux_mom(k as u8).into(), flux.mom[k].node()));
    }
    writes.push(("flux_nrg".to_string(), FieldRef::flux_nrg().into(), flux.nrg.node()));
    writes
}

/// trace an ideal-gas Euler face flux (Newtonian OR relativistic) along sweep `dir` —
/// the gv single source: PLM-reconstruct (rho, every vel_k, pre) to the face, then the
/// canonical `riemann::hlle(regime, IdealGas, L, R, n_hat, 0)` (symbi-hydro). replaces
/// the hand-written `hlle_flux` / `srhd_hlle_flux` Expr builders + their per-component
/// U/F (srhd_side). the reconstruction is a Gv stencil (codegen-only); the HLLE is
/// carrier-generic physics. cartesian: ncomp == ndim == D, sweep coordinate == grid `dir`.
/// generic over the regime (both `Newtonian` and `Srhd` have `Prim<S,D>` / `Cons<S,D>`).
/// `D` is the velocity-component count (ncomp); `ndim` is the reconstruction grid (the
/// stencil shifts along grid axis `dir`); `coord_n` is the sweep COORDINATE (the normal
/// velocity is `vel[coord_n]`, pressure goes on momentum `coord_n`). cartesian: ndim == D,
/// coord_n == dir. cyl r-z: D = 3, ndim = 2, coord_n = axes[dir] (the swirl is the 3rd comp).
fn euler_hlle_flux_gv<const D: usize, R>(
    regime: &R,
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>)
where
    R: Regime<Gv, D, Prim = Prim<Gv, D>, Cons = Cons<Gv, D>>,
{
    begin_trace();
    // the SINGLE-SOURCE physics: reconstructed L/R primitives -> canonical HLLE.
    let (eos, left, right, nhat, vface) = euler_reconstruct::<D>(ndim, dir, coord_n);
    let flux = hlle(regime, &eos, &left, &right, &nhat, vface);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}

/// the adiabatic (ideal-gas Newtonian Euler) face flux — `euler_hlle_flux_gv` at the
/// `Newtonian` regime. replaces the cartesian `hlle_flux(.., has_energy=true)` builder.
/// cartesian: ncomp == ndim == D, sweep coordinate == grid `dir`.
pub fn adiabatic_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_hlle_flux_gv::<D, _>(&Newtonian, D as u8, dir, dir as usize)
}

/// the cyl r-z (axisymmetric swirl) adiabatic face flux: ncomp = 3 (v_phi swirl folds
/// into KE) on a 2D (r, z) grid; the sweep coordinate is `axes[dir]` ([0, 2][dir] — grid
/// axis 1 is the z coordinate). replaces the cyl r-z `hlle_flux` Expr builder.
pub fn adiabatic_flux_cyl_rz_gv(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    let coord_n = [0usize, 2][dir as usize]; // (r, z) grid axes -> coordinates 0, 2
    euler_hlle_flux_gv::<3, _>(&Newtonian, 2, dir, coord_n)
}

/// the SRHD (special-relativistic Euler) face flux — `euler_hlle_flux_gv` at the `Srhd`
/// regime (relativistic U/F/wave speeds via Mignone-Bodo). replaces the `srhd_hlle_flux`
/// Expr builder + `srhd_side`. cartesian-only (srhd has no cyl r-z), ncomp == ndim == D.
pub fn srhd_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_hlle_flux_gv::<D, _>(&Srhd, D as u8, dir, dir as usize)
}

/// the isothermal face flux — ISO-NATIVE through the gv path. traces the iso physics
/// DIRECTLY (no `Regime` trait, no `IdealGas{gamma}` EOS, no energy U/F): U/F have
/// `(den, mom_k)` only, wave speeds use `cs = sqrt(pre / rho)` with prim.pre carrying
/// the locally-isothermal `cs^2(x) * rho` — exactly the substrate's locally-isothermal
/// trick, but the energy nodes never enter the graph in the first place. matches the
/// type-system claim ([[isothermal.rs]]: "zero-overhead isothermal hydrodynamics via
/// the energy model type system") at the gv-trace layer too. ncomp == ndim == D, sweep
/// coordinate == grid `dir` (cartesian).
pub fn iso_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    iso_hlle_flux_gv::<D>(D as u8, dir, dir as usize)
}

/// build the iso HLLE face flux directly using Gv ops — no Regime trait detour, no
/// generic `riemann::hlle` (which would force adiabatic-shaped `Cons<S,D>`). HLLE the
/// algorithm is regime-generic; this is iso-shaped from the first node: U = (den, mom),
/// F = (rho*vn, rho*vn*vel + p*nhat), cs = sqrt(p/rho). ndim is the reconstruction grid
/// (stencil shifts along grid axis `dir`); `coord_n` is the sweep COORDINATE (normal
/// velocity is `vel[coord_n]`, pressure goes on momentum `coord_n`). cartesian: ndim ==
/// D, coord_n == dir.
fn iso_hlle_flux_gv<const D: usize>(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // theta is the only scalar param — iso has no gamma. the substrate's dispatch_flux
    // passes ISO_GAMMA, but `scalars_for` walks the kernel's manifest and only asks for
    // declared scalars, so dropping gamma here drops it from the manifest cleanly.
    let theta = Gv::scalar("theta");

    // primitives at the face: rho, each velocity component, and pre (= cs^2(x) * rho
    // via the substrate's locally-isothermal encoding; the per-cell cs(x) is whatever
    // c2p put into prim.pre).
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl: Vec<Gv> = Vec::with_capacity(D);
    let mut vr: Vec<Gv> = Vec::with_capacity(D);
    for k in 0..D {
        let (l, r) = plm_theta_gv(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), ndim, dir, theta);
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim, dir, theta);

    // iso conserved + flux (the algebra IsoNewtonian writes at Rust level, traced here
    // as Gv ops — no energy slot, so no nrg arithmetic enters the graph). normal-velocity
    // shorthands keep the writes-expression-tree small.
    let vn_l = vl[coord_n];
    let vn_r = vr[coord_n];
    let f_l_den = rho_l * vn_l;
    let f_r_den = rho_r * vn_r;
    let mut f_l_mom: Vec<Gv> = Vec::with_capacity(D);
    let mut f_r_mom: Vec<Gv> = Vec::with_capacity(D);
    let mut u_l_mom: Vec<Gv> = Vec::with_capacity(D);
    let mut u_r_mom: Vec<Gv> = Vec::with_capacity(D);
    for k in 0..D {
        u_l_mom.push(rho_l * vl[k]);
        u_r_mom.push(rho_r * vr[k]);
        let base_l = f_l_den * vl[k];
        let base_r = f_r_den * vr[k];
        // pressure goes on the normal-direction momentum (= delta_{k,coord_n} * p).
        let f_l_k = if k == coord_n { base_l + pre_l } else { base_l };
        let f_r_k = if k == coord_n { base_r + pre_r } else { base_r };
        f_l_mom.push(f_l_k);
        f_r_mom.push(f_r_k);
    }

    // wave speeds: locally-isothermal `cs^2 = pre / rho` (= cs^2(x) since pre carries the
    // per-cell encoding). davis bounds.
    let cs_l = (pre_l / rho_l).sqrt();
    let cs_r = (pre_r / rho_r).sqrt();
    let s_l = (vn_l - cs_l).min(vn_r - cs_r);
    let s_r = (vn_l + cs_l).max(vn_r + cs_r);

    // HLLE arms — same algorithm as `riemann::hlle` (moving-face form: fan
    // comparisons against vface, `f - u*vface` per arm), specialised to iso's
    // (den, mom) tuple. the static binding traces vface = 0*x — exactly zero.
    let vface = mesh_face_velocity_gv(dir);
    let inv = Gv::ONE / (s_r - s_l);
    let den_hll = (f_l_den * s_r - f_r_den * s_l + (rho_r - rho_l) * (s_l * s_r)) * inv;
    let den_u_hll = (rho_r * s_r - rho_l * s_l - f_r_den + f_l_den) * inv;
    let den_flux = Gv::branch(
        s_l.cmp_ge(vface),
        || f_l_den - rho_l * vface,
        || Gv::branch(
            s_r.cmp_le(vface),
            || f_r_den - rho_r * vface,
            || den_hll - den_u_hll * vface,
        ),
    );

    let mut mom_flux: Vec<Gv> = Vec::with_capacity(D);
    for k in 0..D {
        let mom_hll = (f_l_mom[k] * s_r - f_r_mom[k] * s_l
                     + (u_r_mom[k] - u_l_mom[k]) * (s_l * s_r)) * inv;
        let mom_u_hll = (u_r_mom[k] * s_r - u_l_mom[k] * s_l - f_r_mom[k] + f_l_mom[k]) * inv;
        let mk = Gv::branch(
            s_l.cmp_ge(vface),
            || f_l_mom[k] - u_l_mom[k] * vface,
            || Gv::branch(
                s_r.cmp_le(vface),
                || f_r_mom[k] - u_r_mom[k] * vface,
                || mom_hll - mom_u_hll * vface,
            ),
        );
        mom_flux.push(mk);
    }

    let mut writes = vec![("flux_den".to_string(), FieldRef::flux_den().into(), den_flux.node())];
    for k in 0..D {
        writes.push((format!("flux_mom_{k}"), FieldRef::flux_mom(k as u8).into(), mom_flux[k].node()));
    }
    (end_trace(), writes)
}

/// the 3-way minmod for the theta-MC (generalized minmod) limiter, carrier-generic
/// (matches the substrate `minmod3`): the common-signed minimum-magnitude argument iff
/// x,y,z share a strict sign, else 0.
fn minmod3<S: Scalar>(x: S, y: S, z: S) -> S {
    let mn = x.min(y).min(z);
    let mx = x.max(y).max(z);
    let all_pos = mn.cmp_gt(S::ZERO);
    let all_neg = mx.cmp_lt(S::ZERO);
    S::select(all_pos, mn, S::select(all_neg, mx, S::ZERO))
}

/// PLM reconstruct with the free-`theta` theta-MC limiter (matches `plm_reconstruct_theta`,
/// RMHD-only): `plm_slope(vl,vc,vr) = minmod3((vc-vl)*theta, (vr-vl)*0.5, (vr-vc)*theta)`,
/// theta in [1,2] tuning compression (1 == plain minmod). Gv stencil via field_shifted.
fn plm_theta_gv(key: &str, runtime: impl Into<FieldBind>, ndim: u8, dir: u8, theta: Gv) -> (Gv, Gv) {
    let runtime = runtime.into();
    let qm2 = Gv::field_shifted(key, runtime.clone(), ndim, dir, -2);
    let qm1 = Gv::field_shifted(key, runtime.clone(), ndim, dir, -1);
    let q0 = Gv::field_shifted(key, runtime.clone(), ndim, dir, 0);
    let qp1 = Gv::field_shifted(key, runtime, ndim, dir, 1);
    let half = Gv::from_f64(0.5);
    let slope =
        |vl: Gv, vc: Gv, vr: Gv| minmod3((vc - vl) * theta, half * (vr - vl), (vr - vc) * theta);
    let left = qm1 + half * slope(qm2, qm1, q0);
    let right = q0 - half * slope(qm1, q0, qp1);
    (left, right)
}

/// trace the RMHD (relativistic MHD) face flux along sweep `dir` on an `ndim`-grid — the
/// gv single source: theta-MC PLM-reconstruct (rho, vel_{0,1,2}, pre, mag_{0,1,2}) to the
/// face, then `riemann::hlle(Rmhd, IdealGas, L, R, n_hat, 0)` (symbi-hydro — the quartic
/// wave speeds + induction flux, all S::select-traceable). replaces the `rmhd_hlle_flux`
/// Expr builder + `lower_rmhd_side`. RMHD vectors are ALWAYS 3-component; `ndim` selects the
/// reconstruction grid + emit loop. writes the 8 conserved fluxes (D, S_k, tau, B_k).
pub fn rmhd_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // scalar params in the substrate order: gamma (EOS) then theta (limiter compression).
    let gamma = Gv::scalar("gamma");
    let theta = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), ndim, dir, theta);
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim, dir, theta);
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_b{k}"), &format!("prim.mag[{k}]"), ndim, dir, theta);
        bl.push(l);
        br.push(r);
    }

    // the SINGLE-SOURCE physics: reconstructed L/R MHD primitives -> canonical HLLE.
    let eos = IdealGas { gamma };
    let mk = |rho: Gv, v: &[Gv], p: Gv, b: &[Gv]| MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new([v[0], v[1], v[2]]), pre: p },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    // normal B from the staggered FACE field (Gardiner-Stone CT coupling) — reconstructed
    // bcell gives bn_l != bn_r, breaking the constant-Bn assumption. see nmhd_reconstruct.
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let left = mk(rho_l, &vl, pre_l, &bl);
    let right = mk(rho_r, &vr, pre_r, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);

    // wave speeds are NO LONGER recomputed here — they are materialized once per cell by
    // rmhd_wave_speeds_cell_gv into wave_speed_l[dir]/wave_speed_r[dir] (the exact quartic).
    // the HLL fan is the cell-centered Davis estimate over the two cells sharing this face:
    // plm_theta_gv reconstructs L from cell `coord - e_dir` (offset -1) and R from cell `coord`
    // (offset 0), so the fan reads those same two cells' speeds. the rmhd zero-clamp is applied
    // here (the stored per-cell speeds are raw). this strips the 166-register / 12-transcendental
    // quartic out of the flux kernel entirely.
    let dim = ndim;
    let lo = format!("wave_speed_l[{dir}]");
    let hi = format!("wave_speed_r[{dir}]");
    let wsl_m1 = Gv::field_shifted("ws_l", &lo, dim, dir, -1);
    let wsl_0 = Gv::field_shifted("ws_l", &lo, dim, dir, 0);
    let wsr_m1 = Gv::field_shifted("ws_r", &hi, dim, dir, -1);
    let wsr_0 = Gv::field_shifted("ws_r", &hi, dim, dir, 0);
    let s_l = wsl_m1.min(wsl_0).min(Gv::ZERO);
    let s_r = wsr_m1.max(wsr_0).max(Gv::ZERO);
    let flux = hlle_with_speeds(&Rmhd, &eos, &left, &right, &nhat, Gv::ZERO, s_l, s_r);

    let mut writes = vec![("flux_den".to_string(), FieldRef::flux_den().into(), flux.den.node())];
    for k in 0..3 {
        writes.push((format!("flux_mom_{k}"), FieldRef::flux_mom(k as u8).into(), flux.mom[k].node()));
    }
    writes.push(("flux_nrg".to_string(), FieldRef::flux_nrg().into(), flux.nrg.node()));
    for k in 0..3 {
        writes.push((format!("flux_mag_{k}"), format!("flux.mag_{k}").into(), flux.mag[k].node()));
    }

    // Gate 3 smem tile (docs/design/22): reconstruction is 1D ALONG `dir`, so the
    // tile is a thin SLAB — halo on axis `dir` only, transverse axes unextended.
    // the tiled set is derived from the graph (the shifted `LoadAt` fields: the 8
    // reconstructed prim + the 2 per-cell wave speeds), NOT a hand-kept list.
    let k = end_trace();
    let stencil_keys = k.stencil_read_field_keys();
    if stencil_keys.is_empty() {
        return (k, writes);
    }
    let mut halo = vec![0u8; ndim as usize];
    halo[dir as usize] = 2;
    let k = k.with_tile_spec(TileSpec { halo, tiled_field_keys: stencil_keys });
    (k, writes)
}

// =============================================================================
// HLLC face flux — contact-resolving 3-wave solver, regime-specific bodies. one
// builder per regime (Newtonian, SRHD, RMHD) mirroring the HLLE builder shape:
// same PLM reconstruction, same scalar tail (gamma, theta), same write manifest.
// the Riemann solver is the only structural difference. defaulted to the
// Standard shock-smoother arm at trace time — Quirk/Fleischmann are host-time
// dispatch knobs not exposed through the substrate yet.
// =============================================================================

/// adiabatic (ideal-gas Newtonian Euler) HLLC face flux. mirrors
/// `euler_hlle_flux_gv(&Newtonian, ...)` but calls `riemann::hllc` instead of
/// `riemann::hlle`. carrier-generic over Gv; iso is structurally excluded
/// (no contact wave -> HLLE-only).
pub fn adiabatic_hllc_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat, vface) = euler_reconstruct::<D>(D as u8, dir, dir as usize);
    let flux = hllc(&eos, &left, &right, &nhat, vface, ShockwaveLimiter::Standard);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}

/// SRHD HLLC face flux — Mignone-Bodo (2005) quadratic for the contact speed.
/// mirrors `euler_hlle_flux_gv(&Srhd, ...)` but calls `riemann::hllc_srhd`.
pub fn srhd_hllc_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat, vface) = euler_reconstruct::<D>(D as u8, dir, dir as usize);
    let flux = hllc_srhd(&eos, &left, &right, &nhat, vface, ShockwaveLimiter::Standard);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}

/// RMHD HLLC face flux — Mignone-Bodo (2006), null vs non-null normal B-field
/// branch. mirrors `rmhd_flux_gv` (8-component MHD primitive) but routes the
/// reconstructed L/R state through `riemann::hllc_rmhd` instead of `hlle`.
pub fn rmhd_hllc_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hllc_rmhd(&Rmhd, &eos, &left, &right, &nhat, Gv::ZERO, ShockwaveLimiter::Standard);
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// RMHD HLLD face flux — Mignone, Ugliano & Bodo (2009) 5-wave solver, the
/// full magnetosonic/Alfven/contact wave resolution. uses `Scalar::iterate_vec`
/// for the 15-step secant on pressure (freeze-on-converged), eagerly computes
/// HLLE as the divergence fallback, and selects via a success mask at the end.
/// shares the MHD primitive shape with HLLE/HLLC.
pub fn rmhd_hlld_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hlld_rmhd(&Rmhd, &eos, &left, &right, &nhat, Gv::ZERO);
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

// =============================================================================
// CFL wave-speed MAP — symbi-hydro's `Regime::wave_speeds_axis` traced at S=Gv into the
// COMPLETE timestep kernel: per gridded axis the characteristic speed `s_d = max(|sl|,|sr|)`,
// folded against the per-cell inverse PHYSICAL width into `lambda = max_d (s_d * inv_w_d)`.
// ONE gv trace owns physics + geometry + reduction — the regime supplies only
// `wave_speeds_axis` (the SAME function the flux's HLLE uses), the geometry is the in-kernel
// Gv metric (`cfl_inv_widths_gv`). NO Expr quartic, NO splice, NO hand-written iso speed.
//
// `axes[d]` is the COORDINATE gridded axis `d` maps to: identity for cartesian/spherical,
// `[0, 2]` for the cyl r-z swirl. the Euler map reads the normal velocity `vel[axes[d]]`
// directly (`wave_speeds_axis` reads only the normal) and leaves the non-gridded velocity
// slots ZERO — so the swirl CFL reads v_r/v_z but never the folded v_phi, and those zeroed
// slots never enter the graph (dead). the host reduces `lambda` by max -> dt = cfl/lambda_max.
// =============================================================================

/// the per-gridded-axis inverse PHYSICAL width `1/(h_d*width_d)` for the CFL — the Gv mirror
/// of the retired `flux::cfl_inv_widths`. cartesian + uniform: the host's precomputed
/// `inv_dx_d` scalar (one per gridded axis); curvilinear OR non-uniform: the per-cell width
/// from the index (`cell_inv_phys_widths_gv`, so log zones + angular `h_d=r` are tracked). the
/// regime never writes the width — it only supplies wave speeds — so it can never desync it.
fn cfl_inv_widths_gv(coords: Coords, spacing: &[Spacing], axes: &[usize], ndim: usize) -> Vec<Gv> {
    let uniform_cartesian =
        coords == Coords::Cartesian && spacing.iter().all(|&s| s == Spacing::Uniform);
    if uniform_cartesian {
        (0..ndim).map(|d| Gv::scalar(&format!("inv_dx_{d}"))).collect()
    } else {
        cell_inv_phys_widths_gv(coords, spacing, axes, ndim)
    }
}

/// the lambda write list every wave-speed map returns: one scratch output `lambda`.
fn wave_speed_map_writes(root: NodeId) -> Vec<(String, FieldBind, NodeId)> {
    vec![("lambda".to_string(), FieldRef::Scratch.into(), root)]
}

/// trace the COMPLETE ideal-gas Euler CFL wave-speed map at `S = Gv` — the Newtonian regime
/// (which also drives the isothermal CFL at gamma->1) or `Srhd`. reads rho/pre + the gridded
/// normal velocities `vel[axes[d]]` (non-gridded slots left ZERO; `wave_speeds_axis` reads
/// only the normal, so they stay dead) + gamma, then folds `lambda = max_d (max(|sl|,|sr|) *
/// inv_w_d)` over the gridded axes with the in-kernel geometry widths. ONE trace: physics +
/// metric + reduction, replacing the splice-into-`flux::wave_speed_map` composition. always
/// 3-component (the swirl shares the form); `coords`/`spacing`/`axes` select plane + metric.
pub fn euler_wave_speed_map_gv<R>(
    regime: &R,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>)
where
    R: Regime<Gv, 3, Prim = Prim<Gv, 3>, Cons = Cons<Gv, 3>>,
{
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    // the gridded normal velocities only; the non-gridded slots (cyl r-z's v_phi) stay ZERO and
    // never enter the graph — `wave_speeds_axis` reads only the normal velocity `vel[axes[d]]`.
    let mut vel = [Gv::ZERO; 3];
    for d in 0..ndim {
        let c = axes[d];
        vel[c] = Gv::field(&format!("prim_v{c}"), FieldRef::PrimVel(c as u8));
    }
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = Prim::<Gv, 3> { rho, vel: Tensor::new(vel), pre };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    // mesh motion: the cfl signal speed is RELATIVE to the grid, `|s -+ v_g|`
    // with per-axis `v_g = mesh_adot_d * x_centroid + mesh_vtrans_d`
    // (uniform-spacing centroid; the dispatch binds the homologous hubble
    // rate on expanding axes, the translation rate on axis 0, zero
    // otherwise — the static binding makes v_g exactly zero, and `s - 0` /
    // `|s|` are bit-identical).
    let half = Gv::from_f64(0.5);
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let (sl, sr) = regime.wave_speeds_axis(&eos, &prim, axes[d]);
        let xc = Gv::scalar(&format!("x_lo_{d}"))
            + (Gv::coord(d as u8) + half) * Gv::scalar(&format!("dx_{d}"));
        let vg = Gv::scalar(&MeshScalar::Adot(d as u8).name()) * xc
            + Gv::scalar(&MeshScalar::Vtrans(d as u8).name());
        lambda = lambda.max((sl - vg).abs().max((sr - vg).abs()) * inv_w[d]);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}

/// the Newtonian / isothermal CFL wave-speed map (gamma->1 drives isothermal, 1.4 adiabatic) —
/// `Newtonian::wave_speeds_axis` (`|v_d| + cs`) traced to the complete timestep kernel.
pub fn iso_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_wave_speed_map_gv(&Newtonian, coords, spacing, axes, ndim)
}

/// the SRHD CFL wave-speed map — the relativistic Mignone-Bodo per-axis speed (`Srhd::
/// wave_speeds_axis`, the SAME core the SRHD flux's HLLE consumes) traced to the timestep kernel.
pub fn srhd_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_wave_speed_map_gv(&Srhd, coords, spacing, axes, ndim)
}

/// trace the RMHD CFL wave-speed map at `S = Gv` — the MAGNETOSONIC UPPER BOUND
/// (`rmhd_magnetosonic_cfl_speeds`), NOT the full Mignone & Del Zanna quartic. the CFL needs
/// only a stable upper bound on the signal speed, and the bound is ~25x cheaper than the
/// quartic (~30 ops + 1 sqrt vs ~750 ops + ~10 transcendentals, ALL of which trace into the
/// kernel because `S::select` evaluates every arm). the quartic stays on the Riemann/flux
/// path (`rmhd_flux_gv` -> extremal_speeds), where HLLE diffusion needs the tight estimate.
/// see docs/c9fbdcb_perf_study/02. reads the full 3-velocity + 3-magnetic-field (vsq/bsq).
/// RMHD is fixed 3D (identity axes); the folded geometry + max-reduction is the same single-
/// trace form as the Euler map.
pub fn rmhd_wave_speed_map_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let inv_w = cfl_inv_widths_gv(coords, spacing, axes, ndim);
    let mut lambda = Gv::ZERO;
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(axes[d]);
        let (sl, sr) = rmhd_magnetosonic_cfl_speeds(&eos, &prim, &nhat);
        lambda = lambda.max(sl.abs().max(sr.abs()) * inv_w[d]);
    }
    let writes = wave_speed_map_writes(lambda.node());
    (end_trace(), writes)
}

/// trace the PER-CELL RMHD wave speeds — the EXACT Mignone & Del Zanna quartic
/// (`Rmhd::wave_speeds`, raw min/max, NO zero-clamp) evaluated once per cell for each of the
/// 3 directions, writing `wave_speed_l[d] = lambda_min^d` and `wave_speed_r[d] = lambda_max^d`.
///
/// this lifts the wave speed OFF the per-face flux (where it was recomputed for L and R at
/// every face — the dominant 166-register, 12-transcendental cost) onto the cell index space:
/// the quartic's direction-INDEPENDENT guts (rho/h/c_s^2/b_mu^2/...) are CSE-shared across the
/// 3 directions; only vn/bn and the resolvent differ. the flux then reads the two adjacent
/// cells' stored speeds for the Davis fan (`hlle_with_speeds`), and CFL folds the same fields
/// — one computation, three consumers (flux, CFL, UCT-HLL). the zero-clamp for the HLL fan is
/// applied at the FLUX (min/max of the two cells, clamped), so the stored values are raw.
/// RMHD is fixed 3D. reads the full 3-velocity + 3-magnetic-field prim + gamma.
pub fn rmhd_wave_speeds_cell_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    // one (lmin,lmax) pair per spatial sweep direction (ndim of them); the B is always
    // a 3-vector but the grid varies along ndim axes only.
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        // raw quartic min/max — NOT extremal_speeds (no zero-clamp); the flux clamps the fan.
        let (lmin, lmax) = Rmhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((format!("ws_l_{d}"), format!("wave_speed_l[{d}]").into(), lmin.node()));
        writes.push((format!("ws_r_{d}"), format!("wave_speed_r[{d}]").into(), lmax.node()));
    }
    (end_trace(), writes)
}

/// the CLASSICAL (Newtonian ideal-gas) per-cell wave speeds (`NewtonianMhd::wave_speeds` = the fast
/// magnetosonic bound, lmin/lmax = v_n -/+ c_f). materializes `wave_speed_l/r[dir]` so UCT (which
/// reads them for the edge-EMF coefficients) works for NMHD — the classical regimes compute speeds
/// inline in the flux and otherwise do NOT store them. mirror of `rmhd_wave_speeds_cell_gv`.
pub fn nmhd_wave_speeds_cell_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("prim_rho", FieldRef::PrimRho);
    let vel: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let pre = Gv::field("prim_pre", FieldRef::PrimPre);
    let mag: [Gv; 3] =
        std::array::from_fn(|k| Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8)));
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let prim = MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new(vel), pre },
        mag: Tensor::new(mag),
    };
    let mut writes = Vec::with_capacity(2 * ndim);
    for d in 0..ndim {
        let nhat = Tensor::<Gv, 3>::unit(d);
        let (lmin, lmax) = NewtonianMhd.wave_speeds(&eos, &prim, &nhat);
        writes.push((format!("ws_l_{d}"), format!("wave_speed_l[{d}]").into(), lmin.node()));
        writes.push((format!("ws_r_{d}"), format!("wave_speed_r[{d}]").into(), lmax.node()));
    }
    (end_trace(), writes)
}

// =============================================================================
// the conserved-update GODUNOV family in Gv — the finite-volume divergence (the Gv stencil
// `field_shifted(F_i, +e_i) - field(F_i)`, no `MorphismKind::Diff`) composed with the
// forward-Euler / RK2 time update over the conserved set (mass + one scalar law per momentum
// component + optional energy), and the snapshot copy. EOS- AND geometry-generic: snapshot is
// a pure copy (every coord); the CARTESIAN-uniform divergence is `(F_hi - F_lo)/dx_i`, the
// CURVILINEAR is the analytic area-weighted `(1/V)(F_hi*A_hi - F_lo*A_lo)` from the in-kernel
// `cell_geometry_gv` metric, plus the geometric momentum source `S^i = -Gamma^i_jk T^jk`
// (the gv christoffel) on the curvilinear momentum laws. ONE trace per (regime, geom).
// =============================================================================

/// which geometric momentum source the CURVILINEAR godunov adds to the momentum laws. the
/// flux divergence + the conserved-law structure are identical across regimes; only the source
/// expression differs (the "one operator, regime supplies its physics" rule). cartesian = none.
#[derive(Clone, Copy)]
pub enum GeoSource {
    /// hydro / SRHD: well-balanced pressure + (ndim>=2) velocity-quadratic centrifugal/coriolis,
    /// regime-agnostic via the CONSERVED momentum (Newtonian `mom=rho v`, SRHD `mom=rho h W^2 v`).
    Hydro { inertial: bool },
    /// RMHD: pressure + inertial + magnetic tension, from `rmhd_source_quantities` (cons.mom
    /// carries B-momentum so it can't serve the source).
    Rmhd,
    /// newtonian MHD: pressure (p + 1/2|B|^2) + gas inertial (cons.mom IS rho v — the Maxwell
    /// stress lives in the flux, not the momentum) + magnetic tension from the LAB-FRAME B (no
    /// relativistic four-vector). simpler than RMHD: cons.mom serves the inertial directly.
    NewtonianMhd,
    /// isothermal MHD: identical to `NewtonianMhd` but the gas pressure is `cs^2 rho` (no
    /// energy / `prim.pre`); the closure scalar `cs` is read in-kernel.
    IsothermalMhd,
}

/// one cartesian-uniform finite-volume divergence sum over the gridded axes:
/// `sum_i (F_i[coord+e_i] - F_i[coord]) / dx_i`. `base` names the per-direction flux field
/// (`{base}_{i}`, runtime `{base}[{i}]`) — `mass_flux` / `mom_flux_{k}` / `nrg_flux`. the lo
/// read is the direct cell read, the hi a `+e_i` field_shifted (LoadAt); dt is the caller's.
fn gv_divergence_cartesian(base: &str, ndim: u8) -> Gv {
    let mut acc: Option<Gv> = None;
    for ii in 0..ndim {
        let key = format!("{base}_{ii}");
        let rt = format!("{base}[{ii}]");
        let f_lo = Gv::field_shifted(&key, &rt, ndim, ii, 0); // == Gv::field (offset 0)
        let f_hi = Gv::field_shifted(&key, &rt, ndim, ii, 1);
        let dx = Gv::scalar(&format!("dx_{ii}"));
        let term = (f_hi - f_lo) / dx;
        acc = Some(match acc {
            None => term,
            Some(a) => a + term,
        });
    }
    acc.expect("godunov divergence needs ndim >= 1")
}

/// the analytic AREA-WEIGHTED curvilinear divergence: `(1/V) sum_i (F_i[+e_i]*A_hi_i -
/// F_i*A_lo_i)` — each face flux weighted by its face area BEFORE the telescope, the cell sum
/// scaled by `1/V`. the gv mirror of `finite_volume::divergence_sum_weighted`; `geo` carries
/// the in-kernel per-cell areas + inverse volume from `cell_geometry_gv`.
fn gv_divergence_weighted(base: &str, ndim: u8, geo: &CellGeometryGv) -> Gv {
    let mut acc: Option<Gv> = None;
    for ii in 0..ndim {
        let key = format!("{base}_{ii}");
        let rt = format!("{base}[{ii}]");
        let f_lo = Gv::field_shifted(&key, &rt, ndim, ii, 0);
        let f_hi = Gv::field_shifted(&key, &rt, ndim, ii, 1);
        let d = ii as usize;
        let diff = f_hi * geo.area_hi[d] - f_lo * geo.area_lo[d];
        acc = Some(match acc {
            None => diff,
            Some(a) => a + diff,
        });
    }
    acc.expect("godunov divergence needs ndim >= 1") * geo.inv_volume
}

/// the centrifugal/coriolis INERTIAL momentum source per component `S^i = -Gamma^i_jk mom^j
/// v^k` (the velocity-quadratic geometric terms), in Gv. REGIME-AGNOSTIC via the conserved `mom`
/// (Newtonian rho v -> rho v^2; relativistic rho h W^2 v -> wgam2 v^2). `centroid` is
/// COORDINATE-indexed (r at [0], theta at [1]); one source per carried component (`ncomp`).
fn inertial_momentum_sources_gv(
    ncomp: usize,
    coords: Coords,
    mom: &[Gv],
    vel: &[Gv],
    centroid: &[Gv],
) -> Vec<Gv> {
    let mut s = vec![Gv::ZERO; ncomp];
    if coords == Coords::Cartesian {
        return s; // flat space: no inertial source.
    }
    let inv_r = Gv::ONE / centroid[0]; // 1/r (centroid radial)
    match coords {
        // spherical (r, theta, phi): each `rho v_a v_b` term is `mom_a v_b`.
        Coords::Spherical => {
            // S_r = (sum_t mom_t v_t) / r — the carried transverse components.
            let mut sum: Option<Gv> = None;
            for t in 1..ncomp {
                let mvt = mom[t] * vel[t];
                sum = Some(match sum {
                    None => mvt,
                    Some(a) => a + mvt,
                });
            }
            if let Some(sum) = sum {
                s[0] = sum * inv_r;
            }
            if ncomp >= 2 {
                let cot = centroid[1].cos() / centroid[1].sin();
                // S_theta = (mom_phi v_phi cot - mom_r v_theta) / r.
                let mut num = Gv::ZERO - mom[0] * vel[1];
                if ncomp >= 3 {
                    num = num + mom[2] * vel[2] * cot;
                }
                s[1] = num * inv_r;
                if ncomp >= 3 {
                    // S_phi = -mom_phi (v_r + v_theta cot) / r.
                    let inner = vel[0] + vel[1] * cot;
                    s[2] = Gv::ZERO - mom[2] * inner * inv_r;
                }
            }
        }
        // cylindrical (r, phi, z): phi (component 1) is the swirl; z has no inertial source.
        Coords::Cylindrical => {
            if ncomp >= 2 {
                s[0] = mom[1] * vel[1] * inv_r; // S_r = mom_phi v_phi / r
                s[1] = Gv::ZERO - mom[0] * vel[1] * inv_r; // S_phi = -mom_r v_phi / r
            }
        }
        Coords::Cartesian => unreachable!(),
    }
    s
}

/// the FULL geometric momentum source per component `S^i = -Gamma^i_jk T^jk` in Gv, split
/// into the three pieces every
/// regime shares: well-balanced PRESSURE `ptot*(A_hi - A_lo)*inv_V`, INERTIAL `-Gamma(wgam2 v
/// v)`, and (RMHD) MAGNETIC `+Gamma(bmu bmu)`. `gas_mom`/`vel`/`bmu` are the regime quantities;
/// pass EMPTY `gas_mom` to skip the inertial (1D radial). `axes[d]` = the coord of grid axis d.
fn geometric_momentum_sources_gv(
    coords: Coords,
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    geo: &CellGeometryGv,
    ptot: Gv,
    gas_mom: &[Gv],
    vel: &[Gv],
    bmu: Option<&[Gv]>,
) -> Vec<Gv> {
    // COORDINATE-indexed centroid (r at [0], theta at [1]); symmetry slots stay 0 (never read).
    let mut coord_centroid = vec![Gv::ZERO; 3];
    for d in 0..ndim {
        coord_centroid[axes[d]] = geo.centroid[d];
    }
    let inertial = (!gas_mom.is_empty())
        .then(|| inertial_momentum_sources_gv(ncomp, coords, gas_mom, vel, &coord_centroid));
    // the magnetic tension is the SAME christoffel on the four-vector, negated.
    let mag = bmu.map(|b| inertial_momentum_sources_gv(ncomp, coords, b, b, &coord_centroid));
    (0..ncomp)
        .map(|coord| {
            // PRESSURE: only a GRIDDED coordinate has a pressure gradient; written in the
            // divergence's (ptot*A_hi - ptot*A_lo)*inv_V form so a v=0 uniform-ptot state
            // cancels the pressure flux divergence bit-exactly (well-balanced HSE).
            let mut s = if let Some(d) = axes.iter().position(|&c| c == coord) {
                (ptot * geo.area_hi[d] - ptot * geo.area_lo[d]) * geo.inv_volume
            } else {
                Gv::ZERO
            };
            if let Some(ref inert) = inertial {
                s = s + inert[coord];
            }
            if let Some(ref m) = mag {
                s = s - m[coord]; // - magnetic tension (= +Gamma bmu bmu)
            }
            s
        })
        .collect()
}

/// trace the regime's geometric momentum source quantities + form the per-component source.
/// HYDRO/SRHD: total pressure = `prim.pre`, gas momentum density * v = the CONSERVED momentum
/// (cons.mom IS rho v / rho h W^2 v), no magnetic term. RMHD: the gas + magnetic quantities
/// from symbi-hydro's `rmhd_source_quantities` (the SAME carrier-generic source the RMHD flux
/// uses), gas_mom = wgam2*v, bmu = the spatial four-vector. `cons_mom` is the already-read
/// in-place conserved momentum (shared so the gas-inertial reuses it — no duplicate buffer).
fn gv_geometric_source(
    coords: Coords,
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    geo: &CellGeometryGv,
    source: GeoSource,
    cons_mom: &[Gv],
    mag_from_bcell: bool,
) -> Vec<Gv> {
    // the cell-B the magnetic geo source reads. NORMALLY the primitive `prim.mag[k]`. but when
    // this stage is FUSED with the cell-B predictor (which binds `bc_k` in-place), reading mag
    // via the SAME `bc_k` key lets try_fuse merge the two cell-B reads into ONE binding — without
    // it, `prim.mag[k]` and `bc_k` are distinct manifest entries that both resolve to bcell[k] at
    // runtime, aliasing a read-only input to an in-place output (UB on CPU). both keys carry the
    // SAME old-bcell value (the predictor writes after the source evaluates), so it's bit-identical.
    let mag_field = |k: usize| -> Gv {
        if mag_from_bcell {
            Gv::field(&format!("bc_{k}"), FieldRef::BCell(k as u8))
        } else {
            Gv::field(&format!("prim_b{k}"), FieldRef::PrimMag(k as u8))
        }
    };
    match source {
        GeoSource::Hydro { inertial } => {
            let ptot = Gv::field("pre", FieldRef::PrimPre);
            // the velocity-quadratic inertial vanishes for 1D radial — skip it (+ its vel reads).
            let (gas_mom, vel): (Vec<Gv>, Vec<Gv>) = if inertial && ndim >= 2 {
                let v = (0..ndim)
                    .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                    .collect();
                (cons_mom[..ndim].to_vec(), v) // gas_mom = cons.mom (shared, no duplicate read)
            } else {
                (Vec::new(), Vec::new())
            };
            geometric_momentum_sources_gv(coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel, None)
        }
        GeoSource::Rmhd => {
            // the RMHD stress = pressure + gas inertial + magnetic tension: read prim + gamma,
            // trace symbi-hydro's `rmhd_source_quantities` (wgam2, bmu, ptot) at S=Gv.
            let rho = Gv::field("prim_rho", FieldRef::PrimRho);
            let vel: [Gv; 3] =
                std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
            let pre = Gv::field("prim_pre", FieldRef::PrimPre);
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let eos = IdealGas { gamma: Gv::scalar("gamma") };
            let prim = MhdPrim::<Gv, 3> {
                hydro: Prim { rho, vel: Tensor::new(vel), pre },
                mag: Tensor::new(mag),
            };
            let (wgam2, bmu, ptot) = rmhd_source_quantities(&eos, &prim);
            // the inertial + magnetic geometric sources need ALL `ncomp` (DOF) components, not
            // just the `ndim` gridded ones: a 2.5D spherical (r,theta) grid has DOF=3 and the
            // out-of-plane phi momentum (mom[2]) drives the S_theta cot term + the S_phi source.
            let gas_mom: Vec<Gv> = (0..ncomp).map(|k| wgam2 * vel[k]).collect();
            let vel_n: Vec<Gv> = vel[..ncomp].to_vec();
            let bmu_n: Vec<Gv> = (0..ncomp).map(|k| bmu[k]).collect();
            geometric_momentum_sources_gv(
                coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel_n, Some(&bmu_n),
            )
        }
        GeoSource::NewtonianMhd => {
            // newtonian MHD stress: ptot = p + 1/2|B|^2; gas inertial via cons.mom (= rho v,
            // pure gas); magnetic tension via the lab-frame B (the Maxwell stress -B_i B_j has
            // the SAME christoffel form as the inertial, so it reuses the inertial builder, then
            // is subtracted by geometric_momentum_sources_gv). no wgam2 / four-vector.
            // ALL `ncomp` (DOF) components: a 2.5D spherical grid (DOF=3 > ndim=2) needs the
            // out-of-plane phi velocity/momentum/B for the S_theta cot + S_phi geometric sources.
            let vel: Vec<Gv> = (0..ncomp)
                .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                .collect();
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let pre = Gv::field("prim_pre", FieldRef::PrimPre);
            let bsq = mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2];
            let ptot = pre + Gv::from_f64(0.5) * bsq;
            let gas_mom: Vec<Gv> = cons_mom[..ncomp].to_vec();
            let mag_n: Vec<Gv> = (0..ncomp).map(|k| mag[k]).collect();
            geometric_momentum_sources_gv(
                coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel, Some(&mag_n),
            )
        }
        GeoSource::IsothermalMhd => {
            // isothermal MHD stress: ptot = cs^2 rho + 1/2|B|^2 (no prim.pre; cs is a scalar).
            // otherwise identical to NewtonianMhd (gas inertial via cons.mom, lab-frame B tension).
            // ALL `ncomp` (DOF) components (see NewtonianMhd) for the spherical 2.5D out-of-plane source.
            let vel: Vec<Gv> = (0..ncomp)
                .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
                .collect();
            let mag: [Gv; 3] = std::array::from_fn(|k| mag_field(k));
            let rho = Gv::field("prim_rho", FieldRef::PrimRho);
            let cs = Gv::scalar("cs");
            let bsq = mag[0] * mag[0] + mag[1] * mag[1] + mag[2] * mag[2];
            let ptot = cs * cs * rho + Gv::from_f64(0.5) * bsq;
            let gas_mom: Vec<Gv> = cons_mom[..ncomp].to_vec();
            let mag_n: Vec<Gv> = (0..ncomp).map(|k| mag[k]).collect();
            geometric_momentum_sources_gv(
                coords, axes, ndim, ncomp, geo, ptot, &gas_mom, &vel, Some(&mag_n),
            )
        }
    }
}

/// the per-direction inverse divergence operator for `base`: cartesian-uniform `(F_hi -
/// F_lo)/dx_i`, else the area-weighted `(1/V)(F_hi*A_hi - F_lo*A_lo)` from `geo`.
fn gv_divergence(base: &str, ndim: u8, geo: &Option<CellGeometryGv>) -> Gv {
    match geo {
        None => gv_divergence_cartesian(base, ndim),
        Some(g) => gv_divergence_weighted(base, ndim, g),
    }
}

/// `true` iff the flat unweighted `(F_hi-F_lo)/dx` divergence applies (no in-kernel metric).
fn is_cartesian_uniform(coords: Coords, spacing: &[Spacing]) -> bool {
    coords == Coords::Cartesian && spacing.iter().all(|&s| s == Spacing::Uniform)
}

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

/// the single mass-law godunov step to a SEPARATE output buffer (the P2.2 demo):
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
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    godunov_stage_gv_with_fused_sources(coords, spacing, axes, ndim, ncomp, has_energy, source, &[], false)
}

/// per-field NodeId contributions from a list of spec sources, bucketed by
/// `target_field`. consumed by `godunov_stage_gv_with_fused_sources` — the spec
/// vocabulary is spliced once, then each conserved law adds its bucket inside the
/// forward-Euler stage.
///
/// **structural shape contract**: spliced outputs MUST have the expected per-target
/// arity (1 for den/nrg, D for mom); spec authors that violate this get a panic, not a
/// silent wrong-component write.
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

/// **B6-iv Phase 4c/2c — fused-source splice helper**. requires an ACTIVE Gv trace
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
    // the STATE vocabulary the DAG reads `rho`/`vel_k` from (docs/design/33 `StateEnv`). `Some((rho,
    // mom))` binds them (sources read the stage/conserved state); `None` binds NOTHING from state — a
    // pure coordinate prescription (a driven boundary, whose DAG outputs the state rather than reading
    // it). `x_k` (centroid) + scalar params are bound regardless.
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
        // pressure-reading sources (e.g. radiative cooling Lambda(rho, T), T = pre/rho): bind `pre`
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
    // **Phase 2c — LAZY centroid binding**. `x_k` ↔ cell centroid for specs
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
    // (e.g. `g_ext_0` in the mom + nrg specs of uniform_acceleration)
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
/// `fe(u, div, src) = u - dt*div + dt*(geo_src + Σ spec_src)`. one launch folds flux
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
    godunov_stage_gv_with_fused_built(
        coords, spacing, axes, ndim, ncomp, has_energy, source, &src_refs, mag_from_bcell,
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
    spacing: &[Spacing],
    axes: &[usize],
    ndim: u8,
    ncomp: usize,
    has_energy: bool,
    source: GeoSource,
    sources: &[(&str, &symbi_hydro::source_spec::BuiltSource)],
    mag_from_bcell: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let a0 = Gv::scalar("a0");
    let ac = Gv::scalar("ac");
    // the SSP source weight. computed as `ac*dt` so it is BIT-IDENTICAL to the standalone
    // `source_apply_gv` pass's `dt` scalar (the driver fills that with `ac*sim.dt` — the same IEEE
    // f64 product). this is what makes `fused == plain godunov + source_apply` exact, not just
    // ULP-close: the user source is added as a SEPARATE post-combine term with this weight, never
    // folded into the `ac*fe` multiply (which would distribute the rounding differently).
    let ac_dt = ac * dt;
    let geo = (!is_cartesian_uniform(coords, spacing))
        .then(|| cell_geometry_gv(coords, spacing, axes, ndim as usize));
    let rho = Gv::field("rho", FieldRef::cons_den());
    let mom: Vec<Gv> = (0..ncomp)
        .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
        .collect();
    let src = geo
        .as_ref()
        .map(|g| gv_geometric_source(coords, axes, ndim as usize, ncomp, g, source, &mom, mag_from_bcell));

    let contribs = splice_fused_sources_to_contribs(
        coords, spacing, axes, ndim, ncomp, has_energy, &geo, Some((rho, &mom)), sources,
    );

    // the plain forward-Euler stage carries ONLY the flux divergence + the (well-balanced)
    // geometric source — NOT the user sources. `cons_new = a0*u_n + ac*fe`, identical to
    // `godunov_stage_gv`. the homologous-mesh dilution `-mesh_hdil * u` (with
    // `mesh_hdil = ndim * a_dot / a`, the comoving volume-growth rate) rides every
    // conserved law; the static binding mesh_hdil = 0 subtracts an exact zero.
    let h_dil = Gv::scalar("mesh_hdil");
    let fe = |u: Gv, div: Gv, geo_src: Option<Gv>| {
        let mut r = u - dt * div - dt * (h_dil * u);
        if let Some(s) = geo_src {
            r = r + dt * s;
        }
        r
    };
    let combine = |un: Gv, fe: Gv| a0 * un + ac * fe;
    // the USER sources ride as a SEPARATE additive term after the combine: `+ Σ ac*dt*contrib`,
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
    let rho_new = with_sources(
        combine(u_n_rho, fe(rho, gv_divergence("mass_flux", ndim, &geo), None)),
        &contribs.den,
    );
    let mut writes = vec![("rho".to_string(), FieldRef::cons_den().into(), rho_new.node())];
    for k in 0..ncomp {
        let u_n_mom = Gv::field(&format!("u_n_mom_{k}"), FieldRef::un_mom(k as u8));
        let div = gv_divergence(&format!("mom_flux_{k}"), ndim, &geo);
        let mom_new = with_sources(
            combine(u_n_mom, fe(mom[k], div, src.as_ref().map(|s| s[k]))),
            &contribs.mom[k],
        );
        writes.push((format!("mom_{k}"), FieldRef::cons_mom(k as u8).into(), mom_new.node()));
    }
    if has_energy {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        let u_n_nrg = Gv::field("u_n_nrg", FieldRef::un_nrg());
        let nrg_new = with_sources(
            combine(u_n_nrg, fe(nrg, gv_divergence("nrg_flux", ndim, &geo), None)),
            &contribs.nrg,
        );
        writes.push(("nrg".to_string(), FieldRef::cons_nrg().into(), nrg_new.node()));
    }
    (end_trace(), writes)
}

/// the standalone ADDITIVE source pass: `cons += dt * Σ S(prim, x; params)`, in place, per
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
// THE UNIFIED DAG-APPLICATION OPERATOR (docs/design/33 section 7).
//
// `apply_dag_core_gv` is the ONE kernel builder behind BOTH the interior source pass and (future,
// docs/design/33) the driven-boundary pass. it factors out the four decisions the old
// `source_apply_core_gv` baked: WHERE the DAG reads state (`StateEnv`), and HOW its result lands in
// the target field (`WriteMode`). the iteration domain + target-field binding are the dispatch's job
// (the same `dispatch_runtime_ir` + `resolve_path` serve cons.* and prim.*), so this builder is the
// whole difference between a source and a boundary prescription. doc 32's user `combine` projects
// onto `WriteMode`: add/relax -> Accumulate (differ only in the constructed expression), overwrite ->
// Assign.
// =============================================================================

/// the state vocabulary the DAG reads `rho`/`vel_k` from. `Stage` binds them from the SSP stage
/// snapshot `u_stage` (an interior source evaluates at its stage input — the S2 invariant); `Coord`
/// binds NOTHING from state (a pure coordinate prescription — a driven boundary, whose DAG OUTPUTS
/// the state). `x_k` (centroid) + scalar params bind regardless of this.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StateEnv {
    Stage,
    Coord,
}

/// how the DAG result lands in the target field. `Accumulate` is the RHS form `target = read(target)
/// + dt * Σ contrib` (in place; the `dt` scalar is the SSP stage weight) — sources. `Assign` is the
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
            // RHS in place: `cons_slot = cons_slot + Σ dt*contrib`, accumulated exactly as the fused
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
            // here means a mis-routed source; fail loud rather than silently drop it.
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
/// `(target_field, BuiltSource)` values — e.g. `expr_bridge::build_user_source`'s output from a
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

/// DRIVEN-BOUNDARY entry (docs/design/33): prescribe the primitive state from coordinate DAGs — the
/// `(Coord, Assign)` instance of [`apply_dag_core_gv`]. `sources` are `(slot, BuiltSource)` with slot
/// `"den"`/`"mom"`/`"nrg"` mapping to `prim.rho`/`prim.vel_k`/`prim.pre`; each DAG reads only
/// `x_k`/`t`/`p_i` and OUTPUTS the prescribed value. dispatched over a face's ghost band (task 2).
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

// =============================================================================
// in-kernel GEOMETRY — the substrate metric expressed in Gv. `Gv::coord` is the
// index->physical bridge; the coordinate-system formulas are a BUILD-TIME `match` on
// `Coords` (the kernel is generated per geometry, so the branch is resolved at trace
// time, not a runtime select). this is the foundation every curvilinear operator (CFL
// widths, godunov divergence, geometric sources, CT curl) traces through.
// =============================================================================

/// the face at `coord + offset` along axis `ax` as a physical position (offset 0 = lo face,
/// 1 = hi face). `x_lo_{ax}` + `dx_{ax}` are the grid scalars (dx = width for Uniform, the
/// log-slope for Log). the integer coord promotes to f64 against the scalars at lowering.
fn gv_axis_face_at(ax: usize, spacing: Spacing, offset: i64) -> Gv {
    let coord = Gv::coord(ax as u8);
    let start = Gv::scalar(&format!("x_lo_{ax}"));
    let param = Gv::scalar(&format!("dx_{ax}"));
    let i = if offset == 0 { coord } else { coord + Gv::from_f64(offset as f64) };
    match spacing {
        Spacing::Uniform => start + i * param,                      // start + i*dx
        Spacing::Log => start * Gv::from_f64(10.0).powf(i * param), // start * 10^(i*slope)
    }
}

/// the diagonal scale factor `h_dir(pos)` — the metric Lame coefficient. Cartesian: 1;
/// Spherical: (1, r, r*sin(theta)); Cylindrical: (1, r, 1). `pos` is coordinate-indexed
/// (pos[0]=r, pos[1]=theta). the `match` is build-time (Coords is the codegen geometry).
fn gv_scale_factor(coords: Coords, dir: usize, pos: &[Gv]) -> Gv {
    match (coords, dir) {
        (Coords::Cartesian, _) => Gv::ONE,
        (Coords::Spherical, 1) => pos[0],                  // r
        (Coords::Spherical, 2) => pos[0] * pos[1].sin(),   // r*sin(theta)
        (Coords::Spherical, _) => Gv::ONE,
        (Coords::Cylindrical, 1) => pos[0],                // r (phi direction)
        (Coords::Cylindrical, _) => Gv::ONE,
    }
}

/// per-cell PHYSICAL inverse widths `1 / (h_d * width_d)` per gridded axis — the metric-
/// correct CFL length scale (the wave crosses the physical extent `h_d * Δcoord_d`, not the
/// coordinate width), computed in-kernel from the cell index.
/// `axes[d]` is the coordinate gridded axis `d` maps to.
/// (the cartesian-UNIFORM CFL still uses the host's precomputed `inv_dx_d` scalar — this is
/// the curvilinear / non-uniform path.)
pub fn cell_inv_phys_widths_gv(coords: Coords, spacing: &[Spacing], axes: &[usize], ndim: usize) -> Vec<Gv> {
    let half = Gv::from_f64(0.5);
    let lo: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 0)).collect();
    let hi: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 1)).collect();
    let width: Vec<Gv> = (0..ndim).map(|d| hi[d] - lo[d]).collect();
    // coordinate-indexed cell center: scale_factor reads pos by coordinate, so place each
    // gridded axis's center at its coordinate slot (symmetry slots stay 0, never read).
    let mut center = vec![Gv::ZERO; 3];
    for d in 0..ndim {
        center[axes[d]] = (lo[d] + hi[d]) * half;
    }
    (0..ndim)
        .map(|d| {
            let h = gv_scale_factor(coords, axes[d], &center); // h of the coordinate this axis is
            Gv::ONE / (h * width[d]) // 1 / (h_d * width_d)
        })
        .collect()
}

/// per-cell finite-volume geometric factors in Gv:
/// inverse cell volume, per-axis lo/hi face areas, and
/// volume-weighted centroids, all from the cell index. the foundation the curvilinear
/// godunov (area-weighted divergence) + the geometric momentum source trace through.
#[derive(Clone)]
pub struct CellGeometryGv {
    pub inv_volume: Gv,
    pub area_lo: Vec<Gv>,
    pub area_hi: Vec<Gv>,
    pub centroid: Vec<Gv>,
}

/// `a^n` for a small literal power `n >= 1` as repeated multiply — exact (no Pow), so the
/// analytic radial integrals stay byte-form-identical across rebuilds.
fn gv_powi(a: Gv, n: u32) -> Gv {
    let mut acc = a;
    for _ in 1..n {
        acc = acc * a;
    }
    acc
}

/// per axis: `(lo face, hi face, width)` from the index map.
fn gv_faces(spacing: &[Spacing], ndim: usize) -> (Vec<Gv>, Vec<Gv>, Vec<Gv>) {
    let lo: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 0)).collect();
    let hi: Vec<Gv> = (0..ndim).map(|d| gv_axis_face_at(d, spacing[d], 1)).collect();
    let width: Vec<Gv> = (0..ndim).map(|d| hi[d] - lo[d]).collect();
    (lo, hi, width)
}

/// build the per-cell finite-volume geometric factors in Gv (cartesian / spherical /
/// cylindrical), axis-role driven. `axes[d]`
/// is the coordinate gridded axis `d` represents (identity for cartesian/spherical; the cyl
/// r-z swirl folds phi). analytic exact-integral factors + volume-weighted centroids.
pub fn cell_geometry_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
) -> CellGeometryGv {
    let (lo, hi, width) = gv_faces(spacing, ndim);
    match coords {
        Coords::Cartesian => cartesian_geometry_gv(&lo, &hi, &width, ndim),
        Coords::Spherical => spherical_geometry_gv(&lo, &hi, &width, ndim),
        Coords::Cylindrical => cylindrical_geometry_gv(&lo, &hi, &width, axes, ndim),
    }
}

// cartesian: V = prod(width); A_dir = prod_{j!=dir}(width); centroid = arithmetic mid.
fn cartesian_geometry_gv(lo: &[Gv], hi: &[Gv], width: &[Gv], ndim: usize) -> CellGeometryGv {
    let mut vol = width[0];
    for d in 1..ndim {
        vol = vol * width[d];
    }
    let inv_volume = Gv::ONE / vol;
    let half = Gv::from_f64(0.5);
    let mut area_lo = Vec::with_capacity(ndim);
    let mut area_hi = Vec::with_capacity(ndim);
    let mut centroid = Vec::with_capacity(ndim);
    for dir in 0..ndim {
        let mut a: Option<Gv> = None;
        for (j, &w) in width.iter().enumerate() {
            if j == dir {
                continue;
            }
            a = Some(match a {
                None => w,
                Some(acc) => acc * w,
            });
        }
        let area = a.unwrap_or(Gv::ONE); // 1D: unit perpendicular face
        area_lo.push(area);
        area_hi.push(area);
        centroid.push((lo[dir] + hi[dir]) * half); // flat cell centroid = arithmetic mid
    }
    CellGeometryGv { inv_volume, area_lo, area_hi, centroid }
}

// spherical (r, theta, phi): analytic exact-integral factors, volume-weighted centroids
// (radial centroid past the C++ coordinate-center TODO).
fn spherical_geometry_gv(lo: &[Gv], hi: &[Gv], width: &[Gv], ndim: usize) -> CellGeometryGv {
    let pi = std::f64::consts::PI;
    let (rl, rh) = (lo[0], hi[0]);
    let ir1 = (gv_powi(rh, 3) - gv_powi(rl, 3)) / Gv::from_f64(3.0); // int r^2 dr
    let ir2 = (gv_powi(rh, 2) - gv_powi(rl, 2)) / Gv::from_f64(2.0); // int r dr
    let ir_cnum = (gv_powi(rh, 4) - gv_powi(rl, 4)) / Gv::from_f64(4.0); // int r^3 dr
    let centroid_r = ir_cnum / ir1; // (3/4)(rh^4-rl^4)/(rh^3-rl^3)

    let (i_theta, sin_tl, sin_th, centroid_t) = if ndim >= 2 {
        let (tl, th) = (lo[1], hi[1]);
        let (ctl, cth) = (tl.cos(), th.cos());
        let it = ctl - cth; // cos(tl) - cos(th)
        // volume-weighted theta centroid: [(sin th - th cos th)]_{tl}^{th} / Itheta.
        let num = (th.sin() - th * cth) - (tl.sin() - tl * ctl);
        (it, tl.sin(), th.sin(), num / it)
    } else {
        let z = Gv::ZERO;
        (Gv::from_f64(2.0), z, z, Gv::from_f64(pi / 2.0)) // cos(0)-cos(pi)=2; centroid at pi/2
    };
    let i_phi = if ndim >= 3 { width[2] } else { Gv::from_f64(2.0 * pi) };

    let inv_volume = Gv::ONE / (ir1 * i_theta * i_phi);
    let omega = i_theta * i_phi; // angular solid-angle measure for the r-face

    let mut area_lo = vec![Gv::ZERO; ndim];
    let mut area_hi = vec![Gv::ZERO; ndim];
    let mut centroid = vec![Gv::ZERO; ndim];
    area_lo[0] = gv_powi(rl, 2) * omega; // r-face A = r_face^2 * Omega
    area_hi[0] = gv_powi(rh, 2) * omega;
    centroid[0] = centroid_r;
    if ndim >= 2 {
        area_lo[1] = ir2 * sin_tl * i_phi; // theta-face A = Ir2 * sin(theta_face) * Iphi
        area_hi[1] = ir2 * sin_th * i_phi;
        centroid[1] = centroid_t;
    }
    if ndim >= 3 {
        let aphi = ir2 * width[1]; // phi-face A = Ir2 * dtheta
        area_lo[2] = aphi;
        area_hi[2] = aphi;
        centroid[2] = (lo[2] + hi[2]) * Gv::from_f64(0.5); // arithmetic mid (uniform in phi)
    }
    CellGeometryGv { inv_volume, area_lo, area_hi, centroid }
}

// cylindrical (coords 0=r, 1=phi, 2=z): h=(1,r,1), sqrt(g)=r. axis-role driven — one builder
// serves (r,phi)/(r,z)/(r,phi,z); an ungridded coordinate is a symmetry axis (its full-extent
// measure cancels in the divergence).
fn cylindrical_geometry_gv(
    lo: &[Gv],
    hi: &[Gv],
    width: &[Gv],
    axes: &[usize],
    ndim: usize,
) -> CellGeometryGv {
    let pi = std::f64::consts::PI;
    let grid_of = |coord: usize| -> Option<usize> { axes.iter().position(|&c| c == coord) };
    let r_ax = grid_of(0).expect("cylindrical: the radial coordinate (0) must be gridded");
    let phi_ax = grid_of(1);
    let z_ax = grid_of(2);

    let (rl, rh) = (lo[r_ax], hi[r_ax]);
    let ir2 = (gv_powi(rh, 2) - gv_powi(rl, 2)) / Gv::from_f64(2.0); // int r dr
    let ir_cnum = (gv_powi(rh, 3) - gv_powi(rl, 3)) / Gv::from_f64(3.0);
    let centroid_r = ir_cnum / ir2; // (2/3)(rh^3-rl^3)/(rh^2-rl^2)
    let dr = rh - rl;

    // transverse measures: gridded -> the grid width; symmetry -> the full extent constant.
    let i_phi = match phi_ax {
        Some(a) => width[a],
        None => Gv::from_f64(2.0 * pi),
    };
    let i_z = match z_ax {
        Some(a) => width[a],
        None => Gv::ONE,
    };
    let inv_volume = Gv::ONE / (ir2 * i_phi * i_z);

    let half = Gv::from_f64(0.5);
    let mut area_lo = vec![Gv::ZERO; ndim];
    let mut area_hi = vec![Gv::ZERO; ndim];
    let mut centroid = vec![Gv::ZERO; ndim];
    area_lo[r_ax] = rl * i_phi * i_z; // r-face A = r_face * i_phi * i_z
    area_hi[r_ax] = rh * i_phi * i_z;
    centroid[r_ax] = centroid_r;
    if let Some(a) = phi_ax {
        let aphi = dr * i_z; // phi-face A = dr * i_z
        area_lo[a] = aphi;
        area_hi[a] = aphi;
        centroid[a] = (lo[a] + hi[a]) * half;
    }
    if let Some(a) = z_ax {
        let az = ir2 * i_phi; // z-face A = Ir2 * i_phi
        area_lo[a] = az;
        area_hi[a] = az;
        centroid[a] = (lo[a] + hi[a]) * half;
    }
    CellGeometryGv { inv_volume, area_lo, area_hi, centroid }
}

/// the geometry probe: write `inv_volume` + the dir-0 lo/hi face
/// areas + the dir-0 volume-weighted centroid, so a host test bit-diffs them against the
/// analytic formulas (incl. log spacing). identity axes (the probe is always natural-order).
pub fn geometry_probe_gv(
    coords: Coords,
    spacing: &[Spacing],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let axes: Vec<usize> = (0..ndim).collect();
    let g = cell_geometry_gv(coords, spacing, &axes, ndim);
    let writes = vec![
        ("inv_volume".to_string(), "inv_volume".into(), g.inv_volume.node()),
        ("area_lo_0".to_string(), "area_lo_0".into(), g.area_lo[0].node()),
        ("area_hi_0".to_string(), "area_hi_0".into(), g.area_hi[0].node()),
        ("centroid_0".to_string(), "centroid_0".into(), g.centroid[0].node()),
    ];
    (end_trace(), writes)
}

/// SPIKE probe: trace the carrier-generic `symbi_hydro::UniformAccel` source at S=Gv.
/// constructs the source with `g_ext_k` runtime scalars (the same names the splice path
/// declares), reads rho/vel as cell fields, and writes `s_mom_k` + `s_nrg`. a host test
/// renders + evaluates this and asserts the analytical `rho*g_ext` / `rho*(v.g_ext)` — the
/// SAME result `uniform_acceleration_*_source` produces via its hand-built graph, proving the
/// carrier-generic form is a drop-in for the splice path (and is f64==Gv by construction).
pub fn uniform_accel_probe_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("rho", FieldRef::cons_den());
    let vel: [Gv; D] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let g_ext: [Gv; D] = std::array::from_fn(|k| Gv::scalar(&format!("g_ext_{k}")));
    let src = symbi_hydro::UniformAccel::<Gv, D> { g_ext };
    let mom = src.momentum(rho);
    let nrg = src.energy(rho, &vel);
    let mut writes: Vec<(String, FieldBind, NodeId)> = (0..D)
        .map(|k| (format!("s_mom_{k}"), format!("s_mom_{k}").into(), mom[k].node()))
        .collect();
    writes.push(("s_nrg".to_string(), "s_nrg".into(), nrg.node()));
    (end_trace(), writes)
}

/// splice an externally-lowered user-expression `BuiltSource` (a parsed script, bridged into the
/// `symbi-ir` Graph via `symbi_hydro::expr_bridge`, optionally wrapped in a conservation law by
/// `source_spec::user_force_*` / `user_cooling_source`) into a Gv trace — binding each declared
/// param to a runtime Gv scalar of the same name — and write its outputs `s_k`. a user expression
/// FUSES into a kernel graph and RENDERS (CPU + CUDA) through the exact same `splice_built_source_into`
/// path a built-in source uses (the elegance over the C++ `dag/`: the user script becomes compiled
/// kernel code, not a per-cell register-VM walk). carrier-equivalence + the work-energy coupling are
/// gated by `source_term_carrier.rs`.
pub fn splice_user_source_gv(
    built: &symbi_hydro::source_spec::BuiltSource,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // bind every declared param (x_k, t, p_i, ...) to a runtime Gv scalar of the same name;
    // in production the position `x_k` binds to the in-kernel centroid instead.
    let mut name_to_node = std::collections::HashMap::new();
    for p in &built.params {
        name_to_node.insert(p.clone(), Gv::scalar(p).node());
    }
    let outs = with_trace(|t| {
        symbi_hydro::source_spec::splice_built_source_into(built, t.graph(), &name_to_node)
    });
    let writes = outs
        .iter()
        .enumerate()
        .map(|(k, &n)| (format!("s_{k}"), format!("s_{k}").into(), n))
        .collect();
    (end_trace(), writes)
}

/// SPIKE probe: trace the carrier-generic `symbi_hydro::PointMassGravity` source at S=Gv.
/// reads rho/vel as cell fields and the position `x_k`, mass position `xm_k`, and `gm` as
/// runtime scalars (the same names the splice path declares); writes `s_mom_k` + `s_nrg`.
/// a host test renders + evaluates it and asserts `-rho*GM*(x-xm)/|x-xm|^3` — the SAME form
/// `point_mass_{momentum,energy}_source` hand-builds, proving the carrier-generic form is a
/// drop-in (and f64==Gv by construction). the shared `1/|x-xm|^3` is emitted once (hash-cons).
pub fn point_mass_gravity_probe_gv<const D: usize>() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let rho = Gv::field("rho", FieldRef::cons_den());
    let vel: [Gv; D] =
        std::array::from_fn(|k| Gv::field(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8)));
    let x: [Gv; D] = std::array::from_fn(|k| Gv::scalar(&format!("x_{k}")));
    let xm: [Gv; D] = std::array::from_fn(|k| Gv::scalar(&format!("xm_{k}")));
    let gm = Gv::scalar("gm");
    let eps = Gv::scalar("eps");
    let src = symbi_hydro::PointMassGravity::<Gv, D> { gm, xm, eps };
    let mom = src.momentum(rho, &x);
    let nrg = src.energy(rho, &vel, &x);
    let mut writes: Vec<(String, FieldBind, NodeId)> = (0..D)
        .map(|k| (format!("s_mom_{k}"), format!("s_mom_{k}").into(), mom[k].node()))
        .collect();
    writes.push(("s_nrg".to_string(), "s_nrg".into(), nrg.node()));
    (end_trace(), writes)
}

/// the gv inertial-source probe:
/// read the conserved momentum + primitive velocity, compute the centrifugal/coriolis source
/// `S^i = -Gamma^i_jk mom^j v^k` from the in-kernel volume-weighted centroid, write `s_d`. a
/// host test bit-diffs it against the analytic `mom_t v_t / r` forms. identity axes (natural).
pub fn inertial_momentum_probe_gv(
    coords: Coords,
    spacing: &[Spacing],
    ndim: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let axes: Vec<usize> = (0..ndim).collect();
    let mom: Vec<Gv> = (0..ndim)
        .map(|d| Gv::field(&format!("cons_mom_{d}"), FieldRef::cons_mom(d as u8)))
        .collect();
    let vel: Vec<Gv> = (0..ndim)
        .map(|d| Gv::field(&format!("prim_v{d}"), FieldRef::PrimVel(d as u8)))
        .collect();
    let geo = cell_geometry_gv(coords, spacing, &axes, ndim);
    let s = inertial_momentum_sources_gv(ndim, coords, &mom, &vel, &geo.centroid);
    let writes = (0..ndim).map(|d| (format!("s_{d}"), format!("s_{d}").into(), s[d].node())).collect();
    (end_trace(), writes)
}

/// the gv FULL geometric-momentum-source probe — the carrier mirror of the ctx
/// `geometric_momentum_sources` path (+ the rmhd adapter): build the cell geometry, form the
/// per-component source `S^i = -Gamma^i_jk T^jk` via [`gv_geometric_source`], write `s_k`. a host
/// test bit-diffs it against the analytic pressure + inertial (+ rmhd magnetic) forms. `axes`
/// carries the role map (the cyl r-z swirl `[0, 2]` with `ncomp > ndim`).
pub fn geometric_momentum_source_probe_gv(
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    ndim: usize,
    ncomp: usize,
    source: GeoSource,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let geo = cell_geometry_gv(coords, spacing, axes, ndim);
    // hydro shares the conserved momentum (the gas inertial reads cons.mom); rmhd computes its
    // gas momentum density from prim (cons.mom carries B-momentum), so it reads no cons.mom.
    let cons_mom: Vec<Gv> = match source {
        // hydro + newtonian MHD: cons.mom IS the gas momentum density (rho v), read directly.
        // read ALL `ncomp` (DOF) components, not just `ndim`: a 2.5D spherical (r,theta) MHD grid
        // has DOF=3 and the geometric S_theta/S_phi need the out-of-plane phi momentum (mom[2]).
        // hydro has ncomp==ndim so this is unchanged there.
        GeoSource::Hydro { .. } | GeoSource::NewtonianMhd | GeoSource::IsothermalMhd => (0..ncomp)
            .map(|k| Gv::field(&format!("mom_{k}"), FieldRef::cons_mom(k as u8)))
            .collect(),
        GeoSource::Rmhd => Vec::new(),
    };
    let s = gv_geometric_source(coords, axes, ndim, ncomp, &geo, source, &cons_mom, false);
    let writes = (0..ncomp).map(|k| (format!("s_{k}"), format!("s_{k}").into(), s[k].node())).collect();
    (end_trace(), writes)
}

// =============================================================================
// the lattice-map GHOST FILL in Gv (docs/design/11) — the boundary pullback: read the
// primitives at the per-axis integer SOURCE coord (periodic shift / reflect pivot / outflow
// clamp on a runtime `map_type`), write at the cell (in place), with the grade-1 Jacobian
// `vel_sign` flip on the velocity (and B for RMHD). the source coord is PURE INTEGER (the
// `_coord_N` + the I32 `map_type`/`arg` params), so the read is an ordinary multi-axis
// `load_at` — no gather, no float->int cast. the gv multi-axis stencil cap (the integer
// `field_at`) that ghost + CT share, mirroring `pullback::{source_axis, iso_ghost_fill}`.
// =============================================================================

/// the per-axis lattice-map source coord, a pure-integer select on `map_type` (mirror of
/// `pullback::source_axis`): `0` skip -> `c`; `1` periodic -> `c+arg`; `2` reflect -> `arg-c`;
/// `3` outflow -> `arg`. registers `_coord_N` + the I32 `map_type_{ax}`/`arg_{ax}` params.
fn gv_lattice_source(ndim: usize) -> Vec<NodeId> {
    use ElementWiseOp::*;
    with_trace(|t| {
        // register coords, then ALL map_type, then ALL arg (grouped — matching the positional
        // rmhd ghost-fill dispatch ints [map_type_0..D, arg_0..D]).
        let coords: Vec<NodeId> = (0..ndim).map(|ax| t.coord(ax as u8)).collect();
        let map_type: Vec<NodeId> =
            (0..ndim).map(|ax| t.scalar_int(&format!("map_type_{ax}"))).collect();
        let arg: Vec<NodeId> =
            (0..ndim).map(|ax| t.scalar_int(&format!("arg_{ax}"))).collect();
        (0..ndim)
            .map(|ax| {
                let (c, mt, ag) = (coords[ax], map_type[ax], arg[ax]);
                let g = t.graph();
                let zero = g.add_const(ConstValue::I32(0), None);
                let one = g.add_const(ConstValue::I32(1), None);
                let two = g.add_const(ConstValue::I32(2), None);
                let is_skip = g.element_wise(Eq, vec![mt, zero], None);
                let is_periodic = g.element_wise(Eq, vec![mt, one], None);
                let is_reflect = g.element_wise(Eq, vec![mt, two], None);
                let periodic = g.element_wise(Add, vec![c, ag], None); // c + arg
                let reflect = g.element_wise(Sub, vec![ag, c], None); // arg - c
                let pick_reflect = g.select(is_reflect, reflect, ag, None); // else outflow
                let pick_periodic = g.select(is_periodic, periodic, pick_reflect, None);
                g.select(is_skip, c, pick_periodic, None)
            })
            .collect()
    })
}

/// load field `key` at the integer source coord vector `src` (deduped manifest registration) —
/// the gv multi-axis `load_at`, the pullback read. returns the loaded value as a `Gv`.
/// pub(crate): the amr transfer builders (gv_refinement.rs) share this pullback read.
pub(crate) fn gv_load_at(key: &str, runtime: impl Into<FieldBind>, src: &[NodeId]) -> Gv {
    let runtime = runtime.into();
    Gv::of(with_trace(|t| {
        t.register_field(key, runtime);
        t.graph().load_at(Symbol::intern(key), src.to_vec(), None)
    }))
}

/// the isothermal lattice-map ghost fill — pull back rho/vel/pre at the per-axis source coord,
/// write IN PLACE; the velocity component whose coordinate is a GRID axis picks up that axis's
/// wall-normal `vel_sign` (an ungridded swirl coordinate has no wall map -> unflipped). rho/pre
/// are grade-0 copies. `ncomp` velocity components, `ndim` gridded axes; `axes[d]` = the coord
/// of grid axis d. the EOS-generic 3-field pullback the iso/newton/srhd ghost fill share.
pub fn iso_ghost_fill_gv(
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim).map(|ax| Gv::scalar(&format!("vel_sign_{ax}"))).collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src);
        // grade-1 wall flip on the grid axis whose coordinate IS k; ungridded -> unflipped.
        let v = match axes.iter().position(|&c| c == k) {
            Some(ax) => v * vel_sign[ax],
            None => v,
        };
        writes.push((format!("prim_v{k}"), FieldRef::PrimVel(k as u8).into(), v.node()));
    }
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    (end_trace(), writes)
}

/// the SINGLE-SCALAR lattice-map ghost fill: pull back one field "f" at the per-axis
/// integer source coord, times the runtime grade `sign` (+1 for a scalar copy or a
/// tangential staggered component; -1 for a wall-normal component under a reflect
/// map). the staggered `bface` transverse-halo fill dispatches this per component —
/// the field resolves the region's absolute coords against its OWN staggered lo, so
/// the same kernel serves any cell- or face-anchored scalar.
pub fn scalar_ghost_fill_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let sign = Gv::scalar("sign");
    let v = gv_load_at("f", "f", &src) * sign;
    let writes = vec![("f".to_string(), "f".into(), v.node())];
    (end_trace(), writes)
}

// the per-vector-component wall-map sign: the in-plane components (k < ndim) pick up the
// boundary axis's reflect sign (B/vel are grade-1 vectors under the wall map); the out-of-
// plane components (k >= ndim, e.g. Bz/vz in 1.5D/2.5D) are tangential to every grid-axis
// wall, so they copy unchanged (sign = +1). this is why ghost fill loops 0..ncomp (DOF),
// NOT 0..ndim — else the out-of-plane ghosts stay zero and drain the boundary.
fn gv_ghost_sign(k: usize, ndim: usize, vel_sign: &[Gv]) -> Gv {
    if k < ndim { vel_sign[k] } else { Gv::ONE }
}

/// the RMHD lattice-map ghost fill — `iso_ghost_fill_gv` plus the cell-centered B: pull back
/// rho/vel/pre + `mhd.bcell[k]`, the velocity AND B (DOF-vectors) picking up the per-axis
/// `vel_sign` for in-plane components and copying the out-of-plane ones. `ndim` = grid axes
/// (the lattice source + reflect signs), `ncomp` = vector components (DOF).
pub fn rmhd_ghost_fill_gv(ndim: usize, ncomp: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim).map(|k| Gv::scalar(&format!("vel_sign_{k}"))).collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("prim_v{k}"), FieldRef::PrimVel(k as u8).into(), v.node()));
    }
    let pre = gv_load_at("prim_pre", "prim.pre", &src);
    writes.push(("prim_pre".to_string(), FieldRef::PrimPre.into(), pre.node()));
    for k in 0..ncomp {
        let b = gv_load_at(&format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("bcell_{k}"), format!("mhd.bcell[{k}]").into(), b.node()));
    }
    (end_trace(), writes)
}

/// the ISOTHERMAL lattice-map ghost fill — `rmhd_ghost_fill_gv` minus the `pre` field
/// (isothermal MHD has no pressure to fill). rho + vel + bcell only.
pub fn imhd_ghost_fill_gv(ndim: usize, ncomp: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let src = gv_lattice_source(ndim);
    let vel_sign: Vec<Gv> = (0..ndim).map(|k| Gv::scalar(&format!("vel_sign_{k}"))).collect();
    let rho = gv_load_at("prim_rho", "prim.rho", &src);
    let mut writes = vec![("prim_rho".to_string(), FieldRef::PrimRho.into(), rho.node())];
    for k in 0..ncomp {
        let v = gv_load_at(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("prim_v{k}"), FieldRef::PrimVel(k as u8).into(), v.node()));
    }
    for k in 0..ncomp {
        let b = gv_load_at(&format!("bcell_{k}"), &format!("mhd.bcell[{k}]"), &src) * gv_ghost_sign(k, ndim, &vel_sign);
        writes.push((format!("bcell_{k}"), format!("mhd.bcell[{k}]").into(), b.node()));
    }
    (end_trace(), writes)
}

// =============================================================================
// the RMHD CONSTRAINED-TRANSPORT stack in Gv — the staggered curl / edge-EMF / face->cell B /
// cell-B flux-predictor / EMF save+average. all built on the gv multi-axis OFFSET stencil
// `gv_field_at` (the staggered gather: read field at `coord + offsets`). div(B)=0 to machine
// precision is preserved BY THE STENCIL (the discrete curl + divergence telescope the shared
// h-weighted edge EMFs to exactly 0); gated by the rmhd_ct_curl*_divb tests. the input/write
// order matches the hand-built staggered runtime dispatch the RMHD regime binds.
// =============================================================================

/// register a field in the manifest WITHOUT emitting a node — to PIN the buffer order (the
/// staggered runtime dispatch is positional) ahead of the stencil reads that follow.
fn gv_register_field(key: &str, runtime: &str) {
    with_trace(|t| t.register_field(key, runtime));
}

/// load field `key` at `coord + offsets` (per-axis integer offset; all-zero = the cell coord) —
/// the gv multi-axis OFFSET stencil (the CT staggered gather). registers the field (deduped),
/// builds the integer coord arithmetic + `load_at`. like `field_shifted` but a full offset vector.
fn gv_field_at(key: &str, runtime: &str, ndim: usize, offsets: &[i32]) -> Gv {
    Gv::of(with_trace(|t| {
        t.register_field(key, runtime);
        let comps: Vec<NodeId> = (0..ndim)
            .map(|ax| {
                let c = t.coord(ax as u8);
                if offsets[ax] == 0 {
                    c
                } else {
                    let off = t.graph().add_const(ConstValue::I32(offsets[ax]), None);
                    t.graph().element_wise(ElementWiseOp::Add, vec![c, off], None)
                }
            })
            .collect();
        t.graph().load_at(Symbol::intern(key), comps, None)
    }))
}

/// the Gardiner & Stone CT-contact edge EMF (the SOFT-SIGN blend), carrier-generic at S=Gv.
/// a pointwise function of the 4 face EMFs, 4 cell-corner
/// EMFs, and 4 density fluxes: `s = f/(|f|+eps)`; `0.5*((a+b) + s*(a-b))`, transitions
/// continuously through f=0 (= the C++ hard 3-way sign in the |f|>>eps limit). div(B) unaffected.
fn ct_contact_emf_gv(face_e: [Gv; 4], cell_e: [Gv; 4], dflux: [Gv; 4]) -> Gv {
    let [en, es, ee, ew] = face_e;
    let [ene, enw, ese, esw] = cell_e;
    let [fnf, fs, fe, fw] = dflux;
    let two = Gv::from_f64(2.0);
    let eps = Gv::from_f64(1.0e-12);
    let eavg = Gv::from_f64(0.25) * (es + en + ew + ee);
    let soft = |f: Gv, a: Gv, b: Gv| {
        let s = f / (f.abs() + eps);
        Gv::from_f64(0.5) * ((a + b) + s * (a - b))
    };
    let de_jl = soft(fw, two * (es - esw), two * (en - enw)); // west
    let de_jr = soft(fe, two * (ese - es), two * (ene - en)); // east
    let de_kl = soft(fs, two * (ew - esw), two * (ee - ese)); // south
    let de_kr = soft(fnf, two * (enw - ew), two * (ene - ee)); // north
    eavg + Gv::from_f64(0.125) * (de_jl - de_jr + de_kl - de_kr)
}

/// the orthogonal-curl scale-factor weights for the
/// curvilinear induction curl (h_p edge weights + the 1/(h_p1c h_p2c) face-center prefactor + the
/// transverse inverse widths). all Gv, from the cell index via gv_axis_face_at / gv_scale_factor.
struct CtCurlMetricGv {
    h1_here: Gv,
    h1_p2: Gv,
    h2_here: Gv,
    h2_p1: Gv,
    inv_pref: Gv,
    inv_dx_p1: Gv,
    inv_dx_p2: Gv,
}

fn ct_curl_metric_gv(coords: Coords, spacing: &[Spacing], dir: usize) -> CtCurlMetricGv {
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let pos_at = |off: [i64; 3]| -> Vec<Gv> {
        (0..3).map(|ax| gv_axis_face_at(ax, spacing[ax], off[ax])).collect()
    };
    let pos_here = pos_at([0, 0, 0]);
    let mut op2 = [0, 0, 0];
    op2[p2] = 1;
    let pos_p2 = pos_at(op2);
    let mut op1 = [0, 0, 0];
    op1[p1] = 1;
    let pos_p1 = pos_at(op1);
    let h1_here = gv_scale_factor(coords, p1, &pos_here);
    let h1_p2 = gv_scale_factor(coords, p1, &pos_p2);
    let h2_here = gv_scale_factor(coords, p2, &pos_here);
    let h2_p1 = gv_scale_factor(coords, p2, &pos_p1);

    let half = Gv::from_f64(0.5);
    let center: Vec<Gv> = (0..3)
        .map(|ax| {
            if ax == dir {
                gv_axis_face_at(ax, spacing[ax], 0)
            } else {
                (gv_axis_face_at(ax, spacing[ax], 0) + gv_axis_face_at(ax, spacing[ax], 1)) * half
            }
        })
        .collect();
    let inv_pref = Gv::ONE / (gv_scale_factor(coords, p1, &center) * gv_scale_factor(coords, p2, &center));
    let inv_dx_p1 = Gv::ONE / (gv_axis_face_at(p1, spacing[p1], 1) - gv_axis_face_at(p1, spacing[p1], 0));
    let inv_dx_p2 = Gv::ONE / (gv_axis_face_at(p2, spacing[p2], 1) - gv_axis_face_at(p2, spacing[p2], 0));
    CtCurlMetricGv { h1_here, h1_p2, h2_here, h2_p1, inv_pref, inv_dx_p1, inv_dx_p2 }
}

/// the 2.5D in-plane CT curl B-update along ONE face axis `dir` from the single
/// out-of-plane corner EMF Ez (cartesian), in-place on `b` (bface[dir]). PER-DIR
/// (mirroring the 3D `rmhd_ct_curl_3d_dir`) because bx lives on x-faces and by on
/// y-faces — distinct staggered domains, each updated over its own face domain so
/// the high boundary face is covered. dir=0: dBx/dt = -dEz/dy -> b -= dt*idy*(Ez[j+1]-Ez);
/// dir=1: dBy/dt = +dEz/dx -> b += dt*idx*(Ez[i+1]-Ez). div(B)=0 preserved.
/// (the out-of-plane Bz is NOT CT-evolved — it rides the induction-flux divergence.)
pub fn rmhd_ct_curl_2d_dir_gv(dir: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez");
    let dt = Gv::scalar("dt");
    let b_new = if dir == 0 {
        let idy = Gv::scalar("idy");
        let ez_jp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * idy * (ez_jp - ez)
    } else {
        let idx = Gv::scalar("idx");
        let ez_ip = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * idx * (ez_ip - ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}

/// the 2.5D cylindrical r-z (axisymmetric) CT curl from the single out-of-plane edge EMF
/// E_phi (efield[0]), in-place on `b` (bface[dir]). DERIVED from the 3D cyl curl restricted
/// to E_phi with d/dphi = 0 (verified to reproduce the 3D-cyl ct_curl_metric formula):
///   dir=0 (B_r, r-face):  dB_r/dt = +d_z E_phi            (z = grid axis 1; flat, no metric)
///   dir=1 (B_z, z-face):  dB_z/dt = -(1/r) d_r(r E_phi)   (r = grid axis 0; cylindrical metric)
/// r is computed per-cell from gv_axis_face_at(0, ..) (the geom scalars x_lo_0/dx_0). E_phi is
/// the corner field at offsets [0,0]/[+grid]. div(B)=0 preserved by the discrete d∘d.
pub fn rmhd_ct_curl_cyl_rz_gv(dir: usize, spacing: &[Spacing]) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane corner EMF E_phi
    let dt = Gv::scalar("dt");
    // POSITIONAL scalar ABI: the runtime curl dispatch pushes `[dt] ++ push_curvilinear_geom`
    // = [dt, x_lo_0, dx_0, x_lo_1, dx_1] (all grid axes, every dir). scalar_params is fixed at
    // registration order with NO liveness pruning, so BOTH dir branches must register the full
    // geom set in that order — else dir=0 (which only touches axis 1) would bind x_lo_1 to the
    // runtime's x_lo_0 slot. this prelude pins the canonical order; the body's reads dedupe in.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let b_new = if dir == 0 {
        // dB_r/dt = +d_z E_phi : finite difference along grid axis 1 (z). no metric (h_z = 1).
        let inv_dz = Gv::ONE / (gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0));
        let ez_zp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b + dt * inv_dz * (ez_zp - ez)
    } else {
        // dB_z/dt = -(1/r_c) d_r(r E_phi) : the cylindrical metric on the radial derivative.
        // r at the cell's two r-faces (= the corner radii bounding this z-face), cell-center r_c.
        let inv_dr = Gv::ONE / (gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0));
        let r_lo = gv_axis_face_at(0, spacing[0], 0);
        let r_hi = gv_axis_face_at(0, spacing[0], 1);
        let r_c = (r_lo + r_hi) * Gv::from_f64(0.5);
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b - dt * (Gv::ONE / r_c) * inv_dr * (r_hi * ez_rp - r_lo * ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}

/// the 2.5D cylindrical r-phi DISK CT curl from the single out-of-plane edge EMF E_z
/// (efield[0]), in-place on `b` (bface[dir]). DERIVED from the cyl curl restricted to E_z with
/// d/dz = 0 (verified to preserve the staggered cyl div(B) = (1/r)d_r(r B_r) + (1/r)d_phi B_phi):
///   dir=0 (B_r, r-face):   dB_r/dt   = -(1/r) d_phi E_z   (phi = grid axis 1; 1/r metric, r = the r-face radius)
///   dir=1 (B_phi, phi-face): dB_phi/dt = +d_r E_z         (r = grid axis 0; flat, NO metric — mirror of r-z)
/// r is the r-FACE radius (where B_r lives) via gv_axis_face_at(0, .., 0). E_z is the corner field
/// at offsets [0,0]/[+grid]. div(B)=0 preserved by the discrete d∘d (mixed partials cancel).
pub fn rmhd_ct_curl_cyl_rphi_gv(dir: usize, spacing: &[Spacing]) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane corner EMF E_z
    let dt = Gv::scalar("dt");
    // POSITIONAL scalar ABI: the runtime curl dispatch pushes [dt, x_lo_0, dx_0, x_lo_1, dx_1]
    // every dir (see rmhd_ct_curl_cyl_rz_gv). pin the full geom set in canonical order up front.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let b_new = if dir == 0 {
        // dB_r/dt = -(1/r) d_phi E_z : the 1/r metric on the phi-derivative (grid axis 1). r is
        // the r-FACE radius (B_r lives on the r-face = the cell's low r-face, offset 0).
        let r_face = gv_axis_face_at(0, spacing[0], 0);
        let inv_dphi = Gv::ONE / (gv_axis_face_at(1, spacing[1], 1) - gv_axis_face_at(1, spacing[1], 0));
        let ez_phip = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * (Gv::ONE / r_face) * inv_dphi * (ez_phip - ez)
    } else {
        // dB_phi/dt = +d_r E_z : finite difference along grid axis 0 (r). NO metric (the phi-comp
        // of the cyl curl is metric-free; the discrete d∘d still cancels — proven).
        let inv_dr = Gv::ONE / (gv_axis_face_at(0, spacing[0], 1) - gv_axis_face_at(0, spacing[0], 0));
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * inv_dr * (ez_rp - ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}

/// the 2.5D SPHERICAL (r-theta plane, out-of-plane phi) CT curl from the single corner EMF
/// E_phi (efield[0]), in-place on `b` (bface[dir]). Faraday dB/dt = -curl E with E = E_phi phi-hat
/// (axisymmetric) gives the spherical-metric in-plane update:
///   dir=0 (B_r,   r-face):     dB_r/dt   = -(1/(r_f sin th_c)) d_th(sin th * E_phi)   (th = grid axis 1)
///   dir=1 (B_th, theta-face):  dB_th/dt  = +(1/r_c) d_r(r * E_phi)                     (r  = grid axis 0)
/// r_f is the r-FACE radius (where B_r lives); r_c / th_c are the staggered cell centers. mirrors
/// `rmhd_ct_curl_cyl_rz_gv` with the added sin(theta) area weight on the B_r update (and the
/// opposite B_theta sign vs the cylinder's B_z). VALIDATION NOTE: derived from the continuous curl;
/// the staggered div(B)=0 preservation for a POLOIDAL field still needs a dedicated test (the
/// toroidal-injection case exercises this on a zero in-plane field, so it is trivially div-free).
pub fn rmhd_ct_curl_2d_sph_gv(dir: usize, spacing: &[Spacing]) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let b = Gv::field("b", "b");
    let ez = Gv::field("ez", "ez"); // the out-of-plane corner EMF E_phi
    let dt = Gv::scalar("dt");
    // POSITIONAL scalar ABI: the runtime curl dispatch pushes [dt, x_lo_0, dx_0, x_lo_1, dx_1]
    // every dir (see rmhd_ct_curl_cyl_rz_gv). pin the full geom set in canonical order up front.
    for ax in 0..2 {
        let _ = gv_axis_face_at(ax, spacing[ax], 0);
        let _ = gv_axis_face_at(ax, spacing[ax], 1);
    }
    let half = Gv::from_f64(0.5);
    let b_new = if dir == 0 {
        // dB_r/dt = -(1/(r_f sin th_c)) d_th(sin th E_phi). r_f = the low r-face (B_r lives there);
        // th_lo/th_hi are the corner thetas bounding this r-face, th_c the cell-center theta.
        let r_f = gv_axis_face_at(0, spacing[0], 0);
        let th_lo = gv_axis_face_at(1, spacing[1], 0);
        let th_hi = gv_axis_face_at(1, spacing[1], 1);
        let th_c = (th_lo + th_hi) * half;
        let inv_dth = Gv::ONE / (th_hi - th_lo);
        let ez_thp = gv_field_at("ez", "ez", 2, &[0, 1]);
        b - dt * (Gv::ONE / (r_f * th_c.sin())) * inv_dth * (th_hi.sin() * ez_thp - th_lo.sin() * ez)
    } else {
        // dB_th/dt = +(1/r_c) d_r(r E_phi). r_lo/r_hi are the corner radii bounding this theta-face,
        // r_c the cell-center radius (opposite sign to the cylinder's B_z update).
        let r_lo = gv_axis_face_at(0, spacing[0], 0);
        let r_hi = gv_axis_face_at(0, spacing[0], 1);
        let r_c = (r_lo + r_hi) * half;
        let inv_dr = Gv::ONE / (r_hi - r_lo);
        let ez_rp = gv_field_at("ez", "ez", 2, &[1, 0]);
        b + dt * (Gv::ONE / r_c) * inv_dr * (r_hi * ez_rp - r_lo * ez)
    };
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}

/// the 3D CT curl B-update along face axis `dir` (in-place on `b`), mirror of
/// `rmhd::rmhd_ct_curl_3d_dir`: `B_dir += dt*curl`, `curl = dE_p1/dx_p2 - dE_p2/dx_p1`
/// (cartesian, uniform `id_p1`/`id_p2`) or the orthogonal h-weighted curl (curvilinear, via
/// `ct_curl_metric_gv`). reads e_p1/e_p2 at the cell + `+e_p2`/`+e_p1`. div(B)=0 preserved.
pub fn rmhd_ct_curl_3d_dir_gv(
    coords: Coords,
    spacing: &[Spacing],
    dir: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let p1 = (dir + 1) % 3;
    let p2 = (dir + 2) % 3;
    let cartesian = coords == Coords::Cartesian;
    let b = Gv::field("b", "b");
    let dt = Gv::scalar("dt");
    let ids = cartesian.then(|| (Gv::scalar("id_p1"), Gv::scalar("id_p2")));
    let metric = (!cartesian).then(|| ct_curl_metric_gv(coords, spacing, dir));
    // unit offset on a single axis (the +e_p read).
    let off = |ax: usize| -> [i32; 3] {
        let mut o = [0, 0, 0];
        o[ax] = 1;
        o
    };
    let curl = if let Some(m) = metric {
        // (1/(h_p1c h_p2c)) [ d(h_p1 E_p1)/dx_p2 - d(h_p2 E_p2)/dx_p1 ], h-weighted edge EMFs.
        let de = |key: &str, runtime: &str, ax: usize, w_here: Gv, w_plus: Gv, inv_dx: Gv| {
            let e_h = gv_field_at(key, runtime, 3, &[0, 0, 0]);
            let e_p = gv_field_at(key, runtime, 3, &off(ax));
            (w_plus * e_p - w_here * e_h) * inv_dx
        };
        let de1 = de("e_p1", "e_p1", p2, m.h1_here, m.h1_p2, m.inv_dx_p2);
        let de2 = de("e_p2", "e_p2", p1, m.h2_here, m.h2_p1, m.inv_dx_p1);
        m.inv_pref * (de1 - de2)
    } else {
        let (id_p1, id_p2) = ids.expect("cartesian CT curl needs id scalars");
        let ddx = |key: &str, runtime: &str, ax: usize, inv: Gv| {
            let h = gv_field_at(key, runtime, 3, &[0, 0, 0]);
            let p = gv_field_at(key, runtime, 3, &off(ax));
            inv * (p - h)
        };
        let de1 = ddx("e_p1", "e_p1", p2, id_p2);
        let de2 = ddx("e_p2", "e_p2", p1, id_p1);
        de1 - de2
    };
    let b_new = b + dt * curl;
    (end_trace(), vec![("b_new".to_string(), "b".into(), b_new.node())])
}

/// the ISOTHERMAL CT face->cell B interpolation — `bcell_c = 0.5*(bface_c + bface_c[+e_c])`,
/// WITHOUT the 1/2|B|^2 energy correction (isothermal MHD has no energy to correct). reads
/// bface only, writes bcell only — no nrg, no bcell-old read.
pub fn imhd_bcell_from_bface_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let half = Gv::from_f64(0.5);
    let off = |ax: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[ax] = 1;
        o
    };
    // interpolate the ndim in-plane (face-staggered) components; out-of-plane components
    // (if any) are carried cell-centered and untouched here (2.5D / 1.5D — docs/design/30).
    let bf: Vec<Gv> = (0..ndim).map(|c| Gv::field(&format!("bf_{c}"), &format!("bf_{c}"))).collect();
    let writes = (0..ndim)
        .map(|c| {
            let bcc_n = (bf[c] + gv_field_at(&format!("bf_{c}"), &format!("bf_{c}"), ndim, &off(c))) * half;
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bcc_n.node())
        })
        .collect();
    (end_trace(), writes)
}

/// the CT face->cell B interpolation + magnetic-energy correction, mirror of
/// `rmhd::rmhd_bcell_from_bface`: `bcell_c = 0.5*(bface_c + bface_c[+e_c])`,
/// `nrg += 0.5*(|bcell_new|^2 - |bcell_old|^2)`. in-place on bcell + nrg.
pub fn rmhd_bcell_from_bface_gv(ndim: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let half = Gv::from_f64(0.5);
    let off = |ax: usize| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        o[ax] = 1;
        o
    };
    // field order (positional dispatch): all ndim faces, then all ndim old cells, then nrg.
    let bf: Vec<Gv> = (0..ndim).map(|c| Gv::field(&format!("bf_{c}"), &format!("bf_{c}"))).collect();
    let bc: Vec<Gv> = (0..ndim).map(|c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    let nrg = Gv::field("nrg", "nrg");
    // interpolate the ndim in-plane components from their faces; out-of-plane components
    // (Bz in 2.5D) are untouched here, and their |B|^2 term cancels in the energy diff.
    let bc_n: Vec<Gv> = (0..ndim)
        .map(|c| (bf[c] + gv_field_at(&format!("bf_{c}"), &format!("bf_{c}"), ndim, &off(c))) * half)
        .collect();
    let sumsq = |v: &[Gv]| v.iter().fold(Gv::ZERO, |a, &x| a + x * x);
    let nrg_n = nrg + half * (sumsq(&bc_n) - sumsq(&bc));
    let mut writes: Vec<(String, FieldBind, NodeId)> = (0..ndim)
        .map(|c| (format!("bc_{c}_new"), format!("bc_{c}").into(), bc_n[c].node()))
        .collect();
    writes.push(("nrg_new".to_string(), "nrg".into(), nrg_n.node()));
    (end_trace(), writes)
}

/// the CT edge EMF along edge axis `dir`, mirror of `rmhd::rmhd_edge_emf`: gather the 12
/// contact-formula inputs by integer-offset `load_at` (corner cell EMFs v_p2*b_p1 - v_p1*b_p2
/// at coord / -e_p1 / -e_p2 / -e_p1-e_p2; face EMFs from -bflux_a / +bflux_b; density fluxes),
/// then the `ct_contact_emf_gv` soft blend. 8 generic inputs the dispatch binds per edge.
pub fn rmhd_edge_emf_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // g1/g2 are the two GRID offset axes the corner stencil walks (the edge's perpendicular
    // grid plane). they are DECOUPLED from the in-plane physical components p1/p2 the runtime
    // binds to vel_p1/bcell_p1/...: for identity geometries grid axis == component (3D: g1/g2 =
    // (dir+1)%3/(dir+2)%3), but for cyl r-z the grid axes are {0,1} while the components are
    // {r=0, z=2}. the kernel is component-agnostic — only the gather offsets are geometric.
    // pin the 8 inputs in the dispatch's order (vel_p1/p2, bcell_p1/p2, bflux_a/b, fden_p1/p2);
    // the actual values are read at the gather offsets below (gv_field_at, deduped).
    gv_register_field("edge_vp1", "vel_p1");
    gv_register_field("edge_vp2", "vel_p2");
    gv_register_field("edge_bp1", "bcell_p1");
    gv_register_field("edge_bp2", "bcell_p2");
    gv_register_field("edge_bflux_a", "bflux_a");
    gv_register_field("edge_bflux_b", "bflux_b");
    gv_register_field("edge_fden_p1", "fden_p1");
    gv_register_field("edge_fden_p2", "fden_p2");
    // -1 on the listed GRID axes (ndim-length offset; the 2.5D corner walks grid axes 0/1).
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    // cell edge-EMF E_dir = v_p2*b_p1 - v_p1*b_p2 at the given offset.
    let cell = |o: &[i32]| -> Gv {
        let vp1 = gv_field_at("edge_vp1", "vel_p1", ndim, o);
        let vp2 = gv_field_at("edge_vp2", "vel_p2", ndim, o);
        let bp1 = gv_field_at("edge_bp1", "bcell_p1", ndim, o);
        let bp2 = gv_field_at("edge_bp2", "bcell_p2", ndim, o);
        vp2 * bp1 - vp1 * bp2
    };
    let ene = cell(&zero);
    let enw = cell(&cm(&[g1]));
    let ese = cell(&cm(&[g2]));
    let esw = cell(&cm(&[g1, g2]));
    // face EMFs: en=-bflux_a[coord], es=-bflux_a[-e_g2], ee=+bflux_b[coord], ew=+bflux_b[-e_g1].
    let en = Gv::ZERO - gv_field_at("edge_bflux_a", "bflux_a", ndim, &zero);
    let es = Gv::ZERO - gv_field_at("edge_bflux_a", "bflux_a", ndim, &cm(&[g2]));
    let ee = gv_field_at("edge_bflux_b", "bflux_b", ndim, &zero);
    let ew = gv_field_at("edge_bflux_b", "bflux_b", ndim, &cm(&[g1]));
    // density fluxes: fn/fs = fden_p1 at coord / -e_g2; fe/fw = fden_p2 at coord / -e_g1.
    let fnf = gv_field_at("edge_fden_p1", "fden_p1", ndim, &zero);
    let fs = gv_field_at("edge_fden_p1", "fden_p1", ndim, &cm(&[g2]));
    let fe = gv_field_at("edge_fden_p2", "fden_p2", ndim, &zero);
    let fw = gv_field_at("edge_fden_p2", "fden_p2", ndim, &cm(&[g1]));
    let emf = ct_contact_emf_gv([en, es, ee, ew], [ene, enw, ese, esw], [fnf, fs, fe, fw]);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}

/// the per-direction UCT flux/diffusion coefficients at the edge — the (a^L, a^R, d^L, d^R) of the
/// master formula (Mignone & Del Zanna 2020, Eq. 30). `al`/`ar` are the advective flux weights of the
/// upwind/downwind states (a^L + a^R = 1); `dl`/`dr` the dissipative diffusion coefficients (equal
/// for HLL/HLLC's symmetric advection, distinct for HLLD). THIS is the only solver-specific piece:
/// HLL fills it from the fast speeds (regime-generic); HLLC/HLLD swap it for the contact/Alfvén-aware
/// coefficients (Eq. 38 / 44) — the SAME master EMF kernel consumes it.
struct UctDir {
    al: Gv,
    ar: Gv,
    dl: Gv,
    dr: Gv,
}

/// HLL coefficients (Eq. 32) from the edge signal speeds `ap = max(0, lambda_max)`,
/// `am = max(0, -lambda_min)`: a^L = ap/(ap+am), a^R = am/(ap+am), d^L = d^R = ap*am/(ap+am).
fn uct_hll_coeffs(ap: Gv, am: Gv) -> UctDir {
    let eps = Gv::from_f64(1.0e-30);
    let sum = ap + am + eps;
    let d = ap * am / sum;
    UctDir { al: ap / sum, ar: am / sum, dl: d, dr: d }
}

/// HLLC coefficients (Eq. 37-38). the three-wave fan (two fast `ll<=0<=lr` + the contact `lstar`)
/// gives a^L = a^R = 1/2 and the contact-aware diffusion
///   chi^s = -(vx^s - lambda^s)/(lambda^s - lstar),   d^s = ((|lstar|-|lambda^s|)/2) chi^s + |lambda^s|/2
/// (s = L,R). less dissipative than HLL because the transverse-field jump is resolved across the
/// contact, not the fast wave. `vxl`/`vxr` are the L/R normal velocities. classical & relativistic
/// share this algebra; only `lstar` (the contact speed) is regime-specific (computed upstream).
fn uct_hllc_coeffs(ll: Gv, lr: Gv, lstar: Gv, vxl: Gv, vxr: Gv) -> UctDir {
    let half = Gv::from_f64(0.5);
    let eps = Gv::from_f64(1.0e-30);
    // guard the (lambda^s - lstar) denominators away from zero (preserve sign).
    let den_l = ll - lstar;
    let den_r = lr - lstar;
    let den_l = den_l + eps * sign_gv(den_l);
    let den_r = den_r + eps * sign_gv(den_r);
    let chi_l = (Gv::ZERO - (vxl - ll)) / den_l;
    let chi_r = (Gv::ZERO - (vxr - lr)) / den_r;
    // Eq. 38: d^s = ((|lstar| - |lambda^s|)/2) chi^s + |lstar|/2  (the LAST term is |lstar|, the
    // contact speed, NOT |lambda^s|). this is the B_x = 0 DEGENERATE case (for B_x != 0 HLLC == HLL);
    // it is the building block for the HLLD singular limit (Eq. 46, v* = 0), not a standalone solver.
    let dl = ((lstar.abs() - ll.abs()) * half) * chi_l + lstar.abs() * half;
    let dr = ((lstar.abs() - lr.abs()) * half) * chi_r + lstar.abs() * half;
    // clamp to [0, d_HLL]: the HLL diffusion is the stable upper bound (so HLLC is never MORE
    // dissipative than HLL), and 0 the lower bound (no ANTI-diffusion). this also tames the fan
    // degeneracy lambda^s -> lstar where chi^s blows up (an approximate edge-level lstar can push
    // d^s hugely negative -> anti-diffusion -> blow-up; the proper per-face lstar would not, but
    // the clamp is a robust guard regardless).
    // FLOOR (d >= 0): the diffusion coefficient must be DISSIPATIVE. with the per-face lstar, d^s can
    // still dip slightly negative where lstar approaches lambda^s (chi^s grows); allowing it (no
    // floor) yields an unphysically "sharp" result from anti-diffusion (the checkerboard-prone
    // direction). flooring at 0 is the correct physical guard. NO upper cap — HLLC's d legitimately
    // differs from HLL's, and capping at d_HLL artificially over-diffuses.
    let floor = |d: Gv| Gv::ZERO.max(d);
    UctDir { al: half, ar: half, dl: floor(dl), dr: floor(dr) }
}

/// smooth sign (for the HLLC/HLLD denominator guards): f/(|f|+eps), in [-1,1], 0 at f=0.
fn sign_gv(f: Gv) -> Gv {
    let eps = Gv::from_f64(1.0e-300);
    f / (f.abs() + eps)
}

/// the UCT edge EMF in MASTER form (Mignone & Del Zanna 2020, Eq. 33) — the structure that
/// generalizes across Riemann solvers by swapping only the per-direction (a^L, a^R, d) coefficients
/// (`uct_*_coeffs`). for the out-of-plane (z) edge:
/// ```text
///   Ez = -vbar_x (a^L_x B_y^E + a^R_x B_y^W) + d_x (B_y^E - B_y^W)   [x: advect + diffuse B_y]
///       + vbar_y (a^L_y B_x^N + a^R_y B_x^S) - d_y (B_x^N - B_x^S)   [y: advect + diffuse B_x]
/// ```
/// `vbar_t` is the upwind transverse velocity (Eq. 29, from the edge speeds); the `B` are the
/// STAGGERED div-free FACE fields (B_y on y-faces E/W, B_x on x-faces N/S); the edge speeds are the
/// MAX over the 4 surrounding cells (the paper-sanctioned maximal-diffusion edge reconstruction).
/// reduces to `v_y B_x - v_x B_y` in the symmetric-speed limit; the diffusion matches the verified
/// compact Eq. 27. div(B)=0 preserved (a CT curl of one edge EMF, independent of the coefficients).
/// component-agnostic: only the gather offsets are geometric (g1/g2 = the perpendicular grid plane).
pub fn rmhd_edge_emf_uct_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("edge_vp1", "vel_p1");
    gv_register_field("edge_vp2", "vel_p2");
    gv_register_field("edge_bface_a", "bface_a");
    gv_register_field("edge_bface_b", "bface_b");
    gv_register_field("edge_wsr1", "wsr_p1");
    gv_register_field("edge_wsl1", "wsl_p1");
    gv_register_field("edge_wsr2", "wsr_p2");
    gv_register_field("edge_wsl2", "wsl_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let zero_g = Gv::ZERO;
    // cell velocity gathers. corners about the edge (lower-left of cell NE): NE=0, NW=-g1, SE=-g2,
    // SW=-g1-g2. the side velocities are the 2-cell averages straddling the edge.
    let vp1 = |o: &[i32]| gv_field_at("edge_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("edge_vp2", "vel_p2", ndim, o);
    let vx_e = (vp1(&zero) + vp1(&cm(&[g2]))) * half; // East cells (NE, SE)
    let vx_w = (vp1(&cm(&[g1])) + vp1(&cm(&[g1, g2]))) * half; // West (NW, SW)
    let vy_n = (vp2(&zero) + vp2(&cm(&[g1]))) * half; // North (NE, NW)
    let vy_s = (vp2(&cm(&[g2])) + vp2(&cm(&[g1, g2]))) * half; // South (SE, SW)
    // edge signal speeds: MAX over the 4 surrounding cells (maximal-diffusion edge reconstruction).
    let max4 = |key: &str, path: &str| -> Gv {
        let v0 = gv_field_at(key, path, ndim, &zero);
        let v1 = gv_field_at(key, path, ndim, &cm(&[g1]));
        let v2 = gv_field_at(key, path, ndim, &cm(&[g2]));
        let v3 = gv_field_at(key, path, ndim, &cm(&[g1, g2]));
        v0.max(v1).max(v2).max(v3)
    };
    let neg_min4 = |key: &str, path: &str| -> Gv {
        let v0 = zero_g - gv_field_at(key, path, ndim, &zero);
        let v1 = zero_g - gv_field_at(key, path, ndim, &cm(&[g1]));
        let v2 = zero_g - gv_field_at(key, path, ndim, &cm(&[g2]));
        let v3 = zero_g - gv_field_at(key, path, ndim, &cm(&[g1, g2]));
        v0.max(v1).max(v2).max(v3)
    };
    let apx = zero_g.max(max4("edge_wsr1", "wsr_p1"));
    let amx = zero_g.max(neg_min4("edge_wsl1", "wsl_p1"));
    let apy = zero_g.max(max4("edge_wsr2", "wsr_p2"));
    let amy = zero_g.max(neg_min4("edge_wsl2", "wsl_p2"));
    // SOLVER-SPECIFIC coefficients (HLL here; swap uct_hll_coeffs -> hllc/hlld later).
    let cx = uct_hll_coeffs(apx, amx);
    let cy = uct_hll_coeffs(apy, amy);
    // upwind transverse velocities (Eq. 29): vbar_x upwind in x (alpha^+ carries the West/left state),
    // vbar_y upwind in y (alpha^+ carries the South/lower state).
    let eps = Gv::from_f64(1.0e-30);
    let vbar_x = (apx * vx_w + amx * vx_e) / (apx + amx + eps);
    let vbar_y = (apy * vy_s + amy * vy_n) / (apy + amy + eps);
    // staggered face B PLM-reconstructed a half-cell to the EDGE (M&DZ: the staggered transverse
    // field reconstructed from the adjacent interface — the load-bearing 2nd-order piece). geometry
    // VERIFIED vs the CT curl: Ez[i,j] is the corner (i-1/2,j-1/2); B_y is at the corner's y but
    // offset +-1/2 in x (recon along x = its transverse), B_x at the corner's x offset +-1/2 in y.
    // one-sided minmod-theta extrapolation: +1/2 toward the edge from the lower face, -1/2 from the
    // upper. needs the 2nd transverse neighbour -> bface allocated with +-2 transverse halo.
    let theta = Gv::scalar("theta");
    let recon = |key: &str, rt: &str, base: &[i32], axis: usize, sign: f64| -> Gv {
        let off = |d: i32| -> Vec<i32> { let mut o = base.to_vec(); o[axis] += d; o };
        let q0 = gv_field_at(key, rt, ndim, base);
        let qm = gv_field_at(key, rt, ndim, &off(-1));
        let qp = gv_field_at(key, rt, ndim, &off(1));
        let slope = minmod3((q0 - qm) * theta, half * (qp - qm), (qp - q0) * theta);
        q0 + Gv::from_f64(0.5 * sign) * slope
    };
    let by_e = recon("edge_bface_b", "bface_b", &zero, g1, -1.0); // B_y[i,j],   recon -1/2 in x
    let by_w = recon("edge_bface_b", "bface_b", &cm(&[g1]), g1, 1.0); // B_y[i-1,j], recon +1/2 in x
    let bx_n = recon("edge_bface_a", "bface_a", &zero, g2, -1.0); // B_x[i,j],   recon -1/2 in y
    let bx_s = recon("edge_bface_a", "bface_a", &cm(&[g2]), g2, 1.0); // B_x[i,j-1], recon +1/2 in y
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}

/// PLM-reconstruct a staggered face field a half-cell to the EDGE (M&DZ: the staggered transverse
/// field reconstructed from the adjacent interface — the 2nd-order piece that preserves smooth fields,
/// VERIFIED on the field-loop test). `base` the face offset; `axis` the reconstruction direction (the
/// face's TRANSVERSE: x for B_y on y-faces, y for B_x on x-faces); `sign` = +1 reconstructs +1/2
/// toward the edge from the lower face, -1 reconstructs -1/2 from the upper. minmod-theta slope;
/// needs the 2nd transverse neighbour, hence bface's +-2 transverse halo.
fn recon_face_to_edge(ndim: usize, theta: Gv, key: &str, rt: &str, base: &[i32], axis: usize, sign: f64) -> Gv {
    let half = Gv::from_f64(0.5);
    let off = |d: i32| -> Vec<i32> { let mut o = base.to_vec(); o[axis] += d; o };
    let q0 = gv_field_at(key, rt, ndim, base);
    let qm = gv_field_at(key, rt, ndim, &off(-1));
    let qp = gv_field_at(key, rt, ndim, &off(1));
    let slope = minmod3((q0 - qm) * theta, half * (qp - qm), (qp - q0) * theta);
    q0 + Gv::from_f64(0.5 * sign) * slope
}

/// the master-formula edge EMF combination (Eq. 33), shared by every UCT coefficient family. given
/// the per-direction coefficients + the upwind transverse velocities + the staggered face B at the
/// edge:
/// ```text
///   Ez = -vbar_x (a^L_x B_y^E + a^R_x B_y^W) + (d^R_x B_y^E - d^L_x B_y^W)
///       + vbar_y (a^L_y B_x^N + a^R_y B_x^S) - (d^R_y B_x^N - d^L_y B_x^S)
/// ```
/// (signs verified against the compact Eq. 27 diffusion + the symmetric-speed reduction v_y B_x - v_x B_y.)
fn uct_master_emf(cx: &UctDir, cy: &UctDir, vbar_x: Gv, vbar_y: Gv, by_e: Gv, by_w: Gv, bx_n: Gv, bx_s: Gv) -> Gv {
    let zero_g = Gv::ZERO;
    // a^L (= alpha^+/sum) weights the UPWIND face: West for +x (a^L -> by_w), South for +y (a^L -> bx_s)
    // — CONSISTENT with the diffusion's d^L->West/d^R->East pairing and with vbar (apx*vx_w). pairing
    // a^L to the downwind face is anti-upwind: invisible for symmetric speeds (a^L==a^R, subsonic OT)
    // but ADVECTS THE DOWNWIND state at supersonic Mach -> instability (the field-loop blow-up).
    let adv_x = zero_g - vbar_x * (cx.al * by_w + cx.ar * by_e);
    let dif_x = cx.dr * by_e - cx.dl * by_w;
    let adv_y = vbar_y * (cy.al * bx_s + cy.ar * bx_n);
    let dif_y = zero_g - (cy.dr * bx_n - cy.dl * bx_s);
    adv_x + dif_x + adv_y + dif_y
}

/// the master EMF with PER-SIDE velocities (the conservative-flux advective form, Eq. 33 exactly).
/// required when `a^L != a^R` (HLLD): factoring a single `vbar` out turns the asymmetry into a
/// `v* (B^E - B^W)` term that is anti-diffusive and blows up. here each face flux carries its OWN
/// upwind velocity. a^L (= the upwind weight) pairs with the UPWIND face: West for +x, South for +y
/// — `adv_x = -[a^L v_x^W B_y^W + a^R v_x^E B_y^E]` (matching the d^L->West diffusion pairing).
fn uct_master_emf_perside(
    cx: &UctDir, cy: &UctDir, vx_e: Gv, vx_w: Gv, vy_n: Gv, vy_s: Gv, by_e: Gv, by_w: Gv, bx_n: Gv, bx_s: Gv,
) -> Gv {
    let zero_g = Gv::ZERO;
    let adv_x = zero_g - (cx.al * vx_w * by_w + cx.ar * vx_e * by_e);
    let dif_x = cx.dr * by_e - cx.dl * by_w;
    let adv_y = cy.al * vy_s * bx_s + cy.ar * vy_n * bx_n;
    let dif_y = zero_g - (cy.dr * bx_n - cy.dl * bx_s);
    adv_x + dif_x + adv_y + dif_y
}

/// the UCT-HLLC edge EMF (master Eq. 33 + HLLC coefficients Eq. 37-38). same master formula as the
/// HLL kernel, but the diffusion uses the CONTACT speed `lstar` (the three-wave fan) -> less
/// dissipative. CLASSICAL ideal-gas (NMHD): `lstar = m_n^hll/rho^hll` is the HLL-average normal
/// velocity, computed in-kernel from the cell prims with the classical momentum flux
/// `F[m_n] = rho v_n^2 + p + |B|^2/2 - B_n^2`. edge speeds & per-side states use the MAX-over-4-cells
/// / 2-cell-average reconstruction. (IMHD: p = cs^2*rho; RMHD: relativistic conserved/flux.)
pub fn nmhd_edge_emf_uct_hllc_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("e_rho", "rho");
    gv_register_field("e_vp1", "vel_p1");
    gv_register_field("e_vp2", "vel_p2");
    gv_register_field("e_pre", "pre");
    gv_register_field("e_bp1", "bcell_p1");
    gv_register_field("e_bp2", "bcell_p2");
    gv_register_field("e_bout", "bcell_out");
    gv_register_field("e_bface_a", "bface_a");
    gv_register_field("e_bface_b", "bface_b");
    gv_register_field("e_wsr1", "wsr_p1");
    gv_register_field("e_wsl1", "wsl_p1");
    gv_register_field("e_wsr2", "wsr_p2");
    gv_register_field("e_wsl2", "wsl_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let eps = Gv::from_f64(1.0e-30);
    let zero_g = Gv::ZERO;
    let rho = |o: &[i32]| gv_field_at("e_rho", "rho", ndim, o);
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let pre = |o: &[i32]| gv_field_at("e_pre", "pre", ndim, o);
    let bp1 = |o: &[i32]| gv_field_at("e_bp1", "bcell_p1", ndim, o);
    let bp2 = |o: &[i32]| gv_field_at("e_bp2", "bcell_p2", ndim, o);
    let bout = |o: &[i32]| gv_field_at("e_bout", "bcell_out", ndim, o);
    let bsq = |o: &[i32]| {
        let a = bp1(o);
        let b = bp2(o);
        let c = bout(o);
        a * a + b * b + c * c
    };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    // corners about the edge: NE=0, NW=-g1, SE=-g2, SW=-g1-g2.
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // edge signal speeds: MAX over the 4 cells.
    let max4 = |key: &str, path: &str| -> Gv {
        let v0 = gv_field_at(key, path, ndim, &ne);
        let v1 = gv_field_at(key, path, ndim, &nw);
        let v2 = gv_field_at(key, path, ndim, &se);
        let v3 = gv_field_at(key, path, ndim, &sw);
        v0.max(v1).max(v2).max(v3)
    };
    let neg_min4 = |key: &str, path: &str| -> Gv {
        let v0 = zero_g - gv_field_at(key, path, ndim, &ne);
        let v1 = zero_g - gv_field_at(key, path, ndim, &nw);
        let v2 = zero_g - gv_field_at(key, path, ndim, &se);
        let v3 = zero_g - gv_field_at(key, path, ndim, &sw);
        v0.max(v1).max(v2).max(v3)
    };
    let apx = zero_g.max(max4("e_wsr1", "wsr_p1"));
    let amx = zero_g.max(neg_min4("e_wsl1", "wsl_p1"));
    let apy = zero_g.max(max4("e_wsr2", "wsr_p2"));
    let amy = zero_g.max(neg_min4("e_wsl2", "wsl_p2"));
    // per-FACE HLLC coefficients: each face uses ITS OWN two cells (first-order L/R) + Davis face
    // speeds (s_r = max(0, ws_r^L, ws_r^R), s_l = min(0, ws_l^L, ws_l^R)) so lstar = m_n^hll/rho^hll
    // is CONSISTENT (the contact stays inside the fan -> no degeneracy blow-up). then the diffusion
    // is MAX-combined to the edge (the maximal-diffusion edge reconstruction). vn/bn read the normal
    // velocity / normal cell-B; wsr/wsl are the direction's per-cell speed fields.
    let face_d = |l: &[i32], r: &[i32], vn: &dyn Fn(&[i32]) -> Gv, bn: &dyn Fn(&[i32]) -> Gv,
                  wsr_k: &str, wsr_p: &str, wsl_k: &str, wsl_p: &str| -> (Gv, Gv) {
        let (rl, rr) = (rho(l), rho(r));
        let (vl, vr) = (vn(l), vn(r));
        let (pl, pr) = (pre(l), pre(r));
        let (bsl, bsr) = (bsq(l), bsq(r));
        let (bnl, bnr) = (bn(l), bn(r));
        let sr = zero_g.max(gv_field_at(wsr_k, wsr_p, ndim, l).max(gv_field_at(wsr_k, wsr_p, ndim, r)));
        let sl = zero_g.min(gv_field_at(wsl_k, wsl_p, ndim, l).min(gv_field_at(wsl_k, wsl_p, ndim, r)));
        let (mxl, mxr) = (rl * vl, rr * vr);
        let fl = rl * vl * vl + pl + half * bsl - bnl * bnl;
        let fr = rr * vr * vr + pr + half * bsr - bnr * bnr;
        let inv = Gv::ONE / (sr - sl + eps);
        let rho_hll = (sr * rr - sl * rl + mxl - mxr) * inv;
        let mx_hll = (sr * mxr - sl * mxl + fl - fr) * inv;
        let lstar = mx_hll / (rho_hll + eps * sign_gv(rho_hll));
        let c = uct_hllc_coeffs(sl, sr, lstar, vl, vr);
        (c.dl, c.dr)
    };
    // x-faces (normal p1): North NW->NE, South SW->SE; MAX-combine d to the edge.
    let (dln, drn) = face_d(&nw, &ne, &vp1, &bp1, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let (dls, drs) = face_d(&sw, &se, &vp1, &bp1, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let cx = UctDir { al: half, ar: half, dl: avg2(dln, dls), dr: avg2(drn, drs) };
    // y-faces (normal p2): West SW->NW, East SE->NE.
    let (dlw, drw) = face_d(&sw, &nw, &vp2, &bp2, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let (dle, dre) = face_d(&se, &ne, &vp2, &bp2, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let cy = UctDir { al: half, ar: half, dl: avg2(dlw, dle), dr: avg2(drw, dre) };
    // upwind transverse velocities (Eq. 29) for the advective part: alpha^+ carries West / South.
    let vx_w = avg2(vp1(&nw), vp1(&sw));
    let vx_e = avg2(vp1(&ne), vp1(&se));
    let vy_s = avg2(vp2(&sw), vp2(&se));
    let vy_n = avg2(vp2(&nw), vp2(&ne));
    let vbar_x = (apx * vx_w + amx * vx_e) / (apx + amx + eps);
    let vbar_y = (apy * vy_s + amy * vy_n) / (apy + amy + eps);
    // staggered face B PLM-reconstructed a half-cell to the edge (M&DZ transverse reconstruction).
    let theta = Gv::scalar("theta");
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let emf = uct_master_emf(&cx, &cy, vbar_x, vbar_y, by_e, by_w, bx_n, bx_s);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}

/// the UCT-HLLD edge EMF (master Eq. 33 + the five-wave HLLD coefficients Eq. 40-46). the GENUINELY
/// less-diffusive EMF: its rotational/Alfvén waves resolve the transverse field for B_x != 0, which
/// HLL/HLLC cannot. per-face fan (classical ideal-gas NMHD):
///   lambda* = m_n^hll/rho^hll                         (contact, Eq. 41)
///   rho*^s  = rho^s (lambda^s - v_n^s)/(lambda^s - lambda*)
///   lambda*^{L,R} = lambda* -/+ |B_n|/sqrt(rho*^{L,R}) (rotational, Eq. 40; B_n = STAGGERED face)
///   chitilde^s = (v_n^s - lambda*)(lambda^s - lambda*)/(lambda*^s + lambda^s - 2 lambda*)  [Eq. 42,
///                the singular (lambda*^s - lambda^s) factor cancelled; verified vs both stated limits]
///   v^s = (lambda*^s + lambda^s)/(|lambda*^s| + |lambda^s|)              (per side, Eq. 45)
///   v*  = (lambda*^R + lambda*^L)/(|lambda*^R| + |lambda*^L|)            (Eq. 45) -> 0 when the
///         rotational waves collapse (Eq. 46 degenerate guard, B_x -> 0, recovers HLLC == HLL)
///   d^s = (v^s - v*)/2 * chitilde^s + (|lambda*^s| - v* lambda*^s)/2     (Eq. 44)
///   a^L = (1 + v*)/2 ,  a^R = (1 - v*)/2                                 (asymmetric advection)
/// per-face d^{L,R} MAX-combined to the edge; a^L averaged. NO floor (the cancellation + guard keep
/// d well-behaved). (IMHD: isothermal p; RMHD: relativistic fan.)
pub fn nmhd_edge_emf_uct_hlld_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("e_rho", "rho");
    gv_register_field("e_vp1", "vel_p1");
    gv_register_field("e_vp2", "vel_p2");
    gv_register_field("e_pre", "pre");
    gv_register_field("e_bp1", "bcell_p1");
    gv_register_field("e_bp2", "bcell_p2");
    gv_register_field("e_bout", "bcell_out");
    gv_register_field("e_bface_a", "bface_a");
    gv_register_field("e_bface_b", "bface_b");
    gv_register_field("e_wsr1", "wsr_p1");
    gv_register_field("e_wsl1", "wsl_p1");
    gv_register_field("e_wsr2", "wsr_p2");
    gv_register_field("e_wsl2", "wsl_p2");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let one = Gv::ONE;
    let two = Gv::from_f64(2.0);
    let eps = Gv::from_f64(1.0e-30);
    let eps_deg = Gv::from_f64(1.0e-9);
    let tiny = Gv::from_f64(1.0e-30);
    let zero_g = Gv::ZERO;
    let rho = |o: &[i32]| gv_field_at("e_rho", "rho", ndim, o);
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let pre = |o: &[i32]| gv_field_at("e_pre", "pre", ndim, o);
    let bp1 = |o: &[i32]| gv_field_at("e_bp1", "bcell_p1", ndim, o);
    let bp2 = |o: &[i32]| gv_field_at("e_bp2", "bcell_p2", ndim, o);
    let bout = |o: &[i32]| gv_field_at("e_bout", "bcell_out", ndim, o);
    let bsq = |o: &[i32]| {
        let a = bp1(o);
        let b = bp2(o);
        let c = bout(o);
        a * a + b * b + c * c
    };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    let guard = |x: Gv| x + eps * sign_gv(x);
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // per-FACE HLLD coefficients -> (a^L, d^L, d^R). `bn_face` is the STAGGERED div-free normal B at
    // this face (continuous across the fan); `vn`/`bnc` read the normal velocity / normal CELL-B.
    let hlld_face = |l: &[i32], r: &[i32], vn: &dyn Fn(&[i32]) -> Gv, bnc: &dyn Fn(&[i32]) -> Gv,
                     bn_face: Gv, wsr_k: &str, wsr_p: &str, wsl_k: &str, wsl_p: &str| -> (Gv, Gv, Gv) {
        let (rl, rr) = (rho(l), rho(r));
        let (vl, vr) = (vn(l), vn(r));
        let (pl, pr) = (pre(l), pre(r));
        let (bsl, bsr) = (bsq(l), bsq(r));
        let (bncl, bncr) = (bnc(l), bnc(r));
        let sr = zero_g.max(gv_field_at(wsr_k, wsr_p, ndim, l).max(gv_field_at(wsr_k, wsr_p, ndim, r)));
        let sl = zero_g.min(gv_field_at(wsl_k, wsl_p, ndim, l).min(gv_field_at(wsl_k, wsl_p, ndim, r)));
        let (mxl, mxr) = (rl * vl, rr * vr);
        let fml = rl * vl * vl + pl + half * bsl - bncl * bncl;
        let fmr = rr * vr * vr + pr + half * bsr - bncr * bncr;
        let inv = one / (sr - sl + eps);
        let rho_hll = (sr * rr - sl * rl + mxl - mxr) * inv;
        let mx_hll = (sr * mxr - sl * mxl + fml - fmr) * inv;
        let lstar = mx_hll / guard(rho_hll);
        // star densities (Eq. 40).
        let rho_sl = rl * (sl - vl) / guard(sl - lstar);
        let rho_sr = rr * (sr - vr) / guard(sr - lstar);
        // rotational/Alfvén speeds (Eq. 40) — STAGGERED face normal B.
        let abx = bn_face.abs();
        let lrl = lstar - abx / rho_sl.max(tiny).sqrt();
        let lrr = lstar + abx / rho_sr.max(tiny).sqrt();
        // chitilde^s (Eq. 42, singular factor cancelled).
        let chitl = (vl - lstar) * (sl - lstar) / guard(lrl + sl - two * lstar);
        let chitr = (vr - lstar) * (sr - lstar) / guard(lrr + sr - two * lstar);
        // v^s per side (Eq. 45).
        let vsl = (lrl + sl) / (lrl.abs() + sl.abs() + eps);
        let vsr = (lrr + sr) / (lrr.abs() + sr.abs() + eps);
        // v* with the B_x -> 0 degenerate guard (Eq. 46) as a smooth step.
        let vstar_raw = (lrr + lrl) / (lrr.abs() + lrl.abs() + eps);
        let g = (lrr - lrl).abs() - eps_deg * (sr - sl).abs();
        let w = half * (one + sign_gv(g));
        let vstar = w * vstar_raw;
        // d^s (Eq. 44) and a^L.
        let dl = half * (vsl - vstar) * chitl + half * (lrl.abs() - vstar * lrl);
        let dr = half * (vsr - vstar) * chitr + half * (lrr.abs() - vstar * lrr);
        (half * (one + vstar), dl, dr)
    };
    // staggered face B PLM-reconstructed a half-cell to the edge (M&DZ transverse reconstruction);
    // x-faces (bface_a, recon in y), y-faces (bface_b, recon in x). also the master diffusion jumps.
    let theta = Gv::scalar("theta");
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    // x-faces (normal p1): North NW->NE, South SW->SE.
    let (aln, dln, drn) = hlld_face(&nw, &ne, &vp1, &bp1, bx_n, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let (als, dls, drs) = hlld_face(&sw, &se, &vp1, &bp1, bx_s, "e_wsr1", "wsr_p1", "e_wsl1", "wsl_p1");
    let alx = avg2(aln, als);
    let cx = UctDir { al: alx, ar: one - alx, dl: avg2(dln, dls), dr: avg2(drn, drs) };
    // y-faces (normal p2): West SW->NW, East SE->NE.
    let (alw, dlw, drw) = hlld_face(&sw, &nw, &vp2, &bp2, by_w, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let (ale, dle, dre) = hlld_face(&se, &ne, &vp2, &bp2, by_e, "e_wsr2", "wsr_p2", "e_wsl2", "wsl_p2");
    let aly = avg2(alw, ale);
    let cy = UctDir { al: aly, ar: one - aly, dl: avg2(dlw, dle), dr: avg2(drw, dre) };
    // PER-SIDE advective velocities (East/West x-vel, North/South y-vel) — the conservative form the
    // asymmetric a^L = (1+v*)/2 requires (a single vbar would make the v* term anti-diffusive).
    let vx_w = avg2(vp1(&nw), vp1(&sw));
    let vx_e = avg2(vp1(&ne), vp1(&se));
    let vy_s = avg2(vp2(&sw), vp2(&se));
    let vy_n = avg2(vp2(&nw), vp2(&ne));
    let emf = uct_master_emf_perside(&cx, &cy, vx_e, vx_w, vy_n, vy_s, by_e, by_w, bx_n, bx_s);
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}

/// the RELATIVISTIC UCT-HLLD edge EMF (RMHD). built from the WAVE-SUM dissipative flux (Mignone &
/// Del Zanna 2020 Eq. 39 + MUB09 star states), NOT the classical coefficient form (Eq. 44) — that
/// bakes in a CLASSICAL velocity-chi that is invalid relativistically and was VERIFIED wrong
/// (telescoping test, 2026-06-24). derivation + paper proof in `literature/uct_algorithm.md` 3.5.
///
/// the EMF is the centered advection minus the per-direction dissipative flux Phi:
/// ```text
///   E_z = -1/2 (v_x^E B_y^E + v_x^W B_y^W) + 1/2 (v_y^N B_x^N + v_y^S B_x^S) + Phi_x - Phi_y
/// ```
/// where Phi is the EXACT HLLD induction-flux dissipation over the ACTUAL star fields (M&DZ Eq. 39):
/// ```text
///   Phi = 1/2 [ |lambda^L|(B_t^{sL}-B_t^L) + |lambda^{sL}|(B_c-B_t^{sL})
///             + |lambda^{sR}|(B_t^{sR}-B_c) + |lambda^R|(B_t^R-B_t^{sR}) ]
/// ```
/// `B_t^{sL,sR}` single-star (`hlld_rmhd_states.bstar`), `B_c` contact (`.bc`); `lambda` fast (`lam`),
/// `lambda^{s}` Alfven (`alf`). BOUNDED by construction (field differences times |speed| — no ratio,
/// no 1/B_t, no floor, no clamp). reduces EXACTLY to `-F_hlld_rmhd[B_t]` in 1D (verified to machine
/// precision). the STAGGERED transverse face fields are the Riemann L/R (CT consistency, M&DZ p.8) so
/// Phi damps the staggered checkerboard; cell velocities/rho/pre are the 2-cell edge average. gated on
/// `success`: where the secant fails, Phi -> the finite HLL dissipation (the lam are always finite).
pub fn rmhd_edge_emf_uct_hlld_gv(ndim: usize, g1: usize, g2: usize) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    gv_register_field("e_rho", "rho");
    gv_register_field("e_vp1", "vel_p1");
    gv_register_field("e_vp2", "vel_p2");
    gv_register_field("e_vout", "vel_out");
    gv_register_field("e_pre", "pre");
    gv_register_field("e_bp1", "bcell_p1");
    gv_register_field("e_bp2", "bcell_p2");
    gv_register_field("e_bout", "bcell_out");
    gv_register_field("e_bface_a", "bface_a");
    gv_register_field("e_bface_b", "bface_b");
    let cm = |axes: &[usize]| -> Vec<i32> {
        let mut o = vec![0i32; ndim];
        for &ax in axes {
            o[ax] = -1;
        }
        o
    };
    let zero = vec![0i32; ndim];
    let half = Gv::from_f64(0.5);
    let eps = Gv::from_f64(1.0e-30);
    let zero_g = Gv::ZERO;
    let gamma = Gv::scalar("gamma");
    let eos = IdealGas { gamma };
    let avg2 = |a: Gv, b: Gv| (a + b) * half;
    let ne = zero.clone();
    let nw = cm(&[g1]);
    let se = cm(&[g2]);
    let sw = cm(&[g1, g2]);
    // 2-cell-averaged RMHD prim straddling the edge, LOCAL (p1, p2, out) RH basis. the face-NORMAL
    // and the dissipated TRANSVERSE B are both OVERRIDDEN with the staggered div-free face values
    // (the Riemann assumes constant B_n; the transverse is the staggered face that gets dissipated).
    let prim_avg = |o1: &[i32], o2: &[i32], n_idx: usize, bn: Gv, t_idx: usize, bt: Gv| -> MhdPrim<Gv, 3> {
        let avg = |key: &str, path: &str| (gv_field_at(key, path, ndim, o1) + gv_field_at(key, path, ndim, o2)) * half;
        let rho = avg("e_rho", "rho");
        let pre = avg("e_pre", "pre");
        let v = [avg("e_vp1", "vel_p1"), avg("e_vp2", "vel_p2"), avg("e_vout", "vel_out")];
        let mut b = [avg("e_bp1", "bcell_p1"), avg("e_bp2", "bcell_p2"), avg("e_bout", "bcell_out")];
        b[n_idx] = bn;
        b[t_idx] = bt;
        MhdPrim::<Gv, 3> { hydro: Prim { rho, vel: Tensor::new(v), pre }, mag: Tensor::new(b) }
    };
    // staggered face B PLM-reconstructed a half-cell to the edge (M&DZ transverse reconstruction).
    let theta = Gv::scalar("theta");
    let bx_n = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &zero, g2, -1.0);
    let bx_s = recon_face_to_edge(ndim, theta, "e_bface_a", "bface_a", &se, g2, 1.0);
    let by_w = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &nw, g1, 1.0);
    let by_e = recon_face_to_edge(ndim, theta, "e_bface_b", "bface_b", &zero, g1, -1.0);
    // edge normal fields (the single div-free B threading the edge in each direction).
    let bx_edge = avg2(bx_n, bx_s);
    let by_edge = avg2(by_w, by_e);
    // the wave-sum dissipative flux Phi (M&DZ Eq. 39) for a Riemann whose transverse component is `t`,
    // with staggered endpoints `bt_l`,`bt_r` and the single-/double-star fields from `st`. gated on
    // `success` -> HLL dissipation (NaN-safe true select; the HLL branch uses only the finite lam).
    let wave_sum = |st: &HlldStates<Gv, 3>, t: usize, bt_l: Gv, bt_r: Gv| -> Gv {
        let phi_hlld = half
            * (st.lam[0].abs() * (st.bstar[0][t] - bt_l)
                + st.alf[0].abs() * (st.bc[t] - st.bstar[0][t])
                + st.alf[1].abs() * (st.bstar[1][t] - st.bc[t])
                + st.lam[1].abs() * (bt_r - st.bstar[1][t]));
        let ap = zero_g.max(st.lam[1]);
        let am = zero_g.max(zero_g - st.lam[0]);
        let phi_hll = (ap * am / (ap + am + eps)) * (bt_r - bt_l);
        Gv::select(st.success.cmp_gt(half), phi_hlld, phi_hll)
    };
    // x-Riemann (normal p1=0, dissipate B_y=component 1): West (NW,SW) vs East (NE,SE), staggered
    // transverse B_y = by_w / by_e, normal B_x = bx_edge.
    let x_l = prim_avg(&nw, &sw, 0, bx_edge, 1, by_w);
    let x_r = prim_avg(&ne, &se, 0, bx_edge, 1, by_e);
    let st_x = hlld_rmhd_states(&Rmhd, &eos, &x_l, &x_r, &Tensor::<Gv, 3>::unit(0));
    let phi_x = wave_sum(&st_x, 1, by_w, by_e);
    // y-Riemann (normal p2=1, dissipate B_x=component 0): South (SE,SW) vs North (NE,NW), staggered
    // transverse B_x = bx_s / bx_n, normal B_y = by_edge.
    let y_l = prim_avg(&se, &sw, 1, by_edge, 0, bx_s);
    let y_r = prim_avg(&ne, &nw, 1, by_edge, 0, bx_n);
    let st_y = hlld_rmhd_states(&Rmhd, &eos, &y_l, &y_r, &Tensor::<Gv, 3>::unit(1));
    let phi_y = wave_sum(&st_y, 0, bx_s, bx_n);
    // centered advective velocities (2-cell averages straddling the edge).
    let vp1 = |o: &[i32]| gv_field_at("e_vp1", "vel_p1", ndim, o);
    let vp2 = |o: &[i32]| gv_field_at("e_vp2", "vel_p2", ndim, o);
    let vx_w = avg2(vp1(&nw), vp1(&sw));
    let vx_e = avg2(vp1(&ne), vp1(&se));
    let vy_s = avg2(vp2(&sw), vp2(&se));
    let vy_n = avg2(vp2(&nw), vp2(&ne));
    // E_z = -1/2(v_x^E B_y^E + v_x^W B_y^W) + 1/2(v_y^N B_x^N + v_y^S B_x^S) + Phi_x - Phi_y.
    let emf = zero_g - half * (vx_e * by_e + vx_w * by_w)
        + half * (vy_n * bx_n + vy_s * bx_s)
        + phi_x
        - phi_y;
    (end_trace(), vec![("emf".to_string(), "emf".into(), emf.node())])
}

/// the RK2 edge-EMF save `e_n = e` (pointwise copy; the generic 2-buffer copy the runtime also
/// reuses for the bcell^n snapshot). write root == the read field node.
pub fn rmhd_save_efield_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let e = Gv::field("e", "e");
    (end_trace(), vec![("e_n".to_string(), "e_n".into(), e.node())])
}

/// the RK2 edge-EMF time-average `e = 0.5*(e + e_n)`, in-place on e. mirror of
/// `rmhd::rmhd_average_efield`.
pub fn rmhd_average_efield_gv() -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let e = Gv::field("e", "e");
    let en = Gv::field("e_n", "e_n");
    let e_new = Gv::from_f64(0.5) * (e + en);
    (end_trace(), vec![("e_new".to_string(), "e".into(), e_new.node())])
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

/// the OUT-OF-PLANE B component (the one not in `axes`) whose induction curl is METRIC-FREE on the
/// gridded plane, so it must use the PLAIN divergence instead of the gas area-weighted one. the
/// out-of-plane component's curl carries the prefactor 1/(h_g1 h_g2) over the gridded axes — for
/// cylindrical the only non-unit Lame factor is h_phi = r, so this is the AZIMUTHAL component (phi
/// = 1) gridded as r-z (axes [0,2]): (curl E)_phi = d_z E_r - d_r E_z has NO 1/r, yet the gas FV
/// divergence carries h_phi=r in the cell volume (a spurious F_r/r source if reused). the r-phi
/// disk (out-of-plane z, h_z=1) and cartesian are unaffected — their out-of-plane curl IS the
/// area-weighted divergence. returns Some(phi=1) only for the cyl r-z plane.
fn metric_free_oop_component(coords: Coords, axes: &[usize], ncomp: usize) -> Option<usize> {
    if coords != Coords::Cylindrical {
        return None;
    }
    (0..ncomp).find(|c| !axes.contains(c) && *c == 1)
}

/// the RMHD cell-B FLUX PREDICTOR (Euler): `bcell[c] -= dt*div(bflux_c)`, in-place. flux-evolves
/// the cell B as a conserved component (so the energy correction reads the flux-implied b_old).
/// mirror of `rmhd::rmhd_bcell_godunov_euler`; reuses `cell_geometry_gv` on curvilinear grids.
pub fn rmhd_bcell_godunov_euler_gv(
    coords: Coords,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let bc: Vec<Gv> = (0..ncomp).map(|c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    // pin the ndim*ncomp induction-flux inputs in d-outer/c-inner order (the positional dispatch
    // order [bf_0_0, bf_0_1, .., bf_1_0, ..]) before bcell_flux_div_gv reads them (it loops d).
    for d in 0..ndim {
        for c in 0..ncomp {
            gv_register_field(&format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
        }
    }
    let dt = Gv::scalar("dt");
    let (geo, dx) = bcell_godunov_geom(coords, spacing, ndim, axes);
    let oop = metric_free_oop_component(coords, axes, ncomp);
    let writes = (0..ncomp)
        .map(|c| {
            // the metric-free out-of-plane component (cyl r-z B_phi) uses the PLAIN divergence;
            // every other component the gas area-weighted (cartesian dx or the geo metric).
            let div = if Some(c) == oop {
                bcell_flux_div_plain_gv(c, ndim, spacing)
            } else {
                bcell_flux_div_gv(c, ndim, &geo, &dx)
            };
            let bnew = bc[c] - dt * div;
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bnew.node())
        })
        .collect();
    (end_trace(), writes)
}

/// the RMHD cell-B FLUX PREDICTOR (RK2 stage 2): `bcell[c] = 0.5*(bcell_n[c] + (bcell[c] -
/// dt*div(bflux_c)))`, in-place. mirror of `rmhd::rmhd_bcell_godunov_rk2`.
pub fn rmhd_bcell_godunov_rk2_gv(
    coords: Coords,
    spacing: &[Spacing],
    ndim: usize,
    ncomp: usize,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let bcn: Vec<Gv> = (0..ncomp).map(|c| Gv::field(&format!("bcn_{c}"), FieldRef::BCellN(c as u8))).collect();
    let bc: Vec<Gv> = (0..ncomp).map(|c| Gv::field(&format!("bc_{c}"), FieldRef::BCell(c as u8))).collect();
    for d in 0..ndim {
        for c in 0..ncomp {
            gv_register_field(&format!("bf_{d}_{c}"), &format!("bf_{d}_{c}"));
        }
    }
    let dt = Gv::scalar("dt");
    let half = Gv::from_f64(0.5);
    let (geo, dx) = bcell_godunov_geom(coords, spacing, ndim, axes);
    let oop = metric_free_oop_component(coords, axes, ncomp);
    let writes = (0..ncomp)
        .map(|c| {
            let div = if Some(c) == oop {
                bcell_flux_div_plain_gv(c, ndim, spacing)
            } else {
                bcell_flux_div_gv(c, ndim, &geo, &dx)
            };
            let bc_star = bc[c] - dt * div;
            let bnew = half * (bcn[c] + bc_star);
            (format!("bc_{c}_new"), format!("bc_{c}").into(), bnew.node())
        })
        .collect();
    (end_trace(), writes)
}

/// the cell-B godunov geometry: curvilinear -> the gv cell geometry (area-weighted div);
/// cartesian -> the uniform `dx_d` scalars. registered in the order the bcell godunov needs
/// (the bf_d_c fields are read later by `bcell_flux_div_gv`). 3D, identity axes.
fn bcell_godunov_geom(coords: Coords, spacing: &[Spacing], ndim: usize, axes: &[usize]) -> (Option<CellGeometryGv>, Vec<Gv>) {
    if coords == Coords::Cartesian {
        (None, (0..ndim).map(|d| Gv::scalar(&format!("dx_{d}"))).collect())
    } else {
        // axes maps grid axis -> coordinate (identity for sph/3d-cyl; [0,2] for cyl r-z) so the
        // area-weighted divergence uses the right radial axis for the cylindrical metric.
        (Some(cell_geometry_gv(coords, spacing, axes, ndim)), Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_ir::graph::{Graph, Op};

    #[test]
    fn infer_tile_spec_stencil_vs_pointwise() {
        // Gate 3: the rmhd flux builder DECLARES an explicit per-axis SLAB tile
        // (halo on the reconstruction axis `dir`, 0 transverse). a POINTWISE
        // kernel (same-cell reads only) declares no spec -> infers None.
        let (flux, _) = rmhd_flux_gv(1, 0, 0);
        assert!(!flux.coord_components.is_empty(), "flux must be a stencil kernel");
        let ts = flux.infer_tile_spec().expect("rmhd flux -> Some(TileSpec)");
        assert_eq!(ts.halo, vec![2], "PLM reconstruction radius on the single (dir=0) axis");
        assert!(!ts.tiled_field_keys.is_empty(), "tiled fields populated");

        let (c2p, _) = rmhd_c2p_gv(100);
        assert!(c2p.coord_components.is_empty(), "c2p must be pointwise (same-cell)");
        assert!(c2p.infer_tile_spec().is_none(), "pointwise c2p -> no smem tile");
    }

    #[test]
    fn adiabatic_c2p_traces_the_real_physics_to_a_kernel() {
        // the payoff: symbi-hydro's adiabatic c2p, run at S=Gv, yields a dispatchable
        // kernel — the right ABI manifest + the right writes — with NO hand-written builder.
        let (k, writes) = adiabatic_c2p_gv::<1>();
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
    }

    #[test]
    fn boundary_fill_is_the_coord_assign_instance() {
        // docs/design/33 section 7: the SAME operator (apply_dag_core_gv) as the source pass, at the
        // (Coord, Assign) coordinate. proves the abstraction is general — TWO instances, one builder.
        // a prim prescription: rho=2, vel=0.5, pre=1 (consts; a real boundary reads x/t, same path).
        use symbi_ir::graph::ConstValue;
        let mk = |v: f64| {
            let mut g = Graph::new();
            let c = g.add_const(ConstValue::F64(v), None);
            symbi_hydro::source_spec::BuiltSource { graph: g, params: vec![], outputs: vec![c] }
        };
        let (rho, vel, pre) = (mk(2.0), mk(0.5), mk(1.0));
        let sources = [("den", &rho), ("mom", &vel), ("nrg", &pre)];
        let (k, writes) = boundary_fill_from_built_gv(
            Coords::Cartesian, &[Spacing::Uniform], &[0], 1, 1, true, &sources,
        );
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
        // Assign writes the PRIM state (not cons), one DAG per slot.
        let paths: Vec<String> = writes.iter().map(|(_, p, _)| p.name()).collect();
        assert_eq!(paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        // Assign has NO `dt` weight (it is a prescription, not an RHS).
        assert!(!k.scalar_params.contains(&"dt".to_string()), "Assign carries no dt weight");
        // Coord binds NO state -> the kernel reads no `u_stage.*` / `cons.*` inputs (a pure
        // coordinate prescription). the const prims read nothing at all here.
        assert!(
            !k.field_inputs.iter().any(|(_, p)| p.name().starts_with("u_stage") || p.name().starts_with("cons")),
            "Coord/Assign reads no interior state, got inputs {:?}", k.field_inputs,
        );
    }

    #[test]
    fn boundary_fill_prescribes_cell_b_for_mhd() {
        // the toroidal driven boundary: an MHD prescription (ncomp=3) with a `bcell` slot must
        // emit prim.mag[k] writes alongside rho/vel/pre. a purely toroidal injection sets the
        // in-plane B (mag[0],mag[1]) to 0 and the out-of-plane B_phi (mag[2]) to a value.
        use symbi_ir::graph::ConstValue;
        let mk = |vals: &[f64]| {
            let mut g = Graph::new();
            let outs = vals.iter().map(|&v| g.add_const(ConstValue::F64(v), None)).collect();
            symbi_hydro::source_spec::BuiltSource { graph: g, params: vec![], outputs: outs }
        };
        let den = mk(&[1.0]);
        let mom = mk(&[0.1, 0.0, 0.0]);
        let nrg = mk(&[1.0]);
        let bcell = mk(&[0.0, 0.0, 0.5]); // B_r=0, B_theta=0, B_phi=0.5 (purely toroidal)
        let sources = [("den", &den), ("mom", &mom), ("nrg", &nrg), ("bcell", &bcell)];
        let (k, writes) = boundary_fill_from_built_gv(
            Coords::Spherical, &[Spacing::Log, Spacing::Uniform], &[0, 1], 2, 3, true, &sources,
        );
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
        let paths: Vec<String> = writes.iter().map(|(_, p, _)| p.name()).collect();
        assert_eq!(
            paths,
            vec![
                "prim.rho",
                "prim.vel[0]", "prim.vel[1]", "prim.vel[2]",
                "prim.pre",
                "prim.mag[0]", "prim.mag[1]", "prim.mag[2]",
            ],
        );
    }

    #[test]
    fn traces_axpy_to_elementwise_ir() {
        begin_trace();
        let a = Gv::param("a");
        let b = Gv::param("b");
        let c = Gv::param("c");
        let r = a * b + c;
        let root = r.node();
        let g = end_trace().graph;
        // root is Add over [Mul(a, b), c].
        match &g.node(root).op {
            Op::ElementWise(ElementWiseOp::Add, ins) => {
                assert_eq!(ins.len(), 2);
                assert!(matches!(&g.node(ins[0]).op, Op::ElementWise(ElementWiseOp::Mul, _)));
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }

    #[test]
    fn const_literal_materializes_to_const_node_on_use() {
        begin_trace();
        let two = Gv::from_f64(2.0);
        let x = Gv::param("x");
        let r = two * x; // the 2.0 literal materializes to a Const node here
        let root = r.node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::ElementWise(ElementWiseOp::Mul, ins) => {
                let has_two = ins.iter().any(|&i| {
                    matches!(&g.node(i).op, Op::Const(ConstValue::F64(v)) if *v == 2.0)
                });
                assert!(has_two, "the 2.0 literal should be a Const(F64(2.0)) node");
            }
            other => panic!("expected Mul, got {other:?}"),
        }
    }

    #[test]
    fn field_reads_build_the_kernel_abi_manifest() {
        // the input binding (1): the cons fields a c2p reads become field-read nodes
        // whose (ir_key, runtime_path) land — first-seen, deduped — in the manifest the
        // dispatch binds buffers by; declared scalars (gamma) land in the signature.
        begin_trace();
        let _den = Gv::field("cons_den", FieldRef::cons_den());
        let _mx = Gv::field("cons_mom_0", FieldRef::cons_mom(0));
        let _nrg = Gv::field("cons_nrg", FieldRef::cons_nrg());
        let _g = Gv::scalar("gamma");
        let _reread = Gv::field("cons_den", FieldRef::cons_den()); // a re-read dedups
        let k = end_trace();
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
    }

    #[test]
    fn iso_c2p_traces_the_real_physics_to_a_kernel() {
        // symbi-hydro's LOCALLY-isothermal recovery (`Cons::to_primitive` + Isothermal eos,
        // reading cs^2 from the nrg slot) at S = Gv: rho = den, vel = mom/den, pre = cs2*rho.
        // cs2 is a per-cell FIELD (the prescribed temperature) — NO scalar; this is what makes
        // the run able to be locally isothermal (cs varies per cell). global = uniform cs2.
        let (k, writes) = iso_c2p_gv::<1>();
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cs2".to_string(), "cs2".to_string()),
            ]
        );
        assert!(k.scalar_params.is_empty(), "cs2 is a field, not a scalar: {:?}", k.scalar_params);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());

        // pre = recover_pressure = cs2 * rho (the per-cell sound-speed-squared times density).
        let pre_id = writes.iter().find(|(_, rt, _)| rt.name() == "prim.pre").unwrap().2;
        assert!(
            matches!(&k.graph.node(pre_id).op, Op::ElementWise(ElementWiseOp::Mul, _)),
            "expected pre = Mul(cs2, rho), got {:?}", k.graph.node(pre_id).op
        );
    }

    #[test]
    fn srhd_c2p_traces_the_real_iterative_physics_to_a_kernel() {
        // the iterative payoff: symbi-hydro's branch-free `srhd_recover` (a carrier-generic
        // Newton on the pressure root) run at S=Gv yields a dispatchable kernel whose pressure
        // is ONE Op::IterateInline (body traced once) — the deep Newton does NOT unfold into an
        // exponential tree. the manifest + writes match the retired `srhd_c2p` Expr builder.
        let (k, writes) = srhd_c2p_gv::<1>(20);
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["prim.rho", "prim.vel[0]", "prim.pre"]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());

        // the recovered pressure is a fixed-count inline Newton loop.
        let pre_id = writes.iter().find(|(_, rt, _)| rt.name() == "prim.pre").unwrap().2;
        assert!(
            matches!(&k.graph.node(pre_id).op, Op::IterateInline { count: 20, .. }),
            "expected prim.pre = IterateInline(count=20), got {:?}",
            k.graph.node(pre_id).op
        );
    }

    #[test]
    fn rmhd_c2p_traces_the_real_bracketed_physics_to_a_kernel() {
        // the last + hardest c2p: symbi-hydro's `rmhd_recover` (KKC false-position) at
        // S=Gv yields a dispatchable kernel — 8 conserved reads + gamma, the 4 prim writes,
        // and the bracketed solve as a MULTI-accumulator IterateInline (the false-position's
        // 6-state bracket). proves iterate_vec carries the carrier-generic RMHD c2p.
        let (k, writes) = rmhd_c2p_gv(100);
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("cons_den".to_string(), FieldRef::cons_den().name()),
                ("cons_mom_0".to_string(), "cons.mom_0".to_string()),
                ("cons_mom_1".to_string(), "cons.mom_1".to_string()),
                ("cons_mom_2".to_string(), "cons.mom_2".to_string()),
                ("cons_nrg".to_string(), FieldRef::cons_nrg().name()),
                ("cons_mag_0".to_string(), "cons.mag_0".to_string()),
                ("cons_mag_1".to_string(), "cons.mag_1".to_string()),
                ("cons_mag_2".to_string(), "cons.mag_2".to_string()),
            ]
        );
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["prim.rho", "prim.vel[0]", "prim.vel[1]", "prim.vel[2]", "prim.pre"]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());

        // the false-position is a 6-accumulator IterateInline (count=100).
        let has_multi_iter = (0..k.graph.len()).any(|i| {
            matches!(&k.graph.node(NodeId(i as u32)).op,
                Op::IterateInline { accs, count: 100, .. } if accs.len() == 6)
        });
        assert!(has_multi_iter, "expected a 6-accumulator IterateInline(count=100) for the false-position");
    }

    #[test]
    fn adiabatic_flux_traces_recon_plus_hlle_to_a_kernel() {
        // the first gv FLUX: PLM reconstruction (a stencil -> LoadAt) composed with the
        // carrier-generic riemann::hlle (-> Select branches). proves Gv::field_shifted +
        // symbi-hydro's hlle build a dispatchable face-flux kernel — no srhd_side-style
        // hand-written per-component U/F. manifest + writes match the substrate hlle_flux.
        let (k, writes) = adiabatic_flux_gv::<1>(0);
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("prim_rho".to_string(), FieldRef::PrimRho.name()),
                ("prim_v0".to_string(), "prim.vel[0]".to_string()),
                ("prim_pre".to_string(), FieldRef::PrimPre.name()),
            ]
        );
        assert_eq!(
            k.scalar_params,
            vec![
                "gamma".to_string(),
                "theta".to_string(),
                "mesh_adot_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "mesh_vtrans_0".to_string(),
            ]
        );
        assert_eq!(k.coord_components, vec![0]);
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["flux.den", "flux.mom_0", "flux.nrg"]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());

        let has_load_at =
            (0..k.graph.len()).any(|i| matches!(&k.graph.node(NodeId(i as u32)).op, Op::LoadAt(..)));
        let has_select =
            (0..k.graph.len()).any(|i| matches!(&k.graph.node(NodeId(i as u32)).op, Op::Select(..)));
        assert!(has_load_at, "reconstruction should emit LoadAt stencil nodes");
        assert!(has_select, "HLLE should emit Select branches");
    }

    #[test]
    fn srhd_flux_traces_the_relativistic_hlle_to_a_kernel() {
        // same PLM + riemann::hlle pattern at the Srhd regime (relativistic U/F/wave speeds).
        // the only change from adiabatic is the regime — one HLLE source, two physics.
        let (k, writes) = srhd_flux_gv::<1>(0);
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("prim_rho".to_string(), FieldRef::PrimRho.name()),
                ("prim_v0".to_string(), "prim.vel[0]".to_string()),
                ("prim_pre".to_string(), FieldRef::PrimPre.name()),
            ]
        );
        assert_eq!(
            k.scalar_params,
            vec![
                "gamma".to_string(),
                "theta".to_string(),
                "mesh_adot_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "mesh_vtrans_0".to_string(),
            ]
        );
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["flux.den", "flux.mom_0", "flux.nrg"]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
    }

    #[test]
    fn iso_flux_traces_the_newtonian_hlle_minus_energy() {
        // the iso flux is the Newtonian flux at gamma->1 (sound speed sqrt(p/rho) from the
        // reconstructed prim.pre = cs^2(x)*rho — locally isothermal) MINUS the energy flux.
        // so it reconstructs prim.pre and writes only den + mom. it is gamma-INDEPENDENT (the
        // sound speed comes from the reconstructed pressure, not gamma), so the only scalar is
        // the PLM limiter `theta`.
        let (k, writes) = iso_flux_gv::<1>(0);
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![
                ("prim_rho".to_string(), FieldRef::PrimRho.name()),
                ("prim_v0".to_string(), "prim.vel[0]".to_string()),
                ("prim_pre".to_string(), FieldRef::PrimPre.name()),
            ]
        );
        assert_eq!(
            k.scalar_params,
            vec![
                "theta".to_string(),
                "mesh_adot_0".to_string(),
                "x_lo_0".to_string(),
                "dx_0".to_string(),
                "mesh_vtrans_0".to_string(),
            ]
        );
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec!["flux.den", "flux.mom_0"], "iso has no energy flux");
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
    }

    #[test]
    fn rmhd_flux_traces_the_mhd_hlle_to_a_kernel() {
        // RMHD flux: theta-MC PLM (the free-theta limiter) over rho/vel(3)/pre/mag(3),
        // composed with riemann::hlle_with_speeds at the Rmhd regime. the quartic wave speeds
        // are NO LONGER computed here — the flux READS the per-cell wave_speed_l/r (ws_l/ws_r,
        // bound after the 8 prim) and forms the Davis fan. 8 conserved fluxes (D, S_k, tau, B_k).
        let (k, writes) = rmhd_flux_gv(1, 0, 0);
        assert_eq!(k.scalar_params, vec!["gamma".to_string(), "theta".to_string()]);
        assert_eq!(
            k.field_inputs.iter().map(|(key, _)| key.as_str()).collect::<Vec<_>>(),
            vec![
                "prim_rho", "prim_v0", "prim_v1", "prim_v2", "prim_pre", "prim_b0", "prim_b1", "prim_b2",
                "bface_n", // <- the staggered normal-B face field (Gardiner-Stone CT coupling)
                "ws_l", "ws_r", // <- the materialized per-cell wave speeds, read for the fan
            ]
        );
        let write_paths: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(write_paths, vec![
            "flux.den", "flux.mom_0", "flux.mom_1", "flux.mom_2", "flux.nrg",
            "flux.mag_0", "flux.mag_1", "flux.mag_2",
        ]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
        // THE WIN: the quartic's resolvent-cubic transcendentals are GONE from the flux —
        // they live only in rmhd_wave_speeds_cell_gv now (computed once per cell, not per face).
        use symbi_ir::graph::ElementWiseOp as E;
        let has_transcendental = (0..k.graph.len()).any(|i| matches!(
            &k.graph.node(NodeId(i as u32)).op,
            Op::ElementWise(E::Asinh | E::Acosh | E::Cosh | E::Cos | E::Sin | E::Pow, _)));
        assert!(!has_transcendental, "flux must NOT carry the quartic's transcendentals anymore");
    }

    #[test]
    fn field_shifted_traces_a_stencil_load_at() {
        // the stencil cap (foundation for the gv flux / PLM reconstruction): a shifted field
        // read builds a LoadAt at `_coord + offset` and records the field + the coord axis in
        // the manifest; offset 0 dedups to the direct cell read of the same buffer.
        begin_trace();
        let _q0 = Gv::field_shifted("prim_rho", FieldRef::PrimRho, 1, 0, 0); // direct cell read
        let qm1 = Gv::field_shifted("prim_rho", FieldRef::PrimRho, 1, 0, -1); // left neighbour
        let qm1_id = qm1.node();
        let k = end_trace();
        assert!(
            matches!(&k.graph.node(qm1_id).op, Op::LoadAt(..)),
            "shifted read should be a LoadAt, got {:?}",
            k.graph.node(qm1_id).op
        );
        assert_eq!(
            k.field_inputs.iter().map(|(k, b)| (k.clone(), b.name())).collect::<Vec<_>>(),
            vec![("prim_rho".to_string(), FieldRef::PrimRho.name())],
        );
        assert_eq!(k.coord_components, vec![0], "axis 0's _coord recorded once");
    }

    #[test]
    fn iterate_vec_host_runs_a_coupled_recurrence() {
        // f64 multi-state iterate: a 2-state bracketed mean — converge x to the average
        // of the bounds. state=[lo, hi]; body=[(lo+hi)/2, hi]; converged when hi-lo small.
        // checks the host loop advances the whole state + early-breaks on convergence.
        let r = f64::iterate_vec(
            [0.0, 1.0],
            100,
            |[lo, hi]| [(lo + hi) * 0.5, hi],
            |[lo, _], [lo_n, _]| (lo_n - lo).abs().cmp_lt(1e-15),
            0,
        );
        assert!((r - 1.0).abs() < 1e-12, "lo should climb toward hi=1.0, got {r}");
    }

    #[test]
    fn iterate_vec_traces_to_multi_acc_iterate_inline() {
        // the bracketed-iterate cap: a 2-component coupled step traces to ONE
        // multi-accumulator IterateInline (body + per-component freeze recorded once).
        begin_trace();
        let a0 = Gv::param("a0");
        let b0 = Gv::param("b0");
        let r = Gv::iterate_vec(
            [a0, b0],
            7,
            |[a, b]| [b, a + b],            // fibonacci-style coupling
            |_, _| Gv::ZERO.cmp_lt(Gv::ZERO), // never converge (false mask, fixed count)
            0,
        );
        let root = r.node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::IterateInline { accs, inits, steps, count, result, .. } => {
                assert_eq!(*count, 7);
                assert_eq!(*result, 0);
                assert_eq!(accs.len(), 2, "two accumulators");
                assert_eq!(inits.len(), 2);
                assert_eq!(steps.len(), 2);
            }
            other => panic!("expected multi-acc IterateInline, got {other:?}"),
        }
    }

    #[test]
    fn iterate_traces_to_iterate_inline() {
        begin_trace();
        let x0 = Gv::param("x0");
        // x_{n+1} = x_n * 0.5, a fixed 3 steps. the convergence predicate IS traced on Gv:
        // scalar iterate lowers to a single-accumulator IterateInline whose step is a
        // Select(converged, OLD, NEW) — the keep-OLD freeze (carrier equivalence with the
        // host early-break).
        let r = x0.iterate(3, |x| x * Gv::from_f64(0.5), |prev, cur| (cur - prev).cmp_lt(Gv::from_f64(1e-9)));
        let root = r.node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::IterateInline { accs, steps, count, result, .. } => {
                assert_eq!(*count, 3, "fixed bound preserved");
                assert_eq!(accs.len(), 1, "single value accumulator");
                assert_eq!(*result, 0, "result is the value component");
                assert!(
                    matches!(&g.node(steps[0]).op, Op::Select(..)),
                    "the step must be the keep-OLD freeze Select, got {:?}",
                    g.node(steps[0]).op
                );
            }
            other => panic!("expected IterateInline(count=3, 1 acc, Select step), got {other:?}"),
        }
    }

    #[test]
    fn iterate_freezes_on_convergence_carrier_equivalent() {
        // the carrier-equivalence regression for Gv::iterate. a deliberately NON-idempotent
        // body (each step +1, converge at the threshold): the host early-break returns the
        // value AT convergence, while a no-freeze trace would run the full count and overshoot.
        // they agree ONLY if the traced loop freezes — before the fix this returned the
        // run-to-count value, the latent CPU-correct/GPU-wrong bug.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{emit_kernel_cpu, Cpu, CpuField, CpuFieldMut, KernelEmitInputs};

        fn ramp<S: Scalar>(start: S, count: usize, threshold: f64) -> S {
            start.iterate(
                count,
                |x| x + S::ONE,
                move |_prev, cur| cur.cmp_ge(S::from_f64(threshold)),
            )
        }

        // evaluate ramp::<Gv> on a single cell via the CPU interpreter (no nvcc needed).
        fn run_gv(count: usize, threshold: f64, start: f64) -> f64 {
            begin_trace();
            let s = Gv::field("start", "start");
            let root = ramp::<Gv>(s, count, threshold).node();
            let writes = vec![("out".to_string(), "out".into(), root)];
            let k = end_trace();
            assert!(!k.graph.has_errors(), "ramp graph errors: {:?}", k.graph.errors());
            let spec = KernelEmitInputs {
                kernel_name: "ramp",
                ndim: 1,
                target: TargetConfig { target: Target::Cuda, precision: Precision::F64 },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: &writes,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            // also exercise emission (the lowering the AOT path renders).
            let _ = emit_kernel_cpu(&k.graph, &spec);
            let (lo, extent) = ([0i32], [1u32]);
            let start_data = [start];
            let inputs = [CpuField { data: &start_data, lo: &lo, extent: &extent }];
            let mut out_data = [0.0f64];
            let mut outputs = [CpuFieldMut { data: &mut out_data, lo: &lo, extent: &extent }];
            Cpu.run_kernel(&k.graph, &spec, &inputs, &mut outputs, &[], &[1u32], &[0i32]);
            out_data[0]
        }

        // converges within the count: keep-OLD freezes at the last pre-threshold value
        // (4: at prev=4, cur=5 trips `cur >= 5`, so the OLD 4 is kept). the traced loop
        // must freeze there, NOT run to count=20 (the pre-fix value).
        let host = ramp::<f64>(0.0, 20, 5.0);
        let gv = run_gv(20, 5.0, 0.0);
        assert_eq!(host, 4.0, "host keep-OLD freeze returns the last pre-convergence value");
        assert!((gv - host).abs() < 1e-12, "carrier divergence: host={host}, gv={gv}");
        assert!((gv - 20.0).abs() > 0.5, "no freeze: traced loop ran to count ({gv})");

        // never converges within the count: both carriers run the full bound and agree.
        let host_nc = ramp::<f64>(0.0, 3, 5.0);
        let gv_nc = run_gv(3, 5.0, 0.0);
        assert_eq!(host_nc, 3.0);
        assert!((gv_nc - host_nc).abs() < 1e-12, "non-converged divergence: host={host_nc}, gv={gv_nc}");
    }

    #[test]
    fn cond_is_a_lazy_branch_carrier_equivalent_and_renders_if_else() {
        // the DUAL of iterate: `S::cond` is a real data-dependent branch. the
        // untaken arm computes acosh(x) (NaN for x < 1); with `cond` it traces
        // INTO the `if` block and runs ONLY when x > 1 — the carrier-portable
        // C++ early-`if`, not compute-all-paths.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{emit_kernel_from_lowering, Cpu, CpuField, CpuFieldMut, KernelEmitInputs};

        fn pick<S: Scalar>(x: S) -> S {
            S::cond(x.cmp_gt(S::ONE), || x.acosh(), || x * x)
        }

        // 1. TRACE STRUCTURE: the root is Op::IfElse (not Op::Select).
        begin_trace();
        let xp = Gv::param("x");
        let root = pick::<Gv>(xp).node();
        let g = end_trace().graph;
        match &g.node(root).op {
            Op::IfElse { then_results, else_results, .. } => {
                assert_eq!(then_results.len(), 1, "scalar cond → 1 then-result");
                assert_eq!(else_results.len(), 1, "scalar cond → 1 else-result");
            }
            other => panic!("expected Op::IfElse, got {other:?}"),
        }

        // run pick::<Gv> on one cell via the CPU interpreter; return value + the
        // emitted (CUDA) source for the structural check.
        fn run_gv(x: f64) -> (f64, String) {
            begin_trace();
            let xf = Gv::field("x", "x");
            let root = pick::<Gv>(xf).node();
            let writes = vec![("out".to_string(), "out".into(), root)];
            let k = end_trace();
            assert!(!k.graph.has_errors(), "pick graph errors: {:?}", k.graph.errors());
            let spec = KernelEmitInputs {
                kernel_name: "pick",
                ndim: 1,
                target: TargetConfig { target: Target::Cuda, precision: Precision::F64 },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: &writes,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            let src = emit_kernel_from_lowering(&k.graph, &spec).source;
            let (lo, extent) = ([0i32], [1u32]);
            let xdata = [x];
            let inputs = [CpuField { data: &xdata, lo: &lo, extent: &extent }];
            let mut out = [0.0f64];
            let mut outputs = [CpuFieldMut { data: &mut out, lo: &lo, extent: &extent }];
            Cpu.run_kernel(&k.graph, &spec, &inputs, &mut outputs, &[], &[1u32], &[0i32]);
            (out[0], src)
        }

        // 2. CARRIER EQUIVALENCE: f64 host == Gv interp, BIT-identical, on BOTH
        //    arms (x<1 takes else=x*x; x>1 takes then=acosh; near the boundary).
        for &x in &[0.5_f64, 2.0, 1.5, 0.999, 1.0, 3.7] {
            let host = pick::<f64>(x);
            let (gv, _) = run_gv(x);
            assert!(
                gv.to_bits() == host.to_bits() || (gv.is_nan() && host.is_nan()),
                "carrier divergence at x={x}: host={host} gv={gv}",
            );
        }

        // 3. EMITTED SOURCE is a REAL `if (...) { ... } else { ... }`, with the
        //    expensive `acosh` INSIDE the branch (after `if (`), and NO
        //    higher-order placeholder — the structural laziness proof.
        let (_, src) = run_gv(2.0);
        assert!(src.contains("if ("), "no real branch in emitted source:\n{src}");
        assert!(src.contains("} else {"), "no else arm in emitted source:\n{src}");
        assert!(!src.contains("HIGHER_ORDER"), "IfElse not intercepted by emit:\n{src}");
        let if_pos = src.find("if (").expect("if");
        let acosh_pos = src.find("acosh").expect("acosh in then-arm");
        assert!(
            acosh_pos > if_pos,
            "acosh computed BEFORE the branch (not lazy):\n{src}",
        );
    }

    #[test]
    fn cond_vec_is_an_n_output_lazy_branch_carrier_equivalent() {
        // the dual of iterate_vec: ONE branch, TWO outputs from the SAME taken
        // arm. the else arm computes a SHARED expensive value (acosh) feeding
        // both outputs — proving the arm runs once and both outputs project
        // from it: the (sl, sr) wave-speed fast-path shape.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{emit_kernel_from_lowering, Cpu, CpuField, CpuFieldMut, KernelEmitInputs};

        fn pick2<S: Scalar>(x: S) -> [S; 2] {
            // x > 1 -> (acosh(x), 2*acosh(x)) sharing acosh; else -> (x, -x).
            S::cond_vec(
                x.cmp_gt(S::ONE),
                || { let a = x.acosh(); [a, a + a] },
                || [x, S::ZERO - x],
            )
        }

        // 1. TRACE STRUCTURE: two Op::Proj over one Op::IfElse with 2 results.
        begin_trace();
        let xp = Gv::param("x");
        let out = pick2::<Gv>(xp);
        let g = end_trace().graph;
        for (j, gv) in out.iter().enumerate() {
            match &g.node(gv.node()).op {
                Op::Proj { source, index } => {
                    assert_eq!(*index as usize, j, "proj index");
                    match &g.node(*source).op {
                        Op::IfElse { then_results, else_results, .. } => {
                            assert_eq!(then_results.len(), 2, "2 then-results");
                            assert_eq!(else_results.len(), 2, "2 else-results");
                        }
                        other => panic!("proj source not IfElse: {other:?}"),
                    }
                }
                other => panic!("output {j} not a Proj: {other:?}"),
            }
        }

        // 2. CARRIER EQUIVALENCE + shared-arm: run pick2::<Gv> (both outputs)
        //    via the CPU interp, compare bit-for-bit to pick2::<f64> on both
        //    arms; assert the emitted source computes acosh exactly ONCE.
        fn run_gv(x: f64) -> [f64; 2] {
            begin_trace();
            let xf = Gv::field("x", "x");
            let out = pick2::<Gv>(xf);
            let writes = vec![
                ("o0".to_string(), "o0".into(), out[0].node()),
                ("o1".to_string(), "o1".into(), out[1].node()),
            ];
            let k = end_trace();
            assert!(!k.graph.has_errors(), "pick2 graph errors: {:?}", k.graph.errors());
            let spec = KernelEmitInputs {
                kernel_name: "pick2",
                ndim: 1,
                target: TargetConfig { target: Target::Cuda, precision: Precision::F64 },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: &writes,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            let src = emit_kernel_from_lowering(&k.graph, &spec).source;
            assert!(!src.contains("HIGHER_ORDER"), "IfElse/Proj not intercepted:\n{src}");
            assert_eq!(src.matches("acosh").count(), 1, "acosh must be SHARED (computed once):\n{src}");
            let (lo, extent) = ([0i32], [1u32]);
            let xdata = [x];
            let inputs = [CpuField { data: &xdata, lo: &lo, extent: &extent }];
            let mut o0 = [0.0f64];
            let mut o1 = [0.0f64];
            let mut outputs = [
                CpuFieldMut { data: &mut o0, lo: &lo, extent: &extent },
                CpuFieldMut { data: &mut o1, lo: &lo, extent: &extent },
            ];
            Cpu.run_kernel(&k.graph, &spec, &inputs, &mut outputs, &[], &[1u32], &[0i32]);
            [o0[0], o1[0]]
        }

        for &x in &[0.5_f64, 2.0, 1.5, 0.999, 1.0, 3.7] {
            let host = pick2::<f64>(x);
            let gv = run_gv(x);
            for j in 0..2 {
                assert!(
                    gv[j].to_bits() == host[j].to_bits() || (gv[j].is_nan() && host[j].is_nan()),
                    "carrier divergence at x={x} out{j}: host={} gv={}", host[j], gv[j],
                );
            }
        }
    }

    #[test]
    fn srhd_wave_speed_map_traces_the_real_physics() {
        // symbi-hydro's Srhd::wave_speeds_axis (Mignone-Bodo, normal velocity only) at S=Gv,
        // folded with the in-kernel cartesian-uniform widths into ONE timestep kernel — the SAME
        // physics the SRHD flux's HLLE uses. cartesian 2D: reads rho + the GRIDDED normal
        // velocities (v0, v1) + pre — the dead v2 is left ZERO and never enters the graph.
        let (k, writes) = srhd_wave_speed_map_gv(Coords::Cartesian, &[Spacing::Uniform; 2], &[0, 1], 2);
        assert_eq!(writes.len(), 1, "one scratch lambda write");
        assert_eq!(writes[0].1.name(), "scratch");
        assert_eq!(
            k.scalar_params,
            vec![
                "gamma".to_string(),
                "inv_dx_0".into(),
                "inv_dx_1".into(),
                "x_lo_0".into(),
                "dx_0".into(),
                "mesh_adot_0".into(),
                "mesh_vtrans_0".into(),
                "x_lo_1".into(),
                "dx_1".into(),
                "mesh_adot_1".into(),
                "mesh_vtrans_1".into(),
            ]
        );
        let keys: Vec<&str> = k.field_inputs.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["prim_rho", "prim_v0", "prim_v1", "prim_pre"],
            "SRHD CFL reads rho + the gridded normal velocities + pre (no dead v2)"
        );
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
    }

    #[test]
    fn coord_is_the_index_to_physical_bridge() {
        // Gv::coord(ax) is the integer cell index, usable in arithmetic against the f64 grid
        // scalars (auto-promotes) — the foundation for in-kernel geometry. a physical position
        // x = x_lo + coord*dx traces cleanly and records the axis + scalars in the manifest.
        begin_trace();
        let _x = Gv::coord(0) * Gv::scalar("dx_0") + Gv::scalar("x_lo_0");
        let k = end_trace();
        assert_eq!(k.coord_components, vec![0], "axis 0's _coord recorded once");
        assert!(k.scalar_params.contains(&"dx_0".to_string()));
        assert!(k.scalar_params.contains(&"x_lo_0".to_string()));
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
    }

    #[test]
    fn cell_inv_phys_widths_gv_match_the_geometry_per_coords() {
        // the gv metric: cartesian has scale factor 1 (no angular term); spherical's phi axis
        // carries h = r*sin(theta) -> a Sin node. proves the geometry traces in Gv from the
        // cell index, the foundation the curvilinear CFL / divergence / sources will use.
        begin_trace();
        let inv = cell_inv_phys_widths_gv(Coords::Cartesian, &[Spacing::Uniform, Spacing::Uniform], &[0, 1], 2);
        let _r: Vec<NodeId> = inv.iter().map(|g| g.node()).collect();
        let kc = end_trace();
        assert_eq!(inv.len(), 2);
        assert!(!kc.graph.has_errors(), "graph errors: {:?}", kc.graph.errors());
        let has_sin = |g: &Graph| (0..g.len()).any(|i| matches!(&g.node(NodeId(i as u32)).op, Op::ElementWise(ElementWiseOp::Sin, _)));
        assert!(!has_sin(&kc.graph), "cartesian has no angular scale factor");

        begin_trace();
        let inv = cell_inv_phys_widths_gv(Coords::Spherical, &[Spacing::Uniform, Spacing::Uniform, Spacing::Uniform], &[0, 1, 2], 3);
        let _r: Vec<NodeId> = inv.iter().map(|g| g.node()).collect();
        let ks = end_trace();
        assert!(!ks.graph.has_errors(), "graph errors: {:?}", ks.graph.errors());
        assert!(has_sin(&ks.graph), "spherical phi axis needs h = r*sin(theta)");
    }

    #[test]
    fn rmhd_wave_speed_map_traces_the_magnetosonic_bound() {
        // symbi-hydro's rmhd_magnetosonic_cfl_speeds (the cheap c_f^2 = c_s^2 + c_A^2 -
        // c_s^2 c_A^2 UPPER BOUND) at S=Gv, folded into ONE timestep kernel. it reads the full
        // 3-vector prim + gamma (vsq/bsq), same ABI as before, but is ~25x cheaper than the
        // quartic. proves the CFL no longer pays the Mignone & Del Zanna quartic's resolvent
        // cubic (asinh/acosh/cos/cosh) — those stay on the Riemann/flux path only.
        let (k, writes) = rmhd_wave_speed_map_gv(Coords::Cartesian, &[Spacing::Uniform; 3], &[0, 1, 2], 3);
        assert_eq!(writes.len(), 1);
        assert_eq!(
            k.scalar_params,
            vec!["gamma".to_string(), "inv_dx_0".into(), "inv_dx_1".into(), "inv_dx_2".into()]
        );
        let keys: Vec<&str> = k.field_inputs.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            keys,
            vec!["prim_rho", "prim_v0", "prim_v1", "prim_v2", "prim_pre", "prim_b0", "prim_b1", "prim_b2"]
        );
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
        // the magnetosonic bound has NO resolvent-cubic transcendentals — that is the win.
        use symbi_ir::graph::ElementWiseOp as E;
        let expensive = [E::Sin, E::Cos, E::Acos, E::Sinh, E::Cosh, E::Asinh, E::Acosh, E::Pow];
        let has_transcendental = (0..k.graph.len()).any(|i| {
            matches!(&k.graph.node(NodeId(i as u32)).op,
                Op::ElementWise(op, _) if expensive.contains(op))
        });
        assert!(!has_transcendental, "CFL bound must not emit the quartic's transcendentals");
        // it still computes ONE sqrt (the relativistic-addition discriminant).
        let n_sqrt = (0..k.graph.len()).filter(|&i|
            matches!(&k.graph.node(NodeId(i as u32)).op, Op::ElementWise(E::Sqrt, _))).count();
        assert!(n_sqrt >= 1, "magnetosonic bound needs the discriminant sqrt");
    }

    #[test]
    fn rmhd_wave_speeds_cell_traces_the_exact_quartic() {
        // the per-cell wave-speed kernel: the EXACT Mignone & Del Zanna quartic per cell, one
        // (lambda_min, lambda_max) pair per direction -> wave_speed_l[d] / wave_speed_r[d].
        // proves it reads the full prim + gamma, writes 6, and DOES carry the resolvent-cubic
        // transcendentals (it IS the exact quartic — the cost we're lifting off the flux).
        let (k, writes) = rmhd_wave_speeds_cell_gv(3);
        assert_eq!(writes.len(), 6, "lambda_min/max per 3 directions");
        let out_paths: Vec<String> = writes.iter().map(|(_, p, _)| p.name()).collect();
        assert_eq!(out_paths, vec![
            "wave_speed_l[0]", "wave_speed_r[0]",
            "wave_speed_l[1]", "wave_speed_r[1]",
            "wave_speed_l[2]", "wave_speed_r[2]",
        ]);
        assert_eq!(k.scalar_params, vec!["gamma".to_string()]);
        let keys: Vec<&str> = k.field_inputs.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(keys, vec![
            "prim_rho", "prim_v0", "prim_v1", "prim_v2", "prim_pre", "prim_b0", "prim_b1", "prim_b2"
        ]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
        // it IS the exact quartic -> the resolvent cubic's transcendentals ARE present here
        // (the whole point: this kernel pays them ONCE per cell, the flux pays none).
        use symbi_ir::graph::ElementWiseOp as E;
        let has_resolvent = (0..k.graph.len()).any(|i|
            matches!(&k.graph.node(NodeId(i as u32)).op, Op::ElementWise(E::Acosh, _)));
        assert!(has_resolvent, "per-cell kernel must carry the exact quartic (resolvent cubic)");
    }

    #[test]
    fn snapshot_gv_traces_a_pure_copy() {
        // u_n = cons: each write root IS the read field param (a direct buffer copy), no scalars,
        // geometry-free. ncomp=2 + energy -> cons den/mom_0/mom_1/nrg -> u_n.*.
        let (k, writes) = snapshot_gv(2, true);
        assert!(k.scalar_params.is_empty(), "snapshot takes no scalars");
        assert!(k.coord_components.is_empty(), "snapshot is pointwise (no stencil)");
        let in_rt: Vec<String> = k.field_inputs.iter().map(|(_, rt)| rt.name()).collect();
        assert_eq!(in_rt, vec!["cons.den", "cons.mom_0", "cons.mom_1", "cons.nrg"]);
        let out_rt: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(out_rt, vec!["u_n.den", "u_n.mom_0", "u_n.mom_1", "u_n.nrg"]);
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
    }

    #[test]
    fn godunov_stage_gv_traces_the_ssp_combine() {
        // in-place `cons = a0*u_n + ac*(u - dt*div(F))`: cartesian-uniform 2D, ncomp=2 + energy
        // (no geometric source). declares dt + the SSP coefficients a0/ac + the per-axis dx; reads
        // the snapshot `u_n` + the conserved fields + the per-direction fluxes (a +e_i stencil, so
        // coord axes recorded); writes the conserved set in place.
        let (k, writes) = godunov_stage_gv(
            Coords::Cartesian, &[Spacing::Uniform; 2], &[0, 1], 2, 2, true,
            GeoSource::Hydro { inertial: false },
        );
        assert_eq!(
            k.scalar_params,
            vec!["dt".to_string(), "a0".into(), "ac".into(), "mesh_hdil".into(), "dx_0".into(), "dx_1".into()],
        );
        assert_eq!(k.coord_components, vec![0, 1], "the +e_i divergence stencil records both axes");
        let out_rt: Vec<String> = writes.iter().map(|(_, rt, _)| rt.name()).collect();
        assert_eq!(out_rt, vec!["cons.den", "cons.mom_0", "cons.mom_1", "cons.nrg"], "in place");
        let in_rt: Vec<String> = k.field_inputs.iter().map(|(_, rt)| rt.name()).collect();
        // the snapshot reads the SSP `a0*u_n` term needs (held by `snapshot_gv`).
        for rt in ["u_n.den", "u_n.mom_0", "u_n.mom_1", "u_n.nrg"] {
            assert!(in_rt.iter().any(|x| x == rt), "missing snapshot input {rt}; got {in_rt:?}");
        }
        // the flux components the divergence reads (mass + per-momentum + energy, both axes).
        for rt in ["mass_flux[0]", "mass_flux[1]", "mom_flux_0[0]", "mom_flux_1[1]", "nrg_flux[0]"] {
            assert!(in_rt.iter().any(|x| x == rt), "missing flux input {rt}; got {in_rt:?}");
        }
        assert!(!k.graph.has_errors(), "graph errors: {:?}", k.graph.errors());
    }

    #[test]
    fn fused_built_core_matches_spec_adapter_trace() {
        // step-2 split gate: the SourceSpec entry (`godunov_stage_gv_with_fused_sources`) and the
        // BuiltSource core (`godunov_stage_gv_with_fused_built`) MUST emit the IDENTICAL
        // godunov+source kernel — same ABI manifest, same writes, same lowered source. proves the
        // refactor folded the AOT spec path and the runtime BuiltSource path onto ONE trace with
        // no drift. a position- AND energy-dependent family (mom + nrg) exercises the centroid
        // `x_k` binding and the energy overlay, the parts most likely to diverge under a bad split.
        use symbi_ir::emit::{Precision, Target, TargetConfig};
        use symbi_ir::{emit_kernel_from_lowering, KernelEmitInputs};

        let specs = symbi_hydro::source_spec::point_mass_gravity_sources(2, true);
        let spec_refs: Vec<&symbi_hydro::source_spec::SourceSpec> = specs.iter().collect();
        let (coords, spacing, axes) = (Coords::Cartesian, [Spacing::Uniform; 2], [0usize, 1]);
        let geo = GeoSource::Hydro { inertial: false };

        // the compile-time spec path.
        let (k_spec, w_spec) = godunov_stage_gv_with_fused_sources(
            coords, &spacing, &axes, 2, 2, true, geo, &spec_refs, false);

        // the runtime BuiltSource-value path (what `RuntimeSource` feeds).
        let builts: Vec<(&str, symbi_hydro::source_spec::BuiltSource)> = specs.iter()
            .map(|s| (s.target_field, (s.build_source)(2)))
            .collect();
        let src_refs: Vec<(&str, &symbi_hydro::source_spec::BuiltSource)> =
            builts.iter().map(|(t, b)| (*t, b)).collect();
        let (k_built, w_built) = godunov_stage_gv_with_fused_built(
            coords, &spacing, &axes, 2, 2, true, geo, &src_refs, false);

        // the ABI manifest + writes are identical (NodeIds match because both trace the SAME op
        // sequence — building the BuiltSource values outside the trace allocates no trace nodes).
        assert_eq!(k_spec.field_inputs, k_built.field_inputs, "field_inputs drift");
        assert_eq!(k_spec.scalar_params, k_built.scalar_params, "scalar_params drift");
        assert_eq!(k_spec.coord_components, k_built.coord_components, "coord_components drift");
        assert_eq!(w_spec, w_built, "writes drift");

        // the lowered source is byte-identical — the strongest structural equality available
        // (`Graph` has no `PartialEq`; the emitted source captures the full computation).
        let emit = |k: &GvKernel, w: &[(String, FieldBind, NodeId)]| {
            let spec = KernelEmitInputs {
                kernel_name: "fused_eq",
                ndim: 2,
                target: TargetConfig { target: Target::Cuda, precision: Precision::F64 },
                field_inputs: &k.field_inputs,
                scalar_params: &k.scalar_params,
                field_writes: w,
                coord_components: &k.coord_components,
                device_preamble: &[],
                tile_spec: None,
            };
            emit_kernel_from_lowering(&k.graph, &spec).source
        };
        assert_eq!(emit(&k_spec, &w_spec), emit(&k_built, &w_built), "lowered source drift");
    }
}
