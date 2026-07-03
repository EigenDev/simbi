// =============================================================================
// flux.rs
//
// face-flux kernel builders: plm reconstruction composed with the riemann solvers (hlle / hllc / hlld) across regimes.
// =============================================================================

use super::*;
use symbi_hydro::rhd::RhdGr;
use symbi_hydro::RmhdGr;
use symbi_hydro::spatial_metric::SpatialMetric;
use symbi_geometry::{KerrKS, Metric, Schwarzschild, SchwarzschildKS};


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
/// the hand-written `hlle_flux` / `rhd_hlle_flux` Expr builders + their per-component
/// U/F (rhd_side). the reconstruction is a Gv stencil (codegen-only); the HLLE is
/// carrier-generic physics. cartesian: ncomp == ndim == D, sweep coordinate == grid `dir`.
/// generic over the regime (both `Newtonian` and `Rhd` have `Prim<S,D>` / `Cons<S,D>`).
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


/// the RHD (special-relativistic Euler) face flux — `euler_hlle_flux_gv` at the `Rhd`
/// regime (relativistic U/F/wave speeds via Mignone-Bodo). replaces the `rhd_hlle_flux`
/// Expr builder + `rhd_side`. cartesian-only (rhd has no cyl r-z), ncomp == ndim == D.
pub fn rhd_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_hlle_flux_gv::<D, _>(&Rhd, D as u8, dir, dir as usize)
}


/// the RHD face flux on a curved SPATIAL metric — the `_schw`/`_ks` GR path (Valencia covariant U/F +
/// Banyuls-Font coordinate wave speeds). PLM-reconstruct the CONTRAVARIANT-velocity primitive, build
/// the in-kernel `SpatialMetric` (gamma/gamma^{-1}) + lapse from the metric at the radial face, and run
/// `riemann::hlle_with_speeds` at the `RhdGr` regime. `RhdGr` REDUCES to `Rhd` at identity gamma, so at
/// a flat metric this is bit-identical to `rhd_flux_gv`. the kerr-schild shift + the alpha
/// densitization ride the godunov (unchanged). D-generic over the sweep (metric at the swept-axis face,
/// transverse coords at the centroid); baked only for a curved spacetime.
pub fn rhd_flux_gr_gv<const D: usize>(
    dir: u8,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>)
where
    Schwarzschild<Gv>: Metric<Gv, D>,
    SchwarzschildKS<Gv>: Metric<Gv, D>,
    KerrKS<Gv>: Metric<Gv, D>,
{
    begin_trace();
    // `D` is the momentum/velocity DOF; the RECONSTRUCTION grid is `axes.len()` — they differ for
    // the spherical swirl (DOF = 3 on a 2D (r, theta) grid, out-of-plane v_phi reconstructed along
    // the gridded sweeps like any transverse component). the sweep NORMAL is coordinate `axes[dir]`.
    let ndim = axes.len();
    let (eos, left, right, nhat, vface) = euler_reconstruct::<D>(ndim as u8, dir, axes[dir as usize]);
    // the in-kernel spatial metric + lapse at the SWEPT-axis face, transverse GRIDDED coordinates at
    // the cell centroid — the correct face-metric position for a `dir` sweep. an ungridded symmetry
    // slot (the axisymmetric phi) takes zero: the spherical metrics never read phi, and
    // gamma_{phi phi} = r^2 sin^2(theta) needs only the gridded (r, theta).
    let geo = (ndim > 1).then(|| cell_geometry_gv(coords, spacing, axes, ndim));
    let x = Tensor::<Gv, D>::new(std::array::from_fn(|c| {
        if c == axes[dir as usize] {
            gv_axis_face_at(dir as usize, spacing[dir as usize], 0)
        } else {
            match axes.iter().position(|&a| a == c) {
                Some(d) => geo.as_ref().expect("a transverse gridded axis implies ndim > 1").centroid[d],
                None => {
                    assert!(c == 2, "GR flux: only the azimuthal coordinate may be ungridded");
                    Gv::ZERO
                }
            }
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    let (gamma, gamma_inv, alpha) = match spacetime {
        Spacetime::Schwarzschild => {
            let m = Schwarzschild { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x))
        }
        Spacetime::KerrSchild => {
            let m = SchwarzschildKS { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x))
        }
        Spacetime::Kerr => {
            // spinning kerr: non-diagonal gamma_{r phi} at the face — swirl (D = 3) only.
            let m = KerrKS { mass, spin: Gv::scalar("kerr_spin") };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x))
        }
        Spacetime::Minkowski => unreachable!("the GR flux is baked only for a curved spacetime"),
    };
    // spinning kerr: re-reconstruct the AZIMUTHAL velocity in the angular-momentum-carrying
    // variable w = v^phi + (gamma_{r phi} / gamma_{phi phi}) v^r, so a zero-angular-momentum
    // (S_phi = 0) state — whose frame-dragging v^phi exactly cancels against v^r in the covariant
    // lowering — reconstructs to a face pair that STILL cancels: S_phi(face) = E gamma_{phi phi} w
    // exactly, and w = 0 to roundoff for dragging states. reconstructing v^phi raw mixes the
    // geometric dragging profile into the limited slopes and generates S_phi at truncation level.
    // the per-offset coefficient q = gamma_{r phi}/gamma_{phi phi} is evaluated at each stencil
    // cell's VOLUME-WEIGHTED centroid — the exact position the c2p inverted the metric at, so the
    // cell-wise cancellation transfers to the stencil values at roundoff; the face coefficient
    // comes from the SAME face matrices the riemann states lower with. gamma_{r phi} vanishes for
    // every other background, so this block is kerr-only.
    let (left, right) = if spacetime == Spacetime::Kerr {
        assert!(D == 3, "the kerr flux carries the swirl DOF");
        let mass = Gv::scalar("schwarzschild_mass");
        let spin = Gv::scalar("kerr_spin");
        // q at the volume-weighted centroid of the cell `off` steps along the sweep axis; the
        // transverse coordinate sits at THIS cell's centroid (the stencil shifts one axis only).
        let geo = cell_geometry_gv(coords, spacing, axes, ndim);
        let q_at = |off: i32| -> Gv {
            let (r_c, th_c) = if dir == 0 {
                let rl = gv_axis_face_at(0, spacing[0], off as i64);
                let rh = gv_axis_face_at(0, spacing[0], off as i64 + 1);
                let num = gv_powi(rh, 4) - gv_powi(rl, 4);
                let den = gv_powi(rh, 3) - gv_powi(rl, 3);
                (Gv::from_f64(0.75) * num / den, geo.centroid[1])
            } else {
                let tl = gv_axis_face_at(1, spacing[1], off as i64);
                let th = gv_axis_face_at(1, spacing[1], off as i64 + 1);
                // volume-weighted polar centroid: [(sin - t cos)]_{tl}^{th} / (cos tl - cos th).
                let num = (th.sin() - th * th.cos()) - (tl.sin() - tl * tl.cos());
                (geo.centroid[0], num / (tl.cos() - th.cos()))
            };
            let m = KerrKS { mass, spin };
            let gm_c = <KerrKS<Gv> as Metric<Gv, 3>>::spatial_metric(
                &m, Tensor::<Gv, 3>::new([r_c, th_c, Gv::ZERO]),
            );
            gm_c[(0, 2)] / gm_c[(2, 2)]
        };
        let theta_lim = Gv::scalar("theta");
        let stencil = |off: i32| -> Gv {
            let vr = Gv::field_shifted("prim_v0", FieldRef::PrimVel(0), ndim as u8, dir, off);
            let vp = Gv::field_shifted("prim_v2", FieldRef::PrimVel(2), ndim as u8, dir, off);
            vp + q_at(off) * vr
        };
        let (w_l, w_r) = plm_theta_from_stencil(
            stencil(-2), stencil(-1), stencil(0), stencil(1), theta_lim,
        );
        // back to v^phi with the FACE coefficient — the same matrices the riemann states lower
        // with, so the face cancellation is exact to roundoff.
        let q_face = gamma[(0, 2)] / gamma[(2, 2)];
        let mut lv = left;
        let mut rv = right;
        lv.vel[2] = w_l - q_face * lv.vel[0];
        rv.vel[2] = w_r - q_face * rv.vel[0];
        (lv, rv)
    } else {
        (left, right)
    };
    let regime = RhdGr { metric: SpatialMetric { gamma, gamma_inv }, alpha };
    let (s_l, s_r) = regime.extremal_speeds(&eos, &left, &right, &nhat);
    // the kerr-schild charts carry a RADIAL shift: the face flux is the hll solution of the full
    // valencia system d_t U + (1/sqrt(gm)) d_r (sqrt(gm) [alpha F - beta^r U]). with the godunov
    // applying the alpha sqrt(gm) measure to the kernel flux, the exact pieces are: per-side
    // fluxes G = F - (beta^n/alpha) U, signal speeds s - beta^n (the banyuls-font s already
    // carries alpha), and fan dissipation (s_l s_r / alpha) dU — densitization then lands the
    // true central part sqrt(gm)(alpha F - beta U) AND the true dissipation sqrt(gm) s_l s_r dU.
    // beta^theta = beta^phi = 0 on both charts, so the transverse sweeps keep the shift-free
    // path; mesh motion (vface) never composes with a curved spacetime in the bake.
    let radial_shift = matches!(spacetime, Spacetime::KerrSchild | Spacetime::Kerr)
        && axes[dir as usize] == 0;
    let flux = if radial_shift {
        let beta_n = match spacetime {
            Spacetime::KerrSchild => SchwarzschildKS { mass }.shift(x)[0],
            Spacetime::Kerr => {
                KerrKS { mass, spin: Gv::scalar("kerr_spin") }.shift(x)[0]
            }
            _ => unreachable!("the radial shift is a kerr-schild-chart property"),
        };
        let u_l = regime.to_conserved(&eos, &left);
        let u_r = regime.to_conserved(&eos, &right);
        let f_l = regime.to_flux(&left, &nhat, &eos);
        let f_r = regime.to_flux(&right, &nhat, &eos);
        let w = beta_n / alpha;
        let g_l = f_l - u_l * w;
        let g_r = f_r - u_r * w;
        let sh_l = s_l - beta_n;
        let sh_r = s_r - beta_n;
        Gv::branch(
            sh_l.cmp_ge(Gv::ZERO),
            || g_l,
            || {
                Gv::branch(
                    sh_r.cmp_le(Gv::ZERO),
                    || g_r,
                    || {
                        let inv = Gv::ONE / (sh_r - sh_l);
                        (g_l * sh_r - g_r * sh_l + (u_r - u_l) * (sh_l * sh_r / alpha)) * inv
                    },
                )
            },
        )
    } else {
        hlle_with_speeds(&regime, &eos, &left, &right, &nhat, vface, s_l, s_r)
    };
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
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


/// the RMHD face flux on a curved SPATIAL metric — the GRMHD path (Valencia covariant U/F via
/// `RmhdGr` + the fast-magnetosonic-bound coordinate wave speeds). PLM-reconstruct the 8 MHD
/// primitives (the normal B from the staggered face field, Gardiner-Stone), build the in-kernel
/// `SpatialMetric` + lapse at the swept-axis face, and run the HLL fan at the `RmhdGr` regime.
/// wave speeds are INLINE (the bound is quartic-free), so the GR path skips the materialized
/// per-cell quartic the flat kernel reads. on the kerr-schild chart the fan carries the radial
/// shift exactly like the RHD GR flux — with the induction TRANSPOSE term: the true mag-row flux
/// is `(alpha v^n - beta^n) B^i - (alpha v^i - beta^i) B^n`, so beyond the uniform
/// `-(beta^n/alpha) U` subtraction every mag row i gains `+(beta^i/alpha) B^n` (the radial row of
/// a transverse sweep included; B^n is the SHARED face field, so the term is side-symmetric).
/// spinning kerr is excluded until the dragging-consistent reconstruction extends to B (design
/// 44 phase C). the metric's ungridded polar slot takes the equatorial pi/2 (exact for the
/// theta-symmetric 1D radial problem), the azimuthal slot zero.
pub fn rmhd_flux_gr_gv(
    dir: u8,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    hlld: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let ndim = axes.len();
    let coord_n = axes[dir as usize];
    let gamma_eos = Gv::scalar("gamma");
    let theta_lim = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim as u8, dir, theta_lim);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_v{k}"), FieldRef::PrimVel(k as u8), ndim as u8, dir, theta_lim);
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim as u8, dir, theta_lim);
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(&format!("prim_b{k}"), &format!("prim.mag[{k}]"), ndim as u8, dir, theta_lim);
        bl.push(l);
        br.push(r);
    }
    // normal B from the staggered FACE field (Gardiner-Stone CT coupling) — shared by both sides.
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim as u8, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let eos = IdealGas { gamma: gamma_eos };
    let mk = |rho: Gv, v: &[Gv], p: Gv, b: &[Gv]| MhdPrim::<Gv, 3> {
        hydro: Prim { rho, vel: Tensor::new([v[0], v[1], v[2]]), pre: p },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    let left = mk(rho_l, &vl, pre_l, &bl);
    let right = mk(rho_r, &vr, pre_r, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);

    // the metric at the SWEPT-axis face, transverse gridded slots at the cell centroid; the
    // ungridded polar slot is the exact equatorial pi/2, the azimuthal slot zero.
    let geo = (ndim > 1).then(|| cell_geometry_gv(coords, spacing, axes, ndim));
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        if c == coord_n {
            gv_axis_face_at(dir as usize, spacing[dir as usize], 0)
        } else {
            match axes.iter().position(|&a| a == c) {
                Some(d) => geo.as_ref().expect("a transverse gridded axis implies ndim > 1").centroid[d],
                None if c == 1 => Gv::from_f64(std::f64::consts::FRAC_PI_2),
                None => Gv::ZERO,
            }
        }
    }));
    let mass = Gv::scalar("schwarzschild_mass");
    let (gamma, gamma_inv, alpha, beta) = match spacetime {
        Spacetime::Schwarzschild => {
            let m = Schwarzschild { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), <Schwarzschild<Gv> as Metric<Gv, 3>>::shift(&m, x))
        }
        Spacetime::KerrSchild => {
            let m = SchwarzschildKS { mass };
            (m.spatial_metric(x), m.spatial_metric_inv(x), m.lapse(x), <SchwarzschildKS<Gv> as Metric<Gv, 3>>::shift(&m, x))
        }
        Spacetime::Kerr => panic!(
            "spinning-kerr GRMHD is design-44 phase C: the dragging-consistent reconstruction \
             does not yet extend to B"
        ),
        Spacetime::Minkowski => unreachable!("the GRMHD flux is baked only for a curved spacetime"),
    };
    let regime = RmhdGr { metric: SpatialMetric { gamma, gamma_inv }, alpha };
    let (s_l, s_r) = regime.extremal_speeds(&eos, &left, &right, &nhat);
    let has_shift = matches!(spacetime, Spacetime::KerrSchild);
    // GR HLLD (the metric-generalized MUB09 fan): the Schwarzschild chart has ZERO shift, so the
    // solver's intercell flux is the complete kernel flux (the godunov applies alpha). the
    // kerr-schild/kerr charts carry a radial shift whose HLLD moving-interface (x/t = beta) fan
    // is a documented increment — gate loud rather than silently drop it.
    if hlld {
        assert!(
            !has_shift,
            "GR HLLD requires a zero-shift chart (Schwarzschild); the kerr-schild/kerr shifted \
             HLLD fan is a design-44 increment"
        );
        let flux = hlld_rmhd(&regime, &eos, &left, &right, &nhat, Gv::ZERO, &regime.metric);
        let mut writes = vec![("flux_den".to_string(), FieldRef::flux_den().into(), flux.den.node())];
        for k in 0..3 {
            writes.push((format!("flux_mom_{k}"), FieldRef::flux_mom(k as u8).into(), flux.mom[k].node()));
        }
        writes.push(("flux_nrg".to_string(), FieldRef::flux_nrg().into(), flux.nrg.node()));
        for k in 0..3 {
            writes.push((format!("flux_mag_{k}"), format!("flux.mag_{k}").into(), flux.mag[k].node()));
        }
        let k = end_trace();
        let stencil_keys = k.stencil_read_field_keys();
        if stencil_keys.is_empty() {
            return (k, writes);
        }
        let mut halo = vec![0u8; ndim];
        halo[dir as usize] = 2;
        let k = k.with_tile_spec(TileSpec { halo, tiled_field_keys: stencil_keys });
        return (k, writes);
    }
    let flux = if has_shift {
        // the shifted-system HLL (the RHD GR fan) with the induction transpose add per side.
        let beta_n = beta[coord_n];
        let w = beta_n / alpha;
        let u_l = regime.to_conserved(&eos, &left);
        let u_r = regime.to_conserved(&eos, &right);
        let f_l = regime.to_flux(&left, &nhat, &eos);
        let f_r = regime.to_flux(&right, &nhat, &eos);
        let mut g_l = f_l - u_l * w;
        let mut g_r = f_r - u_r * w;
        let transpose = beta.scale(bn_face / alpha);
        g_l.mag = g_l.mag + transpose;
        g_r.mag = g_r.mag + transpose;
        let sh_l = s_l - beta_n;
        let sh_r = s_r - beta_n;
        Gv::branch(
            sh_l.cmp_ge(Gv::ZERO),
            || g_l,
            || {
                Gv::branch(
                    sh_r.cmp_le(Gv::ZERO),
                    || g_r,
                    || {
                        let inv = Gv::ONE / (sh_r - sh_l);
                        (g_l * sh_r - g_r * sh_l + (u_r - u_l) * (sh_l * sh_r / alpha)) * inv
                    },
                )
            },
        )
    } else {
        hlle_with_speeds(&regime, &eos, &left, &right, &nhat, Gv::ZERO, s_l, s_r)
    };

    let mut writes = vec![("flux_den".to_string(), FieldRef::flux_den().into(), flux.den.node())];
    for k in 0..3 {
        writes.push((format!("flux_mom_{k}"), FieldRef::flux_mom(k as u8).into(), flux.mom[k].node()));
    }
    writes.push(("flux_nrg".to_string(), FieldRef::flux_nrg().into(), flux.nrg.node()));
    for k in 0..3 {
        writes.push((format!("flux_mag_{k}"), format!("flux.mag_{k}").into(), flux.mag[k].node()));
    }
    let k = end_trace();
    let stencil_keys = k.stencil_read_field_keys();
    if stencil_keys.is_empty() {
        return (k, writes);
    }
    let mut halo = vec![0u8; ndim];
    halo[dir as usize] = 2;
    let k = k.with_tile_spec(TileSpec { halo, tiled_field_keys: stencil_keys });
    (k, writes)
}


// =============================================================================
// HLLC face flux — contact-resolving 3-wave solver, regime-specific bodies. one
// builder per regime (Newtonian, RHD, RMHD) mirroring the HLLE builder shape:
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


/// adiabatic HLLC-LM face flux: the same builder as `adiabatic_hllc_flux_gv` but with the
/// FLEISCHMANN et al. (2020) low-mach / low-dissipation arm -- the anti-diffusive star-state flux is
/// scaled by the adaptive `phi` (local mach, with shock / interface / alignment overrides), which
/// recovers standard HLLC at supersonic faces and central differencing at zero mach. cures the
/// grid-aligned shock instability AND the HLLC low-mach over-dissipation. newtonian only (the
/// relativistic HLLC bodies ignore the LM correction). the `phi` helpers are fully branchless
/// (`S::select`), so the Fleischmann arm traces at S = Gv just like the Standard arm.
pub fn adiabatic_hllc_lm_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat, vface) = euler_reconstruct::<D>(D as u8, dir, dir as usize);
    let flux = hllc(&eos, &left, &right, &nhat, vface, ShockwaveLimiter::Fleischmann);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}


/// RHD HLLC face flux — Mignone-Bodo (2005) quadratic for the contact speed.
/// mirrors `euler_hlle_flux_gv(&Rhd, ...)` but calls `riemann::hllc_rhd`.
pub fn rhd_hllc_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat, vface) = euler_reconstruct::<D>(D as u8, dir, dir as usize);
    let flux = hllc_rhd(&eos, &left, &right, &nhat, vface, ShockwaveLimiter::Standard);
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
    let flux = hlld_rmhd(&Rmhd, &eos, &left, &right, &nhat, Gv::ZERO, &SpatialMetric::flat());
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}
