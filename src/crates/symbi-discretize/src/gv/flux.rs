// =============================================================================
// flux.rs
//
// face-flux kernel builders: plm reconstruction composed with the riemann solvers (hlle / hllc / hlld) across regimes.
// =============================================================================

use super::*;
use symbi_algebra::Matrix;
use symbi_geometry::{
    KerrKS, KerrKSCartesian, KerrKSCylindrical, Metric, SchwarzschildKS, SchwarzschildKSCartesian,
    SchwarzschildKSCylindrical,
};
use crate::coords::Balance;
use symbi_hydro::RmhdGr;
use symbi_hydro::rhd::RhdGr;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};

/// trace the newtonian-MHD face flux — PLM-reconstruct the 8-component MHD
/// primitive (rho, v_{0,1,2}, pre, B_{0,1,2}) to the face, then the canonical
/// `riemann::hlle(&NewtonianMhd, ...)`. the davis fan speeds are computed inline by
/// `hlle` from the reconstructed L/R states (the closed-form magnetosonic is cheap), so
/// this arm runs off the face states alone, while `rmhd_flux_gv` materializes its quartic
/// speeds in a per-cell field and kernel. `ndim` is the reconstruction grid; `dir`
/// the sweep axis (RMHD/NMHD are fixed 3D in the velocity/field components).
// shared NMHD face-flux reconstruction: bind gamma + theta, PLM-reconstruct the
// 8-component MHD primitive (rho, v_{0..2}, pre, B_{0..2}) to the face. assumes
// begin_trace() is active. returns the eos + L/R primitives + the sweep normal —
// the solver (HLLE / HLLC / HLLD) is the only thing that differs.
// reconstruct the L/R MHD primitives at the `dir`-grid face. the PLM stencil shifts along
// grid axis `dir`; the normal is physical component `coord_n` (= axes[dir]; == dir for
// cartesian/identity, [0,2][dir] for cyl r-z) — nhat and the staggered normal-B override
// both index `coord_n`, while the face field is read along grid `dir`.
fn nmhd_reconstruct(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (IdealGas<Gv>, MhdPrim<Gv, 3>, MhdPrim<Gv, 3>, Tensor<Gv, 3>) {
    let gamma = Gv::scalar("gamma");
    let theta = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            &format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8),
            ndim,
            dir,
            theta,
        );
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim, dir, theta);
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            &format!("prim_b{k}"),
            &format!("prim.mag[{k}]"),
            ndim,
            dir,
            theta,
        );
        bl.push(l);
        br.push(r);
    }
    // the normal field is the staggered, divergence-free face field — read it directly (a
    // single value already living at the face) and override the cell-reconstructed B normal
    // component. gardiner-stone (2005) CT-godunov coupling: reconstructed bcell gives
    // bn_l != bn_r, breaking the riemann solver's constant-Bn assumption (OT noise/blow-up).
    // the face field is read along grid axis `dir`; the overridden component is the physical
    // normal `coord_n` (they coincide for cartesian; differ for the cyl r-z swirl/axisym).
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let mk = |rho: Gv, v: &[Gv], p: Gv, b: &[Gv]| MhdPrim::<Gv, 3> {
        hydro: Prim {
            rho,
            vel: Tensor::new([v[0], v[1], v[2]]),
            pre: p,
        },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    let left = mk(rho_l, &vl, pre_l, &bl);
    let right = mk(rho_r, &vr, pre_r, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);
    (IdealGas { gamma }, left, right, nhat)
}

// the 8 conserved face-flux writes (D, S_{0..2}, nrg, B_{0..2}).
fn nmhd_flux_writes(flux: &MhdCons<Gv, 3>) -> Vec<(String, FieldBind, NodeId)> {
    let mut writes = vec![(
        "flux_den".to_string(),
        FieldRef::flux_den().into(),
        flux.den.node(),
    )];
    for k in 0..3 {
        writes.push((
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8).into(),
            flux.mom[k].node(),
        ));
    }
    writes.push((
        "flux_nrg".to_string(),
        FieldRef::flux_nrg().into(),
        flux.nrg.node(),
    ));
    for k in 0..3 {
        writes.push((
            format!("flux_mag_{k}"),
            format!("flux.mag_{k}").into(),
            flux.mag[k].node(),
        ));
    }
    writes
}

pub fn nmhd_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hlle(&NewtonianMhd, &eos, &left, &right, &nhat, Gv::ZERO);
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// NMHD HLLC face flux — `hllc_newtonian` (Li 2005, contact-resolving, transverse-B
/// continuous) on the reconstructed L/R states. wave speeds inline, from the face states
/// alone (the manifest stays free of ws_l/ws_r).
pub fn nmhd_hllc_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hllc_newtonian(
        &eos,
        &left,
        &right,
        &nhat,
        Gv::ZERO,
        ShockwaveLimiter::Standard,
    );
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// NMHD HLLD face flux — `hlld_newtonian` (miyoshi-kusano 2005, full 5-wave). the
/// robust solver: the algebraic c2p + this closed-form HLLD make orszag-tang stable.
pub fn nmhd_hlld_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hlld_newtonian(&eos, &left, &right, &nhat, Gv::ZERO);
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

// shared isothermal face-flux reconstruction: bind cs + theta, PLM-reconstruct the
// 7-component iso-MHD primitive the isothermal system carries (rho, v_{0..2}, B_{0..2})
// to the face. the face-normal field comes from the staggered face field (bface coupling, see
// nmhd_reconstruct). returns the Isothermal eos + L/R primitives + the sweep normal.
fn imhd_reconstruct(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (
    Isothermal<Gv>,
    IsoMhdPrim<Gv, 3>,
    IsoMhdPrim<Gv, 3>,
    Tensor<Gv, 3>,
) {
    let cs = Gv::scalar("cs");
    let theta = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            &format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8),
            ndim,
            dir,
            theta,
        );
        vl.push(l);
        vr.push(r);
    }
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            &format!("prim_b{k}"),
            &format!("prim.mag[{k}]"),
            ndim,
            dir,
            theta,
        );
        bl.push(l);
        br.push(r);
    }
    // staggered div-free normal face field (gardiner-stone CT coupling): read along grid `dir`,
    // override the physical normal component `coord_n` (= axes[dir]). see nmhd_reconstruct.
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let mk = |rho: Gv, v: &[Gv], b: &[Gv]| IsoMhdPrim::<Gv, 3> {
        hydro: PrimG {
            rho,
            vel: Tensor::new([v[0], v[1], v[2]]),
            pre: Zero::default(),
        },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    let left = mk(rho_l, &vl, &bl);
    let right = mk(rho_r, &vr, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);
    (Isothermal { cs }, left, right, nhat)
}

// the 7 conserved face-flux writes the isothermal system carries: D, S_{0..2}, B_{0..2}.
fn imhd_flux_writes(flux: &IsoMhdCons<Gv, 3>) -> Vec<(String, FieldBind, NodeId)> {
    let mut writes = vec![(
        "flux_den".to_string(),
        FieldRef::flux_den().into(),
        flux.den.node(),
    )];
    for k in 0..3 {
        writes.push((
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8).into(),
            flux.mom[k].node(),
        ));
    }
    for k in 0..3 {
        writes.push((
            format!("flux_mag_{k}"),
            format!("flux.mag_{k}").into(),
            flux.mag[k].node(),
        ));
    }
    writes
}

/// isothermal-MHD HLLE face flux.
pub fn imhd_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = imhd_reconstruct(ndim, dir, coord_n);
    let flux = hlle(&IsothermalMhd, &eos, &left, &right, &nhat, Gv::ZERO);
    let writes = imhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// isothermal-MHD HLLD face flux — `hlld_isothermal` (mignone 2007, 3-state).
pub fn imhd_hlld_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = imhd_reconstruct(ndim, dir, coord_n);
    let flux = hlld_isothermal(&eos, &left, &right, &nhat, Gv::ZERO);
    let writes = imhd_flux_writes(&flux);
    (end_trace(), writes)
}

// =============================================================================
// face flux — PLM reconstruction (Gv stencil) composed with the carrier-generic
// `riemann::hlle` (symbi-hydro). the reconstruction is codegen-only (the host uses
// the compiled kernel in place of a DomainForEach); the HLLE physics has that one source.
// =============================================================================

/// the moving-mesh grid velocity at the face this thread owns:
/// `vface = mesh_adot_{dir} * x_face + mesh_vtrans_{dir}`, with the face coordinate taken
/// through the axis map the cell geometry uses (the thread coordinate on a face domain is
/// the face index). the dispatch decides the semantics per instance: homologous binds
/// `mesh_adot_{dir} = a_dot/a` with physical geometry scalars (so vface = H * r, and zero on
/// non-expanding curvilinear axes); uniform translation binds `mesh_vtrans_{dir} = a_dot` on
/// axis 0. the static binding (both zero) traces arithmetic that is bit-identical to the static
/// flux. the per-axis names follow the convention the wave-speed map uses, minted through
/// `MeshScalar` so the trace and the dispatch stay in lockstep.
///
/// the face position comes from `gv_axis_face_at`, the mapped axis position. on a
/// homologously expanding mesh the grid velocity is multiplied by the face area and differenced
/// against the cell volume, and both of those are built from the mapped faces. in spherical
/// geometry that difference is an exact identity —
///   div(rho vface) = [4 pi H rho r_hi^3 - 4 pi H rho r_lo^3] / [(4 pi/3)(r_hi^3 - r_lo^3)] = 3 H rho
/// — which cancels the dilution term `mesh_hdil = 3 H` for any face positions, uniform or graded.
/// the cancellation holds while vface and the geometry agree on where the face lies; a linear
/// position `x_lo + i*dx` on a graded axis breaks it by the amount the two reconstructions
/// differ, which grows with the grading.
fn mesh_face_velocity_gv(dir: u8) -> Gv {
    let mesh_adot = Gv::scalar(&MeshScalar::Adot(dir).name());
    let x_face = crate::gv::geometry::gv_axis_face_at(dir as usize, Spacing::Uniform, 0);
    mesh_adot * x_face + Gv::scalar(&MeshScalar::Vtrans(dir).name())
}

// shared euler (ideal-gas newtonian/relativistic) face reconstruction: bind the
// scalar tail (gamma, theta), theta-MC PLM-reconstruct the (rho, vel_{0..D}, pre)
// primitive to the `dir`-grid face, and return the IdealGas eos + L/R primitives +
// the sweep normal + the moving-face velocity. the solver (HLLE / HLLC) is the only
// thing that differs. `ndim` is the reconstruction grid (stencil shifts along grid
// axis `dir`); `coord_n` is the sweep coordinate (normal velocity is vel[coord_n]).
/// the well-balanced anchor: what a hydrostatic reconstruction needs to evaluate the body
/// potential at the stencil's own positions. `None` reconstructs the state directly.
#[derive(Clone, Copy)]
pub struct Balanced<'a> {
    pub n_bodies: usize,
    pub coords: Coords,
    pub axes: &'a [usize],
}

fn euler_reconstruct<const D: usize>(
    ndim: u8,
    dir: u8,
    coord_n: usize,
    recon: Recon,
    balanced: Option<Balanced<'_>>,
) -> (Prim<Gv, D>, Prim<Gv, D>, Tensor<Gv, D>, Gv) {
    // theta comes second in the manifest order [gamma, theta]: the caller registers gamma
    // inside the same trace before calling here (the eos construction lives with the
    // caller so the closure can be gamma-law or taub-mathews), and theta is registered
    // on every recon arm so the scalar tail is uniform; the ppm parabola carries its
    // own monotonicity constraint, so theta stays bound-but-inert there.
    let theta = Gv::scalar("theta");
    // well-balancing changes the thermodynamic pair alone: the limiter acts on each cell's
    // departure from the hydrostatic profile, in place of the raw state. everything after
    // this -- the velocity loop, the ppm flattening, the normal and the face velocity -- is
    // shared, which is the point. a second copy of this function silently lost the flattening
    // and hardcoded the normal to the sweep axis.
    // field-registration order is ABI. the traced manifest records fields in first-read
    // order, and every existing plain flux kernel registers [rho, v.., pre] -- so the plain
    // branch must read the velocities between the thermodynamic pair, exactly as it always
    // has, or the manifest of every already-baked kernel silently reorders. the balanced
    // branch computes rho and pre jointly (they share anchors and potentials), so its new
    // kernels register [rho, pre, v..]; that order is theirs from birth and equally stable.
    let (rho_l, rho_r, wb_pre) = match balanced {
        None => {
            let (rl, rr) = recon_gv("prim_rho", "prim.rho", ndim, dir, theta, recon);
            (rl, rr, None)
        }
        Some(b) => {
            let (rl, rr, pl, pr) = balanced_thermo_pair(ndim, dir, recon, theta, b);
            (rl, rr, Some((pl, pr)))
        }
    };
    let mut vl = Vec::with_capacity(D);
    let mut vr = Vec::with_capacity(D);
    for k in 0..D {
        let (l, r) = recon_gv(
            &format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8),
            ndim,
            dir,
            theta,
            recon,
        );
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = match wb_pre {
        Some(pair) => pair,
        None => recon_gv("prim_pre", "prim.pre", ndim, dir, theta, recon),
    };
    let mut rho_lr = (rho_l, rho_r);
    let mut pre_lr = (pre_l, pre_r);
    let mut vel_lr: Vec<(Gv, Gv)> = vl.into_iter().zip(vr).collect();
    if recon == Recon::Ppm {
        // convergence-gated flattening, runtime-dialed: the monotonized
        // parabola's dispersive truncation is anti-diffusive in strongly
        // converging flow, where its small face jumps also starve the riemann
        // solver's entropy-producing upwind dissipation — the pairing destroys
        // entropy (K = p/rho^gamma falls below its lagrangian value) in smooth
        // sustained compressions such as gravitational infall onto a sink,
        // where a limited linear reconstruction holds the adiabat through its
        // larger dissipative jumps. blend each cell's interface values toward
        // its average by the compression the flow crosses per cell, measured
        // against the local isothermal sound speed:
        // c = max(0, -(v_{+1} - v_{-1})/2) / sqrt(p/rho), ramped from
        // `flatten_onset` to full at `flatten_full`.
        //
        // the dials are runtime scalars because the serving pair depends on the
        // regime: the sink-infall vent needs full flatten by c ~ 0.05 (the
        // sealed-wall standing layer; a mid-ramp coefficient there vents and
        // the dip grows with resolution), while trans-sonic turbulence lives
        // at c ~ 0.05-0.3 in every eddy collision — a flatten active there
        // degrades the parabola to first order across the box and its retained
        // kinetic energy falls below even coarse plm. the default (both dials
        // zero) is the pure parabola: `flatten_full <= flatten_onset` zeroes
        // the ramp inverse, so f = 0 everywhere and the blend is exact
        // passthrough. gravity-sink configs declare their own dials.
        // interaction with well-balancing. the blend pulls each face value toward its own
        // cell average, and the two cells either side of a face have different averages -- so
        // where it fires it reintroduces the very jump the balanced reconstruction removes.
        // on a state at rest the sensor -- the velocity convergence across the cell -- is
        // identically zero, so `f = 0` and the blend is exact passthrough.
        // a balanced atmosphere is therefore preserved exactly, and the degradation is confined
        // to genuinely compressing flow, where robustness is the reason the flatten exists.
        // (blending toward the equilibrium value would preserve balance under compression too;
        // the cell-average form is what stays, because the flatten is shared and its meaning
        // has to hold for every kernel that reads it.)
        let onset = Gv::scalar("flatten_onset");
        let full = Gv::scalar("flatten_full");
        let half = Gv::from_f64(0.5);
        let width = full - onset;
        let ramp = Gv::select(width.cmp_gt(Gv::ZERO), Gv::ONE / width, Gv::ZERO);
        let vkey = format!("prim_v{coord_n}");
        let flatten = |cell: i32| -> Gv {
            let vm = Gv::field_shifted(
                &vkey,
                FieldRef::PrimVel(coord_n as u8),
                ndim,
                dir,
                cell - 1,
            );
            let vp = Gv::field_shifted(
                &vkey,
                FieldRef::PrimVel(coord_n as u8),
                ndim,
                dir,
                cell + 1,
            );
            let p0 = Gv::field_shifted("prim_pre", "prim.pre", ndim, dir, cell);
            let r0 = Gv::field_shifted("prim_rho", "prim.rho", ndim, dir, cell);
            let conv = ((vm - vp) * half).max(Gv::ZERO);
            let c = conv / (p0 / r0).sqrt();
            ((c - onset) * ramp).max(Gv::ZERO).min(Gv::ONE)
        };
        // the face's left state is cell -1's right interface, the right state
        // cell 0's left interface; each blends toward its own cell average. the
        // coefficient is the max over the cell and both sweep neighbors — the
        // cell ahead of a steepening front is where the pre-front dispersive
        // error seeds, one cell before the front's own compression registers.
        let f_m2 = flatten(-2);
        let f_m1 = flatten(-1);
        let f_0 = flatten(0);
        let f_p1 = flatten(1);
        let f_l = f_m2.max(f_m1).max(f_0);
        let f_r = f_m1.max(f_0).max(f_p1);
        let blend = |face: Gv, avg: Gv, f: Gv| face + (avg - face) * f;
        rho_lr = (
            blend(
                rho_lr.0,
                Gv::field_shifted("prim_rho", "prim.rho", ndim, dir, -1),
                f_l,
            ),
            blend(rho_lr.1, Gv::field("prim_rho", "prim.rho"), f_r),
        );
        pre_lr = (
            blend(
                pre_lr.0,
                Gv::field_shifted("prim_pre", "prim.pre", ndim, dir, -1),
                f_l,
            ),
            blend(pre_lr.1, Gv::field("prim_pre", "prim.pre"), f_r),
        );
        for (k, lr) in vel_lr.iter_mut().enumerate() {
            let key = format!("prim_v{k}");
            let avg_l =
                Gv::field_shifted(&key, FieldRef::PrimVel(k as u8), ndim, dir, -1);
            let avg_r = Gv::field(&key, FieldRef::PrimVel(k as u8));
            *lr = (blend(lr.0, avg_l, f_l), blend(lr.1, avg_r, f_r));
        }
    }

    let (vl, vr): (Vec<Gv>, Vec<Gv>) = vel_lr.into_iter().unzip();
    let vl_arr: [Gv; D] = vl.try_into().expect("D velocity components");
    let vr_arr: [Gv; D] = vr.try_into().expect("D velocity components");
    let left = Prim::<Gv, D> {
        rho: rho_lr.0,
        vel: Tensor::new(vl_arr),
        pre: pre_lr.0,
    };
    let right = Prim::<Gv, D> {
        rho: rho_lr.1,
        vel: Tensor::new(vr_arr),
        pre: pre_lr.1,
    };
    let nhat = Tensor::<Gv, D>::unit(coord_n);
    let vface = mesh_face_velocity_gv(dir);
    (left, right, nhat, vface)
}

// the D+2 conserved face-flux writes (D, S_{0..D}, nrg) for an euler-shaped Cons.
fn euler_flux_writes<const D: usize>(flux: &Cons<Gv, D>) -> Vec<(String, FieldBind, NodeId)> {
    let mut writes = vec![(
        "flux_den".to_string(),
        FieldRef::flux_den().into(),
        flux.den.node(),
    )];
    for k in 0..D {
        writes.push((
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8).into(),
            flux.mom[k].node(),
        ));
    }
    writes.push((
        "flux_nrg".to_string(),
        FieldRef::flux_nrg().into(),
        flux.nrg.node(),
    ));
    writes
}

/// trace an ideal-gas euler face flux (newtonian or relativistic) along sweep `dir` —
/// the gv single source: PLM-reconstruct (rho, every vel_k, pre) to the face, then the
/// canonical `riemann::hlle(regime, IdealGas, L, R, n_hat, 0)` (symbi-hydro). replaces
/// the hand-written `hlle_flux` / `rhd_hlle_flux` Expr builders + their per-component
/// U/F (rhd_side). the reconstruction is a Gv stencil (codegen-only); the HLLE is
/// carrier-generic physics. cartesian: ncomp == ndim == D, sweep coordinate == grid `dir`.
/// generic over the regime (both `Newtonian` and `Rhd` have `Prim<S,D>` / `Cons<S,D>`).
/// `D` is the velocity-component count (ncomp); `ndim` is the reconstruction grid (the
/// stencil shifts along grid axis `dir`); `coord_n` is the sweep coordinate (the normal
/// velocity is `vel[coord_n]`, pressure goes on momentum `coord_n`). cartesian: ndim == D,
/// coord_n == dir. cyl r-z: D = 3, ndim = 2, coord_n = axes[dir] (the swirl is the 3rd comp).
fn euler_hlle_flux_gv<const D: usize, R>(
    regime: &R,
    ndim: u8,
    dir: u8,
    coord_n: usize,
    recon: Recon,
    eos_arm: EosArm,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>)
where
    R: Regime<Gv, D, Prim = Prim<Gv, D>, Cons = Cons<Gv, D>>,
{
    begin_trace();
    // gamma comes first in the manifest on every arm (under the taub-mathews closure it is
    // bound-but-inert, which keeps the ABI uniform, exactly as theta under ppm).
    let gamma = Gv::scalar("gamma");
    let eos = super::gv_eos(eos_arm, gamma);
    // the single-source physics: reconstructed L/R primitives -> canonical HLLE.
    let (left, right, nhat, vface) = euler_reconstruct::<D>(ndim, dir, coord_n, recon, None);
    let flux = hlle(regime, &eos, &left, &right, &nhat, vface);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}

/// the adiabatic (ideal-gas newtonian euler) face flux — `euler_hlle_flux_gv` at the
/// `Newtonian` regime. replaces the cartesian `hlle_flux(.., has_energy=true)` builder.
/// cartesian: ncomp == ndim == D, sweep coordinate == grid `dir`.
pub fn adiabatic_flux_gv<const D: usize>(
    dir: u8,
    recon: Recon,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_hlle_flux_gv::<D, _>(&Newtonian, D as u8, dir, dir as usize, recon, EosArm::IdealGamma)
}

/// the cyl r-z (axisymmetric swirl) adiabatic face flux: ncomp = 3 (v_phi swirl folds
/// into KE) on a 2D (r, z) grid; the sweep coordinate is `axes[dir]` ([0, 2][dir] — grid
/// axis 1 is the z coordinate). replaces the cyl r-z `hlle_flux` Expr builder.
pub fn adiabatic_flux_cyl_rz_gv(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    let coord_n = [0usize, 2][dir as usize]; // (r, z) grid axes -> coordinates 0, 2
    euler_hlle_flux_gv::<3, _>(&Newtonian, 2, dir, coord_n, Recon::Plm, EosArm::IdealGamma)
}

/// the RHD (special-relativistic euler) face flux — `euler_hlle_flux_gv` at the `Rhd`
/// regime (relativistic U/F/wave speeds via mignone-bodo). replaces the `rhd_hlle_flux`
/// Expr builder + `rhd_side`. cartesian-only (the rhd arm bakes on the cartesian chart),
/// ncomp == ndim == D.
pub fn rhd_flux_gv<const D: usize>(
    dir: u8,
    eos_arm: EosArm,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    euler_hlle_flux_gv::<D, _>(&Rhd, D as u8, dir, dir as usize, Recon::Plm, eos_arm)
}

/// the RHD face flux on a curved spacetime — the `_schw`/`_ks` GR path. PLM-reconstruct the
/// contravariant-velocity primitive, build the in-kernel 3+1 block (gamma/gamma^{-1}, lapse, shift)
/// and the densitization measure `sqrt(det gamma)` from the metric at the swept-axis face, then run
/// `riemann::hlle_with_speeds` at the `RhdGr` regime. the emitted flux is the fully densitized
/// `sqrt(-g)[rho u^n, T^n_i, -(T^n_t + rho u^n)]`, so the godunov differences it in plain
/// coordinates, the measure already carried inside. `RhdGr` reduces to `Rhd` at identity gamma,
/// unit lapse, zero shift and unit measure. D-generic over the sweep (metric at the swept-axis
/// face, transverse coords at the centroid); baked only for a curved spacetime.
pub fn rhd_flux_gr_gv<const D: usize>(
    dir: u8,
    spacetime: Spacetime,
    coords: Coords,
    spacing: &[Spacing],
    axes: &[usize],
    rusanov: bool,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>)
where
    SchwarzschildKS<Gv>: Metric<Gv, D>,
    SchwarzschildKSCartesian<Gv>: Metric<Gv, D>,
    KerrKSCartesian<Gv>: Metric<Gv, D>,
    KerrKSCylindrical<Gv>: Metric<Gv, D>,
    SchwarzschildKSCylindrical<Gv>: Metric<Gv, D>,
    KerrKS<Gv>: Metric<Gv, D>,
{
    begin_trace();
    // `D` is the momentum/velocity DOF; the reconstruction grid is `axes.len()` — they differ for
    // the spherical swirl (DOF = 3 on a 2D (r, theta) grid, out-of-plane v_phi reconstructed along
    // the gridded sweeps like any transverse component). the sweep normal is coordinate `axes[dir]`.
    let ndim = axes.len();
    // the GR arm stays on the gamma-law closure; gamma keeps its first-in-manifest slot.
    let eos = IdealGas {
        gamma: Gv::scalar("gamma"),
    };
    let (left, right, nhat, vface) =
        euler_reconstruct::<D>(ndim as u8, dir, axes[dir as usize], Recon::Plm, None);
    // the in-kernel spatial metric + lapse at the swept-axis face, transverse gridded coordinates at
    // the cell centroid — the correct face-metric position for a `dir` sweep. the spherical metrics
    // read the gridded (r, theta) alone — gamma_{phi phi} = r^2 sin^2(theta) — so an ungridded
    // symmetry slot (the axisymmetric phi) takes zero.
    // the transverse coordinate is the cell's arithmetic midpoint: the face flux is a face
    // average over the transverse coordinate extent, whose second-order sampling point is the
    // midpoint — the same point the cell state densitizes at.
    let mid = gv_cell_midpoints(spacing, ndim);
    let x = Tensor::<Gv, D>::new(std::array::from_fn(|c| {
        if c == axes[dir as usize] {
            gv_axis_face_at(dir as usize, spacing[dir as usize], 0)
        } else {
            match axes.iter().position(|&a| a == c) {
                Some(d) => mid[d],
                None => gv_ungridded_slot(coords, c),
            }
        }
    }));
    // the ADM face block, selected by (spacetime, chart): the kerr-schild spacetime is expressed in
    // the spherical chart (SchwarzschildKS, radial shift) or the cartesian chart
    // (SchwarzschildKSCartesian, non-diagonal, shift along every axis). the shift `beta` is carried
    // out for the per-axis shift term below (zero for the static schwarzschild chart).
    // `volume_factor` is sqrt(det gamma) of the full chart at any instantiated `D`, so a reduced
    // radial or equatorial block still carries the suppressed directions' measure (spherical 1D:
    // r^2/sqrt(f)); `alpha * volume_factor = sqrt(-g)` is the densitization the state and the flux
    // both ride on.
    // spinning kerr on the cartesian chart: the rank-1 kerr-schild update
    // gamma_ij = delta_ij + 2H l_i l_j with the oblate-spheroidal radius; non-diagonal
    // gamma + shift on every axis, DOF == D (the frame dragging rides the swirl of l).
    // spinning kerr on the spherical chart: non-diagonal gamma_{r phi} at the face —
    // swirl (D = 3) only.
    let (gamma, gamma_inv, alpha, beta, sqrt_gamma) = {
        fn adm<const N: usize, M: Metric<Gv, N>>(
            m: &M,
            x: Tensor<Gv, N>,
        ) -> (Matrix<Gv, N>, Matrix<Gv, N>, Gv, Tensor<Gv, N>, Gv) {
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                m.shift(x),
                m.volume_factor(x),
            )
        }
        with_ks_metric!(spacetime, coords, "the GR flux", |m| adm(&m, x))
    };
    // spinning kerr: re-reconstruct the azimuthal velocity in the angular-momentum-carrying
    // variable w = v^phi + (gamma_{r phi} / gamma_{phi phi}) v^r, so a zero-angular-momentum
    // (S_phi = 0) state — whose frame-dragging v^phi exactly cancels against v^r in the covariant
    // lowering — reconstructs to a face pair that still cancels: S_phi(face) = E gamma_{phi phi} w
    // exactly, and w = 0 to roundoff for dragging states. reconstructing v^phi raw mixes the
    // geometric dragging profile into the limited slopes and generates S_phi at truncation level.
    // the per-offset coefficient q = gamma_{r phi}/gamma_{phi phi} is evaluated at each stencil
    // cell's arithmetic midpoint — the exact position the c2p inverted the metric at, so the
    // cell-wise cancellation transfers to the stencil values at roundoff; the face coefficient
    // comes from the face matrices the riemann states lower with. gamma_{r phi} vanishes for
    // every other background, so this block is kerr-only — and spherical-swirl-only: the cartesian
    // kerr chart has DOF == D and carries cartesian velocity components, so it reconstructs the
    // raw v^i (the dragging profile enters the limited slopes at truncation level, which
    // converges away).
    let (left, right) = if spacetime == Spacetime::KerrKS && coords == Coords::Spherical {
        assert!(D == 3, "the kerr flux carries the swirl DOF");
        let mass = Gv::scalar("schwarzschild_mass");
        let spin = Gv::scalar("kerr_spin");
        // q at the arithmetic midpoint of the cell `off` steps along the sweep axis; the
        // transverse coordinate sits at this cell's midpoint (the stencil shifts one axis only).
        let half = Gv::from_f64(0.5);
        let q_at = |off: i32| -> Gv {
            let shifted_mid = |ax: usize| {
                (gv_axis_face_at(ax, spacing[ax], off as i64)
                    + gv_axis_face_at(ax, spacing[ax], off as i64 + 1))
                    * half
            };
            let (r_c, th_c) = if dir == 0 {
                (shifted_mid(0), mid[1])
            } else {
                (mid[0], shifted_mid(1))
            };
            let m = KerrKS { mass, spin };
            let gm_c = <KerrKS<Gv> as Metric<Gv, 3>>::spatial_metric(
                &m,
                Tensor::<Gv, 3>::new([r_c, th_c, Gv::ZERO]),
            );
            gm_c[(0, 2)] / gm_c[(2, 2)]
        };
        let theta_lim = Gv::scalar("theta");
        let stencil = |off: i32| -> Gv {
            let vr = Gv::field_shifted("prim_v0", FieldRef::PrimVel(0), ndim as u8, dir, off);
            let vp = Gv::field_shifted("prim_v2", FieldRef::PrimVel(2), ndim as u8, dir, off);
            vp + q_at(off) * vr
        };
        let (w_l, w_r) =
            plm_theta_from_stencil(stencil(-2), stencil(-1), stencil(0), stencil(1), theta_lim);
        // back to v^phi with the face coefficient — the same matrices the riemann states lower
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
    let regime = RhdGr {
        metric: SpatialMetric::new(Gamma::new(gamma), GammaInv::new(gamma_inv)),
        alpha,
        shift: beta,
        sqrt_gamma,
    };
    let coord_n = axes[dir as usize];
    // rusanov / local lax-friedrichs mode (the FOFC first-order fallback): the light-cone speeds
    // s = +/- alpha sqrt(gamma^{nn}) - beta^n — the state-independent maximal signal bound in
    // coordinate form, matching the shift the flux carries. every fluid characteristic lies inside
    // the light cone, so the bound holds from above right up to the boundary of the physical set;
    // the low-order update keeps the conserved state inside the physical cone.
    let (s_l, s_r) = if rusanov {
        let lam = alpha * regime.metric.gamma_inv.diag(coord_n).sqrt();
        let beta_n = beta[coord_n];
        (Gv::ZERO - lam - beta_n, lam - beta_n)
    } else {
        regime.extremal_speeds(&eos, &left, &right, &nhat)
    };
    // one HLL fan on the densitized pair (U, F^n): both sides carry the one measure sqrt(-g), the
    // shift rides inside F^n, and the signal speeds are the coordinate speeds lambda^n - beta^n,
    // so the fan is complete in coordinate form — every component keeps that single sqrt(-g)
    // weight, and the chart enters through the densitized pair alone. in the bake, mesh motion
    // (vface) pairs with flat spacetime alone.
    let flux = hlle_with_speeds(&regime, &eos, &left, &right, &nhat, vface, s_l, s_r);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}

/// the isothermal face flux — iso-native through the gv path. traces the iso physics
/// directly at the iso shape: U/F carry `(den, mom_k)`, wave speeds use `cs = sqrt(pre / rho)`
/// with prim.pre carrying the locally-isothermal `cs^2(x) * rho` — exactly the substrate's
/// locally-isothermal trick, with the graph built from those nodes alone. matches the
/// type-system claim ([[isothermal.rs]]: "zero-overhead isothermal hydrodynamics via
/// the energy model type system") at the gv-trace layer too. ncomp == ndim == D, sweep
/// coordinate == grid `dir` (cartesian).
pub fn iso_flux_gv<const D: usize>(dir: u8) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    iso_hlle_flux_gv::<D>(D as u8, dir, dir as usize)
}

/// build the iso HLLE face flux directly using Gv ops, at the iso shape (the generic
/// `riemann::hlle` carries adiabatic-shaped `Cons<S,D>`). HLLE the
/// algorithm is regime-generic; this is iso-shaped from the first node: U = (den, mom),
/// F = (rho*vn, rho*vn*vel + p*nhat), cs = sqrt(p/rho). ndim is the reconstruction grid
/// (stencil shifts along grid axis `dir`); `coord_n` is the sweep coordinate (normal
/// velocity is `vel[coord_n]`, pressure goes on momentum `coord_n`). cartesian: ndim ==
/// D, coord_n == dir.
fn iso_hlle_flux_gv<const D: usize>(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // theta is the whole scalar tail — the isothermal closure is set by cs alone. the
    // substrate's dispatch_flux passes ISO_GAMMA, and `scalars_for` walks the kernel's
    // manifest asking for the declared scalars, so leaving gamma out here leaves it out
    // of the manifest cleanly.
    let theta = Gv::scalar("theta");

    // primitives at the face: rho, each velocity component, and pre (= cs^2(x) * rho
    // via the substrate's locally-isothermal encoding; the per-cell cs(x) is whatever
    // c2p put into prim.pre).
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl: Vec<Gv> = Vec::with_capacity(D);
    let mut vr: Vec<Gv> = Vec::with_capacity(D);
    for k in 0..D {
        let (l, r) = plm_theta_gv(
            &format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8),
            ndim,
            dir,
            theta,
        );
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim, dir, theta);

    // iso conserved + flux (the algebra IsoNewtonian writes at Rust level, traced here
    // as Gv ops — the state is (den, mom), so the graph holds that arithmetic alone).
    // normal-velocity shorthands keep the writes-expression-tree small.
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
        || {
            Gv::branch(
                s_r.cmp_le(vface),
                || f_r_den - rho_r * vface,
                || den_hll - den_u_hll * vface,
            )
        },
    );

    let mut mom_flux: Vec<Gv> = Vec::with_capacity(D);
    for k in 0..D {
        let mom_hll =
            (f_l_mom[k] * s_r - f_r_mom[k] * s_l + (u_r_mom[k] - u_l_mom[k]) * (s_l * s_r)) * inv;
        let mom_u_hll = (u_r_mom[k] * s_r - u_l_mom[k] * s_l - f_r_mom[k] + f_l_mom[k]) * inv;
        let mk = Gv::branch(
            s_l.cmp_ge(vface),
            || f_l_mom[k] - u_l_mom[k] * vface,
            || {
                Gv::branch(
                    s_r.cmp_le(vface),
                    || f_r_mom[k] - u_r_mom[k] * vface,
                    || mom_hll - mom_u_hll * vface,
                )
            },
        );
        mom_flux.push(mk);
    }

    let mut writes = vec![(
        "flux_den".to_string(),
        FieldRef::flux_den().into(),
        den_flux.node(),
    )];
    for k in 0..D {
        writes.push((
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8).into(),
            mom_flux[k].node(),
        ));
    }
    (end_trace(), writes)
}

/// trace the RMHD (relativistic MHD) face flux along sweep `dir` on an `ndim`-grid — the
/// gv single source: theta-MC PLM-reconstruct (rho, vel_{0,1,2}, pre, mag_{0,1,2}) to the
/// face, then `riemann::hlle(Rmhd, IdealGas, L, R, n_hat, 0)` (symbi-hydro — the quartic
/// wave speeds + induction flux, all S::select-traceable). replaces the `rmhd_hlle_flux`
/// Expr builder + `lower_rmhd_side`. RMHD vectors are 3-component on every grid; `ndim` selects the
/// reconstruction grid + emit loop. writes the 8 conserved fluxes (D, S_k, tau, B_k).
pub fn rmhd_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // scalar params in the substrate order: gamma (EOS) then theta (limiter compression).
    let gamma = Gv::scalar("gamma");
    let theta = Gv::scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv("prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            &format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8),
            ndim,
            dir,
            theta,
        );
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim, dir, theta);
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            &format!("prim_b{k}"),
            &format!("prim.mag[{k}]"),
            ndim,
            dir,
            theta,
        );
        bl.push(l);
        br.push(r);
    }

    // the single-source physics: reconstructed L/R MHD primitives -> canonical HLLE.
    let eos = IdealGas { gamma };
    let mk = |rho: Gv, v: &[Gv], p: Gv, b: &[Gv]| MhdPrim::<Gv, 3> {
        hydro: Prim {
            rho,
            vel: Tensor::new([v[0], v[1], v[2]]),
            pre: p,
        },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    // normal B from the staggered face field (gardiner-stone CT coupling) — reconstructed
    // bcell gives bn_l != bn_r, breaking the constant-Bn assumption. see nmhd_reconstruct.
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let left = mk(rho_l, &vl, pre_l, &bl);
    let right = mk(rho_r, &vr, pre_r, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);

    // the wave speeds are materialized once per cell by rmhd_wave_speeds_cell_gv into
    // wave_speed_l[dir]/wave_speed_r[dir] (the exact quartic) and read here.
    // the HLL fan is the cell-centered davis estimate over the two cells sharing this face:
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

    let mut writes = vec![(
        "flux_den".to_string(),
        FieldRef::flux_den().into(),
        flux.den.node(),
    )];
    for k in 0..3 {
        writes.push((
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8).into(),
            flux.mom[k].node(),
        ));
    }
    writes.push((
        "flux_nrg".to_string(),
        FieldRef::flux_nrg().into(),
        flux.nrg.node(),
    ));
    for k in 0..3 {
        writes.push((
            format!("flux_mag_{k}"),
            format!("flux.mag_{k}").into(),
            flux.mag[k].node(),
        ));
    }

    // the smem tile: reconstruction is 1D along `dir`, so the
    // tile is a thin slab — halo on axis `dir`, transverse axes at their native extent.
    // the tiled set is derived from the graph (the shifted `LoadAt` fields: the 8
    // reconstructed prim + the 2 per-cell wave speeds), computed automatically from the trace.
    let k = end_trace();
    let stencil_keys = k.stencil_read_field_keys();
    if stencil_keys.is_empty() {
        return (k, writes);
    }
    let mut halo = vec![0u8; ndim as usize];
    halo[dir as usize] = 2;
    let k = k.with_tile_spec(TileSpec {
        halo,
        tiled_field_keys: stencil_keys,
    });
    (k, writes)
}

/// the RMHD face flux on a curved spatial metric — the GRMHD path (valencia covariant U/F via
/// `RmhdGr` + the fast-magnetosonic-bound coordinate wave speeds). PLM-reconstruct the 8 MHD
/// primitives (the normal B from the staggered face field, gardiner-stone), build the in-kernel
/// `SpatialMetric` + lapse at the swept-axis face, and run the HLL fan at the `RmhdGr` regime.
/// wave speeds are inline (the bound is closed-form), so the GR path computes them in the flux
/// kernel where the flat kernel reads a materialized per-cell quartic. on the kerr-schild chart
/// the fan carries the radial shift exactly like the RHD GR flux — with the induction transpose
/// term: the true mag-row flux
/// is `(alpha v^n - beta^n) B^i - (alpha v^i - beta^i) B^n`, so beyond the uniform
/// `-(beta^n/alpha) U` subtraction every mag row i gains `+(beta^i/alpha) B^n` (the radial row of
/// a transverse sweep included; B^n is the shared face field, so the term is side-symmetric).
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
    rusanov: bool,
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
        let (l, r) = plm_theta_gv(
            &format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8),
            ndim as u8,
            dir,
            theta_lim,
        );
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv("prim_pre", "prim.pre", ndim as u8, dir, theta_lim);
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            &format!("prim_b{k}"),
            &format!("prim.mag[{k}]"),
            ndim as u8,
            dir,
            theta_lim,
        );
        bl.push(l);
        br.push(r);
    }
    // normal B from the staggered face field (gardiner-stone CT coupling) — shared by both sides.
    let bn_face = Gv::field_shifted("bface_n", "bface_n", ndim as u8, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let eos = IdealGas { gamma: gamma_eos };
    let mk = |rho: Gv, v: &[Gv], p: Gv, b: &[Gv]| MhdPrim::<Gv, 3> {
        hydro: Prim {
            rho,
            vel: Tensor::new([v[0], v[1], v[2]]),
            pre: p,
        },
        mag: Tensor::new([b[0], b[1], b[2]]),
    };
    let left = mk(rho_l, &vl, pre_l, &bl);
    let right = mk(rho_r, &vr, pre_r, &br);
    let nhat = Tensor::<Gv, 3>::unit(coord_n);

    // the metric at the swept-axis face, transverse gridded slots at the cell centroid; the
    // ungridded polar slot is the exact equatorial pi/2, the azimuthal slot zero.
    let geo = (ndim > 1).then(|| cell_geometry_gv(coords, spacing, axes, ndim));
    let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
        if c == coord_n {
            gv_axis_face_at(dir as usize, spacing[dir as usize], 0)
        } else {
            match axes.iter().position(|&a| a == c) {
                Some(d) => {
                    geo.as_ref()
                        .expect("a transverse gridded axis implies ndim > 1")
                        .centroid[d]
                }
                None => gv_ungridded_slot(coords, c),
            }
        }
    }));
    // spinning kerr on the cartesian chart: the rank-1 kerr-schild update with the
    // oblate-spheroidal radius; non-diagonal gamma + shift on every axis. every spinning
    // kerr chart (ingoing kerr-schild) carries a non-diagonal gamma_{r phi} (the tetrad
    // handles it) and a radial shift (the moving-interface fan handles it); the flux is
    // otherwise metric-generic. the azimuthal momentum (swirl DOF) carries the frame dragging.
    let (gamma, gamma_inv, alpha, beta) = {
        fn adm<M: Metric<Gv, 3>>(
            m: &M,
            x: Tensor<Gv, 3>,
        ) -> (Matrix<Gv, 3>, Matrix<Gv, 3>, Gv, Tensor<Gv, 3>) {
            (
                m.spatial_metric(x),
                m.spatial_metric_inv(x),
                m.lapse(x),
                m.shift(x),
            )
        }
        with_ks_metric!(spacetime, coords, "the GRMHD flux", |m| adm(&m, x))
    };
    // spinning kerr: re-reconstruct the azimuthal velocity in the angular-momentum-carrying variable
    // w = v^phi + (gamma_{r phi}/gamma_{phi phi}) v^r, so a zero-angular-momentum (S_phi = 0) dragging
    // state reconstructs to a face pair that still cancels (S_phi(face) = (rho h W^2 + b^2) gamma_pp w,
    // exact to roundoff); reconstructing v^phi raw mixes the geometric dragging profile into the
    // limited slopes and generates S_phi at truncation level. mirrors the RHD GR flux. kerr-only
    // (gamma_{r phi} vanishes elsewhere). the per-offset q is at each cell's volume-weighted centroid
    // (the c2p metric point), the face q from the face matrix the riemann states lower with.
    // spherical-swirl-only: the cartesian kerr chart reconstructs the raw v^i (cartesian components).
    let (left, right) = if spacetime == Spacetime::KerrKS && coords == Coords::Spherical {
        let mass = Gv::scalar("schwarzschild_mass");
        let spin = Gv::scalar("kerr_spin");
        let geo_c = cell_geometry_gv(coords, spacing, axes, ndim);
        let q_at = |off: i32| -> Gv {
            let (r_c, th_c) = if dir == 0 {
                let rl = gv_axis_face_at(0, spacing[0], off as i64);
                let rh = gv_axis_face_at(0, spacing[0], off as i64 + 1);
                (
                    // the same centroid text `cell_geometry_gv` evaluates -- the c2p
                    // inverted the metric there, and the dragging cancellation
                    // transfers only at the bit-identical position.
                    symbi_geometry::volume_weighted_centroid(
                        symbi_geometry::Geometry::Spherical,
                        0,
                        rl,
                        rh,
                    ),
                    geo_c.centroid[1],
                )
            } else {
                let tl = gv_axis_face_at(1, spacing[1], off as i64);
                let th = gv_axis_face_at(1, spacing[1], off as i64 + 1);
                (
                    geo_c.centroid[0],
                    symbi_geometry::volume_weighted_centroid(
                        symbi_geometry::Geometry::Spherical,
                        1,
                        tl,
                        th,
                    ),
                )
            };
            let gm_c = <KerrKS<Gv> as Metric<Gv, 3>>::spatial_metric(
                &KerrKS { mass, spin },
                Tensor::<Gv, 3>::new([r_c, th_c, Gv::ZERO]),
            );
            gm_c[(0, 2)] / gm_c[(2, 2)]
        };
        let stencil = |off: i32| -> Gv {
            let vr = Gv::field_shifted("prim_v0", FieldRef::PrimVel(0), ndim as u8, dir, off);
            let vp = Gv::field_shifted("prim_v2", FieldRef::PrimVel(2), ndim as u8, dir, off);
            vp + q_at(off) * vr
        };
        let (w_l, w_r) =
            plm_theta_from_stencil(stencil(-2), stencil(-1), stencil(0), stencil(1), theta_lim);
        let q_face = gamma[(0, 2)] / gamma[(2, 2)];
        let mut lv = left;
        let mut rv = right;
        lv.hydro.vel[2] = w_l - q_face * lv.hydro.vel[0];
        rv.hydro.vel[2] = w_r - q_face * rv.hydro.vel[0];
        (lv, rv)
    } else {
        (left, right)
    };
    let regime = RmhdGr {
        metric: SpatialMetric::new(Gamma::new(gamma), GammaInv::new(gamma_inv)),
        alpha,
    };
    let has_shift = matches!(spacetime, Spacetime::SchwarzschildKS | Spacetime::KerrKS);
    // GR HLLD (the orthonormal-frame MUB09 fan): the spatial metric maps (via the tetrad) to the
    // local orthonormal frame where the validated flat solver runs, and the intercell flux maps back
    // exactly. a shifted chart (kerr-schild / kerr) rides the shift as the moving-interface speed
    // vface = beta^n/alpha, so the fan is evaluated at x/t = beta^n/alpha and the kernel flux is
    // F* - (beta^n/alpha) U* (the godunov re-applies alpha). the induction equation carries one more
    // shift term, the transpose +(beta^i/alpha) B^n; B^n is single-valued at the face, so it is a
    // constant added to the magnetic flux after the fan (identical to adding it to both sides).
    // covariant (killing) energy flux: F_ehat/alpha = alpha F_tau + (alpha-1) F_D - beta^i F_{S_i},
    // the linear re-split of the valencia numerical fluxes into the free-index-down energy current,
    // in the face convention the RHD arm stores: the godunov works in the flat coordinate
    // measure (sqrt(gm) = sqrt(gm_flat)/alpha, static) and re-applies one cell lapse to the energy
    // divergence like every other conserved law, so the face slot owes a 1/alpha — a fully
    // self-contained sqrt(-g) current stored here would leave a spurious energy source
    // f_ehat * d_n(ln alpha) on any chart with a varying lapse. both HLLD and HLLE emit the
    // valencia flux in the shifted-G "godunov re-applies alpha" convention
    // (F_X = F_X* - (beta^n/alpha) X*). alpha=1, beta=0 -> F_tau (the flat valencia energy flux).
    let covariant_nrg = |f: &symbi_hydro::MhdCons<Gv, 3>| {
        alpha * f.nrg + (alpha - Gv::ONE) * f.den - beta.dot(&f.mom)
    };
    if hlld && !rusanov {
        let w = if has_shift {
            beta[coord_n] / alpha
        } else {
            Gv::ZERO
        };
        let mut flux = hlld_rmhd_gr_ortho(&eos, &left, &right, coord_n, w, &regime.metric);
        if has_shift {
            flux.mag = flux.mag + beta.scale(bn_face / alpha);
        }
        flux.hydro.nrg = covariant_nrg(&flux);
        let mut writes = vec![(
            "flux_den".to_string(),
            FieldRef::flux_den().into(),
            flux.den.node(),
        )];
        for k in 0..3 {
            writes.push((
                format!("flux_mom_{k}"),
                FieldRef::flux_mom(k as u8).into(),
                flux.mom[k].node(),
            ));
        }
        writes.push((
            "flux_nrg".to_string(),
            FieldRef::flux_nrg().into(),
            flux.nrg.node(),
        ));
        for k in 0..3 {
            writes.push((
                format!("flux_mag_{k}"),
                format!("flux.mag_{k}").into(),
                flux.mag[k].node(),
            ));
        }
        let k = end_trace();
        let stencil_keys = k.stencil_read_field_keys();
        if stencil_keys.is_empty() {
            return (k, writes);
        }
        let mut halo = vec![0u8; ndim];
        halo[dir as usize] = 2;
        let k = k.with_tile_spec(TileSpec {
            halo,
            tiled_field_keys: stencil_keys,
        });
        return (k, writes);
    }
    // the HLL fan speeds. rusanov / local lax-friedrichs (the FOFC first-order fallback):
    // the light-cone speeds s = +/- alpha sqrt(gamma^{nn}) — the state-independent maximal
    // signal bound (the shift is applied by the has_shift fan below); provably
    // admissibility-preserving because it bounds every characteristic from above right up
    // to the boundary of the physical set. otherwise: the per-cell exact-quartic speeds
    // materialized by the wave-speed cell kernel, davis min/max over the two cells sharing
    // this face — the same one-computation-many-consumers layout as the flat flux. the
    // stored values are coordinate speeds (the writing kernel subtracts beta^d at each
    // cell), so the face shift is added back to restore the frame speeds this fan
    // consumes; the relativistic zero-clamp then pins the stationary state inside the fan
    // (stored speeds are raw).
    let (s_l, s_r) = if rusanov {
        let lam = alpha * regime.metric.gamma_inv.diag(coord_n).sqrt();
        (Gv::ZERO - lam, lam)
    } else {
        let lo = format!("wave_speed_l[{dir}]");
        let hi = format!("wave_speed_r[{dir}]");
        let wsl_m1 = Gv::field_shifted("ws_l", &lo, ndim as u8, dir, -1);
        let wsl_0 = Gv::field_shifted("ws_l", &lo, ndim as u8, dir, 0);
        let wsr_m1 = Gv::field_shifted("ws_r", &hi, ndim as u8, dir, -1);
        let wsr_0 = Gv::field_shifted("ws_r", &hi, ndim as u8, dir, 0);
        let beta_n = beta[coord_n];
        (
            (wsl_m1.min(wsl_0) + beta_n).min(Gv::ZERO),
            (wsr_m1.max(wsr_0) + beta_n).max(Gv::ZERO),
        )
    };
    let mut flux = if has_shift {
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
    flux.hydro.nrg = covariant_nrg(&flux);

    let mut writes = vec![(
        "flux_den".to_string(),
        FieldRef::flux_den().into(),
        flux.den.node(),
    )];
    for k in 0..3 {
        writes.push((
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8).into(),
            flux.mom[k].node(),
        ));
    }
    writes.push((
        "flux_nrg".to_string(),
        FieldRef::flux_nrg().into(),
        flux.nrg.node(),
    ));
    for k in 0..3 {
        writes.push((
            format!("flux_mag_{k}"),
            format!("flux.mag_{k}").into(),
            flux.mag[k].node(),
        ));
    }
    let k = end_trace();
    let stencil_keys = k.stencil_read_field_keys();
    if stencil_keys.is_empty() {
        return (k, writes);
    }
    let mut halo = vec![0u8; ndim];
    halo[dir as usize] = 2;
    let k = k.with_tile_spec(TileSpec {
        halo,
        tiled_field_keys: stencil_keys,
    });
    (k, writes)
}

// =============================================================================
// HLLC face flux — contact-resolving 3-wave solver, regime-specific bodies. one
// builder per regime (newtonian, RHD, RMHD) mirroring the HLLE builder shape:
// same PLM reconstruction, same scalar tail (gamma, theta), same write manifest.
// the riemann solver is the only structural difference. defaulted to the
// Standard shock-smoother arm at trace time — fleischmann lives behind
// host-time dispatch knobs the substrate has yet to expose.
// =============================================================================

/// adiabatic (ideal-gas newtonian euler) HLLC face flux. mirrors
/// `euler_hlle_flux_gv(&Newtonian, ...)` at `riemann::hllc`. carrier-generic over
/// Gv; the iso fan carries two waves, so the iso system stays on HLLE.
/// which reference mach number the low-mach ramp saturates at: the published constant, or the
/// runtime kernel scalar the clamp-free arm exposes. a tag, since a `Gv` built at the call
/// site would be built outside the trace this function opens.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MachRef {
    Published,
    Runtime,
}

/// the single adiabatic HLLC-family face flux, over the arms that actually differ.
///
/// four emitters used to spell this body verbatim, differing in a single `ShockwaveLimiter`
/// variant -- while `rhd_hllc_at_arm`, ninety lines below, already factored exactly this shape
/// for the relativistic side. the reference mach number was repeated three times with it, and
/// one copy had already deviated.
///
/// `balance` is independent of `smoother`: well-balancing is a property of the reconstruction
/// and the low-mach ramp a property of the solver, so every pairing is expressible -- including
/// the one the first-order FOFC redo needs, which is HLLE with a balanced reconstruction.
fn adiabatic_hllc_at_arm<const D: usize>(
    dir: u8,
    recon: Recon,
    smoother: ShockwaveLimiter,
    mach_ref: MachRef,
    balance: Balance,
    coords: Coords,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // built inside the trace. a `Gv` handed in as an argument is constructed at the call site,
    // before `begin_trace()` runs here, and every `Gv` op outside an active trace panics. the
    // reference mach number is therefore named by a plain tag and materialized below, which is
    // also why a tag suffices: the choice is one bit of static information.
    let mach_limit = match mach_ref {
        MachRef::Published => Gv::from_f64(symbi_hydro::dissipation::MACH_LIMIT),
        MachRef::Runtime => Gv::scalar("mach_limit"),
    };
    let eos = IdealGas {
        gamma: Gv::scalar("gamma"),
    };
    let balanced = match balance {
        Balance::Plain => None,
        Balance::Hydrostatic => Some(Balanced {
            n_bodies: symbi_ib::MAX_SOURCE_BODIES,
            coords,
            axes,
        }),
    };
    let (left, right, nhat, vface) =
        euler_reconstruct::<D>(D as u8, dir, axes[dir as usize], recon, balanced);
    let flux = hllc(&eos, &left, &right, &nhat, vface, smoother, mach_limit);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}

pub fn adiabatic_hllc_flux_gv<const D: usize>(
    dir: u8,
    recon: Recon,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    adiabatic_hllc_at_arm::<D>(dir, recon, ShockwaveLimiter::Standard, MachRef::Published, Balance::Plain, Coords::Cartesian, &[0, 1, 2][..D])
}


/// adiabatic HLLC-LM face flux as published (Fleischmann, Adami & Adams 2020): the
/// anti-diffusive star-state flux is scaled by `sin(min(1, Ma/0.1) pi/2)` on the face-normal
/// mach number, recovering classical HLLC above Ma = 0.1 and falling with the flow speed below
/// it. cures the grid-aligned shock instability together with the HLLC low-mach
/// over-dissipation, and the pressure jump enters unclamped. newtonian only (the relativistic
/// HLLC bodies use the classical star state). the `phi` helpers are fully branchless
/// (`S::select`), so the fleischmann arm traces at S = Gv just like the Standard arm.
/// adiabatic HLLC-LM face flux, the published scheme (Fleischmann, Adami & Adams 2020): the
/// sine ramp on the acoustic signal speeds, reference mach number a runtime scalar, composable
/// with the well-balanced reconstruction through the `balance` axis. the clamped variant this
/// name once carried is retired -- the balancing removes the hydrostatic residual the clamp
/// damped, and `sealed_column_unclamped` gates the pairing.
pub fn adiabatic_hllc_lm_flux_gv<const D: usize>(
    dir: u8,
    recon: Recon,
    balance: Balance,
    coords: Coords,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    adiabatic_hllc_at_arm::<D>(
        dir,
        recon,
        ShockwaveLimiter::Fleischmann,
        MachRef::Runtime,
        balance,
        coords,
        axes,
    )
}

/// the adiabatic HLLE face flux with a well-balanced reconstruction: the first-order arm the
/// FOFC redo runs. HLLE at theta = 0 is piecewise-constant, and a piecewise-constant
/// reconstruction of departures is exactly balanced -- every departure is zero, so both sides of
/// a face return the profile evaluated there and agree. the redo therefore holds a stratified
/// column that the un-balanced redo would have kicked, and the cells most likely to reach it are
/// the stagnant stratified ones at a solid wall.
/// classical HLLC with a well-balanced reconstruction: the solver-a/b partner of the
/// low-mach arm, so a sweep can flip the solver with the balance held fixed and the
/// comparison stays one-variable.
pub fn adiabatic_hllc_wb_flux_gv<const D: usize>(
    dir: u8,
    recon: Recon,
    coords: Coords,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    adiabatic_hllc_at_arm::<D>(
        dir,
        recon,
        ShockwaveLimiter::Standard,
        MachRef::Published,
        Balance::Hydrostatic,
        coords,
        axes,
    )
}

pub fn adiabatic_hlle_wb_flux_gv<const D: usize>(
    dir: u8,
    recon: Recon,
    coords: Coords,
    axes: &[usize],
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    adiabatic_hllc_at_arm::<D>(
        dir,
        recon,
        ShockwaveLimiter::Standard,
        MachRef::Published,
        Balance::Hydrostatic,
        coords,
        axes,
    )
}

/// adiabatic HLLC face flux with the acoustic-consistency dissipation scaling: identical to
/// `adiabatic_hllc_lm_flux_gv` up to the sensor. the acoustic signal speeds are scaled by how
/// much of the face data obeys the impedance relation `dp = rho c du`, where the LM arm scales
/// by the local mach number against a reference. the traced body is the same.
pub fn adiabatic_hllc_acoustic_flux_gv<const D: usize>(
    dir: u8,
    recon: Recon,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    adiabatic_hllc_at_arm::<D>(dir, recon, ShockwaveLimiter::Acoustic, MachRef::Published, Balance::Plain, Coords::Cartesian, &[0, 1, 2][..D])
}

fn rhd_hllc_at_arm<const D: usize>(
    dir: u8,
    smoother: ShockwaveLimiter,
    eos_arm: EosArm,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    // gamma keeps its first-in-manifest slot on every arm; under the taub-mathews closure
    // it is bound-but-inert, exactly as theta under ppm.
    let gamma = Gv::scalar("gamma");
    let eos = super::gv_eos(eos_arm, gamma);
    let (left, right, nhat, vface) =
        euler_reconstruct::<D>(D as u8, dir, dir as usize, Recon::Plm, None);
    let flux = hllc_rhd(&eos, &left, &right, &nhat, vface, smoother);
    let writes = euler_flux_writes(&flux);
    (end_trace(), writes)
}

/// RHD HLLC face flux — mignone-bodo (2005) quadratic for the contact speed.
/// mirrors `euler_hlle_flux_gv(&Rhd, ...)` but calls `riemann::hllc_rhd`.
pub fn rhd_hllc_flux_gv<const D: usize>(
    dir: u8,
    eos_arm: EosArm,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    rhd_hllc_at_arm::<D>(dir, ShockwaveLimiter::Standard, eos_arm)
}

/// RHD HLLC-LM face flux — the mignone-bodo star states with the acoustic dissipation scaled down
/// at low local mach number. differs from `rhd_hllc_flux_gv` only in the limiter it selects; the
/// scaling is a property of the riemann solver, so the traced kernel is the same body.
pub fn rhd_hllc_lm_flux_gv<const D: usize>(
    dir: u8,
    eos_arm: EosArm,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    rhd_hllc_at_arm::<D>(dir, ShockwaveLimiter::Fleischmann, eos_arm)
}

/// RMHD HLLC face flux — mignone-bodo (2006), null vs non-null normal B-field
/// branch. mirrors `rmhd_flux_gv` (8-component MHD primitive) but routes the
/// reconstructed L/R state through `riemann::hllc_rmhd`; `rmhd_flux_gv` routes through `hlle`.
pub fn rmhd_hllc_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hllc_rmhd(
        &Rmhd,
        &eos,
        &left,
        &right,
        &nhat,
        Gv::ZERO,
        ShockwaveLimiter::Standard,
    );
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// RMHD HLLD face flux — mignone, ugliano & bodo (2009) 5-wave solver, the
/// full magnetosonic/alfven/contact wave resolution. uses `Scalar::iterate_vec`
/// for the 15-step secant on pressure (freeze-on-converged), eagerly computes
/// HLLE as the divergence fallback, and selects via a success mask at the end.
/// shares the MHD primitive shape with HLLE/HLLC.
pub fn rmhd_hlld_flux_gv(
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (GvKernel, Vec<(String, FieldBind, NodeId)>) {
    begin_trace();
    let (eos, left, right, nhat) = nmhd_reconstruct(ndim, dir, coord_n);
    let flux = hlld_rmhd(
        &Rmhd,
        &eos,
        &left,
        &right,
        &nhat,
        Gv::ZERO,
        &SpatialMetric::flat(),
    );
    let writes = nmhd_flux_writes(&flux);
    (end_trace(), writes)
}

/// the thermodynamic face pair of a well-balanced reconstruction: each cell's departure from
/// the hydrostatic profile through it, limited by the ordinary operator, with the profile added
/// back at the face.
///
/// two anchors per face, one per side, and the duplication is load-bearing. the departure at
/// the anchor is exactly zero, so the limiter's one-sided differences about it reduce to
/// `0 - d` and `d - 0` -- the plain differences, exact in floating point. anchoring both sides
/// on one cell would leave every difference as `(q_j - c) - (q_k - c)` and forfeit the
/// gravity-free reduction. the operator is called once per anchor and that side's output is
/// the one kept.
///
/// the transform is independent of which operator consumes it, so plm and ppm share one
/// derivation. in the body-free limit every potential is exactly zero, the enthalpy ratio is
/// exactly one, and the departures are exact differences -- so this returns the plain pair
/// bit-for-bit under plm, and to within roundoff under ppm (a parabola is a weighted sum, whose
/// re-centering rounds). proved in `symbi-hydro/tests/hydrostatic_reconstruction.rs`.
fn balanced_thermo_pair(
    ndim: u8,
    dir: u8,
    recon: Recon,
    theta: Gv,
    b: Balanced<'_>,
) -> (Gv, Gv, Gv, Gv) {
    use symbi_hydro::hydrostatic::LocalEquilibrium;

    // the bake-time spacing enum is vestigial in the potential ladder: face positions come
    // from the runtime per-axis map (`map_kind_{ax}` in `gv_axis_face_at_index`), and odd
    // half-cells land on the map's own cell center (geometric mean on a log axis, arithmetic
    // midpoint otherwise) — the position `set_initial` seeds the column at, which is what
    // makes the anchor departures exactly zero on every grading.
    let spacing = vec![Spacing::Uniform; ndim as usize];
    // offsets the limiter reads, and the anchor index within them for each side of the shared
    // face. the face sits on the lower face of cell 0, which is half-cell 0; a cell center at
    // offset k is half-cell 2k+1.
    let (offsets, anchor_l, anchor_r): (&[i32], usize, usize) = match recon {
        Recon::Plm => (&[-2, -1, 0, 1], 1, 2),
        Recon::Ppm => (&[-3, -2, -1, 0, 1, 2], 2, 3),
    };
    let phi_at = |half_cells: i64| {
        crate::gv_immersed::stencil_potential_gv(
            b.n_bodies,
            b.coords,
            ndim as usize,
            dir as usize,
            b.axes,
            &spacing,
            half_cells,
        )
    };
    let phi_face = phi_at(0);
    let phi: Vec<Gv> = offsets.iter().map(|&k| phi_at(2 * k as i64 + 1)).collect();

    let read = |key: &str, f: &str| -> Vec<Gv> {
        offsets
            .iter()
            .map(|&k| Gv::field_shifted(key, f, ndim, dir, k))
            .collect()
    };
    let rho = read("prim_rho", "prim.rho");
    let pre = read("prim_pre", "prim.pre");

    // one side: departures against that side's own cell, the ordinary operator, profile back.
    // `state_at` returns both components sharing one `powf` -- the pressure exponent exceeds the
    // density exponent by exactly one, so the second transcendental is a multiply.
    let side = |anchor: usize, take_left: bool| -> (Gv, Gv) {
        let eq =
            LocalEquilibrium::through(rho[anchor], pre[anchor], phi[anchor], Gv::scalar("gamma"));
        // the single transform text -- the same function the host proof battery exercises.
        let (d_rho, d_pre) =
            symbi_hydro::hydrostatic::hydrostatic_departures(&rho, &pre, &phi, anchor, Gv::scalar("gamma"));
        let limit = |d: &[Gv]| match recon {
            Recon::Plm => crate::gv::plm_theta_from_stencil(d[0], d[1], d[2], d[3], theta),
            Recon::Ppm => crate::gv::ppm_from_stencil(d[0], d[1], d[2], d[3], d[4], d[5]),
        };
        let (base_rho, base_pre) = eq.state_at(phi_face);
        let pick = |pair: (Gv, Gv)| if take_left { pair.0 } else { pair.1 };
        (
            base_rho + pick(limit(&d_rho)),
            base_pre + pick(limit(&d_pre)),
        )
    };
    let (rho_l, pre_l) = side(anchor_l, true);
    let (rho_r, pre_r) = side(anchor_r, false);
    (rho_l, rho_r, pre_l, pre_r)
}
