// =============================================================================
// flux.rs
//
// face-flux kernel builders: plm reconstruction composed with the riemann solvers (hlle / hllc / hlld) across regimes.
// =============================================================================

use super::*;
use crate::coords::Balance;
use symbi_algebra::Matrix;
use symbi_algebra::{FaceNormal, Normalized, Physical};
use symbi_geometry::{
    KerrKS, KerrKSCartesian, KerrKSCylindrical, Metric, SchwarzschildKS, SchwarzschildKSCartesian,
    SchwarzschildKSCylindrical,
};
use symbi_hydro::RmhdGr;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rhd::RhdGr;
use symbi_hydro::spatial_metric::{Gamma, GammaInv, SpatialMetric};
use symbi_hydro::state::Valencia;
use symbi_ir::{KernelProgram, KernelWrite, KernelWrites, trace_kernel};

/// trace the newtonian-MHD face flux — PLM-reconstruct the 8-component MHD
/// primitive (rho, v_{0,1,2}, pre, B_{0,1,2}) to the face, then the canonical
/// `riemann::hlle(&NewtonianMhd, ...)`. the davis fan speeds are computed inline by
/// `hlle` from the reconstructed L/R states (the closed-form magnetosonic is cheap), so
/// this arm runs off the face states alone, while `rmhd_flux_gv` materializes its quartic
/// speeds in a per-cell field and kernel. `ndim` is the reconstruction grid; `dir`
/// the sweep axis (RMHD/NMHD are fixed 3D in the velocity/field components).
// shared NMHD face-flux reconstruction: bind gamma + theta, PLM-reconstruct the
// 8-component MHD primitive (rho, v_{0..2}, pre, B_{0..2}) to the face. runs inside the
// caller's open trace through `cx`. returns the eos + L/R primitives + the sweep normal —
// the solver (HLLE / HLLC / HLLD) is the only thing that differs.
// reconstruct the L/R MHD primitives at the `dir`-grid face. the PLM stencil shifts along
// grid axis `dir`; the normal is physical component `coord_n` (= axes[dir]; == dir for
// cartesian/identity, [0,2][dir] for cyl r-z) — nhat and the staggered normal-B override
// both index `coord_n`, while the face field is read along grid `dir`.
fn nmhd_reconstruct<'t>(
    cx: TraceCx<'t>,
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (
    IdealGas<Gv<'t>>,
    MhdPrim<Gv<'t>, 3>,
    MhdPrim<Gv<'t>, 3>,
    Normalized<Physical<Gv<'t>, 3>>,
) {
    let gamma = cx.scalar("gamma");
    let theta = cx.scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv(cx, "prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            cx,
            &format!("prim_v{k}"),
            FieldRef::PrimVel(k as u8),
            ndim,
            dir,
            theta,
        );
        vl.push(l);
        vr.push(r);
    }
    let (pre_l, pre_r) = plm_theta_gv(cx, "prim_pre", "prim.pre", ndim, dir, theta);
    let mut bl = Vec::with_capacity(3);
    let mut br = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            cx,
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
    let bn_face = cx.field_shifted("bface_n", FieldRef::BFaceNormal, ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let mk = |rho, v: &[_], p, b: &[_]| {
        MhdPrim::<Gv, 3>::new(
            Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1], v[2]]), Pressure(p)),
            Tensor::new([b[0], b[1], b[2]]),
        )
    };
    let left = mk(rho_l, &vl, pre_l, &bl);
    let right = mk(rho_r, &vr, pre_r, &br);
    let nhat = Normalized::axis(coord_n);
    (IdealGas { gamma }, left, right, nhat)
}

// the 8 conserved face-flux writes (D, S_{0..2}, nrg, B_{0..2}).
fn nmhd_flux_writes<'t>(flux: &MhdCons<Gv<'t>, 3>) -> KernelWrites {
    let mut writes = vec![KernelWrite::new(
        "flux_den",
        FieldRef::flux_den(),
        flux.den().node(),
    )];
    for k in 0..3 {
        writes.push(KernelWrite::new(
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8),
            flux.mom()[k].node(),
        ));
    }
    writes.push(KernelWrite::new(
        "flux_nrg",
        FieldRef::flux_nrg(),
        flux.nrg().node(),
    ));
    for k in 0..3 {
        writes.push(KernelWrite::new(
            format!("flux_mag_{k}"),
            format!("flux.mag_{k}"),
            flux.mag()[k].node(),
        ));
    }
    writes
}

pub fn nmhd_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let (eos, left, right, nhat) = nmhd_reconstruct(cx, ndim, dir, coord_n);
        let flux = hlle(&NewtonianMhd, &eos, &left, &right, &nhat, Gv::ZERO);
        nmhd_flux_writes(&flux)
    })
}

/// NMHD HLLC face flux — `hllc_newtonian` (Li 2005, contact-resolving, transverse-B
/// continuous) on the reconstructed L/R states. wave speeds inline, from the face states
/// alone (the manifest stays free of ws_l/ws_r).
pub fn nmhd_hllc_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let (eos, left, right, nhat) = nmhd_reconstruct(cx, ndim, dir, coord_n);
        let flux = hllc_newtonian(
            &eos,
            &left,
            &right,
            &nhat,
            Gv::ZERO,
            ShockwaveLimiter::Standard,
        );
        nmhd_flux_writes(&flux)
    })
}

/// NMHD HLLD face flux — `hlld_newtonian` (miyoshi-kusano 2005, full 5-wave). the
/// robust solver: the algebraic c2p + this closed-form HLLD make orszag-tang stable.
pub fn nmhd_hlld_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let (eos, left, right, nhat) = nmhd_reconstruct(cx, ndim, dir, coord_n);
        let flux = hlld_newtonian(&eos, &left, &right, &nhat, Gv::ZERO);
        nmhd_flux_writes(&flux)
    })
}

// shared isothermal face-flux reconstruction: bind cs + theta, PLM-reconstruct the
// 7-component iso-MHD primitive the isothermal system carries (rho, v_{0..2}, B_{0..2})
// to the face. the face-normal field comes from the staggered face field (bface coupling, see
// nmhd_reconstruct). returns the Isothermal eos + L/R primitives + the sweep normal.
fn imhd_reconstruct<'t>(
    cx: TraceCx<'t>,
    ndim: u8,
    dir: u8,
    coord_n: usize,
) -> (
    Isothermal<Gv<'t>>,
    IsoMhdPrim<Gv<'t>, 3>,
    IsoMhdPrim<Gv<'t>, 3>,
    Normalized<Physical<Gv<'t>, 3>>,
) {
    // the face sound speed of the isothermal closure: the mean of the two cells' cs^2, the left
    // cell at -1 and the right cell at 0 along the sweep, then the root; a uniform field gives
    // the constant exactly.
    let cs2_l = cx.field_shifted("cs2", FieldRef::IsoCs2, ndim, dir, -1);
    let cs2_r = cx.field_shifted("cs2", FieldRef::IsoCs2, ndim, dir, 0);
    let cs = (Gv::from_f64(0.5) * (cs2_l + cs2_r)).sqrt();
    let theta = cx.scalar("theta");
    let (rho_l, rho_r) = plm_theta_gv(cx, "prim_rho", "prim.rho", ndim, dir, theta);
    let mut vl = Vec::with_capacity(3);
    let mut vr = Vec::with_capacity(3);
    for k in 0..3 {
        let (l, r) = plm_theta_gv(
            cx,
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
            cx,
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
    let bn_face = cx.field_shifted("bface_n", FieldRef::BFaceNormal, ndim, dir, 0);
    bl[coord_n] = bn_face;
    br[coord_n] = bn_face;
    let mk = |rho, v: &[_], b: &[_]| {
        IsoMhdPrim::<Gv, 3>::new(
            PrimG::isothermal(Density(rho), Tensor::new([v[0], v[1], v[2]])),
            Tensor::new([b[0], b[1], b[2]]),
        )
    };
    let left = mk(rho_l, &vl, &bl);
    let right = mk(rho_r, &vr, &br);
    let nhat = Normalized::axis(coord_n);
    (Isothermal { cs }, left, right, nhat)
}

// the 7 conserved face-flux writes the isothermal system carries: D, S_{0..2}, B_{0..2}.
fn imhd_flux_writes<'t>(flux: &IsoMhdCons<Gv<'t>, 3>) -> KernelWrites {
    let mut writes = vec![KernelWrite::new(
        "flux_den",
        FieldRef::flux_den(),
        flux.den().node(),
    )];
    for k in 0..3 {
        writes.push(KernelWrite::new(
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8),
            flux.mom()[k].node(),
        ));
    }
    for k in 0..3 {
        writes.push(KernelWrite::new(
            format!("flux_mag_{k}"),
            format!("flux.mag_{k}"),
            flux.mag()[k].node(),
        ));
    }
    writes
}

/// isothermal-MHD HLLE face flux.
pub fn imhd_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let (eos, left, right, nhat) = imhd_reconstruct(cx, ndim, dir, coord_n);
        let flux = hlle(&IsothermalMhd, &eos, &left, &right, &nhat, Gv::ZERO);
        imhd_flux_writes(&flux)
    })
}

/// isothermal-MHD HLLD face flux — `hlld_isothermal` (mignone 2007, 3-state).
pub fn imhd_hlld_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let (eos, left, right, nhat) = imhd_reconstruct(cx, ndim, dir, coord_n);
        let flux = hlld_isothermal(&eos, &left, &right, &nhat, Gv::ZERO);
        imhd_flux_writes(&flux)
    })
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
fn mesh_face_velocity_gv<'t>(cx: TraceCx<'t>, dir: u8) -> Gv<'t> {
    let mesh_adot = cx.scalar(&MeshScalar::Adot(dir).name());
    let x_face = crate::gv::geometry::gv_axis_face_at(cx, dir as usize, Spacing::Uniform, 0);
    mesh_adot * x_face + cx.scalar(&MeshScalar::Vtrans(dir).name())
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

fn euler_reconstruct<'t, const D: usize>(
    cx: TraceCx<'t>,
    ndim: u8,
    dir: u8,
    coord_n: usize,
    recon: Recon,
    balanced: Option<Balanced<'_>>,
) -> (
    Prim<Gv<'t>, D>,
    Prim<Gv<'t>, D>,
    Normalized<Physical<Gv<'t>, D>>,
    Gv<'t>,
) {
    // theta comes second in the manifest order [gamma, theta]: the caller registers gamma
    // inside the same trace before calling here (the eos construction lives with the
    // caller so the closure can be gamma-law or taub-mathews), and theta is registered
    // on every recon arm so the scalar tail is uniform; the ppm parabola carries its
    // own monotonicity constraint, so theta stays bound-but-inert there.
    let theta = cx.scalar("theta");
    // well-balancing changes the pressure alone: the limiter acts on each cell's pressure
    // departure from the mechanical equilibrium through it, in place of the raw pressure,
    // while density and velocity take the plain reconstruction — the equilibrium density
    // is the piecewise-constant distribution and carries no correction. everything else
    // -- the velocity loop, the ppm flattening, the normal and the face velocity -- is
    // shared, which is the point. a second copy of this function silently lost the
    // flattening and hardcoded the normal to the sweep axis.
    // field-registration order is ABI. the traced manifest records fields in first-read
    // order, and every flux kernel, plain or balanced, registers [rho, v.., pre]: the
    // balanced branch touches only the pressure slot, so the manifest is one order for
    // the whole family.
    let (rho_l, rho_r) = recon_gv(cx, "prim_rho", "prim.rho", ndim, dir, theta, recon);
    let mut vl = Vec::with_capacity(D);
    let mut vr = Vec::with_capacity(D);
    for k in 0..D {
        let (l, r) = recon_gv(
            cx,
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
    let (pre_l, pre_r) = match balanced {
        Some(b) => {
            let balanced_pair = balanced_pressure_pair(cx, ndim, dir, recon, theta, b);
            let plain_pair = recon_gv(cx, "prim_pre", "prim.pre", ndim, dir, theta, recon);
            let spacing = vec![Spacing::Uniform; ndim as usize];
            let left_weight = crate::gv_immersed::surface_handover_weight_gv(
                cx,
                b.n_bodies,
                b.coords,
                ndim as usize,
                dir as usize,
                b.axes,
                &spacing,
                -1,
            );
            let right_weight = crate::gv_immersed::surface_handover_weight_gv(
                cx,
                b.n_bodies,
                b.coords,
                ndim as usize,
                dir as usize,
                b.axes,
                &spacing,
                1,
            );
            (
                balanced_pair.0 + left_weight * (plain_pair.0 - balanced_pair.0),
                balanced_pair.1 + right_weight * (plain_pair.1 - balanced_pair.1),
            )
        }
        None => recon_gv(cx, "prim_pre", "prim.pre", ndim, dir, theta, recon),
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
        let onset = cx.scalar("flatten_onset");
        let full = cx.scalar("flatten_full");
        let half = Gv::from_f64(0.5);
        let width = full - onset;
        let ramp = Gv::select(width.cmp_gt(Gv::ZERO), Gv::ONE / width, Gv::ZERO);
        let vkey = format!("prim_v{coord_n}");
        let flatten = |cell: i32| {
            let vm = cx.field_shifted(&vkey, FieldRef::PrimVel(coord_n as u8), ndim, dir, cell - 1);
            let vp = cx.field_shifted(&vkey, FieldRef::PrimVel(coord_n as u8), ndim, dir, cell + 1);
            let p0 = cx.field_shifted("prim_pre", "prim.pre", ndim, dir, cell);
            let r0 = cx.field_shifted("prim_rho", "prim.rho", ndim, dir, cell);
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
        let blend = |face, avg, f| face + (avg - face) * f;
        rho_lr = (
            blend(
                rho_lr.0,
                cx.field_shifted("prim_rho", "prim.rho", ndim, dir, -1),
                f_l,
            ),
            blend(rho_lr.1, cx.field("prim_rho", "prim.rho"), f_r),
        );
        pre_lr = (
            blend(
                pre_lr.0,
                cx.field_shifted("prim_pre", "prim.pre", ndim, dir, -1),
                f_l,
            ),
            blend(pre_lr.1, cx.field("prim_pre", "prim.pre"), f_r),
        );
        for (k, lr) in vel_lr.iter_mut().enumerate() {
            let key = format!("prim_v{k}");
            let avg_l = cx.field_shifted(&key, FieldRef::PrimVel(k as u8), ndim, dir, -1);
            let avg_r = cx.field(&key, FieldRef::PrimVel(k as u8));
            *lr = (blend(lr.0, avg_l, f_l), blend(lr.1, avg_r, f_r));
        }
    }

    let (vl, vr): (Vec<Gv>, Vec<Gv>) = vel_lr.into_iter().unzip();
    let vl_arr: [Gv; D] = vl.try_into().expect("D velocity components");
    let vr_arr: [Gv; D] = vr.try_into().expect("D velocity components");
    let left = Prim::<Gv, D>::adiabatic(Density(rho_lr.0), Tensor::new(vl_arr), Pressure(pre_lr.0));
    let right =
        Prim::<Gv, D>::adiabatic(Density(rho_lr.1), Tensor::new(vr_arr), Pressure(pre_lr.1));
    let nhat = Normalized::axis(coord_n);
    let vface = mesh_face_velocity_gv(cx, dir);
    (left, right, nhat, vface)
}

// the D+2 conserved face-flux writes (D, S_{0..D}, nrg) for an euler-shaped Cons.
fn euler_flux_writes<'t, const D: usize>(flux: &Cons<Gv<'t>, D>) -> KernelWrites {
    let mut writes = vec![KernelWrite::new(
        "flux_den",
        FieldRef::flux_den(),
        flux.den().node(),
    )];
    for k in 0..D {
        writes.push(KernelWrite::new(
            format!("flux_mom_{k}"),
            FieldRef::flux_mom(k as u8),
            flux.mom()[k].node(),
        ));
    }
    writes.push(KernelWrite::new(
        "flux_nrg",
        FieldRef::flux_nrg(),
        flux.nrg().node(),
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
) -> KernelProgram
where
    R: for<'t> Regime<
            Gv<'t>,
            D,
            Prim = Prim<Gv<'t>, D>,
            Cons = Cons<Gv<'t>, D>,
            Normal = Normalized<Physical<Gv<'t>, D>>,
            Energy = symbi_hydro::energy::Adiabatic,
        >,
{
    trace_kernel(|cx| {
        // gamma comes first in the manifest on every arm (under the taub-mathews closure it is
        // bound-but-inert, which keeps the ABI uniform, exactly as theta under ppm).
        let gamma = cx.scalar("gamma");
        let eos = super::gv_eos(eos_arm, gamma);
        // the single-source physics: reconstructed L/R primitives -> canonical HLLE.
        let (left, right, nhat, vface) =
            euler_reconstruct::<D>(cx, ndim, dir, coord_n, recon, None);
        let flux = hlle(regime, &eos, &left, &right, &nhat, vface);
        euler_flux_writes(&flux)
    })
}

/// the adiabatic (ideal-gas newtonian euler) face flux — `euler_hlle_flux_gv` at the
/// `Newtonian` regime. replaces the cartesian `hlle_flux(.., has_energy=true)` builder.
/// cartesian: ncomp == ndim == D, sweep coordinate == grid `dir`.
pub fn adiabatic_flux_gv<const D: usize>(dir: u8, recon: Recon) -> KernelProgram {
    euler_hlle_flux_gv::<D, _>(
        &Newtonian,
        D as u8,
        dir,
        dir as usize,
        recon,
        EosArm::IdealGamma,
    )
}

/// the cyl r-z (axisymmetric swirl) adiabatic face flux: ncomp = 3 (v_phi swirl folds
/// into KE) on a 2D (r, z) grid; the sweep coordinate is `axes[dir]` ([0, 2][dir] — grid
/// axis 1 is the z coordinate). replaces the cyl r-z `hlle_flux` Expr builder.
pub fn adiabatic_flux_cyl_rz_gv(dir: u8) -> KernelProgram {
    let coord_n = [0usize, 2][dir as usize]; // (r, z) grid axes -> coordinates 0, 2
    euler_hlle_flux_gv::<3, _>(&Newtonian, 2, dir, coord_n, Recon::Plm, EosArm::IdealGamma)
}

/// the RHD (special-relativistic euler) face flux — `euler_hlle_flux_gv` at the `Rhd`
/// regime (relativistic U/F/wave speeds via mignone-bodo). replaces the `rhd_hlle_flux`
/// Expr builder + `rhd_side`. cartesian-only (the rhd arm bakes on the cartesian chart),
/// ncomp == ndim == D.
pub fn rhd_flux_gv<const D: usize>(dir: u8, eos_arm: EosArm) -> KernelProgram {
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
) -> KernelProgram
where
    for<'t> SchwarzschildKS<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> SchwarzschildKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> KerrKSCartesian<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> KerrKSCylindrical<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> SchwarzschildKSCylindrical<Gv<'t>>: Metric<Gv<'t>, D>,
    for<'t> KerrKS<Gv<'t>>: Metric<Gv<'t>, D>,
{
    trace_kernel(|cx| {
        // `D` is the momentum/velocity DOF; the reconstruction grid is `axes.len()` — they differ for
        // the spherical swirl (DOF = 3 on a 2D (r, theta) grid, out-of-plane v_phi reconstructed along
        // the gridded sweeps like any transverse component). the sweep normal is coordinate `axes[dir]`.
        let ndim = axes.len();
        // the GR arm stays on the gamma-law closure; gamma keeps its first-in-manifest slot.
        let eos = IdealGas {
            gamma: cx.scalar("gamma"),
        };
        let (left, right, _nhat, vface) =
            euler_reconstruct::<D>(cx, ndim as u8, dir, axes[dir as usize], Recon::Plm, None);
        // the in-kernel spatial metric + lapse at the swept-axis face, transverse gridded coordinates at
        // the cell centroid — the correct face-metric position for a `dir` sweep. the spherical metrics
        // read the gridded (r, theta) alone — gamma_{phi phi} = r^2 sin^2(theta) — so an ungridded
        // symmetry slot (the axisymmetric phi) takes zero.
        // the transverse coordinate is the cell's arithmetic midpoint: the face flux is a face
        // average over the transverse coordinate extent, whose second-order sampling point is the
        // midpoint — the same point the cell state densitizes at.
        let mid = gv_cell_midpoints(cx, spacing, ndim);
        let x = Tensor::<Gv, D>::new(std::array::from_fn(|c| {
            if c == axes[dir as usize] {
                gv_axis_face_at(cx, dir as usize, spacing[dir as usize], 0)
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
            fn adm<'t, const N: usize, M: Metric<Gv<'t>, N>>(
                m: &M,
                x: Tensor<Gv<'t>, N>,
            ) -> (
                Matrix<Gv<'t>, N>,
                Matrix<Gv<'t>, N>,
                Gv<'t>,
                Tensor<Gv<'t>, N>,
                Gv<'t>,
            ) {
                (
                    m.spatial_metric(x),
                    m.spatial_metric_inv(x),
                    m.lapse(x),
                    m.shift(x),
                    m.volume_factor(x),
                )
            }
            with_ks_metric!(cx, spacetime, coords, "the GR flux", |m| adm(&m, x))
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
            let mass = cx.scalar("schwarzschild_mass");
            let spin = cx.scalar("kerr_spin");
            // q at the arithmetic midpoint of the cell `off` steps along the sweep axis; the
            // transverse coordinate sits at this cell's midpoint (the stencil shifts one axis only).
            let half = Gv::from_f64(0.5);
            let q_at = |off: i32| {
                let shifted_mid = |ax: usize| {
                    (gv_axis_face_at(cx, ax, spacing[ax], off as i64)
                        + gv_axis_face_at(cx, ax, spacing[ax], off as i64 + 1))
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
            let theta_lim = cx.scalar("theta");
            let stencil = |off: i32| {
                let vr = cx.field_shifted("prim_v0", FieldRef::PrimVel(0), ndim as u8, dir, off);
                let vp = cx.field_shifted("prim_v2", FieldRef::PrimVel(2), ndim as u8, dir, off);
                vp + q_at(off) * vr
            };
            let (w_l, w_r) =
                plm_theta_from_stencil(stencil(-2), stencil(-1), stencil(0), stencil(1), theta_lim);
            // back to v^phi with the face coefficient — the same matrices the riemann states lower
            // with, so the face cancellation is exact to roundoff.
            let q_face = gamma[(0, 2)] / gamma[(2, 2)];
            let mut lv = left;
            let mut rv = right;
            let w_face_l = w_l - q_face * lv.vel()[0];
            let w_face_r = w_r - q_face * rv.vel()[0];
            lv.vel_mut()[2] = w_face_l;
            rv.vel_mut()[2] = w_face_r;
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
        // the valencia regime contracts its normal against contravariant
        // velocity and shift: the same axis, witnessed in the coordinate frame.
        let nhat_gr = Normalized::axis(coord_n);
        // on a curved background the reconstructed velocity components are the
        // stored valencia v^i; the wrap states that frame at the regime door.
        let (left, right) = (Valencia(left), Valencia(right));
        let (s_l, s_r) = if rusanov {
            let lam = alpha * regime.metric.gamma_inv.diag(coord_n).sqrt();
            let beta_n = beta[coord_n];
            (Gv::ZERO - lam - beta_n, lam - beta_n)
        } else {
            regime.extremal_speeds(&eos, &left, &right, &nhat_gr)
        };
        // one HLL fan on the densitized pair (U, F^n): both sides carry the one measure sqrt(-g), the
        // shift rides inside F^n, and the signal speeds are the coordinate speeds lambda^n - beta^n,
        // so the fan is complete in coordinate form — every component keeps that single sqrt(-g)
        // weight, and the chart enters through the densitized pair alone. in the bake, mesh motion
        // (vface) pairs with flat spacetime alone.
        let flux = hlle_with_speeds(&regime, &eos, &left, &right, &nhat_gr, vface, s_l, s_r);
        let writes = euler_flux_writes(&flux.0);
        writes
    })
}

/// the isothermal face flux — iso-native through the gv path. traces the iso physics
/// directly at the iso shape: U/F carry `(den, mom_k)`, wave speeds use `cs = sqrt(pre / rho)`
/// with prim.pre carrying the locally-isothermal `cs^2(x) * rho` — exactly the substrate's
/// locally-isothermal trick, with the graph built from those nodes alone. matches the
/// type-system claim ([[isothermal.rs]]: "zero-overhead isothermal hydrodynamics via
/// the energy model type system") at the gv-trace layer too. ncomp == ndim == D, sweep
/// coordinate == grid `dir` (cartesian).
pub fn iso_flux_gv<const D: usize>(dir: u8) -> KernelProgram {
    iso_hlle_flux_gv::<D>(D as u8, dir, dir as usize)
}

/// build the iso HLLE face flux directly using Gv ops, at the iso shape (the generic
/// `riemann::hlle` carries adiabatic-shaped `Cons<S,D>`). HLLE the
/// algorithm is regime-generic; this is iso-shaped from the first node: U = (den, mom),
/// F = (rho*vn, rho*vn*vel + p*nhat), cs = sqrt(p/rho). ndim is the reconstruction grid
/// (stencil shifts along grid axis `dir`); `coord_n` is the sweep coordinate (normal
/// velocity is `vel[coord_n]`, pressure goes on momentum `coord_n`). cartesian: ndim ==
/// D, coord_n == dir.
fn iso_hlle_flux_gv<const D: usize>(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        // theta is the whole scalar tail — the isothermal closure is set by cs alone. the
        // substrate's dispatch_flux passes ISO_GAMMA, and `scalars_for` walks the kernel's
        // manifest asking for the declared scalars, so leaving gamma out here leaves it out
        // of the manifest cleanly.
        let theta = cx.scalar("theta");

        // primitives at the face: rho, each velocity component, and pre (= cs^2(x) * rho
        // via the substrate's locally-isothermal encoding; the per-cell cs(x) is whatever
        // c2p put into prim.pre).
        let (rho_l, rho_r) = plm_theta_gv(cx, "prim_rho", "prim.rho", ndim, dir, theta);
        let mut vl: Vec<Gv> = Vec::with_capacity(D);
        let mut vr: Vec<Gv> = Vec::with_capacity(D);
        for k in 0..D {
            let (l, r) = plm_theta_gv(
                cx,
                &format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                ndim,
                dir,
                theta,
            );
            vl.push(l);
            vr.push(r);
        }
        let (pre_l, pre_r) = plm_theta_gv(cx, "prim_pre", "prim.pre", ndim, dir, theta);

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
        let vface = mesh_face_velocity_gv(cx, dir);
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
            let mom_hll = (f_l_mom[k] * s_r - f_r_mom[k] * s_l
                + (u_r_mom[k] - u_l_mom[k]) * (s_l * s_r))
                * inv;
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

        let mut writes = vec![KernelWrite::new(
            "flux_den",
            FieldRef::flux_den(),
            den_flux.node(),
        )];
        for k in 0..D {
            writes.push(KernelWrite::new(
                format!("flux_mom_{k}"),
                FieldRef::flux_mom(k as u8),
                mom_flux[k].node(),
            ));
        }
        writes
    })
}

/// trace the RMHD (relativistic MHD) face flux along sweep `dir` on an `ndim`-grid — the
/// gv single source: theta-MC PLM-reconstruct (rho, vel_{0,1,2}, pre, mag_{0,1,2}) to the
/// face, then `riemann::hlle(Rmhd, IdealGas, L, R, n_hat, 0)` (symbi-hydro — the quartic
/// wave speeds + induction flux, all S::select-traceable). replaces the `rmhd_hlle_flux`
/// Expr builder + `lower_rmhd_side`. RMHD vectors are 3-component on every grid; `ndim` selects the
/// reconstruction grid + emit loop. writes the 8 conserved fluxes (D, S_k, tau, B_k).
pub fn rmhd_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    let (k, writes) = trace(|cx| {
        // scalar params in the substrate order: gamma (EOS) then theta (limiter compression).
        let gamma = cx.scalar("gamma");
        let theta = cx.scalar("theta");
        let (rho_l, rho_r) = plm_theta_gv(cx, "prim_rho", "prim.rho", ndim, dir, theta);
        let mut vl = Vec::with_capacity(3);
        let mut vr = Vec::with_capacity(3);
        for k in 0..3 {
            let (l, r) = plm_theta_gv(
                cx,
                &format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                ndim,
                dir,
                theta,
            );
            vl.push(l);
            vr.push(r);
        }
        let (pre_l, pre_r) = plm_theta_gv(cx, "prim_pre", "prim.pre", ndim, dir, theta);
        let mut bl = Vec::with_capacity(3);
        let mut br = Vec::with_capacity(3);
        for k in 0..3 {
            let (l, r) = plm_theta_gv(
                cx,
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
        let mk = |rho, v: &[_], p, b: &[_]| {
            MhdPrim::<Gv, 3>::new(
                Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1], v[2]]), Pressure(p)),
                Tensor::new([b[0], b[1], b[2]]),
            )
        };
        // normal B from the staggered face field (gardiner-stone CT coupling) — reconstructed
        // bcell gives bn_l != bn_r, breaking the constant-Bn assumption. see nmhd_reconstruct.
        let bn_face = cx.field_shifted("bface_n", FieldRef::BFaceNormal, ndim, dir, 0);
        bl[coord_n] = bn_face;
        br[coord_n] = bn_face;
        let left = mk(rho_l, &vl, pre_l, &bl);
        let right = mk(rho_r, &vr, pre_r, &br);
        let nhat = Normalized::axis(coord_n);

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
        let wsl_m1 = cx.field_shifted("ws_l", &lo, dim, dir, -1);
        let wsl_0 = cx.field_shifted("ws_l", &lo, dim, dir, 0);
        let wsr_m1 = cx.field_shifted("ws_r", &hi, dim, dir, -1);
        let wsr_0 = cx.field_shifted("ws_r", &hi, dim, dir, 0);
        let s_l = wsl_m1.min(wsl_0).min(Gv::ZERO);
        let s_r = wsr_m1.max(wsr_0).max(Gv::ZERO);
        let flux = hlle_with_speeds(&Rmhd, &eos, &left, &right, &nhat, Gv::ZERO, s_l, s_r);

        let mut writes = vec![KernelWrite::new(
            "flux_den",
            FieldRef::flux_den(),
            flux.den().node(),
        )];
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("flux_mom_{k}"),
                FieldRef::flux_mom(k as u8),
                flux.mom()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "flux_nrg",
            FieldRef::flux_nrg(),
            flux.nrg().node(),
        ));
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("flux_mag_{k}"),
                format!("flux.mag_{k}"),
                flux.mag()[k].node(),
            ));
        }

        writes
    });

    // the smem tile: reconstruction is 1D along `dir`, so the
    // tile is a thin slab — halo on axis `dir`, transverse axes at their native extent.
    // the tiled set is derived from the graph (the shifted `LoadAt` fields: the 8
    // reconstructed prim + the 2 per-cell wave speeds), computed automatically from the trace.
    let stencil_keys = k.stencil_read_field_keys();
    if stencil_keys.is_empty() {
        return KernelProgram::new(k, writes);
    }
    let mut halo = vec![0u8; ndim as usize];
    halo[dir as usize] = 2;
    let k = k.with_tile_spec(TileSpec {
        halo,
        tiled_field_keys: stencil_keys,
    });
    KernelProgram::new(k, writes)
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
) -> KernelProgram {
    let ndim = axes.len();
    let (k, writes) = trace(|cx| {
        let coord_n = axes[dir as usize];
        let gamma_eos = cx.scalar("gamma");
        let theta_lim = cx.scalar("theta");
        let (rho_l, rho_r) = plm_theta_gv(cx, "prim_rho", "prim.rho", ndim as u8, dir, theta_lim);
        let mut vl = Vec::with_capacity(3);
        let mut vr = Vec::with_capacity(3);
        for k in 0..3 {
            let (l, r) = plm_theta_gv(
                cx,
                &format!("prim_v{k}"),
                FieldRef::PrimVel(k as u8),
                ndim as u8,
                dir,
                theta_lim,
            );
            vl.push(l);
            vr.push(r);
        }
        let (pre_l, pre_r) = plm_theta_gv(cx, "prim_pre", "prim.pre", ndim as u8, dir, theta_lim);
        let mut bl = Vec::with_capacity(3);
        let mut br = Vec::with_capacity(3);
        for k in 0..3 {
            let (l, r) = plm_theta_gv(
                cx,
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
        let bn_face = cx.field_shifted("bface_n", FieldRef::BFaceNormal, ndim as u8, dir, 0);
        bl[coord_n] = bn_face;
        br[coord_n] = bn_face;
        let eos = IdealGas { gamma: gamma_eos };
        let mk = |rho, v: &[_], p, b: &[_]| {
            MhdPrim::<Gv, 3>::new(
                Prim::adiabatic(Density(rho), Tensor::new([v[0], v[1], v[2]]), Pressure(p)),
                Tensor::new([b[0], b[1], b[2]]),
            )
        };
        let left = mk(rho_l, &vl, pre_l, &bl);
        let right = mk(rho_r, &vr, pre_r, &br);
        let nhat = Normalized::axis(coord_n);

        // the metric at the swept-axis face, transverse gridded slots at the cell centroid; the
        // ungridded polar slot is the exact equatorial pi/2, the azimuthal slot zero.
        let geo = (ndim > 1).then(|| cell_geometry_gv(cx, coords, spacing, axes, ndim));
        let x = Tensor::<Gv, 3>::new(std::array::from_fn(|c| {
            if c == coord_n {
                gv_axis_face_at(cx, dir as usize, spacing[dir as usize], 0)
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
            fn adm<'t, M: Metric<Gv<'t>, 3>>(
                m: &M,
                x: Tensor<Gv<'t>, 3>,
            ) -> (
                Matrix<Gv<'t>, 3>,
                Matrix<Gv<'t>, 3>,
                Gv<'t>,
                Tensor<Gv<'t>, 3>,
            ) {
                (
                    m.spatial_metric(x),
                    m.spatial_metric_inv(x),
                    m.lapse(x),
                    m.shift(x),
                )
            }
            with_ks_metric!(cx, spacetime, coords, "the GRMHD flux", |m| adm(&m, x))
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
            let mass = cx.scalar("schwarzschild_mass");
            let spin = cx.scalar("kerr_spin");
            let geo_c = cell_geometry_gv(cx, coords, spacing, axes, ndim);
            let q_at = |off: i32| {
                let (r_c, th_c) = if dir == 0 {
                    let rl = gv_axis_face_at(cx, 0, spacing[0], off as i64);
                    let rh = gv_axis_face_at(cx, 0, spacing[0], off as i64 + 1);
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
                    let tl = gv_axis_face_at(cx, 1, spacing[1], off as i64);
                    let th = gv_axis_face_at(cx, 1, spacing[1], off as i64 + 1);
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
            let stencil = |off: i32| {
                let vr = cx.field_shifted("prim_v0", FieldRef::PrimVel(0), ndim as u8, dir, off);
                let vp = cx.field_shifted("prim_v2", FieldRef::PrimVel(2), ndim as u8, dir, off);
                vp + q_at(off) * vr
            };
            let (w_l, w_r) =
                plm_theta_from_stencil(stencil(-2), stencil(-1), stencil(0), stencil(1), theta_lim);
            let q_face = gamma[(0, 2)] / gamma[(2, 2)];
            let mut lv = left;
            let mut rv = right;
            let w_face_l = w_l - q_face * lv.hydro().vel()[0];
            let w_face_r = w_r - q_face * rv.hydro().vel()[0];
            lv.hydro_mut().vel_mut()[2] = w_face_l;
            rv.hydro_mut().vel_mut()[2] = w_face_r;
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
        let covariant_nrg = |f: &symbi_hydro::MhdCons<_, 3>| {
            alpha * f.nrg() + (alpha - Gv::ONE) * f.den() - beta.dot(f.mom())
        };
        if hlld && !rusanov {
            let w = if has_shift {
                beta[coord_n] / alpha
            } else {
                Gv::ZERO
            };
            let mut flux = hlld_rmhd_gr_ortho(&eos, &left, &right, coord_n, w, &regime.metric);
            if has_shift {
                let mag_shifted = *flux.mag() + beta.scale(bn_face / alpha);
                flux = flux.with_mag(mag_shifted);
            }
            let nrg_cov = covariant_nrg(&flux);
            flux = flux.with_hydro(flux.hydro().with_nrg(nrg_cov));
            let mut writes = vec![KernelWrite::new(
                "flux_den",
                FieldRef::flux_den(),
                flux.den().node(),
            )];
            for k in 0..3 {
                writes.push(KernelWrite::new(
                    format!("flux_mom_{k}"),
                    FieldRef::flux_mom(k as u8),
                    flux.mom()[k].node(),
                ));
            }
            writes.push(KernelWrite::new(
                "flux_nrg",
                FieldRef::flux_nrg(),
                flux.nrg().node(),
            ));
            for k in 0..3 {
                writes.push(KernelWrite::new(
                    format!("flux_mag_{k}"),
                    format!("flux.mag_{k}"),
                    flux.mag()[k].node(),
                ));
            }
            return writes;
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
            let wsl_m1 = cx.field_shifted("ws_l", &lo, ndim as u8, dir, -1);
            let wsl_0 = cx.field_shifted("ws_l", &lo, ndim as u8, dir, 0);
            let wsr_m1 = cx.field_shifted("ws_r", &hi, ndim as u8, dir, -1);
            let wsr_0 = cx.field_shifted("ws_r", &hi, ndim as u8, dir, 0);
            let beta_n = beta[coord_n];
            (
                (wsl_m1.min(wsl_0) + beta_n).min(Gv::ZERO),
                (wsr_m1.max(wsr_0) + beta_n).max(Gv::ZERO),
            )
        };
        let flux = if has_shift {
            // the shifted-system HLL (the RHD GR fan) with the induction transpose add per side.
            let beta_n = beta[coord_n];
            let w = beta_n / alpha;
            let u_l = regime.to_conserved(&eos, &Valencia(left));
            let u_r = regime.to_conserved(&eos, &Valencia(right));
            let f_l = regime.to_flux(&Valencia(left), &nhat, &eos);
            let f_r = regime.to_flux(&Valencia(right), &nhat, &eos);
            let mut g_l = f_l - u_l * w;
            let mut g_r = f_r - u_r * w;
            let transpose = beta.scale(bn_face / alpha);
            let gl_mag = *g_l.0.mag() + transpose;
            let gr_mag = *g_r.0.mag() + transpose;
            g_l.0 = g_l.0.with_mag(gl_mag);
            g_r.0 = g_r.0.with_mag(gr_mag);
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
            hlle_with_speeds(
                &regime,
                &eos,
                &Valencia(left),
                &Valencia(right),
                &nhat,
                Gv::ZERO,
                s_l,
                s_r,
            )
        };
        // the fan is done with the witnessed algebra; the write boundary
        // extracts bare components into the flux buffers.
        let mut flux = flux.0;
        let nrg_cov = covariant_nrg(&flux);
        flux = flux.with_hydro(flux.hydro().with_nrg(nrg_cov));

        let mut writes = vec![KernelWrite::new(
            "flux_den",
            FieldRef::flux_den(),
            flux.den().node(),
        )];
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("flux_mom_{k}"),
                FieldRef::flux_mom(k as u8),
                flux.mom()[k].node(),
            ));
        }
        writes.push(KernelWrite::new(
            "flux_nrg",
            FieldRef::flux_nrg(),
            flux.nrg().node(),
        ));
        for k in 0..3 {
            writes.push(KernelWrite::new(
                format!("flux_mag_{k}"),
                format!("flux.mag_{k}"),
                flux.mag()[k].node(),
            ));
        }
        writes
    });

    let stencil_keys = k.stencil_read_field_keys();
    if stencil_keys.is_empty() {
        return KernelProgram::new(k, writes);
    }
    let mut halo = vec![0u8; ndim];
    halo[dir as usize] = 2;
    let k = k.with_tile_spec(TileSpec {
        halo,
        tiled_field_keys: stencil_keys,
    });
    KernelProgram::new(k, writes)
}

// =============================================================================
// HLLC face flux — contact-resolving 3-wave solver, regime-specific bodies. one
// builder per regime (newtonian, RHD, RMHD) mirroring the HLLE builder shape:
// same PLM reconstruction, same scalar tail (gamma, theta), same write manifest.
// the riemann solver is the only structural difference. defaulted to the
// Standard shock-smoother arm at trace time — fleischmann lives behind
// host-time dispatch knobs the substrate has yet to expose.
// =============================================================================

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
#[allow(clippy::too_many_arguments)]
fn adiabatic_hllc_at_arm<const D: usize>(
    dir: u8,
    recon: Recon,
    smoother: ShockwaveLimiter,
    balance: Balance,
    coords: Coords,
    axes: &[usize],
) -> KernelProgram {
    trace_kernel(|cx| {
        let eos = IdealGas {
            gamma: cx.scalar("gamma"),
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
            euler_reconstruct::<D>(cx, D as u8, dir, axes[dir as usize], recon, balanced);
        // the neighborhood pressure ratio the transverse shear viscosity is weighted by. only the
        // HLLC+ arm carries that viscosity, so only it pays the off-axis reads the sensor needs.
        let shear = matches!(smoother, ShockwaveLimiter::HllcPlus).then(|| {
            let gamma = eos.gamma;
            hllc_plus_sensors(cx, D as u8, dir, &move |rho, pre| {
                (gamma * pre / rho).sqrt()
            })
        });
        let flux = hllc(&eos, &left, &right, &nhat, vface, smoother, shear);
        let writes = euler_flux_writes(&flux);
        writes
    })
}

/// the smallest pressure ratio `min(p_a/p_b, p_b/p_a)` across any interface of either cell
/// adjoining the face at `dir`, the shock indicator the transverse shear viscosity is weighted
/// by (Chen, Lin, Li & Yan, SIAM J. Sci. Comput. 42:B921, 2020, eq. 24).
///
/// the neighborhood is what makes the indicator work. the faces the grid-aligned shock
/// instability grows through are the ones transverse to the front, and along a planar front the
/// flow is smooth — so such a face's own two states carry a pressure ratio near one and report
/// no shock at all. the structure that identifies it sits on the shock-normal interfaces of the
/// same two cells, one cell away in a transverse direction, which is why the sensor reads the
/// cross and not the face.
///
/// the face at index `c` along `dir` lies between the cells at offsets -1 and 0, matching the
/// reconstruction stencil, so the interfaces of those two cells are the three consecutive pairs
/// along the sweep together with, for every transverse axis, the pair each cell forms with its
/// two neighbors along that axis. the sweep pairs sit inside the reconstruction's own footprint;
/// the transverse pairs carry the read one cell off-axis, which is the whole of this kernel's
/// halo growth over a plain flux.
/// the stencil quantities both HLLC+ corrections read: how much pressure structure the
/// neighborhood carries, and whether a shock is present in it.
///
/// the pressure ratio alone measures how steep the pressure is across a cell, and a gas bound
/// to a point mass is steep for a reason that has nothing to do with a shock: hydrostatic
/// balance against a `1/r^2` field puts a large ratio across every cell of the atmosphere. read
/// on its own the ratio therefore turns the viscosity on throughout a stratified envelope, and
/// on a smoothly draining accretor — supersonic, unshocked, and strongly sheared near the sink
/// — a full-strength transverse viscosity drives cells out of the admissible set.
///
/// a shock is what the weight is for, and what distinguishes one is that a characteristic speed
/// reverses sign across it: gas ahead of the front outruns a wave that gas behind it cannot,
/// so `u - a` or `u + a` changes sign between neighbors. a hydrostatic gradient reverses
/// nothing, however steep. gating the ratio on that reversal leaves the viscosity at full
/// strength across a front and absent from an atmosphere.
fn hllc_plus_sensors<'t>(
    cx: TraceCx<'t>,
    ndim: u8,
    dir: u8,
    sound_speed: &dyn Fn(Gv<'t>, Gv<'t>) -> Gv<'t>,
) -> symbi_hydro::riemann::HllcPlusSensors<Gv<'t>> {
    let shocked = neighborhood_shock_indicator(cx, ndim, dir, sound_speed);
    symbi_hydro::riemann::HllcPlusSensors {
        pressure_ratio: neighborhood_pressure_ratio(cx, ndim, dir),
        shocked: Gv::select(shocked, Gv::from_f64(1.0), Gv::from_f64(0.0)),
    }
}

/// whether either cell adjoining the face at `dir` sits at a characteristic-speed reversal
/// against one of its neighbors — the shock indicator of Chen, Lin, Li & Yan (SIAM J. Sci.
/// Comput. 42:B921, 2020, eq. 16), evaluated on the acoustic characteristics `u -+ a` along
/// each axis in turn. returns the boolean mask, true where a shock is present.
///
/// `sound_speed` maps `(rho, pre)` to the regime's own sound speed, because the two regimes
/// disagree on it by the specific enthalpy and the newtonian expression evaluated on a
/// relativistic gas exceeds the speed of light. the relativistic characteristics compose the
/// two speeds as `(v -+ a)/(1 -+ v a)`, whose denominator is positive for any subluminal state,
/// so the sign the indicator reads is the sign of `v -+ a` in both regimes and one expression
/// serves each.
fn neighborhood_shock_indicator<'t>(
    cx: TraceCx<'t>,
    ndim: u8,
    dir: u8,
    sound_speed: &dyn Fn(Gv<'t>, Gv<'t>) -> Gv<'t>,
) -> symbi_ir::gv::GvMask<'t> {
    let at = |ax: usize, sweep: i32, off: i32| {
        let mut offsets = vec![0i32; ndim as usize];
        offsets[dir as usize] = sweep;
        offsets[ax] += off;
        let rho = cx.field_offset("prim_rho", FieldRef::PrimRho, ndim, &offsets);
        let pre = cx.field_offset("prim_pre", FieldRef::PrimPre, ndim, &offsets);
        let vel = cx.field_offset("prim_v0", FieldRef::PrimVel(ax as u8), ndim, &offsets);
        let cs = sound_speed(rho, pre);
        (vel - cs, vel + cs)
    };
    // seeded false: a face touches a front only if some pair below reverses.
    let mut shocked = Gv::from_f64(0.0).cmp_gt(Gv::from_f64(1.0));
    // the two cells adjoining this face, each against its neighbors along every axis. a
    // reversal on any of those pairs marks the face as touching a front.
    for ax in 0..ndim as usize {
        for sweep in [-1i32, 0] {
            let (minus_c, plus_c) = at(ax, sweep, 0);
            for off in [-1i32, 1] {
                let (minus_n, plus_n) = at(ax, sweep, off);
                let zero = Gv::from_f64(0.0);
                let slow = minus_c.cmp_gt(zero) & minus_n.cmp_lt(zero);
                let fast = plus_c.cmp_gt(zero) & plus_n.cmp_lt(zero);
                shocked = shocked | slow | fast;
            }
        }
    }
    shocked
}

fn neighborhood_pressure_ratio<'t>(cx: TraceCx<'t>, ndim: u8, dir: u8) -> Gv<'t> {
    let at = |sweep: i32, transverse: Option<(u8, i32)>| {
        let mut offsets = vec![0i32; ndim as usize];
        offsets[dir as usize] = sweep;
        if let Some((ax, off)) = transverse {
            offsets[ax as usize] = off;
        }
        cx.field_offset("prim_pre", FieldRef::PrimPre, ndim, &offsets)
    };
    // the symmetric ratio of an interface, in (0, 1] for positive pressures: one at a smooth
    // interface, falling toward zero as the jump across it strengthens, and blind to which side
    // carries the compressed gas.
    // one reciprocal, not two: for positive pressures the smaller quotient is the smaller
    // pressure over the larger, so `min(a/b, b/a)` and `min(a,b)/max(a,b)` are the same
    // division and agree bit for bit. the sensor reads eleven interfaces per face in 3d,
    // so the halving is eleven reciprocals off the baseline kernel.
    let ratio = |a: Gv<'t>, b: Gv<'t>| a.min(b) / a.max(b);

    // the three interfaces along the sweep, spanning cells -2 through +1.
    let mut weakest = ratio(at(-2, None), at(-1, None))
        .min(ratio(at(-1, None), at(0, None)))
        .min(ratio(at(0, None), at(1, None)));
    // and, for each transverse axis, the two interfaces each adjoining cell forms across it.
    for ax in 0..ndim {
        if ax == dir {
            continue;
        }
        for sweep in [-1i32, 0] {
            let center = at(sweep, None);
            weakest = weakest
                .min(ratio(center, at(sweep, Some((ax, -1)))))
                .min(ratio(center, at(sweep, Some((ax, 1)))));
        }
    }
    weakest
}

/// adiabatic HLLC+ face flux (Chen, Lin, Li & Yan, SIAM J. Sci. Comput. 42:B921, 2020):
/// classical HLLC plus two additive corrections. the first rescales the dissipation on the
/// face's normal velocity jump down to the convective magnitude, restoring the `Ma^2` scaling
/// of pressure fluctuations at low mach number; the second adds a dissipation on the transverse
/// velocity jump across a shock, which is the jump the grid-aligned shock instability grows
/// through. both saturate at the sonic point, so the arm carries no reference mach number and
/// traces no runtime scalar for one.
///
/// composable with the well-balanced reconstruction through the `balance` axis. each correction
/// carries a velocity jump as a factor, so a face in hydrostatic balance — where the two sides
/// share a velocity — sees the classical flux with its full pressure-jump dissipation, and the
/// balance and the corrections act on disjoint structure.
pub fn adiabatic_hllc_plus_flux_gv<const D: usize>(
    dir: u8,
    recon: Recon,
    balance: Balance,
    coords: Coords,
    axes: &[usize],
) -> KernelProgram {
    adiabatic_hllc_at_arm::<D>(
        dir,
        recon,
        ShockwaveLimiter::HllcPlus,
        balance,
        coords,
        axes,
    )
}

pub fn adiabatic_hllc_flux_gv<const D: usize>(dir: u8, recon: Recon) -> KernelProgram {
    adiabatic_hllc_at_arm::<D>(
        dir,
        recon,
        ShockwaveLimiter::Standard,
        Balance::Plain,
        Coords::Cartesian,
        &[0, 1, 2][..D],
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
) -> KernelProgram {
    adiabatic_hllc_at_arm::<D>(
        dir,
        recon,
        ShockwaveLimiter::Standard,
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
) -> KernelProgram {
    adiabatic_hllc_at_arm::<D>(
        dir,
        recon,
        ShockwaveLimiter::Standard,
        Balance::Hydrostatic,
        coords,
        axes,
    )
}

fn rhd_hllc_at_arm<const D: usize>(
    dir: u8,
    smoother: ShockwaveLimiter,
    eos_arm: EosArm,
) -> KernelProgram {
    trace_kernel(|cx| {
        // gamma keeps its first-in-manifest slot on every arm; under the taub-mathews closure
        // it is bound-but-inert, exactly as theta under ppm.
        let gamma = cx.scalar("gamma");
        let eos = super::gv_eos(eos_arm, gamma);
        let (left, right, nhat, vface) =
            euler_reconstruct::<D>(cx, D as u8, dir, dir as usize, Recon::Plm, None);
        let shear = matches!(smoother, ShockwaveLimiter::HllcPlus).then(|| {
            let eos = eos.clone();
            hllc_plus_sensors(cx, D as u8, dir, &move |rho, pre| {
                symbi_hydro::rhd::sound_speed_sq(&eos, rho, pre).sqrt()
            })
        });
        let flux = hllc_rhd(&eos, &left, &right, &nhat, vface, shear);
        let writes = euler_flux_writes(&flux);
        writes
    })
}

/// RHD HLLC face flux — mignone-bodo (2005) quadratic for the contact speed.
/// mirrors `euler_hlle_flux_gv(&Rhd, ...)` but calls `riemann::hllc_rhd`.
pub fn rhd_hllc_flux_gv<const D: usize>(dir: u8, eos_arm: EosArm) -> KernelProgram {
    rhd_hllc_at_arm::<D>(dir, ShockwaveLimiter::Standard, eos_arm)
}

/// relativistic HLLC+ face flux: the Mignone-Bodo star states plus the transverse shear
/// viscosity that carries shock stability. the grid-aligned shock instability grows through
/// the transverse velocity jump in the multidimensional momentum balance, which relativistic
/// jets and blast waves carry as readily as newtonian ones; the coefficient is the enthalpy
/// density `rho h W^2 = e + p` in place of the newtonian mass density.
///
/// the low-mach accuracy term of the newtonian arm stays behind: separating the velocity-jump
/// dissipation from the pressure-jump dissipation in the relativistic flux is its own
/// derivation, and the defect it corrects is a subsonic one.
pub fn rhd_hllc_plus_flux_gv<const D: usize>(dir: u8, eos_arm: EosArm) -> KernelProgram {
    rhd_hllc_at_arm::<D>(dir, ShockwaveLimiter::HllcPlus, eos_arm)
}

/// RMHD HLLC face flux — mignone-bodo (2006), null vs non-null normal B-field
/// branch. mirrors `rmhd_flux_gv` (8-component MHD primitive) but routes the
/// reconstructed L/R state through `riemann::hllc_rmhd`; `rmhd_flux_gv` routes through `hlle`.
pub fn rmhd_hllc_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let (eos, left, right, nhat) = nmhd_reconstruct(cx, ndim, dir, coord_n);
        let flux = hllc_rmhd(
            &Rmhd,
            &eos,
            &left,
            &right,
            &nhat,
            Gv::ZERO,
            ShockwaveLimiter::Standard,
        );
        nmhd_flux_writes(&flux)
    })
}

/// RMHD HLLD face flux — mignone, ugliano & bodo (2009) 5-wave solver, the
/// full magnetosonic/alfven/contact wave resolution. uses `Scalar::iterate_vec`
/// for the 15-step secant on pressure (freeze-on-converged), eagerly computes
/// HLLE as the divergence fallback, and selects via a success mask at the end.
/// shares the MHD primitive shape with HLLE/HLLC.
pub fn rmhd_hlld_flux_gv(ndim: u8, dir: u8, coord_n: usize) -> KernelProgram {
    trace_kernel(|cx| {
        let (eos, left, right, nhat) = nmhd_reconstruct(cx, ndim, dir, coord_n);
        let flux = hlld_rmhd(
            &Rmhd,
            &eos,
            &left,
            &right,
            &nhat,
            Gv::ZERO,
            &SpatialMetric::flat(),
        );
        nmhd_flux_writes(&flux)
    })
}

/// the pressure face pair of a well-balanced reconstruction: each cell's pressure
/// departure from the mechanical equilibrium through it (Kaeppeli & Mishra, A&A 587,
/// A94, 2016), limited by the ordinary operator, with the profile added back at the
/// face. density and velocity take the plain reconstruction, because the equilibrium
/// density is the piecewise-constant distribution itself and carries no correction.
///
/// two anchors per face, one per side, and the duplication is load-bearing. the
/// departure at the anchor is exactly zero, so the limiter's one-sided differences
/// about it reduce to `0 - d` and `d - 0` -- the plain differences, exact in floating
/// point. anchoring both sides on one cell would forfeit the gravity-free reduction.
///
/// the transform is independent of which operator consumes it, so plm and ppm share one
/// derivation, and it commits to no thermal structure: a discretely balanced column of
/// arbitrary entropy stratification presents identical face pressures from both anchors
/// and the flux at rest is exact. proved in
/// `symbi-hydro/tests/hydrostatic_reconstruction.rs`.
fn balanced_pressure_pair<'t>(
    cx: TraceCx<'t>,
    ndim: u8,
    dir: u8,
    recon: Recon,
    theta: Gv<'t>,
    b: Balanced<'_>,
) -> (Gv<'t>, Gv<'t>) {
    use symbi_hydro::hydrostatic::LocalEquilibrium;

    // the bake-time spacing enum is vestigial in the potential ladder: face positions come
    // from the runtime per-axis map (`map_kind_{ax}` in `gv_axis_face_at_index`), and odd
    // half-cells land on the map's own cell center (geometric mean on a log axis, arithmetic
    // midpoint otherwise) — the position `set_initial` seeds the column at, which is what
    // makes the anchor departures exactly zero on every grading.
    let spacing = vec![Spacing::Uniform; ndim as usize];
    // offsets the limiter reads, and the anchor index within them for each side of the shared
    // face. the face sits on the lower face of cell 0, which is half-cell 0; a cell center at
    // offset k is half-cell 2k+1, and the face between offsets k and k+1 is half-cell 2k+2.
    let (offsets, anchor_l, anchor_r): (&[i32], usize, usize) = match recon {
        Recon::Plm => (&[-2, -1, 0, 1], 1, 2),
        Recon::Ppm => (&[-3, -2, -1, 0, 1, 2], 2, 3),
    };
    let phi_at = |half_cells: i64| {
        crate::gv_immersed::stencil_potential_gv(
            cx,
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
    let phi_c: Vec<Gv> = offsets.iter().map(|&k| phi_at(2 * k as i64 + 1)).collect();
    // the interior faces of the stencil, where the piecewise-constant-density segments
    // switch from one cell's density to the next. analytic positions on the face ladder
    // rather than field reads, so the chain needs no ghost width of its own.
    let phi_f: Vec<Gv> = offsets[..offsets.len() - 1]
        .iter()
        .map(|&k| phi_at(2 * k as i64 + 2))
        .collect();

    let read = |key: &str, f: &str| -> Vec<Gv<'t>> {
        offsets
            .iter()
            .map(|&k| cx.field_shifted(key, f, ndim, dir, k))
            .collect()
    };
    let rho = read("prim_rho", "prim.rho");
    let pre = read("prim_pre", "prim.pre");

    // one side: pressure departures against that side's own anchor, the ordinary
    // operator, the anchor's own segment back at the face. the same segment sums are
    // what the equilibrium-pressure body source evaluates for the same cell along the
    // same axis, so the flux and the source follow one and the same profile and their
    // telescoping on an equilibrium is exact.
    //
    // the profile is weighted by how much of its positive domain the anchor's own
    // footprint spends: the potential at the two footprint endpoints, the active
    // reconstruction's own reach either side of the anchor, feeds `balance_weight`.
    // the same endpoints are what the body source reads for the same cell, so the pair
    // fades together, and a footprint that overreaches the segment degrades the whole
    // reconstruction continuously to the plain one instead of building faces on a
    // clamped equilibrium.
    let footprint = 2 * recon.balance_reach();
    let side = |anchor: usize, take_left: bool| {
        let anchor_half = 2 * offsets[anchor] as i64 + 1;
        let rise = symbi_hydro::hydrostatic::potential_rise(
            phi_c[anchor],
            phi_at(anchor_half - footprint),
            phi_at(anchor_half + footprint),
        );
        let weight = symbi_hydro::hydrostatic::balance_weight(rho[anchor], pre[anchor], rise);
        // the single transform text -- the same function the host proof battery exercises.
        let d = symbi_hydro::hydrostatic::hydrostatic_departures(
            anchor, &pre, &rho, &phi_c, &phi_f, weight,
        );
        let limit = |d: &[_]| match recon {
            Recon::Plm => crate::gv::plm_theta_from_stencil(d[0], d[1], d[2], d[3], theta),
            Recon::Ppm => crate::gv::ppm_from_stencil(d[0], d[1], d[2], d[3], d[4], d[5]),
        };
        let eq = LocalEquilibrium::faded(rho[anchor], pre[anchor], phi_c[anchor], rise);
        let pair = limit(&d);
        eq.pressure_at(phi_face) + if take_left { pair.0 } else { pair.1 }
    };
    (side(anchor_l, true), side(anchor_r, false))
}
