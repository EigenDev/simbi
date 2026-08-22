// =============================================================================
// gv_viscous.rs
//
// the traced isothermal viscous operators: per interior cell,
// read the primitive velocity + density on the halo-1 3x3 stencil (via
// `field_offset`), evaluate the same carrier-generic `viscous_mom_update_2d` the
// f64 oracle runs, and accumulate `dt div(tau)` into `cons.mom`. 2D cartesian.
//
//   viscous_iso_gv        constant nu (a uniform viscosity stencil).
//   viscous_iso_alpha_gv  shakura-sunyaev alpha: nu(x) = alpha c_s^2 / Omega_k(r),
//                         Omega_k = sqrt(GM/r^3) about the central body — a
//                         spatially varying viscosity, face-averaged so the flux
//                         divergence stays conservative.
//
// hazard-free in place: the stencil reads are on primitive fields (read-only in
// this pass); the only write is `cons.mom` at the center cell (pointwise), so no
// cell reads a neighbor's half-updated momentum. runs post-c2p (prim current).
// no support ball — the viscous operator acts over the whole interior.
// =============================================================================

use symbi_algebra::Tensor;
use symbi_algebra::algebra::Numeric;
use symbi_geometry::{CylindricalRPhi, DiagonalMetric, Metric, Spherical};
use symbi_hydro::viscous::{
    viscous_mom_update_2d, viscous_mom_update_3d, viscous_mom_update_orthogonal_2d,
};
use symbi_ir::algebra::Scalar as _;
use symbi_ir::gv::Writes;
use symbi_ir::{FieldRef, Gv, GvKernel, begin_trace, end_trace};

use crate::coords::{Coords, Spacing};
use crate::gv::cell_geometry_gv;

fn alpha_viscosity(alpha: Gv, cs2: Gv, mass: Gv, radius: Gv) -> Gv {
    let radius = radius.abs();
    alpha * cs2 * (radius * radius * radius / mass).sqrt()
}

/// read the primitive `(velocity, density)` 3x3 stencil about the current cell.
fn prim_stencil() -> ([[Tensor<Gv, 2>; 3]; 3], [[Gv; 3]; 3]) {
    const NDIM: u8 = 2;
    let mut vst = [[Tensor::<Gv, 2>::zeros(); 3]; 3];
    let mut rst = [[Gv::ZERO; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let off = [ii as i32 - 1, jj as i32 - 1];
            let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
            let v0 = Gv::field_offset("prim_v0", FieldRef::PrimVel(0), NDIM, &off);
            let v1 = Gv::field_offset("prim_v1", FieldRef::PrimVel(1), NDIM, &off);
            vst[jj][ii] = Tensor::new([v0, v1]);
            rst[jj][ii] = rho;
        }
    }
    (vst, rst)
}

// on every newtonian chart prim.vel / cons.mom store physical (orthonormal)
// components — the r-phi inertial source (m_phi v_phi / r), the CFL's
// physical-width crossing rate, and the keplerian-disk balance v_phi =
// sqrt(GM/r) all carry that convention. the orthogonal stress carrier consumes
// physical components, so the stored stencil feeds it directly: scaling by the
// metric h (reading the storage as coordinate-contravariant v^i) shifts the
// shear null from v_phi = Omega r (rigid rotation) to v_phi = const (a sheared
// profile) — an O(1) spurious torque on every rotating disk. (the contravariant
// storage law belongs to the GR valencia path only.)

fn accumulate_mom(dmom: Tensor<Gv, 2>) -> Writes {
    let mom0_c = Gv::field("mom0", FieldRef::cons_mom(0));
    let mom1_c = Gv::field("mom1", FieldRef::cons_mom(1));
    vec![
        (
            "mom_out_0".to_string(),
            FieldRef::cons_mom(0).into(),
            (mom0_c + dmom[0]).node(),
        ),
        (
            "mom_out_1".to_string(),
            FieldRef::cons_mom(1).into(),
            (mom1_c + dmom[1]).node(),
        ),
    ]
}

/// trace the constant-nu isothermal viscous operator, 2D cartesian.
pub fn viscous_iso_gv() -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let nu = Gv::scalar("nu");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");

    let (vst, rst) = prim_stencil();
    // constant nu: a uniform viscosity stencil (the face average 0.5(nu+nu) = nu
    // is bit-identical to a scalar).
    let nust = [[nu; 3]; 3];
    let dmom = viscous_mom_update_2d(&vst, &rst, &nust, dx, dy, dt);
    let writes = accumulate_mom(dmom);
    (end_trace(), writes)
}

/// trace the constant-nu adiabatic viscous operator, 2D cartesian: the same `dt div(tau)` momentum
/// update as `viscous_iso_gv` plus the total-energy increment `dt div(tau . v)` — the viscous energy
/// flux divergence — accumulated onto `cons.nrg`. total energy is conserved (flux form) and the
/// irreversible heating warms the gas. runs post-c2p (prim current).
pub fn viscous_adiabatic_gv() -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let nu = Gv::scalar("nu");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");

    let (vst, rst) = prim_stencil();
    let nust = [[nu; 3]; 3];
    let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_2d(&vst, &rst, &nust, dx, dy, dt);
    let mom0_c = Gv::field("mom0", FieldRef::cons_mom(0));
    let mom1_c = Gv::field("mom1", FieldRef::cons_mom(1));
    let nrg_c = Gv::field("nrg", FieldRef::cons_nrg());
    let writes = vec![
        (
            "mom_out_0".to_string(),
            FieldRef::cons_mom(0).into(),
            (mom0_c + dmom[0]).node(),
        ),
        (
            "mom_out_1".to_string(),
            FieldRef::cons_mom(1).into(),
            (mom1_c + dmom[1]).node(),
        ),
        (
            "nrg_out".to_string(),
            FieldRef::cons_nrg().into(),
            (nrg_c + dnrg).node(),
        ),
    ];
    (end_trace(), writes)
}

/// the shakura-sunyaev alpha viscosity with the local adiabatic sound speed:
/// nu(x) = alpha cs^2(x) / Omega_K(r), cs^2 = gamma p / rho read per stencil cell
/// (the isothermal alpha kernel uses the one global cs; on a varying-cs
/// gas the local read is the shakura-sunyaev prescription). Omega_K from body 0's
/// mass at the in-plane distance. cartesian 2D; carries the viscous heating.
pub fn viscous_adiabatic_alpha_gv() -> (GvKernel, Writes) {
    viscous_adiabatic_alpha_impl(false)
}

/// the DOF = 3 (2.5D magnetized-gas) variant: diffuses the out-of-plane momentum
/// too, same local-cs nu law.
pub fn viscous_adiabatic_alpha_gv_2p5d() -> (GvKernel, Writes) {
    viscous_adiabatic_alpha_impl(true)
}

fn viscous_adiabatic_alpha_impl(dof3: bool) -> (GvKernel, Writes) {
    const NDIM: u8 = 2;
    begin_trace();
    let dt = Gv::scalar("dt");
    let alpha = Gv::scalar("alpha");
    let gamma = Gv::scalar("gamma");
    let gm = Gv::scalar("body_0_mass");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");
    let bx = Gv::scalar("body_0_pos_0");
    let by = Gv::scalar("body_0_pos_1");

    let geo = cell_geometry_gv(
        Coords::Cartesian,
        &vec![Spacing::Uniform; NDIM as usize],
        &(0..NDIM as usize).collect::<Vec<_>>(),
        NDIM as usize,
    );
    let (cx, cy) = (geo.centroid[0], geo.centroid[1]);

    // per-stencil-cell nu from the local cs^2 = gamma p / rho and the keplerian
    // frequency at that cell's in-plane distance from body 0.
    let mut nust = [[Gv::ZERO; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let off = [ii as i32 - 1, jj as i32 - 1];
            let pre = Gv::field_offset("prim_pre", FieldRef::PrimPre, NDIM, &off);
            let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
            let cs2 = gamma * pre / rho;
            let x = cx + Gv::from_f64(ii as f64 - 1.0) * dx;
            let y = cy + Gv::from_f64(jj as f64 - 1.0) * dy;
            let (rx, ry) = (x - bx, y - by);
            let r = (rx * rx + ry * ry).sqrt();
            nust[jj][ii] = alpha_viscosity(alpha, cs2, gm, r);
        }
    }

    let mut writes: Writes = Vec::new();
    if dof3 {
        let (vst, rst) = prim_stencil_2p5d();
        let (dmom, dnrg) =
            symbi_hydro::viscous::viscous_update_2p5d(&vst, &rst, &nust, [dx, dy], dt);
        for c in 0..3 {
            let mom_c = Gv::field(&format!("mom{c}"), FieldRef::cons_mom(c as u8));
            writes.push((
                format!("mom_out_{c}"),
                FieldRef::cons_mom(c as u8).into(),
                (mom_c + dmom[c]).node(),
            ));
        }
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push((
            "nrg_out".to_string(),
            FieldRef::cons_nrg().into(),
            (nrg + dnrg).node(),
        ));
    } else {
        let (vst, rst) = prim_stencil();
        let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_2d(&vst, &rst, &nust, dx, dy, dt);
        for c in 0..2 {
            let mom_c = Gv::field(&format!("mom{c}"), FieldRef::cons_mom(c as u8));
            writes.push((
                format!("mom_out_{c}"),
                FieldRef::cons_mom(c as u8).into(),
                (mom_c + dmom[c]).node(),
            ));
        }
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push((
            "nrg_out".to_string(),
            FieldRef::cons_nrg().into(),
            (nrg + dnrg).node(),
        ));
    }
    (end_trace(), writes)
}

/// the scale factors `(h1, h2)` at a coordinate point, per chart (cartesian -> 1;
/// cylindrical (R, phi) -> (1, R); spherical (r, theta) -> (1, r)). the const-D
/// metric bridge mirrors the geometric-source dispatch — one metric family. shared with the covariant
/// resistive EMF (the dec codifferential is the same lame-coefficient machinery as the viscous stress).
pub(crate) fn scale_factors_at(coords: Coords, ndim: usize, x: &[Gv]) -> Vec<Gv> {
    fn run<M, const D: usize>(m: M, x: &[Gv]) -> Vec<Gv>
    where
        M: Metric<Gv, D> + DiagonalMetric<Gv, D>,
    {
        let h = m.scale_factors(Tensor::from_fn(|i| x[i]));
        (0..D).map(|i| h[i]).collect()
    }
    match (coords, ndim) {
        (Coords::Cartesian, _) => vec![Gv::ONE; ndim],
        (Coords::Cylindrical, 2) => run::<_, 2>(CylindricalRPhi, x),
        (Coords::Spherical, 2) => run::<_, 2>(Spherical, x),
        (c, d) => panic!("viscous scale_factors: unsupported (coords {c:?}, ndim {d})"),
    }
}

/// trace the constant-nu isothermal viscous operator on a general 2D orthogonal
/// chart: read the scale factors `(h1, h2)` at each stencil cell from the chart's
/// `Metric::scale_factors` (evaluated at `centroid + coordinate offset`), then run
/// the one carrier-generic orthogonal operator. `coords` is the bake-time chart —
/// cylindrical and spherical both route here (spherical `h = (1, r)` falls out for
/// free), so this single kernel subsumes the chart-specific curvilinear operators.
/// the adiabatic orthogonal viscous operator: the same scale-factor stencil as
/// the iso one, through the (momentum + heating) carrier pair — div(tau) on the
/// momenta and div(tau . u) onto the total energy. one kernel per chart; the
/// heating and the momentum share their face stresses, so the discrete work
/// telescopes and the pair conserves total energy up to the boundary flux.
pub fn viscous_adiabatic_ortho_gv(coords: Coords) -> (GvKernel, Writes) {
    viscous_adiabatic_ortho_impl(coords, None)
}

/// the adiabatic orthogonal alpha operator: nu(x) = alpha (gamma p / rho) / Omega_K
/// per stencil cell (the local sound speed), with the keplerian frequency from the
/// chart's radial coordinate (the central mass sits on the axis/origin, matching
/// the iso ortho alpha kernel's convention).
pub fn viscous_adiabatic_alpha_ortho_gv(coords: Coords) -> (GvKernel, Writes) {
    viscous_adiabatic_ortho_impl(coords, Some(()))
}

fn viscous_adiabatic_ortho_impl(coords: Coords, alpha_mode: Option<()>) -> (GvKernel, Writes) {
    const NDIM: u8 = 2;
    begin_trace();
    let dt = Gv::scalar("dt");
    let dx1 = Gv::scalar("dx_0");
    let dx2 = Gv::scalar("dx_1");

    let (vst, rst) = prim_stencil();
    let geo = cell_geometry_gv(
        coords,
        &vec![Spacing::Uniform; NDIM as usize],
        &(0..NDIM as usize).collect::<Vec<_>>(),
        NDIM as usize,
    );
    let (c0, c1) = (geo.centroid[0], geo.centroid[1]);

    let mut h1 = [[Gv::ZERO; 3]; 3];
    let mut h2 = [[Gv::ZERO; 3]; 3];
    let mut nust = [[Gv::ZERO; 3]; 3];
    for dj in 0..3usize {
        for di in 0..3usize {
            let x0 = c0 + Gv::from_f64(di as f64 - 1.0) * dx1;
            let x1 = c1 + Gv::from_f64(dj as f64 - 1.0) * dx2;
            let h = scale_factors_at(coords, NDIM as usize, &[x0, x1]);
            h1[dj][di] = h[0];
            h2[dj][di] = h[1];
            nust[dj][di] = if alpha_mode.is_some() {
                // the local cs^2 = gamma p / rho and Omega_K at this cell's radial
                // coordinate (slot 0 on both supported charts).
                let alpha = Gv::scalar("alpha");
                let gamma = Gv::scalar("gamma");
                let gm = Gv::scalar("body_0_mass");
                let off = [di as i32 - 1, dj as i32 - 1];
                let pre = Gv::field_offset("prim_pre", FieldRef::PrimPre, NDIM, &off);
                let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
                let cs2 = gamma * pre / rho;
                alpha_viscosity(alpha, cs2, gm, x0)
            } else {
                Gv::scalar("nu")
            };
        }
    }

    // physical components in, physical force out — same frame as the storage.
    let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_orthogonal_2d(
        &vst, &rst, &nust, &h1, &h2, dx1, dx2, dt,
    );
    let mut writes = accumulate_mom(dmom);
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    writes.push((
        "nrg_out".to_string(),
        FieldRef::cons_nrg().into(),
        (nrg + dnrg).node(),
    ));
    (end_trace(), writes)
}

pub fn viscous_iso_ortho_gv(coords: Coords) -> (GvKernel, Writes) {
    const NDIM: u8 = 2;
    begin_trace();
    let dt = Gv::scalar("dt");
    let nu = Gv::scalar("nu");
    let dx1 = Gv::scalar("dx_0");
    let dx2 = Gv::scalar("dx_1");

    let (vst, rst) = prim_stencil();
    let geo = cell_geometry_gv(
        coords,
        &vec![Spacing::Uniform; NDIM as usize],
        &(0..NDIM as usize).collect::<Vec<_>>(),
        NDIM as usize,
    );
    let (c0, c1) = (geo.centroid[0], geo.centroid[1]);

    let mut h1 = [[Gv::ZERO; 3]; 3];
    let mut h2 = [[Gv::ZERO; 3]; 3];
    for dj in 0..3usize {
        for di in 0..3usize {
            let x0 = c0 + Gv::from_f64(di as f64 - 1.0) * dx1;
            let x1 = c1 + Gv::from_f64(dj as f64 - 1.0) * dx2;
            let h = scale_factors_at(coords, NDIM as usize, &[x0, x1]);
            h1[dj][di] = h[0];
            h2[dj][di] = h[1];
        }
    }

    let nust = [[nu; 3]; 3];
    // physical components in, physical force out — same frame as the storage.
    let dmom = viscous_mom_update_orthogonal_2d(&vst, &rst, &nust, &h1, &h2, dx1, dx2, dt);
    let writes = accumulate_mom(dmom);
    (end_trace(), writes)
}

/// trace the shakura-sunyaev alpha operator on a general 2D orthogonal chart: the
/// same scale-factor operator as `viscous_iso_ortho_gv` but with a spatially
/// varying `nu(R) = alpha c_s^2 / Omega_k(R)`, `Omega_k = sqrt(GM/R^3)`, `R` the
/// radial coordinate `x0` (the orbital radius on both cylindrical and spherical,
/// the central mass on the axis). one alpha kernel for every curvilinear chart.
pub fn viscous_iso_alpha_ortho_gv(coords: Coords) -> (GvKernel, Writes) {
    const NDIM: u8 = 2;
    begin_trace();
    let dt = Gv::scalar("dt");
    let alpha = Gv::scalar("alpha");
    let cs = Gv::scalar("cs");
    let gm = Gv::scalar("body_0_mass");
    let dx1 = Gv::scalar("dx_0");
    let dx2 = Gv::scalar("dx_1");

    let (vst, rst) = prim_stencil();
    let geo = cell_geometry_gv(
        coords,
        &vec![Spacing::Uniform; NDIM as usize],
        &(0..NDIM as usize).collect::<Vec<_>>(),
        NDIM as usize,
    );
    let (c0, c1) = (geo.centroid[0], geo.centroid[1]);
    let cs2 = cs * cs;

    let mut h1 = [[Gv::ZERO; 3]; 3];
    let mut h2 = [[Gv::ZERO; 3]; 3];
    let mut nust = [[Gv::ZERO; 3]; 3];
    for dj in 0..3usize {
        for di in 0..3usize {
            let x0 = c0 + Gv::from_f64(di as f64 - 1.0) * dx1;
            let x1 = c1 + Gv::from_f64(dj as f64 - 1.0) * dx2;
            let h = scale_factors_at(coords, NDIM as usize, &[x0, x1]);
            h1[dj][di] = h[0];
            h2[dj][di] = h[1];
            // nu(R) = alpha cs^2 / Omega_k(R), R the radial coordinate x0.
            nust[dj][di] = alpha_viscosity(alpha, cs2, gm, x0);
        }
    }

    // physical components in, physical force out — same frame as the storage.
    let dmom = viscous_mom_update_orthogonal_2d(&vst, &rst, &nust, &h1, &h2, dx1, dx2, dt);
    let writes = accumulate_mom(dmom);
    (end_trace(), writes)
}

/// read the primitive `(velocity, density)` 3x3x3 stencil about the current cell.
fn prim_stencil_3d() -> ([[[Tensor<Gv, 3>; 3]; 3]; 3], [[[Gv; 3]; 3]; 3]) {
    const NDIM: u8 = 3;
    let mut vst = [[[Tensor::<Gv, 3>::zeros(); 3]; 3]; 3];
    let mut rst = [[[Gv::ZERO; 3]; 3]; 3];
    for kk in 0..3usize {
        for jj in 0..3usize {
            for ii in 0..3usize {
                let off = [ii as i32 - 1, jj as i32 - 1, kk as i32 - 1];
                let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
                let v0 = Gv::field_offset("prim_v0", FieldRef::PrimVel(0), NDIM, &off);
                let v1 = Gv::field_offset("prim_v1", FieldRef::PrimVel(1), NDIM, &off);
                let v2 = Gv::field_offset("prim_v2", FieldRef::PrimVel(2), NDIM, &off);
                vst[kk][jj][ii] = Tensor::new([v0, v1, v2]);
                rst[kk][jj][ii] = rho;
            }
        }
    }
    (vst, rst)
}

/// accumulate the 3D viscous increment onto cons.mom (in place, pointwise center).
fn accumulate_mom_3d(dmom: Tensor<Gv, 3>) -> Writes {
    let mom0_c = Gv::field("mom0", FieldRef::cons_mom(0));
    let mom1_c = Gv::field("mom1", FieldRef::cons_mom(1));
    let mom2_c = Gv::field("mom2", FieldRef::cons_mom(2));
    vec![
        (
            "mom_out_0".to_string(),
            FieldRef::cons_mom(0).into(),
            (mom0_c + dmom[0]).node(),
        ),
        (
            "mom_out_1".to_string(),
            FieldRef::cons_mom(1).into(),
            (mom1_c + dmom[1]).node(),
        ),
        (
            "mom_out_2".to_string(),
            FieldRef::cons_mom(2).into(),
            (mom2_c + dmom[2]).node(),
        ),
    ]
}

/// trace the constant-nu isothermal viscous operator, 3D cartesian.
pub fn viscous_iso_gv_3d() -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let nu = Gv::scalar("nu");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");
    let dz = Gv::scalar("dx_2");

    let (vst, rst) = prim_stencil_3d();
    let nust = [[[nu; 3]; 3]; 3];
    let dmom = viscous_mom_update_3d(&vst, &rst, &nust, [dx, dy, dz], dt);
    let writes = accumulate_mom_3d(dmom);
    (end_trace(), writes)
}

/// trace the constant-nu adiabatic viscous operator, 3D cartesian: the same `dt div(tau)` momentum as
/// `viscous_iso_gv_3d` plus the total-energy increment `dt div(tau . v)` onto `cons.nrg`. serves
/// adiabatic hydro and full-3D MHD alike (viscosity leaves B untouched, so the flux heats the gas with the
/// 1/2 B^2 preserved). runs post-c2p.
pub fn viscous_adiabatic_gv_3d() -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let nu = Gv::scalar("nu");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");
    let dz = Gv::scalar("dx_2");

    let (vst, rst) = prim_stencil_3d();
    let nust = [[[nu; 3]; 3]; 3];
    let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_3d(&vst, &rst, &nust, [dx, dy, dz], dt);
    let mom0 = Gv::field("mom0", FieldRef::cons_mom(0));
    let mom1 = Gv::field("mom1", FieldRef::cons_mom(1));
    let mom2 = Gv::field("mom2", FieldRef::cons_mom(2));
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    let writes = vec![
        (
            "mom_out_0".to_string(),
            FieldRef::cons_mom(0).into(),
            (mom0 + dmom[0]).node(),
        ),
        (
            "mom_out_1".to_string(),
            FieldRef::cons_mom(1).into(),
            (mom1 + dmom[1]).node(),
        ),
        (
            "mom_out_2".to_string(),
            FieldRef::cons_mom(2).into(),
            (mom2 + dmom[2]).node(),
        ),
        (
            "nrg_out".to_string(),
            FieldRef::cons_nrg().into(),
            (nrg + dnrg).node(),
        ),
    ];
    (end_trace(), writes)
}

/// read the primitive `(3-vector velocity, density)` 3x3 in-plane stencil about the current cell — the
/// DOF=3 velocity a 2.5D MHD flow carries on a 2-axis grid (the out-of-plane v_2 is real, just
/// gridless in that axis).
fn prim_stencil_2p5d() -> ([[Tensor<Gv, 3>; 3]; 3], [[Gv; 3]; 3]) {
    const NDIM: u8 = 2;
    let mut vst = [[Tensor::<Gv, 3>::zeros(); 3]; 3];
    let mut rst = [[Gv::ZERO; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let off = [ii as i32 - 1, jj as i32 - 1];
            rst[jj][ii] = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
            let v0 = Gv::field_offset("prim_v0", FieldRef::PrimVel(0), NDIM, &off);
            let v1 = Gv::field_offset("prim_v1", FieldRef::PrimVel(1), NDIM, &off);
            let v2 = Gv::field_offset("prim_v2", FieldRef::PrimVel(2), NDIM, &off);
            vst[jj][ii] = Tensor::new([v0, v1, v2]);
        }
    }
    (vst, rst)
}

/// the constant-nu 2.5D DOF-aware viscous operator (D=2 grid, DOF=3 momentum). two variants share the
/// stencil + `viscous_update_2p5d`: the isothermal twin writes the 3 momentum components; the
/// adiabatic twin also writes the total-energy heating. serves 2.5D MHD (the toroidal velocity
/// diffuses; B is untouched so the heat warms the gas).
pub fn viscous_iso_gv_2p5d() -> (GvKernel, Writes) {
    viscous_2p5d_impl(false)
}
pub fn viscous_adiabatic_gv_2p5d() -> (GvKernel, Writes) {
    viscous_2p5d_impl(true)
}
fn viscous_2p5d_impl(has_energy: bool) -> (GvKernel, Writes) {
    begin_trace();
    let dt = Gv::scalar("dt");
    let nu = Gv::scalar("nu");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");
    let (vst, rst) = prim_stencil_2p5d();
    let nust = [[nu; 3]; 3];
    let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_2p5d(&vst, &rst, &nust, [dx, dy], dt);
    let mut writes: Writes = Vec::new();
    for c in 0..3 {
        let mom_c = Gv::field(&format!("mom{c}"), FieldRef::cons_mom(c as u8));
        writes.push((
            format!("mom_out_{c}"),
            FieldRef::cons_mom(c as u8).into(),
            (mom_c + dmom[c]).node(),
        ));
    }
    if has_energy {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push((
            "nrg_out".to_string(),
            FieldRef::cons_nrg().into(),
            (nrg + dnrg).node(),
        ));
    }
    (end_trace(), writes)
}

/// trace the alpha-viscosity isothermal operator, 3D cartesian. the disk lies in
/// the x-y plane with rotation axis z, so the keplerian frequency is set by the
/// cylindrical radius `R = sqrt((x-x_body)^2 + (y-y_body)^2)`, so `Omega_k =
/// sqrt(GM/R^3)` is a function of R alone and the vertical z-offset drops out. hence `nu(x,y)
/// = alpha c_s^2 / Omega_k(R)` is z-invariant (a cylinder of constant nu about the
/// rotation axis), face-averaged so the flux divergence stays conservative.
/// the alpha-viscosity adiabatic operator, 3D cartesian: `nu(x) = alpha cs^2(x) / Omega_k(R)`
/// with the local sound speed `cs^2 = gamma p / rho` read per stencil cell, and the keplerian
/// frequency set by the cylindrical radius `R = sqrt((x-x_body)^2 + (y-y_body)^2)` about the
/// rotation axis, so `Omega_k` depends on R alone.
///
/// nu here varies with height, where the isothermal 3D twin's is z-invariant: `Omega_k` is
/// z-invariant in both, while the local `cs^2` varies with height through the stratified pressure
/// and density, so each stencil cell carries its own nu. carries the viscous heating onto the total energy, like the other adiabatic forms.
pub fn viscous_adiabatic_alpha_gv_3d() -> (GvKernel, Writes) {
    const NDIM: u8 = 3;
    begin_trace();
    let dt = Gv::scalar("dt");
    let alpha = Gv::scalar("alpha");
    let gamma = Gv::scalar("gamma");
    let gm = Gv::scalar("body_0_mass");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");
    let dz = Gv::scalar("dx_2");
    let bx = Gv::scalar("body_0_pos_0");
    let by = Gv::scalar("body_0_pos_1");

    let (vst, rst) = prim_stencil_3d();

    let geo = cell_geometry_gv(
        Coords::Cartesian,
        &vec![Spacing::Uniform; NDIM as usize],
        &(0..NDIM as usize).collect::<Vec<_>>(),
        NDIM as usize,
    );
    let (cx, cy) = (geo.centroid[0], geo.centroid[1]);

    let mut nust = [[[Gv::ZERO; 3]; 3]; 3];
    for kk in 0..3usize {
        for jj in 0..3usize {
            for ii in 0..3usize {
                let off = [ii as i32 - 1, jj as i32 - 1, kk as i32 - 1];
                let pre = Gv::field_offset("prim_pre", FieldRef::PrimPre, NDIM, &off);
                let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
                let cs2 = gamma * pre / rho;
                let x = cx + Gv::from_f64(ii as f64 - 1.0) * dx;
                let y = cy + Gv::from_f64(jj as f64 - 1.0) * dy;
                let (rx, ry) = (x - bx, y - by);
                let r = (rx * rx + ry * ry).sqrt();
                nust[kk][jj][ii] = alpha_viscosity(alpha, cs2, gm, r);
            }
        }
    }

    let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_3d(&vst, &rst, &nust, [dx, dy, dz], dt);
    let mut writes: Writes = Vec::new();
    for c in 0..3usize {
        let mom_c = Gv::field(&format!("mom{c}"), FieldRef::cons_mom(c as u8));
        writes.push((
            format!("mom_out_{c}"),
            FieldRef::cons_mom(c as u8).into(),
            (mom_c + dmom[c]).node(),
        ));
    }
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    writes.push((
        "nrg_out".to_string(),
        FieldRef::cons_nrg().into(),
        (nrg + dnrg).node(),
    ));
    (end_trace(), writes)
}

pub fn viscous_iso_alpha_gv_3d() -> (GvKernel, Writes) {
    const NDIM: u8 = 3;
    begin_trace();
    let dt = Gv::scalar("dt");
    let alpha = Gv::scalar("alpha");
    let cs = Gv::scalar("cs");
    let gm = Gv::scalar("body_0_mass");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");
    let dz = Gv::scalar("dx_2");
    let bx = Gv::scalar("body_0_pos_0");
    let by = Gv::scalar("body_0_pos_1");

    let (vst, rst) = prim_stencil_3d();

    let geo = cell_geometry_gv(
        Coords::Cartesian,
        &vec![Spacing::Uniform; NDIM as usize],
        &(0..NDIM as usize).collect::<Vec<_>>(),
        NDIM as usize,
    );
    let (cx, cy) = (geo.centroid[0], geo.centroid[1]);
    let cs2 = cs * cs;

    // nu is z-invariant (cylindrical R), so every k-slice of the stencil is equal.
    let mut nust = [[[Gv::ZERO; 3]; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let x = cx + Gv::from_f64(ii as f64 - 1.0) * dx;
            let y = cy + Gv::from_f64(jj as f64 - 1.0) * dy;
            let (rx, ry) = (x - bx, y - by);
            let r = (rx * rx + ry * ry).sqrt();
            let nu = alpha_viscosity(alpha, cs2, gm, r);
            for kk in 0..3usize {
                nust[kk][jj][ii] = nu;
            }
        }
    }

    let dmom = viscous_mom_update_3d(&vst, &rst, &nust, [dx, dy, dz], dt);
    let writes = accumulate_mom_3d(dmom);
    (end_trace(), writes)
}

/// trace the alpha-viscosity isothermal operator, 2D cartesian: `nu(x) =
/// alpha c_s^2 / Omega_k(r)`, `Omega_k = sqrt(GM/r^3)`, `r = |x - x_body|`
/// (G = 1, so GM is the central body mass). the sound speed is the constant
/// `cs` param (globally isothermal). nu vanishes toward the sink (Omega_k -> inf) — the
/// physical alpha-disk `nu ~ r^{3/2}`.
pub fn viscous_iso_alpha_gv() -> (GvKernel, Writes) {
    const NDIM: u8 = 2;
    begin_trace();
    let dt = Gv::scalar("dt");
    let alpha = Gv::scalar("alpha");
    let cs = Gv::scalar("cs");
    let gm = Gv::scalar("body_0_mass");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");
    let bx = Gv::scalar("body_0_pos_0");
    let by = Gv::scalar("body_0_pos_1");

    let (vst, rst) = prim_stencil();

    // the current cell centroid (geometry scaffold); the stencil cell at offset
    // (di, dj) is centroid + (di dx, dj dy) on a uniform cartesian grid.
    let geo = cell_geometry_gv(
        Coords::Cartesian,
        &vec![Spacing::Uniform; NDIM as usize],
        &(0..NDIM as usize).collect::<Vec<_>>(),
        NDIM as usize,
    );
    let (cx, cy) = (geo.centroid[0], geo.centroid[1]);
    let cs2 = cs * cs;

    let mut nust = [[Gv::ZERO; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let x = cx + Gv::from_f64(ii as f64 - 1.0) * dx;
            let y = cy + Gv::from_f64(jj as f64 - 1.0) * dy;
            let (rx, ry) = (x - bx, y - by);
            let r = (rx * rx + ry * ry).sqrt();
            nust[jj][ii] = alpha_viscosity(alpha, cs2, gm, r);
        }
    }

    let dmom = viscous_mom_update_2d(&vst, &rst, &nust, dx, dy, dt);
    let writes = accumulate_mom(dmom);
    (end_trace(), writes)
}

/// the 2.5D orthogonal viscous plane: the 2-axis grid + the frozen third axis
/// whose scale factor rides the in-plane coordinates. cylindrical splits into
/// the (r, phi) disk (out-of-plane z, h = (1, r, 1)) and the (r, z)
/// axisymmetric section (out-of-plane phi, h = (1, 1, r)); the spherical
/// (r, theta) meridian carries the azimuth (h = (1, r, r sin(theta))).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OrthoPlane25 {
    CylRPhi,
    CylRz,
    Sph,
}

fn ortho_25_h(plane: OrthoPlane25, x0: Gv, x1: Gv) -> [Gv; 3] {
    match plane {
        OrthoPlane25::CylRPhi => [Gv::ONE, x0, Gv::ONE],
        OrthoPlane25::CylRz => [Gv::ONE, Gv::ONE, x0],
        OrthoPlane25::Sph => [Gv::ONE, x0, x0 * x1.sin()],
    }
}

// the keplerian orbital radius for the alpha law: the cylindrical radius of
// the chart point (the central mass sits on the symmetry axis).
fn ortho_25_orbital_radius(plane: OrthoPlane25, x0: Gv, x1: Gv) -> Gv {
    match plane {
        OrthoPlane25::CylRPhi | OrthoPlane25::CylRz => x0,
        OrthoPlane25::Sph => x0 * x1.sin(),
    }
}

/// trace the 2.5D (DOF = 3 on a 2-axis grid) orthogonal viscous operator for
/// `plane`: the general scale-factor stress on all three physical momenta,
/// with the heating div(tau . u) onto the total energy when `adiabatic`.
/// `alpha` swaps the constant nu for the shakura-sunyaev law nu = alpha cs^2 /
/// Omega_K(R_cyl) (local cs^2 = gamma p / rho when adiabatic, the global `cs`
/// scalar otherwise), Omega_K about body 0's mass on the symmetry axis.
pub fn viscous_ortho_2p5d_gv(
    plane: OrthoPlane25,
    adiabatic: bool,
    alpha: bool,
) -> (GvKernel, Writes) {
    const NDIM: u8 = 2;
    let (coords, axes): (Coords, [usize; 2]) = match plane {
        OrthoPlane25::CylRPhi => (Coords::Cylindrical, [0, 1]),
        OrthoPlane25::CylRz => (Coords::Cylindrical, [0, 2]),
        OrthoPlane25::Sph => (Coords::Spherical, [0, 1]),
    };
    // the carrier orders components (in-plane-1, in-plane-2, out-of-plane);
    // storage is coordinate-indexed (r, phi, z) / (r, theta, phi). the (r, z)
    // section permutes (storage phi = slot 1 is the out-of-plane component);
    // the disk and the meridian line up with storage already.
    let perm: [usize; 3] = match plane {
        OrthoPlane25::CylRz => [0, 2, 1],
        _ => [0, 1, 2],
    };
    begin_trace();
    let dt = Gv::scalar("dt");
    let dx1 = Gv::scalar("dx_0");
    let dx2 = Gv::scalar("dx_1");
    let (vst_raw, rst) = prim_stencil_2p5d();
    let vst: [[Tensor<Gv, 3>; 3]; 3] = std::array::from_fn(|j| {
        std::array::from_fn(|i| {
            Tensor::new([
                vst_raw[j][i][perm[0]],
                vst_raw[j][i][perm[1]],
                vst_raw[j][i][perm[2]],
            ])
        })
    });
    let geo = cell_geometry_gv(
        coords,
        &vec![Spacing::Uniform; NDIM as usize],
        &axes,
        NDIM as usize,
    );
    let (c0, c1) = (geo.centroid[0], geo.centroid[1]);

    let mut h1 = [[Gv::ZERO; 3]; 3];
    let mut h2 = [[Gv::ZERO; 3]; 3];
    let mut h3 = [[Gv::ZERO; 3]; 3];
    let mut nust = [[Gv::ZERO; 3]; 3];
    for dj in 0..3usize {
        for di in 0..3usize {
            let x0 = c0 + Gv::from_f64(di as f64 - 1.0) * dx1;
            let x1 = c1 + Gv::from_f64(dj as f64 - 1.0) * dx2;
            let h = ortho_25_h(plane, x0, x1);
            h1[dj][di] = h[0];
            h2[dj][di] = h[1];
            h3[dj][di] = h[2];
            nust[dj][di] = if alpha {
                let a = Gv::scalar("alpha");
                let gm = Gv::scalar("body_0_mass");
                let cs2 = if adiabatic {
                    let gamma = Gv::scalar("gamma");
                    let off = [di as i32 - 1, dj as i32 - 1];
                    let pre = Gv::field_offset("prim_pre", FieldRef::PrimPre, NDIM, &off);
                    let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
                    gamma * pre / rho
                } else {
                    let cs = Gv::scalar("cs");
                    cs * cs
                };
                let r = ortho_25_orbital_radius(plane, x0, x1);
                alpha_viscosity(a, cs2, gm, r)
            } else {
                Gv::scalar("nu")
            };
        }
    }

    let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_orthogonal_2p5d(
        &vst, &rst, &nust, &h1, &h2, &h3, dx1, dx2, dt,
    );
    let mut writes: Writes = Vec::new();
    for c in 0..3 {
        // carrier component c lands on its storage slot perm[c].
        let slot = perm[c] as u8;
        let mom_c = Gv::field(&format!("mom{}", perm[c]), FieldRef::cons_mom(slot));
        writes.push((
            format!("mom_out_{}", perm[c]),
            FieldRef::cons_mom(slot).into(),
            (mom_c + dmom[c]).node(),
        ));
    }
    if adiabatic {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push((
            "nrg_out".to_string(),
            FieldRef::cons_nrg().into(),
            (nrg + dnrg).node(),
        ));
    }
    (end_trace(), writes)
}

/// trace the full-3D orthogonal viscous operator for the cylindrical
/// (h = (1, r, 1)) or spherical (h = (1, r, r sin(theta))) chart: the general
/// scale-factor stress + (adiabatic) heating; `alpha` as in the 2.5D twin.
pub fn viscous_ortho_3d_gv(coords: Coords, adiabatic: bool, alpha: bool) -> (GvKernel, Writes) {
    const NDIM: u8 = 3;
    assert!(
        matches!(coords, Coords::Cylindrical | Coords::Spherical),
        "viscous_ortho_3d_gv: cylindrical / spherical charts only"
    );
    begin_trace();
    let dt = Gv::scalar("dt");
    let dx: [Gv; 3] = std::array::from_fn(|a| Gv::scalar(&format!("dx_{a}")));
    let (vst, rst) = prim_stencil_3d();
    let geo = cell_geometry_gv(coords, &vec![Spacing::Uniform; 3], &[0, 1, 2], 3);
    let c: [Gv; 3] = [geo.centroid[0], geo.centroid[1], geo.centroid[2]];

    let mut h1 = [[[Gv::ZERO; 3]; 3]; 3];
    let mut h2 = [[[Gv::ZERO; 3]; 3]; 3];
    let mut h3 = [[[Gv::ZERO; 3]; 3]; 3];
    let mut nust = [[[Gv::ZERO; 3]; 3]; 3];
    for dk in 0..3usize {
        for dj in 0..3usize {
            for di in 0..3usize {
                let x0 = c[0] + Gv::from_f64(di as f64 - 1.0) * dx[0];
                let x1 = c[1] + Gv::from_f64(dj as f64 - 1.0) * dx[1];
                let (hh2, hh3, r_orb) = match coords {
                    Coords::Cylindrical => (x0, Gv::ONE, x0),
                    _ => (x0, x0 * x1.sin(), x0 * x1.sin()),
                };
                h1[dk][dj][di] = Gv::ONE;
                h2[dk][dj][di] = hh2;
                h3[dk][dj][di] = hh3;
                nust[dk][dj][di] = if alpha {
                    let a = Gv::scalar("alpha");
                    let gm = Gv::scalar("body_0_mass");
                    let cs2 = if adiabatic {
                        let gamma = Gv::scalar("gamma");
                        let off = [di as i32 - 1, dj as i32 - 1, dk as i32 - 1];
                        let pre = Gv::field_offset("prim_pre", FieldRef::PrimPre, NDIM, &off);
                        let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
                        gamma * pre / rho
                    } else {
                        let cs = Gv::scalar("cs");
                        cs * cs
                    };
                    alpha_viscosity(a, cs2, gm, r_orb)
                } else {
                    Gv::scalar("nu")
                };
            }
        }
    }

    let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_orthogonal_3d(
        &vst,
        &rst,
        &nust,
        [&h1, &h2, &h3],
        dx,
        dt,
    );
    let mut writes: Writes = Vec::new();
    for cc in 0..3 {
        let mom_c = Gv::field(&format!("mom{cc}"), FieldRef::cons_mom(cc as u8));
        writes.push((
            format!("mom_out_{cc}"),
            FieldRef::cons_mom(cc as u8).into(),
            (mom_c + dmom[cc]).node(),
        ));
    }
    if adiabatic {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push((
            "nrg_out".to_string(),
            FieldRef::cons_nrg().into(),
            (nrg + dnrg).node(),
        ));
    }
    (end_trace(), writes)
}
