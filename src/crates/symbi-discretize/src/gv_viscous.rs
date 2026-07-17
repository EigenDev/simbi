// =============================================================================
// gv_viscous.rs
//
// the traced isothermal viscous operators: per interior cell,
// read the primitive velocity + density on the halo-1 3x3 stencil (via
// `field_offset`), evaluate the SAME carrier-generic `viscous_mom_update_2d` the
// f64 oracle runs, and accumulate `dt div(tau)` into `cons.mom`. 2D cartesian.
//
//   viscous_iso_gv        constant nu (a uniform viscosity stencil).
//   viscous_iso_alpha_gv  Shakura-Sunyaev alpha: nu(x) = alpha c_s^2 / Omega_k(r),
//                         Omega_k = sqrt(GM/r^3) about the central body — a
//                         SPATIALLY VARYING viscosity, face-averaged so the flux
//                         divergence stays conservative.
//
// hazard-free in place: the stencil reads are on PRIMITIVE fields (read-only in
// this pass); the only write is `cons.mom` at the CENTER cell (pointwise), so no
// cell reads a neighbour's half-updated momentum. runs post-c2p (prim current).
// no support ball — the viscous operator acts over the whole interior.
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_algebra::Tensor;
use symbi_geometry::{CylindricalRPhi, DiagonalMetric, Metric, Spherical};
use symbi_hydro::viscous::{
    viscous_mom_update_2d, viscous_mom_update_3d, viscous_mom_update_orthogonal_2d,
};
use symbi_ir::gv::Writes;
use symbi_ir::{begin_trace, end_trace, FieldRef, Gv, GvKernel};

use crate::coords::{Coords, Spacing};
use crate::gv::cell_geometry_gv;

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

// on every newtonian chart prim.vel / cons.mom store PHYSICAL (orthonormal)
// components — the r-phi inertial source (m_phi v_phi / r), the CFL's
// physical-width crossing rate, and the keplerian-disk balance v_phi =
// sqrt(GM/r) all carry that convention. the orthogonal stress carrier consumes
// physical components, so the stored stencil feeds it DIRECTLY: scaling by the
// metric h (reading the storage as coordinate-contravariant v^i) shifts the
// shear null from v_phi = Omega r (rigid rotation) to v_phi = const (a sheared
// profile) — an O(1) spurious torque on every rotating disk. (the contravariant
// storage law belongs to the GR Valencia path only.)

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

/// trace the constant-nu ADIABATIC viscous operator, 2D cartesian: the SAME `dt div(tau)` momentum
/// update as `viscous_iso_gv` PLUS the total-energy increment `dt div(tau . v)` — the viscous energy
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
        ("mom_out_0".to_string(), FieldRef::cons_mom(0).into(), (mom0_c + dmom[0]).node()),
        ("mom_out_1".to_string(), FieldRef::cons_mom(1).into(), (mom1_c + dmom[1]).node()),
        ("nrg_out".to_string(), FieldRef::cons_nrg().into(), (nrg_c + dnrg).node()),
    ];
    (end_trace(), writes)
}

/// the shakura-sunyaev alpha viscosity with the LOCAL adiabatic sound speed:
/// nu(x) = alpha cs^2(x) / Omega_K(r), cs^2 = gamma p / rho read per stencil cell
/// (the isothermal alpha kernel uses the one global cs instead; on a varying-cs
/// gas the local read IS the shakura-sunyaev prescription). Omega_K from body 0's
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
    let floor = Gv::from_f64(1e-30);

    // per-stencil-cell nu from the LOCAL cs^2 = gamma p / rho and the keplerian
    // frequency at that cell's in-plane distance from body 0.
    let mut nust = [[Gv::ZERO; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let off = [ii as i32 - 1, jj as i32 - 1];
            let pre = Gv::field_offset("prim_pre", FieldRef::PrimPre, NDIM, &off);
            let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
            let cs2 = gamma * pre / rho.max(floor);
            let x = cx + Gv::from_f64(ii as f64 - 1.0) * dx;
            let y = cy + Gv::from_f64(jj as f64 - 1.0) * dy;
            let (rx, ry) = (x - bx, y - by);
            let r = (rx * rx + ry * ry).sqrt().max(floor);
            let omega_k = (gm / (r * r * r)).sqrt().max(floor);
            nust[jj][ii] = alpha * cs2 / omega_k;
        }
    }

    let mut writes: Writes = Vec::new();
    if dof3 {
        let (vst, rst) = prim_stencil_2p5d();
        let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_2p5d(&vst, &rst, &nust, [dx, dy], dt);
        for c in 0..3 {
            let mom_c = Gv::field(&format!("mom{c}"), FieldRef::cons_mom(c as u8));
            writes.push((format!("mom_out_{c}"), FieldRef::cons_mom(c as u8).into(), (mom_c + dmom[c]).node()));
        }
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push(("nrg_out".to_string(), FieldRef::cons_nrg().into(), (nrg + dnrg).node()));
    } else {
        let (vst, rst) = prim_stencil();
        let (dmom, dnrg) = symbi_hydro::viscous::viscous_update_2d(&vst, &rst, &nust, dx, dy, dt);
        for c in 0..2 {
            let mom_c = Gv::field(&format!("mom{c}"), FieldRef::cons_mom(c as u8));
            writes.push((format!("mom_out_{c}"), FieldRef::cons_mom(c as u8).into(), (mom_c + dmom[c]).node()));
        }
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push(("nrg_out".to_string(), FieldRef::cons_nrg().into(), (nrg + dnrg).node()));
    }
    (end_trace(), writes)
}

/// the scale factors `(h1, h2)` at a coordinate point, per chart (Cartesian -> 1;
/// cylindrical (R, phi) -> (1, R); spherical (r, theta) -> (1, r)). the const-D
/// metric bridge mirrors the geometric-source dispatch — one metric family. shared with the covariant
/// resistive EMF (the DEC codifferential is the same Lamé-coefficient machinery as the viscous stress).
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

/// trace the constant-nu isothermal viscous operator on a GENERAL 2D ORTHOGONAL
/// chart: read the scale factors `(h1, h2)` at each stencil cell from the chart's
/// `Metric::scale_factors` (evaluated at `centroid + coordinate offset`), then run
/// the one carrier-generic orthogonal operator. `coords` is the bake-time chart —
/// cylindrical and spherical both route here (spherical `h = (1, r)` falls out for
/// free), so this ONE kernel subsumes the chart-specific curvilinear operators.
/// the ADIABATIC orthogonal viscous operator: the same scale-factor stencil as
/// the iso one, through the (momentum + HEATING) carrier pair — div(tau) on the
/// momenta and div(tau . u) onto the total energy. one kernel per chart; the
/// heating and the momentum share their face stresses, so the discrete work
/// telescopes and the pair conserves total energy up to the boundary flux.
pub fn viscous_adiabatic_ortho_gv(coords: Coords) -> (GvKernel, Writes) {
    viscous_adiabatic_ortho_impl(coords, None)
}

/// the ADIABATIC orthogonal ALPHA operator: nu(x) = alpha (gamma p / rho) / Omega_K
/// per stencil cell (the LOCAL sound speed), with the keplerian frequency from the
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
    let floor = Gv::from_f64(1e-30);
    for dj in 0..3usize {
        for di in 0..3usize {
            let x0 = c0 + Gv::from_f64(di as f64 - 1.0) * dx1;
            let x1 = c1 + Gv::from_f64(dj as f64 - 1.0) * dx2;
            let h = scale_factors_at(coords, NDIM as usize, &[x0, x1]);
            h1[dj][di] = h[0];
            h2[dj][di] = h[1];
            nust[dj][di] = if alpha_mode.is_some() {
                // the LOCAL cs^2 = gamma p / rho and Omega_K at this cell's radial
                // coordinate (slot 0 on both supported charts).
                let alpha = Gv::scalar("alpha");
                let gamma = Gv::scalar("gamma");
                let gm = Gv::scalar("body_0_mass");
                let off = [di as i32 - 1, dj as i32 - 1];
                let pre = Gv::field_offset("prim_pre", FieldRef::PrimPre, NDIM, &off);
                let rho = Gv::field_offset("prim_rho", "prim.rho", NDIM, &off);
                let cs2 = gamma * pre / rho.max(floor);
                let r = x0.max(floor);
                let omega_k = (gm / (r * r * r)).sqrt().max(floor);
                alpha * cs2 / omega_k
            } else {
                Gv::scalar("nu")
            };
        }
    }

    // physical components in, physical force out — same frame as the storage.
    let (dmom, dnrg) =
        symbi_hydro::viscous::viscous_update_orthogonal_2d(&vst, &rst, &nust, &h1, &h2, dx1, dx2, dt);
    let mut writes = accumulate_mom(dmom);
    let nrg = Gv::field("nrg", FieldRef::cons_nrg());
    writes.push(("nrg_out".to_string(), FieldRef::cons_nrg().into(), (nrg + dnrg).node()));
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

/// trace the Shakura-Sunyaev ALPHA operator on a GENERAL 2D orthogonal chart: the
/// same scale-factor operator as `viscous_iso_ortho_gv` but with a spatially
/// varying `nu(R) = alpha c_s^2 / Omega_k(R)`, `Omega_k = sqrt(GM/R^3)`, `R` the
/// RADIAL coordinate `x0` (the orbital radius on both cylindrical and spherical,
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
    let floor = Gv::from_f64(1e-30);

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
            let r = x0.max(floor);
            let omega_k = (gm / (r * r * r)).sqrt().max(floor);
            nust[dj][di] = alpha * cs2 / omega_k;
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
        ("mom_out_0".to_string(), FieldRef::cons_mom(0).into(), (mom0_c + dmom[0]).node()),
        ("mom_out_1".to_string(), FieldRef::cons_mom(1).into(), (mom1_c + dmom[1]).node()),
        ("mom_out_2".to_string(), FieldRef::cons_mom(2).into(), (mom2_c + dmom[2]).node()),
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

/// trace the constant-nu ADIABATIC viscous operator, 3D cartesian: the SAME `dt div(tau)` momentum as
/// `viscous_iso_gv_3d` PLUS the total-energy increment `dt div(tau . v)` onto `cons.nrg`. serves both
/// adiabatic hydro AND full-3D MHD (viscosity leaves B untouched, so the flux heats the gas with the
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
        ("mom_out_0".to_string(), FieldRef::cons_mom(0).into(), (mom0 + dmom[0]).node()),
        ("mom_out_1".to_string(), FieldRef::cons_mom(1).into(), (mom1 + dmom[1]).node()),
        ("mom_out_2".to_string(), FieldRef::cons_mom(2).into(), (mom2 + dmom[2]).node()),
        ("nrg_out".to_string(), FieldRef::cons_nrg().into(), (nrg + dnrg).node()),
    ];
    (end_trace(), writes)
}

/// read the primitive `(3-vector velocity, density)` 3x3 IN-PLANE stencil about the current cell — the
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
/// stencil + `viscous_update_2p5d`: the ISOTHERMAL twin writes the 3 momentum components; the
/// ADIABATIC twin ALSO writes the total-energy heating. serves 2.5D MHD (the toroidal velocity
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
        writes.push((format!("mom_out_{c}"), FieldRef::cons_mom(c as u8).into(), (mom_c + dmom[c]).node()));
    }
    if has_energy {
        let nrg = Gv::field("nrg", FieldRef::cons_nrg());
        writes.push(("nrg_out".to_string(), FieldRef::cons_nrg().into(), (nrg + dnrg).node()));
    }
    (end_trace(), writes)
}

/// trace the alpha-viscosity isothermal operator, 3D cartesian. the disk lies in
/// the x-y plane with rotation axis z, so the Keplerian frequency is set by the
/// CYLINDRICAL radius `R = sqrt((x-x_body)^2 + (y-y_body)^2)` — the vertical
/// z-offset from the body does NOT enter `Omega_k = sqrt(GM/R^3)`. hence `nu(x,y)
/// = alpha c_s^2 / Omega_k(R)` is z-invariant (a cylinder of constant nu about the
/// rotation axis), face-averaged so the flux divergence stays conservative.
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
    let floor = Gv::from_f64(1e-30);

    // nu is z-invariant (cylindrical R), so every k-slice of the stencil is equal.
    let mut nust = [[[Gv::ZERO; 3]; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let x = cx + Gv::from_f64(ii as f64 - 1.0) * dx;
            let y = cy + Gv::from_f64(jj as f64 - 1.0) * dy;
            let (rx, ry) = (x - bx, y - by);
            let r = (rx * rx + ry * ry).sqrt().max(floor);
            let omega_k = (gm / (r * r * r)).sqrt().max(floor);
            let nu = alpha * cs2 / omega_k;
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
/// `cs` param (globally isothermal); a locally-isothermal `cs2(x)` field is a
/// later refinement. nu vanishes toward the sink (Omega_k -> inf) — the
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
    let floor = Gv::from_f64(1e-30);

    let mut nust = [[Gv::ZERO; 3]; 3];
    for jj in 0..3usize {
        for ii in 0..3usize {
            let x = cx + Gv::from_f64(ii as f64 - 1.0) * dx;
            let y = cy + Gv::from_f64(jj as f64 - 1.0) * dy;
            let (rx, ry) = (x - bx, y - by);
            let r = (rx * rx + ry * ry).sqrt().max(floor);
            let omega_k = (gm / (r * r * r)).sqrt().max(floor);
            nust[jj][ii] = alpha * cs2 / omega_k;
        }
    }

    let dmom = viscous_mom_update_2d(&vst, &rst, &nust, dx, dy, dt);
    let writes = accumulate_mom(dmom);
    (end_trace(), writes)
}
