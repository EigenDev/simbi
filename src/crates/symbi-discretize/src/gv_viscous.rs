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
use symbi_hydro::viscous::{viscous_mom_update_2d, viscous_mom_update_3d};
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

/// accumulate the viscous increment onto cons.mom (in place, pointwise center).
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
