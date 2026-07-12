// =============================================================================
// gv_viscous.rs
//
// the traced constant-nu isothermal viscous operator (docs/design/54 step 1):
// per interior cell, read the primitive velocity + density on the halo-1 3x3
// stencil (via `field_offset`), evaluate the SAME carrier-generic
// `viscous_mom_update_2d` the f64 oracle runs, and accumulate `dt div(tau)` into
// `cons.mom`. 2D cartesian.
//
// hazard-free in place: the stencil reads are on PRIMITIVE fields (prim.rho /
// prim.vel, read-only during this pass); the only write is `cons.mom` at the
// CENTER cell (pointwise), so no cell reads a neighbour's half-updated momentum.
// the pass therefore runs post-c2p (prim current), once per step after the RK
// combination — the penalize placement law.
//
// scalars: dt, nu (the constant kinematic viscosity), dx_0 / dx_1. no support
// ball — the viscous operator acts over the whole interior (unlike the compactly
// supported drain).
//
// usage (build.rs):
//   let (k, writes) = viscous_iso_gv();
//   emit_gv(out, KernelId::ViscousIso { ndim: 2 }.name(), 2, &k, &writes);
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_algebra::Tensor;
use symbi_hydro::viscous::viscous_mom_update_2d;
use symbi_ir::gv::Writes;
use symbi_ir::{begin_trace, end_trace, FieldRef, Gv, GvKernel};

/// trace the constant-nu isothermal viscous momentum operator, 2D cartesian.
pub fn viscous_iso_gv() -> (GvKernel, Writes) {
    const NDIM: u8 = 2;
    begin_trace();
    let dt = Gv::scalar("dt");
    let nu = Gv::scalar("nu");
    let dx = Gv::scalar("dx_0");
    let dy = Gv::scalar("dx_1");

    // the primitive 3x3 stencil (read-only): v[jj][ii] = prim.vel, rho = prim.rho,
    // at offsets (di, dj) in {-1,0,1}^2. rho diagonals are read but unused by the
    // 5-point face averaging (the trace DCEs them).
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

    let dmom = viscous_mom_update_2d(&vst, &rst, nu, dx, dy, dt);

    // accumulate onto cons.mom (in place, pointwise): mom_new = mom_center + dmom.
    let mom0_c = Gv::field("mom0", FieldRef::cons_mom(0));
    let mom1_c = Gv::field("mom1", FieldRef::cons_mom(1));
    let writes: Writes = vec![
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
    ];

    (end_trace(), writes)
}
