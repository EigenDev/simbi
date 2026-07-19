// =============================================================================
// cylindrical_swirl_evolve.rs
//
// the FIRST end-to-end run of a vector-dimension-lifted sim: an axisymmetric
// cylindrical (r, z) grid carrying a 3-component velocity (v_r, v_phi, v_z) —
// DOF = 3 on an NDIM = 2 grid. it exercises the whole _cyl
// adiabatic kernel family (c2p / flux / godunov / wave_speed / snapshot / ghost,
// all ncomp = 3) through the DOF-generic AdiabaticSubstrateKernelSet + the
// metadata-driven dispatch (each kernel's buffer manifest read off the artifact).
//
// physics check — the AXISYMMETRIC CENTRIFUGAL SOURCE. a uniform swirling gas
// (rho, p uniform; v_phi = v0 const; v_r = v_z = 0) has zero pressure gradient, so
// the only radial force is centrifugal: d v_r / dt = v_phi^2 / r. over a short time
// t (before the pressure feedback matters) v_r(r) ~ v0^2 t / r, i.e., v_r * r is
// constant across r — the unmistakable 1/r signature of the geometric source. a
// cartesian scheme (no source) would leave v_r = 0; matching the magnitude AND the
// 1/r structure proves the source is active, correctly signed, and r-weighted.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cylindrical;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;

#[test]
fn axisymmetric_swirl_centrifugal_source() {
    // a 3-component velocity (v_r, v_phi, v_z) on a 2-axis (r, z) grid: DOF=3, NDIM=2.
    type CylSim =
        SimStateGeneric<Newtonian, 2, 3, Cylindrical, IdealGas<f64>, CpuSpace, HostMemory>;

    // an annulus r in [1, 2] (r_min > 0 avoids the r=0 axis singularity BC); z periodic.
    let (nr, nz) = (48usize, 8usize);
    let (r_lo, r_hi) = (1.0_f64, 2.0_f64);
    let dr = (r_hi - r_lo) / nr as f64;
    let dz = 0.5 / nz as f64;
    let v0 = 1.0_f64; // uniform swirl speed.

    // uniform state: rho = 1, p = 1, v_phi = v0, v_r = v_z = 0. vel is COORDINATE-indexed
    // (0 = r, 1 = phi, 2 = z); the regime folds the full 3-component kinetic term into energy.
    let mut sim = CylSim::build(Newtonian, IdealGas { gamma: GAMMA }, Cylindrical)
        .cells([nr, nz])
        .origin([r_lo, 0.0])
        .spacing([dr, dz])
        // r: outflow (uniform IC -> uniform ghosts, no spurious boundary force); z: periodic.
        .boundaries(Boundaries::per_axis([
            [BoundaryType::Outflow, BoundaryType::Outflow],
            [BoundaryType::Periodic, BoundaryType::Periodic],
        ]))
        .allocate()
        .expect("cylindrical axisymmetric sim construction failed")
        .set_initial(|_x| Prim {
            rho: 1.0,
            vel: Tensor::new([0.0, v0, 0.0]),
            pre: 1.0,
        })
        .build();

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    let t_final = 0.02;
    evolve(&mut sim, &sub, t_final).expect("axisymmetric swirl evolution failed");

    let rho = &sim.fields.prim.rho;
    let vr = &sim.fields.prim.vel[0];
    let vphi = &sim.fields.prim.vel[1];
    let vz = &sim.fields.prim.vel[2];
    let pre = sim.fields.prim.pre_field().expect("prim.pre");

    // sample the interior away from the r-walls (the BC perturbation can reach a few cells
    // in over t_final; the centrifugal signature is under test).
    let mut vr_times_r: Vec<f64> = Vec::new();
    for c in sim.geom.interior.iter() {
        if c[0] < 4 || c[0] >= nr as isize - 4 {
            continue;
        }
        let r = r_lo + (c[0] as f64 + 0.5) * dr;
        let rho_c = *rho.view().at(c);
        let vr_c = *vr.view().at(c);
        let vphi_c = *vphi.view().at(c);
        let vz_c = *vz.view().at(c);
        let p_c = *pre.view().at(c);

        assert!(
            rho_c.is_finite() && vr_c.is_finite() && vphi_c.is_finite() && p_c.is_finite(),
            "non-finite state at {c:?}: rho={rho_c} vr={vr_c} vphi={vphi_c} p={p_c}"
        );
        // centrifugal source pushes outward.
        assert!(
            vr_c > 0.0,
            "v_r must be outward (centrifugal) at r={r:.3}, got {vr_c:.3e}"
        );
        // swirl is preserved (the phi-momentum source ~ -rho v_r v_phi / r is 2nd order while
        // v_r is still tiny), z stays put.
        assert!(
            (vphi_c - v0).abs() < 0.1 * v0,
            "v_phi drifted at r={r:.3}: {vphi_c:.4}"
        );
        assert!(
            vz_c.abs() < 1e-3,
            "v_z must stay ~0 at r={r:.3}: {vz_c:.3e}"
        );

        vr_times_r.push(vr_c * r);
    }
    assert!(vr_times_r.len() > 8, "too few interior samples");

    // the 1/r signature: v_r * r is constant across radius (centrifugal a_r = v0^2/r).
    let mean = vr_times_r.iter().sum::<f64>() / vr_times_r.len() as f64;
    let (lo, hi) = vr_times_r
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), &x| (l.min(x), h.max(x)));
    assert!(
        (hi - lo) / mean < 0.15,
        "v_r*r not radius-constant (the 1/r centrifugal signature): spread {:.1}% (lo={lo:.4e} hi={hi:.4e})",
        100.0 * (hi - lo) / mean
    );

    // the MAGNITUDE: v_r * r ~ v0^2 * t. lenient band — the realized integration time
    // overshoots t_final by < 1 dt and the pressure feedback shaves the late growth.
    let expected = v0 * v0 * t_final;
    assert!(
        mean > 0.5 * expected && mean < 1.4 * expected,
        "centrifugal magnitude off: v_r*r mean = {mean:.4e}, expected ~ v0^2*t = {expected:.4e}"
    );
}
