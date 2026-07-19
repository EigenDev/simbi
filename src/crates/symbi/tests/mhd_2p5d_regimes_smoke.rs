// =============================================================================
// mhd_2p5d_regimes_smoke.rs
//
// cross-regime smoke: the ISOTHERMAL and RELATIVISTIC
// MHD substrates each run a few steps on a GENUINE 2.5D grid (D=2, DOF=3) with an
// Orszag-Tang-with-Bz IC; assert the in-plane staggered div(B) stays at machine zero,
// the state stays finite, and the out-of-plane Bz evolves. companion to
// nmhd_2p5d_divb_under_evolve.rs (the detailed NMHD gate) — these confirm the same
// 2.5D CT substrate path is correctly wired for the other two MHD regimes.
// =============================================================================

use std::f64::consts::PI;

use symbi::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal_mhd::IsothermalMhd;
use symbi_hydro::mhd_state::{MhdPrim, MhdPrimG};
use symbi_hydro::regime::Regime;
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const NX: usize = 16;
const NY: usize = 16;
const B0: f64 = 1.0;
const BZ0: f64 = 0.4;
const V0: f64 = 0.3;

// the OT-with-Bz primitive vectors (vel, B 3-vectors) at a cell CENTER — regime-agnostic.
fn ot_vectors(x: f64, y: f64) -> (Tensor<f64, 3>, Tensor<f64, 3>) {
    let vel = Tensor::new([-V0 * (2.0 * PI * y).sin(), V0 * (2.0 * PI * x).sin(), 0.0]);
    let mag = Tensor::new([-B0 * (2.0 * PI * y).sin(), B0 * (4.0 * PI * x).sin(), BZ0 * (2.0 * PI * x).cos()]);
    (vel, mag)
}

// staggered in-plane B IC (the face "truth"): Bx = -B0 sin(2pi y), By = B0 sin(4pi x).
fn ot_bface(axis: usize, [x, y]: [f64; 2]) -> f64 {
    match axis {
        0 => -B0 * (2.0 * PI * y).sin(),
        _ => B0 * (4.0 * PI * x).sin(),
    }
}

// in-plane staggered div(B) = dBx/dx + dBy/dy, relative to max in-plane |B|.
fn rel_divb<R, E>(s: &SimStateGeneric<R, 2, 3, Cartesian, E, CpuSpace, HostMemory>) -> f64
where
    R: Regime<f64, 2>,
    E: symbi_hydro::eos::Eos<f64>,
{
    let mhd = s.fields.mhd.as_ref().unwrap();
    let (idx, idy) = (NX as f64, NY as f64);
    let mut md = 0.0_f64;
    let mut mb = 1.0_f64;
    for c in s.geom.interior.iter() {
        let bx_lo = *mhd.bface[0].view().at(c);
        let bx_hi = *mhd.bface[0].view().at([c[0] + 1, c[1]]);
        let by_lo = *mhd.bface[1].view().at(c);
        let by_hi = *mhd.bface[1].view().at([c[0], c[1] + 1]);
        md = md.max(((bx_hi - bx_lo) * idx + (by_hi - by_lo) * idy).abs());
        mb = mb.max((bx_lo * bx_lo + by_lo * by_lo).sqrt());
    }
    md / mb
}

// finiteness + the out-of-plane Bz evolution check (vs its analytic IC sample).
fn assert_finite_and_bz_evolved<R, E>(label: &str, sim: &SimStateGeneric<R, 2, 3, Cartesian, E, CpuSpace, HostMemory>)
where
    R: Regime<f64, 2>,
    E: symbi_hydro::eos::Eos<f64>,
{
    let mhd = sim.fields.mhd.as_ref().unwrap();
    let dx = 1.0 / NX as f64;
    let mut max_dbz = 0.0_f64;
    for c in sim.geom.interior.iter() {
        for k in 0..3 {
            assert!(mhd.bcell[k].view().at(c).is_finite(), "{label}: non-finite bcell[{k}] at {c:?}");
        }
        assert!(sim.fields.cons.den.view().at(c).is_finite(), "{label}: non-finite den at {c:?}");
        let x = (c[0] as f64 + 0.5) * dx;
        max_dbz = max_dbz.max((*mhd.bcell[2].view().at(c) - BZ0 * (2.0 * PI * x).cos()).abs());
    }
    assert!(max_dbz > 1e-7, "{label}: out-of-plane Bz did not evolve (max |dBz|={max_dbz:e})");
    eprintln!("[2p5d {label}] OK max|dBz|={max_dbz:e}");
}

#[test]
fn imhd_2p5d_smoke() {
    const CS: f64 = 1.0;
    // iso primitive (no pressure slot); set_initial seeds cons + bcell, seed_faces the in-plane B.
    let rho0 = 1.0;
    let mut sim = SimStateGeneric::<IsothermalMhd, 2, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>::build(
        IsothermalMhd, Isothermal { cs: CS }, Cartesian,
    )
        .cells([NX, NY])
        .spacing([1.0 / NX as f64, 1.0 / NY as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("imhd 2.5d construction")
        .set_initial(|[x, y]| {
            let (vel, mag) = ot_vectors(x, y);
            MhdPrimG::<f64, 3, IsoModel> { hydro: PrimG { rho: rho0, vel, pre: Default::default() }, mag }
        })
        .seed_faces(ot_bface)
        .build();

    let kset = IsothermalMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(CS, 0.3, 1.0, &sim.geom.allocated);
    let mut steps = 0u64;
    evolve_with_callback(&mut sim, &kset, 0.2, 1, |s| {
        let rel = rel_divb(s);
        assert!(rel < 1e-10, "imhd: 2.5D div(B) grew to rel={rel:e} at iter {}", s.iteration);
        steps = s.iteration;
    }).expect("imhd 2.5d evolve failed");
    assert!(steps >= 3, "imhd: only {steps} steps");
    assert_finite_and_bz_evolved("imhd", &sim);
}

#[test]
fn rmhd_2p5d_smoke() {
    const GAMMA: f64 = 5.0 / 3.0;
    // adiabatic primitive; set_initial seeds cons + bcell, seed_faces the in-plane B.
    let rho0 = GAMMA * GAMMA;
    let p0 = GAMMA;
    let mut sim = SimStateGeneric::<Rmhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Rmhd, IdealGas { gamma: GAMMA }, Cartesian,
    )
        .cells([NX, NY])
        .spacing([1.0 / NX as f64, 1.0 / NY as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("rmhd 2.5d construction")
        .set_initial(|[x, y]| {
            let (vel, mag) = ot_vectors(x, y);
            MhdPrim { hydro: Prim { rho: rho0, vel, pre: p0 }, mag }
        })
        .seed_faces(ot_bface)
        .build();

    let kset = RmhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    let mut steps = 0u64;
    evolve_with_callback(&mut sim, &kset, 0.2, 1, |s| {
        let rel = rel_divb(s);
        assert!(rel < 1e-10, "rmhd: 2.5D div(B) grew to rel={rel:e} at iter {}", s.iteration);
        steps = s.iteration;
    }).expect("rmhd 2.5d evolve failed");
    assert!(steps >= 3, "rmhd: only {steps} steps");
    assert_finite_and_bz_evolved("rmhd", &sim);
}
