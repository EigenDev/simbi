// =============================================================================
// viscous_mhd_2p5d.rs
//
// the DOF-aware 2.5D MHD viscous operator (D=2 grid, DOF=3 momentum) must diffuse the OUT-OF-PLANE
// velocity v_z -- the toroidal component a rotating disk carries -- which the plain 2D viscous kernel
// (2 in-plane momentum components) cannot touch. seed a pure out-of-plane shear v_z = V sin(k y): with
// viscosity the out-of-plane kinetic energy decays (mom[2] diffuses by the in-plane Laplacian) and the
// gas heats; the inviscid twin barely moves. this is the case that would silently do NOTHING without
// the _dof3 kernel.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithViscosity;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

// 2.5D MHD: D=2 spatial, DOF=3 momentum (the out-of-plane v_z / B_z).
type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const V0: f64 = 0.05;
const T_FINAL: f64 = 0.1;

fn make() -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("2.5d mhd sim")
        // a PURELY out-of-plane velocity shear v_z = V sin(k y); tiny uniform in-plane B (div-free).
        .set_initial(|[_x, y]| MhdPrim {
            hydro: Prim { rho: 1.0, vel: Tensor::new([0.0, 0.0, V0 * (k * y).sin()]), pre: 1.0 },
            mag: Tensor::new([1e-3, 0.0, 0.0]),
        })
        .seed_faces(|axis, _| if axis == 0 { 1e-3 } else { 0.0 })
        .build()
}

// (out-of-plane kinetic energy, gas internal energy) over the interior.
fn measures(s: &Sim) -> (f64, f64) {
    let m = s.fields.mhd.as_ref().unwrap();
    let den = &s.fields.cons.den;
    let nrg = s.fields.cons.nrg_field().unwrap();
    let (mut kz, mut ie) = (0.0, 0.0);
    for c in s.geom.interior.iter() {
        let rho = *den.view().at(c);
        let mz = *s.fields.cons.mom[2].view().at(c);
        let mut msq = 0.0;
        for k in 0..3 { let mo = *s.fields.cons.mom[k].view().at(c); msq += mo * mo; }
        let mut bsq = 0.0;
        for k in 0..3 { let b = *m.bcell[k].view().at(c); bsq += b * b; }
        kz += 0.5 * mz * mz / rho;
        ie += *nrg.view().at(c) - 0.5 * msq / rho - 0.5 * bsq;
    }
    (kz, ie)
}

fn run(nu: f64) -> (f64, f64, f64, f64) {
    let mut sim = make();
    let (kz0, ie0) = measures(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated)
        .with_viscosity(nu);
    evolve_with_callback(&mut sim, &sub, T_FINAL, u64::MAX, |_| {}).expect("2.5d mhd viscous evolve failed");
    let (kz1, ie1) = measures(&sim);
    (kz0, ie0, kz1, ie1)
}

#[test]
fn viscosity_diffuses_the_out_of_plane_velocity() {
    let (kz0, ie0, kz1, ie1) = run(0.02);
    let (kz0_i, _, kz1_i, _) = run(0.0);

    assert!(kz0 > 0.0, "degenerate seed");
    // the OUT-OF-PLANE kinetic energy decays far more with viscosity than the inviscid floor -- the
    // _dof3 kernel is diffusing mom[2], which the plain 2D kernel would leave untouched.
    let loss_visc = kz0 - kz1;
    let loss_ideal = kz0_i - kz1_i;
    assert!(
        loss_visc > 5.0 * loss_ideal,
        "2.5D MHD viscosity did not diffuse the out-of-plane velocity: viscous KE_z loss {loss_visc:.3e} \
         vs inviscid {loss_ideal:.3e}"
    );
    // and the dissipated toroidal motion heated the gas.
    assert!(ie1 > ie0, "out-of-plane viscous dissipation did not heat the gas: {ie0} -> {ie1}");
}
