// =============================================================================
// resistive_field_decay.rs
//
// the analytic oracle for generic Ohmic resistivity: a sheared field `B_x = B0 sin(k y)` threading
// still gas decays as `exp(-eta k^2 t)` under the resistive induction diffusion `dB/dt = eta lap(B)`
// (the resistive edge EMF `eta J` riding the CT curl). `B0` is tiny so the ideal
// Alfven/magnetic-pressure dynamics are negligible over the run. `eta = 0` must not decay -- the
// bug-injection that proves the resistive term is what does it.
// =============================================================================

use std::f64::consts::PI;
use symbi_hydro::quantity::{Density, Pressure};

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

// 2.5D MHD: D=2 spatial, DOF=3 (the out-of-plane B/vel components), so the explicit generic.
type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const N: usize = 32;
const GAMMA: f64 = 5.0 / 3.0;
const B0: f64 = 1e-2; // tiny: magnetic pressure ~5e-5 << p, Alfven time >> the run
const K: f64 = 2.0 * PI; // one wavelength across [0,1] in y
const T_FINAL: f64 = 0.5;

fn make_sim() -> Sim {
    let dx = 1.0 / N as f64;
    // B_x = B0 sin(k y) is div-free (d/dx of a y-only field is 0). still, uniform gas.
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
        .cells([N, N])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("resistive decay sim construction failed")
        .set_initial(|[_x, y]| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([B0 * (K * y).sin(), 0.0, 0.0])))
        .seed_faces(|axis, [_x, y]| if axis == 0 { B0 * (K * y).sin() } else { 0.0 })
        .build()
}

// the field amplitude: max |B_x| (cell-centered) over the interior.
fn bx_amplitude(s: &Sim) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    s.geom
        .interior
        .iter()
        .map(|c| m.bcell[0].view().at(c).abs())
        .fold(0.0_f64, f64::max)
}

fn evolve(eta: f64) -> f64 {
    let mut sim = make_sim();
    let a0 = bx_amplitude(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 2>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    )
    .with_resistivity(eta);
    evolve_with_callback(&mut sim, &sub, T_FINAL, u64::MAX, |_| {}).expect("evolve failed");
    bx_amplitude(&sim) / a0
}

#[test]
fn resistive_field_decays_at_eta_k_squared() {
    let eta = 0.05;
    let ratio = evolve(eta);
    // the discrete Laplacian eigenvalue of sin(k y) is eta*k^2 to within ~0.3% at this resolution,
    // plus RK2 O(dt^2) error, so allow a modest band around the continuum decay.
    let expected = (-eta * K * K * T_FINAL).exp();
    assert!(
        (ratio - expected).abs() < 0.08 * expected,
        "resistive decay off: got {ratio}, expected ~{expected} (exp(-eta k^2 t))"
    );
}

// ---- 3D (D=3, DOF=3): the three-edge resistive EMF ----
type Sim3 = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
const N3: usize = 32; // y-resolution: same as the 2.5D case, so the numerical floor is small
const NZ3: usize = 4; // thin z-slab: the field is y-only, so z is uninvolved -> keeps it fast

fn make_sim_3d() -> Sim3 {
    let dx = 1.0 / N3 as f64;
    SimState::<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([N3, N3, NZ3])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(0.3)
    .allocate()
    .expect("3d resistive sim construction failed")
    .set_initial(|[_x, y, _z]| {
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
            Tensor::new([B0 * (K * y).sin(), 0.0, 0.0]),
        )
    })
    .seed_faces(|axis, x| {
        if axis == 0 {
            B0 * (K * x[1]).sin()
        } else {
            0.0
        }
    })
    .build()
}

fn bx_amplitude_3d(s: &Sim3) -> f64 {
    let m = s.fields.mhd.as_ref().unwrap();
    s.geom
        .interior
        .iter()
        .map(|c| m.bcell[0].view().at(c).abs())
        .fold(0.0_f64, f64::max)
}

fn evolve_3d(eta: f64) -> f64 {
    let mut sim = make_sim_3d();
    let a0 = bx_amplitude_3d(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    )
    .with_resistivity(eta);
    evolve_with_callback(&mut sim, &sub, T_FINAL, u64::MAX, |_| {}).expect("3d evolve failed");
    bx_amplitude_3d(&sim) / a0
}

#[test]
fn resistive_field_decays_at_eta_k_squared_3d() {
    // the same B_x = B0 sin(k y) diffusion, through the three-edge 3D resistive EMF (only J_z is
    // nonzero for a y-only field, but the full 3-edge dispatch runs). must match exp(-eta k^2 t).
    let eta = 0.05;
    let ratio = evolve_3d(eta);
    let expected = (-eta * K * K * T_FINAL).exp();
    assert!(
        (ratio - expected).abs() < 0.08 * expected,
        "3D resistive decay off: got {ratio}, expected ~{expected}"
    );
}

#[test]
fn resistivity_dominates_the_ideal_numerical_diffusion() {
    // bug-injection: eta = 0 (ideal MHD) still loses a little field to the scheme's own
    // finite-resolution numerical diffusion. the resistive term must cause substantially more loss
    // than that floor -- otherwise the decay in the companion test could be a numerical artifact.
    // eta=0.05 decays to ~0.37, the ideal floor only to ~0.90.
    let ideal = evolve(0.0);
    let resistive = evolve(0.05);
    assert!(
        resistive < 0.6 * ideal,
        "resistivity did not dominate the numerical diffusion: ideal ratio {ideal}, resistive {resistive}"
    );
}
