// =============================================================================
// worker_device_scope.rs
//
// the features a launched worker on device memory refuses: a tracer population, a horizon
// body, and a magnetic-slip body; a plain gas store and a gas store with a gravitating body
// are admitted. the predicate reads the store alone, so it is checked on host stores here; the
// worker applies it to every store on device memory.
//
// usage:
//   cargo test -p symbi --test worker_device_scope
// =============================================================================

use std::f64::consts::PI;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::worker::device_scope_refusal;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 8;

fn gas() -> SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory> {
    let h = 1.0 / N as f64;
    SimState::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
}

fn magnetized() -> SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory> {
    let h = 1.0 / N as f64;
    SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("sim")
        .set_initial(|[_, y]| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)), Tensor::new([(2.0 * PI * y).sin(), 0.0, 0.0])))
        .seed_faces(|axis, [_, y]| if axis == 0 { (2.0 * PI * y).sin() } else { 0.0 })
        .build()
}

#[test]
fn a_plain_gas_store_is_admitted() {
    assert_eq!(device_scope_refusal(&gas().store), None);
    assert_eq!(device_scope_refusal(&magnetized().store), None);
}

#[test]
fn a_gravitating_body_is_admitted() {
    let mut sim = gas();
    sim.attach_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(0, Tensor::new([0.5, 0.5]), Tensor::new([0.0, 0.0]), 1.0, 0.05, 0.02)));
    assert!(sim.has_bodies(), "the body never attached; the admission is vacuous");
    assert_eq!(device_scope_refusal(&sim.store), None);
}

#[test]
fn a_tracer_population_is_refused() {
    let mut sim = gas();
    sim.store.tracers = Some(symbi_sim::tracers::seed_mass_weighted(&sim, 64));
    assert_eq!(device_scope_refusal(&sim.store), Some("a tracer population on device memory"));
}

#[test]
fn a_horizon_body_is_refused() {
    let mut sim = gas();
    sim.attach_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::horizon(0, 0.1, 0.2)));
    assert_eq!(device_scope_refusal(&sim.store), Some("a horizon body on device memory"));
}

#[test]
fn a_magnetic_slip_body_is_refused() {
    let mut sim = magnetized();
    let allocated = sim.geom.allocated.clone();
    sim.store.fields.mhd.as_mut().expect("mhd fields").alloc_slip_quadrature(&allocated).expect("slip quadrature");
    assert_eq!(device_scope_refusal(&sim.store), Some("a magnetic-slip body on device memory"));
}
