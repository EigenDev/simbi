// =============================================================================
// adiabatic_viscous_heating.rs
//
// the ADIABATIC viscous operator books the viscous HEATING in addition to the shear force. a decaying
// shear layer v_x = V sin(k y) in a periodic box must, under viscosity:
//   - CONSERVE total energy (the viscous energy flux div(tau.v) is conservative, like the ideal
//     godunov flux) -- Sum nrg is invariant to round-off,
//   - DISSIPATE kinetic energy into INTERNAL energy (the gas heats up).
// the inviscid twin (nu = 0) loses far less kinetic energy (only the scheme's numerical diffusion) and
// heats far less, isolating the physical viscous dissipation from the numerical floor.
// =============================================================================

use symbi::prelude::Newtonian;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithViscosity;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 32;
const GAMMA: f64 = 1.4;
const V0: f64 = 0.1; // subsonic shear amplitude (cs ~ 1.18)
const P0: f64 = 1.0;
const RHO0: f64 = 1.0;
const T_FINAL: f64 = 0.1;

fn build() -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([0.0, 0.0])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .finish()
        .expect("sim construction failed");
    let cnrg = sim.fields.cons.nrg_field().expect("Newtonian cons.nrg").clone();
    for c in sim.geom.interior.iter() {
        let y = (c[1] as f64 + 0.5) * dx;
        let vx = V0 * (k * y).sin();
        sim.fields.cons.den.view_mut().set(c, RHO0);
        sim.fields.cons.mom[0].view_mut().set(c, RHO0 * vx);
        sim.fields.cons.mom[1].view_mut().set(c, 0.0);
        cnrg.view_mut().set(c, P0 / (GAMMA - 1.0) + 0.5 * RHO0 * vx * vx);
    }
    sim
}

// (total energy, kinetic energy) summed over the interior.
fn energies(s: &Sim) -> (f64, f64) {
    let den = &s.fields.cons.den;
    let nrg = s.fields.cons.nrg_field().unwrap();
    let (mut e_tot, mut ke) = (0.0, 0.0);
    for c in s.geom.interior.iter() {
        let rho = *den.view().at(c);
        let mx = *s.fields.cons.mom[0].view().at(c);
        let my = *s.fields.cons.mom[1].view().at(c);
        e_tot += *nrg.view().at(c);
        ke += 0.5 * (mx * mx + my * my) / rho;
    }
    (e_tot, ke)
}

fn run(nu: f64) -> (f64, f64, f64, f64) {
    let mut sim = build();
    let (e0, ke0) = energies(&sim);
    let sub = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.3, &sim.geom.allocated)
        .with_viscosity(nu);
    evolve(&mut sim, &sub, T_FINAL).expect("adiabatic viscous evolve failed");
    let (e1, ke1) = energies(&sim);
    (e0, ke0, e1, ke1)
}

#[test]
fn adiabatic_viscosity_conserves_energy_and_heats() {
    let (e0, ke0, e1, ke1) = run(0.01);
    let (ie0, ie1) = (e0 - ke0, e1 - ke1);

    // total energy conserved (periodic + conservative viscous energy flux) to round-off.
    let e_rel = (e1 - e0).abs() / e0;
    assert!(e_rel < 1e-10, "adiabatic viscosity did not conserve total energy: rel drift {e_rel:.3e}");
    // kinetic energy dissipated, internal energy risen (the gas heated).
    assert!(ke1 < ke0, "viscous run did not dissipate kinetic energy: {ke0} -> {ke1}");
    assert!(ie1 > ie0, "viscous run did not heat the gas: internal {ie0} -> {ie1}");

    // isolate the PHYSICAL dissipation from the scheme's numerical floor: the inviscid twin loses far
    // less kinetic energy and heats far less.
    let (_, ke0_i, _, ke1_i) = run(0.0);
    let visc_loss = ke0 - ke1;
    let ideal_loss = ke0_i - ke1_i;
    assert!(
        visc_loss > 3.0 * ideal_loss,
        "viscous dissipation did not dominate the numerical floor: viscous KE loss {visc_loss:.3e}, \
         inviscid {ideal_loss:.3e}"
    );
}
