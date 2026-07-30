// =============================================================================
// substrate_adiabatic_sod.rs
//
// the end-to-end gate for the adiabatic substrate: a full ADIABATIC (ideal-gas)
// Euler Sod shock-tube evolution where EVERY operator in the RK2 step — c2p,
// ghost_fill, cfl, flux, godunov_euler, godunov_rk2, snapshot — is the
// substrate-generated kernel (AdiabaticSubstrateKernelSet), run through the real
// `evolve()` loop on a real Newtonian `SimState`.
//
// Sod (gamma=1.4): (rho,p) = (1,1) | (0.125,0.1), v=0, transmissive walls. at the
// short final time the rarefaction + contact + shock stay interior, so: density
// and pressure finite + positive everywhere; the undisturbed edges hold the
// initial state; the middle develops structure (intermediate densities, gas
// accelerated); and mass is conserved (zero-velocity edges => zero boundary flux).
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 1.4;

#[test]
fn full_substrate_adiabatic_euler_sod() {
    let n = 128usize;
    let dx = 1.0 / n as f64;
    // Sod initial conditions, v = 0 => nrg = p/(gamma-1).
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim construction failed")
        .set_initial(|x| {
            let (rho, pre) = if x[0] < 0.5 { (1.0, 1.0) } else { (0.125, 0.1) };
            Prim {
                rho,
                vel: Tensor::new([0.0]),
                pre,
            }
        })
        .build();

    let cells: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
    let mass0: f64 = cells
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(*c))
        .sum::<f64>()
        * dx;

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    // march to t=0.1 — the waves stay well inside [0,1] at n=128.
    evolve(&mut sim, &sub, 0.1).expect("adiabatic evolution failed");

    // finite + positive density and pressure everywhere (a real, stable solve).
    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let mut max_vel = 0.0_f64;
    for c in &cells {
        let rho = *sim.fields.prim.rho.view().at(*c);
        let p = *pre.view().at(*c);
        let v = *sim.fields.prim.vel[0].view().at(*c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        max_vel = max_vel.max(v.abs());
    }

    // the undisturbed edges still hold the initial state (waves are interior).
    let rho_left = *sim.fields.prim.rho.view().at(cells[0]);
    let rho_right = *sim.fields.prim.rho.view().at(cells[n - 1]);
    assert!(
        (rho_left - 1.0).abs() < 1e-9,
        "left edge disturbed: rho = {rho_left}"
    );
    assert!(
        (rho_right - 0.125).abs() < 1e-9,
        "right edge disturbed: rho = {rho_right}"
    );

    // structure developed: intermediate densities exist and the gas accelerated.
    let has_intermediate = cells.iter().any(|c| {
        let r = *sim.fields.prim.rho.view().at(*c);
        r > 0.2 && r < 0.9
    });
    assert!(has_intermediate, "no rarefaction/contact structure formed");
    assert!(
        max_vel > 0.1,
        "gas did not accelerate (max |v| = {max_vel})"
    );

    // mass conserved: zero-velocity edges => zero boundary mass flux.
    let mass1: f64 = cells
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(*c))
        .sum::<f64>()
        * dx;
    assert!(
        (mass1 - mass0).abs() < 1e-9 * mass0,
        "mass drift {:e} (rel {:e})",
        mass1 - mass0,
        (mass1 - mass0) / mass0,
    );

    println!(
        "ADIABATIC SOD: {} steps to t={:.3}, mass rel-drift {:e}, max |v| {:.3}",
        sim.iteration,
        sim.time,
        (mass1 - mass0) / mass0,
        max_vel,
    );
}
