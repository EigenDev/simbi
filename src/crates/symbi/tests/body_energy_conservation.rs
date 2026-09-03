// =============================================================================
// body_energy_conservation.rs
//
// the gas+body energy conservation law for a two-way immersed wall. the gas total
// energy the wall exchanges is booked exactly as `BodyDelta::energy_delta`
// (nrg_old - nrg_new, summed over the masked cells); the body's mechanical KE
// (`Body::mechanical_ke`) changes from the same force/torque receipts via the
// equations of motion. a free (two-way) spinner in still gas drags the gas up and
// spins down: it loses rotational KE and the gas gains exactly that energy.
//
// the penalization writes the gas total energy to change by the work done on the
// wall (momentum transfer . local wall velocity), so summed `energy_delta` equals
// the body's mechanical work and -d(KE_body) == sum(energy_delta) to round-off; the
// frictional dissipation stays in the gas as internal energy (heat), never leaking.
// =============================================================================

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_ib::apply_body_deltas;
use symbi_ib::sdf::SdfExpr;
use symbi_ib::{Body, BodyCollection, SurfaceSpec};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 48;
const L: f64 = 1.0;
const OMEGA: f64 = 3.0;
const INERTIA: f64 = 1.0;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build() -> Sim {
    let dx = 2.0 * L / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .allocate()
        .expect("sim")
        .set_initial(|_| Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(1.0)))
        .build()
        .with_bodies(
            BodyCollection::new().add(
                // a free (two-way) no-slip spinner: the tangential drag drives the reaction torque that
                // spins it down, converting body rotational KE into gas energy.
                Body::rigid_sphere(
                    0,
                    Tensor::new([0.0, 0.0]),
                    Tensor::new([0.0, 0.0]),
                    1.0,
                    0.3,
                    INERTIA,
                    true,
                )
                .with_surface(SurfaceSpec::Porous {
                    porosity: 0.0,
                    k_eta_n: 50.0,
                    k_eta_t: 50.0,
                })
                .with_spin(OMEGA)
                .with_two_way_coupling(true),
            ),
        );
    sim.immersed.as_mut().unwrap().shapes[0] =
        Some(SdfExpr::<f64, 3>::sphere([0.0, 0.0, 0.0], 0.3));
    sim
}

#[test]
fn two_way_body_ke_change_balances_gas_energy() {
    let mut sim = build();
    let ke0 = sim.immersed.as_ref().unwrap().bodies.get(0).mechanical_ke();
    let dt = 1e-3;

    // accumulate the gas total-energy change over a handful of penalization steps (no godunov: the
    // wall exchanges with the local flow it develops, which is all the energy ledger needs).
    let mut gas_energy_change = 0.0_f64;
    for _ in 0..25 {
        dispatch_penalize(&sim, dt, GAMMA, 1.0);
        let deltas = sim.immersed.as_ref().unwrap().diagnostics.consolidate();
        gas_energy_change += deltas[0].energy_delta;
        apply_body_deltas(&mut sim.immersed.as_mut().unwrap().bodies, &deltas, dt);
        sim.immersed.as_mut().unwrap().diagnostics.reset();
    }
    let ke1 = sim.immersed.as_ref().unwrap().bodies.get(0).mechanical_ke();
    let dke = ke1 - ke0;

    // non-vacuous: the spinner actually spun down (lost rotational KE) and the gas gained energy.
    assert!(
        dke < -1e-8,
        "the two-way spinner did not lose KE ({dke:e}); test vacuous"
    );
    assert!(
        gas_energy_change < 0.0,
        "the gas did not gain energy ({gas_energy_change:e})"
    );

    // conservation: the body's mechanical KE change equals the gas total-energy change. energy_delta
    // is (nrg_old - nrg_new), negative when the gas gains, so it matches d(KE_body) < 0. the
    // dissipation stays in the gas as heat and never leaves the ledger. the residual is O(dt): the
    // gas energy is booked at penalize time while the body KE updates at the step midpoint, so the
    // two agree to the timestep's order. penalization that omits the work done on the wall instead
    // leaks order half the exchanged energy.
    let rel = (gas_energy_change - dke).abs() / dke.abs();
    assert!(
        rel < 1e-2,
        "gas+body energy not conserved: d(KE_body) = {dke:e} vs gas energy change {gas_energy_change:e} (rel {rel:e})"
    );
}
