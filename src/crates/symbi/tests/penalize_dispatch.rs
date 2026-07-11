// =============================================================================
// penalize_dispatch.rs
//
// the [Drain] penalization dispatch (docs/design/50 step 2b), end to end on a
// real sim: the kernel runs over the body's declared support box only, drains
// the masked cells in place, leaves the far field BIT-untouched, and the
// reduced deltas land in the diagnostics accumulator — gas loss == body gain
// to machine precision.
// =============================================================================

use symbi::regimes::substrate_kernels::dispatch_penalize;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_ib::{Body, BodyCollection};
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 48;
const L: f64 = 1.0;
const GAMMA: f64 = 1.4;

#[test]
fn penalize_drains_the_mask_and_conserves_gas_plus_body() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let dx = 2.0 * L / N as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .origin([-L, -L])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| Prim {
            rho: 1.0 + 0.2 * (2.0 * x).sin() * (1.5 * y).cos(),
            vel: Tensor::new([0.15, -0.1]),
            pre: 1.0,
        })
        .build()
        .with_bodies(BodyCollection::new().add(Body::black_hole(
            0,
            Tensor::new([0.1, -0.05]),
            Tensor::zeros(),
            1.0,
            0.08,
            0.04,
            0.5,
            0.0,
            0.12, // accretion radius = the mask radius
        )));

    let before: Vec<f64> = sim.geom.interior.iter().map(|c| *sim.fields.cons.den.view().at(c)).collect();
    let nrg_before: Vec<f64> =
        sim.geom.interior.iter().map(|c| *sim.fields.cons.nrg_field().unwrap().view().at(c)).collect();

    let dt = 1e-3;
    dispatch_penalize(&sim, dt, GAMMA, 1.0);

    let im = sim.immersed.as_ref().unwrap();
    let deltas = im.diagnostics.consolidate();
    assert!(deltas[0].mass_delta > 0.0, "the drain never removed mass");
    assert!(deltas[0].energy_delta > 0.0, "the drain never removed energy");

    // gas loss == body gain, machine-exact: sum the cell-integrated conserved
    // change over the interior against the accumulated delta.
    let dv = {
        // the kernel's volume spelling (face-difference widths, reciprocated
        // twice) — mirrored so the sum matches to the bit family.
        let w = ((-L) + dx) - (-L);
        1.0 / (1.0 / (w * w))
    };
    let mut lost_mass = 0.0;
    let mut lost_nrg = 0.0;
    let mut far_changed = 0usize;
    for (i, c) in sim.geom.interior.iter().enumerate() {
        let after = *sim.fields.cons.den.view().at(c);
        lost_mass += (before[i] - after) * dv;
        lost_nrg += (nrg_before[i] - *sim.fields.cons.nrg_field().unwrap().view().at(c)) * dv;
        // far outside the support ball (r_cut = 0.12 + 20 dx ~ 0.95): the
        // dispatch never touches the cell — bit-identical, not just close.
        let x = -L + (c[0] as f64 + 0.5) * dx;
        let y = -L + (c[1] as f64 + 0.5) * dx;
        let r = ((x - 0.1f64).powi(2) + (y + 0.05).powi(2)).sqrt();
        if r > 0.12 + 22.0 * dx && after.to_bits() != before[i].to_bits() {
            far_changed += 1;
        }
    }
    assert_eq!(far_changed, 0, "cells beyond the support ball were touched");
    assert!(
        (lost_mass - deltas[0].mass_delta).abs() <= 1e-12 * deltas[0].mass_delta.abs(),
        "gas mass loss {lost_mass} != body gain {}",
        deltas[0].mass_delta,
    );
    assert!(
        (lost_nrg - deltas[0].energy_delta).abs() <= 1e-12 * deltas[0].energy_delta.abs().max(1e-30),
        "gas energy loss {lost_nrg} != body gain {}",
        deltas[0].energy_delta,
    );
}
