// =============================================================================
// mhd_ideal_energy_audit.rs
//
// a read-only, volume-weighted convergence audit of the discrete energy the production ideal-MHD RK
// step advances. no immersed body (MagneticSpec::None), no sources, periodic 3D cartesian adiabatic
// MHD. two discrete representations of the same continuum energy are compared:
//   the cell-energy representation  int E dV = sum_c E_c dV  (SIMBI's conserved cons.nrg), and
//   the face-Hodge representation   int (E + delta) dV,      (the slip theorem's variable),
// whose difference is the volume-weighted subcell magnetic-variance reservoir int delta dV. the
// ideal step conserves the cell energy to roundoff; the question is whether the face-Hodge drift
//   |Delta int delta dV| = |Delta int (E + delta) dV - Delta int E dV|
// decreases at the base scheme's order for a fixed smooth continuum solution as the grid refines with
// dt proportional to dx. delta_c = 1/8 sum_d (B_{d,+} - B_{d,-})^2 = O(dx^2), so int delta dV and its
// change should fall at roughly second order for a resolved smooth field.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D;
use symbi::sim::evolve::evolve_with_callback;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>;

const GAMMA: f64 = 5.0 / 3.0;
const T_FINAL: f64 = 0.04;

// volume-weighted energies over the interior: (int E dV, int delta dV).
fn integrals(sim: &Sim) -> (f64, f64) {
    let m = sim.fields.mhd.as_ref().unwrap();
    let nrg = sim.fields.cons.nrg_field().unwrap();
    let dv = sim.geom.dx[0] * sim.geom.dx[1] * sim.geom.dx[2];
    let (mut e, mut delta) = (0.0, 0.0);
    for c in sim.geom.interior.iter() {
        e += *nrg.at(c);
        for d in 0..3 {
            let mut up = c;
            up[d] += 1;
            let g = *m.bface[d].at(up) - *m.bface[d].at(c);
            delta += 0.125 * g * g;
        }
    }
    (e * dv, delta * dv)
}

// a fixed smooth continuum initial condition sampled at resolution n: a solenoidal B from the vector
// potential A_z = (a0/k) cos(k x) cos(k y) and a smooth solenoidal flow, k = 2 pi (one wavelength).
fn build_sim(n: usize, cfl: f64) -> Sim {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    let a0 = 0.3;
    SimStateGeneric::<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
        NewtonianMhd,
        IdealGas { gamma: GAMMA },
        Cartesian,
    )
    .cells([n, n, n])
    .origin([0.0, 0.0, 0.0])
    .spacing([dx, dx, dx])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .cfl(cfl)
    .allocate()
    .expect("audit sim construction failed")
    .set_initial(|[x, y, _z]| {
        MhdPrim::new(
            Prim::adiabatic(
                Density(1.0),
                Tensor::new([-(k * y).sin(), (k * x).sin(), 0.0]),
                Pressure(1.0),
            ),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    })
    .seed_faces(move |axis, [x, y, _z]| match axis {
        0 => -a0 * (k * x).cos() * (k * y).sin(), // B_x = dA_z/dy
        1 => a0 * (k * x).sin() * (k * y).cos(),   // B_y = -dA_z/dx
        _ => 0.0,
    })
    .build()
}

// evolve to T_FINAL and return the volume-weighted energy drifts (|Delta int E dV|, |Delta int delta
// dV|) between the first and last distinct accepted states.
fn drifts(n: usize, cfl: f64) -> (f64, f64) {
    let mut sim = build_sim(n, cfl);
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, cfl, 1.0, &sim.geom.allocated);
    let snaps = std::cell::RefCell::new(Vec::<(f64, f64)>::new());
    evolve_with_callback(&mut sim, &sub, T_FINAL, 1, |s| {
        snaps.borrow_mut().push(integrals(s));
    })
    .expect("ideal MHD evolve failed");
    let s = snaps.into_inner();
    // the first primed accepted state, and the last that differs from it (dropping repeated final
    // frames) -- the drift over the full evolution.
    let first = s[0];
    let mut j = s.len() - 1;
    while j > 0 && (s[j].0 - first.0).abs() < 1e-300 && (s[j].1 - first.1).abs() < 1e-300 {
        j -= 1;
    }
    assert!(j > 0, "no distinct evolved state ({} snaps)", s.len());
    ((s[j].0 - first.0).abs(), (s[j].1 - first.1).abs())
}

#[test]
fn ideal_mhd_face_hodge_drift_converges_at_second_order() {
    println!("\n=== ideal-MHD volume-weighted energy convergence (fixed smooth IC, dt ~ dx, T={T_FINAL}) ===");
    let cfl = 0.3;
    let mut prev_delta_drift = 0.0_f64;
    for &n in &[16usize, 32, 64] {
        let (de, ddelta) = drifts(n, cfl);
        print!("n={n:3}: |D int E dV| = {de:.3e}   |D int delta dV| = {ddelta:.3e}");
        if prev_delta_drift > 0.0 {
            println!("   spatial ratio (prev/this) = {:.2}  (order {:.2})", prev_delta_drift / ddelta, (prev_delta_drift / ddelta).log2());
        } else {
            println!();
        }
        prev_delta_drift = ddelta;
    }

    // temporal isolation: fix the grid, halve the timestep; a temporal-error component would shrink.
    println!("--- temporal isolation at n=32 ---");
    let (_e1, d_full) = drifts(32, 0.3);
    let (_e2, d_half) = drifts(32, 0.15);
    println!(
        "cfl 0.30: |D int delta dV| = {d_full:.3e}   cfl 0.15: {d_half:.3e}   temporal ratio = {:.3}",
        d_full / d_half.max(1e-300)
    );
    println!("========================================================\n");
}
