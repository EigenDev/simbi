// =============================================================================
// mhd_ideal_energy_audit.rs
//
// a read-only audit of what discrete energy the production ideal-MHD RK step advances, to decide the
// energy variable for the coupled magnetic-slip method. no immersed body (MagneticSpec::None), no
// sources, periodic 3D cartesian adiabatic MHD. over one accepted H_dt step it measures the change in
// the raw total energy sum E, the face-to-cell magnetic-variance defect sum delta, and the extended
// energy sum (E + delta), plus M_face, M_cell, and e_int. the magnetic-slip substep conserves
// sum(E + delta) exactly; this classifies whether the base ideal step does too, or only conserves the
// raw sum E while delta drifts at the scheme's truncation order.
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

// energies over the interior (cartesian unit face weights).
fn energies(sim: &Sim) -> (f64, f64, f64, f64, f64) {
    // returns (sum E, sum delta, M_face, M_cell, sum e_int).
    let m = sim.fields.mhd.as_ref().unwrap();
    let nrg = sim.fields.cons.nrg_field().unwrap();
    let (mut e, mut delta, mut m_face, mut m_cell, mut eint) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for c in sim.geom.interior.iter() {
        e += *nrg.at(c);
        let mut mcc = 0.0;
        for d in 0..3 {
            let mut up = c;
            up[d] += 1;
            let (bm, bp) = (*m.bface[d].at(c), *m.bface[d].at(up));
            m_face += 0.5 * bm * bm;
            let g = bp - bm;
            delta += 0.125 * g * g;
            let bcell = *m.bcell[d].at(c);
            mcc += 0.5 * bcell * bcell;
        }
        m_cell += mcc;
        // e_int = E - kinetic - M_cell.
        let den = *sim.fields.cons.den.at(c);
        let mut ke = 0.0;
        for d in 0..3 {
            let mom = *sim.fields.cons.mom[d].at(c);
            ke += 0.5 * mom * mom / den;
        }
        eint += *nrg.at(c) - ke - mcc;
    }
    (e, delta, m_face, m_cell, eint)
}

fn build_sim(n: usize, rough: bool) -> Sim {
    let dx = 1.0 / n as f64;
    let k = 2.0 * std::f64::consts::PI;
    // a smooth (or grid-scale) discretely-near-solenoidal B from a vector potential A_z, plus a smooth
    // flow so the ideal step evolves the field. B_x = dA_z/dy, B_y = -dA_z/dx, B_z = 0; A_z = a0
    // cos(k x) cos(m k y) / (m k). rough doubles the transverse wavenumber to load delta.
    let m = if rough { (n / 2).max(1) as f64 } else { 1.0 };
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
    .cfl(0.3)
    .allocate()
    .expect("ideal MHD audit sim construction failed")
    .set_initial(|[x, y, _z]| {
        // a smooth solenoidal velocity (an Orszag-Tang-like shear), uniform density/pressure.
        let vx = -(k * y).sin();
        let vy = (k * x).sin();
        MhdPrim::new(
            Prim::adiabatic(Density(1.0), Tensor::new([vx, vy, 0.0]), Pressure(1.0)),
            Tensor::new([0.0, 0.0, 0.0]),
        )
    })
    .seed_faces(move |axis, [x, y, _z]| match axis {
        0 => -a0 * (k * x).cos() * (m * k * y).sin(),          // B_x = dA_z/dy
        1 => a0 / m * (k * x).sin() * (m * k * y).cos(),       // B_y = -dA_z/dx
        _ => 0.0,
    })
    .build()
}

// run one accepted ideal-MHD step and return the energies before and after it, plus the step's dt.
fn one_step(n: usize, rough: bool, dt: f64) -> ((f64, f64, f64, f64, f64), (f64, f64, f64, f64, f64)) {
    let mut sim = build_sim(n, rough);
    let sub = NewtonianMhdSubstrateKernelSet3D::<HostMemory, f64>::new(GAMMA, 0.3, 1.0, &sim.geom.allocated);
    // capture the energies at each accepted step; compare the second to the first (both are primed,
    // accepted states, so the comparison is one clean H step free of initialization transients).
    let _ = dt;
    let snaps = std::cell::RefCell::new(Vec::<(f64, f64, f64, f64, f64)>::new());
    // evolve a fixed physical time that gives several accepted steps at every resolution, and compare
    // the last two snapshots: both are primed, mid-run, dynamic accepted states, so the difference is
    // one clean H step free of initialization transients.
    evolve_with_callback(&mut sim, &sub, 0.05, 1, |s| {
        snaps.borrow_mut().push(energies(s));
    })
    .expect("ideal MHD evolve failed");
    let s = snaps.into_inner();
    // the callback may repeat the final frame; scan from the end for the last pair of distinct
    // consecutive accepted states (one clean H step between two primed dynamic states).
    let mut i = s.len() - 1;
    while i > 0 && (s[i].0 - s[i - 1].0).abs() < 1e-300 && (s[i].2 - s[i - 1].2).abs() < 1e-300 {
        i -= 1;
    }
    assert!(i > 0, "no two distinct accepted states captured ({} snaps)", s.len());
    (s[i - 1], s[i])
}

#[test]
fn ideal_mhd_step_energy_audit() {
    // classify the extended-energy drift over one ideal step, on a smooth and a grid-scale field, and
    // under spatial refinement. the raw sum E is expected conserved (conservative flux, periodic); the
    // question is delta.
    println!("\n=== ideal-MHD one-step energy audit (periodic 3D adiabatic, no body) ===");
    for &rough in &[false, true] {
        println!("--- {} field ---", if rough { "grid-scale" } else { "smooth" });
        let mut last_d_ext = 0.0_f64;
        for &n in &[16usize, 32] {
            let dt = 2.0e-3;
            let (a, b) = one_step(n, rough, dt);
            let d_e = b.0 - a.0;
            let d_delta = b.1 - a.1;
            let d_ext = (b.0 + b.1) - (a.0 + a.1);
            let d_mface = b.2 - a.2;
            let d_mcell = b.3 - a.3;
            let d_eint = b.4 - a.4;
            let e_scale = a.0.abs().max(1.0);
            println!(
                "n={n:3}: dSE={:.3e} dSdelta={:.3e} dS(E+delta)={:.3e}  dMf={:.3e} dMc={:.3e} deint={:.3e}  (rel dSE={:.2e})",
                d_e, d_delta, d_ext, d_mface, d_mcell, d_eint, d_e.abs() / e_scale
            );
            if n == 32 && last_d_ext.abs() > 1e-14 {
                println!("     dS(E+delta) refinement ratio (n16/n32): {:.3}", last_d_ext.abs() / d_ext.abs().max(1e-300));
            }
            last_d_ext = d_ext;
        }
    }
    println!("========================================================\n");
}
