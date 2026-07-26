// =============================================================================
// viscous_resistive_mhd.rs
//
// resistive-VISCOUS MHD (finite magnetic Prandtl number Pm = nu/eta): both diffusivities active at
// once in a full-3D Newtonian MHD flow. viscosity diffuses the VELOCITY (and heats the gas via the
// energy flux); resistivity diffuses the MAGNETIC field -- ORTHOGONAL, independent operators. a
// sheared v_x = V sin(k y) + a sheared B_x = B0 sin(k y) must, with both on:
//   - lose more KINETIC energy than the ideal run (viscosity is acting),
//   - lose more MAGNETIC energy than the ideal run (resistivity is acting),
//   - GAIN gas internal energy (BOTH viscous heating AND Ohmic heating warm the gas -- each dissipation
//     is conservatively booked into the internal energy),
//   - evolve stably.
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::WithViscosity;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<NewtonianMhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const V0: f64 = 0.05;
const B0: f64 = 0.05;
const T_FINAL: f64 = 0.1;

fn make() -> Sim {
    let dx = 1.0 / N as f64;
    let k = 2.0 * std::f64::consts::PI;
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N, N])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("3d mhd sim")
        .set_initial(|[_x, y, _z]| MhdPrim {
            hydro: Prim {
                rho: 1.0,
                vel: Tensor::new([V0 * (k * y).sin(), 0.0, 0.0]),
                pre: 1.0,
            },
            mag: Tensor::new([B0 * (k * y).sin(), 0.0, 0.0]),
        })
        .seed_faces(|axis, x| {
            if axis == 0 {
                B0 * (k * x[1]).sin()
            } else {
                0.0
            }
        })
        .build()
}

// (kinetic, magnetic, internal-gas) energies summed over the interior.
fn energies(s: &Sim) -> (f64, f64, f64) {
    let m = s.fields.mhd.as_ref().unwrap();
    let den = &s.fields.cons.den;
    let nrg = s.fields.cons.nrg_field().unwrap();
    let (mut ke, mut me, mut ie) = (0.0, 0.0, 0.0);
    for c in s.geom.interior.iter() {
        let rho = *den.view().at(c);
        let mut msq = 0.0;
        for k in 0..3 {
            let mo = *s.fields.cons.mom[k].view().at(c);
            msq += mo * mo;
        }
        let mut bsq = 0.0;
        for k in 0..3 {
            let b = *m.bcell[k].view().at(c);
            bsq += b * b;
        }
        let kin = 0.5 * msq / rho;
        let mag = 0.5 * bsq;
        ke += kin;
        me += mag;
        ie += *nrg.view().at(c) - kin - mag; // gas internal energy = total - kinetic - magnetic
    }
    (ke, me, ie)
}

fn run(eta: f64, nu: f64) -> ((f64, f64, f64), (f64, f64, f64)) {
    let mut sim = make();
    let e0 = energies(&sim);
    let sub = NewtonianMhdSubstrateKernelSet::<HostMemory, f64, 3>::new(
        GAMMA,
        0.3,
        1.0,
        &sim.geom.allocated,
    )
    .with_resistivity(eta)
    .with_viscosity(nu);
    evolve(&mut sim, &sub, T_FINAL).expect("viscous-resistive mhd evolve failed");
    let e1 = energies(&sim);
    (e0, e1)
}

#[test]
fn resistive_viscous_mhd_runs_both_diffusivities() {
    let (i0, i1) = run(0.0, 0.0); // ideal
    let (b0, b1) = run(0.02, 0.02); // resistive + viscous (Pm = 1)

    // sanity: the seeds carry kinetic + magnetic energy.
    assert!(b0.0 > 0.0 && b0.1 > 0.0, "degenerate seed");

    // viscosity dissipates far more KINETIC energy than the ideal numerical floor.
    let ke_loss_visc = b0.0 - b1.0;
    let ke_loss_ideal = i0.0 - i1.0;
    assert!(
        ke_loss_visc > 3.0 * ke_loss_ideal,
        "viscosity not acting: KE loss viscous {ke_loss_visc:.3e} vs ideal {ke_loss_ideal:.3e}"
    );
    // resistivity dissipates far more MAGNETIC energy than the ideal floor.
    let me_loss_res = b0.1 - b1.1;
    let me_loss_ideal = i0.1 - i1.1;
    assert!(
        me_loss_res > 3.0 * me_loss_ideal,
        "resistivity not acting: magnetic-energy loss resistive {me_loss_res:.3e} vs ideal {me_loss_ideal:.3e}"
    );
    // the dissipations (viscous + Ohmic) raised the gas internal energy.
    assert!(
        b1.2 > b0.2,
        "viscous heating did not warm the gas: internal {} -> {}",
        b0.2,
        b1.2
    );
    // and everything stayed finite.
    assert!(
        b1.0.is_finite() && b1.1.is_finite() && b1.2.is_finite(),
        "state went non-finite"
    );
}
