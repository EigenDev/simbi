// =============================================================================
// substrate_rhd_sod.rs
//
// the CARTESIAN sharp relativistic Sod through the real evolve() loop — the
// RHD-robustness control that the smooth-pulse dgeneric smoke did not cover.
// it proves the BASE RHD scheme (full relativistic HLLE wave speeds + flux +
// the iterative pressure-Newton c2p) handles a strong discontinuity and keeps
// the flow subluminal — i.e., the relativistic structure is correct.
//
// Marti & Mueller relativistic Sod (gamma=5/3): (rho,p)=(1,1)|(0.125,0.1), v=0.
// for v=0: D = rho*W = rho, S = 0, tau = rho*h*W^2 - p - D = p/(gamma-1) (W=1).
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

#[test]
fn rhd_cartesian_sharp_sod_stays_subluminal() {
    type Sim = SimState<Rhd, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let gamma = 5.0 / 3.0;
    let n = 128usize;
    let dx = 1.0 / n as f64;
    // Marti & Mueller sharp Sod, v = 0 (reuse the seeding prim; v=0 c2p round-trips exactly).
    let mut sim = Sim::build(Rhd, IdealGas { gamma }, Cartesian)
        .cells([n])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("rhd sim construction failed")
        .set_initial(|x| {
            let (rho, pre) = if x[0] < 0.5 { (1.0, 1.0) } else { (0.125, 0.1) };
            Prim {
                rho,
                vel: Tensor::new([0.0]),
                pre,
            }
        })
        .build();

    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(gamma, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.05).expect("rhd evolution failed");

    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let mut max_vel = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        let v = *sim.fields.prim.vel[0].view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        assert!(v.abs() < 1.0, "superluminal velocity {v} at {c:?}");
        max_vel = max_vel.max(v.abs());
    }
    // the shock accelerates the gas mildly-relativistically, and the relativistic
    // HLLE keeps it strictly subluminal — no NaN, no floor needed.
    assert!(
        max_vel > 0.1,
        "gas did not accelerate (max |v| = {max_vel})"
    );
    println!(
        "RHD CARTESIAN SHARP SOD: {} steps to t={:.3}, max |v| {:.3}",
        sim.iteration, sim.time, max_vel
    );
}
