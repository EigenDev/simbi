// =============================================================================
// substrate_dgeneric_smoke.rs
//
// proves the D-generic AdiabaticSubstrateKernelSet<Mem, Sc, const D> actually
// executes at D > 1 through the real `evolve()` loop — the multi-dimensional
// case. one struct, instantiated at D=2 and D=3.
//
// the 2D test is also a correctness proof: a centered, x<->y-symmetric pressure pulse
// evolved by an isotropic scheme (one per-dir face-flux kernel serving dir 0 and 1)
// stays symmetric to ~machine precision. a miswired dir-1 flux / godunov divergence
// breaks the symmetry immediately, so symmetry certifies both sweep axes.
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::{Domain, Tensor};
use symbi_geometry::Cartesian;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::{Prim, PrimG};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;

// the max asymmetry of the 3D primitive state under the x<->y and x<->z axis swaps.
// an isotropic scheme (one per-dir flux/godunov serving dir 0, 1, 2) on a swap-
// symmetric IC holds both symmetries to ~machine precision: x<->y proves dir-0 ==
// dir-1, x<->z proves dir-0 == dir-2, so all three sweep axes are wired identically.
// scalars (rho, pre) are invariant; velocity components permute with the axes.
fn max_3d_asymmetry(
    interior: &Domain<3>,
    rho: impl Fn([isize; 3]) -> f64,
    pre: impl Fn([isize; 3]) -> f64,
    vel: impl Fn(usize, [isize; 3]) -> f64,
) -> f64 {
    let mut m = 0.0_f64;
    for c in interior.iter() {
        for &(a, b) in &[(0usize, 1usize), (0, 2)] {
            let mut s = c;
            s.swap(a, b);
            let o = 3 - a - b; // the untouched axis
            m = m.max((rho(c) - rho(s)).abs());
            m = m.max((pre(c) - pre(s)).abs());
            m = m.max((vel(a, c) - vel(b, s)).abs());
            m = m.max((vel(b, c) - vel(a, s)).abs());
            m = m.max((vel(o, c) - vel(o, s)).abs());
        }
    }
    m
}

#[test]
fn adiabatic_2d_runs_and_stays_xy_symmetric() {
    type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let n = 32usize;
    let dx = 1.0 / n as f64;
    // a centered Gaussian overpressure, v = 0. symmetric under x<->y (the center is
    // the domain center and r^2 = (x-0.5)^2 + (y-0.5)^2 is i<->j invariant).
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("2D sim construction failed")
        .set_initial(|x| {
            let r2 = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
            let pre = 1.0 + 3.0 * (-r2 / 0.01).exp();
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(pre))
        })
        .build();

    let mass0: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .sum();

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.05).expect("2D adiabatic evolution failed");

    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let mut max_speed = 0.0_f64;
    let mut max_asym = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        let vx = *sim.fields.prim.vel[0].view().at(c);
        let vy = *sim.fields.prim.vel[1].view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        max_speed = max_speed.max((vx * vx + vy * vy).sqrt());
        // x<->y symmetry: scalar fields invariant, vx(i,j) == vy(j,i) (velocity
        // components swap under the reflection). the swapped coord is interior too.
        let t = [c[1], c[0]];
        max_asym = max_asym.max((rho - *sim.fields.prim.rho.view().at(t)).abs());
        max_asym = max_asym.max((p - *pre.view().at(t)).abs());
        max_asym = max_asym.max((vx - *sim.fields.prim.vel[1].view().at(t)).abs());
    }

    // both sweep axes wired correctly => symmetry held to ~machine precision.
    assert!(
        max_asym < 1e-12,
        "x<->y symmetry broken: max asymmetry {max_asym:e}"
    );
    // the pulse actually drove flow in the plane (a real solve).
    assert!(
        max_speed > 0.05,
        "no flow developed (max speed {max_speed})"
    );
    // mass conserved on the periodic box.
    let mass1: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .sum();
    assert!(
        (mass1 - mass0).abs() < 1e-9 * mass0,
        "mass drift {:e}",
        mass1 - mass0
    );

    println!(
        "ADIABATIC 2D: {} steps to t={:.3}, max asym {:e}, max speed {:.3}, mass drift {:e}",
        sim.iteration,
        sim.time,
        max_asym,
        max_speed,
        (mass1 - mass0) / mass0,
    );
}

#[test]
fn adiabatic_3d_runs_and_stays_symmetric() {
    type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let n = 16usize;
    let dx = 1.0 / n as f64;
    // a centered, permutation-symmetric overpressure blast, v = 0.
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n, n, n])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("3D sim construction failed")
        .set_initial(|x| {
            let r2: f64 = (0..3).map(|kk| (x[kk] - 0.5).powi(2)).sum();
            let pre = 1.0 + 3.0 * (-r2 / 0.02).exp();
            Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(pre))
        })
        .build();

    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.03).expect("3D adiabatic evolution failed");

    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let mut max_speed = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        let v: f64 = (0..3)
            .map(|k| sim.fields.prim.vel[k].view().at(c).powi(2))
            .sum();
        max_speed = max_speed.max(v.sqrt());
    }
    let max_asym = max_3d_asymmetry(
        &sim.geom.interior,
        |c| *sim.fields.prim.rho.view().at(c),
        |c| *pre.view().at(c),
        |k, c| *sim.fields.prim.vel[k].view().at(c),
    );
    assert!(
        max_asym < 1e-12,
        "3D x<->y / x<->z symmetry broken: max asymmetry {max_asym:e}"
    );
    assert!(
        max_speed > 0.05,
        "3D blast did not develop flow (max speed {max_speed})"
    );
    println!(
        "ADIABATIC 3D: {} steps to t={:.3}, max asym {:e}, max speed {:.3}",
        sim.iteration, sim.time, max_asym, max_speed
    );
}

#[test]
fn iso_3d_runs_and_stays_symmetric() {
    type Sim = SimState<IsoNewtonian, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let cs = 1.0_f64;
    let n = 16usize;
    let dx = 1.0 / n as f64;
    // a centered, permutation-symmetric density bump (p = cs^2*rho drives the flow), v = 0.
    let mut sim = Sim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([n, n, n])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("3D iso sim construction failed")
        .set_initial(|x| {
            let r2: f64 = (0..3).map(|kk| (x[kk] - 0.5).powi(2)).sum();
            PrimG::<f64, 3, IsoModel>::isothermal(
                Density(1.0 + 0.5 * (-r2 / 0.02).exp()),
                Tensor::new([0.0; 3]),
            )
        })
        .build();

    let mass0: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .sum();

    let sub = IsoSubstrateKernelSet::<HostMemory, f64, 3>::new(cs, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.03).expect("3D iso evolution failed");

    let mut max_speed = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        let v: f64 = (0..3)
            .map(|k| sim.fields.prim.vel[k].view().at(c).powi(2))
            .sum();
        max_speed = max_speed.max(v.sqrt());
    }
    // iso pressure is the substrate-owned field (p = cs^2*rho), symmetric iff rho is.
    let max_asym = max_3d_asymmetry(
        &sim.geom.interior,
        |c| *sim.fields.prim.rho.view().at(c),
        |c| *sub.pre.view().at(c),
        |k, c| *sim.fields.prim.vel[k].view().at(c),
    );
    assert!(
        max_asym < 1e-12,
        "3D iso x<->y / x<->z symmetry broken: max asymmetry {max_asym:e}"
    );
    assert!(
        max_speed > 0.02,
        "3D iso bump did not develop flow (max speed {max_speed})"
    );
    let mass1: f64 = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .sum();
    assert!(
        (mass1 - mass0).abs() < 1e-9 * mass0,
        "iso mass drift {:e}",
        mass1 - mass0
    );
    println!(
        "ISO 3D: {} steps to t={:.3}, max asym {:e}, max speed {:.3}",
        sim.iteration, sim.time, max_asym, max_speed
    );
}

#[test]
fn rhd_3d_runs_and_stays_symmetric() {
    type Sim = SimState<Rhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let gamma = 5.0 / 3.0;
    let n = 16usize;
    let dx = 1.0 / n as f64;
    // centered, permutation-symmetric pressure pulse, v = 0 (reuse the seeding prim; v=0 c2p
    // round-trips => D = rho, S = 0, tau = rho*h - p - rho).
    let mut sim = Sim::build(Rhd, IdealGas { gamma }, Cartesian)
        .cells([n, n, n])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("3D RHD sim construction failed")
        .set_initial(|x| {
            let r2: f64 = (0..3).map(|kk| (x[kk] - 0.5).powi(2)).sum();
            let pre = 1.0 + 2.0 * (-r2 / 0.02).exp();
            Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(pre))
        })
        .build();

    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 3>::new(gamma, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.03).expect("3D RHD evolution failed");

    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let mut max_speed = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        let speed: f64 = (0..3)
            .map(|k| sim.fields.prim.vel[k].view().at(c).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(speed < 1.0, "superluminal speed {speed} at {c:?}");
        max_speed = max_speed.max(speed);
    }
    let max_asym = max_3d_asymmetry(
        &sim.geom.interior,
        |c| *sim.fields.prim.rho.view().at(c),
        |c| *pre.view().at(c),
        |k, c| *sim.fields.prim.vel[k].view().at(c),
    );
    assert!(
        max_asym < 1e-12,
        "3D RHD x<->y / x<->z symmetry broken: max asymmetry {max_asym:e}"
    );
    assert!(
        max_speed > 0.005,
        "3D RHD pulse did not develop flow (max speed {max_speed})"
    );
    println!(
        "RHD 3D: {} steps to t={:.3}, max asym {:e}, max speed {:.4}",
        sim.iteration, sim.time, max_asym, max_speed
    );
}

#[test]
fn rhd_2d_runs_and_stays_xy_symmetric() {
    // the multi-dimensional RHD CFL: the D-generic wave-speed map folds the per-axis
    // relativistic Davis speed over both axes, so RHD runs at 2D. same x<->y symmetry
    // proof as Newton — the relativistic c2p (iterative Newton) + per-axis wave speeds
    // are isotropic, so a symmetric IC stays symmetric to ~machine precision.
    type Sim = SimState<Rhd, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let gamma = 5.0 / 3.0;
    let n = 32usize;
    let dx = 1.0 / n as f64;
    // centered, x<->y-symmetric pressure pulse, v = 0 => W = 1 (reuse the seeding prim; v=0 c2p
    // round-trips => D = rho, S = 0, tau = rho*h - p - rho with h = 1 + gamma/(gamma-1)*p/rho).
    let mut sim = Sim::build(Rhd, IdealGas { gamma }, Cartesian)
        .cells([n, n])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .allocate()
        .expect("2D RHD sim construction failed")
        .set_initial(|x| {
            let r2 = (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2);
            let pre = 1.0 + 2.0 * (-r2 / 0.01).exp();
            Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0]), Pressure(pre))
        })
        .build();

    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 2>::new(gamma, 0.4, &sim.geom.allocated);
    evolve(&mut sim, &sub, 0.04).expect("2D RHD evolution failed");

    let pre = sim.fields.prim.pre_field().expect("prim.pre");
    let mut max_speed = 0.0_f64;
    let mut max_asym = 0.0_f64;
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        let vx = *sim.fields.prim.vel[0].view().at(c);
        let vy = *sim.fields.prim.vel[1].view().at(c);
        assert!(rho.is_finite() && rho > 0.0, "bad density {rho} at {c:?}");
        assert!(p.is_finite() && p > 0.0, "bad pressure {p} at {c:?}");
        let speed = (vx * vx + vy * vy).sqrt();
        assert!(speed < 1.0, "superluminal speed {speed} at {c:?}");
        max_speed = max_speed.max(speed);
        // x<->y symmetry: scalars invariant, vx(i,j) == vy(j,i).
        let t = [c[1], c[0]];
        max_asym = max_asym.max((rho - *sim.fields.prim.rho.view().at(t)).abs());
        max_asym = max_asym.max((p - *pre.view().at(t)).abs());
        max_asym = max_asym.max((vx - *sim.fields.prim.vel[1].view().at(t)).abs());
    }
    assert!(
        max_asym < 1e-12,
        "x<->y symmetry broken: max asymmetry {max_asym:e}"
    );
    assert!(
        max_speed > 0.01,
        "no relativistic flow developed (max speed {max_speed})"
    );
    println!(
        "RHD 2D: {} steps to t={:.3}, max asym {:e}, max speed {:.4}",
        sim.iteration, sim.time, max_asym, max_speed,
    );
}
