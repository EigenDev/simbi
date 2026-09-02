// =============================================================================
// mesh_motion_uni_grid.rs
//
// homologous mesh-motion gates (single grid, cartesian, newtonian hydro):
//
// static equivalence — `MotionState::homologous(1.0, 0.0)` must reproduce the
// default static mesh bit-for-bit: every motion term enters as an exact
// identity (vface = 0*x, |s - 0|, dx*1.0, dilution -dt*(0*u)), so any
// divergence means a motion scalar leaked into the static arithmetic.
//
// free expansion — the exact solution: cold gas coasting homologously
// (v = a_dot*x, uniform rho/p, a_ddot = 0) stays self-similar; on the
// comoving grid the profile is static up to the adiabatic power laws
// rho = rho0/a^3, p = p0/a^(3*gamma), v = a_dot*x. pins the full moving-mesh
// pipeline (ALE flux, relative-speed cfl, physical-width divergence, dilution
// source, stage-time a) against the one problem with a closed-form answer.
// asserted on the inner core (outflow ghosts cannot represent the linear
// velocity profile; the boundary error advects inward at the comoving sound
// speed, which stays far from the core over the run).
//
// usage:
//  cargo test -p symbi --release --test mesh_motion_uni_grid
// =============================================================================

use symbi::regimes::substrate::IsoSubstrateKernelSet;
use symbi::regimes::substrate_kernels::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi::sim::tracers::{ContinuousTracerSet, TracerSet, cell_container_id};
use symbi_algebra::Tensor;
use symbi_geometry::{Cartesian, MotionState, Spherical};
use symbi_hydro::eos::{IdealGas, Isothermal};
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::regime::Regime;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::Prim;
use symbi_sim::mass_transport::ItoOrder;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 32;

type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 3>;

fn build(motion: MotionState<f64>, cfl: f64, fill: impl Fn(&Sim)) -> (Sim, Kset) {
    let dx = 1.0 / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(cfl)
        .finish()
        .unwrap();
    sim.motion = motion;
    fill(&sim);
    let k = Kset::new(GAMMA, cfl, &sim.geom.allocated);
    (sim, k)
}

fn set_prim(sim: &Sim, c: [isize; 3], prim: &Prim<f64, 3>) {
    let cnrg = sim.fields.cons.nrg_field().unwrap();
    let cons = Regime::to_conserved(&sim.physics.regime, &sim.physics.eos, prim);
    sim.fields.cons.den.view_mut().set(c, cons.den());
    for dd in 0..3 {
        sim.fields.cons.mom[dd].view_mut().set(c, cons.mom()[dd]);
    }
    cnrg.view_mut().set(c, cons.nrg());
}

/// a smooth non-trivial state (off-center gaussian pulse in a shear) so the
/// equivalence test exercises real flux arithmetic on a non-equilibrium state.
fn fill_pulse(sim: &Sim) {
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        let r2 = (x[0] - 0.1).powi(2) + x[1] * x[1] + x[2] * x[2];
        let prim = Prim::adiabatic(
            Density(1.0 + 0.5 * (-r2 / 0.02).exp()),
            Tensor::new([0.3 * x[1], -0.2, 0.1]),
            Pressure(1.0 + 0.2 * (-r2 / 0.02).exp()),
        );
        set_prim(sim, c, &prim);
    }
}

#[test]
fn homologous_zero_rate_is_bit_identical_to_static() {
    let (mut a, ka) = build(MotionState::static_mesh(), CFL, fill_pulse);
    let (mut b, kb) = build(MotionState::homologous(1.0, 0.0), CFL, fill_pulse);
    evolve(&mut a, &ka, 0.1).unwrap();
    evolve(&mut b, &kb, 0.1).unwrap();

    let bnrg = b.fields.cons.nrg_field().unwrap();
    let anrg = a.fields.cons.nrg_field().unwrap();
    for c in a.geom.interior.iter() {
        assert_eq!(
            *a.fields.cons.den.view().at(c),
            *b.fields.cons.den.view().at(c),
            "den diverged at {c:?}"
        );
        for dd in 0..3 {
            assert_eq!(
                *a.fields.cons.mom[dd].view().at(c),
                *b.fields.cons.mom[dd].view().at(c),
                "mom{dd} diverged at {c:?}"
            );
        }
        assert_eq!(
            *anrg.view().at(c),
            *bnrg.view().at(c),
            "nrg diverged at {c:?}"
        );
    }
}

/// run free expansion at `cfl` and return the worst core-cell deviations
/// (rho * a^3, p * a^(3 gamma), |v - adot * x| / v_edge).
fn free_expansion_errors(cfl: f64) -> (f64, f64, f64) {
    let (rho0, p0, adot) = (1.0, 1e-3, 0.25);
    let t_final = 2.0;
    let (mut sim, k) = build(MotionState::homologous(1.0, adot), cfl, |s| {
        for c in s.geom.interior.iter() {
            let x = s.geom.centroid(c);
            let prim = Prim::adiabatic(
                Density(rho0),
                Tensor::new([adot * x[0], adot * x[1], adot * x[2]]),
                Pressure(p0),
            );
            set_prim(s, c, &prim);
        }
    });
    evolve(&mut sim, &k, t_final).unwrap();

    let a = sim.motion.a;
    assert!(
        (a - (1.0 + adot * t_final)).abs() < 1e-12,
        "scale factor drifted: a = {a}"
    );

    let pre = sim.fields.prim.pre_field().unwrap();
    let v_scale = adot * 0.5;
    let mut worst_rho = 0.0f64;
    let mut worst_pre = 0.0f64;
    let mut worst_vel = 0.0f64;
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        if x.iter().any(|&xi| xi.abs() > 0.3) {
            continue;
        }
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        worst_rho = worst_rho.max((rho * a.powi(3) / rho0 - 1.0).abs());
        worst_pre = worst_pre.max((p * a.powf(3.0 * GAMMA) / p0 - 1.0).abs());
        for dd in 0..3 {
            let v = *sim.fields.prim.vel[dd].view().at(c);
            worst_vel = worst_vel.max((v - adot * x[dd]).abs() / v_scale);
        }
    }
    eprintln!(
        "[mesh-motion] cfl {cfl}: a = {a:.3}  |d(rho a^3)| {worst_rho:.2e}  \
         |d(p a^(3g))| {worst_pre:.2e}  |dv|/v {worst_vel:.2e}"
    );
    (worst_rho, worst_pre, worst_vel)
}

/// isothermal free expansion: no energy law, cs fixed (no adiabatic cooling
/// by construction), uniform rho — exact coasting. asserts rho * a^3 and the
/// homologous velocity profile on the inner core.
#[test]
fn iso_free_expansion_stays_self_similar() {
    type IsoSim = SimState<IsoNewtonian, 3, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
    let (cs, adot, t_final) = (0.05, 0.25, 2.0);
    let dx = 1.0 / N as f64;
    let mut sim = IsoSim::build(IsoNewtonian, Isothermal { cs }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .finish()
        .unwrap();
    sim.motion = MotionState::homologous(1.0, adot);
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        sim.fields.cons.den.view_mut().set(c, 1.0);
        for dd in 0..3 {
            sim.fields.cons.mom[dd].view_mut().set(c, adot * x[dd]);
        }
    }
    let k = IsoSubstrateKernelSet::<HostMemory, f64, 3>::new(cs, CFL, &sim.geom.allocated);
    evolve(&mut sim, &k, t_final).unwrap();

    let a = sim.motion.a;
    let mut worst_rho = 0.0f64;
    let mut worst_vel = 0.0f64;
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        if x.iter().any(|&xi| xi.abs() > 0.3) {
            continue;
        }
        let rho = *sim.fields.prim.rho.view().at(c);
        worst_rho = worst_rho.max((rho * a.powi(3) - 1.0).abs());
        for dd in 0..3 {
            let v = *sim.fields.prim.vel[dd].view().at(c);
            worst_vel = worst_vel.max((v - adot * x[dd]).abs() / (adot * 0.5));
        }
    }
    eprintln!(
        "[mesh-motion] iso: a = {a:.3}  |d(rho a^3)| {worst_rho:.2e}  |dv|/v {worst_vel:.2e}"
    );
    assert!(
        worst_rho < 2e-2,
        "iso density broke self-similarity: {worst_rho:.3e}"
    );
    assert!(
        worst_vel < 1e-3,
        "iso velocity left the homologous profile: {worst_vel:.3e}"
    );
}

/// rhd free expansion: relativistic coasting is the same exact solution —
/// every parcel moves at constant velocity, so the comoving profile is static
/// and the rest-frame density per comoving cell dilutes as a^-3 (W(x) is
/// time-independent). v_edge = adot/2 = 0.125c keeps the run honest but
/// subluminal everywhere.
#[test]
fn rhd_free_expansion_stays_self_similar() {
    type RhdSim = SimState<Rhd, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    let (rho0, p0, adot, t_final) = (1.0, 1e-3, 0.25, 2.0);
    let dx = 1.0 / N as f64;
    let mut sim = RhdSim::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .finish()
        .unwrap();
    sim.motion = MotionState::homologous(1.0, adot);
    {
        let cnrg = sim.fields.cons.nrg_field().unwrap();
        for c in sim.geom.interior.iter() {
            let x = sim.geom.centroid(c);
            let prim = Prim::adiabatic(
                Density(rho0),
                Tensor::new([adot * x[0], adot * x[1], adot * x[2]]),
                Pressure(p0),
            );
            let cons = Regime::to_conserved(&sim.physics.regime, &sim.physics.eos, &prim);
            sim.fields.cons.den.view_mut().set(c, cons.den());
            for dd in 0..3 {
                sim.fields.cons.mom[dd].view_mut().set(c, cons.mom()[dd]);
            }
            cnrg.view_mut().set(c, cons.nrg());
        }
    }
    let k = RhdSubstrateKernelSet::<HostMemory, f64, 3>::new(GAMMA, CFL, &sim.geom.allocated);
    evolve(&mut sim, &k, t_final).unwrap();

    let a = sim.motion.a;
    let pre = sim.fields.prim.pre_field().unwrap();
    let mut worst_rho = 0.0f64;
    let mut worst_pre = 0.0f64;
    let mut worst_vel = 0.0f64;
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        if x.iter().any(|&xi| xi.abs() > 0.3) {
            continue;
        }
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        worst_rho = worst_rho.max((rho * a.powi(3) / rho0 - 1.0).abs());
        worst_pre = worst_pre.max((p * a.powf(3.0 * GAMMA) / p0 - 1.0).abs());
        for dd in 0..3 {
            let v = *sim.fields.prim.vel[dd].view().at(c);
            worst_vel = worst_vel.max((v - adot * x[dd]).abs() / (adot * 0.5));
        }
    }
    eprintln!(
        "[mesh-motion] rhd: a = {a:.3}  |d(rho a^3)| {worst_rho:.2e}  \
         |d(p a^(3g))| {worst_pre:.2e}  |dv|/v {worst_vel:.2e}"
    );
    assert!(
        worst_rho < 2e-2,
        "rhd density broke self-similarity: {worst_rho:.3e}"
    );
    assert!(
        worst_pre < 8e-2,
        "rhd pressure broke self-similarity: {worst_pre:.3e}"
    );
    assert!(
        worst_vel < 1e-2,
        "rhd velocity left the homologous profile: {worst_vel:.3e}"
    );
}

/// uniform translation, the strongest possible moving-mesh demonstration: a
/// density pulse moving with the grid is an at-rest contact in the grid
/// frame, and hllc resolves at-rest contacts exactly — the pulse must be
/// preserved to roundoff while any static grid would diffuse it under
/// advection. uniform p, v = vtrans everywhere; only rho varies.
#[test]
fn translated_contact_is_preserved_exactly_on_the_comoving_grid() {
    let vtrans = 0.5;
    let fill = |s: &Sim| {
        for c in s.geom.interior.iter() {
            let x = s.geom.centroid(c);
            let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let prim = Prim::adiabatic(
                Density(1.0 + 0.5 * (-r2 / 0.02).exp()),
                Tensor::new([vtrans, 0.0, 0.0]),
                Pressure(1.0),
            );
            set_prim(s, c, &prim);
        }
    };
    let dx = 1.0 / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .origin([-0.5; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .finish()
        .unwrap();
    sim.motion = MotionState::uniform(1.0, vtrans);
    fill(&sim);
    let seed = TracerSet::<3>::seed_stratified(
        &[([-0.25; 3], [0.5; 3])],
        &[cell_container_id(0, 0)],
        &[512],
        1.0,
    );
    let continuous =
        ContinuousTracerSet::<3, HostMemory>::from_discrete(&seed, ItoOrder::Two).unwrap();
    let continuous_initial: [Vec<f64>; 3] = std::array::from_fn(|dd| unsafe {
        std::slice::from_raw_parts(continuous.x[dd].as_ptr::<f64>(), continuous.len).to_vec()
    });
    sim.continuous_tracers = Some(continuous);
    let rho0: Vec<f64> = sim
        .geom
        .interior
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(c))
        .collect();
    let k = Kset::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(Solver::Hllc)
        .expect("valid solver/regime pair");
    evolve(&mut sim, &k, 0.5).unwrap();

    assert!(
        (sim.motion.a - 1.0).abs() == 0.0,
        "uniform translation scaled the mesh"
    );
    let mut worst = 0.0f64;
    for (ii, c) in sim.geom.interior.iter().enumerate() {
        let rho = *sim.fields.cons.den.view().at(c);
        worst = worst.max((rho - rho0[ii]).abs());
    }
    eprintln!("[mesh-motion] translated contact: max |d rho| {worst:.2e}");
    // hllc resolves the at-rest contact exactly in exact arithmetic; the
    // measured drift (~1e-9 over ~100 steps) is accumulated roundoff through
    // the reconstruction + energy-flux nonlinearity — seven orders below the
    // ~1e-2 advection diffusion a static grid produces on the same problem.
    assert!(
        worst < 1e-7,
        "the comoving contact diffused ({worst:.3e}) — hllc at zero relative speed \
         must be exact to accumulated roundoff"
    );
    let continuous = sim.continuous_tracers.as_ref().unwrap();
    for dd in 0..3 {
        let accepted =
            unsafe { std::slice::from_raw_parts(continuous.x[dd].as_ptr::<f64>(), continuous.len) };
        let shift = if dd == 0 { vtrans * sim.time } else { 0.0 };
        let residuals: Vec<_> = accepted
            .iter()
            .zip(&continuous_initial[dd])
            .map(|(actual, initial)| actual - initial - shift)
            .collect();
        let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let variance = residuals
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / (residuals.len() - 1) as f64;
        let standard_error = (variance / residuals.len() as f64).sqrt();
        assert!(
            mean.abs() <= 8.0 * standard_error + 1.0e-12,
            "continuous tracer ensemble drifted from the translating mesh: \
             mean={mean:.3e}, stderr={standard_error:.3e}"
        );
    }
}

/// spherical homologous free expansion: the same coasting solution in
/// spherical coordinates (only r expands; v_theta = 0), exercising the
/// curvilinear geometric source against physical radii and the
/// radial-axis-only grid velocity. theta-uniform, asserted on the radial
/// core away from both r boundaries.
#[test]
fn spherical_homologous_free_expansion_stays_self_similar() {
    type SphSim = SimState<Newtonian, 2, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
    let (rho0, p0, adot, t_final) = (1.0, 1e-3, 0.25, 2.0);
    let (nr, nt) = (64usize, 8usize);
    let (r_lo, r_hi) = (0.5, 2.0);
    let (th_lo, th_hi) = (
        std::f64::consts::FRAC_PI_4,
        3.0 * std::f64::consts::FRAC_PI_4,
    );
    let mut sim = SphSim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([nr, nt])
        .origin([r_lo, th_lo])
        .spacing([(r_hi - r_lo) / nr as f64, (th_hi - th_lo) / nt as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .finish()
        .unwrap();
    sim.motion = MotionState::homologous(1.0, adot);
    sim.seed_cells(|x| {
        Prim::adiabatic(Density(rho0), Tensor::new([adot * x[0], 0.0]), Pressure(p0))
    });
    let k = AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated);
    evolve(&mut sim, &k, t_final).unwrap();

    let a = sim.motion.a;
    assert!(
        (a - (1.0 + adot * t_final)).abs() < 1e-12,
        "scale factor drifted: a = {a}"
    );
    let pre = sim.fields.prim.pre_field().unwrap();
    let mut worst_rho = 0.0f64;
    let mut worst_pre = 0.0f64;
    let mut worst_vel = 0.0f64;
    for c in sim.geom.interior.iter() {
        let x = sim.geom.centroid(c);
        let r = x[0];
        if !(0.8..1.7).contains(&r) {
            continue;
        }
        let rho = *sim.fields.prim.rho.view().at(c);
        let p = *pre.view().at(c);
        worst_rho = worst_rho.max((rho * a.powi(3) / rho0 - 1.0).abs());
        worst_pre = worst_pre.max((p * a.powf(3.0 * GAMMA) / p0 - 1.0).abs());
        let vr = *sim.fields.prim.vel[0].view().at(c);
        worst_vel = worst_vel.max((vr - adot * r).abs() / (adot * r_hi));
    }
    eprintln!(
        "[mesh-motion] spherical: a = {a:.3}  |d(rho a^3)| {worst_rho:.2e}  \
         |d(p a^(3g))| {worst_pre:.2e}  |dv|/v {worst_vel:.2e}"
    );
    assert!(
        worst_rho < 2e-2,
        "spherical density broke self-similarity: {worst_rho:.3e}"
    );
    assert!(
        worst_pre < 8e-2,
        "spherical pressure broke self-similarity: {worst_pre:.3e}"
    );
    assert!(
        worst_vel < 1e-2,
        "spherical velocity left the homologous profile: {worst_vel:.3e}"
    );
}

#[test]
fn free_expansion_stays_self_similar_on_the_comoving_grid() {
    let (rho_1, pre_1, vel_1) = free_expansion_errors(CFL);
    assert!(rho_1 < 2e-2, "density broke self-similarity: {rho_1:.3e}");
    assert!(pre_1 < 8e-2, "pressure broke self-similarity: {pre_1:.3e}");
    assert!(
        vel_1 < 1e-3,
        "velocity left the homologous profile: {vel_1:.3e}"
    );

    // the deviations are pure time-integration error of the a(t)-coupled terms
    // (the velocity profile is exact to roundoff-scale advection) — halving dt
    // must shrink them at second order. the 3.0 bound leaves margin under the
    // ideal 4.0 for the boundary-layer floor.
    let (rho_2, pre_2, _) = free_expansion_errors(CFL / 2.0);
    let order_rho = (rho_1 / rho_2).log2();
    let order_pre = (pre_1 / pre_2).log2();
    eprintln!("[mesh-motion] dt-order: rho {order_rho:.2}  pre {order_pre:.2}");
    assert!(
        order_rho > 1.5 && order_pre > 1.5,
        "mesh-motion time integration degraded below second order \
         (rho {order_rho:.2}, pre {order_pre:.2})"
    );
}

#[test]
fn homologous_mesh_tracers_follow_accepted_geometry() {
    let adot = 0.25;
    let (mut sim, kernels) = build(MotionState::homologous(1.0, adot), CFL, |sim| {
        for coord in sim.geom.interior.iter() {
            let x = sim.geom.centroid(coord);
            set_prim(
                sim,
                coord,
                &Prim::adiabatic(
                    Density(1.0),
                    Tensor::new(std::array::from_fn(|dd| adot * x[dd])),
                    Pressure(1.0e-3),
                ),
            );
        }
    });
    let center = N / 2;
    let owner = cell_container_id(center + N * (center + N * center), 0);
    sim.tracers = Some(TracerSet::seed_stratified(
        &[([0.0; 3], [1.0 / N as f64; 3])],
        &[owner],
        &[2048],
        1.0,
    ));
    {
        let tracers = sim.tracers.as_mut().unwrap();
        tracers.step_owner = tracers.owner.clone();
        tracers.step_flags = tracers.flags.clone();
    }
    let initial = sim.tracers.as_ref().unwrap().clone();

    evolve(&mut sim, &kernels, 0.1).unwrap();

    let accepted = sim.tracers.as_ref().unwrap();
    assert_eq!(accepted.owner, initial.owner);
    assert_eq!(accepted.id, initial.id);
    assert!(
        accepted
            .owner
            .iter()
            .all(|owner| owner.0 < (N * N * N) as u64),
        "homologous coasting sent material to a non-cell reservoir"
    );
    for (position, owner) in accepted.x.iter().zip(&accepted.owner) {
        let linear = owner.0 as usize;
        let nx = N;
        let index = [linear % nx, (linear / nx) % nx, linear / (nx * nx)];
        let expected: [f64; 3] =
            std::array::from_fn(|dd| sim.motion.a * (-0.5 + (index[dd] as f64 + 0.5) / N as f64));
        for dd in 0..3 {
            assert!(
                (position[dd] - expected[dd]).abs() < 1.0e-12,
                "tracer position used stale mesh geometry on axis {dd}: {} vs {}",
                position[dd],
                expected[dd]
            );
        }
    }
}

#[test]
fn translating_mesh_tracers_follow_accepted_ale_mass_flux() {
    let velocity = 0.25;
    let (mut sim, kernels) = build(MotionState::uniform(1.0, velocity), CFL, |sim| {
        for coord in sim.geom.interior.iter() {
            set_prim(
                sim,
                coord,
                &Prim::adiabatic(Density(1.0), Tensor::zeros(), Pressure(1.0)),
            );
        }
    });
    let center = N / 2;
    let owner = cell_container_id(center + N * (center + N * center), 0);
    sim.tracers = Some(TracerSet::seed_stratified(
        &[([0.0; 3], [1.0 / N as f64; 3])],
        &[owner],
        &[2048],
        1.0,
    ));
    {
        let tracers = sim.tracers.as_mut().unwrap();
        tracers.step_owner = tracers.owner.clone();
        tracers.step_flags = tracers.flags.clone();
    }

    evolve(&mut sim, &kernels, 0.1).unwrap();

    let accepted = sim.tracers.as_ref().unwrap();
    assert!(
        accepted.owner != [owner; 2048],
        "the setup exercised no accepted ALE face transfers"
    );
    assert!(
        accepted
            .owner
            .iter()
            .all(|owner| owner.0 < (N * N * N) as u64),
        "interior translating material reached a reservoir"
    );
    for (position, owner) in accepted.x.iter().zip(&accepted.owner) {
        let linear = owner.0 as usize;
        let index = [linear % N, (linear / N) % N, linear / (N * N)];
        let expected: [f64; 3] = std::array::from_fn(|dd| {
            -0.5 + (index[dd] as f64 + 0.5) / N as f64
                + if dd == 0 { velocity * sim.time } else { 0.0 }
        });
        for dd in 0..3 {
            assert!((position[dd] - expected[dd]).abs() < 1.0e-12);
        }
    }
}
