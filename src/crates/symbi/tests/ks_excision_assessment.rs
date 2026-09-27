// =============================================================================
// ks_excision_assessment.rs
//
// an opt-in measurement, ignored by default: the python test_cartesian_ks_mhd_3d
// fixture (rmhd on the cartesian schwarzschild kerr-schild chart, mass 1,
// 32^3 on [-5, 5]^3, rho 1, p 0.1, v 0, B^z = 0.05 / sqrt(gamma) on the
// staggered faces with the kernel's r >= M/2 clamp and the cell field from
// the face average, gamma 4/3, plm theta 1.5, hlle, rk2, cfl 0.3, excision at
// r_ks < 1.4 with floor scales 1.0) built and driven as the python runner
// drives it: Hierarchy::single, primed, evolved to t = 1 with the fofc
// horizon reporting radius 2M set at process start. one line per accepted
// step: candidate and accepted timestep, the retries between them
// (log2 of their ratio, exact since a retry halves), simulation time and
// elapsed wall time. the state at t = 1 (primitives and interior faces) is
// written to KS_DUMP; a run stopped by the step cap or the wall-clock
// watchdog writes no state and reports itself incomplete.
//
// inputs (environment):
// - KS_CT: contact (default) or uct;
// - KS_MAX_STEPS: the step cap (400, the python test's);
// - KS_WALL_SECONDS: the wall-clock watchdog, checked after each accepted step;
// - KS_DUMP: the output path of the state at t = 1.
//
// usage: scripts/ks_excision_assessment.sh builds it at two revisions, runs
// each arm in a fresh process and compares the dumps with
// scripts/ks_excision_compare.py.
//  KS_CT=uct KS_DUMP=/tmp/uct.bin cargo test -p symbi --test ks_excision_assessment -- --ignored --nocapture
// =============================================================================
use std::ops::ControlFlow;
use std::time::Instant;

use symbi::prelude::SimSubstrate;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi::sim::substrate_seam::{KernelSet, Solver, WithExcision, WithViscosity};
use symbi_algebra::Tensor;
use symbi_geometry::SchwarzschildKSCartesian;
use symbi_hydro::eos::{EosSelect, IdealGas};
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Rmhd, 3, 3, SchwarzschildKSCartesian<f64>, EosSelect<f64>, CpuSpace, HostMemory>;

const RES: usize = 32;
const L: f64 = 5.0;
const MASS: f64 = 1.0;
const BZ0: f64 = 0.05;
const GAMMA: f64 = 4.0 / 3.0;
const THETA: f64 = 1.5;
const CFL: f64 = 0.3;
const R_EXC: f64 = 1.4;
const T_END: f64 = 1.0;

/// the fixture's sqrt(gamma) with the kernel's r >= M/2 clamp
fn sqrtg(x: f64, y: f64, z: f64) -> f64 {
    let r_true = (x * x + y * y + z * z).sqrt();
    let r = r_true.max(0.5 * MASS);
    let ll2 = (r_true / r).powi(2);
    (1.0 + (2.0 * MASS / r) * ll2).sqrt()
}

#[test]
#[ignore]
fn assess() {
    let ct = match std::env::var("KS_CT").as_deref() {
        Ok("uct") => CtMethod::Uct,
        _ => CtMethod::Contact,
    };
    let max_steps: u64 = std::env::var("KS_MAX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(400);
    let wall: f64 = std::env::var("KS_WALL_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(1800.0);
    let dump = std::env::var("KS_DUMP").expect("KS_DUMP names the output path");
    eprintln!(
        "CONFIG res={RES} box=[-{L},{L}]^3 mass={MASS} bz0={BZ0} gamma={GAMMA} theta={THETA} solver=hlle ct={ct:?} \
         rk2 cfl={CFL} r_exc={R_EXC} floors=1.0,1.0 horizon_report_radius={} t_end={T_END} max_steps={max_steps} \
         wall={wall}s",
        2.0 * MASS
    );
    let dx = 2.0 * L / RES as f64;
    let bz = |x: f64, y: f64, z: f64| BZ0 / sqrtg(x, y, z);
    symbi::regimes::fofc::fofc_set_horizon_radius(2.0 * MASS);
    let sim = Sim::build(Rmhd, EosSelect::Ideal(IdealGas { gamma: GAMMA }), SchwarzschildKSCartesian { mass: MASS })
        .cells([RES; 3])
        .origin([-L; 3])
        .spacing([dx; 3])
        .ghosts(2)
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y, z]| {
            let cell_bz = 0.5 * (bz(x, y, z - 0.5 * dx) + bz(x, y, z + 0.5 * dx));
            MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.1)), Tensor::new([0.0, 0.0, cell_bz]))
        })
        .seed_faces(|axis, [x, y, z]| if axis == 2 { bz(x, y, z) } else { 0.0 })
        .build();
    let k = sim
        .substrate()
        .theta(THETA)
        .with_solver(Solver::Hlle)
        .expect("solver")
        .ct_method(ct)
        .with_viscosity(0.0)
        .with_resistivity(0.0)
        .with_excision(R_EXC, 1.0, 1.0);
    let mut hier = Hierarchy::single(sim, k);
    hier.prime();
    let clock = Instant::now();
    // the candidate a step starts from: the level's cfl step, capped at the time remaining
    macro_rules! candidate_of {
        ($h:expr) => {{
            let l = &$h.levels[0];
            l.kernels.cfl(&l.state).min(T_END - l.state.time)
        }};
    }
    let mut candidate = candidate_of!(hier);
    let mut last_logged = u64::MAX;
    let mut stopped = None;
    hier.evolve_with_callback(T_END, 1, |h| {
        let s = &h.levels[0].state;
        // the driver reports the final state once more after the last step
        if s.iteration != last_logged {
            last_logged = s.iteration;
            eprintln!(
                "STEP n={} t={:.9e} dt_candidate={:.6e} dt_accepted={:.6e} retries={} elapsed={:.2}",
                s.iteration,
                s.time,
                candidate,
                s.dt,
                (candidate / s.dt).log2().round(),
                clock.elapsed().as_secs_f64()
            );
        }
        candidate = candidate_of!(h);
        if s.iteration >= max_steps {
            stopped = Some("step cap");
            return ControlFlow::Break(());
        }
        if clock.elapsed().as_secs_f64() > wall {
            stopped = Some("wall-clock watchdog");
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    })
    .expect("evolve");
    let s = &hier.levels[0].state;
    if let Some(why) = stopped.filter(|_| s.time < T_END) {
        eprintln!("INCOMPLETE stopped by {why} at t={:.9e} after {} steps", s.time, s.iteration);
        return;
    }
    assert!(s.time >= T_END, "the run ended at t = {} short of {T_END}", s.time);
    // layout: time, steps, then rho, pre, vel_0..2 over the interior (last axis fastest), then
    // the interior faces of each axis
    let mut out: Vec<f64> = vec![s.time, s.iteration as f64];
    let pre = s.fields.prim.pre_field().unwrap();
    for f in [&s.fields.prim.rho, pre, &s.fields.prim.vel[0], &s.fields.prim.vel[1], &s.fields.prim.vel[2]] {
        for c in s.geom.interior.iter() {
            out.push(*f.view().at(c));
        }
    }
    let mhd = s.fields.mhd.as_ref().unwrap();
    for d in 0..3 {
        for c in s.geom.interior.extend(d, 0, 1).iter() {
            out.push(*mhd.bface[d].view().at(c));
        }
    }
    let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&dump, bytes).unwrap();
    eprintln!("COMPLETE t={:.9e} steps={} dump={dump}", s.time, s.iteration);
}
