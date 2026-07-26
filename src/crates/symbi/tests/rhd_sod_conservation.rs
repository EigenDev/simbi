// =============================================================================
// rhd_sod_conservation.rs
//
// the RHD multi-step CONSERVATION + CFL + POSITIVITY gate (T2). complements
// substrate_rhd_sod (single end-state subluminal check) by sampling invariants
// every SAMPLE_EVERY steps under the post-phase-A/D emit path — a regression on
// consolidated IR, identity folding, or view-struct buffer ABIs will surface here
// as drift in a conserved global.
//
// IC: Marti-Mueller relativistic Sod (\gamma = 5/3): (\rho, p) = (1, 1) | (0.125, 0.1),
// v = 0. reflective walls keep the integral diagnostics fully closed (no
// boundary flux of mass / energy) — the test asserts ABSOLUTE conservation up to
// floating drift (1e-9 mass, 1e-8 energy).
//
// step count: t_final chosen so the cfl-driven loop runs \approx 200 steps. waves
// stay interior up to \approx t=0.5 at the chosen tube length; the test sets
// t_final = 0.4 to avoid the first reflection arriving at the wall (mass /
// energy stay strict-conserved without reflective-wall flux subtleties).
//
// invariants checked every SAMPLE_EVERY steps:
//   total_mass    = sum_c (cons.den[c]   * dx)   \equiv  mass_0    (rel < 1e-9)
//   total_energy  = sum_c (cons.nrg[c]   * dx)   \equiv  energy_0  (rel < 1e-8)
//   CFL bound:    dt_next * max_\lambda / dx          <=   CFL_used + 1e-12
//   \rho > 0, p > 0, |v| < 1 at every interior cell every checkpoint
// =============================================================================

use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::{KernelSet, evolve_with_callback};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rhd, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 128;
const SAMPLE_EVERY: u64 = 25;
// t_final = 0.4 keeps the leading shock interior at n=128 (shock speed \lesssim 0.7,
// distance from x=0.5 to either wall is 0.5) and yields ~200 CFL-bounded steps.
const T_FINAL: f64 = 0.4;

// the RHD 1D wave-speed bound used by the kernel cfl: davis estimate on the
// principal axis. cs^2 = \gamma p / (\rho h) with h = 1 + \gamma p / (\rho (\gamma-1)) (ideal gas).
// |\lambda|_max = (|v| + cs) / (1 + |v| cs) on v, cs \in [0,1).
fn max_wavespeed_estimate(rho: f64, p: f64, v: f64) -> f64 {
    let h = 1.0 + GAMMA * p / (rho * (GAMMA - 1.0));
    let cs2 = (GAMMA * p) / (rho * h);
    let cs = cs2.sqrt();
    let av = v.abs();
    (av + cs) / (1.0 + av * cs)
}

#[test]
fn rhd_sod_conserves_mass_energy_and_respects_cfl() {
    let dx = 1.0 / N as f64;
    // Marti-Mueller sharp Sod IC, v = 0 => W = 1 => D = rho, S = 0, tau = p/(gamma-1).
    // cfl == CFL == 0.4 (builder default, omitted).
    let mut sim = Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([dx])
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
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

    let cnrg = sim.fields.cons.nrg_field().expect("Rhd cons.nrg");
    let cells: Vec<[isize; 1]> = sim.geom.interior.iter().collect();
    let mass0: f64 = cells
        .iter()
        .map(|c| *sim.fields.cons.den.view().at(*c))
        .sum::<f64>()
        * dx;
    let energy0: f64 = cells.iter().map(|c| *cnrg.view().at(*c)).sum::<f64>() * dx;

    let sub = RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, CFL, &sim.geom.allocated);

    // walk the production evolve loop; every SAMPLE_EVERY steps the callback
    // exercises mass / energy / cfl / positivity in lock-step with the kernel.
    let mut samples: u32 = 0;
    evolve_with_callback(&mut sim, &sub, T_FINAL, SAMPLE_EVERY, |s| {
        // mass + energy integrals over the reflective interior.
        let cnrg = s.fields.cons.nrg_field().expect("cons.nrg");
        let mass: f64 = cells
            .iter()
            .map(|c| *s.fields.cons.den.view().at(*c))
            .sum::<f64>()
            * dx;
        let energy: f64 = cells.iter().map(|c| *cnrg.view().at(*c)).sum::<f64>() * dx;

        let mass_rel = (mass - mass0).abs() / mass0;
        let energy_rel = (energy - energy0).abs() / energy0;
        assert!(
            mass_rel < 1e-9,
            "MASS CONSERVATION BROKEN at iter {} t={:.4e}: mass {} → {} (rel drift {:e}, > 1e-9)",
            s.iteration,
            s.time,
            mass0,
            mass,
            mass_rel,
        );
        assert!(
            energy_rel < 1e-8,
            "ENERGY CONSERVATION BROKEN at iter {} t={:.4e}: nrg {} → {} (rel drift {:e}, > 1e-8)",
            s.iteration,
            s.time,
            energy0,
            energy,
            energy_rel,
        );

        // positivity + subluminal causality at every interior cell, every checkpoint.
        let pre = s.fields.prim.pre_field().expect("prim.pre");
        let mut max_lambda = 0.0_f64;
        for c in &cells {
            let rho = *s.fields.prim.rho.view().at(*c);
            let p = *pre.view().at(*c);
            let v = *s.fields.prim.vel[0].view().at(*c);
            assert!(
                rho.is_finite() && rho > 0.0,
                "POSITIVITY BROKEN (ρ ≤ 0 or NaN) at iter {} cell {:?}: rho = {}",
                s.iteration,
                c,
                rho,
            );
            assert!(
                p.is_finite() && p > 0.0,
                "POSITIVITY BROKEN (p ≤ 0 or NaN) at iter {} cell {:?}: p = {}",
                s.iteration,
                c,
                p,
            );
            assert!(
                v.abs() < 1.0,
                "CAUSALITY BROKEN (|v| ≥ 1) at iter {} cell {:?}: v = {}",
                s.iteration,
                c,
                v,
            );
            max_lambda = max_lambda.max(max_wavespeed_estimate(rho, p, v));
        }

        // CFL bound: the kernel's NEXT dt — same call evolve uses — must
        // respect dt * max|\lambda| / dx <= CFL. if the kernel returns a dt that
        // violates the CFL on the actual state, this fires.
        let dt_next = sub.cfl(s);
        assert!(
            dt_next > 0.0 && dt_next.is_finite(),
            "CFL DT INVALID at iter {}: dt_next = {}",
            s.iteration,
            dt_next,
        );
        let courant = dt_next * max_lambda / dx;
        assert!(
            courant <= CFL + 1e-12,
            "CFL BOUND VIOLATED at iter {} t={:.4e}: dt·λ/dx = {:e} > CFL ({}) by {:e}",
            s.iteration,
            s.time,
            courant,
            CFL,
            courant - CFL,
        );

        eprintln!(
            "[rhd_sod_cons] iter {:>3} t={:.4e}  mass_rel={:.2e}  E_rel={:.2e}  \
                 dt_next={:.2e}  courant={:.3}",
            s.iteration, s.time, mass_rel, energy_rel, dt_next, courant,
        );
        samples += 1;
    })
    .expect("rhd evolve failed");

    // require at least one mid-run sample fired — otherwise the loop ended
    // before any gate could run.
    assert!(
        samples > 0,
        "no mid-run conservation sample fired (samples={samples})"
    );
    eprintln!(
        "[rhd_sod_cons] DONE iter={} t={:.4e} samples={} (mass0={:.6} energy0={:.6})",
        sim.iteration, sim.time, samples, mass0, energy0,
    );
}
