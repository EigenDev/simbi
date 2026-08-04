// =============================================================================
// lm_clamp_laws.rs
//
// the behavioral law of the HLLC-LM compressibility-consistency clamp: the
// low-mach dissipation reduction survives in the regime it exists for. the
// clamp restores classical dissipation on stratified-balance faces (gated by
// the sealed-column entropy floor in gravity_source_entropy.rs); HERE the
// complementary side is pinned — in smooth subsonic vortical flow the clamped
// scheme must remain strictly less dissipative than classical HLLC, or the
// clamp has crept into the turbulence and quietly reverted the solver.
//
// run: cargo test -p symbi --test lm_clamp_laws
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const CFL: f64 = 0.4;
const N: usize = 48;
const MACH: f64 = 0.06;

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn kinetic_energy_after(solver: Solver, t_end: f64) -> f64 {
    let dx = 1.0 / N as f64;
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(CFL)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim construction failed")
        .set_initial(|[x, y]| {
            let tau = std::f64::consts::TAU;
            Prim {
                rho: 1.0,
                vel: Tensor::new([
                    -MACH * (tau * y).sin() * (tau * x).cos(),
                    MACH * (tau * x).sin() * (tau * y).cos(),
                ]),
                pre: 1.0 / GAMMA,
            }
        })
        .build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 2>::new(GAMMA, CFL, &sim.geom.allocated)
            .with_solver(solver)
            .expect("solver/regime mismatch");
    evolve(&mut sim, &sub, t_end).expect("evolve failed");
    let rho = sim.fields.prim.rho.view();
    let mut e_kin = 0.0;
    for c in sim.geom.interior.iter() {
        let vsq: f64 = (0..2)
            .map(|k| {
                let v = *sim.fields.prim.vel[k].view().at(c);
                v * v
            })
            .sum();
        e_kin += 0.5 * *rho.at(c) * vsq;
    }
    e_kin / (N * N) as f64
}

/// decaying taylor-green flow at mach 0.06: several eddy turnovers of viscous-free
/// decay, where every dissipated joule is numerical. the clamped low-mach scheme
/// must retain strictly more kinetic energy than classical HLLC — the reduction
/// this solver family exists for — and the margin must be substantial: equal
/// retention means the clamp fires broadly enough to revert the scheme.
#[test]
fn the_clamped_low_mach_scheme_stays_less_dissipative_than_classical_hllc() {
    let t_end = 8.0;
    let e_lm = kinetic_energy_after(Solver::HllcLm, t_end);
    let e_std = kinetic_energy_after(Solver::Hllc, t_end);
    let e0 = 0.25 * MACH * MACH; // mean of 1/2 rho v^2 over the taylor-green cell
    eprintln!(
        "E_kin(t={t_end})/E_0: hllc_lm {:.4}, hllc {:.4}",
        e_lm / e0,
        e_std / e0
    );
    assert!(
        e_std > 0.0 && e_lm > e_std,
        "the clamped low-mach scheme retained no more kinetic energy than classical \
         HLLC (lm {e_lm:.6e} vs hllc {e_std:.6e}); the clamp has crept into smooth \
         subsonic turbulence"
    );
    // the retention gap must be a real dissipation difference, not roundoff: the
    // pure-ramp scheme's advantage over classical HLLC on this flow is tens of
    // percent of the dissipated energy.
    let dissipated_std = e0 - e_std;
    assert!(
        (e_lm - e_std) > 0.1 * dissipated_std,
        "the low-mach retention margin collapsed: lm - hllc = {:.3e} against {:.3e} \
         dissipated under hllc",
        e_lm - e_std,
        dissipated_std
    );
}
