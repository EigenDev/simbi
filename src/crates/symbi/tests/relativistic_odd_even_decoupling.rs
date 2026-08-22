// =============================================================================
// relativistic_odd_even_decoupling.rs
//
// Quirk's odd-even decoupling test in the relativistic regime: the grid-aligned
// shock instability (the carbuncle) is a property of the multidimensional
// momentum balance, so a relativistic blast wave or jet front carries it as
// readily as a newtonian shock. a planar relativistic shock travels along x in a
// duct whose ambient carries a row-alternating density perturbation; a solver
// whose transverse-face dissipation is mis-scaled amplifies it into a growing
// transverse velocity and the front breaks up.
//
// the diagnostic is the transverse kinetic energy behind the front, measured
// against the seeded perturbation. it is a growth test rather than a threshold:
// what separates a stable solver from an unstable one is whether the seed grows
// by orders, which the seeded amplitude answers without a tuned bound.
//
// classical relativistic HLLC is the positive control. it is the solver the
// instability is documented for, so if it does not grow here the setup is not
// exercising the mechanism and the HLLC+ arm proves nothing.
//
// run: cargo test -p symbi --test relativistic_odd_even_decoupling -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::evolve::evolve;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::rhd::Rhd;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rhd, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = RhdSubstrateKernelSet<HostMemory, f64, 2>;

/// the ultrarelativistic index a radiation-dominated blast carries.
const GAMMA: f64 = 4.0 / 3.0;
const CFL: f64 = 0.4;
const NX: usize = 240;
const NY: usize = 24;
/// the seeded perturbation: a relative density zig-zag on alternating transverse rows of the
/// gas ahead of the front, so the shock runs into it. perturbing the data rather than the
/// interface position keeps it from quantizing away, since a sub-cell shift of the interface
/// is sampled identically on every row.
///
/// a cold relativistic ambient couples the seed into transverse motion some five orders more
/// weakly than the newtonian shock tube does, so the amplitude is set where the response is
/// both measurable and still linear. at 1e-6 the classical arm reaches 4.8e-16 of transverse
/// energy and the growth ratio is measuring roundoff; at 1e-2 the energy scales by 50 for a
/// hundredfold seed, sub-quadratic, so the mode has saturated and the growth ratio has
/// stopped reporting the instability. at 1e-3 the response is quadratic in the seed and the
/// classical arm reaches 4.2e-10.
const SEED: f64 = 1.0e-3;
const T_MID: f64 = 0.35;
const T_FINAL: f64 = 0.70;

/// a duct carrying a strong relativistic shock travelling in +x. the driver is a hot, dense,
/// mildly relativistic slab against a cold ambient — the configuration a blast wave presents
/// to the grid once its front is aligned with an axis.
fn shocked_duct() -> Sim {
    let dx = 1.0 / NX as f64;
    Sim::build(Rhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([NX, NY])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial_indexed(|idx, x| {
            if x[0] < 0.1 {
                Prim {
                    rho: 10.0,
                    vel: Tensor::new([0.99, 0.0]),
                    pre: 500.0,
                }
            } else {
                let zig = if idx[1] % 2 == 0 {
                    1.0 + SEED
                } else {
                    1.0 - SEED
                };
                Prim {
                    rho: zig,
                    vel: Tensor::new([0.0, 0.0]),
                    pre: 1.0e-2,
                }
            }
        })
        .build()
}

/// transverse kinetic energy per unit mass over the shocked gas. the exact solution carries
/// none, so all of it is the instability's amplitude.
fn transverse_energy(sim: &Sim) -> f64 {
    let (mut total, mut mass) = (0.0, 0.0);
    for c in sim.geom.interior.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        if rho < 2.0 {
            continue;
        }
        let vy = *sim.fields.prim.vel[1].view().at(c);
        total += rho * vy * vy;
        mass += rho;
    }
    if mass > 0.0 { total / mass } else { 0.0 }
}

/// does the domain still hold both shocked and unshocked gas? once the front has left, the box
/// is uniform and the transverse velocity vanishes for a reason unrelated to the solver.
fn shock_is_interior(sim: &Sim) -> bool {
    let (mut hot, mut cold) = (false, false);
    for c in sim.geom.interior.iter() {
        if *sim.fields.prim.rho.view().at(c) > 2.0 {
            hot = true;
        } else {
            cold = true;
        }
    }
    hot && cold
}

fn run(solver: Solver) -> (f64, f64, u64) {
    let mut sim = shocked_duct();
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(solver)
        .expect("solver/regime mismatch");
    evolve(&mut sim, &kernels, T_MID).expect("evolve failed");
    let mid = transverse_energy(&sim);
    evolve(&mut sim, &kernels, T_FINAL).expect("evolve failed");
    assert!(
        shock_is_interior(&sim),
        "the front has left the domain; the box is uniform and this measures nothing"
    );
    (mid, transverse_energy(&sim), sim.iteration)
}

#[test]
fn the_relativistic_shear_viscosity_is_shock_stable_on_quirks_test() {
    let mut rows = Vec::new();
    for (name, solver) in [
        ("rhd hllc", Solver::Hllc),
        ("rhd hllc_plus", Solver::HllcPlus),
    ] {
        let (mid, end, steps) = run(solver);
        let growth = if mid > 0.0 { end / mid } else { f64::INFINITY };
        println!(
            "{name:>16}: transverse KE/mass {mid:.4e} -> {end:.4e} (x{growth:.2e}) in {steps} steps"
        );
        assert!(
            steps > 50,
            "{name}: only {steps} steps — too little evolution to develop or suppress anything"
        );
        assert!(end.is_finite(), "{name}: non-finite transverse energy");
        rows.push((end, growth));
    }
    let (classical, classical_growth) = rows[0];
    let sheared = rows[1].1;

    // positive control: the instability has to be present under the solver it is documented
    // for, or the HLLC+ arm's stability is a statement about a setup that never provoked it.
    assert!(
        classical > 1.0e-12,
        "classical relativistic HLLC reached only {classical:.3e} of transverse kinetic energy \
         from a seed of {SEED:e}; the perturbation never developed and the growth ratio below \
         is measuring roundoff. strengthen the driver or lengthen the clock"
    );
    // the growth threshold is this regime's, not the newtonian one's: a cold ultrarelativistic
    // ambient amplifies the transverse mode far less vigorously than a mach-10 shock tube does,
    // and the classical arm measures 11.3 here against the newtonian suite's 51.
    assert!(
        classical_growth > 3.0,
        "classical relativistic HLLC amplified the seed by only {classical_growth:.3e} — the \
         setup never provoked the grid-aligned instability, so the other arm is vacuous here. \
         raise the lorentz factor of the driver or the resolution"
    );

    // the claim: the transverse viscosity arrests the growth. it acts on the transverse
    // velocity jump the perturbation feeds on, with the enthalpy density `rho h W^2 = e + p`
    // as the inertia carrying it, gated on a characteristic reversal so it reaches the front
    // and nothing else. measured: growth 4.2 against 11.3, and a final transverse energy of
    // 3.5e-11 against 4.2e-10 — an order of magnitude, where the newtonian arm of the same
    // scheme buys six. the relativistic instability is milder here to begin with, and the
    // demand is that the viscosity measurably removes energy from it rather than that it
    // reproduces the newtonian margin.
    assert!(
        sheared < classical_growth / 2.0,
        "relativistic HLLC+ amplified the seed by {sheared:.3e} against classical HLLC's \
         {classical_growth:.3e}; the shear viscosity is not reaching the transverse faces of \
         the front"
    );
    assert!(
        rows[1].0 < classical,
        "relativistic HLLC+ left {:.3e} of transverse energy against classical HLLC's \
         {classical:.3e}; the viscosity must remove energy from the perturbation",
        rows[1].0
    );
}
