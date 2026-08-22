// =============================================================================
// well_balanced_solver_family.rs
//
// division of labour between the well-balancing and the riemann solver, on the
// regime a sealed accretor wall holds its masked cells in: a stagnant, strongly
// stratified, deeply subsonic column.
//
// a hydrostatic state solves the continuum equations, not the discrete ones, so an
// undeclared scheme leaves a truncation residual and deposits it one-signed every
// step. that residual rings at grid scale and shows up as an entropy deficit
// (K = p/rho^gamma below its lagrangian value) in an adiabatic gas that cannot lose
// entropy. two different tools can suppress it:
//
//   - the riemann solver, by keeping enough acoustic dissipation on the stratified
//     faces to damp the ring. this is what a compressibility clamp on the low-mach
//     scaling does, and it costs the low-mach accuracy the scaling exists to buy,
//     across the whole stratified region;
//   - the well-balancing, by measuring the residual once per level and subtracting
//     it every stage, so there is no ring to damp in the first place.
//
// if the second works, the first is not needed here, and a solver that reduces
// dissipation aggressively becomes usable on a stratified problem. that is the
// question: does declaring the target let the acoustic-consistency scaling — which
// fails this column undeclared — hold the floor?
//
// undeclared is the positive control. the deficit must actually appear there, or the
// column is not exercising the imbalance and the declared arm proves nothing.
//
// run: cargo test -p symbi --test well_balanced_solver_family -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// the gravitating mass sits one domain width left of x = 0, so the gas at x feels a
/// bare point mass at radius x + 1 and the domain covers r in [1, 2] with no singularity.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 100.0;
/// enough steps for a per-step imbalance to accumulate well clear of roundoff.
const STEPS: u64 = 400;
/// cells nearest each wall are excluded: the boundary is its own discretization and a
/// wall cell's deficit says nothing about the interior scheme.
const WALL_SKIP: usize = 4;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

use symbi_geometry::Cartesian;

/// the isentropic atmosphere in hydrostatic balance against GM, from the bernoulli
/// invariant `gamma K0/(gamma-1) rho^(gamma-1) - GM/r = const`, normalized to
/// `rho = 1` at the outer edge.
fn hydrostatic(x: [f64; 1]) -> Prim<f64, 1> {
    let r = x[0] + G_OFFSET;
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a - GM / (1.0 + G_OFFSET);
    let rho = (a * (GM / r + c)).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

fn build(solver: Solver) -> Hier {
    build_n(solver, N)
}

fn build_n(solver: Solver, n: usize) -> Hier {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([1.0 / n as f64])
        // a reflecting wall exerts no work on gas at rest, so the hydrostatic state is a
        // fixed point of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(hydrostatic)
        .build();
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(solver)
        .expect("solver/regime mismatch");
    Hierarchy::single(sim, kernels).with_bodies(symbi_ib::BodyCollection::new().add(
        symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new([-G_OFFSET]),
            symbi_algebra::Tensor::zeros(),
            GM,
            1.0e-3,
            0.0,
        ),
    ))
}

/// the smallest `K/K_0` anywhere away from the walls. an adiabatic gas cannot lose
/// entropy, so anything below one is the scheme's own deficit.
fn worst_entropy_ratio(hier: &Hier) -> f64 {
    let st = &hier.levels[0].state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre_field().expect("prim.pre").view();
    let lo = st.geom.interior.spaces[0].lo + WALL_SKIP as isize;
    let hi = st.geom.interior.spaces[0].hi - WALL_SKIP as isize;
    let mut worst = f64::INFINITY;
    for ii in lo..hi {
        let c = [ii];
        let k = *pre.at(c) / rho.at(c).powf(GAMMA);
        worst = worst.min(k / K0);
    }
    worst
}

fn run(solver: Solver, declared: bool) -> f64 {
    let mut hier = build(solver);
    if declared {
        hier = hier.with_equilibrium(hydrostatic).unwrap();
        hier.seed_equilibrium();
    }
    hier.evolve_steps(STEPS).unwrap();
    worst_entropy_ratio(&hier)
}

#[test]
fn declaring_the_target_frees_the_solver_from_damping_the_residual() {
    let solvers = [("hllc", Solver::Hllc), ("hllc_plus", Solver::HllcPlus)];
    println!("\nsealed stratified column, {STEPS} steps — min K/K_0 (deficit = 1 - K/K_0)");
    let mut undeclared = Vec::new();
    let mut declared = Vec::new();
    for (name, solver) in solvers {
        let bare = run(solver, false);
        let with_target = run(solver, true);
        println!(
            "{name:>14}:  undeclared {bare:.12} (deficit {:.3e})   declared {with_target:.12} \
             (deficit {:.3e})",
            (1.0 - bare).max(0.0),
            (1.0 - with_target).max(0.0)
        );
        undeclared.push((name, bare));
        declared.push((name, with_target));
    }
    let deficit = |k: f64| (1.0 - k).max(0.0);

    // positive control: the low-dissipation solver is the one that under-damps the residual,
    // so its undeclared arm must show a deficit. without that this column is not exercising
    // the imbalance and the declared results say nothing.
    let bare_low_mach = undeclared[1].1;
    assert!(
        bare_low_mach < 1.0 - 1.0e-6,
        "undeclared, HLLC+ held the floor at {bare_low_mach:.12} — this column no longer \
         exercises the hydrostatic residual, so the declared arm is vacuous. lengthen the run \
         or steepen the stratification"
    );

    // declaring must never make a solver worse: the correction is the scheme's own measured
    // imbalance, so subtracting it can only remove a deposit.
    for ((name, bare), (_, with_target)) in undeclared.iter().zip(&declared) {
        let (d_bare, d_decl) = (deficit(*bare), deficit(*with_target));
        assert!(
            d_decl <= d_bare + 1.0e-12,
            "{name}: declaring the target made the deficit WORSE ({d_bare:.3e} -> \
             {d_decl:.3e}), which the correction cannot do if it is the measured imbalance"
        );
    }

    // the claim under test: for the solver that cannot damp the residual itself, moving that
    // job to the well-balancing recovers the floor. measured 4.9e-2 undeclared against 2.1e-13
    // declared — eleven orders, where the law asks only for one.
    let (d_bare, d_decl) = (deficit(undeclared[1].1), deficit(declared[1].1));
    assert!(
        d_decl * 10.0 < d_bare,
        "declaring the target improved HLLC+'s deficit only from {d_bare:.3e} to \
         {d_decl:.3e}; the well-balancing is not taking over the job of damping the \
         hydrostatic residual"
    );

    // and what the declaration buys, stated as the gap it closes. undeclared, the two solvers
    // are separated by how much dissipation each applies to the residual: classical HLLC holds
    // the floor outright while HLLC+, which removes the velocity-jump damping the residual
    // would otherwise receive, loses a measurable fraction. declared, both sit at the
    // truncation of the balance itself and the solver has stopped mattering — which is the
    // property that lets a stratified science run choose its solver on the physics rather than
    // on the residual.
    assert!(
        deficit(undeclared[0].1) * 1.0e3 < deficit(undeclared[1].1),
        "undeclared, classical HLLC ({:.3e}) and HLLC+ ({:.3e}) are no longer separated by \
         their dissipation; the gap the declaration closes is what this measures",
        deficit(undeclared[0].1),
        deficit(undeclared[1].1)
    );
    for (name, k) in &declared {
        assert!(
            deficit(*k) < 1.0e-10,
            "{name}: declared, the deficit is {:.3e}; with the target declared every solver \
             must sit at the balance's own truncation rather than at its own dissipation",
            deficit(*k)
        );
    }
}

/// how the undeclared deficit behaves under refinement, per solver.
///
/// the deficit's origin decides whether it is a cost or a defect. a hydrostatic state solves the
/// continuum equations and not the discrete ones, so the residual that drives it is a truncation
/// error and must shrink as the grid refines. a deficit that instead grows with resolution is
/// not truncation at all but an amplification, the signature the acoustic-speed ramp showed on
/// this same column before it was retired — max|v| of 7.6e-11 at 128 cells against 1.4e-3 at
/// 256, seven orders across one refinement.
///
/// run: cargo test --release -p symbi --test well_balanced_solver_family -- --ignored deficit_trend --nocapture
#[test]
#[ignore = "diagnostic: undeclared entropy deficit against resolution, per solver"]
fn diagnose_undeclared_deficit_trend() {
    println!("\nundeclared sealed column, plain reconstruction, {STEPS} steps");
    println!(
        "{:>10} {:>6} {:>12} {:>9}",
        "solver", "n", "deficit", "ratio"
    );
    for solver in [Solver::Hllc, Solver::HllcPlus] {
        let mut prev: Option<f64> = None;
        for n in [64usize, 128, 256] {
            let mut hier = build_n(solver, n);
            hier.evolve_steps(STEPS).unwrap();
            let deficit = (1.0 - worst_entropy_ratio(&hier)).max(0.0);
            let ratio = prev.map_or(f64::NAN, |p: f64| deficit / p);
            println!(
                "{:>10} {n:>6} {deficit:>12.3e} {ratio:>9.3}",
                format!("{solver:?}")
            );
            prev = Some(deficit);
        }
    }
}
