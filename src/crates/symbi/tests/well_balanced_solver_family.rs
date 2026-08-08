// =============================================================================
// well_balanced_solver_family.rs
//
// DIVISION OF LABOUR between the well-balancing and the riemann solver, on the
// regime a sealed accretor wall holds its masked cells in: a stagnant, strongly
// stratified, deeply subsonic column.
//
// a hydrostatic state solves the CONTINUUM equations, not the discrete ones, so an
// undeclared scheme leaves a truncation residual and deposits it one-signed every
// step. that residual rings at grid scale and shows up as an entropy DEFICIT
// (K = p/rho^gamma below its lagrangian value) in an adiabatic gas that cannot lose
// entropy. two different tools can suppress it:
//
//   - the RIEMANN SOLVER, by keeping enough acoustic dissipation on the stratified
//     faces to damp the ring. this is what a compressibility clamp on the low-mach
//     scaling does, and it costs the low-mach accuracy the scaling exists to buy,
//     across the whole stratified region;
//   - the WELL-BALANCING, by measuring the residual once per level and subtracting
//     it every stage, so there is no ring to damp in the first place.
//
// if the second works, the first is not needed HERE, and a solver that reduces
// dissipation aggressively becomes usable on a stratified problem. that is the
// question: does declaring the target let the acoustic-consistency scaling — which
// fails this column undeclared — hold the floor?
//
// UNDECLARED is the positive control. the deficit must actually appear there, or the
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
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
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
    let solvers = [
        ("hllc", Solver::Hllc),
        ("hllc_lm", Solver::HllcLm),
        ("hllc_acoustic", Solver::HllcAcoustic),
    ];
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

    // POSITIVE CONTROL: the acoustic scaling is the one that under-damps the residual,
    // so its UNDECLARED arm must show a deficit. without that this column is not
    // exercising the imbalance and the declared results say nothing.
    let bare_acoustic = undeclared[2].1;
    assert!(
        bare_acoustic < 1.0 - 1.0e-6,
        "undeclared, the acoustic scaling held the floor at {bare_acoustic:.12} — this \
         column no longer exercises the hydrostatic residual, so the declared arm is \
         vacuous. lengthen the run or steepen the stratification"
    );

    // declaring must never make a solver WORSE: the correction is the scheme's own
    // measured imbalance, so subtracting it can only remove a deposit.
    for ((name, bare), (_, with_target)) in undeclared.iter().zip(&declared) {
        let (d_bare, d_decl) = ((1.0 - bare).max(0.0), (1.0 - with_target).max(0.0));
        assert!(
            d_decl <= d_bare + 1.0e-12,
            "{name}: declaring the target made the deficit WORSE ({d_bare:.3e} -> \
             {d_decl:.3e}), which the correction cannot do if it is the measured imbalance"
        );
    }

    // THE CLAIM under test: for the scaling that cannot damp the residual ITSELF, moving
    // that job to the well-balancing recovers most of the floor — the division of labour
    // works, even if it is not exact.
    let (d_bare, d_decl) = (
        (1.0 - undeclared[2].1).max(0.0),
        (1.0 - declared[2].1).max(0.0),
    );
    assert!(
        d_decl * 10.0 < d_bare,
        "declaring the target improved the acoustic scaling's deficit only from \
         {d_bare:.3e} to {d_decl:.3e}; the well-balancing is not taking over the job of \
         damping the hydrostatic residual"
    );

    // and what it does NOT do: the residual deficit still ORDERS by how much dissipation
    // each solver applies, so the declaration relieves the solver without replacing it.
    // recorded here because it is the reason a stratified science run should still prefer
    // the mach-limited scaling over the aggressive one.
    assert!(
        (1.0 - declared[1].1).max(0.0) < (1.0 - declared[2].1).max(0.0),
        "the mach-limited scaling no longer holds the floor better than the aggressive \
         one; the dissipation ordering this conclusion rests on has changed"
    );
}
