// =============================================================================
// sealed_column_unclamped.rs
//
// THE GATE THE HLLC-LM ARC EXISTS FOR: the published low-mach ramp (no
// compressibility clamp) holding the adiabatic entropy floor on a sealed,
// stagnant, strongly stratified column — the regime a solid accretor wall keeps
// its masked cells in, and the regime that motivated bolting a clamp onto the
// published scheme in the first place.
//
// the mechanism under test is a DIVISION OF LABOUR. the ramp removes acoustic
// dissipation below the reference mach number; on a stagnant column that
// dissipation is the only thing damping the hydrostatic truncation residual, so
// the plain reconstruction rings and the floor fails — that arm is this test's
// POSITIVE CONTROL, and it reproduces the failure that motivated the clamp. the
// well-balanced reconstruction removes the RESIDUAL instead of damping it: each
// cell's departure from the isentrope through it is what gets limited, so a
// balanced column presents no face jump and there is nothing to ring. the clamp
// bought the floor by giving up the low-mach reduction across every stratified
// face; this buys it by construction, and the reduction survives.
//
// the column is the isentrope of the PLUMMER-SOFTENED potential — the same
// `body_potential` family the reconstruction balances against and the same
// gravity the source applies (one field, autodiff-proven conservative), so the
// only remaining imbalance is the smooth flux/source discretization mismatch,
// which is second-order and self-limiting rather than a one-signed deposit.
//
// run: cargo test -p symbi --test sealed_column_unclamped -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

use symbi_geometry::Cartesian;

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// the gravitating mass sits one domain width left of x = 0, so the gas at x feels the
/// mass at radius x + 1 and the domain covers r in [1, 2] with no singularity.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 100.0;
/// plummer softening of the body's field. the COLUMN is built from the same softened
/// potential, so the state, the applied gravity and the reconstruction's balance are one
/// field; at r >= 1 the softening correction is ~1e-6 relative and the column is still
/// strongly stratified.
const SOFT: f64 = 1.0e-3;
/// enough steps for the per-step interior imbalance to accumulate well clear of roundoff,
/// while the measurement window below stays CAUSALLY CLEAN of the walls: the fastest wall
/// signal starts at x = 0 with cs ~ 4.7, and 100 steps at this cfl advance t ~ 0.066, so
/// nothing from either wall reaches [0.35, 0.65]. the walls themselves are a HARNESS
/// artifact -- a reflecting ghost mirrors the stratified column, which is not the
/// isentrope's continuation, so wall cells are kicked every step regardless of the interior
/// scheme; the production sealed surface is a penalized interior cell with no such ghost.
const STEPS: u64 = 100;
/// the causally-clean measurement window, in x.
const WINDOW: (f64, f64) = (0.35, 0.65);
/// the stagnancy precondition: the column must stay deeply subsonic (cs ~ O(10) here) or
/// the arms are not exercising the low-mach regime and the gate is vacuous.
const STAGNANT_V: f64 = 1.0e-2;

/// the plummer-softened potential of the test body, evaluated on the gas coordinate.
fn phi(x: f64) -> f64 {
    let r = x + G_OFFSET;
    -GM / (r * r + SOFT * SOFT).sqrt()
}

/// the isentropic atmosphere in hydrostatic balance against the SOFTENED field, from the
/// bernoulli invariant `gamma K0/(gamma-1) rho^(gamma-1) + phi = const`, normalized to
/// `rho = 1` at the outer edge (x = 1).
fn hydrostatic(x: [f64; 1]) -> Prim<f64, 1> {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi(1.0);
    let rho = (a * (c - phi(x[0]))).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn build(balanced: bool) -> Hier {
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
        .with_solver(Solver::HllcLm)
        .expect("solver/regime mismatch")
        .well_balanced_reconstruction(balanced);
    Hierarchy::single(sim, kernels).with_bodies(symbi_ib::BodyCollection::new().add(
        symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new([-G_OFFSET]),
            symbi_algebra::Tensor::zeros(),
            GM,
            SOFT,
            0.0,
        ),
    ))
}

/// the smallest `K/K_0` and the largest |v| away from the walls. an adiabatic gas cannot
/// lose entropy, so anything below one in the first is the scheme's own deficit; the
/// second is the stagnancy precondition.
fn run(balanced: bool) -> (f64, f64) {
    let mut hier = build(balanced);
    hier.evolve_steps(STEPS).unwrap();
    let st = &hier.levels[0].state;
    let rho = st.fields.prim.rho.view();
    let pre = st.fields.prim.pre_field().expect("prim.pre").view();
    let vel = st.fields.prim.vel[0].view();
    let ilo = st.geom.interior.spaces[0].lo;
    let dx = 1.0 / N as f64;
    let mut worst = f64::INFINITY;
    let mut vmax = 0.0_f64;
    let mut in_window = 0usize;
    for ii in st.geom.interior.spaces[0].lo..st.geom.interior.spaces[0].hi {
        let x = ((ii - ilo) as f64 + 0.5) * dx;
        if x < WINDOW.0 || x > WINDOW.1 {
            continue;
        }
        in_window += 1;
        let c = [ii];
        worst = worst.min(*pre.at(c) / rho.at(c).powf(GAMMA) / K0);
        vmax = vmax.max(vel.at(c).abs());
    }
    assert!(in_window > 16, "window too narrow: {in_window} cells");
    (worst, vmax)
}

#[test]
fn the_published_ramp_holds_the_floor_on_a_balanced_reconstruction() {
    let (k_plain, v_plain) = run(false);
    let (k_wb, v_wb) = run(true);
    let (d_plain, d_wb) = ((1.0 - k_plain).max(0.0), (1.0 - k_wb).max(0.0));
    println!(
        "\nsealed stagnant column, published ramp (no clamp), {STEPS} steps\n\
         plain reconstruction:    min K/K_0 {k_plain:.12} (deficit {d_plain:.3e}), max|v| {v_plain:.3e}\n\
         balanced reconstruction: min K/K_0 {k_wb:.12} (deficit {d_wb:.3e}), max|v| {v_wb:.3e}"
    );

    // STAGNANCY PRECONDITION: both arms must stay deeply subsonic, or neither is in the
    // regime the low-mach ramp acts on and the comparison is between two shock problems.
    assert!(
        v_plain < STAGNANT_V && v_wb < STAGNANT_V,
        "the column did not stay stagnant (plain {v_plain:.3e}, balanced {v_wb:.3e} \
         against the {STAGNANT_V:.0e} bound); the gate is not exercising the low-mach \
         regime it exists for"
    );

    // POSITIVE CONTROL: the plain arm must show the deficit that motivated the clamp. if
    // it holds the floor by itself, this column no longer exercises the undamped
    // hydrostatic residual and the balanced arm proves nothing.
    assert!(
        d_plain > 1.0e-6,
        "with a PLAIN reconstruction the unclamped ramp held the floor to {d_plain:.3e}; \
         the column is not exercising the residual and the gate is vacuous. steepen the \
         stratification or lengthen the run"
    );

    // THE CLAIM: the balanced reconstruction removes the residual the ramp cannot damp,
    // so the deficit collapses. the factor is a measured margin, not a derived bound —
    // the reconstruction is exact on this column (T1: face jumps at roundoff), so what
    // remains is the smooth second-order flux/source mismatch, orders below the ring.
    assert!(
        d_wb * 30.0 < d_plain,
        "the balanced reconstruction reduced the unclamped deficit only from \
         {d_plain:.3e} to {d_wb:.3e} (need 30x); the residual is not being removed at \
         the reconstruction and the clamp is still load-bearing"
    );

    // AND THE FLOOR ITSELF: near-exact, not merely better. the bound is measured (see
    // the printed values) with an order of margin; a regression that starts venting
    // entropy through the balanced path lands far above it.
    assert!(
        d_wb < 1.0e-7,
        "the balanced arm's own deficit is {d_wb:.3e}; the sealed column is losing \
         entropy through a path the balanced reconstruction was supposed to close"
    );
}
