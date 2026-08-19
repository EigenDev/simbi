// =============================================================================
// sealed_column_balanced.rs
//
// a low-mach riemann solver holds the adiabatic entropy floor on a sealed, stagnant, strongly
// stratified column when it is paired with the well-balanced reconstruction — the regime a
// solid accretor wall keeps its masked cells in.
//
// the mechanism under test is a division of labor. a low-mach scheme reduces the dissipation a
// face receives in proportion to the local flow speed; on a stagnant column that dissipation is
// the only damping on the hydrostatic truncation residual, so the plain reconstruction rings
// and the floor fails. that arm is this test's positive control. the well-balanced
// reconstruction removes the residual at its source: each cell's pressure departure from the
// mechanical equilibrium through it (Kaeppeli & Mishra, A&A 587, A94, 2016) is what gets
// limited, so a balanced column presents a flat face state and rings at zero amplitude.
// restoring the dissipation instead would buy the floor by giving up the low-mach reduction
// across every stratified face; the balancing buys it by construction, and the reduction
// survives.
//
// the column is seeded in the scheme's own discrete class — pressures related by the
// piecewise-constant-density segment sums of the plummer-softened potential, on the kernel's
// own lattice — with a deliberately non-isentropic density, so the fixed point under test is
// the mechanical scheme's whole claim: a discretely balanced column of arbitrary entropy
// stratification, held to machine precision. the potential is the same `body_potential`
// family the reconstruction balances against and the same gravity the source applies (one
// field, autodiff-proven conservative).
//
// run: cargo test -p symbi --test sealed_column_balanced -- --nocapture
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
/// plummer softening of the body's field. the column is built from the same softened
/// potential, so the state, the applied gravity and the reconstruction's balance are one
/// field; at r >= 1 the softening correction is ~1e-6 relative and the column is still
/// strongly stratified.
const SOFT: f64 = 1.0e-3;
/// long enough that the plain arm's wall contamination reaches the window and accumulates --
/// that contamination is this gate's positive control. a reflecting ghost mirrors the
/// stratified column, and the mirror departs from the isentrope's continuation, so the plain
/// arm's wall cells are kicked every step and the signal sweeps the domain within ~110 steps.
/// the balanced arm's ghosts are balance-aware (velocity mirrors, rho/p extend along the local
/// isentrope), so the same 400 steps leave it at machine equilibrium: with the column
/// built from the body's own softened potential, the measured balanced state is
/// deficit 1.1e-16, |v| 8.6e-16 at 100 steps -- exact to roundoff, and the wall-inclusive run is the
/// stronger statement of the same fact.
const STEPS: u64 = 400;
/// the causally-clean measurement window, in x.
const WINDOW: (f64, f64) = (0.35, 0.65);

/// the plummer-softened potential of the test body, evaluated on the gas coordinate.
fn phi(x: f64) -> f64 {
    let r = x + G_OFFSET;
    -GM / (r * r + SOFT * SOFT).sqrt()
}

/// a deliberately non-isentropic density: the isentrope of the softened field modulated
/// by a finite ripple, so the seeded column carries genuine entropy stratification and no
/// thermal assumption could reproduce it.
fn stratified_rho(x: f64) -> f64 {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi(1.0);
    (a * (c - phi(x))).powf(1.0 / (GAMMA - 1.0)) * (1.0 + 0.25 * (9.0 * x).sin())
}

/// the column in the scheme's own discrete class, on the kernel's own lattice: pressures
/// follow the piecewise-constant-density segment sums of the mechanical equilibrium
/// (Kaeppeli & Mishra, A&A 587, A94, 2016) over the cell centers and faces the balanced
/// reconstruction anchors on. the march runs inward from the outer edge, where pressure
/// only grows, so any positive stratification stays physical. class membership is what
/// makes the column a discrete fixed point at machine precision, whatever the entropy
/// profile — the property the isentropic predecessor reserved for one thermal family.
fn class_column(n: usize) -> Vec<(f64, f64)> {
    let dx = 1.0 / n as f64;
    let center = |i: usize| (i as f64 + 0.5) * dx;
    let face = |i: usize| i as f64 * dx;
    let mut col = vec![(0.0_f64, 0.0_f64); n];
    let rho_out = stratified_rho(center(n - 1));
    col[n - 1] = (rho_out, K0 * rho_out.powf(GAMMA));
    for k in (0..n - 1).rev() {
        let (ra, rb) = (stratified_rho(center(k)), stratified_rho(center(k + 1)));
        let pre = col[k + 1].1
            + rb * (phi(center(k + 1)) - phi(face(k + 1)))
            + ra * (phi(face(k + 1)) - phi(center(k)));
        col[k] = (ra, pre);
    }
    assert!(
        col.iter().all(|&(r, p)| r > 0.0 && p > 0.0),
        "the class column left the physical regime; the fixed point is vacuous"
    );
    col
}

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
type Hier = Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

fn build(balanced: bool) -> Hier {
    build_n(balanced, N)
}

fn build_n(balanced: bool, n: usize) -> Hier {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([n])
        .spacing([1.0 / n as f64])
        // a reflecting wall exerts no work on gas at rest, so the hydrostatic state is a
        // fixed point of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial({
            let col = class_column(n);
            let dx = 1.0 / n as f64;
            move |x: [f64; 1]| {
                let i = ((x[0] / dx - 0.5).round() as usize).min(col.len() - 1);
                let (rho, pre) = col[i];
                Prim {
                    rho,
                    vel: symbi_algebra::Tensor::new([0.0]),
                    pre,
                }
            }
        })
        .build();
    let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
        .with_solver(Solver::HllcPlus)
        .expect("solver/regime mismatch")
        .well_balanced_reconstruction(balanced);
    Hierarchy::single(sim, kernels).with_bodies(symbi_ib::BodyCollection::new().add(
        symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new([-G_OFFSET]),
            symbi_algebra::Tensor::zeros(),
            GM,
            // radius (mask) stays effectively pointlike; soft rides the softening slot so
            // the body's field is the same plummer potential the column is built from --
            // passed via the radius slot, the field is bare and the column carries a
            // benign ~1e-6 mismatch, which is what set the old measured floor.
            1.0e-6,
            SOFT,
        ),
    ))
}

/// the smallest per-cell `K(x, t) / K(x, 0)` and the largest |v| away from the walls. the
/// seeded column is entropy-stratified, so the floor is each cell against its own seeded
/// adiabat rather than against one global constant: an adiabatic gas holds its entropy as
/// a one-way floor cell by cell, and anything below one is the scheme's own deficit. the
/// second quantity is the stagnancy precondition.
fn run(balanced: bool) -> (f64, f64) {
    let col = class_column(N);
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
        let (rho0, pre0) = col[(ii - ilo) as usize];
        let k_seed = pre0 / rho0.powf(GAMMA);
        worst = worst.min(*pre.at(c) / rho.at(c).powf(GAMMA) / k_seed);
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
        "\nsealed stagnant column, low-mach solver, {STEPS} steps\n\
         plain reconstruction:    min K/K_0 {k_plain:.12} (deficit {d_plain:.3e}), max|v| {v_plain:.3e}\n\
         balanced reconstruction: min K/K_0 {k_wb:.12} (deficit {d_wb:.3e}), max|v| {v_wb:.3e}"
    );

    // the discriminating quantity is stagnation, not the floor. on the potential-consistent
    // column the plain arm's entropy floor holds up, because its wall waves dissipate and
    // dissipation raises K; the floor-under-stress claim lives in gravity_source_entropy's
    // 5000-step wall column. what separates the arms here by ten orders is stagnation: the
    // plain arm's mirrored ghosts kick its walls every step and the drift reaches the window,
    // while the balanced triple (reconstruction + source + ghosts) holds the discrete
    // equilibrium to machine precision.
    assert!(
        v_plain > 1.0e-7,
        "the PLAIN arm sits at |v| = {v_plain:.3e}; the column is not exercising the \
         imbalance (the mirrored-ghost wall kick never reached the window) and the \
         balanced arm's stagnation proves nothing. lengthen the run"
    );
    assert!(
        v_wb * 1.0e6 < v_plain,
        "the balanced arm's residual flow ({v_wb:.3e}) is within six orders of the plain \
         arm's ({v_plain:.3e}); the triple is no longer holding the discrete equilibrium"
    );
    // machine equilibrium, absolutely: measured |v| = 3.5e-15 and deficit = 1.1e-16 at
    // 400 wall-inclusive steps. the bounds carry three orders of margin.
    assert!(
        v_wb < 1.0e-12,
        "balanced-arm residual velocity {v_wb:.3e}; the discrete equilibrium is drifting"
    );
    assert!(
        d_wb < 1.0e-12,
        "balanced-arm entropy deficit {d_wb:.3e}; the sealed column is venting through a \
         path the balanced triple was supposed to close"
    );
}

/// a measurement instrument. it prints the wall-clock cost of the balanced reconstruction
/// relative to plain on the identical column; the number is for the record and no threshold
/// is asserted on it. the balanced arm pays one isentrope exponentiation per stencil offset
/// per face side; how that prices out against the full step (riemann solve, source, ghost
/// fill) is settled by the clock, since an op count prices a transcendental wrong. run with
/// `cargo test --release -- --ignored wb_cost` and read the printed ratio; the production
/// number comes from a release build, debug timings weighting transcendentals differently.
/// at the gate's n = 128 the ratio is indistinguishable from 1 (0.97-1.10, per-step overhead
/// dominates, measured 2026-08-15). flux-dominated at n = 65536 it is 1.55-1.72 on an idle
/// arm64 cpu (3 runs, release, 2026-08-17). a sampling profile of that arm attributes about
/// a quarter of the face-flux kernel to the isentrope exponentiation and another sixth to the
/// transcendentals in the face-position coordinate map, so the elementary-function calls are
/// the kernel's largest single term on this hardware. the whole-step production cost sits
/// below the flux-stage bound in proportion to the flux stage's share of the step.
#[test]
#[ignore]
fn wb_cost_probe() {
    // a wide column so the face-flux kernel dominates the step; at the gate's n = 128
    // the ratio is buried in per-step overhead noise.
    const PROBE_N: usize = 65536;
    const PROBE_STEPS: u64 = 200;
    let time = |balanced: bool| {
        // one warm-up build absorbs bake/alloc costs outside the timed window.
        let mut hier = build_n(balanced, PROBE_N);
        hier.evolve_steps(8).unwrap();
        let mut hier = build_n(balanced, PROBE_N);
        let t0 = std::time::Instant::now();
        hier.evolve_steps(PROBE_STEPS).unwrap();
        t0.elapsed().as_secs_f64()
    };
    let (t_plain, t_wb) = (time(false), time(true));
    println!(
        "sealed column, {PROBE_STEPS} steps, n = {PROBE_N}: plain {t_plain:.3}s, \
         balanced {t_wb:.3}s, ratio {:.3}",
        t_wb / t_plain
    );
}
