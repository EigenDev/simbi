// =============================================================================
// sealed_column_graded.rs
//
// the graded-mesh twin of `sealed_column_balanced` / `sealed_column_curvilinear`:
// the balanced triple (reconstruction + equilibrium-pressure source + balance-
// aware ghosts) holding a sealed, stagnant, strongly stratified column on a
// non-uniformly spaced grid — log-radial spherical (the natural bondi-like
// grid) and geometrically graded cartesian. the spacing is what changes the
// statement: every position in the balanced ladder (stencil anchors, source
// face/center potentials, ghost centroids) now comes from the runtime spacing
// map, and machine-exactness holds only because the ladder's cell centers are
// the map's own centers — the geometric mean sqrt(r_lo r_hi) on a log axis,
// the arithmetic face midpoint otherwise — i.e. the exact positions
// `set_initial` seeds the column at through `stagger_coord(Center)`. an
// arithmetic midpoint on the log axis would anchor the ladder O((dr/r)^2 r)
// off every cell, and each gate asserts that displacement is many orders above
// the exactness bound, so the center-definition premise stays live and any
// collapse of it fails the gate loudly.
//
// the plain arm at the same clock is each gate's positive control: its
// analytic rho*g source mismatches the discrete pressure gradient at
// truncation order and its mirrored ghosts kick the walls, so it must vent
// measurably or the balanced arm's stagnation proves nothing.
//
// run: cargo test -p symbi --test sealed_column_graded -- --nocapture
// =============================================================================

use symbi::prelude::Solver;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_geometry::AxisMap;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
const GM: f64 = 100.0;
/// plummer softening of the gravitating body's field. the column is built from the same
/// softened potential, so the state, the applied gravity and the balance are one field.
const SOFT: f64 = 1.0e-3;
const STEPS: u64 = 400;

/// the isentropic column in hydrostatic balance against the softened field of `phi`,
/// from the bernoulli invariant `gamma K0/(gamma-1) rho^(gamma-1) + phi = const`,
/// normalized to `rho = 1` at the outer wall.
fn hydrostatic(phi: impl Fn(f64) -> f64, x: f64, x_outer: f64) -> Prim<f64, 1> {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi(x_outer);
    let rho = (a * (c - phi(x))).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

macro_rules! sealed_graded_gate {
    ($modname:ident, $chart:ty, $chart_val:expr, $map:expr, $body_pos:expr,
     $phi:expr, $x_outer:expr, $window:expr, $v_wb_bound:expr, $d_wb_bound:expr) => {
        mod $modname {
            use super::*;

            type Sim = SimState<Newtonian, 1, $chart, IdealGas<f64>, CpuSpace, HostMemory>;
            type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
            type Hier =
                Hierarchy<Newtonian, 1, 1, $chart, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

            fn build(balanced: bool) -> Hier {
                let map: AxisMap = $map;
                let phi = $phi;
                let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, $chart_val)
                    .cells([N])
                    .origin([map.face(0)])
                    // a nominal uniform width for the builder's validation; the attached
                    // map overrides every position and width the scheme reads.
                    .spacing([map.width(0)])
                    .coord_maps(Some([map]))
                    // a reflecting wall exerts no work on gas at rest, so the hydrostatic
                    // state is a fixed point of the boundary as well as of the interior.
                    .boundaries(Boundaries::uniform(BoundaryType::Reflect))
                    .cfl(CFL)
                    .allocate()
                    .expect("sim construction failed")
                    // the IC closure receives the map's cell centers (`stagger_coord`
                    // honors the attached maps), so the seeded column and the balanced
                    // ladder agree on every cell position — the exactness premise.
                    .set_initial(|x: [f64; 1]| hydrostatic(&phi, x[0], $x_outer))
                    .build();
                let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
                    .with_solver(Solver::HllcPlus)
                    .expect("solver/regime mismatch")
                    .well_balanced_reconstruction(balanced);
                Hierarchy::single(sim, kernels).with_bodies(
                    symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                        0,
                        symbi_algebra::Tensor::new([$body_pos]),
                        symbi_algebra::Tensor::zeros(),
                        GM,
                        // pointlike mask; soft rides the softening slot so the body's
                        // field is the same plummer potential the column is built from.
                        1.0e-6,
                        SOFT,
                    )),
                )
            }

            /// the smallest `K/K_0` and the largest |v| inside the window, with cell
            /// centers read from the same axis map the scheme and the seeding use. an
            /// adiabatic gas holds its entropy as a one-way floor, so anything below one
            /// in the first is the scheme's own deficit; the second is the stagnancy
            /// precondition.
            fn run(balanced: bool) -> (f64, f64) {
                let mut hier = build(balanced);
                hier.evolve_steps(STEPS).unwrap();
                let st = &hier.levels[0].state;
                let map: AxisMap = $map;
                let window: (f64, f64) = $window;
                let rho = st.fields.prim.rho.view();
                let pre = st.fields.prim.pre_field().expect("prim.pre").view();
                let vel = st.fields.prim.vel[0].view();
                let mut worst = f64::INFINITY;
                let mut vmax = 0.0_f64;
                let mut in_window = 0usize;
                for ii in st.geom.interior.spaces[0].lo..st.geom.interior.spaces[0].hi {
                    let x = map.center(ii);
                    if x < window.0 || x > window.1 {
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
            fn the_balanced_triple_holds_the_graded_column() {
                let map: AxisMap = $map;
                // the graded premise, asserted so the gate reports its own irrelevance:
                // the cell widths must genuinely vary across the domain (a map that
                // collapsed to uniform would test nothing new).
                let grading = map.width(N as isize - 1) / map.width(0);
                let g = if grading < 1.0 { 1.0 / grading } else { grading };
                // measured gradings: 1.989x (log over [1, 2] -- the width ratio equals
                // the radius ratio) and 12.4x (geometric, ratio 1.02^127).
                assert!(
                    g > 1.5,
                    "the axis map grades by only {g:.3}x across the domain; this gate \
                     no longer exercises non-uniform spacing"
                );

                let (k_plain, v_plain) = run(false);
                let (k_wb, v_wb) = run(true);
                let (d_plain, d_wb) = ((1.0 - k_plain).max(0.0), (1.0 - k_wb).max(0.0));
                println!(
                    "\nsealed graded column ({}), {STEPS} steps, grading {g:.2}x\n\
                     plain reconstruction:    min K/K_0 {k_plain:.12} (deficit {d_plain:.3e}), max|v| {v_plain:.3e}\n\
                     balanced reconstruction: min K/K_0 {k_wb:.12} (deficit {d_wb:.3e}), max|v| {v_wb:.3e}",
                    stringify!($modname)
                );

                // the discriminating quantity is motion. the plain arm carries both the
                // analytic-source truncation mismatch and the mirrored-ghost wall kick, so
                // it moves measurably -- that is what makes the balanced arm's stagnation a
                // statement about the triple, on a setup that is demonstrably live.
                assert!(
                    v_plain > 1.0e-7,
                    "the PLAIN arm sits at |v| = {v_plain:.3e}; the column is not \
                     exercising the imbalance and the balanced arm's stagnation proves \
                     nothing. lengthen the run"
                );
                assert!(
                    v_wb * 1.0e6 < v_plain,
                    "the balanced arm's residual flow ({v_wb:.3e}) is within six orders \
                     of the plain arm's ({v_plain:.3e}); the triple is no longer holding \
                     the discrete equilibrium"
                );
                assert!(
                    v_wb < $v_wb_bound,
                    "balanced-arm residual velocity {v_wb:.3e}; the discrete equilibrium \
                     is drifting"
                );
                assert!(
                    d_wb < $d_wb_bound,
                    "balanced-arm entropy deficit {d_wb:.3e}; the sealed column is \
                     venting through a path the balanced triple was supposed to close"
                );
            }
        }
    };
}

/// the log-radial map covering r in [1, 2]: face(i) = 10^(i log10(2)/N).
fn log_map() -> AxisMap {
    AxisMap::Log {
        start: 1.0,
        log_slope: 2.0_f64.log10() / N as f64,
    }
}

/// the geometrically graded cartesian map covering x in [0, 1] with width ratio 1.02
/// (widths span ~12x across the domain).
fn geometric_map() -> AxisMap {
    let ratio = 1.02_f64;
    AxisMap::Geometric {
        start: 0.0,
        width: (ratio - 1.0) / (ratio.powi(N as i32) - 1.0),
        ratio,
    }
}

/// the plummer-softened potential of the central body, on the gas radius.
fn phi_central(r: f64) -> f64 {
    -GM / (r * r + SOFT * SOFT).sqrt()
}

/// the plummer-softened potential of the body one domain width left of x = 0, so the
/// gas at x feels the mass at radius x + 1 and the domain covers r in [1, 2].
fn phi_offset(x: f64) -> f64 {
    let r = x + 1.0;
    -GM / (r * r + SOFT * SOFT).sqrt()
}

// machine equilibrium, absolutely: measured at 400 wall-inclusive steps the balanced
// arms sit at |v| = 2.4e-15 / deficit exactly 0 (log spherical) and |v| = 7.1e-16 /
// deficit 1.1e-16, one ulp (geometric cartesian); the plain arms vent at |v| = 7.6e-6 /
// 6.1e-6 with deficits 4.7e-9 / 2.5e-8. the bounds carry roughly three orders of margin
// over the balanced measurements and sit six orders under the plain vents.
//
// on the log axis the map center is the geometric mean, displaced from the arithmetic
// face midpoint by ~r (dr/r)^2 / 8 ~ 4e-6 -- nine orders above the balanced bound. these
// levels therefore pass only for a ladder anchored at the map center; the gate separates
// the two center definitions, with tolerance for either ruled out.
sealed_graded_gate!(
    log_spherical,
    symbi_geometry::Spherical,
    symbi_geometry::Spherical,
    log_map(),
    0.0,
    phi_central,
    2.0,
    (1.35, 1.65),
    1.0e-12,
    1.0e-12
);
sealed_graded_gate!(
    geometric_cartesian,
    symbi_geometry::Cartesian,
    symbi_geometry::Cartesian,
    geometric_map(),
    -1.0,
    phi_offset,
    1.0,
    (0.35, 0.65),
    1.0e-12,
    1.0e-12
);

/// the center-separation premise of the log gate, stated as its own check: the log
/// map's centers (geometric mean) and the arithmetic face midpoints must differ by
/// many orders more than the exactness bound, or the log gate would pass with either
/// center definition and prove nothing about the ladder's choice.
#[test]
fn the_log_map_separates_the_two_center_definitions() {
    let map = log_map();
    let mut max_sep = 0.0_f64;
    for ii in 0..N as isize {
        let arith = 0.5 * (map.face(ii) + map.face(ii + 1));
        max_sep = max_sep.max((arith - map.center(ii)).abs());
    }
    assert!(
        max_sep > 1.0e-7,
        "arithmetic and geometric centers separate by only {max_sep:.3e}; the log \
         gate cannot distinguish the ladder's center definition at the 1e-12 level"
    );
}
