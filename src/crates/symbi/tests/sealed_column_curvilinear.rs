// =============================================================================
// sealed_column_curvilinear.rs
//
// the curvilinear twin of `sealed_column_unclamped`: the balanced triple
// (reconstruction + area-weighted equilibrium-pressure source + balance-aware
// ghosts) holding a sealed, stagnant, strongly stratified radial column on the
// spherical and cylindrical charts. the chart is what changes the statement:
// the momentum update now carries the area-weighted pressure flux divergence
// together with the geometric pressure source `p (A_hi - A_lo)/V`, and the equilibrium
// is a discrete fixed point only because the wb gravity source is exactly the
// three-way telescoping remainder
//
//   S_m = [A_hi (p_eq(phi_hi) - p_eq(phi_c)) - A_lo (p_eq(phi_lo) - p_eq(phi_c))] / V
//
// with A/V the same `cell_geometry` factors the divergence and the geometric
// source use. the plain arm at the same clock is each gate's positive control:
// its analytic rho*g source mismatches the discrete pressure gradient at
// truncation order and its mirrored ghosts kick the walls, so it must vent
// measurably or the balanced arm's stagnation proves nothing.
//
// the column is the isentrope of the plummer-softened potential of a central
// point mass at the chart origin -- the same `body_potential` family the
// reconstruction, source and ghosts all evaluate, so the only remaining
// imbalance is roundoff.
//
// run: cargo test -p symbi --test sealed_column_curvilinear -- --nocapture
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
const GM: f64 = 100.0;
/// plummer softening of the central body's field. the column is built from the same
/// softened potential, so the state, the applied gravity and the balance are one field;
/// at r >= 1 the softening correction is ~1e-6 relative and the column is still
/// strongly stratified.
const SOFT: f64 = 1.0e-3;
/// the radial domain [1, 2]: the central mass sits at the chart origin, one domain
/// width inside the inner wall, so the potential is genuinely curved across the
/// column and the origin singularity is excluded.
const R_IN: f64 = 1.0;
const STEPS: u64 = 400;
/// the causally-clean measurement window, in r.
const WINDOW: (f64, f64) = (1.35, 1.65);

/// the plummer-softened potential of the central body, on the gas radius.
fn phi(r: f64) -> f64 {
    -GM / (r * r + SOFT * SOFT).sqrt()
}

/// the isentropic column in hydrostatic balance against the softened field, from the
/// bernoulli invariant `gamma K0/(gamma-1) rho^(gamma-1) + phi = const`, normalized to
/// `rho = 1` at the outer wall (r = 2). the hydrostatic ode `dp/dr = -rho dphi/dr`
/// carries no chart factor -- the geometric terms of a radial momentum balance cancel
/// between the area-weighted divergence and the geometric source -- so one profile
/// serves both charts.
fn hydrostatic(x: [f64; 1]) -> Prim<f64, 1> {
    let a = (GAMMA - 1.0) / (GAMMA * K0);
    let c = 1.0 / a + phi(2.0);
    let rho = (a * (c - phi(x[0]))).powf(1.0 / (GAMMA - 1.0));
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

macro_rules! sealed_radial_gate {
    ($modname:ident, $chart:ty, $chart_val:expr, $v_wb_bound:expr, $d_wb_bound:expr) => {
        mod $modname {
            use super::*;

            type Sim = SimState<Newtonian, 1, $chart, IdealGas<f64>, CpuSpace, HostMemory>;
            type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;
            type Hier =
                Hierarchy<Newtonian, 1, 1, $chart, IdealGas<f64>, CpuSpace, HostMemory, Kset>;

            fn build(balanced: bool) -> Hier {
                let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, $chart_val)
                    .cells([N])
                    .origin([R_IN])
                    .spacing([1.0 / N as f64])
                    // a reflecting wall exerts no work on gas at rest, so the hydrostatic
                    // state is a fixed point of the boundary as well as of the interior.
                    .boundaries(Boundaries::uniform(BoundaryType::Reflect))
                    .cfl(CFL)
                    .allocate()
                    .expect("sim construction failed")
                    .set_initial(hydrostatic)
                    .build();
                let kernels = Kset::new(GAMMA, CFL, &sim.geom.allocated)
                    .with_solver(Solver::HllcPlus)
                    .expect("solver/regime mismatch")
                    .well_balanced_reconstruction(balanced);
                Hierarchy::single(sim, kernels).with_bodies(
                    symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
                        0,
                        // the chart origin: the radial grid axis embeds on a cartesian
                        // ray from it, so a body at zero is the central point mass.
                        symbi_algebra::Tensor::new([0.0]),
                        symbi_algebra::Tensor::zeros(),
                        GM,
                        // pointlike mask; soft rides the softening slot so the body's
                        // field is the same plummer potential the column is built from.
                        1.0e-6,
                        SOFT,
                    )),
                )
            }

            /// the smallest `K/K_0` and the largest |v| away from the walls. an adiabatic
            /// gas holds its entropy as a one-way floor, so anything below one in the first
            /// is the scheme's own deficit; the second is the stagnancy precondition.
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
                    let r = R_IN + ((ii - ilo) as f64 + 0.5) * dx;
                    if r < WINDOW.0 || r > WINDOW.1 {
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
            fn the_balanced_triple_holds_the_radial_column() {
                let (k_plain, v_plain) = run(false);
                let (k_wb, v_wb) = run(true);
                let (d_plain, d_wb) = ((1.0 - k_plain).max(0.0), (1.0 - k_wb).max(0.0));
                println!(
                    "\nsealed radial column ({}), {STEPS} steps\n\
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

// machine equilibrium, absolutely: measured at 400 wall-inclusive steps |v| = 3.2e-15
// (spherical) / 3.9e-15 (cylindrical) and deficit = 1.1e-16 (one ulp) on both charts;
// the plain arms vent at |v| = 1.6e-5 / 1.8e-5 with deficits 8.4e-10 / 1.7e-10. the
// bounds carry roughly three orders of margin over the balanced measurements and sit
// seven orders under the plain vent.
sealed_radial_gate!(
    spherical,
    symbi_geometry::Spherical,
    symbi_geometry::Spherical,
    1.0e-12,
    1.0e-12
);
sealed_radial_gate!(
    cylindrical,
    symbi_geometry::Cylindrical,
    symbi_geometry::Cylindrical,
    1.0e-12,
    1.0e-12
);
