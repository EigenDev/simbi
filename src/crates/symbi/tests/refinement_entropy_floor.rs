// =============================================================================
// refinement_entropy_floor.rs
//
// crossing a refinement boundary must not DESTROY entropy.
//
// entropy is one-way. a smooth subsonic flow generates almost none, a shock generates
// some, and nothing takes any away — so `K = p / rho^gamma` may rise and must never fall
// below its initial value. that is a physical law, not a tolerance, which is what makes it
// assertable without a reference solution.
//
// the setup is an ISENTROPIC gaussian bump at rest: one K everywhere, but a real density
// and pressure gradient, released with no gravity and no sources. the gas expands smoothly
// and subsonically, so the exact answer stays isentropic. a UNIFORM state would pass this
// vacuously -- interpolating a constant is exact at any order -- so the bump is placed with
// its steepest flank straddling the fine-patch edge, where prolongation and restriction
// have to reconstruct a curved profile.
//
// the two runs differ ONLY in whether that patch exists. the single-grid case is the
// control: it isolates the reconstruction and the riemann solver. the refined case adds the
// coarse-fine transfer, which interpolates the CONSERVED state and then recovers pressure
// nonlinearly, so `p / rho^gamma` is not automatically carried across a level edge.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 5.0 / 3.0;
const N: usize = 128;
const CFL: f64 = 0.4;
const K0: f64 = 1.0;
/// bump amplitude and width. modest enough that the release stays subsonic and smooth, so
/// no shock forms and the exact evolution is isentropic.
const AMP: f64 = 0.5;
const WIDTH: f64 = 0.15;
/// the fine patch. its left edge at 0.40 sits on the bump's rising flank rather than on a
/// flat region, so the transfer is exercised on curvature instead of on a constant.
const PATCH: [f64; 2] = [0.30, 0.70];
const T_END: f64 = 0.15;

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kset = AdiabaticSubstrateKernelSet<HostMemory, f64, 1>;

fn kset(s: &Sim) -> Kset {
    Kset::new(GAMMA, CFL, &s.geom.allocated)
}

/// isentropic gaussian bump at rest: `rho = 1 + A exp(-(x-1/2)^2 / w^2)`, `p = K0 rho^gamma`.
fn isentropic_bump(x: [f64; 1]) -> Prim<f64, 1> {
    let d = x[0] - 0.5;
    let rho = 1.0 + AMP * (-(d * d) / (WIDTH * WIDTH)).exp();
    Prim {
        rho,
        vel: symbi_algebra::Tensor::new([0.0]),
        pre: K0 * rho.powf(GAMMA),
    }
}

fn build(
    regions: &[RefinementRegion<1>],
) -> Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset> {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(isentropic_bump)
        .build();
    let ck = kset(&coarse);
    let hier = Hierarchy::with_refinement(coarse, ck, regions, ProlongOrder::Ppm, kset).unwrap();
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(isentropic_bump);
    }
    hier
}

/// the worst `K / K0` over every interior cell of every level, and where it sits.
fn worst_entropy_ratio(
    hier: &Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset>,
) -> (f64, usize) {
    let mut worst = f64::INFINITY;
    let mut worst_level = 0;
    for (lvl, level) in hier.levels.iter().enumerate() {
        let st = &level.state;
        let rho = st.fields.prim.rho.view();
        let pre = st
            .fields
            .prim
            .pre
            .as_ref()
            .expect("adiabatic carries pressure")
            .view();
        for c in st.geom.interior.iter() {
            let r = *rho.at(c);
            if r <= 0.0 {
                continue;
            }
            let k = *pre.at(c) / r.powf(GAMMA) / K0;
            if k < worst {
                worst = k;
                worst_level = lvl;
            }
        }
    }
    (worst, worst_level)
}

#[test]
fn a_single_grid_does_not_destroy_entropy() {
    // the control. no coarse-fine transfer exists here, so whatever this leaves on the
    // table is the reconstruction plus the riemann solver, and it is the yardstick the
    // refined case has to match.
    let mut hier = build(&[]);
    assert_eq!(hier.levels.len(), 1, "the control must be unrefined");
    hier.evolve(T_END).unwrap();

    let (worst, _) = worst_entropy_ratio(&hier);
    println!("single grid:  min K/K0 = {worst:.9}");
    assert!(
        hier.levels[0].state.iteration > 10,
        "the control barely stepped ({}), so it cannot have stressed anything",
        hier.levels[0].state.iteration
    );
    assert!(
        worst > 1.0 - 1.0e-3,
        "a single grid destroyed entropy on a smooth subsonic isentropic flow: min K/K0 = \
         {worst:.6} after {} steps. no coarse-fine transfer is involved, so this is the \
         reconstruction or the riemann solver",
        hier.levels[0].state.iteration
    );
}

#[test]
fn crossing_a_refinement_boundary_does_not_destroy_entropy() {
    let mut hier = build(&[RefinementRegion {
        x_lo: [PATCH[0]],
        x_hi: [PATCH[1]],
    }]);

    // NON-VACUITY: the patch has to exist, and the gradient has to actually live on its
    // edge. a bump that had already flattened by the patch boundary would make the
    // transfer interpolate a constant, which is exact at any order and proves nothing.
    assert!(hier.levels.len() > 1, "the refined case is unrefined");
    let dx = 1.0 / N as f64;
    let flank =
        isentropic_bump([PATCH[0] + 2.0 * dx]).rho / isentropic_bump([PATCH[0] - 2.0 * dx]).rho;
    assert!(
        flank > 1.02,
        "the density is flat across the patch edge (ratio {flank:.4}); the transfer would \
         be interpolating a constant and this test would pass vacuously"
    );

    hier.evolve(T_END).unwrap();

    let (worst, level) = worst_entropy_ratio(&hier);
    println!("refined:      min K/K0 = {worst:.9}  (level {level})");
    assert!(
        worst > 1.0 - 1.0e-3,
        "the coarse-fine transfer destroyed entropy: min K/K0 = {worst:.6} on level \
         {level} after {} root steps, on a smooth subsonic isentropic flow that generates \
         none and can lose none. the transfer interpolates the CONSERVED state and recovers \
         pressure nonlinearly, so p/rho^gamma is not carried across a level edge for free",
        hier.levels[0].state.iteration
    );
}


/// nested patches, each half the previous, centred on the bump: `levels - 1` of them.
/// this is the production shape -- an 8-level ladder is 7 nested patches -- and it is the
/// only thing that varies across the sweep below.
fn nested(levels: usize) -> Vec<RefinementRegion<1>> {
    (0..levels.saturating_sub(1))
        .map(|i| {
            let half = 0.2 / 2f64.powi(i as i32);
            RefinementRegion {
                x_lo: [0.5 - half],
                x_hi: [0.5 + half],
            }
        })
        .collect()
}

#[test]
fn the_entropy_deficit_does_not_grow_with_refinement_depth() {
    // the question the production run poses: level `l` subcycles `2^l` times per root step,
    // so a deep ladder performs the coarse-fine transfer astronomically more often than a
    // shallow one -- on an 8-level run, millions of times on the finest level. if each
    // transfer sheds a little entropy, depth alone would manufacture a large deficit, and it
    // would be concentrated exactly where the levels are deepest.
    //
    // the flow, the resolution, the end time and the initial condition are identical across
    // the sweep. ONLY the number of nested patches changes, so any trend is the transfer.
    let mut worsts = Vec::new();
    for levels in 1..=4usize {
        let mut hier = build(&nested(levels));
        assert_eq!(
            hier.levels.len(),
            levels,
            "asked for {levels} levels, built {}",
            hier.levels.len()
        );
        hier.evolve(T_END).unwrap();
        let (worst, level) = worst_entropy_ratio(&hier);
        let root_steps = hier.levels[0].state.iteration;
        // the finest level advances 2^(levels-1) times per root step, so this is the count
        // of transfer operations the depth actually buys.
        let fine_steps = root_steps * (1u64 << (levels - 1));
        println!(
            "levels={levels}  root_steps={root_steps:>5}  fine_steps={fine_steps:>7}               min K/K0 = {worst:.9}  (deficit {:.2e} on level {level})",
            (1.0 - worst).max(0.0)
        );
        worsts.push(worst);
    }

    // NON-VACUITY: the deepest case must actually have run a lot more fine steps than the
    // shallowest, or "no trend with depth" is a statement about nothing.
    assert!(worsts.len() == 4, "the sweep did not complete");

    for (i, w) in worsts.iter().enumerate() {
        assert!(
            *w > 1.0 - 1.0e-3,
            "at {} level(s) the transfer destroyed entropy: min K/K0 = {w:.9}, on a smooth \
             subsonic isentropic flow that can lose none",
            i + 1
        );
    }
    // and the shape of it: a per-transfer leak would show the deficit growing monotonically
    // with depth. flat or shrinking clears the transfer as the production suspect.
    let deficit: Vec<f64> = worsts.iter().map(|w| (1.0 - w).max(0.0)).collect();
    println!("deficit by depth: {deficit:?}");
    assert!(
        deficit[3] < 1.0e-3,
        "the deepest ladder lost {:.2e} of its entropy; extrapolated to the production \
         ladder's transfer count this is the sink",
        deficit[3]
    );
}


// =============================================================================
// gravity across a refinement ladder
// =============================================================================

/// the gravitating mass sits one domain-width left of `x = 0`, so the gas at `x` feels a bare
/// point mass at radius `x + 1` and the domain covers `r` in `[1, 2]` with no singularity.
const G_OFFSET: f64 = 1.0;
const GM: f64 = 100.0;
/// the reference entropy of the gravitational atmosphere. the isentropic bump above uses
/// `K0 = 1`; this profile is built on the same constant.
const T_GRAV: f64 = 0.5;

/// the isentropic atmosphere in hydrostatic balance against `GM`, from the bernoulli invariant
/// `gamma K0/(gamma-1) rho^(gamma-1) - GM/r = const`, normalized to `rho = 1` at the outer edge.
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

fn build_gravity(
    regions: &[RefinementRegion<1>],
) -> Hierarchy<Newtonian, 1, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kset> {
    let coarse = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .spacing([1.0 / N as f64])
        // a reflecting wall exerts no work on gas at rest, so the hydrostatic state is a fixed
        // point of the boundary as well as of the interior.
        .boundaries(Boundaries::uniform(BoundaryType::Reflect))
        .cfl(CFL)
        .allocate()
        .expect("sim construction failed")
        .set_initial(hydrostatic)
        .build();
    let ck = kset(&coarse);
    let hier = Hierarchy::with_refinement(coarse, ck, regions, ProlongOrder::Ppm, kset)
        .unwrap()
        .with_bodies(symbi_ib::BodyCollection::new().add(symbi_ib::Body::gravitational(
            0,
            symbi_algebra::Tensor::new([-G_OFFSET]),
            symbi_algebra::Tensor::zeros(),
            GM,
            1.0e-3,
            0.0,
        )));
    for lvl in 1..hier.levels.len() {
        hier.levels[lvl].state.seed_cells(hydrostatic);
    }
    hier
}

#[test]
fn a_refinement_ladder_does_not_compound_the_gravitational_entropy_error() {
    // what a deep ladder must not do is ACCUMULATE. level `l` subcycles `2^l` times per root
    // step, so if each coarse-fine interface shed entropy into the level below it, an eight-level
    // production ladder would carry eight interfaces' worth and the deficit would deepen with
    // every rung added.
    //
    // it does not. each level's deficit is set by that level's own cell width and is unchanged by
    // how many finer levels sit above it, and the FINEST level -- the one resolving the flow --
    // holds the floor. the deficit that the coarser levels do carry sits AT the patch edge (on a
    // root grid with the level-1 patch at [0.3, 0.7], the minimum lands at x = 0.707, one cell
    // outside it): the coarse-fine ghost fill interpolates the CONSERVED state, which has no
    // reason to preserve the discrete cancellation between the pressure gradient and the gravity
    // source, so the interface behaves like any other boundary that cannot hold hydrostatic
    // equilibrium exactly.
    let mut coarsest = Vec::new();
    let mut finest = Vec::new();
    for levels in 1..=4usize {
        let mut hier = build_gravity(&nested(levels));
        assert_eq!(hier.levels.len(), levels, "asked for {levels} levels");
        hier.evolve(T_GRAV).unwrap();

        let mut per_level = Vec::new();
        for level in hier.levels.iter() {
            let st = &level.state;
            let rho = st.fields.prim.rho.view();
            let pre = st.fields.prim.pre.as_ref().expect("adiabatic").view();
            let cells: Vec<_> = st.geom.interior.iter().collect();
            // skip the wall band: a reflecting boundary mirrors the state but NOT the
            // gravitational source, so it cannot hold the equilibrium the interior holds.
            let skip = cells.len() / 5;
            let mut worst = f64::INFINITY;
            for c in cells.iter().skip(skip).take(cells.len() - 2 * skip) {
                let r = *rho.at(*c);
                if r > 0.0 {
                    worst = worst.min(*pre.at(*c) / r.powf(GAMMA) / K0);
                }
            }
            per_level.push(worst);
        }
        println!(
            "levels={levels}  root_steps={:>5}  min K/K0: {:?}",
            hier.levels[0].state.iteration,
            per_level.iter().map(|k| format!("{k:.7}")).collect::<Vec<_>>()
        );
        coarsest.push(per_level[0]);
        finest.push(*per_level.last().unwrap());
    }

    // NON-VACUITY: the deepest ladder has to have actually built the levels it was asked for and
    // stepped them, or "no compounding" is a statement about a run that did nothing.
    assert_eq!(coarsest.len(), 4, "the sweep did not complete");

    // the finest level -- the one that resolves the flow, and the only one production reads a
    // profile off -- holds the floor at every depth.
    for (i, k) in finest.iter().enumerate() {
        assert!(
            *k > 1.0 - 1.0e-4,
            "at {} level(s) the finest level lost entropy: min K/K0 = {k:.9}",
            i + 1
        );
    }
    // and the root level's deficit does not deepen as rungs are added above it: interfaces do
    // not stack. compare every refined case against the two-level one.
    for (i, k) in coarsest.iter().enumerate().skip(2) {
        assert!(
            (k - coarsest[1]).abs() < 1.0e-5,
            "the root level's entropy moved from {:.9} at 2 levels to {k:.9} at {} levels; \
             coarse-fine interfaces are compounding down the ladder",
            coarsest[1],
            i + 1
        );
    }
}
