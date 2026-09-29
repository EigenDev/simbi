// =============================================================================
// refined_correction_locality.rs
//
// a first-order correction on a refined level is local: its redo recomputes
// the level's update from the stage input and keeps the high-order flux on
// every face but the troubled cell's, so beyond the troubled cell's reach the
// corrected stage equals the uncorrected one bit for bit. on a refined root
// that includes the composite step the driver applies to every stage update:
// the covered core, which restriction later replaces, holds its stage input,
// and the covered cells supply the fine level's coarse-fine prolongation data.
//
// the flow is a gaussian overpressure centered on a refined patch (root cells
// 12..20 of a 32 x 32 grid, ratio 2); a root cell at (8, 4), far from the patch
// and at rest, is raised troubled in the third step's predictor and, in a
// second run, its corrector. at the end of that step every fine cell and every
// root cell more than five cells from the flag equals the clean run, and the
// flag changed the cells beside it. the same holds with the initial state
// declared as a stationary target, whose residual the composite step adds back
// to every stage update.
//
// run: cargo test -p symbi --test refined_correction_locality
// =============================================================================

use std::ops::ControlFlow;
use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = AdiabaticSubstrateKernelSet<HostMemory, f64, 2>;
type Hier = Hierarchy<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, Kern>;

const N: usize = 32;
const DX: f64 = 1.0 / N as f64;
const GAMMA: f64 = 1.4;
const CFL: f64 = 0.4;
const FLAG: [isize; 2] = [8, 4];
/// the troubled cell's reach over one RK2 step with a two-cell reconstruction stencil
const REACH: isize = 5;
/// the flagged step (the third) and the step count the comparison runs to
const FLAG_ITERATION: u64 = 2;
const STEPS: usize = 3;

fn bump(x: f64, y: f64) -> f64 {
    0.3 * (-(((x - 0.5) / 0.08).powi(2) + ((y - 0.5) / 0.08).powi(2))).exp()
}

fn initial([x, y]: [f64; 2]) -> Prim<f64, 2> {
    let b = bump(x, y);
    Prim::adiabatic(Density(1.0 + b), Tensor::new([0.0, 0.0]), Pressure(1.0 + b))
}

fn build(declared: bool) -> Hier {
    let root = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([DX; 2])
        .origin([0.0; 2])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(CFL)
        .ghosts(2)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("root sim")
        .set_initial(initial)
        .build();
    let k = Kern::new(GAMMA, CFL, &root.geom.allocated);
    let patch = RefinementRegion { x_lo: [12.0 * DX; 2], x_hi: [20.0 * DX; 2] };
    let mut h = Hier::with_refinement(root, k, &[patch], ProlongOrder::Ppm, |s| Kern::new(GAMMA, CFL, &s.geom.allocated))
        .expect("refined hierarchy");
    h.seed_fine_from_coarse().expect("fine seed");
    if declared {
        h = h.with_equilibrium(initial).expect("declared target");
    }
    h.prime();
    h
}

/// the root and fine conserved density after `STEPS` steps, each cell with its level and its
/// index on that level's interior, the root cell raised troubled at `stage` of the flagged step
/// when given, the initial state declared as the stationary target when `declared` holds
fn run(declared: bool, stage: Option<usize>) -> Vec<(usize, [isize; 2], u64)> {
    let mut h = build(declared);
    if let Some(stage) = stage {
        let s = &h.levels[0].state;
        let cell = std::array::from_fn(|a| s.geom.interior.spaces[a].lo + FLAG[a]);
        *s.workspace.troubled_once.lock().unwrap() = Some((FLAG_ITERATION, stage, cell, false));
    }
    let mut steps = 0;
    h.evolve_with_callback(10.0, 1, |_| {
        steps += 1;
        if steps >= STEPS { ControlFlow::Break(()) } else { ControlFlow::Continue(()) }
    })
    .expect("evolve");
    let mut out = Vec::new();
    for (level, l) in h.levels.iter().enumerate() {
        let s = &l.state;
        let lo: [isize; 2] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo);
        for c in s.geom.interior.iter() {
            let rel: [isize; 2] = std::array::from_fn(|a| c[a] - lo[a]);
            out.push((level, rel, s.fields.cons.den.view().at(c).to_bits()));
        }
    }
    out
}

fn beyond_reach(level: usize, rel: [isize; 2]) -> bool {
    level > 0 || (0..2).any(|a| (rel[a] - FLAG[a]).abs() > REACH)
}

fn assert_local(declared: bool) {
    let clean = run(declared, None);
    for stage in [0usize, 1] {
        let flagged = run(declared, Some(stage));
        assert_eq!(clean.len(), flagged.len());
        let near_changed = clean
            .iter()
            .zip(&flagged)
            .filter(|(a, b)| !beyond_reach(a.0, a.1) && a.2 != b.2)
            .count();
        assert!(near_changed > 0, "stage {stage}: the flag changed no cell beside it; the gate is vacuous");
        let far: Vec<_> = clean
            .iter()
            .zip(&flagged)
            .filter(|(a, b)| beyond_reach(a.0, a.1) && a.2 != b.2)
            .map(|(a, _)| (a.0, a.1))
            .collect();
        assert!(
            far.is_empty(),
            "stage {stage}: the correction at {FLAG:?} changed {} cells beyond its reach, first (level, cell) {:?}",
            far.len(),
            &far[..far.len().min(6)]
        );
    }
}

#[test]
fn a_root_correction_leaves_the_refined_state_beyond_its_reach_untouched() {
    assert_local(false);
}

#[test]
fn a_root_correction_under_a_declared_target_stays_local() {
    assert_local(true);
}
