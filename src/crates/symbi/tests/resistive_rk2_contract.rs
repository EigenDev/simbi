// =============================================================================
// resistive_rk2_contract.rs
//
// the ohmic edge EMF eta J in the rk2 constrained transport: each stage forms
// its complete EMF, ideal and resistive, from its own stage input, and the
// corrector's EMF is the average of the two complete stage EMFs:
// - predictor: E_1 + eta J(B^n), saved, then curled from B^n;
// - corrector: E_2 + eta J(B*), averaged with the saved predictor EMF, then
//   curled from the restored B^n;
// - a first-order correction adds eta J of the stage-input faces to its
//   first-order EMF once; edges away from the flagged cell keep the saved
//   high-order EMF bit for bit.
// the fixture is 3D newtonian MHD, B = b (sin 2 pi y, sin 2 pi z, sin 2 pi x)
// (divergence-free on the staggered faces) in a flow that moves it, so the
// predictor field B* carries a current materially different from J(B^n). the
// resistive part of each stage is isolated against twins that run the same
// stage ideal: A resistive in both stages, B ideal in both, C resistive in the
// predictor and ideal in the corrector. the current J is the code's own
// resistive operator, verified as the curl adjoint elsewhere.
//
// run: cargo test -p symbi --test resistive_rk2_contract
// =============================================================================

use symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
use symbi::sim::evolve::KernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_grid::Field;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::stage::{HookPoint, StageArgs, fold_stage};
use symbi_substrate::regimes::mhd_substrate::apply_resistive_emf_on;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<NewtonianMhd, 3, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = NewtonianMhdSubstrateKernelSet<HostMemory, f64, 3>;

const N: usize = 12;
const GAMMA: f64 = 5.0 / 3.0;
const ETA: f64 = 0.01;
const B1: f64 = 0.3;
const V0: f64 = 0.5;
const DT: f64 = 0.01;
/// the cell poisoned just before a stage's recovery to force a first-order correction
const POISONED: [usize; 3] = [6, 6, 6];

fn sim() -> Sim {
    let tau = std::f64::consts::TAU;
    let dx = 1.0 / N as f64;
    Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y, z]| {
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([V0 * (tau * z).sin(), V0 * (tau * x).sin(), V0 * (tau * y).sin()]), Pressure(1.0)),
                Tensor::new([B1 * (tau * y).sin(), B1 * (tau * z).sin(), B1 * (tau * x).sin()]),
            )
        })
        .seed_faces(|axis, [x, y, z]| B1 * [(tau * y).sin(), (tau * z).sin(), (tau * x).sin()][axis])
        .build()
        .with_rollback_snapshot()
        .expect("rollback snapshot")
}

fn kernels(s: &Sim, eta: f64) -> Kern {
    Kern::new(GAMMA, 0.3, 1.5, &s.geom.allocated).with_resistivity(eta)
}

fn args(stage: usize) -> StageArgs {
    let stages = Timestepping::Rk2.stages();
    StageArgs {
        dt: DT,
        a0: stages[stage].0,
        ac: stages[stage].1,
        stage,
        n_stages: 2,
        injection_weight: symbi_sim::driver::downstream_injection_weight(stages, stage),
        allow_elision: false,
    }
}

fn edges(fields: &[Field<f64, 3, HostMemory>]) -> Vec<Vec<f64>> {
    fields.iter().map(|f| f.domain().iter().map(|c| *f.view().at(c)).collect()).collect()
}

fn faces(s: &Sim) -> Vec<Vec<f64>> {
    let mhd = s.fields.mhd.as_ref().unwrap();
    mhd.bface.b.iter().map(|f| f.domain().iter().map(|c| *f.view().at(c)).collect()).collect()
}

/// one rk2 step with `predictor` and `corrector` kernel sets, the cell poisoned before the
/// recovery of stage `poison` when given: the saved predictor EMF, the corrector's stage-input
/// faces, the final EMF, and whether a correction ran
struct Step {
    predictor_emf: Vec<Vec<f64>>,
    bn: Vec<Vec<f64>>,
    bstar: Vec<Vec<f64>>,
    final_emf: Vec<Vec<f64>>,
    corrected: bool,
}

fn step(eta_predictor: f64, eta_corrector: f64, poison: Option<usize>) -> Step {
    let s = sim();
    let (kp, kc) = (kernels(&s, eta_predictor), kernels(&s, eta_corrector));
    kp.c2p(&s);
    kp.ghost_fill(&s);
    let bn = faces(&s);
    kp.snapshot_retry(&s);
    kp.snapshot(&s);
    let cell: [isize; 3] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo + POISONED[a] as isize);
    let mut corrected = false;
    let mut run = |k: &Kern, stage: usize| {
        let nrg = s.fields.cons.nrg_field().unwrap();
        let mut hook = |hp: HookPoint| {
            if poison == Some(stage) && matches!(hp, HookPoint::BeforeC2p) {
                *nrg.view_mut().at_mut(cell) = -1.0;
            }
        };
        let _ = fold_stage(&s, k, args(stage), &mut hook);
        // the store's own troubled flag at the poisoned cell: the correction's mark
        corrected |= poison == Some(stage) && *s.workspace.fofc_flag.view().at(cell) > 0.0;
    };
    run(&kp, 0);
    let mhd = s.fields.mhd.as_ref().unwrap();
    let predictor_emf = edges(&mhd.efield_n.e);
    let bstar = faces(&s);
    run(&kc, 1);
    let final_emf = edges(&mhd.efield.e);
    Step { predictor_emf, bn, bstar, final_emf, corrected }
}

/// eta J of `face` values through the code's resistive operator, on the fixture's edges
fn eta_j(face_values: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let s = sim();
    let mhd = s.fields.mhd.as_ref().unwrap();
    for (d, f) in mhd.bface.b.iter().enumerate() {
        for (c, v) in f.domain().iter().zip(&face_values[d]) {
            *f.view_mut().at_mut(c) = *v;
        }
    }
    for e in &mhd.efield.e {
        for c in e.domain().iter() {
            *e.view_mut().at_mut(c) = 0.0;
        }
    }
    let faces: Vec<&Field<f64, 3, HostMemory>> = mhd.bface.b.iter().collect();
    apply_resistive_emf_on(&*s, ETA, &faces);
    edges(&mhd.efield.e)
}

fn max_abs(a: &[Vec<f64>]) -> f64 {
    a.iter().flatten().fold(0.0, |m, v| m.max(v.abs()))
}

/// the largest departure of `scale (x - y)` from `expected`, relative to the largest expected value
fn departure(x: &[Vec<f64>], y: &[Vec<f64>], scale: f64, expected: &[Vec<f64>]) -> f64 {
    let mut worst = 0.0f64;
    for ((xa, ya), ea) in x.iter().zip(y).zip(expected) {
        for ((xv, yv), ev) in xa.iter().zip(ya).zip(ea) {
            worst = worst.max((scale * (xv - yv) - ev).abs());
        }
    }
    worst / max_abs(expected)
}

/// both stages carry their own complete resistive EMF, with or without a correction in `poison`
fn assert_contract(poison: Option<usize>, label: &str) {
    let a = step(ETA, ETA, poison);
    let b = step(0.0, 0.0, poison);
    let c = step(ETA, 0.0, poison);
    if poison.is_some() {
        assert!(a.corrected && b.corrected && c.corrected, "{label}: no correction ran; the arm is vacuous");
    }
    let jn = eta_j(&a.bn);
    let jstar = eta_j(&a.bstar);
    let moved = departure(&jstar, &jn, 1.0, &jn);
    assert!(moved > 1e-2, "{label}: J(B*) and J(B^n) differ by {moved:.2e}; an old-state current would pass");
    let predictor = departure(&a.predictor_emf, &b.predictor_emf, 1.0, &jn);
    assert!(predictor < 1e-12, "{label}: the predictor's resistive EMF departs from eta J(B^n) by {predictor:.2e}");
    let corrector = departure(&a.final_emf, &c.final_emf, 2.0, &jstar);
    assert!(corrector < 1e-12, "{label}: the corrector's resistive EMF departs from eta J(B*) by {corrector:.2e}");
}

#[test]
fn each_stage_carries_the_resistive_emf_of_its_own_stage_input() {
    assert_contract(None, "no correction");
}

#[test]
fn a_correction_in_the_corrector_carries_the_resistive_emf_once() {
    assert_contract(Some(1), "correction in the corrector");
}

#[test]
fn a_correction_in_the_predictor_carries_the_resistive_emf_once() {
    assert_contract(Some(0), "correction in the predictor");
}

#[test]
fn edges_away_from_the_correction_keep_the_high_order_emf() {
    let clean = step(ETA, ETA, None);
    let corrected = step(ETA, ETA, Some(1));
    assert!(corrected.corrected, "no correction ran; the gate is vacuous");
    let s = sim();
    let mhd = s.fields.mhd.as_ref().unwrap();
    let lo: [isize; 3] = std::array::from_fn(|a| s.geom.interior.spaces[a].lo);
    let (mut near, mut far) = (0usize, 0usize);
    for (slot, field) in mhd.efield.e.iter().enumerate() {
        for (ii, c) in field.domain().iter().enumerate() {
            if clean.final_emf[slot][ii].to_bits() == corrected.final_emf[slot][ii].to_bits() {
                continue;
            }
            let within = (0..3).all(|a| (c[a] - lo[a] - POISONED[a] as isize).abs() <= 1);
            if within { near += 1 } else { far += 1 }
        }
    }
    assert!(near > 0, "no edge of the flagged cell took the first-order EMF");
    assert_eq!(far, 0, "{far} edges away from the flagged cell left the high-order EMF");
}
