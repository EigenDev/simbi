// =============================================================================
// resistive_rk2_contract.rs
//
// the ohmic edge EMF in the rk2 constrained transport: each stage forms its
// complete EMF, ideal and resistive, from its own stage input, and the
// corrector's EMF is the average of the two complete stage EMFs:
// - predictor: E_1 + eta J(B^n), saved, then curled from B^n;
// - corrector: E_2 + eta J(B*), averaged with the saved predictor EMF, then
//   curled from the restored B^n;
// - a first-order correction adds eta J of the stage-input faces to its
//   first-order EMF once; edges away from the flagged cell keep the saved
//   high-order EMF bit for bit;
// - a correction in both stages of one step: the predictor saves the spliced
//   EMF it curled from, B* is B^n curled by that saved EMF, and the corrector
//   carries eta J of that corrected B*.
// the resistivity is either the uniform eta or, with the uniform one zero, the
// body-localized eta chi J of a resistive immersed sphere that covers the
// flagged cell; the contract holds for each.
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
use symbi_ib::{Body, BodyCollection, MagneticSpec, SurfaceSpec};
use symbi_sim::stage::{HookPoint, StageArgs, fold_stage};
use symbi_substrate::regimes::mhd_substrate::{
    apply_resistive_emf_on, body_resistive_emf_on, ct_curl,
};
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
/// the resistive sphere of the body-localized arms: centered on the poisoned cell
const BODY_CENTER: [f64; 3] = [6.5 / N as f64; 3];
const BODY_RADIUS: f64 = 0.3;

/// where the resistivity lives: the uniform eta of the kernel set, or a resistive immersed body
/// with the uniform eta zero
#[derive(Clone, Copy, Debug, PartialEq)]
enum Resistivity {
    Uniform,
    Body,
}

fn sim(mode: Resistivity) -> Sim {
    let tau = std::f64::consts::TAU;
    let dx = 1.0 / N as f64;
    let s = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N; 3])
        .spacing([dx; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y, z]| {
            MhdPrim::new(
                Prim::adiabatic(
                    Density(1.0),
                    Tensor::new([
                        V0 * (tau * z).sin(),
                        V0 * (tau * x).sin(),
                        V0 * (tau * y).sin(),
                    ]),
                    Pressure(1.0),
                ),
                Tensor::new([
                    B1 * (tau * y).sin(),
                    B1 * (tau * z).sin(),
                    B1 * (tau * x).sin(),
                ]),
            )
        })
        .seed_faces(|axis, [x, y, z]| {
            B1 * [(tau * y).sin(), (tau * z).sin(), (tau * x).sin()][axis]
        })
        .build();
    let s = match mode {
        Resistivity::Uniform => s,
        Resistivity::Body => s.with_bodies(
            BodyCollection::new().add(
                Body::rigid_sphere(
                    0,
                    Tensor::new(BODY_CENTER),
                    Tensor::zeros(),
                    1.0,
                    BODY_RADIUS,
                    1.0,
                    false,
                )
                .with_surface(SurfaceSpec::Porous {
                    porosity: 0.0,
                    k_eta_n: 0.0,
                    k_eta_t: 0.0,
                })
                .with_magnetic(MagneticSpec::Resistive { eta: ETA }),
            ),
        ),
    };
    s.with_rollback_snapshot().expect("rollback snapshot")
}

/// the kernel set and body resistivity of one stage at resistivity `eta`
fn configure(s: &mut Sim, mode: Resistivity, eta: f64) -> Kern {
    let uniform = match mode {
        Resistivity::Uniform => eta,
        Resistivity::Body => {
            let bodies = &mut s.immersed.as_mut().expect("resistive body").bodies;
            bodies.get_mut(0).spec.magnetic = MagneticSpec::Resistive { eta };
            0.0
        }
    };
    Kern::new(GAMMA, 0.3, 1.5, &s.geom.allocated).with_resistivity(uniform)
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
        exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, stage),
    }
}

fn edges(fields: &[Field<f64, 3, HostMemory>]) -> Vec<Vec<f64>> {
    fields
        .iter()
        .map(|f| f.domain().iter().map(|c| *f.view().at(c)).collect())
        .collect()
}

fn faces(s: &Sim) -> Vec<Vec<f64>> {
    let mhd = s.fields.mhd.as_ref().unwrap();
    mhd.bface
        .b
        .iter()
        .map(|f| f.domain().iter().map(|c| *f.view().at(c)).collect())
        .collect()
}

/// one rk2 step with the resistivity `eta_predictor` in the first stage and `eta_corrector` in
/// the second, the cell poisoned before the recovery of each stage in `poison`
struct Step {
    /// the saved predictor EMF the corrector averages with
    predictor_emf: Vec<Vec<f64>>,
    /// the EMF the predictor's faces were curled from, spliced where it corrected
    predictor_curled: Vec<Vec<f64>>,
    bn: Vec<Vec<f64>>,
    bstar: Vec<Vec<f64>>,
    final_emf: Vec<Vec<f64>>,
    corrected: [bool; 2],
}

/// fold one stage; whether a correction ran on the poisoned cell
fn fold(s: &Sim, k: &Kern, stage: usize, poison: bool) -> bool {
    let cell: [isize; 3] =
        std::array::from_fn(|a| s.geom.interior.spaces[a].lo + POISONED[a] as isize);
    let nrg = s.fields.cons.nrg_field().unwrap();
    let mut hook = |hp: HookPoint| {
        if poison && matches!(hp, HookPoint::BeforeC2p) {
            *nrg.view_mut().at_mut(cell) = -1.0;
        }
    };
    let _ = fold_stage(s, k, args(stage), &mut hook);
    // the store's own troubled flag at the poisoned cell: the correction's mark
    poison && *s.workspace.fofc_flag.view().at(cell) > 0.0
}

fn step(mode: Resistivity, eta_predictor: f64, eta_corrector: f64, poison: &[usize]) -> Step {
    let mut s = sim(mode);
    let kp = configure(&mut s, mode, eta_predictor);
    kp.c2p(&s);
    kp.ghost_fill(&s);
    let bn = faces(&s);
    kp.snapshot_retry(&s);
    kp.snapshot(&s);
    let first = fold(&s, &kp, 0, poison.contains(&0));
    let mhd = s.fields.mhd.as_ref().unwrap();
    let predictor_emf = edges(&mhd.efield_n.e);
    let predictor_curled = edges(&mhd.efield.e);
    let bstar = faces(&s);
    let kc = configure(&mut s, mode, eta_corrector);
    let second = fold(&s, &kc, 1, poison.contains(&1));
    let final_emf = edges(&s.fields.mhd.as_ref().unwrap().efield.e);
    Step {
        predictor_emf,
        predictor_curled,
        bn,
        bstar,
        final_emf,
        corrected: [first, second],
    }
}

/// load `face_values` into a fresh fixture's faces and zero its edge EMF
fn loaded(mode: Resistivity, face_values: &[Vec<f64>]) -> Sim {
    let s = sim(mode);
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
    s
}

/// the resistive EMF of `face_values` through the code's resistive operator at eta = ETA: the
/// uniform eta J, or the body's eta chi J
fn eta_j(mode: Resistivity, face_values: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let s = loaded(mode, face_values);
    let mhd = s.fields.mhd.as_ref().unwrap();
    let faces: Vec<&Field<f64, 3, HostMemory>> = mhd.bface.b.iter().collect();
    match mode {
        Resistivity::Uniform => apply_resistive_emf_on(&*s, ETA, &faces),
        Resistivity::Body => body_resistive_emf_on(&*s, &faces),
    }
    edges(&mhd.efield.e)
}

/// the largest |x - y| over matching entries
fn max_diff(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    x.iter()
        .flatten()
        .zip(y.iter().flatten())
        .fold(0.0, |m, (a, b)| m.max((a - b).abs()))
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

/// both stages carry their own complete resistive EMF, with or without corrections in `poison`
fn assert_contract(mode: Resistivity, poison: &[usize], label: &str) -> Step {
    let a = step(mode, ETA, ETA, poison);
    let b = step(mode, 0.0, 0.0, poison);
    let c = step(mode, ETA, 0.0, poison);
    for (twin, arm) in [("A", &a), ("B", &b), ("C", &c)] {
        for stage in 0..2 {
            assert_eq!(
                arm.corrected[stage],
                poison.contains(&stage),
                "{label}: twin {twin} stage {stage} corrected {} where the poison asked {}; the arm is vacuous",
                arm.corrected[stage],
                poison.contains(&stage)
            );
        }
    }
    let jn = eta_j(mode, &a.bn);
    let jstar = eta_j(mode, &a.bstar);
    assert!(
        max_abs(&jn) > 0.0,
        "{label}: the resistive EMF of B^n is zero; the arm is vacuous"
    );
    let moved = departure(&jstar, &jn, 1.0, &jn);
    assert!(
        moved > 1e-2,
        "{label}: J(B*) and J(B^n) differ by {moved:.2e}; an old-state current would pass"
    );
    let predictor = departure(&a.predictor_emf, &b.predictor_emf, 1.0, &jn);
    assert!(
        predictor < 1e-12,
        "{label}: the predictor's resistive EMF departs from eta J(B^n) by {predictor:.2e}"
    );
    let corrector = departure(&a.final_emf, &c.final_emf, 2.0, &jstar);
    assert!(
        corrector < 1e-12,
        "{label}: the corrector's resistive EMF departs from eta J(B*) by {corrector:.2e}"
    );
    a
}

#[test]
fn each_stage_carries_the_resistive_emf_of_its_own_stage_input() {
    assert_contract(Resistivity::Uniform, &[], "no correction");
}

#[test]
fn a_correction_in_the_corrector_carries_the_resistive_emf_once() {
    assert_contract(Resistivity::Uniform, &[1], "correction in the corrector");
}

#[test]
fn a_correction_in_the_predictor_carries_the_resistive_emf_once() {
    assert_contract(Resistivity::Uniform, &[0], "correction in the predictor");
}

#[test]
fn edges_away_from_the_correction_keep_the_high_order_emf() {
    let clean = step(Resistivity::Uniform, ETA, ETA, &[]);
    let corrected = step(Resistivity::Uniform, ETA, ETA, &[1]);
    assert!(
        corrected.corrected[1],
        "no correction ran; the gate is vacuous"
    );
    let s = sim(Resistivity::Uniform);
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
    assert!(
        near > 0,
        "no edge of the flagged cell took the first-order EMF"
    );
    assert_eq!(
        far, 0,
        "{far} edges away from the flagged cell left the high-order EMF"
    );
}

/// the corrector averages with the EMF the corrected predictor curled from, and its B* is B^n
/// curled by that saved EMF
fn assert_corrected_history(mode: Resistivity, label: &str) {
    let a = assert_contract(mode, &[0, 1], label);
    let unspliced = step(mode, ETA, ETA, &[1]);
    assert!(
        max_diff(&unspliced.predictor_curled, &a.predictor_curled) > 0.0,
        "{label}: the predictor's correction left the EMF it curled unchanged; the history gate is vacuous"
    );
    let saved = max_diff(&a.predictor_emf, &a.predictor_curled);
    assert_eq!(
        saved, 0.0,
        "{label}: the saved predictor EMF departs from the spliced one it curled by {saved:.2e}"
    );
    // B^n curled by the saved predictor EMF, on the interior faces
    let s = loaded(mode, &a.bn);
    let mhd = s.fields.mhd.as_ref().unwrap();
    for (e, values) in mhd.efield.e.iter().zip(&a.predictor_emf) {
        for (c, v) in e.domain().iter().zip(values) {
            *e.view_mut().at_mut(c) = *v;
        }
    }
    ct_curl::<3, 3, HostMemory, f64>(&*s, DT);
    let curled = faces(&s);
    let mut worst = 0.0f64;
    for (d, f) in mhd.bface.b.iter().enumerate() {
        for (ii, c) in f.domain().iter().enumerate() {
            if s.geom.interior.contains(c) {
                worst = worst.max((curled[d][ii] - a.bstar[d][ii]).abs());
            }
        }
    }
    assert!(
        worst <= 1e-15 * B1,
        "{label}: B* departs from B^n curled by the saved predictor EMF by {worst:.2e}"
    );
}

#[test]
fn corrections_in_both_stages_carry_the_corrected_predictor_history() {
    assert_corrected_history(Resistivity::Uniform, "corrections in both stages");
}

#[test]
fn a_body_localized_resistivity_follows_the_same_contract() {
    assert_contract(Resistivity::Body, &[], "body, no correction");
    assert_contract(Resistivity::Body, &[0], "body, correction in the predictor");
    assert_contract(Resistivity::Body, &[1], "body, correction in the corrector");
    assert_corrected_history(Resistivity::Body, "body, corrections in both stages");
}
