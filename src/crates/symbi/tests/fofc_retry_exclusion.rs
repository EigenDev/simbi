// =============================================================================
// fofc_retry_exclusion.rs
//
// which freeze acts of the first-order correction reject a step is a property
// of the simulation, carried on its store as its retry exclusion; the
// process-global fofc horizon radius is a reporting split and takes no part.
// the fixture is a magnetized cartesian kerr-schild box (mass 1, no excision,
// UCT) whose stage input is made unrecoverable at one cell near the hole, so
// the redo cannot land it and the correcting select freezes it:
// - with no exclusion the freeze rejects the stage; with the legacy euclidean
//   sphere covering the cell the stage is accepted;
// - each outcome and the state it leaves are bit for bit the same with the
//   reporting radius unset and set over the cell, and whether the cases run
//   in one order or the other in one process;
// - the reporting split still follows the reporting radius;
// - the traced exclusion keeps exactly the freeze acts the legacy host test
//   keeps, cell for cell, so a device run decides as a host run does.
//
// run: cargo test -p symbi --test fofc_retry_exclusion
// =============================================================================

use symbi::regimes::fofc::{fofc_horizon_stats, fofc_reset_stats, fofc_set_horizon_radius, horizon_flagged_count};
use symbi::regimes::substrate_kernels::{Solver, dispatch_fofc_evidence_outside_sphere};
use symbi::regimes::substrate_rmhd::RmhdSubstrateKernelSet3D;
use symbi::sim::evolve::{KernelSet, evolve};
use symbi::sim::state::*;
use symbi::sim::substrate_seam::RetryExclusion;
use symbi_algebra::Tensor;
use symbi_geometry::{Metric, SchwarzschildKSCartesian};
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::rmhd::Rmhd;
use symbi_hydro::state::Prim;
use symbi_sim::stage::{HookPoint, StageArgs, StageOutcome, fold_stage_from_correction, fold_stage_through_mark};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimState<Rmhd, 3, SchwarzschildKSCartesian<f64>, IdealGas<f64>, CpuSpace, HostMemory>;
type Kern = RmhdSubstrateKernelSet3D<HostMemory, f64>;

const N: usize = 16;
const L: f64 = 5.0;
const DX: f64 = 2.0 * L / N as f64;
const MASS: f64 = 1.0;
const T_PRIME: f64 = 0.3;
/// the frozen cell: centroid (0.9375, 0.3125, 0.3125), r = 1.03, inside the horizon r_+ = 2
const CELL: [usize; 3] = [N / 2 + 1, N / 2, N / 2];
/// a sphere covering the cell, and one it lies outside of
const COVERING: f64 = 2.0;
const POISON: f64 = -1.0;

fn primed(exclusion: RetryExclusion) -> (Sim, Kern) {
    let m = SchwarzschildKSCartesian { mass: MASS };
    let bz = move |x: [f64; 3]| 0.05 / m.sqrt_det_gamma(Tensor::new(x));
    let mut sim = Sim::build(Rmhd, IdealGas { gamma: 4.0 / 3.0 }, m)
        .cells([N; 3])
        .spacing([DX; 3])
        .origin([-L; 3])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|x| {
            MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.1)), Tensor::new([0.0, 0.0, bz(x)]))
        })
        .seed_faces(|axis, x| if axis == 2 { bz(x) } else { 0.0 })
        .build();
    sim.retry_exclusion = exclusion;
    let k = Kern::new(4.0 / 3.0, 0.3, 1.5, &sim.geom.allocated).with_solver(Solver::Hlle).expect("solver").ct_method(CtMethod::Uct);
    evolve(&mut sim, &k, T_PRIME).expect("prime");
    (sim, k)
}

/// one rk2 predictor stage whose stage input is unrecoverable at the cell: the outcome, the
/// freeze act there, the conserved state it leaves, the rejection evidence the correction
/// recorded, and every frozen cell as that evidence names one
fn frozen_stage(exclusion: RetryExclusion) -> (StageOutcome, f64, Vec<u64>, Vec<String>, Vec<String>) {
    let (sim, k) = primed(exclusion);
    let cell: [isize; 3] = std::array::from_fn(|a| sim.geom.interior.spaces[a].lo + CELL[a] as isize);
    k.snapshot_retry(&sim);
    k.snapshot(&sim);
    let dt = k.cfl(&sim).min(0.1);
    let args = StageArgs {
        dt,
        a0: 0.0,
        ac: 1.0,
        stage: 0,
        n_stages: 2,
        injection_weight: symbi_sim::driver::downstream_injection_weight(Timestepping::Rk2.stages(), 0),
        allow_elision: false,
    };
    let mut hook = |hp: HookPoint| match hp {
        // after the stage snapshot: the redo restarts from this input and the select holds it
        HookPoint::AfterFlux => *sim.stage_input().nrg_field().unwrap().view_mut().at_mut(cell) = POISON,
        HookPoint::BeforeC2p => *sim.fields.cons.nrg_field().unwrap().view_mut().at_mut(cell) = POISON,
        _ => {}
    };
    let marked = fold_stage_through_mark(&sim, &k, args, &mut hook);
    assert!(marked >= 1, "the poisoned cell was not flagged; the gate is vacuous");
    let outcome = fold_stage_from_correction(&sim, &k, args, false, &mut |_| {});
    let frozen = *sim.workspace.freeze_applied.view().at(cell);
    let cons = &sim.fields.cons;
    let mut state = Vec::new();
    for f in [&cons.den, &cons.mom[0], &cons.mom[1], &cons.mom[2], cons.nrg_field().unwrap()] {
        state.extend(sim.geom.interior.iter().map(|c| f.view().at(c).to_bits()));
    }
    let evidence = std::mem::take(&mut *sim.workspace.rejection_evidence.lock().unwrap());
    let freeze = sim.workspace.freeze_applied.view();
    let frozen_cells = sim
        .geom
        .interior
        .iter()
        .filter(|&c| *freeze.at(c) > 0.0)
        .map(|c| format!("stage tag 1: frozen cell {c:?} at {:?}", sim.geom.centroid(c)))
        .collect();
    (outcome, frozen, state, evidence, frozen_cells)
}

#[test]
fn the_retry_exclusion_decides_and_the_reporting_radius_does_not() {
    let legacy = RetryExclusion::LegacyEuclideanSphere { radius: COVERING };
    let cases = [(RetryExclusion::None, 0.0), (RetryExclusion::None, 3.0), (legacy, 0.0), (legacy, 3.0)];
    type Case = (StageOutcome, f64, Vec<u64>, u64, (Vec<String>, Vec<String>));
    let run = |order: &[usize]| -> Vec<Case> {
        let mut out = vec![None; cases.len()];
        for &ii in order {
            let (exclusion, reporting) = cases[ii];
            fofc_set_horizon_radius(reporting);
            fofc_reset_stats();
            let (outcome, frozen, state, evidence, frozen_cells) = frozen_stage(exclusion);
            out[ii] = Some((outcome, frozen, state, fofc_horizon_stats().1, (evidence, frozen_cells)));
        }
        fofc_set_horizon_radius(0.0);
        out.into_iter().map(|o| o.unwrap()).collect()
    };
    let forward = run(&[0, 1, 2, 3]);
    let reverse = run(&[3, 2, 1, 0]);
    let cell_index = format!("{:?}", std::array::from_fn::<isize, 3, _>(|a| CELL[a] as isize));
    for (ii, ((exclusion, reporting), (outcome, frozen, _, horizon_frozen, (evidence, frozen_cells)))) in cases.iter().zip(&forward).enumerate() {
        assert_eq!(*frozen, 1.0, "case {ii}: the select did not freeze the poisoned cell; the gate is vacuous");
        let expected = if *exclusion == RetryExclusion::None { StageOutcome::RetryStep } else { StageOutcome::Accepted };
        assert!(*outcome == expected, "case {ii} ({exclusion:?}, reporting radius {reporting}): the stage was not {expected:?}");
        // the reporting split books the freeze inside the reporting radius and nowhere else
        assert_eq!(*horizon_frozen > 0, *reporting > 0.0, "case {ii}: the reporting split ignores the reporting radius");
        // a retry request names every eligible frozen cell with its stage tag, the poisoned one
        // among them (all of them with no exclusion); an accepted stage records nothing
        if expected == StageOutcome::RetryStep {
            assert!(frozen_cells.len() <= 8, "case {ii}: more frozen cells than the evidence names");
            assert_eq!(evidence, frozen_cells, "case {ii}: the evidence is not the eligible frozen cells");
            assert!(evidence.iter().any(|e| e.contains(&format!("frozen cell {cell_index}"))), "case {ii}: {evidence:?}");
        } else {
            assert!(evidence.is_empty(), "case {ii}: an accepted stage recorded rejection evidence: {evidence:?}");
        }
    }
    for (a, b) in [(0, 1), (2, 3)] {
        assert!(forward[a].0 == forward[b].0, "cases {a} and {b}: the reporting radius changed the decision");
        assert_eq!(forward[a].2, forward[b].2, "cases {a} and {b}: the reporting radius changed the state");
    }
    for ii in 0..cases.len() {
        assert!(forward[ii].0 == reverse[ii].0 && forward[ii].2 == reverse[ii].2, "case {ii} depends on the order the simulations ran in");
    }
}

#[test]
fn the_traced_exclusion_keeps_the_legacy_host_test_s_freeze_acts() {
    for origin in [[-L; 3], [-4.7, -5.3, -2.1]] {
        let sim = Sim::build(Rmhd, IdealGas { gamma: 4.0 / 3.0 }, SchwarzschildKSCartesian { mass: MASS })
            .cells([N; 3])
            .spacing([DX; 3])
            .origin(origin)
            .boundaries(Boundaries::uniform(BoundaryType::Outflow))
            .allocate()
            .expect("sim")
            .set_initial(|_| MhdPrim::new(Prim::adiabatic(Density(1.0), Tensor::new([0.0; 3]), Pressure(0.1)), Tensor::new([0.0; 3])))
            .seed_faces(|_, _| 0.0)
            .build();
        let freeze = &sim.workspace.freeze_applied;
        for c in sim.geom.interior.iter() {
            *freeze.view_mut().at_mut(c) = 1.0;
        }
        let total = sim.geom.interior.iter().count() as u64;
        for radius in [0.0, 1.0, 2.0, 2.2, 3.7, 9.0] {
            let evidence = &sim.workspace.fofc_evidence;
            dispatch_fofc_evidence_outside_sphere(&sim, radius, freeze, evidence);
            let inside = horizon_flagged_count(freeze, &sim.geom.interior, &sim.geom.x_lo, &sim.geom.dx, &sim.geom.index_offset, radius);
            let mut kept = 0u64;
            for c in sim.geom.interior.iter() {
                let x: [f64; 3] = std::array::from_fn(|a| sim.geom.x_lo[a] + (c[a] as f64 + 0.5) * sim.geom.dx[a]);
                let host_inside = x.iter().map(|v| v * v).sum::<f64>() < radius * radius;
                let traced = *evidence.view().at(c);
                assert_eq!(traced, if host_inside { 0.0 } else { 1.0 }, "origin {origin:?} radius {radius}: cell {c:?} classified differently");
                kept += traced as u64;
            }
            assert_eq!(kept + inside, total, "origin {origin:?} radius {radius}: the counts do not partition the interior");
        }
    }
}
