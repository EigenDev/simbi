// =============================================================================
// mhd_fallback_admissibility.rs
//
// the first-order correction of a magnetized stage accepts only an admissible
// state. its freeze tier holds a cell's stage-input gas (density, momentum,
// energy) while constrained transport has already advanced the faces the cell
// shares with its neighbors, so the frozen cell can carry negative internal
// energy. such an attempt is rejected whole and the step replays from its
// entry state, gas and staggered field together, at a smaller timestep.
//
// the fixture: a 16 x 16 periodic box, uniform density, pressure 1e-3 under
// B = (0.7, 1) (beta ~ 1.3e-3), a converging flow of peak speed 0.5. at twice
// the reference timestep the first stage freezes cells into negative internal
// energy; a few halvings later the step runs clean.
//   - the stage gate pins the defect state (the frozen cells hold their
//     stage-input gas beside re-curled faces, internal energy negative) and
//     requires the correction to request a retry.
//   - the step gate drives the single-grid step transaction from the large
//     timestep: the step is rejected at least once, every cell of the accepted
//     state is admissible, and the accepted state equals bit for bit a clean
//     run taken directly at the accepted timestep.
//
// run: cargo test -p symbi --test mhd_fallback_admissibility
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::sim::evolve::step_once;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::mhd_state::MhdPrim;
use symbi_hydro::newtonian_mhd::NewtonianMhd;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::stage::{StageArgs, StageOutcome, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 16;
const GAMMA: f64 = 5.0 / 3.0;
const PRESSURE: f64 = 1.0e-3;
const BX: f64 = 0.7;
const PEAK_SPEED: f64 = 0.5;
// 0.3 h over the peak speed plus the fast speed sqrt(1 + BX^2) of the uniform field
const DT_REFERENCE: f64 = 0.3 / (N as f64 * (PEAK_SPEED + 1.220_655_561_573_37));
const DT_LARGE: f64 = 2.0 * DT_REFERENCE;

type Sim = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build() -> Sim {
    let h = 1.0 / N as f64;
    let sim = Sim::build(NewtonianMhd, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries([[BoundaryType::Periodic; 2], [BoundaryType::Periodic; 2]]))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            let (dx, dy) = (x - 0.5, y - 0.5);
            let r = (dx * dx + dy * dy).sqrt().max(1e-12);
            let speed = PEAK_SPEED * (r / 0.15).min(1.0) * (-(r * r) / 0.04).exp();
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([-speed * dx / r, -speed * dy / r, 0.0]), Pressure(PRESSURE)),
                Tensor::new([BX, 1.0, 0.0]),
            )
        })
        // uniform faces: divergence free
        .seed_faces(|axis, _| if axis == 1 { 1.0 } else { BX })
        .build();
    let kernels = sim.substrate();
    kernels.c2p(&sim);
    kernels.ghost_fill(&sim);
    sim
}

/// E - |m|^2 / 2 rho - |B_cell|^2 / 2 at `c`
fn internal_energy(store: &FieldStore<2, 3>, c: [isize; 2]) -> f64 {
    let cons = &store.fields.cons;
    let mhd = store.fields.mhd.as_ref().expect("mhd fields");
    let rho = *cons.den.view().at(c);
    let kinetic: f64 = (0..3).map(|k| cons.mom[k].view().at(c).powi(2)).sum::<f64>() / (2.0 * rho);
    let magnetic: f64 = (0..3).map(|k| mhd.bcell.b[k].view().at(c).powi(2)).sum::<f64>() / 2.0;
    *cons.nrg_field().expect("energy").view().at(c) - kinetic - magnetic
}

fn bits(field: &symbi_grid::Field<f64, 2, HostMemory>, domain: &symbi_algebra::Domain<2>) -> Vec<u64> {
    let view = field.view();
    domain.iter().map(|c| view.at(c).to_bits()).collect()
}

/// every conserved component, cell field and face field of the interior, bit patterns
fn state_bits(store: &FieldStore<2, 3>) -> Vec<Vec<u64>> {
    let interior = &store.geom.interior;
    let cons = &store.fields.cons;
    let mhd = store.fields.mhd.as_ref().expect("mhd fields");
    let mut all = vec![bits(&cons.den, interior), bits(cons.nrg_field().expect("energy"), interior)];
    all.extend((0..3).map(|k| bits(&cons.mom[k], interior)));
    all.extend((0..3).map(|k| bits(&mhd.bcell.b[k], interior)));
    all.extend((0..2).map(|d| bits(&mhd.bface.b[d], mhd.bface.b[d].domain())));
    all
}

#[test]
fn a_frozen_cell_left_outside_the_admissible_set_requests_a_retry() {
    let mut sim = build();
    let kernels = sim.substrate();
    sim.dt = DT_LARGE;
    let store = &*sim;
    kernels.snapshot(store);
    let mhd = store.fields.mhd.as_ref().expect("mhd fields");
    let entry_faces: Vec<Vec<u64>> = (0..2).map(|d| bits(&mhd.bface.b[d], mhd.bface.b[d].domain())).collect();
    let stages = Timestepping::Rk2.stages();
    let outcome = fold_stage(
        store,
        &kernels,
        StageArgs {
            dt: DT_LARGE,
            a0: stages[0].0,
            ac: stages[0].1,
            stage: 0,
            n_stages: 2,
            injection_weight: 1.0,
            allow_elision: true,
            exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, 0),
        },
        &mut |_| {},
    );

    let frozen: Vec<[isize; 2]> =
        store.geom.interior.iter().filter(|&c| *store.workspace.freeze_applied.view().at(c) > 0.0).collect();
    assert!(!frozen.is_empty(), "the setup froze no cell; the gate is vacuous");
    // the freeze holds the stage-input gas: at the first stage that is the step entry u_n
    let u_n = &store.workspace.u_n;
    for &c in &frozen {
        assert_eq!(store.fields.cons.den.view().at(c), u_n.den.view().at(c), "frozen density at {c:?}");
        assert_eq!(
            store.fields.cons.nrg_field().unwrap().view().at(c),
            u_n.nrg_field().unwrap().view().at(c),
            "frozen energy at {c:?}"
        );
    }
    let recurled = (0..2).any(|d| bits(&mhd.bface.b[d], mhd.bface.b[d].domain()) != entry_faces[d]);
    assert!(recurled, "the stage left every face at its entry value; no field moved under the frozen gas");
    let (worst, at) = frozen
        .iter()
        .map(|&c| (internal_energy(store, c), c))
        .fold((f64::INFINITY, [0, 0]), |a, b| if b.0 < a.0 { b } else { a });
    assert!(
        worst < 0.0,
        "every frozen cell kept positive internal energy (min {worst:.3e} at {at:?}); the setup no longer \
         produces the inadmissible freeze"
    );
    assert_eq!(
        outcome,
        StageOutcome::RetryStep,
        "the correction accepted a frozen cell with internal energy {worst:.3e} at {at:?}"
    );
}

#[test]
fn a_rejected_step_replays_from_its_entry_and_matches_a_clean_run_at_the_accepted_timestep() {
    let mut retried = build();
    let kernels = retried.substrate();
    step_once(&mut retried, &kernels, DT_LARGE);
    let accepted = retried.dt;
    assert!(accepted < DT_LARGE, "the step was accepted at the large timestep; no retry was exercised");
    for c in retried.geom.interior.iter() {
        let eint = internal_energy(&retried, c);
        assert!(eint > 0.0, "the accepted state carries internal energy {eint:.3e} at {c:?}");
    }

    let mut clean = build();
    let clean_kernels = clean.substrate();
    step_once(&mut clean, &clean_kernels, accepted);
    assert_eq!(clean.dt, accepted, "the clean run was itself rejected at the accepted timestep");
    assert!(
        state_bits(&retried) == state_bits(&clean),
        "the replay differs from a clean step at dt = {accepted:.6e}: the rejected attempt leaked into the replay"
    );
}
