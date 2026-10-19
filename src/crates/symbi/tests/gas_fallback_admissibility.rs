// =============================================================================
// gas_fallback_admissibility.rs
//
// a gas regime's first-order correction accepts a freeze only where it lands in
// the admissible set. a frozen gas cell holds its stage input, so a freeze over
// a valid stage input is admissible and accepted unchanged; no naturally
// occurring inadmissible stage input is known. the guard keeps an artificially
// supplied inadmissible input from being accepted silently. admissibility is
// judged here by the test's own predicate on the conserved state, density
// positive, internal energy E - |m|^2 / 2 rho positive, every value finite,
// independent of the recovery's report.
//
// the fixture: a 32 x 32 periodic shear layer of newtonian gas, two streams
// separating across x = 0.5 at up to unit speed, pressure 1e-4.
//   - a valid freeze: at CFL 1.5 the first-order redo leaves cells
//     unphysical and the select freezes them over their valid stage input;
//     the stage is accepted, every frozen cell holds the stage input bit for
//     bit, and every cell is admissible;
//   - an artificial inadmissible input: a cell's redo input poisoned to
//     negative energy at the reference timestep freezes outside the admissible
//     set; the stage requests a retry, and the step replays from its entry,
//     matching bit for bit a clean step at the accepted timestep;
//   - an inadmissible entry: a cell of the step entry itself carries negative
//     energy; no timestep repairs it, so the step ends after a bounded number
//     of replays at its restored entry with the freeze diagnostic.
//
// run: cargo test -p symbi --test gas_fallback_admissibility
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::sim::evolve::step_once;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::Newtonian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::quantity::{Density, Pressure};
use symbi_hydro::state::Prim;
use symbi_sim::stage::{StageArgs, StageOutcome, fold_stage};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

const N: usize = 32;
const PRESSURE: f64 = 1.0e-4;
const POISONED: [isize; 2] = [8, 20];

type Sim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn build() -> Sim {
    let h = 1.0 / N as f64;
    let sim = Sim::build(Newtonian, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([N, N])
        .spacing([h, h])
        .boundaries(Boundaries([[BoundaryType::Periodic; 2], [BoundaryType::Periodic; 2]]))
        .cfl(0.3)
        .timestepping(Timestepping::Rk2)
        .allocate()
        .expect("sim")
        .set_initial(|[x, y]| {
            let side = if x < 0.5 { -1.0 } else { 1.0 };
            Prim::adiabatic(Density(1.0), Tensor::new([side * (std::f64::consts::TAU * y).sin().abs(), 0.0]), Pressure(PRESSURE))
        })
        .build();
    let kernels = sim.substrate();
    kernels.c2p(&sim);
    kernels.ghost_fill(&sim);
    sim
}

/// the timestep at `courant` crossings of one cell by the peak signal speed
fn timestep(courant: f64) -> f64 {
    courant / (N as f64 * (1.0 + (5.0 / 3.0 * PRESSURE).sqrt()))
}

/// the test's own admissibility predicate on the conserved state at `c`
fn admissible(store: &FieldStore<2, 2>, c: [isize; 2]) -> bool {
    let cons = &store.fields.cons;
    let rho = *cons.den.view().at(c);
    let m2: f64 = (0..2).map(|k| cons.mom[k].view().at(c).powi(2)).sum();
    let e = *cons.nrg_field().unwrap().view().at(c);
    rho.is_finite() && m2.is_finite() && e.is_finite() && rho > 0.0 && e - 0.5 * m2 / rho > 0.0
}

fn first_stage(store: &FieldStore<2, 2>, kernels: &impl KernelSet<2, 2, HostMemory, f64>, dt: f64) -> StageOutcome {
    let stages = Timestepping::Rk2.stages();
    kernels.snapshot(store);
    fold_stage(
        store,
        kernels,
        StageArgs {
            dt,
            a0: stages[0].0,
            ac: stages[0].1,
            stage: 0,
            n_stages: 2,
            injection_weight: 1.0,
            allow_elision: true,
            exit_fraction: symbi_sim::driver::stage_exit_fraction(stages, 0),
        },
        &mut |_| {},
    )
}

fn frozen(store: &FieldStore<2, 2>) -> Vec<[isize; 2]> {
    store.geom.interior.iter().filter(|&c| *store.workspace.freeze_applied.view().at(c) > 0.0).collect()
}

fn bits(store: &FieldStore<2, 2>) -> Vec<u64> {
    let cons = &store.fields.cons;
    let fields = [&cons.den, &cons.mom[0], &cons.mom[1], cons.nrg_field().unwrap()];
    fields.iter().flat_map(|f| store.geom.interior.iter().map(|c| f.view().at(c).to_bits())).collect()
}

#[test]
fn a_freeze_over_a_valid_stage_input_is_accepted_unchanged() {
    let mut sim = build();
    let kernels = sim.substrate();
    sim.dt = timestep(1.5);
    let store = &*sim;
    let outcome = first_stage(store, &kernels, timestep(1.5));
    let cells = frozen(store);
    assert!(!cells.is_empty(), "the redo recovered every cell; no freeze to accept");
    assert_eq!(outcome, StageOutcome::Accepted, "a freeze over a valid stage input was rejected");
    let u_n = &store.workspace.u_n;
    for &c in &cells {
        assert_eq!(store.fields.cons.den.view().at(c), u_n.den.view().at(c), "frozen density at {c:?}");
        assert_eq!(store.fields.cons.nrg_field().unwrap().view().at(c), u_n.nrg_field().unwrap().view().at(c), "frozen energy at {c:?}");
    }
    let bad: Vec<_> = store.geom.interior.iter().filter(|&c| !admissible(store, c)).collect();
    assert!(bad.is_empty(), "the accepted stage carries inadmissible cells {bad:?}");
}

#[test]
fn an_artificially_inadmissible_stage_input_requests_a_retry() {
    let mut sim = build();
    let kernels = sim.substrate();
    *sim.workspace.unrecoverable_once.lock().unwrap() = vec![(0, 0, POISONED, false)];
    sim.dt = timestep(0.5);
    let store = &*sim;
    let outcome = first_stage(store, &kernels, timestep(0.5));
    assert!(frozen(store).contains(&POISONED), "the poisoned cell was not frozen; the gate is vacuous");
    assert!(!admissible(store, POISONED), "the frozen poisoned cell is admissible; the gate is vacuous");
    assert_eq!(outcome, StageOutcome::RetryStep, "the correction accepted a frozen inadmissible cell");
}

#[test]
fn a_step_rejected_for_an_inadmissible_input_matches_a_clean_step_at_the_accepted_timestep() {
    let mut retried = build();
    let kernels = retried.substrate();
    *retried.workspace.unrecoverable_once.lock().unwrap() = vec![(0, 0, POISONED, false)];
    step_once(&mut retried, &kernels, timestep(0.5));
    let accepted = retried.dt;
    assert!(accepted < timestep(0.5), "the step was accepted at its first timestep; no retry was exercised");
    assert!(retried.geom.interior.iter().all(|c| admissible(&retried, c)), "the accepted state carries an inadmissible cell");

    let mut clean = build();
    let clean_kernels = clean.substrate();
    step_once(&mut clean, &clean_kernels, accepted);
    assert_eq!(clean.dt, accepted, "the clean step was itself rejected at the accepted timestep");
    assert!(bits(&retried) == bits(&clean), "the replay differs from a clean step at dt = {accepted:.6e}");
}

#[test]
fn an_inadmissible_step_entry_ends_in_a_bounded_halt_at_the_restored_entry() {
    let mut sim = build();
    let kernels = sim.substrate();
    sim.fields.cons.nrg_field().unwrap().view_mut().set(POISONED, -1.0);
    kernels.c2p(&sim);
    kernels.ghost_fill(&sim);
    assert!(!admissible(&sim, POISONED), "the poisoned entry is admissible; the gate is vacuous");
    let entry = bits(&sim);
    let (time, iteration) = (sim.time, sim.iteration);
    let halted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| step_once(&mut sim, &kernels, timestep(0.5))));
    let message = match halted {
        Err(payload) => payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default(),
        Ok(()) => panic!("a step from an inadmissible entry was accepted"),
    };
    assert!(
        message.contains("FOFC last-resort freeze") || message.contains("retry guard") || message.contains("timestep"),
        "the step ended on another cause: {message}"
    );
    assert_eq!((sim.time, sim.iteration), (time, iteration), "the halt left the step entry's clock");
    assert!(bits(&sim) == entry, "the halt left a state other than the restored entry");
}
