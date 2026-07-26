// =============================================================================
// fofc_stage_input.rs
//
// the FOFC redo must restore from THE stage input (`FieldStore::stage_input()`),
// not from a direct `workspace.u_stage` read: at the first stage of a
// multi-stage scheme the hierarchy driver elides the cons -> u_stage copy
// (u_n IS the stage input), so a direct read hands the redo a stale — on the
// first step, zeroed — buffer. the redo then rebuilds the flow from garbage:
// an unrecoverable poison COMPLETES as a "success" with a zeroed conserved
// buffer (density 0 is finite, so no guard fires), and a recoverable blast
// freezes where it used to recover.
//
// both gates drive the PRODUCTION hierarchy loop (the same single-level wrap
// every python run uses — the raw evolve() pipeline has no fofc phase):
//   1. a finite, unrecoverable energy sink must trip the persistent-freeze
//      halt (a panic naming the freeze) — never march to t_final.
//   2. a recoverable strong blast fires FOFC and conserves the totals with
//      ZERO freezes — the first-order tier recovers every flagged cell.
// =============================================================================

use symbi::prelude::SimSubstrate;
use symbi::regimes::fofc::{fofc_reset_stats, fofc_stats};
use symbi::regimes::substrate_rhd::RhdSubstrateKernelSet;
use symbi::sim::refinement::Hierarchy;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Cartesian;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::expr_bridge::build_user_source;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_hydro::{NEWTONIAN_SPEC, SourceConfig};
use symbi_xpu::{CpuSpace, HostMemory};

const GAMMA: f64 = 1.4;
const N: usize = 100;

// a spatially uniform, enormous energy SINK: finite everywhere (the finiteness
// guards never fire) but unrecoverable — the redo restores the stage input and
// the sink re-poisons it, forcing the freeze tier every substage.
const SINK_JSON: &str = r#"{
    "kind": "raw", "dim": 1, "outputs": [0], "params": [], "target": "nrg",
    "nodes": [ {"op": "CONSTANT", "value": -1.0e6} ]
}"#;

// the fofc counters are process-global atomics: the two gates must not run
// concurrently or one test's events contaminate the other's assertions.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

type Sim = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

fn sod(left_pressure: f64) -> Sim {
    SimState::build(Newtonian, IdealGas { gamma: GAMMA }, Cartesian)
        .cells([N])
        .origin([0.0])
        .spacing([1.0 / N as f64])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .allocate()
        .expect("sim")
        .set_initial(move |[x]| {
            if x < 0.5 {
                Prim {
                    rho: 1.0,
                    vel: Tensor::new([0.0]),
                    pre: left_pressure,
                }
            } else {
                Prim {
                    rho: 0.125,
                    vel: Tensor::new([0.0]),
                    pre: 0.1,
                }
            }
        })
        .build()
}

#[test]
fn unrecoverable_sink_trips_the_persistent_freeze_halt() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let result = std::panic::catch_unwind(|| {
        let sim = sod(1.0);
        let cfg = SourceConfig::from_json(SINK_JSON).expect("parse sink");
        let built = build_user_source(&cfg, &NEWTONIAN_SPEC).expect("lower sink");
        let kset = sim
            .substrate()
            .with_runtime_source(built, cfg.params.clone());
        let mut hier = Hierarchy::single(sim, kset);
        hier.evolve(0.2).expect("hierarchy evolve returned");
        let rho0 = *hier.levels[0]
            .state
            .fields
            .prim
            .rho
            .view()
            .at([N as isize / 2]);
        panic!("no-halt: the poisoned run completed (mid rho = {rho0})");
    });

    let err = result.expect_err("the poisoned run must halt");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.to_lowercase().contains("freeze"),
        "expected the persistent-freeze fail-loud, got: {msg}"
    );
}

#[test]
fn colliding_streams_conserve_with_zero_freezes() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // periodic relativistic colliding streams (v = +0.99 / -0.96, p = 1e-3):
    // the collision shocks drive the high-order c2p unphysical so FOFC fires on
    // hundreds of substages, and the stream strength is tuned so the first-order
    // tier RECOVERS every flagged cell — zero freezes, and periodicity makes the
    // conserved totals exact invariants of any face-telescoping update. a redo
    // restoring from a stale stage input fails both ways at once.
    use symbi_hydro::rhd::Rhd;

    fofc_reset_stats();
    let nx = 400usize;
    let sim = SimState::<Rhd, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>::build(
        Rhd,
        IdealGas { gamma: 4.0 / 3.0 },
        Cartesian,
    )
    .cells([nx])
    .origin([0.0])
    .spacing([1.0 / nx as f64])
    .boundaries(Boundaries::uniform(BoundaryType::Periodic))
    .allocate()
    .expect("sim")
    .set_initial(|[x]| Prim {
        rho: 1.0,
        vel: Tensor::new([if x <= 0.5 { 0.99 } else { -0.96 }]),
        pre: 1e-3,
    })
    .build();
    let interior = sim.geom.interior.clone();
    let totals = |cons: &symbi::sim::state::ConsFieldsGeneric<1, 1, HostMemory, f64>| {
        let (mut d, mut m, mut e) = (0.0f64, 0.0f64, 0.0f64);
        for c in interior.iter() {
            d += *cons.den.view().at(c);
            m += *cons.mom[0].view().at(c);
            e += *cons.nrg_field().unwrap().view().at(c);
        }
        (d, m, e)
    };
    let mut kset =
        RhdSubstrateKernelSet::<HostMemory, f64, 1>::new(4.0 / 3.0, 0.4, &sim.geom.allocated);
    // the production default reconstruction: minmod-MC at theta = 1.5 (the plain
    // minmod theta = 1 is diffusive enough that the collision never overshoots
    // and the gate would not exercise the redo at all).
    kset.theta = 1.5;
    let (d0, m0, e0) = totals(&sim.fields.cons);
    let mut hier = Hierarchy::single(sim, kset);
    hier.evolve(0.1).expect("hierarchy evolve");
    let s = &hier.levels[0].state;

    let (fired, froze) = fofc_stats();
    assert!(
        fired > 0,
        "FOFC never fired — the gate does not exercise the redo"
    );
    assert_eq!(
        froze, 0,
        "the recoverable collision froze {froze} cell-substages"
    );

    let (d1, m1, e1) = totals(&s.fields.cons);
    assert!(
        ((d1 - d0) / d0).abs() < 1e-11,
        "D drifted: {:.3e}",
        (d1 - d0) / d0
    );
    assert!(
        ((m1 - m0) / d0).abs() < 1e-11,
        "S drifted: {:.3e}",
        (m1 - m0) / d0
    );
    assert!(
        ((e1 - e0) / e0).abs() < 1e-11,
        "tau drifted: {:.3e}",
        (e1 - e0) / e0
    );
}
