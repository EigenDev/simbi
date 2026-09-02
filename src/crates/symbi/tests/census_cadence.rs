// =============================================================================
// census_cadence.rs
//
// a census samples on a declared simulation-time interval, at a cadence coarser than the step.
//
// this is a cost control, and a cost control that silently does nothing looks exactly like one
// that works: the numbers are identical, only the row count and the wall clock differ. a sample is
// a full extra sweep of the grid plus its reduction — measured at roughly a third of a hydro step
// on a small 1d problem — so a cadence that quietly degraded to every-step would be paying that
// on every step forever while reporting the same physics.
//
// the interval is in time rather than steps because dt varies over a run, so a fixed step stride
// samples the physics non-uniformly and the spacing of the series becomes an artifact of the
// timestepper.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_amr::refinement::Hierarchy;
use symbi_geometry::Spherical;
use symbi_source_compile::CensusConfig;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_sim::census::{CensusEvaluator, RegisteredCensus};
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 1, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 64;
const GAMMA: f64 = 5.0 / 3.0;
const T_END: f64 = 0.25; // long enough that the step count far exceeds the sample count

fn census(interval: Option<f64>) -> CensusConfig {
    let cadence = match interval {
        Some(dt) => format!(r#""sample_interval": {dt},"#),
        None => String::new(),
    };
    CensusConfig::from_json(&format!(
        r#"{{
            "name": "mass", "op": "add", "axes": [], {cadence}
            "value_names": ["m"], "values": [2], "params": [],
            "nodes": [
                {{"op": "VARIABLE_RHO"}},
                {{"op": "VARIABLE_DV"}},
                {{"op": "MULTIPLY", "left": 0, "right": 1}}
            ]
        }}"#
    ))
    .expect("census parses")
}

/// (samples recorded, steps taken, the sample times)
fn run(interval: Option<f64>) -> (usize, u64, Vec<f64>) {
    let mut sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([1.0])
        .spacing([0.01])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.4)
        .allocate()
        .expect("sim")
        .set_initial(|x| Prim {
            rho: 1.0 + 0.2 * x[0].sin(),
            vel: Tensor::new([0.0]),
            pre: 0.5,
        })
        .build();
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(&census(interval)).expect("census compiles"),
    ));
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, sub);
    hier.evolve(T_END).expect("evolve");
    let s = &hier.levels[0].state;
    let h = &s.store.censuses[0].history;
    (h.len(), s.iteration, h.time().to_vec())
}

#[test]
fn a_declared_interval_samples_far_less_often_than_every_step() {
    let (every_step, steps, _) = run(None);
    let interval = T_END / 5.0;
    let (paced, paced_steps, times) = run(Some(interval));

    // the premise: the run must take many more steps than the cadence allows samples, or the two
    // configurations would be indistinguishable and this proves nothing.
    assert!(
        steps > 20,
        "the run took only {steps} step(s); too few for a cadence to be visible"
    );
    assert_eq!(
        steps, paced_steps,
        "the cadence must not change the evolution"
    );
    assert_eq!(
        every_step as u64, steps,
        "with no interval a census samples every step ({every_step} samples, {steps} steps)"
    );

    // ~5 intervals over the run, plus the initial sample which is always due.
    assert!(
        paced >= 5 && paced <= 8,
        "expected about 6 samples at an interval of {interval} over {T_END}, got {paced}"
    );
    assert!(
        (paced as f64) < 0.5 * every_step as f64,
        "the cadence did not reduce the sample count ({paced} against {every_step}); a cost \
         control that samples every step is indistinguishable from none at all"
    );

    // and the surviving samples sit at least the declared interval apart, which is a stronger
    // statement than merely being fewer.
    for w in times.windows(2) {
        assert!(
            w[1] - w[0] >= interval,
            "samples at {} and {} are closer than the declared interval {interval}",
            w[0],
            w[1]
        );
    }
    // the first sample is the initial state: a census that waited one interval would omit the one
    // row a reader can check against the problem's own setup.
    assert!(
        times[0] <= interval,
        "the initial state was not sampled (first at {})",
        times[0]
    );
}

#[test]
fn a_non_positive_interval_is_refused() {
    // zero already has a spelling — omitting the field — so accepting it here would give one
    // behavior two names, and a negative interval is a computed value nobody chose.
    for bad in [0.0, -1.0] {
        let err = CensusEvaluator::new(&census(Some(bad)))
            .expect_err("a non-positive interval must be refused");
        assert!(err.contains("not positive"), "unhelpful error: {err}");
    }
}
