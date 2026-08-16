// =============================================================================
// census_cost.rs
//
// what a census actually costs, as a fraction of the step it rides on.
//
// the cost is reported as a number, and the assertion is placed elsewhere: a wall-clock
// threshold is either so loose it stays silent or so tight it fails on another machine, and
// neither figure is something a run can be planned around.
//
// the asserted property is where the cost lives — it scales with the size of the per-cell graph
// and stays flat in the number of registered accumulators. that is what lets a registration be
// reasoned about before a job is submitted, and it holds independently of any wall-clock number.
//
// two failure modes this measurement discriminates: a compiled map that re-runs the jit on every
// sample (4.3 ms per sample over 4096 cells, exceeding the step it observes), and a kernel cache
// keyed on the census name — unique within a run, and naturally reused across a parameter sweep in
// one process, so the second run is handed the first one's kernel.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_hydro::CensusConfig;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_sim::census::{CensusEvaluator, RegisteredCensus};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 1, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;
use symbi_geometry::Spherical;

const N: usize = 4096;
const GAMMA: f64 = 5.0 / 3.0;

/// `n_values` accumulators over a single shared graph: each is the same `rho * dv` product
/// re-emitted, so the graph holds a constant size while the output count grows. that separation
/// is what the cost claim rests on.
fn census_with(n_values: usize) -> CensusConfig {
    let names: Vec<String> = (0..n_values).map(|k| format!("\"m{k}\"")).collect();
    let outs: Vec<String> = (0..n_values).map(|_| "2".to_string()).collect();
    CensusConfig::from_json(&format!(
        r#"{{
            "name": "cost", "op": "add", "axes": [],
            "value_names": [{}], "values": [{}], "params": [],
            "nodes": [
                {{"op": "VARIABLE_RHO"}},
                {{"op": "VARIABLE_DV"}},
                {{"op": "MULTIPLY", "left": 0, "right": 1}}
            ]
        }}"#,
        names.join(","),
        outs.join(",")
    ))
    .expect("census parses")
}

fn build() -> Sim {
    Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([1.0])
        .spacing([0.001])
        .boundaries(Boundaries::uniform(BoundaryType::Outflow))
        .cfl(0.4)
        .allocate()
        .expect("sim")
        .set_initial(|x| Prim {
            rho: 1.0 + 0.2 * x[0].sin(),
            vel: Tensor::new([0.0]),
            pre: 0.5,
        })
        .build()
}

fn sample_ms(n_values: usize, reps: usize) -> f64 {
    let sim = build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.c2p(&sim.store);
    let mut sim = sim;
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(&census_with(n_values)).expect("census compiles"),
    ));
    // warm the compile cache and the pages before timing.
    symbi_substrate::census_sample::sample_censuses(&mut sim);
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        symbi_substrate::census_sample::sample_censuses(&mut sim);
    }
    t0.elapsed().as_secs_f64() * 1e3 / reps as f64
}

/// milliseconds per accepted hydro step on the same grid — the thing a census cost is a fraction
/// of. measured through the production driver so it carries the whole step: stages, flux, godunov,
/// recovery, boundaries.
fn hydro_step_ms(window: f64) -> f64 {
    use symbi_amr::refinement::Hierarchy;
    let sim = build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    let mut hier = Hierarchy::single(sim, sub);
    // a warmup step pages the fields in and warms the dispatch caches, then the timed window is a
    // bounded slice of simulation time — the step count that falls out is whatever the cfl gives,
    // and dividing by it makes the result per-step regardless.
    hier.evolve(1.0e-9).expect("warmup step");
    let t_start = hier.levels[0].state.time;
    let n0 = hier.levels[0].state.iteration;
    let t0 = std::time::Instant::now();
    hier.evolve(t_start + window).expect("timed window");
    let taken = hier.levels[0].state.iteration - n0;
    assert!(taken > 0, "the hydro reference took no step");
    t0.elapsed().as_secs_f64() * 1e3 / taken as f64
}

#[test]
fn a_census_sample_is_a_fraction_of_the_step_it_rides_on() {
    // the census sample and the hydro step measured on one grid, in one profile, through the
    // production driver — so the two sides are commensurable.
    let census = sample_ms(1, 20);
    let step = hydro_step_ms(2.0e-3);
    let pct = 100.0 * census / step;
    println!(
        "[{} profile, {N} cells] hydro step {step:.4} ms, census sample {census:.4} ms => census \
         is {pct:.1}% of a step",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
    assert!(
        step > 0.0 && census > 0.0,
        "one of the two did not register on the clock"
    );
}

#[test]
fn the_census_cost_tracks_the_graph_not_the_accumulator_count() {
    let reps = 20;
    let one = sample_ms(1, reps);
    let eight = sample_ms(8, reps);

    // printed for the record, carrying its build profile: these are debug-profile numbers, and
    // the optimized sweep is a different measurement entirely. a "few percent of a hydro step"
    // claim holds only when it is made against a release build.
    println!(
        "census over {N} cells [{} profile]: 1 accumulator {one:.4} ms, 8 accumulators {eight:.4} ms",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );

    // the premise: both must be resolvable above timer noise, or the ratio below is meaningless.
    assert!(
        one > 0.0 && eight > 0.0,
        "census sampling did not register on the clock ({one} ms, {eight} ms); the grid is too \
         small for this to measure anything"
    );

    // the claim: eight accumulators over one shared graph cost far less than eight times one.
    // the per-cell graph is evaluated once either way; what grows is the reduction's value count.
    // a census that re-walked the dag per accumulator would land near 8x.
    let ratio = eight / one;
    assert!(
        ratio < 5.0,
        "eight accumulators cost {ratio:.2}x one ({one:.4} ms -> {eight:.4} ms). the design's cost \
         model says the dag is evaluated once per cell regardless of how many outputs it feeds; \
         a ratio near the output count means it is being re-walked per accumulator."
    );
}

#[test]
fn the_scratch_is_allocated_once_and_reused_across_samples() {
    // the artifacts of a census have a fixed shape for the life of a run — one full-grid field per
    // accumulator plus the bucket — so reallocating them per sample moves that memory for no
    // information. at three million cells with sixteen accumulators that is order 384 mb churned
    // per sample, which is the reason this is a gate and not a preference.
    //
    // identity rather than wall-clock: an allocator that happened to hand back the same pages would
    // make a timing threshold pass while the buffers were genuinely being rebuilt.
    let mut sim = build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.c2p(&sim.store);
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(&census_with(3)).expect("census compiles"),
    ));

    let addr = |sim: &Sim| {
        let f = sim
            .census_scratch_pooled(&sim.store.censuses, 0)
            .expect("host-resident sim has pooled scratch");
        (
            f.values
                .iter()
                .map(|v| v.view().at([0]) as *const f64 as usize)
                .collect::<Vec<_>>(),
            f.segment.view().at([0]) as *const f64 as usize,
        )
    };

    symbi_substrate::census_sample::sample_censuses(&mut sim);
    let first = addr(&sim);
    symbi_substrate::census_sample::sample_censuses(&mut sim);
    let second = addr(&sim);
    assert_eq!(
        first, second,
        "the census scratch was reallocated between samples; its shape is fixed for the run, so \
         the buffers must persist"
    );
    assert_eq!(first.0.len(), 3, "one value buffer per accumulator");

    // the premise: a genuinely fresh allocation must be distinguishable from the pooled one, or
    // the equality above holds for reasons that have nothing to do with reuse.
    let fresh = sim
        .census_scratch(&sim.store.censuses[0].evaluator)
        .expect("host-resident sim allocates scratch");
    assert_ne!(
        fresh.segment.view().at([0]) as *const f64 as usize,
        first.1,
        "a freshly allocated scratch landed on the pooled buffer's address, so comparing addresses \
         cannot tell reuse from reallocation here"
    );
}

#[test]
fn a_reused_scratch_carries_nothing_from_the_previous_sample() {
    // the hazard reuse introduces. a fill that wrote only some cells would leave the rest holding
    // the previous sample's values and buckets, and the reduction would fold them again — a total
    // that is stale rather than wrong-shaped, so every downstream check still passes.
    let mut sim = build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.c2p(&sim.store);
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(&census_with(1)).expect("census compiles"),
    ));

    symbi_substrate::census_sample::sample_censuses(&mut sim);
    let before = sim.store.censuses[0].history.values()[0];

    // triple the density everywhere, so a sample that reflects the new state is exactly 3x and one
    // that reflects the stale scratch is unchanged. the two outcomes are 3x apart.
    for c in sim.geom.allocated.iter() {
        let rho = *sim.fields.prim.rho.view().at(c);
        sim.fields.prim.rho.view_mut().set(c, 3.0 * rho);
    }
    sim.time += 1.0;
    symbi_substrate::census_sample::sample_censuses(&mut sim);
    let after = *sim.store.censuses[0]
        .history
        .values()
        .last()
        .expect("second sample");

    assert!(
        (after / before - 3.0).abs() < 1.0e-12,
        "the second sample is {after} against a first of {before} (ratio {:.6}); tripling the \
         density must triple the mass. a ratio of 1 means the reused scratch was never refilled",
        after / before
    );
}

#[test]
fn a_cell_that_becomes_covered_is_excluded_on_the_reused_scratch() {
    // the specific way a skip-based fill breaks under reuse. a covered cell belongs to the finer
    // level, so it must carry the exclusion marker — but if the fill skips it rather than marking
    // it, the scratch still holds the bucket that cell was assigned when it was a leaf, and the
    // reduction counts that volume on both levels. the total is wrong by exactly the refined
    // volume and otherwise entirely plausible.
    let mut sim = build();
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.c2p(&sim.store);
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(&census_with(1)).expect("census compiles"),
    ));
    let sp = sim.geom.interior.spaces[0].clone();
    let mid = sp.lo + (sp.hi - sp.lo) / 2;
    let span = |lo, hi| {
        symbi_algebra::Domain::new([symbi_algebra::Space {
            name: sp.name,
            lo,
            hi,
        }])
    };

    // uncovered first, which is what leaves a live bucket in every cell of the scratch.
    let full =
        symbi_substrate::census_sample::level_partials(&sim, &sim.store.censuses, 0, None, None);
    let half = span(mid, sp.hi);
    let masked = symbi_substrate::census_sample::level_partials(
        &sim,
        &sim.store.censuses,
        0,
        None,
        Some(&half),
    );

    let (total, part) = (full[0].0[0], masked[0].0[0]);
    assert!(
        part < total * 0.999,
        "covering the outer half changed the total from {total} to {part}; on a reused scratch \
         those cells still hold the bucket they were assigned as leaves"
    );

    // and the complement closes it: the two halves must sum back to the whole, so the masked
    // sample dropped exactly the covered volume rather than an arbitrary amount.
    let outer = span(sp.lo, mid);
    let other = symbi_substrate::census_sample::level_partials(
        &sim,
        &sim.store.censuses,
        0,
        None,
        Some(&outer),
    );
    let sum = part + other[0].0[0];
    assert!(
        (sum / total - 1.0).abs() < 1.0e-12,
        "the two complementary coverings sum to {sum}, not the uncovered total {total}"
    );
}
