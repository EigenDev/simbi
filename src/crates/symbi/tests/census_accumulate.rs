// =============================================================================
// census_accumulate.rs
//
// an accumulating census: every sample folded into one stored row with the census's own reduce
// op, rather than a row apiece.
//
// the mode exists because storage, not compute, is what bounds a fine binning. a two-dimensional
// histogram runs to order a hundred kilobytes per sample, so a run that only ever wanted the
// segment's time average would write thousands of them to disk in order to average them back
// down.
//
// what is asserted here is that the fold is exact — the stored row equals the same samples
// combined by hand — and that it is the census's own operator rather than a second rule. an
// accumulating census that summed where the registration says max, or that quietly held only the
// last sample, would still produce one plausible row and nothing downstream could tell.
// =============================================================================

use symbi::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
use symbi::sim::state::*;
use symbi_algebra::Tensor;
use symbi_geometry::Spherical;
use symbi_source_compile::CensusConfig;
use symbi_hydro::eos::IdealGas;
use symbi_hydro::newtonian::Newtonian;
use symbi_hydro::state::Prim;
use symbi_sim::census::{CensusEvaluator, RegisteredCensus};
use symbi_sim::substrate_seam::KernelSet;
use symbi_xpu::{CpuSpace, HostMemory};

type Sim = SimStateGeneric<Newtonian, 1, 1, Spherical, IdealGas<f64>, CpuSpace, HostMemory>;

const N: usize = 64;
const GAMMA: f64 = 5.0 / 3.0;

/// a global (axis-free) mass census, accumulating or not, under the given reduce op.
fn mass_census(op: &str, accumulate: bool) -> CensusConfig {
    CensusConfig::from_json(&format!(
        r#"{{
            "name": "mass", "op": "{op}", "axes": [],
            "value_names": ["mass"], "values": [2], "params": [],
            "accumulate": {accumulate},
            "nodes": [
                {{"op": "VARIABLE_RHO"}},
                {{"op": "VARIABLE_DV"}},
                {{"op": "MULTIPLY", "left": 0, "right": 1}}
            ]
        }}"#
    ))
    .expect("census parses")
}

fn build(cfg: &CensusConfig) -> Sim {
    let sim = Sim::build(Newtonian, IdealGas { gamma: GAMMA }, Spherical)
        .cells([N])
        .origin([1.0])
        .spacing([0.05])
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
    let sub =
        AdiabaticSubstrateKernelSet::<HostMemory, f64, 1>::new(GAMMA, 0.4, &sim.geom.allocated);
    sub.c2p(&sim.store);
    let mut sim = sim;
    sim.store.censuses.push(RegisteredCensus::new(
        CensusEvaluator::new(cfg).expect("census compiles"),
    ));
    sim
}

/// take `n` samples, scaling the density by `scale(k)` before each so the samples differ and a
/// fold that dropped or repeated one is visible. returns the per-sample totals a non-accumulating
/// run would have stored.
fn samples_with(sim: &mut Sim, n: usize, scale: impl Fn(usize) -> f64) -> Vec<f64> {
    let mut want = Vec::with_capacity(n);
    let base: Vec<f64> = sim
        .geom
        .allocated
        .iter()
        .map(|c| *sim.fields.prim.rho.view().at(c))
        .collect();
    for k in 0..n {
        let s = scale(k);
        for (i, c) in sim.geom.allocated.iter().enumerate() {
            sim.fields.prim.rho.view_mut().set(c, s * base[i]);
        }
        sim.time = k as f64 * 0.5;
        symbi_substrate::census_sample::sample_censuses(sim);
        want.push(s);
    }
    want
}

#[test]
fn an_accumulating_census_stores_one_row_that_is_the_exact_sum_of_its_samples() {
    // the reference: the same five samples, stored per-sample.
    let mut plain = build(&mass_census("add", false));
    let scales = samples_with(&mut plain, 5, |k| 1.0 + 0.3 * k as f64);
    let per_sample = plain.store.censuses[0].history.values().to_vec();
    assert_eq!(per_sample.len(), 5, "the reference stores a row per sample");

    let mut acc = build(&mass_census("add", true));
    let acc_scales = samples_with(&mut acc, 5, |k| 1.0 + 0.3 * k as f64);
    assert_eq!(scales, acc_scales, "both runs drove the same states");

    let history = &acc.store.censuses[0].history;
    assert_eq!(
        history.len(),
        1,
        "an accumulating census stored {} rows; the whole point is that it stores one",
        history.len()
    );
    assert_eq!(history.n_samples(), [5], "five samples were folded in");

    // exactness, not closeness: the fold is a sum of the same numbers in the same order.
    let want: f64 = per_sample.iter().sum();
    let got = history.values()[0];
    assert_eq!(
        got, want,
        "the accumulated row is {got}, but summing the five per-sample rows gives {want}"
    );

    // the premise: the samples must actually differ, or a fold that kept only the last would be
    // indistinguishable from one that summed them.
    let last = *per_sample.last().expect("five rows");
    assert!(
        (got - last).abs() > 0.1 * last,
        "the samples are too alike to tell a sum from a last-write ({got} vs {last})"
    );
}

#[test]
fn an_accumulating_census_folds_with_its_own_reduce_op() {
    // the fold over time is the same commutative monoid as the fold over refinement levels. an
    // extremal census accumulates the extremum over space and time; summing it instead would
    // report a quantity with no physical meaning at all, and it would still be one plausible row.
    let mut sim = build(&mass_census("max", true));
    let scales = samples_with(&mut sim, 4, |k| [1.0, 2.5, 0.4, 1.7][k]);
    assert_eq!(scales.len(), 4);

    let history = &sim.store.censuses[0].history;
    assert_eq!(history.len(), 1);

    // the peak sample was the 2.5x one, and it was not the last — a fold that kept the most
    // recent value would report the 1.7x row instead.
    let mut reference = build(&mass_census("max", false));
    let _ = samples_with(&mut reference, 4, |k| [1.0, 2.5, 0.4, 1.7][k]);
    let rows = reference.store.censuses[0].history.values().to_vec();
    let want = rows.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(
        history.values()[0],
        want,
        "the accumulated row is {} against a per-sample maximum of {want}; the fold is not the \
         registered reduce op",
        history.values()[0]
    );
    assert!(
        want > *rows.last().expect("four rows"),
        "the peak is the last sample, so a last-write fold would pass this vacuously"
    );
}

#[test]
fn the_accumulated_row_carries_the_span_it_covers() {
    // one row is not self-describing. without the sample count there is no way to recover a time
    // average from a running sum, and without the endpoints no way to say what interval the
    // reduction covers — a row from a hundred samples over one dynamical time and a row from two
    // samples over a thousand read identically.
    let mut sim = build(&mass_census("add", true));
    let _ = samples_with(&mut sim, 6, |_| 1.0);

    let history = &sim.store.censuses[0].history;
    assert_eq!(history.n_samples(), [6]);
    assert_eq!(
        history.t_start(),
        [0.0],
        "the first sample was taken at t = 0"
    );
    assert_eq!(
        history.time()[0],
        2.5,
        "the stored time must be the LAST sample folded in, not the first"
    );

    // and the count is what makes the mean recoverable: six identical samples average back to one.
    let one = history.values()[0] / history.n_samples()[0] as f64;
    let mut plain = build(&mass_census("add", false));
    let _ = samples_with(&mut plain, 1, |_| 1.0);
    let single = plain.store.censuses[0].history.values()[0];
    assert!(
        (one - single).abs() <= 1.0e-12 * single,
        "dividing the accumulated row by its sample count gives {one}, not the single-sample \
         total {single}"
    );
}

#[test]
fn accumulation_composes_with_the_sample_cadence() {
    // the two controls are independent: the interval decides which steps are sampled, the mode
    // decides how the samples that were taken are stored. a run declaring both must fold exactly
    // the due samples — folding the skipped ones as zeros would silently deflate every average by
    // the duty cycle.
    let cfg = CensusConfig::from_json(
        r#"{
            "name": "mass", "op": "add", "axes": [],
            "value_names": ["mass"], "values": [2], "params": [],
            "accumulate": true, "sample_interval": 1.0,
            "nodes": [
                {"op": "VARIABLE_RHO"},
                {"op": "VARIABLE_DV"},
                {"op": "MULTIPLY", "left": 0, "right": 1}
            ]
        }"#,
    )
    .expect("census parses");
    let mut sim = build(&cfg);
    // ten sampling calls at t = 0, 0.5, ... 4.5; an interval of 1.0 admits t = 0, 1, 2, 3, 4.
    let _ = samples_with(&mut sim, 10, |_| 1.0);

    let history = &sim.store.censuses[0].history;
    assert_eq!(
        history.n_samples(),
        [5],
        "ten calls at 0.5 apart under an interval of 1.0 must fold 5 samples, not {:?}",
        history.n_samples()
    );
    assert_eq!(history.len(), 1, "still one row");
    assert_eq!(history.t_start(), [0.0]);
    assert_eq!(history.time()[0], 4.0, "the last DUE sample was at t = 4");
}
