// =============================================================================
// census_sample.rs
//
// taking one census sample, for BOTH drivers. the uni-grid loop and the refinement hierarchy
// each run their own step, and a sampling call that lived in only one of them would leave the
// other silently recording nothing — a census that writes no rows is indistinguishable from a
// run that was never asked for one.
//
// usage:
//   census_sample::sample_censuses(sim);   // at the tail of an accepted step
// =============================================================================

use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_sim::state::SimStateGeneric;
use symbi_xpu::{ExecutionSpace, MemorySpace};

/// evaluate every registered census over the current state and append one sample each.
///
/// a tracerless-style inertness holds: with no registrations this is a length check and
/// nothing is allocated or evaluated, so an unregistered run pays nothing.
///
/// the samples are collected before any is recorded because evaluating a census borrows the
/// state immutably while appending to its series needs it mutably.
/// reduce one LEVEL's leaf cells into a partial sample per registered census.
///
/// `covered` is the region a finer level resolves; those cells are not this level's to count.
/// returns one `(values, dropped)` per registration, in registration order.
pub fn level_partials<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    covered: Option<&symbi_algebra::Domain<D>>,
) -> Vec<(Vec<f64>, u64)>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    sim.censuses
        .iter()
        .map(|registered| {
            let spec = registered.evaluator.spec();
            let fields = sim
                .census_fields(&registered.evaluator, covered)
                .unwrap_or_else(|| {
                    panic!(
                        "census '{}' cannot be evaluated on device-resident fields; the binned \
                         reduction's per-cell map is host-only, so run this configuration on the \
                         cpu or drop the registration",
                        spec.name()
                    )
                });
            let values: Vec<_> = fields.values.iter().collect();
            let reduced = crate::regimes::substrate_gpu::field_segmented_reduce(
                &values,
                &fields.segment,
                &sim.geom.interior,
                spec.n_segments(),
                spec.op(),
            );
            (reduced.values, reduced.dropped)
        })
        .collect()
}

/// fold one level's partial into a running total, per registration.
///
/// the combine is the census's OWN reduce op, which is why an accumulator has to be a
/// commutative monoid: the same operator that merges two cells merges two levels, in whatever
/// order the levels happen to be visited.
pub fn combine_partials(
    into: &mut [(Vec<f64>, u64)],
    from: Vec<(Vec<f64>, u64)>,
    ops: &[symbi_ir::emit::ReductionOp],
) {
    for ((acc, dropped), ((add, d), op)) in into
        .iter_mut()
        .zip(from.into_iter().zip(ops.iter().copied()))
    {
        for (a, b) in acc.iter_mut().zip(add) {
            *a = match op {
                symbi_ir::emit::ReductionOp::Add => *a + b,
                symbi_ir::emit::ReductionOp::Min => a.min(b),
                symbi_ir::emit::ReductionOp::Max => a.max(b),
                symbi_ir::emit::ReductionOp::Mul => *a * b,
            };
        }
        *dropped += d;
    }
}

/// append one sample per registered census, from this store's leaf cells alone.
///
/// a tracerless-style inertness holds: with no registrations this is a length check and
/// nothing is allocated or evaluated, so an unregistered run pays nothing.
///
/// the samples are collected before any is recorded because evaluating a census borrows the
/// state immutably while appending to its series needs it mutably.
pub fn sample_censuses<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
) where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    if sim.censuses.is_empty() {
        return;
    }
    let time = sim.time;
    let samples = level_partials(sim, None);
    record_samples(sim, time, samples);
}

/// commit one already-reduced sample per registration.
pub fn record_samples<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    time: f64,
    samples: Vec<(Vec<f64>, u64)>,
) where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    for (registered, (values, dropped)) in sim.store.censuses.iter_mut().zip(samples) {
        registered.history.push(time, &values, dropped);
    }
}
