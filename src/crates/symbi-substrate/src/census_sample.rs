// =============================================================================
// census_sample.rs
//
// taking one census sample, for both drivers. the uni-grid loop and the refinement hierarchy
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
/// reduce one level's leaf cells into a partial sample per registered census.
///
/// `covered` is the region a finer level resolves; those cells are not this level's to count.
/// returns one `(values, dropped)` per registration, in registration order.
///
/// the registrations are supplied rather than read from `sim`, because a census is registered once
/// — on the root, which owns the history — while every level of a hierarchy must be reduced
/// against it. reading them from each level instead makes a refined run silently omit its refined
/// volume: the covered coarse cells are excluded from the parent, and a fine level holding no
/// registrations contributes nothing in their place, so the total is short by exactly the refined
/// region and is otherwise smooth, positive and of the right order.
pub fn level_partials<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    censuses: &[symbi_sim::census::RegisteredCensus],
    level: usize,
    cadence: Option<symbi_sim::census::Cadence>,
    covered: Option<&symbi_algebra::Domain<D>>,
) -> Vec<(Vec<f64>, u64)>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    symbi_sim::driver::prof("census", || {
        let now = sim.time;
        censuses
            .iter()
            .enumerate()
            .map(|(index, registered)| {
            // not due: an empty partial, which `combine_partials` folds as a no-op and
            // `record_samples` skips. the alternative — dropping the entry — would misalign every
            // registration after it, since partials are matched to registrations by position.
            // and a registration whose declared cadence is not the one being driven here: a
            // root-step census must not also sample inside a subcycle, and a per-level one must
            // not also be folded into the composite root sample, or the same physical state would
            // enter the history twice under two different tags. `None` is a driver with a single
            // level, where the two cadences are the same instant.
            let spec_cadence = registered.evaluator.spec().cadence();
            if cadence.is_some_and(|c| c != spec_cadence) || !registered.is_due_at_level(now, level)
            {
                return (Vec::new(), 0u64);
            }
            let spec = registered.evaluator.spec();
            // the compiled map first — the same traced kernel a device runs — falling back to the
            // per-cell interpreter when it does not apply. both write into the same scratch, whose
            // every cell starts excluded, so whichever runs leaves untouched cells out of the
            // reduction rather than in bucket zero.
            let fields = census_map_fields(sim, censuses, index, &registered.evaluator, covered).unwrap_or_else(|| {
                panic!(
                    "census '{}' cannot be evaluated on this store: neither the compiled map nor \
                     the per-cell interpreter applies here",
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
    })
}

/// fold one level's partial into a running total, per registration.
///
/// the combine is the census's own reduce op, which is why an accumulator has to be a
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
        // a level that was not due contributes nothing; folding it would be a no-op anyway, but
        // the lengths would not line up.
        if add.is_empty() || acc.is_empty() {
            continue;
        }
        for (a, b) in acc.iter_mut().zip(add) {
            *a = symbi_sim::census::combine(op, *a, b);
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
    let samples = level_partials(sim, &sim.store.censuses, 0, None, None);
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
    record_samples_at_level(sim, time, 0, samples);
}

/// commit one already-reduced sample per registration, tagged with the level that produced it.
pub fn record_samples_at_level<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    time: f64,
    level: usize,
    samples: Vec<(Vec<f64>, u64)>,
) where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    for (registered, (values, dropped)) in sim.store.censuses.iter_mut().zip(samples) {
        // an empty partial is a registration that was not due this step.
        if values.is_empty() {
            continue;
        }
        registered
            .history
            .push_at_level(time, level as u64, &values, dropped);
        registered.mark_sampled(level, time);
    }
}

/// the per-cell artifacts of registration `index`, by whichever path applies.
///
/// neither fill honours a finer level's coverage — both sweep the whole interior — so the covered
/// cells are marked excluded afterwards, uniformly. that ordering is what makes the scratch
/// reusable: a fill that skipped covered cells would leave the previous sample's bucket standing in
/// a cell that is no longer this level's to count.
pub fn census_map_fields<'a, R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &'a SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    censuses: &[symbi_sim::census::RegisteredCensus],
    index: usize,
    ev: &symbi_sim::census::CensusEvaluator,
    covered: Option<&symbi_algebra::Domain<D>>,
) -> Option<&'a symbi_sim::census::CensusFields<D, Mem>>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let scratch = sim.census_scratch_pooled(censuses, index)?;
    let segment_stamp = if ev.axes_are_geometry_only() {
        Some(static_segment_stamp(sim.motion.a, covered))
    } else {
        None
    };
    let segment_is_cached = segment_stamp.is_some_and(|stamp| {
        scratch
            .segment_stamp
            .lock()
            .unwrap()
            .is_some_and(|cached| cached == stamp)
    });
    let compiled = crate::regimes::substrate_kernels::census_compiled::census_map_compiled(
        &sim.store,
        ev,
        &scratch.values,
        &scratch.segment,
        !segment_is_cached,
    );
    if !compiled {
        sim.census_fill_interpreted(ev, scratch)?;
    }
    // The interpreter always writes the segment. The split compiled kernel does so only when the
    // geometry/coverage stamp changed; a cached segment already includes its covered-cell mask.
    if !compiled || !segment_is_cached {
        if let Some(region) = covered {
            exclude_covered(ev, &scratch.segment, region);
        }
        if let Some(stamp) = segment_stamp {
            *scratch.segment_stamp.lock().unwrap() = Some(stamp);
        }
    }
    Some(scratch)
}

/// identity of everything a geometry-only bucket assignment depends on. The mesh scale factor
/// invalidates expanding coordinates/volumes, and the covered domain invalidates the AMR leaf
/// mask. Grid origin, spacing and allocated layout are immutable for a level's lifetime and the
/// scratch itself belongs to exactly that level, so they need no per-sample contribution.
fn static_segment_stamp<const D: usize>(
    scale_factor: f64,
    covered: Option<&symbi_algebra::Domain<D>>,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    scale_factor.to_bits().hash(&mut h);
    covered.is_some().hash(&mut h);
    if let Some(domain) = covered {
        for space in &domain.spaces {
            space.lo.hash(&mut h);
            space.hi.hash(&mut h);
        }
    }
    h.finish()
}

/// mark a level's covered cells excluded: the cells a finer level resolves are counted there, not
/// here, so they must carry the marker that keeps them out of the reduction.
///
/// a constant fill over the covered region, dispatched through the same generic kernel path every
/// other field operation uses, so a device-resident hierarchy excludes on the device. walking the
/// region on the host instead would work only where the fields happen to be host-accessible — and
/// where they are not, the covered cells keep the bucket the map assigned them and the refined
/// volume is counted on both levels, inflating every extensive total by exactly that volume.
fn exclude_covered<const D: usize, Mem: MemorySpace>(
    ev: &symbi_sim::census::CensusEvaluator,
    segment: &symbi_grid::Field<f64, D, Mem>,
    covered: &symbi_algebra::Domain<D>,
) {
    let name = symbi_ir::KernelId::FieldFill { ndim: D as u8 }.name();
    let excluded = ev.spec().excluded_marker();
    crate::regimes::substrate_kernels::dispatch_fields_each::<f64, Mem, D>(
        name,
        covered,
        &[],
        &[segment],
        &[],
        &[excluded],
    );
}
