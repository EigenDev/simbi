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
use symbi_sim::state::{FieldStore, SimStateGeneric};
use symbi_xpu::{ExecutionSpace, MemorySpace};

/// reduce one level's leaf cells into a partial sample per registered census.
///
/// `covered` is the region a finer level resolves; those cells are not this level's to count.
/// `now` is the time the sample will carry, the time of the state it holds: a registration is due
/// when `now` lies at least its interval past the time its previous row carries.
/// returns one `(values, dropped)` per registration, in registration order.
///
/// the registrations are supplied rather than read from `sim`, because a census is registered once
/// — on the root, which owns the history — while every level of a hierarchy must be reduced
/// against it. reading them from each level instead makes a refined run silently omit its refined
/// volume: the covered coarse cells are excluded from the parent, and a fine level holding no
/// registrations contributes nothing in their place, so the total is short by exactly the refined
/// region and is otherwise smooth, positive and of the right order.
#[allow(clippy::too_many_arguments)]
pub fn store_partials_exact<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    censuses: &[symbi_sim::census::RegisteredCensus],
    level: usize,
    cadence: Option<symbi_sim::census::Cadence>,
    covered: Option<&symbi_algebra::Domain<D>>,
    nonphysical: NonphysicalCells,
    now: f64,
    interpret: &dyn Fn(&symbi_sim::census::CensusEvaluator, &symbi_sim::census::CensusFields<D, Mem>) -> Option<()>,
) -> Vec<symbi_sim::census::CensusPartial> {
    symbi_sim::driver::prof("census", || {
        censuses
            .iter()
            .enumerate()
            .map(|(index, registered)| {
            // not due: an empty partial, which `merge_partials` folds as a no-op and
            // `record_census` skips. the alternative — dropping the entry — would misalign every
            // registration after it, since partials are matched to registrations by position.
            // and a registration whose declared cadence is not the one being driven here: a
            // root-step census must not also sample inside a subcycle, and a per-level one must
            // not also be folded into the composite root sample, or the same physical state would
            // enter the history twice under two different tags. `None` is a driver with a single
            // level, where the two cadences are the same instant.
            let spec_cadence = registered.evaluator.spec().cadence();
            if cadence.is_some_and(|c| c != spec_cadence) || !registered.is_due_at_level(now, level)
            {
                return symbi_sim::census::CensusPartial::None;
            }
            let spec = registered.evaluator.spec();
            // the compiled map first — the same traced kernel a device runs — falling back to the
            // per-cell interpreter when it does not apply. both write into the same scratch, whose
            // every cell starts excluded, so whichever runs leaves untouched cells out of the
            // reduction rather than in bucket zero.
            let fields = census_map_fields(sim, censuses, index, &registered.evaluator, covered, nonphysical, interpret).unwrap_or_else(|| {
                panic!(
                    "census '{}' cannot be evaluated on this store: neither the compiled map nor \
                     the per-cell interpreter applies here",
                    spec.name()
                )
            });
            let values: Vec<_> = fields.values.iter().collect();
            // an additive census on host memory sums exactly, so its partials merge across levels,
            // tiles and workers into the whole grid's sums; the device reduction combines by the op
            if spec.op() == symbi_ir::emit::ReductionOp::Add && !Mem::IS_DEVICE_ACCESSIBLE {
                let (sums, dropped) =
                    symbi_exec::engine::field_segmented_sum_exact(&values, &fields.segment, &sim.geom.interior, spec.n_segments());
                return symbi_sim::census::CensusPartial::Sums { values: sums, dropped };
            }
            let reduced = crate::regimes::substrate_gpu::field_segmented_reduce(
                &values,
                &fields.segment,
                &sim.geom.interior,
                spec.n_segments(),
                spec.op(),
            );
            symbi_sim::census::CensusPartial::Values { values: reduced.values, dropped: reduced.dropped }
        })
        .collect()
    })
}

/// `store_partials_exact` on a tile, which carries no typed metric: the compiled map evaluates
/// every census a tiled run admits
#[allow(clippy::too_many_arguments)]
pub fn tile_partials_exact<const D: usize, const DOF: usize, Mem: MemorySpace>(
    store: &FieldStore<D, DOF, Mem, f64>,
    censuses: &[symbi_sim::census::RegisteredCensus],
    level: usize,
    cadence: Option<symbi_sim::census::Cadence>,
    covered: Option<&symbi_algebra::Domain<D>>,
    excision_radius: Option<f64>,
    now: f64,
) -> Vec<symbi_sim::census::CensusPartial> {
    store_partials_exact(store, censuses, level, cadence, covered, NonphysicalCells { excision_radius }, now, &|_, _| None)
}

/// `store_partials_exact` on a typed state, whose metric lets the per-cell interpreter stand in
/// where the compiled map declines
#[allow(clippy::too_many_arguments)]
pub fn level_partials_exact<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    censuses: &[symbi_sim::census::RegisteredCensus],
    level: usize,
    cadence: Option<symbi_sim::census::Cadence>,
    covered: Option<&symbi_algebra::Domain<D>>,
    nonphysical: NonphysicalCells,
    now: f64,
) -> Vec<symbi_sim::census::CensusPartial>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    store_partials_exact(&sim.store, censuses, level, cadence, covered, nonphysical, now, &|ev, fields| {
        sim.census_fill_interpreted(ev, fields)
    })
}

/// `level_partials_exact` with each partial rounded: one sample per registration of this level
/// alone
#[allow(clippy::too_many_arguments)]
pub fn level_partials<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    censuses: &[symbi_sim::census::RegisteredCensus],
    level: usize,
    cadence: Option<symbi_sim::census::Cadence>,
    covered: Option<&symbi_algebra::Domain<D>>,
    nonphysical: NonphysicalCells,
    now: f64,
) -> Vec<(Vec<f64>, u64)>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    level_partials_exact(sim, censuses, level, cadence, covered, nonphysical, now)
        .iter()
        .map(|p| p.round())
        .collect()
}

/// fold the partials of another level, tile or worker into running totals, per registration, by
/// each census's reduce op
pub fn merge_partials(
    into: &mut [symbi_sim::census::CensusPartial],
    from: &[symbi_sim::census::CensusPartial],
    ops: &[symbi_ir::emit::ReductionOp],
) {
    for ((acc, add), op) in into.iter_mut().zip(from).zip(ops) {
        acc.merge(add, *op);
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
    nonphysical: NonphysicalCells,
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
    let partials = level_partials_exact(sim, &sim.store.censuses, 0, None, None, nonphysical, time);
    sim.store.record_census(time, 0, &partials);
}

/// the per-cell artifacts of registration `index`, by whichever path applies.
///
/// neither fill honours a finer level's coverage — both sweep the whole interior — so the covered
/// cells are marked excluded afterwards, uniformly. that ordering is what makes the scratch
/// reusable: a fill that skipped covered cells would leave the previous sample's bucket standing in
/// a cell that is no longer this level's to count.
pub fn census_map_fields<'a, const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &'a FieldStore<D, DOF, Mem, f64>,
    censuses: &[symbi_sim::census::RegisteredCensus],
    index: usize,
    ev: &symbi_sim::census::CensusEvaluator,
    covered: Option<&symbi_algebra::Domain<D>>,
    nonphysical: NonphysicalCells,
    interpret: &dyn Fn(&symbi_sim::census::CensusEvaluator, &symbi_sim::census::CensusFields<D, Mem>) -> Option<()>,
) -> Option<&'a symbi_sim::census::CensusFields<D, Mem>>
{
    let scratch = sim.census_scratch_pooled(censuses, index)?;
    // the body and excision masks the sample time sees: a moved body changes the stamp, so a
    // cached segment never keeps a cell the body has left out of the reduction
    let masks = nonphysical_masks(sim, nonphysical);
    let segment_stamp = if ev.axes_are_geometry_only() {
        Some(static_segment_stamp(sim.motion.a, covered, &masks))
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
        sim,
        ev,
        &scratch.values,
        &scratch.segment,
        !segment_is_cached,
    );
    if !compiled {
        interpret(ev, scratch)?;
    }
    // The interpreter always writes the segment. The split compiled kernel does so only when the
    // geometry/coverage stamp changed; a cached segment already includes its covered-cell mask.
    if !compiled || !segment_is_cached {
        if let Some(region) = covered {
            exclude_covered(ev, &scratch.segment, region);
        }
        exclude_nonphysical(sim, ev, &scratch.segment, &masks);
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
    masks: &NonphysicalMasks,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    scale_factor.to_bits().hash(&mut h);
    masks.stamp.hash(&mut h);
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

/// the cells of a store a census leaves out beyond its ghosts and its covered cells: a cell whose
/// centroid lies in the excised core of a kerr-schild horizon, of radius `excision_radius`. the
/// cells inside an immersed body's mask come from the store's own bodies.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NonphysicalCells {
    pub excision_radius: Option<f64>,
}

/// the masks a sample applies, posed at the sample's time, and the identity of their geometry
struct NonphysicalMasks {
    /// the excised core's radius and the hole's spin
    excised: Option<(f64, f64)>,
    /// the posed geometry of every body whose surface penalizes the gas: its interior is the
    /// body, not gas
    bodies: Vec<symbi_ib::sdf::SdfExpr<f64, 3>>,
    stamp: u64,
}

/// the excised core and the interiors of the bodies whose surface acts on the gas (an accretor's
/// drain, a porous or rigid wall), each at the position and orientation it holds now. a body that
/// only gravitates masks nothing: the gas it pulls on is gas
fn nonphysical_masks<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    nonphysical: NonphysicalCells,
) -> NonphysicalMasks
{
    use std::hash::{Hash, Hasher};
    use symbi_ib::sdf::SdfExpr;
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let spin = sim.geom.spacetime_scalars.iter().find(|(n, _)| n == "kerr_spin").map_or(0.0, |(_, v)| *v);
    let excised = nonphysical.excision_radius.filter(|r| *r > 0.0).map(|r| (r, spin));
    if let Some((r, a)) = excised {
        (r.to_bits(), a.to_bits()).hash(&mut h);
    }
    let mut bodies = Vec::new();
    if let Some(immersed) = sim.immersed.as_ref() {
        for (b, body) in immersed.bodies.bodies().iter().enumerate() {
            if !(body.has_accretion() || body.has_rigid()) {
                continue;
            }
            // the body's position components name the cartesian axes of the grid's plane
            let mut center = [0.0f64; 3];
            for (k, &cart) in symbi_geometry::body_cartesian_axes(sim.geom.coords, D, &sim.geom.axes).iter().enumerate() {
                if k < D && cart < 3 {
                    center[cart] = body.position[k];
                }
            }
            let shape = match immersed.shapes.get(b).and_then(|s| s.as_ref()) {
                Some(shape) => shape.clone().rotated(body.orientation),
                None => match body.mask_radius() {
                    Some(radius) => SdfExpr::Sphere { center: [0.0; 3], radius },
                    None => continue,
                },
            };
            (b, center.map(f64::to_bits), body.orientation.map(|r| r.map(f64::to_bits))).hash(&mut h);
            body.mask_radius().map(f64::to_bits).hash(&mut h);
            bodies.push(shape.translated(center));
        }
    }
    NonphysicalMasks { excised, bodies, stamp: h.finish() }
}

/// mark the excised cells and the cells inside a masking body excluded, each by the predicate the
/// run's own kernels apply: a cell is excised when the excision fill freezes it (on a cartesian
/// chart its face-midpoint centroid lies strictly inside the kerr-schild radius; on a spherical
/// chart its radial face midpoint lies inside the excision radius, the fill's sampling point), and
/// masked when the cartesian image of its volume centroid lies strictly inside a body (negative
/// signed distance). a whole-cell rule: a cell is gas or it is not
fn exclude_nonphysical<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    ev: &symbi_sim::census::CensusEvaluator,
    segment: &symbi_grid::Field<f64, D, Mem>,
    masks: &NonphysicalMasks,
) {
    if masks.excised.is_none() && masks.bodies.is_empty() {
        return;
    }
    assert!(
        !Mem::IS_DEVICE_ACCESSIBLE,
        "a census over excised or masked cells on device memory awaits its device mask kernel"
    );
    let excluded = ev.spec().excluded_marker();
    let chart = sim.geom.coords;
    for c in sim.geom.interior.iter() {
        // the volume centroid in natural coordinate order, an ungridded coordinate at zero
        let centroid = sim.geom.expression_point(c);
        let mut natural = [0.0f64; 3];
        for (d, &slot) in sim.geom.axes.iter().enumerate() {
            if slot < 3 {
                natural[slot] = centroid[d];
            }
        }
        let excised = masks.excised.is_some_and(|(r, a)| match chart {
            symbi_geometry::Geometry::Cartesian => symbi_ib::excise::ks_excised(&natural, a, r),
            symbi_geometry::Geometry::Spherical => {
                let lo = sim.geom.face_coord(c, 0)[0];
                let mut upper = c;
                upper[0] += 1;
                0.5 * (lo + sim.geom.face_coord(upper, 0)[0]) < r
            }
            symbi_geometry::Geometry::Cylindrical => {
                panic!("a census excision mask on a cylindrical chart, where excision is refused at admission")
            }
        });
        let cartesian = symbi_geometry::chart_to_cartesian(chart, natural);
        if excised || masks.bodies.iter().any(|body| body.dist(cartesian) < 0.0) {
            segment.view_mut().set(c, excluded);
        }
    }
}
