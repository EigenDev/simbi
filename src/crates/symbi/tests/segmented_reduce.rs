// =============================================================================
// segmented_reduce.rs
//
// the segmented Reduce morphism: every cell carries a destination bucket, and each
// value field is combined into that bucket, giving n_segments * n_values
// accumulators from one pass.
//
// the properties that make it usable as the scatter half of a binned reduction:
//   - partition: summing an extensive value over every bucket equals the whole-field
//     reduction of the same value. a bucket assignment that double-counted or lost
//     cells would break here and nowhere else.
//   - the zero-axis case (one bucket) reproduces the whole-field reduction exactly.
//   - cells outside the binning are dropped and counted, so an under-covering
//     binning reads distinctly from a physics result.
//   - the host combine is bit-reproducible across thread counts, per the fixed
//     slab-order contract the whole-field reduce already holds.
// =============================================================================

use symbi::regimes::substrate_gpu::{ReductionOrder, field_reduce, field_segmented_reduce};
use symbi_algebra::{Domain, Space};
use symbi_grid::Field;
use symbi_ir::emit::ReductionOp;
use symbi_xpu::HostMemory;

const NX: isize = 512;
const NY: isize = 512; // 512 x 512 = 2^18 cells: well past the parallel threshold

const N_SEGMENTS: usize = 8;

fn domain() -> Domain<2> {
    Domain::new([
        Space {
            name: "x",
            lo: 0,
            hi: NX,
        },
        Space {
            name: "y",
            lo: 0,
            hi: NY,
        },
    ])
}

/// the bucket a cell lands in. a non-trivial, non-contiguous assignment so a bucket
/// gathers cells scattered across the whole domain rather than one tidy slab — a
/// contiguous assignment would hide an indexing error that a scattered one exposes.
fn bucket(c: [isize; 2]) -> u32 {
    ((c[0] * 7 + c[1] * 3) % N_SEGMENTS as isize) as u32
}

/// adversarial magnitudes: the running sum loses the small addend whenever a large one
/// is present, so any reassociation of the combine is visible in the result. an
/// all-ones field would pass under any order.
fn value(c: [isize; 2], channel: isize) -> f64 {
    let v = if (c[0] + c[1] * channel) % 7 == 0 {
        1.0e16
    } else {
        1.0
    };
    if (c[0] * 3 + c[1]) % 11 == 0 { -v } else { v }
}

fn build(n_values: usize) -> (Vec<Field<f64, 2, HostMemory>>, Field<f64, 2, HostMemory>) {
    let dom = domain();
    let values: Vec<Field<f64, 2, HostMemory>> = (0..n_values)
        .map(|v| {
            let f = Field::<f64, 2, HostMemory>::zeros(&dom).expect("value field");
            for c in dom.iter() {
                f.view_mut().set(c, value(c, v as isize + 1));
            }
            f
        })
        .collect();
    let segment = Field::<f64, 2, HostMemory>::zeros(&dom).expect("segment field");
    for c in dom.iter() {
        segment.view_mut().set(c, bucket(c) as f64);
    }
    (values, segment)
}

#[test]
fn summing_every_bucket_equals_the_whole_field_reduction() {
    // the partition property. the bucket assignment covers the domain exactly once, so
    // the buckets must add back up to the global sum. this catches double counting and
    // lost cells directly, which is why it is the load-bearing check on the mechanism.
    let (values, segment) = build(3);
    let dom = domain();
    let refs: Vec<&Field<f64, 2, HostMemory>> = values.iter().collect();

    let census = field_segmented_reduce(&refs, &segment, &dom, N_SEGMENTS, ReductionOp::Add);
    assert_eq!(census.values.len(), N_SEGMENTS * refs.len());
    assert_eq!(census.dropped, 0, "the binning covers every cell");

    for (v, field) in refs.iter().enumerate() {
        let global = field_reduce(*field, &dom, ReductionOp::Add);
        let binned: f64 = (0..N_SEGMENTS)
            .map(|s| census.values[s * refs.len() + v])
            .sum();
        // the two group the same addends differently, so they agree to reassociation
        // roundoff rather than bitwise. the 1e16-against-1 mix sets the scale.
        let tol = 1.0e-12 * global.abs().max(1.0);
        assert!(
            (binned - global).abs() <= tol,
            "value {v}: buckets sum to {binned:e}, whole-field reduce gives {global:e}"
        );
    }
}

#[test]
fn one_bucket_reproduces_the_whole_field_reduction() {
    // the zero-axis census: with the bin axes empty the census is a global reduction. expressing
    // that is what makes the mechanism a generalization of the reduction it sits beside.
    let dom = domain();
    let field = Field::<f64, 2, HostMemory>::zeros(&dom).expect("value field");
    for c in dom.iter() {
        field.view_mut().set(c, value(c, 1));
    }
    // every cell into bucket zero.
    let segment = Field::<f64, 2, HostMemory>::zeros(&dom).expect("segment field");

    for op in [ReductionOp::Add, ReductionOp::Min, ReductionOp::Max] {
        let census = field_segmented_reduce(&[&field], &segment, &dom, 1, op);
        let global = field_reduce(&field, &dom, op);
        assert_eq!(census.values.len(), 1);
        assert_eq!(census.dropped, 0);
        // both fold the same cells in the same slab order, so they agree BITWISE.
        assert!(
            census.values[0].to_bits() == global.to_bits(),
            "{op:?}: one-bucket census {:e} != whole-field reduce {global:e}",
            census.values[0]
        );
    }
}

#[test]
fn cells_outside_the_binning_are_dropped_and_counted() {
    // a bin index past the last bucket means the cell fell outside the declared edges.
    // it is dropped clear of every bucket and the shortfall is reported, because a silently
    // under-covering binning reads exactly like a physics result.
    let dom = domain();
    let field = Field::<f64, 2, HostMemory>::zeros(&dom).expect("value field");
    let segment = Field::<f64, 2, HostMemory>::zeros(&dom).expect("segment field");
    let mut expect_dropped = 0u64;
    let mut expect_kept = 0.0f64;
    for c in dom.iter() {
        field.view_mut().set(c, 1.0);
        // a third of the domain is placed beyond the last bucket.
        if c[0] % 3 == 0 {
            segment.view_mut().set(c, N_SEGMENTS as f64);
            expect_dropped += 1;
        } else {
            segment.view_mut().set(c, bucket(c) as f64);
            expect_kept += 1.0;
        }
    }

    let census = field_segmented_reduce(&[&field], &segment, &dom, N_SEGMENTS, ReductionOp::Add);
    assert_eq!(
        census.dropped, expect_dropped,
        "every out-of-range cell must be counted"
    );
    let kept: f64 = census.values.iter().sum();
    assert_eq!(
        kept, expect_kept,
        "an out-of-range cell must not be folded into any bucket"
    );
}

#[test]
fn host_buckets_are_bit_reproducible_across_thread_counts() {
    // the fixed slab-order contract, per bucket. the partition is one slab per outer
    // index — a function of the domain shape alone — so the grouping of addends is fixed
    // whatever the machine's thread count. a thread-count-dependent partition would regroup
    // the 1e16-against-1 mix and shift the low bits visibly.
    let (values, segment) = build(2);
    let dom = domain();
    let refs: Vec<&Field<f64, 2, HostMemory>> = values.iter().collect();

    let first = field_segmented_reduce(&refs, &segment, &dom, N_SEGMENTS, ReductionOp::Add);
    assert_eq!(first.order, ReductionOrder::Exact);

    for threads in [1usize, 2, 3, 7] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        let again = pool.install(|| {
            field_segmented_reduce(&refs, &segment, &dom, N_SEGMENTS, ReductionOp::Add)
        });
        for (slot, (a, b)) in again.values.iter().zip(&first.values).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "slot {slot} differs at {threads} threads: {a:e} vs {b:e}"
            );
        }
    }
}

#[test]
fn min_and_max_land_on_the_per_bucket_extrema() {
    // the extrema ops, checked against a direct scan. min/max are order-agnostic, so
    // they are exact under any partition — which is why they stay bit-reproducible in
    // regimes where a sum drifts with the grouping.
    let (values, segment) = build(1);
    let dom = domain();

    let mins = field_segmented_reduce(&[&values[0]], &segment, &dom, N_SEGMENTS, ReductionOp::Min);
    let maxs = field_segmented_reduce(&[&values[0]], &segment, &dom, N_SEGMENTS, ReductionOp::Max);

    let mut expect_min = vec![f64::INFINITY; N_SEGMENTS];
    let mut expect_max = vec![f64::NEG_INFINITY; N_SEGMENTS];
    for c in dom.iter() {
        let s = bucket(c) as usize;
        let v = value(c, 1);
        expect_min[s] = expect_min[s].min(v);
        expect_max[s] = expect_max[s].max(v);
    }
    assert_eq!(mins.values, expect_min);
    assert_eq!(maxs.values, expect_max);
}

#[test]
fn a_poisoned_cell_survives_its_bucket() {
    // min/max must PROPAGATE NaN rather than silently returning the non-NaN operand,
    // which is what the bare `f64::min` does. a dropped NaN would let a poisoned cell
    // pass a census unnoticed, and only the bucket that contains it may be poisoned.
    let dom = domain();
    let field = Field::<f64, 2, HostMemory>::zeros(&dom).expect("value field");
    let segment = Field::<f64, 2, HostMemory>::zeros(&dom).expect("segment field");
    let poisoned = [1isize, 1isize];
    for c in dom.iter() {
        field
            .view_mut()
            .set(c, if c == poisoned { f64::NAN } else { 1.0 });
        segment.view_mut().set(c, bucket(c) as f64);
    }
    let hot = bucket(poisoned) as usize;

    for op in [ReductionOp::Min, ReductionOp::Max, ReductionOp::Add] {
        let census = field_segmented_reduce(&[&field], &segment, &dom, N_SEGMENTS, op);
        assert!(
            census.values[hot].is_nan(),
            "{op:?}: bucket {hot} holds the poisoned cell and must report NaN"
        );
        for (s, v) in census.values.iter().enumerate() {
            if s != hot {
                assert!(!v.is_nan(), "{op:?}: bucket {s} has no poisoned cell");
            }
        }
    }
}

#[test]
#[should_panic(expected = "Mul is not a meaningful segmented reduction")]
fn a_product_over_a_bucket_is_refused() {
    // a product over a bucket's cells overflows to zero or infinity at any realistic
    // cell count and carries no meaning as a census statistic.
    let (values, segment) = build(1);
    let dom = domain();
    field_segmented_reduce(&[&values[0]], &segment, &dom, N_SEGMENTS, ReductionOp::Mul);
}

#[test]
fn the_host_accumulator_is_double_precision_whatever_the_field_carrier_is() {
    // the accumulator width is a property of the reduction, held independent of the fields it
    // reads. that
    // separation is what lets a single-precision run — which is what a device without double
    // support forces — still produce a total that means something: a bin over three million cells
    // summed in f32 loses the low bits of every term once the running sum outgrows them, and the
    // result is a smooth, positive, entirely plausible number that is wrong in its third digit.
    //
    // the discrimination here is absorption. one large term followed by many small ones: in f64
    // every small term lands, in f32 none of them do, and the two answers differ by the whole tail.
    const BIG: f32 = 1.0e8;
    const N_SMALL: isize = 1000;

    let dom = Domain::new([Space {
        name: "x",
        lo: 0,
        hi: N_SMALL + 1,
    }]);
    let value = Field::<f32, 1, HostMemory>::zeros(&dom).expect("value field");
    let segment = Field::<f32, 1, HostMemory>::zeros(&dom).expect("segment field");
    for c in dom.iter() {
        // the large term FIRST in traversal order, so the running sum is already too coarse to
        // resolve the ones that follow.
        value.view_mut().set(c, if c[0] == 0 { BIG } else { 1.0 });
        segment.view_mut().set(c, 0.0);
    }

    let got = field_segmented_reduce(&[&value], &segment, &dom, 1, ReductionOp::Add);
    let want = BIG as f64 + N_SMALL as f64;
    assert_eq!(
        got.values[0], want,
        "the reduction of {N_SMALL} unit terms after one of {BIG} gave {}, not {want}. the \
         accumulator has narrowed to the field's carrier, so the tail of every bin is being \
         absorbed into its running sum.",
        got.values[0]
    );

    // the premise: an f32 accumulator must genuinely lose this tail, or the comparison above
    // holds for reasons unrelated to precision.
    let mut narrow = BIG;
    for _ in 0..N_SMALL {
        narrow += 1.0f32;
    }
    assert_eq!(
        narrow, BIG,
        "single precision did not absorb the tail here ({narrow} vs {BIG}); the magnitudes no \
         longer discriminate between the two accumulator widths"
    );
}
