// =============================================================================
// exact_device_sum.rs
//
// the exact sum of a field (`symbi_exec::engine::field_sum_exact`) on the
// default memory: the gpu build's device memory, the host memory otherwise. a
// tile's partial is the unrounded long accumulator, so tiles merge into the
// whole window's sum and round once, as on the host.
// - a cancellation-sensitive field (1e16, -1e16 and three unit terms by the
//   cell's index mod 5) cut into two tiles: the merged tile partials equal the
//   whole window's accumulator word for word, the host accumulator of the same
//   terms, and the counted total; on device memory the rounded block reduction
//   (`field_reduce`) of the same tiles disagrees with the whole window's, the
//   rounding the exact sum removes
// - non-finite and signed-zero terms follow ieee addition as on the host: a NaN,
//   opposite infinities, one infinity, all negative zeros, an empty window
// - extreme, subnormal and widely spread terms, and a single-precision field
//   whose terms (non-finite and signed zero included) widen exactly, equal the
//   host accumulator word for word
// - a window of 2^21 halves carries through every digit and sums to 2^20
// - cost: the exact sum and the rounded block reduction of a 128^3 window,
//   timed and printed
//
// run: cargo test -p symbi --test exact_device_sum -- --nocapture
//      cargo test -p symbi --features hip --test exact_device_sum -- --nocapture   (on a gpu)
// =============================================================================
use symbi_algebra::exact_sum::ExactSum;
use symbi_algebra::{Domain, Space};
use symbi_exec::engine::{field_reduce, field_sum_exact};
use symbi_grid::Field;
use symbi_ir::emit::ReductionOp;
use symbi_xpu::{DefaultMemory, MemoryStorage};

const DEVICE: bool = <DefaultMemory as MemoryStorage>::IS_DEVICE_ACCESSIBLE;

fn domain(lo: [isize; 3], hi: [isize; 3]) -> Domain<3> {
    Domain::new(std::array::from_fn(|a| Space { name: ["x", "y", "z"][a].into(), lo: lo[a], hi: hi[a] }))
}

/// a field on [lo - 1, hi + 1) (one halo cell each side, outside every window) holding `value` at
/// each interior cell
fn field(lo: [isize; 3], hi: [isize; 3], value: impl Fn([isize; 3]) -> f64) -> Field<f64, 3, DefaultMemory> {
    let alloc = domain(lo.map(|l| l - 1), hi.map(|h| h + 1));
    let f = Field::<f64, 3, DefaultMemory>::zeros(&alloc).expect("field");
    for c in alloc.iter() {
        let inside = (0..3).all(|a| c[a] >= lo[a] && c[a] < hi[a]);
        f.set(c, if inside { value(c) } else { 7.0e300 });
    }
    f
}

fn host_sum(terms: impl Iterator<Item = f64>) -> ExactSum {
    let mut acc = ExactSum::new();
    for x in terms {
        acc.add(x);
    }
    acc
}

fn cancelling(c: [isize; 3]) -> f64 {
    match (c[0] * 64 + c[1] * 8 + c[2]) % 5 {
        0 => 1e16,
        1 => -1e16,
        _ => 1.0,
    }
}

#[test]
fn a_device_tile_partial_merges_as_the_host_accumulator_does() {
    let f = field([0, 0, 0], [8, 8, 8], cancelling);
    let whole = domain([0, 0, 0], [8, 8, 8]);
    let tiles = [domain([0, 0, 0], [8, 8, 4]), domain([0, 0, 4], [8, 8, 8])];

    let exact_whole = field_sum_exact(&f, &whole);
    let mut exact_tiles = ExactSum::new();
    for t in &tiles {
        exact_tiles.merge(&field_sum_exact(&f, t));
    }
    let host = host_sum(whole.iter().map(cancelling));
    assert_eq!(exact_tiles.to_words(), exact_whole.to_words(), "the merged tiles against the whole window");
    assert_eq!(exact_whole.to_words(), host.to_words(), "the whole window against the host accumulator");
    let count = |m: isize| whole.iter().filter(|c| (c[0] * 64 + c[1] * 8 + c[2]) % 5 == m).count() as f64;
    let counted = (count(0) - count(1)) * 1e16 + count(2) + count(3) + count(4);
    assert_eq!(exact_whole.value().to_bits(), counted.to_bits(), "the exact total against the counted {counted}");

    // the rounded block reduction of the same tiles, merged as the body feedback merged them
    let mut rounded_whole = ExactSum::new();
    rounded_whole.add(field_reduce(&f, &whole, ReductionOp::Add));
    let mut rounded_tiles = ExactSum::new();
    for t in &tiles {
        rounded_tiles.add(field_reduce(&f, t, ReductionOp::Add));
    }
    println!(
        "device {DEVICE}: exact {:e}; rounded block reduction: whole {:e}, two tiles {:e}",
        exact_whole.value(),
        rounded_whole.value(),
        rounded_tiles.value()
    );
    if DEVICE {
        assert_ne!(
            rounded_whole.value().to_bits(),
            rounded_tiles.value().to_bits(),
            "the rounded device reduction agrees across the partition on this field; the comparison it frames is vacuous"
        );
    }
}

#[test]
fn non_finite_and_signed_zero_terms_follow_the_host_accumulator() {
    let cases: [(&str, Vec<f64>); 6] = [
        ("a nan", vec![1.0, f64::NAN, 2.0, 3.0]),
        ("opposite infinities", vec![f64::INFINITY, 1.0, f64::NEG_INFINITY, 0.0]),
        ("one infinity", vec![f64::NEG_INFINITY, 5.0, -2.0, 1e300]),
        ("all negative zero", vec![-0.0, -0.0, -0.0, -0.0]),
        ("a positive zero among negative zeros", vec![-0.0, 0.0, -0.0, -0.0]),
        ("cancelling to zero", vec![1.5, -1.5, 2.0, -2.0]),
    ];
    for (label, terms) in cases {
        let f = field([0, 0, 0], [1, 1, terms.len() as isize], |c| terms[c[2] as usize]);
        let got = field_sum_exact(&f, &domain([0, 0, 0], [1, 1, terms.len() as isize]));
        let want = host_sum(terms.iter().copied());
        assert_eq!(got.value().to_bits(), want.value().to_bits(), "{label}: {} against {}", got.value(), want.value());
        if !want.value().is_nan() && want.value().is_finite() {
            assert_eq!(got.to_words(), want.to_words(), "{label}");
        }
    }
    let f = field([0, 0, 0], [2, 2, 2], |_| 1.0);
    let empty = field_sum_exact(&f, &domain([0, 0, 0], [2, 2, 0]));
    assert_eq!(empty, ExactSum::new(), "an empty window is the empty sum");
    assert_eq!(empty.value().to_bits(), 0.0f64.to_bits());
}

/// a deterministic stream spanning many binades and both signs, with the extremes and subnormals
fn spread(n: usize) -> Vec<f64> {
    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut out: Vec<f64> = (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let mantissa = (s >> 11) as f64 / (1u64 << 53) as f64;
            let exponent = ((s >> 3) % 1601) as i32 - 800;
            let sign = if s & 1 == 0 { 1.0 } else { -1.0 };
            sign * mantissa * 2f64.powi(exponent)
        })
        .collect();
    out.extend([f64::MAX, -f64::MAX, f64::from_bits(1), -f64::from_bits(3), f64::MIN_POSITIVE, 1e300, -1e-300]);
    out
}

#[test]
fn extreme_subnormal_and_spread_terms_equal_the_host_accumulator() {
    let terms = spread(4096 - 7);
    assert_eq!(terms.len(), 4096);
    let f = field([0, 0, 0], [16, 16, 16], |c| terms[(c[0] * 256 + c[1] * 16 + c[2]) as usize]);
    let whole = domain([0, 0, 0], [16, 16, 16]);
    let got = field_sum_exact(&f, &whole);
    let want = host_sum(whole.iter().map(|c| terms[(c[0] * 256 + c[1] * 16 + c[2]) as usize]));
    assert_eq!(got.to_words(), want.to_words());
    // every slab partition of the window merges to the same words
    for cut in [1, 5, 8, 15] {
        let mut merged = field_sum_exact(&f, &domain([0, 0, 0], [cut, 16, 16]));
        merged.merge(&field_sum_exact(&f, &domain([cut, 0, 0], [16, 16, 16])));
        assert_eq!(merged.to_words(), want.to_words(), "cut at {cut}");
    }
}

#[test]
fn a_single_precision_field_sums_its_widened_terms_exactly() {
    let terms = spread(1000);
    let alloc = domain([0, 0, -1], [1, 1, 1001]);
    let f = Field::<f32, 3, DefaultMemory>::zeros(&alloc).expect("f32 field");
    for c in alloc.iter() {
        let k = c[2];
        f.set(c, if (0..1000).contains(&k) { terms[k as usize] as f32 } else { 3.0e38 });
    }
    let got = field_sum_exact(&f, &domain([0, 0, 0], [1, 1, 1000]));
    let want = host_sum(terms.iter().take(1000).map(|&x| (x as f32) as f64));
    assert_eq!(got.to_words(), want.to_words());
    // single-precision non-finite and signed-zero terms widen to their f64 values
    let cases: [(&str, Vec<f32>); 4] = [
        ("a nan", vec![1.0, f32::NAN, 2.0]),
        ("opposite infinities", vec![f32::INFINITY, f32::NEG_INFINITY, 1.0]),
        ("one infinity", vec![f32::NEG_INFINITY, 3.0, -2.0]),
        ("all negative zero", vec![-0.0, -0.0, -0.0]),
    ];
    for (label, terms) in cases {
        let n = terms.len() as isize;
        let alloc = domain([0, 0, -1], [1, 1, n + 1]);
        let f = Field::<f32, 3, DefaultMemory>::zeros(&alloc).expect("f32 field");
        for c in alloc.iter() {
            f.set(c, if (0..n).contains(&c[2]) { terms[c[2] as usize] } else { 5.0 });
        }
        let got = field_sum_exact(&f, &domain([0, 0, 0], [1, 1, n]));
        let want = host_sum(terms.iter().map(|&x| x as f64));
        assert_eq!(got.value().to_bits(), want.value().to_bits(), "{label}");
    }
}

#[test]
fn two_million_halves_carry_through_every_digit() {
    let f = field([0, 0, 0], [128, 128, 128], |_| 0.5);
    let got = field_sum_exact(&f, &domain([0, 0, 0], [128, 128, 128]));
    assert_eq!(got.value(), (1u64 << 20) as f64);
    assert_eq!(got.to_words(), host_sum(std::iter::repeat_n(0.5, 1 << 21)).to_words());
}

#[test]
fn the_exact_sum_and_the_block_reduction_are_timed() {
    let f = field([0, 0, 0], [128, 128, 128], |c| ((c[0] * 131 + c[1] * 17 + c[2]) % 97) as f64 * 1.0e-3 - 0.04);
    let window = domain([0, 0, 0], [128, 128, 128]);
    let exact = field_sum_exact(&f, &window);
    let _ = field_reduce(&f, &window, ReductionOp::Add);
    let reps = 20;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        assert_eq!(field_sum_exact(&f, &window), exact);
    }
    let exact_time = start.elapsed().as_secs_f64() / reps as f64;
    let start = std::time::Instant::now();
    for _ in 0..reps {
        std::hint::black_box(field_reduce(&f, &window, ReductionOp::Add));
    }
    let rounded_time = start.elapsed().as_secs_f64() / reps as f64;
    println!(
        "device {DEVICE}: 128^3 window, exact sum {:.3} ms, rounded block reduction {:.3} ms, ratio {:.2}",
        exact_time * 1e3,
        rounded_time * 1e3,
        exact_time / rounded_time
    );
}
