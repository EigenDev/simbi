// =============================================================================
// reduce_visits_every_cell_once.rs
//
// the host fold (`field_reduce`) takes a PARALLEL path above `PAR_THRESHOLD` cells and a serial one
// below. the parallel path slabs the outermost axis across rayon and walks each slab in STORAGE order
// (CONTIGUOUS_AXIS innermost) so the fold streams memory contiguously; walking in index order would stride by `extent[0]` per cell.
//
// that hand-rolled slab walk is exactly the kind of index arithmetic that is correct-looking and
// wrong: skip a cell or double-count one, and a Max reduce still returns a plausible number. so the
// gate is an ADD over values chosen to be exactly representable — the sum pins the MULTISET of
// visited cells; dropping or double-counting a cell shifts the sum even where a max would still look plausible.
//
// what this CANNOT catch, by construction: a transposed or otherwise reordered walk. a reorder is a
// permutation of the same multiset, and every reduce here is order-independent (min/max exactly;
// add/mul over exact integers). so a walk that strides memory the wrong way is invisible to any
// reduce test — which is precisely why the visit ORDER is derived from `symbi_algebra::nest_order`
// and pinned by the layout laws.
//
// run: cargo test -p symbi-exec --test reduce_visits_every_cell_once
// =============================================================================

use symbi_algebra::{Domain, domain, index};
use symbi_exec::engine::field_reduce;
use symbi_grid::Field;
use symbi_ir::emit::ReductionOp;
use symbi_xpu::HostMemory;

// a 3D box with pairwise-distinct extents, above PAR_THRESHOLD (1 << 16 = 65536) so the parallel
// slab path is taken, and with a >1 outermost extent so the slab split actually engages.
fn big_box() -> Domain<3> {
    // 67 * 23 * 43 = 66_263 cells
    domain([
        index("i").over(67),
        index("j").over(23),
        index("k").over(43),
    ])
}

/// fill each cell with its own storage offset, as an f64. every value is an exact integer well under
/// 2^53, so the ADD fold is associative to the bit and the expected sum is closed-form.
fn seeded_field(d: &Domain<3>) -> Field<f64, 3, HostMemory> {
    let f = Field::<f64, 3, HostMemory>::zeros(d).expect("alloc");
    for c in d.iter() {
        f.view_mut().set(c, d.flat_index(c) as f64);
    }
    f
}

#[test]
fn parallel_reduce_visits_every_cell_exactly_once() {
    let d = big_box();
    assert!(
        d.volume() >= 1 << 16,
        "domain must exceed PAR_THRESHOLD to take the slab path"
    );
    let f = seeded_field(&d);
    let n = d.volume() as u64;

    // ADD pins the multiset of visited cells: sum of 0..n-1. a skipped or repeated cell shifts it.
    let want_sum = (n * (n - 1) / 2) as f64;
    let got_sum = field_reduce(&f, &d, ReductionOp::Add);
    assert_eq!(
        got_sum.to_bits(),
        want_sum.to_bits(),
        "parallel ADD reduce != sum(0..{n}): got {got_sum}, want {want_sum} \
         (a cell was skipped or double-counted; a REORDERED walk is invisible here by design)",
    );

    // the extremes: a weak check (any permutation satisfies them), kept only to pin the op dispatch.
    assert_eq!(
        field_reduce(&f, &d, ReductionOp::Max),
        (n - 1) as f64,
        "Max"
    );
    assert_eq!(field_reduce(&f, &d, ReductionOp::Min), 0.0, "Min");
}

#[test]
fn parallel_and_serial_folds_agree() {
    // the same field reduced over a SUB-domain small enough to take the serial path, and over the
    // full domain (parallel). both must agree with an independent fold, so the two paths cannot
    // drift apart the way the traversal helpers did.
    let d = big_box();
    let f = seeded_field(&d);

    // a small window: 8 * 8 * 8 = 512 cells, far below PAR_THRESHOLD -> serial path.
    let small = domain([index("i").over(8), index("j").over(8), index("k").over(8)]);
    assert!(small.volume() < 1 << 16);
    let mut want = 0.0f64;
    for c in small.iter() {
        want += *f.view().at(c);
    }
    let got = field_reduce(&f, &small, ReductionOp::Add);
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "serial ADD reduce disagrees with a direct fold"
    );

    // the parallel path over the full box, checked against a direct serial fold of the same box.
    let mut want_full = 0.0f64;
    for c in d.iter() {
        want_full += *f.view().at(c);
    }
    let got_full = field_reduce(&f, &d, ReductionOp::Add);
    assert_eq!(
        got_full.to_bits(),
        want_full.to_bits(),
        "parallel fold disagrees with a direct serial fold over the same domain",
    );
}
