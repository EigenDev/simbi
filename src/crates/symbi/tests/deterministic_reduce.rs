// =============================================================================
// deterministic_reduce.rs
//
// the fixed-order reduction contract: the host parallel Add reduce combines its
// per-slab partials in slab order, so the result is bit-reproducible run to run
// and across thread counts for a fixed domain shape. the body-feedback sums
// feed the body equations of motion, so a join-order-dependent combine was a
// run-to-run trajectory nondeterminism at production sizes. the field mixes
// magnitudes (1e16 alternating with 1.0) so any reassociation of the combine
// visibly changes the sum — a vacuous all-ones field would pass under any
// order.
// =============================================================================

use symbi::regimes::substrate_gpu::field_reduce;
use symbi_algebra::{Domain, Space};
use symbi_grid::Field;
use symbi_ir::emit::ReductionOp;
use symbi_xpu::HostMemory;

const NX: isize = 512;
const NY: isize = 512; // 512 x 512 = 2^18 cells: well past the parallel threshold

fn build() -> (Field<f64, 2, HostMemory>, Domain<2>) {
    let domain = Domain::new([
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
    ]);
    let field = Field::<f64, 2, HostMemory>::zeros(&domain).expect("field");
    for c in domain.iter() {
        // adversarial magnitudes: the running sum loses the small addend whenever a
        // large one is present, so the grouping of partials is visible in the result.
        let v = if (c[0] + c[1]) % 7 == 0 { 1.0e16 } else { 1.0 };
        field
            .view_mut()
            .set(c, if (c[0] * 3 + c[1]) % 11 == 0 { -v } else { v });
    }
    (field, domain)
}

#[test]
fn parallel_add_reduce_is_bit_reproducible() {
    let (field, domain) = build();
    // run to run: the same call must produce the same bits, repeatedly (a
    // work-stealing-dependent combine tree fails this within a few repeats).
    let first = field_reduce(&field, &domain, ReductionOp::Add);
    for rep in 0..8 {
        let again = field_reduce(&field, &domain, ReductionOp::Add);
        assert!(
            again.to_bits() == first.to_bits(),
            "Add reduce differs across runs (rep {rep}): {again:e} vs {first:e}"
        );
    }

    // against the independently-computed fixed-order reference: per-slab partials
    // over the outermost axis in storage order, folded in slab order — the exact
    // algorithm the contract promises.
    let mut expect = 0.0_f64;
    for ii in 0..NX {
        let mut slab = 0.0_f64;
        for jj in 0..NY {
            slab += *field.view().at([ii, jj]);
        }
        expect += slab;
    }
    assert!(
        first.to_bits() == expect.to_bits(),
        "Add reduce does not match the fixed slab-order reference: {first:e} vs {expect:e}"
    );
}

#[test]
fn min_max_reduce_stay_exact() {
    let (field, domain) = build();
    let mx = field_reduce(&field, &domain, ReductionOp::Max);
    let mn = field_reduce(&field, &domain, ReductionOp::Min);
    assert_eq!(mx, 1.0e16);
    assert_eq!(mn, -1.0e16);
}
