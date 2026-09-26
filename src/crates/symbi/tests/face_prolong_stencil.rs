// =============================================================================
// face_prolong_stencil.rs
//
// the parent faces the coarse-fine face prolongation reads, derived from its
// stencil by `face_prolong_reads`, are exactly the faces the kernel loads: with
// every face component allocated to its derived box the kernel writes the same
// bits as on fields many rows wider, and removing one row from any side of any
// component's box either leaves a load outside the allocation or changes the
// written value. the kernel loads every stencil value of a thread whether or
// not its parity uses it, so the box is the allocation the parent face stencil
// needs. 2D and 3D, every face axis, fine regions with both parities on the
// normal axis and negative fine indices.
//
// run: cargo test -p symbi --test face_prolong_stencil
// =============================================================================
use std::panic::{AssertUnwindSafe, catch_unwind};

use symbi::sim::refinement::transfer::{face_prolong_reads, prolong_face_field};
use symbi::sim::state::axis_name;
use symbi_algebra::{Domain, Space};
use symbi_grid::Field;
use symbi_xpu::HostMemory;

const ALPHA: f64 = 0.375;

fn domain<const D: usize>(lo: [isize; D], hi: [isize; D]) -> Domain<D> {
    Domain::new(std::array::from_fn(|a| Space { name: axis_name(a), lo: lo[a], hi: hi[a] }))
}

/// a smooth, strictly increasing value of a parent face of component `comp` at time level
/// `level`, so every limited slope is nonzero and every read feeds the written value
fn value<const D: usize>(comp: usize, level: usize, c: [isize; D]) -> f64 {
    let rates = [0.051, 0.073, 0.037];
    let x: f64 = (0..D).map(|a| rates[a] * c[a] as f64).sum();
    (x + 0.2 * comp as f64 + 0.11 * level as f64).exp()
}

/// the face components at both time levels over `doms[comp]`
fn fields<const D: usize>(doms: &[Domain<D>]) -> (Vec<Field<f64, D, HostMemory>>, Vec<Field<f64, D, HostMemory>>) {
    let make = |level: usize| -> Vec<Field<f64, D, HostMemory>> {
        doms.iter()
            .enumerate()
            .map(|(comp, dom)| {
                let f = Field::zeros(dom).expect("field");
                for c in dom.iter() {
                    f.view_mut().set(c, value(comp, level, c));
                }
                f
            })
            .collect()
    };
    (make(0), make(1))
}

/// the written fine faces of `region`, or `None` when a load fell outside an allocation
fn prolong<const D: usize>(axis: usize, region: &Domain<D>, doms: &[Domain<D>]) -> Option<Vec<u64>> {
    let (old, new) = fields(doms);
    let dst = Field::<f64, D, HostMemory>::zeros(region).expect("dst");
    let ran = catch_unwind(AssertUnwindSafe(|| prolong_face_field(axis, &old, &new, &dst, region, ALPHA)));
    ran.ok().map(|_| region.iter().map(|c| dst.view().at(c).to_bits()).collect())
}

fn widen<const D: usize>(d: &Domain<D>, by: isize) -> Domain<D> {
    domain(std::array::from_fn(|a| d.spaces[a].lo - by), std::array::from_fn(|a| d.spaces[a].hi + by))
}

fn check<const D: usize>(axis: usize, region: Domain<D>, tight: bool) {
    let reads: Vec<Domain<D>> = face_prolong_reads(axis, &region).into_iter().map(|r| r.expect("a read box")).collect();
    let wide: Vec<Domain<D>> = reads.iter().map(|r| widen(r, 4)).collect();
    let reference = prolong(axis, &region, &wide).expect("the kernel runs on wide fields");
    assert!(reference.iter().all(|b| f64::from_bits(*b).is_finite()), "axis {axis}: a non-finite reference");
    let exact = prolong(axis, &region, &reads).unwrap_or_else(|| panic!("axis {axis}: a load fell outside the derived boxes"));
    assert_eq!(exact, reference, "axis {axis}: the derived boxes change the written faces");
    if !tight {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    for comp in 0..D {
        for ax in 0..D {
            for side in 0..2 {
                let mut shrunk = reads.clone();
                let s = &mut shrunk[comp].spaces[ax];
                if side == 0 { s.lo += 1 } else { s.hi -= 1 }
                let got = prolong(axis, &region, &shrunk);
                assert!(
                    got.as_ref() != Some(&reference),
                    "axis {axis}: component {comp} reads no face on side {side} of axis {ax}; the box is wider than the stencil"
                );
            }
        }
    }
    std::panic::set_hook(previous);
}

#[test]
fn the_derived_boxes_are_exactly_the_2d_face_prolongation_reads() {
    for axis in 0..2 {
        // every read feeds the value: the normal range runs even to even
        let tight: [isize; 2] = std::array::from_fn(|a| if a == axis { 24 } else { 23 });
        let tight_hi: [isize; 2] = std::array::from_fn(|a| if a == axis { 30 } else { 26 });
        check::<2>(axis, domain(tight, tight_hi), true);
        // both parities at the ends of the normal range, negative indices across
        let lo: [isize; 2] = std::array::from_fn(|a| if a == axis { -3 } else { -2 });
        let hi: [isize; 2] = std::array::from_fn(|a| if a == axis { 4 } else { 1 });
        check::<2>(axis, domain(lo, hi), false);
    }
}

#[test]
fn the_derived_boxes_are_exactly_the_3d_face_prolongation_reads() {
    for axis in 0..3 {
        let lo: [isize; 3] = std::array::from_fn(|a| if a == axis { 10 } else { 7 });
        let hi: [isize; 3] = std::array::from_fn(|a| if a == axis { 14 } else { 10 });
        check::<3>(axis, domain(lo, hi), true);
        let lo: [isize; 3] = std::array::from_fn(|a| if a == axis { -1 } else { -3 });
        let hi: [isize; 3] = std::array::from_fn(|a| if a == axis { 4 } else { 0 });
        check::<3>(axis, domain(lo, hi), false);
    }
}
