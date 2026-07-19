// =============================================================================
// layout.rs
//
// the MEMORY-LAYOUT primitive: `Layout<D>` is the single type that knows a
// buffer's memory shape — its per-axis origin (`lo`), allocated `extent`, and
// the `strides` that map a coordinate to a flat offset. one definition of the
// physical-x-fastest stride formula, shared by every view (`Domain`, the runtime
// `CpuField`/`DeviceView`, symbi-grid's `View`). a single definition guarantees
// kernel writes and host reads resolve to the same addresses.
//
// the law (proven in tests): `Layout::at(coord)` is the affine offset
// `sum_a (coord[a] - lo[a]) * strides[a]`, byte-identical to `Domain::flat_index`
// and to the index the emitted kernels compute. the contiguous-axis invariant
// (`strides[0] == 1`, physical-x-fastest) holds by construction.
//
// usage:
//  let lay = Layout::from_domain(&allocated);
//  let off = lay.at([ii, jj, kk]);   // == the manual __idx_cell formula
// =============================================================================

use crate::domain::Domain;

/// THE physical-x-fastest stride prefix product — written ONCE. `strides[0] = 1`,
/// `strides[d] = prod(extent[0..d])`. axis 0 is the fastest-varying in memory, so
/// adjacent x-coords (and CUDA `threadIdx.x` lanes) hit adjacent bytes. every view
/// in the codebase derives its strides from THIS; they cannot drift apart.
///
/// generic over the integer type so the geometric layer (`usize`) and the kernel
/// ABI (`i32`) share the formula. `extent` and `out` must be the same length.
pub fn strides_from_extent<T>(extent: &[T], out: &mut [T])
where
    T: Copy + core::ops::Mul<Output = T> + From<u8>,
{
    debug_assert_eq!(extent.len(), out.len(), "strides_from_extent: length mismatch");
    if extent.is_empty() {
        return;
    }
    out[0] = T::from(1u8);
    for d in 1..extent.len() {
        out[d] = out[d - 1] * extent[d - 1];
    }
}

/// THE contiguous axis: the one `strides_from_extent` gives a stride of 1. every consumer that needs
/// to know "which axis is adjacent in memory" — the CPU loop nest, the unit-stride index term, the
/// vectorized inner loop, the GPU warp axis — must read it from HERE and not assume a fixed rank.
/// assuming the wrong axis is silent: the code stays correct (the offset formula is still affine) and
/// only the memory-access pattern degrades, which no correctness test can see.
pub const CONTIGUOUS_AXIS: usize = 0;

/// the INVERSE of the flat index: recover a coordinate from a canonical iteration index, with
/// [`CONTIGUOUS_AXIS`] varying FASTEST. this is the map every flat driver (`0..total` parallel
/// sweeps, block covers, the IR interpreter) needs; centralizing it here keeps every driver off
/// the trap of walking the SLOWEST axis fastest, which strides the hot loop by
/// `extent[0]*extent[1]` on every cell.
///
/// the defining law, pinned by `unflatten_inverts_the_flat_offset`: for a buffer whose extent equals
/// the iteration extent and whose origin is zero, `flat_offset(unflatten(f)) == f`.
#[inline]
pub fn unflatten(mut flat: usize, extent: &[usize], out: &mut [usize]) {
    debug_assert_eq!(extent.len(), out.len(), "unflatten: length mismatch");
    for a in 0..extent.len() {
        out[a] = flat % extent[a];
        flat /= extent[a];
    }
}

/// the canonical loop-nest order: OUTERMOST axis first, innermost last. the innermost axis is always
/// [`CONTIGUOUS_AXIS`], so the hot loop advances the flat offset by exactly one element per iteration
/// — the precondition both for cache-line reuse and for a compiler to prove a unit-stride access and
/// vectorize the body.
#[inline]
pub fn nest_order(ndim: usize) -> impl DoubleEndedIterator<Item = usize> + Clone {
    (0..ndim).rev()
}

/// the affine flat offset under physical-x-fastest strides: `sum_a (coord[a] - lo[a]) * strides[a]`.
/// THE single value-path index formula — `Layout::at`, `Domain::flat_index`,
/// and the grid `View`/`ViewMut` accessors all route here, so the storage convention lives in
/// exactly one place. a coordinate left of its origin (`coord < lo`) is a caller bug (out of the
/// buffer); the subtraction wraps in release like any index past the end.
#[inline]
pub fn flat_offset<const D: usize>(coord: [isize; D], lo: [isize; D], strides: [usize; D]) -> usize {
    let mut idx = 0usize;
    for a in 0..D {
        idx += (coord[a] - lo[a]) as usize * strides[a];
    }
    idx
}

/// a buffer's memory shape: origin, extent, and the strides derived from them.
/// the canonical "what address does this coordinate live at" type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout<const D: usize> {
    pub lo: [i32; D],
    pub strides: [i32; D],
    pub extent: [u32; D],
}

impl<const D: usize> Layout<D> {
    /// build a layout from origin + allocated extent (strides derived canonically).
    pub fn new(lo: [i32; D], extent: [u32; D]) -> Self {
        let ext_i32: [i32; D] = std::array::from_fn(|a| extent[a] as i32);
        let mut strides = [0i32; D];
        strides_from_extent(&ext_i32, &mut strides);
        // contiguous-axis invariant: physical-x-fastest => axis 0 has stride 1.
        debug_assert!(D == 0 || strides[0] == 1, "Layout: contiguous axis 0 must have stride 1");
        Layout { lo, strides, extent }
    }

    /// the layout of a `Domain`'s allocated cells (origin + size per axis).
    pub fn from_domain(d: &Domain<D>) -> Self {
        Layout::new(
            std::array::from_fn(|a| d.spaces[a].lo as i32),
            std::array::from_fn(|a| d.spaces[a].size() as u32),
        )
    }

    /// the affine flat offset of `coord`: `sum_a (coord[a] - lo[a]) * strides[a]`.
    /// routes through the one [`flat_offset`] formula.
    #[inline]
    pub fn at(&self, coord: [i32; D]) -> usize {
        let coord: [isize; D] = std::array::from_fn(|a| coord[a] as isize);
        let lo: [isize; D] = std::array::from_fn(|a| self.lo[a] as isize);
        let strides: [usize; D] = std::array::from_fn(|a| self.strides[a] as usize);
        flat_offset(coord, lo, strides)
    }

    /// total allocated cells.
    pub fn len(&self) -> usize {
        self.extent.iter().map(|&e| e as usize).product()
    }

    pub fn is_empty(&self) -> bool {
        self.extent.iter().any(|&e| e == 0)
    }

    /// the contiguous (unit-stride) axis — always [`CONTIGUOUS_AXIS`] under physical-x-fastest. the
    /// invariant the lane primitive (`Gv<N>`) will load `N` consecutive lanes along.
    pub const fn contiguous_axis(&self) -> usize {
        CONTIGUOUS_AXIS
    }
}

// =============================================================================
// laws (axioms) — the layout primitive against its definition + the existing
// index paths it unifies.
// =============================================================================
#[cfg(test)]
mod laws {
    use super::*;
    use crate::domain::{domain, index};

    struct Rng(u64);
    impl Rng {
        fn bits(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn in_range(&mut self, lo: isize, hi: isize) -> isize {
            lo + (self.bits() % ((hi - lo) as u64)) as isize
        }
    }

    // AXIOM: the canonical formula is type-agnostic — usize and i32 give the same
    // stride sequence for the same extent (so geometry and the kernel ABI agree).
    #[test]
    fn strides_formula_is_type_agnostic() {
        let mut rng = Rng(0x57AB_1E5);
        for _ in 0..2000 {
            let ext: [usize; 3] = std::array::from_fn(|_| rng.in_range(1, 12) as usize);
            let mut su = [0usize; 3];
            strides_from_extent(&ext, &mut su);
            let exti: [i32; 3] = std::array::from_fn(|a| ext[a] as i32);
            let mut si = [0i32; 3];
            strides_from_extent(&exti, &mut si);
            for a in 0..3 {
                assert_eq!(su[a] as i32, si[a]);
            }
            // contiguous-axis invariant.
            assert_eq!(su[0], 1);
        }
    }

    // AXIOM (the index-equivalence GATE): Layout::at reproduces Domain::flat_index
    // exactly — the View-based offset == the manual offset the kernels compute.
    #[test]
    fn at_equals_domain_flat_index() {
        let mut rng = Rng(0x1DEA_0FF);
        for _ in 0..3000 {
            let dom = Domain::new(std::array::from_fn(|a| {
                let lo = rng.in_range(-4, 4);
                let size = rng.in_range(1, 9);
                crate::domain::Space { name: ["i", "j", "k"][a], lo, hi: lo + size }
            }));
            let lay = Layout::from_domain(&dom);
            for p in dom.iter() {
                let coord = [p[0] as i32, p[1] as i32, p[2] as i32];
                assert_eq!(lay.at(coord), dom.flat_index(p), "Layout::at != Domain::flat_index at {p:?}");
            }
        }
    }

    // AXIOM: co-location — the layout is a pure function of the domain (two views
    // over the same allocated domain share one Layout).
    #[test]
    fn layout_is_a_pure_function_of_the_domain() {
        let d = domain([index("i").over((-2, 30)), index("j").over(20), index("k").over((1, 9))]);
        assert_eq!(Layout::from_domain(&d), Layout::from_domain(&d));
        assert_eq!(Layout::from_domain(&d).contiguous_axis(), 0);
        assert_eq!(Layout::from_domain(&d).len(), d.volume());
    }

    // ---- the traversal laws ---------------------------------------------------------------
    // the layout owns WHERE a cell lives (`strides_from_extent`, `flat_offset`) AND, from here
    // down, the ORDER a sweep visits cells (`unflatten`, `nest_order`). every driver and every
    // code emitter derives from these; the tests below are what makes "derives" enforceable.

    /// the definition: `strides_from_extent` gives the contiguous axis a stride of exactly 1.
    #[test]
    fn contiguous_axis_has_unit_stride() {
        for extent in [vec![7usize], vec![7, 5], vec![7, 5, 3], vec![2, 128, 1]] {
            let mut s = vec![0usize; extent.len()];
            strides_from_extent(&extent, &mut s);
            assert_eq!(
                s[CONTIGUOUS_AXIS], 1,
                "CONTIGUOUS_AXIS={CONTIGUOUS_AXIS} is not unit-stride for extent {extent:?}: {s:?}",
            );
        }
    }

    /// `unflatten` is the exact inverse of the layout's flat index. a driver that unflattens the
    /// OTHER way round (slowest axis fastest) still visits every cell exactly once — so no
    /// correctness test catches it — but it strides the hot loop by `extent[0]*extent[1]`. this law
    /// is what surfaces that as a compile-time-visible bug; without it the mistake is a silent 2x.
    #[test]
    fn unflatten_inverts_the_flat_offset() {
        for extent in [vec![7usize], vec![7, 5], vec![4, 3, 2], vec![2, 128, 1]] {
            let nd = extent.len();
            let mut strides = vec![0usize; nd];
            strides_from_extent(&extent, &mut strides);
            let total: usize = extent.iter().product();
            let mut coord = vec![0usize; nd];
            for flat in 0..total {
                unflatten(flat, &extent, &mut coord);
                // flat_offset over a zero-origin buffer must reproduce the iteration index itself.
                let off: usize = (0..nd).map(|a| coord[a] * strides[a]).sum();
                assert_eq!(off, flat, "unflatten != inverse at {flat} of {extent:?}");
            }
        }
    }

    /// consecutive iteration indices differ by exactly one element in memory. this is the property
    /// a compiler needs to prove a unit-stride access and vectorize; it is also the reason the hot
    /// loop stays on one cache line.
    #[test]
    fn consecutive_iteration_indices_are_adjacent_in_memory() {
        let extent = vec![4usize, 3, 2];
        let mut strides = vec![0usize; 3];
        strides_from_extent(&extent, &mut strides);
        let total: usize = extent.iter().product();
        let (mut a, mut b) = (vec![0usize; 3], vec![0usize; 3]);
        for flat in 0..total - 1 {
            unflatten(flat, &extent, &mut a);
            unflatten(flat + 1, &extent, &mut b);
            let off = |c: &[usize]| -> usize { (0..3).map(|i| c[i] * strides[i]).sum() };
            assert_eq!(
                off(&b) - off(&a),
                strides[CONTIGUOUS_AXIS],
                "consecutive indices {flat},{} are not adjacent in memory",
                flat + 1,
            );
        }
    }

    /// the loop nest's INNERMOST axis is the contiguous one. an emitter that nests the contiguous
    /// axis outermost produces a correct kernel whose hot loop strides across memory — the exact
    /// shape that denies vectorization and thrashes the line.
    #[test]
    fn nest_order_puts_the_contiguous_axis_innermost() {
        for ndim in 1..=3 {
            let order: Vec<usize> = nest_order(ndim).collect();
            assert_eq!(order.len(), ndim);
            assert_eq!(
                *order.last().unwrap(),
                CONTIGUOUS_AXIS,
                "nest_order({ndim}) innermost axis must be CONTIGUOUS_AXIS, got {order:?}",
            );
            let mut seen: Vec<usize> = order.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..ndim).collect::<Vec<_>>(), "nest_order must be a permutation");
        }
    }

    /// walking `nest_order` as a real nest visits cells in exactly `unflatten` order — so the two
    /// halves of the traversal contract (a nested emitter, a flat driver) cannot disagree.
    #[test]
    fn nest_order_walk_agrees_with_unflatten() {
        let extent = [4usize, 3, 2];
        let order: Vec<usize> = nest_order(3).collect(); // [2, 1, 0]
        let mut visited = Vec::new();
        for o in 0..extent[order[0]] {
            for m in 0..extent[order[1]] {
                for i in 0..extent[order[2]] {
                    let mut c = [0usize; 3];
                    c[order[0]] = o;
                    c[order[1]] = m;
                    c[order[2]] = i;
                    visited.push(c);
                }
            }
        }
        let total: usize = extent.iter().product();
        assert_eq!(visited.len(), total);
        let mut c = vec![0usize; 3];
        for (flat, v) in visited.iter().enumerate() {
            unflatten(flat, &extent, &mut c);
            assert_eq!(&c[..], &v[..], "nest walk diverges from unflatten at {flat}");
        }
    }
}
