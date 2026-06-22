// =============================================================================
// layout.rs
//
// the MEMORY-LAYOUT primitive: `Layout<D>` is the single type that knows a
// buffer's memory shape — its per-axis origin (`lo`), allocated `extent`, and
// the `strides` that map a coordinate to a flat offset. one definition of the
// physical-x-fastest stride formula, shared by every view (`Domain`, the runtime
// `CpuField`/`DeviceView`, symbi-grid's `View`) instead of the three hand-synced
// copies that previously had to "agree or kernel writes and host reads land at
// different addresses" (the comment in symbi-aot that this primitive retires).
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

/// the affine flat offset under physical-x-fastest strides: `sum_a (coord[a] - lo[a]) * strides[a]`.
/// THE single value-path index formula (docs/design/38 P3a) — `Layout::at`, `Domain::flat_index`,
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
    /// routes through the one [`flat_offset`] formula (docs/design/38 P3a).
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

    /// the contiguous (unit-stride) axis — always 0 under physical-x-fastest. the
    /// invariant the lane primitive (`Gv<N>`) will load `N` consecutive lanes along.
    pub const fn contiguous_axis(&self) -> usize {
        0
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
}
