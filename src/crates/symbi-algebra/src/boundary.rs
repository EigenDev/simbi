// =============================================================================
// boundary.rs
//
// boundary index maps for ghost cell filling in structured grids.
// pure integer arithmetic — no runtime dependencies.
//
// each map takes a ghost cell index and the interior bounds [lo, hi)
// and returns the source cell index to copy from. for multi-dimensional
// grids, apply per axis via map_nd().
//
// usage:
//   let src = clamp(ghost_idx, lo, hi);
//   let src = mirror(ghost_idx, lo, hi);
//   let src = periodic(ghost_idx, lo, hi);
//   let src_nd = map_nd(idx, lo, hi, [clamp, mirror]);
// =============================================================================

/// clamp (transmissive/outflow): ghost reads nearest interior cell.
/// extrapolates zero-gradient: the ghost cell gets the boundary value.
pub fn clamp(ii: isize, lo: isize, hi: isize) -> isize {
    if ii < lo { lo } else if ii >= hi { hi - 1 } else { ii }
}

/// mirror (reflecting): ghost reads the cell reflected across the boundary.
/// ghost[lo-1] reads interior[lo], ghost[lo-2] reads interior[lo+1], etc.
pub fn mirror(ii: isize, lo: isize, hi: isize) -> isize {
    if ii < lo {
        2 * lo - ii - 1
    } else if ii >= hi {
        2 * hi - ii - 1
    } else {
        ii
    }
}

/// periodic (wrap): ghost reads from the opposite end of the domain.
/// ghost[lo-1] reads interior[hi-1], ghost[hi] reads interior[lo], etc.
pub fn periodic(ii: isize, lo: isize, hi: isize) -> isize {
    let n = hi - lo;
    lo + ((ii - lo) % n + n) % n
}

/// type alias for a boundary index map function.
pub type IndexMap = fn(isize, isize, isize) -> isize;

/// apply a per-axis index map to a D-dimensional index.
/// maps[dd] is applied to idx[dd] with bounds [lo[dd], hi[dd]).
pub fn map_nd<const D: usize>(
    idx: [isize; D],
    lo: [isize; D],
    hi: [isize; D],
    maps: [IndexMap; D],
) -> [isize; D] {
    std::array::from_fn(|dd| maps[dd](idx[dd], lo[dd], hi[dd]))
}

// ============================================================
// tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- clamp ----

    #[test]
    fn test_clamp_interior() {
        // interior indices pass through
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(0, 0, 10), 0);
        assert_eq!(clamp(9, 0, 10), 9);
    }

    #[test]
    fn test_clamp_lo_ghost() {
        // ghost below lo clamps to lo
        assert_eq!(clamp(-1, 0, 10), 0);
        assert_eq!(clamp(-2, 0, 10), 0);
        assert_eq!(clamp(-100, 0, 10), 0);
    }

    #[test]
    fn test_clamp_hi_ghost() {
        // ghost at/above hi clamps to hi-1
        assert_eq!(clamp(10, 0, 10), 9);
        assert_eq!(clamp(11, 0, 10), 9);
        assert_eq!(clamp(100, 0, 10), 9);
    }

    #[test]
    fn test_clamp_nonzero_origin() {
        assert_eq!(clamp(3, 5, 15), 5);
        assert_eq!(clamp(7, 5, 15), 7);
        assert_eq!(clamp(16, 5, 15), 14);
    }

    // ---- mirror ----

    #[test]
    fn test_mirror_interior() {
        assert_eq!(mirror(5, 0, 10), 5);
    }

    #[test]
    fn test_mirror_lo_ghost() {
        // ghost[lo-1] -> interior[lo], ghost[lo-2] -> interior[lo+1]
        assert_eq!(mirror(-1, 0, 10), 0);
        assert_eq!(mirror(-2, 0, 10), 1);
        assert_eq!(mirror(-3, 0, 10), 2);
    }

    #[test]
    fn test_mirror_hi_ghost() {
        // ghost[hi] -> interior[hi-1], ghost[hi+1] -> interior[hi-2]
        assert_eq!(mirror(10, 0, 10), 9);
        assert_eq!(mirror(11, 0, 10), 8);
        assert_eq!(mirror(12, 0, 10), 7);
    }

    #[test]
    fn test_mirror_symmetric() {
        // mirror is symmetric: distance from boundary is preserved
        let (lo, hi) = (0, 10);
        assert_eq!(mirror(lo - 1, lo, hi), lo);
        assert_eq!(mirror(hi, lo, hi), hi - 1);
    }

    #[test]
    fn test_mirror_nonzero_origin() {
        // interior = [5, 15)
        assert_eq!(mirror(4, 5, 15), 5);   // 2*5 - 4 - 1 = 5
        assert_eq!(mirror(3, 5, 15), 6);   // 2*5 - 3 - 1 = 6
        assert_eq!(mirror(15, 5, 15), 14); // 2*15 - 15 - 1 = 14
        assert_eq!(mirror(16, 5, 15), 13); // 2*15 - 16 - 1 = 13
    }

    // ---- periodic ----

    #[test]
    fn test_periodic_interior() {
        assert_eq!(periodic(5, 0, 10), 5);
    }

    #[test]
    fn test_periodic_lo_ghost() {
        // wraps to high end
        assert_eq!(periodic(-1, 0, 10), 9);
        assert_eq!(periodic(-2, 0, 10), 8);
        assert_eq!(periodic(-10, 0, 10), 0);
    }

    #[test]
    fn test_periodic_hi_ghost() {
        // wraps to low end
        assert_eq!(periodic(10, 0, 10), 0);
        assert_eq!(periodic(11, 0, 10), 1);
        assert_eq!(periodic(19, 0, 10), 9);
    }

    #[test]
    fn test_periodic_multi_wrap() {
        // wraps multiple periods
        assert_eq!(periodic(20, 0, 10), 0);
        assert_eq!(periodic(-11, 0, 10), 9);
    }

    #[test]
    fn test_periodic_nonzero_origin() {
        assert_eq!(periodic(4, 5, 15), 14);
        assert_eq!(periodic(15, 5, 15), 5);
        assert_eq!(periodic(3, 5, 15), 13);
    }

    // ---- map_nd ----

    #[test]
    fn test_map_nd_2d() {
        let idx = [-1_isize, 12];
        let lo = [0, 0];
        let hi = [10, 10];
        let result = map_nd(idx, lo, hi, [clamp, periodic]);
        assert_eq!(result[0], 0);  // clamp: -1 -> 0
        assert_eq!(result[1], 2);  // periodic: 12 -> 2
    }

    #[test]
    fn test_map_nd_3d_mixed() {
        let idx = [-2_isize, 5, 20];
        let lo = [0, 0, 0];
        let hi = [10, 10, 10];
        let result = map_nd(idx, lo, hi, [mirror, clamp, periodic]);
        assert_eq!(result[0], 1);  // mirror: -2 -> 1
        assert_eq!(result[1], 5);  // clamp: 5 -> 5 (interior)
        assert_eq!(result[2], 0);  // periodic: 20 -> 0
    }

    #[test]
    fn test_map_nd_1d() {
        let result = map_nd([-1], [0], [10], [clamp]);
        assert_eq!(result[0], 0);
    }

    // ---- edge cases ----

    #[test]
    fn test_clamp_single_cell() {
        // domain with single cell: lo=0, hi=1
        assert_eq!(clamp(-1, 0, 1), 0);
        assert_eq!(clamp(0, 0, 1), 0);
        assert_eq!(clamp(1, 0, 1), 0);
    }

    #[test]
    fn test_periodic_single_cell() {
        assert_eq!(periodic(-1, 0, 1), 0);
        assert_eq!(periodic(0, 0, 1), 0);
        assert_eq!(periodic(1, 0, 1), 0);
    }

    #[test]
    fn test_mirror_single_cell() {
        assert_eq!(mirror(-1, 0, 1), 0);
        assert_eq!(mirror(0, 0, 1), 0);
        assert_eq!(mirror(1, 0, 1), 0);
    }
}
