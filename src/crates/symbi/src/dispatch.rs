// =============================================================================
// dispatch.rs
//
// tiled parallel dispatch over domains. provides the DomainForEach trait
// used by #[symbi::kernel(coord)] generated code.
//
// on CPU: decomposes the domain into rectangular tiles, iterates tiles in
// parallel (rayon), iterates cells within each tile serially. this gives
// cache-friendly access patterns for stencil operations.
//
// the tile size is the CPU analogue of the GPU threadblock size. both are
// configurable via TileSize.
//
// usage (via macro-generated code):
//   DomainForEach::for_each(&domain, |coord| { ... });
//
// usage (direct):
//   domain.for_each_tiled([16, 16], |coord| { ... });
// =============================================================================

use symbi_algebra::Domain;
use rayon::prelude::*;

/// tiled parallel for-each over a domain.
/// the macro #[symbi::kernel(coord)] emits calls to this trait.
pub trait DomainForEach<const D: usize> {
    /// parallel for-each with default tile sizes.
    fn for_each(domain: &Self, f: impl Fn([isize; D]) + Send + Sync);

    /// parallel for-each with explicit tile sizes.
    fn for_each_tiled(domain: &Self, tile: [usize; D], f: impl Fn([isize; D]) + Send + Sync);
}

// default tile sizes per dimension
const fn default_tile<const D: usize>() -> [usize; D] {
    let mut tile = [1usize; D];
    if D >= 1 { tile[0] = 64; }
    if D >= 2 { tile[0] = 16; tile[1] = 16; }
    if D >= 3 { tile[0] = 8; tile[1] = 8; tile[2] = 8; }
    tile
}

impl<const D: usize> DomainForEach<D> for Domain<D> {
    fn for_each(domain: &Self, f: impl Fn([isize; D]) + Send + Sync) {
        Self::for_each_tiled(domain, default_tile::<D>(), f);
    }

    fn for_each_tiled(domain: &Self, tile: [usize; D], f: impl Fn([isize; D]) + Send + Sync) {
        let shape = domain.shape();
        let lo: [isize; D] = std::array::from_fn(|aa| domain.spaces[aa].lo);

        // compute number of tiles per axis
        let num_tiles: [usize; D] = std::array::from_fn(|aa| {
            (shape[aa] + tile[aa] - 1) / tile[aa]
        });
        let total_tiles: usize = num_tiles.iter().product();

        // iterate tiles in parallel, cells within each tile serially
        (0..total_tiles).into_par_iter().for_each(|tile_flat| {
            // unflatten tile index
            let mut tile_idx = [0usize; D];
            let mut remaining = tile_flat;
            for aa in 0..D {
                let stride: usize = num_tiles[aa + 1..].iter().product();
                tile_idx[aa] = remaining / stride;
                remaining %= stride;
            }

            // compute tile bounds (clipped to domain)
            let mut tile_lo = [0isize; D];
            let mut tile_hi = [0isize; D];
            for aa in 0..D {
                tile_lo[aa] = lo[aa] + (tile_idx[aa] * tile[aa]) as isize;
                tile_hi[aa] = (tile_lo[aa] + tile[aa] as isize)
                    .min(lo[aa] + shape[aa] as isize);
            }

            // serial iteration within tile — cache-friendly
            serial_tile_iter::<D>(&tile_lo, &tile_hi, &f);
        });
    }
}

// serial iteration over a rectangular tile [lo, hi) in D dimensions.
// uses nested loops for cache locality. the innermost axis (D-1) is
// the fastest-varying, matching memory layout.
fn serial_tile_iter<const D: usize>(
    lo: &[isize; D],
    hi: &[isize; D],
    f: &impl Fn([isize; D]),
) {
    let mut coord = *lo;
    serial_tile_recurse::<D>(&mut coord, lo, hi, 0, f);
}

fn serial_tile_recurse<const D: usize>(
    coord: &mut [isize; D],
    lo: &[isize; D],
    hi: &[isize; D],
    axis: usize,
    f: &impl Fn([isize; D]),
) {
    if axis == D {
        f(*coord);
        return;
    }
    for ii in lo[axis]..hi[axis] {
        coord[axis] = ii;
        serial_tile_recurse::<D>(coord, lo, hi, axis + 1, f);
    }
}

// ---- parallel reduce ----

/// parallel reduce over a 1D domain.
/// applies `map` to each cell, then combines results with `reduce`.
pub fn parallel_reduce_1d(
    domain: &Domain<1>,
    identity: f64,
    map: impl Fn(isize) -> f64 + Send + Sync,
    reduce: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> f64 {
    let lo = domain.spaces[0].lo;
    let hi = domain.spaces[0].hi;
    (lo..hi)
        .into_par_iter()
        .map(|ii| map(ii))
        .reduce(|| identity, |a, b| reduce(a, b))
}

/// parallel reduce over a 2D domain.
pub fn parallel_reduce_2d(
    domain: &Domain<2>,
    identity: f64,
    map: impl Fn(isize, isize) -> f64 + Send + Sync,
    reduce: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> f64 {
    let lo0 = domain.spaces[0].lo;
    let hi0 = domain.spaces[0].hi;
    let lo1 = domain.spaces[1].lo;
    let hi1 = domain.spaces[1].hi;
    let n1 = (hi1 - lo1) as usize;
    let total = ((hi0 - lo0) as usize) * n1;
    (0..total)
        .into_par_iter()
        .map(|flat| {
            let ii = (flat / n1) as isize + lo0;
            let jj = (flat % n1) as isize + lo1;
            map(ii, jj)
        })
        .reduce(|| identity, |a, b| reduce(a, b))
}

/// parallel reduce over a 3D domain.
pub fn parallel_reduce_3d(
    domain: &Domain<3>,
    identity: f64,
    map: impl Fn(isize, isize, isize) -> f64 + Send + Sync,
    reduce: impl Fn(f64, f64) -> f64 + Send + Sync,
) -> f64 {
    let lo0 = domain.spaces[0].lo;
    let hi0 = domain.spaces[0].hi;
    let lo1 = domain.spaces[1].lo;
    let hi1 = domain.spaces[1].hi;
    let lo2 = domain.spaces[2].lo;
    let hi2 = domain.spaces[2].hi;
    let n1 = (hi1 - lo1) as usize;
    let n2 = (hi2 - lo2) as usize;
    let total = ((hi0 - lo0) as usize) * n1 * n2;
    (0..total)
        .into_par_iter()
        .map(|flat| {
            let ii = (flat / (n1 * n2)) as isize + lo0;
            let jj = ((flat / n2) % n1) as isize + lo1;
            let kk = (flat % n2) as isize + lo2;
            map(ii, jj, kk)
        })
        .reduce(|| identity, |a, b| reduce(a, b))
}

/// kernel metadata trait. generated by #[symbi::kernel(coord)] macro.
pub trait KernelInfo {
    const STENCIL_RADIUS: u32;
    const IS_POINTWISE: bool;
    const NDIM: u8;
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::{Space, domain};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn for_each_visits_all_2d() {
        let dom = domain([
            Space { name: "x", lo: 0, hi: 10 },
            Space { name: "y", lo: 0, hi: 10 },
        ]);
        let count = AtomicUsize::new(0);
        DomainForEach::for_each(&dom, |_coord| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn for_each_visits_all_3d() {
        let dom = domain([
            Space { name: "x", lo: 0, hi: 5 },
            Space { name: "y", lo: 0, hi: 6 },
            Space { name: "z", lo: 0, hi: 7 },
        ]);
        let count = AtomicUsize::new(0);
        DomainForEach::for_each(&dom, |_coord| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::SeqCst), 210);
    }

    #[test]
    fn for_each_tiled_visits_all() {
        let dom = domain([
            Space { name: "x", lo: -3, hi: 13 },
            Space { name: "y", lo: 2, hi: 9 },
        ]);
        let count = AtomicUsize::new(0);
        DomainForEach::for_each_tiled(&dom, [4, 4], |_coord| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        // 16 * 7 = 112
        assert_eq!(count.load(Ordering::SeqCst), 112);
    }

    #[test]
    fn for_each_1d() {
        let dom = domain([Space { name: "x", lo: 0, hi: 100 }]);
        let count = AtomicUsize::new(0);
        DomainForEach::for_each(&dom, |_coord| {
            count.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(count.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn for_each_respects_offsets() {
        let dom = domain([
            Space { name: "x", lo: 5, hi: 8 },
            Space { name: "y", lo: 10, hi: 13 },
        ]);
        let count = AtomicUsize::new(0);
        let in_bounds = AtomicUsize::new(0);
        DomainForEach::for_each(&dom, |coord| {
            count.fetch_add(1, Ordering::Relaxed);
            if coord[0] >= 5 && coord[0] < 8 && coord[1] >= 10 && coord[1] < 13 {
                in_bounds.fetch_add(1, Ordering::Relaxed);
            }
        });
        assert_eq!(count.load(Ordering::SeqCst), 9);
        assert_eq!(in_bounds.load(Ordering::SeqCst), 9);
    }
}
