// =============================================================================
// block.rs
//
// the BLOCK DECOMPOSITION primitive: a `Domain` tiled into fixed-size blocks.
// the blocks are a PARTITION of the domain — pairwise disjoint and exactly
// covering it (the last block on a non-divisible axis is partial). this is the
// disjoint cover that physics fans out over, and the foundation for choosing a
// block size that controls the surface-to-volume (ghost) ratio.
//
// the axiom (proven in the `laws` tests): `blocks()` partitions `domain` —
// `union == domain` and the blocks are pairwise disjoint. the dominant cost
// lever for haloed-stencil sweeps, the ghost ratio `ghost_ratio(width)`, is a
// first-class property here and is monotonically decreasing in block size (the
// measured "larger patches are cheaper" result, now a law).
//
// NOTE: this is the DOMAIN-decomposition primitive (how the index space is cut
// into work units). it is orthogonal to a memory-layout `View` (strides /
// contiguous axis), which concerns how ONE block is laid out in memory.
//
// usage:
//  let grid = BlockGrid::new(interior, [32, 32, 32]);
//  for block in grid.blocks() { dispatch(block); }   // disjoint -> race-free
//  let ratio = grid.ghost_ratio(2);                  // ghost cells / interior
// =============================================================================

use crate::domain::{Domain, Space};

/// a `Domain` partitioned into fixed-size blocks (the last block per axis may be
/// partial). the blocks are a disjoint cover — see the `laws` module.
#[derive(Debug, Clone)]
pub struct BlockGrid<const R: usize> {
    domain: Domain<R>,
    block: [usize; R],
}

impl<const R: usize> BlockGrid<R> {
    /// tile `domain` into `block`-sized boxes. block sizes must be positive.
    pub fn new(domain: Domain<R>, block: [usize; R]) -> Self {
        for aa in 0..R {
            assert!(block[aa] > 0, "BlockGrid: block size 0 on axis {aa}");
        }
        BlockGrid { domain, block }
    }

    pub fn domain(&self) -> &Domain<R> {
        &self.domain
    }

    pub fn block_size(&self) -> [usize; R] {
        self.block
    }

    /// blocks per axis = ceil(axis size / block size).
    pub fn counts(&self) -> [usize; R] {
        std::array::from_fn(|aa| self.domain.spaces[aa].size().div_ceil(self.block[aa]))
    }

    /// total number of blocks.
    pub fn len(&self) -> usize {
        self.counts().iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.domain.volume() == 0
    }

    /// the block at multi-index `idx` (axis-wise). the last block on an axis is
    /// CLIPPED to the domain when the size is not a multiple of the block size.
    pub fn block(&self, idx: [usize; R]) -> Domain<R> {
        Domain::new(std::array::from_fn(|aa| {
            let lo = self.domain.spaces[aa].lo + (idx[aa] * self.block[aa]) as isize;
            let hi = (lo + self.block[aa] as isize).min(self.domain.spaces[aa].hi);
            Space { name: self.domain.spaces[aa].name, lo, hi }
        }))
    }

    /// the `(lo, size)` of the block at LINEAR index `bi` (row-major, axis 0
    /// fastest) WITHOUT minting a `Domain`. for hot parallel dispatch over many
    /// blocks: `blocks()` allocates a `Vec<Domain>` and a `DomainId` atomic PER
    /// block — fine for a few, ruinous for the 32k blocks an 8-edge tile makes of
    /// a 256^3 grid. iterate `0..len()` and call this instead; it is pure index
    /// arithmetic (no alloc, no atomic). the last block per axis is clipped.
    pub fn window(&self, bi: usize) -> ([isize; R], [usize; R]) {
        let counts = self.counts();
        let mut rem = bi;
        let mut lo = [0isize; R];
        let mut size = [0usize; R];
        for aa in 0..R {
            let b = rem % counts[aa];
            rem /= counts[aa];
            let l = self.domain.spaces[aa].lo + (b * self.block[aa]) as isize;
            let hi = (l + self.block[aa] as isize).min(self.domain.spaces[aa].hi);
            lo[aa] = l;
            size[aa] = (hi - l) as usize;
        }
        (lo, size)
    }

    /// the block multi-index that OWNS a domain cell. every cell of the domain
    /// lies in exactly one block; this finds it. (out-of-domain coords clamp to
    /// the nearest block — callers should pass interior coords.)
    pub fn block_of(&self, coord: [isize; R]) -> [usize; R] {
        let counts = self.counts();
        std::array::from_fn(|aa| {
            let off = (coord[aa] - self.domain.spaces[aa].lo).max(0) as usize;
            (off / self.block[aa]).min(counts[aa] - 1)
        })
    }

    /// the disjoint cover: every block, in row-major block-index order (axis 0
    /// fastest, matching the `Domain` stride convention).
    pub fn blocks(&self) -> Vec<Domain<R>> {
        let counts = self.counts();
        let total = self.len();
        let mut out = Vec::with_capacity(total);
        let mut idx = [0usize; R];
        for _ in 0..total {
            out.push(self.block(idx));
            // increment the mixed-radix block index, axis 0 fastest.
            for aa in 0..R {
                idx[aa] += 1;
                if idx[aa] < counts[aa] {
                    break;
                }
                idx[aa] = 0;
            }
        }
        out
    }

    /// the GHOST RATIO for a halo of `width`: ghost cells per interior cell for a
    /// FULL block, `prod(b_a + 2w) / prod(b_a) - 1`. this is the surface-to-volume
    /// quantity that dominates haloed-stencil cost (measured: binary_disk prolong
    /// share fell 60% -> 34% as the block went 16^3 -> 48^3). it is strictly
    /// DECREASING in every block dimension — grow `block` to shrink the overhead.
    pub fn ghost_ratio(&self, width: usize) -> f64 {
        let inner: usize = self.block.iter().product();
        let outer: usize = self.block.iter().map(|&b| b + 2 * width).product();
        (outer - inner) as f64 / inner as f64
    }
}

// =============================================================================
// algebraic laws (axioms)
//
// the block grid is verified against ground-truth set semantics: the blocks
// PARTITION the domain (disjoint + covering), block_of inverts block membership,
// and the ghost ratio is monotone in block size — the "larger patches are
// cheaper" result, encoded as a law.
// =============================================================================
#[cfg(test)]
mod laws {
    use super::*;
    use crate::domain::{domain, index};
    use std::collections::HashSet;

    // deterministic splitmix prng (no external deps; mirrors domain::laws).
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

    fn cells<const R: usize>(d: &Domain<R>) -> HashSet<[isize; R]> {
        d.iter().collect()
    }

    const N3: [&str; 3] = ["i", "j", "k"];
    const ITERS: usize = 3000;

    // AXIOM: blocks() is a PARTITION of the domain — disjoint and covering, for
    // ANY block size (including non-divisible sizes -> partial last blocks, and
    // blocks larger than the domain -> a single block == domain).
    #[test]
    fn blocks_partition_the_domain() {
        let mut rng = Rng(0xB10C_C0DE);
        for _ in 0..ITERS {
            let dom = Domain::new(std::array::from_fn(|a| {
                let lo = rng.in_range(-4, 4);
                let size = rng.in_range(1, 9);
                crate::domain::Space { name: N3[a], lo, hi: lo + size }
            }));
            let block: [usize; 3] = std::array::from_fn(|_| rng.in_range(1, 6) as usize);
            let grid = BlockGrid::new(dom.clone(), block);
            let blocks = grid.blocks();

            // count matches the analytic product.
            assert_eq!(blocks.len(), grid.len());
            assert_eq!(blocks.len(), grid.counts().iter().product::<usize>());

            // pairwise disjoint.
            for ii in 0..blocks.len() {
                for jj in (ii + 1)..blocks.len() {
                    assert!(!blocks[ii].overlaps(&blocks[jj]), "blocks overlap");
                }
            }
            // disjoint => sum of volumes == union cardinality == domain volume.
            let union: HashSet<[isize; 3]> = blocks.iter().flat_map(|b| b.iter()).collect();
            let vol_sum: usize = blocks.iter().map(|b| b.volume()).sum();
            assert_eq!(vol_sum, union.len(), "blocks not disjoint by volume");

            // COVERING: union == the whole domain, exactly.
            assert_eq!(union, cells(&dom), "blocks do not cover the domain");
        }
    }

    // AXIOM: window(bi) is block()'s (lo, size) without minting a Domain — same
    // disjoint cover, allocation-free. (blocks() and window share the row-major
    // axis-0-fastest index order.)
    #[test]
    fn window_matches_block_allocation_free() {
        let mut rng = Rng(0x5EED_F00D);
        for _ in 0..ITERS {
            let dom = Domain::new(std::array::from_fn(|a| {
                let lo = rng.in_range(-4, 4);
                let size = rng.in_range(1, 9);
                crate::domain::Space { name: N3[a], lo, hi: lo + size }
            }));
            let block: [usize; 3] = std::array::from_fn(|_| rng.in_range(1, 6) as usize);
            let grid = BlockGrid::new(dom, block);
            let blocks = grid.blocks();
            for (bi, b) in blocks.iter().enumerate() {
                let (lo, size) = grid.window(bi);
                for a in 0..3 {
                    assert_eq!(lo[a], b.spaces[a].lo, "window lo mismatch at block {bi}");
                    assert_eq!(size[a], b.spaces[a].size(), "window size mismatch at block {bi}");
                }
            }
        }
    }

    // AXIOM: block_of inverts membership — every cell is owned by the unique
    // block that contains it.
    #[test]
    fn block_of_finds_the_owning_block() {
        let mut rng = Rng(0x0FFE_1234);
        for _ in 0..ITERS {
            let dom = Domain::new(std::array::from_fn(|a| {
                let lo = rng.in_range(-4, 4);
                let size = rng.in_range(1, 9);
                crate::domain::Space { name: N3[a], lo, hi: lo + size }
            }));
            let block: [usize; 3] = std::array::from_fn(|_| rng.in_range(1, 6) as usize);
            let grid = BlockGrid::new(dom.clone(), block);
            for c in dom.iter() {
                let owner = grid.block(grid.block_of(c));
                assert!(owner.contains(c), "block_of({c:?}) does not contain the cell");
            }
        }
    }

    // AXIOM: the ghost ratio is STRICTLY DECREASING in block size (for width>0)
    // and zero at width 0 — "larger blocks are cheaper", as a law.
    #[test]
    fn ghost_ratio_is_monotone_in_block_size() {
        let dom = domain([index("i").over(256), index("j").over(256), index("k").over(256)]);
        for width in 1..=4usize {
            let mut prev = f64::INFINITY;
            for b in [8, 16, 24, 32, 48, 64, 96, 128usize] {
                let r = BlockGrid::new(dom.clone(), [b, b, b]).ghost_ratio(width);
                assert!(r > 0.0, "ghost ratio must be positive for width {width}");
                assert!(r < prev, "ghost ratio not decreasing at block {b}, width {width}");
                prev = r;
            }
        }
        // width 0 -> no ghosts -> ratio 0.
        assert_eq!(BlockGrid::new(dom, [32, 32, 32]).ghost_ratio(0), 0.0);
    }

    // the block decomposition COMPOSES with guillotine_difference: a block's halo
    // shell (expanded block minus the block) is the minimal disjoint cover.
    #[test]
    fn block_halo_is_a_guillotine_shell() {
        let grid = BlockGrid::new(
            domain([index("i").over(64), index("j").over(64), index("k").over(64)]),
            [32, 32, 32],
        );
        let b = grid.block([0, 0, 0]);
        let haloed = b.expand(0, 2).expand(1, 2).expand(2, 2);
        let shell = haloed.guillotine_difference(&b);
        // <= 2*R guillotine boxes, and they tile exactly haloed \ block.
        assert!(shell.len() <= 2 * 3);
        let shell_vol: usize = shell.iter().map(|d| d.volume()).sum();
        assert_eq!(shell_vol, haloed.volume() - b.volume());
    }
}
