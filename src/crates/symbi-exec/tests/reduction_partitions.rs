// =============================================================================
// reduction_partitions.rs
//
// the arithmetic of the device block reduction (`field_reduce` on device
// memory), replayed on the host: the window's cells in the kernel's linear
// order (last axis fastest), blocks of REDUCTION_BLOCK_SIZE consecutive cells,
// each block summed by the halving tree (`s[t] += s[t + stride]`), and the
// block partials folded in block order. a tile rounds its window to one f64,
// and the exact merge of the tiles' rounded values depends on the partition:
// - on a cancellation-sensitive field, the whole window and the same window cut
//   into two tiles give different bits
// - the exact accumulation of the same cells (`ExactSum`, the contract the
//   device reduction of body feedback follows) gives the same bits for every
//   partition and equals the correctly rounded sum
//
// run: cargo test -p symbi-exec --test reduction_partitions
// =============================================================================
use symbi_exec::exact_sum::ExactSum;

const BLOCK: usize = 256;
const N: [usize; 3] = [8, 8, 8];

/// a field whose terms are exact and whose partial sums round: 1e16, -1e16 and three units by the
/// cell's global index mod 5. 1e16 + 1.0 rounds back to 1e16 (the spacing there is 2), so a
/// grouping that adds a unit to a large partial loses it and one that adds units together keeps them
fn value(i: usize, j: usize, k: usize) -> f64 {
    match (i * 64 + j * 8 + k) % 5 {
        0 => 1e16,
        1 => -1e16,
        _ => 1.0,
    }
}

/// the cells of the box [lo, hi) in the kernel's linear order
fn window(lo: [usize; 3], hi: [usize; 3]) -> Vec<f64> {
    let mut out = Vec::new();
    for i in lo[0]..hi[0] {
        for j in lo[1]..hi[1] {
            for k in lo[2]..hi[2] {
                out.push(value(i, j, k));
            }
        }
    }
    out
}

/// one window through the device block reduction: per-block halving trees, then the host fold
fn block_reduction(cells: &[f64]) -> f64 {
    let mut acc = 0.0;
    for chunk in cells.chunks(BLOCK) {
        let mut s = [0.0f64; BLOCK];
        s[..chunk.len()].copy_from_slice(chunk);
        let mut stride = BLOCK / 2;
        while stride > 0 {
            for t in 0..stride {
                s[t] += s[t + stride];
            }
            stride /= 2;
        }
        acc += s[0];
    }
    acc
}

fn exact(cells: &[f64]) -> ExactSum {
    let mut acc = ExactSum::new();
    for &x in cells {
        acc.add(x);
    }
    acc
}

#[test]
fn a_tile_rounded_before_the_merge_depends_on_the_partition_and_the_exact_sum_does_not() {
    let whole = window([0, 0, 0], N);
    let tiles = [window([0, 0, 0], [8, 8, 4]), window([0, 0, 4], N)];
    assert_eq!(tiles.iter().map(Vec::len).sum::<usize>(), whole.len());

    // each tile's rounded block reduction, merged exactly as the body feedback merges partials
    let mut rounded_whole = ExactSum::new();
    rounded_whole.add(block_reduction(&whole));
    let mut rounded_tiles = ExactSum::new();
    for tile in &tiles {
        rounded_tiles.add(block_reduction(tile));
    }
    let (one, two) = (rounded_whole.value(), rounded_tiles.value());
    assert_ne!(
        one.to_bits(),
        two.to_bits(),
        "the rounded block reduction agrees across the partition on this field; the demonstration is vacuous"
    );

    // the exact accumulation of the same cells: one integer for every partition, rounded once
    let mut exact_tiles = ExactSum::new();
    for tile in &tiles {
        exact_tiles.merge(&exact(tile));
    }
    let exact_whole = exact(&whole);
    assert_eq!(exact_tiles.to_words(), exact_whole.to_words());
    let total = exact_whole.value();
    println!("block reduction: whole {one:e} ({:#018x}), two tiles {two:e} ({:#018x}); exact {total:e} ({:#018x})",
             one.to_bits(), two.to_bits(), total.to_bits());
    // the correctly rounded total from the counts: the large terms cancel in pairs, the units add
    let count = |m: usize| (0..whole.len()).filter(|g| g % 5 == m).count() as f64;
    let reference = (count(0) - count(1)) * 1e16 + count(2) + count(3) + count(4);
    assert_eq!(total.to_bits(), reference.to_bits(), "the exact total {total} against the counted {reference}");
}
