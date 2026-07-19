// =============================================================================
// config.rs
//
// launch configuration for GPU kernels. grid/block dimensions and shared
// memory size. used by Executor::launch().
//
// usage:
//   let cfg = LaunchConfig::for_2d(256, 256, 16, 16);
//
// block dims are tunable per-process via env vars:
//   SYMBI_BLOCK_1D=256        // 1D substrate kernels
//   SYMBI_BLOCK_2D=32,8       // 2D substrate kernels
//   SYMBI_BLOCK_3D=8,8,4      // 3D substrate kernels
// see `block_dims()`. unset / unparseable falls back to the default and
// the failure is reported ONCE on stderr (silent fallback would mask typos).
// =============================================================================

use std::sync::OnceLock;

/// grid and block dimensions for a kernel launch.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    pub fn for_1d(n: u32, block_size: u32) -> Self {
        LaunchConfig {
            grid: [(n + block_size - 1) / block_size, 1, 1],
            block: [block_size, 1, 1],
            shared_mem_bytes: 0,
        }
    }

    pub fn for_2d(nx: u32, ny: u32, bx: u32, by: u32) -> Self {
        LaunchConfig {
            grid: [(nx + bx - 1) / bx, (ny + by - 1) / by, 1],
            block: [bx, by, 1],
            shared_mem_bytes: 0,
        }
    }

    pub fn for_3d(nx: u32, ny: u32, nz: u32, bx: u32, by: u32, bz: u32) -> Self {
        LaunchConfig {
            grid: [(nx + bx - 1) / bx, (ny + by - 1) / by, (nz + bz - 1) / bz],
            block: [bx, by, bz],
            shared_mem_bytes: 0,
        }
    }
}

/// the block shape for a launch over a window of per-axis `extent` (the exec-domain
/// sizes — the SAME values passed to `LaunchConfig::for_{1,2,3}d`). an explicit
/// `SYMBI_BLOCK_{1D,2D,3D}` env override wins (the escape hatch); otherwise the shape is
/// EXTENT-AWARE (`extent_aware_block`). this replaces the old rank-only `block_dims`,
/// whose fixed `[8,8,4]` idled most of each block on quasi-1D / 2D runs (a 3D kernel over
/// a thin transverse axis — `nz < block.z` left those lanes returning immediately).
pub fn block_for(ndim: usize, extent: &[u32]) -> [u32; 3] {
    env_block(ndim).unwrap_or_else(|| extent_aware_block(ndim, extent))
}

/// derive a block shape from the ACTUAL domain extents, WARP-FIRST on the contiguous axis.
/// the 3D base is `[32,8,1]` — axis 0 (stride-1, `compute_strides[0]=1`) gets a FULL WARP so
/// a warp coalesces into one segment, and the (often thin, CT-transverse-expanded) z axis
/// tiles across GRID blocks. ncu showed the old
/// `[8,8,4]` base starved x: a flux face domain with `nz_face=3` got block.x = 256/(8*3) = 10
/// (below a full warp) -> 59% SM throughput, vs the `nz_face=2` direction at block.x=16 -> 78%.
/// native 2D (Iso/Newtonian) keeps `[16,16,1]` and 1D keeps `[256,1,1]` — only 3D changes.
///   - CLAMP every block dim to its extent (no lanes launched past a thin axis).
///   - REDISTRIBUTE any unspent budget x-FIRST (coalescing), then y, then z; when x is the
///     thin axis (a 2-wide ghost layer) the budget flows to y, filling the block past a half-warp.
pub fn extent_aware_block(ndim: usize, extent: &[u32]) -> [u32; 3] {
    const WARP: u32 = 32;
    let base = [[256u32, 1, 1], [16, 16, 1], [32, 8, 1]][ndim.clamp(1, 3) - 1];
    let target = base[0] * base[1] * base[2];
    let g = |a: usize| -> u32 {
        if a < ndim {
            extent.get(a).copied().unwrap_or(1).max(1)
        } else {
            1
        }
    };

    // clamp each block dim to its extent (no over-provisioned lanes on a thin axis).
    let mut b = [base[0].min(g(0)), base[1].min(g(1)), base[2].min(g(2))];

    // a thin axis leaves the budget unspent — redistribute it to the axes that DO have
    // room, contiguous-axis-FIRST for coalescing (x), then y, then z. when x is the thin
    // one (e.g., a 2-wide ghost layer) the budget flows to y/z, so the block never
    // degenerates to a half-warp. each grow is clamped to its extent + the running budget;
    // x is warp-aligned when it reaches a warp. a FAT domain has no unspent budget, so it
    // keeps its validated base shape (no regression).
    let grow_axis = |b: &mut [u32; 3], axis: usize, g: u32, warp: bool| {
        let others = b
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &x)| x)
            .product::<u32>();
        let room = (target / others.max(1)).max(1);
        let mut v = g.min(room);
        if warp && v >= WARP {
            v -= v % WARP;
        }
        b[axis] = v.max(b[axis]);
    };
    grow_axis(&mut b, 0, g(0), true);
    grow_axis(&mut b, 1, g(1), false);
    grow_axis(&mut b, 2, g(2), false);
    [b[0].max(1), b[1].max(1), b[2].max(1)]
}

/// legacy rank-only block default (env-or-fixed), kept for compatibility. prefer
/// `block_for(ndim, extent)`, which is extent-aware. dims unused by the rank are 1.
pub fn block_dims(ndim: usize) -> [u32; 3] {
    assert!(
        (1..=3).contains(&ndim),
        "block_dims: ndim {ndim} not in 1..=3"
    );
    let fixed = [[256u32, 1, 1], [16, 16, 1], [8, 8, 4]][ndim - 1];
    env_block(ndim).unwrap_or(fixed)
}

/// the EXPLICIT per-process block override from `SYMBI_BLOCK_{1D,2D,3D}`, or `None` if
/// unset (then the extent-aware default applies). parsed once. a malformed value warns
/// once on stderr and is treated as unset.
fn env_block(ndim: usize) -> Option<[u32; 3]> {
    static CACHE: OnceLock<[Option<[u32; 3]>; 3]> = OnceLock::new();
    let dims = CACHE.get_or_init(|| {
        [
            parse_block("SYMBI_BLOCK_1D", 1),
            parse_block("SYMBI_BLOCK_2D", 2),
            parse_block("SYMBI_BLOCK_3D", 3),
        ]
    });
    assert!(
        (1..=3).contains(&ndim),
        "env_block: ndim {ndim} not in 1..=3"
    );
    dims[ndim - 1]
}

/// parse a comma-separated env var of `want` u32 entries into a `[u32; 3]` block (rank-3
/// padded with 1s). `None` on unset; `None` + a one-time stderr warning on malformed input.
fn parse_block(var: &str, want: usize) -> Option<[u32; 3]> {
    let raw = std::env::var(var).ok()?;
    let parts: Vec<&str> = raw.split(',').map(str::trim).collect();
    if parts.len() != want {
        eprintln!(
            "symbi: {var}=\"{raw}\" needs {want} comma-separated u32s; ignoring (using extent-aware default)"
        );
        return None;
    }
    let mut out = [1u32; 3];
    for (i, p) in parts.iter().enumerate() {
        match p.parse::<u32>() {
            Ok(v) if v >= 1 => out[i] = v,
            _ => {
                eprintln!(
                    "symbi: {var}=\"{raw}\" component {i} is not a positive u32; ignoring (using extent-aware default)"
                );
                return None;
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::extent_aware_block;

    // invariants every shape must hold: total <= 256, each dim in [1, extent], block.x a
    // warp multiple (or the whole axis when the axis is sub-warp), and NO lane launched past
    // a thin axis (block dim <= extent).
    fn check(ndim: usize, ext: &[u32]) -> [u32; 3] {
        let b = extent_aware_block(ndim, ext);
        let total = b[0] * b[1] * b[2];
        assert!(
            total <= 256,
            "block {b:?} over {ext:?}: {total} > 256 threads"
        );
        for a in 0..3 {
            let g = if a < ndim { ext[a].max(1) } else { 1 };
            // CORE INVARIANT: no block dim exceeds its extent => no lane launched past a thin axis.
            assert!(
                b[a] >= 1 && b[a] <= g,
                "dim {a} of {b:?} exceeds extent {g}"
            );
            if a >= ndim {
                assert_eq!(b[a], 1, "axis {a} >= ndim must be 1");
            }
        }
        b
    }

    #[test]
    fn extent_aware_shapes() {
        // 3D is WARP-FIRST: x gets a full warp, z tiles across the grid.
        // this is THE fix for the flux face domains (transverse-expanded -> nz=3).
        assert_eq!(check(3, &[1030, 1032, 3]), [32, 8, 1]); // flux_0/1: was [10,8,3] @ 59% -> warp x
        assert_eq!(check(3, &[256, 256, 1]), [32, 8, 1]); // quasi-2D nz=1
        assert_eq!(check(3, &[256, 256, 4]), [32, 8, 1]); // nz=4 tiled across z-blocks
        assert_eq!(check(3, &[64, 64, 64]), [32, 8, 1]); // fat 3D: warp x, z tiled

        // native 2D / 1D base shapes are UNTOUCHED (Iso/Newtonian preserved).
        assert_eq!(check(2, &[256, 256]), [16, 16, 1]);
        assert_eq!(check(1, &[100_000]), [256, 1, 1]);

        // thin transverse y+z: x grows past one warp to refill the budget.
        assert_eq!(check(3, &[256, 1, 1]), [256, 1, 1]); // quasi-1D
        assert_eq!(check(3, &[1024, 4, 1]), [64, 4, 1]);

        // THIN AXIS-0 (a 2-wide ghost layer): x can't grow, so the budget flows to y and
        // the block stays full (a degenerate 16-thread [2,8,1] block would waste most lanes).
        assert_eq!(check(3, &[2, 128, 1]), [2, 128, 1]);
        assert_eq!(check(3, &[2, 1024, 1]), [2, 128, 1]);
        assert_eq!(check(3, &[256, 2, 1]), [128, 2, 1]);

        // tiny / sub-warp axes: take the whole axis (a partial warp is unavoidable).
        check(3, &[16, 16, 16]);
        check(3, &[8, 8, 4]);
    }
}
