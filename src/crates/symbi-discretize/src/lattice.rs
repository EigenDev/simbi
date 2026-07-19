// =============================================================================
// lattice.rs
//
// the lattice-map / pullback algebra. a `LatticeMap` is an
// INTEGER rule "destination cell -> source cell": boundary conditions, AMR
// restrict/prolong, staggered-grid shifts, and MPI halos all reduce to one. it
// pairs with a field PULLBACK — read the source cell, then transform the value by
// a rule the field's grade fixes (a density copies; a velocity flips its
// wall-normal component at a mirror). that makes `ghost_fill` data, not a kernel.
//
// this module is the map itself: pure integer arithmetic on cell indices, with
// no float and no IR. because the source coordinate is an integer function of the
// destination, the pullback (`pullback.rs`) reads it as an ordinary integer-index
// stencil — there is no "data-dependent gather" and no float->int cast.
//
// the per-axis Jacobian sign (`+1`, or `-1` on a reflected axis) is what a field
// value picks up: a scalar ignores it, a vector multiplies its component on that
// axis, a rank-2 tensor twice. that is the entire content of the hand-written
// `vel_sign` parameter — it is the field's grade reacting to the map.
// =============================================================================

/// an integer cell-index remap: given a destination cell, the source cell it
/// reads. boundary conditions, refinement, staggering and halos are all instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LatticeMap {
    /// read self.
    Identity,
    /// wrap: `source[axis] = dest[axis] + shift` (shift = +period for a low-side
    /// ghost, -period for a high-side ghost).
    Periodic { axis: u8, shift: i64 },
    /// mirror about a wall: `source[axis] = pivot2 - dest[axis]`, where
    /// `pivot2 = 2 * wall_index` (the wall sits on a half-index between cells).
    Reflect { axis: u8, pivot2: i64 },
    /// snap to the edge: `source[axis] = edge` (zeroth-order copy of the nearest
    /// interior cell).
    Outflow { axis: u8, edge: i64 },
    /// refinement-lattice child read: `source[axis] = dest[axis] * ratio + offset`.
    /// the amr RESTRICTION pullback — a coarse cell reads its fine children with
    /// `offset` in `0..ratio`. levels share absolute index space (fine index
    /// `ratio*c .. ratio*c + ratio` covers coarse cell `c`), so no coverage-relative
    /// translation appears anywhere.
    Refine { axis: u8, ratio: i64, offset: i64 },
    /// refinement-lattice parent read: `source[axis] = floor_div(dest[axis], ratio)`.
    /// the amr PROLONGATION pullback — a fine cell (ghost indices included, hence
    /// floor division for negatives) reads its coarse parent in absolute indices.
    Coarsen { axis: u8, ratio: i64 },
    /// apply both maps (e.g., a 2D corner ghost: periodic in x and reflect in y).
    /// the maps act on different axes, so order is immaterial.
    Compose(Box<LatticeMap>, Box<LatticeMap>),
}

impl LatticeMap {
    /// the source cell a destination cell reads from.
    pub fn source(&self, dest: &[i64]) -> Vec<i64> {
        match self {
            LatticeMap::Identity => dest.to_vec(),
            LatticeMap::Periodic { axis, shift } => with(dest, *axis, |c| c + *shift),
            LatticeMap::Reflect { axis, pivot2 } => with(dest, *axis, |c| *pivot2 - c),
            LatticeMap::Outflow { axis, edge } => with(dest, *axis, |_| *edge),
            LatticeMap::Refine {
                axis,
                ratio,
                offset,
            } => with(dest, *axis, |c| c * *ratio + *offset),
            LatticeMap::Coarsen { axis, ratio } => with(dest, *axis, |c| c.div_euclid(*ratio)),
            LatticeMap::Compose(a, b) => a.source(&b.source(dest)),
        }
    }

    /// the per-axis sign a field value picks up (the diagonal of the map's
    /// Jacobian): `-1` on a reflected axis, `+1` otherwise. a scalar ignores it;
    /// a vector multiplies its component on that axis by it; a tensor, twice.
    pub fn jacobian_sign(&self, axis: u8) -> i64 {
        match self {
            LatticeMap::Reflect { axis: ra, .. } if *ra == axis => -1,
            LatticeMap::Compose(a, b) => a.jacobian_sign(axis) * b.jacobian_sign(axis),
            _ => 1,
        }
    }
}

fn with(dest: &[i64], axis: u8, f: impl Fn(i64) -> i64) -> Vec<i64> {
    let mut s = dest.to_vec();
    s[axis as usize] = f(dest[axis as usize]);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_wraps_both_sides() {
        // low-side ghost at -1 reads the high interior cell (period 64).
        let lo = LatticeMap::Periodic { axis: 0, shift: 64 };
        assert_eq!(lo.source(&[-1]), vec![63]);
        assert_eq!(lo.source(&[-2]), vec![62]);
        // high-side ghost at 64 reads the low interior cell.
        let hi = LatticeMap::Periodic {
            axis: 0,
            shift: -64,
        };
        assert_eq!(hi.source(&[64]), vec![0]);
        // periodic carries no sign flip.
        assert_eq!(lo.jacobian_sign(0), 1);
    }

    #[test]
    fn reflect_mirrors_and_flips_normal() {
        // wall on the half-index -0.5 (between cell -1 and cell 0): pivot2 = -1.
        let m = LatticeMap::Reflect {
            axis: 0,
            pivot2: -1,
        };
        assert_eq!(m.source(&[-1]), vec![0]); // ghost -1 mirrors to interior 0
        assert_eq!(m.source(&[-2]), vec![1]); // ghost -2 mirrors to interior 1
        // the wall-normal component flips; tangential does not.
        assert_eq!(m.jacobian_sign(0), -1);
        assert_eq!(m.jacobian_sign(1), 1);
    }

    #[test]
    fn outflow_clamps_to_the_edge() {
        let m = LatticeMap::Outflow { axis: 0, edge: 0 };
        assert_eq!(m.source(&[-1]), vec![0]);
        assert_eq!(m.source(&[-3]), vec![0]);
        assert_eq!(m.jacobian_sign(0), 1); // a copy carries no flip
    }

    #[test]
    fn compose_handles_a_corner() {
        // 2D corner ghost: periodic in x, reflecting in y.
        let m = LatticeMap::Compose(
            Box::new(LatticeMap::Periodic { axis: 0, shift: 64 }),
            Box::new(LatticeMap::Reflect {
                axis: 1,
                pivot2: -1,
            }),
        );
        assert_eq!(m.source(&[-1, -1]), vec![63, 0]); // x wraps, y mirrors
        assert_eq!(m.jacobian_sign(0), 1); // x: periodic, no flip
        assert_eq!(m.jacobian_sign(1), -1); // y: reflected, flips
    }

    #[test]
    fn identity_is_a_no_op() {
        assert_eq!(LatticeMap::Identity.source(&[3, 4, 5]), vec![3, 4, 5]);
        assert_eq!(LatticeMap::Identity.jacobian_sign(0), 1);
    }

    #[test]
    fn refine_reads_the_fine_children() {
        // coarse cell 3 reads fine children 6 and 7 at ratio 2 (absolute indices).
        let lo = LatticeMap::Refine {
            axis: 0,
            ratio: 2,
            offset: 0,
        };
        let hi = LatticeMap::Refine {
            axis: 0,
            ratio: 2,
            offset: 1,
        };
        assert_eq!(lo.source(&[3]), vec![6]);
        assert_eq!(hi.source(&[3]), vec![7]);
        // negative coarse indices (ghosts) scale the same way.
        assert_eq!(lo.source(&[-1]), vec![-2]);
        assert_eq!(hi.source(&[-1]), vec![-1]);
        // a refinement carries no sign flip.
        assert_eq!(lo.jacobian_sign(0), 1);
    }

    #[test]
    fn coarsen_reads_the_parent_with_floor_semantics() {
        // fine cells 6 and 7 share coarse parent 3; the negative ghost pair
        // -1, -2 maps to parent -1 (floor division, not truncation).
        let m = LatticeMap::Coarsen { axis: 0, ratio: 2 };
        assert_eq!(m.source(&[6]), vec![3]);
        assert_eq!(m.source(&[7]), vec![3]);
        assert_eq!(m.source(&[-1]), vec![-1]);
        assert_eq!(m.source(&[-2]), vec![-1]);
        assert_eq!(m.source(&[-3]), vec![-2]);
        assert_eq!(m.jacobian_sign(0), 1);
    }

    #[test]
    fn coarsen_inverts_refine_on_every_child() {
        // floor_div(r*c + o, r) == c for all o in 0..r — parent-of-child is identity.
        let r = 2i64;
        for cc in -5..5i64 {
            for oo in 0..r {
                let refine = LatticeMap::Refine {
                    axis: 0,
                    ratio: r,
                    offset: oo,
                };
                let coarsen = LatticeMap::Coarsen { axis: 0, ratio: r };
                assert_eq!(coarsen.source(&refine.source(&[cc])), vec![cc]);
            }
        }
    }
}
