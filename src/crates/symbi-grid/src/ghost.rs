// =============================================================================
// ghost.rs
//
// domain-based ghost cell fill. no flat indices. no stride arithmetic.
// ghost fill is a set-theoretic operation:
//
//   1. ghost_regions = allocated.difference(interior)
//   2. for each region, build a coordinate map (periodic/clamp/mirror)
//   3. field.commit_region(region, field.remap(map), &exec)
//
// the domain algebra produces the regions. the coordinate map transforms
// ghost coordinates to interior coordinates. the field's view handles
// strided memory access. each concern is separate.
//
// usage:
//   ghost_fill(&field, &allocated, &interior, &boundaries, &exec)?;
// =============================================================================

use symbi_algebra::Domain;
use symbi_xpu::{ExecutionSpace, MemorySpace, Executor};
use crate::field::Field;

// =============================================================================
// ghost region classification
// =============================================================================

/// which side of the interior a ghost region contacts in a given axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceSide {
    /// ghost is below the interior in this axis.
    Minus,
    /// ghost is above the interior in this axis.
    Plus,
    /// ghost doesn't contact the interior boundary in this axis.
    None,
}

/// how a ghost region touches the interior: face, edge, or corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GhostType {
    Face,   // contacts on exactly 1 axis
    Edge,   // contacts on exactly 2 axes
    Corner, // contacts on 3+ axes
}

/// a classified ghost region: its domain + how it relates to the interior.
#[derive(Clone, Debug)]
pub struct GhostRegion<const D: usize> {
    pub domain: Domain<D>,
    pub ghost_type: GhostType,
    pub directions: [FaceSide; D],
}

/// analyze ghost regions: compute allocated.difference(interior) and
/// classify each piece by contact type and direction.
pub fn analyze_ghost_regions<const D: usize>(
    allocated: &Domain<D>,
    interior: &Domain<D>,
) -> Vec<GhostRegion<D>> {
    let raw_regions = allocated.difference(interior);
    let mut result = Vec::with_capacity(raw_regions.len());

    for region in &raw_regions {
        let mut directions = [FaceSide::None; D];
        let mut contact_count = 0;

        for ax in 0..D {
            if region.spaces[ax].hi == interior.spaces[ax].lo {
                directions[ax] = FaceSide::Minus;
                contact_count += 1;
            } else if region.spaces[ax].lo == interior.spaces[ax].hi {
                directions[ax] = FaceSide::Plus;
                contact_count += 1;
            }
        }

        let ghost_type = match contact_count {
            0 | 1 => GhostType::Face,
            2 => GhostType::Edge,
            _ => GhostType::Corner,
        };

        result.push(GhostRegion { domain: region.clone(), ghost_type, directions });
    }

    result
}

// =============================================================================
// boundary condition type
// =============================================================================

/// boundary condition type per face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BcType {
    Periodic,
    Outflow,
    Reflect,
    /// skip ghost fill on this face (filled externally, e.g. by AMR prolongation).
    Skip,
}

// =============================================================================
// coordinate maps — pure functions from coord to coord
// =============================================================================

/// periodic coordinate map for one axis.
/// wraps coord[dim] into [start, start + len).
#[inline]
pub fn periodic_remap<const D: usize>(
    coord: [isize; D],
    dim: usize,
    start: isize,
    len: isize,
) -> [isize; D] {
    let mut out = coord;
    let mut val = out[dim] - start;
    val = val % len;
    if val < 0 { val += len; }
    out[dim] = start + val;
    out
}

/// outflow (zero-gradient) coordinate map for one axis.
/// clamps coord[dim] to [lo, hi-1].
#[inline]
pub fn clamp_remap<const D: usize>(
    coord: [isize; D],
    dim: usize,
    lo: isize,
    hi: isize,
) -> [isize; D] {
    let mut out = coord;
    if out[dim] < lo { out[dim] = lo; }
    else if out[dim] >= hi { out[dim] = hi - 1; }
    out
}

/// reflective coordinate map for one axis.
/// mirrors around a pivot: coord[dim] = 2*pivot - 1 - coord[dim].
#[inline]
pub fn mirror_remap<const D: usize>(
    coord: [isize; D],
    dim: usize,
    pivot: isize,
) -> [isize; D] {
    let mut out = coord;
    out[dim] = 2 * pivot - 1 - out[dim];
    out
}

/// build a multi-axis coordinate map for a ghost region.
/// applies the appropriate remap per axis based on the ghost region's
/// direction and the boundary condition.
pub fn build_bc_map<const D: usize>(
    region: &GhostRegion<D>,
    interior: &Domain<D>,
    boundaries: &[[BcType; 2]; D], // [lo, hi] per axis
) -> impl Fn([isize; D]) -> [isize; D] + Clone {
    // collect per-axis remap info
    let remap_info: [(FaceSide, BcType, isize, isize); D] =
        std::array::from_fn(|ax| {
            let side = region.directions[ax];
            let bc = match side {
                FaceSide::Minus => boundaries[ax][0],
                FaceSide::Plus => boundaries[ax][1],
                FaceSide::None => BcType::Periodic, // no contact — use identity (periodic is identity in-range)
            };
            let lo = interior.spaces[ax].lo;
            let hi = interior.spaces[ax].hi;
            (side, bc, lo, hi)
        });

    move |coord: [isize; D]| -> [isize; D] {
        let mut out = coord;
        for ax in 0..D {
            let (side, bc, lo, hi) = remap_info[ax];
            if side == FaceSide::None { continue; }

            match bc {
                BcType::Periodic => {
                    let len = hi - lo;
                    let mut val = out[ax] - lo;
                    val = val % len;
                    if val < 0 { val += len; }
                    out[ax] = lo + val;
                }
                BcType::Outflow => {
                    if out[ax] < lo { out[ax] = lo; }
                    else if out[ax] >= hi { out[ax] = hi - 1; }
                }
                BcType::Reflect => {
                    let pivot = if side == FaceSide::Minus { lo } else { hi };
                    out[ax] = 2 * pivot - 1 - out[ax];
                }
                BcType::Skip => {}
            }
        }
        out
    }
}

// =============================================================================
// the ghost fill: domain algebra + coordinate maps + field remap
// =============================================================================

/// true if ALL contacting faces of a ghost region are BcType::Skip.
/// face ghosts touching a Skip face are filled externally (AMR prolongation).
/// edge/corner ghosts with mixed Skip + physical BCs are processed normally:
/// build_bc_map applies identity for Skip dims and physical BCs for the rest,
/// reading from the prolongated face ghost data at the clamped position.
fn region_touches_skip<const D: usize>(
    region: &GhostRegion<D>,
    boundaries: &[[BcType; 2]; D],
) -> bool {
    let mut any_contact = false;
    for ax in 0..D {
        let is_skip = match region.directions[ax] {
            FaceSide::Minus => { any_contact = true; boundaries[ax][0] == BcType::Skip },
            FaceSide::Plus => { any_contact = true; boundaries[ax][1] == BcType::Skip },
            FaceSide::None => true, // no contact — doesn't count
        };
        if !is_skip { return false; }
    }
    any_contact // at least one contact and all are Skip
}

/// fill ghost cells of a scalar field using domain-based coordinate maps.
/// no flat indices. no stride arithmetic.
/// regions touching BcType::Skip faces are left untouched.
pub fn ghost_fill_field<const D: usize, S: ExecutionSpace, M: MemorySpace>(
    field: &Field<f64, D, M>,
    allocated: &Domain<D>,
    interior: &Domain<D>,
    boundaries: &[[BcType; 2]; D],
    _exec: &Executor<S>,
) -> symbi_xpu::Result<()> {
    let regions = analyze_ghost_regions(allocated, interior);

    for region in &regions {
        if region_touches_skip(region, boundaries) { continue; }
        let bc_map = build_bc_map(region, interior, boundaries);
        let src_view = field.view();

        for coord in &region.domain {
            let src_coord = bc_map(coord);
            let val = *src_view.at(src_coord);
            field.view_mut().set(coord, val);
        }
    }

    Ok(())
}

/// fill ghost cells of ALL fields at each point, in a single pass.
/// iterates ghost coordinates once, applies the coordinate map once per
/// point, and copies all field values at that point.
///
/// `fill_at` is called with (ghost_coord, source_coord) and should copy
/// all relevant field data from source to ghost.
/// regions touching BcType::Skip faces are left untouched.
pub fn ghost_fill_all<const D: usize>(
    allocated: &Domain<D>,
    interior: &Domain<D>,
    boundaries: &[[BcType; 2]; D],
    mut fill_at: impl FnMut([isize; D], [isize; D]),
) {
    let regions = analyze_ghost_regions(allocated, interior);

    for region in &regions {
        if region_touches_skip(region, boundaries) { continue; }
        let bc_map = build_bc_map(region, interior, boundaries);
        for coord in &region.domain {
            let src_coord = bc_map(coord);
            fill_at(coord, src_coord);
        }
    }
}

/// like ghost_fill_all, but the callback also receives a bitmask of which
/// axes have a Reflect BC active for this ghost region. bit `ax` is set
/// when the region contacts a Reflect face on axis `ax`. use this to
/// negate velocity components normal to reflecting faces.
pub fn ghost_fill_all_reflect<const D: usize>(
    allocated: &Domain<D>,
    interior: &Domain<D>,
    boundaries: &[[BcType; 2]; D],
    mut fill_at: impl FnMut([isize; D], [isize; D], u8),
) {
    let regions = analyze_ghost_regions(allocated, interior);

    for region in &regions {
        if region_touches_skip(region, boundaries) { continue; }
        let bc_map = build_bc_map(region, interior, boundaries);

        // build reflect bitmask: bit ax is set if this region contacts
        // a Reflect face on axis ax.
        let mut reflect_mask: u8 = 0;
        for ax in 0..D {
            let is_reflect = match region.directions[ax] {
                FaceSide::Minus => boundaries[ax][0] == BcType::Reflect,
                FaceSide::Plus => boundaries[ax][1] == BcType::Reflect,
                FaceSide::None => false,
            };
            if is_reflect { reflect_mask |= 1 << ax; }
        }

        for coord in &region.domain {
            let src_coord = bc_map(coord);
            fill_at(coord, src_coord, reflect_mask);
        }
    }
}

// =============================================================================
// tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_algebra::{Space, Domain};

    #[test]
    fn ghost_regions_1d() {
        let allocated = Domain::new([Space { name: "i", lo: -2, hi: 12 }]);
        let interior = Domain::new([Space { name: "i", lo: 0, hi: 10 }]);
        let regions = analyze_ghost_regions(&allocated, &interior);
        // 1D: 2 regions (left ghost, right ghost)
        assert_eq!(regions.len(), 2);
        let total: usize = regions.iter().map(|r| r.domain.volume()).sum();
        assert_eq!(total, 4); // 2 + 2
    }

    #[test]
    fn ghost_regions_2d() {
        let allocated = Domain::new([
            Space { name: "i", lo: -2, hi: 12 },
            Space { name: "j", lo: -2, hi: 12 },
        ]);
        let interior = Domain::new([
            Space { name: "i", lo: 0, hi: 10 },
            Space { name: "j", lo: 0, hi: 10 },
        ]);
        let regions = analyze_ghost_regions(&allocated, &interior);
        // 2D: 3^2 - 1 = 8 regions (4 faces + 4 corners)
        assert_eq!(regions.len(), 8);
        let total: usize = regions.iter().map(|r| r.domain.volume()).sum();
        assert_eq!(total, allocated.volume() - interior.volume());
    }

    #[test]
    fn ghost_regions_3d() {
        let ng = 2_isize;
        let nn = 10_isize;
        let allocated = Domain::new([
            Space { name: "i", lo: -ng, hi: nn + ng },
            Space { name: "j", lo: -ng, hi: nn + ng },
            Space { name: "k", lo: -ng, hi: nn + ng },
        ]);
        let interior = allocated.contract(ng);
        let regions = analyze_ghost_regions(&allocated, &interior);
        // 3D: 3^3 - 1 = 26 regions
        assert_eq!(regions.len(), 26);
        let total: usize = regions.iter().map(|r| r.domain.volume()).sum();
        assert_eq!(total, allocated.volume() - interior.volume());
    }

    #[test]
    fn periodic_map_wraps() {
        // 1D: interior [0, 10), ghost at -1 should map to 9
        let coord = periodic_remap([-1], 0, 0, 10);
        assert_eq!(coord, [9]);

        let coord = periodic_remap([-2], 0, 0, 10);
        assert_eq!(coord, [8]);

        let coord = periodic_remap([10], 0, 0, 10);
        assert_eq!(coord, [0]);

        let coord = periodic_remap([11], 0, 0, 10);
        assert_eq!(coord, [1]);
    }

    #[test]
    fn clamp_map_clamps() {
        let coord = clamp_remap([-1], 0, 0, 10);
        assert_eq!(coord, [0]);

        let coord = clamp_remap([10], 0, 0, 10);
        assert_eq!(coord, [9]);

        let coord = clamp_remap([5], 0, 0, 10);
        assert_eq!(coord, [5]);
    }

    #[test]
    fn mirror_map_reflects() {
        // reflect around pivot=0: coord -> 2*0 - 1 - coord = -1 - coord
        let coord = mirror_remap([-1], 0, 0);
        assert_eq!(coord, [0]);

        let coord = mirror_remap([-2], 0, 0);
        assert_eq!(coord, [1]);
    }

    #[test]
    fn ghost_fill_periodic_1d() {
        use symbi_xpu::{CpuSpace, HostMemory, Executor};
        use crate::Field;

        let ng = 2_isize;
        let nn = 10_isize;
        let allocated = Domain::new([Space { name: "i", lo: -ng, hi: nn + ng }]);
        let interior = allocated.contract(ng);
        let exec = Executor::<CpuSpace>::new(0).unwrap();

        let field = Field::<f64, 1, HostMemory>::zeros(&allocated).unwrap();
        // fill interior with 1..10
        for ii in 0..nn {
            field.view_mut().set([ii], (ii + 1) as f64);
        }

        let boundaries = [[BcType::Periodic; 2]; 1];
        ghost_fill_field(&field, &allocated, &interior, &boundaries, &exec).unwrap();

        // left ghosts: [-2] = interior[8] = 9.0, [-1] = interior[9] = 10.0
        assert_eq!(*field.view().at([-2]), 9.0);
        assert_eq!(*field.view().at([-1]), 10.0);

        // right ghosts: [10] = interior[0] = 1.0, [11] = interior[1] = 2.0
        assert_eq!(*field.view().at([10]), 1.0);
        assert_eq!(*field.view().at([11]), 2.0);
    }

    #[test]
    fn ghost_fill_outflow_1d() {
        use symbi_xpu::{CpuSpace, HostMemory, Executor};
        use crate::Field;

        let ng = 2_isize;
        let nn = 10_isize;
        let allocated = Domain::new([Space { name: "i", lo: -ng, hi: nn + ng }]);
        let interior = allocated.contract(ng);
        let exec = Executor::<CpuSpace>::new(0).unwrap();

        let field = Field::<f64, 1, HostMemory>::zeros(&allocated).unwrap();
        for ii in 0..nn {
            field.view_mut().set([ii], (ii + 1) as f64);
        }

        let boundaries = [[BcType::Outflow; 2]; 1];
        ghost_fill_field(&field, &allocated, &interior, &boundaries, &exec).unwrap();

        // left ghosts: clamped to interior[0] = 1.0
        assert_eq!(*field.view().at([-2]), 1.0);
        assert_eq!(*field.view().at([-1]), 1.0);

        // right ghosts: clamped to interior[9] = 10.0
        assert_eq!(*field.view().at([10]), 10.0);
        assert_eq!(*field.view().at([11]), 10.0);
    }

    #[test]
    fn ghost_fill_periodic_2d() {
        use symbi_xpu::{CpuSpace, HostMemory, Executor};
        use crate::Field;

        let ng = 1_isize;
        let nx = 4_isize;
        let ny = 4_isize;
        let allocated = Domain::new([
            Space { name: "i", lo: -ng, hi: nx + ng },
            Space { name: "j", lo: -ng, hi: ny + ng },
        ]);
        let interior = allocated.contract(ng);
        let exec = Executor::<CpuSpace>::new(0).unwrap();

        let field = Field::<f64, 2, HostMemory>::zeros(&allocated).unwrap();
        // fill interior: value = 10*i + j
        for ii in 0..nx {
            for jj in 0..ny {
                field.view_mut().set([ii, jj], (10 * ii + jj) as f64);
            }
        }

        let boundaries = [[BcType::Periodic; 2]; 2];
        ghost_fill_field(&field, &allocated, &interior, &boundaries, &exec).unwrap();

        // i-direction: ghost[-1, 0] = interior[3, 0] = 30.0
        assert_eq!(*field.view().at([-1, 0]), 30.0);
        // i-direction: ghost[4, 0] = interior[0, 0] = 0.0
        assert_eq!(*field.view().at([4, 0]), 0.0);

        // j-direction: ghost[0, -1] = interior[0, 3] = 3.0
        assert_eq!(*field.view().at([0, -1]), 3.0);
        // j-direction: ghost[0, 4] = interior[0, 0] = 0.0
        assert_eq!(*field.view().at([0, 4]), 0.0);

        // corner: ghost[-1, -1] = interior[3, 3] = 33.0
        assert_eq!(*field.view().at([-1, -1]), 33.0);
    }

    #[test]
    fn ghost_fill_periodic_3d() {
        use symbi_xpu::{CpuSpace, HostMemory, Executor};
        use crate::Field;

        let ng = 1_isize;
        let nn = 4_isize;
        let allocated = Domain::new([
            Space { name: "i", lo: -ng, hi: nn + ng },
            Space { name: "j", lo: -ng, hi: nn + ng },
            Space { name: "k", lo: -ng, hi: nn + ng },
        ]);
        let interior = allocated.contract(ng);
        let exec = Executor::<CpuSpace>::new(0).unwrap();

        let field = Field::<f64, 3, HostMemory>::zeros(&allocated).unwrap();
        // fill interior: value = 100*i + 10*j + k
        for coord in &interior {
            let [ii, jj, kk] = coord;
            field.view_mut().set(coord, (100 * ii + 10 * jj + kk) as f64);
        }

        let boundaries = [[BcType::Periodic; 2]; 3];
        ghost_fill_field(&field, &allocated, &interior, &boundaries, &exec).unwrap();

        // face ghost: [-1, 0, 0] = interior[3, 0, 0] = 300.0
        assert_eq!(*field.view().at([-1, 0, 0]), 300.0);

        // edge ghost: [-1, -1, 0] = interior[3, 3, 0] = 330.0
        assert_eq!(*field.view().at([-1, -1, 0]), 330.0);

        // corner ghost: [-1, -1, -1] = interior[3, 3, 3] = 333.0
        assert_eq!(*field.view().at([-1, -1, -1]), 333.0);
    }
}
