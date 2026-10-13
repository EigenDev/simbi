// =============================================================================
// closure_patch.rs
//
// the locally isothermal closure's cs^2 of every refined level of a whole
// patch, built as the single hierarchy builds it, so a tile slices out exactly
// the values the single hierarchy holds at its cells. only scalar fields live:
// the root's whole field, then each level's whole patch in turn, every level
// prolonged from the one before it and the one before released.
//
// two rules, the two regimes' own:
// - Clamp (isothermal hydro): each level prolongs its parent over its interior
//   and clamps every ghost into the interior;
// - Extend (isothermal MHD): each level prolongs its parent over its interior
//   and a band of two ghost cells, clipped to the cells the parent can supply,
//   and continues the rest of the ghosts by the periodic image on an axis
//   periodic at both sides, the nearest cell otherwise. a band the clip bites
//   needs a uniform parent, as on the single hierarchy.
//
// usage:
//  let levels = whole_levels(cells, root_ng, root_boundaries, x_lo, dx, &regions);
//  let peak = build_whole_patch_closure(&levels, root_cs2, order, ClosureRule::Clamp, deepest,
//      |level, cs2| { slice into the tiles of `level` })?;
// =============================================================================

use symbi_algebra::{Domain, Space};
use symbi_grid::Field;
use symbi_sim::state::{BoundaryType, Boundaries, extend_closure_into_ghosts};
use symbi_xpu::MemorySpace;

use super::hierarchy::{RATIO, RefinementRegion, region_to_domain};
use super::transfer::{ProlongOrder, clip_to_prolongable, prolong_field};

/// one level of the whole hierarchy, in its own global index space
#[derive(Clone, Debug)]
pub struct WholeLevel<const D: usize> {
    pub interior: Domain<D>,
    pub allocated: Domain<D>,
    pub boundaries: Boundaries<D>,
}

/// how each level continues its cs^2 past the cells it prolongs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClosureRule {
    Clamp,
    Extend,
}

/// the single hierarchy's levels of a root of `cells` with `root_ng` ghosts and the domain's
/// `boundaries`, refined by `regions` (one per level, nested): each fine level covers its region
/// on its parent's lattice, doubled, with one ghost more than the root, and keeps the parent's
/// boundary on a side flush with the parent's interior, a coarse-fine side elsewhere
pub fn whole_levels<const D: usize>(
    cells: [usize; D],
    root_ng: usize,
    boundaries: Boundaries<D>,
    x_lo: [f64; D],
    dx: [f64; D],
    regions: &[RefinementRegion<D>],
) -> Vec<WholeLevel<D>> {
    let interior = Domain::new(std::array::from_fn(|ax| Space { name: axis(ax), lo: 0, hi: cells[ax] as isize }));
    let allocated = grown(&interior, root_ng);
    let mut levels = vec![WholeLevel { interior, allocated, boundaries }];
    let mut spacing = dx;
    for region in regions {
        let parent = levels.last().expect("the root").clone();
        let coverage = region_to_domain(region, &x_lo, &spacing, &[0; D]);
        let interior = Domain::new(std::array::from_fn(|ax| Space {
            name: axis(ax),
            lo: coverage.spaces[ax].lo * RATIO as isize,
            hi: coverage.spaces[ax].hi * RATIO as isize,
        }));
        let mut fine_boundaries = Boundaries::<D>::uniform(BoundaryType::CoarseFine);
        for ax in 0..D {
            if coverage.spaces[ax].lo == parent.interior.spaces[ax].lo {
                fine_boundaries.0[ax][0] = parent.boundaries.lo(ax);
            }
            if coverage.spaces[ax].hi == parent.interior.spaces[ax].hi {
                fine_boundaries.0[ax][1] = parent.boundaries.hi(ax);
            }
        }
        let allocated = grown(&interior, root_ng + 1);
        levels.push(WholeLevel { interior, allocated, boundaries: fine_boundaries });
        spacing = std::array::from_fn(|ax| spacing[ax] / RATIO as f64);
    }
    levels
}

/// the temporary storage `build_whole_patch_closure` holds at its peak, in bytes: a level's
/// parent, the parent's zero companion and the level itself
pub fn closure_peak_bytes<const D: usize>(levels: &[WholeLevel<D>], deepest: usize) -> usize {
    (1..=deepest.min(levels.len() - 1))
        .map(|l| 8 * (2 * levels[l - 1].allocated.volume() + levels[l].allocated.volume()))
        .max()
        .unwrap_or(0)
        .max(8 * levels[0].allocated.volume())
}

/// build every level's cs^2 from the root's (`root`, over `levels[0].allocated`) down to level
/// `deepest`, handing each to `visit` once it is final and before its parent is released.
/// returns the peak temporary storage in bytes, the root's field included
pub fn build_whole_patch_closure<const D: usize, Mem: MemorySpace>(
    levels: &[WholeLevel<D>],
    root: Field<f64, D, Mem>,
    order: ProlongOrder,
    rule: ClosureRule,
    deepest: usize,
    mut visit: impl FnMut(usize, &Field<f64, D, Mem>) -> Result<(), String>,
) -> Result<usize, String> {
    assert!(same_box(root.domain(), &levels[0].allocated), "the root's cs^2 spans the root's allocated cells");
    let mut parent = root;
    let mut peak = 8 * levels[0].allocated.volume();
    for (l, level) in levels.iter().enumerate().take(deepest + 1).skip(1) {
        let zero = Field::zeros(parent.domain()).map_err(|e| format!("closure scratch: {e:?}"))?;
        let field = Field::zeros(&level.allocated).map_err(|e| format!("level {l} closure: {e:?}"))?;
        peak = peak.max(8 * (2 * parent.domain().volume() + level.allocated.volume()));
        match rule {
            ClosureRule::Clamp => {
                prolong_field(&parent, &zero, &field, &level.interior, order, 0.0);
                clamp_into_ghosts(&field, &level.allocated, &level.interior);
            }
            ClosureRule::Extend => {
                let mut band = level.interior.clone();
                for ax in 0..D {
                    band = band.extend(ax, -2, 2);
                }
                let clipped = clip_to_prolongable(&band, &levels[l - 1].allocated, order).unwrap_or(level.interior.clone());
                if !same_box(&clipped, &band) && !uniform(&parent) {
                    return Err(format!(
                        "level {l}: a refined patch runs to the edge of its parent's allocation, where \
                         the closure continues cs^2 from the fine interior, but the parent's cs^2 is not \
                         one finite value; a spatially varying closure needs the parent's data at that edge"
                    ));
                }
                prolong_field(&parent, &zero, &field, &clipped, order, 0.0);
                extend_closure_into_ghosts(&field, &level.allocated, &clipped, &level.boundaries);
            }
        }
        drop(zero);
        visit(l, &field)?;
        parent = field;
    }
    Ok(peak)
}

/// every ghost of `field` takes the interior cell its coordinates clamp to
pub fn clamp_into_ghosts<const D: usize, Mem: MemorySpace>(field: &Field<f64, D, Mem>, allocated: &Domain<D>, interior: &Domain<D>) {
    for c in allocated.iter() {
        if interior.contains(c) {
            continue;
        }
        let clamped: [isize; D] = std::array::from_fn(|ax| c[ax].clamp(interior.spaces[ax].lo, interior.spaces[ax].hi - 1));
        field.set(c, *field.at(clamped));
    }
}

fn uniform<const D: usize, Mem: MemorySpace>(field: &Field<f64, D, Mem>) -> bool {
    let mut cells = field.domain().iter();
    let Some(first) = cells.next().map(|c| *field.at(c)) else {
        return true;
    };
    first.is_finite() && cells.all(|c| field.at(c).to_bits() == first.to_bits())
}

/// whether two domains cover the same cells
pub fn same_box<const D: usize>(a: &Domain<D>, b: &Domain<D>) -> bool {
    (0..D).all(|ax| (a.spaces[ax].lo, a.spaces[ax].hi) == (b.spaces[ax].lo, b.spaces[ax].hi))
}

fn grown<const D: usize>(interior: &Domain<D>, ng: usize) -> Domain<D> {
    let mut out = interior.clone();
    for ax in 0..D {
        out = out.extend(ax, -(ng as isize), ng as isize);
    }
    out
}

fn axis(ax: usize) -> &'static str {
    ["x", "y", "z"][ax]
}
