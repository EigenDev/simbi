// =============================================================================
// placement.rs
//
// where each tile of a decomposed grid sits on the whole grid it belongs to.
// a whole grid (cells, origin, spacing, coordinate maps, declared boundaries)
// and a partition of its cells determine, for every tile:
// - the whole-grid index of its first cell and its cell count
// - the boundary each face carries: a cut between neighboring tiles, a cut at
//   the domain seam of an axis that wraps across tiles, the declared condition
//   at every other domain edge
// - its physical extent, read from the whole grid's face positions
// placement reads the grid and the partition and returns data; it builds,
// seeds and mutates nothing. a single grid is the one-tile partition: offset
// zero, the whole cell count, the declared boundaries.
//
// usage:
//  let grid = WholeGrid { cells, origin, spacing, maps, boundaries };
//  let tile = grid.place(&partition, tc);
//  let sim = Sim::build(regime, eos, metric).placed(&grid, &tile)...
//  let ring = grid.topology(partition.counts());
// =============================================================================

use crate::decomp::{Partition, Topology};
use crate::state::{Boundaries, BoundaryType};
use symbi_geometry::AxisMap;

/// the lower face of whole-grid cell `i` on one axis: the axis map's face when the grid carries
/// maps, `x_lo + i dx` otherwise. every face position a grid or a tile evaluates on the host comes
/// from this one expression, so a tile's extent and its cells agree bit for bit.
#[inline]
pub fn face_position(x_lo: f64, dx: f64, map: Option<&AxisMap>, i: isize) -> f64 {
    match map {
        Some(map) => map.face(i),
        None => x_lo + (i as f64) * dx,
    }
}

/// the grid a run decomposes: its interior cells, origin, spacing and coordinate maps, and the
/// boundary conditions its domain declares on every face.
#[derive(Clone, Debug)]
pub struct WholeGrid<const D: usize> {
    pub cells: [usize; D],
    pub origin: [f64; D],
    pub spacing: [f64; D],
    pub maps: Option<[AxisMap; D]>,
    pub boundaries: Boundaries<D>,
}

/// one tile's place on its whole grid: the whole-grid index of its first cell, its cell count,
/// and the boundary each of its faces carries.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedTile<const D: usize> {
    pub offset: [usize; D],
    pub cells: [usize; D],
    pub boundaries: Boundaries<D>,
}

impl<const D: usize> WholeGrid<D> {
    /// the lower face of whole-grid cell `i` on axis `ax`.
    pub fn face(&self, ax: usize, i: isize) -> f64 {
        face_position(self.origin[ax], self.spacing[ax], self.maps.as_ref().map(|m| &m[ax]), i)
    }

    /// whether the domain is declared periodic on both faces of `ax`. a one-sided declaration
    /// leaves the axis open.
    pub fn periodic(&self, ax: usize) -> bool {
        self.boundaries.lo(ax) == BoundaryType::Periodic && self.boundaries.hi(ax) == BoundaryType::Periodic
    }

    /// whether `ax` closes into a ring across tiles: periodic and cut into more than one tile. one
    /// tile on a periodic axis spans the domain and wraps onto itself through its own faces.
    pub fn wraps(&self, ax: usize, counts: [usize; D]) -> bool {
        self.periodic(ax) && counts[ax] > 1
    }

    /// the halo topology of a partition with `counts` tiles per axis: a wrapping axis carries the
    /// schedule's end-to-end legs.
    pub fn topology(&self, counts: [usize; D]) -> Topology<D> {
        Topology::wrapping(std::array::from_fn(|ax| self.wraps(ax, counts)))
    }

    /// whether a partition with `counts` tiles per axis can carry this grid's declared boundaries.
    /// an axis periodic on one face only and cut into several tiles cannot: the end tile's periodic
    /// face refills from its own interior, the far side of the domain only when the axis is uncut.
    pub fn admits_cuts(&self, counts: [usize; D]) -> Result<(), String> {
        for ax in (0..D).filter(|&ax| counts[ax] > 1) {
            let periodic = [self.boundaries.lo(ax), self.boundaries.hi(ax)].map(|b| b == BoundaryType::Periodic);
            if periodic[0] != periodic[1] {
                return Err(format!(
                    "the domain is periodic on one face of axis {ax} and that axis is cut into {} tiles; \
                     a tile's periodic face refills from its own interior rather than the far side of \
                     the domain. declare both faces of the axis periodic, or neither",
                    counts[ax]
                ));
            }
        }
        Ok(())
    }

    /// the grid as its own single tile: offset zero, every cell, the declared boundaries.
    pub fn single(&self) -> PlacedTile<D> {
        PlacedTile { offset: [0; D], cells: self.cells, boundaries: self.boundaries.clone() }
    }

    /// the tile at grid coordinate `tc` of `partition`. the partition covers exactly this grid's
    /// cells.
    pub fn place(&self, partition: &Partition<D>, tc: [usize; D]) -> PlacedTile<D> {
        assert_eq!(
            partition.n_cells(),
            self.cells,
            "the partition covers {:?} cells and the grid holds {:?}",
            partition.n_cells(),
            self.cells
        );
        let counts = partition.counts();
        assert!((0..D).all(|ax| tc[ax] < counts[ax]), "tile {tc:?} lies outside a {counts:?} tile grid");
        let ext = partition.tile_extents(tc);
        let boundaries = Boundaries(std::array::from_fn(|ax| {
            let edge = |side: usize| {
                if self.wraps(ax, counts) {
                    BoundaryType::CoarseFine
                } else {
                    self.boundaries.0[ax][side]
                }
            };
            let lo = if tc[ax] == 0 { edge(0) } else { BoundaryType::CoarseFine };
            let hi = if tc[ax] + 1 == counts[ax] { edge(1) } else { BoundaryType::CoarseFine };
            [lo, hi]
        }));
        PlacedTile {
            offset: std::array::from_fn(|ax| ext[ax].0),
            cells: std::array::from_fn(|ax| ext[ax].1),
            boundaries,
        }
    }
}

impl<const D: usize> PlacedTile<D> {
    /// the tile's first cell as the signed whole-grid index the geometry carries.
    pub fn index_offset(&self) -> [isize; D] {
        std::array::from_fn(|ax| self.offset[ax] as isize)
    }

    /// the tile's lower corner: the whole grid's face below its first cell.
    pub fn lower(&self, grid: &WholeGrid<D>) -> [f64; D] {
        std::array::from_fn(|ax| grid.face(ax, self.offset[ax] as isize))
    }

    /// the tile's upper corner: the whole grid's face above its last cell, which is the next
    /// tile's lower corner bit for bit.
    pub fn upper(&self, grid: &WholeGrid<D>) -> [f64; D] {
        std::array::from_fn(|ax| grid.face(ax, (self.offset[ax] + self.cells[ax]) as isize))
    }

    /// the width of the tile's narrowest cell on each axis, from the whole grid's faces.
    pub fn narrowest_cell(&self, grid: &WholeGrid<D>) -> [f64; D] {
        std::array::from_fn(|ax| {
            let lo = self.offset[ax] as isize;
            (lo..lo + self.cells[ax] as isize)
                .map(|i| grid.face(ax, i + 1) - grid.face(ax, i))
                .fold(f64::INFINITY, f64::min)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use BoundaryType::{CoarseFine, Outflow, Periodic, Reflect};

    fn grid2(cells: [usize; 2], boundaries: [[BoundaryType; 2]; 2], maps: Option<[AxisMap; 2]>) -> WholeGrid<2> {
        WholeGrid { cells, origin: [1.5, -0.25], spacing: [1.0 / 48.0, 0.1], maps, boundaries: Boundaries(boundaries) }
    }

    // a ragged partition of a non-dyadic grid: offsets and cell counts are the cut positions
    // stated by hand, and the tiles cover the grid without gap or overlap
    #[test]
    fn a_ragged_partition_places_each_tile_at_its_cut() {
        let grid = grid2([48, 30], [[Outflow, Outflow], [Outflow, Outflow]], None);
        let partition = Partition::explicit([48, 30], [vec![13, 31], vec![11]]).unwrap();
        let want = [([0, 0], [13, 11]), ([13, 0], [18, 11]), ([31, 0], [17, 11]), ([0, 11], [13, 19])];
        for (tc, (offset, cells)) in [[0, 0], [1, 0], [2, 0], [0, 1]].into_iter().zip(want) {
            let tile = grid.place(&partition, tc);
            assert_eq!((tile.offset, tile.cells), (offset, cells), "tile {tc:?}");
        }
    }

    // a cut periodic axis closes through the schedule: both domain-seam faces are cuts and the
    // topology wraps; the open axis keeps its declared conditions at the domain edges
    #[test]
    fn a_cut_periodic_axis_turns_its_seam_faces_into_cuts() {
        let grid = grid2([48, 30], [[Periodic, Periodic], [Outflow, Reflect]], None);
        let partition = Partition::explicit([48, 30], [vec![13, 31], vec![11]]).unwrap();
        let first = grid.place(&partition, [0, 0]).boundaries;
        assert_eq!(first.0, [[CoarseFine, CoarseFine], [Outflow, CoarseFine]]);
        let last = grid.place(&partition, [2, 1]).boundaries;
        assert_eq!(last.0, [[CoarseFine, CoarseFine], [CoarseFine, Reflect]]);
        let ring = grid.topology(partition.counts());
        assert!(ring.is_periodic(0) && !ring.is_periodic(1));
    }

    // the single grid is the one-tile partition: offset zero, every cell, the declared boundaries
    // including an uncut periodic axis that wraps onto itself, and no wrapping leg
    #[test]
    fn one_tile_is_the_whole_grid() {
        let grid = grid2([48, 30], [[Periodic, Periodic], [Outflow, Reflect]], None);
        let partition = Partition::explicit([48, 30], [Vec::new(), Vec::new()]).unwrap();
        let tile = grid.place(&partition, [0, 0]);
        assert_eq!(tile, PlacedTile { offset: [0, 0], cells: [48, 30], boundaries: grid.boundaries.clone() });
        assert_eq!(grid.single(), tile);
        assert_eq!(tile.lower(&grid), grid.origin);
        assert_eq!(tile.upper(&grid), [grid.face(0, 48), grid.face(1, 30)]);
        let ring = grid.topology(partition.counts());
        assert!(!ring.is_periodic(0) && !ring.is_periodic(1));
    }

    // an azimuthal wedge declared with walls is cut like any open axis: its end tiles keep the
    // walls and nothing wraps, while a full periodic azimuth cut the same way closes into a ring
    #[test]
    fn a_non_wrapping_azimuth_keeps_its_walls() {
        let wedge = WholeGrid {
            cells: [8, 12, 30],
            origin: [1.0, 0.1, 0.0],
            spacing: [0.125, 0.25, 0.1],
            maps: None,
            boundaries: Boundaries([[Outflow, Outflow], [Reflect, Reflect], [Reflect, Reflect]]),
        };
        let partition = Partition::explicit([8, 12, 30], [Vec::new(), Vec::new(), vec![10, 20]]).unwrap();
        assert_eq!(wedge.place(&partition, [0, 0, 0]).boundaries.0[2], [Reflect, CoarseFine]);
        assert_eq!(wedge.place(&partition, [0, 0, 2]).boundaries.0[2], [CoarseFine, Reflect]);
        assert!(!wedge.topology(partition.counts()).is_periodic(2));
        let full = WholeGrid { boundaries: Boundaries([[Outflow, Outflow], [Reflect, Reflect], [Periodic, Periodic]]), ..wedge };
        assert_eq!(full.place(&partition, [0, 0, 0]).boundaries.0[2], [CoarseFine, CoarseFine]);
        assert!(full.topology(partition.counts()).is_periodic(2));
    }

    // a periodic declaration on one face only leaves the axis open: the end tile keeps the face
    // as declared and no ring closes
    #[test]
    fn a_one_sided_periodic_declaration_does_not_wrap() {
        let grid = grid2([48, 30], [[Periodic, Outflow], [Outflow, Outflow]], None);
        let partition = Partition::explicit([48, 30], [vec![24], Vec::new()]).unwrap();
        assert_eq!(grid.place(&partition, [0, 0]).boundaries.lo(0), Periodic);
        assert_eq!(grid.place(&partition, [1, 0]).boundaries.hi(0), Outflow);
        assert!(!grid.topology(partition.counts()).is_periodic(0));
    }

    // on a log radial axis the tile corners are r_lo 10^(g s) at the cut index g, stated
    // analytically, and adjacent tiles abut bit for bit
    #[test]
    fn log_axis_corners_follow_the_whole_grid_map() {
        let (r_lo, r_hi, n) = (1.5_f64, 400.0_f64, 48usize);
        let slope = (r_hi / r_lo).log10() / n as f64;
        let maps = [AxisMap::Log { start: r_lo, log_slope: slope }, AxisMap::Uniform { start: -0.25, dx: 0.1 }];
        let grid = grid2([n, 30], [[Outflow, Outflow], [Outflow, Outflow]], Some(maps));
        let partition = Partition::explicit([n, 30], [vec![13, 31], Vec::new()]).unwrap();
        let tiles: Vec<_> = (0..3).map(|t| grid.place(&partition, [t, 0])).collect();
        for (tile, g) in tiles.iter().zip([0usize, 13, 31]) {
            let want = r_lo * 10.0_f64.powf(g as f64 * slope);
            let got = tile.lower(&grid)[0];
            assert!((got - want).abs() <= 4.0 * f64::EPSILON * want, "tile at {g}: {got} against {want}");
        }
        assert!((tiles[2].upper(&grid)[0] - r_hi).abs() <= 4.0 * f64::EPSILON * r_hi);
        for pair in tiles.windows(2) {
            assert_eq!(pair[0].upper(&grid)[0].to_bits(), pair[1].lower(&grid)[0].to_bits());
        }
        // the narrowest cell of a tile on a log axis is its innermost one
        let inner = grid.face(0, 14) - grid.face(0, 13);
        assert_eq!(tiles[1].narrowest_cell(&grid)[0], inner);
    }

    // without maps a face sits at x_lo + i dx on the whole grid's index, the expression the host
    // geometry evaluates; a corner built from a per-tile origin differs from it on this spacing
    #[test]
    fn uniform_corners_use_the_whole_grid_index() {
        let grid = grid2([48, 30], [[Outflow, Outflow], [Outflow, Outflow]], None);
        let partition = Partition::explicit([48, 30], [vec![1, 2], Vec::new()]).unwrap();
        let tile = grid.place(&partition, [1, 0]);
        let upper = tile.upper(&grid)[0];
        assert_eq!(upper.to_bits(), (1.5 + 2.0 * (1.0 / 48.0_f64)).to_bits());
        let per_tile = (1.5 + 1.0 * (1.0 / 48.0_f64)) + 1.0 * (1.0 / 48.0_f64);
        assert_ne!(per_tile.to_bits(), upper.to_bits(), "premise: the per-tile corner must differ here");
    }

    // a one-sided periodic axis is carried uncut and refused cut; a two-sided one is carried cut
    #[test]
    fn a_one_sided_periodic_axis_admits_no_cut() {
        let one_sided = grid2([48, 30], [[Periodic, Outflow], [Outflow, Outflow]], None);
        assert!(one_sided.admits_cuts([1, 2]).is_ok(), "the cut misses the periodic axis");
        let refusal = one_sided.admits_cuts([2, 1]).unwrap_err();
        assert!(refusal.contains("periodic on one face of axis 0"), "{refusal}");
        let both = grid2([48, 30], [[Periodic, Periodic], [Outflow, Outflow]], None);
        assert!(both.admits_cuts([3, 2]).is_ok());
    }

    #[test]
    #[should_panic(expected = "the partition covers")]
    fn a_partition_of_another_grid_is_refused() {
        let grid = grid2([48, 30], [[Outflow, Outflow], [Outflow, Outflow]], None);
        let partition = Partition::explicit([48, 31], [Vec::new(), Vec::new()]).unwrap();
        let _ = grid.place(&partition, [0, 0]);
    }
}
