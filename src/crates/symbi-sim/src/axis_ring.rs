// =============================================================================
// axis_ring.rs
//
// the shared axis edges of a 3D curvilinear grid whose coordinate axis lies on
// the grid: the spherical poles (theta = 0, theta = pi) and the cylindrical
// R = 0 line. every azimuthal copy of an edge lying on the axis is one physical
// segment, so the constrained-transport curl preserves the discrete magnetic
// flux of the cells that touch the axis only when all copies carry one EMF:
// summed over such a cell's faces, the curl leaves dl (E_k+1 - E_k) from the
// two copies bounding its zero-area axis face.
//
// each ring (one pole, one index along the axis) is reconciled to the
// sector-weighted quadrature of its copies around the axis,
//   E = sum_k w_k E_k / sum_k w_k,  w_k = (dphi_{k-1} + dphi_k) / 2,
// with k running over the distinct copies in ascending azimuth. the copy at phi_k serves the two cells beside it, so w_k is its
// share of the ring; on uniform azimuthal spacing every w_k is dphi. the value is
// written to every copy, the closing copy at phi = 2 pi included.
//
// the sum runs in ascending azimuth whatever the partition, so a grid cut
// through its azimuth reconciles bit for bit as the undivided grid does: the
// tiles hand their owned copies (a copy is owned by the tile owning the cell on
// its +phi side) to one assembly that matches them by position.
//
// usage:
//   let rings = axis_rings(store);            // the rings this tile touches
//   reconcile_whole_rings(store);             // a tile holding every copy of its rings
//   reconcile_across(&stores);                // tiles sharing rings cut through the azimuth
//   restore_rings_from(store, &efield_ho);    // keep the reconciled ring through a redo
// =============================================================================

use std::collections::BTreeMap;

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_geometry::Geometry;
use symbi_xpu::MemorySpace;

use crate::state::{EfieldFields, FieldStore, Loc};

/// one ring of axis edges on a tile: the copies of the edges along the axis at one pole, over this
/// tile's cells along the axis and around it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisRing {
    /// the edge-EMF slot of the edges lying along the axis
    pub slot: usize,
    /// the grid axis normal to the zero-area face on the axis, and that face's local index
    pub pole_axis: usize,
    pub pole_face: isize,
    /// which end of the pole axis the ring lies at, 0 low and 1 high
    pub pole_side: u8,
    /// the azimuthal grid axis and this tile's local cells on it
    pub ring_axis: usize,
    pub ring_cells: (isize, isize),
    /// the grid axis running along the coordinate axis and this tile's local cells on it
    pub along_axis: usize,
    pub along_cells: (isize, isize),
}

/// the rings of axis edges a tile touches: none unless the grid is a 3D spherical or cylindrical
/// grid carrying a staggered field whose interior reaches a pole or the R = 0 line
pub fn axis_rings<const D: usize, const DOF: usize, Mem, Sc>(store: &FieldStore<D, DOF, Mem, Sc>) -> Vec<AxisRing>
where
    Mem: symbi_xpu::MemoryStorage,
    Sc: Scalar + OrderedNumeric,
{
    let g = &store.geom;
    if D != 3 || store.fields.mhd.is_none() || (0..D).any(|ax| g.axes[ax] != ax) {
        return Vec::new();
    }
    // (pole axis, azimuthal axis, axis-running axis): theta, phi, r on the sphere; R, phi, z on
    // the cylinder, where only the low R face can lie on the axis
    let (pole_axis, ring_axis, along_axis, sides): (usize, usize, usize, &[u8]) = match g.coords {
        Geometry::Spherical => (1, 2, 0, &[0, 1]),
        Geometry::Cylindrical => (0, 1, 2, &[0]),
        _ => return Vec::new(),
    };
    let int = &g.interior;
    let ring_cells = (int.spaces[ring_axis].lo, int.spaces[ring_axis].hi);
    let along_cells = (int.spaces[along_axis].lo, int.spaces[along_axis].hi);
    sides
        .iter()
        .filter_map(|&side| {
            let pole_face = if side == 0 { int.spaces[pole_axis].lo } else { int.spaces[pole_axis].hi };
            let mut idx = [0isize; D];
            idx[pole_axis] = pole_face;
            idx[ring_axis] = ring_cells.0;
            idx[along_axis] = along_cells.0;
            let loc: [Loc; D] = std::array::from_fn(|ax| if ax == pole_axis { Loc::Face } else { Loc::Center });
            let x = g.stagger_coord(idx, loc)[pole_axis];
            // the axis is detected on the face position itself: a polar bound computed as pi
            // carries sin(theta) at roundoff
            let on_axis = match g.coords {
                Geometry::Spherical => x.sin() * x.sin() < 1.0e-18,
                _ => x * x < 1.0e-24,
            };
            on_axis.then_some(AxisRing {
                slot: along_axis,
                pole_axis,
                pole_face,
                pole_side: side,
                ring_axis,
                ring_cells,
                along_axis,
                along_cells,
            })
        })
        .collect()
}

/// the reconciled EMF of one ring from its distinct copies in ascending global azimuthal order,
/// each given as (the width of the cell on its +phi side, its EMF)
pub fn ring_value(copies: &[(f64, f64)]) -> f64 {
    let n = copies.len();
    let mut num = 0.0;
    let mut den = 0.0;
    for kk in 0..n {
        let w = 0.5 * (copies[(kk + n - 1) % n].0 + copies[kk].0);
        num += w * copies[kk].1;
        den += w;
    }
    num / den
}

fn edge_index<const D: usize>(ring: &AxisRing, along: isize, k: isize) -> [isize; D] {
    let mut idx = [0isize; D];
    idx[ring.pole_axis] = ring.pole_face;
    idx[ring.along_axis] = along;
    idx[ring.ring_axis] = k;
    idx
}

/// the ring passes read and write the edge EMF from the host: host memory, or the unified and
/// managed device memory, after the device's queued kernels have landed their writes
fn require_host<Mem: MemorySpace>() {
    assert!(
        Mem::IS_HOST_ACCESSIBLE,
        "the shared axis-edge EMF is reconciled through host-accessible memory"
    );
    #[cfg(feature = "gpu")]
    if Mem::IS_DEVICE_ACCESSIBLE {
        symbi_xpu::ctx_sync();
    }
}

/// whether this tile holds every azimuthal copy of its rings: its azimuthal cells span 2 pi
fn holds_whole_ring<const D: usize, const DOF: usize, Mem, Sc>(store: &FieldStore<D, DOF, Mem, Sc>, ring: &AxisRing) -> bool
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let span: f64 = (ring.ring_cells.0..ring.ring_cells.1)
        .map(|k| {
            let mut idx = [0isize; D];
            idx[ring.ring_axis] = k;
            store.geom.cell_width(idx, ring.ring_axis)
        })
        .sum();
    (span - std::f64::consts::TAU).abs() <= 1.0e-9 * std::f64::consts::TAU
}

/// whether the tile holds every copy of each axis ring it touches (true for a tile touching none)
pub fn holds_whole_rings<const D: usize, const DOF: usize, Mem, Sc>(store: &FieldStore<D, DOF, Mem, Sc>) -> bool
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    axis_rings(store).iter().all(|ring| holds_whole_ring(store, ring))
}

/// reconcile every ring of a tile that holds each ring whole: the undivided grid, or a tile cut
/// only along the axis or away from it. a tile holding part of a ring panics, since its copies
/// can only be reconciled together with the other tiles' (`reconcile_across`)
pub fn reconcile_whole_rings<const D: usize, const DOF: usize, Mem, Sc>(store: &FieldStore<D, DOF, Mem, Sc>)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let rings = axis_rings(store);
    if rings.is_empty() {
        return;
    }
    require_host::<Mem>();
    let mhd = store.fields.mhd.as_ref().expect("axis rings carry a staggered field");
    for ring in &rings {
        assert!(
            holds_whole_ring(store, ring),
            "a tile holding part of an axis ring reconciles it only across tiles; this driver \
             folds each tile's stage whole"
        );
        let e = &mhd.efield[ring.slot];
        for along in ring.along_cells.0..ring.along_cells.1 {
            let copies: Vec<(f64, f64)> = (ring.ring_cells.0..ring.ring_cells.1)
                .map(|k| {
                    let mut widx = [0isize; D];
                    widx[ring.ring_axis] = k;
                    (store.geom.cell_width(widx, ring.ring_axis), e.at(edge_index::<D>(ring, along, k)).to_f64())
                })
                .collect();
            let value = Sc::from_f64(ring_value(&copies));
            for k in ring.ring_cells.0..=ring.ring_cells.1 {
                e.set(edge_index::<D>(ring, along, k), value);
            }
        }
    }
}

/// a physical coordinate as an exact ordering key: a tile placed by its own origin and one placed
/// by an index offset put a shared position a few ulp apart, and 1e-9 separates every distinct
/// cell of any admitted grid
fn position_key(x: f64) -> i64 {
    (x * 1.0e9).round() as i64
}

/// reconcile the rings the tiles share, each ring assembled from the copies its tiles own (a copy
/// is owned by the tile owning the cell on its +phi side) in ascending azimuth and written to
/// every tile's copies, so the result is the undivided grid's bit for bit. rings and copies are
/// matched across tiles by position, which holds however a tile was placed on the grid
pub fn reconcile_across<const D: usize, const DOF: usize, Mem, Sc>(stores: &[&FieldStore<D, DOF, Mem, Sc>])
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let per_tile: Vec<Vec<AxisRing>> = stores.iter().map(|s| axis_rings(*s)).collect();
    if per_tile.iter().all(Vec::is_empty) {
        return;
    }
    require_host::<Mem>();
    let along_key = |store: &FieldStore<D, DOF, Mem, Sc>, ring: &AxisRing, along: isize| {
        let mut idx = [0isize; D];
        idx[ring.along_axis] = along;
        (ring.pole_side, position_key(store.geom.cell_coord(idx)[ring.along_axis]))
    };
    // (pole side, position along the axis) -> azimuth of the copy -> (width, EMF)
    let mut rings: BTreeMap<(u8, i64), BTreeMap<i64, (f64, f64)>> = BTreeMap::new();
    for (store, tile_rings) in stores.iter().zip(&per_tile) {
        let mhd = store.fields.mhd.as_ref().expect("axis rings carry a staggered field");
        for ring in tile_rings {
            let e = &mhd.efield[ring.slot];
            for along in ring.along_cells.0..ring.along_cells.1 {
                let copies = rings.entry(along_key(store, ring, along)).or_default();
                for k in ring.ring_cells.0..ring.ring_cells.1 {
                    let mut idx = [0isize; D];
                    idx[ring.ring_axis] = k;
                    let azimuth = store.geom.stagger_coord(idx, [Loc::Face; D])[ring.ring_axis];
                    let copy = (store.geom.cell_width(idx, ring.ring_axis), e.at(edge_index::<D>(ring, along, k)).to_f64());
                    let previous = copies.insert(position_key(azimuth), copy);
                    assert!(previous.is_none(), "two tiles own one axis-edge copy at azimuth {azimuth}");
                }
            }
        }
    }
    let values: BTreeMap<(u8, i64), f64> = rings
        .into_iter()
        .map(|(key, copies)| {
            let ordered: Vec<(f64, f64)> = copies.into_values().collect();
            let span: f64 = ordered.iter().map(|c| c.0).sum();
            assert!(
                (span - std::f64::consts::TAU).abs() <= 1.0e-9 * std::f64::consts::TAU,
                "the tiles hold only part of the axis ring {key:?}: its copies span {span}"
            );
            (key, ring_value(&ordered))
        })
        .collect();
    for (store, tile_rings) in stores.iter().zip(&per_tile) {
        let mhd = store.fields.mhd.as_ref().expect("axis rings carry a staggered field");
        for ring in tile_rings {
            let e = &mhd.efield[ring.slot];
            for along in ring.along_cells.0..ring.along_cells.1 {
                let value = Sc::from_f64(values[&along_key(store, ring, along)]);
                for k in ring.ring_cells.0..=ring.ring_cells.1 {
                    e.set(edge_index::<D>(ring, along, k), value);
                }
            }
        }
    }
}

/// write every ring copy of `src` into the live edge EMF: a redo that recomputes the stage EMF
/// keeps the reconciled axis value the stage accepted
pub fn restore_rings_from<const D: usize, const DOF: usize, Mem, Sc>(store: &FieldStore<D, DOF, Mem, Sc>, src: &EfieldFields<D, Mem, Sc>)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let rings = axis_rings(store);
    if rings.is_empty() {
        return;
    }
    require_host::<Mem>();
    let mhd = store.fields.mhd.as_ref().expect("axis rings carry a staggered field");
    for ring in &rings {
        for along in ring.along_cells.0..ring.along_cells.1 {
            for k in ring.ring_cells.0..=ring.ring_cells.1 {
                let idx = edge_index::<D>(ring, along, k);
                mhd.efield[ring.slot].set(idx, *src[ring.slot].at(idx));
            }
        }
    }
}

/// the length of the global ring vector over a grid of `n_cells`: both pole sides, every cell
/// along the axis, every azimuthal copy
pub fn ring_vector_len<const D: usize>(n_cells: [usize; D], coords: Geometry) -> u64 {
    match coords {
        Geometry::Spherical if D == 3 => (2 * n_cells[0] * n_cells[2]) as u64,
        Geometry::Cylindrical if D == 3 => (2 * n_cells[2] * n_cells[1]) as u64,
        _ => 0,
    }
}

fn ring_vector_index<const D: usize>(n_cells: [usize; D], ring: &AxisRing, along: isize, k: isize) -> u64 {
    let n_along = n_cells[ring.along_axis] as u64;
    let n_ring = n_cells[ring.ring_axis] as u64;
    (u64::from(ring.pole_side) * n_along + along as u64) * n_ring + k as u64
}

/// this tile's owned axis-edge copies as (global index, EMF bits) entries of the global ring
/// vector over `n_cells`, for tiles placed on the whole grid by an index offset: every tile that
/// owns a copy supplies it, and no two tiles supply one
pub fn ring_entries<const D: usize, const DOF: usize, Mem, Sc>(store: &FieldStore<D, DOF, Mem, Sc>, n_cells: [usize; D]) -> Vec<(u64, u64)>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let rings = axis_rings(store);
    if rings.is_empty() {
        return Vec::new();
    }
    require_host::<Mem>();
    let mhd = store.fields.mhd.as_ref().expect("axis rings carry a staggered field");
    let off = store.geom.index_offset;
    let mut entries = Vec::new();
    for ring in &rings {
        let e = &mhd.efield[ring.slot];
        for along in ring.along_cells.0..ring.along_cells.1 {
            for k in ring.ring_cells.0..ring.ring_cells.1 {
                let index = ring_vector_index(n_cells, ring, along + off[ring.along_axis], k + off[ring.ring_axis]);
                entries.push((index, e.at(edge_index::<D>(ring, along, k)).to_f64().to_bits()));
            }
        }
    }
    entries
}

/// write each of this tile's rings its reconciled EMF from the assembled global ring vector: the
/// copies in ascending global azimuth with the widths of the whole grid's map, so every tile
/// holding a copy computes the undivided grid's value bit for bit
pub fn apply_assembled<const D: usize, const DOF: usize, Mem, Sc>(store: &FieldStore<D, DOF, Mem, Sc>, n_cells: [usize; D], assembled: &[u64], absent: u64)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let rings = axis_rings(store);
    if rings.is_empty() {
        return;
    }
    require_host::<Mem>();
    let mhd = store.fields.mhd.as_ref().expect("axis rings carry a staggered field");
    let off = store.geom.index_offset;
    for ring in &rings {
        let e = &mhd.efield[ring.slot];
        let n_ring = n_cells[ring.ring_axis] as isize;
        for along in ring.along_cells.0..ring.along_cells.1 {
            let copies: Vec<(f64, f64)> = (0..n_ring)
                .map(|kg| {
                    let bits = assembled[ring_vector_index(n_cells, ring, along + off[ring.along_axis], kg) as usize];
                    assert!(bits != absent, "no worker supplied the axis-edge copy {kg} of a ring this tile holds");
                    let mut widx = [0isize; D];
                    widx[ring.ring_axis] = kg - off[ring.ring_axis];
                    (store.geom.cell_width(widx, ring.ring_axis), f64::from_bits(bits))
                })
                .collect();
            let value = Sc::from_f64(ring_value(&copies));
            for k in ring.ring_cells.0..=ring.ring_cells.1 {
                e.set(edge_index::<D>(ring, along, k), value);
            }
        }
    }
}
