// =============================================================================
// atlas.rs
//
// the index-space description of a decomposed level that a worker needs to
// exchange halos with tiles it does not hold: per-tile layouts derived from the
// partition, per-field layouts derived from each field's centering, and the
// compiled exchange plan of region moves in axis-phase order. everything is
// plain data over index spans; no field store is read to compile a plan, so a
// process holding one tile derives the same plan as a process holding them all.
//
// two digests answer two different questions. the plan digest covers the
// numerical plan (partition, topology, reach, schema, layouts, transfers) and is
// independent of placement. the placement digest covers who holds which tile.
//
// usage:
//  let schema = FieldSchema::new().cell("rho").cell("v1").cell("v2").cell("p");
//  let plan = ExchangePlan::compile(&partition, &Topology::open(), &schema, ng)?;
//  let placement = Placement::new(2, vec![WorkerId(0), WorkerId(0), WorkerId(1), WorkerId(1)])?;
// =============================================================================

use crate::decomp::{Partition, Schedule, Topology, leg_regions, unflatten};
use crate::state::{PartitionGeometry, axis_name};
use symbi_algebra::{Domain, Space};
pub use symbi_fabric::{Digest, Epoch, Fnv, TransferId, WorkerId};

/// a tile of the level, numbered flat over the tile grid (`decomp::flatten`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId(pub u32);

/// a closed-open index span on one axis in a tile's local index space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Span {
    pub lo: isize,
    pub hi: isize,
}

impl Span {
    pub fn size(&self) -> usize {
        (self.hi - self.lo).max(0) as usize
    }

    pub fn of(space: &Space) -> Self {
        Self {
            lo: space.lo,
            hi: space.hi,
        }
    }

    pub fn contains(&self, other: &Span) -> bool {
        other.lo >= self.lo && other.hi <= self.hi
    }

    pub fn overlaps(&self, other: &Span) -> bool {
        self.lo < other.hi && other.lo < self.hi
    }
}

pub fn spans_of<const D: usize>(domain: &Domain<D>) -> [Span; D] {
    std::array::from_fn(|ax| Span::of(&domain.spaces[ax]))
}

pub fn domain_of<const D: usize>(spans: &[Span; D]) -> Domain<D> {
    Domain::new(std::array::from_fn(|ax| Space {
        name: axis_name(ax),
        lo: spans[ax].lo,
        hi: spans[ax].hi,
    }))
}

fn volume<const D: usize>(spans: &[Span; D]) -> usize {
    spans.iter().map(Span::size).product()
}

fn boxes_overlap<const D: usize>(a: &[Span; D], b: &[Span; D]) -> bool {
    (0..D).all(|ax| a[ax].overlaps(&b[ax]))
}

/// a tile's cell layout: interior and allocated span per axis in its local
/// index space, the cell halo width, and the global index of its first
/// interior cell. a tile built by the simulation builder has its interior at
/// zero, so the layout is a function of the partition and ng alone;
/// `of_geometry` reads the same numbers off a built tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileLayout<const D: usize> {
    pub tile: TileId,
    pub interior: [Span; D],
    pub allocated: [Span; D],
    pub ng: usize,
    pub offset: [isize; D],
}

impl<const D: usize> TileLayout<D> {
    pub fn of_partition(partition: &Partition<D>, tile: TileId, ng: usize) -> Self {
        let tc = unflatten(tile.0 as usize, partition.counts());
        let ext = partition.tile_extents(tc);
        let ng_i = ng as isize;
        Self {
            tile,
            interior: std::array::from_fn(|ax| Span {
                lo: 0,
                hi: ext[ax].1 as isize,
            }),
            allocated: std::array::from_fn(|ax| Span {
                lo: -ng_i,
                hi: ext[ax].1 as isize + ng_i,
            }),
            ng,
            offset: std::array::from_fn(|ax| ext[ax].0 as isize),
        }
    }

    pub fn of_geometry(tile: TileId, geom: &PartitionGeometry<D>, offset: [isize; D]) -> Self {
        Self {
            tile,
            interior: spans_of(&geom.interior),
            allocated: spans_of(&geom.allocated),
            ng: geom.ng,
            offset,
        }
    }
}

/// where a field's values sit. a cell field lives on the tile's cell domain; a
/// face field on axis `d` holds one value per face normal to `d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Cell,
    Face(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Element {
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldEntry {
    pub name: String,
    pub kind: FieldKind,
    pub element: Element,
}

/// the fields an exchange carries, in wire order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FieldSchema {
    pub entries: Vec<FieldEntry>,
}

impl FieldSchema {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cell(mut self, name: &str) -> Self {
        self.entries.push(FieldEntry {
            name: name.to_string(),
            kind: FieldKind::Cell,
            element: Element::F64,
        });
        self
    }

    pub fn face(mut self, name: &str, axis: u8) -> Self {
        self.entries.push(FieldEntry {
            name: name.to_string(),
            kind: FieldKind::Face(axis),
            element: Element::F64,
        });
        self
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// one field's own domain on one tile. a cell field shares the tile layout
/// and carries the cell halo on every axis. a face field on axis `d` extends
/// the interior by one face on `d`, carries no halo on `d`, and carries a
/// two-cell halo on every transverse axis whatever the cell halo is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldLayout<const D: usize> {
    pub interior: [Span; D],
    pub allocated: [Span; D],
    pub halo: [usize; D],
}

impl<const D: usize> FieldLayout<D> {
    pub const FACE_TRANSVERSE_HALO: usize = 2;

    pub fn of(tile: &TileLayout<D>, kind: FieldKind) -> Self {
        match kind {
            FieldKind::Cell => Self {
                interior: tile.interior,
                allocated: tile.allocated,
                halo: [tile.ng; D],
            },
            FieldKind::Face(d) => {
                let d = d as usize;
                let t = Self::FACE_TRANSVERSE_HALO as isize;
                let interior: [Span; D] = std::array::from_fn(|ax| Span {
                    lo: tile.interior[ax].lo,
                    hi: tile.interior[ax].hi + if ax == d { 1 } else { 0 },
                });
                Self {
                    interior,
                    allocated: std::array::from_fn(|ax| {
                        if ax == d {
                            interior[ax]
                        } else {
                            Span {
                                lo: interior[ax].lo - t,
                                hi: interior[ax].hi + t,
                            }
                        }
                    }),
                    halo: std::array::from_fn(|ax| {
                        if ax == d {
                            0
                        } else {
                            Self::FACE_TRANSVERSE_HALO
                        }
                    }),
                }
            }
        }
    }

    pub fn cell_of_geometry(geom: &PartitionGeometry<D>) -> Self {
        Self::of(
            &TileLayout::of_geometry(TileId(0), geom, [0; D]),
            FieldKind::Cell,
        )
    }

    pub fn face_of_geometry(geom: &PartitionGeometry<D>, d: usize) -> Self {
        Self::of(
            &TileLayout::of_geometry(TileId(0), geom, [0; D]),
            FieldKind::Face(d as u8),
        )
    }

    pub fn allocated_domain(&self) -> Domain<D> {
        domain_of(&self.allocated)
    }
}

/// the position of an exchange inside a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExchangePoint {
    Prime,
    Stage(u8),
    Rollback,
}

impl ExchangePoint {
    pub const ROLLBACK_WIRE: u8 = 0xF0;

    pub fn to_wire(self) -> u8 {
        match self {
            ExchangePoint::Prime => 0,
            ExchangePoint::Stage(k) => 1 + k,
            ExchangePoint::Rollback => Self::ROLLBACK_WIRE,
        }
    }

    pub fn from_wire(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(ExchangePoint::Prime),
            Self::ROLLBACK_WIRE => Some(ExchangePoint::Rollback),
            Epoch::NO_POINT => None,
            k if k < Self::ROLLBACK_WIRE => Some(ExchangePoint::Stage(k - 1)),
            _ => None,
        }
    }
}

/// one region move: `len` values of field `field` from `src_region` of tile
/// `src` to `dst_region` of tile `dst`, both in the tiles' local index spaces
/// and of equal shape. ids run in plan order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transfer<const D: usize> {
    pub id: TransferId,
    pub axis: u8,
    pub src: TileId,
    pub dst: TileId,
    pub field: u16,
    pub src_region: [Span; D],
    pub dst_region: [Span; D],
    pub len: usize,
}

/// the transfers of one axis, in order. an axis closes before the next opens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisPhase<const D: usize> {
    pub axis: u8,
    pub transfers: Vec<Transfer<D>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    Partition(String),
    ZeroVolume { transfer: TransferId },
    ShapeMismatch { transfer: TransferId },
    OutsideAllocated { transfer: TransferId },
    OverlappingDestinations { a: TransferId, b: TransferId },
    PolarSeam,
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::Partition(detail) => write!(f, "partition: {detail}"),
            PlanError::ZeroVolume { transfer } => write!(f, "transfer {transfer:?} moves no cells"),
            PlanError::ShapeMismatch { transfer } => {
                write!(
                    f,
                    "transfer {transfer:?} has unequal source and destination shapes"
                )
            }
            PlanError::OutsideAllocated { transfer } => {
                write!(
                    f,
                    "transfer {transfer:?} leaves its field's allocated domain"
                )
            }
            PlanError::OverlappingDestinations { a, b } => {
                write!(
                    f,
                    "transfers {a:?} and {b:?} write overlapping destinations"
                )
            }
            PlanError::PolarSeam => write!(f, "polar seams are outside the exchange plan"),
        }
    }
}

impl std::error::Error for PlanError {}

/// the compiled exchange of one level: the tile and field layouts and every
/// region move, grouped by axis in the order the axes close.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExchangePlan<const D: usize> {
    pub reach: usize,
    pub tiles: Vec<TileLayout<D>>,
    /// indexed `[tile][field]`
    pub fields: Vec<Vec<FieldLayout<D>>>,
    pub schema: FieldSchema,
    pub axes: Vec<AxisPhase<D>>,
    pub digest: Digest,
}

impl<const D: usize> ExchangePlan<D> {
    pub fn compile(
        partition: &Partition<D>,
        topology: &Topology<D>,
        schema: &FieldSchema,
        ng: usize,
    ) -> Result<Self, PlanError> {
        let counts = partition.counts();
        let schedule = Schedule::derive(counts, ng, topology);
        if schedule.seam().is_some() {
            return Err(PlanError::PolarSeam);
        }
        let tiles: Vec<TileLayout<D>> = (0..partition.n_tiles())
            .map(|t| TileLayout::of_partition(partition, TileId(t as u32), ng))
            .collect();
        let fields: Vec<Vec<FieldLayout<D>>> = tiles
            .iter()
            .map(|t| {
                schema
                    .entries
                    .iter()
                    .map(|e| FieldLayout::of(t, e.kind))
                    .collect()
            })
            .collect();
        let mut axes: Vec<AxisPhase<D>> = (0..D)
            .map(|ax| AxisPhase {
                axis: ax as u8,
                transfers: Vec::new(),
            })
            .collect();
        let mut next = 0u32;
        for leg in schedule.legs() {
            for (f, entry) in schema.entries.iter().enumerate() {
                // a face field's own axis is the shared interface: the CT curl output on
                // both tiles, never a halo.
                if let FieldKind::Face(d) = entry.kind {
                    if d as usize == leg.axis {
                        continue;
                    }
                }
                let lo = &fields[leg.lo][f];
                let hi = &fields[leg.hi][f];
                let h = lo.halo[leg.axis].min(hi.halo[leg.axis]).min(ng);
                let r = leg_regions(lo, hi, leg, h);
                let mut push =
                    |src: usize, src_region: &Domain<D>, dst: usize, dst_region: &Domain<D>| {
                        let src_region = spans_of(src_region);
                        let dst_region = spans_of(dst_region);
                        axes[leg.axis].transfers.push(Transfer {
                            id: TransferId(next),
                            axis: leg.axis as u8,
                            src: TileId(src as u32),
                            dst: TileId(dst as u32),
                            field: f as u16,
                            len: volume(&dst_region),
                            src_region,
                            dst_region,
                        });
                        next += 1;
                    };
                push(leg.hi, &r.hi_src, leg.lo, &r.lo_ghost);
                push(leg.lo, &r.lo_src, leg.hi, &r.hi_ghost);
            }
        }
        let plan = Self {
            reach: ng,
            tiles,
            fields,
            schema: schema.clone(),
            axes,
            digest: Digest(0),
        };
        plan.validate()?;
        let digest = plan.compute_digest(counts, topology);
        Ok(Self { digest, ..plan })
    }

    fn validate(&self) -> Result<(), PlanError> {
        for t in self.transfers() {
            if t.len == 0 {
                return Err(PlanError::ZeroVolume { transfer: t.id });
            }
            if (0..D).any(|ax| t.src_region[ax].size() != t.dst_region[ax].size()) {
                return Err(PlanError::ShapeMismatch { transfer: t.id });
            }
            let src_alloc = &self.fields[t.src.0 as usize][t.field as usize].allocated;
            let dst_alloc = &self.fields[t.dst.0 as usize][t.field as usize].allocated;
            let inside = (0..D).all(|ax| {
                src_alloc[ax].contains(&t.src_region[ax])
                    && dst_alloc[ax].contains(&t.dst_region[ax])
            });
            if !inside {
                return Err(PlanError::OutsideAllocated { transfer: t.id });
            }
        }
        let all: Vec<&Transfer<D>> = self.transfers().collect();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                if a.dst == b.dst
                    && a.field == b.field
                    && boxes_overlap(&a.dst_region, &b.dst_region)
                {
                    return Err(PlanError::OverlappingDestinations { a: a.id, b: b.id });
                }
            }
        }
        Ok(())
    }

    fn compute_digest(&self, counts: [usize; D], topology: &Topology<D>) -> Digest {
        let mut h = Fnv::new();
        h.u32(D as u32);
        h.u64(self.reach as u64);
        for ax in 0..D {
            h.u64(counts[ax] as u64);
            h.u8(u8::from(topology.is_periodic(ax)));
        }
        h.u32(self.tiles.len() as u32);
        for t in &self.tiles {
            h.u32(t.tile.0);
            h.u64(t.ng as u64);
            for ax in 0..D {
                h.i64(t.interior[ax].lo as i64);
                h.i64(t.interior[ax].hi as i64);
                h.i64(t.allocated[ax].lo as i64);
                h.i64(t.allocated[ax].hi as i64);
                h.i64(t.offset[ax] as i64);
            }
        }
        h.u32(self.schema.len() as u32);
        for e in &self.schema.entries {
            h.str(&e.name);
            match e.kind {
                FieldKind::Cell => h.u8(0),
                FieldKind::Face(d) => {
                    h.u8(1);
                    h.u8(d);
                }
            }
            h.u8(0);
        }
        for per_tile in &self.fields {
            for f in per_tile {
                for ax in 0..D {
                    h.i64(f.interior[ax].lo as i64);
                    h.i64(f.interior[ax].hi as i64);
                    h.i64(f.allocated[ax].lo as i64);
                    h.i64(f.allocated[ax].hi as i64);
                    h.u64(f.halo[ax] as u64);
                }
            }
        }
        for t in self.transfers() {
            h.u32(t.id.0);
            h.u8(t.axis);
            h.u32(t.src.0);
            h.u32(t.dst.0);
            h.u16(t.field);
            h.u64(t.len as u64);
            for ax in 0..D {
                h.i64(t.src_region[ax].lo as i64);
                h.i64(t.src_region[ax].hi as i64);
                h.i64(t.dst_region[ax].lo as i64);
                h.i64(t.dst_region[ax].hi as i64);
            }
        }
        h.finish()
    }

    /// every transfer in plan order.
    pub fn transfers(&self) -> impl Iterator<Item = &Transfer<D>> {
        self.axes.iter().flat_map(|a| a.transfers.iter())
    }

    pub fn n_tiles(&self) -> usize {
        self.tiles.len()
    }
}

/// which worker holds each tile. every worker holds at least one tile and every
/// tile has exactly one holder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    pub workers: u32,
    pub owner: Vec<WorkerId>,
}

impl Placement {
    pub fn new(workers: u32, owner: Vec<WorkerId>) -> Result<Self, String> {
        if workers == 0 {
            return Err("a placement needs at least one worker".into());
        }
        if let Some(bad) = owner.iter().find(|w| w.0 >= workers) {
            return Err(format!(
                "tile owner {bad:?} is outside the {workers} workers"
            ));
        }
        for w in 0..workers {
            if !owner.iter().any(|o| o.0 == w) {
                return Err(format!("worker {w} holds no tile"));
            }
        }
        Ok(Self { workers, owner })
    }

    pub fn digest(&self) -> Digest {
        let mut h = Fnv::new();
        h.u32(self.workers);
        h.u32(self.owner.len() as u32);
        for o in &self.owner {
            h.u32(o.0);
        }
        h.finish()
    }

    pub fn owner_of(&self, tile: TileId) -> WorkerId {
        self.owner[tile.0 as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{
        GlobalGrid, owned_cell_region, owned_face_region, touches_domain_edges,
    };
    use crate::decomp::{Leg, LegRegions, Partition, Topology, flatten};
    use crate::state::{Boundaries, BoundaryType, SimStateGeneric, Timestepping};
    use symbi_geometry::Cartesian;
    use symbi_hydro::eos::IdealGas;
    use symbi_hydro::newtonian::Newtonian;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    use symbi_xpu::{CpuSpace, HostMemory};

    type Hydro = SimStateGeneric<Newtonian, 2, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
    type Mhd = SimStateGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

    const CELLS: [usize; 2] = [40, 24];

    fn ragged() -> Partition<2> {
        Partition::explicit(CELLS, [vec![13, 29], vec![9]]).unwrap()
    }

    fn even() -> Partition<2> {
        Partition::uniform(CELLS, [2, 2]).unwrap()
    }

    fn hydro_tile(p: &Partition<2>, flat: usize, ng: usize) -> Hydro {
        let ext = p.tile_extents(unflatten(flat, p.counts()));
        Hydro::new(
            Newtonian,
            IdealGas { gamma: 1.4 },
            Cartesian,
            [ext[0].1, ext[1].1],
            [ext[0].0 as f64, ext[1].0 as f64],
            [1.0, 1.0],
            ng,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap()
    }

    fn mhd_tile(p: &Partition<2>, flat: usize, ng: usize) -> Mhd {
        let ext = p.tile_extents(unflatten(flat, p.counts()));
        Mhd::new(
            NewtonianMhd,
            IdealGas { gamma: 1.4 },
            Cartesian,
            [ext[0].1, ext[1].1],
            [ext[0].0 as f64, ext[1].0 as f64],
            [1.0, 1.0],
            ng,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap()
    }

    fn hydro_schema() -> FieldSchema {
        FieldSchema::new()
            .cell("rho")
            .cell("v1")
            .cell("v2")
            .cell("p")
    }

    fn mhd_schema() -> FieldSchema {
        hydro_schema()
            .cell("b1")
            .cell("b2")
            .face("bface1", 0)
            .face("bface2", 1)
    }

    #[test]
    fn layout_matches_built_tiles() {
        for p in [ragged(), even()] {
            for ng in [2usize, 3] {
                for flat in 0..p.n_tiles() {
                    let tile = TileId(flat as u32);
                    let derived = TileLayout::of_partition(&p, tile, ng);
                    let sim = hydro_tile(&p, flat, ng);
                    let built = TileLayout::of_geometry(tile, &sim.geom, derived.offset);
                    assert_eq!(derived, built, "tile {flat} ng {ng}");
                }
            }
        }
    }

    #[test]
    fn face_layout_matches_bface_domain() {
        for ng in [2usize, 3] {
            let p = ragged();
            for flat in 0..p.n_tiles() {
                let sim = mhd_tile(&p, flat, ng);
                let mhd = sim.fields.mhd.as_ref().unwrap();
                let tile = TileLayout::of_partition(&p, TileId(flat as u32), ng);
                for d in 0..2 {
                    let layout = FieldLayout::of(&tile, FieldKind::Face(d as u8));
                    assert_eq!(
                        layout.allocated,
                        spans_of(mhd.bface[d].domain()),
                        "tile {flat} ng {ng} bface{d}"
                    );
                    assert_eq!(layout, FieldLayout::face_of_geometry(&sim.geom, d));
                }
                assert_eq!(
                    FieldLayout::of(&tile, FieldKind::Cell),
                    FieldLayout::cell_of_geometry(&sim.geom)
                );
            }
        }
    }

    /// the plan's regions are the leg arithmetic's regions, leg by leg and field by field,
    /// on the built tiles' own geometry.
    #[test]
    fn plan_regions_equal_leg_regions() {
        for ng in [2usize, 3] {
            let p = ragged();
            let plan = ExchangePlan::compile(&p, &Topology::open(), &mhd_schema(), ng).unwrap();
            let tiles: Vec<Mhd> = (0..p.n_tiles()).map(|f| mhd_tile(&p, f, ng)).collect();
            let schedule = Schedule::derive(p.counts(), ng, &Topology::open());
            let mut seen = 0;
            for leg in schedule.legs() {
                for (f, entry) in plan.schema.entries.iter().enumerate() {
                    let (lo, hi) = match entry.kind {
                        FieldKind::Cell => (
                            FieldLayout::cell_of_geometry(&tiles[leg.lo].geom),
                            FieldLayout::cell_of_geometry(&tiles[leg.hi].geom),
                        ),
                        FieldKind::Face(d) if d as usize == leg.axis => continue,
                        FieldKind::Face(d) => (
                            FieldLayout::face_of_geometry(&tiles[leg.lo].geom, d as usize),
                            FieldLayout::face_of_geometry(&tiles[leg.hi].geom, d as usize),
                        ),
                    };
                    let h = lo.halo[leg.axis].min(ng);
                    let LegRegions {
                        lo_ghost,
                        hi_src,
                        hi_ghost,
                        lo_src,
                    } = leg_regions(&lo, &hi, leg, h);
                    let find = |src: usize, dst: usize| {
                        plan.transfers()
                            .find(|t| {
                                t.axis as usize == leg.axis
                                    && t.src.0 as usize == src
                                    && t.dst.0 as usize == dst
                                    && t.field as usize == f
                            })
                            .unwrap_or_else(|| panic!("no transfer {src}->{dst} field {f}"))
                    };
                    let a = find(leg.hi, leg.lo);
                    assert_eq!(a.src_region, spans_of(&hi_src));
                    assert_eq!(a.dst_region, spans_of(&lo_ghost));
                    let b = find(leg.lo, leg.hi);
                    assert_eq!(b.src_region, spans_of(&lo_src));
                    assert_eq!(b.dst_region, spans_of(&hi_ghost));
                    seen += 2;
                }
            }
            assert_eq!(seen, plan.transfers().count(), "every transfer was matched");
            assert!(seen > 0);
        }
    }

    /// a destination lies in the ghost band of its field on the leg's axis and a source lies in
    /// the interior, on cells and on faces alike.
    #[test]
    fn destinations_are_ghosts_and_sources_are_interior() {
        let p = ragged();
        let plan = ExchangePlan::compile(&p, &Topology::open(), &mhd_schema(), 3).unwrap();
        for t in plan.transfers() {
            let ax = t.axis as usize;
            let dst = &plan.fields[t.dst.0 as usize][t.field as usize];
            let src = &plan.fields[t.src.0 as usize][t.field as usize];
            let outside = t.dst_region[ax].hi <= dst.interior[ax].lo
                || t.dst_region[ax].lo >= dst.interior[ax].hi;
            assert!(outside, "transfer {:?} writes interior cells", t.id);
            assert!(
                src.interior[ax].contains(&t.src_region[ax]),
                "transfer {:?} reads ghosts",
                t.id
            );
        }
    }

    #[test]
    fn plan_digest_ignores_placement() {
        let p = even();
        let plan = ExchangePlan::compile(&p, &Topology::open(), &hydro_schema(), 2).unwrap();
        let again = ExchangePlan::compile(&p, &Topology::open(), &hydro_schema(), 2).unwrap();
        assert_eq!(plan, again);
        let a =
            Placement::new(2, vec![WorkerId(0), WorkerId(0), WorkerId(1), WorkerId(1)]).unwrap();
        let b =
            Placement::new(4, vec![WorkerId(0), WorkerId(1), WorkerId(2), WorkerId(3)]).unwrap();
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn plan_digest_changes_with_cuts_reach_schema_and_topology() {
        let base = ExchangePlan::compile(&ragged(), &Topology::open(), &hydro_schema(), 2).unwrap();
        let moved = Partition::explicit(CELLS, [vec![14, 29], vec![9]]).unwrap();
        let cut = ExchangePlan::compile(&moved, &Topology::open(), &hydro_schema(), 2).unwrap();
        let reach =
            ExchangePlan::compile(&ragged(), &Topology::open(), &hydro_schema(), 3).unwrap();
        let schema =
            ExchangePlan::compile(&ragged(), &Topology::open(), &hydro_schema().cell("chi"), 2)
                .unwrap();
        let wrap = ExchangePlan::compile(
            &ragged(),
            &Topology::wrapping([true, false]),
            &hydro_schema(),
            2,
        )
        .unwrap();
        let digests = [
            base.digest,
            cut.digest,
            reach.digest,
            schema.digest,
            wrap.digest,
        ];
        for (i, a) in digests.iter().enumerate() {
            for b in &digests[i + 1..] {
                assert_ne!(a, b);
            }
        }
        assert!(
            wrap.transfers().count() > base.transfers().count(),
            "the wrap legs add transfers"
        );
    }

    #[test]
    fn destinations_disjoint_across_plan() {
        for ng in [2usize, 3] {
            let plan = ExchangePlan::compile(
                &ragged(),
                &Topology::wrapping([true, true]),
                &mhd_schema(),
                ng,
            )
            .unwrap();
            let all: Vec<&Transfer<2>> = plan.transfers().collect();
            for (i, a) in all.iter().enumerate() {
                for b in &all[i + 1..] {
                    if a.dst == b.dst && a.field == b.field {
                        assert!(
                            !boxes_overlap(&a.dst_region, &b.dst_region),
                            "{:?} and {:?}",
                            a.id,
                            b.id
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn an_uncut_grid_compiles_to_no_transfers() {
        let p = Partition::uniform(CELLS, [1, 1]).unwrap();
        let plan = ExchangePlan::compile(&p, &Topology::wrapping([true, true]), &hydro_schema(), 2)
            .unwrap();
        assert_eq!(plan.transfers().count(), 0);
    }

    #[test]
    fn placement_requires_every_worker_to_hold_a_tile() {
        assert!(Placement::new(3, vec![WorkerId(0), WorkerId(1)]).is_err());
        assert!(Placement::new(2, vec![WorkerId(0), WorkerId(2)]).is_err());
        assert!(Placement::new(2, vec![WorkerId(1), WorkerId(0)]).is_ok());
    }

    #[test]
    fn exchange_point_round_trips_the_wire_byte() {
        for p in [
            ExchangePoint::Prime,
            ExchangePoint::Stage(0),
            ExchangePoint::Stage(1),
            ExchangePoint::Rollback,
        ] {
            assert_eq!(ExchangePoint::from_wire(p.to_wire()), Some(p));
        }
        assert_eq!(ExchangePoint::from_wire(Epoch::NO_POINT), None);
    }

    /// over every tile and field kind, the owned boxes are pairwise disjoint and cover the global
    /// dataset exactly: cells with the global halo, faces with the closing face per axis.
    #[test]
    fn owned_boxes_tile_the_grid() {
        for p in [ragged(), even()] {
            for ng in [2usize, 3] {
                let grid = GlobalGrid {
                    cells: CELLS,
                    interior_lo: [0, 0],
                    ng,
                    x_lo: [0.0, 0.0],
                    dx: [1.0, 1.0],
                    maps: None,
                };
                let ng_i = ng as isize;
                let cell_box = [
                    Span {
                        lo: 0,
                        hi: CELLS[0] as isize + 2 * ng_i,
                    },
                    Span {
                        lo: 0,
                        hi: CELLS[1] as isize + 2 * ng_i,
                    },
                ];
                let mut cell_boxes: Vec<[Span; 2]> = Vec::new();
                let mut face_boxes: [Vec<[Span; 2]>; 2] = [Vec::new(), Vec::new()];
                for flat in 0..p.n_tiles() {
                    let t = TileLayout::of_partition(&p, TileId(flat as u32), ng);
                    let size = [t.interior[0].size(), t.interior[1].size()];
                    let (first, last) = touches_domain_edges(&t.offset, &size, &grid);
                    let interior = domain_of(&t.interior);
                    let alloc = domain_of(&t.allocated);
                    let cells = owned_cell_region(&interior, &alloc, &first, &last);
                    // file index of a local cell: global interior index plus the halo width.
                    cell_boxes.push(std::array::from_fn(|ax| Span {
                        lo: cells.spaces[ax].lo + t.offset[ax] + ng_i,
                        hi: cells.spaces[ax].hi + t.offset[ax] + ng_i,
                    }));
                    for d in 0..2 {
                        let faces = owned_face_region(&interior, d, &last);
                        face_boxes[d].push(std::array::from_fn(|ax| Span {
                            lo: faces.spaces[ax].lo + t.offset[ax],
                            hi: faces.spaces[ax].hi + t.offset[ax],
                        }));
                    }
                }
                assert_tiling(&cell_boxes, &cell_box, "cells");
                for d in 0..2 {
                    let face_box: [Span; 2] = std::array::from_fn(|ax| Span {
                        lo: 0,
                        hi: CELLS[ax] as isize + if ax == d { 1 } else { 0 },
                    });
                    assert_tiling(&face_boxes[d], &face_box, &format!("faces {d}"));
                }
            }
        }
    }

    fn assert_tiling(boxes: &[[Span; 2]], whole: &[Span; 2], label: &str) {
        let mut total = 0;
        for (i, a) in boxes.iter().enumerate() {
            assert!(
                (0..2).all(|ax| whole[ax].contains(&a[ax])),
                "{label}: box {a:?} leaves {whole:?}"
            );
            total += volume(a);
            for b in &boxes[i + 1..] {
                assert!(!boxes_overlap(a, b), "{label}: {a:?} overlaps {b:?}");
            }
        }
        assert_eq!(total, volume(whole), "{label}: coverage");
    }

    #[test]
    fn transfer_ids_follow_schedule_order_and_tiles_index_flat() {
        let p = even();
        let plan = ExchangePlan::compile(&p, &Topology::open(), &hydro_schema(), 2).unwrap();
        let ids: Vec<u32> = plan.transfers().map(|t| t.id.0).collect();
        assert_eq!(ids, (0..ids.len() as u32).collect::<Vec<_>>());
        let first = plan.transfers().next().unwrap();
        let leg = Leg {
            axis: 0,
            lo: flatten([0, 0], p.counts()),
            hi: flatten([1, 0], p.counts()),
            clip: [false, true],
        };
        assert_eq!(first.src.0 as usize, leg.hi);
        assert_eq!(first.dst.0 as usize, leg.lo);
    }
}
