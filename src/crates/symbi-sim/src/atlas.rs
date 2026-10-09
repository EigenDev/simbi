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

use crate::decomp::{Partition, PolarSeam, Schedule, Topology, flatten, leg_regions, unflatten};
use crate::state::{PartitionGeometry, axis_name};
use symbi_algebra::{Domain, Side, Space};
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

    /// the same tile in an index space whose first interior cell sits at `lo`: a fine level
    /// indexes its cells in its parent's frame, so its plan's layouts start where its
    /// stores do. the global index of the first interior cell is unchanged.
    pub fn shifted(mut self, lo: [isize; D]) -> Self {
        for ax in 0..D {
            let shift = lo[ax] - self.interior[ax].lo;
            self.interior[ax] = Span { lo: self.interior[ax].lo + shift, hi: self.interior[ax].hi + shift };
            self.allocated[ax] = Span { lo: self.allocated[ax].lo + shift, hi: self.allocated[ax].hi + shift };
        }
        self
    }
}

/// where a field's values sit. a cell field lives on the tile's cell domain; a
/// face field on axis `d` holds one value per face normal to `d`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldKind {
    Cell,
    Face(u8),
    /// a cell field whose exchange covers the `h` ghost layers next to the interior, whatever
    /// halo it is allocated with. it moves at the `Troubled` exchange points and no other:
    /// the troubled-cell flag, which decides the flux on the faces two tiles share.
    CellBand(u8),
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

    pub fn cell_band(mut self, name: &str, h: u8) -> Self {
        self.entries.push(FieldEntry {
            name: name.to_string(),
            kind: FieldKind::CellBand(h),
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
            FieldKind::CellBand(h) => {
                let h = h as usize;
                Self {
                    interior: tile.interior,
                    allocated: std::array::from_fn(|ax| Span {
                        lo: tile.interior[ax].lo - h as isize,
                        hi: tile.interior[ax].hi + h as isize,
                    }),
                    halo: [h; D],
                }
            }
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

    /// a cell field exchanged over the `h` ghost layers next to the interior, whatever halo it
    /// is allocated with: the layout's allocated span is the interior grown by `h`, so a ghost
    /// strip of width `h` starts at the interior's edge.
    pub fn cell_band_of_geometry(geom: &PartitionGeometry<D>, h: usize) -> Self {
        let interior = spans_of(&geom.interior);
        Self {
            interior,
            allocated: std::array::from_fn(|ax| Span {
                lo: interior[ax].lo - h as isize,
                hi: interior[ax].hi + h as isize,
            }),
            halo: [h; D],
        }
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
    /// after the refined levels are seeded and primed: the primed root, whose covered cells and
    /// faces now hold the restriction of the seeded fine level, reaches the neighbors' halos
    /// before the first stage reads them
    Primed,
    Stage(u8),
    /// inside stage `k`, between the troubled-cell mark and the correction: the flags cross
    /// the cuts so both tiles sharing a face take one flux there.
    Troubled(u8),
    /// after the accepted step's post-step operations recovered the primitives: the cut halos
    /// carry them before the next step's first reader and the checkpoint
    PostStep,
    /// after the immersed bodies' surface operations rescaled the conserved state and the
    /// primitives were recovered from it: the cut halos carry the drained gas before the next
    /// step's reconstruction and the checkpoint
    PostBody,
    /// after the accepted step's continuous-tracer coefficients were materialized and their
    /// physical ghosts filled: the cut ghosts carry the neighbors' coefficients before the
    /// tracers interpolate them
    TracerCoefficients,
    /// before the prime of a run that declares a stationary target: the target seeded on every
    /// tile crosses the cuts, so the probe stage that reads its imbalance off sees whole stencils
    EquilibriumPrime,
    /// the probe stage's troubled flags across the cuts
    EquilibriumTroubled,
    Rollback,
    /// after the named exchange: each tile below a cut sends its copy of the shared
    /// interface face to the tile above, which compares it with its own bit for bit. the
    /// audit reads both copies and writes neither.
    Audit(AuditAt),
    /// one point of one fine substep of a refined run: the fine plan moves here, the root
    /// plan carries nothing
    Fine { substep: u8, point: FinePoint },
    /// after the tiles beside a neighbor's patch applied their received register units: the
    /// staggered faces those applications wrote on a cut reach the tile across it before any
    /// tile recovers its cells beside the patch
    CorrectedFaces,
    /// after the restriction and the reflux into the root: the root halos carry the
    /// restricted covered cells before the next step's prolongation snapshot and the
    /// checkpoint read them
    Restrict,
    /// one point of one substep of a nested refined level `level >= 2`: `path` numbers the
    /// substep along the subcycles of every level above it (`path = parent_path * 2 + s`);
    /// that level's plan moves here
    Nested { level: u8, path: u8, point: FinePoint },
    /// after the restriction of level `level + 1` into level `level` within the substep
    /// `path` of level `level`: that level's halos carry the restricted covered cells
    NestedRestrict { level: u8, path: u8 },
    /// at initialization, before level `level >= 2` is seeded by prolongation: the conserved
    /// halos of level `level - 1` cross its cuts
    Seed { level: u8 },
    /// after the horizon-excision sweep of stage `k`, or of the accepted step's post-step
    /// operations when `k` is the stage count: the cut halos carry the swept cells before the
    /// rebuild reads them
    ExciseSweep(u8),
    /// after the excision rebuild of stage `k`, or of the post-step when `k` is the stage count:
    /// the cut halos carry the rebuilt cells before the stage exchange or the next step reads them
    ExciseRebuild(u8),
    /// a declared stationary target's probe on a refined run, at initialization once every level
    /// is seeded: level `level`'s primed target crosses its cuts, or, `troubled`, its probe
    /// stage's troubled flags do
    EquilibriumLevel { level: u8, troubled: bool },
    /// after the root stages of a refined run carrying continuous tracers: the root's accepted ito
    /// coefficients cross the cuts before the fine level prolongs them
    RootCoefficients,
}

/// the points of one fine substep, in order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FinePoint {
    /// after the prolongation and the physical fill: the fine cut halos
    Prolong,
    Troubled(u8),
    Stage(u8),
    Audit(u8),
    /// after the fine tail's recovery and physical fill: the fine cut halos again, and the
    /// EMF contributions of the substep
    PostTail,
    /// the interface audit of the post-tail exchange
    AuditTail,
    /// after the substep's stages of a run carrying continuous tracers: the level's accepted ito
    /// coefficients cross its cuts before a finer level prolongs them and its tracers read them
    Coefficients,
}

/// the exchange an interface audit follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditAt {
    Prime,
    Stage(u8),
    Rollback,
    Restrict,
    /// before a checkpoint is opened, after the step's last exchange
    Checkpoint,
}

impl ExchangePoint {
    /// whether this point is a `Troubled` exchange on any level
    pub fn is_troubled(self) -> bool {
        matches!(
            self,
            ExchangePoint::Troubled(_)
                | ExchangePoint::Fine { point: FinePoint::Troubled(_), .. }
                | ExchangePoint::Nested { point: FinePoint::Troubled(_), .. }
        )
    }

    /// whether this point is an interface audit on any level
    pub fn is_audit(self) -> bool {
        matches!(
            self,
            ExchangePoint::Audit(_)
                | ExchangePoint::Fine { point: FinePoint::Audit(_) | FinePoint::AuditTail, .. }
                | ExchangePoint::Nested { point: FinePoint::Audit(_) | FinePoint::AuditTail, .. }
        )
    }

    /// whether this point belongs to a refined level
    pub fn is_fine(self) -> bool {
        matches!(
            self,
            ExchangePoint::Fine { .. } | ExchangePoint::Nested { .. } | ExchangePoint::NestedRestrict { .. } | ExchangePoint::Seed { .. }
        )
    }

    /// a stable code of the point's kind and indices for the phase table's digest
    pub fn kind_code(self) -> u32 {
        let (kind, a, b): (u32, u32, u32) = match self {
            ExchangePoint::Prime => (0, 0, 0),
            ExchangePoint::Primed => (18, 0, 0),
            ExchangePoint::Stage(k) => (1, k as u32, 0),
            ExchangePoint::Troubled(k) => (2, k as u32, 0),
            ExchangePoint::PostStep => (3, 0, 0),
            ExchangePoint::PostBody => (27, 0, 0),
            ExchangePoint::TracerCoefficients => (28, 0, 0),
            ExchangePoint::EquilibriumPrime => (29, 0, 0),
            ExchangePoint::EquilibriumTroubled => (30, 0, 0),
            ExchangePoint::Rollback => (4, 0, 0),
            ExchangePoint::Audit(AuditAt::Prime) => (5, 0, 0),
            ExchangePoint::Audit(AuditAt::Stage(k)) => (6, k as u32, 0),
            ExchangePoint::Audit(AuditAt::Rollback) => (7, 0, 0),
            ExchangePoint::Audit(AuditAt::Restrict) => (8, 0, 0),
            ExchangePoint::Audit(AuditAt::Checkpoint) => (9, 0, 0),
            ExchangePoint::Restrict => (10, 0, 0),
            ExchangePoint::CorrectedFaces => (17, 0, 0),
            ExchangePoint::Fine { substep, point } => match point {
                FinePoint::Prolong => (11, substep as u32, 0),
                FinePoint::Troubled(k) => (12, substep as u32, k as u32),
                FinePoint::Stage(k) => (13, substep as u32, k as u32),
                FinePoint::Audit(k) => (14, substep as u32, k as u32),
                FinePoint::PostTail => (15, substep as u32, 0),
                FinePoint::AuditTail => (16, substep as u32, 0),
                FinePoint::Coefficients => (34, substep as u32, 0),
            },
            // the level in the high bits of `a`, the path in its low five
            ExchangePoint::Nested { level, path, point } => {
                let a = (level as u32) << 5 | path as u32;
                match point {
                    FinePoint::Prolong => (19, a, 0),
                    FinePoint::Troubled(k) => (20, a, k as u32),
                    FinePoint::Stage(k) => (21, a, k as u32),
                    FinePoint::Audit(k) => (22, a, k as u32),
                    FinePoint::PostTail => (23, a, 0),
                    FinePoint::AuditTail => (24, a, 0),
                    FinePoint::Coefficients => (35, a, 0),
                }
            }
            ExchangePoint::NestedRestrict { level, path } => (25, (level as u32) << 5 | path as u32, 0),
            ExchangePoint::Seed { level } => (26, level as u32, 0),
            ExchangePoint::ExciseSweep(k) => (31, k as u32, 0),
            ExchangePoint::ExciseRebuild(k) => (32, k as u32, 0),
            ExchangePoint::EquilibriumLevel { level, troubled } => (33, level as u32, troubled as u32),
            ExchangePoint::RootCoefficients => (36, 0, 0),
        };
        kind << 16 | a << 8 | b
    }
}

/// a real patch face lying on a cut: the tile holding the fine faces and the neighbor
/// across the cut, with their extents
struct CutFace<const D: usize> {
    ax: usize,
    side: usize,
    face_pos: usize,
    src: TileId,
    src_ext: [(usize, usize); D],
    dst: TileId,
    dst_ext: [(usize, usize); D],
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
    /// an interface audit: the source's copy of a shared interface face, which the
    /// destination compares with its own copy of the same face and never stores.
    pub audit: bool,
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
            PlanError::PolarSeam => write!(f, "a polar seam needs an even azimuth cell count"),
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
    /// the polar seam the plan carries antipodal transfers for: a polar axis face on a
    /// chart whose azimuth is gridded and cut, so the ghost band beyond the pole is the
    /// interior half a period away on other tiles
    pub seam: Option<PolarSeam>,
    /// every (tile, side) whose band beyond the pole the antipodal transfers fill; the band's
    /// odd components take their sign after every state exchange
    pub polar_bands: Vec<(TileId, Side)>,
    /// the axes the domain closes into a ring, as the topology the plan was compiled under
    pub wraps: [bool; D],
    pub digest: Digest,
}

impl<const D: usize> ExchangePlan<D> {
    pub fn compile(
        partition: &Partition<D>,
        topology: &Topology<D>,
        schema: &FieldSchema,
        ng: usize,
    ) -> Result<Self, PlanError> {
        Self::compile_with_seam(partition, topology, schema, ng, None)
    }

    /// `compile` plus the antipodal transfers of a polar seam whose azimuth is cut: for every
    /// tile on a declared pole face, every halo row beyond the pole reads the interior row
    /// the same distance inside the pole of the tiles holding the azimuth window half a
    /// period away, one row per transfer since the mirror reverses the row order. the
    /// transfers run in the mirror axis's phase after its ordinary ones; the axes still to
    /// come clip the window to the interior and carry the band's corners afterwards. the
    /// mirror-axis face field carries no band beyond the pole face, which the seed and the
    /// constrained-transport update own; the azimuth face field carries one face past the
    /// window so a slab's closing face lands on its physical partner.
    pub fn compile_with_seam(
        partition: &Partition<D>,
        topology: &Topology<D>,
        schema: &FieldSchema,
        ng: usize,
        seam: Option<PolarSeam>,
    ) -> Result<Self, PlanError> {
        Self::compile_placed(partition, topology, schema, ng, seam, &|_| None)
    }

    /// `compile` for tiles whose stores index their cells from `interior_lo(tile)` in place
    /// of zero: the fine level of a refined run, whose tiles live in their parents' index
    /// frames. every worker derives the same origins from the partition, so the plan is the
    /// same on every worker.
    pub fn compile_shifted(
        partition: &Partition<D>,
        topology: &Topology<D>,
        schema: &FieldSchema,
        ng: usize,
        interior_lo: &dyn Fn(TileId) -> [isize; D],
    ) -> Result<Self, PlanError> {
        Self::compile_placed(partition, topology, schema, ng, None, &|t| Some(interior_lo(t)))
    }

    fn compile_placed(
        partition: &Partition<D>,
        topology: &Topology<D>,
        schema: &FieldSchema,
        ng: usize,
        seam: Option<PolarSeam>,
        interior_lo: &dyn Fn(TileId) -> Option<[isize; D]>,
    ) -> Result<Self, PlanError> {
        let counts = partition.counts();
        if let Some(s) = &seam {
            if s.slabs.iter().sum::<usize>() % 2 != 0 {
                return Err(PlanError::PolarSeam);
            }
        }
        let schedule = Schedule::derive_with_seam(counts, ng, topology, seam);
        let tiles: Vec<TileLayout<D>> = (0..partition.n_tiles())
            .map(|t| {
                let id = TileId(t as u32);
                let layout = TileLayout::of_partition(partition, id, ng);
                match interior_lo(id) {
                    Some(lo) => layout.shifted(lo),
                    None => layout,
                }
            })
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
                        // the wrap leg of an axis with distinct ends joins two different
                        // faces: nothing to audit.
                        let wraps = unflatten(leg.lo, counts)[leg.axis] + 1 == counts[leg.axis];
                        if wraps && !topology.ends_coincide(leg.axis) {
                            continue;
                        }
                        // the audit of that interface: the lower tile's closing face against
                        // the upper tile's opening face, over the faces both tiles own.
                        let lo = &fields[leg.lo][f];
                        let hi = &fields[leg.hi][f];
                        let face = |tile: usize, at: isize| -> [Span; D] {
                            std::array::from_fn(|ax| {
                                if ax == leg.axis {
                                    Span { lo: at, hi: at + 1 }
                                } else {
                                    tiles[tile].interior[ax]
                                }
                            })
                        };
                        let src_region = face(leg.lo, lo.interior[leg.axis].hi - 1);
                        let dst_region = face(leg.hi, hi.interior[leg.axis].lo);
                        axes[leg.axis].transfers.push(Transfer {
                            id: TransferId(next),
                            axis: leg.axis as u8,
                            src: TileId(leg.lo as u32),
                            dst: TileId(leg.hi as u32),
                            field: f as u16,
                            len: volume(&dst_region),
                            src_region,
                            dst_region,
                            audit: true,
                        });
                        next += 1;
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
                            audit: false,
                        });
                        next += 1;
                    };
                push(leg.hi, &r.hi_src, leg.lo, &r.lo_ghost);
                push(leg.lo, &r.lo_src, leg.hi, &r.hi_ghost);
            }
        }
        if let Some(s) = schedule.seam() {
            let (mirror, az) = (s.mirror, s.azimuth);
            let within = |span: &Span, row: isize| span.lo <= row && row < span.hi;
            for leg in schedule.polar_legs() {
                let (dst_tile, src_tile) = (&tiles[leg.dst], &tiles[leg.src]);
                for k in 0..ng as isize {
                    // row k beyond the pole reads interior row k counted from the pole
                    let (drow, srow) = match leg.side {
                        Side::Lo => (dst_tile.interior[mirror].lo - 1 - k, src_tile.interior[mirror].lo + k),
                        Side::Hi => (dst_tile.interior[mirror].hi + k, src_tile.interior[mirror].hi - 1 - k),
                    };
                    for (f, entry) in schema.entries.iter().enumerate() {
                        let d = match entry.kind {
                            FieldKind::Face(d) => d as usize,
                            _ => D,
                        };
                        if d == mirror {
                            continue;
                        }
                        let (dl, sl) = (&fields[leg.dst][f], &fields[leg.src][f]);
                        if !within(&dl.allocated[mirror], drow) || !within(&sl.allocated[mirror], srow) {
                            continue;
                        }
                        // the azimuth face field carries one face past the window where the
                        // window closes the destination slab; a window ending inside the slab
                        // shares that face with the next leg's opening face
                        let closes = leg.dst_lo + leg.len == dst_tile.interior[az].size() as isize;
                        let region = |layout: &FieldLayout<D>, tile: &TileLayout<D>, row: isize, az_lo: isize| -> [Span; D] {
                            std::array::from_fn(|b| {
                                if b == mirror {
                                    Span { lo: row, hi: row + 1 }
                                } else if b == az {
                                    let lo = tile.interior[az].lo + az_lo;
                                    Span { lo, hi: lo + leg.len + isize::from(d == az && closes) }
                                } else {
                                    let (i, a) = (layout.interior[b], layout.allocated[b]);
                                    Span {
                                        lo: if leg.clip[b][0] { i.lo } else { a.lo },
                                        hi: if leg.clip[b][1] { i.hi } else { a.hi },
                                    }
                                }
                            })
                        };
                        let src_region = region(sl, src_tile, srow, leg.src_lo);
                        let dst_region = region(dl, dst_tile, drow, leg.dst_lo);
                        axes[mirror].transfers.push(Transfer {
                            id: TransferId(next),
                            axis: mirror as u8,
                            src: TileId(leg.src as u32),
                            dst: TileId(leg.dst as u32),
                            field: f as u16,
                            len: volume(&dst_region),
                            src_region,
                            dst_region,
                            audit: false,
                        });
                        next += 1;
                    }
                }
            }
        }
        // the ids run in plan order, axis by axis, the antipodal transfers of the mirror
        // axis included
        let mut next = 0u32;
        for phase in axes.iter_mut() {
            for t in phase.transfers.iter_mut() {
                t.id = TransferId(next);
                next += 1;
            }
        }
        let polar_bands = schedule
            .polar_bands()
            .into_iter()
            .map(|(tile, side)| (TileId(tile as u32), side))
            .collect();
        let plan = Self {
            reach: ng,
            tiles,
            fields,
            schema: schema.clone(),
            axes,
            seam: schedule.seam().cloned(),
            polar_bands,
            wraps: std::array::from_fn(|ax| topology.is_periodic(ax)),
            digest: Digest(0),
        };
        plan.validate()?;
        let digest = plan.compute_digest(counts, topology);
        Ok(Self { digest, ..plan })
    }

    /// the plan of the flux-register contributions of one refined patch of ratio 2 under a
    /// root partition: for every real face of `patch` (root cell ranges per axis) that lies
    /// on a cut, one transfer per register component from the tile holding the fine faces
    /// (the coverage ends at its interior boundary) to the neighbor across the cut, which
    /// holds the corrected coarse cell. the fields are the register slabs themselves,
    /// `2 * D * n_comp` per tile in `(2 * ax + side) * n_comp + comp` order: the contributing
    /// face on the sender, the inbox on the owner, both one cell thick along the face's axis
    /// and the patch's extent clipped to the tile across it, in the tile's root index space.
    /// a plan with no transfer means no patch face lies on a cut.
    pub fn flux_contributions(
        partition: &Partition<D>,
        topology: &Topology<D>,
        patch: [(usize, usize); D],
        ng: usize,
        n_comp: usize,
    ) -> Result<Self, PlanError> {
        let counts = partition.counts();
        let n_tiles = partition.n_tiles();
        let tiles: Vec<TileLayout<D>> = (0..n_tiles).map(|t| TileLayout::of_partition(partition, TileId(t as u32), ng)).collect();
        let mut schema = FieldSchema::new();
        for idx in 0..2 * D {
            for comp in 0..n_comp {
                schema = schema.cell(&format!("register_{idx}_{comp}"));
            }
        }
        let empty = FieldLayout { interior: [Span { lo: 0, hi: 0 }; D], allocated: [Span { lo: 0, hi: 0 }; D], halo: [0; D] };
        let mut fields: Vec<Vec<FieldLayout<D>>> = vec![vec![empty; 2 * D * n_comp]; n_tiles];
        let mut axes: Vec<AxisPhase<D>> = (0..D).map(|ax| AxisPhase { axis: ax as u8, transfers: Vec::new() }).collect();
        let mut next = 0u32;
        for ax in 0..D {
            for side in 0..2 {
                let face_pos = if side == 0 { patch[ax].0 } else { patch[ax].1 };
                for src in 0..n_tiles {
                    let tc = unflatten(src, counts);
                    let ext = partition.tile_extents(tc);
                    // the sender: its interior ends at the patch's face on this side and it
                    // covers part of the patch across the face
                    let ends_at_face = if side == 0 { ext[ax].0 == face_pos } else { ext[ax].0 + ext[ax].1 == face_pos };
                    let covers = (0..D).all(|aa| ext[aa].0 < patch[aa].1 && ext[aa].0 + ext[aa].1 > patch[aa].0);
                    let neighbor_exists = if side == 0 { tc[ax] > 0 } else { tc[ax] + 1 < counts[ax] };
                    if !(ends_at_face && covers && neighbor_exists) {
                        continue;
                    }
                    let mut ntc = tc;
                    ntc[ax] = if side == 0 { tc[ax] - 1 } else { tc[ax] + 1 };
                    let dst = flatten(ntc, counts);
                    let dst_ext = partition.tile_extents(ntc);
                    // the slab across the face: the patch clipped to the tile's extent, which the
                    // neighbor shares on every transverse axis
                    let slab = |tile_ext: [(usize, usize); D], along: isize| -> [Span; D] {
                        std::array::from_fn(|aa| {
                            if aa == ax {
                                Span { lo: along, hi: along + 1 }
                            } else {
                                let lo = patch[aa].0.max(tile_ext[aa].0) as isize - tile_ext[aa].0 as isize;
                                let hi = patch[aa].1.min(tile_ext[aa].0 + tile_ext[aa].1) as isize - tile_ext[aa].0 as isize;
                                Span { lo, hi }
                            }
                        })
                    };
                    let src_region = slab(ext, face_pos as isize - ext[ax].0 as isize);
                    let dst_region = slab(dst_ext, face_pos as isize - dst_ext[ax].0 as isize);
                    for comp in 0..n_comp {
                        let f = (2 * ax + side) * n_comp + comp;
                        fields[src][f] = FieldLayout { interior: src_region, allocated: src_region, halo: [0; D] };
                        fields[dst][f] = FieldLayout { interior: dst_region, allocated: dst_region, halo: [0; D] };
                        axes[ax].transfers.push(Transfer {
                            id: TransferId(next),
                            axis: ax as u8,
                            src: TileId(src as u32),
                            dst: TileId(dst as u32),
                            field: f as u16,
                            len: volume(&dst_region),
                            src_region,
                            dst_region,
                            audit: false,
                        });
                        next += 1;
                    }
                }
            }
        }
        let plan = Self { reach: ng, tiles, fields, schema, axes, seam: None, polar_bands: Vec::new(), wraps: std::array::from_fn(|ax| topology.is_periodic(ax)), digest: Digest(0) };
        plan.validate()?;
        let digest = plan.compute_digest(counts, topology);
        Ok(Self { digest, ..plan })
    }

    /// the sender and owner of every real face of `patch` lying on a cut, with the tiles'
    /// extents: `(ax, side, face_pos, src, src_ext, dst, dst_ext)`
    fn cut_faces(partition: &Partition<D>, patch: [(usize, usize); D]) -> Vec<CutFace<D>> {
        let counts = partition.counts();
        let mut out = Vec::new();
        for ax in 0..D {
            for side in 0..2 {
                let face_pos = if side == 0 { patch[ax].0 } else { patch[ax].1 };
                for src in 0..partition.n_tiles() {
                    let tc = unflatten(src, counts);
                    let ext = partition.tile_extents(tc);
                    let ends_at_face = if side == 0 { ext[ax].0 == face_pos } else { ext[ax].0 + ext[ax].1 == face_pos };
                    let covers = (0..D).all(|aa| ext[aa].0 < patch[aa].1 && ext[aa].0 + ext[aa].1 > patch[aa].0);
                    let neighbor_exists = if side == 0 { tc[ax] > 0 } else { tc[ax] + 1 < counts[ax] };
                    if !(ends_at_face && covers && neighbor_exists) {
                        continue;
                    }
                    let mut ntc = tc;
                    ntc[ax] = if side == 0 { tc[ax] - 1 } else { tc[ax] + 1 };
                    out.push(CutFace {
                        ax,
                        side,
                        face_pos,
                        src: TileId(src as u32),
                        src_ext: ext,
                        dst: TileId(flatten(ntc, counts) as u32),
                        dst_ext: partition.tile_extents(ntc),
                    });
                }
            }
        }
        out
    }

    /// a plan over slabs the caller lays out: `slabs` yields, per transfer, the field index,
    /// the source and destination tiles and their regions in each tile's index space
    fn slab_plan(
        partition: &Partition<D>,
        topology: &Topology<D>,
        ng: usize,
        schema: FieldSchema,
        slabs: Vec<(usize, TileId, [Span; D], TileId, [Span; D])>,
    ) -> Result<Self, PlanError> {
        let counts = partition.counts();
        let n_tiles = partition.n_tiles();
        let tiles: Vec<TileLayout<D>> = (0..n_tiles).map(|t| TileLayout::of_partition(partition, TileId(t as u32), ng)).collect();
        let empty = FieldLayout { interior: [Span { lo: 0, hi: 0 }; D], allocated: [Span { lo: 0, hi: 0 }; D], halo: [0; D] };
        let mut fields: Vec<Vec<FieldLayout<D>>> = vec![vec![empty; schema.len()]; n_tiles];
        let mut axes: Vec<AxisPhase<D>> = (0..D).map(|ax| AxisPhase { axis: ax as u8, transfers: Vec::new() }).collect();
        for (f, src, src_region, dst, dst_region) in slabs {
            fields[src.0 as usize][f] = FieldLayout { interior: src_region, allocated: src_region, halo: [0; D] };
            fields[dst.0 as usize][f] = FieldLayout { interior: dst_region, allocated: dst_region, halo: [0; D] };
            // the axis whose extent is one: the plane's normal
            let ax = (0..D).find(|&a| src_region[a].size() == 1 && dst_region[a].size() == 1).unwrap_or(0);
            axes[ax].transfers.push(Transfer {
                id: TransferId(0),
                axis: ax as u8,
                src,
                dst,
                field: f as u16,
                len: volume(&dst_region),
                src_region,
                dst_region,
                audit: false,
            });
        }
        let mut next = 0u32;
        for phase in axes.iter_mut() {
            for t in phase.transfers.iter_mut() {
                t.id = TransferId(next);
                next += 1;
            }
        }
        let plan = Self { reach: ng, tiles, fields, schema, axes, seam: None, polar_bands: Vec::new(), wraps: std::array::from_fn(|ax| topology.is_periodic(ax)), digest: Digest(0) };
        plan.validate()?;
        let digest = plan.compute_digest(counts, topology);
        Ok(Self { digest, ..plan })
    }

    /// the plan of the EMF-register contributions of one refined patch of ratio 2 under a
    /// root partition: for every real face of `patch` lying on a cut and every edge family
    /// with edges in that plane (the three grid axes in 3D, the missing axis on a 2.5D grid),
    /// one transfer of the in-plane edge slab from the tile holding the fine sub-edges to the
    /// neighbor across the cut, which holds the corrected outside faces. the fields are the
    /// slabs themselves, `2 * D * families` per tile in `(2 * bb + side) * families + family`
    /// order, one node thick along the plane's normal, the patch's node range clipped to the
    /// tile along the corrected faces' axis (both box corners included) and its cell range
    /// along the edge, in the tile's root index space.
    pub fn emf_contributions(
        partition: &Partition<D>,
        topology: &Topology<D>,
        patch: [(usize, usize); D],
        ng: usize,
    ) -> Result<Self, PlanError> {
        let families: Vec<usize> = if D == 3 { vec![0, 1, 2] } else { vec![2] };
        let mut schema = FieldSchema::new();
        for plane in 0..2 * D {
            for family in 0..families.len() {
                schema = schema.cell(&format!("emf_{plane}_{family}"));
            }
        }
        let mut slabs = Vec::new();
        for face in Self::cut_faces(partition, patch) {
            let bb = face.ax;
            for (family, &tt) in families.iter().enumerate() {
                if tt == bb {
                    continue;
                }
                let aa = 3 - tt - bb;
                let region = |ext: [(usize, usize); D]| -> [Span; D] {
                    std::array::from_fn(|axis| {
                        let lo_abs = patch[axis].0.max(ext[axis].0) as isize;
                        let hi_abs = patch[axis].1.min(ext[axis].0 + ext[axis].1) as isize;
                        let origin = ext[axis].0 as isize;
                        if axis == bb {
                            let at = face.face_pos as isize - origin;
                            Span { lo: at, hi: at + 1 }
                        } else if axis == aa {
                            Span { lo: lo_abs - origin, hi: hi_abs + 1 - origin }
                        } else {
                            Span { lo: lo_abs - origin, hi: hi_abs - origin }
                        }
                    })
                };
                let f = (2 * bb + face.side) * families.len() + family;
                slabs.push((f, face.src, region(face.src_ext), face.dst, region(face.dst_ext)));
            }
        }
        Self::slab_plan(partition, topology, ng, schema, slabs)
    }

    /// the plan replacing the neighbor's copy of the staggered faces on a cut that the
    /// restriction and the corner refluxes wrote on the tile holding the fine faces: for
    /// every real face of `patch` lying on a cut, the face field normal to the cut over the
    /// patch's transverse extent widened by one cell on each side (the corner-corrected
    /// faces), from that tile's copy to the neighbor's. the fields are the face fields, one
    /// per axis, in each tile's root index space.
    pub fn interface_faces(
        partition: &Partition<D>,
        topology: &Topology<D>,
        patch: [(usize, usize); D],
        ng: usize,
    ) -> Result<Self, PlanError> {
        let mut schema = FieldSchema::new();
        for ax in 0..D {
            schema = schema.face(&format!("bface{ax}"), ax as u8);
        }
        let mut slabs = Vec::new();
        for face in Self::cut_faces(partition, patch) {
            let region = |ext: [(usize, usize); D]| -> [Span; D] {
                std::array::from_fn(|axis| {
                    let origin = ext[axis].0 as isize;
                    if axis == face.ax {
                        let at = face.face_pos as isize - origin;
                        Span { lo: at, hi: at + 1 }
                    } else {
                        let lo_abs = (patch[axis].0 as isize - 1).max(ext[axis].0 as isize);
                        let hi_abs = (patch[axis].1 as isize + 1).min((ext[axis].0 + ext[axis].1) as isize);
                        Span { lo: lo_abs - origin, hi: hi_abs - origin }
                    }
                })
            };
            slabs.push((face.ax, face.src, region(face.src_ext), face.dst, region(face.dst_ext)));
        }
        Self::slab_plan(partition, topology, ng, schema, slabs)
    }

    /// the plan gathering each refined tile's parent face stencil from the tiles owning those
    /// faces: `boxes[tile][component]` is the stencil box in whole-grid face indices. a tile
    /// owns the faces of component `d` from its first face to its last cell along `d` (its
    /// closing face too where it ends the grid) and its own cells across; along a periodic axis
    /// of `topology` a box face past the domain is owned at its periodic image, the closing face
    /// by its unshifted copy alone. every box face has exactly one owner, the tile itself
    /// included, and a face no tile owns is an error. the
    /// sources are the owners' face fields, the destinations the stencil fields, both in each
    /// tile's own index space; a transfer's field index names the component.
    pub fn face_stencil(
        partition: &Partition<D>,
        topology: &Topology<D>,
        boxes: &[Option<Vec<[Span; D]>>],
        ng: usize,
    ) -> Result<Self, PlanError> {
        Self::face_stencil_shifted(partition, topology, boxes, ng, &|_| [0; D])
    }

    /// `face_stencil` for tiles whose stores index their cells from `interior_lo(tile)` in place
    /// of zero: a nested level's parent, whose tiles live in their own parents' index frames.
    /// the boxes stay in whole-level face indices; sources and destinations land in each tile's
    /// own frame
    pub fn face_stencil_shifted(
        partition: &Partition<D>,
        topology: &Topology<D>,
        boxes: &[Option<Vec<[Span; D]>>],
        ng: usize,
        interior_lo: &dyn Fn(TileId) -> [isize; D],
    ) -> Result<Self, PlanError> {
        let counts = partition.counts();
        let n_tiles = partition.n_tiles();
        assert_eq!(boxes.len(), n_tiles, "one stencil entry per tile");
        let tiles: Vec<TileLayout<D>> = (0..n_tiles)
            .map(|t| TileLayout::of_partition(partition, TileId(t as u32), ng).shifted(interior_lo(TileId(t as u32))))
            .collect();
        let frames: Vec<[isize; D]> = (0..n_tiles).map(|t| interior_lo(TileId(t as u32))).collect();
        let mut schema = FieldSchema::new();
        for d in 0..D {
            schema = schema.face(&format!("stencil{d}"), d as u8);
        }
        let exts: Vec<[(usize, usize); D]> = (0..n_tiles).map(|t| partition.tile_extents(unflatten(t, counts))).collect();
        // each tile's field layout: its face lattice widened to its stencil box
        let mut fields: Vec<Vec<FieldLayout<D>>> = (0..n_tiles)
            .map(|t| (0..D).map(|d| FieldLayout::of(&tiles[t], FieldKind::Face(d as u8))).collect())
            .collect();
        let local = |g: &[Span; D], t: usize| -> [Span; D] {
            let (ext, frame) = (&exts[t], &frames[t]);
            std::array::from_fn(|a| Span { lo: g[a].lo - ext[a].0 as isize + frame[a], hi: g[a].hi - ext[a].0 as isize + frame[a] })
        };
        let mut axes: Vec<AxisPhase<D>> = (0..D).map(|ax| AxisPhase { axis: ax as u8, transfers: Vec::new() }).collect();
        // the cells per axis of the whole grid, and the periodic images of the grid a stencil box
        // may reach: one period either way along each periodic axis of the topology
        let grid: [isize; D] = std::array::from_fn(|a| exts.iter().map(|e| e[a].0 + e[a].1).max().unwrap_or(0) as isize);
        let mut images: Vec<[isize; D]> = vec![[0; D]];
        for a in 0..D {
            if topology.is_periodic(a) {
                images = images
                    .iter()
                    .flat_map(|img| [-1, 0, 1].map(|k| { let mut next = *img; next[a] = k; next }))
                    .collect();
            }
        }
        for (dst, entry) in boxes.iter().enumerate() {
            let Some(entry) = entry else { continue };
            for (d, gbox) in entry.iter().enumerate() {
                let lbox = local(gbox, dst);
                let f = &mut fields[dst][d];
                f.allocated = std::array::from_fn(|a| Span {
                    lo: f.allocated[a].lo.min(lbox[a].lo),
                    hi: f.allocated[a].hi.max(lbox[a].hi),
                });
                let mut covered = 0usize;
                for (src, ext) in exts.iter().enumerate() {
                    let tc = unflatten(src, counts);
                    for image in &images {
                        // the owned faces at this periodic image: the domain's faces 0..=N of the
                        // component's axis belong to the unshifted copies, the closing face N to its
                        // own, so an image shifted along that axis keeps to the faces past them
                        let own: [Span; D] = std::array::from_fn(|a| {
                            let lo = ext[a].0 as isize;
                            let closes = a == d && tc[a] + 1 == counts[a] && image[a] == 0;
                            let shift = image[a] * grid[a];
                            let span = Span { lo: lo + shift, hi: (ext[a].0 + ext[a].1) as isize + isize::from(closes) + shift };
                            if a == d && image[a] > 0 { Span { lo: span.lo.max(grid[a] + 1), hi: span.hi } } else { span }
                        });
                        let piece: [Span; D] = std::array::from_fn(|a| Span { lo: gbox[a].lo.max(own[a].lo), hi: gbox[a].hi.min(own[a].hi) });
                        if piece.iter().any(|s| s.hi <= s.lo) {
                            continue;
                        }
                        covered += volume(&piece);
                        let unshifted: [Span; D] =
                            std::array::from_fn(|a| Span { lo: piece[a].lo - image[a] * grid[a], hi: piece[a].hi - image[a] * grid[a] });
                        let src_region = local(&unshifted, src);
                        let dst_region = local(&piece, dst);
                        axes[d].transfers.push(Transfer {
                            id: TransferId(0),
                            axis: d as u8,
                            src: TileId(src as u32),
                            dst: TileId(dst as u32),
                            field: d as u16,
                            len: volume(&dst_region),
                            src_region,
                            dst_region,
                            audit: false,
                        });
                    }
                }
                if covered != volume(gbox) {
                    return Err(PlanError::Partition(format!(
                        "the parent face stencil of tile {dst} reads faces of component {d} no tile owns: {gbox:?}"
                    )));
                }
            }
        }
        let mut next = 0u32;
        for phase in axes.iter_mut() {
            for t in phase.transfers.iter_mut() {
                t.id = TransferId(next);
                next += 1;
            }
        }
        let plan = Self { reach: ng, tiles, fields, schema, axes, seam: None, polar_bands: Vec::new(), wraps: std::array::from_fn(|ax| topology.is_periodic(ax)), digest: Digest(0) };
        plan.validate()?;
        let digest = plan.compute_digest(counts, topology);
        Ok(Self { digest, ..plan })
    }

    /// the plan carrying the staggered faces a tile beside a neighbor's patch corrected on
    /// one of its own cuts: for every real face of `patch` lying on a cut, the tile across
    /// it applies edge slabs whose corrected faces run along the plane over the patch's node
    /// range; a face at an end node of that range lying on the tile's transverse cut is
    /// held by the tile across that cut too, which receives the corrected copy. one
    /// transfer per such node and edge family, the face field normal to the corrected
    /// faces over the cell beside the plane and the patch's cell range along the edge.
    pub fn corrected_faces(
        partition: &Partition<D>,
        topology: &Topology<D>,
        patch: [(usize, usize); D],
        ng: usize,
    ) -> Result<Self, PlanError> {
        let counts = partition.counts();
        let families: Vec<usize> = if D == 3 { vec![0, 1, 2] } else { vec![2] };
        let mut schema = FieldSchema::new();
        for ax in 0..D {
            schema = schema.face(&format!("bface{ax}"), ax as u8);
        }
        let mut slabs = Vec::new();
        for face in Self::cut_faces(partition, patch) {
            let bb = face.ax;
            let owner = face.dst;
            let owner_tc = unflatten(owner.0 as usize, counts);
            let owner_ext = face.dst_ext;
            // the cell beside the plane on the owner: at the plane beyond a hi face, below it
            // beyond a lo face
            let beside = if face.side == 1 { face.face_pos as isize } else { face.face_pos as isize - 1 };
            for &tt in &families {
                if tt == bb {
                    continue;
                }
                let aa = 3 - tt - bb;
                for (node, upper) in [(patch[aa].0, false), (patch[aa].1, true)] {
                    let on_cut = if upper {
                        node == owner_ext[aa].0 + owner_ext[aa].1 && owner_tc[aa] + 1 < counts[aa]
                    } else {
                        node == owner_ext[aa].0 && owner_tc[aa] > 0
                    };
                    if !on_cut {
                        continue;
                    }
                    let mut across_tc = owner_tc;
                    across_tc[aa] = if upper { owner_tc[aa] + 1 } else { owner_tc[aa] - 1 };
                    let across = TileId(flatten(across_tc, counts) as u32);
                    let across_ext = partition.tile_extents(across_tc);
                    let region = |ext: [(usize, usize); D]| -> [Span; D] {
                        std::array::from_fn(|axis| {
                            let origin = ext[axis].0 as isize;
                            if axis == bb {
                                Span { lo: beside - origin, hi: beside - origin + 1 }
                            } else if axis == aa {
                                Span { lo: node as isize - origin, hi: node as isize - origin + 1 }
                            } else {
                                let lo = patch[axis].0.max(ext[axis].0) as isize - origin;
                                let hi = patch[axis].1.min(ext[axis].0 + ext[axis].1) as isize - origin;
                                Span { lo, hi }
                            }
                        })
                    };
                    slabs.push((aa, owner, region(owner_ext), across, region(across_ext)));
                }
            }
        }
        Self::slab_plan(partition, topology, ng, schema, slabs)
    }

    /// whether `transfer` moves at `point`: an interface audit at the `Audit` points, a banded
    /// cell field at the `Troubled` points, every other transfer at every other root point;
    /// a root plan moves nothing at a fine point.
    pub fn moves_at(&self, transfer: &Transfer<D>, point: ExchangePoint) -> bool {
        let banded = matches!(
            self.schema.entries[transfer.field as usize].kind,
            FieldKind::CellBand(_)
        );
        match point {
            ExchangePoint::Fine { .. }
            | ExchangePoint::CorrectedFaces
            | ExchangePoint::Nested { .. }
            | ExchangePoint::NestedRestrict { .. }
            | ExchangePoint::Seed { .. }
            | ExchangePoint::TracerCoefficients
            | ExchangePoint::RootCoefficients => false,
            ExchangePoint::Audit(_) => transfer.audit,
            ExchangePoint::Troubled(_)
            | ExchangePoint::EquilibriumTroubled
            | ExchangePoint::EquilibriumLevel { troubled: true, .. } => banded,
            _ => !banded && !transfer.audit,
        }
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
        match &self.seam {
            Some(s) => {
                h.u8(1);
                h.u64(s.mirror as u64);
                h.u8(u8::from(s.sides[0]));
                h.u8(u8::from(s.sides[1]));
                h.u64(s.azimuth as u64);
                for w in &s.slabs {
                    h.u64(*w as u64);
                }
            }
            None => h.u8(0),
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
                FieldKind::CellBand(band) => {
                    h.u8(2);
                    h.u8(band);
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
            h.u8(u8::from(t.audit));
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

    /// a placement of a level some workers hold no tile of: the fine level of a refined run,
    /// whose patch reaches only some of the root tiles. every worker still walks the level's
    /// phases; one without a tile sends and receives nothing there.
    pub fn with_idle_workers(workers: u32, owner: Vec<WorkerId>) -> Result<Self, String> {
        if workers == 0 {
            return Err("a placement needs at least one worker".into());
        }
        if let Some(bad) = owner.iter().find(|w| w.0 >= workers) {
            return Err(format!("tile owner {bad:?} is outside the {workers} workers"));
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

    /// a stencil box of y faces past both periodic ends of the y axis: each face past an end is
    /// read at its periodic image, the closing face y = 24 from its own copy alone, every face
    /// once; under the open topology the same box has faces no tile owns
    #[test]
    fn a_face_stencil_past_a_periodic_face_reads_the_periodic_image() {
        let p = ragged();
        let mut boxes: Vec<Option<Vec<[Span; 2]>>> = vec![None; p.n_tiles()];
        let gbox = [Span { lo: 2, hi: 6 }, Span { lo: -2, hi: 27 }];
        boxes[0] = Some(vec![[Span { lo: 2, hi: 6 }, Span { lo: 0, hi: 24 }], gbox]);
        let periodic = Topology::wrapping([false, true]);
        let plan = ExchangePlan::face_stencil(&p, &periodic, &boxes, 2).expect("the periodic image owns every face past the ends");
        // the source face of each destination y face, read off the component-1 transfers
        let mut source_of = std::collections::BTreeMap::new();
        for t in plan.axes[1].transfers.iter().filter(|t| t.field == 1) {
            // the source in whole-grid face indices: the source tile's first cell offsets its region
            let src_lo = p.tile_extents(unflatten(t.src.0 as usize, p.counts()))[1].0 as isize;
            for (dy, sy) in (t.dst_region[1].lo..t.dst_region[1].hi).zip(t.src_region[1].lo..t.src_region[1].hi) {
                assert!(source_of.insert(dy, sy + src_lo).is_none(), "face y = {dy} has two owners");
            }
        }
        let expected: std::collections::BTreeMap<isize, isize> =
            (-2..27).map(|y| (y, if y < 0 { y + 24 } else if y > 24 { y - 24 } else { y })).collect();
        assert_eq!(source_of, expected, "the faces past the ends read their periodic images, the closing face its own copy");
        let err = ExchangePlan::face_stencil(&p, &Topology::open(), &boxes, 2).err().expect("the open topology owns no face past the ends");
        assert!(format!("{err:?}").contains("no tile owns"), "{err:?}");
    }

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
            let schema = mhd_schema().cell_band("fofc_flag", 1);
            let plan = ExchangePlan::compile(&p, &Topology::open(), &schema, ng).unwrap();
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
                        FieldKind::CellBand(band) => (
                            FieldLayout::cell_band_of_geometry(&tiles[leg.lo].geom, band as usize),
                            FieldLayout::cell_band_of_geometry(&tiles[leg.hi].geom, band as usize),
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
            let halos = plan.transfers().filter(|t| !t.audit).count();
            assert_eq!(seen, halos, "every halo transfer was matched");
            assert!(seen > 0);
        }
    }

    /// an interface audit joins the closing face of the lower tile to the opening face of the
    /// upper one. on a wrapping axis with distinct ends (a periodic radius) the wrap leg joins
    /// two different faces and carries no audit; interior cuts and wraps whose ends coincide
    /// carry one audit per leg, and the halo transfers stay the same.
    #[test]
    fn a_wrap_with_distinct_ends_carries_no_audit() {
        let p = even();
        let wrapping = Topology::wrapping([true, true]);
        let audits = |topology: &Topology<2>, axis: u8| {
            let plan = ExchangePlan::compile(&p, topology, &mhd_schema(), 2).unwrap();
            let n = plan.transfers().filter(|t| t.audit && t.axis == axis).count();
            (n, plan.transfers().filter(|t| !t.audit).count())
        };
        let (coincide_x, halos) = audits(&wrapping, 0);
        let (distinct_x, distinct_halos) = audits(&wrapping.with_distinct_ends([true, false]), 0);
        let (distinct_y, _) = audits(&wrapping.with_distinct_ends([true, false]), 1);
        // two tile rows, each with an interior cut and a wrap along x
        assert_eq!(coincide_x, 4, "every x leg of a coinciding wrap carries an audit");
        assert_eq!(distinct_x, 2, "only the interior x cuts carry an audit when the x ends are distinct");
        assert_eq!(distinct_y, 4, "the y wrap keeps its audit");
        assert_eq!(distinct_halos, halos, "the halo transfers are the same");
    }

    /// a destination lies in the ghost band of its field on the leg's axis and a source lies in
    /// the interior, on cells and on faces alike.
    #[test]
    fn destinations_are_ghosts_and_sources_are_interior() {
        let p = ragged();
        let plan = ExchangePlan::compile(&p, &Topology::open(), &mhd_schema(), 3).unwrap();
        for t in plan.transfers().filter(|t| !t.audit) {
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

    /// every cut along a face field's own axis carries exactly one audit of that field: the
    /// lower tile's closing face against the upper tile's opening face, the same global
    /// faces, over the transverse interior both tiles own. an audit moves at the audit points
    /// and no other, and no halo transfer moves there.
    #[test]
    fn each_interface_is_audited_once_over_the_faces_both_tiles_own() {
        let p = ragged();
        let plan = ExchangePlan::compile(&p, &Topology::open(), &mhd_schema(), 3).unwrap();
        let schedule = Schedule::derive(p.counts(), 3, &Topology::open());
        let mut audits = 0;
        for t in plan.transfers() {
            for point in [ExchangePoint::Prime, ExchangePoint::Stage(0), ExchangePoint::Troubled(0)] {
                assert_eq!(plan.moves_at(t, point), !t.audit && point != ExchangePoint::Troubled(0));
            }
            assert_eq!(plan.moves_at(t, ExchangePoint::Audit(AuditAt::Checkpoint)), t.audit);
            if !t.audit {
                continue;
            }
            audits += 1;
            let ax = t.axis as usize;
            assert_eq!(plan.schema.entries[t.field as usize].kind, FieldKind::Face(t.axis));
            let (below, above) = (&plan.tiles[t.src.0 as usize], &plan.tiles[t.dst.0 as usize]);
            assert_eq!(t.src_region[ax], Span { lo: below.interior[ax].hi, hi: below.interior[ax].hi + 1 });
            assert_eq!(t.dst_region[ax], Span { lo: above.interior[ax].lo, hi: above.interior[ax].lo + 1 });
            for other in (0..2).filter(|o| *o != ax) {
                assert_eq!(t.src_region[other], below.interior[other]);
                assert_eq!(t.dst_region[other], above.interior[other]);
                assert_eq!(
                    below.offset[other] + t.src_region[other].lo,
                    above.offset[other] + t.dst_region[other].lo,
                    "the two regions name different global faces"
                );
            }
            assert_eq!(
                below.offset[ax] + t.src_region[ax].lo,
                above.offset[ax] + t.dst_region[ax].lo,
                "the two regions name different global faces"
            );
        }
        let cuts_along = |d: usize| schedule.legs().iter().filter(|l| l.axis == d).count();
        assert_eq!(audits, cuts_along(0) + cuts_along(1), "one audit per cut per normal field");
        assert!(audits > 0);
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

    /// a banded cell field moves the one ghost layer next to the interior under a three-cell
    /// allocation, at the `Troubled` points and no other; every other field moves at every
    /// other point.
    #[test]
    fn a_banded_field_fills_the_layer_next_to_the_interior_at_troubled_points_only() {
        let p = ragged();
        let schema = FieldSchema::new().cell("rho").cell_band("fofc_flag", 1);
        let plan = ExchangePlan::compile(&p, &Topology::open(), &schema, 3).unwrap();
        let mut banded = 0;
        for t in plan.transfers() {
            let is_band = t.field == 1;
            assert_eq!(plan.moves_at(t, ExchangePoint::Troubled(0)), is_band);
            assert_eq!(plan.moves_at(t, ExchangePoint::Stage(0)), !is_band);
            assert_eq!(plan.moves_at(t, ExchangePoint::Prime), !is_band);
            assert_eq!(plan.moves_at(t, ExchangePoint::Rollback), !is_band);
            if !is_band {
                continue;
            }
            banded += 1;
            let ax = t.axis as usize;
            let interior = plan.tiles[t.dst.0 as usize].interior[ax];
            let ghost = t.dst_region[ax];
            assert_eq!(ghost.size(), 1);
            assert!(
                ghost.lo == interior.hi || ghost.hi == interior.lo,
                "ghost {ghost:?} is not the layer next to the interior {interior:?}"
            );
        }
        assert!(banded > 0, "the plan carries no banded transfer");
    }

    /// a patch corner on two cuts: each tile beside a face on a cut corrects the face at the
    /// corner on its own transverse cut, and the tile beyond both cuts receives both copies;
    /// a patch face on one cut with its corner inside the neighbor's interior sends nothing
    #[test]
    fn the_corrected_faces_reach_the_tile_beyond_a_corner_on_two_cuts() {
        let partition = Partition::explicit([32, 32], [vec![17], vec![13]]).unwrap();
        let open = Topology::open();
        let corner = ExchangePlan::corrected_faces(&partition, &open, [(12, 17), (8, 13)], 5).unwrap();
        let moves: Vec<(u32, u32, u16)> = corner.transfers().map(|t| (t.src.0, t.dst.0, t.field)).collect();
        // tiles: 0 = (x lo, y lo), 1 = (x lo, y hi), 2 = (x hi, y lo), 3 = (x hi, y hi)
        assert_eq!(moves.len(), 2, "{moves:?}");
        assert!(moves.contains(&(2, 3, 1)), "the x-face owner sends its corrected y-face to the diagonal: {moves:?}");
        assert!(moves.contains(&(1, 3, 0)), "the y-face owner sends its corrected x-face to the diagonal: {moves:?}");
        for t in corner.transfers() {
            assert_eq!(t.len, 1, "one face per corner node in 2D");
        }
        let face_only = ExchangePlan::corrected_faces(&partition, &open, [(12, 17), (4, 10)], 5).unwrap();
        assert_eq!(face_only.transfers().count(), 0, "the corner lies inside the neighbor's interior");
    }

    /// a root plan moves nothing at a fine point and everything but bands and audits at the
    /// restriction
    #[test]
    fn a_root_plan_is_idle_at_fine_points_and_moves_state_at_the_restriction() {
        let p = ragged();
        let schema = FieldSchema::new().cell("rho").cell_band("fofc_flag", 1);
        let plan = ExchangePlan::compile(&p, &Topology::open(), &schema, 3).unwrap();
        for t in plan.transfers() {
            let is_band = t.field == 1;
            assert!(!plan.moves_at(t, ExchangePoint::Fine { substep: 0, point: FinePoint::Stage(0) }));
            assert!(!plan.moves_at(t, ExchangePoint::Fine { substep: 1, point: FinePoint::Prolong }));
            assert_eq!(plan.moves_at(t, ExchangePoint::Restrict), !is_band && !t.audit);
        }
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
            clip: [[false, false], [false, true]],
        };
        assert_eq!(first.src.0 as usize, leg.hi);
        assert_eq!(first.dst.0 as usize, leg.lo);
    }
}
