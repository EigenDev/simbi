// =============================================================================
// checkpoint.rs
//
// HDF5 checkpoint API for SimState. a thin builder that maps SimState onto a
// `symbi_io::Tree` schema and hands off to `symbi_io::Hdf5Backend`. the I/O
// concerns (file format, field naming, error handling) live in the symbi-io
// crate; this module only describes WHAT SimState contributes to the schema.
//
// the on-disk layout is preserved bit-for-bit (existing `scripts/plot_*.py`
// readers + every existing checkpoint file continue to work). public API
// changes:
//   - `write_checkpoint(sim, path, extras: &Metadata)` — typed extras kill
//     the `&[(&str, &str)]` stringly-typed pattern. callers build
//     `Metadata::new().with("key", value)` with naked typed values.
//   - same for `load_checkpoint` / `read_checkpoint_meta`, which now return
//     `Result<_, symbi_io::IoError>`.
//
// the `CheckpointSchedule` helper is independent of the I/O schema.
// =============================================================================

use std::path::Path;

use crate::state::*;
use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

use symbi_hydro::FieldSpec;
pub use symbi_io::{Attr, IoError, Metadata, Result};
use symbi_io::{DataRef, Dataset, Hdf5Backend, IoBackend, Tree, TreeBuf};

// =============================================================================
// CheckpointSchedule — unchanged, owned here for historical compatibility.
// =============================================================================

#[derive(Clone, Debug)]
pub struct CheckpointSchedule {
    pub next_time: f64,
    pub interval: f64,
    pub dlogt: f64,
    pub tstart: f64,
    pub index: usize,
}

impl CheckpointSchedule {
    pub fn linear(interval: f64) -> Self {
        CheckpointSchedule {
            next_time: interval,
            interval,
            dlogt: 0.0,
            tstart: 0.0,
            index: 0,
        }
    }
    pub fn logarithmic(dlogt: f64, tstart: f64) -> Self {
        CheckpointSchedule {
            next_time: tstart * 10.0_f64.powf(dlogt),
            interval: 0.0,
            dlogt,
            tstart,
            index: 0,
        }
    }
    pub fn should_checkpoint(&self, time: f64) -> bool {
        time >= self.next_time - 1e-14
    }
    pub fn advance(&mut self) {
        self.index += 1;
        if self.dlogt != 0.0 {
            self.next_time = self.tstart * 10.0_f64.powf((self.index + 1) as f64 * self.dlogt);
        } else {
            self.next_time = (self.index + 1) as f64 * self.interval;
        }
    }
    pub fn next(&self) -> f64 {
        self.next_time
    }
    pub fn with_index(mut self, idx: usize) -> Self {
        self.index = idx;
        if self.dlogt != 0.0 {
            self.next_time = self.tstart * 10.0_f64.powf((self.index + 1) as f64 * self.dlogt);
        } else {
            self.next_time = (self.index + 1) as f64 * self.interval;
        }
        self
    }
    pub fn filename(&self, prefix: &str, data_dir: &str) -> String {
        format!("{}/{}_{:04}.h5", data_dir, prefix, self.index)
    }
}

// =============================================================================
// Snapshot — the materialized Vec<f64> buffers a write needs to BORROW
// when building the Tree. one struct, holds every field's interior data,
// no copies during Tree construction.
// =============================================================================

/// the owned per-bucket buffers a write borrows from. each entry is a
/// `(canonical_name, interior_data)` pair — the canonical name comes from
/// `symbi_io::dataset_name(fs, idx)` driven by `R::SPEC.{fields,
/// primitive_fields}`. one source of truth per regime; the writer never
/// hand-spells "m1..mD".
struct Snapshot<const D: usize> {
    resolution: Vec<u64>,
    // the ALLOCATED (padded) cell extent per axis = interior + 2*ng. cell-centered
    // field datasets are written at this full extent so a restart restores the
    // entire field — ghost zones included — not just the interior (which would
    // truncate the halo the next step's stencil reads before the first ghost-fill).
    // the reader trims `halo_radius` (= ng) back to the interior for plotting.
    data_shape: Vec<u64>,
    // interior cell counts in STORAGE (reversed) axis order for the reader's
    // `mesh/global_cells` ([nx3,nx2,nx1]); matches the reversed field `shape` so the
    // plot axes are not transposed (a non-square grid otherwise crashes pcolormesh).
    mesh_cells: Vec<u64>,
    dx_phys: Vec<f64>,
    x_lo_phys: Vec<f64>,
    conserved: Vec<(String, Vec<f64>)>,
    primitive: Vec<(String, Vec<f64>)>,
    bface: Vec<(String, Vec<f64>)>, // canonical "B1".."BD" face-centered B, MHD only
    // per-face (start, fin) index bounds for the reader's `magnetic/Bn/domain` group.
    bface_dom: Vec<(Vec<i64>, Vec<i64>)>,
    // single-partition owned cell range for the `partition_0` group the frozen
    // v2.0 reader expects: start = [0; D], fin = interior cell counts.
    owned_start: Vec<i64>,
    owned_fin: Vec<i64>,
}

/// visit every cell of `domain` in AXIS-0-FASTEST order (x varies fastest) — the on-disk
/// checkpoint layout, so numpy `arr.reshape((Nz, Ny, Nx))` puts physical x on the horizontal
/// `imshow` axis. gather (`extract_field`) and scatter (`restore_field`) MUST share this one
/// walk: if their orders diverge, a written-then-loaded D>=2 field comes back transposed.
fn for_each_cell_axis0<const D: usize>(
    domain: &symbi_algebra::Domain<D>,
    mut visit: impl FnMut([isize; D]),
) {
    let vol = domain.volume();
    let mut coord: [isize; D] = std::array::from_fn(|ax| domain.spaces[ax].lo);
    for _ in 0..vol {
        visit(coord);
        for ax in 0..D {
            coord[ax] += 1;
            if coord[ax] < domain.spaces[ax].hi {
                break;
            }
            coord[ax] = domain.spaces[ax].lo;
        }
    }
}

fn extract_field<const D: usize, Mem: MemorySpace>(
    field: &symbi_grid::Field<f64, D, Mem>,
    domain: &symbi_algebra::Domain<D>,
) -> Vec<f64> {
    let mut data = Vec::with_capacity(domain.volume());
    for_each_cell_axis0(domain, |coord| data.push(*field.view().at(coord)));
    data
}

/// drive the canonical iteration of one bucket's `FieldSpec` list. for each
/// (FieldSpec \times component-idx), the closure dispatches to the right struct
/// member or returns `None` when the field isn't allocated for this regime
/// (e.g., iso has no cons.nrg, non-MHD has no mhd.bcell). returns the
/// `(name, data)` vec the snapshot uses. each cell-centered field is gathered
/// over its OWN allocated domain (`field.domain()` — interior + ghosts), so the
/// written buffer carries the halo and a restart is not truncated.
fn collect_bucket<'a, F, const D: usize, Mem: MemorySpace>(
    fields: &[FieldSpec],
    // the vector-component count for DimVector fields (mom/vel): the momentum DOF, which
    // exceeds the grid dimension for a lifted (swirl) run — every stored component writes.
    dof: usize,
    mut pick: F,
) -> Vec<(String, Vec<f64>)>
where
    F: FnMut(&FieldSpec, usize) -> Option<&'a symbi_grid::Field<f64, D, Mem>>,
{
    let mut out = Vec::new();
    for fs in fields {
        let n = symbi_io::component_count(fs, dof);
        for idx in 0..n {
            if let Some(field) = pick(fs, idx) {
                let name = symbi_io::dataset_name(fs, idx);
                out.push((name, extract_field(field, field.domain())));
            }
        }
    }
    out
}

fn snapshot<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
) -> Snapshot<D>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    // before READING fields here, sync the device so the host sees the
    // committed state from the last RK2 stage (removing the per-launch `ctx_sync()`
    // was a production pipelining win). gate on `IS_DEVICE_ACCESSIBLE`, NOT just the
    // cuda feature: a HOST-memory sim in a cuda-feature build has no CUDA context, so
    // an unconditional `cuCtxSynchronize` panics (CUDA_ERROR_INVALID_CONTEXT). only
    // device-resident memory needs the sync; host memory is already coherent.
    #[cfg(feature = "gpu")]
    if Mem::IS_DEVICE_ACCESSIBLE {
        symbi_xpu::ctx_sync();
    }
    let interior = &sim.geom.interior;
    let a = sim.motion.a;
    let resolution: Vec<u64> = (0..D).map(|ax| interior.spaces[ax].size() as u64).collect();
    // the allocated (padded) cell extent — `den` exists in every regime, so its
    // domain is the canonical cell domain (interior + ng ghosts on every side).
    // cell-centered datasets are written at this extent; the reader trims ng back.
    let alloc = sim.fields.cons.den.domain();
    let data_shape: Vec<u64> = (0..D).map(|ax| alloc.spaces[ax].size() as u64).collect();
    let mesh_cells: Vec<u64> = (0..D)
        .rev()
        .map(|ax| interior.spaces[ax].size() as u64)
        .collect();
    // homologous expansion is RADIAL: a(t) scales the radial coordinate (and, in cartesian,
    // every coordinate isotropically), but NEVER an angular one (theta/phi) — a moving spherical
    // mesh must not report theta -> a*theta. mirror the per-geometry volume jacobian (block.rs:
    // spherical ~a^3 = only r; cylindrical ~a^2 = r,z; cartesian ~a^D = all). axis 0 is x1 (r).
    let metric_geom = sim.physics.metric.geometry();
    let axis_scale = |ax: usize| -> f64 {
        let is_length = match metric_geom {
            symbi_geometry::Geometry::Cartesian => true,
            symbi_geometry::Geometry::Spherical => ax == 0,
            symbi_geometry::Geometry::Cylindrical => ax == 0 || ax == D - 1,
        };
        if is_length { a } else { 1.0 }
    };
    let dx_phys: Vec<f64> = sim.geom.dx[..D]
        .iter()
        .enumerate()
        .map(|(ax, &d)| d * axis_scale(ax))
        .collect();
    let x_lo_phys: Vec<f64> = sim.geom.x_lo[..D]
        .iter()
        .enumerate()
        .map(|(ax, &x)| x * axis_scale(ax))
        .collect();
    // ----- RegimeSpec-driven conserved iteration ------
    let conserved = collect_bucket(R::SPEC.fields, DOF, |fs, idx| match fs.name {
        "den" => Some(&sim.fields.cons.den),
        "mom" => Some(&sim.fields.cons.mom[idx]),
        "nrg" => sim.fields.cons.nrg_field(),
        "mag" => sim.fields.mhd.as_ref().map(|m| &m.bcell[idx]),
        other => panic!("checkpoint write: unknown conserved field '{other}'"),
    });

    // ----- RegimeSpec-driven primitive iteration ------
    let primitive = collect_bucket(R::SPEC.primitive_fields, DOF, |fs, idx| match fs.name {
        "rho" => Some(&sim.fields.prim.rho),
        "vel" => Some(&sim.fields.prim.vel[idx]),
        "pre" => sim.fields.prim.pre_field(),
        "bcell" => sim.fields.mhd.as_ref().map(|m| &m.bcell[idx]),
        other => panic!("checkpoint write: unknown primitive field '{other}'"),
    });

    // ----- face-centered B (CT ground truth) — separate group ----
    let (bface, bface_dom): (Vec<(String, Vec<f64>)>, Vec<(Vec<i64>, Vec<i64>)>) =
        if let Some(ref mhd) = sim.fields.mhd {
            if mhd
                .bface_initialized
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                (0..D)
                    .map(|d| {
                        let face_dom = interior.extend(d, 0, 1);
                        let name = format!("B{}", d + 1);
                        let start: Vec<i64> =
                            (0..D).map(|ax| face_dom.spaces[ax].lo as i64).collect();
                        let fin: Vec<i64> =
                            (0..D).map(|ax| face_dom.spaces[ax].hi as i64).collect();
                        (
                            (name, extract_field(&mhd.bface[d], &face_dom)),
                            (start, fin),
                        )
                    })
                    .unzip()
            } else {
                (Vec::new(), Vec::new())
            }
        } else {
            (Vec::new(), Vec::new())
        };

    // the owned interior index range [0, ncells) per axis, in the SAME reversed (storage) axis order
    // as `mesh_cells` and the `dim_*` geometry. WITHOUT the reverse, owned is (x, y, ..) while
    // global_cells / dims are (.., y, x): the reader then pairs the y geometry with the x extent, and
    // a NON-square AMR fine patch renders transposed / offset (a square grid hides it -- both orders
    // agree). matches the `(0..D).rev()` walk used for `mesh_cells` above.
    let owned_start: Vec<i64> = vec![0; D];
    let owned_fin: Vec<i64> = (0..D).rev().map(|ax| interior.spaces[ax].size() as i64).collect();

    Snapshot {
        resolution,
        data_shape,
        mesh_cells,
        dx_phys,
        x_lo_phys,
        conserved,
        primitive,
        bface,
        bface_dom,
        owned_start,
        owned_fin,
    }
}

// =============================================================================
// build_tree — the schema description SimState contributes. one place that
// owns the on-disk layout; the Hdf5Backend (or any other) walks it.
// =============================================================================

fn regime_name<R: Regime<f64, D>, const D: usize>(r: &R) -> &'static str {
    // `has_energy() == false` marks the isothermal (IsoModel) regimes, which carry
    // no energy equation — distinguish them so the checkpoint regime is faithful.
    if r.is_mhd() {
        if r.is_relativistic() {
            "rmhd"
        } else if r.has_energy() {
            "mhd"
        } else {
            "imhd"
        }
    } else if r.is_relativistic() {
        "rhd"
    } else if r.has_energy() {
        "newtonian"
    } else {
        "isothermal"
    }
}

fn coord_name(g: symbi_geometry::Geometry) -> &'static str {
    match g {
        symbi_geometry::Geometry::Cartesian => "cartesian",
        symbi_geometry::Geometry::Spherical => "spherical",
        symbi_geometry::Geometry::Cylindrical => "cylindrical",
    }
}

fn timestepping_name(t: Timestepping) -> &'static str {
    match t {
        Timestepping::Euler => "euler",
        Timestepping::Rk2 => "rk2",
        Timestepping::Rk3 => "rk3",
    }
}

/// the global `/metadata` group — time/physics/scheme attrs (same across an AMR
/// hierarchy, so authored from the coarse level) + the coarse mesh datasets for
/// single-level readers.
fn build_metadata_group<'a, R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &'a SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    snap: &'a Snapshot<D>,
    extras: &'a Metadata,
) -> Tree<'a>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    // builtins. user extras can override any name here — explicit win.
    let builtins: Vec<(&str, Attr)> = vec![
        ("gamma", Attr::F64(sim.physics.eos.gamma())),
        ("cfl", Attr::F64(sim.cfl)),
        ("time", Attr::F64(sim.time)),
        ("dt", Attr::F64(sim.dt)),
        ("iteration", Attr::U64(sim.iteration as u64)),
        ("dimensions", Attr::U64(D as u64)),
        ("halo_radius", Attr::U64(sim.geom.ng as u64)),
        ("scale_factor", Attr::F64(sim.motion.a)),
        ("scale_factor_dot", Attr::F64(sim.motion.a_dot)),
        ("homologous", Attr::Bool(sim.motion.homologous)),
        ("regime", Attr::Str(regime_name(&sim.physics.regime).into())),
        ("is_mhd", Attr::Bool(sim.physics.regime.is_mhd())),
        (
            "is_relativistic",
            Attr::Bool(sim.physics.regime.is_relativistic()),
        ),
        (
            "timestepping",
            Attr::Str(timestepping_name(sim.timestepping).into()),
        ),
        (
            "coord_system",
            Attr::Str(coord_name(sim.physics.metric.geometry()).into()),
        ),
    ];
    let mut meta = Tree::new("metadata");
    // explicit user extras WIN. start with them, then fill in any built-in
    // the user didn't override.
    for (k, v) in extras {
        meta.push_attr(k.to_string(), v.clone());
    }
    for (k, v) in builtins {
        if extras.get(k).is_some() {
            continue;
        }
        meta.push_attr(k.to_string(), v);
    }
    // coarsest-level mesh info (backward compat with single-level readers).
    meta.push_dataset(Dataset::new(
        "resolution",
        vec![D],
        DataRef::U64(&snap.resolution),
    ));
    meta.push_dataset(Dataset::new("dx", vec![D], DataRef::F64(&snap.dx_phys)));
    meta.push_dataset(Dataset::new("x_lo", vec![D], DataRef::F64(&snap.x_lo_phys)));
    meta
}

/// one `/level_{idx}` group — mesh geometry + partition_0/hydro (+ magnetic) +
/// conserved. each AMR level carries its own resolution / origin / dx, so this is
/// authored per (sim, snap). `idx` is the refinement level (0 = coarse).
fn build_level_group<'a, R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &'a SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    snap: &'a Snapshot<D>,
    idx: usize,
) -> Tree<'a>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    // ---- mesh — frozen v2.0 reader schema: global_cells + geometry ----
    // the reader rebuilds cell centers from (global_cells, per-dim start/end, type),
    // so the mesh carries the geometry description, not the precomputed coordinate
    // arrays. symbi's own `load_checkpoint` reconstructs geometry from config, not
    // from this group, so this layout serves the reader without breaking restart.
    let mut geometry =
        Tree::new("geometry").with_attr("metric", coord_name(sim.physics.metric.geometry()));
    // mesh metadata (global_cells + per-dim geometry) is written in STORAGE
    // (reversed) axis order so it matches the reversed field `shape` below and the
    // reader's [nx3,nx2,nx1] expectation: dim_0 is the SLOWEST-varying screen axis,
    // dim_{D-1} the fastest (x1). without the reverse, global_cells/dims are
    // transposed vs the data -> a SQUARE grid plots mis-oriented and a NON-square
    // grid crashes pcolormesh ("C dims should be one smaller than X and Y").
    for (slot, ax) in (0..D).rev().enumerate() {
        // the interior lower edge — honors an AMR fine level whose interior
        // starts at a non-zero global index (start = global origin offset by
        // the interior origin), so the reader rebuilds cell centers correctly.
        let start = snap.x_lo_phys[ax] + sim.geom.interior.spaces[ax].lo as f64 * snap.dx_phys[ax];
        let end = start + snap.dx_phys[ax] * snap.resolution[ax] as f64;
        // the per-axis spacing the reader reconstructs cell centers from: "linear" -> uniform faces
        // start + i*dx, "log" -> geometric faces start*10^(i*slope). taken from the grid's coordinate
        // maps (uniform when unset). start/end are the axis domain bounds [r_lo, r_hi]; logspace over
        // them recovers the geometric grid.
        let spacing_label = match &sim.geom.maps {
            Some(maps) if !maps[ax].is_uniform() => "log",
            _ => "linear",
        };
        geometry.push_group(
            Tree::new(format!("dim_{slot}"))
                .with_attr("start", start)
                .with_attr("end", end)
                .with_attr("type", spacing_label),
        );
    }
    let mesh = Tree::new("mesh")
        .with_attr("halo_width", sim.geom.ng as u64)
        .with_dataset(Dataset::new(
            "global_cells",
            vec![D],
            DataRef::U64(&snap.mesh_cells),
        ))
        .with_group(geometry);

    // declare shape in REVERSED axis order so it matches the on-disk layout
    // `extract_field` produces (axis-0-fastest in memory -> axis-0 is the LAST
    // dim of the numpy shape). numpy/matplotlib then put physical x on the
    // horizontal screen axis. for 2D OT: shape = [Ny, Nx]; for 3D: [Nz, Ny, Nx].
    // cell datasets carry the PADDED extent (interior + 2*ng); the reader trims
    // `halo_width` per side back to the interior `global_cells` for plotting.
    let shape: Vec<usize> = (0..D)
        .rev()
        .map(|ax| snap.data_shape[ax] as usize)
        .collect();

    // ---- partition_0/hydro/primitives (RegimeSpec-driven) ----
    let mut prim = Tree::new("primitives");
    for (name, data) in &snap.primitive {
        prim.push_dataset(Dataset::new(
            name.clone(),
            shape.clone(),
            DataRef::F64(data),
        ));
    }
    let mut hydro = Tree::new("hydro").with_group(prim);

    // face-centered B (MHD) under hydro/magnetic — each B-face as the GROUP the
    // frozen reader expects: `Bn/{domain/{start,fin}, data}`.
    if !snap.bface.is_empty() {
        let interior = &sim.geom.interior;
        let mut magnetic = Tree::new("magnetic");
        for (face_ax, (name, data)) in snap.bface.iter().enumerate() {
            let face_dom = interior.extend(face_ax, 0, 1);
            let face_shape: Vec<usize> = (0..D)
                .rev()
                .map(|ax| face_dom.spaces[ax].size() as usize)
                .collect();
            let (start, fin) = &snap.bface_dom[face_ax];
            let domain = Tree::new("domain")
                .with_dataset(Dataset::new("start", vec![D], DataRef::I64(start)))
                .with_dataset(Dataset::new("fin", vec![D], DataRef::I64(fin)));
            magnetic.push_group(
                Tree::new(name.clone())
                    .with_group(domain)
                    .with_dataset(Dataset::new("data", face_shape, DataRef::F64(data))),
            );
        }
        hydro.push_group(magnetic);
    }

    let partition_0 = Tree::new("partition_0")
        .with_dataset(Dataset::new(
            "owned_start",
            vec![D],
            DataRef::I64(&snap.owned_start),
        ))
        .with_dataset(Dataset::new(
            "owned_fin",
            vec![D],
            DataRef::I64(&snap.owned_fin),
        ))
        .with_group(hydro);

    // ---- conserved — kept as the primary for symbi's own restart ----
    let mut cons = Tree::new("conserved");
    for (name, data) in &snap.conserved {
        cons.push_dataset(Dataset::new(
            name.clone(),
            shape.clone(),
            DataRef::F64(data),
        ));
    }

    Tree::new(format!("level_{idx}"))
        .with_attr("scale_factor_a", sim.motion.a)
        .with_attr("scale_factor_adot", sim.motion.a_dot)
        .with_group(mesh)
        .with_group(partition_0)
        .with_group(cons)
}

fn build_tree<'a, R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &'a SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    snap: &'a Snapshot<D>,
    extras: &'a Metadata,
) -> Tree<'a>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let mut root = Tree::new("")
        .with_attr("format_version", "2.0")
        .with_attr("symbi_version", "0.1.0");
    root.push_group(build_metadata_group(sim, snap, extras));
    root.push_group(build_level_group(sim, snap, 0));
    root
}

// =============================================================================
// public API: write_checkpoint / load_checkpoint / read_checkpoint_meta
// =============================================================================

/// write a checkpoint. typed `Metadata` carries naked typed values
/// rather than a `&[(&str, &str)]` slice — no `to_string()` boilerplate at call sites:
///
/// ```ignore
/// let extras = Metadata::new()
///     .with("problem", "kepler")
///     .with("ring_r0", 1.0)
///     .with("gm",      gm);
/// write_checkpoint(&sim, "kepler_0001.h5", &extras)?;
/// ```
pub fn write_checkpoint<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
    extras: &Metadata,
) -> Result<()>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let snap = snapshot(sim);
    let tree = build_tree(sim, &snap, extras);
    Hdf5Backend.write(Path::new(path), &tree)
}

/// **AMR checkpoint** — write an entire refinement hierarchy into ONE file as
/// `/level_0`, `/level_1`, … sibling groups (the frozen v2.0 reader walks
/// `while level_i in f`). `levels[0]` is the coarse level and authors the global
/// `/metadata`; every level carries its own mesh + fields. this is the
/// "all levels, one file" layout.
///
/// ```ignore
/// let states: Vec<&SimState<..>> = hier.levels.iter().map(|l| &l.state).collect();
/// write_hierarchy_checkpoint(&states, "run_0001.h5", &extras)?;
/// ```
pub fn write_hierarchy_checkpoint<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    levels: &[&SimStateGeneric<R, D, DOF, M, E, S, Mem>],
    path: &str,
    extras: &Metadata,
) -> Result<()>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    if levels.is_empty() {
        return Err(IoError::MissingPath("hierarchy has no levels".into()));
    }
    // snapshot every level up front so the borrowed field data outlives the tree.
    let snaps: Vec<Snapshot<D>> = levels.iter().map(|s| snapshot(s)).collect();
    let mut root = Tree::new("")
        .with_attr("format_version", "2.0")
        .with_attr("symbi_version", "0.1.0");
    // global metadata authored from the coarse level.
    root.push_group(build_metadata_group(levels[0], &snaps[0], extras));
    for (idx, snap) in snaps.iter().enumerate() {
        root.push_group(build_level_group(levels[idx], snap, idx));
    }
    Hdf5Backend.write(Path::new(path), &root)
}

// =============================================================================
// CheckpointMeta — typed view of the metadata group, parsed once from the
// TreeBuf the read side returns.
// =============================================================================

#[derive(Clone, Debug)]
pub struct CheckpointMeta {
    pub time: f64,
    pub dt: f64,
    pub iteration: u64,
    pub gamma: f64,
    pub dimensions: u64,
    pub regime: String,
    pub coord_system: String,
}

fn read_meta_from(tree: &TreeBuf) -> Result<CheckpointMeta> {
    let m = tree
        .find_group("metadata")
        .ok_or_else(|| IoError::MissingPath("metadata".into()))?;
    Ok(CheckpointMeta {
        time: m
            .find_attr("time")
            .ok_or_else(|| IoError::MissingPath("metadata/time".into()))?
            .as_f64("metadata/time")?,
        dt: m
            .find_attr("dt")
            .ok_or_else(|| IoError::MissingPath("metadata/dt".into()))?
            .as_f64("metadata/dt")?,
        iteration: m
            .find_attr("iteration")
            .ok_or_else(|| IoError::MissingPath("metadata/iteration".into()))?
            .as_u64("metadata/iteration")?,
        gamma: m
            .find_attr("gamma")
            .ok_or_else(|| IoError::MissingPath("metadata/gamma".into()))?
            .as_f64("metadata/gamma")?,
        dimensions: m
            .find_attr("dimensions")
            .ok_or_else(|| IoError::MissingPath("metadata/dimensions".into()))?
            .as_u64("metadata/dimensions")?,
        // strings live as byte-array datasets, not as attrs (on-disk convention)
        regime: read_str_dataset(m, "regime").unwrap_or_else(|_| "unknown".into()),
        coord_system: read_str_dataset(m, "coord_system").unwrap_or_else(|_| "unknown".into()),
    })
}

fn read_str_dataset(tree: &TreeBuf, name: &str) -> Result<String> {
    let ds = tree
        .find_dataset(name)
        .ok_or_else(|| IoError::MissingPath(name.into()))?;
    match &ds.data {
        symbi_io::DataBuf::U8(b) => Ok(String::from_utf8_lossy(b).into_owned()),
        other => Err(IoError::TypeMismatch {
            path: name.into(),
            expected: "u8 (string)",
            actual: match other {
                symbi_io::DataBuf::F64(_) => "f64",
                _ => "?",
            },
        }),
    }
}

pub fn read_checkpoint_meta(path: &str) -> Result<CheckpointMeta> {
    let tree = Hdf5Backend.read(Path::new(path))?;
    read_meta_from(&tree)
}

/// load a checkpoint into an existing SimState. restores cons (and
/// prim if present, and bface if present) from disk; returns the typed
/// `CheckpointMeta` for the caller to consume time/iteration/etc.
pub fn load_checkpoint<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
) -> Result<CheckpointMeta>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let tree = Hdf5Backend.read(Path::new(path))?;
    let meta = read_meta_from(&tree)?;
    sim.time = meta.time;
    sim.dt = meta.dt;
    sim.iteration = meta.iteration;

    let level_0 = tree
        .find_group("level_0")
        .ok_or_else(|| IoError::MissingPath("level_0".into()))?;
    let interior = sim.geom.interior.clone();

    // conserved (primary — c2p will derive prims on restart). RegimeSpec-driven.
    let cons = level_0
        .find_group("conserved")
        .ok_or_else(|| IoError::MissingPath("level_0/conserved".into()))?;
    for fs in R::SPEC.fields {
        let n = symbi_io::component_count(fs, DOF);
        for idx in 0..n {
            let name = symbi_io::dataset_name(fs, idx);
            match fs.name {
                "den" => restore_field(
                    cons,
                    &name,
                    &sim.fields.cons.den,
                    sim.fields.cons.den.domain(),
                )?,
                "mom" => restore_field(
                    cons,
                    &name,
                    &sim.fields.cons.mom[idx],
                    sim.fields.cons.mom[idx].domain(),
                )?,
                "nrg" => {
                    if let Some(nrg) = sim.fields.cons.nrg_field() {
                        restore_field(cons, &name, nrg, nrg.domain())?;
                    }
                }
                "mag" => {
                    if let Some(ref mhd) = sim.fields.mhd {
                        restore_field(cons, &name, &mhd.bcell[idx], mhd.bcell[idx].domain())?;
                    }
                }
                other => panic!("checkpoint read: unknown conserved field '{other}'"),
            }
        }
    }

    // primitives (optional). RegimeSpec-driven iteration mirrors the write.
    if let Some(prim) = level_0.find_group("primitives") {
        for fs in R::SPEC.primitive_fields {
            let n = symbi_io::component_count(fs, DOF);
            for idx in 0..n {
                let name = symbi_io::dataset_name(fs, idx);
                match fs.name {
                    "rho" => restore_field(
                        prim,
                        &name,
                        &sim.fields.prim.rho,
                        sim.fields.prim.rho.domain(),
                    )?,
                    "vel" => restore_field(
                        prim,
                        &name,
                        &sim.fields.prim.vel[idx],
                        sim.fields.prim.vel[idx].domain(),
                    )?,
                    "pre" => {
                        if let Some(pre) = sim.fields.prim.pre_field() {
                            restore_field(prim, &name, pre, pre.domain())?;
                        }
                    }
                    "bcell" => {
                        if let Some(ref mhd) = sim.fields.mhd {
                            if prim.find_dataset(&name).is_some() {
                                restore_field(
                                    prim,
                                    &name,
                                    &mhd.bcell[idx],
                                    mhd.bcell[idx].domain(),
                                )?;
                            }
                        }
                    }
                    other => panic!("checkpoint read: unknown primitive field '{other}'"),
                }
            }
        }
    }

    // face-centered B (CT truth — restores div(B)=0 exactly)
    if let Some(ref mhd) = sim.fields.mhd {
        if let Some(mag) = level_0.find_group("magnetic") {
            let mut all_ok = true;
            for d in 0..D {
                let face_dom = interior.extend(d, 0, 1);
                let name = format!("B{}", d + 1);
                if restore_field(mag, &name, &mhd.bface[d], &face_dom).is_err() {
                    all_ok = false;
                }
            }
            if all_ok {
                mhd.bface_initialized
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    Ok(meta)
}

fn restore_field<const D: usize, Mem: MemorySpace>(
    tree: &TreeBuf,
    name: &str,
    field: &symbi_grid::Field<f64, D, Mem>,
    domain: &symbi_algebra::Domain<D>,
) -> Result<()> {
    let ds = tree
        .find_dataset(name)
        .ok_or_else(|| IoError::MissingPath(name.into()))?;
    let data = ds.data.as_f64().ok_or_else(|| IoError::TypeMismatch {
        path: name.into(),
        expected: "f64",
        actual: "non-f64",
    })?;
    let vol = domain.volume();
    if data.len() != vol {
        return Err(IoError::ShapeMismatch {
            path: name.into(),
            expected: vec![vol],
            actual: vec![data.len()],
        });
    }
    let view = field.view_mut();
    // SAME axis-0-fastest walk as `extract_field` — written-then-loaded is the identity by
    // construction. a `(0..D).rev()` walk would transpose every D>=2 restart.
    let mut ii = 0usize;
    for_each_cell_axis0(domain, |coord| {
        view.set(coord, data[ii]);
        ii += 1;
    });
    Ok(())
}

// =============================================================================
// tests
//
// the checkpoint mesh-coordinate gate for a NONZERO interior origin lives in-crate
// because it exercises `SimStateGeneric::new_at` — the absolute-index amr-internal
// constructor (pub(crate); the public path is `SimBuilder`, which always grids at
// interior_lo = [0; D]). amr fine levels live at absolute indices, so their written
// mesh/x{1,2,3} centers must equal geom.centroid of the ACTUAL interior cells.
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use symbi_geometry::Cartesian;
    use symbi_hydro::eos::IdealGas;
    use symbi_hydro::newtonian::Newtonian;
    use symbi_xpu::{CpuSpace, HostMemory};

    type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

    #[test]
    fn mesh_centers_respect_the_interior_origin() {
        // a fine-level-like state: interior [8, 12)^3 on a global origin at -1.
        let interior_lo = [8isize; 3];
        let sim = Sim::new_at(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            interior_lo,
            [4; 3],
            [-1.0; 3],
            [0.125; 3],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();

        let dir = std::env::temp_dir().join("symbi_checkpoint_offset_origin");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("offset.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        // the frozen v2.0 reader rebuilds cell centers from the geometry
        // description (global_cells + per-dim start/end), so verify THAT honors
        // the interior origin — reconstruct centers exactly as the reader does
        // (center_i = start + (i + 0.5) * (end - start) / n) and match centroid.
        let tree = Hdf5Backend.read(std::path::Path::new(path)).unwrap();
        let geometry = tree
            .find_group("level_0")
            .unwrap()
            .find_group("mesh")
            .unwrap()
            .find_group("geometry")
            .unwrap();
        for ax in 0..3 {
            let dim = geometry.find_group(&format!("dim_{ax}")).unwrap();
            let start = dim.find_attr("start").unwrap().as_f64("start").unwrap();
            let end = dim.find_attr("end").unwrap().as_f64("end").unwrap();
            let n = 4usize;
            let dx = (end - start) / n as f64;
            for ii in 0..n {
                let x = start + (ii as f64 + 0.5) * dx;
                let mut coord = [0isize; 3];
                coord[ax] = interior_lo[ax] + ii as isize;
                let expect = sim.geom.centroid(coord)[ax];
                assert!(
                    (x - expect).abs() < 1e-14,
                    "dim_{ax}[{ii}] reconstructs to {x} but the interior cell center is {expect}"
                );
            }
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_field_layout_2d() {
        // **restart round-trip gate**: write a checkpoint then load it; the conserved state must
        // come back IDENTICAL. a NON-SQUARE grid (5x3) seeded with an ASYMMETRIC pattern
        // (value = i + 100*j, distinct per field) makes any axis transpose between the gather
        // (`extract_field`) and scatter (`restore_field`) a loud failure — the bug that shipped
        // when the two walks used opposite axis orders (`0..D` vs `(0..D).rev()`).
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let build = || {
            Sim2::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize, 0],
                [5usize, 3],
                [0.0, 0.0],
                [0.2, 1.0 / 3.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };

        let sim = build();
        let nrg = sim
            .fields
            .cons
            .nrg_field()
            .expect("Newtonian cons.nrg")
            .clone();
        for c in sim.geom.interior.iter() {
            let (i, j) = (c[0] as f64, c[1] as f64);
            sim.fields.cons.den.view_mut().set(c, 1.0 + i + 100.0 * j);
            sim.fields.cons.mom[0]
                .view_mut()
                .set(c, 10.0 + i + 100.0 * j);
            sim.fields.cons.mom[1]
                .view_mut()
                .set(c, 20.0 + i + 100.0 * j);
            nrg.view_mut().set(c, 30.0 + i + 100.0 * j);
        }

        let dir = std::env::temp_dir().join("symbi_checkpoint_roundtrip_2d");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        let mut loaded = build();
        load_checkpoint(&mut loaded, path).unwrap();

        let lnrg = loaded.fields.cons.nrg_field().unwrap();
        for c in sim.geom.interior.iter() {
            assert_eq!(
                *loaded.fields.cons.den.view().at(c),
                *sim.fields.cons.den.view().at(c),
                "cons.den transposed/garbled at {c:?}"
            );
            for k in 0..2 {
                assert_eq!(
                    *loaded.fields.cons.mom[k].view().at(c),
                    *sim.fields.cons.mom[k].view().at(c),
                    "cons.mom_{k} transposed/garbled at {c:?}"
                );
            }
            assert_eq!(
                *lnrg.view().at(c),
                *nrg.view().at(c),
                "cons.nrg transposed/garbled at {c:?}"
            );
        }
    }

    #[test]
    fn checkpoint_saves_full_allocated_field_including_ghosts() {
        // **truncation gate**: a cell-centered dataset must carry the FULL allocated extent
        // (interior + 2*ng), NOT just the interior — otherwise a restart loses the halo the
        // next stencil reads before the first ghost-fill, and the reader's `halo_width` trim
        // would over-cut interior-only data. seed EVERY allocated cell (ghosts included) with a
        // coord-unique value; assert (a) the on-disk dataset volume is the padded volume, and
        // (b) every ghost cell survives the write -> load round-trip byte-for-byte.
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let ng = 2usize;
        let build = || {
            Sim2::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize, 0],
                [5usize, 3],
                [0.0, 0.0],
                [0.2, 1.0 / 3.0],
                ng,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };

        let sim = build();
        // a value unique per (coord) so a misplaced or dropped ghost is a loud failure.
        let seed = |c: [isize; 2]| 1000.0 + c[0] as f64 + 31.0 * c[1] as f64;
        let alloc = sim.fields.cons.den.domain().clone();
        for c in alloc.iter() {
            sim.fields.cons.den.view_mut().set(c, seed(c));
        }

        let dir = std::env::temp_dir().join("symbi_checkpoint_fullfield");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("full.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        // (a) the den dataset must hold the PADDED volume (5+2*ng)x(3+2*ng), not 5x3.
        let tree = Hdf5Backend.read(std::path::Path::new(path)).unwrap();
        let den_ds = tree
            .find_group("level_0")
            .unwrap()
            .find_group("conserved")
            .unwrap()
            .find_dataset("den")
            .unwrap();
        let on_disk = den_ds.data.as_f64().unwrap().len();
        assert_eq!(
            on_disk,
            alloc.volume(),
            "den dataset holds {on_disk} cells but the allocated field has {} (interior would be 15)",
            alloc.volume()
        );

        // (b) every allocated cell — ESPECIALLY the ghosts outside the interior — round-trips.
        let mut loaded = build();
        load_checkpoint(&mut loaded, path).unwrap();
        let interior = sim.geom.interior.clone();
        let mut ghost_checked = 0usize;
        for c in alloc.iter() {
            assert_eq!(
                *loaded.fields.cons.den.view().at(c),
                seed(c),
                "cons.den lost/garbled at {c:?} (ghost={})",
                !interior.contains(c)
            );
            if !interior.contains(c) {
                ghost_checked += 1;
            }
        }
        assert!(
            ghost_checked > 0,
            "test seeded no ghosts — allocation has no halo"
        );
    }
}
