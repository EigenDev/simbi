// =============================================================================
// checkpoint.rs
//
// HDF5 checkpoint API for SimState. a thin builder that maps SimState onto a
// `symbi_io::Tree` schema and hands off to `symbi_io::Hdf5Backend`. the I/O
// concerns (file format, field naming, error handling) live in the symbi-io
// crate; this module only describes what SimState contributes to the schema.
//
// extras ride in as a typed `symbi_io::Metadata` — callers build
// `Metadata::new().with("key", value)` with naked typed values — and
// `write_checkpoint` / `load_checkpoint` / `read_checkpoint_meta` all return
// `Result<_, symbi_io::IoError>`.
// =============================================================================

use std::path::Path;

use crate::state::*;
use crate::tracers as symbi_sim_tracers;
use symbi_geometry::Metric;
use symbi_hydro::eos::Eos;
use symbi_hydro::regime::Regime;
use symbi_xpu::{ExecutionSpace, MemorySpace};

use symbi_hydro::FieldSpec;
pub use symbi_io::{Attr, IoError, Metadata, Result};
use symbi_io::{DataRef, Dataset, Hdf5Backend, IoBackend, Tree, TreeBuf};

/// the homologous mesh-motion factor applied to axis `ax` of a `d`-dimensional
/// grid: cartesian expands every axis, spherical the radius only, cylindrical
/// the in-plane r and axial z slots. shared by the checkpoint writer (which
/// stores physical, scaled bounds) and the restart region check (which must
/// unscale them back to the comoving grid) so the two cannot disagree about
/// which axes a stored bound was scaled by.
fn motion_axis_scale(geometry: symbi_geometry::Geometry, ax: usize, d: usize, a: f64) -> f64 {
    match geometry {
        symbi_geometry::Geometry::Cartesian => a,
        symbi_geometry::Geometry::Spherical => {
            if ax == 0 {
                a
            } else {
                1.0
            }
        }
        symbi_geometry::Geometry::Cylindrical => {
            if ax == 0 || ax == d - 1 {
                a
            } else {
                1.0
            }
        }
    }
}

fn write_tree_atomic(path: &Path, tree: &Tree<'_>) -> Result<()> {
    let file_name = path.file_name().ok_or_else(|| {
        IoError::MissingPath(format!("checkpoint path has no file name: {path:?}"))
    })?;
    let temporary = path.with_file_name(format!(
        ".{}.tmp.{}",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    if temporary.exists() {
        std::fs::remove_file(&temporary)?;
    }
    if let Err(error) = Hdf5Backend.write(&temporary, tree) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&temporary, path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

/// whether `time` has reached a cadence boundary, tolerant of the roundoff an
/// accumulating clock carries: a step that lands within 32 eps of the boundary
/// counts as having reached it, so a checkpoint is never skipped by one ulp.
pub fn time_at_or_after(time: f64, boundary: f64) -> bool {
    let tolerance = 32.0 * f64::EPSILON * time.abs().max(boundary.abs());
    time >= boundary || (time - boundary).abs() <= tolerance
}

// =============================================================================
// Snapshot — the materialized Vec<f64> buffers a write needs to borrow
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
    // the allocated (padded) cell extent per axis = interior + 2*ng. cell-centered
    // field datasets are written at this full extent so a restart restores the
    // entire field — ghost zones included — not just the interior (which would
    // truncate the halo the next step's stencil reads before the first ghost-fill).
    // the reader trims `halo_radius` (= ng) back to the interior for plotting.
    data_shape: Vec<u64>,
    // interior cell counts in storage (reversed) axis order for the reader's
    // `mesh/global_cells` ([nx3,nx2,nx1]); matches the reversed field `shape` so the
    // plot axes are not transposed (a non-square grid otherwise crashes pcolormesh).
    mesh_cells: Vec<u64>,
    dx_phys: Vec<f64>,
    x_lo_phys: Vec<f64>,
    conserved: Vec<(String, Vec<f64>)>,
    primitive: Vec<(String, Vec<f64>)>,
    bface: Vec<(String, Vec<f64>)>, // canonical "B1".."bd" face-centered B, MHD only
    // per-face (start, fin) index bounds for the reader's `magnetic/Bn/domain` group.
    bface_dom: Vec<(Vec<i64>, Vec<i64>)>,
    // single-partition owned cell range for the `partition_0` group the frozen
    // v2.0 reader expects: start = [0; D], fin = interior cell counts.
    owned_start: Vec<i64>,
    owned_fin: Vec<i64>,
}

/// visit every cell of `domain` in axis-0-fastest order (x varies fastest) — the on-disk
/// checkpoint layout, so numpy `arr.reshape((Nz, Ny, Nx))` puts physical x on the horizontal
/// `imshow` axis. gather (`extract_field`) and scatter (`restore_field`) must share this one
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
/// over its own allocated domain (`field.domain()` — interior + ghosts), so the
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
    // before reading fields here, sync the device so the host sees the
    // committed state from the last RK2 stage (removing the per-launch `ctx_sync()`
    // was a production pipelining win). gate on `IS_DEVICE_ACCESSIBLE`; the cuda
    // feature alone is insufficient: a host-memory sim in a cuda-feature build has no CUDA context, so
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
    // homologous expansion is radial: a(t) scales the radial coordinate (and, in cartesian,
    // every coordinate isotropically), but never an angular one (theta/phi) — a moving spherical
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
    let mut conserved = collect_bucket(R::SPEC.fields, DOF, |fs, idx| match fs.name {
        "den" => Some(&sim.fields.cons.den),
        "mom" => Some(&sim.fields.cons.mom[idx]),
        "nrg" => sim.fields.cons.nrg_field(),
        "mag" => sim.fields.mhd.as_ref().map(|m| &m.bcell[idx]),
        other => panic!("checkpoint write: unknown conserved field '{other}'"),
    });
    // the passive scalar is a run-level opt-in, not a regime field, so it rides
    // outside the spec iteration: present iff allocated.
    if let Some(chi) = sim.fields.cons.chi_field() {
        conserved.push(("chi".to_string(), extract_field(chi, chi.domain())));
    }

    // ----- RegimeSpec-driven primitive iteration ------
    let mut primitive = collect_bucket(R::SPEC.primitive_fields, DOF, |fs, idx| match fs.name {
        "rho" => Some(&sim.fields.prim.rho),
        "vel" => Some(&sim.fields.prim.vel[idx]),
        "pre" => sim.fields.prim.pre_field(),
        "bcell" => sim.fields.mhd.as_ref().map(|m| &m.bcell[idx]),
        other => panic!("checkpoint write: unknown primitive field '{other}'"),
    });
    if let Some(chi) = sim.fields.prim.chi_field() {
        primitive.push(("chi".to_string(), extract_field(chi, chi.domain())));
    }

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

    // the owned interior index range [0, ncells) per axis, in the same reversed (storage) axis order
    // as `mesh_cells` and the `dim_*` geometry. without the reverse, owned is (x, y, ..) while
    // global_cells / dims are (.., y, x): the reader then pairs the y geometry with the x extent, and
    // a non-square AMR fine patch renders transposed / offset (a square grid hides it -- both orders
    // agree). matches the `(0..D).rev()` walk used for `mesh_cells` above.
    let owned_start: Vec<i64> = vec![0; D];
    let owned_fin: Vec<i64> = (0..D)
        .rev()
        .map(|ax| interior.spaces[ax].size() as i64)
        .collect();

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
    // the checkpoint carries the regime in the configuration vocabulary (newtonian, isothermal,
    // rhd, rmhd, nmhd, imhd), so a restart parses it back into the same enum the run was
    // configured from. `has_energy() == false` marks the isothermal (IsoModel) regimes, which
    // carry no energy equation.
    if r.is_mhd() {
        if r.is_relativistic() {
            "rmhd"
        } else if r.has_energy() {
            "nmhd"
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

fn spacetime_name(s: symbi_geometry::Spacetime) -> &'static str {
    match s {
        symbi_geometry::Spacetime::Minkowski => "minkowski",
        symbi_geometry::Spacetime::SchwarzschildKS => "schwarzschild_ks",
        symbi_geometry::Spacetime::KerrKS => "kerr_ks",
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
    let mut builtins: Vec<(&str, Attr)> = vec![
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
        // the background spacetime chart — orthogonal to coord_system. GR readers need
        // this to select the metric (lapse, shift, densitization) when reducing fluxes.
        (
            "spacetime",
            Attr::Str(spacetime_name(sim.geom.spacetime).into()),
        ),
    ];
    // the curved-spacetime scalar params (schwarzschild_mass, kerr_spin) ride as
    // named attrs so a reader can reconstruct the metric; empty on a flat background.
    for (name, value) in &sim.geom.spacetime_scalars {
        builtins.push((name.as_str(), Attr::F64(*value)));
    }
    // isothermal regimes close with p = cs^2 rho at a constant sound speed
    // and store no pressure dataset; record cs so readers can reconstruct
    // pressure-dependent fields. the isothermal eos ignores (rho, pre).
    if !sim.physics.regime.has_energy() {
        builtins.push((
            "sound_speed",
            Attr::F64(sim.physics.eos.sound_speed(
                symbi_hydro::quantity::Density(1.0),
                symbi_hydro::quantity::Pressure(1.0),
            )),
        ));
    }
    let mut meta = Tree::new("metadata");
    // explicit user extras win. start with them, then fill in any built-in
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
    // so the mesh carries the geometry description and no precomputed coordinate
    // arrays. symbi's own `load_checkpoint` reconstructs geometry from config,
    // so this layout serves external readers without breaking restart.
    let mut geometry =
        Tree::new("geometry").with_attr("metric", coord_name(sim.physics.metric.geometry()));
    // mesh metadata (global_cells + per-dim geometry) is written in storage
    // (reversed) axis order so it matches the reversed field `shape` below and the
    // reader's [nx3,nx2,nx1] expectation: dim_0 is the slowest-varying screen axis,
    // dim_{D-1} the fastest (x1). without the reverse, global_cells/dims are
    // transposed vs the data -> a square grid plots mis-oriented and a non-square
    // grid crashes pcolormesh ("C dims should be one smaller than X and Y").
    for (slot, ax) in (0..D).rev().enumerate() {
        // the interior lower edge — honors an AMR fine level whose interior
        // starts at a non-zero global index (start = global origin offset by
        // the interior origin), so the reader rebuilds cell centers correctly.
        let lo_index = sim.geom.interior.spaces[ax].lo;
        let hi_index = sim.geom.interior.spaces[ax].hi;
        let scale = motion_axis_scale(sim.physics.metric.geometry(), ax, D, sim.motion.a);
        let (start, end) = match &sim.geom.maps {
            Some(maps) => (
                maps[ax].face(lo_index) * scale,
                maps[ax].face(hi_index) * scale,
            ),
            None => {
                let start = snap.x_lo_phys[ax] + lo_index as f64 * snap.dx_phys[ax];
                (start, start + snap.dx_phys[ax] * snap.resolution[ax] as f64)
            }
        };
        // the per-axis spacing the reader reconstructs cell centers from: "linear" -> uniform faces
        // start + i*dx, "log" -> geometric faces start*10^(i*slope). taken from the grid's coordinate
        // maps (uniform when unset). start/end are the axis domain bounds [r_lo, r_hi]; logspace over
        // them recovers the geometric grid.
        let (spacing_label, spacing_ratio) = match &sim.geom.maps {
            Some(maps) => match maps[ax] {
                symbi_geometry::AxisMap::Uniform { .. } => ("linear", 1.0),
                symbi_geometry::AxisMap::Log { .. } => ("log", 1.0),
                symbi_geometry::AxisMap::Geometric { ratio, .. } => ("geometric", ratio),
            },
            None => ("linear", 1.0),
        };
        geometry.push_group(
            Tree::new(format!("dim_{slot}"))
                .with_attr("start", start)
                .with_attr("end", end)
                .with_attr("type", spacing_label)
                .with_attr("ratio", spacing_ratio),
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

    // declare shape in reversed axis order so it matches the on-disk layout
    // `extract_field` produces (axis-0-fastest in memory -> axis-0 is the last
    // dim of the numpy shape). numpy/matplotlib then put physical x on the
    // horizontal screen axis. for 2D OT: shape = [Ny, Nx]; for 3D: [Nz, Ny, Nx].
    // cell datasets carry the padded extent (interior + 2*ng); the reader trims
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

    // face-centered B (MHD) under hydro/magnetic — each B-face as the group the
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

    // the level's own clock: every level shares the root's time after a root step, and a finer
    // level's iteration counts its substeps, ratio times the root's per level.
    Tree::new(format!("level_{idx}"))
        .with_attr("scale_factor_a", sim.motion.a)
        .with_attr("scale_factor_adot", sim.motion.a_dot)
        .with_attr("time", sim.time)
        .with_attr("dt", sim.dt)
        .with_attr("iteration", Attr::U64(sim.iteration))
        .with_group(mesh)
        .with_group(partition_0)
        .with_group(cons)
}

// =============================================================================
// body state round-trip: the per-body kinematic + accretion ledger a restart
// must restore. bodies re-attach from the config on restart, so without this
// group a moving body's orbit phase and a sink's cumulative accreted mass
// silently reset — wrong physics for binaries, a broken cumulative ledger for
// accretors.
// =============================================================================

/// the materialized per-body state buffers a write borrows (Snapshot-style).
struct BodyStateSnap {
    pos: Vec<f64>,             // [nb, D]
    vel: Vec<f64>,             // [nb, D]
    mass: Vec<f64>,            // [nb]
    accreted: Vec<f64>,        // [nb] cumulative rest mass (0 for non-sinks)
    rate: Vec<f64>,            // [nb] instantaneous Mdot (0 for non-sinks)
    accreted_energy: Vec<f64>, // [nb] cumulative covariant (killing) energy (GR horizon only)
    energy_rate: Vec<f64>,     // [nb] instantaneous Edot (GR horizon only)
    slip_heat: Vec<f64>,       // [nb] cumulative magnetic-slip heat released by the body's shell
    slip_heat_rate: Vec<f64>,  // [nb] its rate over the last step
    ang_mom: Vec<f64>,         // [nb, 3] world-frame angular momentum L = I omega
    ke_trans: Vec<f64>,        // [nb] translational kinetic energy 0.5 m |v|^2
    ke_rot: Vec<f64>,          // [nb] rotational kinetic energy 0.5 omega.I.omega
    orientation: Vec<f64>,     // [nb, 3, 3] row-major rotation matrix (evolved spin state)
    omega: Vec<f64>,           // [nb, 3] angular-velocity vector
    shape_json: Vec<String>,   // per-body CSG wire (empty = the analytic sphere), for viz
    nb: usize,
}

fn body_state_snap<const D: usize>(im: &ImmersedBodies<D>) -> BodyStateSnap {
    let nb = im.bodies.len();
    let mut snap = BodyStateSnap {
        pos: Vec::with_capacity(nb * D),
        vel: Vec::with_capacity(nb * D),
        mass: Vec::with_capacity(nb),
        accreted: Vec::with_capacity(nb),
        rate: Vec::with_capacity(nb),
        accreted_energy: Vec::with_capacity(nb),
        energy_rate: Vec::with_capacity(nb),
        slip_heat: Vec::with_capacity(nb),
        slip_heat_rate: Vec::with_capacity(nb),
        ang_mom: Vec::with_capacity(nb * 3),
        ke_trans: Vec::with_capacity(nb),
        ke_rot: Vec::with_capacity(nb),
        orientation: Vec::with_capacity(nb * 9),
        omega: Vec::with_capacity(nb * 3),
        shape_json: Vec::with_capacity(nb),
        nb,
    };
    for b in 0..nb {
        // the CSG shape wire (empty for the analytic sphere) — a self-describing silhouette for viz.
        snap.shape_json.push(
            im.shapes
                .get(b)
                .and_then(|s| s.as_ref())
                .map(|s| s.to_json())
                .unwrap_or_default(),
        );
        let body = im.bodies.get(b);
        for a in 0..D {
            snap.pos.push(body.position[a]);
            snap.vel.push(body.velocity[a]);
        }
        // the evolved rigid-body rotation: the orientation matrix (row-major) + angular velocity.
        for i in 0..3 {
            for j in 0..3 {
                snap.orientation.push(body.orientation[i][j]);
            }
        }
        for k in 0..3 {
            snap.omega.push(body.omega[k]);
        }
        // the rigid-body ledgers: world-frame angular momentum + the split kinetic energy, so viz can
        // draw an L glyph and plot the translational/rotational energy budget.
        let l = body.angular_momentum();
        for k in 0..3 {
            snap.ang_mom.push(l[k]);
        }
        snap.ke_trans.push(body.translational_ke());
        snap.ke_rot.push(body.rotational_ke());
        snap.mass.push(body.mass);
        let (acc, rate) = match body.kind {
            symbi_ib::BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } => (total_accreted_mass, accretion_rate),
            // the GR horizon books the shell-flux rest-mass ledger into the same datasets (Mdot).
            symbi_ib::BodyKind::Horizon {
                total_accreted_mass,
                mdot,
                ..
            } => (total_accreted_mass, mdot),
            _ => (0.0, 0.0),
        };
        snap.accreted.push(acc);
        snap.rate.push(rate);
        // the GR horizon also books the covariant (killing) energy ledger (Edot, cumulative E).
        let (acc_e, rate_e) = match body.kind {
            symbi_ib::BodyKind::Horizon {
                total_accreted_energy,
                edot,
                ..
            } => (total_accreted_energy, edot),
            _ => (0.0, 0.0),
        };
        snap.accreted_energy.push(acc_e);
        snap.energy_rate.push(rate_e);
        snap.slip_heat.push(body.slip_heat_total);
        snap.slip_heat_rate.push(body.slip_heat_rate);
    }
    snap
}

/// flattened tracer state for the checkpoint: positions row-major [n, D],
/// ids and flags as f64 (ids stay exact below 2^53; flags are 0/1).
struct TracerSnap {
    n: usize,
    x: Vec<f64>,
    id: Vec<u64>,
    cohort: Vec<u64>,
    owner: Vec<u64>,
    run_seed: u64,
    next_id: u64,
    injection_remainder: f64,
    escaped: Vec<f64>,
    crossed: Vec<f64>,
    crossing_time: Vec<f64>,
    weight: Vec<f64>,
}

fn tracer_snap<const D: usize>(tr: &symbi_sim_tracers::TracerSet<D>) -> TracerSnap {
    let n = tr.len();
    let mut x = Vec::with_capacity(n * D);
    for p in &tr.x {
        x.extend_from_slice(&p[..]);
    }
    TracerSnap {
        n,
        x,
        id: tr.id.clone(),
        cohort: tr.cohort.iter().map(|&cohort| cohort as u64).collect(),
        owner: tr.owner.iter().map(|owner| owner.0).collect(),
        run_seed: tr.run_seed,
        next_id: tr.next_id,
        injection_remainder: tr.injection_remainder,
        escaped: tr.flags.iter().map(|f| f.escaped as u8 as f64).collect(),
        crossed: tr
            .flags
            .iter()
            .map(|f| f.crossed_sink as u8 as f64)
            .collect(),
        crossing_time: tr.flags.iter().map(|f| f.crossing_time).collect(),
        weight: vec![tr.weight],
    }
}

fn tracer_group<const D: usize>(snap: &TracerSnap) -> Tree<'_> {
    Tree::new("tracers")
        .with_attr("n_tracers", snap.n as u64)
        .with_attr("run_seed", snap.run_seed)
        .with_attr("next_id", snap.next_id)
        .with_attr("injection_remainder", snap.injection_remainder)
        .with_dataset(Dataset::new(
            "position",
            vec![snap.n, D],
            DataRef::F64(&snap.x),
        ))
        .with_dataset(Dataset::new("id", vec![snap.n], DataRef::U64(&snap.id)))
        .with_dataset(Dataset::new(
            "cohort",
            vec![snap.n],
            DataRef::U64(&snap.cohort),
        ))
        .with_dataset(Dataset::new(
            "owner",
            vec![snap.n],
            DataRef::U64(&snap.owner),
        ))
        .with_dataset(Dataset::new(
            "escaped",
            vec![snap.n],
            DataRef::F64(&snap.escaped),
        ))
        .with_dataset(Dataset::new(
            "crossed_sink",
            vec![snap.n],
            DataRef::F64(&snap.crossed),
        ))
        .with_dataset(Dataset::new(
            "crossing_time",
            vec![snap.n],
            DataRef::F64(&snap.crossing_time),
        ))
        .with_dataset(Dataset::new("weight", vec![1], DataRef::F64(&snap.weight)))
}

struct ContinuousTracerSnap {
    n: usize,
    order: u64,
    x: Vec<f64>,
    step_x: Vec<f64>,
    id: Vec<u64>,
    cohort: Vec<u64>,
    owner: Vec<u64>,
    escaped: Vec<u64>,
    crossed_sink: Vec<u64>,
    crossing_time: Vec<f64>,
    random_counter: Vec<u64>,
    weight: Vec<f64>,
    run_seed: u64,
    next_id: u64,
    injection_remainder: f64,
}

fn continuous_tracer_snap<const D: usize, Mem: MemorySpace>(
    tracers: &symbi_sim_tracers::ContinuousTracerSet<D, Mem>,
) -> ContinuousTracerSnap {
    assert!(
        Mem::IS_HOST_ACCESSIBLE,
        "continuous tracer checkpointing requires host-accessible storage"
    );
    let n = tracers.len;
    unsafe {
        ContinuousTracerSnap {
            n,
            order: tracers.order as u64,
            x: (0..n)
                .flat_map(|ii| (0..D).map(move |dd| *tracers.x[dd].as_ptr::<f64>().add(ii)))
                .collect(),
            step_x: (0..n)
                .flat_map(|ii| (0..D).map(move |dd| *tracers.step_x[dd].as_ptr::<f64>().add(ii)))
                .collect(),
            id: std::slice::from_raw_parts(tracers.id.as_ptr::<u64>(), n).to_vec(),
            cohort: std::slice::from_raw_parts(tracers.cohort.as_ptr::<u16>(), n)
                .iter()
                .map(|value| *value as u64)
                .collect(),
            owner: std::slice::from_raw_parts(
                tracers.owner.as_ptr::<crate::mass_transport::ContainerId>(),
                n,
            )
            .iter()
            .map(|owner| owner.0)
            .collect(),
            escaped: std::slice::from_raw_parts(tracers.escaped.as_ptr::<u8>(), n)
                .iter()
                .map(|value| *value as u64)
                .collect(),
            crossed_sink: std::slice::from_raw_parts(tracers.crossed_sink.as_ptr::<u8>(), n)
                .iter()
                .map(|value| *value as u64)
                .collect(),
            crossing_time: std::slice::from_raw_parts(tracers.crossing_time.as_ptr::<f64>(), n)
                .to_vec(),
            random_counter: std::slice::from_raw_parts(tracers.random_counter.as_ptr::<u64>(), n)
                .to_vec(),
            weight: vec![tracers.weight],
            run_seed: tracers.run_seed,
            next_id: tracers.next_id,
            injection_remainder: tracers.injection_remainder,
        }
    }
}

fn continuous_tracer_group<const D: usize>(snap: &ContinuousTracerSnap) -> Tree<'_> {
    Tree::new("continuous_tracers")
        .with_attr("n_tracers", snap.n as u64)
        .with_attr("order", snap.order)
        .with_attr("run_seed", snap.run_seed)
        .with_attr("next_id", snap.next_id)
        .with_attr("injection_remainder", snap.injection_remainder)
        .with_dataset(Dataset::new(
            "position",
            vec![snap.n, D],
            DataRef::F64(&snap.x),
        ))
        .with_dataset(Dataset::new(
            "step_position",
            vec![snap.n, D],
            DataRef::F64(&snap.step_x),
        ))
        .with_dataset(Dataset::new("id", vec![snap.n], DataRef::U64(&snap.id)))
        .with_dataset(Dataset::new(
            "cohort",
            vec![snap.n],
            DataRef::U64(&snap.cohort),
        ))
        .with_dataset(Dataset::new(
            "owner",
            vec![snap.n],
            DataRef::U64(&snap.owner),
        ))
        .with_dataset(Dataset::new(
            "escaped",
            vec![snap.n],
            DataRef::U64(&snap.escaped),
        ))
        .with_dataset(Dataset::new(
            "crossed_sink",
            vec![snap.n],
            DataRef::U64(&snap.crossed_sink),
        ))
        .with_dataset(Dataset::new(
            "crossing_time",
            vec![snap.n],
            DataRef::F64(&snap.crossing_time),
        ))
        .with_dataset(Dataset::new(
            "random_counter",
            vec![snap.n],
            DataRef::U64(&snap.random_counter),
        ))
        .with_dataset(Dataset::new("weight", vec![1], DataRef::F64(&snap.weight)))
}

fn combine_continuous_tracer_snaps(
    mut snaps: impl Iterator<Item = ContinuousTracerSnap>,
) -> Option<ContinuousTracerSnap> {
    let mut combined = snaps.next()?;
    for snap in snaps {
        assert_eq!(snap.order, combined.order);
        assert_eq!(snap.run_seed, combined.run_seed);
        assert_eq!(snap.next_id, combined.next_id);
        assert_eq!(
            snap.injection_remainder.to_bits(),
            combined.injection_remainder.to_bits()
        );
        assert_eq!(snap.weight, combined.weight);
        combined.n += snap.n;
        combined.x.extend(snap.x);
        combined.step_x.extend(snap.step_x);
        combined.id.extend(snap.id);
        combined.cohort.extend(snap.cohort);
        combined.owner.extend(snap.owner);
        combined.escaped.extend(snap.escaped);
        combined.crossed_sink.extend(snap.crossed_sink);
        combined.crossing_time.extend(snap.crossing_time);
        combined.random_counter.extend(snap.random_counter);
    }
    Some(combined)
}

/// the dataset naming the magnetic-slip heat by its fate under the run's closure: deposited in
/// the gas (adiabatic) or exported to the cooling bath (isothermal).
fn slip_heat_dataset_name(has_energy: bool) -> &'static str {
    if has_energy {
        "magnetic_slip_heating"
    } else {
        "exported_slip_heat"
    }
}

fn body_state_group<'a, const D: usize>(snap: &'a BodyStateSnap, heat_name: &str) -> Tree<'a> {
    let nb = snap.nb;
    let mut t = Tree::new("bodies")
        .with_attr("n_bodies", nb as u64)
        .with_dataset(Dataset::new(
            "position",
            vec![nb, D],
            DataRef::F64(&snap.pos),
        ))
        .with_dataset(Dataset::new(
            "velocity",
            vec![nb, D],
            DataRef::F64(&snap.vel),
        ))
        .with_dataset(Dataset::new("mass", vec![nb], DataRef::F64(&snap.mass)))
        .with_dataset(Dataset::new(
            "total_accreted_mass",
            vec![nb],
            DataRef::F64(&snap.accreted),
        ))
        .with_dataset(Dataset::new(
            "accretion_rate",
            vec![nb],
            DataRef::F64(&snap.rate),
        ))
        .with_dataset(Dataset::new(
            "total_accreted_energy",
            vec![nb],
            DataRef::F64(&snap.accreted_energy),
        ))
        .with_dataset(Dataset::new(
            "accretion_energy_rate",
            vec![nb],
            DataRef::F64(&snap.energy_rate),
        ))
        .with_dataset(Dataset::new(
            heat_name.to_string(),
            vec![nb],
            DataRef::F64(&snap.slip_heat),
        ))
        .with_dataset(Dataset::new(
            format!("{heat_name}_rate"),
            vec![nb],
            DataRef::F64(&snap.slip_heat_rate),
        ))
        .with_dataset(Dataset::new(
            "orientation",
            vec![nb, 3, 3],
            DataRef::F64(&snap.orientation),
        ))
        .with_dataset(Dataset::new(
            "omega",
            vec![nb, 3],
            DataRef::F64(&snap.omega),
        ))
        .with_dataset(Dataset::new(
            "angular_momentum",
            vec![nb, 3],
            DataRef::F64(&snap.ang_mom),
        ))
        .with_dataset(Dataset::new(
            "ke_translational",
            vec![nb],
            DataRef::F64(&snap.ke_trans),
        ))
        .with_dataset(Dataset::new(
            "ke_rotational",
            vec![nb],
            DataRef::F64(&snap.ke_rot),
        ));
    // the per-body CSG shape wire as a string attr (empty = analytic sphere); viz reconstructs the
    // body silhouette from it + the position/orientation.
    for (b, wire) in snap.shape_json.iter().enumerate() {
        t.push_attr(format!("shape_{b}"), wire.clone());
    }
    t
}

/// root attribute naming the measure the stored conserved state carries. absent means the
/// undensitized Valencia state (`D`, `S_i`, `ehat` per unit coordinate volume).
const CONSERVED_DENSITIZATION_ATTR: &str = "conserved_densitization";
/// the value written for the free-index-down GR-hydro state
/// `sqrt(-g)[rho u^t, T^t_i, -(T^t_t + rho u^t)]`.
const SQRT_MINUS_G: &str = "sqrt_minus_g";

/// the measure the run's conserved state carries, or `None` when it is undensitized. relativistic
/// hydro on a curved spacetime stores the fully densitized state; every other configuration —
/// flat spacetime, and GR MHD, whose induction and CT seam are still Valencia — does not. the two
/// states differ by a per-cell factor `sqrt(-g)(x)`, so reloading one as the other is silently
/// wrong rather than loud, which is why the file records which it holds.
fn conserved_densitization<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
) -> Option<&'static str>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let curved = sim.geom.spacetime != symbi_geometry::Spacetime::Minkowski;
    (curved && sim.fields.mhd.is_none()).then_some(SQRT_MINUS_G)
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
    if let Some(tag) = conserved_densitization(sim) {
        root = root.with_attr(CONSERVED_DENSITIZATION_ATTR, tag);
    }
    root.push_group(build_metadata_group(sim, snap, extras));
    root.push_group(build_level_group(sim, snap, 0));
    // the per-step body-gas exchange series (immersed runs): Mdot(t) is
    // mass_delta/dt, the accretion drag is force. shapes: time/dt [len],
    // mass_delta/energy_delta [len, nb], force [len, nb, D]. the series covers
    // this run segment only (it restarts empty on checkpoint load).
    if let Some(im) = sim.immersed.as_ref()
        && !im.history.is_empty()
    {
        let (n, nb) = (im.history.len(), im.history.n_bodies());
        root.push_group(
            Tree::new("body_diagnostics")
                .with_attr("n_bodies", nb as u64)
                .with_dataset(Dataset::new(
                    "time",
                    vec![n],
                    DataRef::F64(im.history.time()),
                ))
                .with_dataset(Dataset::new("dt", vec![n], DataRef::F64(im.history.dt())))
                .with_dataset(Dataset::new(
                    "mass_delta",
                    vec![n, nb],
                    DataRef::F64(im.history.mass_delta()),
                ))
                .with_dataset(Dataset::new(
                    "energy_delta",
                    vec![n, nb],
                    DataRef::F64(im.history.energy_delta()),
                ))
                .with_dataset(Dataset::new(
                    "force",
                    vec![n, nb, D],
                    DataRef::F64(im.history.force()),
                ))
                .with_dataset(Dataset::new(
                    "force_normal",
                    vec![n, nb, D],
                    DataRef::F64(im.history.force_normal()),
                ))
                .with_dataset(Dataset::new(
                    "torque",
                    vec![n, nb, 3],
                    DataRef::F64(im.history.torque()),
                )),
        );
    }
    for group in census_groups(sim) {
        root.push_group(group);
    }
    root
}

/// the checkpoint tag for a reduce op — what a reader needs to know to interpret the
/// accumulators (a sum combines across restart segments; an extremum does not).
fn reduction_op_tag(op: symbi_ir::emit::ReductionOp) -> String {
    match op {
        symbi_ir::emit::ReductionOp::Add => "add",
        symbi_ir::emit::ReductionOp::Min => "min",
        symbi_ir::emit::ReductionOp::Max => "max",
        symbi_ir::emit::ReductionOp::Mul => {
            unreachable!("a product is refused at census registration")
        }
    }
    .to_string()
}

// =============================================================================
// public API: write_checkpoint / load_checkpoint / read_checkpoint_meta
// =============================================================================

/// write a checkpoint. typed `Metadata` carries naked typed values,
/// so there is no `to_string()` boilerplate at call sites:
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
    let mut tree = build_tree(sim, &snap, extras);
    // per-body kinematic + accretion state (restart round-trip): derived
    // buffers, so they live here and the tree borrows them.
    let body_snap = sim.immersed.as_ref().map(body_state_snap);
    if let Some(bs) = body_snap.as_ref() {
        tree.push_group(body_state_group::<D>(bs, slip_heat_dataset_name(sim.fields.cons.nrg_field().is_some())));
    }
    let tr_snap = sim.tracers.as_ref().map(tracer_snap);
    if let Some(ts) = tr_snap.as_ref() {
        tree.push_group(tracer_group::<D>(ts));
    }
    let continuous_snap = sim.continuous_tracers.as_ref().map(continuous_tracer_snap);
    if let Some(snap) = continuous_snap.as_ref() {
        tree.push_group(continuous_tracer_group::<D>(snap));
    }
    write_tree_atomic(Path::new(path), &tree)
}

/// **AMR checkpoint** — write an entire refinement hierarchy into one file as
/// `/level_0`, `/level_1`, ... sibling groups (the frozen v2.0 reader walks
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
    if let Some(tag) = conserved_densitization(levels[0]) {
        root = root.with_attr(CONSERVED_DENSITIZATION_ATTR, tag);
    }
    // global metadata authored from the coarse level.
    root.push_group(build_metadata_group(levels[0], &snaps[0], extras));
    for (idx, snap) in snaps.iter().enumerate() {
        root.push_group(build_level_group(levels[idx], snap, idx));
    }
    // per-body kinematic + accretion state (restart round-trip) from the finest level carrying
    // the immersed sidecar: the finest level owns every sink's drain and books its accreted mass,
    // while the coarser levels hold gravity-only proxies whose kinematics are copies.
    let body_snap = levels
        .iter()
        .filter_map(|l| l.immersed.as_ref())
        .last()
        .map(body_state_snap);
    if let Some(bs) = body_snap.as_ref() {
        root.push_group(body_state_group::<D>(bs, slip_heat_dataset_name(levels[0].fields.cons.nrg_field().is_some())));
    }
    // the tracer population lives on whichever level carries it (uni-grid:
    // level 0) — same group layout as the single-grid writer.
    let tr_snap = levels
        .iter()
        .filter_map(|l| l.tracers.as_ref())
        .next()
        .map(tracer_snap);
    if let Some(ts) = tr_snap.as_ref() {
        root.push_group(tracer_group::<D>(ts));
    }
    let continuous_snap = combine_continuous_tracer_snaps(levels.iter().filter_map(|level| {
        level
            .continuous_tracers
            .as_ref()
            .map(continuous_tracer_snap)
    }));
    if let Some(snap) = continuous_snap.as_ref() {
        root.push_group(continuous_tracer_group::<D>(snap));
    }
    // the per-step body-gas exchange series: whichever level carries the
    // immersed sidecar (the driver consolidates on one) supplies it — the
    // same group layout the single-grid writer emits.
    if let Some(im) = levels
        .iter()
        .filter_map(|l| l.immersed.as_ref())
        .find(|im| !im.history.is_empty())
    {
        let (n, nb) = (im.history.len(), im.history.n_bodies());
        root.push_group(
            Tree::new("body_diagnostics")
                .with_attr("n_bodies", nb as u64)
                .with_dataset(Dataset::new(
                    "time",
                    vec![n],
                    DataRef::F64(im.history.time()),
                ))
                .with_dataset(Dataset::new("dt", vec![n], DataRef::F64(im.history.dt())))
                .with_dataset(Dataset::new(
                    "mass_delta",
                    vec![n, nb],
                    DataRef::F64(im.history.mass_delta()),
                ))
                .with_dataset(Dataset::new(
                    "energy_delta",
                    vec![n, nb],
                    DataRef::F64(im.history.energy_delta()),
                ))
                .with_dataset(Dataset::new(
                    "force",
                    vec![n, nb, D],
                    DataRef::F64(im.history.force()),
                ))
                .with_dataset(Dataset::new(
                    "force_normal",
                    vec![n, nb, D],
                    DataRef::F64(im.history.force_normal()),
                ))
                .with_dataset(Dataset::new(
                    "torque",
                    vec![n, nb, 3],
                    DataRef::F64(im.history.torque()),
                )),
        );
    }
    // the censuses live on the root level's store, which is where both drivers register and
    // sample them; a refined hierarchy is refused at the sampling site rather than reduced
    // across levels, so there is no second level's history to merge here.
    for group in census_groups(levels[0]) {
        root.push_group(group);
    }
    write_tree_atomic(Path::new(path), &root)
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
        // strings live as byte-array datasets (on-disk convention)
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
    load_checkpoint_level(sim, path, 0)
}

/// how many refinement levels a checkpoint carries.
///
/// a restart may run deeper than the file it resumes from — that is the whole point of a
/// bootstrap ladder, where each rung converges at its own resolution and the next adds one. the
/// levels the file has are loaded; the rest are initialized from their parents. counting them is
/// what tells the two apart, and the count comes from the file rather than from the config so a
/// hand-edited or truncated checkpoint cannot make a level silently start from zeros.
pub fn checkpoint_level_count(path: &str) -> Result<usize> {
    let tree = Hdf5Backend.read(Path::new(path))?;
    let mut n = 0usize;
    while tree.find_group(&format!("level_{n}")).is_some() {
        n += 1;
    }
    if n == 0 {
        return Err(IoError::MissingPath("level_0".into()));
    }
    Ok(n)
}

/// check that a checkpoint's level `level_index` describes the same grid this run built for it.
///
/// a deeper restart only works because level `i` occupies the same region at every depth — true
/// when a config's refinement regions are fixed geometry, false for any schedule that derives them
/// from the level count. in the second case the loaded data would be laid over a different region
/// and produce a field that is smooth, finite and wrong everywhere, with the error appearing as an
/// unexplained profile rather than as a failure.
///
/// so the property is verified rather than assumed: cell counts and physical bounds per axis, from
/// the file's own mesh description.
pub fn verify_checkpoint_level_geometry<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
    level_index: usize,
) -> Result<()>
where
    R: Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: symbi_xpu::MemorySpace,
{
    let tree = Hdf5Backend.read(Path::new(path))?;
    let name = format!("level_{level_index}");
    let mesh = tree
        .find_group(&name)
        .and_then(|level| level.find_group("mesh"))
        .ok_or_else(|| IoError::MissingPath(format!("{name}/mesh")))?;

    let cells = match mesh.find_dataset("global_cells").map(|d| &d.data) {
        Some(symbi_io::DataBuf::U64(v)) => v.clone(),
        _ => return Err(IoError::MissingPath(format!("{name}/mesh/global_cells"))),
    };
    for ax in 0..D {
        let want = sim.geom.interior.spaces[ax].size();
        // `mesh/global_cells` is written in reversed (storage) axis order, matching the reversed
        // dataset shapes so a reader's plot axes are not transposed — see `mesh_cells`. reading it
        // forward compares axis 0 against the last axis's count, which agrees only on a cubic grid
        // and rejects every anisotropic one.
        let got = *cells.get(D - 1 - ax).unwrap_or(&0) as usize;
        if got != want {
            return Err(IoError::Backend(format!(
                "{path}: level {level_index} was written with {got} cell(s) on axis {ax} but this \
                 run builds {want}. a deeper restart requires level {level_index} to occupy the \
                 same grid it did in the checkpoint; a refinement schedule whose regions depend on \
                 the level count does not satisfy that, and loading across the mismatch would \
                 place the data on the wrong region."
            )));
        }
    }

    let geometry = mesh
        .find_group("geometry")
        .ok_or_else(|| IoError::MissingPath(format!("{name}/mesh/geometry")))?;
    // the writer stores physical bounds — the comoving faces scaled by the mesh-motion
    // factor a(t) at write time on the expanding axes — while this sim is freshly built
    // on the comoving grid (a = 1; the motion re-derives a(t) from the resume time, it
    // is never integrated state). unscale the stored bounds by the checkpoint's own
    // scale factor so the comparison is comoving against comoving; a checkpoint from a
    // static-mesh run carries a = 1 and is unchanged.
    let a_checkpoint = tree
        .find_group("metadata")
        .and_then(|meta| meta.find_attr("scale_factor"))
        .and_then(|attr| match attr {
            Attr::F64(v) => Some(*v),
            _ => None,
        })
        .unwrap_or(1.0);
    for ax in 0..D {
        // the geometry groups are named by storage slot, `(0..D).rev().enumerate()` in the writer,
        // so slot `D - 1 - ax` holds axis `ax`. the same reversal as `global_cells`, and equally
        // invisible on a cubic grid or in one dimension.
        let slot = D - 1 - ax;
        let dim = geometry
            .find_group(&format!("dim_{slot}"))
            .ok_or_else(|| IoError::MissingPath(format!("{name}/mesh/geometry/dim_{slot}")))?;
        let bound = |key: &str| match dim.find_attr(key) {
            Some(Attr::F64(v)) => Some(*v),
            _ => None,
        };
        let (Some(start), Some(end)) = (bound("start"), bound("end")) else {
            return Err(IoError::MissingPath(format!(
                "{name}/mesh/geometry/dim_{slot}/start|end"
            )));
        };
        let unscale = motion_axis_scale(sim.physics.metric.geometry(), ax, D, a_checkpoint);
        let start = start / unscale;
        let end = end / unscale;
        // the run's comoving bounds, by the writer's own face arithmetic (the coordinate
        // maps when present, the uniform formula otherwise) so the two sides can only
        // differ by a genuine region mismatch, never by formula drift.
        let lo_index = sim.geom.interior.spaces[ax].lo;
        let hi_index = sim.geom.interior.spaces[ax].hi;
        let (lo, hi) = match &sim.geom.maps {
            Some(maps) => (maps[ax].face(lo_index), maps[ax].face(hi_index)),
            None => {
                let lo = sim.geom.x_lo[ax] + lo_index as f64 * sim.geom.dx[ax];
                (
                    lo,
                    lo + sim.geom.dx[ax] * sim.geom.interior.spaces[ax].size() as f64,
                )
            }
        };
        // relative to the level's own extent: an absolute tolerance would be meaningless across a
        // ladder whose finest level is orders of magnitude smaller than its root.
        let scale = (hi - lo).abs().max(1.0e-300);
        if (start - lo).abs() / scale > 1.0e-9 || (end - hi).abs() / scale > 1.0e-9 {
            return Err(IoError::Backend(format!(
                "{path}: level {level_index} axis {ax} spans [{start:e}, {end:e}] in the checkpoint \
                 (comoving, unscaled by its a = {a_checkpoint}) but [{lo:e}, {hi:e}] in this run. \
                 level {level_index} must occupy the same comoving region at every depth for a \
                 deeper restart to be meaningful."
            )));
        }
    }
    Ok(())
}

pub fn load_checkpoint_level<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    path: &str,
    level_index: usize,
) -> Result<CheckpointMeta>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy,
    E: Eos<f64>,
    S: ExecutionSpace,
    Mem: MemorySpace,
{
    let tree = Hdf5Backend.read(Path::new(path))?;
    // the conserved group is the restart primary, so its measure must match what this run
    // evolves. a densitized GR-hydro run reading an undensitized file (or the reverse) differs by
    // a per-cell sqrt(-g) and would restart onto a physically different state without any symptom
    // the first step could not explain away.
    let stored = match tree.find_attr(CONSERVED_DENSITIZATION_ATTR) {
        Some(Attr::Str(s)) => Some(s.as_str()),
        _ => None,
    };
    let expected = conserved_densitization(sim);
    if stored != expected {
        return Err(IoError::Backend(format!(
            "{path}: the checkpoint stores a {} conserved state but this run evolves a {} one",
            stored.unwrap_or("undensitized"),
            expected.unwrap_or("undensitized"),
        )));
    }
    let meta = read_meta_from(&tree)?;
    sim.time = meta.time;
    sim.dt = meta.dt;
    sim.iteration = meta.iteration;

    let level_name = format!("level_{level_index}");
    let level_0 = tree
        .find_group(&level_name)
        .ok_or_else(|| IoError::MissingPath(level_name.clone()))?;
    // the level's own clock, when the file carries one; older files hold the root's clock alone.
    if let Some(a) = level_0.find_attr("time") {
        sim.time = a.as_f64(&format!("{level_name}/time"))?;
    }
    if let Some(a) = level_0.find_attr("dt") {
        sim.dt = a.as_f64(&format!("{level_name}/dt"))?;
    }
    if let Some(a) = level_0.find_attr("iteration") {
        sim.iteration = a.as_u64(&format!("{level_name}/iteration"))?;
    }
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
    // the passive scalar rides outside the spec iteration (run-level opt-in):
    // restored iff this run allocated it and the file carries it — a dyed
    // restart of an undyed file starts from chi = 0 rather than failing.
    if let Some(chi) = sim.fields.cons.chi_field() {
        if cons.find_dataset("chi").is_some() {
            restore_field(cons, "chi", chi, chi.domain())?;
        }
    }

    // primitives (optional). the canonical v2 tree nests visualization fields
    // under partition_0/hydro; accept the former flat location for old files.
    let hydro = level_0
        .find_group("partition_0")
        .and_then(|partition| partition.find_group("hydro"));
    let primitives = hydro
        .and_then(|group| group.find_group("primitives"))
        .or_else(|| level_0.find_group("primitives"));
    if let Some(prim) = primitives {
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
        // the passive-scalar concentration, outside the spec iteration like its
        // conserved counterpart.
        if let Some(chi) = sim.fields.prim.chi_field() {
            if prim.find_dataset("chi").is_some() {
                restore_field(prim, "chi", chi, chi.domain())?;
            }
        }
    }

    // face-centered B (CT truth — restores div(B)=0 exactly)
    if let Some(ref mhd) = sim.fields.mhd {
        let magnetic = hydro
            .and_then(|group| group.find_group("magnetic"))
            .or_else(|| level_0.find_group("magnetic"));
        if let Some(mag) = magnetic {
            let mut all_ok = true;
            for d in 0..D {
                let face_dom = interior.extend(d, 0, 1);
                let name = format!("B{}", d + 1);
                let restored = mag.find_group(&name).map_or_else(
                    || restore_field(mag, &name, &mhd.bface[d], &face_dom),
                    |face| restore_field(face, "data", &mhd.bface[d], &face_dom),
                );
                if restored.is_err() {
                    all_ok = false;
                }
            }
            if all_ok {
                mhd.bface_initialized
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    // restore per-body kinematic + accretion state over the config-attached
    // collection: without this, a restart resets a moving body's orbit phase
    // and a sink's cumulative accreted mass (the diagnostics.dat ledger would
    // step down at the seam). checkpoints written before the group exists
    // restore fields only — bodies keep their config values.
    // the tracer population: restored whenever the file carries the group —
    // the run continues the previous population (flags, ids, weight intact)
    // regardless of what the fresh-run config would have seeded.
    if let Some(tg) = tree.find_group("tracers") {
        let getf = |name: &str| -> Result<Vec<f64>> {
            Ok(tg
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}")))?
                .data
                .as_f64()
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}: not f64")))?
                .to_vec())
        };
        let getu = |name: &str| -> Result<Vec<u64>> {
            Ok(tg
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}")))?
                .data
                .as_u64()
                .ok_or_else(|| IoError::MissingPath(format!("tracers/{name}: not u64")))?
                .to_vec())
        };
        let xs = getf("position")?;
        let ids = getu("id")?;
        let cohorts = getu("cohort")?;
        let owners = getu("owner")?;
        let (esc, crx, ct) = (
            getf("escaped")?,
            getf("crossed_sink")?,
            getf("crossing_time")?,
        );
        let weight = getf("weight")?.first().copied().unwrap_or(0.0);
        let n = ids.len();
        let run_seed = tg
            .find_attr("run_seed")
            .ok_or_else(|| IoError::MissingPath("tracers/run_seed".to_string()))?
            .as_u64("tracers/run_seed")?;
        let next_id = tg
            .find_attr("next_id")
            .ok_or_else(|| IoError::MissingPath("tracers/next_id".to_string()))?
            .as_u64("tracers/next_id")?;
        let injection_remainder = tg
            .find_attr("injection_remainder")
            .ok_or_else(|| IoError::MissingPath("tracers/injection_remainder".to_string()))?
            .as_f64("tracers/injection_remainder")?;
        let mut tr = symbi_sim_tracers::TracerSet::<D> {
            weight,
            run_seed,
            next_id,
            injection_remainder,
            ..Default::default()
        };
        for i in 0..n {
            let mut p = [0.0; D];
            for a in 0..D {
                p[a] = xs[i * D + a];
            }
            tr.x.push(p);
            tr.id.push(ids[i]);
            tr.cohort.push(
                u16::try_from(cohorts[i])
                    .map_err(|_| IoError::Backend("tracer cohort exceeds u16".to_string()))?,
            );
            tr.owner.push(crate::mass_transport::ContainerId(owners[i]));
            tr.flags.push(symbi_sim_tracers::TracerFlags {
                escaped: esc[i] != 0.0,
                crossed_sink: crx[i] != 0.0,
                crossing_time: ct[i],
            });
        }
        tr.step_owner = tr.owner.clone();
        tr.step_flags = tr.flags.clone();
        sim.tracers = Some(tr);
        let geometry = sim.geom.block_geometry(sim.physics.metric);
        let layout = symbi_sim_tracers::TransportLayout::single(&sim.geom.interior);
        symbi_sim_tracers::refresh_derived_positions_store(&mut sim.store, &geometry, layout);
    }
    if let Some(group) = tree.find_group("continuous_tracers") {
        let getf = |name: &str| -> Result<Vec<f64>> {
            Ok(group
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}")))?
                .data
                .as_f64()
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}: not f64")))?
                .to_vec())
        };
        let getu = |name: &str| -> Result<Vec<u64>> {
            Ok(group
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}")))?
                .data
                .as_u64()
                .ok_or_else(|| IoError::MissingPath(format!("continuous_tracers/{name}: not u64")))?
                .to_vec())
        };
        let x = getf("position")?;
        let step_x = getf("step_position")?;
        let id = getu("id")?;
        let cohort = getu("cohort")?;
        let owner = getu("owner")?;
        let escaped = getu("escaped")?;
        let crossed_sink = getu("crossed_sink")?;
        let crossing_time = getf("crossing_time")?;
        let random_counter = getu("random_counter")?;
        let weight = getf("weight")?.first().copied().unwrap_or(0.0);
        let n = id.len();
        if [
            x.len() / D,
            step_x.len() / D,
            cohort.len(),
            owner.len(),
            escaped.len(),
            crossed_sink.len(),
            crossing_time.len(),
            random_counter.len(),
        ]
        .into_iter()
        .any(|length| length != n)
        {
            return Err(IoError::Backend(
                "continuous tracer checkpoint arrays have inconsistent lengths".to_string(),
            ));
        }
        let order = match group
            .find_attr("order")
            .ok_or_else(|| IoError::MissingPath("continuous_tracers/order".to_string()))?
            .as_u64("continuous_tracers/order")?
        {
            2 => crate::mass_transport::ItoOrder::Two,
            3 => crate::mass_transport::ItoOrder::Three,
            value => {
                return Err(IoError::Backend(format!(
                    "unsupported continuous tracer order {value}"
                )));
            }
        };
        let mut tracers = symbi_sim_tracers::ContinuousTracerSet::<D, Mem>::allocate(n, order)
            .map_err(IoError::Backend)?;
        tracers.weight = weight;
        tracers.run_seed = group
            .find_attr("run_seed")
            .ok_or_else(|| IoError::MissingPath("continuous_tracers/run_seed".to_string()))?
            .as_u64("continuous_tracers/run_seed")?;
        tracers.next_id = group
            .find_attr("next_id")
            .ok_or_else(|| IoError::MissingPath("continuous_tracers/next_id".to_string()))?
            .as_u64("continuous_tracers/next_id")?;
        tracers.injection_remainder = group
            .find_attr("injection_remainder")
            .ok_or_else(|| {
                IoError::MissingPath("continuous_tracers/injection_remainder".to_string())
            })?
            .as_f64("continuous_tracers/injection_remainder")?;
        for ii in 0..n {
            tracers
                .push_host(symbi_sim_tracers::ContinuousTracerRecord {
                    x: std::array::from_fn(|dd| x[ii * D + dd]),
                    step_x: std::array::from_fn(|dd| step_x[ii * D + dd]),
                    id: id[ii],
                    cohort: u16::try_from(cohort[ii]).map_err(|_| {
                        IoError::Backend("continuous tracer cohort exceeds u16".to_string())
                    })?,
                    owner: crate::mass_transport::ContainerId(owner[ii]),
                    escaped: u8::try_from(escaped[ii]).map_err(|_| {
                        IoError::Backend("continuous tracer escaped flag exceeds u8".to_string())
                    })?,
                    crossed_sink: u8::try_from(crossed_sink[ii]).map_err(|_| {
                        IoError::Backend(
                            "continuous tracer crossed-sink flag exceeds u8".to_string(),
                        )
                    })?,
                    crossing_time: crossing_time[ii],
                    random_counter: random_counter[ii],
                })
                .map_err(IoError::Backend)?;
        }
        sim.continuous_tracers = Some(tracers);
    }

    if let (Some(bodies_g), Some(im)) = (tree.find_group("bodies"), sim.immersed.as_mut()) {
        let get = |name: &str| -> Result<Vec<f64>> {
            Ok(bodies_g
                .find_dataset(name)
                .ok_or_else(|| IoError::MissingPath(format!("bodies/{name}")))?
                .data
                .as_f64()
                .ok_or_else(|| IoError::MissingPath(format!("bodies/{name}: not f64")))?
                .to_vec())
        };
        let (pos, vel) = (get("position")?, get("velocity")?);
        let (mass, accreted, rate) = (
            get("mass")?,
            get("total_accreted_mass")?,
            get("accretion_rate")?,
        );
        let nb = im.bodies.len().min(mass.len());
        // the slip heat under either closure's name; files written before the receipt existed
        // leave the counters at zero.
        let optional = |name: &str| -> Option<Vec<f64>> {
            bodies_g.find_dataset(name).and_then(|d| d.data.as_f64().map(|v| v.to_vec()))
        };
        let heat = optional("magnetic_slip_heating").or_else(|| optional("exported_slip_heat"));
        let heat_rate = optional("magnetic_slip_heating_rate").or_else(|| optional("exported_slip_heat_rate"));
        for b in 0..nb {
            let body = im.bodies.get_mut(b);
            for a in 0..D {
                body.position[a] = pos[b * D + a];
                body.velocity[a] = vel[b * D + a];
            }
            body.mass = mass[b];
            if let Some(h) = heat.as_ref() {
                body.slip_heat_total = h[b];
            }
            if let Some(h) = heat_rate.as_ref() {
                body.slip_heat_rate = h[b];
            }
            if let symbi_ib::BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } = &mut body.kind
            {
                *total_accreted_mass = accreted[b];
                *accretion_rate = rate[b];
            }
        }
        // restore the evolved rigid-body rotation (orientation matrix + angular velocity) so a
        // spinning / tumbling body resumes its exact pose. checkpoints written before these datasets
        // existed keep the config values (identity orientation, prescribed omega).
        let get_opt = |name: &str| -> Option<Vec<f64>> {
            bodies_g
                .find_dataset(name)?
                .data
                .as_f64()
                .map(|d| d.to_vec())
        };
        if let (Some(orient), Some(omega)) = (get_opt("orientation"), get_opt("omega")) {
            for b in 0..nb {
                let body = im.bodies.get_mut(b);
                for i in 0..3 {
                    for j in 0..3 {
                        body.orientation[i][j] = orient[b * 9 + i * 3 + j];
                    }
                }
                for k in 0..3 {
                    body.omega[k] = omega[b * 3 + k];
                }
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
    // same axis-0-fastest walk as `extract_field` — written-then-loaded is the identity by
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
// the checkpoint mesh-coordinate gate for a nonzero interior origin lives in-crate
// because it exercises `SimStateGeneric::new_at` — the absolute-index amr-internal
// constructor (pub(crate); the public path is `SimBuilder`, which always grids at
// interior_lo = [0; D]). amr fine levels live at absolute indices, so their written
// mesh/x{1,2,3} centers must equal geom.centroid of the actual interior cells.
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use symbi_geometry::Cartesian;
    use symbi_hydro::eos::IdealGas;
    use symbi_hydro::newtonian::Newtonian;
    use symbi_xpu::{CpuSpace, HostMemory};

    type Sim = SimState<Newtonian, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;

    // the regime slug a checkpoint records is the configuration vocabulary, so a restart parses
    // it back into the enum the run was configured from.
    #[test]
    fn the_checkpoint_regime_slug_is_the_configuration_name() {
        assert_eq!(regime_name::<Newtonian, 3>(&Newtonian), "newtonian");
        assert_eq!(
            regime_name::<symbi_hydro::newtonian_mhd::NewtonianMhd, 3>(
                &symbi_hydro::newtonian_mhd::NewtonianMhd
            ),
            "nmhd"
        );
    }

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
        // description (global_cells + per-dim start/end), so verify that honors
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
    fn checkpoint_records_geometric_spacing_parameters() {
        let ratio = 0.9_f64;
        let cells = 8_usize;
        let width = (ratio - 1.0) / (ratio.powf(cells as f64) - 1.0);
        let maps = [
            symbi_geometry::AxisMap::Geometric {
                start: 0.0,
                width,
                ratio,
            },
            symbi_geometry::AxisMap::Uniform {
                start: 0.0,
                dx: 1.0,
            },
            symbi_geometry::AxisMap::Uniform {
                start: 0.0,
                dx: 1.0,
            },
        ];
        let mut sim = Sim::new(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [cells, 1, 1],
            [0.0; 3],
            [1.0 / cells as f64, 1.0, 1.0],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();
        sim.geom.set_maps(maps);

        let dir = std::env::temp_dir().join("symbi_checkpoint_geometric_spacing");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("geometric.h5");
        write_checkpoint(&sim, path.to_str().unwrap(), &Metadata::new()).unwrap();

        let tree = Hdf5Backend.read(&path).unwrap();
        let dim = tree
            .find_group("level_0")
            .unwrap()
            .find_group("mesh")
            .unwrap()
            .find_group("geometry")
            .unwrap()
            .find_group("dim_2")
            .unwrap();
        assert_eq!(
            dim.find_attr("type").unwrap().as_str("type").unwrap(),
            "geometric"
        );
        assert!((dim.find_attr("ratio").unwrap().as_f64("ratio").unwrap() - ratio).abs() < 1.0e-14);
        assert!((dim.find_attr("start").unwrap().as_f64("start").unwrap()).abs() < 1.0e-14);
        assert!((dim.find_attr("end").unwrap().as_f64("end").unwrap() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn checkpoint_roundtrip_preserves_field_layout_2d() {
        // **restart round-trip gate**: write a checkpoint then load it; the conserved state must
        // come back identical. a non-square grid (5x3) seeded with an asymmetric pattern
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
    fn hierarchy_restart_restores_each_level_from_its_own_group() {
        type Sim1 = SimState<Newtonian, 1, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let build = || {
            Sim1::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize],
                [4usize],
                [0.0],
                [0.25],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };
        let coarse = build();
        let fine = build();
        for coord in coarse.geom.interior.iter() {
            coarse.fields.cons.den.view_mut().set(coord, 2.0);
            fine.fields.cons.den.view_mut().set(coord, 7.0);
        }
        let dir = std::env::temp_dir().join("symbi_hierarchy_restart_levels");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hierarchy.h5");
        let path = path.to_str().unwrap();
        write_hierarchy_checkpoint(&[&coarse, &fine], path, &Metadata::new()).unwrap();

        let mut loaded_coarse = build();
        let mut loaded_fine = build();
        load_checkpoint_level(&mut loaded_coarse, path, 0).unwrap();
        load_checkpoint_level(&mut loaded_fine, path, 1).unwrap();
        for coord in loaded_coarse.geom.interior.iter() {
            assert_eq!(*loaded_coarse.fields.cons.den.view().at(coord), 2.0);
            assert_eq!(*loaded_fine.fields.cons.den.view().at(coord), 7.0);
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_mass_transport_tracers() {
        let build = || {
            Sim::new(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [2, 1, 1],
                [0.0; 3],
                [0.5, 1.0, 1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };
        let mut sim = build();
        sim.tracers = Some(symbi_sim_tracers::TracerSet {
            x: vec![[91.0, 92.0, 93.0], [94.0, 95.0, 96.0]],
            id: vec![u64::MAX - 1, u64::MAX],
            cohort: vec![7, 9],
            flags: vec![Default::default(); 2],
            weight: 3.5,
            owner: vec![
                crate::mass_transport::ContainerId(0),
                crate::mass_transport::ContainerId(1),
            ],
            step_owner: vec![
                crate::mass_transport::ContainerId(0),
                crate::mass_transport::ContainerId(1),
            ],
            step_flags: vec![Default::default(); 2],
            run_seed: u64::MAX - 7,
            next_id: u64::MAX - 2,
            injection_remainder: 0.25,
        });

        let dir = std::env::temp_dir().join("symbi_checkpoint_mass_transport_tracers");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tracers.h5");
        write_checkpoint(&sim, path.to_str().unwrap(), &Metadata::new()).unwrap();

        let mut restored = build();
        load_checkpoint(&mut restored, path.to_str().unwrap()).unwrap();
        let expected = sim.tracers.as_ref().unwrap();
        let actual = restored.tracers.as_ref().unwrap();
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.cohort, expected.cohort);
        assert_eq!(actual.owner, expected.owner);
        assert_eq!(actual.step_owner, expected.owner);
        assert_eq!(actual.x, vec![[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]]);
        assert_ne!(actual.x, expected.x);
        assert_eq!(actual.run_seed, expected.run_seed);
        assert_eq!(actual.next_id, expected.next_id);
        assert_eq!(
            actual.injection_remainder.to_bits(),
            expected.injection_remainder.to_bits()
        );
        assert_eq!(actual.weight.to_bits(), expected.weight.to_bits());
    }

    #[test]
    fn checkpoint_roundtrip_preserves_continuous_tracer_state() {
        let build = || {
            Sim::new(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [2, 1, 1],
                [0.0; 3],
                [0.5, 1.0, 1.0],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap()
        };
        let mut sim = build();
        let mut tracers = symbi_sim_tracers::ContinuousTracerSet::<3, HostMemory>::allocate(
            2,
            crate::mass_transport::ItoOrder::Three,
        )
        .unwrap();
        tracers.weight = 3.5;
        tracers.run_seed = u64::MAX - 7;
        tracers.next_id = u64::MAX - 2;
        tracers.injection_remainder = 0.25;
        for record in [
            symbi_sim_tracers::ContinuousTracerRecord {
                x: [1.0, 2.0, 3.0],
                step_x: [0.5, 1.5, 2.5],
                id: u64::MAX - 1,
                cohort: 7,
                owner: crate::mass_transport::ContainerId(4),
                escaped: 0,
                crossed_sink: 1,
                crossing_time: 2.25,
                random_counter: 19,
            },
            symbi_sim_tracers::ContinuousTracerRecord {
                x: [4.0, 5.0, 6.0],
                step_x: [3.5, 4.5, 5.5],
                id: u64::MAX,
                cohort: 9,
                owner: crate::mass_transport::ContainerId(5),
                escaped: 1,
                crossed_sink: 0,
                crossing_time: 3.25,
                random_counter: 23,
            },
        ] {
            tracers.push_host(record).unwrap();
        }
        sim.continuous_tracers = Some(tracers);

        let dir = std::env::temp_dir().join("symbi_checkpoint_continuous_tracers");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tracers.h5");
        write_checkpoint(&sim, path.to_str().unwrap(), &Metadata::new()).unwrap();

        let mut restored = build();
        load_checkpoint(&mut restored, path.to_str().unwrap()).unwrap();
        let expected = continuous_tracer_snap(sim.continuous_tracers.as_ref().unwrap());
        let actual = continuous_tracer_snap(restored.continuous_tracers.as_ref().unwrap());
        assert_eq!(actual.order, expected.order);
        assert_eq!(actual.x, expected.x);
        assert_eq!(actual.step_x, expected.step_x);
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.cohort, expected.cohort);
        assert_eq!(actual.owner, expected.owner);
        assert_eq!(actual.escaped, expected.escaped);
        assert_eq!(actual.crossed_sink, expected.crossed_sink);
        assert_eq!(actual.crossing_time, expected.crossing_time);
        assert_eq!(actual.random_counter, expected.random_counter);
        assert_eq!(actual.weight, expected.weight);
        assert_eq!(actual.run_seed, expected.run_seed);
        assert_eq!(actual.next_id, expected.next_id);
        assert_eq!(
            actual.injection_remainder.to_bits(),
            expected.injection_remainder.to_bits()
        );
    }

    #[test]
    fn checkpoint_saves_full_allocated_field_including_ghosts() {
        // **truncation gate**: a cell-centered dataset must carry the full allocated extent
        // (interior + 2*ng) — otherwise a restart loses the halo the
        // next stencil reads before the first ghost-fill, and the reader's `halo_width` trim
        // would over-cut interior-only data. seed every allocated cell (ghosts included) with a
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

        // (a) the den dataset must hold the padded volume (5+2*ng)x(3+2*ng).
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

        // (b) every allocated cell — especially the ghosts outside the interior — round-trips.
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

    #[test]
    fn body_state_round_trips_through_a_restart() {
        // the restart contract: a moving sink's orbit phase and cumulative
        // accreted mass survive write -> fresh-config sim -> load. without the
        // bodies group, restart resets both to config values silently.
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let build = || {
            let mut s = Sim2::new_at(
                Newtonian,
                IdealGas { gamma: 5.0 / 3.0 },
                Cartesian,
                [0isize, 0],
                [4usize, 4],
                [0.0, 0.0],
                [0.25, 0.25],
                2,
                Boundaries::uniform(BoundaryType::Outflow),
                0.4,
                Timestepping::Rk2,
                0,
            )
            .unwrap();
            s.attach_bodies(
                symbi_ib::BodyCollection::new().add(symbi_ib::Body::black_hole(
                    0,
                    symbi_algebra::Tensor::new([0.5, 0.5]),
                    symbi_algebra::Tensor::zeros(),
                    1.0,
                    0.1,
                    0.05,
                    0.5,
                    0.0,
                    0.1,
                )),
            );
            s
        };
        let mut sim = build();
        {
            let body = sim.immersed.as_mut().unwrap().bodies.get_mut(0);
            body.position = symbi_algebra::Tensor::new([0.7, 0.3]); // orbit advanced
            body.velocity = symbi_algebra::Tensor::new([-0.1, 0.2]);
            body.mass = 1.25;
            // an evolved rotation state (a spinning/tumbling body): R_z(90) + a tilted omega.
            body.orientation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
            body.omega = symbi_algebra::Tensor::new([0.1, -0.2, 0.3]);
            if let symbi_ib::BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } = &mut body.kind
            {
                *total_accreted_mass = 0.042;
                *accretion_rate = 3.5e-3;
            }
        }
        let dir = std::env::temp_dir().join("symbi_checkpoint_bodystate");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bodystate.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        let mut restored = build(); // fresh: body back at config values
        load_checkpoint(&mut restored, path).unwrap();
        let b = restored.immersed.as_ref().unwrap().bodies.get(0);
        assert_eq!(b.position[0], 0.7);
        assert_eq!(b.position[1], 0.3);
        assert_eq!(b.velocity[0], -0.1);
        assert_eq!(b.mass, 1.25);
        // the evolved rotation survives the restart (else a tumbling body resets to identity/config).
        assert_eq!(
            b.orientation,
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!([b.omega[0], b.omega[1], b.omega[2]], [0.1, -0.2, 0.3]);
        match b.kind {
            symbi_ib::BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } => {
                assert_eq!(total_accreted_mass, 0.042);
                assert_eq!(accretion_rate, 3.5e-3);
            }
            _ => panic!("body kind lost through the round trip"),
        }
    }

    #[test]
    fn body_diagnostics_series_lands_in_the_checkpoint() {
        // the Mdot(t)/F_acc(t) accretion series: pushed per
        // step by evolve_bodies, flushed into every checkpoint as the
        // `body_diagnostics` group. shapes: time/dt [len], mass_delta [len, nb],
        // force [len, nb, D]. this pins the group layout the steady-state
        // detector reads.
        type Sim2 = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let mut sim = Sim2::new_at(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [0isize, 0],
            [4usize, 4],
            [0.0, 0.0],
            [0.25, 0.25],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();
        sim.attach_bodies(
            symbi_ib::BodyCollection::new().add(symbi_ib::Body::black_hole(
                0,
                symbi_algebra::Tensor::new([0.5, 0.5]),
                symbi_algebra::Tensor::zeros(),
                1.0,
                0.1,
                0.05,
                0.5,
                0.0,
                0.1,
            )),
        );

        // two steps' worth of exchanges, distinct so ordering bugs are loud.
        let im = sim.immersed.as_mut().unwrap();
        let mut d = symbi_ib::BodyDelta::<f64, 2>::new(0);
        d.mass_delta = 0.25;
        d.force_delta = symbi_algebra::Tensor::new([1.0, -2.0]);
        im.history.push(0.1, 0.1, &[d]);
        d.mass_delta = 0.5;
        im.history.push(0.2, 0.1, &[d]);

        let dir = std::env::temp_dir().join("symbi_checkpoint_bodydiag");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bodydiag.h5");
        let path = path.to_str().unwrap();
        write_checkpoint(&sim, path, &Metadata::new()).unwrap();

        let tree = Hdf5Backend.read(std::path::Path::new(path)).unwrap();
        let diag = tree
            .find_group("body_diagnostics")
            .expect("body_diagnostics group missing from the checkpoint");
        let time = diag
            .find_dataset("time")
            .unwrap()
            .data
            .as_f64()
            .unwrap()
            .to_vec();
        let mass = diag
            .find_dataset("mass_delta")
            .unwrap()
            .data
            .as_f64()
            .unwrap()
            .to_vec();
        let force = diag
            .find_dataset("force")
            .unwrap()
            .data
            .as_f64()
            .unwrap()
            .to_vec();
        assert_eq!(time, vec![0.1, 0.2]);
        assert_eq!(mass, vec![0.25, 0.5]);
        assert_eq!(force, vec![1.0, -2.0, 1.0, -2.0]);
    }

    #[test]
    fn isothermal_checkpoints_record_the_sound_speed() {
        // the isothermal eos closes with p = cs^2 rho and stores no pressure
        // dataset, so cs must travel in metadata for readers to reconstruct
        // pressure-dependent fields. energy regimes must not carry the attr:
        // their sound speed varies per cell and a constant would be a lie.
        use symbi_hydro::IsoNewtonian;
        use symbi_hydro::eos::Isothermal;

        type IsoSim = SimState<IsoNewtonian, 2, Cartesian, Isothermal<f64>, CpuSpace, HostMemory>;
        let iso = IsoSim::new_at(
            IsoNewtonian,
            Isothermal { cs: 0.75 },
            Cartesian,
            [0isize, 0],
            [4usize, 4],
            [0.0, 0.0],
            [0.25, 0.25],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();

        let dir = std::env::temp_dir().join("symbi_checkpoint_iso_cs");
        std::fs::create_dir_all(&dir).unwrap();
        let iso_path = dir.join("iso.h5");
        write_checkpoint(&iso, iso_path.to_str().unwrap(), &Metadata::new()).unwrap();
        let tree = Hdf5Backend.read(&iso_path).unwrap();
        let meta = tree.find_group("metadata").unwrap();
        let cs = meta
            .find_attr("sound_speed")
            .expect("isothermal metadata must carry sound_speed")
            .as_f64("sound_speed")
            .unwrap();
        assert_eq!(cs, 0.75);

        type AdiSim = SimState<Newtonian, 2, Cartesian, IdealGas<f64>, CpuSpace, HostMemory>;
        let adi = AdiSim::new_at(
            Newtonian,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
            [0isize, 0],
            [4usize, 4],
            [0.0, 0.0],
            [0.25, 0.25],
            2,
            Boundaries::uniform(BoundaryType::Outflow),
            0.4,
            Timestepping::Rk2,
            0,
        )
        .unwrap();
        let adi_path = dir.join("adi.h5");
        write_checkpoint(&adi, adi_path.to_str().unwrap(), &Metadata::new()).unwrap();
        let tree = Hdf5Backend.read(&adi_path).unwrap();
        let meta = tree.find_group("metadata").unwrap();
        assert!(
            meta.find_attr("sound_speed").is_none(),
            "energy-regime metadata must not carry a constant sound_speed"
        );
    }
}

/// the recorded census groups of a store, one per registration carrying at least one sample.
///
/// extracted so both checkpoint writers emit them. the uni-grid writer and the hierarchy writer
/// build their trees separately, and a census group written by only one of them leaves every run
/// on the other driver recording nothing — a checkpoint with no census group reads exactly like a
/// run that registered none, so the omission carries no signal at all.
fn census_groups<'a, R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &'a SimStateGeneric<R, D, DOF, M, E, S, Mem>,
) -> Vec<Tree<'a>>
where
    R: symbi_hydro::regime::Regime<f64, D>,
    M: symbi_geometry::Metric<f64, D> + Copy,
    E: symbi_hydro::eos::Eos<f64>,
    S: symbi_xpu::ExecutionSpace,
    Mem: symbi_xpu::MemorySpace,
{
    // the registered binned reductions, one group each. like the body series these cover
    // this run segment only and restart empty on checkpoint load, so a restart chain
    // concatenates offline rather than the run carrying its whole history forward.
    let mut out = Vec::new();
    for registered in &sim.censuses {
        let spec = registered.evaluator.spec();
        let history = &registered.history;
        if history.is_empty() {
            continue;
        }
        let (n, n_seg, n_val) = (history.len(), history.n_segments(), history.n_values());
        let mut group = Tree::new(&format!("census/{}", spec.name()))
            .with_attr("n_segments", n_seg as u64)
            .with_attr("n_values", n_val as u64)
            // the accumulator labels, in the order the `values` axis carries them, so a
            // reader names a column without re-deriving the registration order.
            .with_attr("value_names", spec.value_names().join(","))
            .with_attr("op", reduction_op_tag(spec.op()))
            // the size of the compiled per-cell graph: what a census actually costs, since
            // the cost scales with the dag rather than with the accumulator count.
            .with_attr("node_count", registered.evaluator.node_count() as u64)
            // an accumulating census stores one row folded from many samples rather than a row
            // apiece, so the row alone does not say what it is an accumulation of. the count and
            // the two endpoints make it self-describing: a reader forms the time average by
            // dividing an additive row by the count, and two run segments combine as a
            // count-weighted sum without either having stored its samples.
            .with_attr("accumulated", u64::from(history.accumulate()))
            .with_attr("cadence", spec.cadence().tag().to_string())
            .with_dataset(Dataset::new("time", vec![n], DataRef::F64(history.time())))
            // segment-major within a sample. a reader reshapes the segment axis to the
            // per-axis bin counts in registration order, last axis varying fastest.
            .with_dataset(Dataset::new(
                "values",
                vec![n, n_seg, n_val],
                DataRef::F64(history.values()),
            ))
            // cells that fell outside the binning. a census that silently under-covers its
            // domain is indistinguishable from a physics result, so the shortfall travels
            // with the numbers.
            .with_dataset(Dataset::new(
                "dropped",
                vec![n],
                DataRef::U64(history.dropped()),
            ))
            // which level produced each row, and the span it covers. an accumulating row is folded
            // from many samples, so without the count there is no way to recover a time average
            // from a running sum, and without the level a per-level row is indistinguishable from a
            // composite one. all ones and all zeros respectively in the ordinary case, which costs
            // sixteen bytes a row and makes every row self-describing.
            .with_dataset(Dataset::new(
                "level",
                vec![n],
                DataRef::U64(history.level()),
            ))
            .with_dataset(Dataset::new(
                "n_samples",
                vec![n],
                DataRef::U64(history.n_samples()),
            ))
            .with_dataset(Dataset::new(
                "t_start",
                vec![n],
                DataRef::F64(history.t_start()),
            ));
        // the edges are a property of the registration, not of a sample, so they are
        // written once per axis rather than per row.
        for (k, axis) in spec.axes().iter().enumerate() {
            group = group
                .with_attr(&format!("axis{k}_name"), axis.name().to_string())
                .with_dataset(Dataset::new(
                    &format!("axis{k}_edges"),
                    vec![axis.edges().len()],
                    DataRef::F64(axis.edges()),
                ));
        }
        out.push(group);
    }
    out
}
