// =============================================================================
// lib.rs
//
// python extension module bridging the python frontend to the rust solver.
// reproduces the pybind11 `cpu_ext.run_simulation` contract exactly so the
// existing `simbi` python package calls into rust with zero frontend changes:
// - parse the `sim_info` dict (pydantic `to_execution_dict`) into a plain Config
// - drain the `prim_gen` python iterator into a typed primitive buffer
// - release the GIL, dispatch on (regime, dims, geometry, eos), run, checkpoint
//
// the on-disk HDF5 layout is written by `symbi_sim::checkpoint::write_checkpoint`
// (schema-compatible with the existing python reader), so results are read back
// by the unchanged `simbi.reader` / `simbi.viz` stack.
//
// usage (from python, unchanged):
//  import simbi.libs.cpu_ext as backend
//  backend.run_simulation(prim_gen=..., staggered_bfields=..., sim_info=...,
//                         a=..., adot=...)
// =============================================================================

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use symbi::prelude::*;
use symbi::sim::refinement::transfer::prolong_field;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi_algebra::Tensor;
use symbi::symbi_grid::Field;
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Eos;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::PrimG;
use symbi_geometry::MotionState;
use symbi_io::Metadata;
use symbi_sim::checkpoint::write_hierarchy_checkpoint;

// =============================================================================
// parsed configuration — a plain-rust mirror of the python exec_dict. the
// monomorphized dispatch below reads these tags to pick the concrete SimState.
// =============================================================================

struct Config {
    regime:              String,
    coord_system:        String,
    cyl_plane:           CylPlane,
    dims:                usize,
    n_cells:             [usize; 3],
    x_lo:                [f64; 3],
    dx:                  [f64; 3],
    boundaries:          Vec<BoundaryType>,
    cfl:                 f64,
    gamma:               f64,
    cs:                  f64,
    locally_isothermal:  bool,
    refinement_enabled:  bool,
    // each region is a flat [lo_0, hi_0, lo_1, hi_1, ..] bound list (2 per axis).
    refinement_regions:  Vec<Vec<f64>>,
    // homologous / translating mesh motion (linear: a_ddot = 0). a0/adot are the
    // scale-factor callables evaluated at start_time (set in run_simulation).
    mesh_motion:         bool,
    is_homologous:       bool,
    scale_a0:            f64,
    scale_adot:          f64,
    solver:              Solver,
    solver_name:         String,
    reconstruction_name: String,
    timestepping:        Timestepping,
    plm_theta:           f64,
    dlogt:               f64,
    viscosity:           f64,
    x1_spacing:          String,
    start_time:          f64,
    checkpoint_index:    u64,
    t_final:             f64,
    checkpoint_interval: f64,
    data_dir:            String,
}

// =============================================================================
// dict extraction helpers
// =============================================================================

/// read a python enum field as its lowercase `.value` string; falls back to the
/// raw value when the object is already a plain string (e.g. `regime`).
fn enum_str(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    let obj = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("sim_info missing '{key}'")))?;
    let s: String = match obj.getattr("value") {
        Ok(v) => v.extract()?,
        Err(_) => obj.extract()?,
    };
    Ok(s.to_lowercase())
}

fn get_f64(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<f64> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("sim_info missing '{key}'")))?
        .extract()
}

fn get_usize(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("sim_info missing '{key}'")))?
        .extract()
}

/// optional float with default (the dict almost always carries these, but a
/// custom config may omit a checkpoint-only field).
fn get_f64_or(dict: &Bound<'_, PyDict>, key: &str, default: f64) -> f64 {
    dict.get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(default)
}

/// optional enum/string with default, read via `.value` then raw.
fn enum_str_or(dict: &Bound<'_, PyDict>, key: &str, default: &str) -> String {
    let Ok(Some(obj)) = dict.get_item(key) else {
        return default.to_string();
    };
    let s: Option<String> = match obj.getattr("value") {
        Ok(v) => v.extract().ok(),
        Err(_) => obj.extract().ok(),
    };
    s.map(|s| s.to_lowercase()).unwrap_or_else(|| default.to_string())
}

fn solver_from_str(s: &str) -> PyResult<Solver> {
    match s {
        "hlle" => Ok(Solver::Hlle),
        "hllc" => Ok(Solver::Hllc),
        "hlld" => Ok(Solver::Hlld),
        other => Err(PyValueError::new_err(format!("unknown solver '{other}'"))),
    }
}

fn timestepping_from_str(s: &str) -> PyResult<Timestepping> {
    match s {
        "euler" => Ok(Timestepping::Euler),
        "rk2" => Ok(Timestepping::Rk2),
        "rk3" => Ok(Timestepping::Rk3),
        other => Err(PyValueError::new_err(format!("unknown timestepping '{other}'"))),
    }
}

fn boundary_from_str(s: &str) -> PyResult<BoundaryType> {
    match s {
        "periodic" => Ok(BoundaryType::Periodic),
        "outflow" => Ok(BoundaryType::Outflow),
        "reflecting" | "reflect" => Ok(BoundaryType::Reflect),
        other => Err(PyValueError::new_err(format!("unsupported boundary '{other}'"))),
    }
}

/// parse the exec_dict into a plain Config. only the fields the solver needs;
/// the rest of the dict (refinement, immersed bodies, expressions) is ignored
/// until those paths are wired.
fn parse_config(dict: &Bound<'_, PyDict>) -> PyResult<Config> {
    let dims = get_usize(dict, "dimensionality")?;

    // resolution is [nx, ny, nz], padded with 1 for unused axes.
    let res: Vec<usize> = dict
        .get_item("resolution")?
        .ok_or_else(|| PyValueError::new_err("sim_info missing 'resolution'"))?
        .extract()?;
    let mut n_cells = [1usize; 3];
    for (ii, &n) in res.iter().take(3).enumerate() {
        n_cells[ii] = n;
    }

    // per-axis bounds from x1/x2/x3_bounds; derive cell widths.
    let mut x_lo = [0.0f64; 3];
    let mut dx = [1.0f64; 3];
    for ii in 0..3 {
        let key = format!("x{}_bounds", ii + 1);
        if let Some(b) = dict.get_item(&key)? {
            let (lo, hi): (f64, f64) = b.extract()?;
            x_lo[ii] = lo;
            dx[ii] = if n_cells[ii] > 0 { (hi - lo) / n_cells[ii] as f64 } else { 1.0 };
        }
    }

    // boundary_conditions is a flat list (lo, hi per axis); map each.
    let bc_objs = dict
        .get_item("boundary_conditions")?
        .ok_or_else(|| PyValueError::new_err("sim_info missing 'boundary_conditions'"))?;
    let mut boundaries = Vec::new();
    for obj in bc_objs.try_iter()? {
        let obj = obj?;
        let s: String = match obj.getattr("value") {
            Ok(v) => v.extract()?,
            Err(_) => obj.extract()?,
        };
        boundaries.push(boundary_from_str(&s.to_lowercase())?);
    }

    let gamma = dict
        .get_item("adiabatic_index")?
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(5.0 / 3.0);

    let solver_name = enum_str(dict, "solver")?;

    // canonicalize the coordinate system: the three cylindrical python variants
    // (cylindrical / axis_cylindrical / planar_cylindrical) all map to the one
    // `Cylindrical` metric, distinguished by the 2D MHD plane selector.
    let raw_coord = enum_str(dict, "coord_system")?;
    let coord_system = if raw_coord.contains("cylindrical") {
        "cylindrical".to_string()
    } else {
        raw_coord.clone()
    };
    let cyl_plane = if raw_coord == "planar_cylindrical" {
        CylPlane::RPhi // the (r, phi) disk plane, out-of-plane B_z
    } else {
        CylPlane::Rz // axisymmetric (r, z), out-of-plane swirl B_phi (the default)
    };

    Ok(Config {
        regime: enum_str(dict, "regime")?,
        coord_system,
        cyl_plane,
        dims,
        n_cells,
        x_lo,
        dx,
        boundaries,
        cfl: get_f64(dict, "cfl_number")?,
        gamma,
        // isothermal sound speed (imhd) — the canonical `ambient_sound_speed`
        // field; adiabatic regimes ignore it.
        cs: {
            let s = get_f64_or(dict, "ambient_sound_speed", 1.0);
            if s > 0.0 { s } else { 1.0 }
        },
        locally_isothermal: dict
            .get_item("locally_isothermal")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        refinement_enabled: dict
            .get_item("refinement_enabled")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        refinement_regions: dict
            .get_item("refinement_regions")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<Vec<Vec<f64>>>().ok())
            .unwrap_or_default(),
        mesh_motion: dict
            .get_item("mesh_motion")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        is_homologous: dict
            .get_item("is_homologous")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        // placeholders; filled from the a/adot callables in run_simulation.
        scale_a0: 1.0,
        scale_adot: 0.0,
        solver: solver_from_str(&solver_name)?,
        solver_name,
        reconstruction_name: enum_str_or(dict, "reconstruction", "plm"),
        timestepping: timestepping_from_str(&enum_str(dict, "timestepping")?)?,
        plm_theta: get_f64_or(dict, "plm_theta", 1.5),
        dlogt: get_f64_or(dict, "dlogt", 0.0),
        viscosity: get_f64_or(dict, "viscosity", 0.0),
        x1_spacing: enum_str_or(dict, "x1_spacing", "linear"),
        start_time: get_f64_or(dict, "start_time", 0.0),
        checkpoint_index: dict
            .get_item("checkpoint_index")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<u64>().ok())
            .unwrap_or(0),
        t_final: get_f64(dict, "end_time")?,
        checkpoint_interval: get_f64(dict, "checkpoint_interval")?,
        data_dir: dict
            .get_item("data_directory")?
            .ok_or_else(|| PyValueError::new_err("sim_info missing 'data_directory'"))?
            .extract()?,
    })
}

/// drain a python primitive-generator into a flat per-cell buffer. each yielded
/// tuple is a hydro primitive row `(rho, v1, .., vD, pre)` of length `2 + D`,
/// in axis-0-fastest order (x inner) — matching both the python generators and
/// the checkpoint write convention. kept arity-generic (a `Vec<f64>` per cell)
/// so one drain serves every dimension.
fn drain_prims(prim_gen: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let mut buf = Vec::new();
    for item in prim_gen.try_iter()? {
        let row: Vec<f64> = item?.extract()?;
        buf.push(row);
    }
    Ok(buf)
}

/// drain the `staggered_bfields` list (one face-field generator per axis: bx, by,
/// bz) into flat per-axis buffers. each generator yields face values in
/// axis-0-fastest order over its staggered (ni+dx)x(nj+dy)x(nk+dz) extent.
fn drain_bfields(staggered: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    let mut bufs = Vec::new();
    for axis_gen in staggered.try_iter()? {
        let axis_gen = axis_gen?;
        let mut buf = Vec::new();
        for v in axis_gen.try_iter()? {
            buf.push(v?.extract::<f64>()?);
        }
        bufs.push(buf);
    }
    Ok(bufs)
}

// =============================================================================
// dispatch + run
// =============================================================================

/// build `Boundaries<D>` from the flat (lo, hi per axis) config list, defaulting
/// to outflow for any face the config didn't specify.
fn boundaries_nd<const D: usize>(bcs: &[BoundaryType]) -> Boundaries<D> {
    Boundaries(std::array::from_fn(|ax| {
        let lo = bcs.get(2 * ax).copied().unwrap_or(BoundaryType::Outflow);
        let hi = bcs.get(2 * ax + 1).copied().unwrap_or(lo);
        [lo, hi]
    }))
}

/// the time-scheduled checkpoint loop. ONE path for every run: a single grid is a
/// 1-level `Hierarchy` (`Hierarchy::single`), an AMR run is a multi-level one — the
/// driver (`evolve_with_callback`) and the writer (`write_hierarchy_checkpoint`,
/// all levels in one file) are identical either way. no separate single-grid vs
/// AMR startup. generic over the concrete monomorphized hierarchy the macro builds.
fn run_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    hier: &mut Hierarchy<R, D, DOF, M, E, S, Mem, K>,
    cfg:  &Config,
) -> Result<(), Box<dyn std::error::Error>>
where
    R:   Regime<f64, D>,
    M:   Metric<f64, D> + Copy + Send + Sync,
    E:   Eos<f64> + Send + Sync,
    S:   ExecutionSpace,
    Mem: MemorySpace + Sync,
    K:   KernelSet<D, DOF, Mem, f64>,
{
    let data_dir = &cfg.data_dir;
    let cp_interval = if cfg.checkpoint_interval > 0.0 {
        cfg.checkpoint_interval
    } else {
        f64::INFINITY
    };
    let mut next_cp = cp_interval;
    let mut cp_index: u64 = cfg.checkpoint_index + 1;

    hier.evolve_with_callback(cfg.t_final, 1, |h| {
        while h.levels[0].state.time + 1e-12 >= next_cp && next_cp.is_finite() {
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let path = format!("{data_dir}{cp_index:04}.h5");
            let _ = write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cp_index));
            cp_index += 1;
            next_cp += cp_interval;
        }
    })?;

    let states: Vec<&_> = hier.levels.iter().map(|l| &l.state).collect();
    let final_path = format!("{data_dir}final.h5");
    write_hierarchy_checkpoint(&states, &final_path, &checkpoint_metadata(cfg, cp_index))?;
    Ok(())
}

/// build `RefinementRegion<D>` boxes from the flat per-region [lo_0, hi_0, lo_1,
/// hi_1, ..] bound lists the python config supplies (2 entries per axis).
fn refinement_regions_nd<const D: usize>(
    regions: &[Vec<f64>],
) -> Result<Vec<RefinementRegion<D>>, String> {
    if regions.is_empty() {
        return Err("refinement_enabled but no refinement_regions provided".to_string());
    }
    let mut out = Vec::with_capacity(regions.len());
    for r in regions {
        if r.len() < 2 * D {
            return Err(format!(
                "refinement region needs {} bounds (lo,hi per axis), got {}",
                2 * D,
                r.len()
            ));
        }
        out.push(RefinementRegion {
            x_lo: std::array::from_fn(|ax| r[2 * ax]),
            x_hi: std::array::from_fn(|ax| r[2 * ax + 1]),
        });
    }
    Ok(out)
}

/// the coarse->fine prolongation order: ONE above the interior reconstruction
/// (pcm -> plm, plm -> ppm), so refinement boundaries never drop a spatial order.
fn prolong_order_for(reconstruction: &str) -> ProlongOrder {
    match reconstruction {
        "pcm" => ProlongOrder::Plm,
        _ => ProlongOrder::Ppm, // plm (the default) -> ppm
    }
}

/// the mesh-motion state from the config: static, homologous expansion (linear,
/// `a += a_dot*dt`, a_ddot = 0), or uniform cartesian translation. `scale_a0` /
/// `scale_adot` are the python scale-factor callables already evaluated at
/// start_time (the rust model integrates a from a constant rate).
fn motion_state(cfg: &Config) -> MotionState<f64> {
    if !cfg.mesh_motion {
        MotionState::static_mesh()
    } else if cfg.is_homologous {
        MotionState::homologous(cfg.scale_a0, cfg.scale_adot)
    } else {
        // uniform translation keeps a = 1; a_dot is the translation velocity.
        MotionState::uniform(1.0, cfg.scale_adot)
    }
}

/// wrap a built sim + its kernel-set into a `Hierarchy`: a single grid (1 level),
/// or — when refinement is requested — a refined hierarchy whose fine interiors
/// are seeded from the coarse level (conservative prolongation at reconstruction
/// order + 1). `$make` rebuilds a fine level's kernel-set. the unified `run_loop`
/// drives either uniformly.
macro_rules! into_hierarchy {
    ($sim:expr, $kernels:expr, $cfg:expr, $d:literal, $make:expr) => {{
        let mut sim = $sim;
        // mesh motion lives on the (coarse) state — set before wrapping. static
        // for the common case; the gates above keep motion to single-grid hydro.
        sim.motion = motion_state($cfg);
        if $cfg.refinement_enabled {
            let regions = refinement_regions_nd::<$d>(&$cfg.refinement_regions)?;
            let prolong = prolong_order_for(&$cfg.reconstruction_name);
            let h = Hierarchy::with_refinement(sim, $kernels, &regions, prolong, $make)
                .map_err(|e| format!("refinement build: {e:?}"))?;
            h.seed_fine_from_coarse().map_err(|e| format!("fine-level seed: {e:?}"))?;
            h
        } else {
            Hierarchy::single(sim, $kernels)
        }
    }};
}

/// build one monomorphized hydro sim (regime x D x geometry, ideal-gas eos),
/// seed it from the drained python buffer (axis-0-fastest linearization), and
/// drive the run loop. the build chain's typestate + per-regime `substrate()`
/// resolve concretely here, so no generic bound-threading is needed.
macro_rules! build_and_run_hydro {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<$regime_ty, $d, $geom_ty, IdealGas<f64>>;

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("allocate failed: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                // axis-0-fastest: lin = i0 + i1*n0 + i2*n0*n1 (matches the generators)
                let mut lin = 0usize;
                let mut stride = 1usize;
                for ax in 0..$d {
                    lin += idx[ax] as usize * stride;
                    stride *= n[ax];
                }
                let row = &prims[lin];
                Prim {
                    rho: row[0],
                    vel: Tensor::new(std::array::from_fn(|k| row[1 + k])),
                    pre: row[1 + $d],
                }
            })
            .build();

        let sub = sim.substrate().with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?;
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d,
            |s| s.substrate().with_solver(solver).expect("fine-level kernel set"));
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// expand the (geometry x dims) arms for one hydro regime. cartesian / spherical
/// / cylindrical are all wired across 1/2/3d (each is a unit-struct `Metric`).
macro_rules! hydro_dispatch {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            (1, "cartesian")   => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 1, Cartesian, Cartesian),
            (2, "cartesian")   => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 2, Cartesian, Cartesian),
            (3, "cartesian")   => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 3, Cartesian, Cartesian),
            (1, "spherical")   => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 1, Spherical, Spherical),
            (2, "spherical")   => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 2, Spherical, Spherical),
            (3, "spherical")   => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 3, Spherical, Spherical),
            (1, "cylindrical") => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 1, Cylindrical, Cylindrical),
            (2, "cylindrical") => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 2, Cylindrical, Cylindrical),
            (3, "cylindrical") => build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 3, Cylindrical, Cylindrical),
            (d, g) => Err(format!("no dispatch arm for (dims={d}, coord={g}) yet")),
        }
    };
}

/// linear (axis-0-fastest) cell index from a `[isize; D]` interior index and the
/// per-axis cell counts — the order the python generators yield.
macro_rules! lin_index {
    ($idx:expr, $n:expr, $d:literal) => {{
        let mut lin = 0usize;
        let mut stride = 1usize;
        for ax in 0..$d {
            lin += $idx[ax] as usize * stride;
            stride *= $n[ax];
        }
        lin
    }};
}

/// build one monomorphized ADIABATIC MHD sim (Rmhd or NewtonianMhd; both are
/// MhdPrim + IdealGas, DOF=3) and drive it. cell state comes from `prim_gen`
/// (rho, vx, vy, vz, p — NO cell B); the staggered face B from `staggered_bfields`:
/// in-grid axes `0..D` seed the CT faces (the divergence-free truth; cell B is the
/// bcell-from-bface kernel's job), transverse axes `D..3` seed cell-centered B.
macro_rules! build_and_run_mhd {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<$regime_ty, $d, 3, $geom_ty, IdealGas<f64>>;

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if bufs.len() < 3 {
            return Err(format!("mhd needs 3 staggered b-field generators, got {}", bufs.len()));
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("allocate failed: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let lin = lin_index!(idx, n, $d);
                let row = &prims[lin];
                let mag_arr: [f64; 3] = std::array::from_fn(|k| if k < $d { 0.0 } else { bufs[k][lin] });
                MhdPrim {
                    hydro: Prim { rho: row[0], vel: Tensor::new([row[1], row[2], row[3]]), pre: row[4] },
                    mag: Tensor::new(mag_arr),
                }
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        let sub = sim.substrate().with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?;
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d,
            |s| s.substrate().with_solver(solver).expect("fine-level kernel set"));
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// build one monomorphized ISOTHERMAL MHD sim (IsothermalMhd + Isothermal eos,
/// DOF=3) and drive it. the iso primitive has NO pressure slot (IsoModel ZST), so
/// `prim_gen` yields (rho, vx, vy, vz); the eos closure is p = cs^2 rho.
macro_rules! build_and_run_imhd {
    ($cfg:expr, $prims:expr, $bufs:expr, $d:literal, $geom:expr, $geom_ty:ty) => {{
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<IsothermalMhd, $d, 3, $geom_ty, Isothermal<f64>>;

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if bufs.len() < 3 {
            return Err(format!("imhd needs 3 staggered b-field generators, got {}", bufs.len()));
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("allocate failed: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let lin = lin_index!(idx, n, $d);
                let row = &prims[lin];
                let mag_arr: [f64; 3] = std::array::from_fn(|k| if k < $d { 0.0 } else { bufs[k][lin] });
                MhdPrimG::<f64, 3, IsoModel> {
                    hydro: PrimG { rho: row[0], vel: Tensor::new([row[1], row[2], row[3]]), pre: Default::default() },
                    mag: Tensor::new(mag_arr),
                }
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        let sub = sim.substrate().with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?;
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d,
            |s| s.substrate().with_solver(solver).expect("fine-level kernel set"));
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// expand the (geometry x dims) arms for an adiabatic mhd regime. cartesian /
/// spherical / cylindrical across 1/2/3d (the cylindrical 2D plane is selected by
/// `cfg.cyl_plane`, threaded into every build).
macro_rules! mhd_dispatch {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            (1, "cartesian")   => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 1, Cartesian, Cartesian),
            (2, "cartesian")   => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 2, Cartesian, Cartesian),
            (3, "cartesian")   => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 3, Cartesian, Cartesian),
            (1, "spherical")   => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 1, Spherical, Spherical),
            (2, "spherical")   => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 2, Spherical, Spherical),
            (3, "spherical")   => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 3, Spherical, Spherical),
            (1, "cylindrical") => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 1, Cylindrical, Cylindrical),
            (2, "cylindrical") => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 2, Cylindrical, Cylindrical),
            (3, "cylindrical") => build_and_run_mhd!($cfg, $prims, $bufs, $regime, $regime_ty, 3, Cylindrical, Cylindrical),
            (d, g) => Err(format!("no mhd dispatch arm for (dims={d}, coord={g}) yet")),
        }
    };
}

/// expand the (geometry x dims) arms for isothermal mhd. cartesian / spherical /
/// cylindrical across 1/2/3d.
macro_rules! imhd_dispatch {
    ($cfg:expr, $prims:expr, $bufs:expr) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            (1, "cartesian")   => build_and_run_imhd!($cfg, $prims, $bufs, 1, Cartesian, Cartesian),
            (2, "cartesian")   => build_and_run_imhd!($cfg, $prims, $bufs, 2, Cartesian, Cartesian),
            (3, "cartesian")   => build_and_run_imhd!($cfg, $prims, $bufs, 3, Cartesian, Cartesian),
            (1, "spherical")   => build_and_run_imhd!($cfg, $prims, $bufs, 1, Spherical, Spherical),
            (2, "spherical")   => build_and_run_imhd!($cfg, $prims, $bufs, 2, Spherical, Spherical),
            (3, "spherical")   => build_and_run_imhd!($cfg, $prims, $bufs, 3, Spherical, Spherical),
            (1, "cylindrical") => build_and_run_imhd!($cfg, $prims, $bufs, 1, Cylindrical, Cylindrical),
            (2, "cylindrical") => build_and_run_imhd!($cfg, $prims, $bufs, 2, Cylindrical, Cylindrical),
            (3, "cylindrical") => build_and_run_imhd!($cfg, $prims, $bufs, 3, Cylindrical, Cylindrical),
            (d, g) => Err(format!("no imhd dispatch arm for (dims={d}, coord={g}) yet")),
        }
    };
}

/// build one monomorphized ISOTHERMAL HYDRO sim (IsoNewtonian + Isothermal eos,
/// DOF=D) and drive it. the iso primitive has NO pressure slot (IsoModel ZST), so
/// `prim_gen` yields (rho, v1..vD). iso is HLLE-only by physics (no contact wave),
/// so `sim.substrate()` is used directly (no solver knob).
///
/// globally isothermal: cs is the uniform scalar from `sound_speed`.
/// locally isothermal: `prim_gen` yields one extra component, the per-cell initial
/// pressure p(x); cs^2(x) = p(x)/rho(x) is derived once (compute_isothermal_cs2)
/// and HELD fixed — the position-dependent "temperature" the substrate flows
/// through c2p / flux / cfl.
macro_rules! build_and_run_iso {
    ($cfg:expr, $prims:expr, $d:literal, $geom:expr, $geom_ty:ty) => {{
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<IsoNewtonian, $d, $geom_ty, Isothermal<f64>>;

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        // locally isothermal carries an extra per-cell pressure component.
        let want = if cfg.locally_isothermal { $d + 2 } else { $d + 1 };
        if let Some(row) = prims.first() {
            if row.len() < want {
                return Err(format!(
                    "isothermal prim row has {} components, expected {want} (rho, v1..v{}{})",
                    row.len(), $d, if cfg.locally_isothermal { ", p_local" } else { "" },
                ));
            }
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("allocate failed: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let lin = lin_index!(idx, n, $d);
                let row = &prims[lin];
                PrimG::<f64, $d, IsoModel> {
                    rho: row[0],
                    vel: Tensor::new(std::array::from_fn(|k| row[1 + k])),
                    pre: Default::default(),
                }
            })
            .build();

        // iso is HLLE-only; the substrate front door gives the kernel-set directly.
        let sub = sim.substrate();

        if cfg.locally_isothermal {
            // derive cs^2(x) = p(x)/rho(x) from the per-cell initial pressure, then HOLD it.
            let pre_ic = Field::<f64, $d, _>::zeros(&sim.geom.allocated)
                .map_err(|e| format!("pre_ic alloc: {e:?}"))?;
            let interior = sim.geom.interior.clone();
            let mut coord: [isize; $d] = std::array::from_fn(|ax| interior.spaces[ax].lo);
            for lin in 0..total {
                pre_ic.view_mut().set(coord, prims[lin][1 + $d]);
                for ax in 0..$d {
                    coord[ax] += 1;
                    if coord[ax] < interior.spaces[ax].hi { break; }
                    coord[ax] = interior.spaces[ax].lo;
                }
            }
            sub.compute_isothermal_cs2(&sim.fields.cons.den, &pre_ic, &interior);
        }

        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| s.substrate());

        // locally isothermal + refinement: the per-cell cs^2(x) "temperature"
        // lives in the iso kernel-set (not the SimState), so prolong it coarse ->
        // fine separately (the cons fields were already seeded). without this the
        // fine levels would fall back to a uniform cs^2 from the eos.
        if cfg.refinement_enabled && cfg.locally_isothermal {
            let order = prolong_order_for(&cfg.reconstruction_name);
            for ll in 1..hier.levels.len() {
                let (lo, hi) = hier.levels.split_at(ll);
                let region = hi[0].state.geom.interior.clone();
                let zero = Field::zeros(&lo[ll - 1].state.geom.allocated)
                    .map_err(|e| format!("cs2 prolong alloc: {e:?}"))?;
                prolong_field(&lo[ll - 1].kernels.cs2, &zero, &hi[0].kernels.cs2,
                              &region, order, 0.0);
            }
        }

        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// expand the (geometry x dims) arms for isothermal hydro. cartesian / spherical
/// / cylindrical across 1/2/3d.
macro_rules! iso_dispatch {
    ($cfg:expr, $prims:expr) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            (1, "cartesian")   => build_and_run_iso!($cfg, $prims, 1, Cartesian, Cartesian),
            (2, "cartesian")   => build_and_run_iso!($cfg, $prims, 2, Cartesian, Cartesian),
            (3, "cartesian")   => build_and_run_iso!($cfg, $prims, 3, Cartesian, Cartesian),
            (1, "spherical")   => build_and_run_iso!($cfg, $prims, 1, Spherical, Spherical),
            (2, "spherical")   => build_and_run_iso!($cfg, $prims, 2, Spherical, Spherical),
            (3, "spherical")   => build_and_run_iso!($cfg, $prims, 3, Spherical, Spherical),
            (1, "cylindrical") => build_and_run_iso!($cfg, $prims, 1, Cylindrical, Cylindrical),
            (2, "cylindrical") => build_and_run_iso!($cfg, $prims, 2, Cylindrical, Cylindrical),
            (3, "cylindrical") => build_and_run_iso!($cfg, $prims, 3, Cylindrical, Cylindrical),
            (d, g) => Err(format!("no isothermal dispatch arm for (dims={d}, coord={g}) yet")),
        }
    };
}

/// runtime dispatch on the config tags → a monomorphized sim. hydro regimes
/// (newtonian/srhd/isothermal) x cartesian (+ curvilinear for adiabatic) x 1/2/3d;
/// the mhd regimes (srmhd/nmhd/imhd) x cartesian x 1/2/3d.
fn dispatch_and_run(cfg: &Config, prims: &[Vec<f64>], bfields: &[Vec<f64>]) -> Result<(), String> {
    // static mesh refinement is wired for hydro (incl. globally-isothermal). the
    // two cases still pending need extra fine-level prolongation:
    if cfg.refinement_enabled
        && cfg.regime.contains("mhd")
        && !(cfg.dims == 3 && cfg.coord_system == "cartesian")
    {
        return Err("mhd refinement requires a 3d cartesian grid (the CT \
                    reflux assumes 1/dx curl coefficients)".to_string());
    }
    // mesh motion is single-grid uniform-spacing hydro only in this pass.
    if cfg.mesh_motion {
        if cfg.refinement_enabled {
            return Err("mesh motion is single-grid only (not wired with refinement)".to_string());
        }
        if cfg.regime.contains("mhd") {
            return Err("mesh motion is not wired for MHD (comoving-field convention pending)".to_string());
        }
    }
    match cfg.regime.as_str() {
        "newtonian"  => hydro_dispatch!(cfg, prims, Newtonian, Newtonian),
        "srhd"       => hydro_dispatch!(cfg, prims, Srhd, Srhd),
        "isothermal" => iso_dispatch!(cfg, prims),
        "srmhd"      => mhd_dispatch!(cfg, prims, bfields, Rmhd, Rmhd),
        "nmhd"       => mhd_dispatch!(cfg, prims, bfields, NewtonianMhd, NewtonianMhd),
        "imhd"       => imhd_dispatch!(cfg, prims, bfields),
        other        => Err(format!("regime '{other}' not wired yet")),
    }
}

/// author the metadata the frozen v2.0 reader requires beyond what
/// `write_checkpoint` derives from SimState (it already writes gamma, cfl, time,
/// dt, iteration, dimensions, halo_radius, regime, coord_system, timestepping,
/// is_mhd, is_relativistic). these are the spatial-scheme + run-control fields,
/// authored from the python config dict (the single source of truth).
fn checkpoint_metadata(cfg: &Config, checkpoint_index: u64) -> Metadata {
    Metadata::new()
        .with("solver", cfg.solver_name.as_str())
        .with("reconstruction", cfg.reconstruction_name.as_str())
        .with("plm_theta", cfg.plm_theta)
        .with("viscosity", cfg.viscosity)
        .with("tend", cfg.t_final)
        .with("dlogt", cfg.dlogt)
        .with("checkpoint_index", checkpoint_index)
        .with("checkpoint_interval", cfg.checkpoint_interval)
        .with("x1_spacing", cfg.x1_spacing.as_str())
        .with("initial_time", cfg.start_time)
}

// =============================================================================
// the pybind11-compatible entry point
// =============================================================================

#[pyfunction]
#[pyo3(signature = (prim_gen, staggered_bfields, sim_info, a, adot))]
fn run_simulation(
    py: Python<'_>,
    prim_gen: &Bound<'_, PyAny>,
    staggered_bfields: &Bound<'_, PyAny>,
    sim_info: &Bound<'_, PyDict>,
    a: &Bound<'_, PyAny>,
    adot: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let mut cfg = parse_config(sim_info)?;
    // evaluate the scale-factor callables at the start time (GIL held). the rust
    // mesh-motion model integrates `a` from the constant rate `a_dot` (linear /
    // free homologous expansion, a_ddot = 0), so a single sample at t0 suffices.
    if cfg.mesh_motion {
        let t0 = cfg.start_time;
        cfg.scale_a0 = a.call1((t0,))?.extract::<f64>()?;
        cfg.scale_adot = adot.call1((t0,))?.extract::<f64>()?;
    }
    let prims = drain_prims(prim_gen)?;
    let bfields = drain_bfields(staggered_bfields)?;

    // the solve is pure rust with no python access — release the GIL so rayon
    // gets real parallelism (and python stays responsive).
    py.allow_threads(|| dispatch_and_run(&cfg, &prims, &bfields))
        .map_err(PyRuntimeError::new_err)
}

#[pymodule]
fn cpu_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    Ok(())
}
