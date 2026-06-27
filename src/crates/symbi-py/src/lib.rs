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

mod afterglow;

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
use symbi_sim::state::CtMethod;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::PrimG;
use symbi_geometry::MotionState;
use symbi_io::Metadata;
use symbi_sim::checkpoint::write_hierarchy_checkpoint;
use symbi_display::{ScreenGuard, SignalGuard, Table};
use symbi_ib::{Body, BodyCollection, BodyKind};

// =============================================================================
// parsed configuration — a plain-rust mirror of the python exec_dict. the
// monomorphized dispatch below reads these tags to pick the concrete SimState.
// =============================================================================

struct Config {
    name:                String,
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
    ct_method:           CtMethod,
    reconstruction_name: String,
    timestepping:        Timestepping,
    plm_theta:           f64,
    dlogt:               f64,
    viscosity:           f64,
    x1_spacing:          String,
    start_time:          f64,
    // the LOG-checkpoint anchor (positive reference for log-spaced cadence). distinct from
    // start_time, which is the physical/resume clock (= checkpoint time on restart). 0 = unset ->
    // fall back to start_time (the common case where they coincide).
    checkpoint_log_anchor: f64,
    checkpoint_index:    u64,
    t_final:             f64,
    checkpoint_interval: f64,
    data_dir:            String,
    // natural time unit for checkpoint names + display: reported time is
    // `time / time_unit`, labeled `time_unit_label` ("t" = code units).
    time_unit:           f64,
    time_unit_label:     String,
    // immersed bodies (gravity / accretion sinks) parsed from the config's
    // `immersed_bodies` list; empty for body-free runs. dimension-agnostic raw
    // form — the typed `BodyCollection<f64, D>` is built per-dim at sim build.
    bodies:              Vec<BodyParams>,
    // a single user source expression in the rust `SourceConfig` wire format
    // (json string), or None. lowered + attached on the hydro path.
    source_json:         Option<String>,
    // mesh-motion scale-factor law a(t)/a_dot(t) as the `serialize_motion` wire (json), or None.
    // when present the time loop evaluates it exactly each (sub)stage (no linearization).
    motion_json:         Option<String>,
    // driven (DYNAMIC) boundary prescriptions as `SourceConfig` json, in Driven-id order
    // (driven_exprs[id] <-> the face marked BoundaryType::Driven(id)). MHD path for now.
    driven_exprs:        Vec<String>,
    // body-diagnostic output cadence in natural units (× time_unit -> code);
    // 0 disables the diagnostics file.
    diagnostic_interval: f64,
}

/// dimension-agnostic raw body parameters from the python `immersed_bodies`
/// list. `capability` is the BodyCapability bitflag (GRAVITATIONAL=1,
/// ACCRETION=2). accretion fields are only meaningful when the ACCRETION bit
/// is set (a black-hole sink); otherwise the body is a fixed-potential mass.
struct BodyParams {
    capability:       u64,
    mass:             f64,
    radius:           f64,
    position:         Vec<f64>,
    velocity:         Vec<f64>,
    softening:        f64,
    accretion_radius: f64,
    sink_rate:        f64,
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

/// read a user source-expression field (already in the rust `SourceConfig` wire
/// format, emitted by python's `CompiledExpr.serialize_source`) and return it as a
/// json string ready for `SourceConfig::from_json`. an empty dict (the default for
/// configs with no source) -> None. the conversion goes through python's
/// `json.dumps`, so the node DAG crosses the boundary without a hand-written
/// PyDict -> serde walk.
fn get_source_json(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(obj) = dict.get_item(key)? else {
        return Ok(None);
    };
    // skip the empty-dict default (`return {}` in the base SimbiProblem).
    if let Ok(d) = obj.downcast::<PyDict>() {
        if d.is_empty() {
            return Ok(None);
        }
    }
    let json = obj.py().import("json")?;
    let s: String = json.call_method1("dumps", (obj,))?.extract()?;
    Ok(Some(s))
}

/// uniform runtime-source attach across the substrate kernel sets the hydro
/// dispatch macro instantiates. the macro body monomorphizes for EVERY regime it
/// covers (newtonian/adiabatic AND srhd), but `with_runtime_source` is inherent
/// only on the substrates that carry a source slot — so the call must go through
/// a trait that ALL of them implement. the relativistic set has no slot yet and
/// reports a clear error rather than failing to compile.
trait AttachRuntimeSource: Sized {
    fn attach_runtime_source(
        self,
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
        params: Vec<f64>,
    ) -> Result<Self, String>;
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    AttachRuntimeSource for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    fn attach_runtime_source(
        self,
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        Ok(self.with_runtime_source(built, params))
    }
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    AttachRuntimeSource for SrhdSubstrateKernelSet<Mem, Sc, D>
{
    fn attach_runtime_source(
        self,
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        Ok(self.with_runtime_source(built, params))
    }
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    AttachRuntimeSource for IsoSubstrateKernelSet<Mem, Sc, D>
{
    fn attach_runtime_source(
        self,
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        Ok(self.with_runtime_source(built, params))
    }
}

// the unified MHD kernel set is generic over the regime R, so this one impl covers
// the nmhd / imhd / rmhd aliases. sources target the hydro slots (den/mom/nrg); the
// per-regime kind validity (rmhd -> raw only) is enforced upstream by build_user_source.
impl<R, Mem, Sc, const D: usize> AttachRuntimeSource
    for symbi::regimes::substrate_mhd::MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace,
    Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric,
{
    fn attach_runtime_source(
        self,
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        Ok(self.with_runtime_source(built, params))
    }
}

fn solver_from_str(s: &str) -> PyResult<Solver> {
    match s {
        "hlle" => Ok(Solver::Hlle),
        "hllc" => Ok(Solver::Hllc),
        "hlld" => Ok(Solver::Hlld),
        other => Err(PyValueError::new_err(format!("unknown solver '{other}'"))),
    }
}

fn ct_method_from_str(s: &str) -> PyResult<CtMethod> {
    match s {
        "contact" => Ok(CtMethod::Contact),
        "uct" => Ok(CtMethod::Uct),
        other => Err(PyValueError::new_err(format!("unknown ct_method '{other}' (contact | uct)"))),
    }
}

fn timestepping_from_str(s: &str) -> PyResult<Timestepping> {
    match s {
        // rk1 is forward euler (first-order); the python `order=1` path emits it.
        "euler" | "rk1" => Ok(Timestepping::Euler),
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
    // per-face boundary-expression field names, in face order (2*axis + side): a `dynamic`
    // (DRIVEN) face reads its prescribed ghost state from the matching field.
    const BX_FIELDS: [&str; 6] = [
        "bx1_inner_expressions", "bx1_outer_expressions",
        "bx2_inner_expressions", "bx2_outer_expressions",
        "bx3_inner_expressions", "bx3_outer_expressions",
    ];
    let mut boundaries = Vec::new();
    // driven (DYNAMIC) boundary expressions in Driven-id order; id == registration order ==
    // the order faces are visited here, so `Driven(id)` on a face matches `driven_exprs[id]`.
    let mut driven_exprs: Vec<String> = Vec::new();
    for (face, obj) in bc_objs.try_iter()?.enumerate() {
        let obj = obj?;
        let s: String = match obj.getattr("value") {
            Ok(v) => v.extract()?,
            Err(_) => obj.extract()?,
        };
        match s.to_lowercase().as_str() {
            "dynamic" => {
                let field = BX_FIELDS.get(face).copied().unwrap_or("bx1_inner_expressions");
                let json = get_source_json(dict, field)?.ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "boundary face {face} is DYNAMIC but '{field}' is empty; \
                         a driven boundary needs a prescribed ghost state"
                    ))
                })?;
                let id = driven_exprs.len() as u16;
                driven_exprs.push(json);
                boundaries.push(BoundaryType::Driven(id));
            }
            other => boundaries.push(boundary_from_str(other)?),
        }
    }

    let gamma = dict
        .get_item("adiabatic_index")?
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(5.0 / 3.0);

    let solver_name = enum_str(dict, "solver")?;
    // constrained-transport edge-EMF scheme; defaults to contact for back-compat.
    let ct_method = ct_method_from_str(&enum_str_or(dict, "ct_method", "contact"))?;

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
        // the problem class name (preserve case); blank when not supplied.
        name: dict
            .get_item("name")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default(),
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
        ct_method,
        reconstruction_name: enum_str_or(dict, "reconstruction", "plm"),
        timestepping: timestepping_from_str(&enum_str(dict, "timestepping")?)?,
        plm_theta: get_f64_or(dict, "plm_theta", 1.5),
        dlogt: get_f64_or(dict, "dlogt", 0.0),
        viscosity: get_f64_or(dict, "viscosity", 0.0),
        x1_spacing: enum_str_or(dict, "x1_spacing", "linear"),
        start_time: get_f64_or(dict, "start_time", 0.0),
        checkpoint_log_anchor: get_f64_or(dict, "checkpoint_log_anchor", 0.0),
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
        time_unit: {
            let u = get_f64_or(dict, "time_unit", 1.0);
            if u > 0.0 { u } else { 1.0 }
        },
        time_unit_label: dict
            .get_item("time_unit_label")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "t".to_string()),
        bodies: parse_bodies(dict),
        diagnostic_interval: get_f64_or(dict, "diagnostic_interval", 0.0),
        // user source expressions (force/cooling/relax/raw) -> the rust source
        // front door. `gravity_source_expressions` is the conventional force slot;
        // `hydro_source_expressions` is the generic self-describing source. one
        // runtime source per run for now (the kernel set holds a single slot).
        source_json: get_source_json(dict, "gravity_source_expressions")?
            .or(get_source_json(dict, "hydro_source_expressions")?),
        motion_json: get_source_json(dict, "scale_factor_expressions")?,
        driven_exprs,
    })
}

/// parse the python `immersed_bodies` list (each a serialized ImmersedBodyConfig
/// dict) into dimension-agnostic `BodyParams`. missing / malformed entries are
/// skipped; a body-free config yields an empty vec.
fn parse_bodies(dict: &Bound<'_, PyDict>) -> Vec<BodyParams> {
    let Ok(Some(obj)) = dict.get_item("immersed_bodies") else {
        return Vec::new();
    };
    let Ok(list) = obj.extract::<Vec<Bound<'_, PyAny>>>() else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(list.len());
    for item in &list {
        let Ok(b) = item.downcast::<PyDict>() else { continue };
        let f = |k: &str| -> f64 {
            b.get_item(k).ok().flatten().and_then(|v| v.extract().ok()).unwrap_or(0.0)
        };
        let v = |k: &str| -> Vec<f64> {
            b.get_item(k).ok().flatten().and_then(|x| x.extract().ok()).unwrap_or_default()
        };
        let capability: u64 = b
            .get_item("capability")
            .ok()
            .flatten()
            .and_then(|x| x.extract().ok())
            .unwrap_or(1);
        let softening = sub_f64(b, "gravitational", "softening_length", 0.0);
        let accretion_radius = sub_f64(b, "accretion", "accretion_radius", 0.0);
        let sink_rate = sub_f64(b, "accretion", "sink_rate", 0.0);
        out.push(BodyParams {
            capability,
            mass: f("mass"),
            radius: f("radius"),
            position: v("position"),
            velocity: v("velocity"),
            softening,
            accretion_radius,
            sink_rate,
        });
    }
    out
}

/// read `body[group][key]` as f64 (the nested ImmersedBodyConfig sub-dicts like
/// `gravitational` / `accretion`), returning `default` when absent or null.
fn sub_f64(body: &Bound<'_, PyDict>, group: &str, key: &str, default: f64) -> f64 {
    body.get_item(group)
        .ok()
        .flatten()
        .and_then(|g| g.downcast::<PyDict>().ok().cloned())
        .and_then(|gd| gd.get_item(key).ok().flatten())
        .and_then(|val| val.extract().ok())
        .unwrap_or(default)
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
    // the checkpoint cadence is in NATURAL units: `checkpoint_interval * time_unit`
    // is the code-unit spacing, so `checkpoint_interval = 0.1` with a binary's
    // orbital `time_unit` means "every 0.1 orbits". default time_unit = 1.0 keeps
    // the cadence in code units, unchanged for ordinary runs.
    let cp_interval = if cfg.checkpoint_interval > 0.0 {
        cfg.checkpoint_interval * cfg.time_unit
    } else {
        f64::INFINITY
    };
    // LOGARITHMIC checkpoint spacing: when dlogt > 0 (the python config enabled
    // log_checkpoints over a positive start_time), the k-th checkpoint lands at
    // start_time*10^(k*dlogt) in code units — dense early, sparse late, the right
    // cadence for a run spanning many decades in time (a relativistic wind from a
    // tiny inner radius out to a huge one). otherwise the cadence is LINEAR at
    // cp_interval. `cp_at(fired)` returns the (fired+1)-th scheduled checkpoint time.
    // the log cadence is anchored at checkpoint_log_anchor (a fixed reference, e.g. the inner
    // light-crossing), NOT start_time — so the schedule is identical across a fresh run and a
    // restart whose clock resumes at the checkpoint time. unset (0) -> start_time (they coincide).
    let cp_anchor = if cfg.checkpoint_log_anchor > 0.0 { cfg.checkpoint_log_anchor } else { cfg.start_time };
    let cp_log = cfg.dlogt > 0.0 && cp_anchor > 0.0;
    let cp_tstart = cp_anchor;
    let cp_dlogt = cfg.dlogt;
    let cp_at = move |fired: u64| -> f64 {
        if cp_log {
            cp_tstart * 10f64.powf((fired + 1) as f64 * cp_dlogt)
        } else if cp_interval.is_finite() {
            (fired + 1) as f64 * cp_interval
        } else {
            f64::INFINITY
        }
    };
    let mut cp_fired: u64 = 0;
    let mut next_cp = cp_at(0);
    let mut cp_index: u64 = cfg.checkpoint_index + 1;
    // LOG-spaced runs are named by the monotonic INDEX, not the time: the fixed-3-decimal
    // time name (`000_790`) collides at small times (0.0001 and 0.0002 both round to
    // `000_000`, silently overwriting the dense early dumps a log run produces). the physical
    // time lives in metadata/time, which every reader uses. size the zero-pad width to the
    // projected checkpoint count (+ any restart offset) so names always sort lexicographically.
    let cp_idx_width: usize = if cp_log {
        // size the zero-pad TIGHTLY to the projected highest index (count + any restart offset).
        // `ceil(log10(max_index + 1))` is the digit count and is robust at the power-of-10 boundary
        // (a projection of 99.99 still yields 2, and exactly 1000 yields 4) so the seam never lands
        // on the run's own last checkpoint. an overshoot extends the width gracefully (format! never
        // truncates: width is a MINIMUM, so 99 -> 100); only a raw `ls` sees a cosmetic seam there,
        // since every reader sorts numerically (metadata/time, viz extract_timestep).
        let projected =
            (cfg.t_final / cp_tstart).log10() / cp_dlogt + cfg.checkpoint_index as f64;
        ((projected.max(1.0) + 1.0).log10().ceil() as usize).max(1)
    } else {
        0
    };

    // the live monitor. dynamic mode self-detects a tty (clearing redraw on a
    // terminal, plain appended frames when piped to a file). the row tracks the
    // root level's clock; checkpoint writes post to the message board.
    let t_final = cfg.t_final;
    let setup = problem_setup_rows(cfg);
    let setup_ref: Vec<[&str; 3]> =
        setup.iter().map(|r| [r[0].as_str(), r[1].as_str(), r[2].as_str()]).collect();
    let title = if cfg.name.is_empty() {
        "SIMBI".to_string()
    } else {
        format!("SIMBI  -  {}", cfg.name)
    };
    let mut table = Table::new(&title, true);
    table.set_problem_setup(&setup_ref);
    table.set_header(&["Iteration", "Time", "dt", "zone-cyc/s"]);
    if let Some(p) = log_path(cfg) {
        let _ = table.set_log_file(std::path::Path::new(&p));
    }

    // zone-cycle throughput: the interior cell count per root step. wall-clock
    // deltas across the row cadence give the instantaneous update rate; the run
    // total gives the average. this is the number a user watches.
    let n_zones: u64 = (0..cfg.dims).map(|ax| cfg.n_cells[ax] as u64).product();
    let cp_width = checkpoint_time_width(cfg);
    // body diagnostics: a separate, user-defined cadence (natural units), only
    // when the run has bodies and a positive interval. one `<dir>diagnostics.dat`
    // table for the whole run.
    let diag_path = if cfg.diagnostic_interval > 0.0 && !cfg.bodies.is_empty() {
        Some(format!("{data_dir}diagnostics.dat"))
    } else {
        None
    };
    let diag_interval = (cfg.diagnostic_interval * cfg.time_unit).max(f64::MIN_POSITIVE);
    let mut next_diag = diag_interval;
    let start = std::time::Instant::now();
    let mut last_inst = start;
    let mut last_iter = hier.levels[0].state.iteration;

    // graceful-interrupt trap: a caught signal (Ctrl-C, scheduler eviction)
    // flips `stop_requested`; we then snapshot a restart checkpoint and break.
    // Drop restores python's handlers + the cursor no matter how the run ends.
    let guard = SignalGuard::install();
    // btop-style live TUI: draw the dashboard in the alternate screen so it
    // leaves no scrollback trail; on exit we restore the primary buffer and
    // re-render one static final frame so the result persists.
    let mut screen = ScreenGuard::enter();

    // prime the IC: derive primitives (c2p) + cell-centered B (bcell-from-bface)
    // from the seeded conserved/face state BEFORE snapshotting, so the t=0
    // checkpoint carries real primitives instead of the zeroed scratch buffers
    // (the reader reads primitives — an unprimed IC plots as all zeros). idempotent
    // with the prime the evolve driver runs at its own start.
    hier.prime();

    // save the initial condition (t = 0) as the start-index checkpoint: a fresh
    // run (clock at zero, or index 0) writes its IC so the first output is the
    // un-evolved state. then render the opening frame.
    {
        let (i0, t0, d0) = {
            let r = &hier.levels[0].state;
            (r.iteration, r.time, r.dt)
        };
        if t0 == 0.0 || cfg.checkpoint_index == 0 {
            let states: Vec<&_> = hier.levels.iter().map(|l| &l.state).collect();
            let ic = checkpoint_name(cfg, &checkpoint_tag(cfg, cp_idx_width, cp_width, t0, cfg.checkpoint_index));
            match write_hierarchy_checkpoint(&states, &ic, &checkpoint_metadata(cfg, cfg.checkpoint_index)) {
                Ok(_) => table.post_success(&format!(
                    "checkpoint {ic}  ({}, initial condition)", fmt_time_msg(cfg, t0),
                )),
                Err(e) => table.post_error(&format!("initial checkpoint failed: {e:?}")),
            }
        }
        if let Some(dp) = &diag_path {
            if let Some(im) = hier.levels.last().and_then(|l| l.state.immersed.as_ref()) {
                let _ = append_diagnostics(dp, t0, &im.bodies);
                table.post_diagnostic(&format!("diagnostics {dp}  ({}, initial)", fmt_time_msg(cfg, t0)));
            }
        }
        set_row(&mut table, i0, t0, d0, t_final, 0.0);
        table.refresh();
    }

    hier.evolve_with_callback(cfg.t_final, 1, |h| {
        let st = &h.levels[0].state;
        let (iter, time, dt) = (st.iteration, st.time, st.dt);

        // signal observed: write a numbered + canonical restart checkpoint so a
        // cluster eviction can resume, then ask the march to stop. the handler
        // has already left the alternate screen, so switch the table to static
        // (no clearing redraw) before any further render of the primary buffer.
        if guard.stop_requested() {
            table.set_dynamic(false);
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let restart = checkpoint_name(cfg, "interrupted");
            let _ = write_hierarchy_checkpoint(&states, &restart, &checkpoint_metadata(cfg, cp_index));
            let _ = write_hierarchy_checkpoint(
                &states, &format!("{data_dir}final.h5"), &checkpoint_metadata(cfg, cp_index),
            );
            table.post_warning(&format!(
                "interrupted ({}) at {}, step {iter} — restart checkpoint {restart}",
                guard.signal_name(), fmt_time_msg(cfg, time),
            ));
            return std::ops::ControlFlow::Break(());
        }

        // MESSAGE BOARD cadence: checkpoints fire on the time schedule. a single
        // large dt can cross MULTIPLE interval boundaries (e.g. a cold-medium CFL
        // step, or a coarse cadence); write EXACTLY ONE checkpoint for the current
        // state and advance next_cp past every boundary it crossed. the skipped
        // intermediate states were never computed, and the file name is keyed by
        // the current time — looping would just re-write the SAME file N times and
        // spam the board with identical entries.
        let mut dirty = false;
        if time + 1e-12 >= next_cp && next_cp.is_finite() {
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let path = checkpoint_name(cfg, &checkpoint_tag(cfg, cp_idx_width, cp_width, time, cp_index));
            match write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cp_index)) {
                Ok(_) => table.post_success(&format!("checkpoint {path}  ({})", fmt_time_msg(cfg, time))),
                Err(e) => table.post_error(&format!("checkpoint {cp_index:04} failed: {e:?}")),
            }
            cp_index += 1;
            cp_fired += 1;
            next_cp = cp_at(cp_fired);
            // advance past every boundary this step crossed (log or linear) so a
            // single large dt yields ONE write, not N identical-named dumps.
            while time + 1e-12 >= next_cp && next_cp.is_finite() {
                cp_fired += 1;
                next_cp = cp_at(cp_fired);
            }
            dirty = true;
        }

        // DIAGNOSTICS cadence: sample body state on its own (finer) schedule,
        // independent of checkpoints — append a row per body to diagnostics.dat.
        // post ONE board line per callback that wrote (not per missed interval, so
        // a dt that spans several intervals collapses to a single notice) and mark
        // the frame dirty so the write is visible the moment it happens.
        if let Some(dp) = &diag_path {
            let mut wrote = false;
            while time + 1e-12 >= next_diag {
                if let Some(im) = h.levels.last().and_then(|l| l.state.immersed.as_ref()) {
                    let _ = append_diagnostics(dp, time, &im.bodies);
                }
                next_diag += diag_interval;
                wrote = true;
            }
            if wrote {
                table.post_diagnostic(&format!("diagnostics {dp}  ({})", fmt_time_msg(cfg, time)));
                dirty = true;
            }
        }

        // BENCHMARK ROW cadence: update the live row every 100 root iterations,
        // faithfully and INDEPENDENT of the checkpoint cadence — the table need
        // not move in lockstep with the message board. the rate is measured over
        // the elapsed 100-iteration window.
        if iter % 100 == 0 {
            let now = std::time::Instant::now();
            let elapsed = now.duration_since(last_inst).as_secs_f64();
            let d_iter = iter.saturating_sub(last_iter);
            let rate = if elapsed > 1e-9 && d_iter > 0 {
                n_zones as f64 * d_iter as f64 / elapsed
            } else {
                0.0
            };
            last_inst = now;
            last_iter = iter;
            set_row(&mut table, iter, time, dt, t_final, rate);
            dirty = true;
        }

        if dirty {
            table.refresh();
        }
        std::ops::ControlFlow::Continue(())
    })?;

    // interrupted: the restart checkpoint is already written and the alternate
    // screen is already torn down by the handler. surface the halt as one static
    // frame on the primary buffer (guard Drop restores python's handlers).
    if guard.stop_requested() {
        screen.leave();
        table.set_dynamic(false);
        let root = &hier.levels[0].state;
        table.post_warning(&format!(
            "run halted at t = {:.4} after {} steps", root.time, root.iteration,
        ));
        table.refresh();
        return Ok(());
    }

    let states: Vec<&_> = hier.levels.iter().map(|l| &l.state).collect();
    let final_path = format!("{data_dir}final.h5");
    write_hierarchy_checkpoint(&states, &final_path, &checkpoint_metadata(cfg, cp_index))?;
    let root = &hier.levels[0].state;
    let wall = start.elapsed().as_secs_f64();
    let avg = if wall > 1e-9 { n_zones as f64 * root.iteration as f64 / wall } else { 0.0 };
    // leave the alternate screen, then render ONE static final frame so the
    // completed dashboard persists on the primary buffer. post the summary
    // first so `draw_row`'s single refresh carries it.
    screen.leave();
    table.set_dynamic(false);
    table.post_success(&format!(
        "done — {} steps to t = {:.4} in {:.2}s (avg {} zone-cyc/s); final checkpoint {final_path}",
        root.iteration, root.time, wall, humanize_rate(avg),
    ));
    draw_row(&mut table, root.iteration, root.time, root.dt, t_final, avg);
    Ok(())
}

/// set the live monitor's benchmark row + progress bar WITHOUT rendering — the
/// caller decides when to refresh (the row cadence is decoupled from the message
/// board). takes primitives (not the generic state) so it is monomorphization-free.
fn set_row(table: &mut Table, iteration: u64, time: f64, dt: f64, t_final: f64, rate_zcps: f64) {
    let frac = (time / t_final).clamp(0.0, 1.0);
    table.update_row(&[
        &iteration.to_string(),
        &format!("{time:.6e}"),
        &format!("{dt:.3e}"),
        &humanize_rate(rate_zcps),
    ]);
    table.set_progress((frac * 100.0) as usize);
}

/// set the row + progress and render one frame (used at start + finalize).
fn draw_row(table: &mut Table, iteration: u64, time: f64, dt: f64, t_final: f64, rate_zcps: f64) {
    set_row(table, iteration, time, dt, t_final, rate_zcps);
    table.refresh();
}

/// si-suffix a zone-cycle rate (k/M/G) for compact display. zero/undefined
/// rate (first frame, no elapsed wall time yet) shows as a dash.
fn humanize_rate(r: f64) -> String {
    if r <= 0.0 {
        "—".to_string()
    } else if r >= 1e9 {
        format!("{:.2}G", r / 1e9)
    } else if r >= 1e6 {
        format!("{:.2}M", r / 1e6)
    } else if r >= 1e3 {
        format!("{:.2}k", r / 1e3)
    } else {
        format!("{r:.0}")
    }
}

/// the live-monitor "PROBLEM SETUP" sub-table rows (category, property, value).
/// the single source of truth for run identification — the parts of the run
/// summary a user actually watches: regime + eos, geometry + zone count, the
/// numerical scheme, boundaries, and the resource estimate.
/// the slope-limiter name for the PLM reconstruction, keyed on `plm_theta` (mirrors `plm_theta_gv`):
/// theta < 0 = van Leer; theta == 1 = minmod; theta == 2 = MC (monotonized central); otherwise the
/// theta-MC family with the compression value shown.
fn limiter_label(theta: f64) -> String {
    if theta < 0.0 {
        "van Leer".to_string()
    } else if (theta - 1.0).abs() < 1e-9 {
        "minmod".to_string()
    } else if (theta - 2.0).abs() < 1e-9 {
        "MC (monotonized central)".to_string()
    } else {
        format!("minmod-MC (theta = {theta:.2})")
    }
}

fn problem_setup_rows(cfg: &Config) -> Vec<[String; 3]> {
    let n_zones: u64 = (0..cfg.dims).map(|ax| cfg.n_cells[ax] as u64).product();
    let res = (0..cfg.dims)
        .map(|ax| cfg.n_cells[ax].to_string())
        .collect::<Vec<_>>()
        .join(" x ");
    // the run reports time in the natural unit. for code units (label "t") the
    // suffix is dropped; otherwise the cadence + t_final read in that unit (the
    // cadence is stored in natural units; t_final in code units -> /time_unit).
    let unit = &cfg.time_unit_label;
    let custom_unit = unit != "t" && cfg.time_unit != 1.0;
    let suffix = if custom_unit { format!(" {unit}") } else { String::new() };
    let cp = if cfg.checkpoint_interval > 0.0 {
        format!("{:.4}{suffix}", cfg.checkpoint_interval)
    } else {
        "final only".to_string()
    };
    let t_final_disp = format!("{:.4}{suffix}", cfg.t_final / cfg.time_unit);
    let mut rows = vec![
        ["Regime".into(), "type".into(), cfg.regime.clone()],
        ["Regime".into(), "eos".into(), eos_label(cfg)],
        ["Geometry".into(), "coords".into(), cfg.coord_system.clone()],
        ["Geometry".into(), "dimensions".into(), format!("{}D", cfg.dims)],
        ["Geometry".into(), "resolution".into(), format!("{res}  ({n_zones} zones)")],
        ["Geometry".into(), "boundaries".into(), boundary_label(cfg)],
        ["Scheme".into(), "solver".into(), cfg.solver_name.clone()],
        ["Scheme".into(), "reconstruction".into(), cfg.reconstruction_name.clone()],
        ["Scheme".into(), "timestepping".into(), timestepping_label(cfg.timestepping)],
        ["Scheme".into(), "cfl".into(), format!("{:.3}", cfg.cfl)],
        ["Run".into(), "t_final".into(), t_final_disp],
        ["Run".into(), "checkpoint dt".into(), cp],
        ["Run".into(), "est. memory".into(), format!("{:.3} GB", est_memory_gb(cfg))],
        ["Run".into(), "output".into(), cfg.data_dir.clone()],
    ];
    // for PLM (2nd-order) runs, name the slope limiter (from plm_theta) under the reconstruction
    // row. pcm (1st order) has no limiter, so the row is omitted there.
    if cfg.reconstruction_name == "plm" {
        if let Some(i) = rows.iter().position(|r| r[1] == "reconstruction") {
            rows.insert(i + 1, ["Scheme".into(), "limiter".into(), limiter_label(cfg.plm_theta)]);
        }
    }
    // document the time unit only when it is not plain code units.
    if custom_unit {
        rows.push(["Run".into(), "time unit".into(),
            format!("1 {unit} = {:.4} code", cfg.time_unit)]);
    }
    rows
}

/// equation-of-state one-liner: ideal gas carries gamma; isothermal regimes
/// carry the (global or position-dependent) sound speed instead.
fn eos_label(cfg: &Config) -> String {
    let isothermal = cfg.regime.contains("iso") || cfg.regime == "imhd";
    if cfg.locally_isothermal {
        "locally isothermal cs(x)".to_string()
    } else if isothermal {
        format!("isothermal (cs = {:.4})", cfg.cs)
    } else {
        format!("ideal gas (gamma = {:.4})", cfg.gamma)
    }
}

/// per-axis boundary tags joined for display (e.g. "reflecting | outflow").
fn boundary_label(cfg: &Config) -> String {
    if cfg.boundaries.is_empty() {
        "—".to_string()
    } else {
        cfg.boundaries
            .iter()
            .map(|b| format!("{b:?}").to_lowercase())
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

fn timestepping_label(ts: Timestepping) -> String {
    match ts {
        Timestepping::Euler => "euler (rk1)".to_string(),
        Timestepping::Rk2 => "rk2".to_string(),
        Timestepping::Rk3 => "rk3".to_string(),
    }
}

/// the working-set memory estimate (GB), mirroring the python rich summary:
/// (cons + prim + flux) buffers of `nvars` f64 per zone, plus the rk2 stage
/// copy. mhd carries 9 vars (rho, v3, B3, e, chi); hydro carries dims + 3.
fn est_memory_gb(cfg: &Config) -> f64 {
    let n_zones: f64 = (0..cfg.dims).map(|ax| cfg.n_cells[ax] as f64).product();
    let is_mhd = cfg.regime.contains("mhd");
    let nvars = if is_mhd { 9.0 } else { cfg.dims as f64 + 3.0 };
    let zbytes = 8.0 * nvars * n_zones;
    let (ncons, nprims, nfluxes) = (1.0, 1.0, cfg.dims as f64);
    let mut bytes = (ncons + nprims + nfluxes) * zbytes;
    if matches!(cfg.timestepping, Timestepping::Rk2) {
        bytes += 2.0 * ncons * zbytes;
    }
    bytes / 1024f64.powi(3)
}

/// the on-disk message log path: `<data_dir>simbi.log`. mirrors every posted
/// message so a redirected/headless run keeps an auditable record.
fn log_path(cfg: &Config) -> Option<String> {
    if cfg.data_dir.is_empty() {
        None
    } else {
        Some(format!("{}simbi.log", cfg.data_dir))
    }
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

/// the effective slope-limiter theta passed to the substrate. PLM uses the
/// config `plm_theta` (default 1.5, theta-MC limiter); PCM — i.e. first-order /
/// `order=1` — maps to theta = 0, which collapses minmod3 to a zero slope, so
/// the reconstruction degenerates to piecewise-constant. the substrate has no
/// separate PCM kernel; this IS how first-order space is selected.
fn build_theta(cfg: &Config) -> f64 {
    if cfg.reconstruction_name == "pcm" {
        0.0
    } else {
        cfg.plm_theta
    }
}

/// build a typed `BodyCollection<f64, D>` from the parsed params. an ACCRETION
/// body becomes a black-hole sink (gravity + accretion onto the body);
/// otherwise it is a fixed-potential gravitating mass. `sink_delta` (the sink
/// smoothing width) uses the example default of 1.0.
fn build_bodies<const D: usize>(params: &[BodyParams]) -> BodyCollection<f64, D> {
    const ACCRETION: u64 = 2;
    let mut coll = BodyCollection::new();
    for (idx, b) in params.iter().enumerate() {
        let pos = Tensor::new(std::array::from_fn(|ax| b.position.get(ax).copied().unwrap_or(0.0)));
        let vel = Tensor::new(std::array::from_fn(|ax| b.velocity.get(ax).copied().unwrap_or(0.0)));
        let body = if b.capability & ACCRETION != 0 {
            Body::black_hole(
                idx, pos, vel, b.mass, b.radius, b.softening, b.sink_rate, 1.0, b.accretion_radius,
            )
        } else {
            Body::gravitational(idx, pos, vel, b.mass, b.radius, b.softening)
        };
        coll = coll.add(body);
    }
    coll
}

/// append one diagnostics row PER BODY to a whitespace-separated table (with a
/// `#`-commented header on first write): the instantaneous body state sampled at
/// the diagnostic cadence — position, velocity, the gas reaction force + torque
/// (the body-feedback accumulator's per-step consolidation), mass, and for
/// black-hole sinks the cumulative accreted mass + instantaneous accretion rate.
/// a plain table (not hdf5): diagnostics are a flat scalar time series, trivially
/// loaded with `numpy.loadtxt`.
fn append_diagnostics<const D: usize>(
    path: &str,
    time: f64,
    bodies: &BodyCollection<f64, D>,
) -> std::io::Result<()> {
    use std::io::Write;
    let fresh = !std::path::Path::new(path).exists();
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    if fresh {
        writeln!(
            f,
            "# time body x y vx vy fx fy torque_z mass accreted_mass accretion_rate"
        )?;
    }
    let comp = |t: &Tensor<f64, D>, ax: usize| if ax < D { t[ax] } else { 0.0 };
    for bb in 0..bodies.len() {
        let b = bodies.get(bb);
        let (accreted, rate) = match b.kind {
            BodyKind::BlackHole { total_accreted_mass, accretion_rate, .. } => {
                (total_accreted_mass, accretion_rate)
            }
            _ => (0.0, 0.0),
        };
        writeln!(
            f,
            "{time:.8e} {bb} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {accreted:.8e} {rate:.8e}",
            comp(&b.position, 0), comp(&b.position, 1),
            comp(&b.velocity, 0), comp(&b.velocity, 1),
            comp(&b.force, 0), comp(&b.force, 1),
            b.torque[2], b.mass,
        )?;
    }
    Ok(())
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
        // the physical clock starts at start_time (t0): the IC is the state AT t0, and a moving-mesh
        // a(t) must be sampled at the physical time, not an elapsed-from-0 clock. (default 0 -> no
        // change for the common case.)
        sim.time = $cfg.start_time;
        // mesh motion lives on the (coarse) state — set before wrapping. static
        // for the common case; the gates above keep motion to single-grid hydro.
        sim.motion = motion_state($cfg);
        // expression-driven mesh motion: build the traced a(t)/a_dot(t) law (autodiff'd a_dot, FD
        // cross-checked) and seed the homologous motion from it at t0 = sim.time. the time loop then
        // evaluates a(t) EXACTLY each (sub)stage instead of the linear scale_a0/scale_adot.
        if let Some(ref mj) = $cfg.motion_json {
            let t0 = sim.time;
            let law = symbi_hydro::motion_law::MotionLaw::from_json(mj, t0, $cfg.t_final)
                .map_err(|e| format!("mesh motion: {e}"))?;
            sim.motion = symbi_geometry::MotionState::homologous(law.a_at(t0), law.adot_at(t0));
            sim.motion_law = Some(law);
        }
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

        // attach immersed bodies (gravity / accretion sinks) when the config
        // declares any; body-free runs keep the original sim untouched.
        let sim = if cfg.bodies.is_empty() {
            sim
        } else {
            sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
        };
        let theta = build_theta(cfg);
        let sub = sim.substrate().theta(theta).with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?;
        // attach a user source expression (force/cooling/relax/raw) when present.
        // lowered against THIS regime's spec via the source front door — the bridge
        // rejects force/cooling/relax on relativistic regimes (use raw). single-grid
        // only: fine levels would not see the source, so refuse the combination.
        let sub = match &cfg.source_json {
            Some(json) => {
                if cfg.refinement_enabled {
                    return Err("user source expressions are not yet supported with mesh refinement".to_string());
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg, <$regime_ty as Regime<f64, $d>>::SPEC,
                ).map_err(|e| format!("source expression lower: {e}"))?;
                sub.attach_runtime_source(built, scfg.params.clone())?
            }
            None => sub,
        };
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d,
            |s| s.substrate().theta(theta).with_solver(solver).expect("fine-level kernel set"));
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

        // attach immersed bodies (gravity / accretion sinks) when the config
        // declares any; body-free runs keep the original sim untouched.
        let sim = if cfg.bodies.is_empty() {
            sim
        } else {
            sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
        };
        let theta = build_theta(cfg);
        let sub = sim.substrate().theta(theta).with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?
            .ct_method(cfg.ct_method);
        // attach a user source expression to the MHD hydro slots (den/mom/nrg).
        // rmhd is relativistic -> only kind="raw"; nmhd takes force/cooling/relax.
        // B is CT-evolved, not a cell source. single-grid only.
        let sub = match &cfg.source_json {
            Some(json) => {
                if cfg.refinement_enabled {
                    return Err("user source expressions are not yet supported with mesh refinement".to_string());
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg, <$regime_ty as Regime<f64, $d>>::SPEC,
                ).map_err(|e| format!("source expression lower: {e}"))?;
                sub.attach_runtime_source(built, scfg.params.clone())?
            }
            None => sub,
        };
        // register DRIVEN (DYNAMIC) boundaries in Driven-id order so `Driven(id)` on a face
        // matches `driven_exprs[id]`. a complete prim prescription incl. the cell B (purely
        // toroidal: in-plane B = 0, out-of-plane B_phi injected). single-grid only.
        let mut sub = sub;
        if !cfg.driven_exprs.is_empty() && cfg.refinement_enabled {
            return Err("driven boundaries are not yet supported with mesh refinement".to_string());
        }
        for json in &cfg.driven_exprs {
            let bcfg = symbi_hydro::SourceConfig::from_json(json)
                .map_err(|e| format!("boundary expression parse: {e}"))?;
            let built = symbi_hydro::expr_bridge::build_boundary_dag(
                &bcfg, <$regime_ty as Regime<f64, $d>>::SPEC,
            ).map_err(|e| format!("boundary expression lower: {e}"))?;
            sub = sub.with_driven_boundary(built, bcfg.params.clone()).0;
        }
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d,
            |s| s.substrate().theta(theta).with_solver(solver).expect("fine-level kernel set"));
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

        // attach immersed bodies (gravity / accretion sinks) when the config
        // declares any; body-free runs keep the original sim untouched.
        let sim = if cfg.bodies.is_empty() {
            sim
        } else {
            sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
        };
        let theta = build_theta(cfg);
        let sub = sim.substrate().theta(theta).with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?
            .ct_method(cfg.ct_method);
        // attach a user source. iso MHD has no energy -> momentum-only force/relax,
        // raw den/mom (raw->nrg rejected); B is CT-evolved. single-grid only.
        let sub = match &cfg.source_json {
            Some(json) => {
                if cfg.refinement_enabled {
                    return Err("user source expressions are not yet supported with mesh refinement".to_string());
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg, <IsothermalMhd as Regime<f64, $d>>::SPEC,
                ).map_err(|e| format!("source expression lower: {e}"))?;
                sub.attach_runtime_source(built, scfg.params.clone())?
            }
            None => sub,
        };
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d,
            |s| s.substrate().theta(theta).with_solver(solver).expect("fine-level kernel set"));
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

        // attach immersed bodies (gravity / accretion sinks) when declared.
        let sim = if cfg.bodies.is_empty() {
            sim
        } else {
            sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
        };
        // iso is HLLE-only; the substrate front door gives the kernel-set directly.
        let theta = build_theta(cfg);
        let sub = sim.substrate().theta(theta);
        // attach a user source expression. iso has NO energy, so build_user_source
        // (against the iso spec) drops the energy overlay for force/relax and rejects
        // raw->nrg; den/mom sources work. single-grid only.
        let sub = match &cfg.source_json {
            Some(json) => {
                if cfg.refinement_enabled {
                    return Err("user source expressions are not yet supported with mesh refinement".to_string());
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg, <IsoNewtonian as Regime<f64, $d>>::SPEC,
                ).map_err(|e| format!("source expression lower: {e}"))?;
                sub.attach_runtime_source(built, scfg.params.clone())?
            }
            None => sub,
        };

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

        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| s.substrate().theta(theta));

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
    // immersed bodies attach to level 0; the AMR body sync (finest-owns-bodies)
    // is not wired through the binding yet, so refined body runs are rejected.
    if !cfg.bodies.is_empty() && cfg.refinement_enabled {
        return Err("immersed bodies are single-grid only in the binding \
                    (AMR body sync not wired yet)".to_string());
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
        .with("time_unit", cfg.time_unit)
        .with("time_unit_label", cfg.time_unit_label.as_str())
}

/// the integer-digit width for the time portion of a checkpoint name, sized
/// from `t_final / time_unit` so every file in a run shares the same width and
/// a directory listing sorts chronologically. minimum 3 (the default).
fn checkpoint_time_width(cfg: &Config) -> usize {
    let t_units = (cfg.t_final / cfg.time_unit).max(1.0);
    let digits = t_units.log10().floor() as usize + 1;
    digits.max(3)
}

/// insert underscores every 3 digits from the RIGHT of a zero-padded integer string,
/// as thousands separators: "001234" -> "001_234", "001234567" -> "001_234_567". `digits`
/// must already be padded to a multiple of 3 so every file in a run shares the grouping.
fn group_thousands(digits: &str) -> String {
    let n = digits.len();
    let mut out = String::with_capacity(n + n / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (n - i) % 3 == 0 {
            out.push('_');
        }
        out.push(ch);
    }
    out
}

/// the time portion of a checkpoint name: the sim time in the natural unit, decimal point
/// rendered as an underscore, fixed 3-digit fraction, and the INTEGER part thousand-grouped
/// so large times stay readable: t/unit = 1.0 -> "001_000", 0.5 -> "000_500", 1234.567 ->
/// "001_234_567". the LAST underscore group is always the fraction; earlier groups are the
/// integer. the integer is zero-padded to a multiple of 3 (sized from t_final) so a directory
/// listing still sorts chronologically. carries on the .9995+ rounding edge (frac never "1000").
fn format_sim_time(value: f64, int_width: usize) -> String {
    let v = value.max(0.0);
    let mut int_part = v.floor() as u64;
    let mut frac = ((v - v.floor()) * 1000.0).round() as u64;
    if frac >= 1000 {
        int_part += 1;
        frac -= 1000;
    }
    // pad the integer to a multiple of 3 so the thousand-grouping is uniform across the run.
    let grouped_width = (int_width + 2) / 3 * 3;
    let padded = format!("{int_part:0grouped_width$}");
    format!("{}_{frac:03}", group_thousands(&padded))
}

/// the `tnow` segment of a checkpoint name. LINEAR runs use the human-readable time
/// (`000_790`); LOG-spaced runs (`idx_width > 0`) use the zero-padded monotonic INDEX
/// (`00042`) instead, because the fixed-decimal time collides at small times. either way the
/// exact physical time is preserved in metadata/time (the source of truth for all readers).
fn checkpoint_tag(
    cfg: &Config, idx_width: usize, time_width: usize, time: f64, index: u64,
) -> String {
    if idx_width > 0 {
        format!("{index:0idx_width$}")
    } else {
        format_sim_time(time / cfg.time_unit, time_width)
    }
}

/// the full checkpoint path: `<dir><zones>.chkpt.<tnow>[.<unit>].h5`. `tnow` is
/// either a formatted time or a status word (interrupted / crashed). the unit
/// segment is appended only for a non-default time unit, so ordinary runs keep
/// the terse `<zones>.chkpt.<time>.h5` form (e.g. `262144.chkpt.000_500.h5`).
fn checkpoint_name(cfg: &Config, tnow: &str) -> String {
    let label = sanitize_unit_label(&cfg.time_unit_label);
    let unit = if label.is_empty() || label == "t" {
        String::new()
    } else {
        format!(".{label}")
    };
    format!("{}{}.chkpt.{tnow}{unit}.h5", cfg.data_dir, resolution_tag(cfg))
}

/// the per-axis resolution tag for a checkpoint name: the interior cell counts
/// joined by `x` (the standard resolution notation, distinct from the `_`
/// decimal in the time). e.g. 1d 100 -> "100", 2d 256x256 -> "256x256",
/// 3d 64x64x64 -> "64x64x64".
fn resolution_tag(cfg: &Config) -> String {
    (0..cfg.dims)
        .map(|ax| cfg.n_cells[ax].to_string())
        .collect::<Vec<_>>()
        .join("x")
}

/// make a user time-unit label safe as a filename segment: keep alphanumerics
/// and underscores, drop everything else (so `t_bondi`, `tjet`, `t/ff`, `t dyn`
/// all become valid path components). the RAW label is still used verbatim in
/// the live display, messages, and checkpoint metadata.
fn sanitize_unit_label(label: &str) -> String {
    label.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect()
}

/// the natural-unit time string for a checkpoint message: "t = 1.0000" in code
/// units, or "t = 0.5000 orbit" when a custom unit is set.
fn fmt_time_msg(cfg: &Config, time: f64) -> String {
    if cfg.time_unit_label == "t" || cfg.time_unit == 1.0 {
        format!("t = {time:.4}")
    } else {
        format!("t = {:.4} {}", time / cfg.time_unit, cfg.time_unit_label)
    }
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

// shared module body. the pyo3 entry-point name below decides the `PyInit_*`
// symbol and the imported module name: `cpu_ext` for the default build,
// `gpu_ext` for the cuda build. both compile the SAME source — cuda only adds
// the NVRTC device path — so the registration is identical and lives here.
fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    afterglow::register(m)?;
    Ok(())
}

// cpu build -> `simbi.libs.cpu_ext`.
#[cfg(not(feature = "cuda"))]
#[pymodule]
fn cpu_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)
}

// cuda build -> `simbi.libs.gpu_ext`. dev.py overrides maturin's module-name to
// match (`--config tool.maturin.module-name="simbi.libs.gpu_ext"`), so the two
// backends coexist instead of overwriting the same `cpu_ext` dylib.
#[cfg(feature = "cuda")]
#[pymodule]
fn gpu_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)
}
