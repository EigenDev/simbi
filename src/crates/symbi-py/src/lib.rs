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
use symbi::symbi_grid::Field;
use symbi_algebra::Tensor;
use symbi_display::{
    Colormap, ExitKind, FieldSlice, LiveDashboard, ScreenGuard, SignalGuard, Table,
};
use symbi_geometry::MotionState;
use symbi_geometry::Schwarzschild;
use symbi_geometry::{KerrKS, SchwarzschildKS, SchwarzschildKSCartesian, SchwarzschildKSCylindrical};
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Eos;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::PrimG;
use symbi_ib::{Body, BodyCollection, BodyKind};
use symbi_io::Metadata;
use symbi_sim::checkpoint::write_hierarchy_checkpoint;
use symbi_sim::state::SimStateGeneric;
use symbi_sim::state::CtMethod;

// =============================================================================
// parsed configuration — a plain-rust mirror of the python exec_dict. the
// monomorphized dispatch below reads these tags to pick the concrete SimState.
// =============================================================================

struct Config {
    name: String,
    regime: String,
    coord_system: String,
    // the spacetime background ("minkowski" default, "schwarzschild" for GR). ORTHOGONAL to
    // coord_system; selects the Schwarzschild metric (lapse/densitization/GR-wavespeed kernels).
    spacetime: String,
    // the Schwarzschild geometric mass M (G = c = 1); only meaningful when spacetime = schwarzschild.
    schwarzschild_mass: f64,
    kerr_spin: f64,
    max_dt: f64,
    cyl_plane: CylPlane,
    dims: usize,
    n_cells: [usize; 3],
    x_lo: [f64; 3],
    dx: [f64; 3],
    boundaries: Vec<BoundaryType>,
    cfl: f64,
    gamma: f64,
    cs: f64,
    locally_isothermal: bool,
    // write a read-only live snapshot to `<data_dir>/.simbi-live/snapshot.bin`
    // each diagnostic cadence, so `simbi attach <data_dir>` can monitor a headless
    // (batch/cluster) run over a shared filesystem.
    live_monitor: bool,
    refinement_enabled: bool,
    // each region is a flat [lo_0, hi_0, lo_1, hi_1, ..] bound list (2 per axis).
    refinement_regions: Vec<Vec<f64>>,
    // homologous / translating mesh motion (linear: a_ddot = 0). a0/adot are the
    // scale-factor callables evaluated at start_time (set in run_simulation).
    mesh_motion: bool,
    is_homologous: bool,
    scale_a0: f64,
    scale_adot: f64,
    solver: Solver,
    solver_name: String,
    ct_method: CtMethod,
    reconstruction_name: String,
    timestepping: Timestepping,
    plm_theta: f64,
    dlogt: f64,
    viscosity: f64,
    x1_spacing: String,
    start_time: f64,
    // the LOG-checkpoint anchor (positive reference for log-spaced cadence). distinct from
    // start_time, which is the physical/resume clock (= checkpoint time on restart). 0 = unset ->
    // fall back to start_time (the common case where they coincide).
    checkpoint_log_anchor: f64,
    checkpoint_index: u64,
    t_final: f64,
    checkpoint_interval: f64,
    data_dir: String,
    // natural time unit for checkpoint names + display: reported time is
    // `time / time_unit`, labeled `time_unit_label` ("t" = code units).
    time_unit: f64,
    time_unit_label: String,
    // immersed bodies (gravity / accretion sinks) parsed from the config's
    // `immersed_bodies` list; empty for body-free runs. dimension-agnostic raw
    // form — the typed `BodyCollection<f64, D>` is built per-dim at sim build.
    bodies: Vec<BodyParams>,
    // a single user source expression in the rust `SourceConfig` wire format
    // (json string), or None. lowered + attached on the hydro path.
    source_json: Option<String>,
    // mesh-motion scale-factor law a(t)/a_dot(t) as the `serialize_motion` wire (json), or None.
    // when present the time loop evaluates it exactly each (sub)stage (no linearization).
    motion_json: Option<String>,
    // driven (DYNAMIC) boundary prescriptions as `SourceConfig` json, in Driven-id order
    // (driven_exprs[id] <-> the face marked BoundaryType::Driven(id)). MHD path only.
    driven_exprs: Vec<String>,
    // body-diagnostic output cadence in natural units (× time_unit -> code);
    // 0 disables the diagnostics file.
    diagnostic_interval: f64,
    // number of gpus to decompose the domain across, intra-node (docs/design/37, 38). 1 =
    // single device (the only path wired today). >1 is validated here but the decomposed run
    // loop (M4) is not yet wired, so it errors -- the runtime knob, not the build backend.
    n_gpus: usize,
}

/// dimension-agnostic raw body parameters from the python `immersed_bodies`
/// list. `capability` is the BodyCapability bitflag (GRAVITATIONAL=1,
/// ACCRETION=2). accretion fields are only meaningful when the ACCRETION bit
/// is set (a black-hole sink); otherwise the body is a fixed-potential mass.
struct BodyParams {
    capability: u64,
    mass: f64,
    radius: f64,
    position: Vec<f64>,
    velocity: Vec<f64>,
    softening: f64,
    accretion_radius: f64,
    sink_rate: f64,
}

// =============================================================================
// dict extraction helpers
// =============================================================================

/// read a python enum field as its lowercase `.value` string; falls back to the
/// raw value when the object is already a plain string (e.g., `regime`).
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
    s.map(|s| s.to_lowercase())
        .unwrap_or_else(|| default.to_string())
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
/// covers (newtonian/adiabatic AND rhd), but `with_runtime_source` is inherent
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
    AttachRuntimeSource for RhdSubstrateKernelSet<Mem, Sc, D>
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
        // fleischmann (2020) low-mach / low-dissipation HLLC (newtonian only).
        "hllc_lm" | "hllc-lm" => Ok(Solver::HllcLm),
        "hlld" => Ok(Solver::Hlld),
        other => Err(PyValueError::new_err(format!("unknown solver '{other}'"))),
    }
}

fn ct_method_from_str(s: &str) -> PyResult<CtMethod> {
    match s {
        "contact" => Ok(CtMethod::Contact),
        "uct" => Ok(CtMethod::Uct),
        other => Err(PyValueError::new_err(format!(
            "unknown ct_method '{other}' (contact | uct)"
        ))),
    }
}

fn timestepping_from_str(s: &str) -> PyResult<Timestepping> {
    match s {
        // rk1 is forward euler (first-order); the python `order=1` path emits it.
        "euler" | "rk1" => Ok(Timestepping::Euler),
        "rk2" => Ok(Timestepping::Rk2),
        "rk3" => Ok(Timestepping::Rk3),
        other => Err(PyValueError::new_err(format!(
            "unknown timestepping '{other}'"
        ))),
    }
}

fn boundary_from_str(s: &str) -> PyResult<BoundaryType> {
    match s {
        "periodic" => Ok(BoundaryType::Periodic),
        "outflow" => Ok(BoundaryType::Outflow),
        "reflecting" | "reflect" => Ok(BoundaryType::Reflect),
        other => Err(PyValueError::new_err(format!(
            "unsupported boundary '{other}'"
        ))),
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
            dx[ii] = if n_cells[ii] > 0 {
                (hi - lo) / n_cells[ii] as f64
            } else {
                1.0
            };
        }
    }

    // boundary_conditions is a flat list (lo, hi per axis); map each.
    let bc_objs = dict
        .get_item("boundary_conditions")?
        .ok_or_else(|| PyValueError::new_err("sim_info missing 'boundary_conditions'"))?;
    // per-face boundary-expression field names, in face order (2*axis + side): a `dynamic`
    // (DRIVEN) face reads its prescribed ghost state from the matching field.
    const BX_FIELDS: [&str; 6] = [
        "bx1_inner_expressions",
        "bx1_outer_expressions",
        "bx2_inner_expressions",
        "bx2_outer_expressions",
        "bx3_inner_expressions",
        "bx3_outer_expressions",
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
                let field = BX_FIELDS
                    .get(face)
                    .copied()
                    .unwrap_or("bx1_inner_expressions");
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
        // the spacetime background; flat ("minkowski") unless the config opts into GR.
        spacetime: dict
            .get_item("spacetime")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .map(|s| s.to_lowercase())
            .unwrap_or_else(|| "minkowski".to_string()),
        schwarzschild_mass: get_f64_or(dict, "schwarzschild_mass", 0.0),
        kerr_spin: get_f64_or(dict, "kerr_spin", 0.0),
        max_dt: get_f64_or(dict, "max_dt", 0.0),
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
        live_monitor: dict
            .get_item("live_monitor")
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
        n_gpus: dict
            .get_item("gpus")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(1)
            .max(1),
        // user source expressions (force/cooling/relax/raw) -> the rust source
        // front door. `gravity_source_expressions` is the conventional force slot;
        // `hydro_source_expressions` is the generic self-describing source. one
        // one runtime source per run (the kernel set holds a single slot).
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
        let Ok(b) = item.downcast::<PyDict>() else {
            continue;
        };
        let f = |k: &str| -> f64 {
            b.get_item(k)
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .unwrap_or(0.0)
        };
        let v = |k: &str| -> Vec<f64> {
            b.get_item(k)
                .ok()
                .flatten()
                .and_then(|x| x.extract().ok())
                .unwrap_or_default()
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
    cfg: &Config,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
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
    // the log cadence is anchored at checkpoint_log_anchor (a fixed reference, e.g., the inner
    // light-crossing), NOT start_time — so the schedule is identical across a fresh run and a
    // restart whose clock resumes at the checkpoint time. unset (0) -> start_time (they coincide).
    let cp_anchor = if cfg.checkpoint_log_anchor > 0.0 {
        cfg.checkpoint_log_anchor
    } else {
        cfg.start_time
    };
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
        let projected = (cfg.t_final / cp_tstart).log10() / cp_dlogt + cfg.checkpoint_index as f64;
        ((projected.max(1.0) + 1.0).log10().ceil() as usize).max(1)
    } else {
        0
    };

    // the live monitor. dynamic mode self-detects a tty (clearing redraw on a
    // terminal, plain appended frames when piped to a file). the row tracks the
    // root level's clock; checkpoint writes post to the message board.
    let t_final = cfg.t_final;
    let setup = problem_setup_rows(cfg);
    let setup_ref: Vec<[&str; 3]> = setup
        .iter()
        .map(|r| [r[0].as_str(), r[1].as_str(), r[2].as_str()])
        .collect();
    let title = if cfg.name.is_empty() {
        "SIMBI".to_string()
    } else {
        format!("SIMBI  -  {}", cfg.name)
    };
    let mut table = Table::new(&title, true);
    // dim title-bar subtitle: regime and the base-grid zone count.
    let base_zones: u64 = (0..cfg.dims).map(|ax| cfg.n_cells[ax] as u64).product();
    table.set_subtitle(&format!("{} · {} zones", cfg.regime, base_zones));
    // live tabbed-dashboard statics: the regime badge and the cfl gauge cap.
    table.set_regime(&cfg.regime.to_uppercase());
    table.set_cfl(cfg.cfl, 1.0);
    // number of fields the `f`-key can cycle (density + pressure / W / |B|).
    table.set_field_count(hier.levels[0].state.field_count());
    table.set_problem_setup(&setup_ref);
    table.set_header(&["Iteration", "Time", "dt", "zone-cyc/s"]);
    if let Some(p) = log_path(cfg) {
        let _ = table.set_log_file(std::path::Path::new(&p));
    }

    // zone-cycle throughput: the ACTUAL interior cell-updates per ROOT step. for a refined run this
    // is NOT just the base grid -- each level ll subcycles RATIO^ll times per root step over its own
    // (finer, larger) interior, so the honest count is sum_ll (interior_cells_ll * RATIO^ll). a
    // single-level run reduces to the base interior, unchanged. without this, AMR reports only the
    // coarse zones while the wall-clock includes all the hidden fine work, so the rate reads ~RATIO^d
    // too low. (RATIO = 2, the baked transfer ratio.)
    let n_zones: u64 = {
        let mut eff = 0u64;
        let mut subcycle = 1u64; // RATIO^ll for level ll
        for level in hier.levels.iter() {
            let cells: u64 = (0..cfg.dims)
                .map(|ax| level.state.geom.interior.spaces[ax].size() as u64)
                .product();
            eff += cells * subcycle;
            subcycle *= 2;
        }
        eff
    };
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
    // FOFC observability: zero the deliberate-fallback counters at run start, track the last-seen
    // totals so the benchmark cadence can post the per-window delta (a limiter that fired is shown,
    // never silent). cumulative totals also close out the run summary.
    symbi::regimes::fofc::fofc_reset_stats();
    let mut last_fofc: (u64, u64) = (0, 0);

    // graceful-interrupt trap: a caught signal (Ctrl-C, scheduler eviction)
    // flips `stop_requested`; the loop then snapshots a restart checkpoint and breaks.
    // Drop restores python's handlers + the cursor no matter how the run ends.
    let guard = SignalGuard::install();
    // btop-style live TUI: draw the dashboard in the alternate screen so it
    // leaves no scrollback trail; on exit the primary buffer is restored and
    // re-render one static final frame so the result persists.
    let mut screen = ScreenGuard::enter();
    // tier 2a: a render thread owns the terminal + input and draws at ~30 fps, so
    // tab / pause respond instantly regardless of step rate. `None` off a tty (the
    // static string path renders headless). the solver publishes snapshots + reads
    // its control flags rather than polling keys inline.
    let mut dash = LiveDashboard::spawn();

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
            let ic = checkpoint_name(
                cfg,
                &checkpoint_tag(cfg, cp_idx_width, cp_width, t0, cfg.checkpoint_index),
            );
            match write_hierarchy_checkpoint(
                &states,
                &ic,
                &checkpoint_metadata(cfg, cfg.checkpoint_index),
            ) {
                Ok(_) => table.post_success(&format!(
                    "checkpoint {ic}  ({}, initial condition)",
                    fmt_time_msg(cfg, t0),
                )),
                Err(e) => table.post_error(&format!("initial checkpoint failed: {e:?}")),
            }
        }
        if let Some(dp) = &diag_path {
            if let Some(im) = hier.levels.last().and_then(|l| l.state.immersed.as_ref()) {
                let _ = append_diagnostics(dp, t0, &im.bodies);
                table.post_diagnostic(&format!(
                    "diagnostics {dp}  ({}, initial)",
                    fmt_time_msg(cfg, t0)
                ));
            }
        }
        set_row(&mut table, i0, t0, d0, t_final, 0.0);
        publish_or_refresh(dash.as_ref(), &mut table);
    }

    // live-dashboard control flags, toggled by keypresses inside the callback.
    // `paused` parks the integrator (no step) while keeping the ui live; `user_quit`
    // requests a graceful stop, handled exactly like a caught signal.
    let mut user_quit = false;

    hier.evolve_with_callback(cfg.t_final, 1, |h| {
        let st = &h.levels[0].state;
        let (iter, time, dt) = (st.iteration, st.time, st.dt);
        let mut dirty = false;

        // live-dashboard input (tier 2a): the render thread owns the keys + sets
        // control flags; the solver only READS them. space parks the integrator
        // here (the render thread keeps drawing); q -> graceful quit; s -> single
        // step; w -> force checkpoint. Ctrl-C is still a SIGINT to the guard.
        let mut want_cp = false;
        if let Some(d) = dash.as_ref() {
            let c = d.controls();
            if c.quit() {
                user_quit = true;
            }
            if c.take_checkpoint() {
                want_cp = true;
            }
            // park while paused (no step); the render thread keeps the ui alive.
            while c.paused() && !user_quit && !guard.stop_requested() {
                if c.quit() {
                    user_quit = true;
                    break;
                }
                if c.take_step() {
                    break;
                }
                if c.take_checkpoint() {
                    want_cp = true;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        }

        // manual checkpoint (w): write the current state immediately, off-schedule.
        if want_cp {
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let path =
                checkpoint_name(cfg, &checkpoint_tag(cfg, cp_idx_width, cp_width, time, cp_index));
            match write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cp_index)) {
                Ok(_) => table.post_success(&format!(
                    "checkpoint {path}  ({}, manual)",
                    fmt_time_msg(cfg, time)
                )),
                Err(e) => table.post_error(&format!("manual checkpoint failed: {e:?}")),
            }
            cp_index += 1;
            dirty = true;
        }

        // interrupt: a caught signal OR a user 'q'. write a numbered + canonical
        // restart checkpoint so a cluster eviction / quit can resume, then stop.
        // the handler has already left the alternate screen on a signal, so switch
        // the table to static before any further render of the primary buffer.
        if guard.stop_requested() || user_quit {
            table.set_dynamic(false);
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let restart = checkpoint_name(cfg, "interrupted");
            let _ =
                write_hierarchy_checkpoint(&states, &restart, &checkpoint_metadata(cfg, cp_index));
            let _ = write_hierarchy_checkpoint(
                &states,
                &checkpoint_name(cfg, "final"),
                &checkpoint_metadata(cfg, cp_index),
            );
            let reason = if user_quit {
                "quit".to_string()
            } else {
                guard.signal_name().to_string()
            };
            table.post_warning(&format!(
                "interrupted ({reason}) at {}, step {iter} — restart checkpoint {restart}",
                fmt_time_msg(cfg, time),
            ));
            return std::ops::ControlFlow::Break(());
        }

        // fatal cfl crash (set by the evolve loop when the wave speed went NaN / collapsed — an
        // unphysical c2p, e.g. V -> 1 at the inner boundary): snapshot the LAST computed state as a
        // `.crashed` checkpoint (+ the `.final` snapshot) so it can be inspected, then stop. mirrors the interrupt
        // path; the post-loop renders it as a crash, not a success.
        if let Some(c) = h.crash {
            table.set_dynamic(false);
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let crashed = checkpoint_name(cfg, "crashed");
            let _ =
                write_hierarchy_checkpoint(&states, &crashed, &checkpoint_metadata(cfg, cp_index));
            let _ = write_hierarchy_checkpoint(
                &states,
                &checkpoint_name(cfg, "final"),
                &checkpoint_metadata(cfg, cp_index),
            );
            table.post_error(&format!(
                "crashed at {} (step {}) — state checkpoint {crashed}",
                fmt_time_msg(cfg, c.time),
                c.iter,
            ));
            return std::ops::ControlFlow::Break(());
        }

        // MESSAGE BOARD cadence: checkpoints fire on the time schedule. a single
        // large dt can cross MULTIPLE interval boundaries (e.g., a cold-medium CFL
        // step, or a coarse cadence); write EXACTLY ONE checkpoint for the current
        // state and advance next_cp past every boundary it crossed. the skipped
        // intermediate states were never computed, and the file name is keyed by
        // the current time — looping would just re-write the SAME file N times and
        // spam the board with identical entries.
        if time + 1e-12 >= next_cp && next_cp.is_finite() {
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let path = checkpoint_name(
                cfg,
                &checkpoint_tag(cfg, cp_idx_width, cp_width, time, cp_index),
            );
            match write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cp_index)) {
                Ok(_) => {
                    table.post_success(&format!("checkpoint {path}  ({})", fmt_time_msg(cfg, time)))
                }
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
            // feed the live throughput chart with this window's instantaneous rate
            // (not the opening zero or the finalize cumulative average).
            table.push_throughput(rate);
            // live tabbed-dashboard metrics + per-level interior cell counts (the
            // amr hierarchy may have regridded since the last update).
            table.push_metrics(iter, time, dt, rate);
            // FOFC observable: surface the deliberate fallbacks that fired since the last window so a
            // limiter is visible rather than silent (a bounded, high-order-preserving first-order
            // correction is expected; a freeze — where neither order recovered a cell — is rarer and
            // worth flagging). only posts when something fired, so a clean run stays quiet.
            let (fb_total, fz_total) = symbi::regimes::fofc::fofc_stats();
            let (d_fb, d_fz) = (fb_total - last_fofc.0, fz_total - last_fofc.1);
            if d_fb > 0 || d_fz > 0 {
                table.post_diagnostic(&format!(
                    "FOFC: {d_fb} first-order fallback cell-steps{} since last window",
                    if d_fz > 0 { format!(", {d_fz} freezes") } else { String::new() },
                ));
                last_fofc = (fb_total, fz_total);
                dirty = true;
            }
            let blocks: Vec<u64> = h
                .levels
                .iter()
                .map(|l| {
                    (0..cfg.dims)
                        .map(|ax| l.state.geom.interior.spaces[ax].size() as u64)
                        .product()
                })
                .collect();
            table.set_blocks_per_level(&blocks);
            // conservation drift + div·B: a host-side interior reduction (skipped
            // on device-resident gpu runs). cheap once per benchmark cadence.
            if let Some(cd) = st.conservation_diag() {
                table.push_conservation(cd.mass, cd.energy, cd.div_b, cd.max_w);
            }
            // machine card: this (compute) node's hostname / cores and the run's
            // resident memory vs the node's physical ram. rss grows, so re-sample
            // each cadence; an attach client reads the compute node, not its own.
            table.set_host(Some(symbi_display::hostinfo::HostStats::sample()));
            // live field heatmap: a screen-sized decimated density slice (2D/3D-mid;
            // None for 1D or device runs), compositing the nested refinement levels
            // so the refined region shows its fine detail. cost bounded by the
            // ~200-cell cap, not the grid size.
            // the `f`-key selects which field to decimate (density / pressure / W /
            // |B|); composite for amr, single-grid fallback for 1D. the render thread
            // owns the colormap, so Inferno here is just a default it overrides.
            let idx = dash.as_ref().map(|d| d.controls().field_kind()).unwrap_or(0);
            // decimate field `kk` (composite over refinement levels, single-grid
            // fallback for 1D) into a display FieldSlice; the render thread owns the
            // colormap, so Inferno here is just a default it overrides.
            let make_slice = |kk: usize| {
                h.field_slice_composite(200, kk)
                    .or_else(|| h.levels[0].state.field_slice(200, kk))
                    .map(|fd| {
                        let label = if cfg.dims >= 3 {
                            format!("{} · z-slice", fd.name)
                        } else {
                            fd.name
                        };
                        FieldSlice {
                            label,
                            width: fd.width,
                            height: fd.height,
                            data: fd.data,
                            vmin: fd.vmin,
                            vmax: fd.vmax,
                            cmap: Colormap::Inferno,
                        }
                    })
            };
            if cfg.live_monitor {
                // build the full field bundle so `simbi attach` can switch fields
                // client-side; the local TUI shows the f-key-selected one. write the
                // read-only snapshot atomically (best-effort — a write failure must
                // never halt the run).
                let bundle: Vec<FieldSlice> =
                    (0..h.levels[0].state.field_count()).filter_map(make_slice).collect();
                if let Some(sel) = bundle.get(idx.min(bundle.len().saturating_sub(1))) {
                    table.set_field(Some(sel.clone()));
                }
                let mut view = table.diagnostic_view();
                view.field_count = bundle.len();
                let _ = symbi_display::snapshot::Snapshot {
                    view,
                    fields: bundle,
                }
                .write_atomic(std::path::Path::new(&cfg.data_dir));
            } else if let Some(sel) = make_slice(idx) {
                table.set_field(Some(sel));
            }
            dirty = true;
        }

        if dirty {
            publish_or_refresh(dash.as_ref(), &mut table);
        }
        std::ops::ControlFlow::Continue(())
    })?;

    // the run loop is done: stop + join the render thread so the main thread has
    // sole terminal ownership for leaving the alt screen + printing the exit frame.
    if let Some(d) = dash.as_mut() {
        d.shutdown();
    }

    // the run has ended (any exit path below): drop the live-monitor snapshot so it
    // does not outlive the run. a still-attached client keeps its last frame.
    if cfg.live_monitor {
        let _ = symbi_display::snapshot::cleanup(std::path::Path::new(&cfg.data_dir));
    }

    // interrupted (a caught signal or a user 'q'): the restart checkpoint is
    // already written. surface the halt as the amber interrupt exit frame on the
    // primary buffer (guard Drop restores python's handlers).
    if guard.stop_requested() || user_quit {
        screen.leave();
        table.set_dynamic(false);
        let root = &hier.levels[0].state;
        let restart = checkpoint_name(cfg, "interrupted");
        let summary = format!(
            "interrupted — {} steps, t = {:.4} · restart {restart}",
            root.iteration, root.time,
        );
        table.post_warning(&summary);
        table.exit_frame(ExitKind::Interrupt, &summary);
        return Ok(());
    }

    // crashed: the observer already snapshotted the `.crashed` + `.final` state. surface the halt as
    // the red crash exit frame, not a success.
    if let Some(c) = hier.crash {
        screen.leave();
        table.set_dynamic(false);
        let crashed = checkpoint_name(cfg, "crashed");
        let summary = format!(
            "crashed — {} steps, t = {:.4} — wave speed collapsed (unphysical c2p near a boundary) · state {crashed}",
            c.iter, c.time,
        );
        table.post_error(&summary);
        table.exit_frame(ExitKind::Crash, &summary);
        return Ok(());
    }

    let states: Vec<&_> = hier.levels.iter().map(|l| &l.state).collect();
    let final_path = checkpoint_name(cfg, "final");
    write_hierarchy_checkpoint(&states, &final_path, &checkpoint_metadata(cfg, cp_index))?;
    let root = &hier.levels[0].state;
    let wall = start.elapsed().as_secs_f64();
    let avg = if wall > 1e-9 {
        n_zones as f64 * root.iteration as f64 / wall
    } else {
        0.0
    };
    // leave the alternate screen, then render the green success exit frame so the
    // run's summary persists on the primary buffer.
    screen.leave();
    table.set_dynamic(false);
    let summary = format!(
        "complete — {} steps, t = {:.4}, {:.2}s, {}/s · final {final_path}",
        root.iteration,
        root.time,
        wall,
        humanize_rate(avg),
    );
    table.post_success(&summary);
    // FOFC run total: report the deliberate fallbacks over the whole run (a quiet run shows nothing).
    let (fb_total, fz_total) = symbi::regimes::fofc::fofc_stats();
    if fb_total > 0 || fz_total > 0 {
        table.post_diagnostic(&format!(
            "FOFC total: {fb_total} first-order fallback cell-steps, {fz_total} freezes"
        ));
    }
    table.exit_frame(ExitKind::Success, &summary);
    dump_profile_if_enabled(root.iteration, n_zones);
    Ok(())
}

/// dump the accumulated per-phase wall-time profile to stderr when `SYMBI_PROFILE` is set (the
/// `prof()` accumulator is otherwise a no-op). mirrors the zone-cycle bench: per phase the wall ms,
/// its share of the instrumented total, and ns/zone-cycle (normalized by steps * interior zones),
/// slowest first. empty (no-op) when profiling is off.
fn dump_profile_if_enabled(steps: u64, n_zones: u64) {
    let mut rows = symbi::sim::evolve::report_profile();
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let total: f64 = rows.iter().map(|(_, ms)| ms).sum();
    let zc = (steps as f64) * (n_zones as f64);
    eprintln!("\n--- per-phase wall time over {steps} steps (SYMBI_PROFILE) ---");
    for (name, ms) in &rows {
        let ns_zc = if zc > 0.0 { ms * 1e6 / zc } else { 0.0 };
        eprintln!(
            "  {name:<18} {ms:>8.1} ms  ({:>4.1}%)   {ns_zc:.0} ns/zone-cycle",
            100.0 * ms / total
        );
    }
    eprintln!("  {:<18} {total:>8.1} ms  (sum of instrumented phases)\n", "TOTAL");
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

/// draw one frame: publish a snapshot to the render thread (live tty, tier 2a) or
/// render the static string frame (headless, no render thread).
fn publish_or_refresh(dash: Option<&LiveDashboard>, table: &mut Table) {
    match dash {
        Some(d) => d.publish(table.diagnostic_view()),
        None => table.refresh(),
    }
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
    let suffix = if custom_unit {
        format!(" {unit}")
    } else {
        String::new()
    };
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
        [
            "Geometry".into(),
            "dimensions".into(),
            format!("{}D", cfg.dims),
        ],
        [
            "Geometry".into(),
            "resolution".into(),
            format!("{res}  ({n_zones} zones)"),
        ],
        ["Geometry".into(), "boundaries".into(), boundary_label(cfg)],
        ["Scheme".into(), "solver".into(), cfg.solver_name.clone()],
        [
            "Scheme".into(),
            "reconstruction".into(),
            cfg.reconstruction_name.clone(),
        ],
        [
            "Scheme".into(),
            "timestepping".into(),
            timestepping_label(cfg.timestepping),
        ],
        ["Scheme".into(), "cfl".into(), format!("{:.3}", cfg.cfl)],
        ["Run".into(), "t_final".into(), t_final_disp],
        ["Run".into(), "checkpoint dt".into(), cp],
        [
            "Run".into(),
            "est. memory".into(),
            format!("{:.3} GB", est_memory_gb(cfg)),
        ],
        ["Run".into(), "output".into(), cfg.data_dir.clone()],
    ];
    // for PLM (2nd-order) runs, name the slope limiter (from plm_theta) under the reconstruction
    // row. pcm (1st order) has no limiter, so the row is omitted there.
    if cfg.reconstruction_name == "plm" {
        if let Some(i) = rows.iter().position(|r| r[1] == "reconstruction") {
            rows.insert(
                i + 1,
                [
                    "Scheme".into(),
                    "limiter".into(),
                    limiter_label(cfg.plm_theta),
                ],
            );
        }
    }
    // document the time unit only when it is not plain code units.
    if custom_unit {
        rows.push([
            "Run".into(),
            "time unit".into(),
            format!("1 {unit} = {:.4} code", cfg.time_unit),
        ]);
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

/// per-axis boundary tags joined for display (e.g., "reflecting | outflow").
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

/// clip each global refinement region to one tile's physical extent `[origin, origin + cells*dx]`,
/// keeping only the regions that genuinely overlap (more than ~half a coarse cell on every axis, so
/// region_to_domain rounds to >= 1 cell). a TILE-LOCAL patch survives in exactly one tile; a patch
/// SPANNING a cut is split into the abutting tiles, each clipped to its own slab (the per-tile
/// hierarchies then share the cut on the fine level, exchanged by `evolve_hierarchy_decomposed`).
fn clip_regions_to_tile<const D: usize>(
    regions: &[RefinementRegion<D>],
    origin: [f64; D],
    cells: [usize; D],
    dx: [f64; D],
) -> Vec<RefinementRegion<D>> {
    let hi: [f64; D] = std::array::from_fn(|a| origin[a] + cells[a] as f64 * dx[a]);
    let mut out = Vec::new();
    for r in regions {
        let lo: [f64; D] = std::array::from_fn(|a| r.x_lo[a].max(origin[a]));
        let up: [f64; D] = std::array::from_fn(|a| r.x_hi[a].min(hi[a]));
        if (0..D).all(|a| up[a] - lo[a] > 0.5 * dx[a]) {
            out.push(RefinementRegion { x_lo: lo, x_hi: up });
        }
    }
    out
}

/// the effective slope-limiter theta passed to the substrate. PLM uses the
/// config `plm_theta` (default 1.5, theta-MC limiter); PCM — i.e., first-order /
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
        let pos = Tensor::new(std::array::from_fn(|ax| {
            b.position.get(ax).copied().unwrap_or(0.0)
        }));
        let vel = Tensor::new(std::array::from_fn(|ax| {
            b.velocity.get(ax).copied().unwrap_or(0.0)
        }));
        let body = if b.capability & ACCRETION != 0 {
            Body::black_hole(
                idx,
                pos,
                vel,
                b.mass,
                b.radius,
                b.softening,
                b.sink_rate,
                1.0,
                b.accretion_radius,
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
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
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
            BodyKind::BlackHole {
                total_accreted_mass,
                accretion_rate,
                ..
            } => (total_accreted_mass, accretion_rate),
            _ => (0.0, 0.0),
        };
        writeln!(
            f,
            "{time:.8e} {bb} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {accreted:.8e} {rate:.8e}",
            comp(&b.position, 0),
            comp(&b.position, 1),
            comp(&b.velocity, 0),
            comp(&b.velocity, 1),
            comp(&b.force, 0),
            comp(&b.force, 1),
            b.torque[2],
            b.mass,
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

/// per-axis coordinate maps for a non-uniform grid. only the radial (x1) axis carries a
/// configurable spacing; angular axes stay uniform. a uniform radial grid returns `None` (the
/// builder keeps the bit-identical `x_lo + i*dx` path). a log-spaced radial axis maps to
/// `face(i) = start * 10^(i * slope)`, the slope set so the last face reaches `r_hi = start + dx*n`
/// (dx being the linear width the binding derived from the bounds). requires `start > 0`.
fn axis_maps<const D: usize>(cfg: &Config) -> Option<[symbi_geometry::AxisMap; D]> {
    use symbi_geometry::AxisMap;
    if !cfg.x1_spacing.eq_ignore_ascii_case("log") {
        return None;
    }
    Some(std::array::from_fn(|ax| {
        let start = cfg.x_lo[ax];
        let n = cfg.n_cells[ax] as f64;
        if ax == 0 && start > 0.0 && n > 0.0 {
            let r_hi = start + cfg.dx[ax] * n;
            AxisMap::Log { start, log_slope: (r_hi / start).log10() / n }
        } else {
            AxisMap::Uniform { start, dx: cfg.dx[ax] }
        }
    }))
}

/// wrap a built sim + its kernel-set into a `Hierarchy`: a single grid (1 level),
/// or — when refinement is requested — a refined hierarchy whose fine interiors
/// are seeded from the coarse level (conservative prolongation at reconstruction
/// order + 1). `$make` rebuilds a fine level's kernel-set. the unified `run_loop`
/// drives either uniformly.
macro_rules! into_hierarchy {
    ($sim:expr, $kernels:expr, $cfg:expr, $d:literal, $make:expr) => {{
        let mut sim = $sim;
        // the user dt clamp (0 = disabled): pins the dt sequence across runs whose CFL
        // estimators differ (kernel cross-validation, temporal convergence studies).
        sim.max_dt = $cfg.max_dt;
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
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty, $d:literal, $dof:literal, $geom:expr, $geom_ty:ty) => {{
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        // `$d` is the grid dimension, `$dof` the momentum-component count; they differ for the
        // spherical swirl (the azimuthal momentum lifted onto a 2D (r, theta) grid).
        type Sim = SimDefaultGeneric<$regime_ty, $d, $dof, $geom_ty, IdealGas<f64>>;

        // gpus>1 -> the decomposed multi-gpu path (validated separately above by
        // validate_gpu_request); gpus<=1 -> the single-device path below, bit-identical.
        // the DOF-lifted (swirl) tile decomposition is not wired; refuse rather than mis-run.
        if cfg.n_gpus > 1 {
            if $dof != $d {
                return Err("DOF-lifted (swirl) runs do not yet support gpus > 1".to_string());
            }
            return build_and_run_hydro_decomposed!(
                $cfg, $prims, $regime, $regime_ty, $d, $geom, $geom_ty
            );
        }
        if $dof != $d && cfg.refinement_enabled {
            return Err("DOF-lifted (swirl) runs do not yet support mesh refinement".to_string());
        }
        if $dof != $d && !cfg.bodies.is_empty() {
            return Err("DOF-lifted (swirl) runs do not yet support immersed bodies".to_string());
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!(
                "prim_gen yielded {} cells, expected {total}",
                prims.len()
            ));
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .coord_maps(axis_maps::<$d>(cfg))
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
                // the generator row is (rho, v_0 .. v_{DOF-1}, pre) — DOF velocities, so the
                // pressure sits at index 1 + DOF (== 1 + $d except for the swirl lift).
                Prim {
                    rho: row[0],
                    vel: Tensor::new(std::array::from_fn(|k| row[1 + k])),
                    pre: row[1 + $dof],
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
        let sub = sim
            .substrate()
            .theta(theta)
            .with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?;
        // attach a user source expression (force/cooling/relax/raw) when present.
        // lowered against THIS regime's spec via the source front door — the bridge
        // rejects force/cooling/relax on relativistic regimes (use raw). single-grid
        // only: fine levels would not see the source, so refuse the combination.
        let sub = match &cfg.source_json {
            Some(json) => {
                if cfg.refinement_enabled {
                    return Err(
                        "user source expressions are not yet supported with mesh refinement"
                            .to_string(),
                    );
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg,
                    <$regime_ty as Regime<f64, $dof>>::SPEC,
                )
                .map_err(|e| format!("source expression lower: {e}"))?;
                sub.attach_runtime_source(built, scfg.params.clone())?
            }
            None => sub,
        };
        // register DRIVEN (DYNAMIC) boundaries in Driven-id order so `Driven(id)` on a face
        // matches `driven_exprs[id]` — the complete prim prescription [rho, vel_0..DOF-1, pre]
        // as coordinate DAGs (docs/design/33). a theta-stratified rotating equilibrium REQUIRES
        // this: no local ghost rule can represent the state beyond a wedge wall.
        let mut sub = sub;
        if !cfg.driven_exprs.is_empty() && cfg.refinement_enabled {
            return Err("driven boundaries are not yet supported with mesh refinement".to_string());
        }
        for json in &cfg.driven_exprs {
            let bcfg = symbi_hydro::SourceConfig::from_json(json)
                .map_err(|e| format!("boundary expression parse: {e}"))?;
            let built = symbi_hydro::expr_bridge::build_boundary_dag(
                &bcfg,
                <$regime_ty as Regime<f64, $dof>>::SPEC,
            )
            .map_err(|e| format!("boundary expression lower: {e}"))?;
            sub = sub.with_driven_boundary(built, bcfg.params.clone()).0;
        }
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| s
            .substrate()
            .theta(theta)
            .with_solver(solver)
            .expect("fine-level kernel set"));
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// the REGIME-AGNOSTIC decomposed run loop (docs/design/37 M4): evolve N pre-built tiles in
/// lockstep with the UNIVERSAL `PeerCopy` transport (real peer where a link exists, staged over
/// managed memory otherwise -- so the SAME code runs on one card with `--gpus 2` and on a node
/// with `--gpus 8`, no machine-specific branch), gathering into `global` for output through the
/// existing single-grid checkpoint writer. every regime's decomposed build feeds this one loop;
/// adding a regime is just a tile-build, not a new loop. v1 cadence is linear `checkpoint_interval`.
fn run_decomposed_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    cfg: &Config,
    mut tiles: Vec<(SimStateGeneric<R, D, DOF, M, E, S, Mem>, K)>,
    global: SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    counts: [usize; D],
) -> Result<(), String>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    use symbi::sim::decomp::{
        enable_peer_mesh, evolve_decomposed, gather_faces, gather_interiors,
    };

    let ntiles = tiles.len();
    let devices: Vec<i32> = (0..ntiles as i32).collect();
    // open peer links once (no-op for pairs that can't peer; those stage).
    enable_peer_mesh(&devices);

    // the UNIVERSAL transport: adaptive peer/staged. single-device builds compile the host arm
    // but never reach this fn (gpus>1 needs a gpu feature; validate_gpu_request enforces it).
    #[cfg(feature = "gpu")]
    let transport = symbi::sim::decomp::PeerCopy;
    #[cfg(not(feature = "gpu"))]
    let transport = symbi::sim::decomp::LocalCopy;

    let cp_dt = if cfg.checkpoint_interval > 0.0 {
        cfg.checkpoint_interval * cfg.time_unit
    } else {
        f64::INFINITY
    };
    let cp_width = checkpoint_time_width(cfg);

    // body diagnostics: a separate user-defined cadence (natural units), only when the run has
    // bodies + a positive interval -- mirrors the single-grid run. the body state is identical on
    // every tile (`step_bodies_decomposed` applies the cross-tile-summed feedback to all tiles), so
    // any tile's bodies ARE the global diagnostic. one `<dir>diagnostics.dat` for the whole run.
    let diag_path = if cfg.diagnostic_interval > 0.0 && !cfg.bodies.is_empty() {
        Some(format!("{}diagnostics.dat", cfg.data_dir))
    } else {
        None
    };
    let diag_interval = (cfg.diagnostic_interval * cfg.time_unit).max(f64::MIN_POSITIVE);
    let mut next_diag = diag_interval;

    // t=start initial condition. a SHARED reborrow of the tiles for the gather (evolve_decomposed
    // takes them by `&mut` below, so the gather views are scoped reborrows on either side).
    if cfg.checkpoint_index == 0 || cfg.start_time == 0.0 {
        let sh: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
        gather_interiors(&global, &sh, counts);
        gather_faces(&global, &sh, counts);
        let tag = checkpoint_tag(cfg, 0, cp_width, cfg.start_time, cfg.checkpoint_index);
        let _ = write_hierarchy_checkpoint(
            &[&global],
            &checkpoint_name(cfg, &tag),
            &checkpoint_metadata(cfg, cfg.checkpoint_index),
        );
        if let Some(dp) = &diag_path {
            if let Some(im) = sh[0].immersed.as_ref() {
                let _ = append_diagnostics(dp, cfg.start_time, &im.bodies);
            }
        }
    }

    let mut next_cp = cfg.start_time + cp_dt;
    let mut cp_index = cfg.checkpoint_index + 1;
    {
        // the decomposed loop owns the tiles by `&mut` (the per-step immersed-body bookkeeping
        // mutates the bodies). build the `&mut` store handles + the `&` kernels from the SAME tiles
        // (disjoint tuple fields). the checkpoint callback receives the shared tile slice it needs
        // for the gather (it can no longer capture `stores` while the loop holds them mutably).
        let mut stores = Vec::with_capacity(ntiles);
        let mut kernels = Vec::with_capacity(ntiles);
        for (s, k) in tiles.iter_mut() {
            stores.push(&mut **s);
            kernels.push(&*k);
        }
        evolve_decomposed(
            &mut stores,
            &kernels,
            counts,
            &devices,
            cfg.timestepping,
            cfg.start_time,
            cfg.t_final,
            1,
            &transport,
            |_iter, time, sh| {
                if time + f64::EPSILON >= next_cp {
                    gather_interiors(&global, sh, counts);
                    gather_faces(&global, sh, counts);
                    let tag = checkpoint_tag(cfg, 0, cp_width, time, cp_index);
                    let _ = write_hierarchy_checkpoint(
                        &[&global],
                        &checkpoint_name(cfg, &tag),
                        &checkpoint_metadata(cfg, cp_index),
                    );
                    while next_cp <= time {
                        next_cp += cp_dt;
                    }
                    cp_index += 1;
                }
                // body diagnostics at their own cadence: any tile's bodies carry the global
                // (cross-tile-summed) force/torque/accreted-mass after step_bodies_decomposed.
                if let Some(dp) = &diag_path {
                    if time + f64::EPSILON >= next_diag {
                        if let Some(im) = sh[0].immersed.as_ref() {
                            let _ = append_diagnostics(dp, time, &im.bodies);
                        }
                        while next_diag <= time {
                            next_diag += diag_interval;
                        }
                    }
                }
                std::ops::ControlFlow::Continue(())
            },
        );
    }

    // canonical final snapshot, mirroring the single-grid run (shared reborrow again).
    {
        let sh: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
        gather_interiors(&global, &sh, counts);
        gather_faces(&global, &sh, counts);
    }
    let _ = write_hierarchy_checkpoint(
        &[&global],
        &checkpoint_name(cfg, "final"),
        &checkpoint_metadata(cfg, cp_index),
    );
    Ok(())
}

/// the multi-gpu (gpus>1) REFINED path: decompose a 2-level static-refinement hierarchy. each tile
/// is a per-tile `Hierarchy` (its root slab + the global refinement region CLIPPED to that slab, or
/// single-level where the region misses it); `evolve_hierarchy_decomposed` drives them in lockstep
/// (root + first-fine-level halo exchange, oracle-proven by decomp_refine_equivalence +
/// decomp_refine_p3_equivalence). for OUTPUT, gather each level into the global hierarchy -- the
/// root over `counts`, the fine over the `fine_subgrid` sub-grid (a decomposition of the global
/// fine level) -- and write all its levels through the existing multi-level checkpoint writer.
/// v1: a SINGLE refined region (the lib driver decomposes the root + first fine level).
fn run_refined_decomposed_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    cfg: &Config,
    mut tiles: Vec<Hierarchy<R, D, DOF, M, E, S, Mem, K>>,
    global: Hierarchy<R, D, DOF, M, E, S, Mem, K>,
    counts: [usize; D],
) -> Result<(), String>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    use symbi::sim::decomp::{enable_peer_mesh, gather_interiors};
    use symbi::sim::refinement::{evolve_hierarchy_decomposed, fine_subgrid};

    let ntiles = tiles.len();
    let devices: Vec<i32> = (0..ntiles as i32).collect();
    enable_peer_mesh(&devices);

    #[cfg(feature = "gpu")]
    let transport = symbi::sim::decomp::PeerCopy;
    #[cfg(not(feature = "gpu"))]
    let transport = symbi::sim::decomp::LocalCopy;

    let cp_dt = if cfg.checkpoint_interval > 0.0 {
        cfg.checkpoint_interval * cfg.time_unit
    } else {
        f64::INFINITY
    };
    let cp_width = checkpoint_time_width(cfg);

    // the fine sub-grid (which tiles carry a fine level + their order/counts) for the fine gather.
    let fg = fine_subgrid(&tiles, counts, &devices);

    // gather each level of the decomposed tiles into the global hierarchy (root over `counts`, fine
    // over the fine sub-grid), then write all the global levels through the multi-level writer.
    let write_cp = |tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], path: &str, cp_index: u64| {
        let roots: Vec<_> = tiles.iter().map(|h| &*h.levels[0].state).collect();
        gather_interiors(&*global.levels[0].state, &roots, counts);
        if let Some(fg) = &fg {
            let fines: Vec<_> = fg.order.iter().map(|&i| &*tiles[i].levels[1].state).collect();
            gather_interiors(&*global.levels[1].state, &fines, fg.counts);
        }
        let states: Vec<_> = global.levels.iter().map(|l| &l.state).collect();
        let _ = write_hierarchy_checkpoint(&states, path, &checkpoint_metadata(cfg, cp_index));
    };

    // t=start initial condition.
    if cfg.checkpoint_index == 0 || cfg.start_time == 0.0 {
        let tag = checkpoint_tag(cfg, 0, cp_width, cfg.start_time, cfg.checkpoint_index);
        write_cp(&tiles, &checkpoint_name(cfg, &tag), cfg.checkpoint_index);
    }

    let mut next_cp = cfg.start_time + cp_dt;
    let mut cp_index = cfg.checkpoint_index + 1;
    evolve_hierarchy_decomposed(
        &mut tiles,
        counts,
        &devices,
        &transport,
        cfg.timestepping,
        cfg.start_time,
        cfg.t_final,
        1,
        |_iter, time, tiles| {
            if time + f64::EPSILON >= next_cp {
                let tag = checkpoint_tag(cfg, 0, cp_width, time, cp_index);
                write_cp(tiles, &checkpoint_name(cfg, &tag), cp_index);
                while next_cp <= time {
                    next_cp += cp_dt;
                }
                cp_index += 1;
            }
            std::ops::ControlFlow::Continue(())
        },
    );

    // canonical final snapshot.
    write_cp(&tiles, &checkpoint_name(cfg, "final"), cp_index);
    Ok(())
}

/// the multi-gpu (gpus>1) REFINED hydro path: per-tile static-refinement hierarchies driven by the
/// oracle-proven `evolve_hierarchy_decomposed` (root + first-fine-level halo exchange; phases 1-3).
/// each tile builds its root slab + the global refinement region CLIPPED to that slab (single-level
/// where the region misses it); a patch that spans a cut is split into the abutting tiles and the
/// fine halos are exchanged at the cut. output gathers each level into the global hierarchy (root
/// over `counts`, fine over the fine sub-grid) and writes the multi-level checkpoint. v1: PLAIN
/// hydro + a single refined region (bodies / sources / motion with refinement-decomp are refused).
macro_rules! build_and_run_hydro_decomposed_refined {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<$regime_ty, $d, $geom_ty, IdealGas<f64>>;
        // the per-tile / global hierarchy type. DOF = D for hydro; the kernel set is the substrate's
        // associated type (matches what `sim.substrate()` yields).
        type Hier = Hierarchy<
            $regime_ty,
            $d,
            $d,
            $geom_ty,
            IdealGas<f64>,
            DefaultSpace,
            DefaultMemory,
            <Sim as symbi::prelude::SimSubstrate<DefaultMemory, f64, $d>>::KernelSet,
        >;

        // refined decomposition v1 = plain hydro. these combinations each need their own cross-level
        // multi-tile handling; refuse rather than silently ignore.
        if !cfg.bodies.is_empty() {
            return Err("gpus>1 + refinement does not yet support immersed bodies; set gpus=1".to_string());
        }
        if cfg.source_json.is_some() {
            return Err("gpus>1 + refinement does not yet support user sources; set gpus=1".to_string());
        }
        if cfg.mesh_motion {
            return Err("gpus>1 + refinement does not yet support mesh motion; set gpus=1".to_string());
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        let counts = decompose_grid(n, cfg.n_gpus)?;
        let m: [usize; $d] = std::array::from_fn(|ax| n[ax] / counts[ax]);
        let dx: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
        let ntiles: usize = counts.iter().product();
        let theta = build_theta(cfg);
        let solver = cfg.solver;
        let prolong = prolong_order_for(&cfg.reconstruction_name);
        let regions = refinement_regions_nd::<$d>(&cfg.refinement_regions)?;

        // build N per-tile hierarchies (root slab + the clipped region, or single-level).
        let mut tiles: Vec<Hier> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            let origin: [f64; $d] =
                std::array::from_fn(|ax| cfg.x_lo[ax] + (tc[ax] * m[ax]) as f64 * cfg.dx[ax]);
            let phys = boundaries_nd::<$d>(&cfg.boundaries);
            let bnd = Boundaries(std::array::from_fn(|ax| {
                let lo = if tc[ax] == 0 { phys.0[ax][0] } else { BoundaryType::CoarseFine };
                let hi = if tc[ax] == counts[ax] - 1 { phys.0[ax][1] } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            let tile_regions = clip_regions_to_tile::<$d>(&regions, origin, m, dx);
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<Hier, String> {
                let sim = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(dx)
                    .boundaries(bnd)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            lin += (tc[ax] * m[ax] + idx[ax] as usize) * stride;
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
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?;
                let make = |s: &Sim| {
                    s.substrate().theta(theta).with_solver(solver).expect("fine kernel set")
                };
                let mut h = if tile_regions.is_empty() {
                    Hierarchy::single(sim, sub)
                } else {
                    let h = Hierarchy::with_refinement(sim, sub, &tile_regions, prolong, make)
                        .map_err(|e| format!("tile {flat} refinement build: {e:?}"))?;
                    h.seed_fine_from_coarse()
                        .map_err(|e| format!("tile {flat} fine seed: {e:?}"))?;
                    h
                };
                h.prime();
                Ok(h)
            })?;
            tiles.push(built);
        }

        // the full-size OUTPUT hierarchy (root + the full region): gather scatters each level's tile
        // interiors into it. lives on device 0 (touched only at output).
        let global = symbi::symbi_xpu::with_device(0, || -> Result<Hier, String> {
            let groot = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
                .cells(n)
                .origin(std::array::from_fn(|ax| cfg.x_lo[ax]))
                .spacing(dx)
                .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
                .cfl(cfg.cfl)
                .timestepping(cfg.timestepping)
                .cyl_plane(cfg.cyl_plane)
                .allocate()
                .map_err(|e| format!("global output allocate: {e:?}"))?
                .set_initial_indexed(|idx, _x| {
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
            let gsub = groot
                .substrate()
                .theta(theta)
                .with_solver(solver)
                .map_err(|e| format!("global substrate/solver: {e:?}"))?;
            let make = |s: &Sim| {
                s.substrate().theta(theta).with_solver(solver).expect("fine kernel set")
            };
            let gh = Hierarchy::with_refinement(groot, gsub, &regions, prolong, make)
                .map_err(|e| format!("global refinement build: {e:?}"))?;
            gh.seed_fine_from_coarse()
                .map_err(|e| format!("global fine seed: {e:?}"))?;
            Ok(gh)
        })?;

        run_refined_decomposed_loop(cfg, tiles, global, counts)
    }};
}

/// the multi-gpu (gpus>1) hydro path (docs/design/37 M4, design/38). decompose the domain into
/// `cfg.n_gpus` tiles, bind each tile to a device, evolve them in lockstep with halo exchange
/// (the oracle-proven `decomp::evolve_decomposed`), and for output gather the tiles into one
/// full-size sim written by the EXISTING single-grid checkpoint path. v1 is single-level hydro:
/// refinement uses the decomposed-hierarchy path above; immersed bodies / user sources are wired.
/// checkpoint cadence is the LINEAR `checkpoint_interval`; the log cadence + live display are
/// single-grid only. correctness is the same oracle contract (decomposed == monolithic).
macro_rules! build_and_run_hydro_decomposed {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<$regime_ty, $d, $geom_ty, IdealGas<f64>>;

        // refinement + gpus>1 takes the decomposed-HIERARCHY path (per-tile hierarchies + the
        // root/fine halo exchange); plain single-level continues below.
        if cfg.refinement_enabled {
            return build_and_run_hydro_decomposed_refined!(
                $cfg, $prims, $regime, $regime_ty, $d, $geom, $geom_ty
            );
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        // choose the tile grid (product == n_gpus), validate even divisibility.
        let counts = decompose_grid(n, cfg.n_gpus)?;
        let m: [usize; $d] = std::array::from_fn(|ax| n[ax] / counts[ax]);
        let ntiles: usize = counts.iter().product();
        let theta = build_theta(cfg);
        let solver = cfg.solver;
        // the physical (domain-external) boundaries; internal faces become CoarseFine cuts.
        let phys = boundaries_nd::<$d>(&cfg.boundaries);

        // build N tiles, each allocated + substrate-built in its own device context.
        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            let origin: [f64; $d] =
                std::array::from_fn(|ax| cfg.x_lo[ax] + (tc[ax] * m[ax]) as f64 * cfg.dx[ax]);
            let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
            let bnd = Boundaries(std::array::from_fn(|ax| {
                let lo = if tc[ax] == 0 { phys.0[ax][0] } else { BoundaryType::CoarseFine };
                let hi = if tc[ax] == counts[ax] - 1 { phys.0[ax][1] } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                let sim = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(spacing)
                    .boundaries(bnd)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        // local cell -> global cell -> global lin (axis-0-fastest, matches the
                        // python generators and the single-grid build).
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            let g = tc[ax] * m[ax] + idx[ax] as usize;
                            lin += g * stride;
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
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their GLOBAL positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically (oracle: decomp_body_equivalence).

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
                };
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?;
                // attach the user source per tile (two-pass via attach_runtime_source). each tile
                // evaluates S at its OWN global coords (the per-tile origin above), so a
                // position-dependent force is correct across cuts -- proven decomposed==monolithic
                // to round-off by `decomp_source_equivalence`. the global output sim carries no
                // source (it is touched only at gather/output).
                let sub = match &cfg.source_json {
                    Some(json) => {
                        let scfg = symbi_hydro::SourceConfig::from_json(json)
                            .map_err(|e| format!("source expression parse: {e}"))?;
                        let built = symbi_hydro::expr_bridge::build_user_source(
                            &scfg,
                            <$regime_ty as Regime<f64, $d>>::SPEC,
                        )
                        .map_err(|e| format!("source expression lower: {e}"))?;
                        sub.attach_runtime_source(built, scfg.params.clone())?
                    }
                    None => sub,
                };
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        // one full-size sim as the OUTPUT view: the gather scatters tile interiors into it and
        // the existing writer serializes it. lives on device 0 (it is only touched at output).
        let global = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
            .cells(n)
            .origin(std::array::from_fn(|ax| cfg.x_lo[ax]))
            .spacing(std::array::from_fn(|ax| cfg.dx[ax]))
            .boundaries(phys)
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("global output sim allocate: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                // seed the full IC (gather overwrites the interior each checkpoint); local idx
                // is the global cell for the full-size grid.
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

        // hand the built tiles + output sim to the regime-agnostic decomposed loop (evolve +
        // gather + checkpoint, universal transport). every regime shares this loop.
        run_decomposed_loop(cfg, tiles, global, counts)
    }};
}

/// expand the (geometry x dims) arms for one hydro regime. cartesian / spherical
/// / cylindrical are all wired across 1/2/3d (each is a unit-struct `Metric`).
macro_rules! hydro_dispatch {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            // C4/M9 fail-loud guard: a non-minkowski spacetime that is NOT one of the baked GR
            // combinations below would otherwise fall through to a flat `(dims, coords)` arm and run
            // SILENTLY on a Minkowski metric (wrong physics, zero warning). the matches! set is the
            // single source of truth for the baked GR-hydro arms; `test_dispatch_rejects_unbaked_gr`
            // asserts it stays in lockstep with the actual arms (guarded-arm-or-Err, never silent-flat).
            (d, c)
                if $cfg.spacetime != "minkowski"
                    && !matches!(
                        (d, c, $cfg.spacetime.as_str()),
                        (2, "cartesian", "kerr_schild")
                            | (1, "spherical", "schwarzschild")
                            | (2, "spherical", "schwarzschild")
                            | (1, "spherical", "kerr_schild")
                            | (2, "spherical", "kerr")
                            | (2, "spherical", "kerr_schild")
                            | (2, "cylindrical", "kerr_schild")
                            | (3, "cylindrical", "kerr_schild")
                    ) =>
            {
                Err(format!(
                    "no baked GR-hydro kernel for (dims={d}, coords={c}, spacetime={}): refusing to \
                     run silently on a flat Minkowski metric. add the (dims, coords, spacetime) arm \
                     + kernel, or use spacetime=minkowski.",
                    $cfg.spacetime
                ))
            }
            (1, "cartesian") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 1, 1, Cartesian, Cartesian)
            }
            // GR (kerr-schild) CARTESIAN: the (x, y) equatorial slice of the horizon-penetrating
            // chart (design 45) — SchwarzschildKSCartesian selects the `_cart` metric-aware c2p +
            // per-sweep flux + light-cone CFL (non-diagonal gamma, shift on every axis, no polar
            // axis). guarded BEFORE the flat cartesian arm; 2D equatorial slice, DOF = 2 (no swirl).
            (2, "cartesian") if $cfg.spacetime == "kerr_schild" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 2, 2,
                SchwarzschildKSCartesian { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCartesian<f64>
            ),
            (2, "cartesian") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 2, 2, Cartesian, Cartesian)
            }
            (3, "cartesian") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 3, 3, Cartesian, Cartesian)
            }
            // GR (Schwarzschild) spherical: select the Schwarzschild metric (lapse-densitized +
            // GR-wavespeed `_schw` kernels). baked for 1D/2D (the Michel accretion targets); 3D
            // spherical Schwarzschild has no baked kernel and is rejected by the fail-loud guard
            // above (never silently run on a flat metric). the spacetime is orthogonal to the regime.
            (1, "spherical") if $cfg.spacetime == "schwarzschild" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 1, 1,
                Schwarzschild { mass: $cfg.schwarzschild_mass }, Schwarzschild<f64>
            ),
            // 2D GR: the generator row length picks the momentum DOF — (rho, v_r, v_theta, pre)
            // is the axisymmetric in-plane flow (DOF = 2); (rho, v_r, v_theta, v_phi, pre) lifts
            // the azimuthal momentum onto the (r, theta) grid (DOF = 3, the `_sph_swirl`
            // kernels: rotating flows — tori, spinning-hole accretion).
            (2, "spherical") if $cfg.spacetime == "schwarzschild" => {
                if $prims.first().map_or(false, |row| row.len() == 5) {
                    build_and_run_hydro!(
                        $cfg, $prims, $regime, $regime_ty, 2, 3,
                        Schwarzschild { mass: $cfg.schwarzschild_mass }, Schwarzschild<f64>
                    )
                } else {
                    build_and_run_hydro!(
                        $cfg, $prims, $regime, $regime_ty, 2, 2,
                        Schwarzschild { mass: $cfg.schwarzschild_mass }, Schwarzschild<f64>
                    )
                }
            }
            // GR (ingoing Kerr-Schild) spherical: the HORIZON-PENETRATING chart (regular across
            // r = 2M) — the `_ks` shift-advection-flux + KS-densitized/wavespeed kernels. reuses the
            // `schwarzschild_mass` scalar. 1D radial + the 2D plane (with the same row-length DOF
            // pick as schwarzschild: 5-tuples lift the azimuthal momentum, `_sph_swirl`).
            (1, "spherical") if $cfg.spacetime == "kerr_schild" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 1, 1,
                SchwarzschildKS { mass: $cfg.schwarzschild_mass }, SchwarzschildKS<f64>
            ),
            // spinning kerr (ingoing kerr-schild coords): the frame-dragging gamma_{r phi}
            // needs the azimuthal momentum DOF, so the 5-tuple (swirl) generator row is
            // REQUIRED — a 4-tuple config is a setup error, not a fallback.
            (2, "spherical") if $cfg.spacetime == "kerr" => {
                if !$prims.first().map_or(false, |row| row.len() == 5) {
                    return Err(
                        "the kerr spacetime requires the azimuthal momentum DOF: yield \
                         5-tuple gas rows (rho, v_r, v_theta, v_phi, pre)".to_string(),
                    );
                }
                build_and_run_hydro!(
                    $cfg, $prims, $regime, $regime_ty, 2, 3,
                    KerrKS { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin }, KerrKS<f64>
                )
            }
            (2, "spherical") if $cfg.spacetime == "kerr_schild" => {
                if $prims.first().map_or(false, |row| row.len() == 5) {
                    build_and_run_hydro!(
                        $cfg, $prims, $regime, $regime_ty, 2, 3,
                        SchwarzschildKS { mass: $cfg.schwarzschild_mass }, SchwarzschildKS<f64>
                    )
                } else {
                    build_and_run_hydro!(
                        $cfg, $prims, $regime, $regime_ty, 2, 2,
                        SchwarzschildKS { mass: $cfg.schwarzschild_mass }, SchwarzschildKS<f64>
                    )
                }
            }
            (1, "spherical") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 1, 1, Spherical, Spherical)
            }
            (2, "spherical") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 2, 2, Spherical, Spherical)
            }
            (3, "spherical") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 3, 3, Spherical, Spherical)
            }
            (1, "cylindrical") => build_and_run_hydro!(
                $cfg,
                $prims,
                $regime,
                $regime_ty,
                1,
                1,
                Cylindrical,
                Cylindrical
            ),
            // GR (kerr-schild) CYLINDRICAL 2D: the plane selector splits the two charts. the (R, phi)
            // equatorial DISK (planar_cylindrical) is DIAGONAL (z = 0, r = R), DOF = 2 (v_R, v_phi);
            // the (R, z) 2.5D axisymmetric-swirl (the default) lifts v_phi, DOF = 3, requiring the
            // 5-tuple. both use the one SchwarzschildKSCylindrical metric (D = 2 disk / D = 3 swirl).
            (2, "cylindrical") if $cfg.spacetime == "kerr_schild" => match $cfg.cyl_plane {
                symbi_sim::state::CylPlane::RPhi => build_and_run_hydro!(
                    $cfg, $prims, $regime, $regime_ty, 2, 2,
                    SchwarzschildKSCylindrical { mass: $cfg.schwarzschild_mass },
                    SchwarzschildKSCylindrical<f64>
                ),
                symbi_sim::state::CylPlane::Rz => {
                    if !$prims.first().map_or(false, |row| row.len() == 5) {
                        return Err(
                            "the cylindrical kerr-schild 2.5D (axisymmetric) chart requires the \
                             azimuthal momentum DOF: yield 5-tuple gas rows (rho, v_R, v_phi, v_z, pre)"
                                .to_string(),
                        );
                    }
                    build_and_run_hydro!(
                        $cfg, $prims, $regime, $regime_ty, 2, 3,
                        SchwarzschildKSCylindrical { mass: $cfg.schwarzschild_mass },
                        SchwarzschildKSCylindrical<f64>
                    )
                }
            },
            (2, "cylindrical") => build_and_run_hydro!(
                $cfg,
                $prims,
                $regime,
                $regime_ty,
                2,
                2,
                Cylindrical,
                Cylindrical
            ),
            // GR (kerr-schild) CYLINDRICAL full 3D (R, phi, z): DOF == NDIM = 3.
            (3, "cylindrical") if $cfg.spacetime == "kerr_schild" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 3, 3,
                SchwarzschildKSCylindrical { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCylindrical<f64>
            ),
            (3, "cylindrical") => build_and_run_hydro!(
                $cfg,
                $prims,
                $regime,
                $regime_ty,
                3,
                3,
                Cylindrical,
                Cylindrical
            ),
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

/// the cell-centered average of a staggered face buffer along its own axis `k`: the two
/// bounding faces of cell `idx`. `faces` is axis-0-fastest over the face domain (cell dims `n`
/// extended +1 on `k`). the curved-spacetime IC seeds the TRUE cell B so the covariant
/// conserved state carries the magnetic terms exactly (the flat path's zero-seed +
/// bcell-from-bface heal applies a EUCLIDEAN energy patch that is wrong under a metric, and
/// never installs the -(v.B) B_i momentum block).
fn face_avg_cell_b<const D: usize>(faces: &[f64], k: usize, idx: [isize; D], n: [usize; D]) -> f64 {
    let mut dims = n;
    dims[k] += 1;
    let lin = |c: [usize; D]| -> usize {
        let mut l = 0usize;
        let mut stride = 1usize;
        for ax in 0..D {
            l += c[ax] * stride;
            stride *= dims[ax];
        }
        l
    };
    let lo: [usize; D] = std::array::from_fn(|ax| idx[ax] as usize);
    let mut hi = lo;
    hi[k] += 1;
    0.5 * (faces[lin(lo)] + faces[lin(hi)])
}

/// slice a GLOBAL per-axis staggered face buffer into the face buffer for one tile, in the
/// axis-0-fastest order `seed_faces_indexed` consumes. `global` is the python `staggered_bfields`
/// generator for axis `d`: axis-0-fastest over the global interior face domain (cell dims `n`
/// extended +1 on `d`). the tile owns `m` cells per axis at tile coord `tc`; its axis-`d` face
/// domain is `m` extended +1 on `d`. the shared internal face (a tile's hi-`d` face == its
/// neighbor's lo-`d` face) maps to the same global index in both tiles, so both seed it
/// identically -- the CT normal-face consistency the decomposition requires, by construction.
fn tile_face_buffer<const D: usize>(
    global: &[f64],
    n: [usize; D],
    m: [usize; D],
    tc: [usize; D],
    d: usize,
) -> Vec<f64> {
    let gdim: [usize; D] = std::array::from_fn(|ax| n[ax] + usize::from(ax == d));
    let ldim: [usize; D] = std::array::from_fn(|ax| m[ax] + usize::from(ax == d));
    let vol: usize = ldim.iter().product();
    let mut out = Vec::with_capacity(vol);
    let mut lc = [0usize; D];
    for _ in 0..vol {
        // local face coord -> global face coord -> global flat (axis-0-fastest).
        let mut gi = 0usize;
        let mut stride = 1usize;
        for ax in 0..D {
            gi += (tc[ax] * m[ax] + lc[ax]) * stride;
            stride *= gdim[ax];
        }
        out.push(global[gi]);
        for ax in 0..D {
            lc[ax] += 1;
            if lc[ax] < ldim[ax] {
                break;
            }
            lc[ax] = 0;
        }
    }
    out
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

        // gpus>1 -> the decomposed multi-gpu path (validated separately above by
        // validate_gpu_request); gpus<=1 -> the single-device path below, bit-identical.
        if cfg.n_gpus > 1 {
            return build_and_run_mhd_decomposed!(
                $cfg, $prims, $bufs, $regime, $regime_ty, $d, $geom, $geom_ty
            );
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!(
                "prim_gen yielded {} cells, expected {total}",
                prims.len()
            ));
        }
        if bufs.len() < 3 {
            return Err(format!(
                "mhd needs 3 staggered b-field generators, got {}",
                bufs.len()
            ));
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .coord_maps(axis_maps::<$d>(cfg))
            .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("allocate failed: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let lin = lin_index!(idx, n, $d);
                let row = &prims[lin];
                // gridded components seed the TRUE cell B (the face average) so the conserved
                // state carries every magnetic term from step zero — the old zero-seed left the
                // relativistic momentum's B^2 v - (v.B) B block missing and the stage-1
                // bcell_from_bface energy heal is exact only at v = 0 (and euclidean-only).
                let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                    if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                });
                MhdPrim {
                    hydro: Prim {
                        rho: row[0],
                        vel: Tensor::new([row[1], row[2], row[3]]),
                        pre: row[4],
                    },
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
        let sub = sim
            .substrate()
            .theta(theta)
            .with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?
            .ct_method(cfg.ct_method);
        // attach a user source expression to the MHD hydro slots (den/mom/nrg).
        // rmhd is relativistic -> only kind="raw"; nmhd takes force/cooling/relax.
        // B is CT-evolved, not a cell source. single-grid only.
        let sub = match &cfg.source_json {
            Some(json) => {
                if cfg.refinement_enabled {
                    return Err(
                        "user source expressions are not yet supported with mesh refinement"
                            .to_string(),
                    );
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg,
                    <$regime_ty as Regime<f64, $d>>::SPEC,
                )
                .map_err(|e| format!("source expression lower: {e}"))?;
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
                &bcfg,
                <$regime_ty as Regime<f64, $d>>::SPEC,
            )
            .map_err(|e| format!("boundary expression lower: {e}"))?;
            sub = sub.with_driven_boundary(built, bcfg.params.clone()).0;
        }
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| s
            .substrate()
            .theta(theta)
            .with_solver(solver)
            .expect("fine-level kernel set"));
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

        // gpus>1 -> the decomposed multi-gpu path (validated separately above by
        // validate_gpu_request); gpus<=1 -> the single-device path below, bit-identical.
        if cfg.n_gpus > 1 {
            return build_and_run_imhd_decomposed!($cfg, $prims, $bufs, $d, $geom, $geom_ty);
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!(
                "prim_gen yielded {} cells, expected {total}",
                prims.len()
            ));
        }
        if bufs.len() < 3 {
            return Err(format!(
                "imhd needs 3 staggered b-field generators, got {}",
                bufs.len()
            ));
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .coord_maps(axis_maps::<$d>(cfg))
            .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("allocate failed: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let lin = lin_index!(idx, n, $d);
                let row = &prims[lin];
                let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                    if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                });
                MhdPrimG::<f64, 3, IsoModel> {
                    hydro: PrimG {
                        rho: row[0],
                        vel: Tensor::new([row[1], row[2], row[3]]),
                        pre: Default::default(),
                    },
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
        let sub = sim
            .substrate()
            .theta(theta)
            .with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?
            .ct_method(cfg.ct_method);
        // attach a user source. iso MHD has no energy -> momentum-only force/relax,
        // raw den/mom (raw->nrg rejected); B is CT-evolved. single-grid only.
        let sub = match &cfg.source_json {
            Some(json) => {
                if cfg.refinement_enabled {
                    return Err(
                        "user source expressions are not yet supported with mesh refinement"
                            .to_string(),
                    );
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg,
                    <IsothermalMhd as Regime<f64, $d>>::SPEC,
                )
                .map_err(|e| format!("source expression lower: {e}"))?;
                sub.attach_runtime_source(built, scfg.params.clone())?
            }
            None => sub,
        };
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| s
            .substrate()
            .theta(theta)
            .with_solver(solver)
            .expect("fine-level kernel set"));
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// the multi-gpu (gpus>1) ADIABATIC MHD path: the MHD analog of `build_and_run_hydro_decomposed`.
/// decompose the domain into `cfg.n_gpus` tiles, bind each to a device, and evolve them in lockstep
/// with the staggered-CT halo exchange (the oracle-proven `decomp::evolve_decomposed`, verified
/// `decomposed == monolithic` to round-off with div(B) exact). cell state seeds `MhdPrim` from the
/// global prim rows; the staggered face B seeds each tile from its slice of the global
/// `staggered_bfields` (`tile_face_buffer`), so the shared internal face is identical in both
/// neighbors by construction. output gathers cell fields + cell B (`gather_interiors`) AND the
/// staggered faces (`gather_faces`) into one global sim written by the existing checkpoint path.
/// single-level only: refinement / bodies / user sources with gpus>1 are refused.
macro_rules! build_and_run_mhd_decomposed {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<$regime_ty, $d, 3, $geom_ty, IdealGas<f64>>;

        // v1 multi-gpu = single-level. these interactions are deferred; refuse rather than silently
        // ignore (each needs its own multi-tile handling).
        if cfg.refinement_enabled {
            return Err("gpus>1 does not yet support mesh refinement; set gpus=1 or disable refinement".to_string());
        }
        if !cfg.driven_exprs.is_empty() {
            return Err("gpus>1 does not yet support driven boundaries; set gpus=1".to_string());
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if bufs.len() < 3 {
            return Err(format!("mhd needs 3 staggered b-field generators, got {}", bufs.len()));
        }
        let counts = decompose_grid(n, cfg.n_gpus)?;
        let m: [usize; $d] = std::array::from_fn(|ax| n[ax] / counts[ax]);
        let ntiles: usize = counts.iter().product();
        let theta = build_theta(cfg);
        let solver = cfg.solver;
        let ct = cfg.ct_method;
        let phys = boundaries_nd::<$d>(&cfg.boundaries);

        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            let origin: [f64; $d] =
                std::array::from_fn(|ax| cfg.x_lo[ax] + (tc[ax] * m[ax]) as f64 * cfg.dx[ax]);
            let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
            let bnd = Boundaries(std::array::from_fn(|ax| {
                let lo = if tc[ax] == 0 { phys.0[ax][0] } else { BoundaryType::CoarseFine };
                let hi = if tc[ax] == counts[ax] - 1 { phys.0[ax][1] } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                // per-tile slice of each global face buffer (axis-0-fastest over the tile face
                // domain); the shared internal face reads the same global value in both neighbors.
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tc, d)).collect();
                let sim = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(spacing)
                    .boundaries(bnd)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        // local cell -> global cell -> global lin (axis-0-fastest, matching the
                        // python generators and the single-grid build); cell B reads the transverse
                        // axes of the global buffers at the same lin.
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            lin += (tc[ax] * m[ax] + idx[ax] as usize) * stride;
                            stride *= n[ax];
                        }
                        let row = &prims[lin];
                        let gidx: [isize; $d] =
                            std::array::from_fn(|ax| (tc[ax] * m[ax]) as isize + idx[ax]);
                        let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        MhdPrim {
                            hydro: Prim {
                                rho: row[0],
                                vel: Tensor::new([row[1], row[2], row[3]]),
                                pre: row[4],
                            },
                            mag: Tensor::new(mag_arr),
                        }
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their GLOBAL positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically (oracle: decomp_body_equivalence).

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
                };
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?
                    .ct_method(ct);
                // attach the user source per tile (two-pass). targets the mhd hydro slots
                // (den/mom/nrg); B is CT-evolved, not a cell source. each tile evaluates S at its
                // own global coords. rmhd is relativistic -> raw only (enforced in build_user_source).
                let sub = match &cfg.source_json {
                    Some(json) => {
                        let scfg = symbi_hydro::SourceConfig::from_json(json)
                            .map_err(|e| format!("source expression parse: {e}"))?;
                        let built = symbi_hydro::expr_bridge::build_user_source(
                            &scfg,
                            <$regime_ty as Regime<f64, $d>>::SPEC,
                        )
                        .map_err(|e| format!("source expression lower: {e}"))?;
                        sub.attach_runtime_source(built, scfg.params.clone())?
                    }
                    None => sub,
                };
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        // the full-size OUTPUT view: gather scatters tile interiors (cells + cell B) and faces into
        // it each checkpoint; seed the faces so `bface_initialized` is set (the gather overwrites
        // the interior). lives on device 0 (touched only at output).
        let global = Sim::build($regime, IdealGas { gamma: cfg.gamma }, $geom)
            .cells(n)
            .origin(std::array::from_fn(|ax| cfg.x_lo[ax]))
            .spacing(std::array::from_fn(|ax| cfg.dx[ax]))
            .boundaries(phys)
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("global output sim allocate: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let mut lin = 0usize;
                let mut stride = 1usize;
                for ax in 0..$d {
                    lin += idx[ax] as usize * stride;
                    stride *= n[ax];
                }
                let row = &prims[lin];
                let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                    if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                });
                MhdPrim {
                    hydro: Prim {
                        rho: row[0],
                        vel: Tensor::new([row[1], row[2], row[3]]),
                        pre: row[4],
                    },
                    mag: Tensor::new(mag_arr),
                }
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        run_decomposed_loop(cfg, tiles, global, counts)
    }};
}

/// the multi-gpu (gpus>1) ISOTHERMAL MHD path: identical tiling/face-seeding/gather to the
/// adiabatic decomposed macro, but the iso primitive has NO pressure slot (`IsoModel` ZST) and the
/// eos closure is `p = cs^2 rho`. single-level only.
macro_rules! build_and_run_imhd_decomposed {
    ($cfg:expr, $prims:expr, $bufs:expr, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<IsothermalMhd, $d, 3, $geom_ty, Isothermal<f64>>;

        if cfg.refinement_enabled {
            return Err("gpus>1 does not yet support mesh refinement; set gpus=1 or disable refinement".to_string());
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if bufs.len() < 3 {
            return Err(format!("imhd needs 3 staggered b-field generators, got {}", bufs.len()));
        }
        let counts = decompose_grid(n, cfg.n_gpus)?;
        let m: [usize; $d] = std::array::from_fn(|ax| n[ax] / counts[ax]);
        let ntiles: usize = counts.iter().product();
        let theta = build_theta(cfg);
        let solver = cfg.solver;
        let ct = cfg.ct_method;
        let phys = boundaries_nd::<$d>(&cfg.boundaries);

        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            let origin: [f64; $d] =
                std::array::from_fn(|ax| cfg.x_lo[ax] + (tc[ax] * m[ax]) as f64 * cfg.dx[ax]);
            let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
            let bnd = Boundaries(std::array::from_fn(|ax| {
                let lo = if tc[ax] == 0 { phys.0[ax][0] } else { BoundaryType::CoarseFine };
                let hi = if tc[ax] == counts[ax] - 1 { phys.0[ax][1] } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tc, d)).collect();
                let sim = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(spacing)
                    .boundaries(bnd)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            lin += (tc[ax] * m[ax] + idx[ax] as usize) * stride;
                            stride *= n[ax];
                        }
                        let row = &prims[lin];
                        let gidx: [isize; $d] =
                            std::array::from_fn(|ax| (tc[ax] * m[ax]) as isize + idx[ax]);
                        let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        MhdPrimG::<f64, 3, IsoModel> {
                            hydro: PrimG {
                                rho: row[0],
                                vel: Tensor::new([row[1], row[2], row[3]]),
                                pre: Default::default(),
                            },
                            mag: Tensor::new(mag_arr),
                        }
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their GLOBAL positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically (oracle: decomp_body_equivalence).

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
                };
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?
                    .ct_method(ct);
                // attach the user source per tile (two-pass). iso mhd has no energy -> momentum-only
                // force/relax, raw den/mom; B is CT-evolved. each tile evaluates S at its own coords.
                let sub = match &cfg.source_json {
                    Some(json) => {
                        let scfg = symbi_hydro::SourceConfig::from_json(json)
                            .map_err(|e| format!("source expression parse: {e}"))?;
                        let built = symbi_hydro::expr_bridge::build_user_source(
                            &scfg,
                            <IsothermalMhd as Regime<f64, $d>>::SPEC,
                        )
                        .map_err(|e| format!("source expression lower: {e}"))?;
                        sub.attach_runtime_source(built, scfg.params.clone())?
                    }
                    None => sub,
                };
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        let global = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
            .cells(n)
            .origin(std::array::from_fn(|ax| cfg.x_lo[ax]))
            .spacing(std::array::from_fn(|ax| cfg.dx[ax]))
            .boundaries(phys)
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("global output sim allocate: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let mut lin = 0usize;
                let mut stride = 1usize;
                for ax in 0..$d {
                    lin += idx[ax] as usize * stride;
                    stride *= n[ax];
                }
                let row = &prims[lin];
                let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                    if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                });
                MhdPrimG::<f64, 3, IsoModel> {
                    hydro: PrimG {
                        rho: row[0],
                        vel: Tensor::new([row[1], row[2], row[3]]),
                        pre: Default::default(),
                    },
                    mag: Tensor::new(mag_arr),
                }
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        run_decomposed_loop(cfg, tiles, global, counts)
    }};
}

/// expand the (geometry x dims) arms for an adiabatic mhd regime. cartesian /
/// spherical / cylindrical across 1/2/3d (the cylindrical 2D plane is selected by
/// `cfg.cyl_plane`, threaded into every build).
macro_rules! mhd_dispatch {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            // C4/M9 fail-loud guard (see hydro_dispatch): reject a non-minkowski spacetime that is
            // not a baked GR-MHD arm rather than silently running it on Minkowski. the matches! set
            // mirrors the baked GR-MHD arms; test_dispatch_rejects_unbaked_gr keeps them in lockstep.
            (d, c)
                if $cfg.spacetime != "minkowski"
                    && !matches!(
                        (d, c, $cfg.spacetime.as_str()),
                        (1, "spherical", "schwarzschild")
                            | (1, "spherical", "kerr_schild")
                            | (2, "spherical", "schwarzschild")
                            | (2, "spherical", "kerr")
                            | (2, "cartesian", "kerr_schild")
                            | (2, "cylindrical", "kerr_schild")
                    ) =>
            {
                Err(format!(
                    "no baked GR-MHD kernel for (dims={d}, coords={c}, spacetime={}): refusing to \
                     run silently on a flat Minkowski metric. add the (dims, coords, spacetime) arm \
                     + kernel, or use spacetime=minkowski.",
                    $cfg.spacetime
                ))
            }
            // GR (Schwarzschild) spherical MHD: the metric type selects the `_schw` GRMHD
            // kernel row (RmhdGr valencia flux + metric-aware KKC c2p + the ideal-MHD stress
            // in the covariant source). baked 1D radial (the magnetized-michel target).
            (1, "spherical") if $cfg.spacetime == "schwarzschild" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 1,
                Schwarzschild { mass: $cfg.schwarzschild_mass }, Schwarzschild<f64>
            ),
            // the horizon-penetrating chart: the `_ks` GRMHD row (the shifted riemann fan with
            // the induction transpose term). the inner boundary can sit below r = 2M.
            (1, "spherical") if $cfg.spacetime == "kerr_schild" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 1,
                SchwarzschildKS { mass: $cfg.schwarzschild_mass }, SchwarzschildKS<f64>
            ),
            // the 2D (r, theta) GRMHD row: the curved-CT machinery (densitized corner EMF +
            // curl + metric-contracted interpolation; contact EMF only — design 44).
            (2, "spherical") if $cfg.spacetime == "schwarzschild" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                Schwarzschild { mass: $cfg.schwarzschild_mass }, Schwarzschild<f64>
            ),
            // the 2D (r, theta) SPINNING-KERR GRMHD row (design 44): the non-diagonal
            // gamma_{r phi} rides the tetrad HLLD, the radial shift the moving-interface fan, and
            // the azimuthal (swirl) momentum the frame dragging. requires the 5-tuple swirl gas rows.
            (2, "spherical") if $cfg.spacetime == "kerr" => {
                if !$prims.first().map_or(false, |row| row.len() == 5) {
                    return Err(
                        "the kerr GRMHD spacetime requires the azimuthal momentum DOF: yield \
                         5-tuple gas rows (rho, v_r, v_theta, v_phi, pre)".to_string(),
                    );
                }
                build_and_run_mhd!(
                    $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                    KerrKS { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin }, KerrKS<f64>
                )
            }
            // the 2D cartesian (x, y) GRMHD row (design 45): the NON-DIAGONAL kerr-schild spatial
            // metric selects the fast-magnetosonic HLLE gas flux + the contact / UCT-HLL densitized
            // CT. the tetrad HLLD wrapper — which the kerr (r, theta) row above already rides on its
            // non-diagonal gamma_{r phi} — is not yet wired for this chart; HLLE here is a follow-on
            // gap, not a metric-diagonality limitation (the Gram-Schmidt tetrad handles non-diagonal
            // spatial metrics). the covariant geodesic + EM-stress source carries the gravity.
            (2, "cartesian") if $cfg.spacetime == "kerr_schild" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                SchwarzschildKSCartesian { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCartesian<f64>
            ),
            (1, "cartesian") => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 1, Cartesian, Cartesian
            ),
            (2, "cartesian") => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 2, Cartesian, Cartesian
            ),
            (3, "cartesian") => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 3, Cartesian, Cartesian
            ),
            (1, "spherical") => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 1, Spherical, Spherical
            ),
            (2, "spherical") => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 2, Spherical, Spherical
            ),
            (3, "spherical") => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 3, Spherical, Spherical
            ),
            (1, "cylindrical") => build_and_run_mhd!(
                $cfg,
                $prims,
                $bufs,
                $regime,
                $regime_ty,
                1,
                Cylindrical,
                Cylindrical
            ),
            // GR (kerr-schild) CYLINDRICAL 2D GRMHD (design 45): the cyl_plane selector (threaded
            // into the geom axes by the builder) splits the two charts — the (R, z) 2.5D poloidal
            // plane (axes [0, 2], non-diagonal gamma_Rz, toroidal E_phi CT) and the (R, phi)
            // equatorial DISK (axes [0, 1], diagonal on the equator, vertical E_z CT). MHD momentum
            // is a full 3-vector in both, so one metric arm serves; HLLE gas flux + contact/UCT-HLL CT.
            (2, "cylindrical") if $cfg.spacetime == "kerr_schild" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                SchwarzschildKSCylindrical { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCylindrical<f64>
            ),
            (2, "cylindrical") => build_and_run_mhd!(
                $cfg,
                $prims,
                $bufs,
                $regime,
                $regime_ty,
                2,
                Cylindrical,
                Cylindrical
            ),
            (3, "cylindrical") => build_and_run_mhd!(
                $cfg,
                $prims,
                $bufs,
                $regime,
                $regime_ty,
                3,
                Cylindrical,
                Cylindrical
            ),
            (d, g) => Err(format!("no mhd dispatch arm for (dims={d}, coord={g}) yet")),
        }
    };
}

/// expand the (geometry x dims) arms for isothermal mhd. cartesian / spherical /
/// cylindrical across 1/2/3d.
macro_rules! imhd_dispatch {
    ($cfg:expr, $prims:expr, $bufs:expr) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            // C4/M9 fail-loud guard: isothermal MHD has NO baked GR kernels, so any non-minkowski
            // spacetime must fail loud rather than silently run flat.
            (d, c) if $cfg.spacetime != "minkowski" => Err(format!(
                "isothermal MHD has no GR kernels; (dims={d}, coords={c}, spacetime={}) is unsupported \
                 — refusing to run silently on a flat Minkowski metric. use spacetime=minkowski.",
                $cfg.spacetime
            )),
            (1, "cartesian") => build_and_run_imhd!($cfg, $prims, $bufs, 1, Cartesian, Cartesian),
            (2, "cartesian") => build_and_run_imhd!($cfg, $prims, $bufs, 2, Cartesian, Cartesian),
            (3, "cartesian") => build_and_run_imhd!($cfg, $prims, $bufs, 3, Cartesian, Cartesian),
            (1, "spherical") => build_and_run_imhd!($cfg, $prims, $bufs, 1, Spherical, Spherical),
            (2, "spherical") => build_and_run_imhd!($cfg, $prims, $bufs, 2, Spherical, Spherical),
            (3, "spherical") => build_and_run_imhd!($cfg, $prims, $bufs, 3, Spherical, Spherical),
            (1, "cylindrical") => {
                build_and_run_imhd!($cfg, $prims, $bufs, 1, Cylindrical, Cylindrical)
            }
            (2, "cylindrical") => {
                build_and_run_imhd!($cfg, $prims, $bufs, 2, Cylindrical, Cylindrical)
            }
            (3, "cylindrical") => {
                build_and_run_imhd!($cfg, $prims, $bufs, 3, Cylindrical, Cylindrical)
            }
            (d, g) => Err(format!(
                "no imhd dispatch arm for (dims={d}, coord={g}) yet"
            )),
        }
    };
}

/// the multi-gpu (gpus>1) ISOTHERMAL path: the iso sibling of `build_and_run_hydro_decomposed!`.
/// builds N iso tiles + a global output sim and hands them to the shared `run_decomposed_loop`
/// (universal transport). v1 is GLOBALLY isothermal (uniform cs); locally-isothermal cs(x) needs
/// per-tile cs^2 setup and is deferred (guarded). same non-AMR / no-bodies / no-source scope.
macro_rules! build_and_run_iso_decomposed {
    ($cfg:expr, $prims:expr, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<IsoNewtonian, $d, $geom_ty, Isothermal<f64>>;

        if cfg.refinement_enabled {
            return Err("gpus>1 does not yet support mesh refinement; set gpus=1".to_string());
        }
        if cfg.locally_isothermal {
            return Err("gpus>1 does not yet support locally-isothermal cs(x); set gpus=1 or use globally isothermal".to_string());
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        let counts = decompose_grid(n, cfg.n_gpus)?;
        let m: [usize; $d] = std::array::from_fn(|ax| n[ax] / counts[ax]);
        let ntiles: usize = counts.iter().product();
        let theta = build_theta(cfg);
        let phys = boundaries_nd::<$d>(&cfg.boundaries);

        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            let origin: [f64; $d] =
                std::array::from_fn(|ax| cfg.x_lo[ax] + (tc[ax] * m[ax]) as f64 * cfg.dx[ax]);
            let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
            let bnd = Boundaries(std::array::from_fn(|ax| {
                let lo = if tc[ax] == 0 { phys.0[ax][0] } else { BoundaryType::CoarseFine };
                let hi = if tc[ax] == counts[ax] - 1 { phys.0[ax][1] } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                let sim = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(spacing)
                    .boundaries(bnd)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            let g = tc[ax] * m[ax] + idx[ax] as usize;
                            lin += g * stride;
                            stride *= n[ax];
                        }
                        let row = &prims[lin];
                        PrimG::<f64, $d, IsoModel> {
                            rho: row[0],
                            vel: Tensor::new(std::array::from_fn(|k| row[1 + k])),
                            pre: Default::default(),
                        }
                    })
                    .build();
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their GLOBAL positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically (oracle: decomp_body_equivalence).

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    sim.with_bodies(build_bodies::<$d>(&cfg.bodies))
                };
                let sub = sim.substrate().theta(theta);
                // attach the user source per tile (two-pass). iso has no energy -> momentum-only
                // force/relax, raw den/mom. each tile evaluates S at its own global coords.
                let sub = match &cfg.source_json {
                    Some(json) => {
                        let scfg = symbi_hydro::SourceConfig::from_json(json)
                            .map_err(|e| format!("source expression parse: {e}"))?;
                        let built = symbi_hydro::expr_bridge::build_user_source(
                            &scfg,
                            <IsoNewtonian as Regime<f64, $d>>::SPEC,
                        )
                        .map_err(|e| format!("source expression lower: {e}"))?;
                        sub.attach_runtime_source(built, scfg.params.clone())?
                    }
                    None => sub,
                };
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        let global = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
            .cells(n)
            .origin(std::array::from_fn(|ax| cfg.x_lo[ax]))
            .spacing(std::array::from_fn(|ax| cfg.dx[ax]))
            .boundaries(phys)
            .cfl(cfg.cfl)
            .timestepping(cfg.timestepping)
            .cyl_plane(cfg.cyl_plane)
            .allocate()
            .map_err(|e| format!("global output sim allocate: {e:?}"))?
            .set_initial_indexed(|idx, _x| {
                let mut lin = 0usize;
                let mut stride = 1usize;
                for ax in 0..$d {
                    lin += idx[ax] as usize * stride;
                    stride *= n[ax];
                }
                let row = &prims[lin];
                PrimG::<f64, $d, IsoModel> {
                    rho: row[0],
                    vel: Tensor::new(std::array::from_fn(|k| row[1 + k])),
                    pre: Default::default(),
                }
            })
            .build();

        run_decomposed_loop(cfg, tiles, global, counts)
    }};
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

        // gpus>1 -> the decomposed iso path; gpus<=1 -> the single-device path below.
        if cfg.n_gpus > 1 {
            return build_and_run_iso_decomposed!($cfg, $prims, $d, $geom, $geom_ty);
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!(
                "prim_gen yielded {} cells, expected {total}",
                prims.len()
            ));
        }
        // locally isothermal carries an extra per-cell pressure component.
        let want = if cfg.locally_isothermal {
            $d + 2
        } else {
            $d + 1
        };
        if let Some(row) = prims.first() {
            if row.len() < want {
                return Err(format!(
                    "isothermal prim row has {} components, expected {want} (rho, v1..v{}{})",
                    row.len(),
                    $d,
                    if cfg.locally_isothermal {
                        ", p_local"
                    } else {
                        ""
                    },
                ));
            }
        }
        let origin: [f64; $d] = std::array::from_fn(|ax| cfg.x_lo[ax]);
        let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);

        let sim = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .coord_maps(axis_maps::<$d>(cfg))
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
                    return Err(
                        "user source expressions are not yet supported with mesh refinement"
                            .to_string(),
                    );
                }
                let scfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("source expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_user_source(
                    &scfg,
                    <IsoNewtonian as Regime<f64, $d>>::SPEC,
                )
                .map_err(|e| format!("source expression lower: {e}"))?;
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
                    if coord[ax] < interior.spaces[ax].hi {
                        break;
                    }
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
                prolong_field(
                    &lo[ll - 1].kernels.cs2,
                    &zero,
                    &hi[0].kernels.cs2,
                    &region,
                    order,
                    0.0,
                );
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
            // C4/M9 fail-loud guard: isothermal hydro has NO baked GR kernels, so any non-minkowski
            // spacetime must fail loud rather than silently run flat.
            (d, c) if $cfg.spacetime != "minkowski" => Err(format!(
                "isothermal hydro has no GR kernels; (dims={d}, coords={c}, spacetime={}) is unsupported \
                 — refusing to run silently on a flat Minkowski metric. use spacetime=minkowski.",
                $cfg.spacetime
            )),
            (1, "cartesian") => build_and_run_iso!($cfg, $prims, 1, Cartesian, Cartesian),
            (2, "cartesian") => build_and_run_iso!($cfg, $prims, 2, Cartesian, Cartesian),
            (3, "cartesian") => build_and_run_iso!($cfg, $prims, 3, Cartesian, Cartesian),
            (1, "spherical") => build_and_run_iso!($cfg, $prims, 1, Spherical, Spherical),
            (2, "spherical") => build_and_run_iso!($cfg, $prims, 2, Spherical, Spherical),
            (3, "spherical") => build_and_run_iso!($cfg, $prims, 3, Spherical, Spherical),
            (1, "cylindrical") => build_and_run_iso!($cfg, $prims, 1, Cylindrical, Cylindrical),
            (2, "cylindrical") => build_and_run_iso!($cfg, $prims, 2, Cylindrical, Cylindrical),
            (3, "cylindrical") => build_and_run_iso!($cfg, $prims, 3, Cylindrical, Cylindrical),
            (d, g) => Err(format!(
                "no isothermal dispatch arm for (dims={d}, coord={g}) yet"
            )),
        }
    };
}

/// runtime dispatch on the config tags → a monomorphized sim. hydro regimes
/// (newtonian/rhd/isothermal) x cartesian (+ curvilinear for adiabatic) x 1/2/3d;
/// the mhd regimes (srmhd/nmhd/imhd) x cartesian x 1/2/3d.
fn dispatch_and_run(cfg: &Config, prims: &[Vec<f64>], bfields: &[Vec<f64>]) -> Result<(), String> {
    // static mesh refinement is wired for hydro (incl. globally-isothermal). the
    // two cases still pending need extra fine-level prolongation:
    if cfg.refinement_enabled
        && cfg.regime.contains("mhd")
        && !(cfg.dims == 3 && cfg.coord_system == "cartesian")
    {
        return Err("mhd refinement requires a 3d cartesian grid (the CT \
                    reflux assumes 1/dx curl coefficients)"
            .to_string());
    }
    // mesh motion is single-grid uniform-spacing hydro only in this pass.
    if cfg.mesh_motion {
        if cfg.refinement_enabled {
            return Err("mesh motion is single-grid only (not wired with refinement)".to_string());
        }
        if cfg.regime.contains("mhd") {
            return Err(
                "mesh motion is not wired for MHD (comoving-field convention pending)".to_string(),
            );
        }
    }
    // immersed bodies attach to level 0; the AMR body sync (finest-owns-bodies)
    // is not wired through the binding yet, so refined body runs are rejected.
    if !cfg.bodies.is_empty() && cfg.refinement_enabled {
        return Err("immersed bodies are single-grid only in the binding \
                    (AMR body sync not wired yet)"
            .to_string());
    }
    // gpus>1 takes the decomposed run loop: single-level hydro (newtonian/rhd/isothermal) and
    // single-level MHD (srmhd/nmhd/imhd, the oracle-proven staggered-CT halo exchange + face
    // gather, docs/design/37 M4). reject every other case HERE so a multi-gpu request never
    // silently runs on one device.
    if cfg.n_gpus > 1 {
        if !matches!(
            cfg.regime.as_str(),
            "newtonian" | "rhd" | "isothermal" | "srmhd" | "nmhd" | "imhd"
        ) {
            return Err(format!(
                "gpus>1 is wired for hydro (newtonian, rhd, isothermal) and mhd (srmhd, nmhd, \
                 imhd); regime '{}' runs single-gpu for now (set gpus=1)",
                cfg.regime
            ));
        }
        if cfg.mesh_motion {
            return Err("gpus>1 does not yet support mesh motion (moving mesh); set gpus=1".to_string());
        }
        // immersed bodies (incl. moving binaries) and their force/accreted-mass diagnostics are
        // wired for gpus>1: the decomposed loop applies the body source per tile, sums the backward
        // feedback across tiles, and advances the prescribed orbit identically (oracle-proven by
        // decomp_body_equivalence). no refusal needed here.
    }
    // a curved spacetime is a RELATIVISTIC construct: only the relativistic regimes compose
    // with it (the non-relativistic kernel rows are never baked with a spacetime slug).
    if cfg.spacetime != "minkowski"
        && !matches!(cfg.regime.as_str(), "rhd" | "srmhd")
    {
        return Err(format!(
            "spacetime '{}' requires a relativistic regime (rhd or srmhd); got '{}'",
            cfg.spacetime, cfg.regime
        ));
    }
    match cfg.regime.as_str() {
        "newtonian" => hydro_dispatch!(cfg, prims, Newtonian, Newtonian),
        "rhd" => hydro_dispatch!(cfg, prims, Rhd, Rhd),
        "isothermal" => iso_dispatch!(cfg, prims),
        "srmhd" => mhd_dispatch!(cfg, prims, bfields, Rmhd, Rmhd),
        "nmhd" => mhd_dispatch!(cfg, prims, bfields, NewtonianMhd, NewtonianMhd),
        "imhd" => imhd_dispatch!(cfg, prims, bfields),
        other => Err(format!("regime '{other}' not wired yet")),
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
    cfg: &Config,
    idx_width: usize,
    time_width: usize,
    time: f64,
    index: u64,
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
/// the terse `<zones>.chkpt.<time>.h5` form (e.g., `262144.chkpt.000_500.h5`).
fn checkpoint_name(cfg: &Config, tnow: &str) -> String {
    let label = sanitize_unit_label(&cfg.time_unit_label);
    let unit = if label.is_empty() || label == "t" {
        String::new()
    } else {
        format!(".{label}")
    };
    format!(
        "{}{}.chkpt.{tnow}{unit}.h5",
        cfg.data_dir,
        resolution_tag(cfg)
    )
}

/// the per-axis resolution tag for a checkpoint name: the interior cell counts
/// joined by `x` (the standard resolution notation, distinct from the `_`
/// decimal in the time). e.g., 1d 100 -> "100", 2d 256x256 -> "256x256",
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
    label
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
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

// validate a multi-gpu request (`Config.n_gpus`) before any heavy work. phase A (docs/design/
// 37, 38): gpus==1 is the only wired path; gpus>1 is validated and rejected with the PRECISE
// reason -- a cpu build, too few visible devices, or the decomposed run loop (M4) not yet
// wired -- so the user gets an actionable message instead of a silent single-device run.
fn validate_gpu_request(n_gpus: usize) -> Result<(), String> {
    if n_gpus <= 1 {
        return Ok(());
    }
    #[cfg(not(feature = "gpu"))]
    {
        Err(format!(
            "gpus={n_gpus} requested, but this is a cpu build. multi-gpu needs a gpu build: \
             `./dev.py install --gpu` (nvidia) or `--hip` (amd)."
        ))
    }
    #[cfg(feature = "gpu")]
    {
        let avail = symbi::symbi_xpu::device_count().unwrap_or(0) as usize;
        if n_gpus > avail {
            // OVERSUBSCRIBE escape hatch (docs/design/37 M2): fold N logical devices onto the
            // available physical ones (distinct contexts via the modulo map in cuda.rs/hip.rs).
            // no real parallelism, but it lets the WHOLE decomposed path (build + scatter +
            // evolve + gather + checkpoint) be validated on a single card -- run the same problem
            // at gpus=1 and gpus=2 and diff the checkpoints. opt-in so a genuine "too few gpus"
            // misconfiguration on a cluster still errors loudly.
            if std::env::var("SYMBI_GPU_OVERSUBSCRIBE").is_err() {
                return Err(format!(
                    "gpus={n_gpus} requested, but only {avail} gpu(s) are visible. select/limit \
                     with CUDA_VISIBLE_DEVICES or ROCR_VISIBLE_DEVICES, lower gpus, or set \
                     SYMBI_GPU_OVERSUBSCRIBE=1 to fold the logical devices onto the {avail} \
                     physical one(s) for a correctness test."
                ));
            }
            eprintln!(
                "warning: gpus={n_gpus} > {avail} visible; oversubscribing onto {avail} physical \
                 device(s) (SYMBI_GPU_OVERSUBSCRIBE) -- correctness check only, no speedup."
            );
        }
        // the decomposed run loop is wired for hydro (docs/design/37 M4); per-regime support is
        // enforced in `dispatch_and_run` so non-hydro regimes error instead of silently falling
        // back to one device.
        Ok(())
    }
}

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
    // fail fast on an unsupported multi-gpu request, before draining generators or allocating.
    validate_gpu_request(cfg.n_gpus).map_err(PyRuntimeError::new_err)?;
    // mesh refinement is cartesian + uniform-spacing ONLY: the coarse-fine prolongation/restriction
    // transfer kernels are geometry-agnostic (equal index-based sub-cells), correct solely for
    // uniform-volume cells. a curvilinear grid (variable r^2 / r cell volumes) or a non-linear axis
    // (unequal sub-cells) would get silently-wrong transfers, so reject it loudly instead.
    if cfg.refinement_enabled {
        if cfg.coord_system != "cartesian" {
            return Err(PyValueError::new_err(format!(
                "mesh refinement is cartesian-only (the coarse-fine transfer ignores curvilinear \
                 cell volumes); got coord_system = '{}'",
                cfg.coord_system
            )));
        }
        if !cfg.x1_spacing.eq_ignore_ascii_case("linear") {
            return Err(PyValueError::new_err(format!(
                "mesh refinement requires uniform (linear) cell spacing (the coarse-fine transfer \
                 assumes equal sub-cells); got x1_spacing = '{}'",
                cfg.x1_spacing
            )));
        }
    }
    // multi-gpu domain decomposition splits the SAME grid across tiles; a log axis needs each
    // tile's local origin offset to the global log position (start*10^(global_lo*slope)), which is
    // not yet wired. reject rather than evolve tiles on a mismatched uniform geometry.
    if cfg.n_gpus > 1 && !cfg.x1_spacing.eq_ignore_ascii_case("linear") {
        return Err(PyValueError::new_err(format!(
            "multi-gpu decomposition does not yet support non-linear ('{}') cell spacing (the \
             per-tile log origin offset is unwired); run single-gpu for log-spaced grids",
            cfg.x1_spacing
        )));
    }
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

/// read-only live monitor: poll `<rundir>/.simbi-live/snapshot.bin` (written by a
/// run started with `live_monitor = true`) and render the dashboard until the user
/// quits or Ctrl-C. blocks on a dedicated terminal — release the gil so the
/// signal + render threads run and python stays interruptible.
#[pyfunction]
#[pyo3(signature = (rundir, poll_ms = 250))]
fn attach_dashboard(py: Python<'_>, rundir: String, poll_ms: u64) -> PyResult<()> {
    py.allow_threads(|| symbi_display::run_attach(std::path::Path::new(&rundir), poll_ms))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

// shared module body. the pyo3 entry-point name below decides the `PyInit_*`
// symbol and the imported module name: `cpu_ext` for the default build,
// `gpu_ext` for the cuda build. both compile the SAME source — cuda only adds
// the NVRTC device path — so the registration is identical and lives here.
fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(attach_dashboard, m)?)?;
    afterglow::register(m)?;
    Ok(())
}

// cpu build -> `simbi.libs.cpu_ext`.
#[cfg(not(feature = "gpu"))]
#[pymodule]
fn cpu_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)
}

// gpu build (cuda or hip) -> `simbi.libs.gpu_ext`. dev.py overrides maturin's module-name to
// match (`--config tool.maturin.module-name="simbi.libs.gpu_ext"`), so the cpu and gpu
// backends coexist instead of overwriting the same `cpu_ext` dylib.
#[cfg(feature = "gpu")]
#[pymodule]
fn gpu_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)
}
