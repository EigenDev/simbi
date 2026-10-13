// =============================================================================
// lib.rs
//
// python extension module bridging the python frontend to the rust solver.
// reproduces the pybind11 `cpu_ext.run_simulation` contract exactly so the
// existing `simbi` python package calls into rust with zero frontend changes:
// - parse the `sim_info` dict (pydantic `to_execution_dict`) into a plain Config
// - drain the `prim_gen` python iterator into a typed primitive buffer
// - release the gil, dispatch on (regime, dims, geometry, eos), run, checkpoint
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
use pyo3::types::{PyDict, PyList};
use symbi_hydro::quantity::{Density, Pressure};

use symbi::prelude::*;
use symbi::sim::decomp::{Partition, Topology};
use symbi::sim::refinement::transfer::prolong_field;
use symbi::sim::refinement::{Hierarchy, ProlongOrder, RefinementRegion};
use symbi::symbi_grid::Field;
use symbi_algebra::Tensor;
use symbi_display::{
    Colormap, ExitKind, FieldSlice, LiveDashboard, ScreenGuard, SignalGuard, Table,
};
use symbi_geometry::MotionState;
use symbi_geometry::{
    KerrKS, KerrKSCartesian, KerrKSCylindrical, SchwarzschildKS, SchwarzschildKSCartesian,
    SchwarzschildKSCylindrical, SchwarzschildKSCylindricalRz,
};
use symbi_hydro::energy::IsoModel;
use symbi_hydro::eos::Eos;
use symbi_hydro::isothermal::IsoNewtonian;
use symbi_hydro::mhd_state::MhdPrimG;
use symbi_hydro::regime::Regime;
use symbi_hydro::state::PrimG;
use symbi_ib::{Body, BodyCollection, BodyKind};
use symbi_io::Metadata;
use symbi_sim::checkpoint::{time_at_or_after, write_hierarchy_checkpoint};
use symbi_sim::placement::{PlacedTile, WholeGrid};
use symbi::regimes::substrate_kernels::GradientBc;
use symbi_sim::state::CtMethod;
use symbi_sim::state::SimStateGeneric;
use symbi_sim::substrate_seam::{WithExcision, WithResistivity, WithViscosity};

const BUILD_GIT_SHA: &str = env!("SIMBI_BUILD_GIT_SHA");
const BUILD_GIT_DIRTY: bool = env!("SIMBI_BUILD_GIT_DIRTY").as_bytes()[0] == b'1';

fn build_source_id() -> String {
    if BUILD_GIT_DIRTY {
        format!("{BUILD_GIT_SHA}-dirty")
    } else {
        BUILD_GIT_SHA.to_string()
    }
}

fn build_source_label() -> String {
    let short: String = BUILD_GIT_SHA.chars().take(12).collect();
    if BUILD_GIT_DIRTY {
        format!("{short}-dirty")
    } else {
        short
    }
}

// =============================================================================
// parsed configuration — a plain-rust mirror of the python exec_dict. the
// monomorphized dispatch below reads these tags to pick the concrete SimState.
// =============================================================================

struct Config {
    name: String,
    /// defining Python configuration file and its content hash. unlike the backend build
    /// identity, these are captured when the run starts because changing a config needs no rebuild.
    config_source: String,
    config_sha256: String,
    regime: String,
    coord_system: String,
    // the spacetime background ("minkowski" default, "schwarzschild" for GR), orthogonal to
    // coord_system; selects the Schwarzschild metric (lapse/densitization/GR-wavespeed kernels).
    spacetime: String,
    // the Schwarzschild geometric mass M (G = c = 1); only meaningful when spacetime = schwarzschild.
    schwarzschild_mass: f64,
    kerr_spin: f64,
    max_dt: f64,
    // the retry guard's thresholds; the built-in values when the problem names none
    retry_max_rejections: u32,
    retry_severe_ratio: f64,
    retry_sustained_steps: u32,
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
    // an initial-condition perturbation as `SourceConfig` json: a position expression
    // supplying a delta on each primitive component, evaluated at each level's own cell
    // centers. the cell generator fills the root grid alone and fine levels inherit it by
    // prolongation, which carries nothing below the root's nyquist; a declared expression
    // is the seam for initial data whose content is finer than the root can represent.
    perturbation_json: Option<String>,
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
    eos_name: String,
    ppm_flatten_onset: f64,
    ppm_flatten_full: f64,
    timestepping: Timestepping,
    plm_theta: f64,
    wb_reconstruction: bool,
    dlogt: f64,
    viscosity: f64,
    /// the shakura-sunyaev alpha-disk coefficient, read from the `viscosity_alpha` key,
    /// not the bare `alpha` key: `alpha` is a generic name a problem may use for its own
    /// quantity (a wave amplitude, a slope), and reading it here would switch on a viscosity
    /// nobody asked for. see `bare_alpha_key`.
    alpha: f64,
    /// whether the config carried a bare `alpha` key. only used to refuse the ambiguous case
    /// loudly: a viscous regime with `alpha` but no `viscosity_alpha` is inherently ambiguous.
    bare_alpha_key: bool,
    resistivity: f64,
    // horizon-excision sphere radius about the chart origin (cartesian kerr-schild,
    // 2d or 3d); 0 disables excision. must sit inside the horizon r_+ = 2M.
    excision_radius: f64,
    excision_rho_scale: f64,
    excision_pre_scale: f64,
    x1_spacing: String,
    x1_spacing_ratio: f64,
    x2_spacing: String,
    x2_spacing_ratio: f64,
    x3_spacing: String,
    x3_spacing_ratio: f64,
    start_time: f64,
    // the log-checkpoint anchor (positive reference for log-spaced cadence). distinct from
    // start_time, which is the physical/resume clock (= checkpoint time on restart). 0 = unset ->
    // fall back to start_time (the common case where they coincide).
    checkpoint_log_anchor: f64,
    checkpoint_index: u64,
    t_final: f64,
    // stop after this many root iterations; 0 = march to t_final. a bounded run
    // exits through the ordinary success path (final checkpoint written), so a
    // smoke gate or profiling probe sees a truncated but otherwise normal run.
    max_steps: u64,
    checkpoint_interval: f64,
    data_dir: String,
    restart_path: Option<String>,
    // natural time unit for checkpoint names + display: reported time is
    // `time / time_unit`, labeled `time_unit_label` ("t" = code units).
    time_unit: f64,
    time_unit_label: String,
    // immersed bodies (gravity / accretion sinks) parsed from the config's
    // `immersed_bodies` list; empty for body-free runs. dimension-agnostic raw
    // form — the typed `BodyCollection<f64, D>` is built per-dim at sim build.
    bodies: Vec<BodyParams>,
    bonded_assembly: Option<BondedAssemblyParams>,
    /// mass-transport tracer count (0 = none): deterministic mass-weighted
    /// seeding over the initial interior density.
    n_tracers: usize,
    tracer_scheme: String,
    // the passive-scalar (dye) initial condition: one value per interior cell,
    // axis-0-fastest, drained from the python `passive_scalar` generator.
    // empty = the run carries no dye; populated after parse when run_simulation drains
    // the generator, independent of the sim_info dict.
    chi_ic: Vec<f64>,
    // immutable initial-material cohort per interior cell.
    cohort_ic: Vec<u16>,
    // the prescribed binary orbit (`body_system.binary_config`), if any; attaches the Keplerian
    // orbit to the body collection so the two components orbit each other. None for non-binary runs.
    binary: Option<BinaryCfg>,
    // ordered user source expressions in the rust `SourceConfig` wire format.
    // contributions are lowered, grouped by target, and added on the hydro path.
    source_jsons: Vec<SourcePayload>,
    // registered binned reductions in the rust `CensusConfig` wire format. each is a
    // pointwise map plus a segmented reduce, emitted as a time series in the checkpoint.
    census_jsons: Vec<SourcePayload>,
    // mesh-motion scale-factor law a(t)/a_dot(t) as the `serialize_motion` wire (json), or None.
    // when present the time loop evaluates it exactly each (sub)stage (no linearization).
    motion_json: Option<String>,
    // the run's stationary target state in the rust `EquilibriumConfig` wire format, or None.
    // when present the scheme measures the target's discrete imbalance once per level and
    // subtracts it back every stage, keeping the target an exact fixed point, immune to the
    // truncation-order drift a plain state would show — most visibly across a coarse-fine
    // interface, where the two grids reduce the same exact solution to different face values.
    equilibrium_json: Option<String>,
    // whether to seed every level from that target before evolving. the state a refined hierarchy
    // holds exactly has covered cells carrying the restriction of the finer target, a value an
    // independently sampled profile only reproduces by coincidence.
    seed_from_equilibrium: bool,
    // driven (dynamic) boundary prescriptions as `SourceConfig` json, in Driven-id order
    // (driven_exprs[id] <-> the face marked BoundaryType::Driven(id)). lowered against each
    // regime's spec at sim build: every regime prescribes the full ghost primitive state; the
    // MHD build additionally prescribes the ghost cell B. rejected with mesh refinement.
    driven_exprs: Vec<String>,
    // gradient (Neumann / Robin) boundary coefficients, in registry order (gradient_bcs[id] <-> the
    // face marked BoundaryType::Neumann(id) / Robin(id)). the convenience prescribed-gradient wall.
    gradient_bcs: Vec<GradientBc>,
    // the config author's own params (subclass fields), grouped, for the live dashboard's
    // problem-setup panel: each is [group, label, value].
    custom_params: Vec<[String; 3]>,
    // body-diagnostic output cadence in natural units (x time_unit -> code);
    // 0 disables the diagnostics file.
    diagnostic_interval: f64,
    // number of gpus to decompose the domain across, intra-node. 1 = single device;
    // >1 splits the grid into that many tiles evolved in lockstep with halo exchange.
    n_gpus: usize,
    // explicit per-axis decomposition cuts (interior cell indices, strictly increasing
    // per axis; one list per dimension). empty = the balanced longest-axis heuristic.
    // the tile count the cuts imply must equal `n_gpus`: each tile binds one device.
    decompose: Vec<Vec<usize>>,
}

struct SourcePayload {
    origin: String,
    json: String,
}

/// dimension-agnostic raw body parameters from the python `immersed_bodies`
/// list. `capability` is the BodyCapability bitflag (gravitational=1,
/// accretion=2). accretion fields are only meaningful when the accretion bit
/// is set (a black-hole sink); otherwise the body is a fixed-potential mass.
struct BodyParams {
    capability: u64,
    mass: f64,
    radius: f64,
    position: Vec<f64>,
    velocity: Vec<f64>,
    softening: f64,
    softening_kind: f64,
    accretion_radius: f64,
    sink_rate: f64,
    /// the porous-surface dial: None keeps the pure drain.
    porosity: Option<f64>,
    k_eta_n: f64,
    k_eta_t: f64,
    /// the torque-free dial: Some(xi) selects the isothermal
    /// torque-free accretor (xi in [0, 1]); None keeps the pure drain. mutually
    /// exclusive with porosity.
    torque_free_xi: Option<f64>,
    /// rigid-wall moment of inertia (capability rigid) — carried for the future
    /// two-way rotational coupling; unused by a static obstacle.
    inertia: f64,
    /// rigid-wall no-slip flag: true relaxes the tangential velocity to the body
    /// (no slip), false is a free-slip wall (the tangential channel is off).
    no_slip: bool,
    /// the rigid-wall shape as a `SdfExpr` json wire (`body["rigid"]["shape"]["wire"]`),
    /// or None for the analytic sphere. a `Some` routes the body to the runtime-JIT'd
    /// arbitrary-shape penalization kernel.
    shape_json: Option<String>,
    /// the prescribed spin rate (radians/time) about `spin_axis`; nonzero makes a shaped wall rotate.
    omega: f64,
    /// the (unit) spin axis; default z.
    spin_axis: [f64; 3],
    /// optional principal moments (I1,I2,I3); all-zero = unspecified (isotropic from `inertia`).
    inertia_principal: [f64; 3],
    /// whether the gas reaction force acts back on the body. black-hole sinks
    /// always feel feedback; this dial adds it to non-accreting gravitating
    /// masses (BodyCollection gates feedback on this flag or the sink kind).
    two_way_coupling: bool,
    /// the body's Ohmic resistivity `eta` (`MagneticSpec::Resistive`): a magnetized immersed sink that
    /// dissipates the field threading it. None = magnetically transparent (`MagneticSpec::None`).
    magnetic_resistivity: Option<f64>,
    /// the force-selective magnetic slip (`MagneticSpec::Slip`), read from the `magnetic.slip` group.
    /// a body carries at most one magnetic coupling; the resistive and slip wires are exclusive.
    magnetic_slip: Option<MagneticSlipParams>,
}

/// the magnetic-slip scales as the body wire carries them under `magnetic.slip`: the Damkohler
/// ratio D_B, the shell mollification width w, the transport-length ratio ell_B / w, the null
/// regularization B_0, and the signed shell placement in units of w. every key is required.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MagneticSlipParams {
    diffusivity_ratio: f64,
    shell_width: f64,
    slip_length_ratio: f64,
    field_regularization: f64,
    placement: f64,
}

/// the public-boundary contract of a magnetic-slip body: exactly one magnetic coupling, a run the
/// slip operator is built for (adiabatic Newtonian MHD on a cartesian grid, 3D or 2.5D), and a
/// surface that removes mass, since the slip coefficient closes on that drain's timescale tau_rho.
fn validate_magnetic_slip_bodies(
    bodies: &[BodyParams],
    regime: &str,
    dims: usize,
    coord_system: &str,
    locally_isothermal: bool,
) -> Result<(), String> {
    const ACCRETION: u64 = 2;
    for (idx, b) in bodies.iter().enumerate() {
        if b.magnetic_slip.is_none() {
            continue;
        }
        if b.magnetic_resistivity.is_some_and(|eta| eta > 0.0) {
            return Err(format!(
                "immersed body {idx} declares both a resistivity and the magnetic slip; a body \
                 carries exactly one magnetic coupling"
            ));
        }
        // the slip runs under two thermal closures: adiabatic Newtonian MHD, where the released
        // magnetic energy heats the gas, and isothermal MHD, where the closure exports it to the
        // cooling bath and the body books it. an adiabatic run flagged locally isothermal is
        // neither and is refused.
        let closure_ok = (regime == "nmhd" && !locally_isothermal) || regime == "imhd";
        if !closure_ok {
            return Err(format!(
                "immersed body {idx}: the magnetic slip runs in adiabatic Newtonian MHD (regime \
                 nmhd, not locally isothermal) or in isothermal MHD (regime imhd); this run is \
                 regime {regime} (locally_isothermal {locally_isothermal})"
            ));
        }
        if !(dims == 3 || dims == 2) || coord_system != "cartesian" {
            return Err(format!(
                "immersed body {idx}: the magnetic slip runs on a cartesian grid in 3D or in 2.5D \
                 (an x-y grid with three vector components); this run is {}D {}",
                dims, coord_system
            ));
        }
        let drains = b.capability & ACCRETION != 0
            && (b.torque_free_xi.is_some() || b.porosity.is_none_or(|p| p > 0.0));
        if !drains {
            return Err(format!(
                "immersed body {idx}: the magnetic slip closes its coefficient on the sink's drain \
                 time, so the body must remove mass (capability ACCRETION with a plain drain, a \
                 torque-free drain, or porosity > 0)"
            ));
        }
    }
    Ok(())
}

/// the public-boundary contract of a magnetized drain: a mass-removing body under any MHD regime
/// relaxes each cell at its local Alfven rate through a kernel baked for the cartesian grid in 3D
/// or 2.5D, so every other chart is refused here, before the grid exists.
fn validate_magnetized_drain_bodies(
    bodies: &[BodyParams],
    regime: &str,
    dims: usize,
    coord_system: &str,
) -> Result<(), String> {
    const ACCRETION: u64 = 2;
    if !regime.ends_with("mhd") {
        return Ok(());
    }
    for (idx, b) in bodies.iter().enumerate() {
        let drains = b.capability & ACCRETION != 0
            && (b.torque_free_xi.is_some() || b.porosity.is_none_or(|p| p > 0.0));
        if !drains {
            continue;
        }
        if !(dims == 3 || dims == 2) || coord_system != "cartesian" {
            return Err(format!(
                "immersed body {idx}: a magnetized drain relaxes at its local Alfven rate on a \
                 cartesian grid in 3D or in 2.5D (an x-y grid with three vector components); this \
                 run is {dims}D {coord_system} under regime {regime}"
            ));
        }
    }
    Ok(())
}

/// immersed bodies on a refined staggered-field hierarchy: adiabatic Newtonian MHD or isothermal
/// MHD on a 3D or 2.5D cartesian grid, every body a drain (transparent, resistive, or magnetic-slip coupling) with
/// no rigid surface. the hierarchy itself checks that each sink's operator support lies inside
/// the finest level once the grid exists.
fn validate_refined_mhd_bodies(
    bodies: &[BodyParams],
    regime: &str,
    dims: usize,
    coord_system: &str,
    locally_isothermal: bool,
) -> Result<(), String> {
    const RIGID: u64 = 1 << 4;
    let closure_ok = (regime == "nmhd" && !locally_isothermal) || regime == "imhd";
    if !closure_ok {
        return Err(format!(
            "immersed bodies with refinement run in adiabatic Newtonian MHD (regime nmhd, not \
             locally isothermal) or in isothermal MHD (regime imhd); this run is regime {regime} \
             (locally_isothermal {locally_isothermal})"
        ));
    }
    if !(dims == 3 || dims == 2) || coord_system != "cartesian" {
        return Err(format!(
            "immersed bodies with refinement in MHD run on a 3D or 2.5D cartesian grid; this run is \
             {dims}D {coord_system}"
        ));
    }
    for (idx, b) in bodies.iter().enumerate() {
        if b.capability & RIGID != 0 {
            return Err(format!(
                "immersed body {idx} carries a rigid surface; a rigid body on a refined staggered \
                 grid is refused (the wall would meet a coarse-fine face)"
            ));
        }
    }
    Ok(())
}

fn validate_porous_body_overlaps(bodies: &[BodyParams]) -> Result<(), String> {
    const ACCRETION: u64 = 2;
    const RIGID: u64 = 1 << 4;
    let mask_radius = |body: &BodyParams| {
        if body.capability & ACCRETION != 0 {
            body.accretion_radius
        } else if body.capability & RIGID != 0 {
            body.radius
        } else {
            0.0
        }
    };
    for ii in 0..bodies.len() {
        for jj in ii + 1..bodies.len() {
            if bodies[ii].porosity.is_none() && bodies[jj].porosity.is_none() {
                continue;
            }
            let radius_sum = mask_radius(&bodies[ii]) + mask_radius(&bodies[jj]);
            if radius_sum <= 0.0 {
                continue;
            }
            let distance_sq = (0..3)
                .map(|axis| {
                    let left = bodies[ii].position.get(axis).copied().unwrap_or(0.0);
                    let right = bodies[jj].position.get(axis).copied().unwrap_or(0.0);
                    (left - right) * (left - right)
                })
                .sum::<f64>();
            if distance_sq < radius_sum * radius_sum {
                return Err(format!(
                    "porous immersed-body masks {ii} and {jj} overlap: sequential wall targets \
                     are order-dependent; separate the masks or represent them as one CSG body"
                ));
            }
        }
    }
    Ok(())
}

/// a bonded-fragment assembly from the `bonded_assembly` config key: a cluster
/// of wall-only rigid spherical fragments (sealed porous surfaces) joined by
/// breakable elastic bonds, with optional soft-sphere contact and mutual
/// gravity. fragments interact through this mutual-gravity/contact/bond path exclusively,
/// separate from the baked gravity/accretion source fan.
struct BondedAssemblyParams {
    positions: Vec<Vec<f64>>,
    masses: Vec<f64>,
    radii: Vec<f64>,
    inertias: Vec<f64>,
    velocities: Vec<Vec<f64>>,
    mobile: Vec<bool>,
    bonds: Vec<(usize, usize)>,
    bond_material: symbi_ib::BondMaterial,
    contact: Option<symbi_ib::ContactMaterial>,
    gravity: Option<symbi_ib::MutualGravity>,
    k_eta_n: f64,
    k_eta_t: f64,
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
/// `json.dumps`, so the node DAG crosses the boundary as a single serialized
/// string, in place of a hand-written PyDict -> serde walk.
fn get_source_json(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    let Some(obj) = dict.get_item(key)? else {
        return Ok(None);
    };
    // skip the empty-dict default (`return {}` in the base SimbiProblem).
    if let Ok(d) = obj.cast::<PyDict>() {
        if d.is_empty() {
            return Ok(None);
        }
    }
    let json = obj.py().import("json")?;
    let s: String = json.call_method1("dumps", (obj,))?.extract()?;
    Ok(Some(s))
}

/// read the registered censuses (already in the rust `CensusConfig` wire format,
/// emitted by python's `Census.serialize`) as json strings. same shape as the source
/// intake: the payload crosses through `json.dumps` as a single serialized string, in
/// place of a hand-written PyDict walk.
fn get_census_jsons(dict: &Bound<'_, PyDict>) -> PyResult<Vec<SourcePayload>> {
    let mut censuses = Vec::new();
    if let Some(obj) = dict.get_item("census_expressions")? {
        let list = obj.cast::<PyList>().map_err(|_| {
            PyValueError::new_err("census_expressions must be a list of census payloads")
        })?;
        let json = obj.py().import("json")?;
        for (index, census) in list.iter().enumerate() {
            censuses.push(SourcePayload {
                origin: format!("census_expressions[{index}]"),
                json: json.call_method1("dumps", (census,))?.extract()?,
            });
        }
    }
    Ok(censuses)
}

/// parse and lower every registered census, so a malformed binning or an
/// unlowerable expression fails at setup, not at the first sample. duplicate
/// names are refused here because the name is the checkpoint group: two censuses
/// sharing one would have the second silently overwrite the first.
fn lower_configured_censuses(
    census_jsons: &[SourcePayload],
) -> Result<Vec<symbi_sim::census::CensusEvaluator>, String> {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    census_jsons
        .iter()
        .map(|census| {
            let config = symbi_source_compile::CensusConfig::from_json(&census.json)
                .map_err(|error| format!("{} parse: {error}", census.origin))?;
            if !seen.insert(config.name.clone()) {
                return Err(format!(
                    "{} reuses the census name '{}'; each names its own output group",
                    census.origin, config.name
                ));
            }
            symbi_sim::census::CensusEvaluator::new(&config)
                .map_err(|error| format!("{} lower: {error}", census.origin))
        })
        .collect()
}

fn get_source_jsons(dict: &Bound<'_, PyDict>) -> PyResult<Vec<SourcePayload>> {
    let mut sources = Vec::new();
    if let Some(obj) = dict.get_item("source_expressions")? {
        let list = obj.cast::<PyList>().map_err(|_| {
            PyValueError::new_err("source_expressions must be a list of source payloads")
        })?;
        let json = obj.py().import("json")?;
        for (index, source) in list.iter().enumerate() {
            sources.push(SourcePayload {
                origin: format!("source_expressions[{index}]"),
                json: json.call_method1("dumps", (source,))?.extract()?,
            });
        }
    }
    Ok(sources)
}

/// uniform runtime-source attach across the substrate kernel sets the hydro
/// dispatch macro instantiates. the macro body monomorphizes for every regime it
/// covers (newtonian/adiabatic and rhd), but `with_runtime_source` is inherent
/// only on the substrates that carry a source slot — so the call must go through
/// a trait every one of them implements. the relativistic set has no slot yet and
/// reports a clear runtime error.
trait AttachRuntimeSource: Sized {
    fn attach_runtime_source(
        self,
        built: symbi_source_compile::AdmittedSources,
        params: Vec<f64>,
    ) -> Result<Self, String>;
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    AttachRuntimeSource for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    fn attach_runtime_source(
        self,
        built: symbi_source_compile::AdmittedSources,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        // fused by default: the user source (and any immersed body) rides inside the godunov stage on
        // host + f64; other carriers / device transparently fall back to the two-pass (bit-identical).
        Ok(self.with_fused_runtime_source(built, params))
    }
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    AttachRuntimeSource for RhdSubstrateKernelSet<Mem, Sc, D>
{
    fn attach_runtime_source(
        self,
        built: symbi_source_compile::AdmittedSources,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        // fused by default on a flat host+f64 run; GR / device / non-f64 fall back to the two-pass.
        Ok(self.with_fused_runtime_source(built, params))
    }
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    AttachRuntimeSource for IsoSubstrateKernelSet<Mem, Sc, D>
{
    fn attach_runtime_source(
        self,
        built: symbi_source_compile::AdmittedSources,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        // fused by default on host + f64 (iso has no energy, so the body stays its own pass — the
        // Cartesian body has its own baked fused kernel); device / non-f64 fall back to the two-pass.
        Ok(self.with_fused_runtime_source(built, params))
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
        built: symbi_source_compile::AdmittedSources,
        params: Vec<f64>,
    ) -> Result<Self, String> {
        Ok(self.with_runtime_source(built, params))
    }
}

/// enable pointwise-source fusion on a source-free substrate — a run with immersed bodies but no user
/// source, where `attach_runtime_source` (which sets the fusion flag) goes uncalled. real only for the
/// adiabatic set (its energy-regime body folds into godunov); a no-op elsewhere (iso folds its body via
/// its own baked kernel; rhd/mhd have no host fused source path). the fused path self-gates on host+f64.
trait EnableSourceFusion: Sized {
    fn enable_source_fusion(self) -> Self;
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    EnableSourceFusion for AdiabaticSubstrateKernelSet<Mem, Sc, D>
{
    fn enable_source_fusion(self) -> Self {
        self.with_source_fusion()
    }
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    EnableSourceFusion for RhdSubstrateKernelSet<Mem, Sc, D>
{
    fn enable_source_fusion(self) -> Self {
        self
    }
}

impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize>
    EnableSourceFusion for IsoSubstrateKernelSet<Mem, Sc, D>
{
    fn enable_source_fusion(self) -> Self {
        self
    }
}

impl<R, Mem, Sc, const D: usize> EnableSourceFusion
    for symbi::regimes::substrate_mhd::MhdSubstrateKernelSet<R, Mem, Sc, D>
where
    R: Regime<Sc, D>,
    Mem: MemorySpace,
    Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric,
{
    fn enable_source_fusion(self) -> Self {
        self
    }
}

/// the conserved state this run's regime builds from primitives, for the sources that
/// relax toward one. the background comes from the config's spacetime and mass, so a
/// source on a curved chart relaxes toward the densitized state the evolution stores
/// rather than its flat shadow.
fn state_law_of(cfg: &Config, spec: &symbi_hydro::RegimeSpec) -> symbi_hydro::state_law::StateLaw {
    use symbi_hydro::state_law::{Background, StateLaw};
    let background = match cfg.spacetime.as_str() {
        "schwarzschild_ks" => Background::SchwarzschildKsCartesian {
            mass: cfg.schwarzschild_mass,
        },
        // a spinning hole drags the frame azimuthally, so its shift is not the
        // non-rotating one and the conserved state built against the wrong shift is
        // wrong in the momentum slots. the spin rides the background rather than
        // being dropped.
        "kerr_ks" => Background::KerrKsCartesian {
            mass: cfg.schwarzschild_mass,
            spin: cfg.kerr_spin,
        },
        _ => Background::Minkowski,
    };
    if spec.is_relativistic {
        StateLaw::relativistic(cfg.gamma, background)
    } else {
        StateLaw::newtonian(cfg.gamma)
    }
}

fn attach_configured_sources<T>(
    substrate: T,
    source_jsons: &[SourcePayload],
    spec: &symbi_hydro::RegimeSpec,
    law: &symbi_hydro::state_law::StateLaw,
) -> Result<T, String>
where
    T: AttachRuntimeSource + EnableSourceFusion,
{
    if source_jsons.is_empty() {
        return Ok(substrate.enable_source_fusion());
    }
    let (built, params) = lower_configured_sources(source_jsons, spec, Some(law))?;
    substrate.attach_runtime_source(built, params)
}

/// the numerics a kernel set carries, read once from the run's configuration for one regime: the
/// slope limiter and reconstruction, the eos and solver, the non-ideal coefficients, the excision,
/// the CT method, and the user sources, driven boundaries and gradient walls in their id order
struct KernelSettings<'c> {
    spec: &'static symbi_hydro::regime_spec::RegimeSpec,
    theta: f64,
    recon: symbi::Recon,
    flatten: (f64, f64),
    well_balanced: bool,
    eos: symbi::EosArm,
    solver: Solver,
    ct_method: CtMethod,
    viscosity: f64,
    alpha: f64,
    resistivity: f64,
    excision: [f64; 3],
    sources: &'c [SourcePayload],
    law: symbi_hydro::state_law::StateLaw,
    driven: &'c [String],
    gradient_walls: &'c [GradientBc],
}

impl<'c> KernelSettings<'c> {
    fn of(cfg: &'c Config, spec: &'static symbi_hydro::regime_spec::RegimeSpec) -> Self {
        Self {
            spec,
            theta: build_theta(cfg),
            recon: build_recon(cfg),
            flatten: (cfg.ppm_flatten_onset, cfg.ppm_flatten_full),
            well_balanced: cfg.wb_reconstruction,
            eos: build_eos(cfg),
            solver: cfg.solver,
            ct_method: cfg.ct_method,
            viscosity: cfg.viscosity,
            alpha: cfg.alpha,
            resistivity: cfg.resistivity,
            excision: [cfg.excision_radius, cfg.excision_rho_scale, cfg.excision_pre_scale],
            sources: &cfg.source_jsons,
            law: state_law_of(cfg, spec),
            driven: &cfg.driven_exprs,
            gradient_walls: &cfg.gradient_bcs,
        }
    }
}

/// the level a kernel set serves in its hierarchy. the root carries the horizon excision; a fine
/// level inside it carries none, since the excised core lives on the root.
#[derive(Clone, Copy, Debug, PartialEq)]
enum LevelRole {
    Root,
    Fine,
}

/// the construction doors both gas kernel sets (newtonian and relativistic hydro) carry as inherent
/// methods, so one gas construction path serves either. each method forwards to the set's own door;
/// the relativistic set accepts the newtonian-only settings at their defaults, which admission
/// guarantees.
trait GasDoors: Sized + WithViscosity + WithResistivity + WithExcision + AttachRuntimeSource + EnableSourceFusion {
    fn slope_theta(self, theta: f64) -> Self;
    fn reconstruct(self, recon: symbi::Recon) -> Self;
    fn flatten(self, onset: f64, full: f64) -> Self;
    fn well_balance(self, on: bool) -> Self;
    fn eos(self, eos: symbi::EosArm) -> Self;
    fn solve_with(self, solver: Solver) -> Result<Self, symbi_sim::state::ConfigError>;
    fn driven(self, built: symbi_source_compile::expr_bridge::BoundaryPrescription, params: Vec<f64>) -> Self;
    fn gradient_wall(self, wall: GradientBc) -> Self;
}

macro_rules! gas_doors {
    ($set:ident) => {
        impl<Mem: MemorySpace, Sc: symbi_hydro::Scalar + symbi_algebra::OrderedNumeric, const D: usize> GasDoors
            for $set<Mem, Sc, D>
        {
            fn slope_theta(self, theta: f64) -> Self {
                $set::theta(self, theta)
            }
            fn reconstruct(self, recon: symbi::Recon) -> Self {
                $set::reconstruction(self, recon)
            }
            fn flatten(self, onset: f64, full: f64) -> Self {
                $set::ppm_flatten(self, onset, full)
            }
            fn well_balance(self, on: bool) -> Self {
                $set::well_balanced_reconstruction(self, on)
            }
            fn eos(self, eos: symbi::EosArm) -> Self {
                $set::with_eos(self, eos)
            }
            fn solve_with(self, solver: Solver) -> Result<Self, symbi_sim::state::ConfigError> {
                $set::with_solver(self, solver)
            }
            fn driven(self, built: symbi_source_compile::expr_bridge::BoundaryPrescription, params: Vec<f64>) -> Self {
                $set::with_driven_boundary(self, built, params).0
            }
            fn gradient_wall(self, wall: GradientBc) -> Self {
                $set::with_gradient_boundary(self, wall).0
            }
        }
    };
}
gas_doors!(AdiabaticSubstrateKernelSet);
gas_doors!(RhdSubstrateKernelSet);

/// the kernel set of one level of a gas run: the limiter, reconstruction and flattening, the
/// balance-aware reconstruction, the eos and solver, viscosity and alpha, the excision on the root,
/// then the user sources, the driven boundaries and the gradient walls in their id order.
fn gas_kernels<K: GasDoors>(sub: K, settings: &KernelSettings, role: LevelRole) -> Result<K, String> {
    let ks = sub
        .slope_theta(settings.theta)
        .reconstruct(settings.recon)
        .flatten(settings.flatten.0, settings.flatten.1)
        .well_balance(settings.well_balanced)
        .eos(settings.eos)
        .solve_with(settings.solver)
        .map_err(|e| format!("substrate/solver: {e:?}"))?
        .with_viscosity(settings.viscosity)
        .with_alpha(settings.alpha)
        .with_resistivity(settings.resistivity);
    let ks = excised(ks, settings, role);
    let ks = attach_configured_sources(ks, settings.sources, settings.spec, &settings.law)?;
    let ks = attach_driven_boundaries(ks, settings.driven, settings.spec, |ks, built, params| ks.driven(built, params))?;
    Ok(attach_gradient_walls(ks, settings.gradient_walls, |ks, wall| ks.gradient_wall(wall)))
}

/// the kernel set of one level of an isothermal hydro run: the limiter, viscosity and alpha, then
/// the user sources, the driven boundaries and the gradient walls in their id order. the local
/// sound-speed closure is seeded from the tile's own cells, outside the construction.
fn iso_kernels<const D: usize>(
    sub: IsoSubstrateKernelSet<DefaultMemory, f64, D>,
    settings: &KernelSettings,
) -> Result<IsoSubstrateKernelSet<DefaultMemory, f64, D>, String> {
    let ks = sub.theta(settings.theta).with_viscosity(settings.viscosity).with_alpha(settings.alpha);
    let ks = attach_configured_sources(ks, settings.sources, settings.spec, &settings.law)?;
    let ks = attach_driven_boundaries(ks, settings.driven, settings.spec, |ks, built, params| {
        ks.with_driven_boundary(built, params).0
    })?;
    Ok(attach_gradient_walls(ks, settings.gradient_walls, |ks, wall| ks.with_gradient_boundary(wall).0))
}

/// the kernel set of one level of a magnetized run (newtonian, relativistic or isothermal MHD): the
/// limiter, the solver and the CT method, viscosity and alpha, resistivity, the excision on the
/// root, then the user sources and the driven boundaries in their id order.
fn mhd_kernels<R: Regime<f64, D>, const D: usize>(
    sub: symbi::regimes::substrate_mhd::MhdSubstrateKernelSet<R, DefaultMemory, f64, D>,
    settings: &KernelSettings,
    role: LevelRole,
) -> Result<symbi::regimes::substrate_mhd::MhdSubstrateKernelSet<R, DefaultMemory, f64, D>, String> {
    let ks = sub
        .theta(settings.theta)
        .with_solver(settings.solver)
        .map_err(|e| format!("substrate/solver: {e:?}"))?
        .ct_method(settings.ct_method)
        .with_viscosity(settings.viscosity)
        .with_alpha(settings.alpha)
        .with_resistivity(settings.resistivity);
    let ks = excised(ks, settings, role);
    let ks = attach_configured_sources(ks, settings.sources, settings.spec, &settings.law)?;
    attach_driven_boundaries(ks, settings.driven, settings.spec, |ks, built, params| ks.with_driven_boundary(built, params).0)
}

/// the horizon excision on a root kernel set; a fine level keeps the set as it is
fn excised<K: WithExcision>(ks: K, settings: &KernelSettings, role: LevelRole) -> K {
    match role {
        LevelRole::Root => ks.with_excision(settings.excision[0], settings.excision[1], settings.excision[2]),
        LevelRole::Fine => ks,
    }
}

/// the static regime spec a config's `regime` string names. the spec drives every
/// source-term validation (which conservation laws are well posed, whether an energy
/// equation exists), so resolving it here is what lets a source be lowered at preflight,
/// not only at dispatch, where the regime is a type parameter.
fn regime_spec_for(regime: &str) -> Result<&'static symbi_hydro::RegimeSpec, String> {
    match regime {
        "newtonian" => Ok(&symbi_hydro::NEWTONIAN_SPEC),
        "isothermal" => Ok(&symbi_hydro::ISO_NEWTONIAN_SPEC),
        "rhd" => Ok(&symbi_hydro::RHD_SPEC),
        "rmhd" => Ok(&symbi_hydro::RMHD_SPEC),
        "nmhd" => Ok(&symbi_hydro::NEWTONIAN_MHD_SPEC),
        "imhd" => Ok(&symbi_hydro::ISO_MHD_SPEC),
        other => Err(format!("unknown regime '{other}'")),
    }
}

fn lower_configured_sources(
    source_jsons: &[SourcePayload],
    spec: &symbi_hydro::RegimeSpec,
    law: Option<&symbi_hydro::state_law::StateLaw>,
) -> Result<(symbi_source_compile::AdmittedSources, Vec<f64>), String> {
    let configs = source_jsons
        .iter()
        .map(|source| {
            let config = symbi_source_compile::SourceConfig::from_json(&source.json)
                .map_err(|error| format!("{} parse: {error}", source.origin))?;
            symbi_source_compile::expr_bridge::build_user_source_with_law(&config, spec, law)
                .map_err(|error| format!("{} lower: {error}", source.origin))?;
            Ok(config)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (built, params) =
        symbi_source_compile::expr_bridge::build_user_sources_with_law(&configs, spec, law)
            .map_err(|error| format!("source expression lower: {error}"))?;
    Ok((built, params))
}

#[cfg(test)]
mod source_collection_tests {
    use super::*;

    #[test]
    fn lower_error_names_collection_index() {
        let sources = [SourcePayload {
            origin: "source_expressions[3]".to_string(),
            json: r#"{
                "kind":"force", "dim":1, "outputs":[2], "params":[],
                "vocabulary":{"reads":[],"params":[]},
                "nodes":[{"op":"CONSTANT","value":1.0},
                         {"op":"CONSTANT","value":2.0},
                         {"op":"MOD","left":0,"right":1}]
            }"#
            .to_string(),
        }];
        let error = match lower_configured_sources(&sources, &symbi_hydro::NEWTONIAN_SPEC, None) {
            Err(error) => error,
            Ok(_) => panic!("unsupported source operation was accepted"),
        };
        assert!(error.contains("source_expressions[3] lower:"), "{error}");
    }

    #[test]
    fn parse_error_names_collection_index() {
        let sources = [SourcePayload {
            origin: "source_expressions[2]".to_string(),
            json: "{".to_string(),
        }];
        let error = match lower_configured_sources(&sources, &symbi_hydro::NEWTONIAN_SPEC, None) {
            Err(error) => error,
            Ok(_) => panic!("malformed source json was accepted"),
        };
        assert!(error.contains("source_expressions[2] parse:"), "{error}");
    }

    #[test]
    fn a_source_granted_rho_alone_that_reads_vel_is_refused_at_the_binding() {
        // the wire a python graph emits when its context granted `density()` alone while the
        // dag carries a velocity leaf: the config door refuses the contribution with the
        // observed read named, and the collection index points at the source.
        let sources = [SourcePayload {
            origin: "source_expressions[0]".to_string(),
            json: r#"{
                "kind":"raw", "dim":2, "target":"den", "outputs":[2], "params":[],
                "vocabulary":{"reads":["rho"],"params":[]},
                "nodes":[{"op":"VARIABLE_RHO"}, {"op":"VARIABLE_VEL1"},
                         {"op":"MULTIPLY","left":0,"right":1}]
            }"#
            .to_string(),
        }];
        let error = match lower_configured_sources(&sources, &symbi_hydro::NEWTONIAN_SPEC, None) {
            Err(error) => error,
            Ok(_) => panic!("a read outside the granted vocabulary was accepted"),
        };
        assert!(error.contains("source_expressions[0] lower:"), "{error}");
        assert!(error.contains("UndeclaredRead"), "{error}");
        assert!(error.contains("Vel(Axis(0))"), "{error}");
    }

    fn census_payload(origin: &str, name: &str) -> SourcePayload {
        SourcePayload {
            origin: origin.to_string(),
            json: format!(
                r#"{{ "name":"{name}", "axes":[], "values":[2], "value_names":["mass"],
                      "op":"add", "params":[],
                      "nodes":[{{"op":"VARIABLE_RHO"}}, {{"op":"VARIABLE_DV"}},
                               {{"op":"MULTIPLY","left":0,"right":1}}] }}"#
            ),
        }
    }

    #[test]
    fn a_well_formed_census_lowers_at_setup() {
        let censuses = [census_payload("census_expressions[0]", "conservation")];
        let evaluated = lower_configured_censuses(&censuses).expect("census lowers");
        assert_eq!(evaluated.len(), 1);
        assert_eq!(evaluated[0].spec().name(), "conservation");
        // no bin axes is a global reduction over the grid: one bucket.
        assert_eq!(evaluated[0].spec().n_segments(), 1);
    }

    #[test]
    fn census_lower_error_names_the_registration_index() {
        // an unlowerable expression must be reported against the registration that
        // carries it, not as an anonymous failure at the first sample.
        let censuses = [SourcePayload {
            origin: "census_expressions[3]".to_string(),
            json: r#"{ "name":"bad", "axes":[], "values":[2], "value_names":["v"],
                       "op":"add", "params":[],
                       "nodes":[{"op":"CONSTANT","value":1.0},
                                {"op":"CONSTANT","value":2.0},
                                {"op":"MOD","left":0,"right":1}] }"#
                .to_string(),
        }];
        let error = lower_configured_censuses(&censuses).expect_err("unsupported op was accepted");
        assert!(error.contains("census_expressions[3] lower:"), "{error}");
    }

    #[test]
    fn census_parse_error_names_the_registration_index() {
        let censuses = [SourcePayload {
            origin: "census_expressions[2]".to_string(),
            json: "{".to_string(),
        }];
        let error = lower_configured_censuses(&censuses).expect_err("malformed json was accepted");
        assert!(error.contains("census_expressions[2] parse:"), "{error}");
    }

    #[test]
    fn a_duplicate_census_name_is_refused_at_setup() {
        // the name is the checkpoint group, so a collision would have the second
        // registration silently overwrite the first.
        let censuses = [
            census_payload("census_expressions[0]", "shells"),
            census_payload("census_expressions[1]", "shells"),
        ];
        let error = lower_configured_censuses(&censuses).expect_err("duplicate name was accepted");
        assert!(error.contains("reuses the census name 'shells'"), "{error}");
    }

    #[test]
    fn a_census_binning_is_validated_at_setup() {
        // non-increasing edges make a bin no cell can land in. the registration must
        // fail here, not produce a census that quietly bins nothing.
        let censuses = [SourcePayload {
            origin: "census_expressions[0]".to_string(),
            json: r#"{ "name":"shells",
                       "axes":[{"name":"r","expr":0,"edges":[2.0,1.0]}],
                       "values":[0], "value_names":["v"], "op":"add", "params":[],
                       "nodes":[{"op":"VARIABLE_RHO"}] }"#
                .to_string(),
        }];
        let error = lower_configured_censuses(&censuses).expect_err("bad edges accepted");
        assert!(error.contains("strictly increase"), "{error}");
    }
}

fn solver_from_str(s: &str) -> PyResult<Solver> {
    match s {
        "hlle" => Ok(Solver::Hlle),
        "hllc" => Ok(Solver::Hllc),
        // the retired low-mach family. both scaled the acoustic signal speeds, which
        // multiplies the pressure jump along with the velocity jump, so both left the
        // hydrostatic truncation residual of a stagnant stratified column undamped and both
        // amplified it under refinement. hllc_plus rescales the two velocity jumps instead,
        // leaving every signal speed classical, and covers the low-mach accuracy and the
        // grid-aligned shock instability at once.
        "hllc_lm" | "hllc-lm" | "hllc_lm_plain" | "hllc-lm-plain" | "hllc_acoustic"
        | "hllc-acoustic" => Err(PyValueError::new_err(format!(
            "solver '{s}' was retired on 2026-08-17. it scaled the acoustic signal speeds, \
             which damps the pressure jump along with the velocity jump, so a stagnant \
             stratified column lost the dissipation holding its hydrostatic residual down \
             and the residual grew with resolution. use solver='hllc_plus', which rescales \
             the velocity jumps alone and needs no reference mach number; pair it with \
             wb_reconstruction=True for stratified problems."
        ))),
        // anti-dissipation pressure correction: rescales the velocity-jump dissipation alone,
        // needing no reference mach number (newtonian only).
        "hllc_plus" | "hllc-apc" => Ok(Solver::HllcPlus),
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
        "axis" => Ok(BoundaryType::Axis),
        other => Err(PyValueError::new_err(format!(
            "unsupported boundary '{other}'"
        ))),
    }
}

/// the gradient wall a python `Neumann`/`Robin` object (attributes `rho`, `velocity`, `pressure`)
/// prescribes, per variable in prim order: rho, then exactly `n_vel` velocity entries, then pre.
/// Neumann carries one normal derivative `q` per variable, Robin one `(a, b, c)` triple. an omitted
/// trailing velocity component takes the documented default, a zero gradient for neumann and the
/// dirichlet triple `(1, 0, 0)` for robin; more components than the regime carries are refused by
/// name.
fn extract_gradient_spec(obj: &Bound<'_, PyAny>, kind: &str, n_vel: usize) -> PyResult<GradientBc> {
    let rho = obj.getattr("rho")?;
    let velocity = obj.getattr("velocity")?;
    let pressure = obj.getattr("pressure")?;
    let given = velocity.len()?;
    if given > n_vel {
        return Err(PyValueError::new_err(format!(
            "a {kind} wall gives {given} velocity components; the regime carries {n_vel}"
        )));
    }
    if kind == "neumann" {
        let mut q = vec![rho.extract::<f64>()?];
        for v in velocity.try_iter()? {
            q.push(v?.extract::<f64>()?);
        }
        q.extend(std::iter::repeat_n(0.0, n_vel - given));
        q.push(pressure.extract::<f64>()?);
        Ok(GradientBc::Neumann(q))
    } else {
        let mut triples: Vec<[f64; 3]> = Vec::new();
        let (a, b, c): (f64, f64, f64) = rho.extract()?;
        triples.push([a, b, c]);
        for v in velocity.try_iter()? {
            let (a, b, c): (f64, f64, f64) = v?.extract()?;
            triples.push([a, b, c]);
        }
        triples.extend(std::iter::repeat_n([1.0, 0.0, 0.0], n_vel - given));
        let (a, b, c): (f64, f64, f64) = pressure.extract()?;
        triples.push([a, b, c]);
        Ok(GradientBc::Robin(triples))
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
    // (`Driven`) face reads its prescribed ghost state from the matching field.
    const BX_FIELDS: [&str; 6] = [
        "bx1_inner_expressions",
        "bx1_outer_expressions",
        "bx2_inner_expressions",
        "bx2_outer_expressions",
        "bx3_inner_expressions",
        "bx3_outer_expressions",
    ];
    let mut boundaries = Vec::new();
    // driven (`dynamic`) boundary expressions in Driven-id order; id == registration order ==
    // the order faces are visited here, so `Driven(id)` on a face matches `driven_exprs[id]`.
    let mut driven_exprs: Vec<String> = Vec::new();
    // gradient (Neumann / Robin) boundary coefficients in registry order; id == registration order,
    // so `Neumann(id)` / `Robin(id)` on a face matches `gradient_bcs[id]`.
    let mut gradient_bcs: Vec<GradientBc> = Vec::new();
    // the velocity components a primitive carries: three for a magnetized regime, one per
    // spatial axis otherwise
    let n_vel = if enum_str(dict, "regime")?.contains("mhd") { 3 } else { dims };
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
            kind @ ("neumann" | "robin") => {
                let spec = extract_gradient_spec(&obj, kind, n_vel)?;
                let id = gradient_bcs.len() as u16;
                gradient_bcs.push(spec);
                boundaries.push(if kind == "neumann" {
                    BoundaryType::Neumann(id)
                } else {
                    BoundaryType::Robin(id)
                });
            }
            other => boundaries.push(boundary_from_str(other)?),
        }
    }

    let gamma = dict
        .get_item("adiabatic_index")?
        .and_then(|v| v.extract::<f64>().ok())
        .unwrap_or(5.0 / 3.0);
    // gamma = 1 on an energy regime is a degenerate EOS: the adiabatic sound
    // speed gamma(gamma-1)e is identically zero, the CFL wave speed vanishes
    // on quiescent gas, and the first dt spans the whole run — one giant step
    // to NaN, reported as success. reject at parse with the actionable fix.
    {
        let regime_s = enum_str(dict, "regime")?;
        let has_energy = !(regime_s.contains("iso") || regime_s == "imhd");
        if has_energy && gamma <= 1.0 {
            return Err(PyValueError::new_err(format!(
                "adiabatic_index = {gamma} is degenerate for the energy regime '{regime_s}'                  (adiabatic sound speed = 0): gamma = 1 is the ISOTHERMAL equation of state —                  set regime = Regime.ISOTHERMAL (and ambient_sound_speed), or use gamma > 1",
            )));
        }
    }

    let solver_name = enum_str(dict, "solver")?;
    // constrained-transport edge-emf scheme; contact is the default method.
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

    // the config author's grouped custom params for the dashboard: a list of [group, label, value].
    let custom_params: Vec<[String; 3]> = match dict.get_item("custom_params")? {
        Some(obj) if !obj.is_none() => {
            let mut v = Vec::new();
            for row in obj.try_iter()? {
                let row = row?;
                v.push([
                    row.get_item(0)?.extract()?,
                    row.get_item(1)?.extract()?,
                    row.get_item(2)?.extract()?,
                ]);
            }
            v
        }
        _ => Vec::new(),
    };

    Ok(Config {
        // the problem class name (preserve case); blank if omitted.
        name: dict
            .get_item("name")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_default(),
        config_source: dict
            .get_item("config_source")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "<unknown>".to_string()),
        config_sha256: dict
            .get_item("config_sha256")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "unavailable".to_string()),
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
        retry_max_rejections: get_f64_or(dict, "retry_max_rejections", symbi_sim::retry_guard::RetryGuard::default().max_rejections as f64) as u32,
        retry_severe_ratio: get_f64_or(dict, "retry_severe_ratio", symbi_sim::retry_guard::RetryGuard::default().severe_ratio),
        retry_sustained_steps: get_f64_or(dict, "retry_sustained_steps", symbi_sim::retry_guard::RetryGuard::default().sustained_steps as f64) as u32,
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
        eos_name: enum_str_or(dict, "eos", "ideal"),
        ppm_flatten_onset: get_f64_or(dict, "ppm_flatten_onset", 0.0),
        ppm_flatten_full: get_f64_or(dict, "ppm_flatten_full", 0.0),
        timestepping: timestepping_from_str(&enum_str(dict, "timestepping")?)?,
        plm_theta: get_f64_or(dict, "plm_theta", 1.5),
        wb_reconstruction: dict
            .get_item("wb_reconstruction")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        dlogt: get_f64_or(dict, "dlogt", 0.0),
        viscosity: get_f64_or(dict, "viscosity", 0.0),
        alpha: get_f64_or(dict, "viscosity_alpha", 0.0),
        bare_alpha_key: dict.get_item("alpha").ok().flatten().is_some(),
        resistivity: get_f64_or(dict, "resistivity", 0.0),
        excision_radius: get_f64_or(dict, "excision_radius", 0.0),
        excision_rho_scale: 1.0,
        excision_pre_scale: 1.0,
        x1_spacing: enum_str_or(dict, "x1_spacing", "linear"),
        x1_spacing_ratio: get_f64_or(dict, "x1_spacing_ratio", 1.0),
        x2_spacing: enum_str_or(dict, "x2_spacing", "linear"),
        x2_spacing_ratio: get_f64_or(dict, "x2_spacing_ratio", 1.0),
        x3_spacing: enum_str_or(dict, "x3_spacing", "linear"),
        x3_spacing_ratio: get_f64_or(dict, "x3_spacing_ratio", 1.0),
        start_time: get_f64_or(dict, "start_time", 0.0),
        checkpoint_log_anchor: get_f64_or(dict, "checkpoint_log_anchor", 0.0),
        checkpoint_index: dict
            .get_item("checkpoint_index")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<u64>().ok())
            .unwrap_or(0),
        t_final: get_f64(dict, "end_time")?,
        max_steps: dict
            .get_item("max_steps")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<u64>().ok())
            .unwrap_or(0),
        checkpoint_interval: get_f64(dict, "checkpoint_interval")?,
        data_dir: dict
            .get_item("data_directory")?
            .ok_or_else(|| PyValueError::new_err("sim_info missing 'data_directory'"))?
            .extract()?,
        restart_path: dict
            .get_item("checkpoint_file")
            .ok()
            .flatten()
            .and_then(|value| value.extract::<String>().ok())
            .filter(|path| !path.is_empty()),
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
        bonded_assembly: parse_bonded_assembly(dict)?,
        n_tracers: dict
            .get_item("n_tracers")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<i64>().ok())
            .map(|v| v.max(0) as usize)
            .unwrap_or(0),
        tracer_scheme: dict
            .get_item("tracer_scheme")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<String>().ok())
            .unwrap_or_else(|| "discrete".to_string()),
        chi_ic: Vec::new(),
        cohort_ic: Vec::new(),
        binary: parse_binary(dict),
        diagnostic_interval: get_f64_or(dict, "diagnostic_interval", 0.0),
        n_gpus: dict
            .get_item("gpus")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(1)
            .max(1),
        decompose: dict
            .get_item("decompose")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<Vec<Vec<usize>>>().ok())
            .unwrap_or_default(),
        // ordered source configs use the sole public source-expression wire.
        source_jsons: get_source_jsons(dict)?,
        // registered binned reductions, lowered at setup so a malformed one fails there.
        census_jsons: get_census_jsons(dict)?,
        motion_json: get_source_json(dict, "scale_factor_expressions")?,
        equilibrium_json: get_source_json(dict, "equilibrium_expressions")?,
        seed_from_equilibrium: dict
            .get_item("seed_from_equilibrium")
            .ok()
            .flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false),
        perturbation_json: get_source_json(dict, "perturbation_expressions")?,
        driven_exprs,
        gradient_bcs,
        custom_params,
    })
}

/// parse the python `immersed_bodies` list (each a serialized ImmersedBodyConfig
/// dict) into dimension-agnostic `BodyParams`. missing / malformed entries are
/// skipped; a body-free config yields an empty vec.
/// parse the `bonded_assembly` wire (the python `BondedAssembly.to_backend`
/// dict). the python side validates shape/lengths; the extraction here still
/// fails loud on any missing or mistyped key -- a fragment cluster silently
/// dropped to zero bodies would be the body-system parse bug all over again.
fn parse_bonded_assembly(dict: &Bound<'_, PyDict>) -> PyResult<Option<BondedAssemblyParams>> {
    let Some(obj) = dict.get_item("bonded_assembly")? else {
        return Ok(None);
    };
    if obj.is_none() {
        return Ok(None);
    }
    let d = obj
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("bonded_assembly must be a dict"))?;
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(k)?
            .ok_or_else(|| PyValueError::new_err(format!("bonded_assembly missing key '{k}'")))
    };
    let positions: Vec<Vec<f64>> = get("positions")?.extract()?;
    let masses: Vec<f64> = get("masses")?.extract()?;
    let radii: Vec<f64> = get("radii")?.extract()?;
    let inertias: Vec<f64> = get("inertias")?.extract()?;
    let velocities: Vec<Vec<f64>> = get("velocities")?.extract()?;
    let mobile: Vec<bool> = get("mobile")?.extract()?;
    let bonds: Vec<(usize, usize)> = get("bonds")?
        .extract::<Vec<Vec<usize>>>()?
        .into_iter()
        .map(|p| (p[0], p[1]))
        .collect();
    let n = positions.len();
    if masses.len() != n
        || radii.len() != n
        || inertias.len() != n
        || velocities.len() != n
        || mobile.len() != n
    {
        return Err(PyValueError::new_err(format!(
            "bonded_assembly arrays disagree on the fragment count {n}"
        )));
    }
    for &(i, j) in &bonds {
        if i >= n || j >= n || i == j {
            return Err(PyValueError::new_err(format!(
                "bonded_assembly bond ({i}, {j}) outside the {n} fragments"
            )));
        }
    }
    let mat = get("bond_material")?;
    let md = mat
        .cast::<PyDict>()
        .map_err(|_| PyValueError::new_err("bond_material must be a dict"))?;
    let mf = |k: &str| -> PyResult<f64> {
        md.get_item(k)?
            .ok_or_else(|| PyValueError::new_err(format!("bond_material missing '{k}'")))?
            .extract()
    };
    let bond_material = symbi_ib::BondMaterial {
        k_n: mf("k_n")?,
        k_t: mf("k_t")?,
        gamma: mf("gamma")?,
        area: mf("area")?,
        sigma_t: mf("sigma_t")?,
        tau_s: mf("tau_s")?,
    };
    let contact = match d.get_item("contact")? {
        Some(c) if !c.is_none() => {
            let cd = c
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("contact must be a dict"))?
                .clone();
            let cf = |k: &str| -> PyResult<f64> {
                cd.get_item(k)?
                    .ok_or_else(|| PyValueError::new_err(format!("contact missing '{k}'")))?
                    .extract()
            };
            Some(symbi_ib::ContactMaterial {
                k_n: cf("k_n")?,
                k_t: cf("k_t")?,
                gamma_n: cf("gamma_n")?,
                mu: cf("mu")?,
            })
        }
        _ => None,
    };
    let gravity = match d.get_item("gravity")? {
        Some(g) if !g.is_none() => {
            let gd = g
                .cast::<PyDict>()
                .map_err(|_| PyValueError::new_err("gravity must be a dict"))?
                .clone();
            let gf = |k: &str| -> PyResult<f64> {
                gd.get_item(k)?
                    .ok_or_else(|| PyValueError::new_err(format!("gravity missing '{k}'")))?
                    .extract()
            };
            Some(symbi_ib::MutualGravity {
                g: gf("g")?,
                softening: gf("softening")?,
            })
        }
        _ => None,
    };
    let kf = |k: &str| -> PyResult<f64> { get(k)?.extract() };
    Ok(Some(BondedAssemblyParams {
        positions,
        masses,
        radii,
        inertias,
        velocities,
        mobile,
        bonds,
        bond_material,
        contact,
        gravity,
        k_eta_n: kf("k_eta_n")?,
        k_eta_t: kf("k_eta_t")?,
    }))
}

fn parse_bodies(dict: &Bound<'_, PyDict>) -> Vec<BodyParams> {
    let mut out: Vec<BodyParams> = Vec::new();
    // the gravitational body-system branch (`body_system.binary_config`): the binary components are
    // gravitating accretors whose initial positions/velocities come from the Keplerian orbit (the
    // config leaves the component positions at the origin, delegating the ICs to the backend). the
    // orbit itself is attached separately via `parse_binary` -> `with_binary_params`.
    parse_binary_components(dict, &mut out);
    let Ok(Some(obj)) = dict.get_item("immersed_bodies") else {
        return out;
    };
    let Ok(list) = obj.extract::<Vec<Bound<'_, PyAny>>>() else {
        return out;
    };
    out.reserve(list.len());
    for item in &list {
        let Ok(b) = item.cast::<PyDict>() else {
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
        let two_way_coupling: bool = b
            .get_item("two_way_coupling")
            .ok()
            .flatten()
            .and_then(|x| x.extract().ok())
            .unwrap_or(false);
        let softening = sub_f64(b, "gravitational", "softening_length", 0.0);
        // the family selector rides as a string so a config reads as physics, not a flag.
        // the vocabulary is validated on the python side (`GravitationalProperties`), which is
        // where a typo can be reported against the line that wrote it.
        let softening_kind =
            if sub_str(b, "gravitational", "softening_kind", "plummer") == "compact" {
                1.0
            } else {
                0.0
            };
        let accretion_radius = sub_f64(b, "accretion", "accretion_radius", 0.0);
        let sink_rate = sub_f64(b, "accretion", "sink_rate", 0.0);
        let porosity = sub_f64_opt(b, "accretion", "porosity");
        let torque_free_xi = sub_f64_opt(b, "accretion", "torque_free_xi");
        // the rigid wall (capability rigid = 1<<4) is the drain-off porous surface; its
        // dials live in the `rigid` block (a rigid body has no `accretion` block). free
        // slip (apply_no_slip False) zeroes the tangential channel exactly.
        const RIGID_BIT: u64 = 1 << 4;
        let is_rigid = capability & RIGID_BIT != 0;
        let inertia = sub_f64(b, "rigid", "inertia", 0.0);
        let no_slip = sub_bool(b, "rigid", "apply_no_slip", true);
        let shape_json = get_shape_json(b);
        let omega = sub_f64(b, "rigid", "omega", 0.0);
        let spin_axis = sub_vec3(b, "rigid", "spin_axis", [0.0, 0.0, 1.0]);
        // all-zero sentinel = unspecified (isotropic); a valid tensor has all moments > 0.
        let inertia_principal = sub_vec3(b, "rigid", "inertia_principal", [0.0, 0.0, 0.0]);
        let (k_eta_n, k_eta_t) = if is_rigid {
            (
                sub_f64(b, "rigid", "k_eta_n", 1.0),
                if no_slip {
                    sub_f64(b, "rigid", "k_eta_t", 1.0)
                } else {
                    0.0
                },
            )
        } else {
            (
                sub_f64(b, "accretion", "k_eta_n", 0.0),
                sub_f64(b, "accretion", "k_eta_t", 0.0),
            )
        };
        out.push(BodyParams {
            capability,
            mass: f("mass"),
            radius: f("radius"),
            position: v("position"),
            velocity: v("velocity"),
            softening,
            softening_kind,
            accretion_radius,
            sink_rate,
            porosity,
            k_eta_n,
            k_eta_t,
            torque_free_xi,
            inertia,
            no_slip,
            shape_json,
            omega,
            spin_axis,
            inertia_principal,
            two_way_coupling,
            magnetic_resistivity: sub_f64_opt(b, "magnetic", "resistivity"),
            magnetic_slip: parse_magnetic_slip(b),
        });
    }
    out
}

/// the `magnetic.slip` group, when the body declares the magnetic slip. a declared group with a
/// missing or non-numeric scale is a wire fault and fails here rather than silently defaulting.
fn parse_magnetic_slip(body: &Bound<'_, PyDict>) -> Option<MagneticSlipParams> {
    let slip = body
        .get_item("magnetic")
        .ok()
        .flatten()
        .and_then(|g| g.cast::<PyDict>().ok().cloned())
        .and_then(|gd| gd.get_item("slip").ok().flatten())
        .and_then(|sl| sl.cast::<PyDict>().ok().cloned())?;
    let f = |key: &str| -> f64 {
        slip.get_item(key)
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| panic!("immersed body `magnetic.slip.{key}` is missing or not a number"))
    };
    Some(MagneticSlipParams {
        diffusivity_ratio: f("diffusivity_ratio"),
        shell_width: f("shell_width"),
        slip_length_ratio: f("slip_length_ratio"),
        field_regularization: f("field_regularization"),
        placement: f("placement"),
    })
}

/// the prescribed binary orbit params (`body_system.binary_config`): total mass, semi-major axis,
/// eccentricity, mass ratio. `None` when there is no gravitational binary. threaded to
/// `BodyCollection::with_binary_params` so the two components follow the Keplerian orbit each step.
struct BinaryCfg {
    total_mass: f64,
    semi_major: f64,
    eccentricity: f64,
    mass_ratio: f64,
}

/// the `body_system.binary_config` sub-dict, if the config carries a gravitational binary.
fn binary_config_dict<'py>(dict: &Bound<'py, PyDict>) -> Option<Bound<'py, PyDict>> {
    let bs = dict.get_item("body_system").ok().flatten()?;
    let bs = bs.cast_into::<PyDict>().ok()?;
    let bc = bs.get_item("binary_config").ok().flatten()?;
    bc.cast_into::<PyDict>().ok()
}

fn parse_binary(dict: &Bound<'_, PyDict>) -> Option<BinaryCfg> {
    let bc = binary_config_dict(dict)?;
    Some(BinaryCfg {
        total_mass: get_f64_or(&bc, "total_mass", 1.0),
        semi_major: get_f64_or(&bc, "semi_major", 1.0),
        eccentricity: get_f64_or(&bc, "eccentricity", 0.0),
        mass_ratio: get_f64_or(&bc, "mass_ratio", 1.0),
    })
}

/// append the binary components as gravitating (and, if `is_an_accretor`, accreting) bodies, with
/// their initial positions/velocities from the circular Keplerian orbit about the com (the config
/// leaves the component positions at the origin and delegates the ICs to `keplerian_binary`).
fn parse_binary_components(dict: &Bound<'_, PyDict>, out: &mut Vec<BodyParams>) {
    const ACCRETION: u64 = 2;
    let Some(bc) = binary_config_dict(dict) else {
        return;
    };
    let total_mass = get_f64_or(&bc, "total_mass", 1.0);
    let semi_major = get_f64_or(&bc, "semi_major", 1.0);
    let mass_ratio = get_f64_or(&bc, "mass_ratio", 1.0);
    let (p1, v1, _m1, p2, v2, _m2) =
        symbi_ib::keplerian_binary::<f64>(total_mass, semi_major, mass_ratio);
    let ics = [(p1, v1), (p2, v2)];
    let Ok(Some(comps)) = bc.get_item("components") else {
        return;
    };
    let Ok(comps) = comps.extract::<Vec<Bound<'_, PyAny>>>() else {
        return;
    };
    for (i, comp) in comps.iter().enumerate() {
        let Ok(c) = comp.cast::<PyDict>() else {
            continue;
        };
        let has_accretion = c.get_item("accretion").ok().flatten().is_some();
        let (pos, vel) = ics.get(i).copied().unwrap_or((p1, v1));
        let two_way = c
            .get_item("two_way_coupling")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or(false);
        out.push(BodyParams {
            capability: if has_accretion { ACCRETION } else { 1 },
            mass: get_f64_or(c, "mass", 0.0),
            radius: get_f64_or(c, "radius", 0.0),
            position: vec![pos[0], pos[1]],
            velocity: vec![vel[0], vel[1]],
            softening: sub_f64(c, "gravitational", "softening_length", 0.0),
            softening_kind: if sub_str(c, "gravitational", "softening_kind", "plummer") == "compact"
            {
                1.0
            } else {
                0.0
            },
            accretion_radius: sub_f64(c, "accretion", "accretion_radius", 0.0),
            sink_rate: sub_f64(c, "accretion", "sink_rate", 0.0),
            porosity: sub_f64_opt(c, "accretion", "porosity"),
            k_eta_n: sub_f64(c, "accretion", "k_eta_n", 0.0),
            k_eta_t: sub_f64(c, "accretion", "k_eta_t", 0.0),
            torque_free_xi: sub_f64_opt(c, "accretion", "torque_free_xi"),
            inertia: 0.0,
            no_slip: true,
            shape_json: None,
            omega: 0.0,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0, 0.0, 0.0],
            two_way_coupling: two_way,
            magnetic_resistivity: sub_f64_opt(c, "magnetic", "resistivity"),
            magnetic_slip: parse_magnetic_slip(c),
        });
    }
}

/// read `body[group][key]` as f64 (the nested ImmersedBodyConfig sub-dicts like
/// `gravitational` / `accretion`), returning `default` when absent or null.
fn sub_f64(body: &Bound<'_, PyDict>, group: &str, key: &str, default: f64) -> f64 {
    sub_f64_opt(body, group, key).unwrap_or(default)
}

/// the string twin of `sub_f64`: a nested `body[group][key]` string, or `default` when the
/// group or key is absent. used by selectors that name a physical model, not set a number.
fn sub_str(body: &Bound<'_, PyDict>, group: &str, key: &str, default: &str) -> String {
    body.get_item(group)
        .ok()
        .flatten()
        .and_then(|g| g.cast::<PyDict>().ok().cloned())
        .and_then(|gd| gd.get_item(key).ok().flatten())
        .and_then(|val| val.extract::<String>().ok())
        .unwrap_or_else(|| default.to_string())
}

/// like `sub_f64` but absent / null / non-numeric is `None` — for dials whose
/// absence means a different code path (e.g. porosity: None = the pure drain).
fn sub_f64_opt(body: &Bound<'_, PyDict>, group: &str, key: &str) -> Option<f64> {
    body.get_item(group)
        .ok()
        .flatten()
        .and_then(|g| g.cast::<PyDict>().ok().cloned())
        .and_then(|gd| gd.get_item(key).ok().flatten())
        .and_then(|val| val.extract().ok())
}

/// read the rigid-wall shape CSG (`body["rigid"]["shape"]["wire"]`) and return it as a json string
/// ready for `SdfExpr::from_json`; None when no shape is declared (the analytic sphere). the dict
/// crosses the boundary through `json.dumps`, matching the source-expression wire convention.
fn get_shape_json(body: &Bound<'_, PyDict>) -> Option<String> {
    let rigid = body
        .get_item("rigid")
        .ok()
        .flatten()?
        .cast::<PyDict>()
        .ok()?
        .clone();
    let shape = rigid
        .get_item("shape")
        .ok()
        .flatten()?
        .cast::<PyDict>()
        .ok()?
        .clone();
    let wire = shape.get_item("wire").ok().flatten()?;
    let json = wire.py().import("json").ok()?;
    json.call_method1("dumps", (wire,)).ok()?.extract().ok()
}

/// read a 3-vector from a body sub-block (`body[group][key]` = `[x, y, z]`); absent / wrong-arity
/// -> `default`.
fn sub_vec3(body: &Bound<'_, PyDict>, group: &str, key: &str, default: [f64; 3]) -> [f64; 3] {
    let v: Option<Vec<f64>> = body
        .get_item(group)
        .ok()
        .flatten()
        .and_then(|g| g.cast::<PyDict>().ok().cloned())
        .and_then(|gd| gd.get_item(key).ok().flatten())
        .and_then(|val| val.extract().ok());
    match v {
        Some(v) if v.len() == 3 => [v[0], v[1], v[2]],
        _ => default,
    }
}

/// read a bool from a body sub-block (`body[group][key]`); absent / non-bool -> `default`.
fn sub_bool(body: &Bound<'_, PyDict>, group: &str, key: &str, default: bool) -> bool {
    body.get_item(group)
        .ok()
        .flatten()
        .and_then(|g| g.cast::<PyDict>().ok().cloned())
        .and_then(|gd| gd.get_item(key).ok().flatten())
        .and_then(|val| val.extract().ok())
        .unwrap_or(default)
}

/// the scale factor and its rate at the start time, sampled from the problem's callables (gil
/// held): the mesh-motion model integrates `a` from the constant rate `a_dot` (linear or free
/// homologous expansion, a_ddot = 0), so one sample at t0 sets the motion
fn sample_scale_factor(cfg: &mut Config, a: Option<&Bound<'_, PyAny>>, adot: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    if cfg.mesh_motion {
        let (Some(a), Some(adot)) = (a.filter(|f| !f.is_none()), adot.filter(|f| !f.is_none())) else {
            return Err(PyValueError::new_err("mesh motion declared without the scale factor callables a(t), a_dot(t)"));
        };
        // the callable motion integrates a at the constant rate sampled at the start, so a resumed
        // run continues the integrated a and its rate the checkpoint records; the callables would
        // place the mesh on their own curve instead. a traced law evaluates a(t) exactly and keeps
        // the start sample
        if let (Some(path), None) = (cfg.restart_path.as_deref(), cfg.motion_json.as_ref()) {
            let meta = symbi_io::read_attrs(std::path::Path::new(path), "metadata")
                .map_err(|e| PyValueError::new_err(format!("restart from '{path}': {e}")))?;
            let recorded = |name: &str| -> PyResult<f64> {
                match meta.iter().find(|(key, _)| key == name).map(|(_, attr)| attr) {
                    Some(symbi_io::Attr::F64(v)) => Ok(*v),
                    _ => Err(PyValueError::new_err(format!(
                        "restart from '{path}': the checkpoint records no metadata/{name} for the mesh motion"
                    ))),
                }
            };
            cfg.scale_a0 = recorded("scale_factor")?;
            cfg.scale_adot = recorded("scale_factor_dot")?;
            return Ok(());
        }
        let t0 = cfg.start_time;
        cfg.scale_a0 = a.call1((t0,))?.extract::<f64>()?;
        cfg.scale_adot = adot.call1((t0,))?.extract::<f64>()?;
    }
    Ok(())
}

/// the passive-scalar generator as one flat cell-centered buffer in the prim order (None or an
/// empty generator: an undyed run)
fn drain_dye(chi_field: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
    let mut vals = Vec::new();
    if let Some(chi_field) = chi_field.filter(|f| !f.is_none()) {
        for v in chi_field.try_iter()? {
            vals.push(v?.extract::<f64>()?);
        }
    }
    Ok(vals)
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

fn initial_excision_scales(prims: &[Vec<f64>]) -> Result<(f64, f64), String> {
    let rho_scale = prims
        .iter()
        .filter_map(|row| row.first().copied())
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    let pre_scale = prims
        .iter()
        .filter_map(|row| row.last().copied())
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::INFINITY, f64::min);
    if !rho_scale.is_finite() || !pre_scale.is_finite() {
        return Err(
            "excision atmosphere requires positive finite initial density and pressure".to_string(),
        );
    }
    Ok((rho_scale, pre_scale))
}

#[cfg(test)]
mod excision_scale_tests {
    use super::initial_excision_scales;

    #[test]
    fn initial_excision_scales_follow_state_units() {
        let prims = vec![vec![2.0, 0.1, 0.0, 0.5], vec![5.0, -0.2, 0.1, 3.0]];

        for factor in [1e-100_f64, 1.0, 1e100] {
            let scaled: Vec<Vec<f64>> = prims
                .iter()
                .map(|row| vec![row[0] * factor, row[1], row[2], row[3] * factor])
                .collect();
            let (rho, pre) = initial_excision_scales(&scaled).unwrap();
            assert_eq!(rho, 2.0 * factor);
            assert_eq!(pre, 0.5 * factor);
        }
    }

    #[test]
    fn initial_excision_scales_reject_nonpositive_atmospheres() {
        let prims = vec![vec![1.0, 0.0, 0.0, 0.0]];
        assert!(initial_excision_scales(&prims).is_err());
    }
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

/// the driven (dynamic) boundary prescriptions in registry order, `Driven(id)` naming the id-th:
/// each lowered against the run's regime and handed to `register` with its parameters
fn attach_driven_boundaries<K>(
    mut ks: K,
    driven: &[String],
    spec: &symbi_hydro::regime_spec::RegimeSpec,
    register: impl Fn(K, symbi_source_compile::expr_bridge::BoundaryPrescription, Vec<f64>) -> K,
) -> Result<K, String> {
    for json in driven {
        let bcfg = symbi_source_compile::SourceConfig::from_json(json)
            .map_err(|e| format!("boundary expression parse: {e}"))?;
        let built = symbi_source_compile::expr_bridge::build_boundary_dag(&bcfg, spec)
            .map_err(|e| format!("boundary expression lower: {e}"))?;
        ks = register(ks, built, bcfg.params.clone());
    }
    Ok(ks)
}

/// register the configured neumann / robin walls on a kernel set in registry order, so the
/// `Neumann(id)` / `Robin(id)` faces name the same coefficients on every level, tile and worker. a
/// level or tile away from every wall carries no such face and the table stays unread.
fn attach_gradient_walls<K>(
    mut sub: K,
    walls: &[GradientBc],
    register: impl Fn(K, GradientBc) -> K,
) -> K {
    for wall in walls {
        sub = register(sub, wall.clone());
    }
    sub
}

/// the polar seam a decomposed run carries, in process or launched: a spherical shell gridded
/// over a pole, or a cylinder holding the R = 0 axis, with the azimuth wrapping and cut into
/// tiles. the ghost band beyond such a pole is the interior half a turn away on other tiles, so
/// the schedule carries antipodal transfers for it and the pole faces of the tiles are cuts. an
/// uncut azimuth needs none: the tile's own half-turn fill covers the band. admission holds a
/// gridded azimuth under an axis face to a periodic full turn of even cell count, the domain the
/// half-turn maps onto itself. `slab(azimuth, i)` is the i-th azimuth tile's interior cell count.
fn polar_seam<const D: usize>(
    coord_system: &str,
    phys: &Boundaries<D>,
    topology: &symbi::sim::decomp::Topology<D>,
    counts: [usize; D],
    slab: impl Fn(usize, usize) -> usize,
) -> Option<symbi::sim::decomp::PolarSeam> {
    if D != 3 {
        return None;
    }
    let (mirror, azimuth) = match coord_system {
        "spherical" => (1, 2),
        "cylindrical" => (0, 1),
        _ => return None,
    };
    let sides = [phys.lo(mirror) == BoundaryType::Axis, phys.hi(mirror) == BoundaryType::Axis];
    if !(sides[0] || sides[1]) || !topology.is_periodic(azimuth) || counts[azimuth] < 2 {
        return None;
    }
    Some(symbi::sim::decomp::PolarSeam {
        mirror,
        sides,
        azimuth,
        slabs: (0..counts[azimuth]).map(|i| slab(azimuth, i)).collect(),
    })
}

/// a tile's pole faces on a seam are cuts the antipodal transfers fill
fn seam_faces_as_cuts<const D: usize>(
    mut bnd: Boundaries<D>,
    seam: Option<&symbi::sim::decomp::PolarSeam>,
) -> Boundaries<D> {
    if let Some(s) = seam {
        for side in 0..2 {
            if s.sides[side] && bnd.0[s.mirror][side] == BoundaryType::Axis {
                bnd.0[s.mirror][side] = BoundaryType::CoarseFine;
            }
        }
    }
    bnd
}

/// what a decomposed run may carry once an axis wraps. closing the domain seam with exchanged
/// legs moves the periodic law out of the tile faces and into the schedule. what still reads a
/// tile's own faces to learn it is refused rather than approximated.
fn wrap_capability(periodic: bool, wb_reconstruction: bool, has_bodies: bool) -> Result<(), String> {
    if !periodic {
        return Ok(());
    }
    if wb_reconstruction && has_bodies {
        return Err(
            "a periodic axis with an immersed body cannot use the balance-aware \
                    reconstruction: the periodic images sit at different body potentials, so \
                    the isentrope carried across the seam is a real state change. use outflow \
                    or reflect walls, or run the plain reconstruction."
                .to_string(),
        );
    }
    Ok(())
}

/// the whole grid a run decomposes: the configured cells, origin, spacing, coordinate maps and
/// declared boundaries every tile is placed on.
fn whole_grid<const D: usize>(cfg: &Config) -> WholeGrid<D> {
    WholeGrid {
        cells: std::array::from_fn(|ax| cfg.n_cells[ax]),
        origin: std::array::from_fn(|ax| cfg.x_lo[ax]),
        spacing: std::array::from_fn(|ax| cfg.dx[ax]),
        maps: axis_maps::<D>(cfg),
        boundaries: boundaries_nd::<D>(&cfg.boundaries),
    }
}

/// the halo topology on the run's chart: a wrapping axis the metric depends on carries distinct
/// end faces. the radius of a cylinder or a sphere and the polar angle vary face areas and edge
/// lengths along themselves; a curved spacetime varies them along every axis.
fn chart_topology<const D: usize>(cfg: &Config, whole: &WholeGrid<D>, counts: [usize; D]) -> Topology<D> {
    let curved = cfg.spacetime != "minkowski";
    let distinct = std::array::from_fn(|ax| {
        curved
            || match cfg.coord_system.as_str() {
                "cylindrical" => ax == 0,
                "spherical" => ax <= 1,
                _ => false,
            }
    });
    whole.topology(counts).with_distinct_ends(distinct)
}

fn seed_configured_tracers<const D: usize, const DOF: usize, Mem: symbi::symbi_xpu::MemorySpace>(
    sim: &symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    cfg: &Config,
) -> symbi_sim::tracers::TracerSet<D> {
    if cfg.cohort_ic.is_empty() {
        symbi_sim::tracers::seed_mass_weighted(sim, cfg.n_tracers)
    } else {
        symbi_sim::tracers::seed_mass_weighted_with_cohorts(sim, cfg.n_tracers, &cfg.cohort_ic)
            .unwrap_or_else(|detail| panic!("tracer cohort seeding: {detail}"))
    }
}

fn configured_ito_order(
    cfg: &Config,
) -> Result<Option<symbi_sim::mass_transport::ItoOrder>, String> {
    match cfg.tracer_scheme.as_str() {
        "discrete" => Ok(None),
        "ito2" => Ok(Some(symbi_sim::mass_transport::ItoOrder::Two)),
        "ito3" => Ok(Some(symbi_sim::mass_transport::ItoOrder::Three)),
        scheme => Err(format!(
            "tracer_scheme must be one of discrete, ito2, or ito3; got '{scheme}'"
        )),
    }
}

/// lay the declared perturbation over `$target` (a sim or a hierarchy) at its own cell centers: a
/// delta on density, each velocity component and, on an energy regime, the pressure. a fresh start
/// only; a restart restores the state wholesale. yields whether a perturbation was laid.
macro_rules! lay_perturbation {
    ($target:expr, $cfg:expr, $dof:literal, $has_energy:expr) => {{
        let cfg: &Config = $cfg;
        match cfg.perturbation_json.as_ref().filter(|_| cfg.restart_path.is_none()) {
            None => false,
            Some(declared) => {
                let cfg_p = symbi_source_compile::EquilibriumConfig::from_json(declared)
                    .map_err(|e| format!("perturbation parse: {e}"))?;
                let expression = cfg_p
                    .to_expression()
                    .map_err(|e| format!("perturbation lower: {e:?}"))?;
                let has_energy: bool = $has_energy;
                let expected = 1 + $dof + usize::from(has_energy);
                if cfg_p.outputs.len() != expected {
                    return Err(format!(
                        "perturbation_expressions supplies {} component(s); this run needs \
                         {expected} — density, {} velocity component(s){}",
                        cfg_p.outputs.len(),
                        $dof,
                        if has_energy { ", and pressure" } else { "" }
                    ));
                }
                $target.perturb_cells(|x, p| {
                    let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                    let d = expression.eval(at(0), at(1), at(2), 0.0);
                    let mut p = p;
                    p = p.with_rho(Density(p.rho() + d[0]));
                    for ax in 0..$dof {
                        p.vel_mut()[ax] += d[1 + ax];
                    }
                    // an energy-free regime's slot discards whatever it is handed, so the
                    // pressure delta needs no branch: it is a no-op there by construction.
                    {
                        use symbi_hydro::energy::EnergySlot;
                        let dp = d.get(1 + $dof).copied().unwrap_or(0.0);
                        p = p.with_pre(p.pre().add(EnergySlot::from_scalar(dp)));
                    }
                    p
                });
                true
            }
        }
    }};
}

/// a locally isothermal tile's closure cs^2 = p / rho on every allocated cell, read from the whole
/// grid's initial rows at the nearest whole-grid interior cell: the interior and the cut ghosts
/// take their own cells' values and the physical ghosts the clamped continuation, the values the
/// single grid holds at the same cells. the closure is fixed in time, so no exchange carries it.
macro_rules! seed_local_cs2 {
    ($sim:expr, $sub:expr, $cfg:expr, $prims:expr, $tile_lo:expr, $n:expr, $d:literal) => {{
        let cfg: &Config = $cfg;
        if cfg.locally_isothermal {
            let ilo: [isize; $d] = std::array::from_fn(|ax| $sim.geom.interior.spaces[ax].lo);
            for c in $sim.geom.allocated.iter() {
                let g: [isize; $d] = std::array::from_fn(|ax| {
                    (c[ax] - ilo[ax] + $tile_lo[ax] as isize).clamp(0, $n[ax] as isize - 1)
                });
                let row = &$prims[global_lin::<$d>(&[0; $d], &g, &$n)];
                let cs2 = row[1 + $d] / row[0];
                $sub.cs2.set(c, cs2);
                if let Some(dst) = $sim.fields.cs2.as_ref() {
                    dst.set(c, cs2);
                }
            }
        }
    }};
}

/// an isothermal MHD tile's closure cs^2 = p / rho on every allocated cell, read from the whole
/// grid's initial rows (pressure in the fifth column) at the cell the single grid continues it
/// from: the periodic image on an axis the domain closes into a ring, the nearest whole-grid
/// interior cell otherwise. the closure is fixed in time, so no exchange carries it.
macro_rules! seed_mhd_local_cs2 {
    ($sim:expr, $cfg:expr, $prims:expr, $tile_lo:expr, $n:expr, $grid:expr, $d:literal) => {{
        let cfg: &Config = $cfg;
        if cfg.locally_isothermal {
            if $prims.first().is_some_and(|row: &Vec<f64>| row.len() < 5) {
                return Err(format!(
                    "locally isothermal MHD prim row has {} components, expected 5 (rho, v1..v3, p_local)",
                    $prims[0].len()
                ));
            }
            let wraps: [bool; $d] = std::array::from_fn(|ax| $grid.periodic(ax));
            let ilo: [isize; $d] = std::array::from_fn(|ax| $sim.geom.interior.spaces[ax].lo);
            let cs2 = $sim.fields.cs2.as_ref().expect("the isothermal closure field exists on an energy-free regime");
            for c in $sim.geom.allocated.iter() {
                let g: [isize; $d] = std::array::from_fn(|ax| {
                    let whole = c[ax] - ilo[ax] + $tile_lo[ax] as isize;
                    let cells = $n[ax] as isize;
                    if wraps[ax] { whole.rem_euclid(cells) } else { whole.clamp(0, cells - 1) }
                });
                let row = &$prims[global_lin::<$d>(&[0; $d], &g, &$n)];
                cs2.set(c, row[4] / row[0]);
            }
        }
    }};
}

/// a tile's sim with the passive scalar allocated and seeded from the global dye (an undyed run
/// passes the sim through)
macro_rules! with_tile_dye {
    ($sim:expr, $cfg:expr, $tile_lo:expr, $n:expr, $flat:expr) => {{
        let sim = $sim;
        if $cfg.chi_ic.is_empty() {
            sim
        } else {
            let sim = sim
                .with_passive_scalar()
                .map_err(|e| format!("tile {} dye allocation: {e:?}", $flat))?;
            seed_tile_dye(&sim, &$cfg.chi_ic, $tile_lo, $n);
            sim
        }
    }};
}

/// seed a tile's passive scalar from the global initial dye: cell by cell, the tile's local cell
/// maps to its global cell (`tile_lo` + local offset) and the conserved dye is rho * chi.
fn seed_tile_dye<const D: usize, const DOF: usize, Mem: symbi::symbi_xpu::MemorySpace>(
    store: &symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    chi_ic: &[f64],
    tile_lo: [usize; D],
    n: [usize; D],
) {
    let ilo: [isize; D] = std::array::from_fn(|ax| store.geom.interior.spaces[ax].lo);
    let cons_chi = store.fields.cons.chi_field().expect("cons chi");
    let prim_chi = store.fields.prim.chi_field().expect("prim chi");
    for c in store.geom.interior.iter() {
        let mut lin = 0usize;
        let mut stride = 1usize;
        for ax in 0..D {
            lin += (tile_lo[ax] + (c[ax] - ilo[ax]) as usize) * stride;
            stride *= n[ax];
        }
        let chi = chi_ic[lin];
        let rho = *store.fields.cons.den.view().at(c);
        cons_chi.view_mut().set(c, rho * chi);
        prim_chi.view_mut().set(c, chi);
    }
}

/// install the configured censuses onto a store. lowering already happened at preflight, where
/// a malformed registration is rejected before any grid is allocated; this is the step that makes
/// the run actually carry them. omitting it leaves a census that validates, reports no error, and
/// records nothing — a checkpoint with no census group reads identically to a run that skipped
/// registration entirely.
fn attach_configured_censuses<
    const D: usize,
    const DOF: usize,
    Mem: symbi::symbi_xpu::MemorySpace,
>(
    store: &mut symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    cfg: &Config,
) -> Result<(), String> {
    for evaluator in lower_configured_censuses(&cfg.census_jsons)? {
        store
            .censuses
            .push(symbi_sim::census::RegisteredCensus::new(evaluator));
    }
    Ok(())
}

fn attach_configured_tracers<
    const D: usize,
    const DOF: usize,
    Mem: symbi::symbi_xpu::MemorySpace,
>(
    sim: &mut symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    cfg: &Config,
) -> Result<(), String> {
    let seed = seed_configured_tracers(sim, cfg);
    if let Some(order) = configured_ito_order(cfg)? {
        sim.continuous_tracers = Some(
            symbi_sim::tracers::ContinuousTracerSet::from_discrete(&seed, order)
                .map_err(|detail| format!("continuous tracer seeding: {detail}"))?,
        );
    } else {
        sim.tracers = Some(seed);
    }
    Ok(())
}

fn convert_seeded_tracers<const D: usize, const DOF: usize, Mem: symbi::symbi_xpu::MemorySpace>(
    sim: &mut symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    order: symbi_sim::mass_transport::ItoOrder,
) -> Result<(), String> {
    let seed = sim
        .tracers
        .take()
        .ok_or_else(|| "continuous tracer conversion requires a discrete seed".to_string())?;
    sim.continuous_tracers = Some(
        symbi_sim::tracers::ContinuousTracerSet::from_discrete(&seed, order)
            .map_err(|detail| format!("continuous tracer conversion: {detail}"))?,
    );
    Ok(())
}

fn partition_configured_tracers<
    const D: usize,
    const DOF: usize,
    Mem: symbi::symbi_xpu::MemorySpace,
>(
    sim: &symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    cfg: &Config,
    partition: &Partition<D>,
) -> Vec<symbi_sim::tracers::TracerSet<D>> {
    if cfg.cohort_ic.is_empty() {
        symbi_sim::tracers::seed_and_partition(sim, cfg.n_tracers, partition)
    } else {
        symbi_sim::tracers::seed_and_partition_with_cohorts(
            sim,
            cfg.n_tracers,
            partition,
            &cfg.cohort_ic,
        )
        .unwrap_or_else(|detail| panic!("tracer cohort partition: {detail}"))
    }
}

/// the number of scheduled checkpoint boundaries already at or before `resume_time`, i.e. the
/// count a restart must skip so the next write lands on the first boundary strictly in the future.
/// `boundary(fired)` is the (fired+1)-th scheduled checkpoint time and is monotonic increasing
/// (log: anchor*10^((fired+1)*dlogt); linear: (fired+1)*interval). this skip keeps a restart from
/// re-dumping the checkpoint it resumed from, which would duplicate a file and shift every later
/// index by one, whenever the schedule is anchored at a fixed reference below the resume clock.
/// a fresh run (`is_restart` false) skips nothing. the loop terminates: a finite monotonic
/// cadence exceeds any finite `resume_time`, and a disabled cadence (`boundary` returns a
/// non-finite sentinel) stops at once.
fn checkpoints_at_or_before(
    is_restart: bool,
    resume_time: f64,
    boundary: impl Fn(u64) -> f64,
) -> u64 {
    if !is_restart {
        return 0;
    }
    let mut fired = 0u64;
    while boundary(fired).is_finite() && time_at_or_after(resume_time, boundary(fired)) {
        fired += 1;
    }
    fired
}

/// the checkpoint schedule every driver walks, single grid, host tiles and launched workers
/// alike. a log cadence (`dlogt > 0` over a positive anchor, `checkpoint_log_anchor` or else the
/// start time) lands the k-th checkpoint at anchor*10^(k*dlogt); a linear one at k*interval from
/// time zero, so a resumed run walks the fresh run's boundaries. a restart skips every boundary
/// at or before its clock, so its first write lands on the next future one and reuses no index.
/// a log run names its files by index, zero-padded to the projected count; a linear one by time.
#[derive(Debug, Clone)]
struct CheckpointCadence {
    log: bool,
    anchor: f64,
    dlogt: f64,
    interval: f64,
    fired: u64,
    next: f64,
    /// the index of the next file written
    index: u64,
    idx_width: usize,
}

impl CheckpointCadence {
    fn of(cfg: &Config) -> Self {
        let interval = if cfg.checkpoint_interval > 0.0 { cfg.checkpoint_interval * cfg.time_unit } else { f64::INFINITY };
        let anchor = if cfg.checkpoint_log_anchor > 0.0 { cfg.checkpoint_log_anchor } else { cfg.start_time };
        let log = cfg.dlogt > 0.0 && anchor > 0.0;
        let mut cadence = Self { log, anchor, dlogt: cfg.dlogt, interval, fired: 0, next: f64::INFINITY, index: cfg.checkpoint_index + 1, idx_width: 0 };
        cadence.fired = checkpoints_at_or_before(cfg.checkpoint_index > 0, cfg.start_time, |k| cadence.boundary(k));
        cadence.next = cadence.boundary(cadence.fired);
        // `ceil(log10(max_index + 1))` digits: robust at a power-of-ten boundary; an overshoot only
        // widens later names, every reader sorting numerically
        if log {
            let projected = (cfg.t_final / anchor).log10() / cfg.dlogt + cfg.checkpoint_index as f64;
            cadence.idx_width = ((projected.max(1.0) + 1.0).log10().ceil() as usize).max(1);
        }
        cadence
    }

    /// the (k+1)-th scheduled checkpoint time
    fn boundary(&self, k: u64) -> f64 {
        if self.log {
            self.anchor * 10f64.powf((k + 1) as f64 * self.dlogt)
        } else if self.interval.is_finite() {
            (k + 1) as f64 * self.interval
        } else {
            f64::INFINITY
        }
    }

    /// a scheduled checkpoint is due at `time`
    fn due(&self, time: f64) -> bool {
        self.next.is_finite() && time_at_or_after(time, self.next)
    }

    /// the name tag of a file written at `time` under index `index`
    fn tag(&self, cfg: &Config, time_width: usize, time: f64, index: u64) -> String {
        checkpoint_tag(cfg, self.idx_width, time_width, time, index)
    }

    /// after the scheduled write at `time`: the next index, and the schedule advanced past every
    /// boundary the step crossed, one write covering them all
    fn advance(&mut self, time: f64) {
        self.index += 1;
        self.fired += 1;
        self.next = self.boundary(self.fired);
        while self.due(time) {
            self.fired += 1;
            self.next = self.boundary(self.fired);
        }
    }
}

#[cfg(test)]
mod checkpoint_schedule_tests {
    use super::{checkpoints_at_or_before, time_at_or_after};

    #[test]
    fn clock_boundary_comparison_tracks_time_units() {
        for factor in [1e-100_f64, 1.0, 1e100] {
            let boundary = factor;
            let roundoff = 16.0 * f64::EPSILON * factor;
            assert!(time_at_or_after(boundary - roundoff, boundary));
            assert!(!time_at_or_after(0.9 * boundary, boundary));
        }
    }

    // log cadence anchored at a fixed reference: the k-th checkpoint lands at anchor*10^(k*dlogt).
    // cp_at(fired) is the (fired+1)-th boundary. a restart at index k resumes with the clock at the
    // k-th boundary, so the skip must consume exactly k boundaries and leave the next write on the
    // (k+1)-th — same index it would carry in an uninterrupted run, writing exactly once at the
    // resume time. this is the "restart from the correct index" invariant.
    #[test]
    fn log_restart_skips_to_the_next_future_boundary() {
        let anchor = 1.0_f64;
        let dlogt = 0.1_f64;
        let cp_at = |fired: u64| anchor * 10f64.powf((fired + 1) as f64 * dlogt);

        for k in 1..=8u64 {
            let resume_time = anchor * 10f64.powf(k as f64 * dlogt);
            let skipped = checkpoints_at_or_before(true, resume_time, cp_at);
            assert_eq!(skipped, k, "restart at index {k} must skip {k} boundaries");
            // the next scheduled write is strictly in the future and is the (k+1)-th boundary.
            assert!(
                cp_at(skipped) > resume_time + 1e-12,
                "restart at index {k} would re-dump the resumed checkpoint"
            );
            assert!((cp_at(skipped) - anchor * 10f64.powf((k + 1) as f64 * dlogt)).abs() < 1e-12);
        }
    }

    // when the anchor defaults to the resume clock (no fixed checkpoint_log_anchor) the schedule
    // re-anchors at start_time, so the first boundary is already in the future and nothing is
    // skipped — cp_fired stays 0, matching the pre-existing correct behavior for that anchoring.
    #[test]
    fn log_restart_reanchored_at_resume_skips_nothing() {
        let resume_time = 4.2_f64;
        let dlogt = 0.1_f64;
        let cp_at = |fired: u64| resume_time * 10f64.powf((fired + 1) as f64 * dlogt);
        assert_eq!(checkpoints_at_or_before(true, resume_time, cp_at), 0);
    }

    // linear cadence: boundary (fired+1) at (fired+1)*interval. a restart at index k (clock at
    // k*interval) skips k boundaries and resumes on the (k+1)-th.
    #[test]
    fn linear_restart_skips_to_the_next_future_boundary() {
        let interval = 0.2_f64;
        let cp_at = |fired: u64| (fired + 1) as f64 * interval;
        let resume_time = 5.0 * interval;
        assert_eq!(checkpoints_at_or_before(true, resume_time, cp_at), 5);
        assert!((cp_at(5) - 6.0 * interval).abs() < 1e-12);
    }

    // a fresh run skips zero boundaries, regardless of the schedule, so the first checkpoint keeps index 0->1.
    #[test]
    fn fresh_run_skips_nothing() {
        let cp_at = |fired: u64| (fired + 1) as f64 * 0.2;
        assert_eq!(checkpoints_at_or_before(false, 1_000.0, cp_at), 0);
    }

    // a disabled cadence returns a non-finite sentinel; the skip loop must terminate at once
    // on that sentinel.
    #[test]
    fn disabled_cadence_terminates() {
        let cp_at = |_fired: u64| f64::INFINITY;
        assert_eq!(checkpoints_at_or_before(true, 1_000.0, cp_at), 0);
    }
}

/// 1-level `Hierarchy` (`Hierarchy::single`), an AMR run is a multi-level one — the
/// driver (`evolve_with_callback`) and the writer (`write_hierarchy_checkpoint`,
/// all levels in one file) are identical either way. no separate single-grid vs
/// AMR startup. generic over the concrete monomorphized hierarchy the macro builds.
fn run_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    hier: &mut Hierarchy<R, D, DOF, M, E, S, Mem, K>,
    cfg: &Config,
) -> Result<symbi_sim::run_diagnostics::RunDiagnostics, Box<dyn std::error::Error>>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    let data_dir = &cfg.data_dir;
    let guard = retry_guard_of(cfg);
    let exclusion = retry_exclusion_of(cfg);
    for level in hier.levels.iter_mut() {
        level.state.retry_guard = guard;
        level.state.retry_exclusion = exclusion;
    }
    if let Some(path) = cfg.restart_path.as_deref() {
        // a restart may run deeper than the file it resumes from — the bootstrap ladder, where a
        // rung converges at its own resolution and the next resumes it with one more level. the
        // sequence (count the file's levels, verify their grids, load them, inject the rest) lives
        // on the hierarchy so it is reachable from a test, not only from here.
        for level in hier.levels.iter_mut() {
            level.state.tracers = None;
            level.state.continuous_tracers = None;
        }
        hier.restore_from_checkpoint(path)?;
        if hier.levels.len() > 1
            && hier.levels.iter().any(|level| {
                level.state.tracers.is_some() || level.state.continuous_tracers.is_some()
            })
        {
            return Err(format!(
                "refined checkpoint '{path}' contains tracers; hierarchy restart requires \
                 repartitioning the global tracer population"
            )
            .into());
        }
    }
    announce_retry_guard(&hier.levels[0].state);
    // the checkpoint cadence is in natural units: `checkpoint_interval * time_unit`
    // is the code-unit spacing, so `checkpoint_interval = 0.1` with a binary's
    // orbital `time_unit` means "every 0.1 orbits". default time_unit = 1.0 keeps
    // the cadence in code units, unchanged for ordinary runs.
    // the checkpoint schedule: log or linear, a restart skipping the boundaries already behind it
    let mut cadence = CheckpointCadence::of(cfg);
    let cp_idx_width = cadence.idx_width;

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

    // zone-cycle throughput: the actual interior cell-updates per root step. for a refined run this
    // counts every level's own work -- each level ll subcycles ratio^ll times per root step over its
    // own (finer, larger) interior, so the honest count is sum_ll (interior_cells_ll * ratio^ll). a
    // single-level run reduces to the base interior, unchanged. this accounting is what keeps AMR
    // from reporting only the coarse zones while the wall-clock includes all the hidden fine work,
    // which would read the rate ~ratio^d too low. (ratio = 2, the baked transfer ratio.)
    let n_zones: u64 = {
        let mut eff = 0u64;
        let mut subcycle = 1u64; // ratio^ll for level ll
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
    // the existing diagnostics.dat is the authority for the restart seam: whether the
    // resume-clock sample is already on disk, and where the cadence resumes so no
    // historical interval is re-emitted.
    let diag_plan = diag_path.as_ref().map(|dp| {
        plan_diagnostics(last_diagnostics_time(dp), cfg.start_time, diag_interval)
            .unwrap_or_else(|detail| panic!("{detail}"))
    });
    let mut next_diag = diag_plan.as_ref().map_or(diag_interval, |p| p.next);
    let start = std::time::Instant::now();
    // checkpoint-write seconds, accumulated on every run regardless of SYMBI_PROFILE: the
    // exit summary reports the sustained integration rate with i/o excluded — on a
    // short run the h5 writes are a large slice of wall and dilute the displayed rate.
    let io_secs = std::cell::Cell::new(0.0f64);
    let mut last_inst = start;
    let mut last_iter = hier.levels[0].state.iteration;
    // FOFC observability: zero the deliberate-fallback counters at run start, track the last-seen
    // totals so the benchmark cadence can post the per-window delta (a limiter that fired always
    // shows). cumulative totals also close out the run summary.
    symbi::regimes::fofc::fofc_reset_stats();
    // on a cartesian black-hole run, the FOFC counters are reported split at the outer horizon
    // r_+ = M + sqrt(M^2 - a^2), the true spin-dependent radius (cells in (r_+, 2M) at nonzero
    // spin sit outside the horizon): the exterior subtotal is the production signal.
    symbi::regimes::fofc::fofc_set_horizon_radius(outer_horizon_radius(cfg));
    let mut last_fofc: (u64, u64) = (0, 0);
    let mut last_fofc_h: (u64, u64) = (0, 0);
    let mut horizon_note_posted = false;

    // graceful-interrupt trap: a caught signal (Ctrl-C, scheduler eviction)
    // flips `stop_requested`; the loop then snapshots a restart checkpoint and breaks.
    // Drop restores python's handlers + the cursor no matter how the run ends.
    let guard = SignalGuard::install();
    // btop-style live tui: draw the dashboard in the alternate screen so it
    // leaves no scrollback trail; on exit the primary buffer is restored and
    // re-render one static final frame so the result persists.
    let mut screen = ScreenGuard::enter();
    // a render thread owns the terminal + input and draws at ~30 fps, so
    // tab / pause respond instantly regardless of step rate. `None` off a tty (the
    // static string path renders headless). the solver publishes snapshots + reads
    // its control flags, leaving key polling entirely to the render thread.
    let mut dash = LiveDashboard::spawn();

    // prime the IC: derive primitives (c2p) + cell-centered B (bcell-from-bface)
    // from the seeded conserved/face state before snapshotting, so the t=0
    // checkpoint carries real primitives (the reader reads primitives — an
    // unprimed IC's zeroed scratch buffers plot as all zeros). idempotent
    // with the prime the evolve driver runs at its own start.
    if cfg.restart_path.is_none() {
        hier.prime();
    }

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
            if diag_plan.as_ref().is_some_and(|p| p.write_initial) {
                if let Some(im) = hier.levels.last().and_then(|l| l.state.immersed.as_ref()) {
                    let _ = append_diagnostics(dp, cfg.start_time, &im.bodies);
                    table.post_diagnostic(&format!(
                        "diagnostics {dp}  ({}, initial)",
                        fmt_time_msg(cfg, cfg.start_time)
                    ));
                }
            }
        }
        set_row(&mut table, i0, t0, d0, t_final, 0.0);
        publish_or_refresh(dash.as_ref(), &mut table);
    }

    // live-dashboard control flags, toggled by keypresses inside the callback.
    // `paused` parks the integrator (no step) while keeping the ui live; `user_quit`
    // requests a graceful stop, handled exactly like a caught signal.
    let mut user_quit = false;
    // the energy-split warning fires at most once per run; see the check at the
    // conservation-diagnostic site.
    let mut ke_split_warned = false;

    let callback = |h: &Hierarchy<R, D, DOF, M, E, S, Mem, K>| {
        let st = &h.levels[0].state;
        let (iter, time, dt) = (st.iteration, st.time, st.dt);
        let mut dirty = false;

        // bounded march: stop at the `max_steps` root iteration (0 = unbounded). the break
        // exits through the success epilogue, which writes the final checkpoint — a truncated
        // but otherwise ordinary run.
        if step_cap_reached(cfg, iter) {
            return std::ops::ControlFlow::Break(());
        }

        // live-dashboard input: the render thread owns the keys + sets
        // control flags; the solver only reads them. space parks the integrator
        // here (the render thread keeps drawing); q -> graceful quit; s -> single
        // step; w -> force checkpoint. Ctrl-C is still a sigint to the guard.
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
            let path = checkpoint_name(
                cfg,
                &cadence.tag(cfg, cp_width, time, cadence.index),
            );
            let t_io = std::time::Instant::now();
            let res =
                write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cadence.index));
            io_secs.set(io_secs.get() + t_io.elapsed().as_secs_f64());
            match res {
                Ok(_) => table.post_success(&format!(
                    "checkpoint {path}  ({}, manual)",
                    fmt_time_msg(cfg, time)
                )),
                Err(e) => table.post_error(&format!("manual checkpoint failed: {e:?}")),
            }
            cadence.index += 1;
            dirty = true;
        }

        // interrupt: a caught signal or a user 'q'. write a numbered + canonical
        // restart checkpoint so a cluster eviction / quit can resume, then stop.
        // the handler has already left the alternate screen on a signal, so switch
        // the table to static before any further render of the primary buffer.
        if guard.stop_requested() || user_quit {
            table.set_dynamic(false);
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let restart =
                checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Interrupted));
            let _ =
                write_hierarchy_checkpoint(&states, &restart, &checkpoint_metadata(cfg, cadence.index));
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

        // fatal crash (set by the evolve loop: the cfl watchdog on a NaN / collapsed wave speed,
        // or a panic caught inside the step — the FOFC freeze-streak halt): snapshot the last
        // computed state as a `.crashed` checkpoint so it can be inspected, then stop.
        if let Some(c) = &h.crash {
            table.set_dynamic(false);
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
            let _ =
                write_hierarchy_checkpoint(&states, &crashed, &checkpoint_metadata(cfg, cadence.index));
            table.post_error(&format!(
                "crashed at {} (step {}) — state checkpoint {crashed}",
                fmt_time_msg(cfg, c.time),
                c.iter,
            ));
            return std::ops::ControlFlow::Break(());
        }

        // message board cadence: checkpoints fire on the time schedule. a single
        // large dt can cross multiple interval boundaries (e.g., a cold-medium CFL
        // step, or a coarse cadence); write exactly one checkpoint for the current
        // state and advance the schedule past every boundary it crossed. the skipped
        // intermediate states were never computed, and the file name is keyed by
        // the current time — looping would just re-write the same file N times and
        // spam the board with identical entries.
        if cadence.due(time) {
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let path = checkpoint_name(
                cfg,
                &cadence.tag(cfg, cp_width, time, cadence.index),
            );
            let t_io = std::time::Instant::now();
            let res = symbi_sim::driver::prof("checkpoint_io", || {
                write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cadence.index))
            });
            io_secs.set(io_secs.get() + t_io.elapsed().as_secs_f64());
            match res {
                Ok(_) => {
                    table.post_success(&format!("checkpoint {path}  ({})", fmt_time_msg(cfg, time)))
                }
                Err(e) => table.post_error(&format!("checkpoint {:04} failed: {e:?}", cadence.index)),
            }
            // advance past every boundary this step crossed (log or linear) so a
            // single large dt yields one write covering all of them.
            cadence.advance(time);
            dirty = true;
        }

        // diagnostics cadence: sample body state on its own (finer) schedule,
        // independent of checkpoints — append a row per body to diagnostics.dat.
        // post one board line per callback that wrote (not per missed interval, so
        // a dt that spans several intervals collapses to a single notice) and mark
        // the frame dirty so the write is visible the moment it happens.
        if let Some(dp) = &diag_path {
            if diagnostics_due(time, &mut next_diag, diag_interval) {
                if let Some(im) = h.levels.last().and_then(|l| l.state.immersed.as_ref()) {
                    let _ = append_diagnostics(dp, time, &im.bodies);
                }
                table.post_diagnostic(&format!("diagnostics {dp}  ({})", fmt_time_msg(cfg, time)));
                dirty = true;
            }
        }

        // benchmark row cadence: update the live row every 100 root iterations,
        // faithfully and independent of the checkpoint cadence — the table need
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
            // limiter is visible (a bounded, high-order-preserving first-order
            // correction is expected; a freeze — where neither order recovered a cell — is rarer and
            // worth flagging). only posts when something fired, so a clean run stays quiet.
            let (fb_total, fz_total) = symbi::regimes::fofc::fofc_stats();
            let (fb_h, fz_h) = symbi::regimes::fofc::fofc_horizon_stats();
            // the exterior deltas are the meaningful signal: fire inside the horizon
            // (the excised interior, the metric-guard ring) is expected and steady,
            // so it gets one note per run.
            let (d_fb_ext, d_fz_ext) = (
                (fb_total - fb_h) - (last_fofc.0 - last_fofc_h.0),
                (fz_total - fz_h) - (last_fofc.1 - last_fofc_h.1),
            );
            let (d_fb_h, d_fz_h) = (fb_h - last_fofc_h.0, fz_h - last_fofc_h.1);
            if d_fb_ext > 0 || d_fz_ext > 0 {
                table.post_diagnostic(&format!(
                    "FOFC: {d_fb_ext} exterior first-order fallback cell-steps{} since last window",
                    if d_fz_ext > 0 {
                        format!(", {d_fz_ext} freezes")
                    } else {
                        String::new()
                    },
                ));
            }
            if (d_fb_h > 0 || d_fz_h > 0) && !horizon_note_posted {
                horizon_note_posted = true;
                table.post_diagnostic(&format!(
                    "FOFC: firing inside the horizon ({d_fb_h} cell-steps this window) — \
                     causally disconnected, expected on an excised run; further interior \
                     fire is tallied silently (run total at exit)"
                ));
            }
            last_fofc = (fb_total, fz_total);
            last_fofc_h = (fb_h, fz_h);
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
            // conservation drift + div(B): a host-side interior reduction (skipped
            // on device-resident gpu runs). cheap once per benchmark cadence.
            if let Some(cd) = st.conservation_diag() {
                table.push_conservation(cd.mass, cd.energy, cd.div_b, cd.max_w);
                // the energy split's conditioning. recovering internal energy from the total
                // is a subtraction whose operands exceed the result by this ratio, so the
                // inversion sheds about log10(ratio) digits each time. an ordinary mach-M
                // flow sits near gamma(gamma-1)M^2/2 -- about 56 at M = 10 -- so the warning
                // threshold is set well above any merely supersonic run and marks a cold,
                // kinetically dominated flow whose internal energy is poorly conditioned.
                // reported once per run: the failure is a slow drift, not a per-step event,
                // and a message board that repeats it every cadence buries everything else.
                if let Some(ratio) = cd.max_ke_over_eint
                    && ratio > KE_OVER_EINT_WARN
                    && !ke_split_warned
                {
                    ke_split_warned = true;
                    table.post_warning(&format!(
                        "energy split ill-conditioned: max KE/e_int = {ratio:.3e}. internal \
                         energy is a 1/{:.0} fraction of the total, so c2p sheds ~{:.1} digits \
                         per inversion and any entropy loss it causes is silent (a cooling gas \
                         keeps the sound speed, and the timestep, comfortable)",
                        1.0 + ratio,
                        ratio.log10(),
                    ));
                }
            }
            // machine card: this (compute) node's hostname / cores and the run's
            // resident memory vs the node's physical ram. rss grows, so re-sample
            // each cadence; an attach client reads the compute node's stats, never the client's own host.
            table.set_host(Some(symbi_display::hostinfo::HostStats::sample()));
            // the bound accelerator and the block shape the root level's interior sweep
            // launches with. the root extent is the representative one: finer levels are
            // smaller and flux-face domains are transverse-expanded.
            let dev_extent: Vec<u32> = (0..cfg.dims)
                .map(|ax| h.levels[0].state.geom.interior.spaces[ax].size() as u32)
                .collect();
            table.set_device(device_stats(cfg.dims, &dev_extent));
            // live field heatmap: a screen-sized decimated density slice (2D/3D-mid;
            // None for 1D or device runs), compositing the nested refinement levels
            // so the refined region shows its fine detail. cost bounded by the
            // ~200-cell cap.
            // the `f`-key selects which field to decimate (density / pressure / W /
            // |B|); composite for amr, single-grid fallback for 1D. the render thread
            // owns the colormap, so Inferno here is just a default it overrides.
            let idx = dash
                .as_ref()
                .map(|d| d.controls().field_kind())
                .unwrap_or(0);
            // the o-key's 3D slice orientation (z / y / x mid-plane); 1D/2D ignore it.
            let orient = dash
                .as_ref()
                .map(|d| d.controls().slice_orient())
                .unwrap_or(0);
            // the +/- zoom exponent: 2^k magnification about the domain center.
            let zoom = dash
                .as_ref()
                .map(|d| d.controls().zoom_level())
                .unwrap_or(0);
            // decimate field `kk` (composite over refinement levels, single-grid
            // fallback for 1D) into a display FieldSlice; the render thread owns the
            // colormap, so Inferno here is just a default it overrides.
            let make_slice = |kk: usize| {
                h.field_slice_composite(200, kk, orient, zoom)
                    .or_else(|| {
                        h.levels[0]
                            .state
                            .field_slice_oriented(200, kk, orient, zoom)
                    })
                    .map(|fd| {
                        let mut label = if cfg.dims >= 3 {
                            let plane = ["z-slice", "y-slice", "x-slice"][orient % 3];
                            format!("{} · {plane}", fd.name)
                        } else {
                            fd.name
                        };
                        if zoom > 0 {
                            label = format!("{label} · {}x", 1usize << zoom.min(4));
                        }
                        let label = label;
                        FieldSlice {
                            label,
                            width: fd.width,
                            height: fd.height,
                            data: fd.data,
                            vmin: fd.vmin,
                            vmax: fd.vmax,
                            cmap: Colormap::Inferno,
                            preserve_aspect: fd.preserve_aspect,
                            log_scale: false,
                        }
                    })
            };
            if cfg.live_monitor {
                // build the full field bundle so `simbi attach` can switch fields
                // client-side; the local tui shows the f-key-selected one. write the
                // read-only snapshot atomically (best-effort — a write failure must
                // never halt the run).
                let bundle: Vec<FieldSlice> = (0..h.levels[0].state.field_count())
                    .filter_map(make_slice)
                    .collect();
                if let Some(sel) = bundle.get(idx.min(bundle.len().saturating_sub(1))) {
                    table.set_field(Some(sel.clone()));
                }
                let mut view = table.diagnostic_view();
                view.field_count = bundle.len();
                // the snapshot is the only channel an attached viewer has, so the solver's
                // own pause state has to travel in it: the viewer cannot reach these
                // controls and must not invent a badge from its local keys. a batch run is
                // off-tty, has no controls at all, and therefore reports integrating.
                view.paused = dash.as_ref().is_some_and(|d| d.controls().paused());
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
    };
    // a run resumed at its step cap takes no step and ends through the same epilogue
    let diagnostics = if step_cap_reached(cfg, hier.levels[0].state.iteration) {
        symbi_sim::run_diagnostics::RunDiagnostics::default()
    } else if cfg.restart_path.is_some() {
        hier.resume_with_callback(cfg.t_final, 1, callback)?
    } else {
        hier.evolve_with_callback(cfg.t_final, 1, callback)?
    };

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
        let restart = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Interrupted));
        let summary = format!(
            "interrupted — {} steps, t = {:.4} · restart {restart}",
            root.iteration, root.time,
        );
        table.post_warning(&summary);
        table.exit_frame(ExitKind::Interrupt, &summary);
        return Ok(diagnostics);
    }

    // crashed: the observer already snapshotted the `.crashed` state. surface the halt as the red
    // crash exit frame.
    if let Some(c) = &hier.crash {
        screen.leave();
        table.set_dynamic(false);
        let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
        // report which halt condition fired and its numbers. these are different failures with
        // different fixes: a caught step panic carries its own message (the FOFC freeze-streak
        // halt names the offending cell and state), a non-finite rate is a poisoned cell, a
        // non-positive one a degenerate state. a positive increase is allowed: it means the
        // maximum signal speed fell and is not itself unsafe. asserting one diagnosis for all
        // sends every investigation down the same wrong path.
        let cause = crash_cause(c.dt_cfl, c.panic.as_deref(), c.halt.as_ref());
        let summary = format!(
            "crashed — {} steps, t = {:.4} — {cause} · state {crashed}",
            c.iter, c.time,
        );
        table.post_error(&summary);
        table.exit_frame(ExitKind::Crash, &summary);
        // every crash ends with a failing exit, so schedulers and chains read a failure signal: a
        // caught panic, a halted root step and a timestep the march could not take alike
        return Err(summary.into());
    }

    let root = &hier.levels[0].state;
    // fail-loud completion: a run whose final state is non-finite must never
    // wear the green box — the in-loop NaN guard only fires on the next cfl,
    // so a run that ends on its bad step (e.g. one degenerate-EOS giant dt)
    // would otherwise be reported as success over garbage.
    let final_finite = root
        .geom
        .interior
        .iter()
        .all(|c| root.fields.cons.den.view().at(c).is_finite());
    if !final_finite {
        let states: Vec<&_> = hier.levels.iter().map(|level| &level.state).collect();
        let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
        write_hierarchy_checkpoint(&states, &crashed, &checkpoint_metadata(cfg, cadence.index))?;
        screen.leave();
        table.set_dynamic(false);
        let summary = format!(
            "CRASHED — final state is non-finite (NaN/inf density) after {} steps at t = {:.4}; \
             diagnostic state {crashed}",
            root.iteration, root.time,
        );
        table.post_error(&summary);
        table.exit_frame(ExitKind::Crash, &summary);
        return Err(summary.into());
    }

    let states: Vec<&_> = hier.levels.iter().map(|level| &level.state).collect();
    let final_path = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Completed));
    let t_io = std::time::Instant::now();
    symbi_sim::driver::prof("checkpoint_io", || {
        write_hierarchy_checkpoint(&states, &final_path, &checkpoint_metadata(cfg, cadence.index))
    })?;
    io_secs.set(io_secs.get() + t_io.elapsed().as_secs_f64());
    let wall = start.elapsed().as_secs_f64();
    // the sustained integration rate excludes checkpoint i/o: what the solver
    // delivers between writes, and the number comparable across run lengths
    // (wall-based rates dilute with checkpoint cadence on short runs).
    let compute = (wall - io_secs.get()).max(1e-9);
    let avg = if wall > 1e-9 {
        n_zones as f64 * root.iteration as f64 / compute
    } else {
        0.0
    };
    // leave the alternate screen, then render the green success exit frame so the
    // run's summary persists on the primary buffer.
    screen.leave();
    table.set_dynamic(false);
    let summary = format!(
        "complete — {} steps, t = {:.4}, {:.2}s ({:.2}s io), {}/s sustained · final {final_path}",
        root.iteration,
        root.time,
        wall,
        io_secs.get(),
        humanize_rate(avg),
    );
    table.post_success(&summary);
    // FOFC run total: report the deliberate fallbacks over the whole run (a quiet run shows
    // nothing). on a horizon-split run the exterior count is the acceptance criterion
    // (exterior == 0 for a production run); the interior tally is informational.
    let (fb_total, fz_total) = symbi::regimes::fofc::fofc_stats();
    let (fb_h, fz_h) = symbi::regimes::fofc::fofc_horizon_stats();
    if fb_total > 0 || fz_total > 0 {
        if fb_h > 0 || fz_h > 0 {
            table.post_diagnostic(&format!(
                "FOFC total: exterior {} fallback cell-steps + {} freezes; inside the horizon \
                 {fb_h} + {fz_h} (causally disconnected)",
                fb_total - fb_h,
                fz_total - fz_h,
            ));
        } else {
            table.post_diagnostic(&format!(
                "FOFC total: {fb_total} first-order fallback cell-steps, {fz_total} freezes"
            ));
        }
    }
    table.exit_frame(ExitKind::Success, &summary);
    dump_profile_if_enabled(root.iteration, n_zones);
    dump_dispatch_profile_if_enabled();
    Ok(diagnostics)
}

/// dump the dispatch micro-profile to stderr when `SYMBI_DISPATCH_PROF` is set. the
/// accumulators are written only under that variable, so a zero call count means it was
/// unset and there is nothing to report.
///
/// the question it answers: of the per-call cost in the AMR transfer path, how much is the
/// registry name lookup (a wide match over kernel names) against the kernel execution
/// itself. a lookup share rivalling execution says the dispatch is the cost rather than the
/// arithmetic, which is a scheduling problem and not a numerics one.
/// the KE/e_int ratio above which the energy split is reported as ill-conditioned.
///
/// an ordinary mach-M flow carries `gamma (gamma - 1) M^2 / 2`, which is about 56 at
/// M = 10 and 500 at M = 30 for gamma = 5/3, so this sits above any merely supersonic run
/// and marks a cold, kinetically dominated one. at this ratio the recovered internal
/// energy is a 0.1 percent fraction of the total and c2p sheds ~3 significant digits per
/// inversion.
const KE_OVER_EINT_WARN: f64 = 1.0e3;

fn dump_dispatch_profile_if_enabled() {
    let (calls, lookup_ns, exec_ns) = symbi::symbi_exec::policy::report_dispatch_profile();
    let total = (lookup_ns + exec_ns) as f64;
    if calls == 0 || total <= 0.0 {
        return;
    }
    let row = |label: &str, ns: u64| {
        eprintln!(
            "  {label:<18} {:>8.1} ms  ({:>4.1}%)   {:.0} ns/call",
            ns as f64 / 1e6,
            100.0 * ns as f64 / total,
            ns as f64 / calls as f64,
        );
    };
    eprintln!("\n--- dispatch overhead over {calls} calls (SYMBI_DISPATCH_PROF) ---");
    row("registry lookup", lookup_ns);
    row("kernel execution", exec_ns);
    eprintln!(
        "  {:<18} {:>8.1} ms  ({:.0} ns/call)\n",
        "TOTAL",
        total / 1e6,
        total / calls as f64
    );
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
    eprintln!(
        "  {:<18} {total:>8.1} ms  (sum of instrumented phases)\n",
        "TOTAL"
    );
}

/// set the live monitor's benchmark row + progress bar without rendering — the
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

/// draw one frame: publish a snapshot to the render thread (live tty) or
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

/// the live-monitor "problem setup" sub-table rows (category, property, value).
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
    // physical domain extent per axis: [x_lo, x_lo + dx*n].
    let domain = (0..cfg.dims)
        .map(|ax| {
            let hi = cfg.x_lo[ax] + cfg.dx[ax] * cfg.n_cells[ax] as f64;
            format!("[{:.3}, {:.3}]", cfg.x_lo[ax], hi)
        })
        .collect::<Vec<_>>()
        .join(" x ");
    // build the panel section by section (the display renders each `[section, ..]` group under a
    // divider header). order = the section order shown. keep each section's rows contiguous.
    let mut rows: Vec<[String; 3]> = Vec::new();
    let mut push = |sec: &str, prop: &str, val: String| rows.push([sec.into(), prop.into(), val]);

    // physics: the regime, EOS, and whatever physics the run actually engages (present-only).
    push("Physics", "regime", cfg.regime.clone());
    push("Physics", "eos", eos_label(cfg));
    if !cfg.bodies.is_empty() {
        push("Physics", "immersed bodies", cfg.bodies.len().to_string());
    }
    // the operator support of every sink on a refined staggered grid, in finest cells: the
    // accretion sphere, the drain mask and the slip shell to their f64 support, and the stencil
    // reach. the finest level contains it, so the margin from the sink to the innermost region's
    // boundary is the number to read when a run is refused.
    if cfg.refinement_enabled && cfg.regime.contains("mhd") {
        if let Some(inner) = cfg.refinement_regions.last() {
            let levels = cfg.refinement_regions.len() as i32;
            let dx_fine = cfg.dx[0] / 2f64.powi(levels);
            let widths = symbi_ib::sdf::mask_support_widths();
            for (idx, b) in cfg.bodies.iter().enumerate() {
                if b.capability & 2 == 0 {
                    continue;
                }
                let r = b.accretion_radius;
                let drain = r + widths * dx_fine;
                let slip = b.magnetic_slip.map_or(r, |m| r + m.placement * m.shell_width + widths * m.shell_width);
                let support = drain.max(slip) + 3.0 * dx_fine;
                let margin = (0..cfg.dims)
                    .map(|ax| {
                        let (lo, hi) = (inner[2 * ax], inner[2 * ax + 1]);
                        let p = b.position.get(ax).copied().unwrap_or(0.0);
                        (p - lo).min(hi - p)
                    })
                    .fold(f64::INFINITY, f64::min);
                push(
                    "Physics",
                    &format!("sink {idx} support"),
                    format!(
                        "{:.1} fine cells of {:.1} to the innermost boundary",
                        support / dx_fine,
                        margin / dx_fine
                    ),
                );
            }
        }
    }
    if !cfg.source_jsons.is_empty() {
        push("Physics", "source term", "active".into());
    }
    if cfg.motion_json.is_some() {
        push("Physics", "mesh motion", "active".into());
    }
    let n_bc = cfg.driven_exprs.len() + cfg.gradient_bcs.len();
    if n_bc > 0 {
        push("Physics", "driven/gradient BCs", n_bc.to_string());
    }

    // geometry
    push("Geometry", "coords", cfg.coord_system.clone());
    push("Geometry", "dimensions", format!("{}D", cfg.dims));
    push(
        "Geometry",
        "resolution",
        format!("{res}  ({n_zones} zones)"),
    );
    push("Geometry", "domain", domain);
    push("Geometry", "boundaries", boundary_label(cfg));

    // numerics: the discretization + solver knobs.
    push("Numerics", "solver", cfg.solver_name.clone());
    push(
        "Numerics",
        "reconstruction",
        cfg.reconstruction_name.clone(),
    );
    if cfg.reconstruction_name == "plm" {
        push("Numerics", "limiter", limiter_label(cfg.plm_theta));
    }
    if cfg.eos_name != "ideal" {
        push("Numerics", "eos", cfg.eos_name.clone());
    }
    if cfg.ppm_flatten_full > cfg.ppm_flatten_onset && cfg.ppm_flatten_full > 0.0 {
        push(
            "Numerics",
            "ppm flatten",
            format!(
                "{:.3} -> {:.3}",
                cfg.ppm_flatten_onset, cfg.ppm_flatten_full
            ),
        );
    }
    push(
        "Numerics",
        "timestepping",
        timestepping_label(cfg.timestepping),
    );
    push("Numerics", "cfl", format!("{:.3}", cfg.cfl));
    if cfg.n_gpus > 1 {
        push("Numerics", "gpus", cfg.n_gpus.to_string());
    }

    // scales: characteristic numbers the physics sets (present-only). timescales
    // are quoted at a unit fiducial radius r = 1 (code units); Omega_k = sqrt(GM/r^3)
    // uses the central body's mass. shown only when the run engages viscosity — the
    // viscous dt cap in particular explains a step pinned far below the advective CFL.
    if cfg.viscosity > 0.0 || cfg.alpha > 0.0 {
        let pi = std::f64::consts::PI;
        let min_dx = (0..cfg.dims)
            .map(|ax| cfg.dx[ax])
            .fold(f64::INFINITY, f64::min);
        // central mass + center: body 0 when present, else GM = 1 about the origin.
        let gm = cfg.bodies.first().map(|b| b.mass).unwrap_or(1.0);
        let center = cfg
            .bodies
            .first()
            .map(|b| b.position.clone())
            .unwrap_or_else(|| vec![0.0; cfg.dims]);
        let omega_k = |r: f64| (gm / (r * r * r)).sqrt();
        // outer radius: the farthest domain corner from the center in the disk plane
        // (first two axes) — where alpha's nu(r) = alpha cs^2 / Omega_k(r) is largest,
        // so it sets the viscous CFL cap (matches the substrate's nu_max).
        let plane = cfg.dims.min(2);
        let mut r_out2 = 0.0_f64;
        for corner in 0..(1usize << plane) {
            let mut d2 = 0.0;
            for a in 0..plane {
                let hi = cfg.x_lo[a] + cfg.dx[a] * cfg.n_cells[a] as f64;
                let x = if corner & (1 << a) != 0 {
                    hi
                } else {
                    cfg.x_lo[a]
                };
                let d = x - center.get(a).copied().unwrap_or(0.0);
                d2 += d * d;
            }
            r_out2 = r_out2.max(d2);
        }
        let r_out = r_out2.sqrt().max(1e-30);
        let is_alpha = cfg.alpha > 0.0;
        let nu_at = |r: f64| {
            if is_alpha {
                cfg.alpha * cfg.cs * cfg.cs / omega_k(r)
            } else {
                cfg.viscosity
            }
        };
        // the explicit parabolic CFL cap dt <= C_visc dx^2 / nu_max, C_visc = 0.1
        // (the 2D/3D Navier-Stokes von-Neumann factor). nu_max at the outer radius.
        let dt_cap = 0.1 * min_dx * min_dx / nu_at(r_out);
        push(
            "Scales",
            if is_alpha { "alpha" } else { "nu" },
            format!("{:.4}", if is_alpha { cfg.alpha } else { cfg.viscosity }),
        );
        push("Scales", "viscous dt cap", format!("{dt_cap:.2e}"));
        // viscous diffusion time t_nu = r^2 / nu at the fiducial radius.
        let nu1 = nu_at(1.0);
        let t_nu = 1.0 / nu1;
        if cfg.bodies.is_empty() {
            push("Scales", "viscous time @r=1", format!("{t_nu:.3}"));
        } else {
            let t_orb = 2.0 * pi / omega_k(1.0);
            push(
                "Scales",
                "viscous time @r=1",
                format!("{t_nu:.2}  ({:.1} orbits)", t_nu / t_orb),
            );
            push("Scales", "orbital time @r=1", format!("{t_orb:.3}"));
            // Reynolds = r v_kep / nu, Mach = v_kep / cs, with v_kep = Omega_k r at r=1.
            let v_kep = omega_k(1.0);
            push("Scales", "Reynolds @r=1", format!("{:.0}", v_kep / nu1));
            if cfg.cs > 0.0 {
                push("Scales", "Mach @r=1", format!("{:.1}", v_kep / cfg.cs));
            }
        }
    }

    // run: the schedule + outputs.
    push("Run", "t_final", t_final_disp);
    push("Run", "checkpoint dt", cp);
    push("Run", "backend build", build_source_label());
    push(
        "Run",
        "config sha256",
        cfg.config_sha256.chars().take(12).collect(),
    );
    if custom_unit {
        push(
            "Run",
            "time unit",
            format!("1 {unit} = {:.4} code", cfg.time_unit),
        );
    }
    push(
        "Run",
        "est. memory",
        format!("{:.3} GB", est_memory_gb(cfg)),
    );
    push("Run", "output", cfg.data_dir.clone());

    drop(push);
    // the config author's own params, grouped by their ProblemParam(group=...); appended last so the
    // core panel reads first, then each config's bespoke parameters.
    rows.extend(cfg.custom_params.iter().cloned());
    rows
}

/// equation-of-state one-liner: ideal gas carries gamma; isothermal regimes
/// carry the (global or position-dependent) sound speed instead. the taub-mathews
/// gas is parameter-free — the gamma it carries is an inert placeholder, and printing
/// it would name a closure the run is not integrating.
fn eos_label(cfg: &Config) -> String {
    let isothermal = cfg.regime.contains("iso") || cfg.regime == "imhd";
    if cfg.locally_isothermal {
        "locally isothermal cs(x)".to_string()
    } else if isothermal {
        format!("isothermal (cs = {:.4})", cfg.cs)
    } else if cfg.eos_name == "synge" {
        "taub-mathews (gamma_eff: 5/3 cold -> 4/3 hot)".to_string()
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

/// the working-set memory estimate (gb), mirroring the python rich summary:
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

/// clip each global refinement region to one tile's physical extent, keeping only the regions that
/// genuinely overlap (more than about half a cell on every axis, so region_to_domain rounds to at
/// least one cell). a tile-local patch survives in exactly one tile; a patch spanning a cut is split
/// into the abutting tiles, each clipped to its own slab, and the per-tile hierarchies share the cut
/// on the fine level, exchanged by `evolve_hierarchy_decomposed`.
///
/// the tile's corners and cell widths are the whole grid's face positions at the tile's own cell
/// indices, so on a log radial axis the far corner sits where the tile's last cell ends. the
/// survival test compares each clipped extent against the tile's narrowest cell, the "narrower
/// than half a cell" rule expressed so it holds when the cells differ in size.
fn clip_regions_to_tile<const D: usize>(
    regions: &[RefinementRegion<D>],
    grid: &WholeGrid<D>,
    tile: &PlacedTile<D>,
) -> Vec<(RefinementRegion<D>, RefinementRegion<D>)> {
    let (origin, hi) = (tile.lower(grid), tile.upper(grid));
    let min_w = tile.narrowest_cell(grid);
    // region k refines level k, whose cells are 2^-k of the root's: a tile keeps the regions down
    // to the first one it holds no cell of, so its levels are the run's first levels in order
    let mut out = Vec::new();
    for (k, r) in regions.iter().enumerate() {
        let lo: [f64; D] = std::array::from_fn(|a| r.x_lo[a].max(origin[a]));
        let up: [f64; D] = std::array::from_fn(|a| r.x_hi[a].min(hi[a]));
        let level_w = 0.5f64.powi(k as i32);
        if !(0..D).all(|a| up[a] - lo[a] > 0.5 * min_w[a] * level_w) {
            break;
        }
        out.push((RefinementRegion { x_lo: lo, x_hi: up }, *r));
    }
    out
}

/// the effective slope-limiter theta passed to the substrate. PLM uses the
/// config `plm_theta` (default 1.5, theta-MC limiter); PCM — i.e., first-order /
/// `order=1` — maps to theta = 0, which collapses minmod3 to a zero slope, so
/// the reconstruction degenerates to piecewise-constant. the substrate has no
/// separate PCM kernel; this is how first-order space is selected.
fn build_theta(cfg: &Config) -> f64 {
    if cfg.reconstruction_name == "pcm" {
        0.0
    } else {
        cfg.plm_theta
    }
}

/// the evolution reconstruction selector for the kernel set. pcm rides the plm kernel
/// at theta = 0 (see `build_theta`); ppm is its own baked kernel family.
fn build_recon(cfg: &Config) -> symbi::Recon {
    if cfg.reconstruction_name == "ppm" {
        symbi::Recon::Ppm
    } else {
        symbi::Recon::Plm
    }
}

/// allocated ghost width: the ppm face pair loads -3..+2 along the sweep, one cell
/// beyond the plm default of 2.
fn ghost_width(cfg: &Config) -> usize {
    if cfg.reconstruction_name == "ppm" {
        3
    } else {
        2
    }
}

/// the eos closure arm for the kernel set. "synge" is the taub-mathews
/// relativistic perfect gas (h = 2.5 theta + sqrt(2.25 theta^2 + 1),
/// parameter-free — the gamma scalar stays bound but inert); anything else is
/// the gamma-law closure.
fn build_eos(cfg: &Config) -> symbi::EosArm {
    if cfg.eos_name == "synge" {
        symbi::EosArm::TaubMathews
    } else {
        symbi::EosArm::IdealGamma
    }
}

/// the host-side closure the `SimState` carries, which must name the same gas as the kernel
/// arm `build_eos` selects. `build_eos` owns c2p, the fluxes and the wave speeds; this one owns
/// the primitive -> conserved conversion that seeds the initial condition, so the two are read
/// at different moments on the same run. seeding through a gamma law and recovering through
/// taub-mathews leaves D = rho W intact and misplaces the rho/W split, which corrupts t = 0 on
/// every synge run. both arms derive from `cfg.eos_name` and sit adjacent so they cannot drift.
/// the run's retry safeguards: the built-in thresholds unless the problem overrides them
/// the outer horizon radius r_+ = M + sqrt(M^2 - a^2) of a cartesian black-hole run, 0 on any
/// other: the radius the FOFC census splits its report at
fn outer_horizon_radius(cfg: &Config) -> f64 {
    if cfg.spacetime != "minkowski" && cfg.coord_system == "cartesian" && cfg.schwarzschild_mass > 0.0 {
        let m = cfg.schwarzschild_mass;
        let a = cfg.kerr_spin.abs().min(m); // |a| <= M; clamp guards the sqrt
        m + (m * m - a * a).sqrt()
    } else {
        0.0
    }
}

/// the retry exclusion this runner configures for a run, applied to every tile and level of every
/// run shape: on a cartesian black-hole run the freeze acts inside the euclidean sphere of the outer
/// horizon radius do not reject a step or count toward the freeze-streak halt, and on every other
/// run each freeze act is evidence. a compatibility choice that reproduces this runner's decision
/// from before the policy was a property of the simulation; it states nothing about which cells
/// the scheme may exclude.
fn retry_exclusion_of(cfg: &Config) -> symbi_sim::substrate_seam::RetryExclusion {
    match outer_horizon_radius(cfg) {
        radius if radius > 0.0 => symbi_sim::substrate_seam::RetryExclusion::LegacyEuclideanSphere { radius },
        _ => symbi_sim::substrate_seam::RetryExclusion::None,
    }
}

fn retry_guard_of(cfg: &Config) -> symbi_sim::retry_guard::RetryGuard {
    symbi_sim::retry_guard::RetryGuard {
        max_rejections: cfg.retry_max_rejections,
        severe_ratio: cfg.retry_severe_ratio,
        sustained_steps: cfg.retry_sustained_steps,
    }
}

/// the retry guard's opening report for a run whose root store is `store`, on stderr
fn announce_retry_guard<const D: usize, const DOF: usize, Mem: symbi::symbi_xpu::MemorySpace>(store: &symbi_sim::state::FieldStore<D, DOF, Mem, f64>) {
    eprintln!(
        "SIMBI {}",
        symbi_sim::retry_guard::opening_report(&store.retry_guard, store.retry_guard_recorded.as_ref(), store.severe_streak)
    );
    eprintln!("SIMBI retry exclusion: {}", store.retry_exclusion);
}

fn host_eos(cfg: &Config) -> EosSelect<f64> {
    if cfg.eos_name == "synge" {
        EosSelect::Tm(TaubMathews)
    } else {
        EosSelect::Ideal(IdealGas { gamma: cfg.gamma })
    }
}

/// build a typed `BodyCollection<f64, D>` from the parsed params. an accretion
/// body becomes a black-hole sink (gravity + accretion onto the body);
/// otherwise it is a fixed-potential gravitating mass. the `two_way_coupling`
/// feedback flag is carried from config.
/// parse each body's optional shape wire into an `SdfExpr`, parallel to `build_bodies`. `None` =
/// the analytic sphere; a malformed shape json fails loud at build, never a silent sphere.
fn build_body_shapes(params: &[BodyParams]) -> Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>> {
    params
        .iter()
        .map(|b| {
            b.shape_json.as_ref().map(|j| {
                symbi_ib::sdf::SdfExpr::<f64, 3>::from_json(j)
                    .unwrap_or_else(|e| panic!("immersed body shape: {e}"))
            })
        })
        .collect()
}

fn build_bodies<const D: usize>(
    params: &[BodyParams],
    binary: Option<&BinaryCfg>,
) -> BodyCollection<f64, D> {
    const ACCRETION: u64 = 2;
    const RIGID: u64 = 1 << 4;
    let mut coll = BodyCollection::new();
    for (idx, b) in params.iter().enumerate() {
        // the body position is cartesian on every grid: the immersed-body kernels
        // map the cell centroid to Cartesian and take |x_cart - position|, so a
        // spherical/cylindrical run gives the body's (x, y, z); the position is
        // never the native (r, theta, phi) tuple. a centered accretor is the origin (0, 0, 0) either way.
        let pos = Tensor::new(std::array::from_fn(|ax| {
            b.position.get(ax).copied().unwrap_or(0.0)
        }));
        let vel = Tensor::new(std::array::from_fn(|ax| {
            b.velocity.get(ax).copied().unwrap_or(0.0)
        }));
        let body = if b.capability & ACCRETION != 0 {
            let bh = Body::black_hole(
                idx,
                pos,
                vel,
                b.mass,
                b.radius,
                b.softening,
                b.sink_rate,
                // sink_delta: the sunset dittmann torque-free dial. the baked
                // uniform-scaling drain ignores it; the backend field lingers
                // only as the body_{idx}_delta expression scalar (defaults 1.0).
                1.0,
                b.accretion_radius,
            );
            // the declared surface stack selects the penalization kernel: a
            // torque-free xi or a porosity dial
            // switches off the pure drain. they are mutually exclusive
            // (the python config rejects declaring both).
            match (b.torque_free_xi, b.porosity) {
                (Some(xi), _) => bh.with_surface(symbi_ib::SurfaceSpec::TorqueFree { xi }),
                (None, Some(porosity)) => bh.with_surface(symbi_ib::SurfaceSpec::Porous {
                    porosity,
                    k_eta_n: b.k_eta_n,
                    k_eta_t: b.k_eta_t,
                }),
                (None, None) => bh,
            }
        } else if b.capability & RIGID != 0 {
            // a rigid wall: the drain-off porous surface. porosity 0 seals the drain
            // channel (no mass removed); the normal channel (k_eta_n) enforces
            // no-penetration and the tangential channel (k_eta_t, zero for free slip)
            // enforces no-slip, both relaxing the gas velocity toward the body.
            {
                let mut body =
                    Body::rigid_sphere(idx, pos, vel, b.mass, b.radius, b.inertia, b.no_slip)
                        .with_surface(symbi_ib::SurfaceSpec::Porous {
                            porosity: 0.0,
                            k_eta_n: b.k_eta_n,
                            k_eta_t: b.k_eta_t,
                        })
                        .with_spin_about(b.omega, Tensor::new(b.spin_axis));
                // anisotropic principal moments override the isotropic default when specified
                // (all-zero sentinel = unspecified).
                if b.inertia_principal.iter().any(|&m| m > 0.0) {
                    body = body.with_inertia_principal(b.inertia_principal);
                }
                body
            }
        } else {
            Body::gravitational(idx, pos, vel, b.mass, b.radius, b.softening)
        };
        // the softening family, applied after the capability branch so neither arm can miss it.
        // an accreting body and a bare gravitating one both carry a softening length, and both
        // need to say which field that length describes.
        let body = body.with_softening_kind(if b.softening_kind > 0.5 {
            symbi_ib::SofteningKind::Compact
        } else {
            symbi_ib::SofteningKind::Plummer
        });
        // the magnetic coupling, applied on top of the surface stack: the force-selective slip when
        // the wire carries the `magnetic.slip` group, the Ohmic sink for a positive resistivity, and
        // transparency otherwise (a no-op on B for a hydro body). `validate_magnetic_slip_bodies`
        // has already refused a body declaring both.
        let body = match (b.magnetic_slip, b.magnetic_resistivity) {
            (Some(slip), _) => body.with_magnetic(symbi_ib::MagneticSpec::Slip {
                diffusivity_ratio: slip.diffusivity_ratio,
                shell_width: slip.shell_width,
                slip_length_ratio: slip.slip_length_ratio,
                field_regularization: slip.field_regularization,
                placement: slip.placement,
            }),
            (None, Some(eta)) if eta > 0.0 => {
                body.with_magnetic(symbi_ib::MagneticSpec::Resistive { eta })
            }
            _ => body,
        };
        coll = coll.add(body.with_two_way_coupling(b.two_way_coupling));
    }
    // attach the prescribed binary orbit so `apply_body_deltas` advances the two components on their
    // Keplerian orbit each step (the components were built at the Keplerian ICs in parse_binary_components).
    if let Some(bin) = binary {
        // `as_binary` flips the orbital-advance capability `advance_binary` gates on; `with_binary_params`
        // supplies the orbit. both are required (the flag and the params are separate).
        coll = coll
            .as_binary()
            .with_binary_params(symbi_ib::BinaryParams::new(
                bin.total_mass,
                bin.semi_major,
                bin.eccentricity,
                bin.mass_ratio,
            ));
    }
    coll
}

/// build the immersed bodies and, for a cartesian kerr-schild excision run, auto-append the GR
/// excision horizon as a first-class diagnostic body. the horizon is not a user-placed body — it is
/// the excision surface itself — so it is synthesized from the spacetime config: the shell-flux
/// accretion ledger is measured through a coordinate sphere at `diagnostic_radius = 1.5 r_+ = 3 M`
/// (outside the horizon, where the flux is well-posed and, with the covariant energy, radius-
/// invariant at steady state). cartesian GR always uses the kerr-schild chart (schwarzschild is
/// spherical), so `spacetime != minkowski && coord == cartesian && r_exc > 0` selects it.
/// append the bonded-assembly fragments beyond the source prefix and build
/// the pair-physics carrier. every fragment is a sealed rigid sphere (the
/// drain-off porous wall); bond indices shift by the source-body count. the
/// caller pushes the returned physics through `attach_fragment_physics` and
/// keeps `shapes` parallel to the collection (fragments are analytic spheres).
fn append_fragments<const D: usize>(
    coll: BodyCollection<f64, D>,
    shapes: &mut Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>>,
    asm: &BondedAssemblyParams,
) -> (BodyCollection<f64, D>, symbi_ib::FragmentPhysics) {
    let base = coll.len();
    let mut coll = coll;
    for k in 0..asm.positions.len() {
        let pos = Tensor::new(std::array::from_fn(|ax| {
            asm.positions[k].get(ax).copied().unwrap_or(0.0)
        }));
        let vel = Tensor::new(std::array::from_fn(|ax| {
            asm.velocities[k].get(ax).copied().unwrap_or(0.0)
        }));
        let body = Body::rigid_sphere(
            base + k,
            pos,
            vel,
            asm.masses[k],
            asm.radii[k],
            asm.inertias[k],
            true,
        )
        .with_surface(symbi_ib::SurfaceSpec::Porous {
            porosity: 0.0,
            k_eta_n: asm.k_eta_n,
            k_eta_t: asm.k_eta_t,
        })
        .with_two_way_coupling(asm.mobile[k]);
        coll = coll.add_fragment(body);
        shapes.push(None);
    }
    let bonds = asm
        .bonds
        .iter()
        .map(|&(i, j)| {
            symbi_ib::Bond::form(
                base + i,
                base + j,
                coll.get(base + i),
                coll.get(base + j),
                asm.bond_material,
            )
        })
        .collect();
    let physics = symbi_ib::FragmentPhysics {
        bonds,
        contacts: asm.contact.map(symbi_ib::Contacts::new),
        gravity: asm.gravity,
    };
    (coll, physics)
}

/// whether a run carries the horizon body, whose ledger books the accretion through the shell
/// around the excised core: an excised curved spacetime on the cartesian chart
fn builds_horizon_body(cfg: &Config) -> bool {
    cfg.spacetime != "minkowski" && cfg.coord_system == "cartesian" && cfg.excision_radius > 0.0
}

/// the radius of the horizon ledger's diagnostic shell: 1.5 r_+ of the schwarzschild hole, outside
/// the horizon and the excised core
fn horizon_diagnostic_radius(cfg: &Config) -> f64 {
    3.0 * cfg.schwarzschild_mass
}

/// whether the run books a horizon ledger: the horizon body rides with the run's immersed bodies,
/// so an excised run carrying none excises its core and books no ledger
fn carries_horizon_ledger(cfg: &Config) -> bool {
    builds_horizon_body(cfg) && (!cfg.bodies.is_empty() || cfg.bonded_assembly.is_some())
}

/// the horizon ledger is the root's (single owning level): the root books the flux through the
/// diagnostic shell once per accepted root step, as an unrefined grid does, and every refinement
/// region keeps clear of the shell and the band of root cells whose state the shell faces read. a
/// patch over that band would replace the root's evolution there with restricted and refluxed fine
/// data, so the ledger would book a flux the root never applied. the shell stays where the horizon
/// sets it; a patch that needs to cross it is refused, and a composite ledger booking each level's
/// share of the shell is the alternative this restriction defers.
///
/// the band reaches `(reach + 1) dx + diagonal / 2` either side of the oblate kerr-schild shell
/// `r_ks = r_diag`: a shell face reads root cells up to `reach` along its normal from a cell whose
/// centroid lies within half a root-cell diagonal of the shell, and a covered cell beside one of
/// those cells would have the reflux correct it, so a box farther than that from the shell covers
/// no cell the shell faces read. a region box meets the band when it reaches inside the outer
/// spheroid and does not lie wholly inside the inner one; the spheroid's quadratic form is
/// separable, so its least and greatest values over a box are exact
fn check_horizon_shell_clearance(cfg: &Config) -> Result<(), String> {
    if !(cfg.refinement_enabled && carries_horizon_ledger(cfg)) {
        return Ok(());
    }
    let reach = match cfg.reconstruction_name.as_str() {
        "pcm" => 1.0,
        "ppm" => 3.0,
        _ => 2.0,
    };
    horizon_shell_clearance(&cfg.refinement_regions, &cfg.dx[..cfg.dims], reach, horizon_diagnostic_radius(cfg), cfg.kerr_spin)
}

/// the band test of `check_horizon_shell_clearance` over the regions' boxes on a root of spacing
/// `dx` (one entry per axis), a stencil of `reach` cells, the shell `r_diag` and the spin
fn horizon_shell_clearance(regions: &[Vec<f64>], dx: &[f64], reach: f64, r_diag: f64, spin: f64) -> Result<(), String> {
    let dims = dx.len();
    let diagonal = dx.iter().map(|d| d * d).sum::<f64>().sqrt();
    let widest = dx.iter().copied().fold(0.0, f64::max);
    let width = (reach + 1.0) * widest + 0.5 * diagonal;
    let (r_out, r_in) = (r_diag + width, r_diag - width);
    let a2 = spin * spin;
    // the spheroid r_ks = r: (x^2 + y^2) / (r^2 + a^2) + z^2 / r^2 = 1; the polar axis is z in 3d
    let weights = |r: f64| -> Vec<f64> {
        (0..dims).map(|ax| if dims == 3 && ax == 2 { 1.0 / (r * r) } else { 1.0 / (r * r + a2) }).collect()
    };
    let least_square = |lo: f64, hi: f64| if lo > 0.0 { lo * lo } else if hi < 0.0 { hi * hi } else { 0.0 };
    for region in regions.iter().filter(|r| r.len() >= 2 * dims) {
        let w_out = weights(r_out);
        let least: f64 = (0..dims).map(|ax| w_out[ax] * least_square(region[2 * ax], region[2 * ax + 1])).sum();
        let inside_inner = r_in > 0.0 && {
            let w_in = weights(r_in);
            let greatest: f64 = (0..dims)
                .map(|ax| w_in[ax] * region[2 * ax].powi(2).max(region[2 * ax + 1].powi(2)))
                .sum();
            greatest < 1.0
        };
        if least <= 1.0 && !inside_inner {
            return Err(format!(
                "refinement region {region:?} meets the horizon ledger's diagnostic shell (r_ks = \
                 {r_diag:.3}) or its band of {width:.3} either side, the root cells the shell faces \
                 read; the root books the horizon ledger and a patch there would replace the flux it \
                 applies. keep the refinement region outside r_ks = {r_out:.3} or inside r_ks = \
                 {:.3}",
                r_in.max(0.0)
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod horizon_shell_clearance_tests {
    use super::horizon_shell_clearance;

    /// a plm root of spacing 0.1 around a unit-mass hole, the shell at r = 3: the band reaches
    /// 3 cells and half a diagonal, 0.3 + 0.087 = 0.387, either side
    fn check(region: [f64; 6], spin: f64) -> Result<(), String> {
        horizon_shell_clearance(&[region.to_vec()], &[0.1; 3], 2.0, 3.0, spin)
    }

    #[test]
    fn a_region_across_the_shell_is_refused_by_name() {
        let err = check([2.5, 4.0, -0.2, 0.2, -0.2, 0.2], 0.0).expect_err("admitted");
        assert!(err.contains("meets the horizon ledger's diagnostic shell"), "{err}");
    }

    #[test]
    fn a_region_inside_the_band_is_refused_and_one_clear_of_it_admitted() {
        assert!(check([3.3, 4.5, -0.1, 0.1, -0.1, 0.1], 0.0).is_err());
        assert!(check([3.45, 4.5, -0.1, 0.1, -0.1, 0.1], 0.0).is_ok());
    }

    #[test]
    fn a_region_inside_the_inner_spheroid_is_admitted() {
        // wholly inside r_ks = 3 - 0.387 = 2.61, and refused when it reaches into the band
        assert!(check([1.5, 2.0, -0.3, 0.3, -0.3, 0.3], 0.0).is_ok());
        assert!(check([1.5, 2.7, -0.3, 0.3, -0.3, 0.3], 0.0).is_err());
    }

    #[test]
    fn the_shell_is_the_oblate_spheroid_of_the_spin() {
        // a = 0.9 widens the equator to sqrt(r^2 + a^2): a box clear of the a = 0 band at x = 3.45
        // meets the spinning shell, and along the spin axis the same box stays clear
        assert!(check([3.45, 4.5, -0.1, 0.1, -0.1, 0.1], 0.9).is_err());
        assert!(check([-0.1, 0.1, -0.1, 0.1, 3.45, 4.5], 0.9).is_ok());
    }
}

/// the shapes parallel to `build_bodies_and_horizon`: each user body's shape, then the horizon's
/// analytic sphere when the run carries it
fn build_body_shapes_and_horizon(cfg: &Config) -> Vec<Option<symbi_ib::sdf::SdfExpr<f64, 3>>> {
    let mut shapes = build_body_shapes(&cfg.bodies);
    if builds_horizon_body(cfg) {
        shapes.push(None);
    }
    shapes
}

fn build_bodies_and_horizon<const D: usize>(cfg: &Config) -> BodyCollection<f64, D> {
    let mut coll = build_bodies::<D>(&cfg.bodies, cfg.binary.as_ref());
    if builds_horizon_body(cfg) {
        let diagnostic_radius = horizon_diagnostic_radius(cfg);
        let idx = coll.len();
        coll = coll.add(symbi_ib::Body::horizon(
            idx,
            cfg.excision_radius,
            diagnostic_radius,
        ));
    }
    coll
}

/// append one diagnostics row per body to a whitespace-separated table (with a
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
    // the row is 3-component for every grid dimension (unused axes exactly
    // zero): one schema serves 1d/2d/3d runs, and a 3d run's z-components are
    // never dropped. the header line is the schema declaration — readers parse
    // column names from it, so old 2d-shaped files stay loadable.
    if fresh {
        writeln!(
            f,
            "# time body x y z vx vy vz fx fy fz torque_x torque_y torque_z \
             mass accreted_mass accretion_rate"
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
            "{time:.8e} {bb} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} {:.8e} \
             {:.8e} {:.8e} {:.8e} {:.8e} {accreted:.8e} {rate:.8e}",
            comp(&b.position, 0),
            comp(&b.position, 1),
            comp(&b.position, 2),
            comp(&b.velocity, 0),
            comp(&b.velocity, 1),
            comp(&b.velocity, 2),
            comp(&b.force, 0),
            comp(&b.force, 1),
            comp(&b.force, 2),
            b.torque[0],
            b.torque[1],
            b.torque[2],
            b.mass,
        )?;
    }
    Ok(())
}

/// the clock of the last sample recorded in a `diagnostics.dat` table, or `None` when
/// the file is absent or carries only its `#`-commented header. the schema puts the
/// time first on every row, so the final data line's first field is the tail sample's
/// timestamp; a sample spans one row per body, all sharing that clock.
fn last_diagnostics_time(path: &str) -> Option<f64> {
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .rev()
        .find(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
        .and_then(|l| l.split_whitespace().next())
        .and_then(|field| field.parse::<f64>().ok())
}

/// two clocks that agree to within the cadence comparator's roundoff tolerance.
fn same_diagnostic_time(a: f64, b: f64) -> bool {
    time_at_or_after(a, b) && time_at_or_after(b, a)
}

/// the diagnostics write plan for a run's start: whether to emit the seam sample at the
/// resume clock, and the first cadence boundary strictly beyond both the resume clock
/// and any tail already on disk. the existing `diagnostics.dat` is the authority for
/// whether the seam sample is already present, so the plan never assumes what a prior
/// run wrote. the resulting table holds strictly increasing sample times with at most
/// one sample at `resume_time`.
#[derive(Debug)]
struct DiagnosticsPlan {
    write_initial: bool,
    next: f64,
}

/// derive the write plan from the file's recorded tail and the resume clock. a tail
/// strictly beyond `resume_time` is a restarted branch grafted onto a later timeline
/// and is rejected — appending would interleave two histories, and trimming the file
/// is a policy the caller must state. a fresh file (or one whose tail predates the
/// resume clock) takes the seam sample; a tail already at the resume clock cedes it to
/// the run that wrote it.
fn plan_diagnostics(
    last_written: Option<f64>,
    resume_time: f64,
    interval: f64,
) -> Result<DiagnosticsPlan, String> {
    let write_initial = match last_written {
        None => true,
        Some(last) if same_diagnostic_time(last, resume_time) => false,
        Some(last) if time_at_or_after(resume_time, last) => true,
        Some(last) => {
            return Err(format!(
                "diagnostics.dat records t = {last:.8e} beyond the restart clock \
                 t = {resume_time:.8e}: appending would splice a restarted branch into \
                 a later timeline. trim the file to the restart time to proceed."
            ));
        }
    };
    // the first cadence multiple strictly beyond both the resume clock and the recorded
    // tail, so no historical interval is ever re-emitted. the direct estimate lands
    // within one interval of the boundary; the corrections settle it in O(1) steps.
    let floor = last_written.map_or(resume_time, |last| last.max(resume_time));
    let mut next = ((floor / interval).floor().max(0.0) + 1.0) * interval;
    while time_at_or_after(floor, next) {
        next += interval;
    }
    while next > interval && !time_at_or_after(floor, next - interval) {
        next -= interval;
    }
    Ok(DiagnosticsPlan {
        write_initial,
        next,
    })
}

/// advance a cadence cursor past every boundary at or before `time`, reporting whether
/// a sample is due. one call per callback collapses a step that spans many intervals to
/// a single sample stamped at `time`, keeping the table's sample times strictly
/// increasing. both the single-grid and decomposed drivers schedule through this, so
/// their cadence cannot diverge.
fn diagnostics_due(time: f64, next: &mut f64, interval: f64) -> bool {
    if !time_at_or_after(time, *next) {
        return false;
    }
    while time_at_or_after(time, *next) {
        *next += interval;
    }
    true
}

// the public boundary of the magnetic-slip sink: the parsed body wire becomes `MagneticSpec::Slip`,
// the contract refuses runs the operator is not built for, and a configured slip body allocates its
// workspace and drives the palindromic coupled step through the ordinary root advance.
#[cfg(test)]
mod magnetic_slip_boundary_tests {
    use super::*;
    use symbi::sim::refinement::Hierarchy;
    use symbi::sim::refinement::hierarchy::{slip_schedule_arm, slip_schedule_take};
    use symbi::sim::state::{Boundaries, BoundaryType, SimStateGeneric};
    use symbi::symbi_xpu::{CpuSpace, HostMemory};
    use symbi_geometry::Cartesian;
    use symbi_hydro::eos::IdealGas;
    use symbi_hydro::mhd_state::MhdPrim;
    use symbi_hydro::newtonian_mhd::NewtonianMhd;
    use symbi_hydro::state::Prim;

    const SLIP: MagneticSlipParams = MagneticSlipParams {
        diffusivity_ratio: 2.0,
        shell_width: 0.12,
        slip_length_ratio: 1.0,
        field_regularization: 0.1,
        placement: 0.0,
    };

    // a gravitating accretor with a plain drain at the box center: the surface stack a slip
    // coupling closes on.
    fn draining_sink(slip: Option<MagneticSlipParams>, resistivity: Option<f64>) -> BodyParams {
        BodyParams {
            capability: 3, // gravitational | accretion
            mass: 1.0,
            radius: 0.22,
            position: vec![0.5, 0.5, 0.5],
            velocity: vec![0.0, 0.0, 0.0],
            softening: 0.05,
            softening_kind: 0.0,
            accretion_radius: 0.22,
            sink_rate: 1.0,
            porosity: None,
            k_eta_n: 0.0,
            k_eta_t: 0.0,
            torque_free_xi: None,
            inertia: 0.0,
            no_slip: false,
            shape_json: None,
            omega: 0.0,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0; 3],
            two_way_coupling: false,
            magnetic_resistivity: resistivity,
            magnetic_slip: slip,
        }
    }

    #[test]
    fn build_bodies_carries_exactly_one_magnetic_coupling() {
        let slip = build_bodies::<3>(&[draining_sink(Some(SLIP), None)], None);
        assert_eq!(
            slip.get(0).spec.magnetic,
            symbi_ib::MagneticSpec::Slip {
                diffusivity_ratio: 2.0,
                shell_width: 0.12,
                slip_length_ratio: 1.0,
                field_regularization: 0.1,
                placement: 0.0,
            }
        );
        let resistive = build_bodies::<3>(&[draining_sink(None, Some(0.1))], None);
        assert_eq!(resistive.get(0).spec.magnetic, symbi_ib::MagneticSpec::Resistive { eta: 0.1 });
        let transparent = build_bodies::<3>(&[draining_sink(None, None)], None);
        assert_eq!(transparent.get(0).spec.magnetic, symbi_ib::MagneticSpec::None);
    }

    #[test]
    fn prescribed_binary_build_keeps_each_components_magnetic_coupling() {
        let params = vec![
            draining_sink(Some(SLIP), None),
            draining_sink(None, Some(0.25)),
        ];
        let binary = BinaryCfg {
            total_mass: 2.0,
            semi_major: 1.0,
            eccentricity: 0.0,
            mass_ratio: 1.0,
        };
        let bodies = build_bodies::<2>(&params, Some(&binary));
        assert!(bodies.is_binary());
        assert!(matches!(
            bodies.get(0).spec.magnetic,
            symbi_ib::MagneticSpec::Slip { .. }
        ));
        assert_eq!(
            bodies.get(1).spec.magnetic,
            symbi_ib::MagneticSpec::Resistive { eta: 0.25 }
        );
    }

    #[test]
    fn a_magnetized_drain_is_admitted_on_cartesian_grids_alone() {
        let sink = draining_sink(None, None);
        validate_magnetized_drain_bodies(std::slice::from_ref(&sink), "nmhd", 3, "cartesian")
            .expect("a 3D cartesian magnetized drain");
        validate_magnetized_drain_bodies(std::slice::from_ref(&sink), "imhd", 2, "cartesian")
            .expect("a 2.5D cartesian isothermal magnetized drain");
        validate_magnetized_drain_bodies(std::slice::from_ref(&sink), "newtonian", 2, "spherical")
            .expect("a hydro drain runs on every chart");
        let err = validate_magnetized_drain_bodies(std::slice::from_ref(&sink), "nmhd", 2, "cylindrical")
            .expect_err("a magnetized drain on the cylindrical section is refused");
        assert!(err.contains("local Alfven rate") && err.contains("2D cylindrical"), "{err}");
        let err = validate_magnetized_drain_bodies(std::slice::from_ref(&sink), "rmhd", 3, "spherical")
            .expect_err("a magnetized drain on a spherical grid is refused");
        assert!(err.contains("3D spherical"), "{err}");
    }

    #[test]
    fn the_boundary_admits_the_slip_only_where_the_operator_is_built() {
        let check = |bodies: &[BodyParams], regime: &str, dims: usize, coords: &str, iso: bool| {
            validate_magnetic_slip_bodies(bodies, regime, dims, coords, iso)
        };
        check(&[draining_sink(Some(SLIP), None)], "nmhd", 3, "cartesian", false)
            .expect("an adiabatic 3D cartesian draining sink carries the slip");
        check(&[draining_sink(Some(SLIP), None)], "nmhd", 2, "cartesian", false)
            .expect("an adiabatic 2.5D cartesian draining sink carries the slip");
        // the contract binds slip bodies only; other couplings run wherever they already run.
        check(&[draining_sink(None, Some(0.1))], "imhd", 2, "spherical", true)
            .expect("a resistive sink is outside this contract");
        let open_porous = {
            let mut b = draining_sink(Some(SLIP), None);
            b.porosity = Some(0.5);
            b
        };
        check(&[open_porous], "nmhd", 3, "cartesian", false).expect("an open porous surface drains");
        let torque_free = {
            let mut b = draining_sink(Some(SLIP), None);
            b.torque_free_xi = Some(0.5);
            b
        };
        check(&[torque_free], "nmhd", 3, "cartesian", false).expect("a torque-free drain drains");

        let sealed = {
            let mut b = draining_sink(Some(SLIP), None);
            b.porosity = Some(0.0);
            b
        };
        let no_sink = {
            let mut b = draining_sink(Some(SLIP), None);
            b.capability = 1;
            b
        };
        // the isothermal slip is built too: its heat leaves through the cooling reservoir.
        check(&[draining_sink(Some(SLIP), None)], "imhd", 3, "cartesian", false)
            .expect("an isothermal 3D cartesian draining sink carries the slip");
        let refused: [(&str, Vec<BodyParams>, &str, usize, &str, bool); 7] = [
            ("both couplings", vec![draining_sink(Some(SLIP), Some(0.1))], "nmhd", 3, "cartesian", false),
            ("relativistic regime", vec![draining_sink(Some(SLIP), None)], "rmhd", 3, "cartesian", false),
            ("locally isothermal", vec![draining_sink(Some(SLIP), None)], "nmhd", 3, "cartesian", true),
            ("one dimension", vec![draining_sink(Some(SLIP), None)], "nmhd", 1, "cartesian", false),
            ("spherical chart", vec![draining_sink(Some(SLIP), None)], "nmhd", 3, "spherical", false),
            ("sealed porous surface", vec![sealed], "nmhd", 3, "cartesian", false),
            ("no sink", vec![no_sink], "nmhd", 3, "cartesian", false),
        ];
        for (label, bodies, regime, dims, coords, iso) in refused {
            let err = check(&bodies, regime, dims, coords, iso)
                .expect_err(&format!("{label}: the boundary admitted the slip"));
            assert!(err.contains("magnetic slip") || err.contains("immersed body"), "{label}: {err}");
        }
    }

    // the same wire on a 2.5D grid: the workspace carries the cell B_z complex, and the root advance
    // runs the coupled composition.
    #[test]
    fn a_configured_slip_body_drives_the_coupled_step_on_a_2p5d_grid() {
        let n = 12usize;
        let dx = 1.0 / n as f64;
        let k = 2.0 * std::f64::consts::PI;
        let (a0, b0) = (0.3, 0.2);
        let sim = SimStateGeneric::<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>, CpuSpace, HostMemory, f64>::build(
            NewtonianMhd,
            IdealGas { gamma: 5.0 / 3.0 },
            Cartesian,
        )
        .cells([n, n])
        .origin([0.0, 0.0])
        .spacing([dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("sim construction")
        .set_initial(move |[x, y]| {
            let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
            let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new([
                    0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                    0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                    b0 * (k * x).cos() * (k * y).cos(),
                ]),
            )
        })
        .seed_faces(move |axis, [x, y]| match axis {
            0 => -a0 * (k * x).cos() * (k * y).sin(),
            _ => a0 * (k * x).sin() * (k * y).cos(),
        })
        .build();
        let mut body = draining_sink(Some(SLIP), None);
        body.position = vec![0.5, 0.5];
        body.velocity = vec![0.0, 0.0];
        let sim = sim.with_bodies(build_bodies::<2>(&[body], None));
        {
            let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
            let ws = mhd.magnetic_slip.as_ref().expect("the slip workspace was not allocated");
            assert!(ws.z.is_some(), "the 2.5D workspace carries no cell B_z complex");
        }
        let sub = symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet::<
            HostMemory,
            f64,
            2,
        >::new(5.0 / 3.0, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        slip_schedule_arm();
        hier.step_root_with_dt(2.0e-3);
        let trace = slip_schedule_take().expect("the schedule spy was armed");
        let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
        assert_eq!(ops, ["D", "M", "H", "M", "D"], "the 2.5D root advance did not run the coupled step");
    }

    // the wire a config produces, built into a 3D adiabatic MHD state: the slip workspace and the
    // quadrature scratch are allocated when the bodies attach, and one root advance runs the
    // palindromic D(dt/2) M(dt/2) H(dt) M(dt/2) D(dt/2) composition.
    #[test]
    fn a_configured_slip_body_allocates_its_workspace_and_drives_the_coupled_step() {
        let n = 12usize;
        let dx = 1.0 / n as f64;
        let k = 2.0 * std::f64::consts::PI;
        let a0 = 0.3;
        let sim = SimStateGeneric::<
            NewtonianMhd,
            3,
            3,
            Cartesian,
            IdealGas<f64>,
            CpuSpace,
            HostMemory,
            f64,
        >::build(NewtonianMhd, IdealGas { gamma: 5.0 / 3.0 }, Cartesian)
        .cells([n, n, n])
        .origin([0.0, 0.0, 0.0])
        .spacing([dx, dx, dx])
        .boundaries(Boundaries::uniform(BoundaryType::Periodic))
        .cfl(0.3)
        .allocate()
        .expect("sim construction")
        // the cell field is the exact average of its bounding faces, so the seeding deposits the
        // magnetic energy the first primitive recovery subtracts.
        .set_initial(move |[x, y, _z]| {
            let bx = |xf: f64| -a0 * (k * xf).cos() * (k * y).sin();
            let by = |yf: f64| a0 * (k * x).sin() * (k * yf).cos();
            MhdPrim::new(
                Prim::adiabatic(Density(1.0), Tensor::new([0.0, 0.0, 0.0]), Pressure(1.0)),
                Tensor::new([
                    0.5 * (bx(x - 0.5 * dx) + bx(x + 0.5 * dx)),
                    0.5 * (by(y - 0.5 * dx) + by(y + 0.5 * dx)),
                    0.0,
                ]),
            )
        })
        .seed_faces(move |axis, [x, y, _z]| match axis {
            0 => -a0 * (k * x).cos() * (k * y).sin(),
            1 => a0 * (k * x).sin() * (k * y).cos(),
            _ => 0.0,
        })
        .build();
        let sim = sim.with_bodies(build_bodies::<3>(&[draining_sink(Some(SLIP), None)], None));
        {
            let mhd = sim.fields.mhd.as_ref().expect("mhd fields");
            assert!(mhd.magnetic_slip.is_some(), "the slip midpoint workspace was not allocated");
            assert!(mhd.slip_quadrature.is_some(), "the slip quadrature scratch was not allocated");
        }
        let sub = symbi::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet3D::<
            HostMemory,
            f64,
        >::new(5.0 / 3.0, 0.3, 1.0, &sim.geom.allocated);
        let mut hier = Hierarchy::single(sim, sub);
        hier.prime();
        slip_schedule_arm();
        hier.step_root_with_dt(2.0e-3);
        let trace = slip_schedule_take().expect("the schedule spy was armed");
        let ops: Vec<&str> = trace.iter().map(|(op, _)| *op).collect();
        assert_eq!(ops, ["D", "M", "H", "M", "D"], "the root advance did not run the coupled step");
    }
}

#[cfg(test)]
mod diagnostics_tests {
    use super::*;

    fn porous_body(x: f64) -> BodyParams {
        BodyParams {
            capability: 2,
            mass: 1.0,
            radius: 0.0,
            position: vec![x, 0.0],
            velocity: vec![0.0, 0.0],
            softening: 0.05,
            softening_kind: 0.0,
            accretion_radius: 0.2,
            sink_rate: 1.0,
            porosity: Some(0.5),
            k_eta_n: 1.0,
            k_eta_t: 0.0,
            torque_free_xi: None,
            inertia: 0.0,
            no_slip: false,
            shape_json: None,
            omega: 0.0,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0; 3],
            two_way_coupling: false,
            magnetic_resistivity: None,
            magnetic_slip: None,
        }
    }

    #[test]
    fn porous_masks_reject_order_dependent_overlap() {
        let error = validate_porous_body_overlaps(&[porous_body(0.0), porous_body(0.3)])
            .expect_err("overlapping porous masks were accepted");
        assert!(error.contains("order-dependent"), "{error}");
        validate_porous_body_overlaps(&[porous_body(0.0), porous_body(0.5)])
            .expect("separated porous masks were rejected");
    }

    // the diagnostics row carries all three components for every grid
    // dimension — a 3d run's z position/velocity/force and the full torque
    // vector survive, and the header names every column (readers parse it).
    #[test]
    fn diagnostics_row_is_three_component_in_3d() {
        let mut b = symbi_ib::Body::<f64, 3>::black_hole(
            0,
            Tensor::new([1.0, 2.0, 3.0]),
            Tensor::new([0.1, 0.2, 0.3]),
            1.0,
            0.1,
            0.05,
            0.5,
            0.0,
            0.2,
        );
        b.force = Tensor::new([4.0, 5.0, 6.0]);
        b.torque = Tensor::new([7.0, 8.0, 9.0]);
        let bodies = BodyCollection::new().add(b);

        // the process id keeps concurrent test binaries off each other's file. a fixed
        // path under the system temp directory is shared by every run on the machine,
        // and a second one appending while this one reads yields a row that parses as
        // text rather than as numbers.
        let dir = std::env::temp_dir().join(format!("symbi_diag_dat_3d_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diagnostics.dat");
        let _ = std::fs::remove_file(&path);
        append_diagnostics(path.to_str().unwrap(), 0.5, &bodies).unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines
            .next()
            .unwrap()
            .trim_start_matches('#')
            .split_whitespace()
            .collect();
        let row: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(header.len(), row.len(), "header names every column");
        let col = |name: &str| row[header.iter().position(|h| *h == name).unwrap()];
        assert_eq!(col("z"), 3.0);
        assert_eq!(col("vz"), 0.3);
        assert_eq!(col("fz"), 6.0);
        assert_eq!(col("torque_x"), 7.0);
        assert_eq!(col("torque_z"), 9.0);
        assert_eq!(col("mass"), 1.0);
    }

    // a 2d run pads the unused axis with exact zeros — one schema for every D.
    #[test]
    fn diagnostics_row_pads_2d_with_zeros() {
        let bodies = BodyCollection::new().add(symbi_ib::Body::<f64, 2>::gravitational(
            0,
            Tensor::new([1.0, 2.0]),
            Tensor::zeros(),
            1.0,
            0.1,
            0.05,
        ));
        // the process id keeps concurrent test binaries off each other's file. a fixed
        // path under the system temp directory is shared by every run on the machine,
        // and a second one appending while this one reads yields a row that parses as
        // text rather than as numbers.
        let dir = std::env::temp_dir().join(format!("symbi_diag_dat_2d_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diagnostics.dat");
        let _ = std::fs::remove_file(&path);
        append_diagnostics(path.to_str().unwrap(), 0.0, &bodies).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let header: Vec<&str> = text
            .lines()
            .next()
            .unwrap()
            .trim_start_matches('#')
            .split_whitespace()
            .collect();
        let row: Vec<f64> = text
            .lines()
            .nth(1)
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let col = |name: &str| row[header.iter().position(|h| *h == name).unwrap()];
        assert_eq!(col("z"), 0.0);
        assert_eq!(col("vz"), 0.0);
        assert_eq!(col("fz"), 0.0);
    }

    // the two-way-coupling flag reaches the built body: build_bodies must carry
    // it from config (feedback onto a non-accreting gravitating mass is opt-in
    // via this flag; a hardcoded coupling = false would drop it).
    #[test]
    fn build_bodies_carries_two_way_coupling() {
        let params = vec![BodyParams {
            capability: 2, // accretion
            mass: 1.0,
            radius: 0.1,
            position: vec![0.0, 0.0],
            velocity: vec![0.0, 0.0],
            softening: 0.05,
            softening_kind: 0.0,
            accretion_radius: 0.2,
            sink_rate: 0.5,
            porosity: None,
            k_eta_n: 0.0,
            k_eta_t: 0.0,
            torque_free_xi: None,
            inertia: 0.0,
            no_slip: true,
            shape_json: None,
            omega: 0.0,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0, 0.0, 0.0],
            two_way_coupling: true,
            magnetic_resistivity: None,
            magnetic_slip: None,
        }];
        let coll = build_bodies::<2>(&params, None);
        assert!(coll.get(0).two_way_coupling);
    }

    // =====================================================================
    // diagnostics restart seam: the file on disk is the authority for whether
    // the resume-clock sample already exists, and the cadence resumes strictly
    // beyond the recorded tail so no historical interval is re-emitted. the
    // single-grid and decomposed drivers schedule through the same helpers, so
    // these unit tests bind both.
    // =====================================================================

    // a fresh file (or a run at t=0) writes the seam and resumes one interval later.
    #[test]
    fn plan_writes_the_seam_on_a_fresh_file() {
        let dt = 0.25;
        let plan = plan_diagnostics(None, 0.0, dt).unwrap();
        assert!(plan.write_initial, "a fresh file must record the initial sample");
        assert!(
            (plan.next - dt).abs() < 1e-12,
            "a fresh run resumes the cadence at the first interval, got {}",
            plan.next
        );
    }

    // a restart with a fresh diagnostics file still records the resume-clock sample,
    // and the cadence resumes on the first multiple strictly beyond the resume clock.
    #[test]
    fn plan_writes_the_seam_when_the_file_is_fresh_on_restart() {
        let dt = 0.25;
        let resume = 7.1; // between the 28th and 29th boundary (28*dt = 7.0)
        let plan = plan_diagnostics(None, resume, dt).unwrap();
        assert!(plan.write_initial, "a fresh file records the seam even on restart");
        assert!(plan.next > resume, "cadence resumes past the resume clock");
        assert!(
            plan.next <= resume + dt + 1e-12,
            "cadence resumes on the very next boundary, not further, got {}",
            plan.next
        );
    }

    // a tail already at the resume clock is owned by the run that wrote it: the
    // resumed run cedes the seam and resumes strictly beyond it.
    #[test]
    fn plan_cedes_a_seam_already_present() {
        let dt = 0.5;
        let resume = 5.0; // exactly the 10th boundary
        let plan = plan_diagnostics(Some(resume), resume, dt).unwrap();
        assert!(!plan.write_initial, "the prior run owns the resume-clock sample");
        assert!(plan.next > resume, "cadence resumes strictly beyond the seam");
        assert!((plan.next - (resume + dt)).abs() < 1e-12, "got {}", plan.next);
    }

    // a tail behind the resume clock leaves the seam missing: the resumed run fills it.
    #[test]
    fn plan_fills_a_missing_seam() {
        let dt = 0.5;
        let resume = 5.0;
        let tail = 4.0; // last sample predates the checkpoint we resumed from
        let plan = plan_diagnostics(Some(tail), resume, dt).unwrap();
        assert!(plan.write_initial, "a tail behind the resume clock leaves the seam to fill");
        assert!(plan.next > resume, "cadence resumes past the resume clock, got {}", plan.next);
    }

    // a tail strictly beyond the resume clock is a restarted branch grafted onto a
    // later timeline; the plan refuses it rather than interleaving two histories.
    #[test]
    fn plan_rejects_a_tail_beyond_the_resume_clock() {
        let dt = 0.5;
        let resume = 5.0;
        let tail = 6.5; // output already past where this restart begins
        let err = plan_diagnostics(Some(tail), resume, dt)
            .expect_err("a tail beyond the restart clock must be rejected");
        assert!(err.contains("beyond the restart clock"), "{err}");
    }

    // a single step spanning many intervals reports one due sample and advances the
    // cursor past every crossed boundary — no burst of same-timestamp rows.
    #[test]
    fn a_large_step_yields_one_sample_not_a_burst() {
        let dt = 0.1;
        let mut next = dt;
        let jump = 5.05; // spans ~50 intervals in one step
        assert!(diagnostics_due(jump, &mut next, dt), "the crossed cadence is due");
        assert!(next > jump, "the cursor advances past the step, got {next}");
        assert!(
            !diagnostics_due(jump, &mut next, dt),
            "the same clock yields no second sample"
        );
    }

    // last_diagnostics_time round-trips through the real writer: absent -> None,
    // header-only -> None, and after writes -> the tail sample's clock.
    #[test]
    fn last_recorded_time_reads_the_file_tail() {
        let bodies = BodyCollection::new().add(symbi_ib::Body::<f64, 2>::gravitational(
            0,
            Tensor::new([1.0, 2.0]),
            Tensor::zeros(),
            1.0,
            0.1,
            0.05,
        ));
        let dir = std::env::temp_dir().join(format!("symbi_diag_tail_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diagnostics.dat");
        let ps = path.to_str().unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(last_diagnostics_time(ps), None, "an absent file has no tail");
        for t in [0.0_f64, 0.5, 1.25] {
            append_diagnostics(ps, t, &bodies).unwrap();
        }
        let tail = last_diagnostics_time(ps).expect("a written file has a tail");
        assert!((tail - 1.25).abs() < 1e-9, "tail clock is the last sample, got {tail}");
    }

    // end-to-end over the real file writer: an initial run emits a seam-and-cadence
    // series, a restart reads the file to decide the seam, and the concatenated
    // sample clocks are strictly increasing with exactly one sample per cadence point.
    // the two restart geometries are the seam landing on a boundary (ceded) and
    // between boundaries (filled).
    fn replay_across_restart(dt: f64, run1_samples: &[f64], resume: f64, run2_end: f64) -> Vec<f64> {
        let bodies = BodyCollection::new().add(symbi_ib::Body::<f64, 2>::gravitational(
            0,
            Tensor::zeros(),
            Tensor::zeros(),
            1.0,
            0.1,
            0.05,
        ));
        let dir = std::env::temp_dir().join(format!(
            "symbi_diag_restart_{}_{}",
            std::process::id(),
            (resume * 1e6) as u64
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("diagnostics.dat");
        let ps = path.to_str().unwrap();
        let _ = std::fs::remove_file(&path);

        // run 1's recorded series: the samples on disk when the run stopped, tail at
        // or behind the checkpoint the restart resumes from.
        for &t in run1_samples {
            append_diagnostics(ps, t, &bodies).unwrap();
        }

        // run 2: restart at `resume`, letting the file decide the seam, then step the
        // cadence forward to run2_end.
        let p2 = plan_diagnostics(last_diagnostics_time(ps), resume, dt).unwrap();
        if p2.write_initial {
            append_diagnostics(ps, resume, &bodies).unwrap();
        }
        let mut next = p2.next;
        let mut t = resume;
        while t < run2_end - 1e-12 {
            t = (t + dt).min(run2_end);
            if diagnostics_due(t, &mut next, dt) {
                append_diagnostics(ps, t, &bodies).unwrap();
            }
        }

        // the recorded sample clocks: one distinct value per cadence point (a sample
        // is one row per body; a single body here makes rows and samples coincide).
        let text = std::fs::read_to_string(&path).unwrap();
        text.lines()
            .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .map(|l| l.split_whitespace().next().unwrap().parse::<f64>().unwrap())
            .collect()
    }

    fn assert_strictly_increasing_single_seam(times: &[f64], seam: f64) {
        for w in times.windows(2) {
            assert!(
                w[1] > w[0] + 1e-12,
                "sample clocks must strictly increase across the restart, saw {} then {}",
                w[0],
                w[1]
            );
        }
        let at_seam = times.iter().filter(|t| (**t - seam).abs() < 1e-9).count();
        assert!(at_seam <= 1, "at most one sample at the restart clock, found {at_seam}");
    }

    // the run ends between boundaries, so the checkpoint we resume from predates the
    // last-written sample: the resumed run fills the seam once.
    #[test]
    fn restart_between_boundaries_fills_the_seam_once() {
        let dt = 0.5;
        // run1's tail is the 3.0 boundary; the checkpoint sits between boundaries at 3.2,
        // so the resume-clock sample is missing and the resumed run fills it.
        let run1 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let resume = 3.2;
        let times = replay_across_restart(dt, &run1, resume, 6.0);
        assert_strictly_increasing_single_seam(&times, resume);
        assert!(
            times.iter().any(|t| (*t - resume).abs() < 1e-9),
            "the missing seam at {resume} was filled"
        );
    }

    // the run ends exactly on a boundary, so the resume-clock sample is already on
    // disk: the resumed run cedes it, and no duplicate appears at the seam.
    #[test]
    fn restart_on_a_boundary_does_not_duplicate_the_seam() {
        let dt = 0.5;
        // run1 wrote a sample at exactly the checkpoint clock; the resumed run cedes it.
        let run1 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let resume = 3.0;
        let times = replay_across_restart(dt, &run1, resume, 6.0);
        assert_strictly_increasing_single_seam(&times, resume);
        let at_seam = times.iter().filter(|t| (*t - resume).abs() < 1e-9).count();
        assert_eq!(at_seam, 1, "exactly one sample at the seam, neither dropped nor doubled");
    }

    // the softening family must survive `build_bodies` on both capability arms. an accreting body
    // and a bare gravitating one are constructed by different branches, and each carries a
    // softening length that means a different field: plummer is below newtonian at every radius
    // (0.354 of it at r = h), compact is exactly newtonian outside h. a body that lost the
    // selector would silently run plummer, weakening gravity across the whole domain rather than
    // only within h -- a systematic bias in any measurement that fits a power law in radius.
    #[test]
    fn build_bodies_carries_the_softening_family_on_every_capability() {
        let body = |capability: u64, softening_kind: f64| {
            let params = vec![BodyParams {
                capability,
                mass: 1.0,
                radius: 0.1,
                position: vec![0.0, 0.0],
                velocity: vec![0.0, 0.0],
                softening: 0.05,
                softening_kind,
                accretion_radius: 0.2,
                sink_rate: 0.5,
                porosity: None,
                k_eta_n: 0.0,
                k_eta_t: 0.0,
                torque_free_xi: None,
                inertia: 0.0,
                no_slip: true,
                shape_json: None,
                omega: 0.0,
                spin_axis: [0.0, 0.0, 1.0],
                inertia_principal: [0.0, 0.0, 0.0],
                two_way_coupling: false,
                magnetic_resistivity: None,
                magnetic_slip: None,
            }];
            *build_bodies::<2>(&params, None).get(0)
        };

        // capability 2 = accretion, 1 = gravitational: the two arms of the construction branch.
        for capability in [2, 1] {
            assert_eq!(
                body(capability, 1.0).softening_kind(),
                Some(symbi_ib::SofteningKind::Compact.as_scalar()),
                "capability {capability}: a compact-softened body was built as plummer, so its \
                 field is below newtonian everywhere instead of exact outside the softening length"
            );
            assert_eq!(
                body(capability, 0.0).softening_kind(),
                Some(symbi_ib::SofteningKind::Plummer.as_scalar()),
                "capability {capability}: the default softening family is not plummer"
            );
        }
    }

    // regression: a gravitational binary (`body_system.binary_config`) must build two orbiting
    // accretors with the prescribed Keplerian orbit attached -- the `body_system` payload was being
    // dropped (parse_bodies read only `immersed_bodies`), yielding zero bodies.
    #[test]
    fn binary_cfg_builds_a_prescribed_orbiting_pair() {
        let accretor = |x: f64| BodyParams {
            capability: 2, // accretion
            mass: 0.5,
            radius: 0.0,
            position: vec![x, 0.0],
            velocity: vec![0.0, 0.0],
            softening: 0.05,
            softening_kind: 0.0,
            accretion_radius: 0.2,
            sink_rate: 0.0,
            porosity: None,
            k_eta_n: 0.0,
            k_eta_t: 0.0,
            torque_free_xi: None,
            inertia: 0.0,
            no_slip: true,
            shape_json: None,
            omega: 0.0,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0, 0.0, 0.0],
            two_way_coupling: false,
            magnetic_resistivity: None,
            magnetic_slip: None,
        };
        let params = vec![accretor(0.5), accretor(-0.5)];
        let binary = BinaryCfg {
            total_mass: 1.0,
            semi_major: 1.0,
            eccentricity: 0.0,
            mass_ratio: 1.0,
        };
        let coll = build_bodies::<2>(&params, Some(&binary));
        assert_eq!(coll.len(), 2, "the binary must build two bodies");
        assert!(
            coll.is_binary(),
            "the collection must carry the prescribed binary orbit"
        );
        assert!(
            coll.get(0).has_gravity() && coll.get(1).has_gravity(),
            "both components gravitate"
        );
        // guard: without the binary cfg the same bodies are not a prescribed binary (the orbit is the
        // thing the body_system payload adds).
        assert!(!build_bodies::<2>(&params, None).is_binary());
    }

    // a rigid-capability body builds a non-accreting rigid sphere with the drain-off
    // porous wall (porosity 0), so it penalizes but removes no mass; free slip
    // (no_slip false) zeroes the tangential channel.
    #[test]
    fn build_bodies_rigid_is_drain_off_porous_wall() {
        let params = vec![BodyParams {
            capability: 1 << 4, // rigid
            mass: 0.0,
            radius: 0.3,
            position: vec![0.0, 0.0],
            velocity: vec![0.0, 0.0],
            softening: 0.0,
            softening_kind: 0.0,
            accretion_radius: 0.0,
            sink_rate: 0.0,
            porosity: None,
            k_eta_n: 2.0,
            k_eta_t: 0.0, // free slip: the parse zeroes the tangential dial
            torque_free_xi: None,
            inertia: 1.0,
            no_slip: false,
            shape_json: None,
            omega: 0.0,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0, 0.0, 0.0],
            two_way_coupling: false,
            magnetic_resistivity: None,
            magnetic_slip: None,
        }];
        let coll = build_bodies::<2>(&params, None);
        let body = coll.get(0);
        // shape stays None here: the rigid body defaults to its analytic sphere.
        assert!(build_body_shapes(&params)[0].is_none());
        // the wall masks to the body's physical radius (not an accretion radius).
        assert_eq!(body.accretion_radius(), None);
        assert_eq!(body.mask_radius(), Some(0.3));
        // the surface is the sealed (porosity 0) porous wall, tangential channel off.
        match body.spec.surface {
            symbi_ib::SurfaceSpec::Porous {
                porosity,
                k_eta_n,
                k_eta_t,
            } => {
                assert_eq!(porosity, 0.0);
                assert_eq!(k_eta_n, 2.0);
                assert_eq!(k_eta_t, 0.0);
            }
            other => panic!("rigid body must be a porous wall, got {other:?}"),
        }
    }

    // a shape wire on a rigid body parses to the corresponding SdfExpr (the runtime-JIT'd
    // arbitrary wall); the bodyless / shapeless entries stay None.
    #[test]
    fn build_body_shapes_parses_the_wire() {
        let rigid = |shape_json: Option<String>| BodyParams {
            capability: 1 << 4,
            mass: 0.0,
            radius: 0.3,
            position: vec![0.0, 0.0],
            velocity: vec![0.0, 0.0],
            softening: 0.0,
            softening_kind: 0.0,
            accretion_radius: 0.0,
            sink_rate: 0.0,
            porosity: None,
            k_eta_n: 1.0,
            k_eta_t: 1.0,
            torque_free_xi: None,
            inertia: 1.0,
            no_slip: true,
            shape_json,
            omega: 0.0,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0, 0.0, 0.0],
            two_way_coupling: false,
            magnetic_resistivity: None,
            magnetic_slip: None,
        };
        let params = vec![
            rigid(Some(
                r#"{"kind":"box","center":[0.0,0.0,0.0],"half_extents":[0.5,0.3,0.2]}"#.to_string(),
            )),
            rigid(None),
        ];
        let shapes = build_body_shapes(&params);
        assert!(matches!(
            shapes[0],
            Some(symbi_ib::sdf::SdfExpr::Cuboid { .. })
        ));
        assert!(shapes[1].is_none());
    }

    // the prescribed spin rate reaches the built body (Body::with_spin), so a shaped wall rotates.
    #[test]
    fn build_bodies_carries_spin() {
        let params = vec![BodyParams {
            capability: 1 << 4,
            mass: 0.0,
            radius: 0.3,
            position: vec![0.0, 0.0],
            velocity: vec![0.0, 0.0],
            softening: 0.0,
            softening_kind: 0.0,
            accretion_radius: 0.0,
            sink_rate: 0.0,
            porosity: None,
            k_eta_n: 1.0,
            k_eta_t: 1.0,
            torque_free_xi: None,
            inertia: 1.0,
            no_slip: true,
            shape_json: Some(
                r#"{"kind":"box","center":[0.0,0.0,0.0],"half_extents":[0.2,0.1,1.0]}"#.to_string(),
            ),
            omega: 3.5,
            spin_axis: [0.0, 0.0, 1.0],
            inertia_principal: [0.0, 0.0, 0.0],
            two_way_coupling: false,
            magnetic_resistivity: None,
            magnetic_slip: None,
        }];
        let coll = build_bodies::<2>(&params, None);
        // omega = rate * spin_axis = 3.5 * (0,0,1).
        let w = coll.get(0).omega;
        assert!(
            w[0] == 0.0 && w[1] == 0.0 && (w[2] - 3.5).abs() < 1e-12,
            "spin omega = {w:?}"
        );
    }
}

/// the coarse->fine prolongation order: degree above the reach of the evolution kernel
/// (plm -> ppm, ppm -> quartic), so refinement boundaries never drop a spatial order. pcm runs
/// the plm kernel at a zero slope and carries its reach, so it takes the plm prolongation.
fn prolong_order_for(reconstruction: &str) -> ProlongOrder {
    match reconstruction {
        "ppm" => ProlongOrder::Quartic,
        _ => ProlongOrder::Ppm,
    }
}

/// which halt condition stopped a march, with its numbers: a caught step panic carries its own
/// message (the FOFC freeze-streak halt names the offending cell and state), a halted root step
/// the retry guard's or the timestep floor's evidence, a non-finite rate is a poisoned cell, a
/// non-positive one a degenerate state, and a positive one a reduced retry that could no longer
/// advance time.
fn crash_cause(dt_cfl: f64, panic: Option<&str>, step: Option<&symbi_sim::step_transaction::StepHalt>) -> String {
    if let Some(msg) = panic {
        format!("step panic: {msg}")
    } else if let Some(step) = step {
        step.detail().to_string()
    } else if dt_cfl.is_nan() {
        "cfl dt is NaN — a non-finite wave speed (poisoned cell or boundary)".to_string()
    } else if dt_cfl <= 0.0 {
        format!("cfl dt is non-positive ({dt_cfl:.6e})")
    } else {
        format!("invalid cfl dt ({dt_cfl:.6e})")
    }
}

/// whether a march whose root level stands at `iteration` has reached the configured step cap.
/// `max_steps` bounds the root iteration itself, a resumed run's checkpointed iterations included,
/// so a run resumed at or past the cap takes no step; zero leaves the march unbounded.
fn step_cap_reached(cfg: &Config, iteration: u64) -> bool {
    cfg.max_steps > 0 && iteration >= cfg.max_steps
}

/// pin a state's clock to the run's start time and give it the user timestep clamp. the march
/// advances from start_time, the checkpoint's time on a restart, and every level, tile and worker
/// clamps its step alike.
fn pin_run_clock<const D: usize, const DOF: usize, Mem: MemorySpace>(
    store: &mut symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    cfg: &Config,
) {
    store.time = cfg.start_time;
    store.max_dt = cfg.max_dt;
}

/// seed a freshly built state for the run: its clock and clamp, and the mesh-motion state from
/// the scale-factor scalars or the traced a(t) law, sampled at the physical start time. every
/// single grid, level, tile and worker seeds through it, so all advance the identical law in
/// lockstep. a resumed state keeps the motion its checkpoint restored and only re-pins its clock.
fn seed_run_state<const D: usize, const DOF: usize, Mem: MemorySpace>(
    store: &mut symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
    cfg: &Config,
) -> Result<(), String> {
    pin_run_clock(store, cfg);
    store.motion = motion_state(cfg);
    if let Some(ref mj) = cfg.motion_json {
        let t0 = store.time;
        let law = symbi_source_compile::motion_law::MotionLaw::from_json(mj, t0, cfg.t_final)
            .map_err(|e| format!("mesh motion: {e}"))?;
        store.motion = symbi_geometry::MotionState::homologous(law.a_at(t0), law.adot_at(t0));
        store.motion_law = Some(law);
    }
    Ok(())
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

/// the tile partition for a decomposed run: the config's explicit per-axis cuts when
/// given, the balanced longest-axis heuristic otherwise. an explicit partition must
/// yield exactly `n_gpus` tiles, since each tile binds one device, and every partition
/// must carry the declared boundaries: an axis periodic on one face only stays uncut.
fn tile_partition<const D: usize>(n: [usize; D], cfg: &Config) -> Result<Partition<D>, String> {
    let partition = if cfg.decompose.is_empty() {
        Partition::auto(n, cfg.n_gpus)?
    } else {
        if cfg.decompose.len() != D {
            return Err(format!(
                "decompose carries {} axis cut lists for a {D}-dimensional run",
                cfg.decompose.len()
            ));
        }
        let cuts: [Vec<usize>; D] = std::array::from_fn(|ax| cfg.decompose[ax].clone());
        let partition = Partition::explicit(n, cuts)?;
        if partition.n_tiles() != cfg.n_gpus {
            return Err(format!(
                "decompose yields {} tiles for {} gpus; each tile binds one device",
                partition.n_tiles(),
                cfg.n_gpus
            ));
        }
        partition
    };
    // the declared boundaries the cuts must carry
    whole_grid::<D>(cfg).admits_cuts(partition.counts())?;
    // a tile's halo fills from the tiles beside it, so every tile spans at least the halo width
    let ng = ghost_width(cfg);
    let counts = partition.counts();
    for ax in 0..D {
        for k in 0..counts[ax] {
            let mut tc = [0; D];
            tc[ax] = k;
            let width = partition.tile_extents(tc)[ax].1;
            if width < ng {
                return Err(format!(
                    "tile {k} along axis {ax} spans {width} cell(s), fewer than the {ng}-cell halo the \
                     tiles beside it fill; place the cuts so every tile spans at least {ng} cells"
                ));
            }
        }
    }
    Ok(partition)
}

fn axis_maps<const D: usize>(cfg: &Config) -> Option<[symbi_geometry::AxisMap; D]> {
    use symbi_geometry::AxisMap;
    let spacings = [&cfg.x1_spacing, &cfg.x2_spacing, &cfg.x3_spacing];
    let ratios = [
        cfg.x1_spacing_ratio,
        cfg.x2_spacing_ratio,
        cfg.x3_spacing_ratio,
    ];
    if spacings[..D]
        .iter()
        .all(|spacing| spacing.eq_ignore_ascii_case("linear"))
    {
        return None;
    }
    Some(std::array::from_fn(|ax| {
        let spacing = spacings[ax];
        let start = cfg.x_lo[ax];
        let n = cfg.n_cells[ax] as f64;
        if spacing.eq_ignore_ascii_case("linear") {
            return AxisMap::Uniform {
                start,
                dx: cfg.dx[ax],
            };
        }
        if spacing.eq_ignore_ascii_case("log") {
            assert!(
                start > 0.0 && n > 0.0,
                "log spacing on x{} requires positive bounds",
                ax + 1
            );
            let end = start + cfg.dx[ax] * n;
            return AxisMap::Log {
                start,
                log_slope: (end / start).log10() / n,
            };
        }
        assert!(
            spacing.eq_ignore_ascii_case("geometric"),
            "unsupported x{} cell spacing '{}'",
            ax + 1,
            spacing
        );
        let ratio = ratios[ax];
        assert!(
            ratio.is_finite() && ratio > 0.0,
            "x{}_spacing_ratio must be positive and finite",
            ax + 1
        );
        let extent = cfg.dx[ax] * n;
        let width = if (ratio - 1.0).abs() < 1.0e-12 {
            cfg.dx[ax]
        } else {
            extent * (ratio - 1.0) / (ratio.powf(n) - 1.0)
        };
        AxisMap::Geometric {
            start,
            width,
            ratio,
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
        // the clock, the user dt clamp and the mesh motion, seeded on the coarse state before it
        // is wrapped; the motion is sampled at the physical start time
        seed_run_state(&mut sim, $cfg)?;
        if $cfg.refinement_enabled {
            let regions = refinement_regions_nd::<$d>(&$cfg.refinement_regions)?;
            let prolong = prolong_order_for(&$cfg.reconstruction_name);
            let mut h = Hierarchy::with_refinement(sim, $kernels, &regions, prolong, $make)
                .map_err(|e| format!("refinement build: {e:?}"))?;
            h.seed_fine_from_coarse().map_err(|e| format!("fine-level seed: {e:?}"))?;
            if $cfg.n_tracers > 0 {
                h.attach_mass_tracers($cfg.n_tracers);
                if let Some(order) = configured_ito_order($cfg)? {
                    for level in &mut h.levels {
                        convert_seeded_tracers(&mut level.state, order)?;
                    }
                }
            }
            h
        } else {
            Hierarchy::single(sim, $kernels)
        }
    }};
}

/// attach the config's immersed bodies (gravity / accretion sinks + shaped walls) plus any bonded
/// fragment assembly to a built sim, at their global positions. the single site both the
/// single-device and the decomposed (per-tile) hydro builds call, so a body / shape / fragment
/// feature is wired once -- a decomposed copy silently dropping the shapes (as it once did) is
/// exactly what a shared attach prevents. the caller owns the empty / refined guard (a refined run
/// attaches to the hierarchy's finest level instead).
/// the immersed bodies and their shaped walls, at their global positions: a magnetized regime
/// carries no bonded fragments (refused at admission), so its bodies are the sinks and walls alone
macro_rules! attach_bodies_and_shapes {
    ($sim:expr, $cfg:expr, $d:literal) => {{
        let cfg: &Config = $cfg;
        let mut sim = $sim.with_bodies(build_bodies_and_horizon::<$d>(cfg));
        sim.attach_body_shapes(build_body_shapes_and_horizon(cfg));
        sim
    }};
}

macro_rules! attach_bodies_and_fragments {
    ($sim:expr, $cfg:expr, $d:literal) => {{
        let cfg: &Config = $cfg;
        let mut coll = build_bodies_and_horizon::<$d>(cfg);
        let mut shapes = build_body_shapes_and_horizon(cfg);
        let physics = cfg.bonded_assembly.as_ref().map(|asm| {
            let (with_frags, physics) = append_fragments::<$d>(coll.clone(), &mut shapes, asm);
            coll = with_frags;
            physics
        });
        let mut sim = $sim.with_bodies(coll);
        sim.attach_body_shapes(shapes);
        if let Some(physics) = physics {
            sim.attach_fragment_physics(physics);
        }
        sim
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
        type Sim = SimDefaultGeneric<$regime_ty, $d, $dof, $geom_ty, EosSelect<f64>>;


        // gpus>1 -> the decomposed multi-gpu path (validated separately above by
        // validate_gpu_request); gpus<=1 -> the single-device path below, bit-identical.
        // the DOF-lifted (swirl) tile decomposition is wired: the decomposed build is
        // DOF-generic and the transport carries the out-of-plane momentum across cuts.
        // refinement / bodies with swirl are
        // not, matching the single-device swirl guards below.
        if cfg.n_gpus > 1 {
            if $dof != $d && cfg.refinement_enabled {
                return Err(
                    "DOF-lifted (swirl) runs do not yet support mesh refinement".to_string(),
                );
            }
            if $dof != $d && !cfg.bodies.is_empty() {
                return Err(
                    "DOF-lifted (swirl) runs do not yet support immersed bodies".to_string(),
                );
            }
            return build_and_run_hydro_decomposed!(
                $cfg, $prims, $regime, $regime_ty, $d, $dof, $geom, $geom_ty
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

        let sim = Sim::build($regime, host_eos(cfg), $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .ghosts(ghost_width(cfg))
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
                Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $dof]))
            })
            .build();

        // attach immersed bodies (gravity / accretion sinks + bonded fragments) when the config
        // declares any. a refined run attaches them to the hierarchy instead (finest level owns
        // the sinks), so skip the sim-level attach here and defer to `hier.with_bodies` after
        // `into_hierarchy` (fragments reject refinement upstream).
        let has_any_body = !cfg.bodies.is_empty() || cfg.bonded_assembly.is_some();
        let sim = if !has_any_body || cfg.refinement_enabled {
            sim
        } else {
            attach_bodies_and_fragments!(sim, cfg, $d)
        };
        // seed the passive scalar (dye): cons.chi = rho*chi and the primitive
        // concentration over the interior; the evolve-entry ghost fill covers
        // the halo before the first flux reads it.
        let sim = if cfg.chi_ic.is_empty() {
            sim
        } else {
            let sim = sim
                .with_passive_scalar()
                .map_err(|e| format!("passive-scalar allocation: {e:?}"))?;
            let interior = sim.geom.interior.clone();
            let lo: Vec<isize> = interior.spaces.iter().map(|s| s.lo).collect();
            let ns: Vec<usize> = interior.spaces.iter().map(|s| (s.hi - s.lo) as usize).collect();
            let n_total: usize = ns.iter().product();
            if cfg.chi_ic.len() != n_total {
                return Err(format!(
                    "passive_scalar yielded {} values for {} interior cells",
                    cfg.chi_ic.len(),
                    n_total
                ));
            }
            {
                let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
                let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
                for c in interior.iter() {
                    let mut lin = 0usize;
                    let mut stride = 1usize;
                    for ax in 0..$d {
                        lin += ((c[ax] - lo[ax]) as usize) * stride;
                        stride *= ns[ax];
                    }
                    let chi_v = cfg.chi_ic[lin];
                    let rho = *sim.fields.cons.den.view().at(c);
                    cons_chi.view_mut().set(c, rho * chi_v);
                    prim_chi.view_mut().set(c, chi_v);
                }
            }
            sim
        };
        let sim = {
            let mut sim = sim;
            if cfg.n_tracers != 0 {
                attach_configured_tracers(&mut sim, cfg)?;
            }
            // unconditional: the census list is empty on a run that registered none, so this
            // costs nothing there, and a run that did register one must carry it. attaching
            // only under a flag is how the registration reached preflight and stopped.
            attach_configured_censuses(&mut sim, cfg)?;
            sim
        };
        // the numerics every level carries: the base level with the excision, each fine level of
        // a refined run without it, so a fine set never runs a scheme other than the base's
        let settings = KernelSettings::of(cfg, <$regime_ty as Regime<f64, $dof>>::SPEC);
        let sub = gas_kernels(sim.substrate(), &settings, LevelRole::Root)?;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            gas_kernels(s.substrate(), &settings, LevelRole::Fine).expect("fine-level kernel set")
        });
        // a refined run attaches its immersed bodies to the hierarchy: the finest level owns the full
        // (accreting) bodies, coarser levels carry a gravity-only proxy (finest-owns-bodies, so the
        // sink applies once). the sink sphere must lie inside the finest level (asserted there).
        if !cfg.bodies.is_empty() && cfg.refinement_enabled {
            hier = hier.with_bodies(build_bodies_and_horizon::<$d>(cfg));
            hier.attach_body_shapes(build_body_shapes_and_horizon(cfg));
        }

        // the run's stationary target, when one is declared: its discrete imbalance is measured
        // once per level and added back at every stage, so the target becomes an exact fixed point
        // rather than a state that drifts at truncation order. this runs after the bodies attach
        // because the imbalance is read off a real stage, and a gravitational body's source is
        // half of the balance being measured.
        if let Some(ref target) = cfg.equilibrium_json {
            let declared = symbi_source_compile::EquilibriumConfig::from_json(target)
                .map_err(|e| format!("stationary target parse: {e}"))?;
            hier = hier
                .with_equilibrium_expression(&declared)
                .map_err(|e| format!("stationary target: {e:?}"))?;
            // a restart restores the checkpoint after this point, so a resumed run continues from
            // its own state and the seed applies only to a fresh start.
            if cfg.seed_from_equilibrium {
                hier.seed_equilibrium();
            }
        }

        // the declared perturbation, laid over the base state at every level's own cell
        // centers: the cell generator fills only the root, so fine levels otherwise hold nothing
        // the root cannot represent
        let has_energy = hier.levels[0].state.fields.cons.has_energy();
        if lay_perturbation!(hier, cfg, $dof, has_energy) {
            hier.sync_perturbed();
        }
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// the halo schedule of a decomposed run on its chart: a polar axis face whose azimuth is cut
/// hands its ghost band to the schedule's antipodal legs, so the pole tiles' axis faces become
/// cuts their own fill skips, and the seam names the azimuth slabs the legs rotate across
fn tiled_schedule<const D: usize, const DOF: usize, Mem: MemorySpace>(
    cfg: &Config,
    counts: [usize; D],
    stores: &mut [&mut symbi_sim::state::FieldStore<D, DOF, Mem, f64>],
) -> symbi::sim::decomp::Schedule<D> {
    let topology = chart_topology(cfg, &whole_grid::<D>(cfg), counts);
    // each azimuth tile's interior cell count, read off the tiles in the partition's order
    let seam = polar_seam::<D>(&cfg.coord_system, &boundaries_nd::<D>(&cfg.boundaries), &topology, counts, |az, i| {
        let mut tc = [0usize; D];
        tc[az] = i;
        let g = &stores[symbi::sim::decomp::flatten(tc, counts)].geom.interior;
        (g.spaces[az].hi - g.spaces[az].lo) as usize
    });
    if let Some(seam) = seam.as_ref() {
        for (flat, store) in stores.iter_mut().enumerate() {
            let tc = symbi::sim::decomp::unflatten(flat, counts);
            for (side, on_pole) in [(0usize, tc[seam.mirror] == 0), (1usize, tc[seam.mirror] + 1 == counts[seam.mirror])] {
                if seam.sides[side] && on_pole {
                    store.boundaries.0[seam.mirror][side] = BoundaryType::CoarseFine;
                }
            }
        }
    }
    symbi::sim::decomp::Schedule::derive_with_seam(counts, stores[0].geom.ng, &topology, seam)
}

/// a stationary target's discrete imbalance on every tile, as the single hierarchy reads it off
/// its level: the tiles already hold the target on their interiors; they are primed across the
/// cuts, each keeps the target's anchor state, their interface fluxes are evaluated, and one
/// forward-euler stage of the tiled pipeline at the smallest cfl step of any tile (the single
/// grid's step) advances them, so `R = (target - advanced) / dt` on each tile's cells. returns
/// each tile's anchor and residual; the caller restores the state it saved.
fn tiled_equilibrium_probe<const D: usize, const DOF: usize, Mem, K>(
    cfg: &Config,
    stores: &mut [&mut symbi_sim::state::FieldStore<D, DOF, Mem, f64>],
    kernels: &[&K],
    counts: [usize; D],
) -> Result<
    (
        Vec<symbi::sim::refinement::equilibrium::GasState<D, DOF, Mem>>,
        Vec<symbi_sim::state::ConsFieldsGeneric<D, DOF, Mem>>,
    ),
    String,
>
where
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    use symbi::sim::refinement::equilibrium::{imbalance_from_stage, save_gas_state};
    // the probe is no accepted evolution: it runs before the march opens its guard scope, so no
    // guard act is booked, and a stage books no body delta (feedback and the surface operations
    // follow the stages), both checked below; the restored state and the reset stage clock leave
    // the tiles at their build state
    if symbi_sim::guard_ledger::scope_is_open() {
        return Err("the stationary target's probe ran inside a guard scope, where its acts would be booked".to_string());
    }
    let schedule = tiled_schedule(cfg, counts, stores);
    let devices: Vec<i32> = (0..stores.len() as i32).collect();
    #[cfg(feature = "gpu")]
    let transport = symbi::sim::decomp::PeerCopy;
    #[cfg(not(feature = "gpu"))]
    let transport = symbi::sim::decomp::LocalCopy;
    let sh: Vec<&symbi_sim::state::FieldStore<D, DOF, Mem, f64>> = stores.iter().map(|s| &**s).collect();
    symbi::sim::decomp::prime_tiles(&sh, kernels, &schedule, &devices, &transport);
    let anchors = sh
        .iter()
        .map(|s| save_gas_state(*s))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("stationary target anchor: {e:?}"))?;
    for (store, kernel) in sh.iter().zip(kernels) {
        kernel.wave_speeds(*store);
        for dd in 0..D {
            kernel.flux(*store, dd);
        }
    }
    let probe_dt = sh.iter().zip(kernels).map(|(store, kernel)| kernel.cfl(*store)).fold(f64::INFINITY, f64::min);
    if !(probe_dt.is_finite() && probe_dt > 0.0) {
        return Err(format!(
            "the stationary target has no finite cfl step ({probe_dt:.3e}), so it cannot be advanced to read off its imbalance"
        ));
    }
    symbi::sim::decomp::probe_stage(&sh, kernels, &schedule, &devices, &transport, probe_dt)
        .map_err(|e| format!("the stationary target is not a state the scheme can carry: {e}"))?;
    if sh.iter().any(|store| store.immersed.as_ref().is_some_and(|im| !im.diagnostics.consolidate().is_empty())) {
        return Err("the stationary target's probe booked a body delta".to_string());
    }
    // the probe's step, reported as the single hierarchy reports its level's
    eprintln!("SIMBI equilibrium probe: level=0 dt_bits={:016x}", probe_dt.to_bits());
    for (k, store) in sh.iter().enumerate() {
        let err = symbi_sim::hydro_ops::scan_c2p_errors(*store);
        if err.is_err() {
            return Err(format!(
                "advancing the stationary target by one cfl step left an unrecoverable state on tile {k} ({err}), so its imbalance cannot be read off a linear response"
            ));
        }
    }
    let residuals = sh
        .iter()
        .zip(&anchors)
        .map(|(store, anchor)| imbalance_from_stage(anchor.cons(), &store.fields.cons, &store.geom.allocated, probe_dt))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("stationary target residual: {e:?}"))?;
    Ok((anchors, residuals))
}

/// the tiles' stationary-target residuals when the run declares a target: each tile's interior is
/// seeded from the declared expression at its cell centres (t = 0; the whole-grid origin and
/// index offset place a whole-grid cell bit for bit), the tiled probe reads the imbalance off,
/// the declared stationary target's expression for a `$d`-dimensional run (`None` when the run
/// declares none): parsed, its component count and dimension checked against the run, and a
/// passive scalar refused, as the single hierarchy refuses them
macro_rules! declared_equilibrium {
    ($cfg:expr, $d:literal, $has_energy:expr) => {{
        let cfg: &Config = $cfg;
        match cfg.equilibrium_json.as_ref() {
            None => None,
            Some(target) => {
                let declared = symbi_source_compile::EquilibriumConfig::from_json(target)
                    .map_err(|e| format!("stationary target parse: {e}"))?;
                let expected = 1 + $d + usize::from($has_energy);
                if declared.outputs.len() != expected || declared.dim != $d {
                    return Err(format!(
                        "the declared stationary target supplies {} primitive components on a {}-dimensional grid; this run needs {expected} on a {}-dimensional grid",
                        declared.outputs.len(),
                        declared.dim,
                        $d
                    ));
                }
                Some(declared.to_expression().map_err(|e| format!("the declared stationary target does not load: {e:?}"))?)
            }
        }
    }};
}

/// and the tiles' own state comes back; a fresh start seeded from the target takes the anchor.
macro_rules! tiled_equilibrium {
    ($tiles:expr, $cfg:expr, $counts:expr, $d:literal) => {{
        let cfg: &Config = $cfg;
        match cfg.equilibrium_json.as_ref() {
            None => None,
            Some(target) => {
                use symbi::sim::refinement::equilibrium::{overwrite, restore_gas_state, save_gas_state};
                let declared = symbi_source_compile::EquilibriumConfig::from_json(target)
                    .map_err(|e| format!("stationary target parse: {e}"))?;
                let has_energy = $tiles[0].0.fields.cons.has_energy();
                let expected = 1 + $d + usize::from(has_energy);
                if declared.outputs.len() != expected || declared.dim != $d {
                    return Err(format!(
                        "the declared stationary target supplies {} primitive components on a {}-dimensional grid; this run needs {expected} on a {}-dimensional grid",
                        declared.outputs.len(),
                        declared.dim,
                        $d
                    ));
                }
                let expression = declared.to_expression().map_err(|e| format!("the declared stationary target does not load: {e:?}"))?;
                let saved = $tiles
                    .iter()
                    .map(|(s, _)| save_gas_state(&**s))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("stationary target save: {e:?}"))?;
                for (s, _) in $tiles.iter() {
                    s.seed_cells(|x| {
                        let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                        symbi_hydro::state::PrimFromSlots::from_slots(&expression.eval(at(0), at(1), at(2), 0.0))
                    });
                }
                let (anchors, residuals) = {
                    let mut stores = Vec::with_capacity($tiles.len());
                    let mut kernels = Vec::with_capacity($tiles.len());
                    for (s, k) in $tiles.iter_mut() {
                        stores.push(&mut **s);
                        kernels.push(&*k);
                    }
                    tiled_equilibrium_probe(cfg, &mut stores, &kernels, $counts)?
                };
                for ((s, _), state) in $tiles.iter().zip(&saved) {
                    restore_gas_state(&**s, state);
                }
                // a restart restores the checkpoint after this point, so a resumed run continues from
                // its own state and the seed applies only to a fresh start
                if cfg.seed_from_equilibrium {
                    for ((s, _), anchor) in $tiles.iter().zip(&anchors) {
                        overwrite(anchor.cons(), &s.fields.cons, &s.geom.allocated);
                    }
                }
                Some(residuals)
            }
        }
    }};
}

/// the regime-agnostic decomposed run loop: evolve N pre-built tiles in
/// lockstep with the universal `PeerCopy` transport (real peer where a link exists, staged over
/// managed memory otherwise -- so the same code runs on one card with `--ngpus 2` and on a node
/// with `--ngpus 8`, no machine-specific branch), gathering into `global` for output through the
/// existing single-grid checkpoint writer. every regime's decomposed build feeds this one loop;
/// adding a regime is just a tile-build. the checkpoint cadence is the linear `checkpoint_interval`.
fn run_decomposed_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    cfg: &Config,
    mut tiles: Vec<(SimStateGeneric<R, D, DOF, M, E, S, Mem>, K)>,
    offsets: Vec<[isize; D]>,
    grid: symbi_sim::checkpoint::GlobalGrid<D>,
    counts: [usize; D],
    residuals: Option<Vec<symbi_sim::state::ConsFieldsGeneric<D, DOF, Mem>>>,
) -> Result<symbi_sim::run_diagnostics::RunDiagnostics, String>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    Cartesian: Metric<f64, D>,
{
    use symbi::sim::decomp::{enable_peer_mesh, evolve_scheduled_hooked};
    use symbi_sim::checkpoint::{
        LevelTiles, TileView, load_partitioned_level, write_partitioned_checkpoint,
    };

    // a restart reads each tile's own block of the checkpoint's global grid, whatever partition
    // wrote it; the exchange rebuilds the cut halos when the march primes.
    if let Some(path) = cfg.restart_path.as_deref() {
        for (flat, (tile, _)) in tiles.iter_mut().enumerate() {
            load_partitioned_level(tile, path, 0, offsets[flat], &grid)
                .map_err(|e| format!("restart of tile {flat} from '{path}': {e}"))?;
        }
    }
    // one checkpoint write from the tiles as they stand: every tile streams its owned block.
    let identity = symbi_sim::checkpoint::PhysicsIdentity::of(&tiles[0].0);
    let write_tiles = |states: &[&symbi_sim::state::FieldStore<D, DOF, Mem, f64>],
                       path: &str,
                       cp_index: u64|
     -> Result<f64, String> {
        let level = LevelTiles {
            tiles: states
                .iter()
                .zip(&offsets)
                .map(|(state, &offset)| TileView { state: *state, offset })
                .collect(),
            grid: grid.clone(),
            header: None,
        };
        let t_io = std::time::Instant::now();
        write_partitioned_checkpoint::<R, D, DOF, Mem>(&identity, &[level], path, &checkpoint_metadata(cfg, cp_index))
            .map_err(|e| checkpoint_write_error(path, e))?;
        Ok(t_io.elapsed().as_secs_f64())
    };

    let ntiles = tiles.len();
    let devices: Vec<i32> = (0..ntiles as i32).collect();
    let n_zones: u64 = tiles
        .iter()
        .map(|(s, _)| (0..D).map(|ax| s.geom.interior.spaces[ax].size() as u64).product::<u64>())
        .sum();
    let slabs: Vec<Vec<usize>> = (0..D)
        .map(|ax| {
            (0..counts[ax])
                .map(|i| {
                    let mut tc = [0usize; D];
                    tc[ax] = i;
                    let g = &tiles[symbi::sim::decomp::flatten(tc, counts)].0.geom.interior;
                    g.spaces[ax].size()
                })
                .collect()
        })
        .collect();
    let mut reporter = DecomposedReporter::new(cfg, n_zones, &counts, &slabs);
    // open peer links once (no-op for pairs that can't peer; those stage).
    enable_peer_mesh(&devices);
    reporter.milestone("peer links opened");

    // the universal transport: adaptive peer/staged. single-device builds compile the host arm
    // but never reach this fn (gpus>1 needs a gpu feature; validate_gpu_request enforces it).
    #[cfg(feature = "gpu")]
    let transport = symbi::sim::decomp::PeerCopy;
    #[cfg(not(feature = "gpu"))]
    let transport = symbi::sim::decomp::LocalCopy;

    // the checkpoint schedule the single grid walks
    let mut cadence = CheckpointCadence::of(cfg);
    let cp_width = checkpoint_time_width(cfg);

    // body diagnostics: a separate user-defined cadence (natural units), only when the run has
    // bodies + a positive interval -- mirrors the single-grid run. the body state is identical on
    // every tile (`step_bodies_decomposed` applies the cross-tile-summed feedback to all tiles), so
    // any tile's bodies are the global diagnostic. one `<dir>diagnostics.dat` for the whole run.
    let diag_path = if cfg.diagnostic_interval > 0.0 && !cfg.bodies.is_empty() {
        Some(format!("{}diagnostics.dat", cfg.data_dir))
    } else {
        None
    };
    let diag_interval = (cfg.diagnostic_interval * cfg.time_unit).max(f64::MIN_POSITIVE);
    // the existing diagnostics.dat is the authority for the restart seam: whether the
    // resume-clock sample is already on disk, and where the cadence resumes so no
    // historical interval is re-emitted.
    let diag_plan = diag_path.as_ref().map(|dp| {
        plan_diagnostics(last_diagnostics_time(dp), cfg.start_time, diag_interval)
            .unwrap_or_else(|detail| panic!("{detail}"))
    });
    let mut next_diag = diag_plan.as_ref().map_or(diag_interval, |p| p.next);

    // pin every tile's physical clock to the start time (nonzero on restart): the decomposed
    // loop advances these per step, and the checkpoint writer records the gather target's
    // clock — synced from tile 0 before every write below. every tile carries the user
    // timestep clamp the march applies.
    for ((s, _), offset) in tiles.iter_mut().zip(&offsets) {
        pin_run_clock(s, cfg);
        // the whole grid's interior in the tile's `index + offset` frame
        s.geom.whole_span = Some(grid.interior_span(&s.geom, *offset));
    }
    if cfg.n_tracers > 0 {
        reporter.milestone(&format!(
            "tracers: {} seeded on the whole grid at initialization, a whole-grid allocation that \
             lives for the seeding alone",
            cfg.n_tracers
        ));
    }
    // t=start initial condition. every device finishes its seeding before the staging reads.
    if cfg.checkpoint_index == 0 || cfg.start_time == 0.0 {
        // the builders seed the conserved state; the initial checkpoint publishes the primitives
        // the march starts from, recovered on every tile as the single grid's priming does
        for (k, (s, kern)) in tiles.iter().enumerate() {
            symbi::symbi_xpu::with_device(devices[k], || kern.c2p(&**s));
        }
        symbi::sim::decomp::drain_devices::<Mem>(&devices);
        let sh: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
        let tag = cadence.tag(cfg, cp_width, cfg.start_time, cfg.checkpoint_index);
        let path = checkpoint_name(cfg, &tag);
        match write_tiles(&sh, &path, cfg.checkpoint_index) {
            Ok(io) => reporter.checkpoint_written(cfg, &path, cfg.start_time, io),
            Err(msg) => {
                reporter.fail(&msg);
                return Err(msg);
            }
        }
    }

    // the seam sample rides the plan, independent of the checkpoint guard above: a
    // fresh file (or a tail behind the resume clock) records the resume-clock sample
    // even on restart, while a tail already at the resume clock cedes it.
    if let Some(dp) = &diag_path {
        if diag_plan.as_ref().is_some_and(|p| p.write_initial) {
            if let Some(im) = tiles[0].0.immersed.as_ref() {
                let _ = append_diagnostics(dp, cfg.start_time, &im.bodies);
            }
        }
    }

    let cp_error: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
    // a run resumed at its step cap takes no step and ends through the same epilogue
    let diagnostics = if step_cap_reached(cfg, tiles[0].0.iteration) {
        symbi_sim::run_diagnostics::RunDiagnostics::default()
    } else {
        // the decomposed loop owns the tiles by `&mut` (the per-step immersed-body bookkeeping
        // mutates the bodies). build the `&mut` store handles + the `&` kernels from the same tiles
        // (disjoint tuple fields). the checkpoint callback receives the shared tile slice it needs
        // for the gather (it cannot capture `stores` while the loop holds them mutably).
        let mut stores = Vec::with_capacity(ntiles);
        let mut kernels = Vec::with_capacity(ntiles);
        for (s, k) in tiles.iter_mut() {
            stores.push(&mut **s);
            kernels.push(&*k);
        }
        let schedule = tiled_schedule(cfg, counts, &mut stores);
        reporter.milestone(&format!(
            "halo schedule: {} legs{}; priming primitives, halos and the first exchange",
            schedule.legs().len(),
            if schedule.polar_legs().is_empty() {
                String::new()
            } else {
                format!(" + {} antipodal legs", schedule.polar_legs().len())
            }
        ));
        let guard = retry_guard_of(cfg);
        let exclusion = retry_exclusion_of(cfg);
        for store in stores.iter_mut() {
            store.retry_guard = guard;
            store.retry_exclusion = exclusion;
        }
        announce_retry_guard(&*stores[0]);
        // a declared stationary target's residual, added back before the recovery of every stage
        // (the admissibility redo's included): the target becomes an exact fixed point of the
        // tiled scheme as of the single grid's
        let mut stage_hook = |tile: usize,
                              store: &symbi_sim::state::FieldStore<D, DOF, Mem, f64>,
                              point: symbi_sim::stage::HookPoint,
                              args: symbi_sim::stage::StageArgs| {
            if let (symbi_sim::stage::HookPoint::BeforeC2p, Some(residuals)) = (point, residuals.as_ref()) {
                symbi::sim::refinement::equilibrium::accumulate(
                    &residuals[tile],
                    &store.fields.cons,
                    &store.geom.interior,
                    args.ac * args.dt,
                );
            }
        };
        evolve_scheduled_hooked(
            &mut stores,
            &kernels,
            &schedule,
            &devices,
            cfg.timestepping,
            cfg.start_time,
            cfg.t_final,
            1,
            &transport,
            &mut stage_hook,
            |iter, time, sh| {
                reporter.progress(iter, time);
                // bounded march: stop at the `max_steps` root iteration (0 = unbounded), ahead of
                // the cadence and the diagnostics as on the single grid; the final snapshot
                // records the step that ends the march.
                if step_cap_reached(cfg, sh[0].iteration) {
                    return std::ops::ControlFlow::Break(());
                }
                if cadence.due(time) {
                    let tag = cadence.tag(cfg, cp_width, time, cadence.index);
                    let path = checkpoint_name(cfg, &tag);
                    match write_tiles(sh, &path, cadence.index) {
                        Ok(io) => reporter.checkpoint_written(cfg, &path, time, io),
                        Err(msg) => {
                            *cp_error.borrow_mut() = Some(msg);
                            return std::ops::ControlFlow::Break(());
                        }
                    }
                    cadence.advance(time);
                }
                // body diagnostics at their own cadence: any tile's bodies carry the global
                // (cross-tile-summed) force/torque/accreted-mass after step_bodies_decomposed.
                if let Some(dp) = &diag_path {
                    if diagnostics_due(time, &mut next_diag, diag_interval) {
                        if let Some(im) = sh[0].immersed.as_ref() {
                            let _ = append_diagnostics(dp, time, &im.bodies);
                        }
                    }
                }
                std::ops::ControlFlow::Continue(())
            },
        )
    };

    if let Some(msg) = cp_error.into_inner() {
        reporter.fail(&msg);
        return Err(msg);
    }
    symbi::sim::decomp::drain_devices::<Mem>(&devices);
    let sh: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
    // a march halted on a timestep it could not take publishes its last state as crashed and no
    // final snapshot, as the single grid does
    if let Some(h) = diagnostics.halt.clone() {
        let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
        if let Err(msg) = write_tiles(&sh, &crashed, cadence.index) {
            reporter.fail(&msg);
            return Err(msg);
        }
        let summary = format!(
            "crashed — {} steps, t = {:.4} — {} · state {crashed}",
            h.iter,
            h.time,
            crash_cause(h.dt_cfl, None, h.step.as_ref())
        );
        reporter.fail(&summary);
        // a halted march ends with a failing exit, a timestep it could not take included
        return Err(summary);
    }
    // canonical final snapshot, mirroring the single-grid run; the march drained every device
    // before returning, and the drain here pins that boundary in front of the staging reads.
    let final_path = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Completed));
    let (iter, time) = (tiles[0].0.iteration, tiles[0].0.time);
    match write_tiles(&sh, &final_path, cadence.index) {
        Ok(io) => {
            reporter.checkpoint_written(cfg, &final_path, time, io);
            reporter.finish(iter, time, &final_path);
        }
        Err(msg) => {
            reporter.fail(&msg);
            return Err(msg);
        }
    }
    Ok(diagnostics)
}

/// the global grid a decomposed run's tiles partition: the run's cell counts, origin, spacing
/// and coordinate maps, with the tiles' halo width.
fn decomposed_grid<const D: usize>(cfg: &Config, ng: usize) -> symbi_sim::checkpoint::GlobalGrid<D> {
    symbi_sim::checkpoint::GlobalGrid {
        cells: std::array::from_fn(|ax| cfg.n_cells[ax]),
        interior_lo: [0; D],
        ng,
        x_lo: std::array::from_fn(|ax| cfg.x_lo[ax]),
        dx: std::array::from_fn(|ax| cfg.dx[ax]),
        maps: axis_maps::<D>(cfg),
    }
}

/// each tile's global interior offset per axis, in the partition's flat tile order.
/// the axis-0-fastest linear index on the whole grid of the tile-local cell `idx` of a tile whose
/// first cell is `tile_lo`: the order the python generators yield rows in
fn global_lin<const D: usize>(tile_lo: &[usize; D], idx: &[isize; D], n: &[usize; D]) -> usize {
    let mut lin = 0usize;
    let mut stride = 1usize;
    for ax in 0..D {
        lin += (tile_lo[ax] + idx[ax] as usize) * stride;
        stride *= n[ax];
    }
    lin
}

fn tile_offsets<const D: usize>(partition: &Partition<D>) -> Vec<[isize; D]> {
    let counts = partition.counts();
    (0..partition.n_tiles())
        .map(|flat| {
            let tc = symbi::sim::decomp::unflatten(flat, counts);
            let ext = partition.tile_extents(tc);
            std::array::from_fn(|ax| ext[ax].0 as isize)
        })
        .collect()
}

/// the multi-gpu (gpus>1) refined path: decompose a nested static-refinement hierarchy. each tile
/// is a per-tile `Hierarchy` (its root slab + the global refinement regions clipped to that slab,
/// down to the first region it misses); `evolve_hierarchy_decomposed` drives them in lockstep with
/// a halo exchange on every level, reproducing the single hierarchy bit for bit. for output, each
/// level is written from the sub-grid of the tiles carrying it (a decomposition of the global
/// level) through the partitioned multi-level checkpoint writer.
fn run_refined_decomposed_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    cfg: &Config,
    mut tiles: Vec<Hierarchy<R, D, DOF, M, E, S, Mem, K>>,
    root_offsets: Vec<[isize; D]>,
    root_grid: symbi_sim::checkpoint::GlobalGrid<D>,
    counts: [usize; D],
) -> Result<symbi_sim::run_diagnostics::RunDiagnostics, String>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
{
    use symbi::sim::decomp::enable_peer_mesh;
    use symbi::sim::refinement::{evolve_hierarchy_decomposed, fine_subgrid_of_level};
    use symbi_sim::checkpoint::{
        LevelTiles, PhysicsIdentity, TileView, load_partitioned_level, write_partitioned_checkpoint,
    };

    let ntiles = tiles.len();
    let devices: Vec<i32> = (0..ntiles as i32).collect();
    // zone-cycles per root step over every tile and level: level ll subcycles 2^ll times over
    // its own interior, the same accounting the single-grid hierarchy reports.
    let n_zones: u64 = tiles
        .iter()
        .map(|h| {
            h.levels
                .iter()
                .enumerate()
                .map(|(ll, level)| {
                    let cells: u64 = (0..D)
                        .map(|ax| level.state.geom.interior.spaces[ax].size() as u64)
                        .product();
                    cells * (1u64 << ll)
                })
                .sum::<u64>()
        })
        .sum();
    let slabs: Vec<Vec<usize>> = (0..D)
        .map(|ax| {
            (0..counts[ax])
                .map(|i| {
                    let mut tc = [0usize; D];
                    tc[ax] = i;
                    let g = &tiles[symbi::sim::decomp::flatten(tc, counts)].levels[0].state.geom.interior;
                    g.spaces[ax].size()
                })
                .collect()
        })
        .collect();
    let mut reporter = DecomposedReporter::new(cfg, n_zones, &counts, &slabs);
    enable_peer_mesh(&devices);
    reporter.milestone("peer links opened");

    #[cfg(feature = "gpu")]
    let transport = symbi::sim::decomp::PeerCopy;
    #[cfg(not(feature = "gpu"))]
    let transport = symbi::sim::decomp::LocalCopy;

    // the checkpoint schedule the single grid walks
    let mut cadence = CheckpointCadence::of(cfg);
    let cp_width = checkpoint_time_width(cfg);
    // body diagnostics at their own cadence, from the finest bodies, as the single hierarchy
    // records them: every tile's finest bodies carry the cross-tile-summed reaction
    let diag_path = if cfg.diagnostic_interval > 0.0 && !cfg.bodies.is_empty() {
        Some(format!("{}diagnostics.dat", cfg.data_dir))
    } else {
        None
    };
    let diag_interval = (cfg.diagnostic_interval * cfg.time_unit).max(f64::MIN_POSITIVE);
    let diag_plan = diag_path.as_ref().map(|dp| {
        plan_diagnostics(last_diagnostics_time(dp), cfg.start_time, diag_interval)
            .unwrap_or_else(|detail| panic!("{detail}"))
    });
    let mut next_diag = diag_plan.as_ref().map_or(diag_interval, |p| p.next);

    // per refined level: the sub-grid of the tiles carrying it, each tile's offset on the level's
    // global grid, and that grid. a tile's level-l interior starts at 2^l times its root offset
    // plus its own level-l interior origin (every refinement ratio is two), and the grid spans the
    // tiles' level-l interiors with the root's origin at 2^-l of the root spacing.
    let depth = tiles.iter().map(|h| h.levels.len()).max().unwrap_or(1);
    let fine: Vec<(symbi::sim::refinement::FineSubgrid<D>, Vec<[isize; D]>, symbi_sim::checkpoint::GlobalGrid<D>)> = (1..depth)
        .map(|level| {
            let fg = fine_subgrid_of_level(&tiles, counts, &devices, level)
                .expect("the tiles carrying a level of the run");
            let offsets: Vec<[isize; D]> = fg
                .order
                .iter()
                .map(|&i| {
                    let f = &tiles[i].levels[level].state.geom.interior;
                    std::array::from_fn(|ax| (root_offsets[i][ax] << level) + f.spaces[ax].lo)
                })
                .collect();
            let cells: [usize; D] = std::array::from_fn(|ax| {
                (0..fg.counts[ax])
                    .map(|k| {
                        let mut rc = [0usize; D];
                        rc[ax] = k;
                        let i = fg.order[symbi::sim::decomp::flatten(rc, fg.counts)];
                        tiles[i].levels[level].state.geom.interior.spaces[ax].size()
                    })
                    .sum()
            });
            let interior_lo: [isize; D] = std::array::from_fn(|ax| offsets.iter().map(|o| o[ax]).min().unwrap_or(0));
            let grid = symbi_sim::checkpoint::GlobalGrid {
                cells,
                interior_lo,
                ng: tiles[fg.order[0]].levels[level].state.geom.ng,
                x_lo: root_grid.x_lo,
                dx: std::array::from_fn(|ax| root_grid.dx[ax] / (1u64 << level) as f64),
                maps: None,
            };
            (fg, offsets, grid)
        })
        .collect();
    reporter.milestone(&format!(
        "refinement: {} of {ntiles} tiles carry a fine level, {depth} levels",
        fine.first().map_or(0, |(fg, _, _)| fg.order.len())
    ));
    // every level of every tile records its whole level's interior in the tile's own index frame:
    // a bound that reads the level's extent (the alpha viscosity's slowest orbit) takes the whole
    // level's, as the single hierarchy does, whichever tile holds a part of it
    for (i, tile) in tiles.iter_mut().enumerate() {
        let span = root_grid.interior_span(&tile.levels[0].state.geom, root_offsets[i]);
        tile.levels[0].state.geom.whole_span = Some(span);
    }
    for (level, (fg, offsets, grid)) in fine.iter().enumerate().map(|(k, f)| (k + 1, f)) {
        for (k, &i) in fg.order.iter().enumerate() {
            let span = grid.interior_span(&tiles[i].levels[level].state.geom, offsets[k]);
            tiles[i].levels[level].state.geom.whole_span = Some(span);
        }
    }
    // a restart reads every level of every tile from the checkpoint's global grids.
    if let Some(path) = cfg.restart_path.as_deref() {
        let stored = symbi_sim::checkpoint::checkpoint_level_count(path).map_err(|e| format!("{e}"))?;
        // a deeper checkpoint would lose the levels it carries beyond this run's, a shallower one
        // would leave levels unseeded: both are refused before any level is read
        if stored != depth {
            return Err(format!(
                "restart from '{path}': the checkpoint holds {stored} level(s), this run refines to \
                 {depth}; a refined decomposed run resumes from a checkpoint of its own depth"
            ));
        }
        for (i, tile) in tiles.iter_mut().enumerate() {
            load_partitioned_level(&mut tile.levels[0].state, path, 0, root_offsets[i], &root_grid)
                .map_err(|e| format!("restart of tile {i} root from '{path}': {e}"))?;
        }
        for (level, (fg, offsets, grid)) in fine.iter().enumerate().map(|(k, f)| (k + 1, f)) {
            for (k, &i) in fg.order.iter().enumerate() {
                load_partitioned_level(&mut tiles[i].levels[level].state, path, level, offsets[k], grid)
                    .map_err(|e| format!("restart of tile {i} level {level} from '{path}': {e}"))?;
            }
        }
    }
    let identity = PhysicsIdentity::of(&tiles[0].levels[0].state);

    // gather each level of the decomposed tiles into the global hierarchy (root over `counts`, fine
    // over the fine sub-grid), then write all the global levels through the multi-level writer.
    let write_cp = |tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], path: &str, cp_index: u64| {
        let mut levels = vec![LevelTiles {
            tiles: tiles
                .iter()
                .zip(&root_offsets)
                .map(|(h, &offset)| TileView { state: &h.levels[0].state.store, offset })
                .collect(),
            grid: root_grid.clone(),
            header: None,
        }];
        for (level, (fg, offsets, grid)) in fine.iter().enumerate().map(|(k, f)| (k + 1, f)) {
            levels.push(LevelTiles {
                tiles: fg
                    .order
                    .iter()
                    .zip(offsets)
                    .map(|(&i, &offset)| TileView { state: &tiles[i].levels[level].state.store, offset })
                    .collect(),
                grid: grid.clone(),
                header: None,
            });
        }
        let t_io = std::time::Instant::now();
        write_partitioned_checkpoint::<R, D, DOF, Mem>(&identity, &levels, path, &checkpoint_metadata(cfg, cp_index))
            .map_err(|e| checkpoint_write_error(path, e))
            .map(|()| t_io.elapsed().as_secs_f64())
    };

    // t=start initial condition. every device finishes its seeding before the staging reads.
    if cfg.checkpoint_index == 0 || cfg.start_time == 0.0 {
        symbi::sim::decomp::drain_devices::<Mem>(&devices);
        let tag = cadence.tag(cfg, cp_width, cfg.start_time, cfg.checkpoint_index);
        let path = checkpoint_name(cfg, &tag);
        match write_cp(&tiles, &path, cfg.checkpoint_index) {
            Ok(io) => reporter.checkpoint_written(cfg, &path, cfg.start_time, io),
            Err(msg) => {
                reporter.fail(&msg);
                return Err(msg);
            }
        }
    }

    if let Some(dp) = &diag_path {
        if diag_plan.as_ref().is_some_and(|p| p.write_initial) {
            if let Some(im) = tiles[0].levels.last().and_then(|l| l.state.immersed.as_ref()) {
                let _ = append_diagnostics(dp, cfg.start_time, &im.bodies);
            }
        }
    }

    let cp_error: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
    let guard = retry_guard_of(cfg);
    let exclusion = retry_exclusion_of(cfg);
    for tile in tiles.iter_mut() {
        for level in tile.levels.iter_mut() {
            level.state.retry_guard = guard;
            level.state.retry_exclusion = exclusion;
        }
    }
    announce_retry_guard(&tiles[0].levels[0].state);
    reporter.milestone("priming the hierarchy levels, their halos and the first exchange");
    // a run resumed at its step cap takes no step and ends through the same epilogue
    let diagnostics = if step_cap_reached(cfg, tiles[0].levels[0].state.iteration) {
        symbi_sim::run_diagnostics::RunDiagnostics::default()
    } else {
        evolve_hierarchy_decomposed(
        &mut tiles,
        counts,
        &devices,
        &transport,
        cfg.timestepping,
        cfg.start_time,
        cfg.t_final,
        1,
        |iter, time, tiles| {
            reporter.progress(iter, time);
            // a crashed tile halts the march; its state is published as crashed below, never
            // through the cadence
            if tiles.iter().any(|h| h.crash.is_some()) {
                return std::ops::ControlFlow::Break(());
            }
            // bounded march: stop at the `max_steps` root iteration (0 = unbounded), ahead of the
            // cadence as on the single grid; the final snapshot records the step that ends it.
            if step_cap_reached(cfg, tiles[0].levels[0].state.iteration) {
                return std::ops::ControlFlow::Break(());
            }
            if cadence.due(time) {
                let tag = cadence.tag(cfg, cp_width, time, cadence.index);
                let path = checkpoint_name(cfg, &tag);
                match write_cp(tiles, &path, cadence.index) {
                    Ok(io) => reporter.checkpoint_written(cfg, &path, time, io),
                    Err(e) => {
                        *cp_error.borrow_mut() = Some(e);
                        return std::ops::ControlFlow::Break(());
                    }
                }
                cadence.advance(time);
            }
            if let Some(dp) = &diag_path {
                if diagnostics_due(time, &mut next_diag, diag_interval) {
                    if let Some(im) = tiles[0].levels.last().and_then(|l| l.state.immersed.as_ref()) {
                        let _ = append_diagnostics(dp, time, &im.bodies);
                    }
                }
            }
            std::ops::ControlFlow::Continue(())
        },
    )
    };

    if let Some(msg) = cp_error.into_inner() {
        reporter.fail(&msg);
        return Err(msg);
    }
    symbi::sim::decomp::drain_devices::<Mem>(&devices);
    // a tile that halted on a timestep it could not take, a halted root step or a caught step
    // panic records the crash: the run publishes its state as crashed and no final snapshot, as
    // the single grid does, and a halted step or a panic ends with a failing exit
    if let Some(c) = tiles.iter().find_map(|h| h.crash.as_ref()) {
        let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
        let summary = format!(
            "crashed — {} steps, t = {:.4} — {} · state {crashed}",
            c.iter,
            c.time,
            crash_cause(c.dt_cfl, c.panic.as_deref(), c.halt.as_ref())
        );

        if let Err(msg) = write_cp(&tiles, &crashed, cadence.index) {
            reporter.fail(&msg);
            return Err(msg);
        }
        reporter.fail(&summary);
        // every crash ends with a failing exit, a timestep the march could not take included
        return Err(summary);
    }
    // canonical final snapshot, after every device drained.
    let final_path = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Completed));
    match write_cp(&tiles, &final_path, cadence.index) {
        Ok(io) => {
            let root = &tiles[0].levels[0].state;
            let (iter, time) = (root.iteration, root.time);
            reporter.checkpoint_written(cfg, &final_path, time, io);
            reporter.finish(iter, time, &final_path);
        }
        Err(msg) => {
            reporter.fail(&msg);
            return Err(msg);
        }
    }
    Ok(diagnostics)
}

/// the multi-gpu (gpus>1) refined hydro path: per-tile static-refinement hierarchies driven by the
/// `evolve_hierarchy_decomposed` (a halo exchange on every level).
/// each tile builds its root slab + the global refinement regions clipped to that slab (down to the
/// first region it misses); a patch that spans a cut is split into the abutting tiles and the
/// halos of every refined level are exchanged at the cut. output writes each level from the tiles
/// carrying it through the multi-level checkpoint writer. hydro with nested refined regions,
/// carrying user sources and driven boundaries, and immersed bodies (with shapes) under one
/// refined region -- each attached per tile and evolved in lockstep, matching the single
/// hierarchy bit for bit. mass-transport tracers use global owners and migrate complete records
/// across cuts. mesh motion and bonded fragments are carried by the single hierarchy alone.
macro_rules! build_and_run_hydro_decomposed_refined {
    (
        $cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty,
        $eos:expr, $eos_ty:ty, prim = |$row:ident| $prim:expr, kernels = |$s:ident, $role:ident| $kernels:expr
    ) => {{
        use symbi::sim::decomp::unflatten;
        use symbi::sim::refinement::launched::receiving_faces_of_tile;
        use symbi_sim::atlas::TileId;
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<$regime_ty, $d, $geom_ty, $eos_ty>;
        // the per-tile / global hierarchy type. DOF = D for hydro; the kernel set is the substrate's
        // associated type (matches what `sim.substrate()` yields).
        type Hier = Hierarchy<
            $regime_ty,
            $d,
            $d,
            $geom_ty,
            $eos_ty,
            DefaultSpace,
            DefaultMemory,
            <Sim as symbi::prelude::SimSubstrate<DefaultMemory, f64, $d>>::KernelSet,
        >;

        // the refined decomposition carries plain hydro. the combinations below each need their own
        // cross-level multi-tile handling, so they are refused.

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        let partition = tile_partition(n, cfg)?;
        let counts = partition.counts();
        let dx: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
        let ntiles: usize = counts.iter().product();
        let prolong = prolong_order_for(&cfg.reconstruction_name);
        let regions = refinement_regions_nd::<$d>(&cfg.refinement_regions)?;
        // the level-1 patch in root cells, whose faces on cuts carry register units across them
        let root_patch = launch_patch::<$d>(cfg)?;
        // immersed bodies give each tile's own deepest level the full bodies; the body couples to
        // the run's finest level alone, so a tile holding fewer levels adds no reaction

        let whole = whole_grid::<$d>(cfg);
        // build N per-tile hierarchies (root slab + the clipped region, or single-level).
        let mut tiles: Vec<Hier> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            // the tile's place on the whole grid, and the refinement regions its extent holds
            let tile = whole.place(&partition, tc);
            let tile_lo = tile.offset;
            let tile_regions = clip_regions_to_tile::<$d>(&regions, &whole, &tile);
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<Hier, String> {
                let sim = Sim::build($regime, $eos, $geom)
                    .placed(&whole, &tile)
                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            lin += (tile_lo[ax] + idx[ax] as usize) * stride;
                            stride *= n[ax];
                        }
                        let $row = &prims[lin];
                        $prim
                    })
                    .build();
                // the physical clock starts at start_time on the root, and the hierarchy built
                // from it starts every fine level there, as on the single hierarchy; the root
                // carries the user timestep clamp the hierarchy applies
                let mut sim = sim;
                seed_run_state(&mut sim, cfg)?;
                // the dye rides on the root; the hierarchy carries it onto the fine levels
                let sim = if cfg.chi_ic.is_empty() {
                    sim
                } else {
                    let sim = sim
                        .with_passive_scalar()
                        .map_err(|e| format!("tile {flat} dye allocation: {e:?}"))?;
                    seed_tile_dye(&sim, &cfg.chi_ic, tile_lo, n);
                    sim
                };
                // every level of the tile carries the configured numerics, the user sources, the
                // driven boundaries and the gradient walls; the tile root carries the excision
                let kernels_of = |$s: &Sim, $role: LevelRole| $kernels;
                let sub = kernels_of(&sim, LevelRole::Root);
                let make = |s: &Sim| kernels_of(s, LevelRole::Fine);
                // a patch face on a root cut contributes its register units to the tile across the
                // cut, and a tile beside such a face holds the receiving register of the cells it
                // corrects
                let mut h = if tile_regions.is_empty() {
                    let h = Hierarchy::single(sim, sub);
                    match receiving_faces_of_tile(&partition, root_patch, TileId(flat as u32)) {
                        Some((patch_local, faces)) => h
                            .with_receiving_register(&patch_local, faces)
                            .map_err(|e| format!("tile {flat} receiving register: {e:?}"))?,
                        None => h,
                    }
                } else {
                    // a patch spanning a periodic axis cut into tiles has no face along it and its
                    // level wraps across the tiles, as the root does
                    let (clipped, whole_patch): (Vec<_>, Vec<_>) = tile_regions.iter().copied().unzip();
                    let ring = whole.topology(counts);
                    Hierarchy::with_refinement_placed(sim, sub, &clipped, &whole_patch, prolong, make, n, std::array::from_fn(|ax| ring.is_periodic(ax)))
                        .map_err(|e| format!("tile {flat} refinement build: {e:?}"))?
                };
                // immersed bodies on every tile hierarchy at their global positions: the finest
                // level owns the full (accreting) bodies, coarser levels a gravity-only proxy —
                // finest-owns-bodies per tile. the decomposed driver sums the backward feedback
                // across tiles and advances every tile's bodies identically; the clipped sink
                // containment (sphere overlap with a tile must lie inside its fine level) is
                // asserted each step.
                if !cfg.bodies.is_empty() {
                    h = h.with_bodies(build_bodies_and_horizon::<$d>(cfg));
                    h.attach_body_shapes(build_body_shapes_and_horizon(cfg));
                }
                // the root exchange closes a periodic axis cut into tiles across its seam; a refined
                // patch keeps clear of the seam, whose faces are cuts to the refinement build
                let ring = whole.topology(counts);
                h.set_root_layout(n, tile_lo, std::array::from_fn(|ax| ring.is_periodic(ax)));
                Ok(h)
            })?;
            tiles.push(built);
        }
        // fine seeding + prime, decomposition-aware and after every tile exists: the seed
        // prolongs conserved components, whose cut ghosts only the dedicated cons exchange
        // fills -- seeded per tile inside the loop, a patch spanning a cut prolongs from the
        // tile's standalone boundary fill and the run differs from its monolithic twin before
        // the first step. prime runs a c2p audit per level, so it follows the seed.
        {
            // one tile per device, and the same transport selection as the evolve loop
            // (`run_refined_decomposed_loop`): peer copies on gpu, plain copies on host.
            let devices: Vec<i32> = (0..tiles.len() as i32).collect();
            #[cfg(feature = "gpu")]
            let seed_transport = symbi::sim::decomp::PeerCopy;
            #[cfg(not(feature = "gpu"))]
            let seed_transport = symbi::sim::decomp::LocalCopy;
            symbi::sim::refinement::seed_decomposed_fine_from_coarse(
                &tiles,
                counts,
                &devices,
                &seed_transport,
            )
            .map_err(|e| format!("decomposed fine seed: {e:?}"))?;
            // the declared stationary target on every level of every tile, its imbalance read off
            // the tiled probe per level; a fresh run seeded from it starts there, as the single
            // hierarchy does
            if let Some(target) = cfg.equilibrium_json.as_ref() {
                use symbi_hydro::state::PrimFromSlots;
                let declared = symbi_source_compile::EquilibriumConfig::from_json(target)
                    .map_err(|e| format!("stationary target parse: {e}"))?;
                let has_energy = tiles[0].levels[0].state.fields.cons.has_energy();
                let expected = 1 + $d + usize::from(has_energy);
                if declared.outputs.len() != expected || declared.dim != $d {
                    return Err(format!(
                        "the declared stationary target supplies {} components on a {}-dimensional grid; this run needs {expected} on a {}-dimensional grid",
                        declared.outputs.len(), declared.dim, $d
                    ));
                }
                let expression = declared
                    .to_expression()
                    .map_err(|e| format!("the declared stationary target does not load: {e:?}"))?;
                symbi::sim::refinement::with_equilibrium_decomposed(&mut tiles, counts, &devices, &seed_transport, |x| {
                    let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                    <<$regime_ty as Regime<f64, $d>>::Prim as PrimFromSlots<f64, $d>>::from_slots(&expression.eval(at(0), at(1), at(2), 0.0))
                })
                .map_err(|e| format!("stationary target: {}", e.detail))?;
                if cfg.seed_from_equilibrium {
                    for (k, h) in tiles.iter_mut().enumerate() {
                        symbi::symbi_xpu::with_device(devices[k], || h.seed_equilibrium());
                    }
                }
            }
            for (k, h) in tiles.iter_mut().enumerate() {
                // the declared perturbation over the seeded levels at their own cell centers,
                // the covered coarse cells re-restricted, as the single hierarchy lays it
                let has_energy = h.levels[0].state.fields.cons.has_energy();
                if lay_perturbation!(h, cfg, $d, has_energy) {
                    symbi::symbi_xpu::with_device(devices[k], || h.sync_perturbed());
                }
                symbi::symbi_xpu::with_device(devices[k], || h.prime());
            }
        }

        // the full-size output hierarchy (root + the full region): gather scatters each level's tile
        // interiors into it. lives on device 0 (touched only at output).
        let offsets = tile_offsets(&partition);
        let grid = decomposed_grid::<$d>(cfg, tiles[0].levels[0].state.geom.ng);
        // the population seeds once from the whole seeded hierarchy, as the single hierarchy seeds
        // its own, and each tile keeps the records its cells own; the whole hierarchy lives for that
        // seeding alone
        if cfg.n_tracers > 0 {
            let global = symbi::symbi_xpu::with_device(0, || -> Result<Hier, String> {
                let kernels_of = |$s: &Sim, $role: LevelRole| $kernels;
                let groot = Sim::build($regime, $eos, $geom)
                    .cells(n)
                    .ghosts(ghost_width(cfg))
                    .origin(std::array::from_fn(|ax| cfg.x_lo[ax]))
                    .spacing(dx)
                    .coord_maps(axis_maps::<$d>(cfg))
                    .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("global seeding allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let $row = &prims[global_lin::<$d>(&[0; $d], &idx, &n)];
                        $prim
                    })
                    .build();
                let gsub = kernels_of(&groot, LevelRole::Fine);
                let gh = Hierarchy::with_refinement(groot, gsub, &regions, prolong, |s: &Sim| kernels_of(s, LevelRole::Fine))
                    .map_err(|e| format!("global refinement build: {e:?}"))?;
                gh.seed_fine_from_coarse().map_err(|e| format!("global fine seed: {e:?}"))?;
                Ok(gh)
            })?;
            symbi::sim::refinement::seed_decomposed_hierarchy_tracers(&global, &mut tiles, cfg.n_tracers);
            // an ito run carries the seeded population as continuous tracers on every level of every
            // tile, as the single hierarchy converts its own
            if let Some(order) = configured_ito_order(cfg)? {
                for tile in &mut tiles {
                    for level in &mut tile.levels {
                        convert_seeded_tracers(&mut level.state, order)?;
                    }
                }
            }
        }
        run_refined_decomposed_loop(cfg, tiles, offsets, grid, counts)
    }};
}

/// the refined magnetized tiles (newtonian, relativistic or isothermal MHD): each tile hierarchy is
/// its root slab seeded from its slice of the global staggered faces and the refinement regions
/// clipped to it, with the numerics the single hierarchy installs on every level; the fine levels
/// seed from their parents after the root cut halos are exchanged. `$prim` builds a cell's primitive
/// from its row and its cell field.
macro_rules! build_and_run_magnetized_decomposed_refined {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty, $eos:expr, $eos_ty:ty, $prim:expr) => {{
        use symbi::sim::decomp::unflatten;
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<$regime_ty, $d, 3, $geom_ty, $eos_ty>;

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if bufs.len() < 3 {
            return Err(format!("mhd needs 3 staggered b-field generators, got {}", bufs.len()));
        }
        let partition = tile_partition(n, cfg)?;
        let counts = partition.counts();
        let dx: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
        let ntiles: usize = counts.iter().product();
        let prolong = prolong_order_for(&cfg.reconstruction_name);
        let regions = refinement_regions_nd::<$d>(&cfg.refinement_regions)?;
        if regions.len() > 1 && !cfg.bodies.is_empty() && !gated_nested_magnetized_bodies(cfg, $d, regions.len()) {
            return Err(format!(
                "immersed bodies with {} nested refinement levels on magnetized tiles (gpus > 1) are \
                 gated for one draining sink on a 2.5D cartesian newtonian MHD run alone (PLM, RK2, \
                 HLLE, contact EMF, linear spacing, outflow walls, three levels, no resistive, \
                 viscous, excised, driven or user-source term); run on one device or keep one \
                 refinement region",
                regions.len() + 1
            ));
        }
        // the kernels the single magnetized hierarchy installs on every level: the solver, the edge
        // EMF method, the viscosity, the resistivity and the driven boundaries in Driven-id order;
        // the tile root alone carries the excision
        let settings = KernelSettings::of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC);
        let kernels_of = |s: &Sim| mhd_kernels(s.substrate(), &settings, LevelRole::Fine).expect("refined kernel set");

        let whole = whole_grid::<$d>(cfg);
        let mut tiles = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            // the tile's place on the whole grid, and the refinement regions its extent holds
            let tile = whole.place(&partition, tc);
            let (tile_lo, m) = (tile.offset, tile.cells);
            let tile_regions = clip_regions_to_tile::<$d>(&regions, &whole, &tile);
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<_, String> {
                // the tile's slice of each global face buffer: a face shared by two tiles reads the
                // same global value in both
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                let mut sim = Sim::build($regime, $eos, $geom)
                    .placed(&whole, &tile)
                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                        let gidx: [isize; $d] = std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        $prim(&prims[lin], mag)
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // the physical clock and the user timestep clamp ride on the root, as on the single
                // hierarchy
                seed_run_state(&mut sim, cfg)?;
                let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                let ring = whole.topology(counts);
                let mut h = if tile_regions.is_empty() {
                    Hierarchy::single(sim, sub)
                } else {
                    // the build sees the whole grid and its wrapping axes: a patch spanning a
                    // periodic axis cut into tiles has no face along it and wraps across the seam
                    let (clipped, whole_patch): (Vec<_>, Vec<_>) = tile_regions.iter().copied().unzip();
                    Hierarchy::with_refinement_placed(sim, sub, &clipped, &whole_patch, prolong, kernels_of, n, std::array::from_fn(|ax| ring.is_periodic(ax)))
                        .map_err(|e| format!("tile {flat} refinement build: {e:?}"))?
                };
                // the sinks on each tile's finest level at their global positions, gravity-only
                // proxies below, as on the single hierarchy
                if !cfg.bodies.is_empty() {
                    h = h.try_with_bodies(build_bodies_and_horizon::<$d>(cfg)).map_err(|e| format!("tile {flat}: {e}"))?;
                }
                h.set_root_layout(n, tile_lo, std::array::from_fn(|ax| ring.is_periodic(ax)));
                Ok(h)
            })?;
            tiles.push(built);
        }
        // the fine levels seed from their parents once every tile exists and the root cut halos are
        // exchanged; prime audits each level's recovery after the seed
        {
            let devices: Vec<i32> = (0..tiles.len() as i32).collect();
            #[cfg(feature = "gpu")]
            let seed_transport = symbi::sim::decomp::PeerCopy;
            #[cfg(not(feature = "gpu"))]
            let seed_transport = symbi::sim::decomp::LocalCopy;
            symbi::sim::refinement::seed_decomposed_fine_from_coarse(&tiles, counts, &devices, &seed_transport)
                .map_err(|e| format!("decomposed fine seed: {e:?}"))?;
            for (k, h) in tiles.iter_mut().enumerate() {
                symbi::symbi_xpu::with_device(devices[k], || h.prime());
            }
        }
        let offsets = tile_offsets(&partition);
        let grid = decomposed_grid::<$d>(cfg, tiles[0].levels[0].state.geom.ng);
        // the population seeds once from the whole seeded hierarchy, as the single hierarchy seeds
        // its own, and each tile keeps the records its cells own; the whole hierarchy lives for that
        // seeding alone
        if cfg.n_tracers > 0 {
            let global = {
                let groot = Sim::build($regime, $eos, $geom)
                    .cells(n)
                    .ghosts(ghost_width(cfg))
                    .origin(std::array::from_fn(|ax| cfg.x_lo[ax]))
                    .spacing(dx)
                    .coord_maps(axis_maps::<$d>(cfg))
                    .boundaries(boundaries_nd::<$d>(&cfg.boundaries))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("global seeding allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&[0; $d], &idx, &n);
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                        });
                        $prim(&prims[lin], mag)
                    })
                    .seed_faces_indexed(&bufs[0..$d])
                    .build();
                let gsub = kernels_of(&groot);
                let gh = Hierarchy::with_refinement(groot, gsub, &regions, prolong, kernels_of)
                    .map_err(|e| format!("global refinement build: {e:?}"))?;
                gh.seed_fine_from_coarse().map_err(|e| format!("global fine seed: {e:?}"))?;
                gh
            };
            symbi::sim::refinement::seed_decomposed_hierarchy_tracers(&global, &mut tiles, cfg.n_tracers);
            if let Some(order) = configured_ito_order(cfg)? {
                for tile in &mut tiles {
                    for level in &mut tile.levels {
                        convert_seeded_tracers(&mut level.state, order)?;
                    }
                }
            }
        }
        run_refined_decomposed_loop(cfg, tiles, offsets, grid, counts)
    }};
}

/// the multi-gpu (gpus>1) hydro path. decompose the domain into
/// `cfg.n_gpus` tiles, bind each tile to a device, evolve them in lockstep with halo exchange
/// (`decomp::evolve_decomposed`), and for output gather the tiles into one
/// full-size sim written by the existing single-grid checkpoint path. the scope is single-level
/// hydro: refinement uses the decomposed-hierarchy path above; immersed bodies / user sources are
/// wired. checkpoint cadence is the linear `checkpoint_interval`; the log cadence + live display
/// are single-grid only. the correctness contract is decomposed == monolithic.
macro_rules! build_and_run_hydro_decomposed {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty, $d:literal, $dof:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::unflatten;
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        // `$d` is the grid dimension, `$dof` the momentum-component count; they differ for the
        // swirl lift (the azimuthal momentum on a 2D (r, z)/(r, theta) grid, DOF = 3).
        type Sim = SimDefaultGeneric<$regime_ty, $d, $dof, $geom_ty, EosSelect<f64>>;

        // refinement + gpus>1 takes the decomposed-hierarchy path (per-tile hierarchies + the
        // root/fine halo exchange); plain single-level continues below.
        if cfg.refinement_enabled {
            let settings = KernelSettings::of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC);
            return build_and_run_hydro_decomposed_refined!(
                $cfg, $prims, $regime, $regime_ty, $d, $geom, $geom_ty, host_eos(cfg), EosSelect<f64>,
                prim = |row| Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $d])),
                kernels = |s, role| gas_kernels(s.substrate(), &settings, role).expect("tile kernel set")
            );
        }
        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        // the dye IC (when present) spans the whole global grid in the same axis-0-fastest order
        // as the prims; each tile reads its own sub-box out of it below.
        if !cfg.chi_ic.is_empty() && cfg.chi_ic.len() != total {
            return Err(format!(
                "passive_scalar yielded {} dye values, expected {total}",
                cfg.chi_ic.len()
            ));
        }
        // choose the tile grid (product == n_gpus), validate even divisibility.
        let partition = tile_partition(n, cfg)?;
        let counts = partition.counts();
        let ntiles: usize = counts.iter().product();
        // the whole grid the tiles partition; faces between tiles become CoarseFine cuts.
        let whole = whole_grid::<$d>(cfg);
        let settings = KernelSettings::of(cfg, <$regime_ty as Regime<f64, $dof>>::SPEC);

        // build N tiles, each allocated + substrate-built in its own device context.
        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            // the tile's place on the whole grid: its first cell, its cells and the boundaries its
            // faces carry; every position it evaluates is the whole grid's bit for bit
            let tile = whole.place(&partition, tc);
            let tile_lo = tile.offset;
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                let sim = Sim::build($regime, host_eos(cfg), $geom)
                    .placed(&whole, &tile)
                    .ghosts(ghost_width(cfg))
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
                            let g = tile_lo[ax] + idx[ax] as usize;
                            lin += g * stride;
                            stride *= n[ax];
                        }
                        let row = &prims[lin];
                        Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $dof]))
                    })
                    .build();
                // seed the passive scalar (dye) on this tile: cons.chi = rho*chi, prim.chi = chi
                // over the tile interior, indexed from the global dye IC by the same axis-0-fastest
                // global lin as the prim seed above. the transport carries prim.chi across cuts
                // (derived into the exchange set from the store), so the decomposed dye matches the
                // monolithic run to round-off.
                let sim = if cfg.chi_ic.is_empty() {
                    sim
                } else {
                    let sim = sim
                        .with_passive_scalar()
                        .map_err(|e| format!("tile {flat} dye allocation: {e:?}"))?;
                    seed_tile_dye(&sim, &cfg.chi_ic, tile_lo, n);
                    sim
                };
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their global positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically.

                // gravity / accretion sinks plus bonded fragments + shaped walls, attached per
                // tile at their global positions (refinement takes the refined decomposed path
                // earlier, so no hierarchy branch here). unshaped point bodies get an empty shape
                // list (a no-op), so only shaped / fragment runs differ from a bare with_bodies.
                let has_any_body = !cfg.bodies.is_empty() || cfg.bonded_assembly.is_some();
                let mut sim = if !has_any_body {
                    sim
                } else {
                    attach_bodies_and_fragments!(sim, cfg, $d)
                };
                // clock + mesh motion per tile: every tile carries the identical a(t) law and
                // the decomposed loop advances them in lockstep with the shared dt.
                seed_run_state(&mut sim, cfg)?;
                // the declared perturbation at the tile's own cell centers, the single grid's; with a
                // declared stationary target it is laid over the target, after the probe
                let has_energy = sim.fields.cons.has_energy();
                if cfg.equilibrium_json.is_none() {
                    lay_perturbation!(sim, cfg, $dof, has_energy);
                }
                // the numerics, sources, driven boundaries and gradient walls the single grid
                // installs; each tile evaluates its sources and prescriptions at the whole grid's
                // coordinates of its cells, so a tile carries the single grid's scheme bit for bit
                let sub = gas_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        // one full-size sim as the output view: the gather scatters tile interiors into it and
        // the existing writer serializes it. lives on device 0 (it is only touched at output).
        let offsets = tile_offsets(&partition);
        let grid = decomposed_grid::<$d>(cfg, tiles[0].0.geom.ng);
        // the tracer population seeds once from the whole-grid density and splits across the
        // tiles by position, so the run starts from the particles a single grid would carry;
        // the whole-grid state lives for that seeding alone.
        if cfg.n_tracers > 0 {
        let global = Sim::build($regime, host_eos(cfg), $geom)
            .placed(&whole, &whole.single())
            .ghosts(ghost_width(cfg))
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
                Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $dof]))
            })
            .build();
        // the gather copies each tile's dye into this output view (data_fields includes chi), so
        // the destination slot must exist; the interior is overwritten every checkpoint, so it is
        // allocated but not seeded here.
        let mut global = if cfg.chi_ic.is_empty() {
            global
        } else {
            global
                .with_passive_scalar()
                .map_err(|e| format!("global output dye allocation: {e:?}"))?
        };

        // lagrangian tracers: seed the population once from the global density (the monolithic
        // seeding) and split it across the tiles by initial position, so a decomposed run starts
        // from the identical particles a single-grid run would. the output view carries an empty
        // set the checkpoint gather refills from the tiles each write.
        if cfg.n_tracers > 0 {
            let per_tile = partition_configured_tracers(&global, cfg, &partition);
            if let Some(order) = configured_ito_order(cfg)? {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.continuous_tracers = Some(
                        symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                            .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                    );
                }
                global.continuous_tracers = Some(
                    symbi_sim::tracers::ContinuousTracerSet::allocate(0, order)
                        .map_err(|detail| format!("continuous tracer output: {detail}"))?,
                );
            } else {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.tracers = Some(set);
                }
                global.tracers = Some(symbi_sim::tracers::TracerSet::default());
            }
        }

        // hand the built tiles + output sim to the regime-agnostic decomposed loop (evolve +
        // gather + checkpoint, universal transport). every regime shares this loop.
        }
        let residuals = tiled_equilibrium!(tiles, cfg, counts, $d);
        // the declared perturbation over the target, as the single grid lays it after the target
        if cfg.equilibrium_json.is_some() {
            let has_energy = tiles[0].0.fields.cons.has_energy();
            for (sim, _) in tiles.iter_mut() {
                lay_perturbation!(sim, cfg, $dof, has_energy);
            }
        }
        run_decomposed_loop(cfg, tiles, offsets, grid, counts, residuals)
    }};
}

/// the (dims, chart, spacetime) combinations with a baked GR-hydro kernel: the single source of
/// truth for the dispatch guard and the launch scope
fn baked_gr_hydro(dims: usize, coords: &str, spacetime: &str) -> bool {
    matches!(
        (dims, coords, spacetime),
        (3, "cartesian", "schwarzschild_ks")
            | (3, "cartesian", "kerr_ks")
            | (1, "spherical", "schwarzschild_ks")
            | (2, "spherical", "kerr_ks")
            | (2, "spherical", "schwarzschild_ks")
            | (2, "cylindrical", "schwarzschild_ks")
            | (3, "cylindrical", "schwarzschild_ks")
            | (3, "cylindrical", "kerr_ks")
    )
}

/// the baked GR-MHD combinations: the arms of `mhd_dispatch`, which the launched rmhd workers build
/// alike
fn baked_gr_mhd(dims: usize, coords: &str, spacetime: &str) -> bool {
    matches!(
        (dims, coords, spacetime),
        (1, "spherical", "schwarzschild_ks")
            | (2, "spherical", "kerr_ks")
            | (3, "cartesian", "schwarzschild_ks")
            | (3, "cartesian", "kerr_ks")
            | (2, "cylindrical", "schwarzschild_ks")
    )
}

/// the curved spacetimes the launched refined workers build: the relativistic gas and the
/// relativistic magnetized gas on the three-dimensional kerr-schild box, its root excised or not
fn launched_refined_gr(cfg: &Config) -> bool {
    matches!(cfg.regime.as_str(), "rhd" | "rmhd") && cfg.dims == 3 && cfg.coord_system == "cartesian" && matches!(cfg.spacetime.as_str(), "schwarzschild_ks" | "kerr_ks")
}

/// the baked GR-hydro combinations the launched gas workers build: the kerr-schild box in three
/// dimensions, the radial line, the spherical meridional plane and the cylindrical charts, the arms
/// of `charted!` in `run_worker_process`
fn launched_gr_hydro(dims: usize, coords: &str, spacetime: &str) -> bool {
    baked_gr_hydro(dims, coords, spacetime) && matches!((dims, coords), (3, "cartesian") | (1 | 2, "spherical") | (2 | 3, "cylindrical"))
}

/// expand the (geometry x dims) arms for one hydro regime. cartesian / spherical
/// / cylindrical are all wired across 1/2/3d (each is a unit-struct `Metric`).
macro_rules! hydro_dispatch {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            // fail-loud guard: a non-minkowski spacetime that is not one of the baked GR
            // combinations below would otherwise fall through to a flat `(dims, coords)` arm and run
            // silently on a Minkowski metric (wrong physics, zero warning). baked_gr_hydro is the
            // single source of truth for the baked GR-hydro arms; `test_dispatch_rejects_unbaked_gr`
            // asserts it stays in lockstep with the actual arms (guarded-arm-or-Err, never silent-flat).
            (d, c) if $cfg.spacetime != "minkowski" && !baked_gr_hydro(d, c, &$cfg.spacetime) =>
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
            // GR (kerr-schild) cartesian: the (x, y) equatorial slice of the horizon-penetrating
            // chart — SchwarzschildKSCartesian selects the `_cart` metric-aware c2p +
            // per-sweep flux + light-cone CFL (non-diagonal gamma, shift on every axis, no polar
            // axis). guarded before the flat cartesian arm; 2D equatorial slice, DOF = 2 (no swirl).
            (2, "cartesian") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 2, 2,
                SchwarzschildKSCartesian { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCartesian<f64>
            ),
            // GR (kerr-schild) cartesian 3D: the full horizon-penetrating box — no polar axis
            // anywhere, so a torus resolves its poles like any other direction. DOF = NDIM = 3.
            (3, "cartesian") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 3, 3,
                SchwarzschildKSCartesian { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCartesian<f64>
            ),
            // spinning kerr on the cartesian chart (spin about z): the rank-1 kerr-schild
            // metric with the oblate-spheroidal radius — non-diagonal gamma, shift on every
            // axis, frame dragging in the swirl of l. DOF == NDIM (no extra momentum slot; the
            // cartesian components already span the dragging). the 2D instance is the exact
            // equatorial slice (l_z = 0 at z = 0).
            (2, "cartesian") if $cfg.spacetime == "kerr_ks" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 2, 2,
                KerrKSCartesian { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin },
                KerrKSCartesian<f64>
            ),
            (3, "cartesian") if $cfg.spacetime == "kerr_ks" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 3, 3,
                KerrKSCartesian { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin },
                KerrKSCartesian<f64>
            ),
            (2, "cartesian") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 2, 2, Cartesian, Cartesian)
            }
            (3, "cartesian") => {
                build_and_run_hydro!($cfg, $prims, $regime, $regime_ty, 3, 3, Cartesian, Cartesian)
            }
            // GR (ingoing Kerr-Schild) spherical: the horizon-penetrating chart, regular across
            // r = 2M — the `_ks` shift-advection-flux + KS-densitized/wavespeed kernels. baked for
            // 1D radial (the michel / bondi accretion targets) and the 2D (r, theta) plane; 3D
            // spherical has no baked kernel and is rejected by the fail-loud guard above, never
            // silently run on a flat metric. the spacetime is orthogonal to the regime.
            //
            // in 2D the generator row length picks the momentum DOF: (rho, v_r, v_theta, pre) is
            // the axisymmetric in-plane flow (DOF = 2), while (rho, v_r, v_theta, v_phi, pre) lifts
            // the azimuthal momentum onto the grid (DOF = 3, the `_sph_swirl` kernels — rotating
            // flows: tori, spinning-hole accretion).
            (1, "spherical") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 1, 1,
                SchwarzschildKS { mass: $cfg.schwarzschild_mass }, SchwarzschildKS<f64>
            ),
            // spinning kerr (ingoing kerr-schild coords): the frame-dragging gamma_{r phi}
            // needs the azimuthal momentum DOF, which `primitive_row_dof` requires of the rows
            (2, "spherical") if $cfg.spacetime == "kerr_ks" => {
                build_and_run_hydro!(
                    $cfg, $prims, $regime, $regime_ty, 2, 3,
                    KerrKS { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin }, KerrKS<f64>
                )
            }
            (2, "spherical") if $cfg.spacetime == "schwarzschild_ks" => {
                if primitive_row_dof($cfg, $prims.first().map_or(0, |row| row.len()))? == 3 {
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
            // GR (kerr-schild) cylindrical 2D: the plane selector splits the two charts. the (R, phi)
            // equatorial disk (planar_cylindrical) is diagonal (z = 0, r = R), DOF = 2 (v_R, v_phi);
            // the (R, z) 2.5D axisymmetric-swirl (the default) lifts v_phi, DOF = 3, requiring the
            // 5-tuple. the disk takes SchwarzschildKSCylindrical (evaluated at z = 0), the meridional
            // plane SchwarzschildKSCylindricalRz (evaluated at (R, 0, z)).
            (2, "cylindrical") if $cfg.spacetime == "schwarzschild_ks" => match $cfg.cyl_plane {
                symbi_sim::state::CylPlane::RPhi => build_and_run_hydro!(
                    $cfg, $prims, $regime, $regime_ty, 2, 2,
                    SchwarzschildKSCylindrical { mass: $cfg.schwarzschild_mass },
                    SchwarzschildKSCylindrical<f64>
                ),
                symbi_sim::state::CylPlane::Rz => {
                    build_and_run_hydro!(
                        $cfg, $prims, $regime, $regime_ty, 2, 3,
                        SchwarzschildKSCylindricalRz { mass: $cfg.schwarzschild_mass },
                        SchwarzschildKSCylindricalRz<f64>
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
            // GR (kerr-schild) cylindrical full 3D (R, phi, z): DOF == NDIM = 3.
            (3, "cylindrical") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 3, 3,
                SchwarzschildKSCylindrical { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCylindrical<f64>
            ),
            // spinning kerr on the full 3D cylindrical chart: the rank-1 metric with the
            // frame dragging in l_phi, shift on every axis. DOF == NDIM = 3; the 2.5D
            // (R, z) swirl at spin needs the dragging-consistent azimuthal reconstruction
            // and stays guard-rejected.
            (3, "cylindrical") if $cfg.spacetime == "kerr_ks" => build_and_run_hydro!(
                $cfg, $prims, $regime, $regime_ty, 3, 3,
                KerrKSCylindrical { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin },
                KerrKSCylindrical<f64>
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
/// extended +1 on `k`). the curved-spacetime IC seeds the true cell B so the covariant
/// conserved state carries the magnetic terms exactly (the flat path's zero-seed +
/// bcell-from-bface heal applies a euclidean energy patch that is wrong under a metric, and
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

/// slice a global per-axis staggered face buffer into the face buffer for one tile, in the
/// axis-0-fastest order `seed_faces_indexed` consumes. `global` is the python `staggered_bfields`
/// generator for axis `d`: axis-0-fastest over the global interior face domain (cell dims `n`
/// extended +1 on `d`). the tile owns `m` cells per axis; its axis-`d` face
/// domain is `m` extended +1 on `d`, offset `tile_lo` cells into the global grid. the shared internal face (a tile's hi-`d` face == its
/// neighbor's lo-`d` face) maps to the same global index in both tiles, so both seed it
/// identically -- the CT normal-face consistency the decomposition requires, by construction.
fn tile_face_buffer<const D: usize>(
    global: &[f64],
    n: [usize; D],
    m: [usize; D],
    tile_lo: [usize; D],
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
            gi += (tile_lo[ax] + lc[ax]) * stride;
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

/// build one monomorphized adiabatic MHD sim (Rmhd or NewtonianMhd; both are
/// MhdPrim + an energy-bearing closure, DOF=3) and drive it. cell state comes from `prim_gen`
/// (rho, vx, vy, vz, p — no cell B); the staggered face B from `staggered_bfields`:
/// in-grid axes `0..D` seed the CT faces (the divergence-free truth; cell B is the
/// bcell-from-bface kernel's job), transverse axes `D..3` seed cell-centered B.
macro_rules! build_and_run_mhd {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<$regime_ty, $d, 3, $geom_ty, EosSelect<f64>>;

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

        let sim = Sim::build($regime, host_eos(cfg), $geom)
            .cells(n)
            .origin(origin)
            .spacing(spacing)
            .ghosts(ghost_width(cfg))
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
                // gridded components seed the true cell B (the face average) so the conserved
                // state carries every magnetic term from step zero. seeding a zero cell B instead
                // drops the relativistic momentum's B^2 v - (v.B) B block, and the stage-1
                // bcell_from_bface energy heal recovers it only at v = 0 (and euclidean-only).
                let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                    if k < $d {
                        face_avg_cell_b::<$d>(&bufs[k], k, idx, n)
                    } else {
                        bufs[k][lin]
                    }
                });
                MhdPrim::new(
                    Prim::adiabatic(
                        Density(row[0]),
                        Tensor::new([row[1], row[2], row[3]]),
                        Pressure(row[4]),
                    ),
                    Tensor::new(mag_arr),
                )
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        // attach immersed bodies (gravity / accretion sinks) when the config declares any;
        // body-free runs keep the original sim untouched. a refined run attaches them to the
        // hierarchy instead: the finest level owns the sinks, the coarser levels carry
        // gravity-only proxies.
        let sim = if cfg.bodies.is_empty() || cfg.refinement_enabled {
            sim
        } else {
            attach_bodies_and_shapes!(sim, cfg, $d)
        };
        let sim = {
            let mut sim = sim;
            if cfg.n_tracers != 0 {
                attach_configured_tracers(&mut sim, cfg)?;
            }
            // unconditional: the census list is empty on a run that registered none, so this
            // costs nothing there, and a run that did register one must carry it. attaching
            // only under a flag is how the registration reached preflight and stopped.
            attach_configured_censuses(&mut sim, cfg)?;
            sim
        };
        // the numerics every level carries: the base level with the excision, each fine level of
        // a refined run without it, the CT method and the driven boundaries on every level
        let settings = KernelSettings::of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC);
        let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root)?;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            mhd_kernels(s.substrate(), &settings, LevelRole::Fine).expect("fine-level kernel set")
        });
        // the sinks on the finest level, gravity-only proxies below; a sink whose operator
        // support (accretion sphere, drain mask, slip shell, stencil reach) leaves the finest
        // level is a configuration fault reported here.
        if !cfg.bodies.is_empty() && cfg.refinement_enabled {
            hier = hier.try_with_bodies(build_bodies_and_horizon::<$d>(cfg))?;
        }
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// build one monomorphized isothermal MHD sim (IsothermalMhd + Isothermal eos,
/// DOF=3) and drive it. the iso primitive has no pressure slot (IsoModel zst), so
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
                    if k < $d {
                        face_avg_cell_b::<$d>(&bufs[k], k, idx, n)
                    } else {
                        bufs[k][lin]
                    }
                });
                MhdPrimG::<f64, 3, IsoModel>::new(
                    PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])),
                    Tensor::new(mag_arr),
                )
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        // a locally isothermal run carries a fifth per-cell component, the initial pressure
        // p(x): cs^2(x) = p(x) / rho(x) is derived once and held fixed, the closure field the
        // isothermal kernels read; its ghosts are continued from the interior. a globally
        // isothermal run keeps the constructor's uniform cs^2.
        if cfg.locally_isothermal {
            if prims.first().is_some_and(|row| row.len() < 5) {
                return Err(format!(
                    "locally isothermal MHD prim row has {} components, expected 5 (rho, v1..v3, \
                     p_local)",
                    prims[0].len()
                ));
            }
            let pre_ic = Field::<f64, $d, _>::zeros(&sim.geom.allocated)
                .map_err(|e| format!("pre_ic alloc: {e:?}"))?;
            let interior = sim.geom.interior.clone();
            let mut coord: [isize; $d] = std::array::from_fn(|ax| interior.spaces[ax].lo);
            for lin in 0..total {
                pre_ic.view_mut().set(coord, prims[lin][4]);
                for ax in 0..$d {
                    coord[ax] += 1;
                    if coord[ax] < interior.spaces[ax].hi {
                        break;
                    }
                    coord[ax] = interior.spaces[ax].lo;
                }
            }
            sim.set_isothermal_cs2_from_pressure(&pre_ic);
        }

        // attach immersed bodies (gravity / accretion sinks) when the config declares any;
        // body-free runs keep the original sim untouched. a refined run attaches them to the
        // hierarchy instead: the finest level owns the sinks, the coarser levels carry
        // gravity-only proxies.
        let sim = if cfg.bodies.is_empty() || cfg.refinement_enabled {
            sim
        } else {
            attach_bodies_and_shapes!(sim, cfg, $d)
        };
        let sim = {
            let mut sim = sim;
            if cfg.n_tracers != 0 {
                attach_configured_tracers(&mut sim, cfg)?;
            }
            // unconditional: the census list is empty on a run that registered none, so this
            // costs nothing there, and a run that did register one must carry it. attaching
            // only under a flag is how the registration reached preflight and stopped.
            attach_configured_censuses(&mut sim, cfg)?;
            sim
        };
        // the numerics every level carries, the CT method and the driven boundaries included
        let settings = KernelSettings::of(cfg, <IsothermalMhd as Regime<f64, $d>>::SPEC);
        let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root)?;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            mhd_kernels(s.substrate(), &settings, LevelRole::Fine).expect("fine-level kernel set")
        });
        // the sinks on the finest level, gravity-only proxies below; a sink whose operator
        // support leaves the finest level is a configuration fault reported here.
        if !cfg.bodies.is_empty() && cfg.refinement_enabled {
            hier = hier.try_with_bodies(build_bodies_and_horizon::<$d>(cfg))?;
        }
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// the multi-gpu (gpus>1) adiabatic MHD path: the MHD analog of `build_and_run_hydro_decomposed`.
/// decompose the domain into `cfg.n_gpus` tiles, bind each to a device, and evolve them in lockstep
/// with the staggered-CT halo exchange (`decomp::evolve_decomposed`, which holds
/// `decomposed == monolithic` to round-off with div(B) exact). cell state seeds `MhdPrim` from the
/// global prim rows; the staggered face B seeds each tile from its slice of the global
/// `staggered_bfields` (`tile_face_buffer`), so the shared internal face is identical in both
/// neighbors by construction. output gathers cell fields + cell B (`gather_interiors`) and the
/// staggered faces (`gather_faces`) into one global sim written by the existing checkpoint path.
/// single-level only: refinement with gpus>1 is refused; immersed bodies attach per tile at
/// their global positions.
macro_rules! build_and_run_mhd_decomposed {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::unflatten;
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<$regime_ty, $d, 3, $geom_ty, EosSelect<f64>>;

        if cfg.refinement_enabled {
            return build_and_run_magnetized_decomposed_refined!(
                cfg, prims, bufs, $regime, $regime_ty, $d, $geom, $geom_ty, host_eos(cfg), EosSelect<f64>,
                |row: &Vec<f64>, mag: [f64; 3]| MhdPrim::new(
                    Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])),
                    Tensor::new(mag),
                )
            );
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if bufs.len() < 3 {
            return Err(format!("mhd needs 3 staggered b-field generators, got {}", bufs.len()));
        }
        let partition = tile_partition(n, cfg)?;
        let counts = partition.counts();
        let ntiles: usize = counts.iter().product();
        let settings = KernelSettings::of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC);
        let whole = whole_grid::<$d>(cfg);

        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            // the tile's place on the whole grid: its first cell, its cells and the boundaries its
            // faces carry; every position it evaluates is the whole grid's bit for bit
            let tile = whole.place(&partition, tc);
            let (tile_lo, m) = (tile.offset, tile.cells);
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                // per-tile slice of each global face buffer (axis-0-fastest over the tile face
                // domain); the shared internal face reads the same global value in both neighbors.
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                let sim = Sim::build($regime, host_eos(cfg), $geom)
                    .placed(&whole, &tile)
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
                            lin += (tile_lo[ax] + idx[ax] as usize) * stride;
                            stride *= n[ax];
                        }
                        let row = &prims[lin];
                        let gidx: [isize; $d] =
                            std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                        let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        MhdPrim::new(Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])), Tensor::new(mag_arr))
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their global positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically.

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    attach_bodies_and_shapes!(sim, cfg, $d)
                };
                // the numerics, sources and driven boundaries the single grid installs, each tile
                // evaluating its sources and prescriptions at the whole grid's coordinates
                let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        // the full-size output view: gather scatters tile interiors (cells + cell B) and faces into
        // it each checkpoint; seed the faces so `bface_initialized` is set (the gather overwrites
        // the interior). lives on device 0 (touched only at output).
        let offsets = tile_offsets(&partition);
        let grid = decomposed_grid::<$d>(cfg, tiles[0].0.geom.ng);
        // the tracer population seeds once from the whole-grid density and splits across the
        // tiles by position, so the run starts from the particles a single grid would carry;
        // the whole-grid state lives for that seeding alone.
        if cfg.n_tracers > 0 {
        let mut global = Sim::build($regime, host_eos(cfg), $geom)
            .placed(&whole, &whole.single())
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
                MhdPrim::new(Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])), Tensor::new(mag_arr))
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        if cfg.n_tracers > 0 {
            let per_tile = partition_configured_tracers(&global, cfg, &partition);
            if let Some(order) = configured_ito_order(cfg)? {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.continuous_tracers = Some(
                        symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                            .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                    );
                }
                global.continuous_tracers = Some(
                    symbi_sim::tracers::ContinuousTracerSet::allocate(0, order)
                        .map_err(|detail| format!("continuous tracer output: {detail}"))?,
                );
            } else {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.tracers = Some(set);
                }
                global.tracers = Some(symbi_sim::tracers::TracerSet::default());
            }
        }
        }
        run_decomposed_loop(cfg, tiles, offsets, grid, counts, None)
    }};
}

/// the multi-gpu (gpus>1) isothermal MHD path: identical tiling/face-seeding/gather to the
/// adiabatic decomposed macro, but the iso primitive has no pressure slot (`IsoModel` zst) and the
/// eos closure is `p = cs^2 rho`. single-level only.
macro_rules! build_and_run_imhd_decomposed {
    ($cfg:expr, $prims:expr, $bufs:expr, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::unflatten;
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<IsothermalMhd, $d, 3, $geom_ty, Isothermal<f64>>;

        if cfg.refinement_enabled {
            // a locally isothermal closure on refined tiles needs each fine level's closure from
            // the whole patch, which a tile's clipped patch does not hold
            if cfg.locally_isothermal {
                return Err(
                    "a locally isothermal closure with mesh refinement and gpus > 1 awaits the whole-patch \
                     closure on each tile; run it on one device"
                        .to_string(),
                );
            }
            return build_and_run_magnetized_decomposed_refined!(
                cfg, prims, bufs, IsothermalMhd, IsothermalMhd, $d, $geom, $geom_ty, Isothermal { cs: cfg.cs }, Isothermal<f64>,
                |row: &Vec<f64>, mag: [f64; 3]| MhdPrimG::<f64, 3, IsoModel>::new(
                    PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])),
                    Tensor::new(mag),
                )
            );
        }

        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if bufs.len() < 3 {
            return Err(format!("imhd needs 3 staggered b-field generators, got {}", bufs.len()));
        }
        let partition = tile_partition(n, cfg)?;
        let counts = partition.counts();
        let ntiles: usize = counts.iter().product();
        let settings = KernelSettings::of(cfg, <IsothermalMhd as Regime<f64, $d>>::SPEC);
        let whole = whole_grid::<$d>(cfg);

        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            // the tile's place on the whole grid: its first cell, its cells and the boundaries its
            // faces carry; every position it evaluates is the whole grid's bit for bit
            let tile = whole.place(&partition, tc);
            let (tile_lo, m) = (tile.offset, tile.cells);
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                let sim = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
                    .placed(&whole, &tile)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            lin += (tile_lo[ax] + idx[ax] as usize) * stride;
                            stride *= n[ax];
                        }
                        let row = &prims[lin];
                        let gidx: [isize; $d] =
                            std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                        let mag_arr: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])), Tensor::new(mag_arr))
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their global positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically.

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    attach_bodies_and_shapes!(sim, cfg, $d)
                };
                // the numerics, sources and driven boundaries the single grid installs, each tile
                // evaluating its sources and prescriptions at the whole grid's coordinates
                let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                seed_mhd_local_cs2!(sim, cfg, prims, tile_lo, n, &whole, $d);
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        let offsets = tile_offsets(&partition);
        let grid = decomposed_grid::<$d>(cfg, tiles[0].0.geom.ng);
        // the tracer population seeds once from the whole-grid density and splits across the
        // tiles by position, so the run starts from the particles a single grid would carry;
        // the whole-grid state lives for that seeding alone.
        if cfg.n_tracers > 0 {
        let mut global = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
            .placed(&whole, &whole.single())
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
                MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])), Tensor::new(mag_arr))
            })
            .seed_faces_indexed(&bufs[0..$d])
            .build();

        if cfg.n_tracers > 0 {
            let per_tile = partition_configured_tracers(&global, cfg, &partition);
            if let Some(order) = configured_ito_order(cfg)? {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.continuous_tracers = Some(
                        symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                            .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                    );
                }
                global.continuous_tracers = Some(
                    symbi_sim::tracers::ContinuousTracerSet::allocate(0, order)
                        .map_err(|detail| format!("continuous tracer output: {detail}"))?,
                );
            } else {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.tracers = Some(set);
                }
                global.tracers = Some(symbi_sim::tracers::TracerSet::default());
            }
        }
        }
        run_decomposed_loop(cfg, tiles, offsets, grid, counts, None)
    }};
}

/// expand the (geometry x dims) arms for an adiabatic mhd regime. cartesian /
/// spherical / cylindrical across 1/2/3d (the cylindrical 2D plane is selected by
/// `cfg.cyl_plane`, threaded into every build).
macro_rules! mhd_dispatch {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            // fail-loud guard (mirroring hydro_dispatch): reject a non-minkowski spacetime with
            // no baked gr-mhd arm; silently running it on Minkowski applies the wrong metric. the matches! set
            // mirrors the baked gr-mhd arms; test_dispatch_rejects_unbaked_gr keeps them in lockstep.
            (d, c)
                if $cfg.spacetime != "minkowski" && !baked_gr_mhd(d, c, &$cfg.spacetime) =>
            {
                Err(format!(
                    "no baked GR-MHD kernel for (dims={d}, coords={c}, spacetime={}): refusing to \
                     run silently on a flat Minkowski metric. add the (dims, coords, spacetime) arm \
                     + kernel, or use spacetime=minkowski.",
                    $cfg.spacetime
                ))
            }
            // GR spherical MHD on the horizon-penetrating chart: the `_ks` GRMHD kernel row
            // (RmhdGr valencia flux with the shifted riemann fan + induction transpose,
            // metric-aware KKC c2p, and the ideal-MHD stress in the covariant source). baked 1D
            // radial (the magnetized-michel target); the inner boundary can sit below r = 2M.
            (1, "spherical") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 1,
                SchwarzschildKS { mass: $cfg.schwarzschild_mass }, SchwarzschildKS<f64>
            ),
            // the 2D (r, theta) spinning-kerr GRMHD row: the non-diagonal
            // gamma_{r phi} rides the tetrad HLLD, the radial shift the moving-interface fan, and
            // the azimuthal (swirl) momentum the frame dragging. requires the 5-tuple swirl gas rows.
            (2, "spherical") if $cfg.spacetime == "kerr_ks" => {
                build_and_run_mhd!(
                    $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                    KerrKS { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin }, KerrKS<f64>
                )
            }
            // the 2D cartesian (x, y) GRMHD row: the non-diagonal kerr-schild spatial
            // metric selects the fast-magnetosonic HLLE gas flux + the contact / UCT-HLL densitized
            // CT. the tetrad HLLD wrapper — which the kerr (r, theta) row above already rides on its
            // non-diagonal gamma_{r phi} — is unbaked for this chart, so the flux is HLLE. the
            // metric's non-diagonality is not the obstruction (the Gram-Schmidt tetrad handles
            // non-diagonal spatial metrics). the covariant geodesic + EM-stress source carries the gravity.
            (2, "cartesian") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                SchwarzschildKSCartesian { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCartesian<f64>
            ),
            // GRMHD on the full 3D cartesian kerr-schild box: no polar axis, the densitized
            // contact CT (the UCT families are unbaked at 3D GR and fail loud by name).
            (3, "cartesian") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 3,
                SchwarzschildKSCartesian { mass: $cfg.schwarzschild_mass },
                SchwarzschildKSCartesian<f64>
            ),
            // GRMHD on the spinning kerr cartesian chart (2d equatorial slice + full 3d box):
            // the rank-1 non-diagonal metric with the frame dragging in the swirl of l; the
            // densitized contact CT telescopes for any face weight, so the CT chain is the
            // same family as the a = 0 chart with the kerr metric arms.
            (2, "cartesian") if $cfg.spacetime == "kerr_ks" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                KerrKSCartesian { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin },
                KerrKSCartesian<f64>
            ),
            (3, "cartesian") if $cfg.spacetime == "kerr_ks" => build_and_run_mhd!(
                $cfg, $prims, $bufs, $regime, $regime_ty, 3,
                KerrKSCartesian { mass: $cfg.schwarzschild_mass, spin: $cfg.kerr_spin },
                KerrKSCartesian<f64>
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
            // GR (kerr-schild) cylindrical 2D GRMHD: the cyl_plane selector (threaded
            // into the geom axes by the builder) splits the two charts — the (R, z) 2.5D poloidal
            // plane (axes [0, 2], non-diagonal gamma_Rz, toroidal E_phi CT) and the (R, phi)
            // equatorial disk (axes [0, 1], diagonal on the equator, vertical E_z CT). MHD momentum
            // is a full 3-vector in both, so one metric arm serves; HLLE gas flux + contact/UCT-HLL CT.
            // the host metric follows the plane: the equatorial disk evaluates at z = 0, the
            // meridional plane at (R, 0, z)
            (2, "cylindrical") if $cfg.spacetime == "schwarzschild_ks" => match $cfg.cyl_plane {
                symbi_sim::state::CylPlane::RPhi => build_and_run_mhd!(
                    $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                    SchwarzschildKSCylindrical { mass: $cfg.schwarzschild_mass },
                    SchwarzschildKSCylindrical<f64>
                ),
                symbi_sim::state::CylPlane::Rz => build_and_run_mhd!(
                    $cfg, $prims, $bufs, $regime, $regime_ty, 2,
                    SchwarzschildKSCylindricalRz { mass: $cfg.schwarzschild_mass },
                    SchwarzschildKSCylindricalRz<f64>
                ),
            },
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
            // fail-loud guard: isothermal MHD has no baked GR kernels, so any non-minkowski
            // spacetime must fail loud; silently running it flat would drop the curvature.
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

/// the multi-gpu (gpus>1) isothermal path: the iso sibling of `build_and_run_hydro_decomposed!`.
/// builds N iso tiles + a global output sim and hands them to the shared `run_decomposed_loop`
/// (universal transport). the scope is globally isothermal (uniform cs); locally-isothermal cs(x)
/// needs per-tile cs^2 setup and is refused. same non-AMR / no-bodies / no-source scope.
macro_rules! build_and_run_iso_decomposed {
    ($cfg:expr, $prims:expr, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::unflatten;
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<IsoNewtonian, $d, $geom_ty, Isothermal<f64>>;

        if cfg.refinement_enabled {
            // a locally isothermal closure on refined tiles needs each fine level's closure from
            // the whole patch, which a tile's clipped patch does not hold
            if cfg.locally_isothermal {
                return Err(
                    "a locally isothermal closure with mesh refinement and gpus > 1 awaits the whole-patch \
                     closure on each tile; run it on one device"
                        .to_string(),
                );
            }
            let settings = KernelSettings::of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC);
            return build_and_run_hydro_decomposed_refined!(
                cfg, prims, IsoNewtonian, IsoNewtonian, $d, $geom, $geom_ty, Isothermal { cs: cfg.cs }, Isothermal<f64>,
                prim = |row| PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k]))),
                kernels = |s, _role| iso_kernels(s.substrate(), &settings).expect("tile kernel set")
            );
        }
        let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
        let total: usize = n.iter().product();
        if prims.len() != total {
            return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
        }
        if !cfg.chi_ic.is_empty() && cfg.chi_ic.len() != total {
            return Err(format!(
                "passive_scalar yielded {} values for {total} interior cells",
                cfg.chi_ic.len()
            ));
        }
        let partition = tile_partition(n, cfg)?;
        let counts = partition.counts();
        let ntiles: usize = counts.iter().product();
        let whole = whole_grid::<$d>(cfg);
        let settings = KernelSettings::of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC);

        let mut tiles: Vec<(Sim, _)> = Vec::with_capacity(ntiles);
        for flat in 0..ntiles {
            let tc = unflatten(flat, counts);
            // the tile's place on the whole grid: its first cell, its cells and the boundaries its
            // faces carry; every position it evaluates is the whole grid's bit for bit
            let tile = whole.place(&partition, tc);
            let tile_lo = tile.offset;
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                let sim = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
                    .placed(&whole, &tile)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let mut lin = 0usize;
                        let mut stride = 1usize;
                        for ax in 0..$d {
                            let g = tile_lo[ax] + idx[ax] as usize;
                            lin += g * stride;
                            stride *= n[ax];
                        }
                        let row = &prims[lin];
                        PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])))
                    })
                    .build();
                // attach the immersed bodies per tile (gravity + accretion sink). all tiles share the
                // bodies at their global positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically.

                // gravity / accretion sinks plus bonded fragments and shaped walls, attached per
                // tile at their global positions, as the energy-regime tiles attach them
                let has_any_body = !cfg.bodies.is_empty() || cfg.bonded_assembly.is_some();
                let sim = if !has_any_body {
                    sim
                } else {
                    attach_bodies_and_fragments!(sim, cfg, $d)
                };
                // the dye on this tile, indexed from the global dye by the tile's global cells;
                // the halo exchange moves prim.chi across the cuts
                let sim = with_tile_dye!(sim, cfg, tile_lo, n, flat);
                // clock + mesh motion per tile: every tile carries the identical a(t) law and
                // the decomposed loop advances them in lockstep with the shared dt.
                let mut sim = sim;
                seed_run_state(&mut sim, cfg)?;
                // the declared perturbation at the tile's own cell centers, the single grid's; with a
                // declared stationary target it is laid over the target, after the probe
                if cfg.equilibrium_json.is_none() {
                    lay_perturbation!(sim, cfg, $d, false);
                }
                let sim = sim;
                // the limiter, viscosity, sources, driven boundaries and gradient walls the single
                // grid installs, each tile evaluating them at the whole grid's coordinates
                let sub = iso_kernels(sim.substrate(), &settings).map_err(|e| format!("tile {flat}: {e}"))?;
                seed_local_cs2!(sim, sub, cfg, prims, tile_lo, n, $d);
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        // the tracer population seeds once from the whole-grid density and splits across the
        // tiles by position, so the run starts from the particles a single grid would carry;
        // the whole-grid state lives for that seeding alone.
        if cfg.n_tracers > 0 {
            let global = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
                .placed(&whole, &whole.single())
                .cfl(cfg.cfl)
                .timestepping(cfg.timestepping)
                .cyl_plane(cfg.cyl_plane)
                .allocate()
                .map_err(|e| format!("global seeding sim allocate: {e:?}"))?
                .set_initial_indexed(|idx, _x| {
                    let row = &prims[global_lin::<$d>(&[0; $d], &idx, &n)];
                    PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])))
                })
                .build();
            let per_tile = partition_configured_tracers(&global, cfg, &partition);
            if let Some(order) = configured_ito_order(cfg)? {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.continuous_tracers = Some(
                        symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                            .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                    );
                }
            } else {
                for ((tile, _), set) in tiles.iter_mut().zip(per_tile) {
                    tile.tracers = Some(set);
                }
            }
        }

        let offsets = tile_offsets(&partition);
        let grid = decomposed_grid::<$d>(cfg, tiles[0].0.geom.ng);
        let residuals = tiled_equilibrium!(tiles, cfg, counts, $d);
        // the declared perturbation over the target, as the single grid lays it after the target
        if cfg.equilibrium_json.is_some() {
            let has_energy = false;
            for (sim, _) in tiles.iter_mut() {
                lay_perturbation!(sim, cfg, $d, has_energy);
            }
        }
        run_decomposed_loop(cfg, tiles, offsets, grid, counts, residuals)
    }};
}

/// build one monomorphized isothermal hydro sim (IsoNewtonian + Isothermal eos,
/// DOF=D) and drive it. the iso primitive has no pressure slot (IsoModel zst), so
/// `prim_gen` yields (rho, v1..vD). iso is HLLE-only by physics (no contact wave),
/// so `sim.substrate()` is used directly (no solver knob).
///
/// globally isothermal: cs is the uniform scalar from `sound_speed`.
/// locally isothermal: `prim_gen` yields one extra component, the per-cell initial
/// pressure p(x); cs^2(x) = p(x)/rho(x) is derived once (compute_isothermal_cs2)
/// and held fixed — the position-dependent "temperature" the substrate flows
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
                PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])))
            })
            .build();

        // attach immersed bodies (gravity / accretion sinks + bonded fragments) when
        // declared. a refined run attaches them to the hierarchy instead (finest level
        // owns the sinks), exactly like the energy-regime hydro macro.
        let has_any_body = !cfg.bodies.is_empty() || cfg.bonded_assembly.is_some();
        let sim = if !has_any_body || cfg.refinement_enabled {
            sim
        } else {
            attach_bodies_and_fragments!(sim, cfg, $d)
        };
        // seed the passive scalar (dye): cons.chi = rho*chi and the primitive concentration over
        // the interior, as on the energy-regime path; the isothermal kernel set transports it
        let sim = if cfg.chi_ic.is_empty() {
            sim
        } else {
            if cfg.chi_ic.len() != total {
                return Err(format!(
                    "passive_scalar yielded {} values for {total} interior cells",
                    cfg.chi_ic.len()
                ));
            }
            let sim = sim
                .with_passive_scalar()
                .map_err(|e| format!("passive-scalar allocation: {e:?}"))?;
            seed_tile_dye(&sim, &cfg.chi_ic, [0; $d], n);
            sim
        };
        let sim = {
            let mut sim = sim;
            if cfg.n_tracers != 0 {
                attach_configured_tracers(&mut sim, cfg)?;
            }
            // unconditional: the census list is empty on a run that registered none, so this
            // costs nothing there, and a run that did register one must carry it. attaching
            // only under a flag is how the registration reached preflight and stopped.
            attach_configured_censuses(&mut sim, cfg)?;
            sim
        };
        // iso is HLLE-only; the substrate front door gives the kernel-set directly.
        // the limiter, viscosity, sources, driven boundaries and gradient walls every level carries
        let settings = KernelSettings::of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC);
        let sub = iso_kernels(sim.substrate(), &settings)?;

        if cfg.locally_isothermal {
            // derive cs^2(x) = p(x)/rho(x) from the per-cell initial pressure, then hold it.
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
            // the temperature field is held fixed, so its ghost values are set once here:
            // clamped zero-gradient continuation into every ghost cell. without it the
            // ghosts keep the constructor's uniform cs^2 and the ghost-pressure pass
            // books an alien temperature into every boundary flux.
            sub.extend_cs2_into_ghosts(&sim.geom.allocated, &interior);
            // the closure field the shared isothermal kernels (drains, viscosity, the godunov
            // parachute) read: the substrate's cs^2, copied once.
            if let Some(dst) = sim.fields.cs2.as_ref() {
                for c in sim.geom.allocated.iter() {
                    dst.set(c, *sub.cs2.at(c));
                }
            }
        }

        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            iso_kernels(s.substrate(), &settings).expect("fine-level kernel set")
        });
        // a refined run attaches its immersed bodies to the hierarchy: the finest
        // level owns the full (accreting) bodies, coarser levels a gravity-only
        // proxy (finest-owns-bodies, so the sink applies once).
        if !cfg.bodies.is_empty() && cfg.refinement_enabled {
            hier = hier.with_bodies(build_bodies_and_horizon::<$d>(cfg));
            hier.attach_body_shapes(build_body_shapes_and_horizon(cfg));
        }

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
                // the prolongation covers the fine interior only; extend the fixed
                // temperature into the fine ghosts (incl. coarse-fine bands) the same
                // way the base level does, else they keep the uniform constructor cs^2.
                hi[0]
                    .kernels
                    .extend_cs2_into_ghosts(&hi[0].state.geom.allocated, &region);
                if let Some(dst) = hi[0].state.fields.cs2.as_ref() {
                    for c in hi[0].state.geom.allocated.iter() {
                        dst.set(c, *hi[0].kernels.cs2.at(c));
                    }
                }
            }
        }

        // the run's stationary target, when one is declared: its discrete imbalance is measured
        // once per level and added back at every stage, so the target becomes an exact fixed point
        // rather than a state that drifts at truncation order.
        //
        // this must follow every input the target's flux is evaluated against, because the
        // imbalance is read off a real stage: the immersed bodies, whose gravity is half of the
        // balance being measured, and the per-cell cs^2(x) on the fine levels, which sets the
        // sound speed the interface flux carries.
        if let Some(ref target) = cfg.equilibrium_json {
            let declared = symbi_source_compile::EquilibriumConfig::from_json(target)
                .map_err(|e| format!("stationary target parse: {e}"))?;
            hier = hier
                .with_equilibrium_expression(&declared)
                .map_err(|e| format!("stationary target: {e:?}"))?;
            // a restart restores the checkpoint after this point, so a resumed run continues from
            // its own state and the seed applies only to a fresh start.
            if cfg.seed_from_equilibrium {
                hier.seed_equilibrium();
            }
        }

        // the declared perturbation at every level's own cell centers, the isothermal gas
        // carrying no pressure delta
        if lay_perturbation!(hier, cfg, $d, false) {
            hier.sync_perturbed();
        }

        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// expand the (geometry x dims) arms for isothermal hydro. cartesian / spherical
/// / cylindrical across 1/2/3d.
macro_rules! iso_dispatch {
    ($cfg:expr, $prims:expr) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            // fail-loud guard: isothermal hydro has no baked GR kernels, so any non-minkowski
            // spacetime must fail loud; silently running it flat would drop the curvature.
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

/// an `axis` face is admitted only on the polar or cylindrical axis of a 2D (2.5D) chart; the
/// placement rule lives beside `BoundaryType`.
fn validate_axis_boundaries(cfg: &Config) -> Result<(), String> {
    if !cfg.boundaries.contains(&BoundaryType::Axis) {
        return Ok(());
    }
    let coords = match cfg.coord_system.as_str() {
        "cartesian" => symbi_geometry::Geometry::Cartesian,
        "spherical" => symbi_geometry::Geometry::Spherical,
        "cylindrical" => symbi_geometry::Geometry::Cylindrical,
        other => return Err(format!("unknown coordinate system '{other}'")),
    };
    let x_hi: [f64; 3] =
        std::array::from_fn(|ax| cfg.x_lo[ax] + cfg.dx[ax] * cfg.n_cells[ax] as f64);
    // the mirror and the half-turn are index maps, so the mirrored axis and, on a gridded
    // azimuth, the rotated axis must be uniformly spaced for an index shift to be the angle.
    let spacing = [
        cfg.x1_spacing.as_str(),
        cfg.x2_spacing.as_str(),
        cfg.x3_spacing.as_str(),
    ];
    let bnd3 = boundaries_nd::<3>(&cfg.boundaries);
    let mut uniform_axes: Vec<usize> = (0..cfg.dims)
        .filter(|&ax| bnd3.lo(ax) == BoundaryType::Axis || bnd3.hi(ax) == BoundaryType::Axis)
        .collect();
    if cfg.dims == 3 && !uniform_axes.is_empty() {
        uniform_axes.push(if cfg.coord_system == "spherical" { 2 } else { 1 });
        if cfg.n_tracers > 0 {
            return Err("tracers do not yet cross a polar axis on a gridded azimuth; set n_tracers=0 or use reflecting".to_string());
        }
    }
    for ax in uniform_axes {
        if spacing[ax] != "linear" {
            return Err(format!(
                "an axis face needs linear spacing on axis {ax} (the mirrored axis and the rotated azimuth); got '{}'",
                spacing[ax]
            ));
        }
    }
    let violation = match cfg.dims {
        1 => symbi_sim::state::axis_boundary_violation::<1>(
            coords,
            cfg.cyl_plane,
            &[cfg.x_lo[0]],
            &[x_hi[0]],
            &[cfg.n_cells[0]],
            &boundaries_nd::<1>(&cfg.boundaries),
        ),
        2 => symbi_sim::state::axis_boundary_violation::<2>(
            coords,
            cfg.cyl_plane,
            &[cfg.x_lo[0], cfg.x_lo[1]],
            &[x_hi[0], x_hi[1]],
            &[cfg.n_cells[0], cfg.n_cells[1]],
            &boundaries_nd::<2>(&cfg.boundaries),
        ),
        _ => symbi_sim::state::axis_boundary_violation::<3>(
            coords,
            cfg.cyl_plane,
            &cfg.x_lo,
            &x_hi,
            &cfg.n_cells,
            &bnd3,
        ),
    };
    violation.map_or(Ok(()), Err)
}


/// the one-process report of a decomposed run: the setup panel, the initialization milestones,
/// the periodic progress row, checkpoint notices, and the completion or failure frame, on the
/// same table the single-grid run draws. throughput is zone-cycles per second: the interior
/// cells summed over every tile, times the iterations completed in a wall-clock window,
/// divided by that window, so it is comparable to the single-grid figure for the same grid.
/// the row's dt is the mean step over the window. one table for the whole run; no tile
/// posts on its own and nothing here synchronizes a device.
struct DecomposedReporter {
    table: Table,
    dash: Option<LiveDashboard>,
    screen: ScreenGuard,
    n_zones: u64,
    t_final: f64,
    start: std::time::Instant,
    last_inst: std::time::Instant,
    last_iter: u64,
    last_time: f64,
    io_secs: f64,
    live: bool,
}

impl DecomposedReporter {
    fn new(cfg: &Config, n_zones: u64, counts: &[usize], slabs: &[Vec<usize>]) -> Self {
        let mut setup = problem_setup_rows(cfg);
        setup.push(["Decomposition".into(), "gpus".into(), cfg.n_gpus.to_string()]);
        setup.push([
            "Decomposition".into(),
            "tile grid".into(),
            counts.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" x "),
        ]);
        for (ax, widths) in slabs.iter().enumerate() {
            if widths.len() > 1 {
                setup.push([
                    "Decomposition".into(),
                    format!("axis {ax} slabs"),
                    widths.iter().map(|w| w.to_string()).collect::<Vec<_>>().join(", "),
                ]);
            }
        }
        setup.push([
            "Decomposition".into(),
            "global zones".into(),
            (0..cfg.dims).map(|ax| cfg.n_cells[ax].to_string()).collect::<Vec<_>>().join(" x "),
        ]);
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
        table.set_subtitle(&format!(
            "{} · {} zones · {} gpus",
            cfg.regime,
            (0..cfg.dims).map(|ax| cfg.n_cells[ax] as u64).product::<u64>(),
            cfg.n_gpus
        ));
        table.set_regime(&cfg.regime.to_uppercase());
        table.set_cfl(cfg.cfl, 1.0);
        table.set_problem_setup(&setup_ref);
        table.set_header(&["Iteration", "Time", "dt", "zone-cyc/s"]);
        if let Some(p) = log_path(cfg) {
            let _ = table.set_log_file(std::path::Path::new(&p));
        }
        table.set_host(Some(symbi_display::hostinfo::HostStats::sample()));
        let screen = ScreenGuard::enter();
        let dash = LiveDashboard::spawn();
        let now = std::time::Instant::now();
        let mut me = Self {
            table,
            dash,
            screen,
            n_zones,
            t_final: cfg.t_final,
            start: now,
            last_inst: now,
            last_iter: 0,
            last_time: cfg.start_time,
            io_secs: 0.0,
            live: cfg.live_monitor,
        };
        me.milestone(&format!(
            "decomposed run: {} tiles over {} gpus, {} zones per root step",
            counts.iter().product::<usize>(),
            cfg.n_gpus,
            n_zones
        ));
        me
    }

    fn refresh(&mut self) {
        publish_or_refresh(self.dash.as_ref(), &mut self.table);
    }

    /// an initialization step completed; visible at once, so a stall shows where it stopped.
    fn milestone(&mut self, msg: &str) {
        self.table.post_info(msg);
        self.refresh();
    }

    /// the progress row every hundred iterations, rated over the window since the last row.
    fn progress(&mut self, iter: u64, time: f64) {
        if iter == 0 || iter % 100 != 0 {
            return;
        }
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_inst).as_secs_f64();
        let d_iter = iter.saturating_sub(self.last_iter);
        let rate = if elapsed > 1e-9 && d_iter > 0 {
            self.n_zones as f64 * d_iter as f64 / elapsed
        } else {
            0.0
        };
        let dt = if d_iter > 0 { (time - self.last_time) / d_iter as f64 } else { 0.0 };
        self.last_inst = now;
        self.last_iter = iter;
        self.last_time = time;
        set_row(&mut self.table, iter, time, dt, self.t_final, rate);
        self.table.push_throughput(rate);
        self.table.push_metrics(iter, time, dt, rate);
        self.refresh();
    }

    /// a checkpoint landed; the write's wall time is kept out of the sustained rate.
    fn checkpoint_written(&mut self, cfg: &Config, path: &str, time: f64, io: f64) {
        self.io_secs += io;
        self.table.post_success(&format!("checkpoint {path}  ({})", fmt_time_msg(cfg, time)));
        self.refresh();
    }

    fn finish(mut self, iter: u64, time: f64, final_path: &str) {
        let wall = self.start.elapsed().as_secs_f64();
        let compute = (wall - self.io_secs).max(1e-9);
        let avg = self.n_zones as f64 * iter as f64 / compute;
        self.screen.leave();
        self.table.set_dynamic(false);
        let summary = format!(
            "complete — {iter} steps, t = {time:.4}, {wall:.2}s ({:.2}s io), {}/s sustained · final {final_path}",
            self.io_secs,
            humanize_rate(avg),
        );
        self.table.post_success(&summary);
        self.table.exit_frame(ExitKind::Success, &summary);
        let _ = self.live;
    }

    fn fail(mut self, msg: &str) {
        self.screen.leave();
        self.table.set_dynamic(false);
        self.table.post_error(msg);
        self.table.exit_frame(ExitKind::Crash, msg);
    }
}

/// the error a failed checkpoint write raises: the target path and the writer's own detail,
/// so a run never reports success over a missing or partial file.
fn checkpoint_write_error(path: &str, err: impl std::fmt::Debug) -> String {
    format!("checkpoint write failed: {path}: {err:?}")
}


/// the mesh motion a run admits: hydro on an unrefined grid
fn validate_mesh_motion(cfg: &Config) -> Result<(), String> {
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
    Ok(())
}

/// the passive scalar a run admits: newtonian or isothermal gas on a cartesian chart, and a
/// driven face prescribing the injected fluid's dye. every driver checks it before seeding.
fn validate_passive_scalar(cfg: &Config) -> Result<(), String> {
    if cfg.chi_ic.is_empty() {
        return Ok(());
    }
    // the conserved dye is a slot on the conserved state, orthogonal to the energy slot:
    // `D_chi = rho chi` involves no energy, so newtonian and isothermal carry one alike. the
    // relativistic and MHD regimes have no dye transport wired.
    if !matches!(cfg.regime.as_str(), "newtonian" | "isothermal") {
        return Err(format!(
            "passive_scalar is wired for the newtonian and isothermal regimes; got '{}'",
            cfg.regime
        ));
    }
    if cfg.coord_system != "cartesian" {
        return Err(format!(
            "passive_scalar requires a cartesian chart; got '{}'",
            cfg.coord_system
        ));
    }
    // every grid feature carries the dye now. gpus > 1 seeds it per tile and the halo exchange
    // moves prim.chi across cuts; refinement allocates fine-level slots, prolongs the
    // concentration onto coarse-fine ghosts, restricts it back, and refluxes the materialized
    // interface dye flux alongside mass; immersed bodies drain it by the same factor as the
    // density; and a homologously expanding mesh dilutes `D_chi` exactly as it dilutes `rho`,
    // leaving the concentration invariant.
    // gradient (neumann / robin) faces carry the dye at zero normal derivative, the scalar
    // reading of the per-variable prescription the registry holds for the prim state.
    //
    // a driven face has to say more than that. it prescribes the state of fluid entering the
    // domain, and the dye of that fluid is independent of everything inside, so no local rule
    // recovers it. a prescription that stops at the prim state leaves the dye ghost band on
    // that face written by nobody, which reads downstream as clean inflow rather than as a
    // missing boundary condition. require the extra output instead.
    for (id, json) in cfg.driven_exprs.iter().enumerate() {
        let bcfg = symbi_source_compile::SourceConfig::from_json(json)
            .map_err(|e| format!("driven boundary {id}: expression parse: {e}"))?;
        let spec = if cfg.regime == "isothermal" {
            &symbi_hydro::regime_spec::ISO_NEWTONIAN_SPEC
        } else {
            &symbi_hydro::regime_spec::NEWTONIAN_SPEC
        };
        let want = symbi_source_compile::expr_bridge::boundary_prim_arity(spec, bcfg.dim) + 1;
        if bcfg.outputs.len() != want {
            return Err(format!(
                "driven boundary {id} prescribes {} outputs, but a run carrying a passive \
                 scalar needs {want}: the full prim state plus a trailing dye concentration \
                 for the injected fluid",
                bcfg.outputs.len(),
            ));
        }
    }
    Ok(())
}

/// the stack a run executes on: the dispatch onto a run's monomorphized driver holds every
/// driver's state in one frame, about 12 MB optimized and over 100 MB unoptimized, past the 8 MiB
/// a python main thread is commonly given. the reservation is virtual; a run commits the pages its
/// frames touch
const RUN_STACK_BYTES: usize = 512 * 1024 * 1024;

/// run `f` on a thread of its own with `RUN_STACK_BYTES` of stack, the caller blocked until it
/// returns; a panic on that thread resumes on the caller. the projection ledger the run booked on
/// its thread becomes the caller's, where `projection_ledger_report` reads it
fn on_run_stack<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    let (value, (attempted, accepted)) = std::thread::scope(|scope| {
        let run = std::thread::Builder::new()
            .name("simbi-run".into())
            .stack_size(RUN_STACK_BYTES)
            .spawn_scoped(scope, || (f(), symbi_sim::projection_ledger::ledger_report()))
            .expect("the run thread starts");
        match run.join() {
            Ok(done) => done,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    });
    symbi_sim::projection_ledger::ledger_adopt(attempted, accepted);
    value
}

#[cfg(test)]
mod run_stack_tests {
    use super::on_run_stack;
    use symbi_sim::projection_ledger::{self, ProjectionReceipt};

    /// the projection ledger a run books on its own thread is the one `projection_ledger_report`
    /// reads on the caller once the run returns
    #[test]
    fn the_caller_reads_the_projection_ledger_the_run_booked() {
        projection_ledger::ledger_reset();
        on_run_stack(|| {
            let scope = projection_ledger::open_scope();
            let mut receipt = ProjectionReceipt::empty();
            receipt.fired = true;
            receipt.projected_cells = 3;
            receipt.min_theta = 0.25;
            projection_ledger::record(&receipt, 1.0);
            projection_ledger::step_commit();
            drop(scope);
        });
        let (attempted, accepted) = projection_ledger::ledger_report();
        assert_eq!((attempted.passes_fired, attempted.projected_cells), (1, 3));
        assert_eq!((accepted.passes_fired, accepted.projected_cells, accepted.min_theta), (1, 3, 0.25));
    }

    /// a panic on the run thread resumes on the caller with its own payload
    #[test]
    fn a_panic_on_the_run_thread_resumes_on_the_caller() {
        let caught = std::panic::catch_unwind(|| on_run_stack(|| panic!("the run failed")));
        let payload = caught.expect_err("the panic reached the caller");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"the run failed"));
    }
}

/// runtime dispatch on the config tags -> a monomorphized sim. hydro regimes
/// (newtonian/rhd/isothermal) x cartesian (+ curvilinear for adiabatic) x 1/2/3d;
/// the mhd regimes (rmhd/nmhd/imhd) x cartesian x 1/2/3d.
fn dispatch_and_run(
    cfg: &Config,
    prims: &[Vec<f64>],
    bfields: &[Vec<f64>],
) -> Result<symbi_sim::run_diagnostics::RunDiagnostics, String> {
    validate_run(cfg)?;
    validate_declared_target(cfg, validate_primitive_rows(cfg, prims)?)?;
    if cfg.n_gpus > 1 {
        validate_tiled_march(cfg)?;
        validate_tiled_balance(cfg)?;
    }
    match cfg.regime.as_str() {
        "newtonian" => hydro_dispatch!(cfg, prims, Newtonian, Newtonian),
        "rhd" => hydro_dispatch!(cfg, prims, Rhd, Rhd),
        "isothermal" => iso_dispatch!(cfg, prims),
        "rmhd" => mhd_dispatch!(cfg, prims, bfields, Rmhd, Rmhd),
        "nmhd" => mhd_dispatch!(cfg, prims, bfields, NewtonianMhd, NewtonianMhd),
        "imhd" => imhd_dispatch!(cfg, prims, bfields),
        other => Err(format!("regime '{other}' not wired yet")),
    }
}

/// the momentum components a configuration's primitive rows carry, read from the row width: rho,
/// the velocities, then the pressure on an energy regime. an isothermal row may end with one
/// p/cs entry, required as the closure pressure when the closure is local. the velocity count is
/// the grid dimension except on MHD (three everywhere) and on the kerr-schild 2D charts that lift
/// the azimuthal velocity: the (r, theta) plane takes either form, spinning kerr and the (R, z)
/// plane require the lift. any other width is refused by name
fn primitive_row_dof(cfg: &Config, width: usize) -> Result<usize, String> {
    let d = cfg.dims;
    let accepted: Vec<(usize, usize)> = match cfg.regime.as_str() {
        "nmhd" | "rmhd" => vec![(5, 3)],
        "imhd" if cfg.locally_isothermal => vec![(5, 3)],
        "imhd" => vec![(4, 3), (5, 3)],
        "isothermal" if cfg.locally_isothermal => vec![(d + 2, d)],
        "isothermal" => vec![(d + 1, d), (d + 2, d)],
        _ => match (d, cfg.coord_system.as_str(), cfg.spacetime.as_str()) {
            (2, "spherical", "schwarzschild_ks") => vec![(4, 2), (5, 3)],
            (2, "spherical", "kerr_ks") => vec![(5, 3)],
            (2, "cylindrical", "schwarzschild_ks") if cfg.cyl_plane == symbi_sim::state::CylPlane::Rz => vec![(5, 3)],
            _ => vec![(d + 2, d)],
        },
    };
    accepted.iter().find(|(w, _)| *w == width).map(|&(_, dof)| dof).ok_or_else(|| {
        let widths: Vec<String> = accepted.iter().map(|(w, _)| w.to_string()).collect();
        format!(
            "a primitive row of {width} entries on the {} regime on the {}: the dispatched kernels \
             read rows of {} entries (rho, the velocities{})",
            cfg.regime,
            grid_name(cfg),
            widths.join(" or "),
            if matches!(cfg.regime.as_str(), "isothermal" | "imhd") { ", then an optional p/cs" } else { ", then the pressure" }
        )
    })
}

/// every primitive row has one width the configuration's layout accepts
fn validate_primitive_rows(cfg: &Config, prims: &[Vec<f64>]) -> Result<usize, String> {
    let width = prims.first().map_or(0, |row| row.len());
    if let Some((ii, row)) = prims.iter().enumerate().find(|(_, row)| row.len() != width) {
        return Err(format!("primitive row {ii} has {} entries where row 0 has {width}", row.len()));
    }
    primitive_row_dof(cfg, width)
}

/// a declared stationary target supplies density, one velocity per grid axis and the pressure;
/// a swirl run (`dof` velocities on a grid of fewer axes) holds a state no declared target
/// describes, so every driver refuses the pair before it builds anything
fn validate_declared_target(cfg: &Config, dof: usize) -> Result<(), String> {
    if cfg.equilibrium_json.is_some() && dof != cfg.dims {
        return Err(format!(
            "a declared stationary target on a swirl run ({dof} velocity components on the {}-dimensional {}) \
             is outside every driver: the target supplies one velocity per grid axis",
            cfg.dims, cfg.coord_system
        ));
    }
    Ok(())
}

/// the semantic admission every driver applies before it builds anything: a configured setting
/// either reaches the kernels the run dispatches or is refused here by name. the single grid, the
/// host tiles and the launched workers all call it, so a capability decision lives in one place
fn validate_run(cfg: &Config) -> Result<(), String> {
    validate_porous_body_overlaps(&cfg.bodies)?;
    // a user source acts on a magnetized run's hydro slots on the single grid; the refined
    // magnetized march carries none, on any driver
    if matches!(cfg.regime.as_str(), "nmhd" | "imhd" | "rmhd") && cfg.refinement_enabled && !cfg.source_jsons.is_empty() {
        return Err("user source expressions are not yet supported with mesh refinement on a magnetized regime".to_string());
    }
    // a declared periodic face wraps the grid on one device and across tiles and workers alike
    wrap_capability(cfg.boundaries.contains(&BoundaryType::Periodic), cfg.wb_reconstruction, !cfg.bodies.is_empty())?;
    validate_axis_boundaries(cfg)?;
    validate_magnetic_slip_bodies(
        &cfg.bodies,
        &cfg.regime,
        cfg.dims,
        &cfg.coord_system,
        cfg.locally_isothermal,
    )?;
    validate_magnetized_drain_bodies(&cfg.bodies, &cfg.regime, cfg.dims, &cfg.coord_system)?;
    check_horizon_shell_clearance(cfg)?;
    // static mesh refinement is wired for hydro (incl. globally-isothermal). the two cases
    // refused below need fine-level prolongation the transfer set does not carry:
    if cfg.refinement_enabled
        && cfg.regime.contains("mhd")
        && !((cfg.dims == 3 || cfg.dims == 2) && cfg.coord_system == "cartesian")
    {
        return Err("mhd refinement requires a 3d or 2.5d cartesian grid (the CT \
                    reflux assumes 1/dx curl coefficients)"
            .to_string());
    }
    validate_mesh_motion(cfg)?;
    // immersed bodies + refinement: the finest-owns-bodies AMR sync (`hier.with_bodies` — full
    // bodies on the finest level, gravity-only proxy on coarser) is wired for every hydro regime
    // and, for the staggered field, for adiabatic Newtonian MHD and isothermal MHD on a 3D or 2.5D
    // cartesian grid with draining bodies (transparent, resistive, or magnetic-slip coupling). the finest level
    // owns the drain, the slip, and the receipts; the coarser levels carry gravity-only
    // proxies; the coupled root step is a transaction over every level. a rigid or shaped body
    // on a refined staggered grid is refused.
    if !cfg.bodies.is_empty() && cfg.refinement_enabled && cfg.regime.contains("mhd") {
        validate_refined_mhd_bodies(&cfg.bodies, &cfg.regime, cfg.dims, &cfg.coord_system, cfg.locally_isothermal)?;
    }
    // a curved spacetime is a relativistic construct: only the relativistic regimes compose
    // with it (the non-relativistic kernel rows are never baked with a spacetime slug).
    if cfg.spacetime != "minkowski" && !matches!(cfg.regime.as_str(), "rhd" | "rmhd") {
        return Err(format!(
            "spacetime '{}' requires a relativistic regime (rhd or rmhd); got '{}'",
            cfg.spacetime, cfg.regime
        ));
    }
    // bonded fragments ride the uni-grid cartesian newtonian/isothermal paths:
    // the per-fragment support-box dispatch plus the host bonded subcycle.
    // every other combination has no fragment step yet and must fail loud —
    // a silently frozen cluster would read as a valid static wall array.
    if let Some(asm) = &cfg.bonded_assembly {
        let n = asm.positions.len();
        if !matches!(cfg.regime.as_str(), "newtonian" | "isothermal") {
            return Err(format!(
                "bonded_assembly ({n} fragments) is wired for newtonian/isothermal hydro; \
                 got regime '{}'",
                cfg.regime
            ));
        }
        if cfg.coord_system != "cartesian" {
            return Err(format!(
                "bonded_assembly ({n} fragments) requires a cartesian chart (per-fragment \
                 support boxes); got '{}'",
                cfg.coord_system
            ));
        }
        if cfg.refinement_enabled {
            return Err(format!(
                "bonded_assembly ({n} fragments) does not support refinement yet (no \
                 fragment step on the hierarchy)"
            ));
        }
        // gpus > 1 is wired: the decomposed body step sums each fragment's per-tile fluid load and
        // runs the bonded dem subcycle on the total, replicated across tiles
        // so the assembly evolves identically. refinement with fragments stays refused above.
        if cfg.dims < 2 {
            return Err(format!(
                "bonded_assembly ({n} fragments) needs a 2d or 3d grid (bond torques are \
                 degenerate in 1d)"
            ));
        }
        if cfg.mesh_motion {
            return Err(format!(
                "bonded_assembly ({n} fragments) is not wired with mesh motion"
            ));
        }
    }
    // the passive scalar: the chi kernels are baked cartesian, and every unwired combination
    // fails loud, since a silently undyed or wrongly dyed run reads as valid science
    validate_passive_scalar(cfg)?;
    // mass-transport tracers consume the accepted finite-volume density flux.
    // curvilinear decomposition and mesh motion still require explicit
    // material-volume geometry in the decomposed driver.
    // non-ideal transport. the baked matrix is narrower than the config surface, so an
    // unsupported chart would otherwise surface mid-run as an unbaked-kernel panic rather than a
    // config-time refusal — a worse failure than the loud startup rejection every other capability
    // gets. the accepted sets below mirror `gen_viscous` / `gen_resistive` in symbi-aot/build.rs.
    // scoped to the regimes whose kernel sets actually dispatch a viscous operator. the
    // relativistic regimes carry none, so a viscosity or `viscosity_alpha` configured on one is
    // refused by name; a bare `alpha` there is a problem's own quantity (a wave amplitude, say)
    // and is left alone.
    let viscous_regime = matches!(
        cfg.regime.as_str(),
        "newtonian" | "isothermal" | "nmhd" | "imhd"
    );
    if !viscous_regime && (cfg.viscosity != 0.0 || cfg.alpha != 0.0) {
        return Err(format!(
            "viscous transport is not built for the {} regime: the configured viscosity = {} and \
             viscosity_alpha = {} would have no effect; set both to 0",
            cfg.regime, cfg.viscosity, cfg.alpha
        ));
    }
    alpha_key_verdict(&cfg.regime, cfg.bare_alpha_key, cfg.alpha)?;
    // the alpha law reads the local sound speed from the pressure, which isothermal MHD carries
    // no field for; isothermal hydro reads the constant cs
    if cfg.regime == "imhd" && cfg.alpha != 0.0 {
        return Err(format!(
            "alpha-disk viscosity on the imhd regime: the configured viscosity_alpha = {} reads the              local sound speed from a pressure isothermal MHD does not carry; use a constant              viscosity",
            cfg.alpha
        ));
    }
    if viscous_regime && (cfg.viscosity > 0.0 || cfg.alpha > 0.0) {
        // the shear operator is a face-centered stencil over at least two grid axes; there is no
        // 1d instance (a single-axis shear has no transverse gradient to diffuse).
        if cfg.dims < 2 {
            return Err(format!(
                "viscosity needs a 2d or 3d grid (the shear stencil has no transverse axis in 1d); \
                 got dims = {}",
                cfg.dims
            ));
        }
        if !matches!(
            cfg.coord_system.as_str(),
            "cartesian" | "cylindrical" | "axis_cylindrical" | "planar_cylindrical" | "spherical"
        ) {
            return Err(format!(
                "viscosity is baked for the cartesian, cylindrical, and spherical charts; got \
                 coord_system = '{}'",
                cfg.coord_system
            ));
        }
        // the alpha law reads Omega_K from the central body's mass, so a body must exist to
        // define it. `alpha` without one silently divides by a mass that was never set.
        if cfg.alpha > 0.0 && cfg.bodies.is_empty() {
            return Err(
                "alpha-disk viscosity reads the keplerian frequency from immersed body 0; the \
                 config declares no bodies"
                    .to_string(),
            );
        }
    }
    if cfg.resistivity > 0.0 {
        if !cfg.regime.contains("mhd") {
            return Err(format!(
                "resistivity is an MHD operator (the eta*J edge EMF); got regime '{}'",
                cfg.regime
            ));
        }
        if cfg.dims < 2 {
            return Err(format!(
                "resistivity needs a 2d or 3d grid (the edge EMF curl has no transverse axis in \
                 1d); got dims = {}",
                cfg.dims
            ));
        }
        if !matches!(
            cfg.coord_system.as_str(),
            "cartesian" | "cylindrical" | "axis_cylindrical" | "planar_cylindrical" | "spherical"
        ) {
            return Err(format!(
                "resistivity is baked for the cartesian, cylindrical, and spherical charts; got \
                 coord_system = '{}'",
                cfg.coord_system
            ));
        }
    }
    if cfg.n_tracers > 0 {
        if !matches!(cfg.tracer_scheme.as_str(), "discrete" | "ito2" | "ito3") {
            return Err(format!(
                "tracer_scheme must be one of discrete, ito2, or ito3; got '{}'",
                cfg.tracer_scheme
            ));
        }
        if !matches!(
            cfg.regime.as_str(),
            "newtonian" | "rhd" | "isothermal" | "nmhd" | "rmhd" | "imhd"
        ) {
            return Err(format!(
                "n_tracers = {} is wired for hydro and mhd; got '{}'",
                cfg.n_tracers, cfg.regime
            ));
        }
        if cfg.coord_system != "cartesian" || cfg.spacetime != "minkowski" {
            return Err(format!(
                "n_tracers = {} requires a flat cartesian chart until curvilinear decomposed \
                 transport geometry is wired; got ({}, {})",
                cfg.n_tracers, cfg.coord_system, cfg.spacetime
            ));
        }
        if !cfg.cohort_ic.is_empty() && cfg.refinement_enabled {
            return Err(
                "tracer_cohort seeding on a refined hierarchy requires a composite-level \
                 cohort field and is not wired yet"
                    .to_string(),
            );
        }
    }
    if !cfg.cohort_ic.is_empty() && cfg.n_tracers == 0 {
        return Err("tracer_cohort requires n_tracers > 0".to_string());
    }
    // ppm evolution reconstruction: baked for the flat cartesian newtonian ideal-gas family
    if cfg.reconstruction_name == "ppm" {
        if cfg.regime != "newtonian" {
            return Err(format!(
                "ppm reconstruction is wired for the newtonian (adiabatic ideal-gas) \
                 regime only; got '{}'",
                cfg.regime
            ));
        }
        if cfg.coord_system != "cartesian" || cfg.spacetime != "minkowski" {
            return Err(format!(
                "ppm reconstruction requires a flat cartesian chart; got ({}, {})",
                cfg.coord_system, cfg.spacetime
            ));
        }
    }
    // the synge (taub-mathews) eos: baked for the flat rhd family only
    if cfg.eos_name == "synge" {
        if cfg.regime != "rhd" {
            return Err(format!(
                "eos = 'synge' (taub-mathews) is a relativistic closure and is wired \
                 for the rhd regime only; got '{}'",
                cfg.regime
            ));
        }
        if cfg.spacetime != "minkowski" {
            return Err(format!(
                "eos = 'synge' requires a flat (minkowski) spacetime; got '{}'",
                cfg.spacetime
            ));
        }
    }
    // the stationary target declares the gas alone; a passive scalar has no target concentration
    // for its interface flux to difference against
    if cfg.equilibrium_json.is_some() && !cfg.chi_ic.is_empty() {
        return Err(
            "the passive scalar carries no declared target concentration, so its interface flux \
             has nothing to difference against; drop the passive scalar or the stationary target"
                .to_string(),
        );
    }
    // a neumann or robin wall maps to a skipped ghost fill on the magnetized substrates, which
    // register no gradient pass: the prescribed gradient would never be applied
    if cfg.regime.contains("mhd") && !cfg.gradient_bcs.is_empty() {
        return Err(format!(
            "a neumann or robin wall with the {} regime: the magnetized substrates fill no \
             gradient ghost; use outflow, reflect, periodic or a driven face",
            cfg.regime
        ));
    }
    // the balanced reconstruction is installed on the newtonian substrate alone
    if cfg.wb_reconstruction && cfg.regime != "newtonian" {
        return Err(format!(
            "wb_reconstruction with the {} regime: the balanced reconstruction is built for the \
             newtonian regime only; unset it",
            cfg.regime
        ));
    }
    // the isothermal hydro substrate evaluates the hlle flux
    if cfg.regime == "isothermal" && cfg.solver != Solver::Hlle {
        return Err(format!(
            "solver = '{}' with the isothermal regime: isothermal hydro evaluates the hlle flux; \
             set solver = hlle",
            format!("{:?}", cfg.solver).to_lowercase()
        ));
    }
    Ok(())
}

/// the admission of every tiled march, in-process host tiles and launched workers alike: the
/// operators the single grid applies that no tiled driver carries yet are refused by name
fn validate_tiled_march(cfg: &Config) -> Result<(), String> {
    // the gas and isothermal tiles, flat and refined, read a stationary target's residual off the
    // tiled pipeline and lay a declared perturbation over it; the magnetized tiles carry none
    if cfg.equilibrium_json.is_some() && !matches!(cfg.regime.as_str(), "newtonian" | "rhd" | "isothermal") {
        return Err(format!(
            "an equilibrium prescription with several tiles awaits its tiled residual on the {} \
             regime; run it on one device",
            cfg.regime
        ));
    }
    // the tiled marches step no magnetic slip body
    if cfg.bodies.iter().any(|b| b.magnetic_slip.is_some()) {
        return Err("a magnetic slip body with several tiles awaits its tiled slip step; run it on one device".to_string());
    }
    // the tiles wrap a periodic axis cut into them on the root and on a first patch spanning that
    // axis; a nested level wraps within its own tile, so a nested patch keeps clear of the
    // periodic faces of a cut axis
    if cfg.refinement_enabled && cfg.refinement_regions.len() > 1 {
        let reached = match cfg.dims {
            1 => nested_patch_on_periodic_face::<1>(cfg, true)?,
            2 => nested_patch_on_periodic_face::<2>(cfg, true)?,
            _ => nested_patch_on_periodic_face::<3>(cfg, true)?,
        };
        if let Some(ax) = reached {
            return Err(format!(
                "a nested refined patch reaching the periodic face of axis {ax}, which is cut into tiles, \
                 awaits nested levels that wrap across the tiles; keep the nested regions clear of that \
                 face or run on one device"
            ));
        }
    }
    // a census reduces over the whole grid; the tiled marches carry no cross-tile census
    if !cfg.census_jsons.is_empty() {
        return Err(
            "a census with several tiles awaits a cross-tile census reduction; run on one device or \
             drop the census"
                .to_string(),
        );
    }
    Ok(())
}

/// author the metadata the frozen v2.0 reader requires beyond what
/// `write_checkpoint` derives from SimState (it already writes gamma, cfl, time,
/// dt, iteration, dimensions, halo_radius, regime, coord_system, timestepping,
/// is_mhd, is_relativistic). these are the spatial-scheme + run-control fields,
/// authored from the python config dict (the single source of truth).
fn checkpoint_metadata(cfg: &Config, checkpoint_index: u64) -> Metadata {
    let boundary_conditions = cfg
        .boundaries
        .iter()
        .map(|boundary| match boundary {
            BoundaryType::Periodic => "periodic",
            BoundaryType::Outflow => "outflow",
            BoundaryType::Reflect => "reflecting",
            BoundaryType::Axis => "axis",
            BoundaryType::CoarseFine => "coarse_fine",
            BoundaryType::Driven(_) => "dynamic",
            BoundaryType::Neumann(_) => "neumann",
            BoundaryType::Robin(_) => "robin",
        })
        .collect::<Vec<_>>()
        .join(",");
    Metadata::new()
        .with("build_git_sha", BUILD_GIT_SHA)
        .with("build_git_dirty", BUILD_GIT_DIRTY)
        .with("build_source_id", build_source_id())
        .with("config_source", cfg.config_source.as_str())
        .with("config_sha256", cfg.config_sha256.as_str())
        .with("solver", cfg.solver_name.as_str())
        .with("reconstruction", cfg.reconstruction_name.as_str())
        .with("eos", cfg.eos_name.as_str())
        .with("ppm_flatten_onset", cfg.ppm_flatten_onset)
        .with("ppm_flatten_full", cfg.ppm_flatten_full)
        .with("plm_theta", cfg.plm_theta)
        .with("viscosity", cfg.viscosity)
        .with("tend", cfg.t_final)
        .with("dlogt", cfg.dlogt)
        .with("checkpoint_index", checkpoint_index)
        .with("checkpoint_interval", cfg.checkpoint_interval)
        .with("x1_spacing", cfg.x1_spacing.as_str())
        .with("x1_spacing_ratio", cfg.x1_spacing_ratio)
        .with("x2_spacing", cfg.x2_spacing.as_str())
        .with("x2_spacing_ratio", cfg.x2_spacing_ratio)
        .with("x3_spacing", cfg.x3_spacing.as_str())
        .with("x3_spacing_ratio", cfg.x3_spacing_ratio)
        .with("boundary_conditions", boundary_conditions)
        // the reconstruction-balance discriminator. `solver = hllc_lm` changed meaning when
        // the clamped variant was retired (2026-08-15): a checkpoint recording the solver
        // name without this attribute is clamp-era by construction, and the python restart
        // guard refuses to continue it under the new numerics. every new file must record
        // it, else the guard would refuse its own series on first resume.
        .with("wb_reconstruction", cfg.wb_reconstruction)
        .with("initial_time", cfg.start_time)
        .with("time_unit", cfg.time_unit)
        .with("time_unit_label", cfg.time_unit_label.as_str())
        // the stationary target the scheme is well-balanced against, verbatim. it is not a field,
        // so nothing else in the file records it, and a run resumed against a different target
        // integrates different equations while looking identical from the outside. an empty string
        // records the ordinary case of no declared target.
        .with(
            "equilibrium_target",
            cfg.equilibrium_json.as_deref().unwrap_or(""),
        )
        // the edge EMF of a magnetized run: a scheme choice with no field of its own, so a run
        // resumed under the other scheme would continue the file's faces under a different
        // induction update. an empty string records a run without a staggered field.
        .with("ct_method", ct_method_name(cfg))
}

/// the recorded edge-EMF name of a magnetized run, empty for a run without a staggered field
fn ct_method_name(cfg: &Config) -> &'static str {
    if !cfg.regime.contains("mhd") {
        return "";
    }
    match cfg.ct_method {
        CtMethod::Contact => "contact",
        CtMethod::Uct => "uct",
    }
}

/// the integer-digit width for the time portion of a checkpoint name, sized
/// from `t_final / time_unit` so every file in a run shares the same width and
/// a directory listing sorts chronologically. minimum 3 (the default).
fn checkpoint_time_width(cfg: &Config) -> usize {
    let t_units = (cfg.t_final / cfg.time_unit).max(1.0);
    let digits = t_units.log10().floor() as usize + 1;
    digits.max(3)
}

/// insert underscores every 3 digits from the right of a zero-padded integer string,
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
/// rendered as an underscore, fixed 3-digit fraction, and the integer part thousand-grouped
/// so large times stay readable: t/unit = 1.0 -> "001_000", 0.5 -> "000_500", 1234.567 ->
/// "001_234_567". the last underscore group is always the fraction; earlier groups are the
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

/// the `tnow` segment of a checkpoint name. linear runs use the human-readable time
/// (`000_790`); log-spaced runs (`idx_width > 0`) use the zero-padded monotonic index
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CheckpointOutcome {
    Completed,
    Interrupted,
    Crashed,
}

/// the canonical checkpoint status is mutually exclusive: `final` certifies
/// completion, while interrupted and crashed states remain distinguishable.
fn checkpoint_status_tag(outcome: CheckpointOutcome) -> &'static str {
    match outcome {
        CheckpointOutcome::Completed => "final",
        CheckpointOutcome::Interrupted => "interrupted",
        CheckpointOutcome::Crashed => "crashed",
    }
}

#[cfg(test)]
mod checkpoint_status_tests {
    use super::{CheckpointOutcome, checkpoint_status_tag};

    #[test]
    fn final_is_reserved_for_successful_completion() {
        assert_eq!(checkpoint_status_tag(CheckpointOutcome::Completed), "final");
        assert_eq!(
            checkpoint_status_tag(CheckpointOutcome::Interrupted),
            "interrupted"
        );
        assert_eq!(checkpoint_status_tag(CheckpointOutcome::Crashed), "crashed");
    }
}

/// the full checkpoint path: `<dir><zones>.chkpt.<tnow>[.<unit>].h5`. `tnow` is
/// either a formatted time or a status word (final / interrupted / crashed). the unit
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
/// all become valid path components). the raw label is still used verbatim in
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

// validate a multi-gpu request (`Config.n_gpus`) before any heavy work. an unsatisfiable
// request is rejected with the precise reason -- a cpu build, or too few visible devices
// without the oversubscribe opt-in -- so the request never silently degrades to one device.
/// the outer horizon r_+ containment gate for a GR accretion run. a well-posed
/// accretion-rate certificate needs the innermost flux surface to be causally one-way;
/// the domain's inner radius must sit on the correct side of the horizon for the chart:
///   schwarzschild  singular at r_+ = 2M (lapse -> 0)  -> r_min > r_+   (stay outside)
///   schwarzschild_ks  horizon-penetrating, r_+ = 2M      -> r_min < r_+   (swallow it)
///   kerr_ks           horizon-penetrating, r_+ = M + sqrt(M^2 - a^2)  -> r_min < r_+
/// the gate is a radial (spherical/cylindrical) condition; the cartesian equatorial
/// slice places the origin inside the box, a containment condition on the box bounds,
/// so it is left to those bounds. flat minkowski has no horizon.
fn check_horizon_containment(
    spacetime: &str,
    mass: f64,
    spin: f64,
    coord_system: &str,
    r_min: f64,
) -> Result<(), String> {
    if spacetime == "minkowski" {
        return Ok(());
    }
    if mass <= 0.0 {
        return Err(format!(
            "GR spacetime '{spacetime}' requires a positive mass; got M = {mass}"
        ));
    }
    let r_plus = match spacetime {
        "schwarzschild" | "schwarzschild_ks" => 2.0 * mass,
        "kerr_ks" => {
            if spin.abs() > mass {
                return Err(format!(
                    "kerr spin |a| = {} exceeds mass M = {mass}: no horizon (a naked \
                     singularity), the accretion certificate is undefined",
                    spin.abs()
                ));
            }
            mass + (mass * mass - spin * spin).sqrt()
        }
        other => return Err(format!("unknown GR spacetime '{other}'")),
    };
    // the radial-coordinate check applies to spherical charts only: coordinate
    // slot 0 is the cylindrical R on a cylindrical chart (the spherical radius
    // depends on the z bounds too) and a box coordinate on cartesian. and only
    // the singular schwarzschild chart forbids a radius — a kerr-schild patch
    // entirely outside the horizon is a legitimate chart choice (the metric is
    // regular everywhere); the excision-request gate separately enforces the
    // swallow-the-horizon geometry where excision demands it.
    if coord_system == "spherical" && spacetime == "schwarzschild" && r_min <= r_plus {
        return Err(format!(
            "schwarzschild inner radius r_min = {r_min} <= r_+ = 2M = {r_plus}: the \
             metric is singular at and inside the horizon (lapse alpha = sqrt(1 - 2M/r) \
             is imaginary). use r_min > 2M, or the horizon-penetrating schwarzschild_ks chart."
        ));
    }
    Ok(())
}

/// the excision-request gate: a positive excision radius is only meaningful on a
/// baked combination — the 2d or 3d cartesian kerr-schild chart with the sphere strictly
/// inside the horizon r_+ = 2M (excising exterior gas would delete causally connected
/// flow) and strictly above the metric-guard radius M/2 (below it the metric is frozen
/// and the fill would read constant-metric cells as if they were physical). refinement
/// and multi-gpu paths do not carry the excision pass.
fn check_excision_request(
    excision_radius: f64,
    spacetime: &str,
    coord_system: &str,
    dims: usize,
    mass: f64,
    spin: f64,
    refinement_enabled: bool,
    refinement_regions: &[Vec<f64>],
    n_gpus: usize,
) -> Result<(), String> {
    if excision_radius <= 0.0 {
        return Ok(());
    }
    // every relativistic regime excises: MHD fills the gas state only (a vacuum floor
    // on rho/v/p + a magnetized conserved rebuild reading the cell B); the staggered
    // faces stay CT-owned, so div(sqrt(gamma) B) is untouched. non-relativistic
    // regimes never reach here (a curved spacetime on a newtonian/iso regime is
    // rejected upstream).
    // the admissible charts. 3d cartesian: the excised region is the kerr-schild-radius level set,
    // staircased across the lattice. 1d radial and 2d (r, theta) spherical: r is a coordinate, so
    // the excision surface is a coordinate surface — no staircase, and the region is an exact slab
    // of innermost cells.
    //
    // a 2d cartesian slice is excluded for a reason that does not carry over: it is
    // z-translation-invariant, a black string rather than a point black hole, so its spherical
    // metric evaluated at z = 0 is inconsistent with the planar dynamics, and its staircased
    // excision circle seeds an m = 4 mode that grows into the exterior. the spherical reductions
    // are genuine symmetry reductions (spherical symmetry in 1d, axisymmetry in 2d), not slices of
    // a 3d problem, and their excision surface carries no grid imprint at all.
    let chart_ok = match coord_system {
        "cartesian" => dims == 3,
        "spherical" => dims == 1 || dims == 2,
        _ => false,
    };
    if !matches!(spacetime, "schwarzschild_ks" | "kerr_ks") || !chart_ok {
        return Err(format!(
            "excision_radius = {excision_radius} requires a horizon-penetrating kerr-schild chart \
             (spacetime schwarzschild_ks or kerr_ks) on a 3d cartesian box or a 1d/2d spherical \
             grid; got (dims={dims}, coords={coord_system}, spacetime={spacetime}). a 2d cartesian \
             slice is z-translation-invariant — a black string, not a point black hole: its \
             spherical metric evaluated at z = 0 is inconsistent with the planar dynamics, and the \
             excision circle is grid-staircased (it seeds an m = 4 mode that grows into the \
             exterior)."
        ));
    }
    // the excision surface is the kerr-schild-radius level set r_ks = r_exc, which
    // must sit strictly inside the outer horizon: r_+ = 2M at a = 0,
    // r_+ = M + sqrt(M^2 - a^2) at spin (|a| <= M is validated by the horizon gate).
    let r_plus = if spacetime == "kerr_ks" {
        mass + (mass * mass - spin * spin).max(0.0).sqrt()
    } else {
        2.0 * mass
    };
    if excision_radius >= r_plus {
        return Err(format!(
            "excision_radius = {excision_radius} >= r_+ = {r_plus}: the excision surface \
             must sit strictly inside the horizon (excised cells are causally disconnected \
             ONLY there)."
        ));
    }
    if excision_radius <= 0.5 * mass {
        return Err(format!(
            "excision_radius = {excision_radius} <= M/2 = {}: the metric is clamped constant \
             below M/2, so the excision surface must sit above it (recommended ~0.7 r_+ = {}).",
            0.5 * mass,
            0.7 * r_plus
        ));
    }
    // refinement + decomposition + excision compose: the decomposed driver runs the root excise
    // (`level_tail_excise`) in the same position the uni-grid tail does, and the overlap check below
    // — which applies to every gpu count — keeps fine patches off the excised surface, which is the
    // condition that makes a root-only excise correct in the first place.
    let _ = n_gpus;
    if refinement_enabled {
        // the excise pass runs on the root level only; a fine patch overlapping the
        // excised region would evolve its copy of those cells and restrict them back
        // over the fill. the excised spheroid spans +-sqrt(r^2 + a^2) equatorially
        // and +-r on the spin axis; reject any refinement region whose box intersects
        // it, with one root cell of margin so a box that merely abuts the surface is
        // rejected too.
        let semi_xy = (excision_radius * excision_radius + spin * spin).sqrt();
        for region in refinement_regions {
            if region.len() < 2 * dims {
                continue;
            }
            let mut overlaps = true;
            for ax in 0..dims {
                let ext = if dims == 3 && ax == 2 {
                    excision_radius
                } else {
                    semi_xy
                };
                let (lo, hi) = (region[2 * ax], region[2 * ax + 1]);
                if hi < -ext || lo > ext {
                    overlaps = false;
                    break;
                }
            }
            if overlaps {
                return Err(format!(
                    "refinement region {region:?} overlaps the excised region \
                     (equatorial extent {semi_xy:.3}, polar {excision_radius:.3}); the \
                     excise pass runs on the root level only, so a fine patch there \
                     would restrict un-excised values back over the fill. move the \
                     refinement region off the horizon."
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod excision_gate_tests {
    use super::check_excision_request;

    #[test]
    fn zero_radius_is_always_fine() {
        assert!(
            check_excision_request(0.0, "minkowski", "spherical", 1, 0.0, 0.0, true, &[], 4)
                .is_ok()
        );
    }

    #[test]
    fn excision_needs_a_3d_cartesian_ks_chart() {
        // a 2d cartesian slice is a z-translation-invariant black string, not a point hole;
        // excision demands a genuine 3d cartesian box (or a spherical / cylindrical chart
        // that hides the horizon behind r_min).
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_ok()
        );
        assert!(
            check_excision_request(1.4, "kerr_ks", "cartesian", 3, 1.0, 0.9, false, &[], 1).is_ok()
        );
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "cartesian",
                2,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_err()
        );
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild",
                "cartesian",
                3,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_err()
        );
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "spherical",
                3,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_err()
        );
        assert!(
            check_excision_request(1.4, "kerr_ks", "spherical", 3, 1.0, 0.9, false, &[], 1)
                .is_err()
        );
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "cartesian",
                1,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_err()
        );
    }

    #[test]
    fn excision_surface_must_sit_between_the_guard_and_the_horizon() {
        // a = 0, M = 1: the valid band is (M/2, 2M) = (0.5, 2.0).
        assert!(
            check_excision_request(
                2.0,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_err()
        );
        assert!(
            check_excision_request(
                0.5,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_err()
        );
        assert!(
            check_excision_request(
                0.6,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_ok()
        );
        assert!(
            check_excision_request(
                1.9,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                false,
                &[],
                1
            )
            .is_ok()
        );
    }

    #[test]
    fn spinning_horizon_shrinks_the_band() {
        // a = 0.9, M = 1: r_+ = 1 + sqrt(1 - 0.81) ~ 1.436.
        assert!(
            check_excision_request(1.9, "kerr_ks", "cartesian", 3, 1.0, 0.9, false, &[], 1)
                .is_err()
        );
        assert!(
            check_excision_request(1.4, "kerr_ks", "cartesian", 3, 1.0, 0.9, false, &[], 1).is_ok()
        );
        assert!(
            check_excision_request(0.5, "kerr_ks", "cartesian", 3, 1.0, 0.9, false, &[], 1)
                .is_err()
        );
    }

    #[test]
    fn decomposition_refinement_and_their_combination_are_all_allowed() {
        // the excise pass is owned by whichever level contains the excised region, and the decomposed
        // driver runs the root excise in the same position the uni-grid tail does, so decomposition
        // and refinement compose. the condition that makes a root-only excise correct is that no fine
        // patch overlaps the excised surface, which the overlap check enforces at every gpu count.
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                false,
                &[],
                2
            )
            .is_ok()
        );
        let far = vec![vec![5.0, 8.0, 5.0, 8.0, 5.0, 8.0]];
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                true,
                &far,
                1
            )
            .is_ok()
        );
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                true,
                &far,
                2
            )
            .is_ok()
        );
    }

    #[test]
    fn refinement_region_on_the_horizon_is_rejected() {
        // a central fine box overlapping the excised spheroid would restrict
        // un-excised values back over the root-level fill.
        let central = vec![vec![-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]];
        assert!(
            check_excision_request(
                1.4,
                "schwarzschild_ks",
                "cartesian",
                3,
                1.0,
                0.0,
                true,
                &central,
                1
            )
            .is_err()
        );
        // at spin the equatorial extent widens to sqrt(r^2 + a^2): a box clear of
        // r_exc on x but inside the widened extent still overlaps.
        let graze = vec![vec![1.5, 3.0, -0.5, 0.5, -0.5, 0.5]];
        assert!(
            check_excision_request(1.4, "kerr_ks", "cartesian", 3, 1.0, 0.9, true, &graze, 1)
                .is_err()
        );
        let clear = vec![vec![2.0, 4.0, 2.0, 4.0, -0.5, 0.5]];
        assert!(
            check_excision_request(1.4, "kerr_ks", "cartesian", 3, 1.0, 0.9, true, &clear, 1)
                .is_ok()
        );
    }

    #[test]
    fn the_horizon_overlap_check_governs_at_every_gpu_count() {
        // the protection that matters is the patch/spheroid overlap test, not the gpu count: a fine
        // patch on the horizon would evolve its own copy of the excised cells and restrict them back
        // over the root fill, and that is equally wrong on one device or many. so the same region is
        // accepted or refused identically however the grid is split.
        let far = vec![vec![5.0, 8.0, 5.0, 8.0, 5.0, 8.0]];
        let central = vec![vec![-2.0, 2.0, -2.0, 2.0, -2.0, 2.0]];
        for gpus in [1usize, 2, 4] {
            assert!(
                check_excision_request(
                    1.4,
                    "schwarzschild_ks",
                    "cartesian",
                    3,
                    1.0,
                    0.0,
                    true,
                    &far,
                    gpus
                )
                .is_ok(),
                "a far patch was refused at gpus = {gpus}"
            );
            assert!(
                check_excision_request(
                    1.4,
                    "schwarzschild_ks",
                    "cartesian",
                    3,
                    1.0,
                    0.0,
                    true,
                    &central,
                    gpus
                )
                .is_err(),
                "a patch ON the excised spheroid was accepted at gpus = {gpus}"
            );
        }
    }
}

#[cfg(test)]
mod horizon_gate_tests {
    use super::check_horizon_containment;

    #[test]
    fn minkowski_has_no_horizon_to_gate() {
        assert!(check_horizon_containment("minkowski", 0.0, 0.0, "spherical", 0.5).is_ok());
    }

    #[test]
    fn schwarzschild_must_stay_outside_the_horizon() {
        // the standard-chart configs use r_min = 3 > 2M = 2.
        assert!(check_horizon_containment("schwarzschild", 1.0, 0.0, "spherical", 3.0).is_ok());
        // at or inside r_+ = 2M the lapse is imaginary.
        assert!(check_horizon_containment("schwarzschild", 1.0, 0.0, "spherical", 2.0).is_err());
        assert!(check_horizon_containment("schwarzschild", 1.0, 0.0, "spherical", 1.5).is_err());
    }

    #[test]
    fn schwarzschild_ks_allows_any_regular_patch() {
        // the horizon-penetrating chart is regular everywhere: inside-horizon
        // inner boundaries (the accretion configs) and entirely-outside patches
        // are both legitimate. only excision demands the swallow geometry, and
        // its own request gate enforces that.
        assert!(check_horizon_containment("schwarzschild_ks", 1.0, 0.0, "spherical", 1.5).is_ok());
        assert!(check_horizon_containment("schwarzschild_ks", 1.0, 0.0, "spherical", 3.0).is_ok());
        // the cylindrical chart's slot 0 is a cylindrical R, so the spherical-radius gate is exempt.
        assert!(
            check_horizon_containment("schwarzschild_ks", 1.0, 0.0, "cylindrical", 4.0).is_ok()
        );
    }

    #[test]
    fn kerr_rejects_only_the_naked_singularity() {
        assert!(check_horizon_containment("kerr_ks", 1.0, 0.6, "spherical", 1.7).is_ok());
        // |a| > M is a naked singularity: no horizon at all.
        assert!(check_horizon_containment("kerr_ks", 1.0, 1.5, "spherical", 0.5).is_err());
    }

    #[test]
    fn cartesian_slice_is_gated_by_box_bounds_not_r_min() {
        // the origin sits inside the box; r_min (a corner distance) does not gate it.
        assert!(check_horizon_containment("schwarzschild_ks", 1.0, 0.0, "cartesian", 5.0).is_ok());
    }

    #[test]
    fn gr_chart_requires_positive_mass() {
        assert!(check_horizon_containment("schwarzschild_ks", 0.0, 0.0, "spherical", 1.0).is_err());
        assert!(check_horizon_containment("schwarzschild", -1.0, 0.0, "spherical", 3.0).is_err());
    }
}

fn validate_gpu_request(n_gpus: usize) -> Result<(), String> {
    if n_gpus <= 1 {
        return Ok(());
    }
    #[cfg(not(feature = "gpu"))]
    {
        // the same switch that folds logical devices onto too few physical ones runs the
        // decomposed path as host tiles here: the whole build + scatter + exchange + gather +
        // checkpoint path on one cpu, so a decomposed run can be diffed against the single grid
        // without a device. no parallelism.
        if std::env::var("SYMBI_GPU_OVERSUBSCRIBE").is_ok() {
            eprintln!(
                "warning: gpus={n_gpus} on a cpu build; running {n_gpus} host tiles \
                 (SYMBI_GPU_OVERSUBSCRIBE) -- correctness check only, no speedup."
            );
            return Ok(());
        }
        Err(format!(
            "gpus={n_gpus} requested, but this is a cpu build. multi-gpu needs a gpu build: \
             `./dev.py install --gpu` (nvidia) or `--hip` (amd)."
        ))
    }
    #[cfg(feature = "gpu")]
    {
        let avail = symbi::symbi_xpu::device_count().unwrap_or(0) as usize;
        if n_gpus > avail {
            // oversubscribe escape hatch: fold N logical devices onto the
            // available physical ones (distinct contexts via the modulo map in cuda.rs/hip.rs).
            // no real parallelism, but it lets the whole decomposed path (build + scatter +
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
        // the decomposed run loop supports hydro; per-regime support is
        // enforced in `dispatch_and_run` so non-hydro regimes error and never silently fall
        // back to one device.
        Ok(())
    }
}

// =============================================================================
// run-diagnostics transport
//
// frozen, read-only carriers that move one run's owned diagnostics across the
// python boundary by named getters. private-facing (`_Native*`): the scientific
// api is the python `RunResult` dataclass tree the frontend builds from these.
// =============================================================================

/// a count of cells and the subset inside a configured horizon.
#[pyclass(frozen, name = "_NativeCellCount")]
struct NativeCellCount {
    inner: symbi_sim::guard_ledger::CellCount,
}

#[pymethods]
impl NativeCellCount {
    #[getter]
    fn total(&self) -> u64 {
        self.inner.total
    }
    #[getter]
    fn inside_horizon(&self) -> u64 {
        self.inner.inside_horizon
    }
}

/// the FOFC guard acts of a run: troubled and frozen cell counts.
#[pyclass(frozen, name = "_NativeGuardDiagnostics")]
struct NativeGuardDiagnostics {
    inner: symbi_sim::guard_ledger::GuardTotals,
}

#[pymethods]
impl NativeGuardDiagnostics {
    #[getter]
    fn troubled_cells(&self) -> NativeCellCount {
        NativeCellCount {
            inner: self.inner.troubled_cells,
        }
    }
    #[getter]
    fn frozen_cells(&self) -> NativeCellCount {
        NativeCellCount {
            inner: self.inner.frozen_cells,
        }
    }
}

/// the admissible-boundary projection's accepted intervention totals.
#[pyclass(frozen, name = "_NativeProjectionDiagnostics")]
struct NativeProjectionDiagnostics {
    inner: symbi_sim::projection_ledger::ProjectionLedger,
}

#[pymethods]
impl NativeProjectionDiagnostics {
    #[getter]
    fn passes_fired(&self) -> u64 {
        self.inner.passes_fired
    }
    #[getter]
    fn projected_cells(&self) -> u64 {
        self.inner.projected_cells
    }
    #[getter]
    fn min_theta(&self) -> f64 {
        self.inner.min_theta
    }
    /// the signed direct projection contribution to the accepted conserved mass.
    #[getter]
    fn injected_den_signed(&self) -> f64 {
        self.inner.injected_den.signed
    }
    /// the gross (L1) projection activity in the accepted conserved mass.
    #[getter]
    fn injected_den_gross(&self) -> f64 {
        self.inner.injected_den.abs
    }
    /// the signed direct projection contribution to the accepted conserved energy.
    #[getter]
    fn injected_nrg_signed(&self) -> f64 {
        self.inner.injected_nrg.signed
    }
    /// the gross (L1) projection activity in the accepted conserved energy.
    #[getter]
    fn injected_nrg_gross(&self) -> f64 {
        self.inner.injected_nrg.abs
    }
}

/// one run's accepted diagnostics: the projection and guard evidence of the
/// states that survived into the solution.
#[pyclass(frozen, name = "_NativeRunDiagnostics")]
struct NativeRunDiagnostics {
    inner: symbi_sim::run_diagnostics::RunDiagnostics,
}

#[pymethods]
impl NativeRunDiagnostics {
    #[getter]
    fn projection(&self) -> NativeProjectionDiagnostics {
        NativeProjectionDiagnostics {
            inner: self.inner.projection,
        }
    }
    #[getter]
    fn guards(&self) -> NativeGuardDiagnostics {
        NativeGuardDiagnostics {
            inner: self.inner.guards,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (
    prim_gen,
    staggered_bfields,
    sim_info,
    a,
    adot,
    chi_field=None,
    cohort_field=None
))]
fn run_simulation(
    py: Python<'_>,
    prim_gen: &Bound<'_, PyAny>,
    staggered_bfields: &Bound<'_, PyAny>,
    sim_info: &Bound<'_, PyDict>,
    a: &Bound<'_, PyAny>,
    adot: &Bound<'_, PyAny>,
    chi_field: Option<&Bound<'_, PyAny>>,
    cohort_field: Option<&Bound<'_, PyAny>>,
) -> PyResult<NativeRunDiagnostics> {
    let mut cfg = parse_config(sim_info)?;
    validate_config_preflight(&cfg).map_err(PyValueError::new_err)?;
    eprintln!(
        "SIMBI provenance: backend={} config={} config_sha256={}",
        build_source_id(),
        cfg.config_source,
        cfg.config_sha256
    );
    // drain the passive-scalar generator (None = undyed). one flat cell-centered
    // buffer; the per-arm seeding checks the count against the interior.
    cfg.chi_ic = drain_dye(chi_field)?;
    cfg.cohort_ic = drain_cohort(cohort_field)?;
    validate_config_preflight(&cfg).map_err(PyValueError::new_err)?;
    // a non-linear (log) radial axis is supported under decomposition: each tile carries the global
    // per-cell slope and an origin advanced multiplicatively to its first global cell
    // (`start * 10^(g * slope)`), so a tile's local index i lands exactly where the undecomposed grid
    // puts g + i, and refinement regions are clipped against the tile's map rather than a linear
    // extent. gated by tile_coord_tests.
    // evaluate the scale-factor callables at the start time (gil held). the rust
    // mesh-motion model integrates `a` from the constant rate `a_dot` (linear /
    // free homologous expansion, a_ddot = 0), so a single sample at t0 suffices.
    sample_scale_factor(&mut cfg, Some(a), Some(adot))?;
    let prims = drain_prims(prim_gen)?;
    if cfg.excision_radius > 0.0 {
        (cfg.excision_rho_scale, cfg.excision_pre_scale) =
            initial_excision_scales(&prims).map_err(PyValueError::new_err)?;
    }
    let bfields = drain_bfields(staggered_bfields)?;

    // the solve is pure rust with no python access — release the gil so rayon
    // gets real parallelism (and python stays responsive).
    let diagnostics = py
        .detach(|| on_run_stack(|| dispatch_and_run(&cfg, &prims, &bfields)))
        .map_err(PyRuntimeError::new_err)?;
    Ok(NativeRunDiagnostics { inner: diagnostics })
}

/// whether a config's `alpha`-key spelling is unambiguous.
///
/// `alpha` used to be read as the shakura-sunyaev coefficient straight off the config dict, which
/// collides with problems that use the name for their own quantity — a wave amplitude, a slope.
/// the coefficient now comes from `viscosity_alpha`, and a leftover bare `alpha` is only a problem
/// where it could plausibly have meant viscosity: on a regime whose kernel set dispatches a
/// viscous operator, with no `viscosity_alpha` to disambiguate it. guessing there is silent either
/// way — read it as viscosity and a transport term appears uninvited, ignore it and one silently
/// vanishes — so the ambiguous case is refused and the config is asked to say which it meant.
fn alpha_key_verdict(
    regime: &str,
    bare_alpha_key: bool,
    viscosity_alpha: f64,
) -> Result<(), String> {
    let viscous_regime = matches!(regime, "newtonian" | "isothermal" | "nmhd" | "imhd");
    if viscous_regime && bare_alpha_key && viscosity_alpha == 0.0 {
        return Err(format!(
            "the config declares a bare `alpha` field on regime '{regime}', which dispatches \
             viscosity. `alpha` is no longer read as the shakura-sunyaev coefficient, because the \
             name collides with problems that use it for their own quantity. rename it to \
             `viscosity_alpha` if it IS the alpha-disk coefficient; if it means something else, \
             declare `viscosity_alpha` explicitly (0.0 for no alpha viscosity) to say so."
        ));
    }
    Ok(())
}

#[cfg(test)]
mod alpha_key_tests {
    use super::alpha_key_verdict;

    // the ambiguous case: a viscous regime carrying the old spelling and nothing to disambiguate.
    #[test]
    fn a_bare_alpha_on_a_viscous_regime_is_refused() {
        for regime in ["newtonian", "isothermal", "nmhd", "imhd"] {
            assert!(
                alpha_key_verdict(regime, true, 0.0).is_err(),
                "{regime} accepted an ambiguous bare `alpha`"
            );
        }
    }

    // the relativistic regimes dispatch no viscous operator, so `alpha` there cannot have meant
    // the coefficient. this is the case that would have refused the isentropic-wave problem,
    // whose `alpha` is a wave amplitude.
    #[test]
    fn a_bare_alpha_on_a_nonviscous_regime_is_fine() {
        for regime in ["rhd", "rmhd"] {
            assert!(
                alpha_key_verdict(regime, true, 0.0).is_ok(),
                "{regime} refused an `alpha` it could never have read as viscosity"
            );
        }
    }

    // declaring the namespaced key resolves the ambiguity, whatever the bare one meant.
    #[test]
    fn viscosity_alpha_disambiguates() {
        assert!(alpha_key_verdict("newtonian", true, 0.05).is_ok());
        assert!(alpha_key_verdict("newtonian", false, 0.05).is_ok());
        assert!(alpha_key_verdict("newtonian", false, 0.0).is_ok());
    }
}

fn validate_config_preflight(cfg: &Config) -> Result<(), String> {
    validate_gpu_request(cfg.n_gpus)?;
    validate_config_checks(cfg)
}

/// the configuration checks every entry point applies before draining the initial state; the
/// single-process entry adds its device request on top
fn validate_config_checks(cfg: &Config) -> Result<(), String> {
    // the cartesian constrained-transport curl reads one uniform width per axis; on a mapped axis
    // the staggered field would evolve on the uniform lattice while the gas moves on the mapped one
    let spacings = [&cfg.x1_spacing, &cfg.x2_spacing, &cfg.x3_spacing];
    if cfg.regime.contains("mhd")
        && cfg.coord_system == "cartesian"
        && cfg.dims >= 2
        && spacings[..cfg.dims.min(3)].iter().any(|s| !s.eq_ignore_ascii_case("linear"))
    {
        return Err(format!(
            "the {} regime on a non-uniform cartesian grid: the cartesian constrained-transport \
             curl reads uniform widths; use linear spacing on every axis",
            cfg.regime
        ));
    }
    // the axis placement rule runs here as well as at dispatch, so a misplaced or unsupported
    // axis face fails before a queue slot is spent.
    validate_axis_boundaries(cfg)?;
    check_horizon_containment(
        &cfg.spacetime,
        cfg.schwarzschild_mass,
        cfg.kerr_spin,
        &cfg.coord_system,
        cfg.x_lo[0],
    )?;
    check_excision_request(
        cfg.excision_radius,
        &cfg.spacetime,
        &cfg.coord_system,
        cfg.dims,
        cfg.schwarzschild_mass,
        cfg.kerr_spin,
        cfg.refinement_enabled,
        &cfg.refinement_regions,
        cfg.n_gpus,
    )?;
    if cfg.refinement_enabled {
        if cfg.coord_system != "cartesian" {
            return Err(format!(
                "mesh refinement is cartesian-only; got coord_system = '{}'",
                cfg.coord_system
            ));
        }
        for (axis, spacing) in [&cfg.x1_spacing, &cfg.x2_spacing, &cfg.x3_spacing]
            .into_iter()
            .enumerate()
            .take(cfg.dims)
        {
            if !spacing.eq_ignore_ascii_case("linear") {
                return Err(format!(
                    "mesh refinement requires linear cell spacing; got x{}_spacing = '{}'",
                    axis + 1,
                    spacing
                ));
            }
        }
    }
    for source in &cfg.source_jsons {
        let parsed = symbi_source_compile::SourceConfig::from_json(&source.json)
            .map_err(|err| format!("{} parse: {err}", source.origin))?;
        if parsed.kind == "rotating_frame" && cfg.coord_system != "cartesian" {
            return Err(format!(
                "{} rotating_frame requires cartesian coordinates; got '{}'",
                source.origin, cfg.coord_system,
            ));
        }
    }
    // lower every source, not just parse it. the bridge is where a source is checked
    // against the regime it will run under — a newtonian conservation law asked for on a
    // relativistic regime, a cooling term on a regime with no energy equation, an operator
    // with no carrier primitive, a reference to the cell measure. parsing alone accepts all
    // of those and defers the failure to dispatch, after a queue slot has been spent.
    // the same law the run will attach under, so the pre-flight check validates the
    // configuration that will actually execute rather than a flat stand-in.
    let preflight_spec = regime_spec_for(&cfg.regime)?;
    lower_configured_sources(
        &cfg.source_jsons,
        preflight_spec,
        Some(&state_law_of(cfg, preflight_spec)),
    )?;
    // each driven face lowers against the run's regime here, so a prescription that cannot is
    // refused by name before any driver builds a level; every level of every tile and worker
    // then registers a prescription known to lower
    for (ii, json) in cfg.driven_exprs.iter().enumerate() {
        let bcfg = symbi_source_compile::SourceConfig::from_json(json)
            .map_err(|err| format!("driven boundary {ii} parse: {err}"))?;
        symbi_source_compile::expr_bridge::build_boundary_dag(&bcfg, preflight_spec)
            .map_err(|err| format!("driven boundary {ii} lower: {err}"))?;
    }
    if let Some(json) = &cfg.motion_json {
        symbi_source_compile::motion_law::MotionLaw::from_json(json, cfg.start_time, cfg.t_final)
            .map_err(|err| format!("mesh motion parse: {err}"))?;
    }
    // compile every registered census here so an unlowerable expression, a malformed
    // binning or a duplicate name is reported before the run starts rather than at the
    // first sample, when a queue slot has already been spent.
    lower_configured_censuses(&cfg.census_jsons)?;
    Ok(())
}

#[pyfunction]
fn validate_simulation(sim_info: &Bound<'_, PyDict>) -> PyResult<()> {
    let cfg = parse_config(sim_info)?;
    validate_config_preflight(&cfg).map_err(PyValueError::new_err)
}

/// read-only live monitor: poll `<rundir>/.simbi-live/snapshot.bin` (written by a
/// run started with `live_monitor = true`) and render the dashboard until the user
/// quits or Ctrl-C. blocks on a dedicated terminal — release the gil so the
/// signal + render threads run and python stays interruptible.
#[pyfunction]
#[pyo3(signature = (rundir, poll_ms = 250))]
fn attach_dashboard(py: Python<'_>, rundir: String, poll_ms: u64) -> PyResult<()> {
    py.detach(|| symbi_display::run_attach(std::path::Path::new(&rundir), poll_ms))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// the analytic transonic bondi state at radius `r` (bondi radii, code units
/// G*M = c_inf = rho_inf = 1): `(rho, u, pre)` with `u` the inflow speed
/// magnitude (the radial velocity is `-u * rhat`). the config-side initial
/// condition for a spherical accretion run — seed the transonic profile directly.
#[pyfunction]
fn bondi_profile(r: f64, gamma: f64) -> (f64, f64, f64) {
    let s = symbi_ib::bondi_profile(r, gamma);
    (s.rho, s.u, s.pre)
}

/// the analytic bondi accretion rate 4 pi lambda_c(gamma) in code units —
/// the target the emergent drain rate is validated against.
#[pyfunction]
fn mdot_bondi(gamma: f64) -> f64 {
    symbi_ib::mdot_bondi(gamma)
}

/// the sonic radius (5 - 3 gamma)/4 in bondi radii. the well-posedness
/// constraint is r_mask < r_s; degenerate 0 at gamma = 5/3.
#[pyfunction]
fn bondi_sonic_radius(gamma: f64) -> f64 {
    symbi_ib::sonic_radius(gamma)
}

/// the guard-activation census for the run that just finished: `(fallback, freeze,
/// fallback_inside_horizon, freeze_inside_horizon)` in cell-substage events. `run_simulation`
/// zeroes these at run start, so the values are that run's totals.
///
/// the `fallback` fields count TROUBLED CELLS — cells the recovery flagged at the c2p-status
/// decode — summed over every substage, including substages later rejected and replayed. they
/// are attempted-event counts, not a per-cell count of applied first-order flux (the fallback is
/// applied per face in the splice). the run-owned `RunDiagnostics.guards` reports the accepted
/// troubled and frozen cell counts of the surviving solution instead; this census is the legacy
/// process-global attempted counter, kept for gate acceptance thresholds.
///
/// this is the one number that says whether a passing gate passed on its own merits or on a
/// limiter. it covers the whole defensive surface of a smooth GR-hydro run, not just FOFC:
/// `fofc_orchestrate` early-returns when no cell is flagged, and both the admissible-boundary
/// projection and the first-order redo run inside it, so a zero fallback count proves neither
/// acted; the relativistic velocity ceiling only binds an out-of-cone state, which is exactly what
/// sets the flag. a smooth, warm, shock-free flow has no physical business tripping any of them, so
/// its acceptance criterion is zero — "a limiter is fine as long as it is visible", made checkable.
#[pyfunction]
fn guard_census() -> (u64, u64, u64, u64) {
    let (fb, fz) = symbi::regimes::fofc::fofc_stats();
    let (fb_h, fz_h) = symbi::regimes::fofc::fofc_horizon_stats();
    (fb, fz, fb_h, fz_h)
}

/// zero the guard-activation counters. `run_simulation` already does this at run start; a caller
/// measuring across several runs resets explicitly.
#[pyfunction]
fn reset_guard_census() {
    symbi::regimes::fofc::fofc_reset_stats();
}

/// the projection ledger: the admissible-boundary projection's booked evidence for the run.
/// two flat tuples `(attempted, accepted)`, each
/// `(passes, passes_fired, projected_cells, min_theta,
/// [den_signed, den_gross, nrg_signed, nrg_gross],   # intervention (raw, weight one)
/// [den_signed, den_gross, nrg_signed, nrg_gross])`. # injected (SSP-weighted)
/// `den` is the mass slot, `nrg` the total energy slot the projection moved. intervention sums
/// every raw per-pass delta; injected scales each pass by its downstream shu-osher propagation
/// weight (euler 1; rk2 1/2, 1; rk3 1/6, 2/3, 1). the injected `signed` is the exact direct
/// projection contribution to the accepted conserved total; the `gross` is the scheme-weighted
/// L1 budget `sum w|delta|`, which bounds the absolute net from above rather than equaling it.
/// attempted books every projection pass; accepted books only whole accepted steps. the ledger
/// is evidence about production physics, not a correction — it records what the live projection
/// already did. host single-grid runs collect it; the totals persist until the next run clears
/// them. all zeros (min_theta = 1) when the projection never fired.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn projection_ledger_report() -> (
    (u64, u64, u64, f64, [f64; 4], [f64; 4]),
    (u64, u64, u64, f64, [f64; 4], [f64; 4]),
) {
    let flat = |t: &symbi_sim::projection_ledger::ProjectionLedger| {
        (
            t.passes,
            t.passes_fired,
            t.projected_cells,
            t.min_theta,
            [
                t.intervention_den.signed,
                t.intervention_den.abs,
                t.intervention_nrg.signed,
                t.intervention_nrg.abs,
            ],
            [
                t.injected_den.signed,
                t.injected_den.abs,
                t.injected_nrg.signed,
                t.injected_nrg.abs,
            ],
        )
    };
    let (attempted, accepted) = symbi_sim::projection_ledger::ledger_report();
    (flat(&attempted), flat(&accepted))
}

/// zero the projection ledger on this thread (call between runs when measuring per run).
#[pyfunction]
fn reset_projection_ledger() {
    symbi_sim::projection_ledger::ledger_reset();
}

// shared module body. the pyo3 entry-point name below decides the `PyInit_*`
// symbol and the imported module name: `cpu_ext` for the default build,
// `gpu_ext` for the cuda build. both compile the same source — cuda only adds
// the NVRTC device path — so the registration is identical and lives here.
fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(launch_worker, m)?)?;
    m.add_function(wrap_pyfunction!(config_digest, m)?)?;
    m.add_class::<NativeRunDiagnostics>()?;
    m.add_class::<NativeProjectionDiagnostics>()?;
    m.add_class::<NativeGuardDiagnostics>()?;
    m.add_class::<NativeCellCount>()?;
    m.add_function(wrap_pyfunction!(validate_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(attach_dashboard, m)?)?;
    m.add_function(wrap_pyfunction!(bondi_profile, m)?)?;
    m.add_function(wrap_pyfunction!(mdot_bondi, m)?)?;
    m.add_function(wrap_pyfunction!(bondi_sonic_radius, m)?)?;
    m.add_function(wrap_pyfunction!(guard_census, m)?)?;
    m.add_function(wrap_pyfunction!(reset_guard_census, m)?)?;
    m.add_function(wrap_pyfunction!(projection_ledger_report, m)?)?;
    m.add_function(wrap_pyfunction!(reset_projection_ledger, m)?)?;
    // Queryable before a run so submission tooling can print the identity of the
    // installed extension it is about to execute. checkpoints carry the same value.
    m.add("BUILD_GIT_SHA", BUILD_GIT_SHA)?;
    m.add("BUILD_GIT_DIRTY", BUILD_GIT_DIRTY)?;
    m.add("BUILD_SOURCE_ID", build_source_id())?;
    // the feature handshake: config keys the backend does not know are silently
    // absorbed by the `_or` defaults above, so a new python front end driving an
    // old extension would drop a declared knob without a word — a run asking for
    // one physics and silently getting another. the front end checks its
    // non-default knobs against this list before running; a knob absent here
    // refuses with "rebuild the extension" instead of running the wrong physics.
    m.add("FEATURES", vec!["eos", "reconstruction", "ppm_flatten"])?;
    afterglow::register(m)?;
    Ok(())
}

// =============================================================================
// machine-card accelerator description
// =============================================================================

/// the accelerator bound to this process, or None on a cpu build. `extent` is the
/// interior cell count per axis, which selects the launch block shape: the derivation
/// widens the contiguous (stride-1) axis to a full warp before filling the transverse
/// ones, so a thin domain and a fat one do not launch the same geometry. an explicit
/// `SYMBI_BLOCK_{1,2,3}D` in the environment overrides the derivation, and reporting
/// `block_for` rather than the fixed base keeps the card honest about what actually ran.
#[cfg(feature = "gpu")]
fn device_stats(ndim: usize, extent: &[u32]) -> Option<symbi_display::hostinfo::DeviceStats> {
    let info = symbi::symbi_xpu::device_info().ok()?;
    Some(symbi_display::hostinfo::DeviceStats {
        name: info.name,
        count: info.device_count.max(0) as usize,
        mem_total: info.total_memory_bytes,
        block: symbi::symbi_xpu::block_for(ndim, extent),
    })
}

#[cfg(not(feature = "gpu"))]
fn device_stats(_ndim: usize, _extent: &[u32]) -> Option<symbi_display::hostinfo::DeviceStats> {
    None
}

/// the launch parameters one worker of a `simbi launch` session receives.
struct WorkerLaunch {
    worker: usize,
    workers: usize,
    owner: Vec<usize>,
    rendezvous: String,
    credential: u64,
    /// the address this worker listens on and the host its peers reach it at
    bind: String,
    advertise: String,
    staging_cells: usize,
    /// the digest of the effective execution configuration every worker must share
    config_digest: u64,
}

/// a fault a validation run asks one worker to produce, from `SIMBI_FABRIC_INJECT`:
/// `reject:WORKER:STEP:STAGE` reports a stage rejection, `bad_cfl:WORKER:STEP` reports an
/// invalid timestep candidate, `exit:WORKER:STEP` ends the process after that step as a lost
/// node would, `face:WORKER:STEP` moves one of that worker's shared-face copies by a unit in
/// the last place after that step so the next audit names it. `trouble:STEP:STAGE:I:J[:K]` names a global cell and no worker: whichever worker
/// holds the cell raises its troubled flag at that stage, so the same value drives a run on
/// any partition. `reject_level:WORKER:STEP:LEVEL:SUBSTEP:STAGE` rejects a stage of a refined
/// level on that worker's tiles holding the level, the substeps counted across the root step.
/// `cap:STEP:BITS` caps the agreed timestep of that step at the f64 of those bits on every
/// worker, the clean control of a rejection. unset, every worker runs clean.
#[derive(Default)]
struct LaunchFault {
    injection: symbi_sim::worker::Injection,
    exit_after: Option<u64>,
    /// after this step, this worker's opening copy of one shared face is moved by one unit in
    /// the last place: the next interface audit must name it
    perturb_face_after: Option<u64>,
    /// a rejection of a refined level's stage on this worker's tiles: (step, level, substep, stage)
    level_rejection: Option<(u64, usize, u64, usize)>,
}

impl LaunchFault {
    fn from_env(me: usize) -> Result<Self, String> {
        let Ok(spec) = std::env::var("SIMBI_FABRIC_INJECT") else {
            return Ok(Self::default());
        };
        let parts: Vec<&str> = spec.split(':').collect();
        let number = |i: usize| -> Result<u64, String> {
            parts
                .get(i)
                .and_then(|p| p.parse::<u64>().ok())
                .ok_or_else(|| format!("SIMBI_FABRIC_INJECT='{spec}': field {i} is not a number"))
        };
        let mut fault = Self::default();
        if parts[0] == "cap" {
            // cap:STEP:BITS, on every worker alike
            fault.injection.cap_dt_at = Some((number(1)?, number(2)?));
            return Ok(fault);
        }
        if parts[0] == "trouble" {
            // trouble:STEP:STAGE:I:J[:K], the third cell index absent on a plane
            let k = if parts.len() > 5 { number(5)? as isize } else { 0 };
            fault.injection.trouble_at = Some((
                number(1)?,
                number(2)? as usize,
                [number(3)? as isize, number(4)? as isize, k],
            ));
            return Ok(fault);
        }
        if number(1)? as usize != me {
            match parts[0] {
                "reject" | "reject_level" | "bad_cfl" | "exit" | "face" => return Ok(fault),
                other => return Err(format!("SIMBI_FABRIC_INJECT: unknown fault '{other}'")),
            }
        }
        match parts[0] {
            "reject" => fault.injection.reject_at = vec![(number(2)?, number(3)? as usize)],
            "reject_level" => {
                fault.level_rejection = Some((number(2)?, number(3)? as usize, number(4)?, number(5)? as usize))
            }
            "bad_cfl" => fault.injection.invalid_cfl_at = Some(number(2)?),
            "exit" => fault.exit_after = Some(number(2)?),
            "face" => fault.perturb_face_after = Some(number(2)?),
            other => return Err(format!("SIMBI_FABRIC_INJECT: unknown fault '{other}'")),
        }
        Ok(fault)
    }
}

/// whether the operator has opted into the combinations the launched path builds without a
/// distributed gate
fn launch_experimental_opt_in() -> bool {
    std::env::var("SIMBI_LAUNCH_EXPERIMENTAL").is_ok_and(|v| v == "1")
}

/// the balanced reconstruction on several tiles, host or launched: the balanced face pair and
/// body source read the cells beside each face, which the halo exchange fills, and are gated on
/// the cartesian grids, the spherical line and (r, theta) shell and the cylindrical line; under
/// refinement, on the 2D and 3D cartesian grids, the balanced coarse-fine transfer and the seam
/// band rewrite start their chains from cells of the whole patch: the tiles carry a band's chains
/// across one cut inside it
fn validate_tiled_balance(cfg: &Config) -> Result<(), String> {
    if !cfg.wb_reconstruction {
        return Ok(());
    }
    let charted = match cfg.coord_system.as_str() {
        "cartesian" => matches!(cfg.dims, 1..=3),
        "spherical" => matches!(cfg.dims, 1 | 2),
        "cylindrical" => cfg.dims == 1,
        _ => false,
    };
    if !charted || (cfg.refinement_enabled && (cfg.coord_system != "cartesian" || cfg.dims == 1)) {
        return Err(format!(
            "wb_reconstruction on the {} with several tiles is outside the gated tiled runs (the \
             cartesian grids, the spherical line and (r, theta) shell and the cylindrical line, \
             refined on the 2D and 3D cartesian grids alone); run on one device or unset it",
            grid_name(cfg)
        ));
    }
    if cfg.refinement_enabled {
        let crossings = match cfg.dims {
            2 => cuts_in_balanced_bands::<2>(cfg)?,
            _ => cuts_in_balanced_bands::<3>(cfg)?,
        };
        // the tiles carry a band's chains across every cut inside it, hop by hop from the seam; a
        // cut on the patch face leaves the uncovered cell the chains start from on a tile beside the
        // band
        if let Some(c) = crossings.iter().find(|c| c.on_face) {
            let (level, ax, cut, face, band) = (c.level, c.axis, c.cut, c.face, c.band);
            let at = if level == 0 {
                format!("the patch face at {face}")
            } else {
                format!("the level-{} patch face at level-{level} cell {face}", level + 1)
            };
            return Err(format!(
                "wb_reconstruction with mesh refinement on several tiles: the cut at root cell {cut} on \
                 axis {ax} crosses the {band}-cell balanced restriction band inside {at}, which rewrites \
                 its covered cells from the uncovered cell beyond the face; keep every cut at least \
                 {band} cells of that level inside the face or beyond it"
            ));
        }
    }
    Ok(())
}

/// the seams whose balanced band a cut runs inside, as `(parent level, axis, high)` in the order
/// the restriction rewrites them: by level, then axis, the low side first. every launched worker
/// opens the band point of each at every restriction into its level
fn launched_band_seams<const D: usize>(cfg: &Config) -> Result<Vec<symbi_sim::phase_table::BandHops>, String> {
    if !cfg.wb_reconstruction || !cfg.refinement_enabled {
        return Ok(Vec::new());
    }
    let mut seams: Vec<symbi_sim::phase_table::BandHops> = Vec::new();
    for c in cuts_in_balanced_bands::<D>(cfg)?.iter().filter(|c| !c.on_face) {
        let at = c.cut << c.level;
        let distance = if c.high { c.face - at } else { at - c.face } as isize;
        let (level, axis) = (c.level as u8, c.axis as u8);
        match seams.iter_mut().find(|s| (s.level, s.axis, s.high) == (level, axis, c.high)) {
            Some(s) => s.distances.push(distance),
            None => seams.push(symbi_sim::phase_table::BandHops { level, axis, high: c.high, distances: vec![distance] }),
        }
    }
    for s in seams.iter_mut() {
        s.distances.sort_unstable();
        s.distances.dedup();
    }
    seams.sort_by_key(|s| (s.level, s.axis, s.high));
    Ok(seams)
}

/// a cut of the tile partition inside or on the face of a seam band of a refined patch
struct BandCrossing {
    /// the parent level of the band, the axis of the cut, the root cut, the patch face in parent
    /// cells and the band's width
    level: usize,
    axis: usize,
    cut: usize,
    face: usize,
    band: usize,
    /// the cut lies on the patch face itself
    on_face: bool,
    /// the band lies inside the patch's high face along the axis
    high: bool,
}

/// every cut of the tile partition that falls in a seam band of a refined patch: the balanced
/// restriction rewrites the parent cells within `band` cells of each seam from the uncovered
/// parent cell beyond it. `band` is the reconstruction reach plus two parent cells, capped by the
/// unrolled chain; a seam lies wherever the patch face is inside the domain. region `ii` refines
/// level `ii`, whose cells are the root cells split by 2^ii, so a root cut at `c` falls on
/// level-`ii` cell `c * 2^ii`
fn cuts_in_balanced_bands<const D: usize>(cfg: &Config) -> Result<Vec<BandCrossing>, String> {
    let reach = match cfg.reconstruction_name.as_str() {
        "pcm" => 1,
        "ppm" => 3,
        _ => 2,
    };
    let band = (reach + 2).min(symbi::WB_CF_CHAIN_MAX as usize);
    let n: [usize; D] = std::array::from_fn(|ax| cfg.n_cells[ax]);
    let partition = tile_partition(n, cfg)?;
    let counts = partition.counts();
    let regions = refinement_regions_nd::<D>(&cfg.refinement_regions)?;
    let mut crossings = Vec::new();
    for (level, region) in regions.iter().enumerate() {
        let scale = 1usize << level;
        let cell = |x: f64, ax: usize| ((x - cfg.x_lo[ax]) / cfg.dx[ax] * scale as f64).round() as usize;
        for ax in 0..D {
            let (lo, hi) = (cell(region.x_lo[ax], ax), cell(region.x_hi[ax], ax));
            for k in 1..counts[ax] {
                let mut tc = [0usize; D];
                tc[ax] = k;
                let cut = partition.tile_extents(tc)[ax].0;
                let at = cut * scale;
                if lo > 0 && (lo..lo + band).contains(&at) {
                    crossings.push(BandCrossing { level, axis: ax, cut, face: lo, band, on_face: at == lo, high: false });
                }
                if hi < n[ax] * scale && (hi.saturating_sub(band) + 1..=hi).contains(&at) {
                    crossings.push(BandCrossing { level, axis: ax, cut, face: hi, band, on_face: at == hi, high: true });
                }
            }
        }
    }
    Ok(crossings)
}

/// the launch authorization of a combination: a gated one (`ungated` is `None`) runs; an ungated
/// one is refused by name unless the operator opted in, and then runs as experimental, the
/// property returned for the announcement
/// whether immersed bodies over `nested` refined regions on magnetized tiles lie in the gated
/// scope: one gravitating draining sink with no magnetic coupling of its own on a 2.5D cartesian
/// newtonian MHD run with PLM, RK2, HLLE and the contact EMF, linear spacing, outflow walls and
/// three levels, carrying no resistive, viscous, excised, driven, balanced or user-source term
fn gated_nested_magnetized_bodies(cfg: &Config, ndim: usize, nested: usize) -> bool {
    const ACCRETION: u64 = 2;
    const RIGID: u64 = 1 << 4;
    let sink = |body: &BodyParams| {
        body.capability & ACCRETION != 0
            && body.capability & RIGID == 0
            && body.porosity.is_none()
            && body.torque_free_xi.is_none()
            && body.shape_json.is_none()
            && body.magnetic_resistivity.is_none()
            && body.magnetic_slip.is_none()
    };
    ndim == 2
        && nested == 2
        && cfg.regime == "nmhd"
        && cfg.coord_system == "cartesian"
        && cfg.reconstruction_name == "plm"
        && matches!(cfg.timestepping, Timestepping::Rk2)
        && cfg.solver_name == "hlle"
        && matches!(cfg.ct_method, CtMethod::Contact)
        && [&cfg.x1_spacing, &cfg.x2_spacing].iter().all(|spacing| spacing.as_str() == "linear")
        && cfg.boundaries.iter().all(|bc| *bc == BoundaryType::Outflow)
        && cfg.resistivity == 0.0
        && cfg.viscosity == 0.0
        && cfg.alpha == 0.0
        && cfg.excision_radius == 0.0
        && !cfg.wb_reconstruction
        && !cfg.mesh_motion
        && cfg.driven_exprs.is_empty()
        && cfg.gradient_bcs.is_empty()
        && cfg.source_jsons.is_empty()
        && cfg.binary.is_none()
        && cfg.bonded_assembly.is_none()
        && cfg.n_tracers == 0
        && cfg.bodies.len() == 1
        && sink(&cfg.bodies[0])
}

fn authorize_launch(ungated: Option<String>, opt_in: bool) -> Result<Option<String>, String> {
    match ungated {
        None => Ok(None),
        Some(property) if opt_in => Ok(Some(property)),
        Some(property) => Err(format!(
            "{property} is installed on the launched path without a distributed gate; set \
             SIMBI_LAUNCH_EXPERIMENTAL=1 to run it as an experimental, unvalidated combination"
        )),
    }
}

#[cfg(test)]
mod launch_authorization_tests {
    use super::authorize_launch;

    #[test]
    fn a_gated_combination_runs_with_or_without_the_opt_in() {
        assert_eq!(authorize_launch(None, false), Ok(None));
        assert_eq!(authorize_launch(None, true), Ok(None));
    }

    #[test]
    fn an_ungated_combination_is_refused_by_name_without_the_opt_in() {
        let err = authorize_launch(Some("the x regime on the y grid".into()), false).expect_err("an ungated combination ran");
        assert!(err.starts_with("the x regime on the y grid is installed on the launched path without a distributed gate"), "{err}");
        assert!(err.contains("SIMBI_LAUNCH_EXPERIMENTAL=1"), "{err}");
    }

    #[test]
    fn an_ungated_combination_runs_as_experimental_under_the_opt_in() {
        assert_eq!(authorize_launch(Some("the x regime on the y grid".into()), true), Ok(Some("the x regime on the y grid".into())));
    }
}

/// whether the launch of `cfg`'s regime on its grid carries a distributed gate: every gas and
/// magnetized regime on the cartesian grid, and on the spherical and cylindrical grids in one, two
/// and three dimensions (lines, the (r, z) and (r, phi) planes, shells and annuli); on a curved
/// spacetime the relativistic MHD charts are the baked GR-MHD set. a combination outside it runs
/// only under the experimental opt-in
fn gated_grid(cfg: &Config) -> bool {
    let curvilinear = matches!(cfg.coord_system.as_str(), "spherical" | "cylindrical");
    if cfg.coord_system == "cartesian" {
        return true;
    }
    match cfg.regime.as_str() {
        "newtonian" | "rhd" | "isothermal" | "nmhd" | "imhd" => curvilinear,
        // a curved spacetime's charts are the GR-MHD set the launch scope checks
        "rmhd" if cfg.spacetime != "minkowski" => baked_gr_mhd(cfg.dims, &cfg.coord_system, &cfg.spacetime),
        "rmhd" => curvilinear,
        _ => false,
    }
}

/// the grid of a configuration as a refusal names it
fn grid_name(cfg: &Config) -> String {
    let chart = match (cfg.coord_system.as_str(), cfg.dims, cfg.cyl_plane) {
        ("cylindrical", 2, CylPlane::RPhi) => "(r, phi) cylindrical".to_string(),
        ("cylindrical", 2, CylPlane::Rz) => "(r, z) cylindrical".to_string(),
        (other, _, _) => other.to_string(),
    };
    format!("{}-dimensional {chart} grid", cfg.dims)
}

fn ungated_launch_combination(cfg: &Config) -> Option<String> {
    if !gated_grid(cfg) {
        return Some(format!("the {} regime on the {}", cfg.regime, grid_name(cfg)));
    }
    None
}

/// whether the run spaces any axis non-uniformly
fn axis_maps_any(cfg: &Config) -> bool {
    match cfg.dims {
        1 => axis_maps::<1>(cfg).is_some(),
        2 => axis_maps::<2>(cfg).is_some(),
        _ => axis_maps::<3>(cfg).is_some(),
    }
}

fn validate_launch_scope(cfg: &Config) -> Result<(), String> {
    let refused = |what: &str| Err(format!("{what} is outside the distributed first release"));
    if !matches!(cfg.regime.as_str(), "newtonian" | "rhd" | "isothermal" | "nmhd" | "imhd" | "rmhd") {
        return refused(&format!("the {} regime", cfg.regime));
    }
    // the magnetized substrate installs the limiter (plm, or pcm at a zero slope), the solver and
    // the edge EMF and has no door for a flattening or an equation of state; a request for one is
    // refused here so a comparison never appears to exercise a setting the run ignores
    if matches!(cfg.regime.as_str(), "nmhd" | "rmhd") {
        if cfg.ppm_flatten_full > 0.0 || cfg.ppm_flatten_onset > 0.0 {
            return refused(&format!("ppm flattening with the {} regime", cfg.regime));
        }
        if cfg.eos_name != "ideal" {
            return refused(&format!("the {} equation of state with the {} regime", cfg.eos_name, cfg.regime));
        }
        if cfg.wb_reconstruction {
            return refused(&format!("well-balanced reconstruction with the {} regime", cfg.regime));
        }
    }
    if !matches!(cfg.dims, 1..=3) {
        return refused(&format!("a {}-dimensional grid", cfg.dims));
    }
    // a curved spacetime carries the relativistic gas and the relativistic magnetized gas on the
    // kerr-schild charts, with or without excision, on a static mesh without bodies, refined on the
    // three-dimensional kerr-schild box
    if cfg.spacetime != "minkowski" {
        // the single grid runs the baked GR-hydro and GR-MHD combinations, the launched workers the
        // ones the worker chart arms build
        let built = match cfg.regime.as_str() {
            "rhd" => launched_gr_hydro(cfg.dims, &cfg.coord_system, &cfg.spacetime),
            "rmhd" => baked_gr_mhd(cfg.dims, &cfg.coord_system, &cfg.spacetime),
            _ => return refused(&format!("the {} spacetime with the {} regime", cfg.spacetime, cfg.regime)),
        };
        if !built {
            return refused(&format!("the {} spacetime on the {}-dimensional {} chart", cfg.spacetime, cfg.dims, cfg.coord_system));
        }
        if cfg.refinement_enabled && !launched_refined_gr(cfg) {
            return refused(&format!("the {} spacetime with mesh refinement", cfg.spacetime));
        }
        if !cfg.bodies.is_empty() || cfg.bonded_assembly.is_some() {
            return refused(&format!("immersed bodies on the {} spacetime", cfg.spacetime));
        }
    }

    if cfg.refinement_enabled {
        validate_refined_launch_scope(cfg)?;
    }
    // the launched hydro workers move the mesh under the single grid's admission; the refined
    // and magnetized worker builders attach no motion, so an expression law there is refused
    validate_mesh_motion(cfg)?;
    if cfg.motion_json.is_some() && (cfg.refinement_enabled || cfg.regime.contains("mhd")) {
        return refused(&format!(
            "an expression mesh motion with {}",
            if cfg.refinement_enabled { "mesh refinement".to_string() } else { format!("the {} regime", cfg.regime) }
        ));
    }
    // the launched gas, isothermal and magnetized tiles carry the discrete and the ito
    // populations on a flat grid
    if cfg.n_tracers > 0 {
        // the refined workers carry the discrete and the ito populations
        if !matches!(cfg.regime.as_str(), "newtonian" | "rhd" | "isothermal" | "nmhd" | "rmhd" | "imhd") {
            return refused(&format!("a tracer population with the {} regime", cfg.regime));
        }
    }
    // the dye rides the launched gas and isothermal tiles under the single grid's admission
    validate_passive_scalar(cfg)?;
    // a combination the launched tiles build as the single path does but that no distributed
    // gate has validated: refused by name, or run as experimental under an explicit opt-in.
    // successful construction is no evidence of validated support
    validate_tiled_balance(cfg)?;
    if let Some(property) = authorize_launch(ungated_launch_combination(cfg), launch_experimental_opt_in())? {
        eprintln!("SIMBI launch: experimental combination without a distributed gate: {property}");
    }
    Ok(())
}

/// the refined launch's gated scope: newtonian, relativistic or isothermal hydro or MHD on a 2D
/// (2.5D for MHD) or 3D uniform cartesian grid, hydro with nested refined levels up to the depth
/// the phase table's wire byte holds and MHD with one, every level of ratio two; open or periodic
/// boundaries with the patch clear of a periodic face; pcm, plm or
/// ppm with its flattening, a constant viscosity, a resistivity, user sources and gradient walls
/// where the outer scope admits them for the regime. every other refined combination is named and
/// refused.
fn validate_refined_launch_scope(cfg: &Config) -> Result<(), String> {
    let refused = |what: &str| Err(format!("{what} with mesh refinement is outside the distributed first release"));
    if !matches!(cfg.regime.as_str(), "newtonian" | "rhd" | "isothermal" | "nmhd" | "imhd" | "rmhd") {
        return refused(&format!("the {} regime", cfg.regime));
    }
    if !matches!(cfg.dims, 1..=3) {
        return refused(&format!("a {}-dimensional grid", cfg.dims));
    }
    if cfg.coord_system != "cartesian" {
        return refused(&format!("the {} chart", cfg.coord_system));
    }
    let mapped = axis_maps_any(cfg);
    if mapped {
        return refused("a non-uniform spacing");
    }
    let depth = 1 + cfg.refinement_regions.len();
    if cfg.refinement_regions.is_empty() {
        return refused("no refined region");
    }
    // the phase table of every level's subcycle fits the wire byte up to an implementation depth;
    // a deeper run is refused here, before any session is built, never truncated
    let stages = cfg.timestepping.stages().len() as u8;
    let band_seams = match cfg.dims {
        2 => launched_band_seams::<2>(cfg)?,
        3 => launched_band_seams::<3>(cfg)?,
        _ => Vec::new(),
    };
    if let Err(e) = symbi_sim::phase_table::PhaseTable::nested_banded(stages, 2, depth as u8, &band_seams) {
        return refused(&format!("{depth} levels, the root included ({e}),"));
    }
    // the root and seed plans carry the wrap of a periodic axis cut into tiles; the fine level and
    // its registers exchange inside the domain, so on a cut periodic axis the patch keeps clear of
    // the faces. an uncut periodic axis wraps within each tile, and the fine level of a patch
    // spanning it wraps with it; the root's parent face stencil reads the periodic image, and the
    // nested levels' parent face stencils read inside the domain, so a magnetized nested run keeps
    // clear of every periodic face
    let magnetized = matches!(cfg.regime.as_str(), "nmhd" | "imhd" | "rmhd");
    let nested = cfg.refinement_regions.len();
    if magnetized && nested > 1 && !cfg.bodies.is_empty() && !gated_nested_magnetized_bodies(cfg, cfg.dims, nested) {
        return Err(format!(
            "immersed bodies with {} nested refinement levels on magnetized launched workers are \
             gated for one draining sink on a 2.5D cartesian newtonian MHD run alone (PLM, RK2, \
             HLLE, contact EMF, linear spacing, outflow walls, three levels, no resistive, viscous, \
             excised, driven or user-source term); run on one device or keep one refinement region",
            nested + 1
        ));
    }
    let seam = match cfg.dims {
        1 => patch_on_periodic_face::<1>(cfg)?,
        2 => patch_on_periodic_face::<2>(cfg)?,
        _ => patch_on_periodic_face::<3>(cfg)?,
    };
    if let Some(ax) = seam {
        return refused(&format!("a refined patch on the periodic face of axis {ax}"));
    }
    // a magnetized nested level reads its parent's faces through stencils compiled on the parent's
    // open partition, so a nested patch keeps clear of every periodic face; the first patch may
    // span an uncut periodic axis, whose levels wrap within each tile
    if magnetized && nested > 1 {
        let nested_seam = match cfg.dims {
            1 => nested_patch_on_periodic_face::<1>(cfg, false)?,
            2 => nested_patch_on_periodic_face::<2>(cfg, false)?,
            _ => nested_patch_on_periodic_face::<3>(cfg, false)?,
        };
        if let Some(ax) = nested_seam {
            return refused(&format!("a nested magnetized patch on the periodic face of axis {ax}"));
        }
    }
    // the refined worker hierarchy seeds no per-cell closure field on its levels
    if cfg.locally_isothermal {
        return refused("a locally isothermal closure");
    }
    if cfg.excision_radius > 0.0 && !launched_refined_gr(cfg) {
        return refused("excision");
    }
    Ok(())
}

/// the first periodic axis cut into tiles whose domain face the launch's refined patch reaches,
/// short of a patch spanning the whole axis: that patch has no face along it and its level, its
/// staggered faces included, wraps across the tiles as the root does
fn patch_on_periodic_face<const D: usize>(cfg: &Config) -> Result<Option<usize>, String> {
    let whole = whole_grid::<D>(cfg);
    let patch = launch_patch::<D>(cfg)?;
    let n: [usize; D] = std::array::from_fn(|ax| cfg.n_cells[ax]);
    let counts = tile_partition(n, cfg)?.counts();
    Ok((0..D).find(|&ax| {
        let spans = patch[ax].0 == 0 && patch[ax].1 >= cfg.n_cells[ax];
        whole.periodic(ax) && counts[ax] > 1 && (patch[ax].0 == 0 || patch[ax].1 >= cfg.n_cells[ax]) && !spans
    }))
}

/// the first periodic axis a nested refinement region (every region past the first) reaches the
/// domain face of; with `cut_only`, among the axes cut into tiles
fn nested_patch_on_periodic_face<const D: usize>(cfg: &Config, cut_only: bool) -> Result<Option<usize>, String> {
    let whole = whole_grid::<D>(cfg);
    let regions = refinement_regions_nd::<D>(&cfg.refinement_regions)?;
    let cell = |x: f64, ax: usize| ((x - cfg.x_lo[ax]) / cfg.dx[ax]).round() as usize;
    let n: [usize; D] = std::array::from_fn(|ax| cfg.n_cells[ax]);
    let counts = tile_partition(n, cfg)?.counts();
    for region in regions.iter().skip(1) {
        for ax in 0..D {
            let considered = !cut_only || counts[ax] > 1;
            if considered && whole.periodic(ax) && (cell(region.x_lo[ax], ax) == 0 || cell(region.x_hi[ax], ax) >= cfg.n_cells[ax]) {
                return Ok(Some(ax));
            }
        }
    }
    Ok(None)
}

/// the refined root patch of the launch's one region in root cells per axis, snapped as the
/// hierarchy snaps a region to its parent's cells
fn launch_patch<const D: usize>(cfg: &Config) -> Result<[(usize, usize); D], String> {
    let regions = refinement_regions_nd::<D>(&cfg.refinement_regions)?;
    let r = &regions[0];
    let cell = |x: f64, ax: usize| ((x - cfg.x_lo[ax]) / cfg.dx[ax]).round();
    Ok(std::array::from_fn(|ax| (cell(r.x_lo[ax], ax) as usize, cell(r.x_hi[ax], ax) as usize)))
}

/// the root tiling of a refined launch as this worker sees it: the partition, the owners of its
/// tiles, the tiles this worker holds, and the refined patch in root cells
struct RefinedLayout<const D: usize> {
    n: [usize; D],
    partition: symbi::sim::decomp::Partition<D>,
    owner: Vec<symbi_fabric::WorkerId>,
    placement: symbi_sim::atlas::Placement,
    me: symbi_fabric::WorkerId,
    mine: Vec<usize>,
    patch: [(usize, usize); D],
}

fn refined_layout<const D: usize>(cfg: &Config, launch: &WorkerLaunch, cells: usize) -> Result<RefinedLayout<D>, String> {
    use symbi_fabric::WorkerId;
    use symbi_sim::atlas::{Placement, TileId};
    let n: [usize; D] = std::array::from_fn(|ax| cfg.n_cells[ax]);
    let total: usize = n.iter().product();
    if cells != total {
        return Err(format!("prim_gen yielded {cells} cells, expected {total}"));
    }
    if !cfg.chi_ic.is_empty() && cfg.chi_ic.len() != total {
        return Err(format!("passive_scalar yielded {} values for {total} interior cells", cfg.chi_ic.len()));
    }
    let partition = tile_partition(n, cfg)?;
    let n_root = partition.n_tiles();
    if launch.owner.len() != n_root {
        return Err(format!("the owner map names {} tiles; the partition has {n_root}", launch.owner.len()));
    }
    let owner: Vec<WorkerId> = launch.owner.iter().map(|&w| WorkerId(w as u32)).collect();
    let placement = Placement::new(launch.workers as u32, owner.clone())?;
    let me = WorkerId(launch.worker as u32);
    let mine: Vec<usize> = (0..n_root).filter(|&f| placement.owner_of(TileId(f as u32)) == me).collect();
    if mine.is_empty() {
        return Err(format!("worker {} holds no tile", launch.worker));
    }
    let patch = launch_patch::<D>(cfg)?;
    Ok(RefinedLayout { n, partition, owner, placement, me, mine, patch })
}

/// one held root tile's hierarchy, the tile at its own origin so the levels index the parent's
/// absolute cells from it: the patch clipped to the tile with the whole patch beside it (a patch
/// face on a cut as a contributing register face), or the receiving register of the shell beside
/// the patch; a magnetized tile carries the step-entry faces its neighbors' parent face stencils
/// gather, and a refined one the stencil itself
/// a refined worker's share of the discrete population: the population seeds once from the whole
/// seeded hierarchy, built from `whole` with the levels' kernels, as the single hierarchy seeds its
/// own, and the held tiles keep the records their cells own
fn seed_refined_worker_tracers<R, const D: usize, const DOF: usize, M, E, K>(
    cfg: &Config,
    tiles: &mut [Hierarchy<R, D, DOF, M, E, DefaultSpace, DefaultMemory, K>],
    whole: impl FnOnce() -> Result<SimDefaultGeneric<R, D, DOF, M, E>, String>,
    kernels_of: impl Fn(&SimDefaultGeneric<R, D, DOF, M, E>) -> K,
) -> Result<(), String>
where
    R: Regime<f64, D, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF> + Copy,
    M: Metric<f64, D> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    K: KernelSet<D, DOF, DefaultMemory, f64>,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
    <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
{
    if cfg.n_tracers == 0 {
        return Ok(());
    }
    let sim = whole()?;
    let regions = refinement_regions_nd::<D>(&cfg.refinement_regions)?;
    let sub = kernels_of(&sim);
    let global = Hierarchy::with_refinement(sim, sub, &regions, prolong_order_for(&cfg.reconstruction_name), |s| kernels_of(s))
        .map_err(|e| format!("tracer seeding hierarchy: {e:?}"))?;
    global.seed_fine_from_coarse().map_err(|e| format!("tracer seeding fine seed: {e:?}"))?;
    symbi::sim::refinement::seed_held_hierarchy_tracers(&global, tiles, cfg.n_tracers);
    // an ito run carries the seeded population as continuous tracers, advancing against the whole
    // domain's extent and boundaries where a tile's own faces include cuts
    if let Some(order) = configured_ito_order(cfg)? {
        let bounds: [(f64, f64); D] = std::array::from_fn(|ax| (cfg.x_lo[ax], cfg.x_lo[ax] + cfg.n_cells[ax] as f64 * cfg.dx[ax]));
        let boundaries = boundaries_nd::<D>(&cfg.boundaries);
        for tile in tiles.iter_mut() {
            tile.set_continuous_domain(bounds, boundaries);
            for level in &mut tile.levels {
                convert_seeded_tracers(&mut level.state, order)?;
            }
        }
    }
    Ok(())
}

fn refined_tile<R, const D: usize, const DOF: usize, M, E, K>(
    cfg: &Config,
    layout: &RefinedLayout<D>,
    flat: usize,
    grid: &WholeGrid<D>,
    tile: &PlacedTile<D>,
    sim: SimDefaultGeneric<R, D, DOF, M, E>,
    kernels_of: impl Fn(&SimDefaultGeneric<R, D, DOF, M, E>, LevelRole) -> K,
) -> Result<Hierarchy<R, D, DOF, M, E, DefaultSpace, DefaultMemory, K>, String>
where
    R: Regime<f64, D> + Copy,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    K: KernelSet<D, DOF, DefaultMemory, f64>,
{
    use symbi::sim::refinement::launched::receiving_faces_of_tile;
    use symbi_sim::atlas::TileId;
    let regions = refinement_regions_nd::<D>(&cfg.refinement_regions)?;
    // the physical clock starts at start_time on the root, and the hierarchy built from it
    // starts every fine level there, as on the single hierarchy; the root carries the user
    // timestep clamp the hierarchy applies
    let mut sim = sim;
    seed_run_state(&mut sim, cfg)?;
    // the root carries the excision; every fine level is built as the single hierarchy builds it
    let sub = kernels_of(&sim, LevelRole::Root);
    let clipped = clip_regions_to_tile::<D>(&regions, grid, tile);
    let mut h = if clipped.is_empty() {
        let h = Hierarchy::single(sim, sub);
        match receiving_faces_of_tile(&layout.partition, layout.patch, TileId(flat as u32)) {
            Some((patch_local, faces)) => h
                .with_receiving_register(&patch_local, faces)
                .map_err(|e| format!("tile {flat} receiving register: {e:?}"))?,
            None => h,
        }
    } else {
        let (clip, whole): (Vec<_>, Vec<_>) = clipped.into_iter().unzip();
        let prolong = prolong_order_for(&cfg.reconstruction_name);
        let ring = grid.topology(layout.partition.counts());
        Hierarchy::with_refinement_placed(sim, sub, &clip, &whole, prolong, |s| kernels_of(s, LevelRole::Fine), layout.n, std::array::from_fn(|ax| ring.is_periodic(ax)))
            .map_err(|e| format!("tile {flat} refinement build: {e:?}"))?
    };
    // immersed bodies at their global positions: the finest level owns the full bodies, coarser
    // levels a gravity-only proxy, as on the in-process refined tiles
    if !cfg.bodies.is_empty() {
        h = h.try_with_bodies(build_bodies_and_horizon::<D>(cfg)).map_err(|e| format!("tile {flat}: {e}"))?;
        h.attach_body_shapes(build_body_shapes_and_horizon(cfg));
    }
    h.enable_face_stencil().map_err(|e| format!("tile {flat} face stencil: {e:?}"))?;
    h.enable_nested_face_stencils(1 + regions.len()).map_err(|e| format!("tile {flat} nested face stencils: {e:?}"))?;
    // the tracer containers address the whole root lattice from the tile's first cell, the root
    // closed across its periodic seams
    let ring = grid.topology(layout.partition.counts());
    h.set_root_layout(layout.n, tile.offset, std::array::from_fn(|ax| ring.is_periodic(ax)));
    Ok(h)
}

/// evolve this worker's refined hierarchies through the fabric, the tiles built per regime:
/// newtonian, relativistic or isothermal hydro, or newtonian, relativistic or isothermal MHD with
/// each tile seeded from its slice of the global staggered faces, on a 2D or 3D grid
fn run_refined_worker_process(
    cfg: &Config,
    prims: &[Vec<f64>],
    bufs: &[Vec<f64>],
    launch: &WorkerLaunch,
    fault: LaunchFault,
    drained: std::time::Duration,
) -> Result<(), String> {
    use symbi::sim::decomp::unflatten;
    // the arms of one grid dimension
    macro_rules! refined_worker_of_dims {
        ($d:literal) => {{
    let build_clock = std::time::Instant::now();
            let layout = refined_layout::<$d>(cfg, launch, prims.len())?;
            let n = layout.n;
            let counts = layout.partition.counts();
            // each held tile's place on the whole grid
            let whole = whole_grid::<$d>(cfg);
            let place = |flat: usize| whole.place(&layout.partition, unflatten(flat, counts));
            // the refined tiles of a gas regime (newtonian or relativistic hydro): each level installs
            // the numerics the single hierarchy installs
            macro_rules! refined_gas {
                ($regime:ident, $geom:expr, $geom_ty:ty) => {{
                    type Sim = SimDefault<$regime, $d, $geom_ty, EosSelect<f64>>;
                    // the numerics, sources, driven boundaries and gradient walls the single hierarchy
                    // installs, on every level a worker holds; the root alone carries the excision
                    let settings = KernelSettings::of(cfg, <$regime as Regime<f64, $d>>::SPEC);
                    let kernels_of = |s: &Sim, role: LevelRole| gas_kernels(s.substrate(), &settings, role).expect("refined kernel set");
                    let mut tiles = Vec::with_capacity(layout.mine.len());
                    for &flat in &layout.mine {
                        let tile = place(flat);
                        let tile_lo = tile.offset;
                        let sim = Sim::build($regime, host_eos(cfg), $geom)
                            .placed(&whole, &tile)
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let row = &prims[global_lin::<$d>(&tile_lo, &idx, &n)];
                                Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $d]))
                            })
                            .build();
                        // the dye on the root; the hierarchy carries it onto the fine levels
                        let sim = with_tile_dye!(sim, cfg, tile_lo, n, flat);
                        tiles.push(refined_tile(cfg, &layout, flat, &whole, &tile, sim, kernels_of)?);
                    }
                    seed_refined_worker_tracers(cfg, &mut tiles, || {
                        Ok(Sim::build($regime, host_eos(cfg), $geom)
                            .placed(&whole, &whole.single())
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tracer seeding allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let row = &prims[global_lin::<$d>(&[0; $d], &idx, &n)];
                                Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $d]))
                            })
                            .build())
                    }, |s: &Sim| kernels_of(s, LevelRole::Fine))?;
                    // the declared stationary target, seeded on every held level for the probe the
                    // session runs once the levels are seeded; a fresh run seeded from it starts there
                    let has_energy = tiles[0].levels[0].state.fields.cons.has_energy();
                    let expression = declared_equilibrium!(cfg, $d, has_energy);
                    let mut seed = |h: &Hierarchy<$regime, $d, $d, $geom_ty, EosSelect<f64>, DefaultSpace, DefaultMemory, _>| -> Result<(), String> {
                        let Some(expression) = expression.as_ref() else { return Ok(()) };
                        h.seed_declared_target(&|x: [f64; $d]| {
                            let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                            symbi_hydro::state::PrimFromSlots::from_slots(&expression.eval(at(0), at(1), at(2), 0.0))
                        })
                    };
                    let declared = expression.is_some();
                    refined_session(cfg, launch, fault, drained, build_clock, &layout, tiles, declared.then_some(&mut seed as _), |h| {
                        if declared && cfg.seed_from_equilibrium {
                            h.seed_equilibrium();
                        }
                        // the declared perturbation over the seeded levels, as the single hierarchy lays it
                        let has_energy = h.levels[0].state.fields.cons.has_energy();
                        if lay_perturbation!(h, cfg, $d, has_energy) {
                            h.sync_perturbed();
                        }
                        Ok(())
                    })
                }};
            }
            // the refined tiles of a magnetized regime on the 2.5D plane (newtonian or relativistic MHD),
            // each seeded from its slice of the global staggered faces
            macro_rules! refined_mhd {
                ($regime:ident, $geom:expr, $geom_ty:ty) => {{
                    type Sim = SimDefaultGeneric<$regime, $d, 3, $geom_ty, EosSelect<f64>>;
                    if bufs.len() < 3 {
                        return Err(format!("mhd needs 3 staggered b-field generators, got {}", bufs.len()));
                    }
                    // the numerics, sources, driven boundaries and gradient walls the single hierarchy
                    // installs, on every level a worker holds; the root alone carries the excision
                    let settings = KernelSettings::of(cfg, <$regime as Regime<f64, $d>>::SPEC);
                    let kernels_of = |s: &Sim, role: LevelRole| mhd_kernels(s.substrate(), &settings, role).expect("refined kernel set");
                    let mut tiles = Vec::with_capacity(layout.mine.len());
                    for &flat in &layout.mine {
                        let tile = place(flat);
                        let (tile_lo, m) = (tile.offset, tile.cells);
                        let tile_faces: Vec<Vec<f64>> = (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                        // the rollback snapshot restores every level's faces when a step is rejected
                        let sim = Sim::build($regime, host_eos(cfg), $geom)
                            .placed(&whole, &tile)
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                                let row = &prims[lin];
                                let gidx: [isize; $d] = std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                                let mag: [f64; 3] = std::array::from_fn(|k| if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] });
                                MhdPrim::new(Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])), Tensor::new(mag))
                            })
                            .seed_faces_indexed(&tile_faces)
                            .build()
                            .with_rollback_snapshot()
                            .map_err(|e| format!("tile {flat} rollback snapshot: {e:?}"))?;
                        tiles.push(refined_tile(cfg, &layout, flat, &whole, &tile, sim, kernels_of)?);
                    }
                    seed_refined_worker_tracers(cfg, &mut tiles, || {
                        Ok(Sim::build($regime, host_eos(cfg), $geom)
                            .placed(&whole, &whole.single())
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tracer seeding allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let lin = global_lin::<$d>(&[0; $d], &idx, &n);
                                let row = &prims[lin];
                                let mag: [f64; 3] = std::array::from_fn(|k| if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] });
                                MhdPrim::new(Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])), Tensor::new(mag))
                            })
                            .seed_faces_indexed(&bufs[0..$d])
                            .build())
                    }, |s: &Sim| kernels_of(s, LevelRole::Fine))?;
                    refined_session(cfg, launch, fault, drained, build_clock, &layout, tiles, None, |_| Ok(()))
                }};
            }
            match cfg.regime.as_str() {
                "newtonian" => refined_gas!(Newtonian, Cartesian, Cartesian),
                // the relativistic gas on the horizon-penetrating kerr-schild box
                "rhd" if cfg.spacetime == "schwarzschild_ks" => refined_gas!(
                    Rhd,
                    SchwarzschildKSCartesian { mass: cfg.schwarzschild_mass },
                    SchwarzschildKSCartesian<f64>
                ),
                "rhd" if cfg.spacetime == "kerr_ks" => refined_gas!(
                    Rhd,
                    KerrKSCartesian { mass: cfg.schwarzschild_mass, spin: cfg.kerr_spin },
                    KerrKSCartesian<f64>
                ),
                "rhd" => refined_gas!(Rhd, Cartesian, Cartesian),
                "nmhd" => refined_mhd!(NewtonianMhd, Cartesian, Cartesian),
                // the relativistic magnetized gas on the kerr-schild box, its face transfers carrying
                // the densitized flux
                "rmhd" if cfg.spacetime == "schwarzschild_ks" => refined_mhd!(
                    Rmhd,
                    SchwarzschildKSCartesian { mass: cfg.schwarzschild_mass },
                    SchwarzschildKSCartesian<f64>
                ),
                "rmhd" if cfg.spacetime == "kerr_ks" => refined_mhd!(
                    Rmhd,
                    KerrKSCartesian { mass: cfg.schwarzschild_mass, spin: cfg.kerr_spin },
                    KerrKSCartesian<f64>
                ),
                "rmhd" => refined_mhd!(Rmhd, Cartesian, Cartesian),
                "isothermal" => {
                    type Sim = SimDefault<IsoNewtonian, $d, Cartesian, Isothermal<f64>>;
                    // the numerics, sources, driven boundaries and gradient walls the single hierarchy
                    // installs, on every level a worker holds; a launched refined run carries no
                    // excision, so its root is built as every other level
                    let settings = KernelSettings::of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC);
                    let kernels_of = |s: &Sim, _role: LevelRole| iso_kernels(s.substrate(), &settings).expect("refined kernel set");
                    let mut tiles = Vec::with_capacity(layout.mine.len());
                    for &flat in &layout.mine {
                        let tile = place(flat);
                        let tile_lo = tile.offset;
                        let sim = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, Cartesian)
                            .placed(&whole, &tile)
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let row = &prims[global_lin::<$d>(&tile_lo, &idx, &n)];
                                PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])))
                            })
                            .build();
                        // the dye on the root; the hierarchy carries it onto the fine levels
                        let sim = with_tile_dye!(sim, cfg, tile_lo, n, flat);
                        tiles.push(refined_tile(cfg, &layout, flat, &whole, &tile, sim, kernels_of)?);
                    }
                    seed_refined_worker_tracers(cfg, &mut tiles, || {
                        Ok(Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, Cartesian)
                            .placed(&whole, &whole.single())
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tracer seeding allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let row = &prims[global_lin::<$d>(&[0; $d], &idx, &n)];
                                PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])))
                            })
                            .build())
                    }, |s: &Sim| kernels_of(s, LevelRole::Fine))?;
                    let expression = declared_equilibrium!(cfg, $d, false);
                    let mut seed = |h: &Hierarchy<IsoNewtonian, $d, $d, Cartesian, Isothermal<f64>, DefaultSpace, DefaultMemory, _>| -> Result<(), String> {
                        let Some(expression) = expression.as_ref() else { return Ok(()) };
                        h.seed_declared_target(&|x: [f64; $d]| {
                            let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                            symbi_hydro::state::PrimFromSlots::from_slots(&expression.eval(at(0), at(1), at(2), 0.0))
                        })
                    };
                    let declared = expression.is_some();
                    refined_session(cfg, launch, fault, drained, build_clock, &layout, tiles, declared.then_some(&mut seed as _), |h| {
                        if declared && cfg.seed_from_equilibrium {
                            h.seed_equilibrium();
                        }
                        // the declared perturbation over the seeded levels, as the single hierarchy lays it
                        if lay_perturbation!(h, cfg, $d, false) {
                            h.sync_perturbed();
                        }
                        Ok(())
                    })
                }
                "imhd" => {
                    type Sim = SimDefaultGeneric<IsothermalMhd, $d, 3, Cartesian, Isothermal<f64>>;
                    if bufs.len() < 3 {
                        return Err(format!("imhd needs 3 staggered b-field generators, got {}", bufs.len()));
                    }
                    // the numerics, sources, driven boundaries and gradient walls the single hierarchy
                    // installs, on every level a worker holds; a launched refined run carries no
                    // excision, so its root is built as every other level
                    let settings = KernelSettings::of(cfg, <IsothermalMhd as Regime<f64, $d>>::SPEC);
                    let kernels_of = |s: &Sim, role: LevelRole| mhd_kernels(s.substrate(), &settings, role).expect("refined kernel set");
                    let mut tiles = Vec::with_capacity(layout.mine.len());
                    for &flat in &layout.mine {
                        let tile = place(flat);
                        let (tile_lo, m) = (tile.offset, tile.cells);
                        let tile_faces: Vec<Vec<f64>> = (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                        // the rollback snapshot restores every level's faces when a step is rejected
                        let sim = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, Cartesian)
                            .placed(&whole, &tile)
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                                let row = &prims[lin];
                                let gidx: [isize; $d] = std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                                let mag: [f64; 3] = std::array::from_fn(|k| if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] });
                                MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])), Tensor::new(mag))
                            })
                            .seed_faces_indexed(&tile_faces)
                            .build()
                            .with_rollback_snapshot()
                            .map_err(|e| format!("tile {flat} rollback snapshot: {e:?}"))?;
                        tiles.push(refined_tile(cfg, &layout, flat, &whole, &tile, sim, kernels_of)?);
                    }
                    seed_refined_worker_tracers(cfg, &mut tiles, || {
                        Ok(Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, Cartesian)
                            .placed(&whole, &whole.single())
                            .ghosts(ghost_width(cfg))
                            .cfl(cfg.cfl)
                            .timestepping(cfg.timestepping)
                            .allocate()
                            .map_err(|e| format!("tracer seeding allocate: {e:?}"))?
                            .set_initial_indexed(|idx, _x| {
                                let lin = global_lin::<$d>(&[0; $d], &idx, &n);
                                let row = &prims[lin];
                                let mag: [f64; 3] = std::array::from_fn(|k| if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] });
                                MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])), Tensor::new(mag))
                            })
                            .seed_faces_indexed(&bufs[0..$d])
                            .build())
                    }, |s: &Sim| kernels_of(s, LevelRole::Fine))?;
                    refined_session(cfg, launch, fault, drained, build_clock, &layout, tiles, None, |_| Ok(()))
                }
                other => Err(format!("the {other} regime with mesh refinement is outside the distributed first release")),
            }
        }};
    }
    match cfg.dims {
        1 => refined_worker_of_dims!(1),
        2 => refined_worker_of_dims!(2),
        3 => refined_worker_of_dims!(3),
        d => Err(format!("a {d}-dimensional grid with mesh refinement is outside the distributed first release")),
    }
}

/// the refined session over the held tiles: the fine level partitioned under the root's cuts;
/// the plans of the root state, the root seed, the fine level and the flux contributions, and
/// for a magnetized run the EMF contributions, the interface faces, the corrected faces and the
/// two temporal levels of the parent face stencil; the refined transaction with the checkpoints
/// of both levels at the linear cadence, a final checkpoint, and a restart onto any partition
#[allow(clippy::too_many_arguments)]
fn refined_session<R, const D: usize, const DOF: usize, M, E, K>(
    cfg: &Config,
    launch: &WorkerLaunch,
    fault: LaunchFault,
    drained: std::time::Duration,
    build_clock: std::time::Instant,
    layout: &RefinedLayout<D>,
    mut tiles: Vec<Hierarchy<R, D, DOF, M, E, DefaultSpace, DefaultMemory, K>>,
    seed_target: Option<&mut dyn FnMut(&Hierarchy<R, D, DOF, M, E, DefaultSpace, DefaultMemory, K>) -> Result<(), String>>,
    on_seeded: impl FnMut(&mut Hierarchy<R, D, DOF, M, E, DefaultSpace, DefaultMemory, K>) -> Result<(), String>,
) -> Result<(), String>
where
    R: Regime<f64, D, Energy = <R as Regime<f64, DOF>>::Energy> + Regime<f64, DOF> + Copy,
    M: Metric<f64, D> + Metric<f64, DOF> + Copy + Send + Sync,
    E: symbi_hydro::eos::EosFor<f64, <R as Regime<f64, D>>::Energy> + Copy + Send + Sync,
    K: KernelSet<D, DOF, DefaultMemory, f64>,
    <R as Regime<f64, DOF>>::Cons: symbi_hydro::state::SeedableCons<f64, DOF>,
    <R as Regime<f64, DOF>>::Prim: symbi_hydro::state::PrimFromSlots<f64, DOF>,
{
    use symbi::sim::decomp::{FinePartition, LocalCopy, flatten, plan_cons_schema, plan_schema, unflatten};
    use symbi::sim::refinement::launched::{
        Held, NestedHeld, NestedPlans, RefinedConfig, RefinedGrids, RefinedPlans, evolve_refined_worker, face_stencil_boxes_of_tile,
        load_refined_tiles, refined_checkpoint,
    };
    use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
    use symbi_fabric::{Fnv, WorkerId};
    use symbi_sim::atlas::{ExchangePlan, Placement, Span, TileId};
    use symbi_sim::checkpoint::PhysicsIdentity;
    use symbi_sim::distributed_checkpoint::{CheckpointRequest, block_credit_for};
    use symbi_sim::phase_table::PhaseTable;
    use symbi_sim::plan_exchange::{PlanExchange, PlanRole, fabric_for};
    use symbi_sim::substrate_seam::RegimeKind;
    use symbi_sim::worker::{WorkerConfig, WorkerError};

    let RefinedLayout { n, partition, owner, placement, me, mine, patch } = layout;
    let (n, patch, me) = (*n, *patch, *me);
    if let Some((step, level, substep, stage)) = fault.level_rejection {
        for h in tiles.iter_mut().filter(|h| h.levels.len() > level) {
            h.reject_level_once_at(step, level, substep, stage);
        }
    }
    let counts = partition.counts();
    let n_root = partition.n_tiles();
    let total: usize = n.iter().product();

    // the fine level's tiles under the root's cuts, each owned by the worker owning its root tile
    let fine_partition = FinePartition::of(partition, patch, 2)?;
    let fine_counts = fine_partition.partition.counts();
    let n_fine = fine_partition.partition.n_tiles();
    let fine_owner: Vec<WorkerId> =
        (0..n_fine).map(|f| owner[flatten(fine_partition.root_tile_of(unflatten(f, fine_counts)), counts)]).collect();
    let fine_placement = Placement::with_idle_workers(launch.workers as u32, fine_owner)?;
    let root_ids: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let fine_ids: Vec<Option<TileId>> = mine
        .iter()
        .map(|&f| fine_partition.fine_tile_of(unflatten(f, counts)).map(|tc| TileId(flatten(tc, fine_counts) as u32)))
        .collect();
    for (k, h) in tiles.iter().enumerate() {
        if (h.levels.len() > 1) != fine_ids[k].is_some() {
            return Err(format!("tile {}: the hierarchy and the fine partition disagree on its fine level", mine[k]));
        }
    }

    // the plans of the session in their fabric id order
    let root_store = &*tiles[0].levels[0].state;
    let ng = root_store.geom.ng;
    let magnetized = root_store.fields.mhd.is_some();
    let schema = plan_schema(root_store);
    let cons_schema = plan_cons_schema(root_store);
    let open = symbi::sim::decomp::Topology::open();
    // the root level wraps a periodic axis across its seam; the fine level lies inside the domain
    let wrap = chart_topology(cfg, &whole_grid::<D>(cfg), counts);
    let root_plan = ExchangePlan::compile(partition, &wrap, &schema, ng).map_err(|e| e.to_string())?;
    let seed_plan = ExchangePlan::compile(partition, &wrap, &cons_schema, ng).map_err(|e| e.to_string())?;
    let fine_lo = |t: TileId| -> [isize; D] {
        let tc = unflatten(t.0 as usize, fine_counts);
        let root_ext = partition.tile_extents(fine_partition.root_tile_of(tc));
        std::array::from_fn(|a| (2 * patch[a].0.saturating_sub(root_ext[a].0)) as isize)
    };
    // the fine level wraps an axis the root wraps where its patch spans the whole axis
    let fine_wraps: [bool; D] = std::array::from_fn(|ax| wrap.is_periodic(ax) && patch[ax] == (0, n[ax]));
    let fine_topology = symbi::sim::decomp::Topology::wrapping(fine_wraps);
    let fine_plan = ExchangePlan::compile_shifted(&fine_partition.partition, &fine_topology, &schema, ng + 1, &fine_lo)
        .map_err(|e| e.to_string())?;
    // the register components: the conserved gas fields and, on a magnetized plane, the
    // out-of-plane cell field
    let n_comp = cons_schema.len() + if magnetized { DOF - D } else { 0 };
    let flux_plan = ExchangePlan::flux_contributions(partition, &open, patch, ng, n_comp).map_err(|e| e.to_string())?;
    let mut digests = vec![root_plan.digest, seed_plan.digest, fine_plan.digest, flux_plan.digest];
    let magnetized_plans = if magnetized {
        let emf = ExchangePlan::emf_contributions(partition, &open, patch, ng).map_err(|e| e.to_string())?;
        let faces = ExchangePlan::interface_faces(partition, &open, patch, ng).map_err(|e| e.to_string())?;
        let corrected = ExchangePlan::corrected_faces(partition, &open, patch, ng).map_err(|e| e.to_string())?;
        // every refined tile's stencil boxes in whole-grid face indices, derived from the
        // partition and the patch
        let mut boxes: Vec<Option<Vec<[Span; D]>>> = Vec::with_capacity(n_root);
        for f in 0..n_root {
            let ext = partition.tile_extents(unflatten(f, counts));
            let tile = match face_stencil_boxes_of_tile(partition, patch, TileId(f as u32)) {
                None => None,
                Some(components) => Some(
                    components
                        .iter()
                        .map(|d| {
                            let d = d.as_ref().ok_or_else(|| format!("tile {f}: a face component without a stencil box"))?;
                            Ok(std::array::from_fn(|a| Span { lo: d.spaces[a].lo + ext[a].0 as isize, hi: d.spaces[a].hi + ext[a].0 as isize }))
                        })
                        .collect::<Result<Vec<_>, String>>()?,
                ),
            };
            boxes.push(tile);
        }
        // the parent face stencil of a patch spanning a periodic axis reads past the domain face,
        // at the periodic image of the faces, on an uncut axis the tile's own
        let whole = whole_grid::<D>(cfg);
        let periodic = symbi::sim::decomp::Topology::wrapping(std::array::from_fn(|ax| whole.periodic(ax)));
        let stencil = ExchangePlan::face_stencil(partition, &periodic, &boxes, ng).map_err(|e| e.to_string())?;
        digests.extend([emf.digest, faces.digest, corrected.digest, stencil.digest]);
        Some((emf, faces, corrected, stencil))
    } else {
        None
    };
    // the nested levels: each refines its parent level over its region, its tiles cut where the
    // parent's cuts cross the region and owned by the worker owning the root tile beneath; every
    // worker derives every level's tiles, owners and index frames from the partition alone
    let regions = refinement_regions_nd::<D>(&cfg.refinement_regions)?;
    let depth = 1 + regions.len();
    let fine_frames: Vec<[isize; D]> = (0..n_fine).map(|t| fine_lo(TileId(t as u32))).collect();
    let fine_roots: Vec<usize> =
        (0..n_fine).map(|t| flatten(fine_partition.root_tile_of(unflatten(t, fine_counts)), counts)).collect();
    // per nested level: its partition, each tile's index frame, its root tile, its grid origin
    let mut nested_parts: Vec<FinePartition<D>> = Vec::new();
    let mut nested_frames: Vec<Vec<[isize; D]>> = Vec::new();
    let mut nested_roots: Vec<Vec<usize>> = Vec::new();
    let mut nested_origins: Vec<[isize; D]> = Vec::new();
    let mut parent_origin: [isize; D] = std::array::from_fn(|a| (2 * patch[a].0) as isize);
    for level in 2..depth {
        let (parent_part, parent_frames, parent_roots) = if level == 2 {
            (&fine_partition.partition, &fine_frames, &fine_roots)
        } else {
            (&nested_parts[level - 3].partition, &nested_frames[level - 3], &nested_roots[level - 3])
        };
        let r = &regions[level - 1];
        let scale = (1u64 << (level - 1)) as f64;
        let cell = |x: f64, a: usize| ((x - cfg.x_lo[a]) / (cfg.dx[a] / scale)).round() as isize;
        let mut rel = [(0usize, 0usize); D];
        for a in 0..D {
            let (lo, hi) = (cell(r.x_lo[a], a) - parent_origin[a], cell(r.x_hi[a], a) - parent_origin[a]);
            if lo < 0 || hi <= lo {
                return Err(format!("refinement region {} lies outside its parent level on axis {a}", level - 1));
            }
            rel[a] = (lo as usize, hi as usize);
        }
        let part = FinePartition::of(parent_part, rel, 2)?;
        let level_counts = part.partition.counts();
        let parent_counts = parent_part.counts();
        let mut frames = Vec::with_capacity(part.partition.n_tiles());
        let mut roots = Vec::with_capacity(part.partition.n_tiles());
        for t in 0..part.partition.n_tiles() {
            let parent_tc = part.root_tile_of(unflatten(t, level_counts));
            let parent = flatten(parent_tc, parent_counts);
            let ext = parent_part.tile_extents(parent_tc);
            frames.push(std::array::from_fn(|a| 2 * (parent_frames[parent][a] + rel[a].0.saturating_sub(ext[a].0) as isize)));
            roots.push(parent_roots[parent]);
        }
        let origin: [isize; D] = std::array::from_fn(|a| 2 * (parent_origin[a] + rel[a].0 as isize));
        nested_parts.push(part);
        nested_frames.push(frames);
        nested_roots.push(roots);
        nested_origins.push(origin);
        parent_origin = origin;
    }
    let nested_placements: Vec<Placement> = nested_roots
        .iter()
        .map(|roots| Placement::with_idle_workers(launch.workers as u32, roots.iter().map(|&r| owner[r]).collect()))
        .collect::<Result<_, _>>()?;
    let nested_state_plans: Vec<ExchangePlan<D>> = (0..nested_parts.len())
        .map(|i| {
            let frames = &nested_frames[i];
            ExchangePlan::compile_shifted(&nested_parts[i].partition, &open, &schema, ng + 1, &|t: TileId| frames[t.0 as usize])
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    // a continuous population's ito coefficients cross every level's cuts through their own plans,
    // each on its level's partition with the level's own ghost width
    let continuous_traced = cfg.n_tracers > 0 && configured_ito_order(cfg)?.is_some();
    let coefficient_plans: Vec<ExchangePlan<D>> = if continuous_traced {
        let coefficient_schema = (0..3 * D).fold(symbi_sim::atlas::FieldSchema::new(), |schema, ii| schema.cell(&format!("ito_coefficient_{ii}")));
        let mut plans = vec![
            ExchangePlan::compile(partition, &wrap, &coefficient_schema, ng).map_err(|e| e.to_string())?,
            ExchangePlan::compile_shifted(&fine_partition.partition, &fine_topology, &coefficient_schema, ng + 1, &fine_lo).map_err(|e| e.to_string())?,
        ];
        for i in 0..nested_parts.len() {
            let frames = &nested_frames[i];
            plans.push(
                ExchangePlan::compile_shifted(&nested_parts[i].partition, &open, &coefficient_schema, ng + 1, &|t: TileId| frames[t.0 as usize])
                    .map_err(|e| e.to_string())?,
            );
        }
        plans
    } else {
        Vec::new()
    };
    // the balanced bands a cut runs inside: at each such seam's band point the restricted density
    // and the band equilibria cross the cuts on the plans of the band's coarse and fine levels,
    // each on its level's partition with the level's own ghost width
    let band_seams = launched_band_seams::<D>(cfg)?;
    let band_plans: Vec<ExchangePlan<D>> = if band_seams.is_empty() {
        Vec::new()
    } else {
        let band_schema = symbi_sim::decomp::plan_band_schema();
        let mut plans = vec![
            ExchangePlan::compile(partition, &wrap, &band_schema, ng).map_err(|e| e.to_string())?,
            ExchangePlan::compile_shifted(&fine_partition.partition, &fine_topology, &band_schema, ng + 1, &fine_lo).map_err(|e| e.to_string())?,
        ];
        for i in 0..nested_parts.len() {
            let frames = &nested_frames[i];
            plans.push(
                ExchangePlan::compile_shifted(&nested_parts[i].partition, &open, &band_schema, ng + 1, &|t: TileId| frames[t.0 as usize])
                    .map_err(|e| e.to_string())?,
            );
        }
        plans
    };
    digests.extend(band_plans.iter().map(|p| p.digest));
    let nested_seed_plans: Vec<ExchangePlan<D>> = (0..nested_parts.len())
        .map(|i| {
            let (parent_part, frames) =
                if i == 0 { (&fine_partition.partition, &fine_frames) } else { (&nested_parts[i - 1].partition, &nested_frames[i - 1]) };
            ExchangePlan::compile_shifted(parent_part, &open, &cons_schema, ng + 1, &|t: TileId| frames[t.0 as usize])
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    // a magnetized nested level reads its parent's faces past a tile's face lattice from a stencil
    // gathered on the parent's partition: the boxes of the level's patch on every parent tile, in
    // whole-parent-level face indices, the parent partition's outer edges its coarse-fine boundary
    let nested_stencil_plans: Vec<Option<ExchangePlan<D>>> = (0..nested_parts.len())
        .map(|i| {
            if !magnetized {
                return Ok(None);
            }
            let (parent_part, frames) =
                if i == 0 { (&fine_partition.partition, &fine_frames) } else { (&nested_parts[i - 1].partition, &nested_frames[i - 1]) };
            let rel = nested_parts[i].patch;
            let parent_counts = parent_part.counts();
            let mut boxes: Vec<Option<Vec<[Span; D]>>> = Vec::with_capacity(parent_part.n_tiles());
            for t in 0..parent_part.n_tiles() {
                let ext = parent_part.tile_extents(unflatten(t, parent_counts));
                let tile = match symbi::sim::refinement::launched::face_stencil_boxes_of_level_tile(parent_part, rel, TileId(t as u32), false) {
                    None => None,
                    Some(components) => Some(
                        components
                            .iter()
                            .map(|d| {
                                let d = d.as_ref().ok_or_else(|| format!("level {}: a face component without a stencil box", i + 2))?;
                                Ok(std::array::from_fn(|a| Span { lo: d.spaces[a].lo + ext[a].0 as isize, hi: d.spaces[a].hi + ext[a].0 as isize }))
                            })
                            .collect::<Result<Vec<_>, String>>()?,
                    ),
                };
                boxes.push(tile);
            }
            ExchangePlan::face_stencil_shifted(parent_part, &open, &boxes, ng + 1, &|t: TileId| frames[t.0 as usize])
                .map(Some)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, String>>()?;
    for ((state, seed), stencil) in nested_state_plans.iter().zip(&nested_seed_plans).zip(&nested_stencil_plans) {
        digests.extend([state.digest, seed.digest]);
        if let Some(stencil) = stencil {
            digests.push(stencil.digest);
        }
    }
    // each held root tile's tile on every nested level, when the level reaches it
    let nested_ids: Vec<Vec<Option<TileId>>> = (0..nested_parts.len())
        .map(|i| {
            mine.iter()
                .map(|&f| (0..nested_roots[i].len()).find(|&t| nested_roots[i][t] == f).map(|t| TileId(t as u32)))
                .collect()
        })
        .collect();
    for (k, h) in tiles.iter().enumerate() {
        for (i, ids) in nested_ids.iter().enumerate() {
            if (h.levels.len() > i + 2) != ids[k].is_some() {
                return Err(format!("tile {}: the hierarchy and the partition of level {} disagree on its tile", mine[k], i + 2));
            }
        }
    }
    let stages = cfg.timestepping.stages().len() as u8;
    let table = PhaseTable::nested_banded(stages, 2, depth as u8, &band_seams).map_err(|e| format!("{depth} refinement levels: {e}"))?;
    // each plan takes the fabric ids after the last one's
    fn placed<'a, const D2: usize>(
        plan: &'a ExchangePlan<D2>,
        placement: &'a Placement,
        me: WorkerId,
        table: &PhaseTable,
        role: PlanRole,
        offset: &mut u32,
    ) -> PlanExchange<'a, D2> {
        let p = PlanExchange::new(plan, placement, me, table.clone()).with_role(role, *offset);
        *offset += p.lens().len() as u32;
        p
    }
    let root = PlanExchange::new(&root_plan, placement, me, table.clone());
    let mut offset = root.lens().len() as u32;
    let mut next = |plan, placement, role| placed(plan, placement, me, &table, role, &mut offset);
    let seed = next(&seed_plan, placement, PlanRole::RootSeed);
    let fine = next(&fine_plan, &fine_placement, PlanRole::Fine);
    let transported = flux_plan.transfers().count() > 0;
    let flux = transported.then(|| next(&flux_plan, placement, PlanRole::FluxContribution));
    let (emf, faces, corrected, stencil) = match magnetized_plans.as_ref() {
        Some((emf, faces, corrected, stencil)) => {
            let emf = transported.then(|| next(emf, placement, PlanRole::EmfContribution));
            let faces = transported.then(|| next(faces, placement, PlanRole::InterfaceFace));
            let corrected = (corrected.transfers().count() > 0).then(|| next(corrected, placement, PlanRole::CorrectedFace));
            let old = next(stencil, placement, PlanRole::FaceStencil);
            let new = next(stencil, placement, PlanRole::FaceStencil);
            (emf, faces, corrected, Some((old, new)))
        }
        None => (None, None, None, None),
    };
    // the nested levels' plans take the ids after every plan of the first refined level, the
    // order every worker lists the plans in
    let mut nested_plans = Vec::with_capacity(nested_state_plans.len());
    for i in 0..nested_state_plans.len() {
        let level = (i + 2) as u8;
        let parent_placement = if i == 0 { &fine_placement } else { &nested_placements[i - 1] };
        let state = next(&nested_state_plans[i], &nested_placements[i], PlanRole::Nested { level });
        let seed = next(&nested_seed_plans[i], parent_placement, PlanRole::NestedSeed { level });
        let stencil = nested_stencil_plans[i].as_ref().map(|plan| {
            let old = next(plan, parent_placement, PlanRole::NestedFaceStencil { level });
            let new = next(plan, parent_placement, PlanRole::NestedFaceStencil { level });
            (old, new)
        });
        nested_plans.push(NestedPlans { state, seed, stencil });
    }
    // the coefficient plans take the ids after every level's plans
    let coefficients: Vec<_> = coefficient_plans
        .iter()
        .enumerate()
        .map(|(level, plan)| {
            let placement = match level {
                0 => placement,
                1 => &fine_placement,
                _ => &nested_placements[level - 2],
            };
            next(plan, placement, PlanRole::LevelCoefficients { level: level as u8 })
        })
        .collect();
    // the band plans take the ids after the coefficient plans
    let bands: Vec<_> = band_plans
        .iter()
        .enumerate()
        .map(|(level, plan)| {
            let placement = match level {
                0 => placement,
                1 => &fine_placement,
                _ => &nested_placements[level - 2],
            };
            next(plan, placement, PlanRole::BandSeam { level: level as u8 })
        })
        .collect();
    let plan_digest = {
        let mut h = Fnv::new();
        for d in digests {
            h.u32((d.0 >> 32) as u32);
            h.u32(d.0 as u32);
        }
        h.finish()
    };
    let budget = launch.staging_cells.max(1);
    let credit = block_credit_for(budget, D);
    let bind: std::net::IpAddr = launch.bind.parse().map_err(|e| format!("bind address '{}': {e}", launch.bind))?;
    let rendezvous = Rendezvous {
        me,
        coordinator: std::net::SocketAddr::new(bind, 0),
        announce: Some(std::path::PathBuf::from(&launch.rendezvous)),
        bind: Some(bind),
        advertise: Some(launch.advertise.clone()),
        identity: Identity {
            credential: launch.credential,
            build_id: build_source_id(),
            config_digest: symbi_fabric::Digest(launch.config_digest),
            plan_digest,
            placement_digest: placement.digest(),
            phase_table_digest: table.digest(),
            ng: ng as u32,
            workers: launch.workers as u32,
            block_credit: credit,
        },
        startup: std::time::Duration::from_secs(120),
    };
    let deadline = std::time::Duration::from_secs(600);
    let mut all: Vec<&PlanExchange<'_, D>> = vec![&root, &seed, &fine];
    all.extend(flux.as_ref());
    all.extend(emf.as_ref());
    all.extend(faces.as_ref());
    all.extend(corrected.as_ref());
    if let Some((old, new)) = stencil.as_ref() {
        all.extend([old, new]);
    }
    for level in &nested_plans {
        all.extend([&level.state, &level.seed]);
        if let Some((old, new)) = level.stencil.as_ref() {
            all.extend([old, new]);
        }
    }
    all.extend(coefficients.iter());
    all.extend(bands.iter());
    let max_payload = (all.iter().flat_map(|p| p.lens()).max().unwrap_or(0) * 8).max(credit as usize);
    let built = build_clock.elapsed();
    let connect_clock = std::time::Instant::now();
    let (link, session) = connect(&rendezvous, max_payload).map_err(|e| e.to_string())?;
    {
        use std::io::Write;
        let line = format!(
            "SIMBI launch startup: worker={}/{} drain={:.3}s build={:.3}s connect={:.3}s refined=1\n",
            launch.worker,
            launch.workers,
            drained.as_secs_f64(),
            built.as_secs_f64(),
            connect_clock.elapsed().as_secs_f64(),
        );
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
    let mut fabric = fabric_for(&all, link, session).with_block_credit(credit);
    let plans = RefinedPlans { root, seed, fine, flux, emf, faces, corrected, stencil, nested: nested_plans, coefficients, bands, band_seams };
    let root_devices = vec![0i32; n_root];
    let fine_devices = vec![0i32; n_fine];
    let nested_held: Vec<NestedHeld> = nested_ids
        .into_iter()
        .zip(&nested_parts)
        .map(|(ids, part)| {
            let n = part.partition.n_tiles();
            NestedHeld { ids, n, devices: vec![0i32; n] }
        })
        .collect();
    let held = Held {
        root_ids: &root_ids,
        fine_ids: &fine_ids,
        n_root,
        n_fine,
        root_devices: &root_devices,
        fine_devices: &fine_devices,
        nested: &nested_held,
    };
    let root_grid = decomposed_grid::<D>(cfg, ng);
    let mut grids = RefinedGrids::of(n, root_grid.x_lo, root_grid.dx, ng, patch);
    for (part, origin) in nested_parts.iter().zip(&nested_origins) {
        grids = grids.with_nested(part.partition.n_cells(), *origin);
    }
    symbi::sim::refinement::launched::record_level_spans(&mut tiles, &held, &plans, &grids);
    let restart = cfg.restart_path.is_some();
    if let Some(path) = cfg.restart_path.as_deref() {
        load_refined_tiles(&mut tiles, &held, &plans, &grids, path).map_err(|e| e.to_string())?;
    }
    let identity = PhysicsIdentity::of(&tiles[0].levels[0].state);
    let refined_cfg = RefinedConfig {
        base: WorkerConfig {
            regime: RegimeKind::of::<f64, D, R>(),
            timestepping: cfg.timestepping,
            start_time: cfg.start_time,
            t_final: cfg.t_final,
            max_steps: cfg.max_steps,
            deadline,
            injection: fault.injection.clone(),
            audit_every_point: std::env::var("SIMBI_FABRIC_AUDIT").is_ok_and(|v| v == "all"),
        },
        invalid_level_cfl_at: None,
        fine_trouble_at: None,
        contribution_fault: None,
        restart,
        perturb_fine_face_after: None,
        clock_fault: None,
    };
    let slabs: Vec<Vec<usize>> = (0..D).map(|ax| (0..counts[ax]).map(|i| partition.tile_range(ax, i).1).collect()).collect();
    let mut reporter = (me == WorkerId(0)).then(|| DecomposedReporter::new(cfg, total as u64, &counts, &slabs));
    // the checkpoint schedule the single grid walks
    let mut cadence = CheckpointCadence::of(cfg);
    let cp_width = checkpoint_time_width(cfg);
    // the last accepted step whose state an audit certified: the final checkpoint after a cadence
    // write of the same step reuses that certification, its audit epoch being spent
    let certified = std::cell::Cell::new(None::<u64>);
    let write = |tiles: &[Hierarchy<R, D, DOF, M, E, DefaultSpace, DefaultMemory, K>], fabric: &mut symbi_fabric::Fabric<_>, path: &str, cp_index: u64, accepted: u64| -> Result<(), WorkerError> {
        let audit = certified.get() != Some(accepted);
        certified.set(Some(accepted));
        let extras = checkpoint_metadata(cfg, cp_index);
        let req = CheckpointRequest {
            path,
            extras: &extras,
            budget,
            deadline,
            drop_block: None,
            duplicate_first_block: false,
            vote_no: false,
        };
        refined_checkpoint(tiles, &held, &plans, &grids, &identity, &req, fabric, accepted, audit)
    };
    let write_initial = !restart && (cfg.checkpoint_index == 0 || cfg.start_time == 0.0);
    // the body diagnostics, written by the coordinator from its first tile's finest bodies, which
    // carry the cross-tile-summed reaction as every tile's do; one `<dir>diagnostics.dat` per run
    let diag_path = if me == WorkerId(0) && cfg.diagnostic_interval > 0.0 && !cfg.bodies.is_empty() {
        Some(format!("{}diagnostics.dat", cfg.data_dir))
    } else {
        None
    };
    let diag_interval = (cfg.diagnostic_interval * cfg.time_unit).max(f64::MIN_POSITIVE);
    let diag_plan = match diag_path.as_ref() {
        Some(dp) => Some(plan_diagnostics(last_diagnostics_time(dp), cfg.start_time, diag_interval)?),
        None => None,
    };
    let mut next_diag = diag_plan.as_ref().map_or(diag_interval, |p| p.next);
    if let Some(dp) = &diag_path {
        if diag_plan.as_ref().is_some_and(|p| p.write_initial) {
            if let Some(im) = tiles[0].levels.last().and_then(|l| l.state.immersed.as_ref()) {
                let _ = append_diagnostics(dp, cfg.start_time, &im.bodies);
            }
        }
    }
    let guard = retry_guard_of(cfg);
    let exclusion = retry_exclusion_of(cfg);
    for tile in tiles.iter_mut() {
        for level in tile.levels.iter_mut() {
            level.state.retry_guard = guard;
            level.state.retry_exclusion = exclusion;
        }
    }
    if me == WorkerId(0) {
        announce_retry_guard(&tiles[0].levels[0].state);
    }
    let outcome = evolve_refined_worker(&mut tiles, &held, &plans, &LocalCopy, &mut fabric, &refined_cfg, seed_target, on_seeded, |accepted, time, tiles, fabric| {
        if accepted == 0 {
            if write_initial {
                let tag = cadence.tag(cfg, cp_width, cfg.start_time, cfg.checkpoint_index);
                let path = checkpoint_name(cfg, &tag);
                write(tiles, fabric, &path, cfg.checkpoint_index, 0)?;
                if let Some(r) = reporter.as_mut() {
                    r.checkpoint_written(cfg, &path, cfg.start_time, 0.0);
                }
            }
            return Ok(std::ops::ControlFlow::Continue(()));
        }
        if fault.exit_after == Some(accepted) {
            std::process::exit(86);
        }
        if let Some(r) = reporter.as_mut() {
            r.progress(accepted, time);
        }
        // the step that reaches the `max_steps` root iteration publishes through the final
        // snapshot alone, as on the single grid
        if step_cap_reached(cfg, tiles[0].levels[0].state.iteration) {
            return Ok(std::ops::ControlFlow::Continue(()));
        }
        if cadence.due(time) {
            let tag = cadence.tag(cfg, cp_width, time, cadence.index);
            let path = checkpoint_name(cfg, &tag);
            write(tiles, fabric, &path, cadence.index, accepted)?;
            if let Some(r) = reporter.as_mut() {
                r.checkpoint_written(cfg, &path, time, 0.0);
            }
            cadence.advance(time);
        }
        if let Some(dp) = &diag_path {
            if diagnostics_due(time, &mut next_diag, diag_interval) {
                if let Some(im) = tiles[0].levels.last().and_then(|l| l.state.immersed.as_ref()) {
                    let _ = append_diagnostics(dp, time, &im.bodies);
                }
            }
        }
        Ok(std::ops::ControlFlow::Continue(()))
    });
    let result: Result<(u64, f64, Option<String>), String> = outcome.map_err(|e| e.to_string()).and_then(|report| {
        // a march every worker agreed to halt publishes its step entry, the last accepted state,
        // as crashed in place of the final snapshot
        let outcome = if report.halt.is_some() { CheckpointOutcome::Crashed } else { CheckpointOutcome::Completed };
        let final_path = checkpoint_name(cfg, checkpoint_status_tag(outcome));
        write(&tiles, &mut fabric, &final_path, cadence.index, report.steps).map_err(|e| e.to_string())?;
        if let Some(r) = reporter.as_mut() {
            r.checkpoint_written(cfg, &final_path, report.time, 0.0);
        }
        fabric.finish(deadline).map_err(|e| e.to_string())?;
        {
            use std::io::Write;
            let line = format!(
                "SIMBI launch: worker={}/{} steps={} rejections={} refined=1\n",
                launch.worker, launch.workers, report.steps, report.rejections
            );
            let _ = std::io::stderr().write_all(line.as_bytes());
        }
        Ok((report.steps, report.time, report.halt))
    });
    match result {
        Ok((iter, time, Some(cause))) => halted_launch(cfg, reporter, iter, time, &cause),
        Ok((iter, time, None)) => {
            if let Some(r) = reporter {
                let final_path = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Completed));
                r.finish(iter, time, &final_path);
            }
            Ok(())
        }
        Err(msg) => {
            fabric.abort(&msg);
            if let Some(r) = reporter {
                r.fail(&msg);
            }
            Err(msg)
        }
    }
}

/// evolve this worker's tiles through the fabric: the per-tile build of the decomposed path
/// restricted to the tiles this worker holds, the rendezvous under the session credential,
/// the worker loop with the coordinator-written checkpoints at the linear cadence, and the
/// final checkpoint before the session ends. worker 0 reports through the decomposed table.
/// `bufs` holds the staggered face-field generators' output for a magnetized regime, one
/// buffer per component, and is empty for hydro.
fn run_worker_process(
    cfg: &Config,
    prims: &[Vec<f64>],
    bufs: &[Vec<f64>],
    launch: &WorkerLaunch,
    drained: std::time::Duration,
) -> Result<(), String> {
    use symbi::sim::decomp::unflatten;
    use symbi_fabric::WorkerId;
    use symbi_sim::atlas::{Placement, TileId};

    validate_run(cfg)?;
    validate_declared_target(cfg, validate_primitive_rows(cfg, prims)?)?;
    validate_tiled_march(cfg)?;
    validate_launch_scope(cfg)?;
    let fault = LaunchFault::from_env(launch.worker)?;
    if cfg.refinement_enabled {
        return run_refined_worker_process(cfg, prims, bufs, launch, fault, drained);
    }
    // the tiles are built and evolved at the grid's dimension; every helper below is
    // dimension-generic, so one body serves the plane and the volume
    macro_rules! worker_of_dims {
        ($d:tt) => {{
    let n: [usize; $d] = std::array::from_fn(|ax| cfg.n_cells[ax]);
    let total: usize = n.iter().product();
    if prims.len() != total {
        return Err(format!("prim_gen yielded {} cells, expected {total}", prims.len()));
    }
    if !cfg.chi_ic.is_empty() && cfg.chi_ic.len() != total {
        return Err(format!("passive_scalar yielded {} values for {total} interior cells", cfg.chi_ic.len()));
    }
    let build_clock = std::time::Instant::now();
    let partition = tile_partition(n, cfg)?;
    let counts = partition.counts();
    if launch.owner.len() != partition.n_tiles() {
        return Err(format!(
            "the owner map names {} tiles; the partition has {}",
            launch.owner.len(),
            partition.n_tiles()
        ));
    }
    let owner: Vec<WorkerId> = launch.owner.iter().map(|&w| WorkerId(w as u32)).collect();
    let placement = Placement::new(launch.workers as u32, owner)?;
    let me = WorkerId(launch.worker as u32);
    // every tile is placed on the whole grid by the index of its first cell, so each cell sits
    // bitwise where the whole grid puts it
    let whole = whole_grid::<$d>(cfg);
    let topology = chart_topology(cfg, &whole, counts);
    let seam = polar_seam::<$d>(&cfg.coord_system, &whole.boundaries, &topology, partition.counts(), |az, i| partition.tile_range(az, i).1);
    let mine: Vec<usize> = (0..partition.n_tiles())
        .filter(|&f| placement.owner_of(TileId(f as u32)) == me)
        .collect();
    if mine.is_empty() {
        return Err(format!("worker {} holds no tile", launch.worker));
    }
    let offsets = tile_offsets(&partition);
    // the gas regimes share one tile build: a conserved vector of density, momentum and energy
    // over the same primitive rows, with every numerics door of the single path
    // the tail every regime shares: the restart from a partitioned checkpoint, the clock, and
    // the fabric session over the built tiles
    macro_rules! session_over {
        ($regime_ty:ty, $dof:literal, $tiles:ident) => {{
            session_over!($regime_ty, $dof, $tiles, None, None)
        }};
        ($regime_ty:ty, $dof:literal, $tiles:ident, $seed:expr, $perturb:expr) => {{
            let ng = $tiles[0].0.geom.ng;
            let grid = decomposed_grid::<$d>(cfg, ng);
            for ((sim, _), &flat) in $tiles.iter_mut().zip(&mine) {
                pin_run_clock(sim, cfg);
                // the whole grid's interior in the tile's `index + offset` frame
                sim.geom.whole_span = Some(grid.interior_span(&sim.geom, offsets[flat]));
            }
            let built = build_clock.elapsed();
            worker_session::<$regime_ty, _, _, _, $d, $dof>(
                cfg, launch, fault, drained, built, &partition, placement, &topology, me, &mine, &offsets, $tiles, $seed, $perturb,
            )
        }};
    }
    macro_rules! gas_tiles {
        ($geom:expr, $geom_ty:ty, $regime:expr, $regime_ty:ty) => {
            gas_tiles!($geom, $geom_ty, $regime, $regime_ty, $d)
        };
        // `$dof` momentum components on the `$d`-dimensional grid: one more on a swirl run, whose
        // azimuthal momentum rides a two-dimensional kerr-schild grid
        ($geom:expr, $geom_ty:ty, $regime:expr, $regime_ty:ty, $dof:tt) => {{
            type Sim = SimDefaultGeneric<$regime_ty, $d, $dof, $geom_ty, EosSelect<f64>>;
            let mut tiles = Vec::with_capacity(mine.len());
            for &flat in &mine {
                let tc = unflatten(flat, counts);
                // the tile's place on the whole grid, its pole faces on a polar seam made cuts
                let mut tile = whole.place(&partition, tc);
                tile.boundaries = seam_faces_as_cuts(tile.boundaries.clone(), seam.as_ref());
                let tile_lo = tile.offset;
                let sim = Sim::build($regime, host_eos(cfg), $geom)
                    .placed(&whole, &tile)                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                        let row = &prims[lin];
                        Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $dof]))
                    })
                    .build();
                // the dye on this tile, indexed from the global dye by the tile's global cells;
                // the halo exchange moves prim.chi across the cuts
                let sim = with_tile_dye!(sim, cfg, tile_lo, n, flat);
                // gravity / accretion sinks plus bonded fragments and shaped walls, every tile
                // holding the whole set at their global positions
                let sim = if cfg.bodies.is_empty() && cfg.bonded_assembly.is_none() {
                    sim
                } else {
                    attach_bodies_and_fragments!(sim, cfg, $d)
                };
                // the clock and the mesh motion, every tile carrying the same a(t)
                let mut sim = sim;
                seed_run_state(&mut sim, cfg)?;
                // the declared perturbation at the tile's own cell centers, the single grid's; with a
                // declared stationary target the session lays it over the target, after the probe
                let has_energy = sim.fields.cons.has_energy();
                if cfg.equilibrium_json.is_none() {
                    lay_perturbation!(sim, cfg, $dof, has_energy);
                }
                // the numerics, sources, driven boundaries and gradient walls the single grid
                // installs, each tile evaluating them at the whole grid's coordinates of its cells
                let settings = KernelSettings::of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC);
                let sub = gas_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                tiles.push((sim, sub));
            }
            // the tracer population seeds once from the whole-grid density, as the single grid's,
            // and each worker keeps its own tiles' share; the whole-grid state lives for the
            // seeding alone
            if cfg.n_tracers > 0 {
                let global = Sim::build($regime, host_eos(cfg), $geom)
                    .placed(&whole, &whole.single())
                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("global seeding sim allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let row = &prims[global_lin::<$d>(&[0; $d], &idx, &n)];
                        Prim::adiabatic(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])), Pressure(row[1 + $dof]))
                    })
                    .build();
                let mut per_tile = partition_configured_tracers(&global, cfg, &partition);
                let order = configured_ito_order(cfg)?;
                for ((sim, _), &flat) in tiles.iter_mut().zip(&mine) {
                    let set = std::mem::take(&mut per_tile[flat]);
                    match order {
                        Some(order) => {
                            sim.continuous_tracers = Some(
                                symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                                    .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                            )
                        }
                        None => sim.tracers = Some(set),
                    }
                }
            }
            // the declared stationary target, seeded on each tile at its cell centres (t = 0) for the
            // probe the session runs before the restart restores the state
            let has_energy = tiles[0].0.fields.cons.has_energy();
            let expression = declared_equilibrium!(cfg, $d, has_energy);
            let seed = expression.as_ref().map(|expression| {
                move |sim: &Sim| {
                    sim.seed_cells(|x| {
                        let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                        symbi_hydro::state::PrimFromSlots::from_slots(&expression.eval(at(0), at(1), at(2), 0.0))
                    })
                }
            });
            let perturb = |sim: &mut Sim| -> Result<(), String> {
                lay_perturbation!(sim, cfg, $dof, has_energy);
                Ok(())
            };
            session_over!($regime_ty, $dof, tiles, seed.as_ref().map(|f| f as &dyn Fn(&Sim)), expression.as_ref().map(|_| &perturb as &dyn Fn(&mut Sim) -> Result<(), String>))
        }};
    }
    macro_rules! nmhd_tiles {
        ($geom:expr, $geom_ty:ty) => {{
            type Sim = SimDefaultGeneric<NewtonianMhd, $d, 3, $geom_ty, EosSelect<f64>>;
            if bufs.len() < 3 {
                return Err(format!("mhd needs 3 staggered b-field generators, got {}", bufs.len()));
            }
            let mut tiles = Vec::with_capacity(mine.len());
            for &flat in &mine {
                let tc = unflatten(flat, counts);
                // the tile's place on the whole grid, its pole faces on a polar seam made cuts
                let mut tile = whole.place(&partition, tc);
                tile.boundaries = seam_faces_as_cuts(tile.boundaries.clone(), seam.as_ref());
                let (tile_lo, m) = (tile.offset, tile.cells);
                // each tile's slice of the global staggered buffers: a shared interface face
                // reads the same global value on both of its tiles
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                let sim = Sim::build(NewtonianMhd, host_eos(cfg), $geom)
                    .placed(&whole, &tile)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                        let row = &prims[lin];
                        let gidx: [isize; $d] =
                            std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        MhdPrim::new(
                            Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])),
                            Tensor::new(mag),
                        )
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // the immersed bodies and their shaped walls, every tile holding the whole set
                let sim = if cfg.bodies.is_empty() { sim } else { attach_bodies_and_shapes!(sim, cfg, $d) };
                // the numerics, sources, driven boundaries and gradient walls the single grid
                // installs, each tile evaluating them at the whole grid's coordinates of its cells
                let settings = KernelSettings::of(cfg, <NewtonianMhd as Regime<f64, $d>>::SPEC);
                let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                tiles.push((sim, sub));
            }
            // the tracer population seeds once from the whole-grid density, as the single grid's,
            // and each worker keeps its own tiles' share; the whole-grid state lives for the
            // seeding alone
            if cfg.n_tracers > 0 {
                let global = Sim::build(NewtonianMhd, host_eos(cfg), $geom)
                    .placed(&whole, &whole.single())
                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("global seeding sim allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&[0; $d], &idx, &n);
                        let row = &prims[lin];
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                        });
                        MhdPrim::new(Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])), Tensor::new(mag))
                    })
                    .seed_faces_indexed(&bufs[0..$d])
                    .build();
                let mut per_tile = partition_configured_tracers(&global, cfg, &partition);
                let order = configured_ito_order(cfg)?;
                for ((sim, _), &flat) in tiles.iter_mut().zip(&mine) {
                    let set = std::mem::take(&mut per_tile[flat]);
                    match order {
                        Some(order) => {
                            sim.continuous_tracers = Some(
                                symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                                    .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                            )
                        }
                        None => sim.tracers = Some(set),
                    }
                }
            }
            session_over!(NewtonianMhd, 3, tiles)
        }};
    }
    macro_rules! rmhd_tiles {
        ($geom:expr, $geom_ty:ty) => {{
            type Sim = SimDefaultGeneric<Rmhd, $d, 3, $geom_ty, EosSelect<f64>>;
            if bufs.len() < 3 {
                return Err(format!("mhd needs 3 staggered b-field generators, got {}", bufs.len()));
            }
            let mut tiles = Vec::with_capacity(mine.len());
            for &flat in &mine {
                let tc = unflatten(flat, counts);
                // the tile's place on the whole grid, its pole faces on a polar seam made cuts
                let mut tile = whole.place(&partition, tc);
                tile.boundaries = seam_faces_as_cuts(tile.boundaries.clone(), seam.as_ref());
                let (tile_lo, m) = (tile.offset, tile.cells);
                // each tile's slice of the global staggered buffers: a shared interface face
                // reads the same global value on both of its tiles
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                let sim = Sim::build(Rmhd, host_eos(cfg), $geom)
                    .placed(&whole, &tile)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                        let row = &prims[lin];
                        let gidx: [isize; $d] =
                            std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        MhdPrim::new(
                            Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])),
                            Tensor::new(mag),
                        )
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // the immersed bodies and their shaped walls, every tile holding the whole set
                let sim = if cfg.bodies.is_empty() { sim } else { attach_bodies_and_shapes!(sim, cfg, $d) };
                // the numerics, sources, driven boundaries and gradient walls the single grid
                // installs, each tile evaluating them at the whole grid's coordinates of its cells
                let settings = KernelSettings::of(cfg, <Rmhd as Regime<f64, $d>>::SPEC);
                let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                tiles.push((sim, sub));
            }
            // the tracer population seeds once from the whole-grid density, as the single grid's,
            // and each worker keeps its own tiles' share; the whole-grid state lives for the
            // seeding alone
            if cfg.n_tracers > 0 {
                let global = Sim::build(Rmhd, host_eos(cfg), $geom)
                    .placed(&whole, &whole.single())
                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("global seeding sim allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&[0; $d], &idx, &n);
                        let row = &prims[lin];
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                        });
                        MhdPrim::new(Prim::adiabatic(Density(row[0]), Tensor::new([row[1], row[2], row[3]]), Pressure(row[4])), Tensor::new(mag))
                    })
                    .seed_faces_indexed(&bufs[0..$d])
                    .build();
                let mut per_tile = partition_configured_tracers(&global, cfg, &partition);
                let order = configured_ito_order(cfg)?;
                for ((sim, _), &flat) in tiles.iter_mut().zip(&mine) {
                    let set = std::mem::take(&mut per_tile[flat]);
                    match order {
                        Some(order) => {
                            sim.continuous_tracers = Some(
                                symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                                    .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                            )
                        }
                        None => sim.tracers = Some(set),
                    }
                }
            }
            session_over!(Rmhd, 3, tiles)
        }};
    }
    macro_rules! iso_tiles {
        ($geom:expr, $geom_ty:ty) => {{
            type Sim = SimDefault<IsoNewtonian, $d, $geom_ty, Isothermal<f64>>;
            let mut tiles = Vec::with_capacity(mine.len());
            for &flat in &mine {
                let tc = unflatten(flat, counts);
                // the tile's place on the whole grid, its pole faces on a polar seam made cuts
                let mut tile = whole.place(&partition, tc);
                tile.boundaries = seam_faces_as_cuts(tile.boundaries.clone(), seam.as_ref());
                let tile_lo = tile.offset;
                let sim = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
                    .placed(&whole, &tile)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                        let row = &prims[lin];
                        PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])))
                    })
                    .build();
                // the dye on this tile, indexed from the global dye by the tile's global cells;
                // the halo exchange moves prim.chi across the cuts
                let sim = with_tile_dye!(sim, cfg, tile_lo, n, flat);
                // gravity / accretion sinks plus bonded fragments and shaped walls, every tile
                // holding the whole set at their global positions
                let sim = if cfg.bodies.is_empty() && cfg.bonded_assembly.is_none() {
                    sim
                } else {
                    attach_bodies_and_fragments!(sim, cfg, $d)
                };
                // the clock and the mesh motion, every tile carrying the same a(t)
                let mut sim = sim;
                seed_run_state(&mut sim, cfg)?;
                // the declared perturbation at the tile's own cell centers, the single grid's; with a
                // declared stationary target the session lays it over the target, after the probe
                if cfg.equilibrium_json.is_none() {
                    lay_perturbation!(sim, cfg, $d, false);
                }
                // the numerics, sources, driven boundaries and gradient walls the single grid
                // installs, each tile evaluating them at the whole grid's coordinates of its cells
                let settings = KernelSettings::of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC);
                let sub = iso_kernels(sim.substrate(), &settings).map_err(|e| format!("tile {flat}: {e}"))?;
                seed_local_cs2!(sim, sub, cfg, prims, tile_lo, n, $d);
                tiles.push((sim, sub));
            }
            // the tracer population seeds once from the whole-grid density, as the single grid's,
            // and each worker keeps its own tiles' share; the whole-grid state lives for the
            // seeding alone
            if cfg.n_tracers > 0 {
                let global = Sim::build(IsoNewtonian, Isothermal { cs: cfg.cs }, $geom)
                    .placed(&whole, &whole.single())
                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("global seeding sim allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let row = &prims[global_lin::<$d>(&[0; $d], &idx, &n)];
                        PrimG::<f64, $d, IsoModel>::isothermal(Density(row[0]), Tensor::new(std::array::from_fn(|k| row[1 + k])))
                    })
                    .build();
                let mut per_tile = partition_configured_tracers(&global, cfg, &partition);
                let order = configured_ito_order(cfg)?;
                for ((sim, _), &flat) in tiles.iter_mut().zip(&mine) {
                    let set = std::mem::take(&mut per_tile[flat]);
                    match order {
                        Some(order) => {
                            sim.continuous_tracers = Some(
                                symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                                    .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                            )
                        }
                        None => sim.tracers = Some(set),
                    }
                }
            }
            // the declared stationary target, seeded on each tile at its cell centres (t = 0) for the
            // probe the session runs before the restart restores the state
            let has_energy = tiles[0].0.fields.cons.has_energy();
            let expression = declared_equilibrium!(cfg, $d, has_energy);
            let seed = expression.as_ref().map(|expression| {
                move |sim: &Sim| {
                    sim.seed_cells(|x| {
                        let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                        symbi_hydro::state::PrimFromSlots::from_slots(&expression.eval(at(0), at(1), at(2), 0.0))
                    })
                }
            });
            let perturb = |sim: &mut Sim| -> Result<(), String> {
                lay_perturbation!(sim, cfg, $d, false);
                Ok(())
            };
            session_over!(IsoNewtonian, $d, tiles, seed.as_ref().map(|f| f as &dyn Fn(&Sim)), expression.as_ref().map(|_| &perturb as &dyn Fn(&mut Sim) -> Result<(), String>))
        }};
    }
    macro_rules! imhd_tiles {
        ($geom:expr, $geom_ty:ty) => {{
            type Sim = SimDefaultGeneric<IsothermalMhd, $d, 3, $geom_ty, Isothermal<f64>>;
            if bufs.len() < 3 {
                return Err(format!("imhd needs 3 staggered b-field generators, got {}", bufs.len()));
            }
            let mut tiles = Vec::with_capacity(mine.len());
            for &flat in &mine {
                let tc = unflatten(flat, counts);
                // the tile's place on the whole grid, its pole faces on a polar seam made cuts
                let mut tile = whole.place(&partition, tc);
                tile.boundaries = seam_faces_as_cuts(tile.boundaries.clone(), seam.as_ref());
                let (tile_lo, m) = (tile.offset, tile.cells);
                let tile_faces: Vec<Vec<f64>> =
                    (0..$d).map(|d| tile_face_buffer::<$d>(&bufs[d], n, m, tile_lo, d)).collect();
                let sim = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
                    .placed(&whole, &tile)
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("tile {flat} allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&tile_lo, &idx, &n);
                        let row = &prims[lin];
                        let gidx: [isize; $d] =
                            std::array::from_fn(|ax| tile_lo[ax] as isize + idx[ax]);
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, gidx, n) } else { bufs[k][lin] }
                        });
                        MhdPrimG::<f64, 3, IsoModel>::new(
                            PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])),
                            Tensor::new(mag),
                        )
                    })
                    .seed_faces_indexed(&tile_faces)
                    .build();
                // the immersed bodies and their shaped walls, every tile holding the whole set
                let sim = if cfg.bodies.is_empty() { sim } else { attach_bodies_and_shapes!(sim, cfg, $d) };
                // the numerics, sources, driven boundaries and gradient walls the single grid
                // installs, each tile evaluating them at the whole grid's coordinates of its cells
                let settings = KernelSettings::of(cfg, <IsothermalMhd as Regime<f64, $d>>::SPEC);
                let sub = mhd_kernels(sim.substrate(), &settings, LevelRole::Root).map_err(|e| format!("tile {flat}: {e}"))?;
                seed_mhd_local_cs2!(sim, cfg, prims, tile_lo, n, &whole, $d);
                tiles.push((sim, sub));
            }
            // the tracer population seeds once from the whole-grid density, as the single grid's,
            // and each worker keeps its own tiles' share; the whole-grid state lives for the
            // seeding alone
            if cfg.n_tracers > 0 {
                let global = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
                    .placed(&whole, &whole.single())
                    .ghosts(ghost_width(cfg))
                    .cfl(cfg.cfl)
                    .timestepping(cfg.timestepping)
                    .cyl_plane(cfg.cyl_plane)
                    .allocate()
                    .map_err(|e| format!("global seeding sim allocate: {e:?}"))?
                    .set_initial_indexed(|idx, _x| {
                        let lin = global_lin::<$d>(&[0; $d], &idx, &n);
                        let row = &prims[lin];
                        let mag: [f64; 3] = std::array::from_fn(|k| {
                            if k < $d { face_avg_cell_b::<$d>(&bufs[k], k, idx, n) } else { bufs[k][lin] }
                        });
                        MhdPrimG::<f64, 3, IsoModel>::new(PrimG::isothermal(Density(row[0]), Tensor::new([row[1], row[2], row[3]])), Tensor::new(mag))
                    })
                    .seed_faces_indexed(&bufs[0..$d])
                    .build();
                let mut per_tile = partition_configured_tracers(&global, cfg, &partition);
                let order = configured_ito_order(cfg)?;
                for ((sim, _), &flat) in tiles.iter_mut().zip(&mine) {
                    let set = std::mem::take(&mut per_tile[flat]);
                    match order {
                        Some(order) => {
                            sim.continuous_tracers = Some(
                                symbi_sim::tracers::ContinuousTracerSet::from_discrete(&set, order)
                                    .map_err(|detail| format!("continuous tracer partition: {detail}"))?,
                            )
                        }
                        None => sim.tracers = Some(set),
                    }
                }
            }
            session_over!(IsothermalMhd, 3, tiles)
        }};
    }
    // one arm per chart the flat spacetime is gridded on
    // the meridional plane of a black hole on the horizon-penetrating chart, built only on a
    // two-dimensional grid: the rows choose one momentum per axis or the swirl's three, as the
    // single grid's dispatch chooses (spinning kerr always carries the swirl)
    macro_rules! kerr_schild_plane {
        (2, $arm:ident, $regime:expr, $regime_ty:ty) => {{
            let dof = primitive_row_dof(cfg, prims.first().map_or(0, |row| row.len()))?;
            match (cfg.spacetime.as_str(), dof) {
                ("kerr_ks", _) => $arm!(KerrKS { mass: cfg.schwarzschild_mass, spin: cfg.kerr_spin }, KerrKS<f64>, $regime, $regime_ty, 3),
                (_, 3) => $arm!(SchwarzschildKS { mass: cfg.schwarzschild_mass }, SchwarzschildKS<f64>, $regime, $regime_ty, 3),
                _ => $arm!(SchwarzschildKS { mass: cfg.schwarzschild_mass }, SchwarzschildKS<f64>, $regime, $regime_ty, 2),
            }
        }};
        ($other:tt, $arm:ident, $regime:expr, $regime_ty:ty) => {
            Err(format!("the {} spacetime on the {}-dimensional spherical chart is outside the distributed first release", cfg.spacetime, $other))
        };
    }
    // a black hole on the horizon-penetrating cylindrical chart, built as the single grid builds
    // it: the (R, phi) plane with one momentum per axis, the (R, z) plane with the azimuthal
    // momentum, the full three-dimensional chart, schwarzschild or spinning
    // the relativistic magnetized gas on the horizon-penetrating charts, built as the single grid's
    // `mhd_dispatch` builds it: the radial line, the spinning (r, theta) plane, the three-dimensional
    // box and the cylindrical planes, each only on its own dimension
    macro_rules! kerr_schild_mhd {
        (1) => {
            match (cfg.coord_system.as_str(), cfg.spacetime.as_str()) {
                ("spherical", "schwarzschild_ks") => rmhd_tiles!(SchwarzschildKS { mass: cfg.schwarzschild_mass }, SchwarzschildKS<f64>),
                (chart, spacetime) => Err(format!("the {spacetime} spacetime on the 1-dimensional {chart} chart is outside the distributed first release")),
            }
        };
        (2) => {
            match (cfg.coord_system.as_str(), cfg.spacetime.as_str(), cfg.cyl_plane) {
                ("spherical", "kerr_ks", _) => rmhd_tiles!(KerrKS { mass: cfg.schwarzschild_mass, spin: cfg.kerr_spin }, KerrKS<f64>),
                ("cylindrical", "schwarzschild_ks", CylPlane::RPhi) => {
                    rmhd_tiles!(SchwarzschildKSCylindrical { mass: cfg.schwarzschild_mass }, SchwarzschildKSCylindrical<f64>)
                }
                ("cylindrical", "schwarzschild_ks", CylPlane::Rz) => {
                    rmhd_tiles!(SchwarzschildKSCylindricalRz { mass: cfg.schwarzschild_mass }, SchwarzschildKSCylindricalRz<f64>)
                }
                (chart, spacetime, _) => Err(format!("the {spacetime} spacetime on the 2-dimensional {chart} chart is outside the distributed first release")),
            }
        };
        (3) => {
            match (cfg.coord_system.as_str(), cfg.spacetime.as_str()) {
                ("cartesian", "schwarzschild_ks") => {
                    rmhd_tiles!(SchwarzschildKSCartesian { mass: cfg.schwarzschild_mass }, SchwarzschildKSCartesian<f64>)
                }
                ("cartesian", "kerr_ks") => {
                    rmhd_tiles!(KerrKSCartesian { mass: cfg.schwarzschild_mass, spin: cfg.kerr_spin }, KerrKSCartesian<f64>)
                }
                (chart, spacetime) => Err(format!("the {spacetime} spacetime on the 3-dimensional {chart} chart is outside the distributed first release")),
            }
        };
    }
    macro_rules! kerr_schild_cylinder {
        (2, $arm:ident, $regime:expr, $regime_ty:ty) => {
            match (cfg.spacetime.as_str(), cfg.cyl_plane) {
                ("schwarzschild_ks", CylPlane::RPhi) => {
                    $arm!(SchwarzschildKSCylindrical { mass: cfg.schwarzschild_mass }, SchwarzschildKSCylindrical<f64>, $regime, $regime_ty, 2)
                }
                ("schwarzschild_ks", CylPlane::Rz) => {
                    $arm!(SchwarzschildKSCylindricalRz { mass: cfg.schwarzschild_mass }, SchwarzschildKSCylindricalRz<f64>, $regime, $regime_ty, 3)
                }
                (spacetime, _) => Err(format!("the {spacetime} spacetime on the 2-dimensional cylindrical chart is outside the distributed first release")),
            }
        };
        (3, $arm:ident, $regime:expr, $regime_ty:ty) => {
            match cfg.spacetime.as_str() {
                "kerr_ks" => $arm!(KerrKSCylindrical { mass: cfg.schwarzschild_mass, spin: cfg.kerr_spin }, KerrKSCylindrical<f64>, $regime, $regime_ty, 3),
                _ => $arm!(SchwarzschildKSCylindrical { mass: cfg.schwarzschild_mass }, SchwarzschildKSCylindrical<f64>, $regime, $regime_ty, 3),
            }
        };
        ($other:tt, $arm:ident, $regime:expr, $regime_ty:ty) => {
            Err(format!("the {} spacetime on the {}-dimensional cylindrical chart is outside the distributed first release", cfg.spacetime, $other))
        };
    }
    macro_rules! charted {
        ($arm:ident) => {
            match cfg.coord_system.as_str() {
                "cartesian" => $arm!(Cartesian, Cartesian),
                "spherical" => $arm!(Spherical, Spherical),
                "cylindrical" => $arm!(Cylindrical, Cylindrical),
                other => Err(format!("the {other} chart is outside the distributed first release")),
            }
        };
        // a gas regime also on the horizon-penetrating charts of a black hole: the three-dimensional
        // kerr-schild box and the radial line, charts the single grid bakes with one momentum per axis
        ($arm:ident, $regime:expr, $regime_ty:ty) => {
            match (cfg.coord_system.as_str(), cfg.spacetime.as_str(), $d) {
                ("cartesian", "minkowski", _) => $arm!(Cartesian, Cartesian, $regime, $regime_ty),
                ("spherical", "minkowski", _) => $arm!(Spherical, Spherical, $regime, $regime_ty),
                ("cylindrical", "minkowski", _) => $arm!(Cylindrical, Cylindrical, $regime, $regime_ty),
                ("cartesian", "schwarzschild_ks", 3) => $arm!(
                    SchwarzschildKSCartesian { mass: cfg.schwarzschild_mass },
                    SchwarzschildKSCartesian<f64>,
                    $regime,
                    $regime_ty
                ),
                ("cartesian", "kerr_ks", 3) => $arm!(
                    KerrKSCartesian { mass: cfg.schwarzschild_mass, spin: cfg.kerr_spin },
                    KerrKSCartesian<f64>,
                    $regime,
                    $regime_ty
                ),
                ("spherical", "schwarzschild_ks", 1) => $arm!(
                    SchwarzschildKS { mass: cfg.schwarzschild_mass },
                    SchwarzschildKS<f64>,
                    $regime,
                    $regime_ty
                ),
                ("spherical", "schwarzschild_ks" | "kerr_ks", 2) => kerr_schild_plane!($d, $arm, $regime, $regime_ty),
                ("cylindrical", "schwarzschild_ks" | "kerr_ks", _) => kerr_schild_cylinder!($d, $arm, $regime, $regime_ty),
                (chart, "minkowski", _) => Err(format!("the {chart} chart is outside the distributed first release")),
                (chart, spacetime, d) => Err(format!(
                    "the {spacetime} spacetime on the {d}-dimensional {chart} chart is outside the distributed first release"
                )),
            }
        };
    }
    // the tiles this worker holds, built as the single path builds them, per regime and chart
    match cfg.regime.as_str() {
        "newtonian" => charted!(gas_tiles, Newtonian, Newtonian),
        "rhd" => charted!(gas_tiles, Rhd, Rhd),
        "nmhd" => charted!(nmhd_tiles),
        "rmhd" if cfg.spacetime != "minkowski" => kerr_schild_mhd!($d),
        "rmhd" => charted!(rmhd_tiles),
        "isothermal" => charted!(iso_tiles),
        "imhd" => charted!(imhd_tiles),
        other => Err(format!("the {other} regime is outside the distributed first release")),
    }
        }};
    }
    match cfg.dims {
        1 => worker_of_dims!(1),
        2 => worker_of_dims!(2),
        3 => worker_of_dims!(3),
        other => Err(format!("a {other}-dimensional grid is outside the distributed first release")),
    }
}

/// the fabric session of one worker over its built tiles, whatever the regime: the
/// rendezvous under the session credential, the worker loop with the coordinator-written
/// checkpoints at the linear cadence, and the final checkpoint before the session ends.
/// worker 0 reports through the decomposed table.
#[allow(clippy::too_many_arguments)]
fn worker_session<R, G, E, K, const D: usize, const DOF: usize>(
    cfg: &Config,
    launch: &WorkerLaunch,
    fault: LaunchFault,
    drained: std::time::Duration,
    built: std::time::Duration,
    partition: &Partition<D>,
    placement: symbi_sim::atlas::Placement,
    topology: &symbi::sim::decomp::Topology<D>,
    me: symbi_fabric::WorkerId,
    mine: &[usize],
    offsets: &[[isize; D]],
    mut tiles: Vec<(SimDefaultGeneric<R, D, DOF, G, E>, K)>,
    seed_target: Option<&dyn Fn(&SimDefaultGeneric<R, D, DOF, G, E>)>,
    perturb_over_target: Option<&dyn Fn(&mut SimDefaultGeneric<R, D, DOF, G, E>) -> Result<(), String>>,
) -> Result<(), String>
where
    symbi::symbi_geometry::Cartesian: symbi::symbi_geometry::Metric<f64, D>,
    R: Regime<f64, D>,
    G: symbi_geometry::Metric<f64, D> + Copy,
    E: Eos<f64>,
    K: symbi_sim::substrate_seam::KernelSet<D, DOF, DefaultMemory, f64>,
{
    use symbi::sim::decomp::{LocalCopy, plan_schema};
    use symbi_fabric::rendezvous::{Identity, Rendezvous, connect};
    use symbi_fabric::{Fabric, WorkerId};
    use symbi_sim::atlas::{ExchangePlan, TileId};
    use symbi_sim::checkpoint::{LevelTiles, PhysicsIdentity, TileView};
    use symbi_sim::distributed_checkpoint::{AuditBasis, CheckpointRequest, audited_checkpoint, block_credit_for};
    use symbi_sim::plan_exchange::PlanExchange;
    use symbi_sim::phase_table::PhaseTable;
    use symbi_sim::substrate_seam::RegimeKind;
    use symbi_sim::worker::{WorkerConfig, WorkerError, evolve_worker_traced};

    let n: [usize; D] = std::array::from_fn(|ax| cfg.n_cells[ax]);
    let total: usize = n.iter().product();
    let counts = partition.counts();
    let ng = tiles[0].0.geom.ng;
    let grid = decomposed_grid::<D>(cfg, ng);
    let identity = PhysicsIdentity::of(&tiles[0].0);
    let tile_ids: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
    let my_offsets: Vec<[isize; D]> = mine.iter().map(|&f| offsets[f]).collect();
    let schema = plan_schema(&*tiles[0].0);
    let seam = polar_seam::<D>(&cfg.coord_system, &boundaries_nd::<D>(&cfg.boundaries), topology, partition.counts(), |az, i| partition.tile_range(az, i).1);
    let plan = ExchangePlan::compile_with_seam(partition, topology, &schema, ng, seam).map_err(|e| e.to_string())?;
    let table = PhaseTable::flat(cfg.timestepping.stages().len() as u8).map_err(|e| e.to_string())?;
    let exchange = PlanExchange::new(&plan, &placement, me, table.clone());
    // a continuous-tracer population's coefficients cross the cuts through their own plan, its
    // fabric ids after the state plan's
    let continuous_traced = cfg.n_tracers > 0 && configured_ito_order(cfg)?.is_some();
    let coefficient_plan = if continuous_traced {
        let schema = (0..3 * D).fold(symbi_sim::atlas::FieldSchema::new(), |schema, ii| schema.cell(&format!("ito_coefficient_{ii}")));
        let seam = polar_seam::<D>(&cfg.coord_system, &boundaries_nd::<D>(&cfg.boundaries), topology, partition.counts(), |az, i| partition.tile_range(az, i).1);
        Some(ExchangePlan::compile_with_seam(partition, topology, &schema, ng, seam).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let coefficients = coefficient_plan.as_ref().map(|p| {
        PlanExchange::new(p, &placement, me, table.clone())
            .with_role(symbi_sim::plan_exchange::PlanRole::TracerCoefficients, exchange.lens().len() as u32)
    });
    let budget = launch.staging_cells.max(1);
    let credit = block_credit_for(budget, D);
    let bind: std::net::IpAddr = launch
        .bind
        .parse()
        .map_err(|e| format!("bind address '{}': {e}", launch.bind))?;
    let rendezvous = Rendezvous {
        me,
        coordinator: std::net::SocketAddr::new(bind, 0),
        announce: Some(std::path::PathBuf::from(&launch.rendezvous)),
        bind: Some(bind),
        advertise: Some(launch.advertise.clone()),
        identity: Identity {
            credential: launch.credential,
            build_id: build_source_id(),
            config_digest: symbi_fabric::Digest(launch.config_digest),
            plan_digest: plan.digest,
            placement_digest: placement.digest(),
            phase_table_digest: exchange.table().digest(),
            ng: ng as u32,
            workers: launch.workers as u32,
            block_credit: credit,
        },
        startup: std::time::Duration::from_secs(120),
    };
    let deadline = std::time::Duration::from_secs(600);
    let mut lens = exchange.lens().to_vec();
    if let Some(c) = coefficients.as_ref() {
        lens.extend(c.lens().iter().copied());
    }
    let max_payload = (lens.iter().max().copied().unwrap_or(0) * 8).max(credit as usize);
    let connect_clock = std::time::Instant::now();
    let (link, session) = connect(&rendezvous, max_payload).map_err(|e| e.to_string())?;
    // the startup timeline, one write: the generator drain (python, under the interpreter
    // lock), the tile build with its kernel compilation, and the rendezvous
    {
        use std::io::Write;
        let line = format!(
            "SIMBI launch startup: worker={}/{} drain={:.3}s build={:.3}s connect={:.3}s cpus={}\n",
            launch.worker,
            launch.workers,
            drained.as_secs_f64(),
            built.as_secs_f64(),
            connect_clock.elapsed().as_secs_f64(),
            std::thread::available_parallelism().map_or(0, |n| n.get()),
        );
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
    let sends_per_peer_axis = exchange.sends_per_peer_axis();
    let mut fabric = match coefficients.as_ref() {
        Some(c) => symbi_sim::plan_exchange::fabric_for(&[&exchange, c], link, session),
        None => exchange.fabric(link, session),
    }
    .with_block_credit(credit);
    // the whole domain's faces and boundary conditions the continuous tracers meet
    let continuous_context = coefficients.as_ref().map(|c| symbi_sim::worker::ContinuousTracerContext {
        coefficients: c,
        bounds: symbi_sim::worker_tracers::whole_bounds(&tiles[0].0.geom, n),
        boundaries: {
            let phys = boundaries_nd::<D>(&cfg.boundaries);
            symbi_sim::state::Boundaries::per_axis(std::array::from_fn(|dd| {
                if topology.is_periodic(dd) {
                    [symbi_sim::state::BoundaryType::Periodic; 2]
                } else {
                    [phys.lo(dd), phys.hi(dd)]
                }
            }))
        },
    });
    let devices = vec![0i32; partition.n_tiles()];
    let ng_session = tiles[0].0.geom.ng;
    let restart_grid = decomposed_grid::<D>(cfg, ng_session);
    // a declared stationary target's imbalance, read off one probe stage of the launched pipeline
    // before the restart restores the state, as the single hierarchy reads its level's: the
    // tiles hold the target, the probe primes and advances it at the smallest cfl step of any
    // tile, and the tiles' own state comes back
    let residuals: Option<Vec<symbi_sim::state::ConsFieldsGeneric<D, DOF, DefaultMemory>>> = match seed_target {
        None => None,
        Some(seed) => {
            use symbi::sim::refinement::equilibrium::{imbalance_from_stage, overwrite, restore_gas_state, save_gas_state};
            if symbi_sim::guard_ledger::scope_is_open() {
                return Err("the stationary target's probe ran inside a guard scope, where its acts would be booked".to_string());
            }
            let saved = tiles
                .iter()
                .map(|(s, _)| save_gas_state(&**s))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("stationary target save: {e:?}"))?;
            for (sim, _) in tiles.iter() {
                seed(sim);
            }
            let tile_ids: Vec<TileId> = mine.iter().map(|&f| TileId(f as u32)).collect();
            let sh: Vec<&symbi_sim::state::FieldStore<D, DOF, DefaultMemory, f64>> = tiles.iter().map(|(s, _)| &**s).collect();
            let kernels: Vec<&K> = tiles.iter().map(|(_, k)| k).collect();
            let deadline = std::time::Duration::from_secs(600);
            let mut anchors = Vec::new();
            let probe_dt = symbi_sim::worker::probe_equilibrium(
                &exchange,
                &tile_ids,
                &sh,
                &kernels,
                &devices,
                &LocalCopy,
                &mut fabric,
                deadline,
                &mut |fabric: &mut Fabric<_>| {
                    anchors = sh
                        .iter()
                        .map(|s| save_gas_state(*s))
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| WorkerError::Numerics(format!("stationary target anchor: {e:?}")))?;
                    for (store, kernel) in sh.iter().zip(&kernels) {
                        kernel.wave_speeds(*store);
                        for dd in 0..D {
                            kernel.flux(*store, dd);
                        }
                    }
                    let local = sh.iter().zip(&kernels).map(|(store, kernel)| kernel.cfl(*store)).fold(f64::INFINITY, f64::min);
                    let agreed = f64::from_bits(fabric.collective(symbi_fabric::OpKind::Min, local.to_bits(), deadline)?);
                    if !(agreed.is_finite() && agreed > 0.0) {
                        return Err(WorkerError::Numerics(format!(
                            "the stationary target has no finite cfl step ({agreed:.3e}), so it cannot be advanced to read off its imbalance"
                        )));
                    }
                    Ok(agreed)
                },
            )
            .map_err(|e| format!("the stationary target is not a state the scheme can carry: {e}"))?;
            // every worker learns whether any tile's probe left an unrecoverable state
            let unrecoverable = sh.iter().any(|store| symbi_sim::hydro_ops::scan_c2p_errors(*store).is_err());
            if fabric.collective(symbi_fabric::OpKind::Any, u64::from(unrecoverable), deadline).map_err(|e| e.to_string())? != 0 {
                return Err("advancing the stationary target by one cfl step left an unrecoverable state, so its imbalance cannot be read off a linear response".to_string());
            }
            if sh.iter().any(|store| store.immersed.as_ref().is_some_and(|im| !im.diagnostics.consolidate().is_empty())) {
                return Err("the stationary target's probe booked a body delta".to_string());
            }
            eprintln!("SIMBI equilibrium probe: level=0 dt_bits={:016x}", probe_dt.to_bits());
            let residuals = sh
                .iter()
                .zip(&anchors)
                .map(|(store, anchor)| imbalance_from_stage(anchor.cons(), &store.fields.cons, &store.geom.allocated, probe_dt))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("stationary target residual: {e:?}"))?;
            for ((s, _), state) in tiles.iter().zip(&saved) {
                restore_gas_state(&**s, state);
            }
            // a restart restores the checkpoint below, so the seed applies only to a fresh start
            if cfg.seed_from_equilibrium {
                for ((s, _), anchor) in tiles.iter().zip(&anchors) {
                    overwrite(anchor.cons(), &s.fields.cons, &s.geom.allocated);
                }
            }
            Some(residuals)
        }
    };
    // the declared perturbation over the target, as the single grid lays it after the target
    if let Some(perturb) = perturb_over_target {
        for (sim, _) in tiles.iter_mut() {
            perturb(sim)?;
        }
    }
    // the restart from a partitioned checkpoint, after the stationary target's probe
    if let Some(path) = cfg.restart_path.as_deref() {
        for (&flat, (sim, _)) in mine.iter().zip(tiles.iter_mut()) {
            symbi_sim::checkpoint::load_partitioned_level(sim, path, 0, offsets[flat], &restart_grid)
                .map_err(|e| format!("restart of tile {flat} from '{path}': {e}"))?;
        }
    }

    // the coordinator reports through the decomposed table
    let slabs: Vec<Vec<usize>> = (0..D)
        .map(|ax| (0..counts[ax]).map(|i| partition.tile_range(ax, i).1).collect())
        .collect();
    let mut reporter = if me == WorkerId(0) {
        let mut r = DecomposedReporter::new(cfg, total as u64, &counts, &slabs);
        r.milestone(&format!(
            "fabric session over {} workers; plan {:016x}, placement {:016x}",
            launch.workers, plan.digest.0, placement.digest().0
        ));
        Some(r)
    } else {
        None
    };
    // the checkpoint schedule the single grid walks
    let mut cadence = CheckpointCadence::of(cfg);
    let cp_width = checkpoint_time_width(cfg);
    let extras_of = |cp_index: u64| checkpoint_metadata(cfg, cp_index);

    // one checkpoint through the coordinator from this worker's tiles as they stand: the
    // shared interface faces are audited first when the plan has any, then the file is
    // written. the basis says whether the state is the primed one, written before the march
    // and certified by the prime audit, or the one that completed a step. returns the audit's
    // and the checkpoint's own wall times apart.
    let audit_total = std::cell::Cell::new((0u64, symbi_sim::plan_exchange::AuditCost::default()));
    // the last step whose state an audit certified: a second checkpoint of the same step (the
    // final one after a cadence write) reuses that certification, its audit epoch being spent
    let certified = std::cell::Cell::new(None::<u64>);
    let write = |sh: &[&symbi_sim::state::FieldStore<D, DOF, DefaultMemory, f64>],
                 fabric: &mut Fabric<_>,
                 path: &str,
                 cp_index: u64,
                 basis: AuditBasis|
     -> Result<f64, String> {
        let basis = match basis {
            AuditBasis::AfterStep(step) if certified.get() == Some(step) => AuditBasis::Certified(step),
            AuditBasis::AfterStep(step) => {
                certified.set(Some(step));
                basis
            }
            other => other,
        };
        let level = LevelTiles {
            tiles: sh
                .iter()
                .zip(&my_offsets)
                .map(|(s, &offset)| TileView { state: *s, offset })
                .collect(),
            grid: grid.clone(),
            header: None,
        };
        let extras = extras_of(cp_index);
        let req = CheckpointRequest {
            path,
            extras: &extras,
            budget,
            deadline,
            drop_block: None,
            duplicate_first_block: false,
            vote_no: false,
        };
        let cost = audited_checkpoint::<R, D, DOF, _, _>(
            fabric, &exchange, &identity, &level, &tile_ids, &req, basis,
        )
        .map_err(|e| checkpoint_write_error(path, symbi_io::IoError::Backend(e.to_string())))?;
        if basis != AuditBasis::Primed && exchange.has_audits() {
            let (n, mut sum) = audit_total.get();
            sum.faces += cost.audit.faces;
            sum.wire_bytes += cost.audit.wire_bytes;
            sum.elapsed += cost.audit.elapsed;
            audit_total.set((n + 1, sum));
        }
        Ok(cost.checkpoint.as_secs_f64())
    };

    let mut stores: Vec<&mut symbi_sim::state::FieldStore<D, DOF, DefaultMemory, f64>> = Vec::new();
    let mut kernels = Vec::new();
    for (s, k) in tiles.iter_mut() {
        stores.push(&mut **s);
        kernels.push(&*k);
    }
    // the body diagnostics, on the coordinator alone: every worker's bodies are the same, at the
    // cadence the single grid keeps
    let diag_path = if me == WorkerId(0) && cfg.diagnostic_interval > 0.0 && !cfg.bodies.is_empty() {
        Some(format!("{}diagnostics.dat", cfg.data_dir))
    } else {
        None
    };
    let diag_interval = (cfg.diagnostic_interval * cfg.time_unit).max(f64::MIN_POSITIVE);
    let diag_plan = diag_path.as_ref().map(|dp| {
        plan_diagnostics(last_diagnostics_time(dp), cfg.start_time, diag_interval)
            .unwrap_or_else(|detail| panic!("{detail}"))
    });
    let mut next_diag = diag_plan.as_ref().map_or(diag_interval, |p| p.next);
    let outcome: Result<(u64, f64, Option<String>), String> = (|| {
        if cfg.checkpoint_index == 0 || cfg.start_time == 0.0 {
            let sh: Vec<&symbi_sim::state::FieldStore<D, DOF, DefaultMemory, f64>> = stores.iter().map(|s| &**s).collect();
            // the builders seed the conserved state; the initial checkpoint publishes the
            // primitives the march starts from, recovered on every held tile as the single
            // grid's priming does (a recovery touches no face)
            for (s, k) in sh.iter().zip(&kernels) {
                k.c2p(s);
            }
            let tag = cadence.tag(cfg, cp_width, cfg.start_time, cfg.checkpoint_index);
            let path = checkpoint_name(cfg, &tag);
            // before the march: the prime audit certified this state and nothing has touched
            // a face since
            let io = write(&sh, &mut fabric, &path, cfg.checkpoint_index, AuditBasis::Primed)?;
            if let Some(r) = reporter.as_mut() {
                r.checkpoint_written(cfg, &path, cfg.start_time, io);
            }
        }
        // the start-time body diagnostics row on a fresh file (or a tail behind the resume clock)
        if let Some(dp) = &diag_path {
            if diag_plan.as_ref().is_some_and(|p| p.write_initial) {
                if let Some(im) = stores[0].immersed.as_ref() {
                    let _ = append_diagnostics(dp, cfg.start_time, &im.bodies);
                }
            }
        }
        let worker_cfg = WorkerConfig {
            regime: RegimeKind::of::<f64, D, R>(),
            timestepping: cfg.timestepping,
            start_time: cfg.start_time,
            t_final: cfg.t_final,
            max_steps: cfg.max_steps,
            deadline,
            injection: fault.injection,
            // SIMBI_FABRIC_AUDIT=all audits the shared interface faces after every exchange
            // point; the default audits them after the prime exchange
            audit_every_point: std::env::var("SIMBI_FABRIC_AUDIT").is_ok_and(|v| v == "all"),
        };
        let guard = retry_guard_of(cfg);
        let exclusion = retry_exclusion_of(cfg);
        for store in stores.iter_mut() {
            store.retry_guard = guard;
            store.retry_exclusion = exclusion;
        }
        if launch.worker == 0 {
            announce_retry_guard(&*stores[0]);
        }
        let report = evolve_worker_traced(
            &tile_ids,
            &mut stores,
            &kernels,
            &devices,
            &exchange,
            &LocalCopy,
            &mut fabric,
            &worker_cfg,
            continuous_context.as_ref(),
            // the declared target's residual, added back before the recovery of every stage
            &mut |tile: usize,
                  store: &symbi_sim::state::FieldStore<D, DOF, DefaultMemory, f64>,
                  point: symbi_sim::stage::HookPoint,
                  args: symbi_sim::stage::StageArgs| {
                if let (symbi_sim::stage::HookPoint::BeforeC2p, Some(residuals)) = (point, residuals.as_ref()) {
                    symbi::sim::refinement::equilibrium::accumulate(
                        &residuals[tile],
                        &store.fields.cons,
                        &store.geom.interior,
                        args.ac * args.dt,
                    );
                }
            },
            |iter, time, sh, fabric| {
                if fault.exit_after == Some(iter) {
                    // the process ends here as a lost node would: no relay, no unwinding, the
                    // sockets closed by the operating system
                    std::process::exit(86);
                }
                if fault.perturb_face_after == Some(iter) {
                    if let Some(mhd) = sh[0].fields.mhd.as_ref() {
                        // the second x face of the tile's first interior row, one cell up
                        let field = &mhd.bface.b[0];
                        let at: [isize; D] =
                            std::array::from_fn(|ax| sh[0].geom.interior.spaces[ax].lo + isize::from(ax == 1));
                        let old = *field.view().at(at);
                        *field.view_mut().at_mut(at) = f64::from_bits(old.to_bits() + 1);
                    }
                }
                if let Some(r) = reporter.as_mut() {
                    r.progress(iter, time);
                }
                // the step that reaches the `max_steps` root iteration publishes through the
                // final snapshot alone, as on the single grid
                if step_cap_reached(cfg, sh[0].iteration) {
                    return Ok(std::ops::ControlFlow::Continue(()));
                }
                if cadence.due(time) {
                    let tag = cadence.tag(cfg, cp_width, time, cadence.index);
                    let path = checkpoint_name(cfg, &tag);
                    // the state completed step `iter`, whose exchanges ran under step index iter - 1
                    let io = write(sh, fabric, &path, cadence.index, AuditBasis::AfterStep(iter - 1)).map_err(WorkerError::Checkpoint)?;
                    if let Some(r) = reporter.as_mut() {
                        r.checkpoint_written(cfg, &path, time, io);
                    }
                    cadence.advance(time);
                }
                if let Some(dp) = &diag_path {
                    if diagnostics_due(time, &mut next_diag, diag_interval) {
                        if let Some(im) = sh[0].immersed.as_ref() {
                            let _ = append_diagnostics(dp, time, &im.bodies);
                        }
                    }
                }
                Ok(std::ops::ControlFlow::Continue(()))
            },
        )
        .map_err(|e| e.to_string())?;
        let sh: Vec<&symbi_sim::state::FieldStore<D, DOF, DefaultMemory, f64>> = stores.iter().map(|s| &**s).collect();
        // a march every worker agreed to halt publishes its step entry, the last accepted state,
        // as crashed in place of the final snapshot
        let outcome = if report.halt.is_some() { CheckpointOutcome::Crashed } else { CheckpointOutcome::Completed };
        let final_path = checkpoint_name(cfg, checkpoint_status_tag(outcome));
        let io = write(
            &sh,
            &mut fabric,
            &final_path,
            cadence.index,
            report.steps.checked_sub(1).map_or(AuditBasis::Primed, AuditBasis::AfterStep),
        )?;
        if let Some(r) = reporter.as_mut() {
            r.checkpoint_written(cfg, &final_path, report.time, io);
        }
        fabric.finish(deadline).map_err(|e| e.to_string())?;
        // one write, so the lines of concurrent workers sharing a stream stay whole
        {
            use std::io::Write;
            let t = report.timing;
            // the interface audits run before each checkpoint transaction opened, apart from
            // the checkpoints' own time: their count, the faces compared, the bytes sent, and
            // their wall time
            let (audits, audit) = audit_total.get();
            let line = format!(
                "SIMBI launch: worker={}/{} steps={} rejections={} compute={:.6}s collectives={:.6}s \
                 exchange={:.6}s final_checkpoint={:.6}s checkpoint_audits={} audit_faces={} \
                 audit_bytes={} checkpoint_audit={:.6}s\n",
                launch.worker,
                launch.workers,
                report.steps,
                report.rejections,
                t.compute.as_secs_f64(),
                t.collectives.as_secs_f64(),
                t.exchange.as_secs_f64(),
                io,
                audits,
                audit.faces,
                audit.wire_bytes,
                audit.elapsed.as_secs_f64(),
            );
            let _ = std::io::stderr().write_all(line.as_bytes());
            // the tracers that crossed tiles, every move and those through a corner
            if cfg.n_tracers > 0 {
                let line = format!(
                    "SIMBI launch tracers: worker={}/{} migrations={} corner_migrations={}
",
                    launch.worker, launch.workers, report.tracer_migrations.moved, report.tracer_migrations.corner,
                );
                let _ = std::io::stderr().write_all(line.as_bytes());
            }
            // the halo frames one exchange point sends each peer on each axis, and how the
            // session's waiting passes divided between advancing and idle
            let st = fabric.stats();
            let line = format!(
                "SIMBI launch link: worker={}/{} sends_per_peer_axis={:?} halo_frames_sent={} \
                 halo_frames_received={} passes={} advancing_passes={} idle_pauses={}\n",
                launch.worker,
                launch.workers,
                sends_per_peer_axis,
                st.halo_frames_sent,
                st.halo_frames_received,
                st.passes,
                st.advancing_passes,
                st.idle_pauses,
            );
            let _ = std::io::stderr().write_all(line.as_bytes());
        }
        Ok((report.steps, report.time, report.halt))
    })();
    match outcome {
        Ok((iter, time, Some(cause))) => halted_launch(cfg, reporter, iter, time, &cause),
        Ok((iter, time, None)) => {
            if let Some(r) = reporter {
                let final_path = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Completed));
                r.finish(iter, time, &final_path);
            }
            Ok(())
        }
        Err(msg) => {
            fabric.abort(&msg);
            if let Some(r) = reporter {
                r.fail(&msg);
            }
            Err(msg)
        }
    }
}

/// the failing end of a launch every worker agreed to halt: the session finished after the
/// crashed snapshot of the step entry, so the halt is reported and the worker exits failing
fn halted_launch(cfg: &Config, reporter: Option<DecomposedReporter>, iter: u64, time: f64, cause: &str) -> Result<(), String> {
    let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
    let summary = format!("crashed — {iter} steps, t = {time:.4} — {cause} · state {crashed}");
    if let Some(r) = reporter {
        r.fail(&summary);
    }
    Err(summary)
}

/// the canonical form of one execution-configuration value. the supported set is exactly
/// what the execution dict carries: none, booleans, integers, floats, strings (including
/// the string-valued enums, which encode as their value), and sequences and mappings of
/// those. a mapping encodes its entries in key order, so insertion order plays no part; a
/// sequence keeps its order, since a sequence's order is data.
#[derive(Debug, Clone, PartialEq)]
enum Canon {
    None,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Seq(Vec<Canon>),
    Map(Vec<(String, Canon)>),
}

fn canon_of(value: &Bound<'_, PyAny>, path: &str) -> PyResult<Canon> {
    use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PyString, PyTuple};
    if value.is_none() {
        return Ok(Canon::None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Canon::Bool(value.extract()?));
    }
    if value.is_instance_of::<PyFloat>() {
        return Ok(Canon::Float(value.extract()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(Canon::Str(value.extract()?));
    }
    if let Ok(i) = value.extract::<i128>() {
        return Ok(Canon::Int(i));
    }
    if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        let mut items = Vec::new();
        for (i, item) in value.try_iter()?.enumerate() {
            items.push(canon_of(&item?, &format!("{path}[{i}]"))?);
        }
        return Ok(Canon::Seq(items));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut entries = Vec::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            let Ok(key) = key.extract::<String>() else {
                return Err(PyValueError::new_err(format!(
                    "configuration key at {path} is not a string"
                )));
            };
            let canon = canon_of(&item, &format!("{path}.{key}"))?;
            entries.push((key, canon));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        return Ok(Canon::Map(entries));
    }
    // a prescribed-gradient wall (`Neumann` / `Robin`): its kind and its per-primitive
    // coefficients, which are floats and tuples of floats
    if let (Ok(kind), Ok(rho), Ok(velocity), Ok(pressure)) = (
        value.getattr("value"),
        value.getattr("rho"),
        value.getattr("velocity"),
        value.getattr("pressure"),
    ) {
        if let Ok(kind) = kind.extract::<String>() {
            if matches!(kind.as_str(), "neumann" | "robin") {
                return Ok(Canon::Map(vec![
                    ("kind".to_string(), Canon::Str(kind)),
                    ("pressure".to_string(), canon_of(&pressure, &format!("{path}.pressure"))?),
                    ("rho".to_string(), canon_of(&rho, &format!("{path}.rho"))?),
                    ("velocity".to_string(), canon_of(&velocity, &format!("{path}.velocity"))?),
                ]));
            }
        }
    }
    Err(PyValueError::new_err(format!(
        "configuration value at '{path}' has unsupported type {}",
        value.get_type().name()?
    )))
}

/// the canonical byte encoding, folded into the digest: a tag per value, lengths before
/// containers and strings, integers as 128-bit little-endian, floats as their bit pattern.
fn encode_canon(value: &Canon, h: &mut symbi_fabric::Fnv) {
    match value {
        Canon::None => h.u8(0),
        Canon::Bool(b) => {
            h.u8(1);
            h.u8(u8::from(*b));
        }
        Canon::Int(i) => {
            h.u8(2);
            h.bytes(&i.to_le_bytes());
        }
        Canon::Float(f) => {
            h.u8(3);
            h.u64(f.to_bits());
        }
        Canon::Str(s) => {
            h.u8(4);
            h.str(s);
        }
        Canon::Seq(items) => {
            h.u8(5);
            h.u32(items.len() as u32);
            for item in items {
                encode_canon(item, h);
            }
        }
        Canon::Map(entries) => {
            h.u8(6);
            h.u32(entries.len() as u32);
            for (key, item) in entries {
                h.str(key);
                encode_canon(item, h);
            }
        }
    }
}

fn digest_of_canon(source_sha256: &str, canon: &Canon) -> u64 {
    let mut h = symbi_fabric::Fnv::new();
    h.str(source_sha256);
    encode_canon(canon, &mut h);
    h.finish().0
}

/// the digest of the effective execution configuration: every resolved flag and
/// environment-derived parameter, canonically encoded after the source digest. two workers
/// agree on it exactly when their execution dicts are equal as values.
fn effective_config_digest(sim_info: &Bound<'_, PyDict>, source_sha256: &str) -> PyResult<u64> {
    let canon = canon_of(sim_info.as_any(), "")?;
    Ok(digest_of_canon(source_sha256, &canon))
}

/// the effective configuration digest of an execution dict, as the launch handshake
/// computes it: mappings in key order, sequences in order, unsupported values refused.
#[pyfunction]
fn config_digest(sim_info: &Bound<'_, PyDict>) -> PyResult<u64> {
    let source = sim_info
        .get_item("config_sha256")?
        .map(|v| v.extract::<String>())
        .transpose()?
        .unwrap_or_default();
    effective_config_digest(sim_info, &source)
}

#[cfg(test)]
mod canon_tests {
    use super::{Canon, digest_of_canon};

    fn nested(order: &[(&str, Canon)]) -> Canon {
        let mut entries: Vec<(String, Canon)> = order.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        Canon::Map(entries)
    }

    #[test]
    fn insertion_order_plays_no_part_at_any_depth() {
        let inner_a = nested(&[("x", Canon::Int(1)), ("y", Canon::Float(0.5))]);
        let inner_b = nested(&[("y", Canon::Float(0.5)), ("x", Canon::Int(1))]);
        let a = nested(&[("cfl", Canon::Float(0.4)), ("inner", inner_a)]);
        let b = nested(&[("inner", inner_b), ("cfl", Canon::Float(0.4))]);
        assert_eq!(digest_of_canon("s", &a), digest_of_canon("s", &b));
    }

    #[test]
    fn a_changed_numerical_parameter_changes_the_digest() {
        let a = nested(&[("cfl", Canon::Float(0.4)), ("theta", Canon::Float(1.5))]);
        let b = nested(&[("cfl", Canon::Float(0.4000001)), ("theta", Canon::Float(1.5))]);
        assert_ne!(digest_of_canon("s", &a), digest_of_canon("s", &b));
        let c = nested(&[("cfl", Canon::Float(0.4)), ("theta", Canon::Float(1.5)), ("n", Canon::Int(64))]);
        let d = nested(&[("cfl", Canon::Float(0.4)), ("theta", Canon::Float(1.5)), ("n", Canon::Int(65))]);
        assert_ne!(digest_of_canon("s", &c), digest_of_canon("s", &d));
    }

    #[test]
    fn sequence_order_and_types_are_data() {
        let a = Canon::Seq(vec![Canon::Int(1), Canon::Int(2)]);
        let b = Canon::Seq(vec![Canon::Int(2), Canon::Int(1)]);
        assert_ne!(digest_of_canon("s", &a), digest_of_canon("s", &b));
        assert_ne!(digest_of_canon("s", &Canon::Int(1)), digest_of_canon("s", &Canon::Float(1.0)));
        assert_ne!(digest_of_canon("s", &Canon::Bool(true)), digest_of_canon("s", &Canon::Int(1)));
        assert_ne!(digest_of_canon("a", &Canon::None), digest_of_canon("b", &Canon::None));
    }
}

/// drain the tracer-cohort generator (None = no cohort field): one label per interior cell in
/// the gas state's traversal order, each in 0..=65534 (65535 marks injected material)
fn drain_cohort(cohort_field: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<u16>> {
    let Some(cohort_field) = cohort_field.filter(|field| !field.is_none()) else {
        return Ok(Vec::new());
    };
    let mut vals = Vec::new();
    for value in cohort_field.try_iter()? {
        let cohort = value?.extract::<i64>()?;
        vals.push(u16::try_from(cohort).map_err(|_| {
            PyValueError::new_err(format!(
                "tracer cohort {cohort} is outside the supported range 0..=65534"
            ))
        })?);
    }
    if vals.iter().any(|&cohort| cohort == u16::MAX) {
        return Err(PyValueError::new_err(
            "tracer cohort 65535 is reserved for injected material",
        ));
    }
    Ok(vals)
}

/// one worker of a `simbi launch` session. `launch` carries the worker id, the worker count,
/// the tile owner map, the per-axis cuts, the rendezvous file, the session credential (zero
/// leaves it to the rendezvous file), the bind address and advertised host, and the checkpoint
/// block size in cells.
#[pyfunction]
#[pyo3(signature = (prim_gen, staggered_bfields, sim_info, launch, chi_field=None, a=None, adot=None, cohort_field=None))]
#[allow(clippy::too_many_arguments)]
fn launch_worker(
    py: Python<'_>,
    prim_gen: &Bound<'_, PyAny>,
    staggered_bfields: &Bound<'_, PyAny>,
    sim_info: &Bound<'_, PyDict>,
    launch: &Bound<'_, PyDict>,
    chi_field: Option<&Bound<'_, PyAny>>,
    a: Option<&Bound<'_, PyAny>>,
    adot: Option<&Bound<'_, PyAny>>,
    cohort_field: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut cfg = parse_config(sim_info)?;
    let get = |key: &str| -> PyResult<Bound<'_, PyAny>> {
        launch
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("launch: missing '{key}'")))
    };
    let owner: Vec<usize> = get("owner")?.extract()?;
    let cuts: Vec<Vec<usize>> = get("cuts")?.extract()?;
    let params = WorkerLaunch {
        worker: get("worker")?.extract()?,
        workers: get("workers")?.extract()?,
        owner,
        rendezvous: get("rendezvous")?.extract()?,
        credential: get("credential")?.extract()?,
        bind: get("bind")?.extract()?,
        advertise: get("advertise")?.extract()?,
        staging_cells: get("staging_cells")?.extract()?,
        config_digest: effective_config_digest(sim_info, &cfg.config_sha256)?,
    };
    cfg.n_gpus = params.owner.len();
    cfg.decompose = cuts;
    // the layout, not the config's device count, sets the tiles; each worker runs on the host
    validate_config_checks(&cfg).map_err(PyValueError::new_err)?;
    // one write, so the lines of concurrent workers sharing a stream stay whole
    {
        use std::io::Write;
        let line = format!(
            "SIMBI provenance: backend={} config={} config_sha256={} worker={}/{}\n",
            build_source_id(),
            cfg.config_source,
            cfg.config_sha256,
            params.worker,
            params.workers
        );
        let _ = std::io::stderr().write_all(line.as_bytes());
    }
    let drain_clock = std::time::Instant::now();
    let prims = drain_prims(prim_gen)?;
    let bufs = if matches!(cfg.regime.as_str(), "nmhd" | "imhd" | "rmhd") { drain_bfields(staggered_bfields)? } else { Vec::new() };
    cfg.chi_ic = drain_dye(chi_field)?;
    cfg.cohort_ic = drain_cohort(cohort_field)?;
    sample_scale_factor(&mut cfg, a, adot)?;
    // the excision floors scale with the whole grid's initial atmosphere, as on the single grid:
    // every worker reads the same rows, fresh or resumed
    if cfg.excision_radius > 0.0 {
        (cfg.excision_rho_scale, cfg.excision_pre_scale) =
            initial_excision_scales(&prims).map_err(PyValueError::new_err)?;
    }
    let drained = drain_clock.elapsed();
    py.detach(|| on_run_stack(|| run_worker_process(&cfg, &prims, &bufs, &params, drained)))
        .map_err(PyRuntimeError::new_err)
}

// cpu build -> `simbi.libs.cpu_ext`.
#[cfg(not(feature = "gpu"))]
#[pymodule]
fn cpu_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)
}

// gpu build (cuda or hip) -> `simbi.libs.gpu_ext`. maturin derives the expected init symbol
// from the crate `[lib] name` (`cpu_ext`), so it warns that `PyInit_cpu_ext` is missing and
// writes the dylib under the `cpu_ext` filename. dev.py's `_finalize_gpu_ext` renames the file
// to `gpu_ext.<suffix>.so` afterward so name and `PyInit_gpu_ext` symbol agree, letting the cpu
// and gpu backends coexist; a shared `cpu_ext` dylib would overwrite one with the other.
#[cfg(feature = "gpu")]
#[pymodule]
fn gpu_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register(m)
}

#[cfg(test)]
mod tile_clip_tests {
    use super::{PlacedTile, RefinementRegion, WholeGrid, clip_regions_to_tile};
    use symbi::sim::state::{Boundaries, BoundaryType};
    use symbi_geometry::AxisMap;

    // the clip reads the tile's far corner from the whole grid's faces. on a log radial axis the
    // cells widen outward, so a linear estimate `origin + cells * innermost width` falls well short
    // of where the tile's last cell ends and would drop a region that really overlaps the tile.
    // this region sits in that gap: inside the true extent, outside the linear estimate.
    #[test]
    fn log_tile_clipping_keeps_a_region_the_linear_extent_would_drop() {
        let (m, start) = (16usize, 1.5_f64);
        let slope = (400.0_f64 / start).log10() / 64.0;
        let map = AxisMap::Log { start, log_slope: slope };
        let uniform = AxisMap::Uniform { start: 0.0, dx: 1.0 };
        let grid = WholeGrid {
            cells: [64, 1, 1],
            origin: [start, 0.0, 0.0],
            spacing: [map.face(1) - map.face(0), 1.0, 1.0],
            maps: Some([map, uniform, uniform]),
            boundaries: Boundaries::uniform(BoundaryType::Outflow),
        };
        // the second tile of four along the radius
        let tile = PlacedTile { offset: [m, 0, 0], cells: [m, 1, 1], boundaries: Boundaries::uniform(BoundaryType::Outflow) };
        let (lo, true_hi) = (map.face(m as isize), map.face(2 * m as isize));
        let linear_hi = lo + m as f64 * (map.face(m as isize + 1) - lo);
        assert!(linear_hi < true_hi, "setup is vacuous: the linear extent {linear_hi} must fall short of {true_hi}");
        let r = RefinementRegion { x_lo: [0.5 * (linear_hi + true_hi), 0.0, 0.0], x_hi: [true_hi, 1.0, 1.0] };
        let kept = clip_regions_to_tile::<3>(&[r], &grid, &tile);
        assert_eq!(kept.len(), 1, "the clip dropped a region that overlaps the tile (true hi {true_hi}, linear hi {linear_hi})");
        assert_eq!(kept[0].0.x_hi[0], true_hi, "the clipped region ends at the tile's far face");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BUILD_GIT_DIRTY, BUILD_GIT_SHA, build_source_id, build_source_label, wrap_capability,
    };
    use symbi::sim::decomp::Topology;

    // the seam exists exactly when an axis face meets a wrapping azimuth cut into tiles, and
    // carries each azimuth tile's width
    #[test]
    fn a_polar_seam_needs_an_axis_face_on_a_wrapping_cut_azimuth() {
        use super::polar_seam;
        use symbi::sim::state::{Boundaries, BoundaryType::{Axis, Outflow, Periodic}};
        let pole = Boundaries([[Outflow, Outflow], [Axis, Axis], [Periodic, Periodic]]);
        let wrapping = Topology::wrapping([false, false, true]);
        let seam = polar_seam::<3>("spherical", &pole, &wrapping, [1, 1, 3], |_, i| [6, 4, 2][i]).expect("the seam");
        assert_eq!((seam.mirror, seam.azimuth, seam.sides, seam.slabs.clone()), (1, 2, [true, true], vec![6, 4, 2]));
        let open = Topology::wrapping([false, false, false]);
        assert!(polar_seam::<3>("spherical", &pole, &open, [1, 1, 3], |_, _| 4).is_none(), "the azimuth must wrap");
        assert!(polar_seam::<3>("spherical", &pole, &wrapping, [2, 1, 1], |_, _| 12).is_none(), "an uncut azimuth needs none");
        let walls = Boundaries([[Outflow, Outflow], [Outflow, Outflow], [Periodic, Periodic]]);
        assert!(polar_seam::<3>("spherical", &walls, &wrapping, [1, 1, 3], |_, _| 4).is_none(), "the pole must be an axis face");
    }

    #[test]
    fn compiled_backend_carries_a_self_consistent_source_identity() {
        assert!(!BUILD_GIT_SHA.is_empty());
        assert!(!BUILD_GIT_SHA.chars().any(char::is_whitespace));
        let id = build_source_id();
        assert!(id.starts_with(BUILD_GIT_SHA));
        assert_eq!(id.ends_with("-dirty"), BUILD_GIT_DIRTY);
        assert!(build_source_label().len() <= 18);
    }

    // a run with no periodic face carries whatever it declares; the refusal is about the wrap.
    #[test]
    fn an_open_grid_carries_everything() {
        assert!(wrap_capability(false, true, true).is_ok());
    }

    // the balance-aware reconstruction reads a body potential across the seam, which the wrap
    // cannot make sound, so it is refused by name. each half alone is carried.
    #[test]
    fn a_wrapping_grid_refuses_what_it_cannot_carry() {
        assert!(wrap_capability(true, false, false).is_ok());
        assert!(wrap_capability(true, true, false).is_ok(), "balancing alone is fine");
        assert!(wrap_capability(true, false, true).is_ok(), "a body alone is fine");

        let bodies = wrap_capability(true, true, true).unwrap_err();
        assert!(bodies.contains("body potentials"), "{bodies}");
    }
}

#[cfg(test)]
mod kernel_construction_tests {
    use super::*;

    const GAMMA: f64 = 1.4;

    // a force pushing +x, a fixed inflow state on the driven face, and a zero-gradient wall: one
    // entry in each registry, so the ids each construction assigns are observable
    fn force() -> Vec<SourcePayload> {
        vec![SourcePayload {
            origin: "test force".into(),
            json: r#"{"kind": "force", "dim": 2, "outputs": [0, 1], "params": [0.5],
                      "vocabulary": {"reads": [], "params": [0]},
                      "nodes": [{"op": "PARAMETER", "param_idx": 0}, {"op": "CONSTANT", "value": 0.0}]}"#
                .into(),
        }]
    }
    fn inflow() -> Vec<String> {
        vec![r#"{"kind": "dirichlet", "dim": 2, "outputs": [0, 1, 2, 3], "params": [],
                 "nodes": [{"op": "CONSTANT", "value": 1.3}, {"op": "CONSTANT", "value": 0.2},
                           {"op": "CONSTANT", "value": 0.0}, {"op": "CONSTANT", "value": 1.1}]}"#
            .into()]
    }
    fn walls() -> Vec<GradientBc> {
        vec![GradientBc::Neumann(vec![0.0; 4])]
    }

    // every door away from its constructor default: theta 1.0, plm, no flattening, plain
    // reconstruction, hlle, contact CT, zero viscosity, alpha, resistivity and excision
    fn settings<'c>(
        spec: &'static symbi_hydro::regime_spec::RegimeSpec,
        sources: &'c [SourcePayload],
        driven: &'c [String],
        gradient_walls: &'c [GradientBc],
    ) -> KernelSettings<'c> {
        KernelSettings {
            spec,
            theta: 1.5,
            recon: symbi::Recon::Ppm,
            flatten: (0.3, 0.8),
            well_balanced: true,
            eos: symbi::EosArm::IdealGamma,
            solver: Solver::Hllc,
            ct_method: CtMethod::Uct,
            viscosity: 2.0e-3,
            alpha: 1.0e-2,
            resistivity: 3.0e-3,
            excision: [0.25, 2.0, 3.0],
            sources,
            law: symbi_hydro::state_law::StateLaw::newtonian(GAMMA),
            driven,
            gradient_walls,
        }
    }

    type GasSim = SimDefault<Newtonian, 2, Cartesian, EosSelect<f64>>;

    // a 2:1 box with a driven inflow on x_lo, a neumann wall on x_hi and periodic y, seeded with a
    // smooth density and pressure bump so every door acts on a nonuniform state
    fn gas_sim() -> GasSim {
        GasSim::build(Newtonian, EosSelect::Ideal(IdealGas { gamma: GAMMA }), Cartesian)
            .cells([32, 16])
            .ghosts(3)
            .bounds([0.0, 0.0], [2.0, 1.0])
            .boundaries(Boundaries([
                [BoundaryType::Driven(0), BoundaryType::Neumann(0)],
                [BoundaryType::Periodic, BoundaryType::Periodic],
            ]))
            .allocate()
            .expect("gas sim")
            .set_initial(|x| {
                let b = 0.2 * (-((x[0] - 1.0).powi(2) + (x[1] - 0.5).powi(2)) / 0.02).exp();
                Prim::adiabatic(Density(1.0 + b), Tensor::new([0.1, 0.0]), Pressure(1.0 + b))
            })
            .build()
    }

    #[test]
    fn the_gas_set_carries_every_configured_door() {
        let (sources, driven, walls) = (force(), inflow(), walls());
        let s = settings(&symbi_hydro::regime_spec::NEWTONIAN_SPEC, &sources, &driven, &walls);
        let ks = gas_kernels(gas_sim().substrate(), &s, LevelRole::Root).expect("gas kernels");
        assert_eq!(ks.theta, 1.5);
        assert_eq!(ks.recon, symbi::Recon::Ppm);
        assert_eq!(ks.flatten, (0.3, 0.8));
        assert_ne!(ks.balance, gas_sim().substrate().balance, "the balance-aware reconstruction is on");
        assert_eq!(ks.solver, Solver::Hllc);
        assert_eq!((ks.viscosity, ks.alpha), (2.0e-3, 1.0e-2));
        assert!(ks.runtime_source.is_some(), "the configured source is attached");
        assert_eq!(ks.boundary_dags.len(), 1, "the driven face has its prescription");
        assert_eq!(ks.gradient_bcs, walls);
    }

    // the shared construction runs the scheme its doors name: a short run matches a set whose doors
    // are set one at a time, bit for bit, and differs from the default doors
    #[test]
    fn the_gas_set_runs_the_scheme_its_settings_name() {
        let (sources, driven, walls) = (force(), inflow(), walls());
        let mut s = settings(&symbi_hydro::regime_spec::NEWTONIAN_SPEC, &sources, &driven, &walls);
        // balance-aware reconstruction acts through a body potential, which this box has none of
        s.well_balanced = false;
        let run = |ks: AdiabaticSubstrateKernelSet<DefaultMemory, f64, 2>| -> Vec<u64> {
            let mut h = Hierarchy::single(gas_sim(), ks);
            h.evolve(0.05).expect("evolve");
            let st = &h.levels[0].state;
            st.geom.interior.iter().map(|c| st.fields.cons.den.view().at(c).to_bits()).collect()
        };
        let registries = |ks: AdiabaticSubstrateKernelSet<DefaultMemory, f64, 2>| {
            let ks = attach_configured_sources(ks, &sources, &symbi_hydro::regime_spec::NEWTONIAN_SPEC, &s.law).expect("source");
            let bcfg = symbi_source_compile::SourceConfig::from_json(&driven[0]).expect("inflow parse");
            let built = symbi_source_compile::expr_bridge::build_boundary_dag(&bcfg, &symbi_hydro::regime_spec::NEWTONIAN_SPEC).expect("inflow lower");
            ks.with_driven_boundary(built, bcfg.params.clone()).0.with_gradient_boundary(walls[0].clone()).0
        };
        let shared = run(gas_kernels(gas_sim().substrate(), &s, LevelRole::Root).expect("gas kernels"));
        let by_door = run(registries(
            gas_sim()
                .substrate()
                .theta(1.5)
                .reconstruction(symbi::Recon::Ppm)
                .ppm_flatten(0.3, 0.8)
                .with_solver(Solver::Hllc)
                .expect("hllc")
                .with_viscosity(2.0e-3)
                .with_alpha(1.0e-2),
        ));
        let defaults = run(registries(gas_sim().substrate()));
        assert_ne!(by_door, defaults, "premise: the configured doors must change the run");
        assert_eq!(shared, by_door, "the shared construction runs a scheme other than its settings name");
    }

    #[test]
    fn the_isothermal_set_carries_every_configured_door() {
        let (sources, driven) = (force(), inflow());
        let walls = vec![GradientBc::Robin(vec![[1.0, 0.0, 1.0]; 3])];
        let mut s = settings(&symbi_hydro::regime_spec::ISO_NEWTONIAN_SPEC, &sources, &driven[..0], &walls);
        s.law = symbi_hydro::state_law::StateLaw::newtonian(1.0);
        let domain = gas_sim().geom.allocated.clone();
        let ks = iso_kernels::<2>(IsoSubstrateKernelSet::new(1.0, 0.4, &domain), &s).expect("iso kernels");
        assert_eq!(ks.theta, 1.5);
        assert_eq!((ks.viscosity, ks.alpha), (2.0e-3, 1.0e-2));
        assert!(ks.runtime_source.is_some(), "the configured source is attached");
        assert_eq!(ks.gradient_bcs, walls);
    }

    // the root carries the excision, a fine level none; every other door is the same on both
    #[test]
    fn the_magnetized_set_carries_every_configured_door_and_excises_on_the_root() {
        let mut s = settings(&symbi_hydro::regime_spec::NEWTONIAN_MHD_SPEC, &[], &[], &[]);
        s.solver = Solver::Hlld;
        let domain = gas_sim().geom.allocated.clone();
        let fresh = || {
            symbi::regimes::substrate_mhd::MhdSubstrateKernelSet::<NewtonianMhd, DefaultMemory, f64, 2>::new(GAMMA, 0.4, 1.0, &domain)
        };
        let root = mhd_kernels(fresh(), &s, LevelRole::Root).expect("root kernels");
        let fine = mhd_kernels(fresh(), &s, LevelRole::Fine).expect("fine kernels");
        for ks in [&root, &fine] {
            assert_eq!(ks.theta, 1.5);
            assert_eq!(ks.solver, Solver::Hlld);
            assert_eq!(ks.ct_method, CtMethod::Uct);
            assert_eq!((ks.viscosity, ks.alpha, ks.resistivity), (2.0e-3, 1.0e-2, 3.0e-3));
        }
        assert_eq!(root.excision_radius, 0.25);
        assert_eq!(fine.excision_radius, 0.0);
    }
}
