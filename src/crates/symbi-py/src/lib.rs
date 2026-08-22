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

use symbi::prelude::*;
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
    SchwarzschildKSCylindrical,
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
use symbi_sim::state::CtMethod;
use symbi_sim::state::SimStateGeneric;
use symbi_sim::substrate_seam::{WithExcision, WithResistivity, WithViscosity};

// =============================================================================
// parsed configuration — a plain-rust mirror of the python exec_dict. the
// monomorphized dispatch below reads these tags to pick the concrete SimState.
// =============================================================================

struct Config {
    name: String,
    regime: String,
    coord_system: String,
    // the spacetime background ("minkowski" default, "schwarzschild" for GR), orthogonal to
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
    gradient_bcs: Vec<GradientBcSpec>,
    // the config author's own params (subclass fields), grouped, for the live dashboard's
    // problem-setup panel: each is [group, label, value].
    custom_params: Vec<[String; 3]>,
    // body-diagnostic output cadence in natural units (x time_unit -> code);
    // 0 disables the diagnostics file.
    diagnostic_interval: f64,
    // number of gpus to decompose the domain across, intra-node. 1 = single device;
    // >1 splits the grid into that many tiles evolved in lockstep with halo exchange.
    n_gpus: usize,
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
            let config = symbi_hydro::CensusConfig::from_json(&census.json)
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
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
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
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
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
        built: Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
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
) -> Result<
    (
        Vec<(String, symbi_hydro::source_spec::BuiltSource)>,
        Vec<f64>,
    ),
    String,
> {
    let configs = source_jsons
        .iter()
        .map(|source| {
            let config = symbi_hydro::SourceConfig::from_json(&source.json)
                .map_err(|error| format!("{} parse: {error}", source.origin))?;
            symbi_hydro::expr_bridge::build_user_source_with_law(&config, spec, law)
                .map_err(|error| format!("{} lower: {error}", source.origin))?;
            Ok(config)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let (built, params) =
        symbi_hydro::expr_bridge::build_user_sources_with_law(&configs, spec, law)
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
        other => Err(PyValueError::new_err(format!(
            "unsupported boundary '{other}'"
        ))),
    }
}

/// a gradient-boundary (Neumann / Robin) prescription in registry order: the kind plus the flattened
/// per-variable coefficients in prim order `[rho, vel.., pre]` — Neumann is one `q` per variable,
/// Robin one `(a, b, c)` triple per variable. built from a python `Neumann`/`Robin` object.
struct GradientBcSpec {
    kind: String,
    coeffs: Vec<f64>,
}

/// read the per-variable coefficients off a python `Neumann`/`Robin` object (attributes `rho`,
/// `velocity`, `pressure`). Neumann flattens to `[rho, vel.., pre]`; Robin flattens each variable's
/// `(a, b, c)` triple in the same order.
fn extract_gradient_spec(obj: &Bound<'_, PyAny>, kind: &str) -> PyResult<GradientBcSpec> {
    let rho = obj.getattr("rho")?;
    let velocity = obj.getattr("velocity")?;
    let pressure = obj.getattr("pressure")?;
    let mut coeffs = Vec::new();
    if kind == "neumann" {
        coeffs.push(rho.extract::<f64>()?);
        for v in velocity.try_iter()? {
            coeffs.push(v?.extract::<f64>()?);
        }
        coeffs.push(pressure.extract::<f64>()?);
    } else {
        let (a, b, c): (f64, f64, f64) = rho.extract()?;
        coeffs.extend_from_slice(&[a, b, c]);
        for v in velocity.try_iter()? {
            let (a, b, c): (f64, f64, f64) = v?.extract()?;
            coeffs.extend_from_slice(&[a, b, c]);
        }
        let (a, b, c): (f64, f64, f64) = pressure.extract()?;
        coeffs.extend_from_slice(&[a, b, c]);
    }
    Ok(GradientBcSpec {
        kind: kind.to_string(),
        coeffs,
    })
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
    let mut gradient_bcs: Vec<GradientBcSpec> = Vec::new();
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
                let spec = extract_gradient_spec(&obj, kind)?;
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
        });
    }
    out
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
            magnetic_resistivity: None,
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
    counts: [usize; D],
) -> Vec<symbi_sim::tracers::TracerSet<D>> {
    if cfg.cohort_ic.is_empty() {
        symbi_sim::tracers::seed_and_partition(sim, cfg.n_tracers, counts)
    } else {
        symbi_sim::tracers::seed_and_partition_with_cohorts(
            sim,
            cfg.n_tracers,
            counts,
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
    // the checkpoint cadence is in natural units: `checkpoint_interval * time_unit`
    // is the code-unit spacing, so `checkpoint_interval = 0.1` with a binary's
    // orbital `time_unit` means "every 0.1 orbits". default time_unit = 1.0 keeps
    // the cadence in code units, unchanged for ordinary runs.
    let cp_interval = if cfg.checkpoint_interval > 0.0 {
        cfg.checkpoint_interval * cfg.time_unit
    } else {
        f64::INFINITY
    };
    // logarithmic checkpoint spacing: when dlogt > 0 (the python config enabled
    // log_checkpoints over a positive start_time), the k-th checkpoint lands at
    // start_time*10^(k*dlogt) in code units — dense early, sparse late, the right
    // cadence for a run spanning many decades in time (a relativistic wind from a
    // tiny inner radius out to a huge one). otherwise the cadence is linear at
    // cp_interval. `cp_at(fired)` returns the (fired+1)-th scheduled checkpoint time.
    // the log cadence is anchored at checkpoint_log_anchor (a fixed reference, e.g., the inner
    // light-crossing) so the schedule is identical across a fresh run and a restart whose clock
    // resumes at the checkpoint time; anchoring on start_time would shift it. unset (0) -> start_time (they coincide).
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
    // on a restart the clock resumes mid-schedule at start_time; skip every boundary already at or
    // before it so the first write lands on the next future boundary and reuses no index. seeding
    // cp_fired at 0 would re-dump the resumed checkpoint (a duplicate file, every later index shifted
    // by one) whenever cp_at is anchored at a fixed checkpoint_log_anchor below the resume clock. a
    // restart is signalled by a nonzero loaded checkpoint_index; a fresh run skips nothing.
    let mut cp_fired: u64 =
        checkpoints_at_or_before(cfg.checkpoint_index > 0, cfg.start_time, &cp_at);
    let mut next_cp = cp_at(cp_fired);
    let mut cp_index: u64 = cfg.checkpoint_index + 1;
    // log-spaced runs are named by the monotonic index: the fixed-3-decimal
    // time name (`000_790`) collides at small times (0.0001 and 0.0002 both round to
    // `000_000`, silently overwriting the dense early dumps a log run produces). the physical
    // time lives in metadata/time, which every reader uses. size the zero-pad width to the
    // projected checkpoint count (+ any restart offset) so names always sort lexicographically.
    let cp_idx_width: usize = if cp_log {
        // size the zero-pad tightly to the projected highest index (count + any restart offset).
        // `ceil(log10(max_index + 1))` is the digit count and is robust at the power-of-10 boundary
        // (a projection of 99.99 still yields 2, and exactly 1000 yields 4) so the digit-count boundary stays
        // clear of the run's own last checkpoint. an overshoot extends the width gracefully (format! only
        // pads or grows: width is a floor, so 99 -> 100); only a raw `ls` sees a cosmetic width jump there,
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
    let mut next_diag = diag_interval;
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
    // on a cartesian black-hole run, split the FOFC counters at the outer horizon
    // r_+ = M + sqrt(M^2 - a^2): everything inside is causally disconnected (and the
    // near-horizon infall band fires steadily by design), so the exterior signal is the
    // acceptance criterion for a production run (exterior events == 0). the split is
    // load-bearing — the freeze-streak halt gates on the exterior count — so it uses the
    // true spin-dependent r_+, not the Schwarzschild 2M: cells in (r_+, 2M) at nonzero spin
    // sit outside the horizon, where a poison must still halt the run.
    symbi::regimes::fofc::fofc_set_horizon_radius(
        if cfg.spacetime != "minkowski"
            && cfg.coord_system == "cartesian"
            && cfg.schwarzschild_mass > 0.0
        {
            let m = cfg.schwarzschild_mass;
            let a = cfg.kerr_spin.abs().min(m); // |a| <= M; clamp guards the sqrt
            m + (m * m - a * a).sqrt()
        } else {
            0.0
        },
    );
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
    // the energy-split warning fires at most once per run; see the check at the
    // conservation-diagnostic site.
    let mut ke_split_warned = false;

    let callback = |h: &Hierarchy<R, D, DOF, M, E, S, Mem, K>| {
        let st = &h.levels[0].state;
        let (iter, time, dt) = (st.iteration, st.time, st.dt);
        let mut dirty = false;

        // bounded march: stop after `max_steps` root iterations (0 = unbounded).
        // the break exits through the success epilogue, which writes the final
        // checkpoint — a truncated but otherwise ordinary run.
        if cfg.max_steps > 0 && iter >= cfg.max_steps {
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
                &checkpoint_tag(cfg, cp_idx_width, cp_width, time, cp_index),
            );
            let t_io = std::time::Instant::now();
            let res =
                write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cp_index));
            io_secs.set(io_secs.get() + t_io.elapsed().as_secs_f64());
            match res {
                Ok(_) => table.post_success(&format!(
                    "checkpoint {path}  ({}, manual)",
                    fmt_time_msg(cfg, time)
                )),
                Err(e) => table.post_error(&format!("manual checkpoint failed: {e:?}")),
            }
            cp_index += 1;
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
                write_hierarchy_checkpoint(&states, &restart, &checkpoint_metadata(cfg, cp_index));
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
        // unphysical c2p, e.g. V -> 1 at the inner boundary): snapshot the last computed state as a
        // `.crashed` checkpoint so it can be inspected, then stop.
        if let Some(c) = h.crash {
            table.set_dynamic(false);
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
            let _ =
                write_hierarchy_checkpoint(&states, &crashed, &checkpoint_metadata(cfg, cp_index));
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
        // state and advance next_cp past every boundary it crossed. the skipped
        // intermediate states were never computed, and the file name is keyed by
        // the current time — looping would just re-write the same file N times and
        // spam the board with identical entries.
        if time_at_or_after(time, next_cp) && next_cp.is_finite() {
            let states: Vec<&_> = h.levels.iter().map(|l| &l.state).collect();
            let path = checkpoint_name(
                cfg,
                &checkpoint_tag(cfg, cp_idx_width, cp_width, time, cp_index),
            );
            let t_io = std::time::Instant::now();
            let res = symbi_sim::driver::prof("checkpoint_io", || {
                write_hierarchy_checkpoint(&states, &path, &checkpoint_metadata(cfg, cp_index))
            });
            io_secs.set(io_secs.get() + t_io.elapsed().as_secs_f64());
            match res {
                Ok(_) => {
                    table.post_success(&format!("checkpoint {path}  ({})", fmt_time_msg(cfg, time)))
                }
                Err(e) => table.post_error(&format!("checkpoint {cp_index:04} failed: {e:?}")),
            }
            cp_index += 1;
            cp_fired += 1;
            next_cp = cp_at(cp_fired);
            // advance past every boundary this step crossed (log or linear) so a
            // single large dt yields one write covering all of them.
            while time_at_or_after(time, next_cp) && next_cp.is_finite() {
                cp_fired += 1;
                next_cp = cp_at(cp_fired);
            }
            dirty = true;
        }

        // diagnostics cadence: sample body state on its own (finer) schedule,
        // independent of checkpoints — append a row per body to diagnostics.dat.
        // post one board line per callback that wrote (not per missed interval, so
        // a dt that spans several intervals collapses to a single notice) and mark
        // the frame dirty so the write is visible the moment it happens.
        if let Some(dp) = &diag_path {
            let mut wrote = false;
            while time_at_or_after(time, next_diag) {
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
    if cfg.restart_path.is_some() {
        hier.resume_with_callback(cfg.t_final, 1, callback)?;
    } else {
        hier.evolve_with_callback(cfg.t_final, 1, callback)?;
    }

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
        return Ok(());
    }

    // crashed: the observer already snapshotted the `.crashed` state. surface the halt as the red
    // crash exit frame.
    if let Some(c) = hier.crash {
        screen.leave();
        table.set_dynamic(false);
        let crashed = checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Crashed));
        // report which halt condition fired and its numbers. the three are different failures with
        // different fixes: a non-finite rate is a poisoned cell, a non-positive one a degenerate
        // state, and a sudden jump means the cfl rate collapsed (the wave speeds went to zero, or
        // the scratch the reduction reads was left holding something other than a rate). asserting
        // one diagnosis for all three sends every investigation down the same wrong path.
        let cause = if c.dt_cfl.is_nan() {
            "cfl dt is NaN — a non-finite wave speed (poisoned cell or boundary)".to_string()
        } else if c.dt_cfl <= 0.0 {
            format!("cfl dt is non-positive ({:.6e})", c.dt_cfl)
        } else {
            format!(
                "cfl dt jumped {:.3e}x in one step ({:.6e} -> {:.6e}) — the cfl rate collapsed",
                c.dt_cfl / c.dt_prev,
                c.dt_prev,
                c.dt_cfl
            )
        };
        let summary = format!(
            "crashed — {} steps, t = {:.4} — {cause} · state {crashed}",
            c.iter, c.time,
        );
        table.post_error(&summary);
        table.exit_frame(ExitKind::Crash, &summary);
        return Ok(());
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
        write_hierarchy_checkpoint(&states, &crashed, &checkpoint_metadata(cfg, cp_index))?;
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
        write_hierarchy_checkpoint(&states, &final_path, &checkpoint_metadata(cfg, cp_index))
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
    Ok(())
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

/// clip each global refinement region to one tile's physical extent `[origin, origin + cells*dx]`,
/// keeping only the regions that genuinely overlap (more than ~half a coarse cell on every axis, so
/// region_to_domain rounds to >= 1 cell). a tile-local patch survives in exactly one tile; a patch
/// spanning a cut is split into the abutting tiles, each clipped to its own slab (the per-tile
/// hierarchies then share the cut on the fine level, exchanged by `evolve_hierarchy_decomposed`).
/// clip the global refinement regions to one tile's physical box. `maps` carries the tile's own
/// coordinate maps, so the tile's far corner and its cell widths come from the map rather than from
/// `origin + cells * dx`: on a log radial axis the widths grow with radius and the linear form would
/// place the tile's outer edge far short of where its last cell actually ends, silently dropping
/// refinement regions that do overlap the tile.
///
/// the survival test compares each clipped extent against the width of the smallest cell in the
/// overlap rather than a single global `dx`, which is the same "narrower than half a cell" rule the
/// uniform path applied, expressed so it still means that when the cells are not all one size.
fn clip_regions_to_tile<const D: usize>(
    regions: &[RefinementRegion<D>],
    origin: [f64; D],
    cells: [usize; D],
    maps: &[symbi_geometry::AxisMap; D],
) -> Vec<RefinementRegion<D>> {
    let hi: [f64; D] = std::array::from_fn(|a| maps[a].face(cells[a] as isize));
    // the narrowest cell on each axis: for a uniform map every cell has this width, for a log map it
    // is the innermost one, so the test never discards a region an actual cell could resolve.
    let min_w: [f64; D] = std::array::from_fn(|a| {
        (0..cells[a])
            .map(|i| maps[a].face(i as isize + 1) - maps[a].face(i as isize))
            .fold(f64::INFINITY, f64::min)
    });
    let mut out = Vec::new();
    for r in regions {
        let lo: [f64; D] = std::array::from_fn(|a| r.x_lo[a].max(origin[a]));
        let up: [f64; D] = std::array::from_fn(|a| r.x_hi[a].min(hi[a]));
        if (0..D).all(|a| up[a] - lo[a] > 0.5 * min_w[a]) {
            out.push(RefinementRegion { x_lo: lo, x_hi: up });
        }
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
        // a magnetized sink: the body dissipates the field threading it (MHD runs only; a no-op on B
        // for a hydro/None body). applied on top of the surface stack.
        let body = match b.magnetic_resistivity {
            Some(eta) if eta > 0.0 => body.with_magnetic(symbi_ib::MagneticSpec::Resistive { eta }),
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

fn build_bodies_and_horizon<const D: usize>(cfg: &Config) -> BodyCollection<f64, D> {
    let mut coll = build_bodies::<D>(&cfg.bodies, cfg.binary.as_ref());
    if cfg.spacetime != "minkowski" && cfg.coord_system == "cartesian" && cfg.excision_radius > 0.0
    {
        let diagnostic_radius = 3.0 * cfg.schwarzschild_mass; // 1.5 r_+, outside the horizon
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
        }];
        let coll = build_bodies::<2>(&params, None);
        assert!(coll.get(0).two_way_coupling);
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

/// the coarse->fine prolongation order: degree at least one above the interior
/// reconstruction (pcm -> plm, plm -> ppm, ppm -> quartic), so refinement
/// boundaries never drop a spatial order.
fn prolong_order_for(reconstruction: &str) -> ProlongOrder {
    match reconstruction {
        "pcm" => ProlongOrder::Plm,
        "ppm" => ProlongOrder::Quartic,
        _ => ProlongOrder::Ppm, // plm (the default) -> ppm
    }
}

/// the mesh-motion state from the config: static, homologous expansion (linear,
/// `a += a_dot*dt`, a_ddot = 0), or uniform cartesian translation. `scale_a0` /
/// `scale_adot` are the python scale-factor callables already evaluated at
/// start_time (the rust model integrates a from a constant rate).
/// seed a sim's clock + mesh-motion state from the config: physical time starts at
/// start_time (a moving-mesh a(t) samples the physical clock), the motion state from the
/// scale-factor scalars, and the traced a(t) law when the config carries one — the same
/// seeding the shared uni-grid prep performs; decomposed tiles call it per tile so every
/// tile advances the identical law in lockstep.
fn attach_motion<R, const D: usize, const DOF: usize, M, E, S, Mem>(
    sim: &mut SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    cfg: &Config,
) -> Result<(), String>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
{
    sim.time = cfg.start_time;
    sim.motion = motion_state(cfg);
    if let Some(ref mj) = cfg.motion_json {
        let t0 = sim.time;
        let law = symbi_hydro::motion_law::MotionLaw::from_json(mj, t0, cfg.t_final)
            .map_err(|e| format!("mesh motion: {e}"))?;
        sim.motion = symbi_geometry::MotionState::homologous(law.a_at(t0), law.adot_at(t0));
        sim.motion_law = Some(law);
    }
    Ok(())
}

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
/// the per-axis coordinate maps for one tile of a decomposed grid, and the tile's physical origin.
///
/// a tile holds local cell indices `0..m` positioned by an origin, so the map it carries must place
/// local index `i` where the global grid places `g + i` (`g` = the tile's first global cell on that
/// axis).
///
/// for a uniform axis that composes additively — `x_lo + g dx` then `+ i dx` — which is what the
/// decomposed builder has always done. a log axis maps `face(i) = start * 10^(i * slope)`, so the
/// composition is multiplicative: the tile's origin is `start * 10^(g * slope)`.
///
/// the slope is global and must be inherited, never re-derived from the tile's own extent: it is a
/// per-cell stretching, so a tile that recomputed it from its local start and cell count would give
/// itself a different stretching and the grid would break at every seam. re-deriving it from the
/// correct tile origin is consistent — `log10(hi_local/start_local)/m = log10(10^(m s))/m = s` — so
/// inheriting is the same value, obtained without the chance of getting it wrong.
fn tile_axis_maps<const D: usize>(
    cfg: &Config,
    tile_lo: [usize; D],
) -> Option<[symbi_geometry::AxisMap; D]> {
    let global = axis_maps::<D>(cfg)?;
    Some(std::array::from_fn(|ax| {
        shift_axis_map(global[ax], tile_lo[ax])
    }))
}

/// advance one axis map to a tile whose first cell is global index `tile_lo`, so the tile's local
/// index `i` lands where the global grid puts `tile_lo + i`. uniform composes additively, log
/// multiplicatively; the log slope is a per-cell stretching and is carried through unchanged.
fn shift_axis_map(global: symbi_geometry::AxisMap, tile_lo: usize) -> symbi_geometry::AxisMap {
    use symbi_geometry::AxisMap;
    match global {
        AxisMap::Log { start, log_slope } => AxisMap::Log {
            start: start * 10.0_f64.powf(tile_lo as f64 * log_slope),
            log_slope,
        },
        AxisMap::Uniform { start, dx } => AxisMap::Uniform {
            start: start + tile_lo as f64 * dx,
            dx,
        },
        AxisMap::Geometric {
            start,
            width,
            ratio,
        } => AxisMap::Geometric {
            start: AxisMap::Geometric {
                start,
                width,
                ratio,
            }
            .face(tile_lo as isize),
            width: width * ratio.powf(tile_lo as f64),
            ratio,
        },
    }
}

/// the physical lower corner of a decomposed tile whose first global cell is `tile_lo`. the one
/// place this formula lives: a uniform axis advances additively by `g dx`, a log axis
/// multiplicatively by `10^(g slope)`, and a caller that assumed the uniform form on a log grid
/// would silently place the tile at the wrong radius.
fn tile_origin<const D: usize>(cfg: &Config, tile_lo: [usize; D]) -> [f64; D] {
    match tile_axis_maps::<D>(cfg, tile_lo) {
        Some(maps) => std::array::from_fn(|ax| match maps[ax] {
            symbi_geometry::AxisMap::Log { start, .. } => start,
            symbi_geometry::AxisMap::Uniform { start, .. } => start,
            symbi_geometry::AxisMap::Geometric { start, .. } => start,
        }),
        None => std::array::from_fn(|ax| cfg.x_lo[ax] + tile_lo[ax] as f64 * cfg.dx[ax]),
    }
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
        // the user dt clamp (0 = disabled): pins the dt sequence across runs whose CFL
        // estimators differ (kernel cross-validation, temporal convergence studies).
        sim.max_dt = $cfg.max_dt;
        // the physical clock starts at start_time (t0): the IC is the state at t0, and a moving-mesh
        // a(t) must be sampled at the physical time; an elapsed-from-0 clock would mis-phase it. (default 0 -> no
        // change for the common case.)
        sim.time = $cfg.start_time;
        // mesh motion lives on the (coarse) state — set before wrapping. static
        // for the common case; the gates above keep motion to single-grid hydro.
        sim.motion = motion_state($cfg);
        // expression-driven mesh motion: build the traced a(t)/a_dot(t) law (autodiff'd a_dot, fd
        // cross-checked) and seed the homologous motion from it at t0 = sim.time. the time loop then
        // evaluates a(t) exactly each (sub)stage.
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
macro_rules! attach_bodies_and_fragments {
    ($sim:expr, $cfg:expr, $d:literal) => {{
        let cfg: &Config = $cfg;
        let mut coll = build_bodies_and_horizon::<$d>(cfg);
        let mut shapes = build_body_shapes(&cfg.bodies);
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

        // the declared perturbation is a per-cell primitive delta, so it is chart- and
        // regime-generic; what it cannot reach is the decomposed build, whose tiles each
        // hold their own hierarchy. refused rather than silently dropped.
        if cfg.perturbation_json.is_some() && cfg.n_gpus > 1 {
            return Err(
                "perturbation_expressions is not wired for the decomposed (multi-gpu) \
                 build"
                    .to_string(),
            );
        }

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
                Prim {
                    rho: row[0],
                    vel: Tensor::new(std::array::from_fn(|k| row[1 + k])),
                    pre: row[1 + $dof],
                }
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
        let theta = build_theta(cfg);
        let sub = sim
            .substrate()
            .theta(theta)
            .reconstruction(build_recon(cfg))
            .ppm_flatten(cfg.ppm_flatten_onset, cfg.ppm_flatten_full)
                .well_balanced_reconstruction(cfg.wb_reconstruction)
            .with_eos(build_eos(cfg))
            .with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?
            .with_viscosity(cfg.viscosity)
            .with_resistivity(cfg.resistivity)
            .with_excision(
                cfg.excision_radius,
                cfg.excision_rho_scale,
                cfg.excision_pre_scale,
            );
        // attach a user source expression (force/cooling/relax/raw) when present.
        // lowered against this regime's spec via the source front door — the bridge rejects
        // force/cooling/relax on relativistic regimes (use raw). the base level attaches it here;
        // refined runs re-attach the same source to each fine level in the `into_hierarchy` `$make`
        // closure below, so it acts on every level it overlaps.
        let sub = attach_configured_sources(
            sub,
            &cfg.source_jsons,
            <$regime_ty as Regime<f64, $dof>>::SPEC,
            &state_law_of(cfg, <$regime_ty as Regime<f64, $dof>>::SPEC),
        )?;
        // register driven (dynamic) boundaries in Driven-id order so `Driven(id)` on a face
        // matches `driven_exprs[id]` — the complete prim prescription [rho, vel_0..DOF-1, pre]
        // as coordinate DAGs. a theta-stratified rotating equilibrium requires
        // this: no local ghost rule can represent the state beyond a wedge wall.
        let mut sub = sub;
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
        // register the Neumann/Robin gradient boundaries (the convenience prescribed-gradient walls);
        // the flattened coeffs are re-grouped into the registry entry (Robin: (a,b,c) triples).
        for spec in &cfg.gradient_bcs {
            use symbi::regimes::substrate_kernels::GradientBc;
            let gbc = match spec.kind.as_str() {
                "neumann" => GradientBc::Neumann(spec.coeffs.clone()),
                "robin" => GradientBc::Robin(
                    spec.coeffs.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                ),
                other => return Err(format!("unknown gradient boundary kind '{other}'")),
            };
            sub = sub.with_gradient_boundary(gbc).0;
        }
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            // the fine set carries the same non-ideal knobs as the base: the hierarchy
            // applies the viscous pass on the finest level only, so a fine set without
            // nu would make the whole refined run silently inviscid. the reconstruction
            // is likewise per-level: a fine set left on plm under ppm evolution would
            // silently evolve every refined region one order low.
            let ks = s
                .substrate()
                .theta(theta)
                .reconstruction(build_recon(cfg))
                .ppm_flatten(cfg.ppm_flatten_onset, cfg.ppm_flatten_full)
                .well_balanced_reconstruction(cfg.wb_reconstruction)
                .with_eos(build_eos(cfg))
                .with_solver(solver)
                .expect("fine-level kernel set")
                .with_viscosity(cfg.viscosity)
                .with_resistivity(cfg.resistivity);
            // register the same driven-boundary DAGs on each fine level, in Driven-id order: a
            // fine level flush against a driven physical face inherits `Driven(id)` there (an
            // interior fine level has only CoarseFine faces and never consults the dags). the
            // prescription is a pure coordinate DAG, so the fine level evaluates it at its own
            // finer ghost coordinates; the fill runs at the tail of ghost_fill, after
            // prolong_cf, so it deterministically owns the driven/coarse-fine corner overlap.
            // already validated at the base registration.
            let mut ks = ks;
            for json in &cfg.driven_exprs {
                let bcfg = symbi_hydro::SourceConfig::from_json(json)
                    .expect("fine-level boundary parse");
                let built = symbi_hydro::expr_bridge::build_boundary_dag(
                    &bcfg,
                    <$regime_ty as Regime<f64, $dof>>::SPEC,
                )
                .expect("fine-level boundary lower");
                ks = ks.with_driven_boundary(built, bcfg.params.clone()).0;
            }
            for spec in &cfg.gradient_bcs {
                use symbi::regimes::substrate_kernels::GradientBc;
                let boundary = match spec.kind.as_str() {
                    "neumann" => GradientBc::Neumann(spec.coeffs.clone()),
                    "robin" => GradientBc::Robin(
                        spec.coeffs
                            .chunks_exact(3)
                            .map(|coeff| [coeff[0], coeff[1], coeff[2]])
                            .collect(),
                    ),
                    other => panic!("unknown fine-level gradient boundary kind '{other}'"),
                };
                ks = ks.with_gradient_boundary(boundary).0;
            }
            let ks = ks;
            // attach the same user source to each fine level as the base level, so a source
            // overlapping a refined region still acts there (a base-only attach would be restricted
            // away by the fine solution). the source was already validated at the base attach.
            attach_configured_sources(
                ks,
                &cfg.source_jsons,
                <$regime_ty as Regime<f64, $dof>>::SPEC,
                &state_law_of(cfg, <$regime_ty as Regime<f64, $dof>>::SPEC),
            )
            .expect("fine-level source attach")
        });
        // a refined run attaches its immersed bodies to the hierarchy: the finest level owns the full
        // (accreting) bodies, coarser levels carry a gravity-only proxy (finest-owns-bodies, so the
        // sink applies once). the sink sphere must lie inside the finest level (asserted there).
        if !cfg.bodies.is_empty() && cfg.refinement_enabled {
            hier = hier.with_bodies(build_bodies_and_horizon::<$d>(cfg));
            hier.attach_body_shapes(build_body_shapes(&cfg.bodies));
        }

        // the run's stationary target, when one is declared: its discrete imbalance is measured
        // once per level and added back at every stage, so the target becomes an exact fixed point
        // rather than a state that drifts at truncation order. this runs after the bodies attach
        // because the imbalance is read off a real stage, and a gravitational body's source is
        // half of the balance being measured.
        if let Some(ref target) = cfg.equilibrium_json {
            let declared = symbi_hydro::EquilibriumConfig::from_json(target)
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
        // centers: a delta on each primitive component. the cell generator fills only the
        // root, so fine levels otherwise hold nothing the root cannot represent. skipped on
        // restart, where the checkpoint restore replaces the state wholesale.
        if cfg.restart_path.is_none() {
            if let Some(ref declared) = cfg.perturbation_json {
                let cfg_p = symbi_hydro::EquilibriumConfig::from_json(declared)
                    .map_err(|e| format!("perturbation parse: {e}"))?;
                let expression = cfg_p
                    .to_expression()
                    .map_err(|e| format!("perturbation lower: {e:?}"))?;
                let has_energy = hier.levels[0].state.fields.cons.has_energy();
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
                hier.perturb_cells(|x, p| {
                    let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                    let d = expression.eval(at(0), at(1), at(2), 0.0);
                    let mut p = p;
                    p.rho += d[0];
                    for ax in 0..$dof {
                        p.vel[ax] += d[1 + ax];
                    }
                    // an energy-free regime's slot discards whatever it is handed, so the
                    // pressure delta needs no branch: it is a no-op there by construction.
                    {
                        use symbi_hydro::energy::EnergySlot;
                        let dp = d.get(1 + $dof).copied().unwrap_or(0.0);
                        p.pre = p.pre.add(EnergySlot::from_scalar(dp));
                    }
                    p
                });
                hier.sync_perturbed();
            }
        }
        run_loop(&mut hier, cfg).map_err(|e| e.to_string())
    }};
}

/// the regime-agnostic decomposed run loop: evolve N pre-built tiles in
/// lockstep with the universal `PeerCopy` transport (real peer where a link exists, staged over
/// managed memory otherwise -- so the same code runs on one card with `--gpus 2` and on a node
/// with `--gpus 8`, no machine-specific branch), gathering into `global` for output through the
/// existing single-grid checkpoint writer. every regime's decomposed build feeds this one loop;
/// adding a regime is just a tile-build. the checkpoint cadence is the linear `checkpoint_interval`.
fn run_decomposed_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    cfg: &Config,
    mut tiles: Vec<(SimStateGeneric<R, D, DOF, M, E, S, Mem>, K)>,
    mut global: SimStateGeneric<R, D, DOF, M, E, S, Mem>,
    counts: [usize; D],
) -> Result<(), String>
where
    R: Regime<f64, D>,
    M: Metric<f64, D> + Copy + Send + Sync,
    E: Eos<f64> + Send + Sync,
    S: ExecutionSpace,
    Mem: MemorySpace + Sync,
    K: KernelSet<D, DOF, Mem, f64>,
    Cartesian: Metric<f64, D>,
{
    use symbi::sim::decomp::{
        enable_peer_mesh, evolve_decomposed, gather_faces, gather_interiors, gather_tracers,
    };

    let ntiles = tiles.len();
    let devices: Vec<i32> = (0..ntiles as i32).collect();
    // open peer links once (no-op for pairs that can't peer; those stage).
    enable_peer_mesh(&devices);

    // the universal transport: adaptive peer/staged. single-device builds compile the host arm
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
    // any tile's bodies are the global diagnostic. one `<dir>diagnostics.dat` for the whole run.
    let diag_path = if cfg.diagnostic_interval > 0.0 && !cfg.bodies.is_empty() {
        Some(format!("{}diagnostics.dat", cfg.data_dir))
    } else {
        None
    };
    let diag_interval = (cfg.diagnostic_interval * cfg.time_unit).max(f64::MIN_POSITIVE);
    let mut next_diag = diag_interval;

    // pin every tile's physical clock to the start time (nonzero on restart): the decomposed
    // loop advances these per step, and the checkpoint writer records the gather target's
    // clock — synced from tile 0 before every write below.
    for (s, _) in tiles.iter_mut() {
        s.time = cfg.start_time;
    }
    global.time = cfg.start_time;

    // t=start initial condition. a shared reborrow of the tiles for the gather (evolve_decomposed
    // takes them by `&mut` below, so the gather views are scoped reborrows on either side).
    if cfg.checkpoint_index == 0 || cfg.start_time == 0.0 {
        let sh: Vec<_> = tiles.iter().map(|(s, _)| &**s).collect();
        global.time = sh[0].time;
        global.iteration = sh[0].iteration;
        global.motion = sh[0].motion;
        gather_interiors(&global, &sh, counts);
        gather_faces(&global, &sh, counts);
        gather_tracers(&mut global, &sh);
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
        // mutates the bodies). build the `&mut` store handles + the `&` kernels from the same tiles
        // (disjoint tuple fields). the checkpoint callback receives the shared tile slice it needs
        // for the gather (it cannot capture `stores` while the loop holds them mutably).
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
                    // the writer records the gather target's clock/scale factor: sync from
                    // tile 0 (all tiles advance in lockstep) or the checkpoint carries the
                    // start-time forever.
                    global.time = sh[0].time;
                    global.iteration = sh[0].iteration;
                    global.motion = sh[0].motion;
                    gather_interiors(&global, sh, counts);
                    gather_faces(&global, sh, counts);
                    gather_tracers(&mut global, sh);
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
        global.time = sh[0].time;
        global.iteration = sh[0].iteration;
        global.motion = sh[0].motion;
        gather_interiors(&global, &sh, counts);
        gather_faces(&global, &sh, counts);
        gather_tracers(&mut global, &sh);
    }
    let _ = write_hierarchy_checkpoint(
        &[&global],
        &checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Completed)),
        &checkpoint_metadata(cfg, cp_index),
    );
    Ok(())
}

/// the multi-gpu (gpus>1) refined path: decompose a 2-level static-refinement hierarchy. each tile
/// is a per-tile `Hierarchy` (its root slab + the global refinement region clipped to that slab, or
/// single-level where the region misses it); `evolve_hierarchy_decomposed` drives them in lockstep
/// (root + first-fine-level halo exchange), reproducing the monolithic hierarchy to round-off.
/// for output, gather each level into the global hierarchy -- the
/// root over `counts`, the fine over the `fine_subgrid` sub-grid (a decomposition of the global
/// fine level) -- and write all its levels through the existing multi-level checkpoint writer.
/// exactly one refined region is carried (the lib driver decomposes the root + first fine level).
fn run_refined_decomposed_loop<R, const D: usize, const DOF: usize, M, E, S, Mem, K>(
    cfg: &Config,
    mut tiles: Vec<Hierarchy<R, D, DOF, M, E, S, Mem, K>>,
    mut global: Hierarchy<R, D, DOF, M, E, S, Mem, K>,
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
    use symbi::sim::refinement::{
        evolve_hierarchy_decomposed, fine_subgrid, gather_decomposed_hierarchy_tracers,
    };

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
    let mut write_cp =
        |tiles: &[Hierarchy<R, D, DOF, M, E, S, Mem, K>], path: &str, cp_index: u64| {
            // the writer records the gather target's clock: sync every global level from tile 0
            // (all tiles advance in lockstep; between root steps the fine clock equals the root's)
            // or the checkpoint carries the start-time forever.
            let t_now = tiles[0].levels[0].state.time;
            let it_now = tiles[0].levels[0].state.iteration;
            for l in global.levels.iter_mut() {
                l.state.time = t_now;
                l.state.iteration = it_now;
            }
            let roots: Vec<_> = tiles.iter().map(|h| &*h.levels[0].state).collect();
            gather_interiors(&*global.levels[0].state, &roots, counts);
            if let Some(fg) = &fg {
                let fines: Vec<_> = fg
                    .order
                    .iter()
                    .map(|&i| &*tiles[i].levels[1].state)
                    .collect();
                gather_interiors(&*global.levels[1].state, &fines, fg.counts);
            }
            if cfg.n_tracers > 0 {
                gather_decomposed_hierarchy_tracers(&mut global, tiles);
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
    write_cp(
        &tiles,
        &checkpoint_name(cfg, checkpoint_status_tag(CheckpointOutcome::Completed)),
        cp_index,
    );
    Ok(())
}

/// the multi-gpu (gpus>1) refined hydro path: per-tile static-refinement hierarchies driven by the
/// `evolve_hierarchy_decomposed` (root + first-fine-level halo exchange).
/// each tile builds its root slab + the global refinement region clipped to that slab (single-level
/// where the region misses it); a patch that spans a cut is split into the abutting tiles and the
/// fine halos are exchanged at the cut. output gathers each level into the global hierarchy (root
/// over `counts`, fine over the fine sub-grid) and writes the multi-level checkpoint. hydro + a
/// single refined region, carrying immersed bodies (with shapes), user sources, and driven
/// boundaries -- each attached per tile and evolved in lockstep, matching the monolithic
/// hierarchy to round-off. mass-transport tracers use global owners and
/// migrate complete records across cuts. mesh motion, the passive scalar, and bonded fragments are
/// not carried here.
macro_rules! build_and_run_hydro_decomposed_refined {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        type Sim = SimDefault<$regime_ty, $d, $geom_ty, EosSelect<f64>>;
        // the per-tile / global hierarchy type. DOF = D for hydro; the kernel set is the substrate's
        // associated type (matches what `sim.substrate()` yields).
        type Hier = Hierarchy<
            $regime_ty,
            $d,
            $d,
            $geom_ty,
            EosSelect<f64>,
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
            // the tile's first global cell on each axis; the origin and the coordinate maps both
            // derive from it, so a log radial axis advances multiplicatively rather than by g*dx.
            let tile_lo: [usize; $d] = std::array::from_fn(|ax| tc[ax] * m[ax]);
            let origin: [f64; $d] = tile_origin::<$d>(cfg, tile_lo);
            let phys = boundaries_nd::<$d>(&cfg.boundaries);
            let bnd = Boundaries(std::array::from_fn(|ax| {
                let lo = if tc[ax] == 0 { phys.0[ax][0] } else { BoundaryType::CoarseFine };
                let hi = if tc[ax] == counts[ax] - 1 { phys.0[ax][1] } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            let tile_maps: [symbi_geometry::AxisMap; $d] = tile_axis_maps::<$d>(cfg, tile_lo)
                .unwrap_or_else(|| std::array::from_fn(|ax| symbi_geometry::AxisMap::Uniform {
                    start: origin[ax], dx: dx[ax],
                }));
            let tile_regions = clip_regions_to_tile::<$d>(&regions, origin, m, &tile_maps);
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<Hier, String> {
                let sim = Sim::build($regime, host_eos(cfg), $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(dx)
                    .coord_maps(tile_axis_maps::<$d>(cfg, tile_lo))
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
                // the tile set carries the same non-ideal + excision knobs as the
                // monolithic base — a tile set without them makes the whole decomposed
                // run silently ideal/unexcised.
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?
                    .with_viscosity(cfg.viscosity)
                    .with_resistivity(cfg.resistivity)
                    .with_excision(
                        cfg.excision_radius,
                        cfg.excision_rho_scale,
                        cfg.excision_pre_scale,
                    );
                // register the driven (dynamic) boundary DAGs on the tile root and every fine
                // level, in Driven-id order: an edge tile's exterior faces carry Driven(id)
                // from the physical-boundary copy above, and a fine level flush against one
                // inherits it; interior faces and cuts are CoarseFine and never consult the
                // dags. each level evaluates the coordinate prescription at its own global
                // coordinates (tiles and their hierarchies share the global origin).
                let register = |mut ks: <Sim as symbi::prelude::SimSubstrate<DefaultMemory, f64, $d>>::KernelSet|
                    -> <Sim as symbi::prelude::SimSubstrate<DefaultMemory, f64, $d>>::KernelSet {
                    for json in &cfg.driven_exprs {
                        let bcfg = symbi_hydro::SourceConfig::from_json(json)
                            .expect("tile boundary parse");
                        let built = symbi_hydro::expr_bridge::build_boundary_dag(
                            &bcfg,
                            <$regime_ty as Regime<f64, $d>>::SPEC,
                        )
                        .expect("tile boundary lower");
                        ks = ks.with_driven_boundary(built, bcfg.params.clone()).0;
                    }
                    for spec in &cfg.gradient_bcs {
                        use symbi::regimes::substrate_kernels::GradientBc;
                        let boundary = match spec.kind.as_str() {
                            "neumann" => GradientBc::Neumann(spec.coeffs.clone()),
                            "robin" => GradientBc::Robin(
                                spec.coeffs
                                    .chunks_exact(3)
                                    .map(|coeff| [coeff[0], coeff[1], coeff[2]])
                                    .collect(),
                            ),
                            other => panic!("unknown tile gradient boundary kind '{other}'"),
                        };
                        ks = ks.with_gradient_boundary(boundary).0;
                    }
                    // the same user source on the tile root and every fine level: each level of
                    // each tile evaluates S at its own global coordinates, so a position-dependent
                    // force is correct across cuts and level seams; the canonical per-level stage
                    // drives source_apply. already validated at the front door.
                    attach_configured_sources(
                        ks,
                        &cfg.source_jsons,
                        <$regime_ty as Regime<f64, $d>>::SPEC,
                        &state_law_of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC),
                    )
                    .expect("tile source attach")
                };
                let sub = register(sub);
                let make = |s: &Sim| {
                    // same non-ideal knobs as the base (the viscous pass runs on the
                    // finest level only).
                    register(
                        s.substrate()
                            .theta(theta)
                            .with_solver(solver)
                            .expect("fine kernel set")
                            .with_viscosity(cfg.viscosity)
                            .with_resistivity(cfg.resistivity),
                    )
                };
                let mut h = if tile_regions.is_empty() {
                    Hierarchy::single(sim, sub)
                } else {
                    Hierarchy::with_refinement(sim, sub, &tile_regions, prolong, make)
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
                    h.attach_body_shapes(build_body_shapes(&cfg.bodies));
                }
                h.set_tracer_root_layout(n, tile_lo);
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
            for (k, h) in tiles.iter_mut().enumerate() {
                symbi::symbi_xpu::with_device(devices[k], || h.prime());
            }
        }

        // the full-size output hierarchy (root + the full region): gather scatters each level's tile
        // interiors into it. lives on device 0 (touched only at output).
        let mut global = symbi::symbi_xpu::with_device(0, || -> Result<Hier, String> {
            let groot = Sim::build($regime, host_eos(cfg), $geom)
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
                // same non-ideal knobs as the base (the viscous pass runs on the finest
                // level only).
                s.substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .expect("fine kernel set")
                    .with_viscosity(cfg.viscosity)
                    .with_resistivity(cfg.resistivity)
            };
            let gh = Hierarchy::with_refinement(groot, gsub, &regions, prolong, make)
                .map_err(|e| format!("global refinement build: {e:?}"))?;
            gh.seed_fine_from_coarse()
                .map_err(|e| format!("global fine seed: {e:?}"))?;
            Ok(gh)
        })?;
        if cfg.n_tracers > 0 {
            symbi::sim::refinement::seed_decomposed_hierarchy_tracers(
                &global,
                &mut tiles,
                cfg.n_tracers,
            );
            global.attach_mass_tracers(cfg.n_tracers);
            if let Some(order) = configured_ito_order(cfg)? {
                for tile in &mut tiles {
                    for level in &mut tile.levels {
                        convert_seeded_tracers(&mut level.state, order)?;
                    }
                }
                for level in &mut global.levels {
                    convert_seeded_tracers(&mut level.state, order)?;
                }
            }
        }

        run_refined_decomposed_loop(cfg, tiles, global, counts)
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
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        // `$d` is the grid dimension, `$dof` the momentum-component count; they differ for the
        // swirl lift (the azimuthal momentum on a 2D (r, z)/(r, theta) grid, DOF = 3).
        type Sim = SimDefaultGeneric<$regime_ty, $d, $dof, $geom_ty, EosSelect<f64>>;

        // refinement + gpus>1 takes the decomposed-hierarchy path (per-tile hierarchies + the
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
        // the dye IC (when present) spans the whole global grid in the same axis-0-fastest order
        // as the prims; each tile reads its own sub-box out of it below.
        if !cfg.chi_ic.is_empty() && cfg.chi_ic.len() != total {
            return Err(format!(
                "passive_scalar yielded {} dye values, expected {total}",
                cfg.chi_ic.len()
            ));
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
            // the tile's first global cell on each axis; the origin and the coordinate maps both
            // derive from it, so a log radial axis advances multiplicatively rather than by g*dx.
            let tile_lo: [usize; $d] = std::array::from_fn(|ax| tc[ax] * m[ax]);
            let origin: [f64; $d] = tile_origin::<$d>(cfg, tile_lo);
            let spacing: [f64; $d] = std::array::from_fn(|ax| cfg.dx[ax]);
            let bnd = Boundaries(std::array::from_fn(|ax| {
                let lo = if tc[ax] == 0 { phys.0[ax][0] } else { BoundaryType::CoarseFine };
                let hi = if tc[ax] == counts[ax] - 1 { phys.0[ax][1] } else { BoundaryType::CoarseFine };
                [lo, hi]
            }));
            let dev = flat as i32;
            let built = symbi::symbi_xpu::with_device(dev, || -> Result<(Sim, _), String> {
                let sim = Sim::build($regime, host_eos(cfg), $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(spacing)
                    .coord_maps(tile_axis_maps::<$d>(cfg, tile_lo))
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
                            pre: row[1 + $dof],
                        }
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
                    let ilo: [isize; $d] =
                        std::array::from_fn(|ax| sim.geom.interior.spaces[ax].lo);
                    {
                        let cons_chi = sim.fields.cons.chi_field().expect("cons chi");
                        let prim_chi = sim.fields.prim.chi_field().expect("prim chi");
                        for c in sim.geom.interior.iter() {
                            let mut lin = 0usize;
                            let mut stride = 1usize;
                            for ax in 0..$d {
                                let g = tc[ax] * m[ax] + (c[ax] - ilo[ax]) as usize;
                                lin += g * stride;
                                stride *= n[ax];
                            }
                            let chi_v = cfg.chi_ic[lin];
                            let rho = *sim.fields.cons.den.view().at(c);
                            cons_chi.view_mut().set(c, rho * chi_v);
                            prim_chi.view_mut().set(c, chi_v);
                        }
                    }
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
                attach_motion(&mut sim, cfg)?;
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?;
                // attach the user source per tile (two-pass via attach_runtime_source). each tile
                // evaluates S at its own global coords (the per-tile origin above), so a
                // position-dependent force is correct across cuts -- decomposed == monolithic
                // to round-off. the global output sim carries no
                // source (it is touched only at gather/output).
                let sub = attach_configured_sources(
                    sub,
                    &cfg.source_jsons,
                    <$regime_ty as Regime<f64, $dof>>::SPEC,
                    &state_law_of(cfg, <$regime_ty as Regime<f64, $dof>>::SPEC),
                )?;
                // register the driven (dynamic) boundary DAGs on every tile, in Driven-id order:
                // the ids ride the boundary enum copied from the physical faces, so only edge
                // tiles carry Driven faces and interior tiles hold the dags inert. each tile
                // evaluates the coordinate prescription at its own global coords, the same
                // contract as the per-tile user source (decomposed == monolithic to round-off).
                let sub = {
                    let mut sub = sub;
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
                    sub
                };
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        // one full-size sim as the output view: the gather scatters tile interiors into it and
        // the existing writer serializes it. lives on device 0 (it is only touched at output).
        let global = Sim::build($regime, host_eos(cfg), $geom)
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
                    pre: row[1 + $dof],
                }
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
            let per_tile = partition_configured_tracers(&global, cfg, counts);
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
        run_decomposed_loop(cfg, tiles, global, counts)
    }};
}

/// expand the (geometry x dims) arms for one hydro regime. cartesian / spherical
/// / cylindrical are all wired across 1/2/3d (each is a unit-struct `Metric`).
macro_rules! hydro_dispatch {
    ($cfg:expr, $prims:expr, $regime:expr, $regime_ty:ty) => {
        match ($cfg.dims, $cfg.coord_system.as_str()) {
            // fail-loud guard: a non-minkowski spacetime that is not one of the baked GR
            // combinations below would otherwise fall through to a flat `(dims, coords)` arm and run
            // silently on a Minkowski metric (wrong physics, zero warning). the matches! set is the
            // single source of truth for the baked GR-hydro arms; `test_dispatch_rejects_unbaked_gr`
            // asserts it stays in lockstep with the actual arms (guarded-arm-or-Err, never silent-flat).
            (d, c)
                if $cfg.spacetime != "minkowski"
                    && !matches!(
                        (d, c, $cfg.spacetime.as_str()),
                        (3, "cartesian", "schwarzschild_ks")
                            | (3, "cartesian", "kerr_ks")
                            | (1, "spherical", "schwarzschild_ks")
                            | (2, "spherical", "kerr_ks")
                            | (2, "spherical", "schwarzschild_ks")
                            | (2, "cylindrical", "schwarzschild_ks")
                            | (3, "cylindrical", "schwarzschild_ks")
                            | (3, "cylindrical", "kerr_ks")
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
            // needs the azimuthal momentum DOF, so the 5-tuple (swirl) generator row is
            // required — a 4-tuple config is a setup error with no fallback.
            (2, "spherical") if $cfg.spacetime == "kerr_ks" => {
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
            (2, "spherical") if $cfg.spacetime == "schwarzschild_ks" => {
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
            // GR (kerr-schild) cylindrical 2D: the plane selector splits the two charts. the (R, phi)
            // equatorial disk (planar_cylindrical) is diagonal (z = 0, r = R), DOF = 2 (v_R, v_phi);
            // the (R, z) 2.5D axisymmetric-swirl (the default) lifts v_phi, DOF = 3, requiring the
            // 5-tuple. both use the one SchwarzschildKSCylindrical metric (D = 2 disk / D = 3 swirl).
            (2, "cylindrical") if $cfg.spacetime == "schwarzschild_ks" => match $cfg.cyl_plane {
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
            sim.with_bodies(build_bodies_and_horizon::<$d>(cfg))
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
        let theta = build_theta(cfg);
        let sub = sim
            .substrate()
            .theta(theta)
            .with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?
            .ct_method(cfg.ct_method)
            .with_viscosity(cfg.viscosity)
            .with_resistivity(cfg.resistivity)
            .with_excision(
                cfg.excision_radius,
                cfg.excision_rho_scale,
                cfg.excision_pre_scale,
            );
        // attach a user source expression to the MHD hydro slots (den/mom/nrg).
        // rmhd is relativistic -> only kind="raw"; nmhd takes force/cooling/relax.
        // B is CT-evolved, so it takes no cell source. single-grid only.
        if !cfg.source_jsons.is_empty() && cfg.refinement_enabled {
            return Err(
                "user source expressions are not yet supported with mesh refinement".to_string(),
            );
        }
        let sub = attach_configured_sources(
            sub,
            &cfg.source_jsons,
            <$regime_ty as Regime<f64, $d>>::SPEC,
            &state_law_of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC),
        )?;
        // register driven (dynamic) boundaries in Driven-id order so `Driven(id)` on a face
        // matches `driven_exprs[id]`. a complete prim prescription incl. the cell B (purely
        // toroidal: in-plane B = 0, out-of-plane B_phi injected). single-grid only.
        let mut sub = sub;
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
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            // the fine kernel set mirrors the base: the same ct method (the constructor
            // default is Contact, which would silently downgrade a UCT run's fine levels)
            // and the same driven-boundary DAGs in Driven-id order — a fine level flush
            // against a driven physical face inherits Driven(id) and evaluates the
            // coordinate DAG at its own finer ghost coordinates. already validated at the
            // base registration.
            let mut ks = s
                .substrate()
                .theta(theta)
                .with_solver(solver)
                .expect("fine-level kernel set")
                .ct_method(cfg.ct_method)
                .with_viscosity(cfg.viscosity)
                .with_resistivity(cfg.resistivity);
            for json in &cfg.driven_exprs {
                let bcfg =
                    symbi_hydro::SourceConfig::from_json(json).expect("fine-level boundary parse");
                let built = symbi_hydro::expr_bridge::build_boundary_dag(
                    &bcfg,
                    <$regime_ty as Regime<f64, $d>>::SPEC,
                )
                .expect("fine-level boundary lower");
                ks = ks.with_driven_boundary(built, bcfg.params.clone()).0;
            }
            ks
        });
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
            sim.with_bodies(build_bodies_and_horizon::<$d>(cfg))
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
        let theta = build_theta(cfg);
        let sub = sim
            .substrate()
            .theta(theta)
            .with_solver(cfg.solver)
            .map_err(|e| format!("substrate/solver: {e:?}"))?
            .ct_method(cfg.ct_method)
            .with_viscosity(cfg.viscosity)
            .with_resistivity(cfg.resistivity);
        // attach a user source. iso MHD has no energy -> momentum-only force/relax,
        // raw den/mom (raw->nrg rejected); B is CT-evolved. single-grid only.
        if !cfg.source_jsons.is_empty() && cfg.refinement_enabled {
            return Err(
                "user source expressions are not yet supported with mesh refinement".to_string(),
            );
        }
        let sub = attach_configured_sources(
            sub,
            &cfg.source_jsons,
            <IsothermalMhd as Regime<f64, $d>>::SPEC,
            &state_law_of(cfg, <IsothermalMhd as Regime<f64, $d>>::SPEC),
        )?;
        // register driven (dynamic) boundaries in Driven-id order so `Driven(id)` on a face
        // matches `driven_exprs[id]`. the iso-MHD prescription is [rho, vel.., B..] (no
        // pressure slot; the eos closure p = cs^2 rho covers the ghosts). purely toroidal
        // injection: in-plane B = 0, out-of-plane B_phi injected (div-free by axisymmetry).
        let mut sub = sub;
        for json in &cfg.driven_exprs {
            let bcfg = symbi_hydro::SourceConfig::from_json(json)
                .map_err(|e| format!("boundary expression parse: {e}"))?;
            let built = symbi_hydro::expr_bridge::build_boundary_dag(
                &bcfg,
                <IsothermalMhd as Regime<f64, $d>>::SPEC,
            )
            .map_err(|e| format!("boundary expression lower: {e}"))?;
            sub = sub.with_driven_boundary(built, bcfg.params.clone()).0;
        }
        let solver = cfg.solver;
        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            // the fine kernel set mirrors the base: the same ct method (the constructor
            // default is Contact, which would silently downgrade a UCT run's fine levels)
            // and the same driven-boundary DAGs in Driven-id order — a fine level flush
            // against a driven physical face inherits Driven(id) and evaluates the
            // coordinate DAG at its own finer ghost coordinates. already validated at the
            // base registration.
            let mut ks = s
                .substrate()
                .theta(theta)
                .with_solver(solver)
                .expect("fine-level kernel set")
                .ct_method(cfg.ct_method)
                .with_viscosity(cfg.viscosity)
                .with_resistivity(cfg.resistivity);
            for json in &cfg.driven_exprs {
                let bcfg =
                    symbi_hydro::SourceConfig::from_json(json).expect("fine-level boundary parse");
                let built = symbi_hydro::expr_bridge::build_boundary_dag(
                    &bcfg,
                    <IsothermalMhd as Regime<f64, $d>>::SPEC,
                )
                .expect("fine-level boundary lower");
                ks = ks.with_driven_boundary(built, bcfg.params.clone()).0;
            }
            ks
        });
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
/// single-level only: refinement / bodies / user sources with gpus>1 are refused.
macro_rules! build_and_run_mhd_decomposed {
    ($cfg:expr, $prims:expr, $bufs:expr, $regime:expr, $regime_ty:ty, $d:literal, $geom:expr, $geom_ty:ty) => {{
        use symbi::sim::decomp::{decompose_grid, unflatten};
        let cfg: &Config = $cfg;
        let prims: &[Vec<f64>] = $prims;
        let bufs: &[Vec<f64>] = $bufs;
        type Sim = SimDefaultGeneric<$regime_ty, $d, 3, $geom_ty, EosSelect<f64>>;

        // multi-gpu MHD is single-level: refinement needs its own cross-level multi-tile
        // handling, so it is refused.
        if cfg.refinement_enabled {
            return Err("gpus>1 does not yet support mesh refinement; set gpus=1 or disable refinement".to_string());
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
            // the tile's first global cell on each axis; the origin and the coordinate maps both
            // derive from it, so a log radial axis advances multiplicatively rather than by g*dx.
            let tile_lo: [usize; $d] = std::array::from_fn(|ax| tc[ax] * m[ax]);
            let origin: [f64; $d] = tile_origin::<$d>(cfg, tile_lo);
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
                let sim = Sim::build($regime, host_eos(cfg), $geom)
                    .cells(m)
                    .origin(origin)
                    .spacing(spacing)
                    .coord_maps(tile_axis_maps::<$d>(cfg, tile_lo))
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
                // bodies at their global positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically.

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    sim.with_bodies(build_bodies_and_horizon::<$d>(cfg))
                };
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?
                    .ct_method(ct)
                    .with_viscosity(cfg.viscosity)
                    .with_resistivity(cfg.resistivity)
                    .with_excision(
                        cfg.excision_radius,
                        cfg.excision_rho_scale,
                        cfg.excision_pre_scale,
                    );
                // attach the user source per tile (two-pass). targets the mhd hydro slots
                // (den/mom/nrg); B is CT-evolved, so it takes no cell source. each tile evaluates S at its
                // own global coords. rmhd is relativistic -> raw only (enforced in build_user_source).
                let sub = attach_configured_sources(
                    sub,
                    &cfg.source_jsons,
                    <$regime_ty as Regime<f64, $d>>::SPEC,
                    &state_law_of(cfg, <$regime_ty as Regime<f64, $d>>::SPEC),
                )?;
                // register the driven (dynamic) boundary DAGs on every tile, in Driven-id order:
                // only edge tiles carry Driven faces (interior cuts are CoarseFine), and each
                // tile evaluates the coordinate prescription at its own global coords -- the
                // same contract as the per-tile user source. the prescription covers the hydro
                // prims + the cell B; the staggered face B rides the CT ghost fill and the
                // transverse halo exchange, identically to the monolithic run.
                let sub = {
                    let mut sub = sub;
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
                    sub
                };
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        // the full-size output view: gather scatters tile interiors (cells + cell B) and faces into
        // it each checkpoint; seed the faces so `bface_initialized` is set (the gather overwrites
        // the interior). lives on device 0 (touched only at output).
        let mut global = Sim::build($regime, host_eos(cfg), $geom)
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

        if cfg.n_tracers > 0 {
            let per_tile = partition_configured_tracers(&global, cfg, counts);
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

        run_decomposed_loop(cfg, tiles, global, counts)
    }};
}

/// the multi-gpu (gpus>1) isothermal MHD path: identical tiling/face-seeding/gather to the
/// adiabatic decomposed macro, but the iso primitive has no pressure slot (`IsoModel` zst) and the
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
            // the tile's first global cell on each axis; the origin and the coordinate maps both
            // derive from it, so a log radial axis advances multiplicatively rather than by g*dx.
            let tile_lo: [usize; $d] = std::array::from_fn(|ax| tc[ax] * m[ax]);
            let origin: [f64; $d] = tile_origin::<$d>(cfg, tile_lo);
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
                    .coord_maps(tile_axis_maps::<$d>(cfg, tile_lo))
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
                // bodies at their global positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically.

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    sim.with_bodies(build_bodies_and_horizon::<$d>(cfg))
                };
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_solver(solver)
                    .map_err(|e| format!("tile {flat} substrate/solver: {e:?}"))?
                    .ct_method(ct)
                    .with_viscosity(cfg.viscosity)
                    .with_resistivity(cfg.resistivity)
                    .with_excision(
                        cfg.excision_radius,
                        cfg.excision_rho_scale,
                        cfg.excision_pre_scale,
                    );
                // attach the user source per tile (two-pass). iso mhd has no energy -> momentum-only
                // force/relax, raw den/mom; B is CT-evolved. each tile evaluates S at its own coords.
                let sub = attach_configured_sources(
                    sub,
                    &cfg.source_jsons,
                    <IsothermalMhd as Regime<f64, $d>>::SPEC,
                    &state_law_of(cfg, <IsothermalMhd as Regime<f64, $d>>::SPEC),
                )?;
                // register the driven (dynamic) boundary DAGs on every tile, in Driven-id order:
                // only edge tiles carry Driven faces (interior cuts are CoarseFine), and each
                // tile evaluates the coordinate prescription at its own global coords. the iso-mhd
                // prescription is [rho, vel.., B..] (no pressure slot; the eos closure is
                // p = cs^2 rho); the staggered face B rides the CT ghost fill + halo exchange.
                let sub = {
                    let mut sub = sub;
                    for json in &cfg.driven_exprs {
                        let bcfg = symbi_hydro::SourceConfig::from_json(json)
                            .map_err(|e| format!("boundary expression parse: {e}"))?;
                        let built = symbi_hydro::expr_bridge::build_boundary_dag(
                            &bcfg,
                            <IsothermalMhd as Regime<f64, $d>>::SPEC,
                        )
                        .map_err(|e| format!("boundary expression lower: {e}"))?;
                        sub = sub.with_driven_boundary(built, bcfg.params.clone()).0;
                    }
                    sub
                };
                Ok((sim, sub))
            })?;
            tiles.push(built);
        }

        let mut global = Sim::build(IsothermalMhd, Isothermal { cs: cfg.cs }, $geom)
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

        if cfg.n_tracers > 0 {
            let per_tile = partition_configured_tracers(&global, cfg, counts);
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

        run_decomposed_loop(cfg, tiles, global, counts)
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
                if $cfg.spacetime != "minkowski"
                    && !matches!(
                        (d, c, $cfg.spacetime.as_str()),
                        (1, "spherical", "schwarzschild_ks")
                            | (2, "spherical", "kerr_ks")
                            | (3, "cartesian", "schwarzschild_ks")
                            | (3, "cartesian", "kerr_ks")
                            | (2, "cylindrical", "schwarzschild_ks")
                    ) =>
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
            (2, "cylindrical") if $cfg.spacetime == "schwarzschild_ks" => build_and_run_mhd!(
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
            // the tile's first global cell on each axis; the origin and the coordinate maps both
            // derive from it, so a log radial axis advances multiplicatively rather than by g*dx.
            let tile_lo: [usize; $d] = std::array::from_fn(|ax| tc[ax] * m[ax]);
            let origin: [f64; $d] = tile_origin::<$d>(cfg, tile_lo);
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
                    .coord_maps(tile_axis_maps::<$d>(cfg, tile_lo))
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
                // bodies at their global positions; each applies the source to its own cells. the
                // decomposed loop sums the backward feedback across tiles + advances the prescribed
                // binary orbit identically.

                let sim = if cfg.bodies.is_empty() {
                    sim
                } else {
                    sim.with_bodies(build_bodies_and_horizon::<$d>(cfg))
                };
                // clock + mesh motion per tile: every tile carries the identical a(t) law and
                // the decomposed loop advances them in lockstep with the shared dt.
                let mut sim = sim;
                attach_motion(&mut sim, cfg)?;
                let sim = sim;
                let sub = sim
                    .substrate()
                    .theta(theta)
                    .with_viscosity(cfg.viscosity)
                    .with_alpha(cfg.alpha);
                // attach the user source per tile (two-pass). iso has no energy -> momentum-only
                // force/relax, raw den/mom. each tile evaluates S at its own global coords.
                let sub = attach_configured_sources(
                    sub,
                    &cfg.source_jsons,
                    <IsoNewtonian as Regime<f64, $d>>::SPEC,
                    &state_law_of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC),
                )?;
                // register the driven (dynamic) boundary DAGs on every tile, in Driven-id order:
                // only edge tiles carry Driven faces (interior cuts are CoarseFine), and each
                // tile evaluates the coordinate prescription at its own global coords -- the same
                // contract as the per-tile user source. iso prescribes [rho, vel..]; the ghost
                // pressure is the eos closure p = cs^2 rho (globally isothermal on this path).
                let sub = {
                    let mut sub = sub;
                    for json in &cfg.driven_exprs {
                        let bcfg = symbi_hydro::SourceConfig::from_json(json)
                            .map_err(|e| format!("boundary expression parse: {e}"))?;
                        let built = symbi_hydro::expr_bridge::build_boundary_dag(
                            &bcfg,
                            <IsoNewtonian as Regime<f64, $d>>::SPEC,
                        )
                        .map_err(|e| format!("boundary expression lower: {e}"))?;
                        sub = sub.with_driven_boundary(built, bcfg.params.clone()).0;
                    }
                    sub
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
                PrimG::<f64, $d, IsoModel> {
                    rho: row[0],
                    vel: Tensor::new(std::array::from_fn(|k| row[1 + k])),
                    pre: Default::default(),
                }
            })
            .build();

        // attach immersed bodies (gravity / accretion sinks + bonded fragments) when
        // declared. a refined run attaches them to the hierarchy instead (finest level
        // owns the sinks), exactly like the energy-regime hydro macro.
        let has_any_body = !cfg.bodies.is_empty() || cfg.bonded_assembly.is_some();
        let sim = if !has_any_body || cfg.refinement_enabled {
            sim
        } else {
            let mut coll = build_bodies_and_horizon::<$d>(cfg);
            let mut shapes = build_body_shapes(&cfg.bodies);
            let physics = cfg.bonded_assembly.as_ref().map(|asm| {
                let (with_frags, physics) = append_fragments::<$d>(coll.clone(), &mut shapes, asm);
                coll = with_frags;
                physics
            });
            let mut sim = sim.with_bodies(coll);
            sim.attach_body_shapes(shapes);
            if let Some(physics) = physics {
                sim.attach_fragment_physics(physics);
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
        // iso is HLLE-only; the substrate front door gives the kernel-set directly.
        let theta = build_theta(cfg);
        // the constant-nu viscosity — the iso path has its own
        // build macro, so it needs its own .with_viscosity (the base hydro build
        // at build_and_run_hydro does not cover it).
        let sub = sim
            .substrate()
            .theta(theta)
            .with_viscosity(cfg.viscosity)
            .with_alpha(cfg.alpha);
        // attach a user source expression. iso has no energy, so build_user_source
        // (against the iso spec) drops the energy overlay for force/relax and rejects
        // raw->nrg; den/mom sources work. refined runs re-attach the same source to
        // each fine level in the into_hierarchy make-closure below.
        let sub = attach_configured_sources(
            sub,
            &cfg.source_jsons,
            <IsoNewtonian as Regime<f64, $d>>::SPEC,
            &state_law_of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC),
        )?;

        // register driven (dynamic) boundaries in Driven-id order, lowered against the iso
        // spec: the prescription is [rho, vel..] only (no pressure slot; the ghost pressure
        // re-derives as cs^2 * rho from the held temperature after every fill).
        let sub = {
            let mut sub = sub;
            for json in &cfg.driven_exprs {
                let bcfg = symbi_hydro::SourceConfig::from_json(json)
                    .map_err(|e| format!("boundary expression parse: {e}"))?;
                let built = symbi_hydro::expr_bridge::build_boundary_dag(
                    &bcfg,
                    <IsoNewtonian as Regime<f64, $d>>::SPEC,
                )
                .map_err(|e| format!("boundary expression lower: {e}"))?;
                sub = sub.with_driven_boundary(built, bcfg.params.clone()).0;
            }
            sub
        };
        // register the Neumann/Robin gradient boundaries; iso re-derives pre = cs^2*rho at the ghost
        // (the pressure coefficients are ignored — the substrate feeds cs^2 to the shared kernel).
        let sub = {
            let mut sub = sub;
            for spec in &cfg.gradient_bcs {
                use symbi::regimes::substrate_kernels::GradientBc;
                let gbc = match spec.kind.as_str() {
                    "neumann" => GradientBc::Neumann(spec.coeffs.clone()),
                    "robin" => GradientBc::Robin(
                        spec.coeffs.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                    ),
                    other => return Err(format!("unknown gradient boundary kind '{other}'")),
                };
                sub = sub.with_gradient_boundary(gbc).0;
            }
            sub
        };

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
        }

        let mut hier = into_hierarchy!(sim, sub, cfg, $d, |s| {
            let ks = s
                .substrate()
                .theta(theta)
                .with_viscosity(cfg.viscosity)
                .with_alpha(cfg.alpha);
            // register the same driven-boundary DAGs on each fine level, in Driven-id order: a
            // fine level flush against a driven physical face inherits `Driven(id)` there; an
            // interior fine level has only CoarseFine faces and never consults the dags. the
            // fine ghost pressure re-derives as cs2 * rho after the fill (the fine cs2 is
            // prolonged + clamp-extended). already validated at the base registration.
            let mut ks = ks;
            for json in &cfg.driven_exprs {
                let bcfg = symbi_hydro::SourceConfig::from_json(json)
                    .expect("fine-level boundary parse");
                let built = symbi_hydro::expr_bridge::build_boundary_dag(
                    &bcfg,
                    <IsoNewtonian as Regime<f64, $d>>::SPEC,
                )
                .expect("fine-level boundary lower");
                ks = ks.with_driven_boundary(built, bcfg.params.clone()).0;
            }
            for spec in &cfg.gradient_bcs {
                use symbi::regimes::substrate_kernels::GradientBc;
                let boundary = match spec.kind.as_str() {
                    "neumann" => GradientBc::Neumann(spec.coeffs.clone()),
                    "robin" => GradientBc::Robin(
                        spec.coeffs
                            .chunks_exact(3)
                            .map(|coeff| [coeff[0], coeff[1], coeff[2]])
                            .collect(),
                    ),
                    other => panic!("unknown fine-level gradient boundary kind '{other}'"),
                };
                ks = ks.with_gradient_boundary(boundary).0;
            }
            let ks = ks;
            // attach the same user source to each fine level (a base-only attach
            // would be restricted away by the fine solution). already validated
            // at the base attach.
            attach_configured_sources(
                ks,
                &cfg.source_jsons,
                <IsoNewtonian as Regime<f64, $d>>::SPEC,
                &state_law_of(cfg, <IsoNewtonian as Regime<f64, $d>>::SPEC),
            )
            .expect("fine-level source attach")
        });
        // a refined run attaches its immersed bodies to the hierarchy: the finest
        // level owns the full (accreting) bodies, coarser levels a gravity-only
        // proxy (finest-owns-bodies, so the sink applies once).
        if !cfg.bodies.is_empty() && cfg.refinement_enabled {
            hier = hier.with_bodies(build_bodies_and_horizon::<$d>(cfg));
            hier.attach_body_shapes(build_body_shapes(&cfg.bodies));
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
            let declared = symbi_hydro::EquilibriumConfig::from_json(target)
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

        // the declared perturbation at every level's own cell centers (see the adiabatic
        // build): an energy-free regime's pressure slot discards its delta, so the same
        // component contract reads one entry shorter here.
        if cfg.restart_path.is_none() {
            if let Some(ref declared) = cfg.perturbation_json {
                let cfg_p = symbi_hydro::EquilibriumConfig::from_json(declared)
                    .map_err(|e| format!("perturbation parse: {e}"))?;
                let expression = cfg_p
                    .to_expression()
                    .map_err(|e| format!("perturbation lower: {e:?}"))?;
                if cfg_p.outputs.len() != 1 + $d {
                    return Err(format!(
                        "perturbation_expressions supplies {} component(s); an isothermal \
                         {}-dimensional run needs {} — density and {} velocity component(s)",
                        cfg_p.outputs.len(),
                        $d,
                        1 + $d,
                        $d
                    ));
                }
                hier.perturb_cells(|x, p| {
                    let at = |aa: usize| x.get(aa).copied().unwrap_or(0.0);
                    let d = expression.eval(at(0), at(1), at(2), 0.0);
                    let mut p = p;
                    p.rho += d[0];
                    for ax in 0..$d {
                        p.vel[ax] += d[1 + ax];
                    }
                    p
                });
                hier.sync_perturbed();
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

/// runtime dispatch on the config tags -> a monomorphized sim. hydro regimes
/// (newtonian/rhd/isothermal) x cartesian (+ curvilinear for adiabatic) x 1/2/3d;
/// the mhd regimes (rmhd/nmhd/imhd) x cartesian x 1/2/3d.
fn dispatch_and_run(cfg: &Config, prims: &[Vec<f64>], bfields: &[Vec<f64>]) -> Result<(), String> {
    validate_porous_body_overlaps(&cfg.bodies)?;
    // static mesh refinement is wired for hydro (incl. globally-isothermal). the two cases
    // refused below need fine-level prolongation the transfer set does not carry:
    if cfg.refinement_enabled
        && cfg.regime.contains("mhd")
        && !(cfg.dims == 3 && cfg.coord_system == "cartesian")
    {
        return Err("mhd refinement requires a 3d cartesian grid (the CT \
                    reflux assumes 1/dx curl coefficients)"
            .to_string());
    }
    // mesh motion is single-grid uniform-spacing hydro only.
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
    // immersed bodies + refinement: the finest-owns-bodies AMR sync (`hier.with_bodies` — full
    // bodies on the finest level, gravity-only proxy on coarser) is wired for every hydro regime
    // (newtonian/rhd/isothermal — the iso body source/feedback/penalize kernels are baked and the
    // build macro is shared). MHD is refused: coupling a body to a staggered face field is
    // not wired.
    if !cfg.bodies.is_empty() && cfg.refinement_enabled && cfg.regime.contains("mhd") {
        return Err("immersed bodies with refinement are not wired for MHD (staggered-B body coupling pending)"
            .to_string());
    }
    // gpus>1 takes the decomposed run loop: single-level hydro (newtonian/rhd/isothermal) and
    // single-level MHD (rmhd/nmhd/imhd, the equivalence-tested staggered-CT halo exchange + face
    // gather). reject every other case here so a multi-gpu request never
    // silently runs on one device.
    if cfg.n_gpus > 1 {
        if !matches!(
            cfg.regime.as_str(),
            "newtonian" | "rhd" | "isothermal" | "rmhd" | "nmhd" | "imhd"
        ) {
            return Err(format!(
                "gpus>1 is wired for hydro (newtonian, rhd, isothermal) and mhd (rmhd, nmhd, \
                 imhd); regime '{}' runs single-gpu for now (set gpus=1)",
                cfg.regime
            ));
        }
        // immersed bodies (incl. moving binaries) and their force/accreted-mass diagnostics are
        // wired for gpus>1: the decomposed loop applies the body source per tile, sums the backward
        // feedback across tiles, and advances the prescribed orbit identically, so no refusal is
        // needed here.
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
    // the passive scalar (dye) rides the uni-grid cartesian newtonian path: the chi
    // kernels are baked cartesian, the drain/wall
    // every unwired combination fails loud — a silently undyed or wrongly-dyed run reads as
    // valid science.
    if !cfg.chi_ic.is_empty() {
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
            let bcfg = symbi_hydro::SourceConfig::from_json(json)
                .map_err(|e| format!("driven boundary {id}: expression parse: {e}"))?;
            let want = symbi_hydro::expr_bridge::boundary_prim_arity(
                &symbi_hydro::regime_spec::NEWTONIAN_SPEC,
                bcfg.dim,
            ) + 1;
            if bcfg.outputs.len() != want {
                return Err(format!(
                    "driven boundary {id} prescribes {} outputs, but a run carrying a passive \
                     scalar needs {want}: the full prim state plus a trailing dye concentration \
                     for the injected fluid",
                    bcfg.outputs.len(),
                ));
            }
        }
    }
    // mass-transport tracers consume the accepted finite-volume density flux.
    // curvilinear decomposition and mesh motion still require explicit
    // material-volume geometry in the decomposed driver.
    // non-ideal transport. the baked matrix is narrower than the config surface, so an
    // unsupported chart would otherwise surface mid-run as an unbaked-kernel panic rather than a
    // config-time refusal — a worse failure than the loud startup rejection every other capability
    // gets. the accepted sets below mirror `gen_viscous` / `gen_resistive` in symbi-aot/build.rs.
    // scoped to the regimes whose kernel sets actually dispatch a viscous operator. the
    // relativistic regimes accept the coefficient and ignore it, and `alpha` is a bare config key
    // that other problems legitimately use for their own meaning (a wave amplitude, say), so
    // keying the refusal on the value alone would reject configs that never asked for viscosity.
    let viscous_regime = matches!(
        cfg.regime.as_str(),
        "newtonian" | "isothermal" | "nmhd" | "imhd"
    );
    alpha_key_verdict(&cfg.regime, cfg.bare_alpha_key, cfg.alpha)?;
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
        // multi-device hydro uses global container identities and migrates
        // complete tracer records across decomposition cuts.
        if cfg.n_gpus > 1
            && !matches!(
                cfg.regime.as_str(),
                "newtonian" | "rhd" | "nmhd" | "rmhd" | "imhd"
            )
        {
            return Err(format!(
                "n_tracers with gpus > 1 is wired for hydro/mhd; regime '{}' not yet",
                cfg.regime
            ));
        }
    }
    if !cfg.cohort_ic.is_empty() && cfg.n_tracers == 0 {
        return Err("tracer_cohort requires n_tracers > 0".to_string());
    }
    // ppm evolution reconstruction: baked for the flat cartesian newtonian ideal-gas
    // family only. refinement is refused because the coarse-fine transfer has no
    // prolongation one order above ppm — a refined run would silently lose an order at
    // every level boundary inside the domain. decomposed (gpus > 1) waits on a ppm
    // cut-equivalence gate. every unsupported combination refuses here, before any
    // build, with the reason attached.
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
        if cfg.n_gpus > 1 {
            return Err(
                "ppm reconstruction with gpus > 1 awaits the decomposed cut-equivalence \
                 gate for the widened (-3..+2) exchange"
                    .to_string(),
            );
        }
    }
    // the synge (taub-mathews) eos: baked for the flat rhd family only. every
    // unsupported combination refuses here, before any build, with the reason
    // attached.
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
        if cfg.n_gpus > 1 {
            return Err(
                "eos = 'synge' with gpus > 1 awaits the decomposed equivalence gate".to_string(),
            );
        }
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
            BoundaryType::CoarseFine => "coarse_fine",
            BoundaryType::Driven(_) => "dynamic",
            BoundaryType::Neumann(_) => "neumann",
            BoundaryType::Robin(_) => "robin",
        })
        .collect::<Vec<_>>()
        .join(",");
    Metadata::new()
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
) -> PyResult<()> {
    let mut cfg = parse_config(sim_info)?;
    validate_config_preflight(&cfg).map_err(PyValueError::new_err)?;
    // drain the passive-scalar generator (None = undyed). one flat cell-centered
    // buffer; the per-arm seeding checks the count against the interior.
    if let Some(chi_field) = chi_field {
        if !chi_field.is_none() {
            let mut vals = Vec::new();
            for v in chi_field.try_iter()? {
                vals.push(v?.extract::<f64>()?);
            }
            cfg.chi_ic = vals;
        }
    }
    if let Some(cohort_field) = cohort_field {
        if !cohort_field.is_none() {
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
            cfg.cohort_ic = vals;
        }
    }
    validate_config_preflight(&cfg).map_err(PyValueError::new_err)?;
    // a non-linear (log) radial axis is supported under decomposition: each tile carries the global
    // per-cell slope and an origin advanced multiplicatively to its first global cell
    // (`start * 10^(g * slope)`), so a tile's local index i lands exactly where the undecomposed grid
    // puts g + i, and refinement regions are clipped against the tile's map rather than a linear
    // extent. gated by tile_coord_tests.
    // evaluate the scale-factor callables at the start time (gil held). the rust
    // mesh-motion model integrates `a` from the constant rate `a_dot` (linear /
    // free homologous expansion, a_ddot = 0), so a single sample at t0 suffices.
    if cfg.mesh_motion {
        let t0 = cfg.start_time;
        cfg.scale_a0 = a.call1((t0,))?.extract::<f64>()?;
        cfg.scale_adot = adot.call1((t0,))?.extract::<f64>()?;
    }
    let prims = drain_prims(prim_gen)?;
    if cfg.excision_radius > 0.0 {
        (cfg.excision_rho_scale, cfg.excision_pre_scale) =
            initial_excision_scales(&prims).map_err(PyValueError::new_err)?;
    }
    let bfields = drain_bfields(staggered_bfields)?;

    // the solve is pure rust with no python access — release the gil so rayon
    // gets real parallelism (and python stays responsive).
    py.detach(|| dispatch_and_run(&cfg, &prims, &bfields))
        .map_err(PyRuntimeError::new_err)
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
    if cfg.restart_path.is_some() && cfg.n_gpus > 1 {
        return Err("checkpoint restart is not yet supported with decomposition".to_string());
    }
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
        let parsed = symbi_hydro::SourceConfig::from_json(&source.json)
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
    for (ii, json) in cfg.driven_exprs.iter().enumerate() {
        symbi_hydro::SourceConfig::from_json(json)
            .map_err(|err| format!("driven boundary {ii} parse: {err}"))?;
    }
    if let Some(json) = &cfg.motion_json {
        symbi_hydro::motion_law::MotionLaw::from_json(json, cfg.start_time, cfg.t_final)
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

// shared module body. the pyo3 entry-point name below decides the `PyInit_*`
// symbol and the imported module name: `cpu_ext` for the default build,
// `gpu_ext` for the cuda build. both compile the same source — cuda only adds
// the NVRTC device path — so the registration is identical and lives here.
fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(validate_simulation, m)?)?;
    m.add_function(wrap_pyfunction!(attach_dashboard, m)?)?;
    m.add_function(wrap_pyfunction!(bondi_profile, m)?)?;
    m.add_function(wrap_pyfunction!(mdot_bondi, m)?)?;
    m.add_function(wrap_pyfunction!(bondi_sonic_radius, m)?)?;
    m.add_function(wrap_pyfunction!(guard_census, m)?)?;
    m.add_function(wrap_pyfunction!(reset_guard_census, m)?)?;
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
mod tile_coord_tests {
    use super::shift_axis_map;
    use symbi_geometry::AxisMap;

    // the law a decomposed grid must satisfy: a tile holds local indices, so its local cell `i` has
    // to land exactly where the undecomposed grid puts global cell `tile_lo + i`. if it does not, the
    // tiles describe different physical domains and the halo exchange glues together grids that do
    // not meet.
    #[test]
    fn tile_local_faces_reproduce_the_global_grid_on_a_log_axis() {
        let (n, tiles) = (64usize, 4usize);
        let (start, r_hi) = (1.5_f64, 400.0_f64);
        let global = AxisMap::Log {
            start,
            log_slope: (r_hi / start).log10() / n as f64,
        };
        let m = n / tiles;
        let mut checked = 0;
        for t in 0..tiles {
            let g = t * m;
            let local = shift_axis_map(global, g);
            // every face of the tile including the far one must coincide with the global face.
            for i in 0..=m {
                let want = global.face((g + i) as isize);
                let got = local.face(i as isize);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs(),
                    "tile {t} local face {i} (global {}) sits at {got}, global grid puts it at {want}",
                    g + i
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            tiles * (m + 1),
            "the sweep did not cover every tile face"
        );
        // the seam: tile t's far face is tile t+1's origin, so tiles abut with no gap or overlap.
        for t in 0..tiles - 1 {
            let far = shift_axis_map(global, t * m).face(m as isize);
            let next = shift_axis_map(global, (t + 1) * m).face(0);
            assert!(
                (far - next).abs() <= 1e-12 * next.abs(),
                "tiles {t}/{} do not abut: {far} vs {next}",
                t + 1
            );
        }
    }

    #[test]
    fn tile_local_faces_reproduce_the_global_grid_on_a_geometric_axis() {
        let (n, tiles) = (60usize, 4usize);
        let ratio = 0.97_f64;
        let extent = 12.0_f64;
        let width = extent * (ratio - 1.0) / (ratio.powf(n as f64) - 1.0);
        let global = AxisMap::Geometric {
            start: -2.0,
            width,
            ratio,
        };
        let cells_per_tile = n / tiles;

        for tile in 0..tiles {
            let offset = tile * cells_per_tile;
            let local = shift_axis_map(global, offset);
            for ii in 0..=cells_per_tile {
                let got = local.face(ii as isize);
                let want = global.face((offset + ii) as isize);
                assert!(
                    (got - want).abs() <= 1.0e-12 * want.abs().max(1.0),
                    "tile {tile} face {ii}: local {got}, global {want}"
                );
            }
        }
    }

    // the slope is a per-cell stretching and is global; a tile inherits it rather than re-deriving it
    // from its own extent. re-derivation would give each tile its own stretching -- the failure mode
    // this pins. the check is that re-deriving from the correct tile endpoints returns the same
    // number, so inheriting is consistent rather than a special case.
    #[test]
    fn a_tile_re_deriving_its_slope_would_get_the_global_one() {
        let (n, m) = (64usize, 16usize);
        let (start, r_hi) = (1.5_f64, 400.0_f64);
        let s_global = (r_hi / start).log10() / n as f64;
        let global = AxisMap::Log {
            start,
            log_slope: s_global,
        };
        for t in 0..4 {
            let local = shift_axis_map(global, t * m);
            match local {
                AxisMap::Log { log_slope, .. } => {
                    assert_eq!(log_slope, s_global, "tile {t} carries a different slope");
                    let (lo, hi) = (local.face(0), local.face(m as isize));
                    let s_local = (hi / lo).log10() / m as f64;
                    assert!(
                        (s_local - s_global).abs() <= 1e-12 * s_global,
                        "tile {t} endpoints imply slope {s_local}, global is {s_global}"
                    );
                }
                _ => panic!("tile {t} lost its log map"),
            }
        }
    }

    // on a uniform axis the map reduces to the additive origin the linear form assumes, so the two
    // agree there.

    // the clip must use the tile's map, not `origin + cells * dx`. on a log radial axis the cells
    // widen outward, so the linear form puts the tile's far corner well inside where its last cell
    // actually ends and silently drops refinement regions that really do overlap the tile. this
    // region sits in that gap: it is inside the true tile extent and outside the linear estimate.
    #[test]
    fn log_tile_clipping_keeps_a_region_the_linear_extent_would_drop() {
        use super::{RefinementRegion, clip_regions_to_tile};
        let (m, start) = (16usize, 1.5_f64);
        let slope = (400.0_f64 / start).log10() / 64.0;
        let map = AxisMap::Log {
            start,
            log_slope: slope,
        };
        let maps = [
            map,
            AxisMap::Uniform {
                start: 0.0,
                dx: 1.0,
            },
            AxisMap::Uniform {
                start: 0.0,
                dx: 1.0,
            },
        ];
        let origin = [map.face(0), 0.0, 0.0];
        let true_hi = map.face(m as isize);
        // the linear extent estimate: origin + m * (innermost width).
        let dx0 = map.face(1) - map.face(0);
        let linear_hi = origin[0] + m as f64 * dx0;
        assert!(
            linear_hi < true_hi,
            "setup is vacuous: the linear extent {linear_hi} must fall short of the true {true_hi}"
        );
        // a region living strictly between the two: genuinely on the tile, invisible to linear.
        let r = RefinementRegion {
            x_lo: [0.5 * (linear_hi + true_hi), 0.0, 0.0],
            x_hi: [true_hi, 1.0, 1.0],
        };
        let kept = clip_regions_to_tile::<3>(&[r], origin, [m, 1, 1], &maps);
        assert_eq!(
            kept.len(),
            1,
            "the map-aware clip dropped a region that overlaps the tile (true hi {true_hi}, \
             linear hi {linear_hi})"
        );
        assert!(
            kept[0].x_hi[0] <= true_hi + 1e-12 * true_hi,
            "clipped region extends past the tile's far face"
        );
    }

    #[test]
    fn a_uniform_axis_keeps_its_additive_origin() {
        let global = AxisMap::Uniform {
            start: -2.0,
            dx: 0.125,
        };
        for g in [0usize, 16, 48] {
            let local = shift_axis_map(global, g);
            for i in 0..=8 {
                let want = global.face((g + i) as isize);
                let got = local.face(i as isize);
                assert!(
                    (got - want).abs() <= 1e-15 * want.abs().max(1.0),
                    "uniform tile at {g}, local face {i}: {got} != {want}"
                );
            }
        }
    }
}
