// =============================================================================
// symbi-source-compile
//
// how source descriptions become executable programs. this crate sits between
// the physics declarations (`symbi-hydro`: regimes, source laws, conservation
// lifts, all over `S: Scalar`) and the compiler (`symbi-ir`): it interprets
// user expressions in the carrier algebra, traces and splices `SourceProgram`s,
// composes overlays, and evaluates lowered programs on the host and the GPU.
//
// organization:
//   source_spec       — SourceSpec registry + the traced source builders
//   source_effects    — typed source signature, observed effects + the additive fold law
//   expr_bridge       — symbi-expr DAG interpreted at the carrier
//   simulation_laws   — regime + overlay composition and validation
//   state_law_gv      — StateLaw's traced conserved-state conversion
//   source_evaluator  — per-cell host evaluation of lowered programs
//   motion_law        — mesh-motion a(t)/adot(t) compiled once, evaluated per stage
//   gpu_source_kernel — standalone CUDA source emission per overlay field
//   gpu_launcher      — NVRTC JIT launch wiring (feature `gpu`)
//
// usage:
//  let sim = SimulationLaws::new(&NEWTONIAN_SPEC).with_gravity(point_mass_gravity_sources(3, true));
//  let program = sim.build_total_source("mom", 3);
// =============================================================================

pub mod expr_bridge;
#[cfg(feature = "gpu")]
pub mod gpu_launcher;
pub mod gpu_source_kernel;
pub mod motion_law;
pub mod simulation_laws;
pub mod source_effects;
pub mod source_evaluator;
pub mod source_spec;
pub mod state_law_gv;

pub use expr_bridge::{BoundaryPrescription, CensusProgram};
#[cfg(feature = "gpu")]
pub use gpu_launcher::launch_source_kernel;
pub use gpu_source_kernel::GpuSourceKernel;
pub use simulation_laws::{
    CompositionError, FusedSourceFamily, Overlay, SimulationLaws, point_mass, uniform_accel,
};
pub use source_effects::{
    AdmittedSources, SourceContributionEffects, SourceParameter, SourceSignature, SourceTarget,
    TypedParameterSet, TypedReadSet, admit_user_contribution,
};
pub use source_evaluator::SourceEvaluator;
pub use source_spec::{
    SourceKind, SourceProgram, SourceSpec, cartesian_geometric_sources,
    cylindrical_geometric_sources, point_mass_gravity_sources, rigid_body_penalty_sources,
    spherical_geometric_sources, uniform_acceleration_sources, user_cooling_source,
    user_defined_source, user_force_energy_source, user_force_momentum_source,
};
pub use state_law_gv::StateLawGv;

// the python front door's wire format — re-exported so the source API (SourceConfig +
// expr_bridge::build_user_source) is one import surface.
pub use symbi_expr::{CensusAxisConfig, CensusConfig, EquilibriumConfig, SourceConfig};
