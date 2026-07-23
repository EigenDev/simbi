// =============================================================================
// symbi-hydro
//
// compressible hydrodynamics for the symbi framework. provides equation of state,
// conservative/primitive state types, the physics regimes, and the riemann solvers
// — the carrier-generic (S: Scalar) single source the substrate traces at S=Gv. pure
// math, no runtime dependency on symbi. the substrate is the production path.
//
// usage:
//   use symbi_hydro::{IdealGas, Prim, Cons, Newtonian, hlle};
// =============================================================================

pub mod admissible;
pub mod c2p_result;
pub mod energy;
pub mod eos;
pub mod state;
pub mod spatial_metric;
pub mod mhd_state;
pub mod regime;
pub mod regime_spec;
pub mod source_spec;
pub mod source_term;
pub mod boundary_term;
pub mod expr_bridge;
pub mod motion_law;
pub mod simulation_laws;
pub mod source_evaluator;
pub mod gpu_source_kernel;
#[cfg(feature = "gpu")]
pub mod gpu_launcher;
pub mod newtonian;
pub mod newtonian_mhd;
pub mod isothermal_mhd;
pub mod rhd;
pub mod rmhd;
pub mod dissipation;
pub mod viscous;
pub mod riemann;
pub mod isothermal;
pub use symbi_ir::algebra::Scalar;
pub use symbi_algebra::Tensor;
pub use c2p_result::{ErrorCode, C2pResult};
pub use symbi_geometry;
pub use energy::{EnergyModel, EnergySlot, Adiabatic, IsoModel, Zero};
pub use eos::{Eos, IdealGas, Isothermal};
pub use state::{Prim, Cons, PrimG, ConsG, Magnetic, NonMagnetic};
pub use regime::Regime;
pub use regime_spec::{
    RegimeSpec, FieldSpec, FieldKind, EosKind, C2pKind,
    NEWTONIAN_SPEC, ISO_NEWTONIAN_SPEC, RHD_SPEC, RMHD_SPEC, NEWTONIAN_MHD_SPEC, ISO_MHD_SPEC,
};
pub use source_spec::{
    SourceSpec, BuiltSource, SourceKind,
    spherical_geometric_sources, cylindrical_geometric_sources,
    cartesian_geometric_sources, point_mass_gravity_sources,
    rigid_body_penalty_sources,
    user_defined_source, uniform_acceleration_sources,
    user_force_momentum_source, user_force_energy_source, user_cooling_source,
};
pub use source_term::{PointMassGravity, UniformAccel};
pub use simulation_laws::{
    point_mass, uniform_accel, CompositionError, FusedSourceFamily, Overlay, SimulationLaws,
};
pub use source_evaluator::SourceEvaluator;
// the python front door's wire format — re-exported so the source API (SourceConfig +
// expr_bridge::build_user_source) is one import surface.
pub use symbi_expr::SourceConfig;
pub use gpu_source_kernel::GpuSourceKernel;
#[cfg(feature = "gpu")]
pub use gpu_launcher::launch_source_kernel;
pub use newtonian::Newtonian;
pub use newtonian_mhd::NewtonianMhd;
pub use isothermal_mhd::{IsothermalMhd, imhd_recover};
pub use rhd::{Rhd, RhdGr};
pub use rmhd::{Rmhd, RmhdGr};
pub use isothermal::{IsoNewtonian, IsoPrim, IsoCons};
pub use mhd_state::{MhdPrim, MhdCons, MhdPrimG, MhdConsG, IsoMhdPrim, IsoMhdCons};
pub use riemann::{hllc, hllc_rhd, hllc_rmhd, hllc_newtonian, hlld_rmhd, hlld_newtonian, hlld_isothermal, hlle};
pub use dissipation::{adaptive_phi, ShockwaveLimiter};
