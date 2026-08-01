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
pub mod boundary_term;
pub mod c2p_result;
pub mod constraints;
pub mod dissipation;
pub mod energy;
pub mod eos;
pub mod expr_bridge;
#[cfg(feature = "gpu")]
pub mod gpu_launcher;
pub mod gpu_source_kernel;
pub mod isothermal;
pub mod isothermal_mhd;
pub mod mhd_state;
pub mod motion_law;
pub mod newtonian;
pub mod newtonian_mhd;
pub mod regime;
pub mod regime_spec;
pub mod rhd;
pub mod riemann;
pub mod rmhd;
pub mod simulation_laws;
pub mod source_evaluator;
pub mod source_spec;
pub mod source_term;
pub mod spatial_metric;
pub mod state;
pub mod viscous;
pub use c2p_result::{C2pResult, ErrorCode};
pub use energy::{Adiabatic, EnergyModel, EnergySlot, IsoModel, Zero};
pub use eos::{Eos, IdealGas, Isothermal};
pub use regime::Regime;
pub use regime_spec::{
    C2pKind, EosKind, FieldKind, FieldSpec, ISO_MHD_SPEC, ISO_NEWTONIAN_SPEC, NEWTONIAN_MHD_SPEC,
    NEWTONIAN_SPEC, RHD_SPEC, RMHD_SPEC, RegimeSpec,
};
pub use simulation_laws::{
    CompositionError, FusedSourceFamily, Overlay, SimulationLaws, point_mass, uniform_accel,
};
pub use source_evaluator::SourceEvaluator;
pub use source_spec::{
    BuiltSource, SourceKind, SourceSpec, cartesian_geometric_sources,
    cylindrical_geometric_sources, point_mass_gravity_sources, rigid_body_penalty_sources,
    spherical_geometric_sources, uniform_acceleration_sources, user_cooling_source,
    user_defined_source, user_force_energy_source, user_force_momentum_source,
};
pub use source_term::{PointMassGravity, UniformAccel};
pub use state::{Cons, ConsG, Magnetic, NonMagnetic, Prim, PrimG};
pub use symbi_algebra::Tensor;
pub use symbi_geometry;
pub use symbi_ir::algebra::Scalar;
// the python front door's wire format — re-exported so the source API (SourceConfig +
// expr_bridge::build_user_source) is one import surface.
pub use dissipation::{ShockwaveLimiter, adaptive_phi};
#[cfg(feature = "gpu")]
pub use gpu_launcher::launch_source_kernel;
pub use gpu_source_kernel::GpuSourceKernel;
pub use isothermal::{IsoCons, IsoNewtonian, IsoPrim};
pub use isothermal_mhd::{IsothermalMhd, imhd_recover};
pub use mhd_state::{IsoMhdCons, IsoMhdPrim, MhdCons, MhdConsG, MhdPrim, MhdPrimG};
pub use newtonian::Newtonian;
pub use newtonian_mhd::NewtonianMhd;
pub use rhd::{Rhd, RhdGr};
pub use riemann::{
    hllc, hllc_newtonian, hllc_rhd, hllc_rmhd, hlld_isothermal, hlld_newtonian, hlld_rmhd, hlle,
};
pub use rmhd::{Rmhd, RmhdGr};
pub use symbi_expr::{CensusAxisConfig, CensusConfig, EquilibriumConfig, SourceConfig};
