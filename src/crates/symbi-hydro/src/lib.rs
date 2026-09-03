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

pub mod admissibility;
pub mod admissible;
pub mod boundary_term;
pub mod c2p_result;
pub mod constraints;
pub mod dissipation;
pub mod energy;
pub mod eos;
pub mod hydrostatic;
pub mod isothermal;
pub mod isothermal_mhd;
pub mod mhd_state;
pub mod newtonian;
pub mod newtonian_mhd;
pub mod quantity;
pub mod recovery;
pub mod regime;
pub mod regime_spec;
pub mod rhd;
pub mod riemann;
pub mod rmhd;
pub mod source_term;
pub mod spatial_metric;
pub mod state;
pub mod state_law;
pub mod traced_recovery;
pub mod viscous;
pub use c2p_result::ErrorCode;
pub use dissipation::{ShockwaveLimiter, mach_scale, shear_weight};
pub use energy::{Adiabatic, EnergyModel, EnergySlot, IsoModel, Zero};
pub use eos::{Eos, IdealGas, Isothermal};
pub use isothermal::{IsoCons, IsoNewtonian, IsoPrim};
pub use isothermal_mhd::{IsothermalMhd, imhd_recover};
pub use mhd_state::{IsoMhdCons, IsoMhdPrim, MhdCons, MhdConsG, MhdPrim, MhdPrimG};
pub use newtonian::Newtonian;
pub use newtonian_mhd::NewtonianMhd;
pub use quantity::{Density, EnergyDensity, Pressure, SpecificInternalEnergy, VelocitySquared};
pub use admissibility::{
    Admissible, AdmissibilityLaw, WuTang, WuTangMargins, WuTangOutside, WuTangState,
    WuTangVerdict, WuTangWitness,
};
pub use recovery::{DiagnosticOnly, Recovered, Recovery, RecoveryFailure, RecoveryIssues};
pub use regime::Regime;
pub use regime_spec::{
    C2pKind, EosKind, FieldKind, FieldSpec, ISO_MHD_SPEC, ISO_NEWTONIAN_SPEC, NEWTONIAN_MHD_SPEC,
    NEWTONIAN_SPEC, RHD_SPEC, RMHD_SPEC, RegimeSpec,
};
pub use rhd::{Rhd, RhdGr};
pub use riemann::{
    hllc, hllc_newtonian, hllc_rhd, hllc_rmhd, hlld_isothermal, hlld_newtonian, hlld_rmhd, hlle,
};
pub use rmhd::{Rmhd, RmhdGr};
pub use source_term::{PointMassGravity, UniformAccel};
pub use state::{Cons, ConsG, Magnetic, NonMagnetic, Prim, PrimG};
pub use symbi_algebra::Tensor;
pub use symbi_carrier::Scalar;
pub use symbi_geometry;
pub use traced_recovery::{KernelC2pStatus, TracedRecovery};
