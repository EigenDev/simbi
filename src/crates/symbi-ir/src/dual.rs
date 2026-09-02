// =============================================================================
// dual.rs
//
// the forward-mode derivative carrier lives in the foundation crate
// `symbi-carrier`; re-exported so compiler-side code keeps the
// `symbi_ir::dual::Dual` path.
//
// usage:
//  use symbi_ir::dual::Dual;           // == symbi_carrier::Dual
// =============================================================================

pub use symbi_carrier::dual::*;
