// =============================================================================
// riemann.rs (module root)
//
// approximate riemann solvers for compressible hydrodynamics — ONE solver per
// file (mirrors the C++ `solvers/` layout). all solvers use nhat (unit normal
// vector) for direction, so ONE implementation handles every dimension and
// direction, and all accept a `vface` ALE grid velocity (pass 0 for a static
// mesh).
//
//   hlle  (hlle.rs)  regime-generic 2-wave solver, any Regime. GPU-traceable.
//   hllc  (hllc.rs)  3-wave (contact-resolving): newtonian / srhd / rmhd, one
//                    function per regime; newtonian takes a `ShockwaveLimiter`
//                    (Standard / Fleischmann LM / Quirk-fallback).
//   hlld  (hlld.rs)  5-wave RMHD (fast/alfven/contact), host-only secant iter.
//
// usage:
//   let nhat = Tensor::unit(0);
//   let flux = hlle(&regime, &eos, &prim_l, &prim_r, &nhat, 0.0);
// =============================================================================

mod hlle;
mod hllc;
mod hlld;

pub use hlle::{hlle, hlle_with_speeds};
pub use hllc::{hllc, hllc_rmhd, hllc_srhd, hllc_newtonian};
pub use hlld::{hlld_rmhd, hlld_rmhd_states, hlld_newtonian, hlld_newtonian_coeffs, hlld_isothermal, hlld_isothermal_coeffs, HlldStates};

// shared solver tolerances (visible to the submodules via `super::`).
/// guard against division by zero in intermediate expressions.
const DIVZERO_GUARD: f64 = 1e-30;
/// threshold below which a magnetic field component is treated as zero.
const NULL_FIELD_THRESHOLD: f64 = 1e-14;
