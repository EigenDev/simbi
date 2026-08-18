// =============================================================================
// riemann.rs (module root)
//
// approximate riemann solvers for compressible hydrodynamics — one solver per
// file. all solvers use nhat (unit normal
// vector) for direction, so one implementation handles every dimension and
// direction, and all accept a `vface` ALE grid velocity (pass 0 for a static
// mesh).
//
//   hlle  (hlle.rs)  regime-generic 2-wave solver, any Regime. GPU-traceable.
//   hllc  (hllc.rs)  3-wave (contact-resolving): newtonian / rhd / rmhd, one
//                    function per regime; newtonian takes a `ShockwaveLimiter`
//                    (Standard / Fleischmann LM).
//   hlld  (hlld.rs)  5-wave RMHD (fast/alfven/contact), host-only secant iter.
//
// usage:
//   let nhat = Tensor::unit(0);
//   let flux = hlle(&regime, &eos, &prim_l, &prim_r, &nhat, 0.0);
// =============================================================================

mod hllc;
mod hlld;
mod hlle;

pub use hllc::{HllcPlusSensors, hllc, hllc_newtonian, hllc_rhd, hllc_rmhd};
pub use hlld::{
    HlldStates, hlld_isothermal, hlld_isothermal_coeffs, hlld_newtonian, hlld_newtonian_coeffs,
    hlld_rmhd, hlld_rmhd_gr_ortho, hlld_rmhd_states, hlld_rmhd_states_gr_ortho,
};
pub use hlle::{hlle, hlle_with_speeds};

// shared solver tolerances (visible to the submodules via `super::`).
