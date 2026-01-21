// =============================================================================
// lib.rs
//
// base: pure numerical methods for computational fluid dynamics.
// provides mathematical primitives and utilities with zero runtime overhead.
//
// design:
//   - no_std compatible (embedded/gpu kernels)
//   - compile-time evaluation where possible
//   - zero-cost abstractions via const generics
//   - no device awareness (pure math)
//
// usage:
//   use base::...;
// =============================================================================

#![cfg_attr(not(feature = "std"), no_std)]
#![allow(dead_code)]

// placeholder for future numerical primitives:
// - constants (pi, e, physical constants)
// - unit conversions
// - mathematical functions (if not using libm)
// - array utilities

#[cfg(test)]
mod tests {
    #[test]
    fn base_exists() {
        assert!(true);
    }
}
