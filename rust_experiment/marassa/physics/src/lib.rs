// =============================================================================
// lib.rs
//
// physics: riemann solvers, equations of state, and conservation laws.
// provides numerical methods for solving hyperbolic pdes in fluid dynamics.
//
// design:
//   - device-agnostic (works with any xpu backend)
//   - compile-time physics selection via features
//   - zero-cost abstractions for different regimes
//
// modules:
//   - hydro: newtonian hydrodynamics (euler equations)
//   - mhd: magnetohydrodynamics (future)
//
// usage:
//   use physics::hydro::{hlle_flux, ideal_gas_eos};
// =============================================================================

#![allow(dead_code)]

pub mod boundary;
pub mod config;
pub mod hydro;

#[cfg(feature = "mhd")]
pub mod mhd;

// re-export commonly used types
pub use boundary::{BoundaryCondition, OutflowBC, PeriodicBC, ReflectingBC, UserDefinedBC};
pub use config::{
    Configuration, Dim1, Dim2, Dim3, Dimensionality, Euler1D, Euler1DFirstOrder, Euler2D, Euler3D,
    FieldLayout, HlleSolver, Newtonian, PhysicsConfig, PlmReconstruction, Regime, RiemannSolver,
};
pub use hydro::{
    hlle_flux, hlle_flux_moving, ideal_gas_pressure, ideal_gas_sound_speed, Conserved1D, Flux1D,
    Primitive1D,
};

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        assert!(true);
    }
}
