// =============================================================================
// config.rs
//
// simulation configuration extracted from python dicts. pure rust, no pyo3.
// enums mirror the type-level dispatch (RegimeType, GeometryType, EosType)
// so that symbi-py can build a SimConfig from a dict, then dispatch to
// monomorphized SimState construction.
//
// usage:
//   let cfg = SimConfig { regime: RegimeType::Newtonian, ... };
//   // dispatch on (cfg.regime, cfg.dims, cfg.geometry, cfg.eos)
// =============================================================================

use crate::state::{BoundaryType, Timestepping, Reconstruction, Solver, CtMethod};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegimeType {
    Newtonian,
    Rhd,
    Rmhd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeometryType {
    Cartesian,
    Spherical,
    Cylindrical,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EosType {
    IdealGas { gamma: f64 },
    Isothermal { cs: f64 },
}

/// simulation configuration. built from a python dict by symbi-py,
/// consumed by the dispatch macro to construct a monomorphized SimState.
pub struct SimConfig {
    pub regime: RegimeType,
    pub geometry: GeometryType,
    pub eos: EosType,
    pub dims: usize,
    /// cell counts per axis, padded with 1 for unused dims.
    pub n_cells: [usize; 3],
    /// physical lower bounds, padded with 0.0.
    pub x_lo: [f64; 3],
    /// cell widths per axis, padded with 1.0.
    pub dx: [f64; 3],
    pub boundaries: [BoundaryType; 6],
    pub cfl: f64,
    pub timestepping: Timestepping,
    pub solver: Solver,
    pub reconstruction: Reconstruction,
    pub ct_method: CtMethod,
    pub t_final: f64,
    pub checkpoint_interval: f64,
    pub data_dir: String,
    pub prefix: String,
    pub title: String,
}
