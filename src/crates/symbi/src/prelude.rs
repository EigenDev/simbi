// =============================================================================
// prelude.rs
//
// `use symbi::prelude::*;` — the one-import surface for building, seeding, running, and
// inspecting a simulation. it gathers the types a typical sim touches, which otherwise come
// from four crates (symbi_hydro / symbi_geometry / symbi_xpu / symbi) in a 7-15 line import
// block: the physics regimes, equations of state, geometries, execution / memory spaces, the
// `Sim` types + builder, the `evolve` entry points, the per-regime godunov / constrained-
// transport `KernelSet`s, and the primitive / conserved value types.
//
// purely ADDITIVE — every fully-qualified path still works; this just spares the boilerplate.
//
// usage:
//   use symbi::prelude::*;
//   type Sim = SimCpu<Newtonian, 1, Cartesian, IdealGas<f64>>;            // hydro, natural DOF
//   type Mhd = SimCpuGeneric<NewtonianMhd, 2, 3, Cartesian, IdealGas<f64>>; // 2.5D MHD (DOF != D)
// =============================================================================

// --- physics regimes ---
pub use symbi_hydro::newtonian::Newtonian;
pub use symbi_hydro::newtonian_mhd::NewtonianMhd;
pub use symbi_hydro::isothermal_mhd::IsothermalMhd;
pub use symbi_hydro::rmhd::Rmhd;
pub use symbi_hydro::srhd::Srhd;

// --- equations of state ---
pub use symbi_hydro::eos::{IdealGas, Isothermal};

// --- geometry: the coordinate systems + the metric trait ---
pub use symbi_geometry::{Cartesian, Cylindrical, Metric, Spherical};

// --- execution + memory spaces (the `S` / `Mem` backend params) ---
pub use symbi_xpu::{CpuSpace, ExecutionSpace, HostMemory, MemorySpace};
#[cfg(feature = "cuda")]
pub use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
// the FEATURE-SELECTED backend: GPU (CudaSpace / UnifiedMemory) under `--features cuda`, else CPU
// (CpuSpace / HostMemory) — re-exported so an example drops its per-file `#[cfg(feature="cuda")]`
// Space/Mem block in favor of `SimDefault*` (below).
pub use symbi_xpu::{DefaultMemory, DefaultSpace};

// --- the simulation state, fluent builder, and setup enums ---
pub use crate::sim::state::{
    Boundaries, BoundaryType, ConfigError, CylPlane, NeedsCells, NeedsGrid, Ready, SimBuilder,
    SimState, SimStateGeneric, Timestepping,
};

// --- the evolve loop + the KernelSet contract + classification enums (sim seam) ---
pub use crate::sim::evolve::{evolve, evolve_with_callback};
pub use crate::sim::substrate_seam::{KernelSet, Solver};

// --- the regime -> KernelSet map + the `sim.substrate()` front door (substrate layer) ---
pub use crate::regimes::regime_substrate::{RegimeSubstrate, SimSubstrate};

// --- per-regime substrate KernelSets (godunov gas stage + constrained transport) ---
pub use crate::regimes::substrate::IsoSubstrateKernelSet;
pub use crate::regimes::substrate_isothermal_mhd::IsothermalMhdSubstrateKernelSet;
pub use crate::regimes::substrate_newton::AdiabaticSubstrateKernelSet;
pub use crate::regimes::substrate_newtonian_mhd::NewtonianMhdSubstrateKernelSet;
pub use crate::regimes::substrate_rmhd::RmhdSubstrateKernelSet;
pub use crate::regimes::substrate_srhd::SrhdSubstrateKernelSet;

// --- primitive / conserved value types (for IC closures + field read-back) ---
pub use symbi_algebra::Tensor;
pub use symbi_hydro::mhd_state::{MhdCons, MhdPrim};
pub use symbi_hydro::state::{Cons, Prim};

// --- convenience type aliases: pin the CPU / host-memory / f64 backend so a sim is just
// `SimCpu<Regime, D, Metric, Eos>` (the 8-param `SimStateGeneric` collapses to 4). ---

/// a CPU / host-memory / f64 sim at the NATURAL vector dimension (DOF = D) — hydro, and 3D MHD.
pub type SimCpu<R, const D: usize, M, E> = SimState<R, D, M, E, CpuSpace, HostMemory, f64>;

/// a CPU / host-memory / f64 sim with an EXPLICIT vector DOF (the 1.5D / 2.5D MHD lift where the
/// 3-vector B / momentum rides a lower-dimensional grid, DOF != D).
pub type SimCpuGeneric<R, const D: usize, const DOF: usize, M, E> =
    SimStateGeneric<R, D, DOF, M, E, CpuSpace, HostMemory, f64>;

/// a sim on the FEATURE-SELECTED backend (GPU under `--features cuda`, else CPU), natural DOF = D
/// — the example default: ONE binary that runs on whichever backend it's built for, no per-file
/// `#[cfg(feature="cuda")]` Space/Mem block.
pub type SimDefault<R, const D: usize, M, E> =
    SimState<R, D, M, E, DefaultSpace, DefaultMemory, f64>;
/// feature-selected backend with an EXPLICIT vector DOF (1.5D / 2.5D MHD lift).
pub type SimDefaultGeneric<R, const D: usize, const DOF: usize, M, E> =
    SimStateGeneric<R, D, DOF, M, E, DefaultSpace, DefaultMemory, f64>;
