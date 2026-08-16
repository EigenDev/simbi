// =============================================================================
// symbi-substrate/src/lib.rs
//
// the substrate crate: the live per-regime substrate
// `KernelSet`s (iso / adiabatic / rhd / rmhd / mhd), the regime -> KernelSet map
// (`RegimeSubstrate` + the `SimSubstrate` front door), the shared substrate-kernel
// dispatch (binding / params / dispatch / runtime_source / boundary), and the
// regime-agnostic kernel support (cfl_from_lambda, GhostFillDriver, ...).
//
// sits above `symbi-sim` (it implements `KernelSet` over `FieldStore` and names the
// classification enums) and `symbi-exec` (it dispatches through the launch engine);
// below the `symbi` top crate (which drives it via `evolve` + AMR). it names no
// integrator and no refinement — those depend down on it.
//
// usage:
//  use symbi_substrate::regimes::substrate::IsoSubstrateKernelSet;
//  use symbi_substrate::regimes::regime_substrate::SimSubstrate;
// =============================================================================

// the live substrate KernelSets (iso / adiabatic / rhd / rmhd) + the shared
// dispatch + the NVRTC GPU runtime path. see regimes/mod.rs.
pub mod census_sample;
pub mod regimes;

// shared kernel-set support (cfl_from_lambda, GhostFillDriver, ...), regime-agnostic.
pub mod kernels;
