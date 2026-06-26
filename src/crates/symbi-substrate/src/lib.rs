// =============================================================================
// symbi-substrate/src/lib.rs
//
// the substrate crate (docs/design/41 step 3): the live per-regime substrate
// `KernelSet`s (iso / adiabatic / srhd / rmhd / mhd), the regime -> KernelSet map
// (`RegimeSubstrate` + the `SimSubstrate` front door), the shared substrate-kernel
// dispatch (binding / params / dispatch / runtime_source / boundary), and the
// regime-agnostic kernel support (cfl_from_lambda, GhostFillDriver, ...).
//
// sits ABOVE `symbi-sim` (it implements `KernelSet` over `FieldStore` and names the
// classification enums) and `symbi-exec` (it dispatches through the launch engine);
// BELOW the `symbi` top crate (which drives it via `evolve` + AMR). it names NO
// integrator and NO refinement — those depend DOWN on it.
//
// usage:
//  use symbi_substrate::regimes::substrate::IsoSubstrateKernelSet;
//  use symbi_substrate::regimes::regime_substrate::SimSubstrate;
// =============================================================================

// the live substrate KernelSets (iso / adiabatic / srhd / rmhd) + the shared
// dispatch + the NVRTC GPU runtime path. see regimes/mod.rs.
pub mod regimes;

// shared kernel-set support (cfl_from_lambda, GhostFillDriver, ...) — the
// hand-written per-regime kernel sets were retired (docs/design/18).
pub mod kernels;
