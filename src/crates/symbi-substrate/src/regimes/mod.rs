// =============================================================================
// regimes/mod.rs
//
// the live substrate KernelSets: each `substrate_*` module is a D-generic
// KernelSet whose every method dispatches to a build-time AOT substrate
// kernel (symbi-discretize -> symbi-ir/tensor -> symbi-aot -> render_from_ir)
// via the structured binding ABI + the generated `kernel_by_name` registry.
// `substrate_kernels` is the shared dispatch (geom_suffix, cfl, godunov,
// flux, body source/feedback); `substrate_gpu` is the NVRTC runtime path.
// =============================================================================

pub mod fofc;
pub mod mhd_substrate;
pub mod regime_substrate;
pub mod source_config;
pub mod substrate;
pub mod substrate_gpu;
pub mod substrate_isothermal_mhd;
pub mod substrate_kernels;
pub mod substrate_mhd;
pub mod substrate_newton;
pub mod substrate_newtonian_mhd;
pub mod substrate_rhd;
pub mod substrate_rmhd;
