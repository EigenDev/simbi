// =============================================================================
// stub_kernels.cu
//
// no-op placeholder for the hand-written CUDA kernels in
// crates/symbi/kernels_deprecated/. compiled by build.rs into PTX, then
// embedded as the EMPTY_PTX const in src/kernels.rs. every entry-point name
// the GpuKernels::load() loader looks up is defined here as an empty kernel
// so cuModuleLoadData and cuModuleGetFunction succeed.
//
// the traced dispatch in src/sim/evolve.rs routes around these placeholders.
// an unmigrated deprecated path that launches one of these stubs computes
// nothing, and the simulation produces wrong physics.
// =============================================================================

// hydro/regime kernels
extern "C" __global__ void c2p() {}
extern "C" __global__ void flux_hllc() {}
extern "C" __global__ void max_wave_speed() {}
extern "C" __global__ void godunov_euler() {}
extern "C" __global__ void godunov_rk2() {}
extern "C" __global__ void save_state() {}
extern "C" __global__ void ghost_periodic() {}

// CT (constrained transport for MHD)
extern "C" __global__ void ct_edge_efield() {}
extern "C" __global__ void ct_curl() {}
extern "C" __global__ void ct_bface_to_bcell() {}
extern "C" __global__ void ct_efield_save() {}
extern "C" __global__ void ct_efield_average() {}
extern "C" __global__ void ct_efield_save_all() {}
extern "C" __global__ void ct_efield_avg_all() {}
extern "C" __global__ void ct_curl_2d() {}

// MHD ghost fill
extern "C" __global__ void ghost_periodic_mhd() {}

// AMR ops
extern "C" __global__ void amr_restrict() {}
extern "C" __global__ void amr_prolong_pcm() {}
extern "C" __global__ void amr_prolong_plm() {}
extern "C" __global__ void amr_prolong_ppm() {}
extern "C" __global__ void amr_save_old() {}
extern "C" __global__ void amr_flux_reg_zero() {}
extern "C" __global__ void amr_flux_reg_accum() {}
extern "C" __global__ void amr_flux_reg_apply() {}
extern "C" __global__ void amr_emf_reg_accum() {}
extern "C" __global__ void amr_restrict_bface() {}
extern "C" __global__ void amr_bface_to_bcell_nrg() {}
extern "C" __global__ void amr_emf_apply_2d() {}
extern "C" __global__ void amr_emf_apply_3d() {}

// body source / sink reduce
extern "C" __global__ void body_source_kernel() {}
extern "C" __global__ void sink_reduce() {}
