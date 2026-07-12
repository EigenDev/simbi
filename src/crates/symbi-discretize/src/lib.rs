// =============================================================================
// symbi-discretize
//
// the SRP-clean bridge between the carrier-generic physics (symbi-hydro over
// `S: Scalar`) and the discrete-IR layer (symbi-ir). instantiating that physics
// at `S = Gv` TRACES it into a stencil DAG `Graph` — Gv is the sole IR front end.
//
// the substrate cut:
//
//     symbi-hydro   "what the physics is"      (carrier-generic, S: Scalar)
//          |
//          |  run the physics at S = Gv (gv::*_gv builders)
//          v
//     symbi-ir      "what the kernel computes" (stencil DAG)
//          |
//          |  emit_kernel / emit_cuda
//          v
//     PTX / x86 / ...
//
// every production kernel — c2p, flux, wave-speed, godunov, ghost-fill, CT curl,
// immersed source — is built this way (`gv.rs` / `gv_immersed.rs`); the geometry
// is traced in-kernel from the cell index (`gv::cell_geometry_gv`).
// =============================================================================

pub mod coords;
pub mod kernel_slug;
pub mod gv;
pub mod gv_refinement;
pub mod gv_immersed;
pub mod gv_penalize;
pub mod ibm;
pub mod lattice;

pub use coords::{Coords, Spacing, Spacetime};
// facade: the carrier types live in symbi-ir alongside Op + Graph (consolidated
// 2026-05-30; symbi-core was folded in). re-export them so the builder return
// types (GvKernel) stay nameable by downstream callers (symbi-aot/build.rs).
pub use symbi_ir::{Gv, GvKernel};
pub use gv_penalize::{
    penalize_drain_gv, penalize_drain_iso_gv, penalize_porous_gv, penalize_torque_free_iso_gv,
};
pub use gv_immersed::{
    body_feedback_drain_gv, body_feedback_grav_gv, body_feedback_gv, body_feedback_iso_gv,
    body_source_built, body_source_gv, body_source_iso_gv,
};
pub use gv::{
    adiabatic_c2p_gv, adiabatic_flux_cyl_rz_gv, adiabatic_flux_gv, adiabatic_hllc_flux_gv,
    geometric_momentum_source_probe_gv,
    geometry_probe_gv, godunov_mass_gv, godunov_stage_gv, inertial_momentum_probe_gv,
    fofc_select_gv, fofc_select_with_body_gv, fofc_splice_gv, fofc_bflux_splice_gv, fofc_copy_gv, fofc_probe_gv, fofc_freeze_probe_gv, state_finite_probe_gv,
    point_mass_gravity_probe_gv, source_apply_gv, source_apply_from_built_gv, boundary_fill_from_built_gv,
    splice_user_source_gv, uniform_accel_probe_gv,
    iso_c2p_gv, iso_flux_gv, iso_ghost_fill_gv, iso_pre_gv, iso_wave_speed_map_gv,
    neumann_ghost_fill_gv, robin_ghost_fill_gv,
    imhd_bcell_from_bface_gv, imhd_c2p_gv, imhd_flux_gv, imhd_ghost_fill_gv, imhd_hlld_flux_gv, imhd_wave_speed_map_gv,
    nmhd_c2p_gv, nmhd_flux_gv, nmhd_hllc_flux_gv, nmhd_hlld_flux_gv, nmhd_wave_speed_map_gv, rmhd_average_efield_gv,
    rmhd_bcell_from_bface_gv, rmhd_bcell_godunov_euler_gv, rmhd_bcell_godunov_rk2_gv, rmhd_c2p_gv,
    rmhd_ct_curl_2d_dir_gv, rmhd_ct_curl_2d_sph_gv, rmhd_ct_curl_3d_dir_gv, rmhd_ct_curl_cyl_rz_gv, rmhd_ct_curl_cyl_rphi_gv, rmhd_edge_emf_gv, fofc_emf_splice_gv, rmhd_edge_emf_uct_gv, nmhd_edge_emf_uct_hllc_gv, nmhd_edge_emf_uct_hlld_gv, imhd_edge_emf_uct_hlld_gv, rmhd_edge_emf_uct_hlld_gv, uct_master_emf_proof_kernel, hlld_wave_sum_proof_kernel, rmhd_flux_gv, rmhd_ghost_fill_gv,
    rmhd_hllc_flux_gv, rmhd_hlld_flux_gv, rmhd_save_efield_gv, rmhd_wave_speed_map_gv,
    rmhd_wave_speeds_cell_gv, nmhd_wave_speeds_cell_gv, imhd_wave_speeds_cell_gv, scalar_ghost_fill_gv, snapshot_gv,
    rhd_c2p_gv, rhd_c2p_gr_gv, rhd_flux_gv, rhd_flux_gr_gv,
    rmhd_c2p_gr_gv, rmhd_flux_gr_gv, gr_light_cone_wave_speed_map_gv,
    rmhd_ct_curl_2d_sph_gr_gv, rmhd_edge_emf_gr_gv, rmhd_bcell_from_bface_gr_gv,
    rmhd_edge_emf_uct_gr_gv, rmhd_wave_speeds_cell_gr_gv, rmhd_edge_emf_uct_hlld_gr_gv,
    kerr_wave_speed_map_gv, rhd_hllc_flux_gv,
    rhd_wave_speed_map_gv, GeoSource,
};
pub use gv_refinement::{
    field_lerp_multi_gv, refine_acc_edge_gv, refine_acc_face_gv, refine_prolong_1t_gv,
    refine_prolong_face_gv, refine_prolong_gv, refine_prolong_multi_1t_gv, refine_prolong_multi_gv,
    refine_prolong_sweep_multi_gv,
    refine_restrict_face_gv, refine_restrict_gv, field_axpy_shift_gv, field_copy_gv, field_fill_gv,
    ProlongOrder,
};
pub use lattice::LatticeMap;

/// whether a kernel's buffers all share one allocated layout, so the cell index
/// can be computed once and shared across reads. true for the single-layout
/// cell-centered kernels — c2p (cons<->prim), wave-speed maps (prim -> scalar
/// scratch), and the pure-hydro adiabatic/rhd face flux (no staggered ct efield).
/// false for mhd `*face_flux*` (writes a staggered edge efield) and amr
/// prolong/restrict (read one grid, write another). classified by kernel name HERE
/// in the hydro layer so the IR (`KernelEmitInputs::coalesce_layout`) stays
/// domain-agnostic and merely carries the producer-set flag. the carrier oracle
/// catches any misclassification (a wrong index diverges from the f64 oracle).
/// PROTOTYPE: to be replaced by real per-field layout identity.
pub fn kernel_coalesces_layout(kernel_name: &str) -> bool {
    kernel_name.contains("c2p")
        || kernel_name.contains("wave_speed_map")
        || kernel_name.contains("adiabatic_face_flux")
        || kernel_name.contains("rhd_face_flux")
}
