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
// is traced in-kernel from the cell index (`gv::cell_geometry_gv`). the legacy
// `Expr` DSL + `DiscretizeCtx` lowering is retired.
// =============================================================================

pub mod coords;
pub mod gv;
pub mod gv_refinement;
pub mod gv_immersed;
pub mod lattice;

pub use coords::{Coords, Spacing};
// facade: the carrier types live in symbi-ir alongside Op + Graph (consolidated
// 2026-05-30; symbi-core was folded in). re-export them so the builder return
// types (GvKernel) stay nameable by downstream callers (symbi-aot/build.rs).
pub use symbi_ir::{Gv, GvKernel};
pub use gv_immersed::{body_feedback_gv, body_source_gv};
pub use gv::{
    adiabatic_c2p_gv, adiabatic_flux_cyl_rz_gv, adiabatic_flux_gv, adiabatic_hllc_flux_gv,
    geometric_momentum_source_probe_gv,
    geometry_probe_gv, godunov_mass_gv, godunov_stage_gv, inertial_momentum_probe_gv,
    point_mass_gravity_probe_gv, source_apply_gv, source_apply_from_built_gv, boundary_fill_from_built_gv,
    splice_user_source_gv, uniform_accel_probe_gv,
    iso_c2p_gv, iso_flux_gv, iso_ghost_fill_gv, iso_wave_speed_map_gv,
    imhd_bcell_from_bface_gv, imhd_c2p_gv, imhd_flux_gv, imhd_ghost_fill_gv, imhd_hlld_flux_gv, imhd_wave_speed_map_gv,
    nmhd_c2p_gv, nmhd_flux_gv, nmhd_hllc_flux_gv, nmhd_hlld_flux_gv, nmhd_wave_speed_map_gv, rmhd_average_efield_gv,
    rmhd_bcell_from_bface_gv, rmhd_bcell_godunov_euler_gv, rmhd_bcell_godunov_rk2_gv, rmhd_c2p_gv,
    rmhd_ct_curl_2d_dir_gv, rmhd_ct_curl_3d_dir_gv, rmhd_ct_curl_cyl_rz_gv, rmhd_ct_curl_cyl_rphi_gv, rmhd_edge_emf_gv, rmhd_flux_gv, rmhd_ghost_fill_gv,
    rmhd_hllc_flux_gv, rmhd_hlld_flux_gv, rmhd_save_efield_gv, rmhd_wave_speed_map_gv,
    rmhd_wave_speeds_cell_gv, scalar_ghost_fill_gv, snapshot_gv,
    srhd_c2p_gv, srhd_flux_gv, srhd_hllc_flux_gv,
    srhd_wave_speed_map_gv, GeoSource,
};
pub use gv_refinement::{
    refine_acc_edge_gv, refine_acc_face_gv, refine_prolong_face_gv, refine_prolong_gv, refine_prolong_multi_gv,
    refine_restrict_face_gv, refine_restrict_gv, field_axpy_shift_gv, field_copy_gv, field_fill_gv,
    ProlongOrder,
};
pub use lattice::LatticeMap;
