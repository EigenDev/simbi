// =============================================================================
// receipt_rounding_scope.rs
//
// the kernels that round term by term on the device: every conservation receipt kernel, per
// channel and batched, on both axisymmetric planes, and no other kernel, so contraction stays the
// compiler's default on the kernels that evolve the state.
//
// run: cargo test -p symbi-discretize --test receipt_rounding_scope
// =============================================================================
use symbi_discretize::{LedgerBatch, LedgerChannel, kernel_device_preamble, kernel_keeps_receipt_rounding};

#[test]
fn every_receipt_kernel_per_channel_and_batched_carries_the_directive() {
    let stems: Vec<String> = LedgerChannel::all().into_iter().map(LedgerChannel::stem).chain(LedgerBatch::all().into_iter().map(LedgerBatch::stem)).collect();
    assert!(stems.len() > LedgerChannel::all().len(), "the batched stems are missing; the check covers the per-channel kernels alone");
    for stem in stems {
        for sfx in ["_sph_swirl", "_cyl_rz"] {
            let name = format!("{stem}{sfx}_2d");
            assert!(kernel_keeps_receipt_rounding(&name), "{name}");
            assert_eq!(kernel_device_preamble(&name), ["#pragma clang fp contract(off)"], "{name}");
        }
    }
}

#[test]
fn no_kernel_that_evolves_the_state_carries_it() {
    for name in [
        "adiabatic_godunov_stage_sph_swirl_2d",
        "body_source_face_work_sph_swirl_2d",
        "body_feedback_sph_swirl_2d",
        "adiabatic_fofc_select_with_body_sph_swirl_2d",
        "outward_wall_1_high_sph_swirl_2d",
        "wb_band_encode_0_1d",
        "wb_band_decode_3d",
        "imhd_c2p_3d",
        "refine_restrict_3d",
    ] {
        assert!(!kernel_keeps_receipt_rounding(name), "{name}");
        assert!(kernel_device_preamble(name).is_empty(), "{name}");
    }
}
