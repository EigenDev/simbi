// =============================================================================
// chain_rounding_scope.rs
//
// the kernels that round term by term: every balanced band encode and decode, and every
// conservation receipt kernel, per channel and batched, and nothing else, so contraction stays
// the compiler's default on every other kernel.
//
// run: cargo test -p symbi-discretize --test chain_rounding_scope
// =============================================================================
use symbi_discretize::{LedgerBatch, LedgerChannel, kernel_device_preamble, kernel_keeps_chain_rounding, kernel_keeps_receipt_rounding};

#[test]
fn every_band_chain_kernel_and_no_other_carries_the_directive() {
    for name in ["wb_band_encode_0_1d", "wb_band_encode_2_3d", "wb_band_encode_anchored_0_3d", "wb_band_encode_anchored_2_2d", "wb_band_decode_3d", "wb_band_decode_1d"] {
        assert!(kernel_keeps_chain_rounding(name), "{name}");
        assert_eq!(kernel_device_preamble(name), ["#pragma clang fp contract(off)"], "{name}");
    }
    for name in ["imhd_c2p_3d", "adiabatic_godunov_stage_3d", "band_energy_3d", "wb_cf_lerp_encode_3d", "body_source_3d", "refine_restrict_3d", "body_feedback_sph_swirl_2d"] {
        assert!(!kernel_keeps_chain_rounding(name), "{name}");
        assert!(!kernel_keeps_receipt_rounding(name), "{name}");
        assert!(kernel_device_preamble(name).is_empty(), "{name}");
    }
}

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
