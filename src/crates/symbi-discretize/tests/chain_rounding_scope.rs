// =============================================================================
// chain_rounding_scope.rs
//
// the kernels that round their chains term by term: every balanced band encode and decode, and
// nothing else, so contraction stays the compiler's default on every other kernel.
//
// run: cargo test -p symbi-discretize --test chain_rounding_scope
// =============================================================================
use symbi_discretize::{kernel_device_preamble, kernel_keeps_chain_rounding};

#[test]
fn every_band_chain_kernel_and_no_other_carries_the_directive() {
    for name in ["wb_band_encode_0_1d", "wb_band_encode_2_3d", "wb_band_encode_anchored_0_3d", "wb_band_encode_anchored_2_2d", "wb_band_decode_3d", "wb_band_decode_1d"] {
        assert!(kernel_keeps_chain_rounding(name), "{name}");
        assert_eq!(kernel_device_preamble(name), ["#pragma clang fp contract(off)"], "{name}");
    }
    for name in ["imhd_c2p_3d", "adiabatic_godunov_stage_3d", "band_energy_3d", "wb_cf_lerp_encode_3d", "body_source_3d", "refine_restrict_3d"] {
        assert!(!kernel_keeps_chain_rounding(name), "{name}");
        assert!(kernel_device_preamble(name).is_empty(), "{name}");
    }
}
