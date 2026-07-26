// =============================================================================
// rmhd_uct_hlld_wave_sum_symbolic.rs
//
// the SYMBOLIC proof of the HLLD wave-sum EMF DISSIPATION-SIGN PAIRING (M8) — the invariant the
// upwind proof (nmhd_uct_emf_upwind_symbolic) explicitly does NOT cover, because the HLLD edge EMF
// kernels (`rmhd_edge_emf_uct_hlld_gv` + `_gr_gv`) build a wave-sum dissipative flux Phi (M&DZ Eq. 39),
// an assembly outside the `uct_master_emf` coefficient form.
//
// the wave-sum dissipation for one transverse B across the 4-wave fan is
//   Phi = 1/2 [ |lam_L|(bstar_L - bt_L) + |alf_L|(bc - bstar_L)
//             + |alf_R|(bstar_R - bc)  + |lam_R|(bt_R - bstar_R) ]
// `hlld_wave_sum_terms` (the un-halved form both HLLD kernels compose through) is LINEAR in the five
// staggered/star transverse fields, so `LinForm` reads each field's coefficient in the four opaque
// wave speeds. the load-bearing invariant — correct (diffusive) dissipation — is
// a coefficient PAIRING check:
//   - the LEFT endpoint bt_L is diffused by the LEFT fast wave |lam_L| ONLY  (coeff -1 on |lam_L|, coeff 0 on |lam_R|)
//   - the RIGHT endpoint bt_R by the RIGHT fast wave |lam_R| ONLY            (coeff +1 on |lam_R|, coeff 0 on |lam_L|)
//   - the star / central states telescope (each between its two bracketing waves)
// mispairing an endpoint with the opposite fast wave flips the dissipation sign — the anti-diffusive
// bug that is invisible subsonically (|lam_L| == |lam_R|) and to every div(B) test (the curl
// preserves div(B) for ANY emf). this proves it structurally, with no evolve loop.
// =============================================================================

use symbi_discretize::hlld_wave_sum_proof_kernel;
use symbi_ir::proof::{LinForm, Poly};

// the five reconstructed transverse-field reads are the "fields"; the four absolute fan speeds are
// opaque scalars. bt_l/bt_r are the staggered endpoints; bstar_l/bstar_r the fast-star states; bc
// the central (contact) state.
const FIELDS: &[&str] = &["bt_l", "bstar_l", "bc", "bstar_r", "bt_r"];
const SCALARS: &[&str] = &["alam_l", "aalf_l", "aalf_r", "alam_r"];

// the symbolic param leaves carry no LoadAt, so every field read resolves to the offset-free key.
fn coeff<'a>(lf: &'a LinForm, key: &str) -> &'a Poly {
    lf.terms.get(&(key.to_string(), vec![])).unwrap_or_else(|| {
        panic!("wave-sum has no coefficient for field `{key}` — extraction changed shape")
    })
}

#[test]
fn hlld_wave_sum_dissipation_pairing_symbolic() {
    let (kernel, writes) = hlld_wave_sum_proof_kernel(false);
    let lf = LinForm::extract(&kernel.graph, writes[0].2, FIELDS, SCALARS);

    // the FAST-WAVE ENDPOINT PAIRING (the dissipation sign): each staggered endpoint is diffused by
    // the fast wave on ITS OWN side, and by no other.
    let btl = coeff(&lf, "bt_l");
    assert_eq!(
        btl.coefficient_of(&["alam_l"]),
        -1,
        "bt_l must be diffused by the LEFT fast wave"
    );
    assert_eq!(
        btl.coefficient_of(&["alam_r"]),
        0,
        "bt_l must NOT be diffused by the RIGHT fast wave"
    );
    let btr = coeff(&lf, "bt_r");
    assert_eq!(
        btr.coefficient_of(&["alam_r"]),
        1,
        "bt_r must be diffused by the RIGHT fast wave"
    );
    assert_eq!(
        btr.coefficient_of(&["alam_l"]),
        0,
        "bt_r must NOT be diffused by the LEFT fast wave"
    );

    // the intermediate states telescope, each bracketed by its two adjacent wave speeds.
    let bsl = coeff(&lf, "bstar_l");
    assert_eq!(
        bsl.coefficient_of(&["alam_l"]),
        1,
        "bstar_l telescopes with the LEFT fast wave"
    );
    assert_eq!(
        bsl.coefficient_of(&["aalf_l"]),
        -1,
        "bstar_l telescopes with the LEFT Alfven wave"
    );
    let bcc = coeff(&lf, "bc");
    assert_eq!(
        bcc.coefficient_of(&["aalf_l"]),
        1,
        "bc telescopes with the LEFT Alfven wave"
    );
    assert_eq!(
        bcc.coefficient_of(&["aalf_r"]),
        -1,
        "bc telescopes with the RIGHT Alfven wave"
    );
    let bsr = coeff(&lf, "bstar_r");
    assert_eq!(
        bsr.coefficient_of(&["aalf_r"]),
        1,
        "bstar_r telescopes with the RIGHT Alfven wave"
    );
    assert_eq!(
        bsr.coefficient_of(&["alam_r"]),
        -1,
        "bstar_r telescopes with the RIGHT fast wave"
    );
}

// NEGATIVE control: mispair the two fast-wave endpoints (|lam_L| <-> |lam_R|) — the anti-diffusive
// bug. the checker must reject it (bt_l now lands on the wrong wave), so the pairing proof is not
// vacuously green.
#[test]
fn hlld_wave_sum_symbolic_detects_swapped_fast_waves() {
    let (kernel, writes) = hlld_wave_sum_proof_kernel(true);
    let lf = LinForm::extract(&kernel.graph, writes[0].2, FIELDS, SCALARS);
    let btl = coeff(&lf, "bt_l");
    assert_eq!(
        btl.coefficient_of(&["alam_l"]),
        0,
        "bug: bt_l no longer on the LEFT fast wave"
    );
    assert_eq!(
        btl.coefficient_of(&["alam_r"]),
        -1,
        "bug: bt_l wrongly diffused by the RIGHT fast wave"
    );
}
