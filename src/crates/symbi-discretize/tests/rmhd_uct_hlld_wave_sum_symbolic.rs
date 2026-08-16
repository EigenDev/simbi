// =============================================================================
// rmhd_uct_hlld_wave_sum_symbolic.rs
//
// the symbolic proof of the HLLD wave-sum EMF dissipation-sign pairing — the invariant left open by
// the upwind proof (nmhd_uct_emf_upwind_symbolic), because the HLLD edge EMF
// kernels (`rmhd_edge_emf_uct_hlld_gv` + `_gr_gv`) build a wave-sum dissipative flux Phi (M&DZ Eq. 39),
// an assembly outside the `uct_master_emf` coefficient form.
//
// the wave-sum dissipation for one transverse B across the 4-wave fan is
//   Phi = 1/2 [ |lam_L|(bstar_L - bt_L) + |alf_L|(bc - bstar_L)
//             + |alf_R|(bstar_R - bc)  + |lam_R|(bt_R - bstar_R) ]
// `hlld_wave_sum_terms` (the un-halved form both HLLD kernels compose through) is linear in the five
// staggered/star transverse fields, so `LinForm` reads each field's coefficient in the four opaque
// wave speeds. the load-bearing invariant — correct (diffusive) dissipation — is
// a coefficient pairing check:
//   - the left endpoint bt_L is diffused by the left fast wave |lam_L| alone (coeff -1 on |lam_L|, coeff 0 on |lam_R|)
//   - the right endpoint bt_R by the right fast wave |lam_R| alone          (coeff +1 on |lam_R|, coeff 0 on |lam_L|)
//   - the star / central states telescope (each between its two bracketing waves)
// mispairing an endpoint with the opposite fast wave flips the dissipation sign — the anti-diffusive
// bug that hides subsonically (|lam_L| == |lam_R|) and behind every div(B) test (the curl
// preserves div(B) for any emf). this proves the pairing structurally, straight from the graph.
// =============================================================================

use symbi_discretize::hlld_wave_sum_proof_kernel;
use symbi_ir::proof::{LinForm, Poly};

// the five reconstructed transverse-field reads are the "fields"; the four absolute fan speeds are
// opaque scalars. bt_l/bt_r are the staggered endpoints; bstar_l/bstar_r the fast-star states; bc
// the central (contact) state.
const FIELDS: &[&str] = &["bt_l", "bstar_l", "bc", "bstar_r", "bt_r"];
const SCALARS: &[&str] = &["alam_l", "aalf_l", "aalf_r", "alam_r"];

// every field read resolves to the offset-free key: the symbolic param leaves stand in for LoadAt
// nodes.
fn coeff<'a>(lf: &'a LinForm, key: &str) -> &'a Poly {
    lf.terms.get(&(key.to_string(), vec![])).unwrap_or_else(|| {
        panic!("wave-sum has no coefficient for field `{key}` — extraction changed shape")
    })
}

#[test]
fn hlld_wave_sum_dissipation_pairing_symbolic() {
    let (kernel, writes) = hlld_wave_sum_proof_kernel(false);
    let lf = LinForm::extract(&kernel.graph, writes[0].2, FIELDS, SCALARS);

    // the fast-wave endpoint pairing (the dissipation sign): each staggered endpoint is diffused by
    // the fast wave on its own side alone.
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

// negative control: mispair the two fast-wave endpoints (|lam_L| <-> |lam_R|), which makes the
// dissipation anti-diffusive. the checker rejects it (bt_l lands on the wrong wave), so the
// pairing proof reports a real failure when the pairing breaks.
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
