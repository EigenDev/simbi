// =============================================================================
// nmhd_uct_emf_upwind_symbolic.rs
//
// the SYMBOLIC proof of the uct edge-emf upwind-pairing invariant — the instant,
// structural counterpart to the numerical blow-up gate
// (nmhd_uct_supersonic_emf_upwind.rs). instead of advecting a supersonic field
// loop for N steps and watching the magnetic energy, it reads the invariant
// straight off the traced DAG at graph-build time.
//
// the MASTER-FORM uct emf kernels — the solver-agnostic `eq:emf2D` composition (nmhd/imhd HLL +
// HLLD, rmhd HLL, and the GR ortho path) — all compose through the SAME `uct_master_emf`; THIS proof
// covers exactly those. it does NOT cover the wave-sum HLLD EMF kernels `rmhd_edge_emf_uct_hlld_gv` /
// `rmhd_edge_emf_uct_hlld_gr_gv`, which assemble centered advection + a dissipative Phi (M&DZ Eq. 39)
// rather than the master coefficient form; their dissipation-sign pairing is proven SEPARATELY in
// `rmhd_uct_hlld_wave_sum_symbolic.rs` (M8, via `hlld_wave_sum_proof_kernel`), not here.
// `uct_master_emf_proof_kernel` traces the master form in isolation with symbolic param
// leaves, so the result is LINEAR in the four staggered face reads {by_w, by_e, bx_n, bx_s} (all the
// wave-speed nonlinearity lives upstream in cx/cy, which here are opaque scalars al/ar/dl/dr). the
// master form is
//   emf = -vbar_x (a^L by_w + a^R by_e) + (d^R by_e - d^L by_w)
//       + vbar_y (a^L by_s + a^R by_n) - (d^R by_n - d^L by_s).
// `LinForm` extracts each face's coefficient polynomial; the upwind invariant is
// then a coefficient check: a^L (the alpha^+/sum weight) must multiply the UPWIND
// face — by_w for +x, bx_s for +y — NOT the downwind one. the ct_emf.rs:577
// anti-upwind bug (a^L paired with the downwind face) is invisible to the div(B)
// tests (the curl preserves div(B) for any emf) and invisible subsonically
// (a^L==a^R); this proves the pairing for ALL kernels with no evolve loop.
// =============================================================================

use symbi_discretize::uct_master_emf_proof_kernel;
use symbi_ir::proof::LinForm;

// the four staggered face reads are the "fields"; the uct coefficients are opaque
// scalars. by_w/bx_s are the UPWIND faces (+x West / +y South); by_e/bx_n the
// downwind faces.
const FIELDS: &[&str] = &["by_w", "by_e", "bx_n", "bx_s"];
const SCALARS: &[&str] = &[
    "vbar_x", "vbar_y", "al_x", "ar_x", "dl_x", "dr_x", "al_y", "ar_y", "dl_y", "dr_y",
];

// the symbolic param leaves carry no LoadAt, so every field read resolves to the
// zero-length (offset-free) stencil key.
fn coeff<'a>(lf: &'a LinForm, key: &str) -> &'a symbi_ir::proof::Poly {
    lf.terms
        .get(&(key.to_string(), vec![]))
        .unwrap_or_else(|| panic!("emf has no coefficient for face `{key}` — extraction changed shape"))
}

#[test]
fn uct_emf_upwind_pairing_symbolic() {
    let (kernel, writes) = uct_master_emf_proof_kernel(false);
    let root = writes[0].2;
    let lf = LinForm::extract(&kernel.graph, root, FIELDS, SCALARS);

    // +x advection: a^L (al_x) weights the UPWIND West face by_w; a^R (ar_x) the
    // downwind East face by_e. the products are -vbar_x*al_x and -vbar_x*ar_x.
    let by_w = coeff(&lf, "by_w");
    let by_e = coeff(&lf, "by_e");
    assert_eq!(by_w.coefficient_of(&["vbar_x", "al_x"]), -1, "a^L must weight the upwind face by_w");
    assert_eq!(by_w.coefficient_of(&["vbar_x", "ar_x"]), 0, "a^R must NOT weight the upwind face by_w");
    assert_eq!(by_e.coefficient_of(&["vbar_x", "ar_x"]), -1, "a^R must weight the downwind face by_e");
    assert_eq!(by_e.coefficient_of(&["vbar_x", "al_x"]), 0, "a^L must NOT weight the downwind face by_e");

    // the diffusion pairing rides along the same faces: d^L->West (by_w), d^R->East (by_e).
    assert_eq!(by_w.coefficient_of(&["dl_x"]), -1, "d^L must weight the West face by_w");
    assert_eq!(by_e.coefficient_of(&["dr_x"]), 1, "d^R must weight the East face by_e");

    // +y advection: a^L (al_y) weights the UPWIND South face bx_s; a^R (ar_y) the North bx_n.
    let bx_s = coeff(&lf, "bx_s");
    let bx_n = coeff(&lf, "bx_n");
    assert_eq!(bx_s.coefficient_of(&["vbar_y", "al_y"]), 1, "a^L must weight the upwind face bx_s");
    assert_eq!(bx_s.coefficient_of(&["vbar_y", "ar_y"]), 0, "a^R must NOT weight the upwind face bx_s");
    assert_eq!(bx_n.coefficient_of(&["vbar_y", "ar_y"]), 1, "a^R must weight the downwind face bx_n");
    assert_eq!(bx_n.coefficient_of(&["vbar_y", "al_y"]), 0, "a^L must NOT weight the downwind face bx_n");
}

// the negative control: feeding the master the anti-upwind (swapped by_w/by_e)
// pairing — exactly the ct_emf.rs:577 bug — MUST violate the invariant. proves the
// check is not vacuously green.
#[test]
fn uct_emf_anti_upwind_pairing_is_rejected() {
    let (kernel, writes) = uct_master_emf_proof_kernel(true);
    let root = writes[0].2;
    let lf = LinForm::extract(&kernel.graph, root, FIELDS, SCALARS);

    // with by_w/by_e swapped, a^L now lands on the DOWNWIND face: the upwind face
    // by_w carries a^R, not a^L. the production invariant above would fail.
    let by_w = coeff(&lf, "by_w");
    assert_eq!(by_w.coefficient_of(&["vbar_x", "al_x"]), 0, "bug: a^L no longer on by_w");
    assert_eq!(by_w.coefficient_of(&["vbar_x", "ar_x"]), -1, "bug: a^R wrongly weights the upwind face");
}
