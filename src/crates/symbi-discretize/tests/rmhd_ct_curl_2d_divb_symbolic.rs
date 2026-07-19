// =============================================================================
// rmhd_ct_curl_2d_divb_symbolic.rs
//
// the SYMBOLIC proof that the 2D CARTESIAN constrained-transport curl (rmhd_ct_curl_2d_dir_gv)
// preserves div(B) = 0 — by polynomial-coefficient cancellation on the traced IR DAG at graph-build
// time; no numeric evolve loop is run. closes the M10 gap "the 2D cartesian curl is numeric-only" (the 3D cartesian curl was
// already proven by rmhd_ct_curl3d_divb_symbolic; the 2D builder is a distinct kernel and was not).
//
// the 2D curl from the single out-of-plane corner EMF E_z (`ez`), inverse-width form:
//   dir=0 (B_x, x-face): dB_x/dt = -idy (E_z[+y] - E_z)
//   dir=1 (B_y, y-face): dB_y/dt = +idx (E_z[+x] - E_z)
// the point-form divergence idx (B_x[+x] - B_x) + idy (B_y[+y] - B_y), applied symbolically by
// shifting each curl's edge-emf reads, telescopes to the ZERO linear form (the mixed corner reads
// E_z[1,1]/E_z[1,0]/E_z[0,1]/E_z[0,0] cancel) for ANY input field — that is the proof.
// =============================================================================

use symbi_discretize::rmhd_ct_curl_2d_dir_gv;
use symbi_ir::proof::{LinForm, Poly};

const FIELDS: &[&str] = &["ez", "b"];
const SCALARS: &[&str] = &["dt", "idx", "idy"];

// strip the old-field `b` leaf: it reproduces div(B_old), invariant under the update — not part of
// the "the update preserves div" proof.
fn curl_only(mut lf: LinForm) -> LinForm {
    lf.terms.retain(|(key, _), _| key != "b");
    lf
}

#[test]
fn divb_2d_cartesian_symbolic_telescoping() {
    // div(B) = idx (B_x[+x] - B_x) + idy (B_y[+y] - B_y). B_x is dir 0 (weighted by idx, shift +x),
    // B_y is dir 1 (weighted by idy, shift +y).
    let mut div = LinForm::default();
    for (dir, id_div, e_dir) in [(0usize, "idx", [1i32, 0]), (1usize, "idy", [0i32, 1])] {
        let (kernel, writes) = rmhd_ct_curl_2d_dir_gv(dir);
        assert_eq!(writes.len(), 1, "curl builder must write exactly b_new");
        let curl = curl_only(LinForm::extract(&kernel.graph, writes[0].2, FIELDS, SCALARS));
        assert!(!curl.is_zero(), "dir {dir}: curl is empty — extractor saw no emf reads");
        let mut diff = curl.shifted(&e_dir);
        diff.add(&curl.shifted(&[0, 0]).neg_form());
        div.add(&diff.scale_var(id_div));
    }
    assert!(
        div.is_zero(),
        "2D cartesian div(curl B) != 0 symbolically — residual edge-emf coefficients:\n{:#?}",
        div.residual()
    );
}

// a NEGATIVE control: mismatched coefficients do NOT cancel, so the checker is not vacuously green —
// exactly the residual a sign/offset bug in the curl would leave.
#[test]
fn divb_2d_cartesian_symbolic_detects_broken() {
    let mut lf = LinForm::default();
    lf.add(&LinForm::single(("ez".into(), vec![0, 0]), Poly::var("idx")));
    lf.add(&LinForm::single(("ez".into(), vec![1, 0]), Poly::var("idy")));
    assert!(!lf.is_zero(), "mismatched coefficients must NOT cancel");
    assert_eq!(lf.residual().len(), 2);
}
